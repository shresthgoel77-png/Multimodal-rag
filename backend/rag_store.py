import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


EMBED_MODEL = "gemini-embedding-2"
DEFAULT_DIMENSIONS = 768
CHUNK_WORDS = 170
CHUNK_OVERLAP = 35
INLINE_MEDIA_LIMIT_BYTES = 18 * 1024 * 1024
FILE_API_POLL_SECONDS = 2
FILE_API_MAX_WAIT_SECONDS = 90

CHROMA_COLLECTION = "rag_chunks"
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"


MODALITY_COLORS = {
    "text": "#9fc9a2",
    "url": "#9fbbe0",
    "pdf": "#c08532",
    "image": "#c0a8dd",
    "audio": "#dfa88f",
    "video": "#e6e5e0",
    "query": "#f54e00",
}


@dataclass
class RackChunk:
    id: str
    source_id: str
    title: str
    modality: str
    text: str
    vector: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RackSource:
    id: str
    title: str
    modality: str
    summary: str
    chunks: int
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _chunk_text(text: str) -> list[str]:
    words = _clean_text(text).split()
    if not words:
        return []
    if len(words) <= CHUNK_WORDS:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    for start in range(0, len(words), step):
        chunk = words[start : start + CHUNK_WORDS]
        if len(chunk) >= 25:
            chunks.append(" ".join(chunk))
    return chunks


def _blend_vectors(primary: list[float], secondary: list[float], secondary_weight: float = 0.32) -> list[float]:
    primary_weight = 1.0 - secondary_weight
    blended = [
        (left * primary_weight) + (right * secondary_weight)
        for left, right in zip(primary, secondary)
    ]
    norm = math.sqrt(sum(value * value for value in blended)) or 1.0
    return [value / norm for value in blended]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-12:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def _orthogonalize(vector: list[float], components: list[list[float]]) -> list[float]:
    adjusted = vector[:]
    for component in components:
        projection = _dot(adjusted, component)
        adjusted = [value - projection * component[index] for index, value in enumerate(adjusted)]
    return adjusted


class MultimodalRagStore:
    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS):
        self.dimensions = dimensions
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.sources: list[RackSource] = []
        self.chunks: list[RackChunk] = []
        self.events: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.embedding_provider = "gemini-embedding-2"

        self.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", str(DEFAULT_CHROMA_DIR))
        self.chroma_error: str | None = None
        self._chroma_client: Any = None
        self._collection: Any = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            self.chroma_error = (
                "ChromaDB is not installed. Run: pip install 'chromadb>=1.5.9,<1.6.0'"
            )
            return

        try:
            path = Path(self.persist_directory).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            if not os.access(path, os.W_OK):
                raise PermissionError(f"directory exists but is not writable: {path}")

            self._chroma_client = chromadb.PersistentClient(
                path=str(path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            self.chroma_error = (
                f"ChromaDB is unavailable with CHROMA_PERSIST_DIRECTORY={self.persist_directory!r}: {exc}"
            )
            self._chroma_client = None
            self._collection = None
            return

        try:
            self._reload_from_chroma()
        except Exception as exc:
            self.chroma_error = (
                f"ChromaDB could not be read from CHROMA_PERSIST_DIRECTORY={self.persist_directory!r}: {exc}"
            )
            self._collection = None
            return

    def _require_collection(self) -> Any:
        if self.chroma_error:
            raise RuntimeError(self.chroma_error)
        if self._collection is None:
            raise RuntimeError(f"ChromaDB storage is not available at {self.persist_directory!r}.")
        return self._collection

    def _reload_from_chroma(self) -> None:
        collection = self._collection
        try:
            record = collection.get(include=["metadatas", "documents"])
        except Exception as exc:
            raise RuntimeError(f"ChromaDB failed to read stored chunks: {exc}") from exc

        self.chunks = []
        self.sources = []
        by_source: dict[str, dict[str, Any]] = {}

        ids = list(record.get("ids") or [])
        documents = list(record.get("documents") or [])
        metadatas = list(record.get("metadatas") or [])
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            meta = metadata or {}
            source_id = str(meta.get("source_id", ""))
            chunk = RackChunk(
                id=chunk_id,
                source_id=source_id,
                title=str(meta.get("title", "")),
                modality=str(meta.get("modality", "text")),
                text=document or "",
                vector=[],
                metadata=dict(meta),
                created_at=float(meta.get("created_at", 0.0) or 0.0),
            )
            self.chunks.append(chunk)
            if not source_id:
                continue
            aggregate = by_source.get(source_id)
            if aggregate is None:
                aggregate = {
                    "title": chunk.title,
                    "modality": chunk.modality,
                    "summary": str(meta.get("summary", "")),
                    "created_at": chunk.created_at,
                    "source_metadata": {},
                }
                try:
                    encoded = meta.get("source_metadata")
                    if encoded:
                        aggregate["source_metadata"] = json.loads(encoded)
                except (TypeError, ValueError):
                    aggregate["source_metadata"] = {}
                by_source[source_id] = aggregate
            aggregate["count"] = aggregate.get("count", 0) + 1

        for source_id, aggregate in by_source.items():
            self.sources.append(
                RackSource(
                    id=source_id,
                    title=aggregate["title"],
                    modality=aggregate["modality"],
                    summary=aggregate["summary"],
                    chunks=int(aggregate.get("count", 0)),
                    created_at=aggregate["created_at"],
                    metadata=dict(aggregate.get("source_metadata", {})),
                )
            )

    # ------------------------------------------------------------------ helpers

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "at": time.time(), **payload})
        self.events = self.events[-80:]

    def _require_client(self) -> genai.Client:
        if not self.client:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini Embedding 2.")
        return self.client

    def _embed_text(self, text: str, task_prefix: str) -> list[float]:
        content = f"{task_prefix}: {text}"
        client = self._require_client()

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[content],
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
        )
        return result.embeddings[0].values

    def _embed_uploaded_file(self, data: bytes, mime_type: str, title: str) -> list[float]:
        client = self._require_client()
        suffix = Path(title).suffix or self._suffix_from_mime(mime_type)
        uploaded = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
                temp_file.write(data)
                temp_file.flush()
                uploaded = client.files.upload(
                    file=temp_file.name,
                    config=types.UploadFileConfig(mime_type=mime_type, display_name=title),
                )

            waited = 0
            while getattr(getattr(uploaded, "state", None), "name", "") == "PROCESSING":
                if waited >= FILE_API_MAX_WAIT_SECONDS:
                    raise ValueError("Gemini is still processing this media file. Try a shorter clip or upload it again in a moment.")
                time.sleep(FILE_API_POLL_SECONDS)
                waited += FILE_API_POLL_SECONDS
                uploaded = client.files.get(name=uploaded.name)

            state_name = getattr(getattr(uploaded, "state", None), "name", "")
            if state_name and state_name not in {"ACTIVE", "SUCCEEDED"}:
                raise ValueError(f"Gemini could not process this media file. File state: {state_name}.")

            part = types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime_type)
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[part],
                config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
            )
            return result.embeddings[0].values
        finally:
            uploaded_name = getattr(uploaded, "name", None)
            if uploaded_name:
                try:
                    client.files.delete(name=uploaded_name)
                except Exception as exc:
                    self._emit("file_cleanup_failed", {"name": uploaded_name, "error": str(exc)})

    def _embed_file(self, data: bytes, mime_type: str, title: str, notes: str) -> tuple[list[float], str]:
        client = self._require_client()

        use_file_api = (
            len(data) > INLINE_MEDIA_LIMIT_BYTES
            or mime_type.startswith("video/")
            or mime_type.startswith("audio/")
        )
        if use_file_api:
            return self._embed_uploaded_file(data, mime_type, title), "gemini-file-api"

        part = types.Part.from_bytes(data=data, mime_type=mime_type)
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[part],
                config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
            )
            return result.embeddings[0].values, "gemini-inline"
        except Exception:
            if mime_type.startswith(("video/", "audio/")) or mime_type == "application/pdf":
                return self._embed_uploaded_file(data, mime_type, title), "gemini-file-api"
            raise

    def _validate_vector(self, vector: list[float]) -> None:
        if not isinstance(vector, (list, tuple)):
            raise ValueError("Chunk embeddings must be provided as a vector (list of floats).")
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Malformed embedding for a chunk: expected {self.dimensions} dimensions, got {len(vector)}."
            )
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ValueError(f"Chunk embeddings must contain only numeric values, got vector of length {len(vector)}.")

    def _pca_projection(self, vectors: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        if not vectors:
            return {}

        ids = list(vectors)
        rows = [vectors[item_id][: self.dimensions] for item_id in ids]
        if len(rows) == 1:
            return {ids[0]: {"x": 0.0, "y": 0.0, "z": 0.0}}

        means = [sum(row[index] for row in rows) / len(rows) for index in range(self.dimensions)]
        centered = [[row[index] - means[index] for index in range(self.dimensions)] for row in rows]
        components: list[list[float]] = []

        for component_index in range(3):
            candidate = [
                math.sin((index + 1) * (component_index + 1) * 0.017)
                + math.cos((index + 1) * (component_index + 2) * 0.013)
                for index in range(self.dimensions)
            ]
            candidate = _normalize(_orthogonalize(candidate, components))

            for _ in range(24):
                scores = [_dot(row, candidate) for row in centered]
                next_candidate = [0.0] * self.dimensions
                for score, row in zip(scores, centered):
                    for index, value in enumerate(row):
                        next_candidate[index] += score * value
                next_candidate = _normalize(_orthogonalize(next_candidate, components))
                if not any(next_candidate):
                    break
                candidate = next_candidate

            if not any(candidate):
                candidate = [0.0] * self.dimensions
                candidate[min(component_index, self.dimensions - 1)] = 1.0
            components.append(candidate)

        raw = {
            item_id: [_dot(row, component) for component in components]
            for item_id, row in zip(ids, centered)
        }
        max_radius = max(
            math.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])
            for values in raw.values()
        ) or 1.0
        scale = 2.65 / max_radius
        return {
            item_id: {
                "x": round(values[0] * scale, 4),
                "y": round(values[1] * scale, 4),
                "z": round(values[2] * scale, 4),
            }
            for item_id, values in raw.items()
        }

    def _source_vector(self, source_id: str) -> list[float]:
        collection = self._require_collection()
        try:
            record = collection.get(where={"source_id": source_id}, include=["embeddings"])
        except Exception as exc:
            raise RuntimeError(f"ChromaDB failed to read chunks for source {source_id}: {exc}") from exc

        embeddings = record.get("embeddings")
        if embeddings is None:
            return [0.0] * self.dimensions
        embeddings = list(embeddings)
        if not embeddings:
            return [0.0] * self.dimensions

        vector = [0.0] * self.dimensions
        for embedding in embeddings:
            for index, value in enumerate(embedding[: self.dimensions]):
                vector[index] += value
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _source_vectors(self) -> dict[str, list[float]]:
        return {source.id: self._source_vector(source.id) for source in self.sources}

    def _source_point(self, source: RackSource, projection: dict[str, float]) -> dict[str, Any]:
        return {
            "id": source.id,
            "source_id": source.id,
            "title": source.title,
            "modality": source.modality,
            "projection": projection,
            "color": MODALITY_COLORS.get(source.modality, "#94a3b8"),
            "preview": source.summary,
        }

    @staticmethod
    def _chunk_meta(
        source_id: str,
        index: int,
        title: str,
        modality: str,
        created_at: float,
        summary: str,
        source_metadata: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source_id": source_id,
            "chunk_index": index,
            "title": title,
            "modality": modality,
            "created_at": created_at,
            "summary": summary,
        }
        if source_metadata:
            meta["source_metadata"] = json.dumps(source_metadata, sort_keys=True)
        if extra:
            meta.update(extra)
        return meta

    @staticmethod
    def _source_id_for_text(title: str, text: str, modality: str) -> str:
        canonical = f"text::{modality}::{title}::{_clean_text(text)}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _source_id_for_file(title: str, data: bytes, mime_type: str, notes: str, modality: str) -> str:
        content_hash = hashlib.sha256(data).hexdigest()
        canonical = f"file::{modality}::{title}::{mime_type}::{notes}::{content_hash}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ ingestion

    def add_text_source(self, title: str, text: str, modality: str = "text", seed: bool = False) -> RackSource:
        with self._lock:
            self._require_collection()
            chunks_text = _chunk_text(text)
            if not chunks_text:
                raise ValueError("Source text is empty.")

            cleaned = _clean_text(text)
            title = title.strip() or f"{modality.title()} source"
            source_id = self._source_id_for_text(title, cleaned, modality)
            chunk_metadatas = [{"chunk_index": index + 1} for index in range(len(chunks_text))]

            return self._ingest_chunks(
                source_id=source_id,
                title=title,
                modality=modality,
                summary=cleaned[:220],
                chunk_texts=chunks_text,
                chunk_metadatas=chunk_metadatas,
                source_metadata={},
                seed=seed,
            )

    def add_file_source(self, title: str, data: bytes, mime_type: str, notes: str = "") -> RackSource:
        with self._lock:
            self._require_collection()
            modality = self._modality_from_mime(mime_type)
            title = title.strip() or "Uploaded source"
            display_text = _clean_text(notes) or f"{title} ({mime_type}) embedded natively in Gemini Embedding 2."
            annotation_text = _clean_text(f"{title}. {display_text}")

            media_vector, embedding_path = self._embed_file(data, mime_type, title, display_text)
            annotation_vector = self._embed_text(annotation_text, "task: retrieval document")
            vector = _blend_vectors(media_vector, annotation_vector)
            self._validate_vector(vector)

            source_id = self._source_id_for_file(title, data, mime_type, notes, modality)
            source_metadata = {
                "mime_type": mime_type,
                "bytes": len(data),
                "embedding_path": embedding_path,
                "annotation_blended": True,
            }
            chunk_metadata = {
                "mime_type": mime_type,
                "bytes": len(data),
                "native_multimodal": bool(self.client),
                "embedding_path": embedding_path,
                "annotation_blended": True,
            }

            return self._ingest_chunks(
                source_id=source_id,
                title=title,
                modality=modality,
                summary=display_text[:220],
                chunk_texts=[display_text],
                chunk_metadatas=[chunk_metadata],
                source_metadata=source_metadata,
                vectors=[vector],
                seed=False,
            )

    def _ingest_chunks(
        self,
        source_id: str,
        title: str,
        modality: str,
        summary: str,
        chunk_texts: list[str],
        chunk_metadatas: list[dict[str, Any]],
        source_metadata: dict[str, Any],
        seed: bool,
        vectors: list[list[float]] | None = None,
    ) -> RackSource:
        collection = self._require_collection()
        existing = next((source for source in self.sources if source.id == source_id), None)
        created_at = existing.created_at if existing else time.time()

        if vectors is None:
            vectors = []
            for chunk_text in chunk_texts:
                vector = self._embed_text(chunk_text, "task: retrieval document")
                self._validate_vector(vector)
                vectors.append(vector)
        # For file sources the caller provides pre-blended vectors; still validate them.
        for vector in vectors:
            self._validate_vector(vector)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for index, (chunk_text, extra) in enumerate(zip(chunk_texts, chunk_metadatas)):
            ids.append(f"{source_id}::{index + 1}")
            documents.append(chunk_text)
            metadatas.append(
                self._chunk_meta(source_id, index + 1, title, modality, created_at, summary, source_metadata, extra)
            )

        # Replace any previously stored chunks for this source (upsert-by-replace),
        # so re-ingesting the same source never leaves duplicate chunks behind.
        if existing is not None:
            self._delete_chroma_chunks_for_source(collection, source_id)

        try:
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
        except Exception as exc:
            raise RuntimeError(f"ChromaDB failed to store chunks for source {source_id}: {exc}") from exc

        self._rebuild_source_mirror(
            source_id=source_id,
            title=title,
            modality=modality,
            summary=summary,
            created_at=created_at,
            chunk_texts=chunk_texts,
            chunk_metadatas=chunk_metadatas,
            source_metadata=source_metadata,
        )

        if existing is not None:
            self._emit("source_updated", {"source_id": source_id, "title": title, "chunks": len(ids)})
        elif not seed:
            self._emit("source_added", {"source_id": source_id, "title": title, "chunks": len(ids)})

        return self._source_record(source_id)

    def _delete_chroma_chunks_for_source(self, collection: Any, source_id: str) -> None:
        try:
            collection.delete(where={"source_id": source_id})
        except Exception as exc:
            raise RuntimeError(f"ChromaDB failed to delete chunks for source {source_id}: {exc}") from exc

    def _rebuild_source_mirror(
        self,
        source_id: str,
        title: str,
        modality: str,
        summary: str,
        created_at: float,
        chunk_texts: list[str],
        chunk_metadatas: list[dict[str, Any]],
        source_metadata: dict[str, Any],
    ) -> None:
        self.sources = [source for source in self.sources if source.id != source_id]
        self.chunks = [chunk for chunk in self.chunks if chunk.source_id != source_id]

        self.sources.append(
            RackSource(
                id=source_id,
                title=title,
                modality=modality,
                summary=summary,
                chunks=len(chunk_texts),
                created_at=created_at,
                metadata=dict(source_metadata),
            )
        )
        for index, (chunk_text, extra) in enumerate(zip(chunk_texts, chunk_metadatas)):
            meta = self._chunk_meta(source_id, index + 1, title, modality, created_at, summary, source_metadata, extra)
            self.chunks.append(
                RackChunk(
                    id=f"{source_id}::{index + 1}",
                    source_id=source_id,
                    title=title,
                    modality=modality,
                    text=chunk_text,
                    vector=[],
                    metadata=meta,
                    created_at=created_at,
                )
            )

    def _source_record(self, source_id: str) -> RackSource:
        source = next((item for item in self.sources if item.id == source_id), None)
        if source is None:
            raise RuntimeError(f"Source {source_id} was not registered after ingestion.")
        return source

    # ------------------------------------------------------------------ deletion

    def remove_source(self, source_id: str) -> bool:
        with self._lock:
            collection = self._require_collection()
            source = next((item for item in self.sources if item.id == source_id), None)
            if source is None:
                return False

            self._delete_chroma_chunks_for_source(collection, source_id)
            self.sources = [item for item in self.sources if item.id != source_id]
            self.chunks = [chunk for chunk in self.chunks if chunk.source_id != source_id]
            self._emit("source_removed", {"source_id": source_id, "title": source.title})
            return True

    # ------------------------------------------------------------------ retrieval

    def search(self, query: str, top_k: int = 6) -> dict[str, Any]:
        with self._lock:
            collection = self._require_collection()
            query_vector = self._embed_text(query, "task: question answering | query")
            self._validate_vector(query_vector)
            query_id = f"query-{uuid.uuid4().hex[:8]}"

            source_vectors = self._source_vectors()
            projections = self._pca_projection({**source_vectors, query_id: query_vector})
            query_point = {
                "id": query_id,
                "source_id": "query",
                "title": query,
                "modality": "query",
                "projection": projections.get(query_id, {"x": 0.0, "y": 0.0, "z": 0.0}),
                "color": MODALITY_COLORS["query"],
                "score": 1,
                "preview": "Query embedding projected with the active source set.",
            }

            count = self._collection_count(collection)
            if count == 0:
                self._emit("query_embedded", {"query": query, "matches": []})
                return {
                    "query_point": query_point,
                    "matches": [],
                    "space": self.snapshot(projections=projections),
                }

            source_by_id = {source.id: source for source in self.sources}
            nearest = self._query_collection(collection, query_vector, min(top_k * 3, count))
            source_matches: dict[str, dict[str, Any]] = {}
            for chunk_id, text, metadata, distance in zip(
                nearest["ids"], nearest["documents"], nearest["metadatas"], nearest["distances"]
            ):
                meta = metadata or {}
                source_id = str(meta.get("source_id", ""))
                source = source_by_id.get(source_id)
                if source is None:
                    continue
                score = round(1.0 - float(distance), 4)
                current = source_matches.get(source_id)
                if not current or score > current["score"]:
                    source_matches[source_id] = {
                        "id": source.id,
                        "source_id": source.id,
                        "title": source.title,
                        "modality": source.modality,
                        "text": text,
                        "score": score,
                        "projection": projections.get(source.id, {"x": 0.0, "y": 0.0, "z": 0.0}),
                        "metadata": {"best_chunk": chunk_id, **meta},
                    }

            matches = sorted(source_matches.values(), key=lambda item: item["score"], reverse=True)[:top_k]
            self._emit("query_embedded", {"query": query, "matches": [m["id"] for m in matches]})
            return {
                "query_point": query_point,
                "matches": matches,
                "space": self.snapshot(projections=projections),
            }

    def _collection_count(self, collection: Any) -> int:
        try:
            return collection.count()
        except Exception as exc:
            raise RuntimeError(f"ChromaDB failed to count stored chunks: {exc}") from exc

    def _query_collection(self, collection: Any, vector: list[float], n_results: int) -> dict[str, list[Any]]:
        try:
            result = collection.query(
                query_embeddings=[vector],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            raise RuntimeError(f"ChromaDB query failed: {exc}") from exc
        return self._normalize_query_result(result)

    @staticmethod
    def _normalize_query_result(result: Any) -> dict[str, list[Any]]:
        def unwrap(value: Any) -> list[Any]:
            if not value:
                return []
            wrapped = value if isinstance(value[0], list) else [value]
            return wrapped[0] if wrapped else []

        return {
            "ids": unwrap(result["ids"]),
            "distances": unwrap(result["distances"]),
            "metadatas": unwrap(result["metadatas"]),
            "documents": unwrap(result["documents"]),
        }

    # ------------------------------------------------------------------ space

    def snapshot(self, projections: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
        with self._lock:
            source_vectors = self._source_vectors()
            projection_map = projections or self._pca_projection(source_vectors)
            points = [
                self._source_point(source, projection_map.get(source.id, {"x": 0.0, "y": 0.0, "z": 0.0}))
                for source in self.sources
            ]
            return {
                "sources": [source.__dict__ for source in self.sources],
                "points": points,
                "events": self.events,
                "provider": self.embedding_provider,
                "dimensions": self.dimensions,
                "model": EMBED_MODEL,
                "projection": {
                    "method": "pca_3d",
                    "basis": "current source vectors plus active query when present",
                },
            }

    def retrieval_tool(self, query: str, top_k: int = 5) -> dict[str, Any]:
        results = self.search(query, top_k=top_k)
        return self.retrieval_payload(results)

    def retrieval_payload(self, results: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": self.embedding_provider,
            "matches": [
                {
                    "citation": match["id"],
                    "source": match["title"],
                    "modality": match["modality"],
                    "similarity": match["score"],
                    "evidence": match["text"],
                }
                for match in results["matches"]
            ],
        }

    def space_tool(self) -> dict[str, Any]:
        with self._lock:
            self._require_collection()
            source_modalities: dict[str, int] = {}
            chunk_modalities: dict[str, int] = {}
            for source in self.sources:
                source_modalities[source.modality] = source_modalities.get(source.modality, 0) + 1
            for chunk in self.chunks:
                chunk_modalities[chunk.modality] = chunk_modalities.get(chunk.modality, 0) + 1
            return {
                "sources": len(self.sources),
                "chunks": len(self.chunks),
                "dimensions": self.dimensions,
                "provider": self.embedding_provider,
                "modalities": source_modalities,
                "chunk_modalities": chunk_modalities,
                "projection": "pca_3d",
            }

    @staticmethod
    def _modality_from_mime(mime_type: str) -> str:
        if mime_type == "application/pdf":
            return "pdf"
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type.startswith("video/"):
            return "video"
        return "text"

    @staticmethod
    def _suffix_from_mime(mime_type: str) -> str:
        if mime_type == "application/pdf":
            return ".pdf"
        if mime_type.startswith("image/"):
            return f".{mime_type.split('/', 1)[1].split(';', 1)[0]}"
        if mime_type.startswith("audio/"):
            return ".mp3"
        if mime_type.startswith("video/"):
            return ".mp4"
        return ".bin"