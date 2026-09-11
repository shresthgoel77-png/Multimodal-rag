import hashlib
import math

import pytest

from rag_store import MultimodalRagStore, _chunk_text


DIM = MultimodalRagStore().dimensions
LONG_TEXT = (" ".join(["querystellar"] * 400)) + ". " + (" ".join(["unrelated"] * 100))
SHORT_TEXT = "The chromium storage layer persists embeddings across restarts."


def _deterministic_vector(text: str, dim: int = DIM) -> list[float]:
    state = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    vector = []
    for _ in range(dim):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        vector.append((state / 0xFFFFFFFF) - 0.5)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _mock_embeddings(store: MultimodalRagStore) -> None:
    store._embed_text = lambda text, task_prefix: _deterministic_vector(text)
    store._embed_file = lambda data, mime_type, title, notes: (_deterministic_vector(f"{title}::{notes}"), "test-file-api")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    instance = MultimodalRagStore()
    _mock_embeddings(instance)
    return instance


def _restart_store(store: MultimodalRagStore):
    """Simulate a backend restart: a brand-new store over the same persist directory."""
    fresh = MultimodalRagStore()
    _mock_embeddings(fresh)
    return fresh


def test_persists_across_restart(store):
    source = store.add_text_source("Persist me", LONG_TEXT, "text")
    query = _chunk_text(LONG_TEXT)[1]

    restarted = _restart_store(store)
    assert len(restarted.sources) == 1
    assert restarted.sources[0].id == source.id
    assert restarted._collection.count() == len(_chunk_text(LONG_TEXT))

    results = restarted.search(query, top_k=3)
    assert results["matches"], "Expected a match after restart without re-ingesting"
    top = results["matches"][0]
    assert top["source_id"] == source.id
    assert top["title"] == "Persist me"
    assert top["score"] == pytest.approx(1.0, abs=0.001)
    assert top["text"] == query


def test_reingest_same_source_no_duplicates(store):
    first = store.add_text_source("Dedupe me", LONG_TEXT, "text")
    first_count = store._collection.count()
    first_chunk_ids = {chunk.id for chunk in store.chunks}

    second = store.add_text_source("Dedupe me", LONG_TEXT, "text")

    assert second.id == first.id, "Deterministic source identity expected"
    assert len(store.sources) == 1
    assert store._collection.count() == first_count, "Chunk count must not double"
    assert len(store.chunks) == first_count
    assert {chunk.id for chunk in store.chunks} == first_chunk_ids
    assert store._collection.get()["ids"].__len__() == first_count


def test_reingest_upsert_keeps_created_at(store):
    first = store.add_text_source("Timed", LONG_TEXT, "text")
    second = store.add_text_source("Timed", LONG_TEXT, "text")
    assert second.created_at == first.created_at


def test_delete_removes_all_chunks_from_chroma(store):
    source = store.add_text_source("Delete me", LONG_TEXT, "text")
    for chunk in store.chunks:
        assert store._collection.get(ids=[chunk.id])["ids"] == [chunk.id]

    removed = store.remove_source(source.id)
    assert removed is True
    assert store._collection.count() == 0
    assert store._collection.get(where={"source_id": source.id})["ids"] == []
    assert len(store.sources) == 0
    assert len(store.chunks) == 0

    results = store.search("querystellar", top_k=3)
    assert results["matches"] == []


def test_delete_unknown_source_returns_false(store):
    assert store.remove_source("does-not-exist") is False


def test_empty_database_query_is_graceful(store):
    results = store.search("anything at all", top_k=6)
    assert results["matches"] == []
    assert results["query_point"]["modality"] == "query"
    assert results["space"]["points"] == []
    assert results["space"]["sources"] == []

    snapshot = store.snapshot()
    assert snapshot["sources"] == []
    assert snapshot["points"] == []

    tool = store.space_tool()
    assert tool["sources"] == 0
    assert tool["chunks"] == 0


def test_chroma_error_invalid_persist_path(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("this is a file, not a directory")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(blocker / "chroma"))

    broken = MultimodalRagStore()
    assert broken.chroma_error is not None
    assert "ChromaDB" in broken.chroma_error

    with pytest.raises(RuntimeError, match="ChromaDB"):
        broken.space_tool()
    with pytest.raises(RuntimeError, match="ChromaDB"):
        broken.add_text_source("t", SHORT_TEXT, "text")
    with pytest.raises(RuntimeError, match="ChromaDB"):
        broken.search("what is chroma?", top_k=3)
    with pytest.raises(RuntimeError, match="ChromaDB"):
        broken.remove_source("anything")


def test_file_source_metadata_survives_restart(store):
    source = store.add_file_source(
        title="note.pdf",
        data=b"%PDF-1.4 fake pdf bytes",
        mime_type="application/pdf",
        notes="my pdf notes",
    )
    assert source.modality == "pdf"
    assert source.metadata["mime_type"] == "application/pdf"
    assert source.summary

    restarted = _restart_store(store)
    rebuilt = restarted.sources[0]
    assert rebuilt.id == source.id
    assert rebuilt.modality == "pdf"
    assert rebuilt.metadata.get("mime_type") == "application/pdf"
    assert rebuilt.metadata.get("embedding_path") == "test-file-api"
    assert restarted._collection.count() == 1


def test_malformed_embedding_rejected(store):
    store._embed_text = lambda text, task_prefix: [0.5] * 3
    with pytest.raises(ValueError, match="Malformed embedding"):
        store.add_text_source("Bad vector", SHORT_TEXT, "text")


def test_retrieval_payload_shape(store):
    store._embed_text = lambda text, task_prefix: _deterministic_vector(text)
    store.add_text_source("Citation", SHORT_TEXT, "text")
    results = store.search(SHORT_TEXT, top_k=3)
    payload = store.retrieval_payload(results)
    assert payload["matches"]
    match = payload["matches"][0]
    assert set(match) == {"citation", "source", "modality", "similarity", "evidence"}
    assert len(payload["matches"]) == len(results["matches"])


def test_default_persist_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("CHROMA_PERSIST_DIRECTORY", raising=False)
    store = MultimodalRagStore()
    assert store.persist_directory.endswith("backend/chroma_db")