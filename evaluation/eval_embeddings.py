"""Deterministic, API-free embeddings for the fixed evaluation corpus.

The benchmark corpus is ingested into ChromaDB using these embeddings instead
of Gemini Embedding 2 so that the phase-5 evaluation harness can run fully
offline and reproducibly (the same approach the repo's existing unit tests
use when they monkeypatch ``_embed_text``). Retrieval quality is lexical:
features are word unigrams + word bigrams hashed into a fixed-dimension,
L2-normalized vector with feature hashing.

Consistency contract for Phases 6/7:
  - Ingest the corpus with ``install_deterministic_embeddings(store)`` set.
  - Run every query with the *same* ``embed_text`` function installed so the
    query embedding lives in the same space as the stored chunk vectors.
When ``GOOGLE_API_KEY`` is available, a future phase may instead re-ingest the
same unchanged corpus text with the real embedding provider; the benchmark
file's ``expected_sources`` refer to the deterministic source ids recorded in
``corpus/manifest.json``, which derive from title+content only and therefore
change value but not *meaning* if the embedding provider changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

DETERMINISTIC_DIM = 768

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "its", "not", "of", "on", "or", "than", "that",
    "the", "these", "this", "those", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "will", "can", "would",
}


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def text_features(text: str) -> dict[tuple[str, ...], float]:
    words = [w for w in tokenize(text) if w not in _STOPWORDS and len(w) > 1]
    features: dict[tuple[str, ...], float] = {}
    for word in words:
        features[("w", word)] = features.get(("w", word), 0.0) + 1.0
    for left, right in zip(words, words[1:]):
        features[("b", left, right)] = features.get(("b", left, right), 0.0) + 1.0
    return features


def embed_text(text: str, task_prefix: str = "") -> list[float]:
    """Embed `text` into a DETERMINISTIC_DIM-dimensional L2-normalized vector.

    `task_prefix` is accepted for signature compatibility with
    ``MultimodalRagStore._embed_text`` but intentionally ignored so that the
    chunk-time and query-time embeddings live in the same space.
    """
    features = text_features(text)
    vector = [0.0] * DETERMINISTIC_DIM
    for feature, count in features.items():
        raw = hashlib.sha256(repr(feature).encode("utf-8")).digest()
        digest_int = int.from_bytes(raw[:8], "little")
        index = digest_int % DETERMINISTIC_DIM
        sign = 1.0 if (digest_int >> 62) & 1 else -1.0
        vector[index] += sign * math.sqrt(count)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def install_deterministic_embeddings(store: Any) -> dict[str, Any]:
    """Monkeypatch a MultimodalRagStore to use deterministic embeddings.

    Mirrors the technique used in backend/tests/test_rag_store.py and changes
    no Phase 1-4 library code: the store object's ``_embed_text`` slot is
    replaced, and store-level behavior is untouched.

    Returns a small info dict describing the override.
    """
    store._embed_text = lambda text, task_prefix="": embed_text(text)  # type: ignore[method-assign]
    return {
        "embedding_provider": "deterministic-feature-hash",
        "dimensions": DETERMINISTIC_DIM,
    }