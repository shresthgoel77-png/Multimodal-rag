import hashlib
import json
import math
import time

import pytest

from rag_store import MultimodalRagStore, _chunk_text
from reranker import (
    RELEVANCE_MAX,
    RELEVANCE_MIN,
    GeminiReranker,
    RerankTimeoutError,
    rerank_and_select,
)


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


def _candidate(cid: str, similarity: float, source_id: str | None = None, text: str = "chunk text") -> dict:
    return {
        "id": cid,
        "source_id": source_id or cid.split("::")[0],
        "title": "Source",
        "modality": "text",
        "text": text,
        "similarity": similarity,
        "metadata": {},
    }


def _reranker(text: str) -> GeminiReranker:
    instance = GeminiReranker(client=object())
    instance._client_generate = lambda prompt: text
    return instance


class _ScoreAllReranker:
    available = True
    timeout = 1.0

    def __init__(self, relevance: float = 0.9) -> None:
        self.relevance = relevance

    def score_candidates(self, query: str, candidates: list[dict]) -> dict:
        return {
            "scores": {c["id"]: {"relevance": self.relevance, "reason": "scored"} for c in candidates},
            "errors": [],
        }


class _FailReranker:
    available = True
    timeout = 1.0

    def score_candidates(self, query: str, candidates: list[dict]) -> dict:
        raise RuntimeError("rerank backend exploded")


class _UnavailableReranker:
    available = False

    def score_candidates(self, query: str, candidates: list[dict]) -> dict:
        raise AssertionError("should never be called")


# ------------------------------------------------------------------ normal path

def test_normal_rerank_uses_both_scores():
    candidates = [_candidate(f"c{i}", 0.9 - i * 0.05) for i in range(12)]
    reranker = _reranker(
        json.dumps(
            {
                "results": [
                    {"id": "c0", "relevance": 0.9, "reason": "direct match"},
                    {"id": "c1", "relevance": 0.8, "reason": "close"},
                    {"id": "c2", "relevance": 0.7, "reason": ""},
                    {"id": "c3", "relevance": 0.6, "reason": None},
                    {"id": "c4", "relevance": 0.5, "reason": "partial"},
                ]
            }
        )
    )

    outcome = rerank_and_select(reranker, "the question", candidates, final_k=5)

    assert outcome["used_reranking"] is True
    assert outcome["fallback"] is False
    assert outcome["reason"] == ""
    assert [e["id"] for e in outcome["evidence"]] == ["c0", "c1", "c2", "c3", "c4"]
    top = outcome["evidence"][0]
    assert top["relevance"] == 0.9
    assert top["reason"] == "direct match"
    assert top["similarity"] == 0.9
    assert outcome["evidence"][2]["reason"] is None, "blank reason normalizes to None"
    assert outcome["evidence"][3]["reason"] is None


# ------------------------------------------------------------------ fallback paths

def test_malformed_json_falls_back_to_similarity():
    candidates = [_candidate(f"c{i}", 0.9 - i * 0.05) for i in range(6)]
    reranker = _reranker("this is not json {{{")

    outcome = rerank_and_select(reranker, "q", candidates, final_k=5)

    assert outcome["used_reranking"] is False
    assert outcome["fallback"] is True
    assert "not valid JSON" in outcome["errors"][0]
    assert [e["id"] for e in outcome["evidence"]] == ["c0", "c1", "c2", "c3", "c4"]
    assert all(e["relevance"] is None for e in outcome["evidence"])


def test_wrong_schema_falls_back():
    candidates = [_candidate(f"c{i}", 0.9 - i * 0.05) for i in range(4)]
    reranker = _reranker(json.dumps([{"id": "c0", "relevance": 0.9}]))

    outcome = rerank_and_select(reranker, "q", candidates, final_k=2)

    assert outcome["fallback"] is True
    assert outcome["used_reranking"] is False
    assert any("schema" in error for error in outcome["errors"])


def test_empty_reranker_response_falls_back():
    candidates = [_candidate("c0", 0.9), _candidate("c1", 0.8)]
    outcome = rerank_and_select(_reranker("   "), "q", candidates, final_k=2)
    assert outcome["fallback"] is True
    assert "empty response" in outcome["errors"][0]


def test_reranker_generic_failure_falls_back():
    candidates = [_candidate(f"c{i}", 0.9 - i * 0.05) for i in range(4)]
    outcome = rerank_and_select(_FailReranker(), "q", candidates, final_k=2)
    assert outcome["fallback"] is True
    assert outcome["used_reranking"] is False
    assert "rerank backend exploded" in outcome["reason"]
    assert outcome["evidence"], "generation must still get evidence"


def test_reranker_timeout_falls_back():
    candidates = [_candidate(f"c{i}", 0.9 - i * 0.05) for i in range(4)]
    slow = GeminiReranker(client=object(), timeout=0.05)
    slow._client_generate = lambda prompt: time.sleep(2.0) or ""

    started = time.monotonic()
    outcome = rerank_and_select(slow, "q", candidates, final_k=2)
    elapsed = time.monotonic() - started

    assert outcome["fallback"] is True
    assert outcome["used_reranking"] is False
    assert "timed out" in outcome["reason"]
    assert elapsed < 1.5, "timeout must cut the call short"


def test_unavailable_reranker_falls_back():
    candidates = [_candidate(f"c{i}", 0.9 - i * 0.05) for i in range(4)]
    outcome = rerank_and_select(_UnavailableReranker(), "q", candidates, final_k=2)
    assert outcome["fallback"] is True
    assert "unavailable" in outcome["reason"]
    assert outcome["evidence"], "fallback evidence still selected"


def test_timeout_error_class():
    with pytest.raises(RerankTimeoutError):
        raise RerankTimeoutError("boom")


# ------------------------------------------------------------------ validation rules

def test_unknown_id_is_discarded():
    candidates = [_candidate("c0", 0.9)]
    reranker = _reranker(
        json.dumps(
            {
                "results": [
                    {"id": "ghost-id", "relevance": 1.0},
                    {"id": "c0", "relevance": 0.6},
                ]
            }
        )
    )

    outcome = rerank_and_select(reranker, "q", candidates, final_k=1)

    assert outcome["used_reranking"] is True
    assert [e["id"] for e in outcome["evidence"]] == ["c0"]
    assert any("unknown candidate" in error for error in outcome["errors"])


def test_out_of_range_relevance_is_clamped():
    candidates = [_candidate("c0", 0.9), _candidate("c1", 0.8), _candidate("c2", 0.7)]
    reranker = _reranker(
        json.dumps(
            {
                "results": [
                    {"id": "c0", "relevance": 5.0},
                    {"id": "c1", "relevance": -3.0},
                    {"id": "c2", "relevance": 0.7},
                ]
            }
        )
    )

    outcome = rerank_and_select(reranker, "q", candidates, final_k=3)

    by_id = {e["id"]: e for e in outcome["evidence"]}
    assert by_id["c0"]["relevance"] == RELEVANCE_MAX
    assert by_id["c1"]["relevance"] == RELEVANCE_MIN
    assert by_id["c2"]["relevance"] == 0.7
    assert [e["id"] for e in outcome["evidence"]] == ["c0", "c2", "c1"]
    assert len(outcome["errors"]) == 2


def test_duplicate_keeps_first_occurrence_and_missing_fills_by_similarity():
    candidates = [_candidate(f"c{i}", 0.95 - i * 0.1) for i in range(5)]
    reranker = _reranker(
        json.dumps(
            {
                "results": [
                    {"id": "c0", "relevance": 0.5},
                    {"id": "c0", "relevance": 0.9},
                    {"id": "c2", "relevance": 0.8},
                ]
            }
        )
    )

    outcome = rerank_and_select(reranker, "q", candidates, final_k=3)

    by_id = {e["id"]: e for e in outcome["evidence"]}
    assert by_id["c0"]["relevance"] == 0.5, "first occurrence wins"
    assert by_id["c2"]["relevance"] == 0.8
    assert [e["id"] for e in outcome["evidence"]] == ["c2", "c0", "c1"]
    assert outcome["evidence"][2]["relevance"] is None, "missing candidate fills slot unscored"
    assert any("duplicate" in error for error in outcome["errors"])


def test_missing_candidates_excluded_when_enough_scored():
    candidates = [_candidate(f"c{i}", 0.9 - i * 0.05) for i in range(5)]
    reranker = _reranker(
        json.dumps({"results": [{"id": "c0", "relevance": 0.9}, {"id": "c1", "relevance": 0.8}]})
    )
    outcome = rerank_and_select(reranker, "q", candidates, final_k=5)
    assert len(outcome["evidence"]) == 5
    scored_ids = [e["id"] for e in outcome["evidence"] if e["relevance"] is not None]
    assert scored_ids == ["c0", "c1"], "only scored candidates are ranked by relevance"
    assert outcome["used_reranking"] is True


def test_non_numeric_relevance_discarded():
    candidates = [_candidate("c0", 0.9), _candidate("c1", 0.8)]
    reranker = _reranker(
        json.dumps(
            {
                "results": [
                    {"id": "c0", "relevance": "very relevant"},
                    {"id": "c1", "relevance": True},
                ]
            }
        )
    )
    outcome = rerank_and_select(reranker, "q", candidates, final_k=2)
    assert outcome["fallback"] is True
    assert any("not numeric" in e for e in outcome["errors"])


# ------------------------------------------------------------------ empty / small corpus

def test_empty_corpus_no_crash():
    outcome = rerank_and_select(None, "q", [], final_k=5)
    assert outcome["evidence"] == []
    assert outcome["used_reranking"] is False
    assert outcome["fallback"] is False
    assert "No candidates" in outcome["reason"]


def test_retrieve_candidates_fewer_than_candidate_k_no_crash(store):
    store.add_text_source("Short", SHORT_TEXT, "text")
    retrieval = store.retrieve_candidates("chromium storage", candidate_k=12)
    assert 1 <= len(retrieval["candidates"]) < 12

    outcome = rerank_and_select(_ScoreAllReranker(), "chromium storage", retrieval["candidates"], final_k=5)
    assert len(outcome["evidence"]) == len(retrieval["candidates"])
    assert all(e["relevance"] is not None for e in outcome["evidence"])
    assert outcome["used_reranking"] is True


def test_retrieve_candidates_returns_chunk_level(store):
    store.add_text_source("Alpha", LONG_TEXT, "text")
    store.add_text_source("Beta", LONG_TEXT, "text")
    retrieval = store.retrieve_candidates("querystellar question", candidate_k=4)

    assert retrieval["query_point"]["modality"] == "query"
    assert retrieval["candidates"], "candidates expected after ingestion"
    assert len(retrieval["candidates"]) <= 4
    required = {"id", "source_id", "title", "modality", "text", "similarity", "metadata"}
    for candidate in retrieval["candidates"]:
        assert required.issubset(candidate.keys())
        assert "::" in candidate["id"], "candidate ids are chunk-level"
        assert isinstance(candidate["similarity"], (int, float))
    assert "space" in retrieval and "query_point" in retrieval


def test_full_pipeline_via_store_and_reranker(store):
    store.add_text_source("Alpha", LONG_TEXT, "text")
    store.add_text_source("Beta", LONG_TEXT, "text")
    retrieval = store.retrieve_candidates("querystellar", candidate_k=4)
    score_reranker = _ScoreAllReranker(relevance=0.88)
    outcome = rerank_and_select(score_reranker, "querystellar", retrieval["candidates"], final_k=3)

    assert outcome["used_reranking"] is True
    assert len(outcome["evidence"]) == 3
    for item in outcome["evidence"]:
        assert item["similarity"] is not None
        assert item["relevance"] == 0.88
        assert item["reason"] == "scored"
    assert outcome["evidence"] == sorted(
        outcome["evidence"], key=lambda e: (-e["relevance"], -e["similarity"], e["id"])
    )


# ------------------------------------------------------------------ reranker availability

def test_reranker_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    reranker = GeminiReranker()
    assert reranker.client is None
    assert reranker.available is False


def test_reranker_available_with_fake_client():
    assert GeminiReranker(client=object()).available is True


# ------------------------------------------------------------------ /ask wiring

def _canned_retrieval(candidates: list[dict]) -> dict:
    return {
        "query": "q",
        "query_point": {
            "id": "query-abc123",
            "source_id": "query",
            "title": "q",
            "modality": "query",
            "projection": {"x": 0.0, "y": 0.0, "z": 0.0},
            "color": "#f54e00",
            "score": 1,
            "preview": "",
        },
        "candidates": candidates,
        "space": {
            "sources": [],
            "points": [
                {"source_id": "src1", "projection": {"x": 1.0, "y": 2.0, "z": 3.0}},
                {"source_id": "src2", "projection": {"x": 4.0, "y": 5.0, "z": 6.0}},
            ],
            "events": [],
            "provider": "gemini-embedding-2",
            "dimensions": 768,
            "model": "test",
            "projection": {},
        },
    }


@pytest.fixture()
def server_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import server

    return server


class _StandardRouter:
    def classify(self, query):
        return {
            "strategy": "STANDARD",
            "reason": "simple factual question",
            "subqueries": [],
            "fell_back": False,
            "error": "",
        }


def _patch_router_and_adk(server_module, monkeypatch):
    monkeypatch.setattr(server_module, "ROUTER", _StandardRouter())

    async def _fake_adk(question, payload, insufficient_evidence=False):
        return "canned answer"

    monkeypatch.setattr(server_module, "_run_adk_agent", _fake_adk)


def test_ask_reranked_response_shape(server_module, monkeypatch):
    from fastapi.testclient import TestClient

    _patch_router_and_adk(server_module, monkeypatch)
    candidates = [
        _candidate("src1::1", 0.7, source_id="src1", text="alpha one"),
        _candidate("src1::2", 0.6, source_id="src1", text="alpha two"),
        _candidate("src2::1", 0.5, source_id="src2", text="beta one"),
    ]
    monkeypatch.setattr(server_module.RAG_STORE, "retrieve_candidates", lambda query, candidate_k: _canned_retrieval(candidates))
    monkeypatch.setattr(server_module.RAG_STORE, "embedding_provider", "gemini-embedding-2")

    class FakeReranker:
        available = True
        timeout = 1.0

        def score_candidates(self, query, candidates):
            return {
                "scores": {c["id"]: {"relevance": 0.9, "reason": "matches"} for c in candidates},
                "errors": [],
            }

    monkeypatch.setattr(server_module, "RERANKER", FakeReranker())

    res = TestClient(server_module.app).post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["reranked"] is True
    assert data["answer"] == "canned answer"
    assert data["strategy"] == "STANDARD"
    assert data["router_fell_back"] is False
    assert data["subqueries"] == []
    assert data["insufficient_evidence"] is False
    assert len(data["matches"]) == 2
    for match in data["matches"]:
        for key in ("id", "source_id", "title", "modality", "text", "score", "similarity", "relevance", "reason", "projection", "metadata"):
            assert key in match
        assert match["score"] == match["similarity"]
        assert match["relevance"] == 0.9
        assert match["reason"] == "matches"
    assert data["matches"][0]["id"] == "src1::1", "top final_k selected by relevance then similarity"
    assert data["query_point"]["modality"] == "query"
    assert "space" in data and "trace" in data


def test_ask_fallback_response_shape(server_module, monkeypatch):
    from fastapi.testclient import TestClient

    _patch_router_and_adk(server_module, monkeypatch)
    candidates = [
        _candidate("src1::1", 0.7, source_id="src1", text="alpha one"),
        _candidate("src1::2", 0.6, source_id="src1", text="alpha two"),
        _candidate("src2::1", 0.5, source_id="src2", text="beta one"),
    ]
    monkeypatch.setattr(server_module.RAG_STORE, "retrieve_candidates", lambda query, candidate_k: _canned_retrieval(candidates))
    monkeypatch.setattr(server_module.RAG_STORE, "embedding_provider", "gemini-embedding-2")
    monkeypatch.setattr(server_module, "RERANKER", _UnavailableReranker())

    res = TestClient(server_module.app).post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["reranked"] is False
    assert data["answer"] == "canned answer", "generation must still proceed on fallback"
    assert len(data["matches"]) == 2
    assert [m["id"] for m in data["matches"]] == ["src1::1", "src1::2"], "fallback ranked by similarity"
    assert all(m["relevance"] is None for m in data["matches"])
    assert all(m["reason"] is None for m in data["matches"])
    assert any(t["agent"] == "reranker" and t["status"] == "fallback" for t in data["trace"])