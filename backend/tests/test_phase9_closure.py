"""Phase 9 closure tests: end-to-end /ask + ingestion-API coverage.

Earlier phases covered each fallback in isolation (store-level, reranker-level,
router-level, verifier-level, metrics-level). These tests close the remaining
gaps at the HTTP API level, using a fresh tmp Chroma store with deterministic
mock embeddings so they run with no API key and no network:

  - normal retrieval: ingest via POST /sources/text, ask a supported question
  - empty database: /ask with nothing ingested -> insufficient evidence
  - irrelevant query: /ask with no relevant content -> insufficient evidence
  - source deletion reflected in /ask: delete then re-ask -> insufficient
  - duplicate ingestion via API: POST the same source twice -> no duplicates
"""

import hashlib
import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from rag_store import MultimodalRagStore

DIM = 768
SUPPORTED_TEXT = "Paris is the capital of France. The Seine river flows through Paris."
COOKING_TEXT = "Baking sourdough bread requires flour, water, salt, and a live starter culture."
IRRELEVANT_Q = "What is the quantum tunneling rate in Josephson junctions?"


def _deterministic_vector(text: str, dim: int = DIM) -> list[float]:
    state = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    vector = []
    for _ in range(dim):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        vector.append((state / 0xFFFFFFFF) - 0.5)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class _StandardRouter:
    def classify(self, query):
        return {
            "strategy": "STANDARD",
            "reason": "Simple single-fact question.",
            "subqueries": [],
            "fell_back": False,
            "error": "",
        }


class _ScoreAllReranker:
    available = True
    timeout = 1.0

    def __init__(self, relevance: float = 0.9) -> None:
        self.relevance = relevance

    def score_candidates(self, query, candidates):
        return {
            "scores": {c["id"]: {"relevance": self.relevance, "reason": "matches"} for c in candidates},
            "errors": [],
        }


def _verification_ok(verifier, question, answer, evidence, citations=None):
    if not evidence:
        return {
            "unavailable": True,
            "reason": "Verification skipped: no evidence was available to ground or verify this answer.",
            "grounded": None,
            "score": None,
            "claims": [],
            "unsupported_claims": [],
            "errors": [],
        }
    return {
        "unavailable": False,
        "reason": "",
        "grounded": True,
        "score": 1.0,
        "claims": [{"claim": "c", "supported": True, "evidence_ids": [evidence[0]["id"]]}] if evidence else [],
        "unsupported_claims": [],
        "errors": [],
    }


async def _fake_adk(question, retrieval, insufficient_evidence=False):
    if insufficient_evidence:
        return "The available evidence is insufficient to answer this question."
    return "Paris is the capital of France."


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    store = MultimodalRagStore()
    store._embed_text = lambda text, task_prefix: _deterministic_vector(f"{task_prefix}::{text}")
    monkeypatch.setattr(server, "RAG_STORE", store)
    monkeypatch.setattr(server, "ROUTER", _StandardRouter())
    monkeypatch.setattr(server, "RERANKER", _ScoreAllReranker(relevance=0.9))
    monkeypatch.setattr(server, "_run_adk_agent", _fake_adk)
    monkeypatch.setattr(server, "run_verification", _verification_ok)
    return TestClient(server.app), store


def _ingest(client, title, text):
    res = client.post("/sources/text", json={"title": title, "text": text, "modality": "text"})
    assert res.status_code == 200, res.text
    return res.json()


def test_closure_normal_retrieval_supported_answer(api):
    client, _ = api
    _ingest(client, "France", SUPPORTED_TEXT)

    res = client.post("/ask", json={"question": SUPPORTED_TEXT, "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "Paris is the capital of France."
    assert data["insufficient_evidence"] is False
    assert data["matches"], "a supported query must return evidence"
    assert data["reranked"] is True
    assert data["strategy"] == "STANDARD"
    assert data["grounded"] is True
    agents = [t["agent"] for t in data["trace"]]
    for expected in ("query_router", "retrieval_tool", "reranker", "answer_synthesizer", "grounding_verifier"):
        assert expected in agents


def test_closure_empty_database_ask(api):
    client, _ = api
    res = client.post("/ask", json={"question": "Anything at all?", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["matches"] == []
    assert data["candidates"] == []
    assert data["candidate_count"] == 0
    assert data["insufficient_evidence"] is True
    assert "insufficient" in data["answer"].lower()
    assert "verification_unavailable" in data


def test_closure_irrelevant_query(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(server, "RERANKER", _ScoreAllReranker(relevance=0.0))
    _ingest(client, "Cooking", COOKING_TEXT)

    res = client.post("/ask", json={"question": IRRELEVANT_Q, "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["insufficient_evidence"] is True
    assert "insufficient" in data["answer"].lower()


def test_closure_source_deletion_reflected_in_ask(api):
    client, _ = api
    created = _ingest(client, "France", SUPPORTED_TEXT)
    source_id = created["source"]["id"]

    before = client.post("/ask", json={"question": SUPPORTED_TEXT, "top_k": 2})
    assert before.json()["matches"], "evidence present before deletion"

    deleted = client.delete(f"/sources/{source_id}")
    assert deleted.status_code == 200

    after = client.post("/ask", json={"question": SUPPORTED_TEXT, "top_k": 2})
    assert after.status_code == 200
    data = after.json()
    assert data["matches"] == []
    assert data["insufficient_evidence"] is True
    assert "insufficient" in data["answer"].lower()


def test_closure_duplicate_ingestion_via_api_creates_no_duplicates(api):
    client, store = api
    first = _ingest(client, "Dedupe", SUPPORTED_TEXT)
    count = store._collection.count()

    second = _ingest(client, "Dedupe", SUPPORTED_TEXT)
    assert second["source"]["id"] == first["source"]["id"]
    assert store._collection.count() == count
    assert len(client.get("/space").json()["sources"]) == 1
