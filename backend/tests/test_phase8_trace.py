"""Phase 8 tests: query trace payload + read-only evaluation summary endpoint."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from server import app


def _canned_space():
    return {
        "sources": [],
        "points": [
            {
                "source_id": "s1",
                "projection": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        ],
        "events": [],
        "provider": "test",
        "dimensions": 8,
        "model": "test",
    }


def _canned_query_point():
    return {
        "id": "query-abc",
        "source_id": "query",
        "title": "q",
        "modality": "query",
        "projection": {"x": 0.0, "y": 0.0, "z": 0.0},
        "color": "#f54e00",
        "score": 1,
        "preview": "",
    }


def _candidates():
    return [
        {
            "id": "s1::1",
            "source_id": "s1",
            "title": "Source One",
            "modality": "text",
            "text": "chunk one text",
            "similarity": 0.9,
            "metadata": {},
        },
        {
            "id": "s2::1",
            "source_id": "s2",
            "title": "Source Two",
            "modality": "text",
            "text": "chunk two text",
            "similarity": 0.7,
            "metadata": {},
        },
    ]


class _FakeStore:
    sources = ["s1"]
    chunks = ["c1", "c2"]
    dimensions = 8
    embedding_provider = "test-provider"

    def retrieve_candidates(self, question, candidate_k):
        return {
            "candidates": _candidates(),
            "space": _canned_space(),
            "query_point": _canned_query_point(),
        }


class _FakeRouter:
    def classify(self, question):
        return {
            "strategy": "STANDARD",
            "reason": "Simple single-fact question.",
            "subqueries": [],
            "fell_back": False,
            "error": "",
        }


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "RAG_STORE", _FakeStore())
    monkeypatch.setattr(server, "ROUTER", _FakeRouter())
    monkeypatch.setattr(server, "RERANKER", object())
    monkeypatch.setattr(server, "VERIFIER", object())

    async def _fake_agent(question, retrieval, insufficient_evidence=False):
        return " grounded answer text"

    monkeypatch.setattr(server, "_run_adk_agent", _fake_agent)
    return TestClient(app)


def _patch_rerank_verify(monkeypatch, *, rerank_ok=True, verify_ok=True):
    if rerank_ok:
        def _rerank(reranker, question, candidates, final_k):
            return {
                "evidence": [
                    {**candidates[0], "relevance": 0.95, "reason": "Directly answers."},
                    {**candidates[1], "relevance": 0.4, "reason": "Partially relevant."},
                ][:final_k],
                "used_reranking": True,
                "fallback": False,
                "reason": "",
                "errors": [],
            }
    else:
        def _rerank(reranker, question, candidates, final_k):
            return {
                "evidence": [
                    {**candidates[0], "relevance": None, "reason": None},
                    {**candidates[1], "relevance": None, "reason": None},
                ][:final_k],
                "used_reranking": False,
                "fallback": True,
                "reason": "Reranker unavailable (test); using similarity ranking.",
                "errors": [],
            }

    if verify_ok:
        def _verify(verifier, question, answer, evidence, citations):
            return {
                "unavailable": False,
                "reason": "",
                "grounded": True,
                "score": 1.0,
                "claims": [
                    {"claim": "Chunk one says X.", "supported": True, "evidence_ids": ["s1::1"]}
                ],
                "unsupported_claims": [],
                "errors": [],
            }
    else:
        def _verify(verifier, question, answer, evidence, citations):
            return {
                "unavailable": True,
                "reason": "Verifier unavailable (test); verification not run.",
                "grounded": None,
                "score": None,
                "claims": [],
                "unsupported_claims": [],
                "errors": [],
            }

    monkeypatch.setattr(server, "rerank_and_select", _rerank)
    monkeypatch.setattr(server, "run_verification", _verify)


def test_ask_trace_fields_present(client, monkeypatch):
    _patch_rerank_verify(monkeypatch, rerank_ok=True, verify_ok=True)
    res = client.post("/ask", json={"question": "What is X?", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    # Pre-existing fields still present (additive change only).
    for field in ("answer", "matches", "reranked", "strategy", "subqueries",
                  "router_fell_back", "trace", "space", "query_point"):
        assert field in data, f"missing pre-existing field {field}"

    # New trace fields.
    assert data["query"] == "What is X?"
    assert data["candidate_count"] == 2
    assert isinstance(data["candidates"], list) and len(data["candidates"]) == 2
    assert data["candidates"][0]["similarity"] == 0.9
    for stage in ("retrieval_latency", "reranking_latency", "generation_latency",
                  "verification_latency", "total_latency"):
        assert stage in data and isinstance(data[stage], (int, float)), stage
    assert isinstance(data["latencies"], dict)
    assert data["latencies"]["total_latency"] == data["total_latency"]

    # Scores on final evidence.
    assert data["matches"][0]["similarity"] == 0.9
    assert data["matches"][0]["relevance"] == 0.95
    assert data["rerank_fallback"] is False

    # Verification surfaced.
    assert data["grounded"] is True
    assert data["score"] == 1.0
    assert data["unsupported_claims"] == []


def test_ask_trace_degrades_on_fallbacks(client, monkeypatch):
    _patch_rerank_verify(monkeypatch, rerank_ok=False, verify_ok=False)
    res = client.post("/ask", json={"question": "What is X?", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["rerank_fallback"] is True
    assert "similarity" in data["rerank_reason"].lower() or "unavailable" in data["rerank_reason"].lower()
    assert data["matches"][0]["relevance"] is None
    assert "verification_unavailable" in data
    assert "grounded" not in data  # unavailable path must not fabricate a verdict
    for stage in ("retrieval_latency", "reranking_latency", "generation_latency",
                  "verification_latency", "total_latency"):
        assert stage in data


def test_evaluation_summary_with_results(client):
    res = client.get("/evaluation/summary")
    assert res.status_code == 200
    data = res.json()
    assert set(data.keys()) == {"benchmark", "latest_run", "latest_comparison", "notes"}
    # Real repo files exist: benchmark + at least one run + one comparison.
    assert data["benchmark"] is not None
    assert data["benchmark"]["question_count"] == 25
    assert data["latest_run"] is not None
    assert isinstance(data["latest_run"]["retrieval"], dict)
    assert data["latest_comparison"] is not None
    assert "recall_at_5" in (data["latest_comparison"]["differences_improved_minus_baseline"] or {})


def test_evaluation_summary_empty_state(client, monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(server, "EVAL_RESULTS_DIR", empty)
    monkeypatch.setattr(server, "BENCHMARK_PATH", empty / "benchmark_v1.json")
    res = client.get("/evaluation/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["benchmark"] is None
    assert data["latest_run"] is None
    assert data["latest_comparison"] is None
    assert len(data["notes"]) >= 3  # clear empty state, not a blank response


def test_evaluation_summary_malformed_file(client, monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "run_20250101T000000.json").write_text("{not valid json", encoding="utf-8")
    (results / "comparison_20250101T000000.json").write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(server, "EVAL_RESULTS_DIR", results)
    monkeypatch.setattr(server, "BENCHMARK_PATH", results / "missing.json")
    res = client.get("/evaluation/summary")
    assert res.status_code == 200  # defensive: never errors on malformed data
    data = res.json()
    assert data["latest_run"] is None
    assert data["latest_comparison"] is None
    assert any("malformed" in note for note in data["notes"])


def test_frontend_fallback_states_present():
    """Static check: new UI sections render explicit fallback states."""
    text = (Path(__file__).resolve().parents[2] / "frontend" / "src" / "App.tsx").read_text(
        encoding="utf-8"
    )
    assert "/evaluation/summary" in text
    assert "No evaluation results yet" in text
    assert "verification unavailable" in text.lower() or "Verification did not run" in text
    assert "fallback" in text.lower()
    assert "VectorSpace" in text  # 3D visualization still present
    assert "makeGlowTexture" in text  # Three.js code untouched
