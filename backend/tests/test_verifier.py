import json
import time

import pytest

from verifier import (
    GeminiVerifier,
    run_verification,
    validate_verifier_output,
)


# ------------------------------------------------------------------ helpers

def _evidence(cid: str, text: str = "evidence text", relevance: float = 0.9, similarity: float = 0.85) -> dict:
    return {
        "id": cid,
        "source_id": cid.split("::")[0],
        "title": "Source",
        "modality": "text",
        "text": text,
        "similarity": similarity,
        "metadata": {},
        "relevance": relevance,
        "reason": "scored",
    }


def _verifier(text: str, timeout: float = 5.0) -> GeminiVerifier:
    instance = GeminiVerifier(client=object(), timeout=timeout)
    instance._client_generate = lambda prompt: text
    return instance


def _payload(claims, grounded=True, score=1.0, unsupported=None) -> str:
    return json.dumps(
        {
            "grounded": grounded,
            "score": score,
            "claims": claims,
            "unsupported_claims": unsupported if unsupported is not None else [],
        }
    )


# ------------------------------------------------------------------ validate_verifier_output unit tests

def test_validate_output_not_an_object():
    cleaned, notes = validate_verifier_output([1, 2], {"src1::1"})
    assert cleaned is None
    assert notes


def test_validate_output_grounded_must_be_boolean():
    for bad in ("yes", 1, None):
        cleaned, notes = validate_verifier_output({"grounded": bad, "score": 1.0, "claims": []}, set())
        assert cleaned is None


def test_validate_output_score_must_be_numeric():
    for bad in ("high", True, None, [1.0]):
        cleaned, notes = validate_verifier_output(
            {"grounded": True, "score": bad, "claims": []}, set()
        )
        assert cleaned is None


def test_validate_output_claims_must_be_list():
    cleaned, notes = validate_verifier_output(
        {"grounded": True, "score": 1.0, "claims": "nope"}, set()
    )
    assert cleaned is None


def test_validate_output_clamps_out_of_range_score():
    cleaned, notes = validate_verifier_output(
        {
            "grounded": True,
            "score": 5.0,
            "claims": [{"claim": "c", "supported": True, "evidence_ids": ["src1::1"]}],
        },
        {"src1::1"},
    )
    assert cleaned is not None
    assert cleaned["score"] == 1.0
    assert any("Clamped" in note for note in notes)


def test_validate_output_discards_unknown_evidence_ids():
    cleaned, notes = validate_verifier_output(
        {
            "grounded": True,
            "score": 1.0,
            "claims": [{"claim": "c", "supported": True, "evidence_ids": ["ghost", "src1::1"]}],
        },
        {"src1::1"},
    )
    assert cleaned["claims"][0]["evidence_ids"] == ["src1::1"]
    assert any("ghost" in note for note in notes)


def test_validate_output_demotes_supported_claim_with_no_valid_evidence():
    cleaned, notes = validate_verifier_output(
        {
            "grounded": True,
            "score": 1.0,
            "claims": [{"claim": "c", "supported": True, "evidence_ids": ["ghost"]}],
        },
        {"src1::1"},
    )
    claim = cleaned["claims"][0]
    assert claim["supported"] is False
    assert claim["evidence_ids"] == []
    assert cleaned["unsupported_claims"] == ["c"]
    assert cleaned["grounded"] is False
    assert any("Demoted" in note for note in notes)


def test_validate_output_drops_malformed_claims_but_keeps_valid_ones():
    cleaned, notes = validate_verifier_output(
        {
            "grounded": True,
            "score": 1.0,
            "claims": [
                {"claim": "", "supported": True, "evidence_ids": []},
                {"claim": "good", "supported": True, "evidence_ids": ["src1::1"]},
                {"claim": "bad", "supported": "maybe", "evidence_ids": []},
                "not a dict",
            ],
        },
        {"src1::1"},
    )
    assert [c["claim"] for c in cleaned["claims"]] == ["good"]
    assert cleaned["grounded"] is True
    assert cleaned["score"] == 1.0
    assert len(notes) == 3


def test_validate_output_all_claims_malformed_is_unusable():
    cleaned, notes = validate_verifier_output(
        {
            "grounded": True,
            "score": 1.0,
            "claims": [{"claim": "", "supported": True, "evidence_ids": []}, {"right_field_here": 1}],
        },
        set(),
    )
    assert cleaned is None


def test_validate_output_empty_claims_vacuously_grounded():
    cleaned, notes = validate_verifier_output(
        {"grounded": True, "score": 1.0, "claims": []}, set()
    )
    assert cleaned["grounded"] is True
    assert cleaned["score"] == 1.0
    assert cleaned["unsupported_claims"] == []


# ------------------------------------------------------------------ normal paths

def test_normal_verification_grounded():
    evidence = [
        _evidence("src1::1", text="Paris is the capital of France."),
        _evidence("src2::1", text="The Seine flows through Paris."),
    ]
    verifier = _verifier(
        _payload(
            [
                {"claim": "Paris is the capital of France.", "supported": True, "evidence_ids": ["src1::1"]},
                {"claim": "The Seine flows through Paris.", "supported": True, "evidence_ids": ["src2::1"]},
            ]
        )
    )

    result = verifier.verify("What is the capital of France?", "Paris is the capital of France.", evidence)

    assert result["unavailable"] is False
    assert result["grounded"] is True
    assert result["score"] == 1.0
    assert result["claims"] == [
        {"claim": "Paris is the capital of France.", "supported": True, "evidence_ids": ["src1::1"]},
        {"claim": "The Seine flows through Paris.", "supported": True, "evidence_ids": ["src2::1"]},
    ]
    assert result["unsupported_claims"] == []


def test_unsupported_claim_surfaces_and_lowers_score():
    evidence = [_evidence("src1::1", text="Paris is the capital of France.")]
    verifier = _verifier(
        _payload(
            [
                {"claim": "Paris is the capital of France.", "supported": True, "evidence_ids": ["src1::1"]},
                {"claim": "Paris has 12 million residents.", "supported": False, "evidence_ids": []},
            ],
            grounded=False,
            score=0.5,
        )
    )

    result = verifier.verify("Tell me about Paris.", "Paris is the capital; it has 12 million residents.", evidence)

    assert result["unavailable"] is False
    assert result["grounded"] is False
    assert result["score"] == 0.5
    assert result["unsupported_claims"] == ["Paris has 12 million residents."]


def test_score_is_proportion_of_supported_claims():
    evidence = [_evidence("src1::1"), _evidence("src2::1")]
    one_of_three = _verifier(
        _payload(
            [
                {"claim": "a", "supported": True, "evidence_ids": ["src1::1"]},
                {"claim": "b", "supported": False, "evidence_ids": []},
                {"claim": "c", "supported": False, "evidence_ids": []},
            ],
            grounded=False,
            score=0.33,
        )
    )
    result = one_of_three.verify("q", "answer", evidence)
    assert result["score"] == round(1 / 3, 4)
    assert result["grounded"] is False
    assert result["unsupported_claims"] == ["b", "c"]


def test_no_claims_is_vacuously_grounded():
    verifier = _verifier(_payload([], grounded=True, score=1.0))
    result = verifier.verify("q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is False
    assert result["grounded"] is True
    assert result["score"] == 1.0
    assert result["claims"] == []


def test_unsupported_claims_recomputed_from_claims_list():
    verifier = _verifier(
        _payload(
            [
                {"claim": "Good one.", "supported": True, "evidence_ids": ["src1::1"]},
                {"claim": "Bad one.", "supported": False, "evidence_ids": []},
            ],
            grounded=False,
            score=0.5,
            unsupported=["Good one."],
        )
    )
    result = verifier.verify("q", "answer", [_evidence("src1::1")])
    assert result["unsupported_claims"] == ["Bad one."]
    assert any("recomputed" in error for error in result["errors"])


# ------------------------------------------------------------------ failure: call / timeout / output

def test_verification_call_failure_is_unavailable():
    broken = _verifier("")
    broken._client_generate = lambda prompt: (_ for _ in ()).throw(RuntimeError("network exploded"))

    result = broken.verify("q", "answer", [_evidence("src1::1")])

    assert result["unavailable"] is True
    assert "network exploded" in result["reason"]
    assert result["grounded"] is None
    assert result["score"] is None
    assert result["claims"] == []


def test_verification_timeout_is_unavailable():
    slow = GeminiVerifier(client=object(), timeout=0.05)
    slow._client_generate = lambda prompt: time.sleep(2.0) or ""

    started = time.monotonic()
    result = slow.verify("q", "answer", [_evidence("src1::1")])
    elapsed = time.monotonic() - started

    assert result["unavailable"] is True
    assert "timed out" in result["reason"]
    assert elapsed < 1.5, "timeout must cut the verification call short"


def test_verification_malformed_json_is_unavailable():
    result = _verifier("this is {{ not json").verify("q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is True
    assert "not valid JSON" in result["reason"]


def test_verification_empty_response_is_unavailable():
    result = _verifier("   ").verify("q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is True
    assert "empty" in result["reason"]


def test_verification_schema_mismatch_is_unavailable():
    verifier = _verifier(_payload([{"claim": "x", "supported": True, "evidence_ids": ["src1::1"]}], grounded="yes"))
    result = verifier.verify("q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is True
    assert "schema" in result["reason"]


def test_non_numeric_score_is_unavailable():
    verifier = _verifier(
        json.dumps(
            {
                "grounded": True,
                "score": "high",
                "claims": [{"claim": "c", "supported": True, "evidence_ids": ["src1::1"]}],
            }
        )
    )
    result = verifier.verify("q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is True


def test_out_of_range_score_clamped_and_not_trusted():
    verifier = _verifier(
        _payload(
            [{"claim": "Supported claim.", "supported": True, "evidence_ids": ["src1::1"]}],
            grounded=True,
            score=5.0,
        )
    )
    result = verifier.verify("q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is False
    assert result["score"] == 1.0, "derived score replaces the raw out-of-range value"
    assert any("Clamped" in error for error in result["errors"])


def test_negative_out_of_range_score_clamped():
    verifier = _verifier(
        _payload(
            [{"claim": "Unsupported claim.", "supported": False, "evidence_ids": []}],
            grounded=False,
            score=-2.0,
        )
    )
    result = verifier.verify("q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is False
    assert result["score"] == 0.0
    assert any("Clamped" in error for error in result["errors"])


# ------------------------------------------------------------------ evidence membership rules

def test_unknown_evidence_id_discarded_not_trusted():
    evidence = [_evidence("src1::1")]
    verifier = _verifier(
        _payload(
            [
                {"claim": "Real chunk claim.", "supported": True, "evidence_ids": ["src1::1"]},
                {"claim": "Ghost chunk claim.", "supported": True, "evidence_ids": ["ghost-id"]},
            ]
        )
    )
    result = verifier.verify("q", "answer", evidence)

    by_claim = {c["claim"]: c for c in result["claims"]}
    assert by_claim["Real chunk claim."]["evidence_ids"] == ["src1::1"]
    ghost = by_claim["Ghost chunk claim."]
    assert ghost["evidence_ids"] == [], "unknown id discarded"
    assert ghost["supported"] is False, "support with no real evidence is not trusted"
    assert "Ghost chunk claim." in result["unsupported_claims"]
    assert any("ghost-id" in error for error in result["errors"])
    assert result["grounded"] is False
    assert result["score"] == 0.5


def test_fabricated_citation_detected():
    evidence = [_evidence("src1::1")]
    verifier = _verifier(
        _payload([{"claim": "Full fabrication.", "supported": True, "evidence_ids": ["made-up::9"]}])
    )
    result = verifier.verify("q", "answer", evidence)
    assert result["unavailable"] is False
    assert result["claims"][0]["evidence_ids"] == []
    assert result["unsupported_claims"] == ["Full fabrication."]


def test_citation_warning_when_citations_reference_missing_ids():
    verifier = _verifier(_payload([]))
    result = verifier.verify("q", "answer", [_evidence("src1::1")], citations=["does-not-exist::1"])
    assert result["unavailable"] is False
    assert any("does-not-exist::1" in error for error in result["errors"])


# ------------------------------------------------------------------ empty evidence / availability

def test_empty_evidence_skipped_gracefully():
    verifier = _verifier("should never be called")
    result = verifier.verify("q", "answer", [])
    assert result["unavailable"] is True
    assert "no evidence" in result["reason"]
    assert result["claims"] == []


def test_empty_answer_skipped_gracefully():
    result = _verifier("x").verify("q", "   ", [_evidence("src1::1")])
    assert result["unavailable"] is True
    assert "empty" in result["reason"].lower()


def test_verifier_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = GeminiVerifier(client=None).verify("q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is True
    assert "unavailable" in result["reason"]


def test_run_verification_none_verifier():
    result = run_verification(None, "q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is True
    assert "not configured" in result["reason"]


class _ExplodingVerifier:
    def verify(self, *args, **kwargs):
        raise RuntimeError("backend exploded")


def test_run_verification_wraps_verifier_exceptions():
    result = run_verification(_ExplodingVerifier(), "q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is True
    assert "backend exploded" in result["reason"]


def test_run_verification_delegates_on_success():
    verifier = _verifier(
        _payload([{"claim": "c", "supported": True, "evidence_ids": ["src1::1"]}])
    )
    result = run_verification(verifier, "q", "answer", [_evidence("src1::1")])
    assert result["unavailable"] is False
    assert result["grounded"] is True


# ------------------------------------------------------------------ /ask wiring

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


def _canned_retrieval(candidates: list[dict]) -> dict:
    points = []
    seen = set()
    for candidate in candidates:
        source_id = candidate["source_id"]
        if source_id in seen:
            continue
        seen.add(source_id)
        points.append({"source_id": source_id, "projection": {"x": 0.0, "y": 0.0, "z": 0.0}})
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
            "points": points,
            "events": [],
            "provider": "gemini-embedding-2",
            "dimensions": 768,
            "model": "test",
            "projection": {},
        },
    }


class _StandardRouter:
    def classify(self, query):
        return {
            "strategy": "STANDARD",
            "reason": "simple factual question",
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


@pytest.fixture()
def server_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import server

    return server


def _patch_pipeline(server_module, monkeypatch, candidates, relevance=0.9):
    monkeypatch.setattr(server_module, "ROUTER", _StandardRouter())
    monkeypatch.setattr(server_module, "RERANKER", _ScoreAllReranker(relevance=relevance))
    monkeypatch.setattr(
        server_module.RAG_STORE,
        "retrieve_candidates",
        lambda query, candidate_k: _canned_retrieval(candidates),
    )
    monkeypatch.setattr(server_module.RAG_STORE, "embedding_provider", "gemini-embedding-2")

    async def _fake_adk(question, payload, insufficient_evidence=False):
        if insufficient_evidence:
            return "The available evidence is insufficient to answer this question."
        return "canned answer"

    monkeypatch.setattr(server_module, "_run_adk_agent", _fake_adk)


def test_ask_includes_verification_fields(server_module, monkeypatch):
    from fastapi.testclient import TestClient

    candidates = [_candidate("src1::1", 0.7, source_id="src1", text="alpha one")]
    _patch_pipeline(server_module, monkeypatch, candidates)

    def fake_verification(verifier, question, answer, evidence, citations=None):
        return {
            "unavailable": False,
            "reason": "",
            "grounded": True,
            "score": 0.8,
            "claims": [
                {"claim": "alpha one", "supported": True, "evidence_ids": ["src1::1"]},
                {"claim": "related fact", "supported": False, "evidence_ids": []},
            ],
            "unsupported_claims": ["related fact"],
            "errors": [],
        }

    monkeypatch.setattr(server_module, "run_verification", fake_verification)

    res = TestClient(server_module.app).post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["answer"] == "canned answer"
    assert data["grounded"] is True
    assert data["score"] == 0.8
    assert data["claims"][0]["evidence_ids"] == ["src1::1"]
    assert data["unsupported_claims"] == ["related fact"]
    assert "verification_unavailable" not in data

    assert data["strategy"] == "STANDARD"
    assert data["reranked"] is True
    assert data["insufficient_evidence"] is False
    assert data["matches"][0]["id"] == "src1::1"
    assert data["trace"][-1]["agent"] == "grounding_verifier"
    assert data["trace"][-1]["status"] == "complete"


def test_ask_verifier_failure_still_returns_answer(server_module, monkeypatch):
    from fastapi.testclient import TestClient

    candidates = [_candidate("src1::1", 0.7, source_id="src1", text="alpha one")]
    _patch_pipeline(server_module, monkeypatch, candidates)
    monkeypatch.setattr(server_module, "VERIFIER", _ExplodingVerifier())

    res = TestClient(server_module.app).post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["answer"] == "canned answer", "original answer is still returned"
    assert data["verification_unavailable"] == "Verification unexpectedly failed (backend exploded)."
    assert "grounded" not in data
    assert "score" not in data
    assert "claims" not in data
    assert "unsupported_claims" not in data
    assert data["trace"][-1]["agent"] == "grounding_verifier"
    assert data["trace"][-1]["status"] == "unavailable"


def test_ask_verifier_not_configured_marks_unavailable(server_module, monkeypatch):
    from fastapi.testclient import TestClient

    candidates = [_candidate("src1::1", 0.7, source_id="src1", text="alpha one")]
    _patch_pipeline(server_module, monkeypatch, candidates)
    monkeypatch.setattr(server_module, "VERIFIER", None)

    res = TestClient(server_module.app).post("/ask", json={"question": "q", "top_k": 2})
    data = res.json()

    assert data["answer"] == "canned answer"
    assert "verification_unavailable" in data
    assert "not configured" in data["verification_unavailable"]
    assert "grounded" not in data


def test_ask_insufficient_evidence_marks_verification_unavailable(server_module, monkeypatch):
    from fastapi.testclient import TestClient

    candidates = [
        _candidate("src1::1", 0.03, source_id="src1", text="unrelated filler"),
        _candidate("src1::2", 0.04, source_id="src1", text="still unrelated"),
    ]
    _patch_pipeline(server_module, monkeypatch, candidates, relevance=0.0)

    res = TestClient(server_module.app).post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["insufficient_evidence"] is True
    assert data["answer"] == "The available evidence is insufficient to answer this question."
    assert data["verification_unavailable"] == (
        "Verification skipped: no evidence was available to ground or verify this answer."
    )
    assert "grounded" not in data


def test_ask_real_verifier_unavailable_without_key(server_module, monkeypatch):
    from fastapi.testclient import TestClient

    candidates = [_candidate("src1::1", 0.7, source_id="src1", text="alpha one")]
    _patch_pipeline(server_module, monkeypatch, candidates)
    assert server_module.ADK_AVAILABLE is False
    assert server_module.VERIFIER.available is False

    res = TestClient(server_module.app).post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["answer"] == "canned answer"
    assert "verification_unavailable" in data
    assert "unavailable" in data["verification_unavailable"]
    assert "grounded" not in data


def test_ask_does_not_disturb_phase2_and_phase3_fields(server_module, monkeypatch):
    from fastapi.testclient import TestClient

    candidates = [
        _candidate("src1::1", 0.9, source_id="src1", text="alpha one"),
        _candidate("src2::1", 0.8, source_id="src2", text="beta one"),
    ]
    _patch_pipeline(server_module, monkeypatch, candidates)
    monkeypatch.setattr(server_module, "VERIFIER", None)

    res = TestClient(server_module.app).post("/ask", json={"question": "q", "top_k": 2})
    data = res.json()

    for key in (
        "answer",
        "matches",
        "reranked",
        "strategy",
        "router_reason",
        "subqueries",
        "router_fell_back",
        "insufficient_evidence",
        "query_point",
        "trace",
        "space",
    ):
        assert key in data, f"existing field {key} must remain"
    for match in data["matches"]:
        for key in ("id", "source_id", "title", "modality", "text", "score", "similarity", "relevance", "reason", "projection", "metadata"):
            assert key in match
    assert data["query_point"]["modality"] == "query"