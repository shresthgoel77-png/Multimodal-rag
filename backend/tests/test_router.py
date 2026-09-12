import json
import time

import pytest

from query_router import (
    COMPLEX,
    MULTI_HOP,
    ROUTER_COMPLEX_CANDIDATE_K,
    ROUTER_MAX_SUBQUERIES,
    ROUTER_MULTIHOP_CANDIDATE_K,
    ROUTER_RELEVANCE_THRESHOLD,
    STANDARD,
    QueryRouter,
    evidence_is_sufficient,
    merge_candidates_keep_best,
    validate_router_output,
)
from reranker import RERANK_CANDIDATE_K


# ------------------------------------------------------------------ helpers

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


def _router(text: str, timeout: float = 5.0) -> QueryRouter:
    instance = QueryRouter(client=object(), timeout=timeout)
    instance._client_generate = lambda prompt: text
    return instance


def _router_result(strategy: str, reason: str = "because", subqueries=None, fell_back=False, error=""):
    return {
        "strategy": strategy,
        "reason": reason,
        "subqueries": subqueries if subqueries is not None else [],
        "fell_back": fell_back,
        "error": error,
    }


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


class _FakeRouter:
    def __init__(self, result: dict) -> None:
        self.result = result

    def classify(self, query: str) -> dict:
        return self.result


class _AdkRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, question, retrieval, insufficient_evidence=False):
        self.calls.append(
            {
                "question": question,
                "retrieval": retrieval,
                "insufficient_evidence": insufficient_evidence,
            }
        )
        if insufficient_evidence:
            return "The available evidence is insufficient to answer this question."
        return "canned answer"


@pytest.fixture()
def server_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import server

    return server


# ------------------------------------------------------------------ validation unit tests

def test_validate_standard():
    cleaned = validate_router_output({"strategy": "STANDARD", "reason": "simple", "subqueries": []})
    assert cleaned == {"strategy": "STANDARD", "reason": "simple", "subqueries": []}


def test_validate_standard_discards_subqueries():
    cleaned = validate_router_output(
        {"strategy": "STANDARD", "reason": "nope", "subqueries": ["sneaky subquery"]}
    )
    assert cleaned["strategy"] == "STANDARD"
    assert cleaned["subqueries"] == [], "subqueries with non-MULTI_HOP are discarded"


def test_validate_complex():
    cleaned = validate_router_output({"strategy": "COMPLEX", "reason": "broad", "subqueries": []})
    assert cleaned["strategy"] == "COMPLEX"


def test_validate_multihop_two_subqueries():
    cleaned = validate_router_output(
        {"strategy": "MULTI_HOP", "reason": "combine", "subqueries": ["one fact", "two fact"]}
    )
    assert cleaned["strategy"] == "MULTI_HOP"
    assert cleaned["subqueries"] == ["one fact", "two fact"]


def test_validate_multihop_three_subqueries():
    cleaned = validate_router_output(
        {"strategy": "MULTI_HOP", "reason": "combine", "subqueries": ["a", "b", "  c  "]}
    )
    assert cleaned["subqueries"] == ["a", "b", "c"], "whitespace trimmed"


def test_validate_invalid_strategy_rejected():
    for bad in ("standard", "STANDard", "DEEP_DIVE", "none", "standard strategy", "  STANDARD  "):
        assert validate_router_output({"strategy": bad, "reason": "x", "subqueries": []}) is None
    assert validate_router_output(["not", "a", "dict"]) is None


def test_validate_multihop_too_few_subqueries_rejected():
    assert validate_router_output({"strategy": "MULTI_HOP", "reason": "x", "subqueries": ["only one"]}) is None
    assert validate_router_output({"strategy": "MULTI_HOP", "reason": "x", "subqueries": ["", "  "]}) is None
    assert validate_router_output({"strategy": "MULTI_HOP", "reason": "x", "subqueries": []}) is None
    assert validate_router_output({"strategy": "MULTI_HOP", "reason": "x"}) is None


def test_validate_multihop_four_subqueries_truncated():
    cleaned = validate_router_output(
        {
            "strategy": "MULTI_HOP",
            "reason": "x",
            "subqueries": ["a", "b", "c", "d", "e"],
        }
    )
    assert cleaned["strategy"] == "MULTI_HOP"
    assert len(cleaned["subqueries"]) == ROUTER_MAX_SUBQUERIES == 3
    assert cleaned["subqueries"] == ["a", "b", "c"]


def test_validate_multihop_empty_strings_inside_list_rejected_or_normalized():
    assert validate_router_output(
        {"strategy": "MULTI_HOP", "reason": "x", "subqueries": ["a", ""]}
    ) is None
    assert validate_router_output(
        {"strategy": "MULTI_HOP", "reason": "x", "subqueries": ["a", "b", "", ""]}
    ) == {"strategy": "MULTI_HOP", "reason": "x", "subqueries": ["a", "b"]}


# ------------------------------------------------------------------ dedup / merge unit tests

def test_merge_dedups_by_id_keeping_highest_similarity():
    first = [_candidate("a::1", 0.9), _candidate("shared::1", 0.6, source_id="shared")]
    second = [_candidate("b::1", 0.8), _candidate("shared::1", 0.95, source_id="shared")]

    merged = merge_candidates_keep_best([first, second])

    ids = [c["id"] for c in merged]
    assert ids == ["a::1", "shared::1", "b::1"], "shared chunk appears once, first-seen position"
    by_id = {c["id"]: c for c in merged}
    assert by_id["shared::1"]["similarity"] == 0.95, "highest-similarity occurrence wins"


def test_merge_tie_keeps_first_seen():
    first = [_candidate("shared::1", 0.8, source_id="shared", text="first text")]
    second = [_candidate("shared::1", 0.8, source_id="shared", text="second text")]

    merged = merge_candidates_keep_best([first, second])

    assert len(merged) == 1
    assert merged[0]["text"] == "first text"


def test_merge_handles_empty_sets():
    assert merge_candidates_keep_best([]) == []
    assert merge_candidates_keep_best([[], [], []]) == []


def test_evidence_sufficient_checks():
    assert evidence_is_sufficient([], ROUTER_RELEVANCE_THRESHOLD) is False
    weak = [_candidate("a::1", 0.05, text="x")]
    strong = [_candidate("a::1", 0.9, text="x")]
    assert evidence_is_sufficient([weak[0]], ROUTER_RELEVANCE_THRESHOLD) is False
    assert evidence_is_sufficient([strong[0]], ROUTER_RELEVANCE_THRESHOLD) is True
    assert evidence_is_sufficient(
        [{**weak[0], "relevance": 0.0}], ROUTER_RELEVANCE_THRESHOLD
    ) is False
    assert evidence_is_sufficient(
        [{**weak[0], "relevance": 0.7}], ROUTER_RELEVANCE_THRESHOLD
    ) is True
    assert evidence_is_sufficient(
        [{**strong[0], "relevance": None}], ROUTER_RELEVANCE_THRESHOLD
    ) is True, "similarity proxy used when relevance is None (rerank fallback)"


# ------------------------------------------------------------------ router classify unit tests

def test_classify_standard():
    result = _router(json.dumps({"strategy": "STANDARD", "reason": "simple", "subqueries": []})).classify("q")
    assert result["strategy"] == "STANDARD"
    assert result["fell_back"] is False
    assert result["subqueries"] == []


def test_classify_complex():
    result = _router(json.dumps({"strategy": "COMPLEX", "reason": "broad", "subqueries": []})).classify("q")
    assert result["strategy"] == "COMPLEX"
    assert result["fell_back"] is False


def test_classify_multihop():
    result = _router(
        json.dumps({"strategy": "MULTI_HOP", "reason": "needs two facts", "subqueries": ["fact one", "fact two"]})
    ).classify("q")
    assert result["strategy"] == "MULTI_HOP"
    assert result["subqueries"] == ["fact one", "fact two"]
    assert result["fell_back"] is False


def test_classify_malformed_json_falls_back():
    result = _router("this is {{ not json").classify("q")
    assert result["strategy"] == "STANDARD"
    assert result["fell_back"] is True
    assert "not valid JSON" in result["error"]


def test_classify_invalid_strategy_falls_back():
    result = _router(json.dumps({"strategy": "TELEPORT", "reason": "x", "subqueries": []})).classify("q")
    assert result["strategy"] == "STANDARD"
    assert result["fell_back"] is True
    assert "validation" in result["error"]


def test_classify_four_subqueries_truncated():
    subqueries = ["a", "b", "c", "d", "e"]
    result = _router(
        json.dumps({"strategy": "MULTI_HOP", "reason": "many", "subqueries": subqueries})
    ).classify("q")
    assert result["strategy"] == "MULTI_HOP"
    assert result["fell_back"] is False
    assert len(result["subqueries"]) == ROUTER_MAX_SUBQUERIES


def test_classify_too_few_subqueries_falls_back():
    result = _router(
        json.dumps({"strategy": "MULTI_HOP", "reason": "x", "subqueries": ["only one"]})
    ).classify("q")
    assert result["strategy"] == "STANDARD"
    assert result["fell_back"] is True


def test_classify_subqueries_for_non_multihop_discarded():
    result = _router(
        json.dumps({"strategy": "STANDARD", "reason": "x", "subqueries": ["stray"]})
    ).classify("q")
    assert result["strategy"] == "STANDARD"
    assert result["subqueries"] == []
    assert result["fell_back"] is False


def test_classify_timeout_falls_back_quickly():
    slow = _router("", timeout=0.05)
    slow._client_generate = lambda prompt: time.sleep(2.0) or ""

    started = time.monotonic()
    result = slow.classify("q")
    elapsed = time.monotonic() - started

    assert result["strategy"] == "STANDARD"
    assert result["fell_back"] is True
    assert "timed out" in result["reason"]
    assert elapsed < 1.5, "timeout must cut the router call short"


def test_classify_when_unavailable_falls_back(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = QueryRouter().classify("q")
    assert result["strategy"] == "STANDARD"
    assert result["fell_back"] is True
    assert "unavailable" in result["reason"]


def test_classify_empty_question_falls_back():
    result = _router(json.dumps({"strategy": "COMPLEX", "reason": "x", "subqueries": []})).classify("   ")
    assert result["strategy"] == "STANDARD"
    assert result["fell_back"] is True


def test_classify_router_call_exception_falls_back():
    broken = _router("")
    broken._client_generate = lambda prompt: (_ for _ in ()).throw(RuntimeError("network exploded"))
    result = broken.classify("q")
    assert result["strategy"] == "STANDARD"
    assert result["fell_back"] is True
    assert "network exploded" in result["reason"]


# ------------------------------------------------------------------ /ask endpoint wiring

@pytest.fixture()
def ask_env(server_module, monkeypatch):
    def _install(
        router_result: dict | None = None,
        relevance: float = 0.9,
        retrieval_by_query: dict[str, list[dict]] | None = None,
        fail_queries: set[str] | None = None,
    ):
        from fastapi.testclient import TestClient

        recorder = _AdkRecorder()
        calls: list[tuple[str, int]] = []
        monkeypatch.setattr(server_module, "ROUTER", _FakeRouter(router_result))
        monkeypatch.setattr(server_module, "RERANKER", _ScoreAllReranker(relevance=relevance))
        monkeypatch.setattr(server_module, "_run_adk_agent", recorder)
        monkeypatch.setattr(server_module.RAG_STORE, "embedding_provider", "gemini-embedding-2")

        by_query = retrieval_by_query or {
            "q": [
                _candidate("src1::1", 0.95, source_id="src1"),
                _candidate("src1::2", 0.85, source_id="src1"),
                _candidate("src2::1", 0.75, source_id="src2"),
            ]
        }
        fail = fail_queries or set()

        def _retrieve(query: str, candidate_k: int) -> dict:
            calls.append((query, candidate_k))
            if query in fail:
                raise RuntimeError("retrieval exploded")
            return _canned_retrieval(list(by_query.get(query, [])))

        monkeypatch.setattr(server_module.RAG_STORE, "retrieve_candidates", _retrieve)
        return TestClient(server_module.app), calls, recorder

    return _install


def test_ask_standard_uses_phase2_flow_unchanged(ask_env):
    client, calls, recorder = ask_env(_router_result(STANDARD))
    res = client.post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["strategy"] == "STANDARD"
    assert data["router_fell_back"] is False
    assert data["router_reason"] == "because"
    assert data["subqueries"] == []
    assert data["insufficient_evidence"] is False
    assert data["reranked"] is True
    assert calls == [("q", RERANK_CANDIDATE_K)], "STANDARD uses the Phase 2 candidate_k"
    assert data["answer"] == "canned answer"
    assert len(data["matches"]) == 2
    assert recorder.calls and recorder.calls[0]["insufficient_evidence"] is False
    assert data["trace"][0]["agent"] == "query_router"
    assert data["trace"][0]["status"] == "complete"


def test_ask_complex_uses_larger_candidate_k(ask_env):
    assert ROUTER_COMPLEX_CANDIDATE_K > RERANK_CANDIDATE_K
    client, calls, recorder = ask_env(_router_result(COMPLEX, reason="broad question"))
    res = client.post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["strategy"] == "COMPLEX"
    assert data["router_fell_back"] is False
    assert calls == [("q", ROUTER_COMPLEX_CANDIDATE_K)], "COMPLEX uses the configured larger candidate_k"
    assert data["reranked"] is True


def test_ask_multihop_two_subqueries_merged_and_deduped(ask_env):
    overlap = {
        "What is the capital of France?": [
            _candidate("france::1", 0.9, source_id="france", text="Paris"),
            _candidate("shared::1", 0.6, source_id="shared", text="shared low"),
        ],
        "What river is in Paris?": [
            _candidate("river::1", 0.8, source_id="river", text="Seine"),
            _candidate("shared::1", 0.95, source_id="shared", text="shared high"),
        ],
    }
    client, calls, recorder = ask_env(
        _router_result(MULTI_HOP, reason="two facts", subqueries=list(overlap)),
        retrieval_by_query=overlap,
    )
    res = client.post("/ask", json={"question": "q", "top_k": 5})
    assert res.status_code == 200
    data = res.json()

    assert data["strategy"] == "MULTI_HOP"
    assert data["subqueries"] == list(overlap)
    assert calls == [
        ("What is the capital of France?", ROUTER_MULTIHOP_CANDIDATE_K),
        ("What river is in Paris?", ROUTER_MULTIHOP_CANDIDATE_K),
    ], "independent retrieval per subquery"
    assert data["matches"][0]["id"] == "shared::1"
    assert data["matches"][0]["similarity"] == 0.95, "dedup keeps highest similarity occurrence"
    assert [m["id"] for m in data["matches"]] == ["shared::1", "france::1", "river::1"]
    assert data["trace"][2]["agent"] == "retrieval_tool"
    assert "2 subqueries" in data["trace"][2]["detail"]
    assert data["insufficient_evidence"] is False


def test_ask_multihop_three_subqueries(ask_env):
    subqs = [
        "What is the capital of France?",
        "What river is in Paris?",
        "What is the tallest building in Paris?",
    ]
    by_query = {
        subqs[0]: [_candidate("x::1", 0.9, source_id="x", text="Paris")],
        subqs[1]: [_candidate("y::1", 0.8, source_id="y", text="Seine")],
        subqs[2]: [_candidate("z::1", 0.7, source_id="z", text="Eiffel")],
    }
    client, calls, recorder = ask_env(
        _router_result(MULTI_HOP, reason="three facts", subqueries=subqs),
        retrieval_by_query=by_query,
    )
    res = client.post("/ask", json={"question": "q", "top_k": 5})
    assert res.status_code == 200
    data = res.json()

    assert data["strategy"] == "MULTI_HOP"
    assert calls == [(sub, ROUTER_MULTIHOP_CANDIDATE_K) for sub in subqs]
    assert {m["id"] for m in data["matches"]} == {"x::1", "y::1", "z::1"}


def test_ask_multihop_subquery_retrieves_nothing_does_not_crash(ask_env):
    client, calls, recorder = ask_env(
        _router_result(MULTI_HOP, reason="x", subqueries=["sub-ok", "sub-empty"]),
        retrieval_by_query={
            "sub-ok": [_candidate("a::1", 0.9, source_id="a")],
            "sub-empty": [],
        },
    )
    res = client.post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["strategy"] == "MULTI_HOP"
    assert [m["id"] for m in data["matches"]] == ["a::1"]


def test_ask_multihop_subquery_retrieval_error_does_not_crash(ask_env):
    client, calls, recorder = ask_env(
        _router_result(MULTI_HOP, reason="x", subqueries=["sub-boom", "sub-ok"]),
        retrieval_by_query={"sub-ok": [_candidate("a::1", 0.9, source_id="a")]},
        fail_queries={"sub-boom"},
    )
    res = client.post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["strategy"] == "MULTI_HOP"
    assert data["matches"], "the healthy subquery still contributes evidence"


def test_ask_multihop_merged_empty_feeds_insufficient_path(ask_env):
    client, calls, recorder = ask_env(
        _router_result(MULTI_HOP, reason="x", subqueries=["sub-one", "sub-two"]),
        retrieval_by_query={"sub-one": [], "sub-two": []},
    )
    res = client.post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["strategy"] == "MULTI_HOP"
    assert data["matches"] == []
    assert data["insufficient_evidence"] is True
    assert recorder.calls[0]["insufficient_evidence"] is True
    assert recorder.calls[0]["retrieval"]["matches"] == []
    assert "insufficient" in data["answer"].lower()


def test_ask_malformed_router_json_falls_back_to_standard(ask_env):
    client, calls, recorder = ask_env(
        _router_result(STANDARD, reason="Router output was not valid JSON; using STANDARD.", fell_back=True, error="Router output was not valid JSON.")
    )
    res = client.post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["strategy"] == "STANDARD"
    assert data["router_fell_back"] is True, "fallback is distinguishable from a normal STANDARD"
    assert data["answer"] == "canned answer", "query still answered after fallback"
    assert calls == [("q", RERANK_CANDIDATE_K)]
    assert data["trace"][0]["status"] == "fallback"


def test_ask_router_invalid_strategy_falls_back_and_answers(ask_env):
    client, calls, recorder = ask_env(_router_result(STANDARD, reason="invalid; fallback", fell_back=True))
    res = client.post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["strategy"] == "STANDARD"
    assert data["router_fell_back"] is True
    assert data["answer"] == "canned answer"


def test_ask_no_relevant_content_states_insufficient_evidence(ask_env):
    client, calls, recorder = ask_env(
        _router_result(STANDARD),
        relevance=0.0,
        retrieval_by_query={
            "q": [
                _candidate("src1::1", 0.04, source_id="src1", text="unrelated filler"),
                _candidate("src1::2", 0.05, source_id="src1", text="still unrelated"),
            ]
        },
    )
    res = client.post("/ask", json={"question": "q", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["insufficient_evidence"] is True, "weak evidence triggers the insufficient path"
    assert len(data["matches"]) == 2, "found matches still surfaced for citation context"
    assert recorder.calls[0]["insufficient_evidence"] is True
    assert recorder.calls[0]["retrieval"]["matches"] == [], "no weak evidence forwarded to generation"
    assert "insufficient" in data["answer"].lower(), "answer explicitly states insufficiency"