"""Full evaluation runner for the Phase 5 benchmark (Phase 6 metrics).

Usage:
    python -m evaluation.run_evaluation [OPTIONS]

The runner invokes the **actual Phase 1-4 pipeline** programmatically for
every benchmark question, without going through the HTTP API. Each stage
(routing, retrieval, reranking, insufficient-evidence check, generation,
verification, judge) is exercised in the same order the ``/ask`` endpoint
uses; timing is recorded around every stage with ``time.perf_counter()``.

When ``GOOGLE_API_KEY`` is absent (the benchmark's deterministic-embedding
configuration), generation, verification, and judge all degrade gracefully:
generation is recorded as unavailable, verification and judge each record
``status="unavailable"`` in the results. Retrieval metrics (Recall@K, MRR)
are always computed and always fully functional offline.

Results are written to ``<results_dir>/run_<UTC-timestamp>.json`` as a
single JSON file containing per-question records, aggregate metrics, corpus-
drift flags, and run metadata. Phase 7 consumes this file for comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Paths must resolve correctly when run as ``python -m evaluation.run_evaluation``.
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
EVAL_DIR = REPO_ROOT / "evaluation"
BENCHMARK_PATH = EVAL_DIR / "benchmark" / "benchmark_v1.json"
MANIFEST_PATH = EVAL_DIR / "corpus" / "manifest.json"
CORPUS_PATH = EVAL_DIR / "corpus" / "corpus.json"
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

# Backend singletons — imported *inside* functions so the module can be
# imported by tests without triggering Chroma init or Gemini client
# construction at module-import time.


def _load_shared_pipeline():
    """Return the pipeline singletons, installing deterministic embeddings.

    This is the same store/router/reranker/verifier that server.py creates
    in app_state.py. Calling this once at startup is sufficient; the store
    is stateful and the remaining callables (router, reranker, verifier)
    are reusable across questions.
    """
    from eval_embeddings import install_deterministic_embeddings
    from app_state import RAG_STORE, RERANKER, ROUTER, VERIFIER

    info = install_deterministic_embeddings(RAG_STORE)
    return RAG_STORE, RERANKER, ROUTER, VERIFIER, info


def _load_benchmark(path: Path) -> dict[str, Any]:
    """Load and validate the benchmark file. Fails loudly on any problem."""
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. "
            "Run ``python -m evaluation.scripts.ingest_corpus`` first."
        )
    if not path.is_file():
        raise IsADirectoryError(f"Expected a file but got a directory: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Could not read benchmark file: {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Benchmark file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(
            f"Benchmark file does not have the expected structure (missing 'questions'): {path}"
        )
    questions = data["questions"]
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError(f"Benchmark 'questions' list is empty or not a list: {path}")
    required_keys = {"id", "question", "expected_answer", "expected_sources", "category", "difficulty"}
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError(f"Non-dict question entry: {item!r}")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(
                f"Question {item.get('id', '?')!r} is missing keys: {missing}"
            )
    return data


def _detect_corpus_drift(
    expected_sources: list[str], live_source_ids: set[str]
) -> dict[str, Any]:
    """Return corpus-drift info for one question.

    A source id referenced in expected_sources that is not in the live store
    is flagged explicitly so it is not silently scored as a retrieval miss
    indistinguishable from a real retrieval failure.
    """
    missing = [sid for sid in expected_sources if sid not in live_source_ids]
    return {"present": bool(missing), "missing_sources": missing}


# ---------------------------------------------------------------------------
# Timing wrapper
# ---------------------------------------------------------------------------


def _timed(fn, *args, **kwargs) -> tuple[Any, float]:
    """Call ``fn(*args, **kwargs)`` and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


async def _atimed(fn, *args, **kwargs) -> tuple[Any, float]:
    """Call ``fn(*args, **kwargs)`` and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = await fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# ADK generation helper (mirrors server._run_adk_agent)
# ---------------------------------------------------------------------------

APP_NAME = "evaluation_run"
_USER_ID = "eval-bot"


async def _generate_answer(
    question: str,
    retrieval_payload: dict[str, Any],
    insufficient_evidence: bool = False,
    session_service: Any = None,
    adk_available: bool = False,
) -> tuple[str, float]:
    """Run the ADK agent to generate an answer. Mirrors server._run_adk_agent.

    Returns (answer, elapsed_ms). When ADK is unavailable, returns a
    deterministic insufficiency marker.
    """
    if not adk_available:
        return "", 0.0

    from google.adk.runners import Runner
    from agentic_rag_agent.agent import build_agent

    def retrieve_relevant_context(query: str, top_k: int = 6) -> dict:
        return retrieval_payload

    request_agent = build_agent(retrieve_relevant_context)
    request_runner = Runner(agent=request_agent, app_name=APP_NAME, session_service=session_service)
    session = await session_service.create_session(app_name=APP_NAME, user_id=_USER_ID)

    instruction = (
        "Question: {question}\nUse the retrieval tool result for this exact question."
        if not insufficient_evidence
        else (
            "Question: {question}\n"
            "The retrieved evidence is insufficient to answer this question. "
            "State clearly that the available evidence is insufficient to answer "
            "the question, and do not invent an answer."
        )
    )

    from google.genai import types as genai_types

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=instruction.format(question=question))],
    )
    start = time.perf_counter()
    final_text = ""
    async for event in request_runner.run_async(
        user_id=_USER_ID,
        session_id=session.id,
        new_message=content,
    ):
        if getattr(event, "content", None) and event.content.parts:
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if text:
                    final_text = text
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return final_text, elapsed_ms


# ---------------------------------------------------------------------------
# Evidence payload (mirrors server._evidence_payload)
# ---------------------------------------------------------------------------


def _evidence_payload(evidence: list[dict[str, Any]], store: Any) -> dict[str, Any]:
    """Shape the reranked evidence for the ADK generator."""
    return {
        "provider": store.embedding_provider,
        "matches": [
            {
                "citation": item.get("id", ""),
                "source": item.get("title", ""),
                "modality": item.get("modality", "text"),
                "similarity": item.get("similarity", 0.0),
                "evidence": item.get("text", ""),
            }
            for item in evidence
        ],
    }


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------


def _store_source_ids(store: Any) -> set[str]:
    return {source.id for source in store.sources}


async def _run_question(
    question: dict[str, Any],
    store: Any,
    router: Any,
    reranker: Any,
    verifier: Any,
    judge: Any,
    live_source_ids: set[str],
    *,
    k_candidates: int,
    final_k: int,
    skip_generation: bool = False,
    session_service: Any = None,
    adk_available: bool = False,
) -> dict[str, Any]:
    """Run one benchmark question through the full pipeline; return a record."""
    from query_router import (
        COMPLEX,
        MULTI_HOP,
        ROUTER_COMPLEX_CANDIDATE_K,
        ROUTER_MULTIHOP_CANDIDATE_K,
        ROUTER_RELEVANCE_THRESHOLD,
        evidence_is_sufficient,
        merge_candidates_keep_best,
    )
    from reranker import rerank_and_select
    from verifier import run_verification

    qid = question["id"]
    question_text = question["question"]
    expected_sources = list(question["expected_sources"])
    expected_answer = question["expected_answer"]
    is_unanswerable = question["category"] == "unanswerable"

    timing: dict[str, float] = {}
    total_start = time.perf_counter()

    # --- Router ---
    route, router_ms = _timed(router.classify, question_text)
    strategy = route["strategy"]
    router_fell_back = bool(route["fell_back"])
    subqueries = list(route.get("subqueries") or [])
    timing["router_ms"] = router_ms

    # --- Retrieval ---
    is_multi_hop = strategy == MULTI_HOP
    is_complex = strategy == COMPLEX

    if is_multi_hop:
        candidate_k = max(1, ROUTER_MULTIHOP_CANDIDATE_K)
    else:
        candidate_k = max(
            1, ROUTER_COMPLEX_CANDIDATE_K if is_complex else k_candidates
        )

    candidate_sets: list[list[dict[str, Any]]] = []
    retrieval_ms = 0.0
    if is_multi_hop:
        start = time.perf_counter()
        for subquery in subqueries:
            try:
                retrieval = store.retrieve_candidates(subquery, candidate_k)
                candidate_sets.append(retrieval.get("candidates") or [])
            except Exception:
                candidate_sets.append([])
        candidates = merge_candidates_keep_best(candidate_sets)
        retrieval_ms = (time.perf_counter() - start) * 1000.0
    else:
        retrieval, retrieval_ms = _timed(
            store.retrieve_candidates, question_text, candidate_k
        )
        candidates = retrieval.get("candidates") or []

    timing["retrieval_ms"] = retrieval_ms

    # --- Corpus drift check ---
    corpus_drift = _detect_corpus_drift(expected_sources, live_source_ids)

    # --- Rerank ---
    outcome, rerank_ms = _timed(rerank_and_select, reranker, question_text, candidates, final_k)
    evidence = outcome["evidence"]
    timing["rerank_ms"] = rerank_ms

    # --- Insufficient-evidence check ---
    insufficient = not evidence_is_sufficient(evidence, ROUTER_RELEVANCE_THRESHOLD)

    # --- Generation ---
    # For insufficient questions the real agent is still invoked in server.py
    # (with insufficient_evidence=True) and responds with an explicit
    # insufficiency statement; we mirror that here. Offline (no Gemini client),
    # generation is unavailable and we emit a deterministic insufficiency-only
    # proxy when the question trips the Phase 3 gate so the judge input is a
    # stated insufficiency rather than empty text.
    payload_evidence: list[dict[str, Any]] = [] if insufficient else evidence
    retrieval_payload = _evidence_payload(payload_evidence, store)

    generation_ms = 0.0
    answer = ""
    generation_status = "complete"
    if skip_generation:
        generation_status = "skipped"
        answer = ""
    elif adk_available:
        generation_status = "complete"
        answer, generation_ms = await _generate_answer(
            question_text,
            retrieval_payload,
            insufficient_evidence=insufficient,
            session_service=session_service,
            adk_available=adk_available,
        )
    elif insufficient:
        generation_status = "unavailable"
        answer = "The available evidence is insufficient to answer the question."
    else:
        generation_status = "unavailable"
        answer = ""
    timing["generation_ms"] = generation_ms

    # --- Verification ---
    verification_result: dict[str, Any]
    if not answer or not answer.strip():
        verification_result = {
            "unavailable": True,
            "reason": "No generated answer to verify.",
            "grounded": None,
            "score": None,
            "claims": [],
            "unsupported_claims": [],
            "errors": [],
        }
        timing["verification_ms"] = 0.0
    else:
        verification_result, verification_ms = _timed(
            run_verification,
            verifier,
            question_text,
            answer,
            payload_evidence,
            [item.get("id", "") for item in payload_evidence],
        )
        timing["verification_ms"] = verification_ms

    # --- LLM judge ---
    # A question that tripped the Phase 3 gate is judged on its stated
    # insufficiency (grounded by construction per the judge's definition).
    judge_ms = 0.0
    if judge is None or not judge.available:
        judge_result: dict[str, Any] = {
            "status": "unavailable",
            "reason": "Judge not configured; judgment not run.",
            "correctness": None,
            "groundedness": None,
            "citation_correctness": None,
            "errors": [],
        }
    elif answer and answer.strip():
        judge_result, judge_ms = _atimed(
            judge.judge,
            question=question_text,
            expected_answer=expected_answer,
            answer=answer,
            evidence=payload_evidence,
            insufficient_evidence=insufficient,
        )
    else:
        judge_result = {
            "status": "unavailable",
            "reason": "No generated answer to score.",
            "correctness": None,
            "groundedness": None,
            "citation_correctness": None,
            "errors": [],
        }
    timing["judge_ms"] = judge_ms

    timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0

    # --- Compute retrieval metrics ---
    from metrics import source_id_sequence, recall_at_k, mrr, precision_at_k

    metric_ids = source_id_sequence(candidates)
    recall5 = recall_at_k(expected_sources, metric_ids, 5)
    recall10 = recall_at_k(expected_sources, metric_ids, 10)
    mrr_val = mrr(expected_sources, metric_ids)
    precision5 = precision_at_k(expected_sources, metric_ids, 5)
    precision10 = precision_at_k(expected_sources, metric_ids, 10)

    # --- Unanswerable handling ---
    unanswerable_result: dict[str, Any] | None = None
    if is_unanswerable:
        unanswerable_result = {
            "passed": insufficient,  # system correctly refused to answer
            "insufficient_evidence": insufficient,
            "retrieved_source_ids": metric_ids,
            "final_evidence_count": len(evidence),
        }

    # --- Outcome marker ---
    if is_unanswerable:
        outcome_marker = "unanswerable_handled" if insufficient else "unanswerable_missed"
    elif generation_status == "skipped":
        outcome_marker = "generation_skipped"
    elif generation_status == "unavailable":
        outcome_marker = "generation_unavailable"
    elif verification_result.get("unavailable"):
        outcome_marker = "verification_unavailable"
    else:
        outcome_marker = "complete"

    # --- Metric IDs tracked per question ---
    # Final evidence ids (chunk-level) used for generation
    evidence_ids = [item.get("id", "") for item in payload_evidence]

    record: dict[str, Any] = {
        "question_id": qid,
        "question": question_text,
        "category": question["category"],
        "difficulty": question["difficulty"],
        "expected_answer": expected_answer,
        "expected_sources": expected_sources,
        "unanswerable": is_unanswerable,
        "strategy": strategy,
        "router_fell_back": router_fell_back,
        "subqueries": subqueries,
        "retrieved_ids": [item.get("id", "") for item in candidates],
        "retrieved_source_ids": metric_ids,
        "final_evidence_ids": evidence_ids,
        "final_evidence_count": len(evidence),
        "insufficient_evidence": insufficient,
        "answer": answer,
        "verification": verification_result,
        "metrics": {
            "retrieval": {
                "recall_at_5": recall5,
                "recall_at_10": recall10,
                "mrr": mrr_val,
                "precision_at_5": precision5,
                "precision_at_10": precision10,
            },
            "judge": judge_result,
        },
        "latency_ms": timing,
        "corpus_drift": corpus_drift,
        "generation_status": generation_status,
        "outcome": outcome_marker,
    }
    if unanswerable_result is not None:
        record["unanswerable_result"] = unanswerable_result
    return record


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------


async def _run_benchmark(
    benchmark: dict[str, Any],
    *,
    results_dir: Path,
    limit: int | None = None,
    skip_judge: bool = False,
    k_candidates: int = 12,
    final_k: int = 5,
) -> dict[str, Any]:
    """Run every benchmark question and return the full results dict."""
    from metrics import LLMJudge, aggregate_retrieval_metrics

    store, reranker, router, verifier, embed_info = _load_shared_pipeline()
    live_source_ids = _store_source_ids(store)
    questions = benchmark["questions"][:limit] if limit else benchmark["questions"]

    # ADK / judge clients
    adk_available = False
    session_service = None
    try:
        from google.adk.sessions import InMemorySessionService
        adk_available = bool(os.getenv("GOOGLE_API_KEY"))
        if adk_available:
            session_service = InMemorySessionService()
    except Exception:
        adk_available = False

    judge = None
    if not skip_judge:
        # LLMJudge builds a Gemini client only when GOOGLE_API_KEY is set;
        # otherwise it reports itself unavailable and records generation
        # metrics as unavailable.
        judge = LLMJudge()

    per_question_records: list[dict[str, Any]] = []
    run_errors: list[str] = []

    for question in questions:
        try:
            record = await _run_question(
                question=question,
                store=store,
                router=router,
                reranker=reranker,
                verifier=verifier,
                judge=judge,
                live_source_ids=live_source_ids,
                k_candidates=k_candidates,
                final_k=final_k,
                session_service=session_service,
                adk_available=adk_available,
            )
            per_question_records.append(record)
        except Exception as exc:
            run_errors.append(f"{question['id']}: {exc}")
            per_question_records.append(
                {
                    "question_id": question["id"],
                    "question": question["question"],
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "expected_answer": question["expected_answer"],
                    "expected_sources": question["expected_sources"],
                    "unanswerable": question["category"] == "unanswerable",
                    "error": str(exc),
                    "outcome": "error",
                }
            )

    # --- Aggregates ---
    retrieval_agg = aggregate_retrieval_metrics(per_question_records)

    # --- Unanswerable aggregate ---
    unanswerable_records = [
        r for r in per_question_records if r.get("unanswerable")
    ]
    correctly_handled = sum(
        1 for r in unanswerable_records
        if r.get("unanswerable_result", {}).get("passed", False)
    )
    unanswerable_agg = {
        "count": len(unanswerable_records),
        "correctly_handled": correctly_handled,
        "miscount": len(unanswerable_records) - correctly_handled,
    }

    # --- Judge aggregate ---
    judge_records = [
        r["metrics"]["judge"]
        for r in per_question_records
        if "judge" in r.get("metrics", {})
    ]
    judge_ok = [j for j in judge_records if j.get("status") == "ok"]
    judge_agg: dict[str, Any] = {
        "total": len(judge_records),
        "available": len(judge_ok),
        "unavailable": len(judge_records) - len(judge_ok),
    }
    if judge_ok:
        from metrics import _mean
        judge_agg["correctness_mean"] = _mean([j["correctness"] for j in judge_ok])
        judge_agg["groundedness_mean"] = _mean([j["groundedness"] for j in judge_ok])
        judge_agg["citation_correctness_mean"] = _mean(
            [j["citation_correctness"] for j in judge_ok]
        )

    # --- Corpus drift aggregate ---
    drift_questions = [
        r["question_id"]
        for r in per_question_records
        if r.get("corpus_drift", {}).get("present", False)
    ]

    run_timestamp = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "run": {
            "timestamp": run_timestamp.isoformat(),
            "benchmark_name": benchmark.get("name", ""),
            "benchmark_version": benchmark.get("version", ""),
            "benchmark_path": str(BENCHMARK_PATH),
            "question_count": len(per_question_records),
            "api_available": adk_available,
            "skip_judge": skip_judge,
            "judge_model": judge.model if judge else "none",
            "embedding": embed_info,
            "k_candidates": k_candidates,
            "final_k": final_k,
        },
        "aggregates": {
            "retrieval": retrieval_agg,
            "unanswerable": unanswerable_agg,
            "judge": judge_agg,
            "corpus_drift": {
                "count": len(drift_questions),
                "questions": drift_questions,
            },
            "errors": run_errors,
        },
        "per_question": per_question_records,
    }

    # --- Persist ---
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"run_{run_timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    results_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    result["_results_path"] = str(results_path)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 5 benchmark through Phase 6 metrics."
    )
    parser.add_argument(
        "--benchmark",
        default=str(BENCHMARK_PATH),
        help=f"Benchmark JSON path (default: {BENCHMARK_PATH})",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory for results files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the LLM judge for generation metrics (judge unavailable).",
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        default=12,
        help="Number of raw retrieval candidates to fetch (default 12 = RERANK_CANDIDATE_K).",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
        help="Number of evidence items for generation (default 5 = RERANK_FINAL_K).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the benchmark. Called from ``__main__`` and from tests."""
    args = _parse_args(argv)
    benchmark_path = Path(args.benchmark)
    benchmark = _load_benchmark(benchmark_path)
    result = asyncio.run(
        _run_benchmark(
            benchmark,
            results_dir=Path(args.results_dir),
            limit=args.limit,
            skip_judge=args.skip_judge,
            k_candidates=args.k_candidates,
            final_k=args.final_k,
        )
    )
    return result


if __name__ == "__main__":
    result = main()
    path = result.get("_results_path", "")
    agg = result["aggregates"]
    retrieval = agg["retrieval"]
    unans = agg["unanswerable"]
    judge = agg["judge"]
    print(
        f"Done — {result['run']['question_count']} questions. "
        f"Results: {path}"
    )
    print(
        f"  Recall@5={retrieval['recall_at_5']:.4f}  "
        f"Recall@10={retrieval['recall_at_10']:.4f}  "
        f"MRR={retrieval['mrr']:.4f}  "
        f"(answerable={retrieval['answerable_count']})"
    )
    print(
        f"  Unanswerable: {unans['count']} total, "
        f"{unans['correctly_handled']} correctly handled"
    )
    if judge.get("correctness_mean") is not None:
        print(
            f"  Judge (n={judge['available']}/{judge['total']}): "
            f"correctness={judge['correctness_mean']:.4f}  "
            f"groundedness={judge['groundedness_mean']:.4f}  "
            f"citation_correctness={judge['citation_correctness_mean']:.4f}"
        )
    else:
        print(f"  Judge: {judge['unavailable']} unavailable of {judge['total']}")
    if agg["corpus_drift"]["count"]:
        print(f"  ⚠ Corpus drift: {agg['corpus_drift']['questions']}")
    if agg["errors"]:
        print(f"  Errors: {agg['errors']}")
