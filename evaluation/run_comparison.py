"""Phase 7 baseline-vs-improved comparison (orchestration only).

This module implements the central experiment of the project: a rigorous,
apples-to-apples comparison between a simple baseline RAG pipeline and the
full improved pipeline built across Phases 1-4.

It modifies no Phase 1-6 logic. Specifically it does NOT touch retrieval,
reranking, routing, verification, the benchmark content, or the metric
definitions. It only orchestrates two pipelines through the existing Phase 6
metrics and produces an executed comparison:

- BASELINE (reconstructed here, isolated code path): Question -> embedding ->
  raw Chroma similarity search (``MultimodalRagStore.retrieve_candidates``,
  the pre-reranking function from Phase 1) -> top K -> generation via the same
  ADK answer flow used elsewhere. No reranking, no adaptive routing, no query
  decomposition, no grounding verification. Baseline still produces citations
  in the same existing format; it only skips reranking/routing/verification.
- IMPROVED (the real Phase 2-4 pipeline, called as-is): adaptive routing ->
  candidate retrieval -> reranking -> evidence selection -> grounded
  generation -> verification, invoked through
  ``evaluation.run_evaluation._run_question`` unchanged, NOT reimplemented.

Both pipelines run against the identical fixed corpus (no re-ingestion), the
identical Phase 5 benchmark question set, the same embedding model
(deterministic feature-hash when offline, Gemini when keyed -- installed once
via ``_load_shared_pipeline``), and the identical Phase 6 metrics functions
(``recall_at_k`` / ``mrr`` / ``aggregate_retrieval_metrics`` / ``LLMJudge``).

Usage:
    python -m evaluation.run_comparison
    python -m evaluation.run_comparison --limit 3
    python -m evaluation.run_comparison --skip-judge
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

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
EVAL_DIR = REPO_ROOT / "evaluation"
BENCHMARK_PATH = EVAL_DIR / "benchmark" / "benchmark_v1.json"
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

# Backend singletons are imported *inside* functions so this module stays
# importable by tests without triggering Chroma init at import time.


def _baseline_k(final_k: int) -> int:
    """Baseline top-K is defined to equal the improved pipeline's final_k."""
    return final_k


def _corpus_fingerprint(store: Any) -> dict[str, Any]:
    """Snapshot the live corpus state so cross-pipeline drift is detectable."""
    try:
        source_ids = sorted(source.id for source in store.sources)
    except Exception:
        source_ids = []
    try:
        chunk_count = store._collection_count(store._collection)
    except Exception:
        chunk_count = None
    return {"source_ids": source_ids, "chunk_count": chunk_count}


async def _run_baseline_question(
    question: dict[str, Any],
    store: Any,
    judge: Any,
    live_source_ids: set[str],
    *,
    final_k: int,
    session_service: Any = None,
    adk_available: bool = False,
) -> dict[str, Any]:
    """Run one benchmark question through the isolated baseline pipeline.

    Baseline path: raw Chroma similarity search (``retrieve_candidates`` with
    ``candidate_k == final_k``) -> top-K evidence -> same ADK generation flow.
    No router call, no reranking, no insufficient-evidence threshold gate, no
    Phase 4 verifier call. Never raises for per-question pipeline failures;
    only unexpected harness bugs propagate (the caller still guards those).
    """
    from run_evaluation import (
        _detect_corpus_drift,
        _evidence_payload,
        _generate_answer,
        _timed,
    )

    qid = question["id"]
    question_text = question["question"]
    expected_sources = list(question["expected_sources"])
    expected_answer = question["expected_answer"]
    is_unanswerable = question["category"] == "unanswerable"

    timing: dict[str, float] = {}
    total_start = time.perf_counter()
    top_k = _baseline_k(final_k)

    # --- Retrieval: raw similarity search only (pre-reranking Phase 1 path) ---
    retrieval_error = ""
    try:
        start = time.perf_counter()
        retrieval = store.retrieve_candidates(question_text, top_k)
        retrieval_ms = (time.perf_counter() - start) * 1000.0
        candidates = retrieval.get("candidates") or []
    except Exception as exc:  # baseline retrieval failure: record, don't abort
        retrieval_ms = (time.perf_counter() - start) * 1000.0
        candidates = []
        retrieval_error = str(exc)
    timing["retrieval_ms"] = retrieval_ms
    # Router/rerank stages do not exist on this path; keys stay present (0.0)
    # so per-stage latency dicts remain comparable across pipelines.
    timing["router_ms"] = 0.0
    timing["rerank_ms"] = 0.0

    # --- Corpus drift check (same function as the improved pipeline) ---
    corpus_drift = _detect_corpus_drift(expected_sources, live_source_ids)

    # --- Evidence = top-K similarity hits directly (no reranking) ---
    evidence: list[dict[str, Any]] = [
        {**item, "relevance": None, "reason": None} for item in candidates
    ]

    # Baseline has no Phase 3 threshold gate: it generates from whatever the
    # similarity search returned. Only a genuinely empty evidence set counts
    # as insufficient (nothing to generate from).
    insufficient = len(evidence) == 0
    payload_evidence = list(evidence)
    retrieval_payload = _evidence_payload(payload_evidence, store)

    # --- Generation: same ADK answer flow, same payload shape as improved ---
    generation_ms = 0.0
    answer = ""
    generation_status = "complete"
    generation_error = ""
    try:
        if adk_available:
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
    except Exception as exc:  # generation failure: mark, don't abort the run
        generation_ms = 0.0
        answer = ""
        generation_status = "failed"
        generation_error = str(exc)
    timing["generation_ms"] = generation_ms

    # --- Verification: baseline never runs the Phase 4 internal verifier ---
    verification_result: dict[str, Any] = {
        "unavailable": True,
        "reason": "Baseline pipeline skips Phase 4 verification by design.",
        "grounded": None,
        "score": None,
        "claims": [],
        "unsupported_claims": [],
        "errors": [],
    }
    timing["verification_ms"] = 0.0

    # --- LLM judge: same Phase 6 judge, same scale, against baseline's own
    # answer/citations (separate from the skipped internal verifier) ---
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
        judge_result, judge_ms = _timed(
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

    # --- Retrieval metrics: same Phase 6 functions as the improved path ---
    from metrics import source_id_sequence, recall_at_k, mrr, precision_at_k

    metric_ids = source_id_sequence(candidates)

    unanswerable_result: dict[str, Any] | None = None
    if is_unanswerable:
        unanswerable_result = {
            "passed": insufficient,
            "insufficient_evidence": insufficient,
            "retrieved_source_ids": metric_ids,
            "final_evidence_count": len(evidence),
        }

    if is_unanswerable:
        outcome_marker = "unanswerable_handled" if insufficient else "unanswerable_missed"
    elif generation_status == "failed":
        outcome_marker = "generation_failed"
    elif generation_status == "unavailable":
        outcome_marker = "generation_unavailable"
    else:
        outcome_marker = "complete"

    evidence_ids = [item.get("id", "") for item in payload_evidence]
    record: dict[str, Any] = {
        "pipeline": "baseline",
        "question_id": qid,
        "question": question_text,
        "category": question["category"],
        "difficulty": question["difficulty"],
        "expected_answer": expected_answer,
        "expected_sources": expected_sources,
        "unanswerable": is_unanswerable,
        "strategy": "BASELINE",
        "router_fell_back": False,
        "subqueries": [],
        "retrieved_ids": [item.get("id", "") for item in candidates],
        "retrieved_source_ids": metric_ids,
        "final_evidence_ids": evidence_ids,
        "citations": evidence_ids,
        "final_evidence_count": len(evidence),
        "insufficient_evidence": insufficient,
        "answer": answer,
        "verification": verification_result,
        "metrics": {
            "retrieval": {
                "recall_at_5": recall_at_k(expected_sources, metric_ids, 5),
                "recall_at_10": recall_at_k(expected_sources, metric_ids, 10),
                "mrr": mrr(expected_sources, metric_ids),
                "precision_at_5": precision_at_k(expected_sources, metric_ids, 5),
                "precision_at_10": precision_at_k(expected_sources, metric_ids, 10),
            },
            "judge": judge_result,
        },
        "latency_ms": timing,
        "corpus_drift": corpus_drift,
        "generation_status": generation_status,
        "outcome": outcome_marker,
        "top_k": top_k,
    }
    if retrieval_error:
        record["retrieval_error"] = retrieval_error
    if generation_error:
        record["generation_error"] = generation_error
    if unanswerable_result is not None:
        record["unanswerable_result"] = unanswerable_result
    return record


def _aggregate_pipeline(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate retrieval + judge + unanswerable metrics for one pipeline.

    Uses the same Phase 6 aggregation helpers as ``run_evaluation`` so both
    pipelines are scored identically.
    """
    from metrics import _mean, aggregate_retrieval_metrics

    retrieval_agg = aggregate_retrieval_metrics(records)

    unanswerable_records = [r for r in records if r.get("unanswerable")]
    correctly_handled = sum(
        1
        for r in unanswerable_records
        if r.get("unanswerable_result", {}).get("passed", False)
    )
    unanswerable_agg = {
        "count": len(unanswerable_records),
        "correctly_handled": correctly_handled,
        "miscount": len(unanswerable_records) - correctly_handled,
    }

    judge_records = [
        r["metrics"]["judge"]
        for r in records
        if "judge" in r.get("metrics", {})
    ]
    judge_ok = [j for j in judge_records if j.get("status") == "ok"]
    judge_agg: dict[str, Any] = {
        "total": len(judge_records),
        "available": len(judge_ok),
        "unavailable": len(judge_records) - len(judge_ok),
    }
    if judge_ok:
        judge_agg["correctness_mean"] = _mean([j["correctness"] for j in judge_ok])
        judge_agg["groundedness_mean"] = _mean([j["groundedness"] for j in judge_ok])
        judge_agg["citation_correctness_mean"] = _mean(
            [j["citation_correctness"] for j in judge_ok]
        )

    drift_questions = [
        r["question_id"]
        for r in records
        if r.get("corpus_drift", {}).get("present", False)
    ]
    failed = [r["question_id"] for r in records if r.get("outcome") == "error"]
    generation_failed = [
        r["question_id"] for r in records if r.get("outcome") == "generation_failed"
    ]
    return {
        "retrieval": retrieval_agg,
        "unanswerable": unanswerable_agg,
        "judge": judge_agg,
        "corpus_drift": {"count": len(drift_questions), "questions": drift_questions},
        "failed_questions": failed,
        "generation_failed_questions": generation_failed,
    }


def _diff_or_none(improved: Any, baseline: Any) -> Any:
    """Compute improved - baseline; None when either side is unavailable."""
    if improved is None or baseline is None:
        return None
    try:
        return round(float(improved) - float(baseline), 4)
    except (TypeError, ValueError):
        return None


def _compute_differences(
    baseline_agg: dict[str, Any], improved_agg: dict[str, Any]
) -> dict[str, Any]:
    """Real computed differences (improved - baseline) from actual aggregates."""
    b_ret, i_ret = baseline_agg["retrieval"], improved_agg["retrieval"]
    diffs: dict[str, Any] = {
        "recall_at_5": _diff_or_none(i_ret.get("recall_at_5"), b_ret.get("recall_at_5")),
        "recall_at_10": _diff_or_none(i_ret.get("recall_at_10"), b_ret.get("recall_at_10")),
        "mrr": _diff_or_none(i_ret.get("mrr"), b_ret.get("mrr")),
        "precision_at_5": _diff_or_none(
            i_ret.get("precision_at_5"), b_ret.get("precision_at_5")
        ),
        "precision_at_10": _diff_or_none(
            i_ret.get("precision_at_10"), b_ret.get("precision_at_10")
        ),
    }
    b_judge, i_judge = baseline_agg["judge"], improved_agg["judge"]
    for key in (
        "correctness_mean",
        "groundedness_mean",
        "citation_correctness_mean",
    ):
        diffs[key] = _diff_or_none(i_judge.get(key), b_judge.get(key))
    b_un, i_un = baseline_agg["unanswerable"], improved_agg["unanswerable"]
    diffs["unanswerable_correctly_handled"] = (i_un["correctly_handled"] or 0) - (
        b_un["correctly_handled"] or 0
    )
    return diffs


async def _run_comparison(
    benchmark: dict[str, Any],
    *,
    results_dir: Path,
    limit: int | None = None,
    skip_judge: bool = False,
    k_candidates: int = 12,
    final_k: int = 5,
) -> dict[str, Any]:
    """Run every benchmark question through both pipelines; return results."""
    from metrics import LLMJudge
    from run_evaluation import (
        _load_benchmark as _unused,  # noqa: F401 (kept for parity check)
        _load_shared_pipeline,
        _run_question,
        _store_source_ids,
    )
    from reranker import RERANK_CANDIDATE_K, RERANK_FINAL_K

    # Document and enforce the K contract: baseline top-K == improved final_k.
    baseline_k = _baseline_k(final_k)

    store, reranker, router, verifier, embed_info = _load_shared_pipeline()
    live_source_ids = _store_source_ids(store)
    questions = benchmark["questions"][:limit] if limit else benchmark["questions"]

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
        judge = LLMJudge()

    fingerprint_before = _corpus_fingerprint(store)

    baseline_records: list[dict[str, Any]] = []
    improved_records: list[dict[str, Any]] = []
    run_errors: list[str] = []

    async def _run_all() -> None:
        # --- Baseline pass (isolated similarity-search path) ---
        for question in questions:
            try:
                record = await _run_baseline_question(
                    question=question,
                    store=store,
                    judge=judge,
                    live_source_ids=live_source_ids,
                    final_k=final_k,
                    session_service=session_service,
                    adk_available=adk_available,
                )
                baseline_records.append(record)
            except Exception as exc:  # never abort the run on one question
                run_errors.append(f"baseline:{question['id']}: {exc}")
                baseline_records.append(
                    {
                        "pipeline": "baseline",
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

        fingerprint_mid = _corpus_fingerprint(store)

        # --- Improved pass (real Phase 2-4 pipeline, called as-is) ---
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
                record["pipeline"] = "improved"
                record["citations"] = list(record.get("final_evidence_ids") or [])
                improved_records.append(record)
            except Exception as exc:  # never abort the run on one question
                run_errors.append(f"improved:{question['id']}: {exc}")
                improved_records.append(
                    {
                        "pipeline": "improved",
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

        fingerprint_after = _corpus_fingerprint(store)
        corpus_state_check.update(
            {
                "before": fingerprint_before,
                "between_pipelines": fingerprint_mid,
                "after": fingerprint_after,
                "mismatch": not (
                    fingerprint_before == fingerprint_mid == fingerprint_after
                ),
            }
        )
        if corpus_state_check["mismatch"]:
            corpus_state_check["warning"] = (
                "Corpus state changed between pipeline runs; "
                "baseline-vs-improved numbers may reflect mismatched data."
            )

    corpus_state_check: dict[str, Any] = {
        "before": fingerprint_before,
        "between_pipelines": None,
        "after": None,
        "mismatch": False,
    }
    try:
        await _run_all()
    finally:
        # Partial results are always aggregated and persisted, even when the
        # process is interrupted partway through the benchmark.
        baseline_agg = _aggregate_pipeline(
            [r for r in baseline_records if "metrics" in r]
            + [
                {
                    "expected_sources": r.get("expected_sources", []),
                    "retrieved_source_ids": [],
                }
                for r in baseline_records
                if "metrics" not in r
            ]
        ) if baseline_records else _aggregate_pipeline([])
        improved_agg = _aggregate_pipeline(
            [r for r in improved_records if "metrics" in r]
            + [
                {
                    "expected_sources": r.get("expected_sources", []),
                    "retrieved_source_ids": [],
                }
                for r in improved_records
                if "metrics" not in r
            ]
        ) if improved_records else _aggregate_pipeline([])
        differences = _compute_differences(baseline_agg, improved_agg)

        run_timestamp = datetime.now(timezone.utc)
        result: dict[str, Any] = {
            "run": {
                "timestamp": run_timestamp.isoformat(),
                "benchmark_name": benchmark.get("name", ""),
                "benchmark_version": benchmark.get("version", ""),
                "benchmark_path": str(BENCHMARK_PATH),
                "question_count": len(questions),
                "api_available": adk_available,
                "skip_judge": skip_judge,
                "judge_model": judge.model if judge else "none",
                "embedding": embed_info,
                "k_candidates": k_candidates,
                "final_k": final_k,
                "baseline_top_k": baseline_k,
                "baseline_top_k_matches_final_k": baseline_k == final_k,
                "repo_final_k": RERANK_FINAL_K,
                "repo_candidate_k": RERANK_CANDIDATE_K,
            },
            "aggregates": {
                "baseline": baseline_agg,
                "improved": improved_agg,
                "differences_improved_minus_baseline": differences,
                "errors": run_errors,
            },
            "corpus_state_check": corpus_state_check,
            "per_question": {
                "baseline": baseline_records,
                "improved": improved_records,
            },
        }

        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = (
            results_dir / f"comparison_{run_timestamp.strftime('%Y%m%dT%H%M%S')}.json"
        )
        results_path.write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
        )
        result["_results_path"] = str(results_path)

    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: baseline-vs-improved comparison over the fixed benchmark."
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
        help="Improved pipeline candidate count (default 12 = RERANK_CANDIDATE_K).",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
        help="Final evidence count for improved AND baseline top-K (default 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the comparison. Called from ``__main__`` and from tests."""
    from run_evaluation import _load_benchmark

    args = _parse_args(argv)
    benchmark = _load_benchmark(Path(args.benchmark))
    return asyncio.run(
        _run_comparison(
            benchmark,
            results_dir=Path(args.results_dir),
            limit=args.limit,
            skip_judge=args.skip_judge,
            k_candidates=args.k_candidates,
            final_k=args.final_k,
        )
    )


if __name__ == "__main__":
    result = main()
    path = result.get("_results_path", "")
    base = result["aggregates"]["baseline"]["retrieval"]
    imp = result["aggregates"]["improved"]["retrieval"]
    diffs = result["aggregates"]["differences_improved_minus_baseline"]
    print(f"Done - {result['run']['question_count']} questions per pipeline. Results: {path}")
    print(f"  baseline K={result['run']['baseline_top_k']} "
          f"(matches final_k={result['run']['final_k']}: "
          f"{result['run']['baseline_top_k_matches_final_k']})")
    for label, agg in (("baseline", base), ("improved", imp)):
        print(
            f"  {label}: Recall@5={agg['recall_at_5']:.4f}  "
            f"Recall@10={agg['recall_at_10']:.4f}  MRR={agg['mrr']:.4f}  "
            f"(answerable={agg['answerable_count']})"
        )
    print(
        f"  diff (improved-baseline): Recall@5={diffs['recall_at_5']:+.4f}  "
        f"Recall@10={diffs['recall_at_10']:+.4f}  MRR={diffs['mrr']:+.4f}"
    )
    for side in ("baseline", "improved"):
        j = result["aggregates"][side]["judge"]
        if j.get("correctness_mean") is not None:
            print(
                f"  judge {side} (n={j['available']}/{j['total']}): "
                f"correctness={j['correctness_mean']:.4f}  "
                f"groundedness={j['groundedness_mean']:.4f}  "
                f"citation_correctness={j['citation_correctness_mean']:.4f}"
            )
        else:
            print(f"  judge {side}: {j['unavailable']} unavailable of {j['total']}")
    if result["corpus_state_check"].get("mismatch"):
        print(f"  WARNING: {result['corpus_state_check'].get('warning')}")
    if result["aggregates"]["errors"]:
        print(f"  Errors: {result['aggregates']['errors']}")
