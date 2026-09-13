"""Phase 7 tests: baseline-vs-improved comparison orchestration.

Covers (per the phase spec):
  - baseline runs for a known question, returns evidence + citations
  - baseline top-K genuinely matches the improved pipeline's final_k
  - Recall@K/MRR/judge computed identically (same functions, same scale)
  - one question's generation failure does not abort the full run
  - final output holds both aggregates + real computed (not hardcoded) diffs
  - repeated runs against the unchanged corpus/benchmark behave consistently

Store-backed tests need the ingested Chroma store (like Phase 6's slow
integration tests) and are marked slow. Pure unit tests run anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"
BACKEND_DIR = REPO_ROOT / "backend"
BENCHMARK_PATH = EVAL_DIR / "benchmark" / "benchmark_v1.json"


@pytest.fixture(autouse=True)
def _paths():
    for directory in (EVAL_DIR, BACKEND_DIR):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    yield


def _load_benchmark() -> dict[str, Any]:
    from run_evaluation import _load_benchmark

    return _load_benchmark(BENCHMARK_PATH)


def _first_answerable(benchmark: dict[str, Any]) -> dict[str, Any]:
    for question in benchmark["questions"]:
        if question["category"] != "unanswerable":
            return question
    raise AssertionError("benchmark has no answerable question")


# ---------------------------------------------------------------------------
# Pure unit tests (no store, no API key)
# ---------------------------------------------------------------------------


class TestKContract:
    def test_baseline_k_equals_final_k(self):
        from run_comparison import _baseline_k

        for final_k in (1, 3, 5, 7):
            assert _baseline_k(final_k) == final_k

    def test_repo_final_k_is_five(self):
        """Documents the K value: RERANK_FINAL_K=5, used by both pipelines."""
        from reranker import RERANK_FINAL_K

        assert RERANK_FINAL_K == 5
        from run_comparison import _baseline_k

        assert _baseline_k(RERANK_FINAL_K) == 5


class TestDifferencesAreComputed:
    def _aggs(self, b_ret: dict, i_ret: dict) -> tuple[dict, dict]:
        base = {
            "retrieval": b_ret,
            "unanswerable": {"count": 2, "correctly_handled": 0, "miscount": 2},
            "judge": {"total": 1, "available": 0, "unavailable": 1},
        }
        imp = {
            "retrieval": i_ret,
            "unanswerable": {"count": 2, "correctly_handled": 1, "miscount": 1},
            "judge": {"total": 1, "available": 0, "unavailable": 1},
        }
        return base, imp

    def test_diff_is_improved_minus_baseline(self):
        from run_comparison import _compute_differences

        b, i = self._aggs(
            {"recall_at_5": 0.5, "recall_at_10": 0.5, "mrr": 0.25,
             "precision_at_5": 0.2, "precision_at_10": 0.1},
            {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 0.5,
             "precision_at_5": 0.1, "precision_at_10": 0.05},
        )
        diffs = _compute_differences(b, i)
        assert diffs["recall_at_5"] == pytest.approx(0.5)
        assert diffs["recall_at_10"] == pytest.approx(0.5)
        assert diffs["mrr"] == pytest.approx(0.25)
        # Negative diffs are preserved, never hidden.
        assert diffs["precision_at_5"] == pytest.approx(-0.1)
        assert diffs["precision_at_10"] == pytest.approx(-0.05)
        assert diffs["unanswerable_correctly_handled"] == 1

    def test_diff_none_when_judge_unavailable(self):
        from run_comparison import _compute_differences

        b, i = self._aggs(
            {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0,
             "precision_at_5": 0.5, "precision_at_10": 0.5},
            {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0,
             "precision_at_5": 0.5, "precision_at_10": 0.5},
        )
        diffs = _compute_differences(b, i)
        assert diffs["correctness_mean"] is None
        assert diffs["groundedness_mean"] is None
        assert diffs["citation_correctness_mean"] is None


class TestIdenticalMetricFunctions:
    def test_same_recall_mrr_functions_for_both_pipelines(self):
        """Both pipelines' metric dicts equal direct Phase 6 function output."""
        from metrics import mrr, recall_at_k, source_id_sequence
        from run_comparison import _aggregate_pipeline

        candidates = [
            {"id": "s1::1", "source_id": "s1"},
            {"id": "s2::1", "source_id": "s2"},
        ]
        seq = source_id_sequence(candidates)
        expected = ["s1"]
        records = [
            {"expected_sources": expected, "retrieved_source_ids": seq,
             "metrics": {"judge": {"status": "unavailable"}},
             "unanswerable": False},
        ]
        agg = _aggregate_pipeline(records)
        assert agg["retrieval"]["recall_at_5"] == recall_at_k(expected, seq, 5)
        assert agg["retrieval"]["recall_at_10"] == recall_at_k(expected, seq, 10)
        assert agg["retrieval"]["mrr"] == mrr(expected, seq)


# ---------------------------------------------------------------------------
# Store-backed tests (need backend/chroma_db ingested corpus)
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline():
    from run_evaluation import _load_shared_pipeline
    from run_evaluation import _store_source_ids

    store, reranker, router, verifier, _ = _load_shared_pipeline()
    live_ids = _store_source_ids(store)
    return store, reranker, router, verifier, live_ids


class TestBaselinePipeline:
    @pytest.mark.slow
    def test_baseline_runs_with_citations(self, pipeline):
        """Baseline returns evidence + same-format citations for a question."""
        import asyncio

        from run_comparison import _run_baseline_question

        store, _, _, _, live_ids = pipeline
        benchmark = _load_benchmark()
        question = _first_answerable(benchmark)
        record = asyncio.run(
            _run_baseline_question(
                question=question,
                store=store,
                judge=None,
                live_source_ids=live_ids,
                final_k=5,
                adk_available=False,
            )
        )
        assert record["pipeline"] == "baseline"
        assert record["question_id"] == question["id"]
        # Top-K evidence retrieved via raw similarity search.
        assert len(record["retrieved_ids"]) <= 5
        assert len(record["final_evidence_ids"]) <= 5
        # Citations present in the same chunk-id format (not skipped).
        assert record["citations"] == record["final_evidence_ids"]
        assert all("::" in cid for cid in record["citations"])
        # Per-stage latency recorded; router/rerank/verifier skipped by design.
        for key in ("retrieval_ms", "generation_ms", "judge_ms", "total_ms"):
            assert key in record["latency_ms"]
        assert record["verification"]["unavailable"] is True
        # Judge unavailable offline, recorded explicitly (never a fake zero).
        assert record["metrics"]["judge"]["status"] == "unavailable"

    @pytest.mark.slow
    def test_baseline_top_k_matches_final_k(self, pipeline):
        import asyncio

        from run_comparison import _run_baseline_question

        store, _, _, _, live_ids = pipeline
        benchmark = _load_benchmark()
        question = _first_answerable(benchmark)
        for final_k in (3, 5):
            record = asyncio.run(
                _run_baseline_question(
                    question=question,
                    store=store,
                    judge=None,
                    live_source_ids=live_ids,
                    final_k=final_k,
                    adk_available=False,
                )
            )
            assert record["top_k"] == final_k
            assert len(record["retrieved_ids"]) <= final_k

    @pytest.mark.slow
    def test_baseline_generation_failure_is_recorded(self, pipeline):
        """A generation blowup marks that question failed, sans exception."""
        import asyncio

        import run_evaluation
        from run_comparison import _run_baseline_question

        store, _, _, _, live_ids = pipeline
        benchmark = _load_benchmark()
        question = _first_answerable(benchmark)

        async def boom(*args, **kwargs):
            raise RuntimeError("simulated baseline generation blowup")

        real_generate = run_evaluation._generate_answer
        run_evaluation._generate_answer = boom
        try:
            record = asyncio.run(
                _run_baseline_question(
                    question=question,
                    store=store,
                    judge=None,
                    live_source_ids=live_ids,
                    final_k=5,
                    # Force the generation branch so the patched call fires.
                    adk_available=True,
                )
            )
        finally:
            run_evaluation._generate_answer = real_generate

        assert record["outcome"] == "generation_failed"
        assert record["generation_status"] == "failed"
        assert "blowup" in record["generation_error"]
        # Retrieval/citation evidence is still recorded, not dropped.
        assert record["retrieved_ids"]
        assert record["metrics"]["retrieval"]["recall_at_5"] >= 0.0

    @pytest.mark.slow
    def test_run_continues_after_single_question_failure(self, tmp_path, pipeline):
        """A retrieval failure on one question is recorded; the run continues."""
        import asyncio

        from run_comparison import _run_comparison

        store, _, _, _, _ = pipeline
        benchmark = _load_benchmark()
        subset = {
            "questions": [q for q in benchmark["questions"] if q["category"] != "unanswerable"][:3]
        }
        first_text = subset["questions"][0]["question"]

        real_retrieve = store.retrieve_candidates

        def flaky_retrieve(query: str, candidate_k: int = 12):
            if query == first_text:
                raise RuntimeError("simulated retrieval blowup")
            return real_retrieve(query, candidate_k)

        store.retrieve_candidates = flaky_retrieve
        try:
            result = asyncio.run(
                _run_comparison(
                    subset,
                    results_dir=tmp_path,
                    skip_judge=True,
                    k_candidates=12,
                    final_k=5,
                )
            )
        finally:
            store.retrieve_candidates = real_retrieve

        # All questions still present on both sides; errors listed, not raised.
        assert len(result["per_question"]["baseline"]) == 3
        assert len(result["per_question"]["improved"]) == 3
        assert any("blowup" in e for e in result["aggregates"]["errors"])
        assert Path(result["_results_path"]).exists()


class TestComparisonOutput:
    @pytest.mark.slow
    def test_output_has_both_aggregates_and_real_diffs(self, tmp_path):
        from run_comparison import main

        result = main(
            ["--results-dir", str(tmp_path), "--limit", "4", "--skip-judge"]
        )
        agg = result["aggregates"]
        assert "baseline" in agg and "improved" in agg
        for side in ("baseline", "improved"):
            ret = agg[side]["retrieval"]
            for key in ("recall_at_5", "recall_at_10", "mrr"):
                assert 0.0 <= ret[key] <= 1.0
        diffs = agg["differences_improved_minus_baseline"]
        # Differences are recomputed from the aggregates in-test: no hardcode.
        for key in ("recall_at_5", "recall_at_10", "mrr"):
            expected = round(
                agg["improved"]["retrieval"][key]
                - agg["baseline"]["retrieval"][key],
                4,
            )
            assert diffs[key] == pytest.approx(expected, abs=1e-9)
        assert result["run"]["baseline_top_k"] == result["run"]["final_k"]
        assert result["run"]["baseline_top_k_matches_final_k"] is True
        assert result["corpus_state_check"]["mismatch"] is False
        # Partial-run safety: every question accounted for on both sides.
        assert len(result["per_question"]["baseline"]) == 4
        assert len(result["per_question"]["improved"]) == 4

    @pytest.mark.slow
    def test_repeated_runs_are_structurally_consistent(self, tmp_path):
        """Two runs over the unchanged corpus give identical retrieval numbers."""
        from run_comparison import main

        first = main(["--results-dir", str(tmp_path), "--limit", "4", "--skip-judge"])
        second = main(["--results-dir", str(tmp_path), "--limit", "4", "--skip-judge"])
        for side in ("baseline", "improved"):
            assert first["aggregates"][side]["retrieval"] == (
                second["aggregates"][side]["retrieval"]
            )
        first_ids = [
            (r["question_id"], r["retrieved_ids"])
            for r in first["per_question"]["baseline"]
        ]
        second_ids = [
            (r["question_id"], r["retrieved_ids"])
            for r in second["per_question"]["baseline"]
        ]
        assert first_ids == second_ids
