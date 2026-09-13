"""Unit tests for evaluation/metrics.py and the evaluation runner logic.

These tests cover:
  - Recall@K computed correctly for a synthetic case (K=5, K=10)
  - MRR computed correctly including no relevant result case
  - Unanswerable questions excluded from aggregation and scored separately
  - LLM-judge output validation: well-formed accepted, out-of-range rejected,
    malformed JSON handled as unavailable
  - Corpus-drift detection for a deleted source
  - Precision@K (since it was implemented)
  - Full integration run over the real Phase 5 benchmark (requires store)

Run:  python -m pytest evaluation/tests/test_metrics.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"
BENCHMARK_PATH = EVAL_DIR / "benchmark" / "benchmark_v1.json"
MANIFEST_PATH = EVAL_DIR / "corpus" / "manifest.json"


@pytest.fixture(autouse=True)
def _backend_on_path():
    """Make run_evaluation/metrics importable."""
    for directory in (EVAL_DIR, REPO_ROOT / "backend"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    yield

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------


def _make_candidate(source_id: str, similarity: float = 0.9) -> dict[str, Any]:
    """Build a minimal chunk candidate dict for metric tests."""
    return {
        "id": f"{source_id}::1",
        "source_id": source_id,
        "title": "test",
        "modality": "text",
        "text": "chunk text",
        "similarity": similarity,
    }


# A synthetic ranking: five unique sources, ordered by similarity
SYNTHETIC_CANDIDATES = [
    _make_candidate("src_A", 0.95),
    _make_candidate("src_B", 0.88),
    _make_candidate("src_C", 0.82),
    _make_candidate("src_A", 0.70),  # duplicate source, should be deduped
    _make_candidate("src_D", 0.65),
    _make_candidate("src_E", 0.50),
    _make_candidate("src_B", 0.45),  # duplicate source
    _make_candidate("src_F", 0.40),
    _make_candidate("src_G", 0.35),
    _make_candidate("src_H", 0.30),
    _make_candidate("src_I", 0.25),
    _make_candidate("src_J", 0.20),
]


# ---------------------------------------------------------------------------
# Metrics imports (module-level but no Chroma/Gemini side effects)
# ---------------------------------------------------------------------------

from metrics import (
    LLMJudge,
    aggregate_retrieval_metrics,
    mrr,
    precision_at_k,
    recall_at_k,
    source_id_sequence,
    validate_judge_output,
)


# ===================================================================
# Recall@K
# ===================================================================


class TestRecallAtK:
    """Test Recall@K on a hand-crafted case."""

    def test_deduped_source_sequence(self):
        """Deduplication preserves first-seen order."""
        ids = source_id_sequence(SYNTHETIC_CANDIDATES)
        assert ids == ["src_A", "src_B", "src_C", "src_D", "src_E",
                       "src_F", "src_G", "src_H", "src_I", "src_J"]

    def test_recall_at_5_all_found(self):
        """Expected sources all inside top-5 deduped retrieval."""
        ids = source_id_sequence(SYNTHETIC_CANDIDATES)
        # top-5 = [src_A, src_B, src_C, src_D, src_E]
        expected = ["src_A", "src_C", "src_E"]
        assert recall_at_k(expected, ids, 5) == 1.0

    def test_recall_at_5_partial(self):
        """One expected source falls outside top-5."""
        ids = source_id_sequence(SYNTHETIC_CANDIDATES)
        # src_G is at rank 7, outside top-5
        expected = ["src_A", "src_G"]
        assert recall_at_k(expected, ids, 5) == 0.5  # 1/2 found

    def test_recall_at_10_all_found(self):
        """Expected sources all inside top-10."""
        ids = source_id_sequence(SYNTHETIC_CANDIDATES)
        expected = ["src_A", "src_G", "src_J"]
        assert recall_at_k(expected, ids, 10) == 1.0

    def test_recall_at_10_one_missing(self):
        """src_G at rank 7, src_J at rank 10, src_K not in ranking."""
        ids = source_id_sequence(SYNTHETIC_CANDIDATES)
        expected = ["src_A", "src_G", "src_J", "src_K"]
        assert recall_at_k(expected, ids, 10) == 0.75  # 3/4

    def test_recall_at_10_two_missing(self):
        """Two expected sources not in retrieval at all."""
        ids = source_id_sequence(SYNTHETIC_CANDIDATES)
        expected = ["src_A", "src_MISSING1", "src_MISSING2"]
        assert recall_at_k(expected, ids, 10) == pytest.approx(1/3, abs=1e-3)

    def test_empty_retrieval(self):
        """No candidates retrieved → recall = 0."""
        assert recall_at_k(["src_A"], [], 5) == 0.0

    def test_empty_expected(self):
        """No expected sources → recall = 0 (defensive; shouldn't appear for answerable)."""
        assert recall_at_k([], ["src_A"], 5) == 0.0

    def test_k_zero(self):
        """k=0 → recall = 0."""
        assert recall_at_k(["src_A"], ["src_A"], 0) == 0.0

    def test_k_larger_than_retrieval(self):
        """K exceeds retrieval length; still works correctly."""
        ids = ["src_A", "src_B"]
        assert recall_at_k(["src_A", "src_B", "src_C"], ids, 10) == pytest.approx(2/3, abs=1e-3)

    def test_single_expected_at_rank_1(self):
        """MRR-like single-source case."""
        ids = ["src_X", "src_Y"]
        assert recall_at_k(["src_X"], ids, 5) == 1.0


# ===================================================================
# Precision@K
# ===================================================================


class TestPrecisionAtK:
    def test_precision_at_5_known(self):
        ids = source_id_sequence(SYNTHETIC_CANDIDATES)
        expected = ["src_A", "src_B", "src_C", "src_D", "src_E"]
        # top-5 ids all relevant → precision = 5/5 = 1.0
        assert precision_at_k(expected, ids, 5) == 1.0

    def test_precision_at_5_partial(self):
        ids = source_id_sequence(SYNTHETIC_CANDIDATES)
        expected = ["src_A", "src_C"]  # 2 relevant; top-5 = 5 ids
        assert precision_at_k(expected, ids, 5) == pytest.approx(2/5, abs=1e-3)

    def test_empty_retrieval(self):
        assert precision_at_k(["src_A"], [], 5) == 0.0


# ===================================================================
# MRR
# ===================================================================


class TestMRR:
    def test_first_rank(self):
        """Expected source at rank 1 → RR = 1.0."""
        ids = ["src_A", "src_B", "src_C"]
        assert mrr(["src_A"], ids) == 1.0

    def test_second_rank(self):
        """Expected source at rank 2 → RR = 0.5."""
        ids = ["src_B", "src_A", "src_C"]
        assert mrr(["src_A"], ids) == pytest.approx(0.5, abs=1e-4)

    def test_fourth_rank(self):
        """Expected source at rank 4 → RR = 0.25."""
        ids = ["src_B", "src_C", "src_D", "src_A"]
        assert mrr(["src_A"], ids) == pytest.approx(0.25, abs=1e-4)

    def test_multiple_expected_first_relevant_rank2(self):
        """Multiple expected sources; RR based on the first hit."""
        ids = ["src_B", "src_A", "src_C"]
        assert mrr(["src_A", "src_C"], ids) == pytest.approx(0.5, abs=1e-4)

    def test_no_relevant_result(self):
        """No expected source appears → RR = 0."""
        ids = ["src_X", "src_Y"]
        assert mrr(["src_MISSING"], ids) == 0.0

    def test_empty_retrieval(self):
        """Empty retrieval → RR = 0."""
        assert mrr(["src_A"], []) == 0.0

    def test_empty_expected(self):
        """Empty expected → RR = 0."""
        assert mrr([], ["src_A"]) == 0.0


# ===================================================================
# Aggregate retrieval metrics
# ===================================================================


class TestAggregateRetrievalMetrics:
    def test_answerable_only(self):
        """Unanswerable questions are excluded from aggregation."""
        records = [
            # Answerable
            {
                "expected_sources": ["src_A"],
                "retrieved_source_ids": ["src_A", "src_B"],
            },
            # Unanswerable — excluded
            {
                "expected_sources": [],
                "retrieved_source_ids": ["src_X"],
            },
            # Answerable
            {
                "expected_sources": ["src_C"],
                "retrieved_source_ids": ["src_B", "src_C"],
            },
        ]
        agg = aggregate_retrieval_metrics(records)
        assert agg["answerable_count"] == 2
        # Record 1: recall@5=1.0 (src_A at rank 1), mrr=1.0
        # Record 3: recall@5=1.0 (src_C at rank 2), mrr=0.5
        assert agg["recall_at_5"] == pytest.approx(1.0, abs=1e-4)
        assert agg["mrr"] == pytest.approx(0.75, abs=1e-4)

    def test_all_unanswerable(self):
        records = [
            {"expected_sources": [], "retrieved_source_ids": ["src_A"]},
            {"expected_sources": [], "retrieved_source_ids": ["src_B"]},
        ]
        agg = aggregate_retrieval_metrics(records)
        assert agg["answerable_count"] == 0
        assert agg["recall_at_5"] == 0.0
        assert agg["mrr"] == 0.0


# ===================================================================
# LLM-judge output validation
# ===================================================================


class TestValidateJudgeOutput:
    def test_well_formed_accepted(self):
        parsed = {"correctness": 1.0, "groundedness": 0.85, "citation_correctness": 0.9}
        cleaned, notes = validate_judge_output(parsed)
        assert cleaned is not None
        assert cleaned["correctness"] == 1.0
        assert cleaned["groundedness"] == 0.85
        assert cleaned["citation_correctness"] == 0.9
        assert notes == []

    def test_all_zeros_accepted(self):
        parsed = {"correctness": 0.0, "groundedness": 0.0, "citation_correctness": 0.0}
        cleaned, notes = validate_judge_output(parsed)
        assert cleaned is not None
        assert all(cleaned[k] == 0.0 for k in ("correctness", "groundedness", "citation_correctness"))

    def test_out_of_range_rejected(self):
        """Out-of-range value (1.5) causes the output to be rejected, not clamped."""
        parsed = {"correctness": 1.5, "groundedness": 0.8, "citation_correctness": 0.9}
        cleaned, notes = validate_judge_output(parsed)
        assert cleaned is None
        assert len(notes) >= 1
        assert "out of range" in notes[0].lower() or "out-of-range" in notes[0].lower()

    def test_negative_rejected(self):
        parsed = {"correctness": -0.1, "groundedness": 0.5, "citation_correctness": 0.5}
        cleaned, notes = validate_judge_output(parsed)
        assert cleaned is None

    def test_missing_key_rejected(self):
        parsed = {"correctness": 0.8, "groundedness": 0.8}  # missing citation_correctness
        cleaned, notes = validate_judge_output(parsed)
        assert cleaned is None
        assert any("citation_correctness" in n for n in notes)

    def test_non_numeric_rejected(self):
        parsed = {"correctness": "high", "groundedness": 0.8, "citation_correctness": 0.8}
        cleaned, notes = validate_judge_output(parsed)
        assert cleaned is None

    def test_bool_rejected(self):
        """Boolean should be rejected (not treated as numeric)."""
        parsed = {"correctness": True, "groundedness": 0.8, "citation_correctness": 0.8}
        cleaned, notes = validate_judge_output(parsed)
        assert cleaned is None

    def test_not_object_rejected(self):
        cleaned, notes = validate_judge_output("just a string")
        assert cleaned is None

    def test_none_rejected(self):
        cleaned, notes = validate_judge_output(None)
        assert cleaned is None

    def test_extra_fields_ignored(self):
        parsed = {
            "correctness": 0.8,
            "groundedness": 0.9,
            "citation_correctness": 1.0,
            "extra_field": 42,
        }
        cleaned, notes = validate_judge_output(parsed)
        assert cleaned is not None
        assert "extra_field" not in cleaned


# ===================================================================
# LLMJudge when client is unavailable
# ===================================================================


class TestLLMJudgeUnavailable:
    def test_judge_returns_unavailable_when_no_client(self):
        """Without a Gemini client, judge records unavailable."""
        judge = LLMJudge(client=None)
        assert not judge.available
        result = judge.judge(
            question="test?",
            expected_answer="yes",
            answer="yes",
            evidence=[],
        )
        assert result["status"] == "unavailable"
        assert result["correctness"] is None
        assert result["groundedness"] is None
        assert result["citation_correctness"] is None
        assert result["reason"] != ""


# ===================================================================
# Corpus drift detection
# ===================================================================


class TestCorpusDrift:
    def test_detect_corpus_drift_flag(self):
        """From run_evaluation._detect_corpus_drift"""
        from run_evaluation import _detect_corpus_drift

        live_ids = {"src_A", "src_B"}
        expected = ["src_A", "src_MISSING"]
        drift = _detect_corpus_drift(expected, live_ids)
        assert drift["present"] is True
        assert "src_MISSING" in drift["missing_sources"]

    def test_no_corpus_drift(self):
        from run_evaluation import _detect_corpus_drift

        live_ids = {"src_A", "src_B", "src_C"}
        expected = ["src_A", "src_B"]
        drift = _detect_corpus_drift(expected, live_ids)
        assert drift["present"] is False
        assert drift["missing_sources"] == []

    def test_unanswerable_no_drift(self):
        from run_evaluation import _detect_corpus_drift

        drift = _detect_corpus_drift([], {"any"})
        assert drift["present"] is False


# ===================================================================
# Unanswerable questions are excluded from Recall@K/MRR
# ===================================================================


class TestUnanswerableExclusion:
    def test_unanswerable_excluded_from_aggregate(self):
        records = [
            # Answerable
            {
                "question_id": "q01",
                "category": "factual",
                "expected_sources": ["src_A"],
                "retrieved_source_ids": ["src_A", "src_B"],
            },
            # Unanswerable — should be excluded
            {
                "question_id": "q24",
                "category": "unanswerable",
                "expected_sources": [],
                "retrieved_source_ids": ["src_X", "src_Y", "src_Z"],
            },
            # Answerable
            {
                "question_id": "q03",
                "category": "factual",
                "expected_sources": ["src_B"],
                "retrieved_source_ids": ["src_A", "src_B"],
            },
        ]
        agg = aggregate_retrieval_metrics(records)
        assert agg["answerable_count"] == 2
        assert agg["recall_at_5"] == pytest.approx(1.0, abs=1e-4)
        assert agg["mrr"] == pytest.approx(0.75, abs=1e-4)

    def test_unanswerable_separate_scoring(self):
        """A runner record for q24 has unanswerable_result not metrics."""
        record = {
            "question_id": "q24",
            "category": "unanswerable",
            "expected_sources": [],
            "metrics": {
                "retrieval": {
                    "recall_at_5": 0.0,
                    "recall_at_10": 0.0,
                    "mrr": 0.0,
                },
            },
            "unanswerable_result": {
                "passed": True,
                "insufficient_evidence": True,
                "retrieved_source_ids": ["src_A"],
                "final_evidence_count": 0,
            },
            "outcome": "unanswerable_handled",
        }
        # The aggregate should not include this record
        agg = aggregate_retrieval_metrics([record])
        assert agg["answerable_count"] == 0


# ===================================================================
# LLMJudge with a fake Gemini client (no API key needed)
# ===================================================================


class _FakeResponse:
    """Mimics google.genai GenerateContentResponse's .text attribute."""

    def __init__(self, text: str):
        self.text = text


class _SlowResponse:
    """A response whose generation blocks longer than any judge timeout."""

    def __init__(self, text: str, delay: float = 5.0):
        self._text = text
        self._delay = delay

    @property
    def text(self) -> str:
        import time

        time.sleep(self._delay)
        return self._text


class _FakeGeminiClient:
    """Returns a canned response; records whether generate was called."""

    def __init__(self, response: Any, *, delay: float = 0.0):
        self._response = response
        self._delay = delay
        self.generate_calls = []

        class _Models:
            def __init__(self, outer):
                self._outer = outer

            def generate_content(self, *args, **kwargs):
                self._outer.generate_calls.append((args, kwargs))
                return self._outer._response

        self.models = _Models(self)

    def __call__(self, *args, **kwargs):
        return None


JUDGE_EVIDENCE = [
    {
        "id": "48a12d8e4a0af28f::1",
        "source_id": "48a12d8e4a0af28f",
        "title": "Kelp Forests",
        "text": "Kelp anchors to the seafloor with a root-like grip called a holdfast.",
        "similarity": 0.12,
        "relevance": None,
    }
]


class TestLLMJudgeFakeClient:
    def test_judge_ok_with_well_formed_output(self):
        judge = LLMJudge(
            client=_FakeGeminiClient(
                _FakeResponse(
                    json.dumps(
                        {"correctness": 0.9, "groundedness": 0.8, "citation_correctness": 1.0}
                    )
                )
            )
        )
        result = judge.judge(
            question="What anchors kelp?",
            expected_answer="a holdfast",
            answer="kelp uses a holdfast",
            evidence=JUDGE_EVIDENCE,
        )
        assert result["status"] == "ok"
        assert result["correctness"] == 0.9
        assert result["groundedness"] == 0.8
        assert result["citation_correctness"] == 1.0
        assert result["reason"] == ""

    def test_judge_prompt_includes_inputs(self):
        """build_judge_prompt must pass question, expected answer, answer, evidence."""
        from metrics import build_judge_prompt

        prompt = build_judge_prompt(
            question="What anchors kelp?",
            expected_answer="a holdfast",
            answer="kelp uses a holdfast",
            evidence=JUDGE_EVIDENCE,
            insufficient_evidence=False,
        )
        assert "What anchors kelp?" in prompt
        assert "a holdfast" in prompt
        assert "kelp uses a holdfast" in prompt
        assert "48a12d8e4a0af28f::1" in prompt

    def test_judge_malformed_json_unavailable(self):
        judge = LLMJudge(
            client=_FakeGeminiClient(_FakeResponse("this is not json {{{"))
        )
        result = judge.judge(
            question="q",
            expected_answer="a",
            answer="b",
            evidence=[],
        )
        assert result["status"] == "unavailable"
        assert result["correctness"] is None
        assert "not valid JSON" in result["reason"]

    def test_judge_schema_invalid_unavailable(self):
        """Missing field or out-of-range → unavailable."""
        judge = LLMJudge(
            client=_FakeGeminiClient(
                _FakeResponse(
                    json.dumps({"correctness": 1.4, "groundedness": 0.5, "citation_correctness": 0.5})
                )
            )
        )
        result = judge.judge(
            question="q", expected_answer="a", answer="b", evidence=[]
        )
        assert result["status"] == "unavailable"
        assert result["correctness"] is None
        assert "schema validation" in result["reason"]

    def test_judge_timeout_unavailable(self):
        judge = LLMJudge(client=_FakeGeminiClient(object()), timeout=0.5)
        # Patch _client_generate to block longer than the timeout.
        original = judge._client_generate

        def slow_generate(prompt):
            import time

            time.sleep(3.0)
            return '{"correctness": 1, "groundedness": 1, "citation_correctness": 1}'

        judge._client_generate = slow_generate
        try:
            result = judge.judge(
                question="q", expected_answer="a", answer="b", evidence=[]
            )
            assert result["status"] == "unavailable"
            assert "timed out" in result["reason"]
        finally:
            judge._client_generate = original

    def test_judge_empty_answer_unavailable(self):
        judge = LLMJudge(client=_FakeGeminiClient(_FakeResponse("{}")))
        result = judge.judge(
            question="q", expected_answer="a", answer="", evidence=[]
        )
        assert result["status"] == "unavailable"
        assert "no generated answer" in result["reason"]


# ===================================================================
# Load and validate real benchmark (schema + source IDs in manifest)
# ===================================================================


class TestBenchmarkLoading:
    def test_benchmark_loads(self):
        data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        assert "questions" in data
        assert len(data["questions"]) == 25

    def test_all_expected_sources_in_manifest(self):
        data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        live_ids = {source["id"] for source in manifest["sources"]}
        for q in data["questions"]:
            for sid in q["expected_sources"]:
                assert sid in live_ids, (
                    f"{q['id']}: expected_source {sid} is not in the live store"
                )


# ===================================================================
# Full integration run over the real Phase 5 benchmark
# ===================================================================


class TestFullBenchmarkRun:
    """Requires the real ingested store in backend/chroma_db.

    Runs all 25 questions with judge unavailability (no API key), and
    confirms the results file has per-question records and aggregate metrics.
    Marked with pytest.mark.slow so CI can skip it.
    """

    @pytest.mark.slow
    def test_full_run_produces_results_file(self, tmp_path):
        from run_evaluation import main

        results = main(["--results-dir", str(tmp_path), "--skip-judge"])

        # Results file exists
        path = results["_results_path"]
        assert Path(path).exists()
        assert Path(path).suffix == ".json"

        # All 25 questions present
        assert len(results["per_question"]) == 25
        assert results["run"]["question_count"] == 25

        # Aggregate retrieval metrics exist
        agg = results["aggregates"]
        retrieval = agg["retrieval"]
        assert "answerable_count" in retrieval
        assert retrieval["answerable_count"] == 23  # 25 - 2 unanswerable
        assert 0.0 <= retrieval["recall_at_5"] <= 1.0
        assert 0.0 <= retrieval["recall_at_10"] <= 1.0
        assert 0.0 <= retrieval["mrr"] <= 1.0

        # Unanswerable aggregate
        unans = agg["unanswerable"]
        assert unans["count"] == 2
        assert unans["correctly_handled"] + unans["miscount"] == 2

        # Every question has either outcome=complete or an explicit marker
        for record in results["per_question"]:
            assert record.get("outcome") is not None
            assert record.get("question_id") is not None
            assert "metrics" in record or record.get("outcome") == "error"

    @pytest.mark.slow
    def test_small_run_smoke(self, tmp_path):
        """Run just 3 questions to verify no crashes."""
        from run_evaluation import main

        results = main(["--results-dir", str(tmp_path), "--limit", "3", "--skip-judge"])
        assert results["run"]["question_count"] == 3
        assert len(results["per_question"]) == 3
        for record in results["per_question"]:
            assert record.get("outcome") is not None
            assert "metrics" in record

    @pytest.mark.slow
    def test_results_file_roundtrip(self, tmp_path):
        """Verify results file is valid JSON and re-readable."""
        from run_evaluation import main

        results = main(["--results-dir", str(tmp_path), "--limit", "2", "--skip-judge"])
        path = results["_results_path"]
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded["run"]["question_count"] == 2
        assert isinstance(loaded["per_question"], list)
        assert len(loaded["per_question"]) == 2
