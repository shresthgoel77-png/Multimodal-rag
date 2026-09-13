"""Phase 6 evaluation metrics for the fixed Phase 5 benchmark.

This module owns the metric *definitions* and the LLM-judge generation
scoring. It performs no benchmark I/O and holds no Phase 1-4 state, so it can
be imported by tests with zero side effects. The end-to-end harness that runs
every benchmark question through the real pipeline lives in
``evaluation/run_evaluation.py``; this module only computes numbers from the
data the harness feeds it.

Matching rule (documented and applied consistently everywhere)
--------------------------------------------------------------
``expected_sources`` in the benchmark are **source-level** ids (16-hex SHA-256
of title+content, see ``evaluation/corpus/manifest.json``). Retrieval returns
**chunk-level** candidates whose metadata carries ``source_id``. Every metric
here therefore *deduplicates retrieved candidates by ``source_id`` preserving
first-seen order* and then compares source ids against expected source ids.
Chunk identity is irrelevant to the ground truth, and matching at the source
level is exactly what Phase 5's ``verify_benchmark.py`` did, so Recall@K/MRR
are comparable with the Phase 5 verification and are not distorted by how a
source happens to be chunked.

Scoring definitions (consistent 0-1 scale, applied to every question)
---------------------------------------------------------------------
- **correctness**: how much of the information conveyed by ``expected_answer``
  is also conveyed by the generated answer. 1.0 = the answer conveys all the
  required information with no contradicting content; 0.0 = none of the
  required information and/or outright contradiction; intermediate values for
  partial coverage (some required facts present, minor omissions, harmless
  extra detail).
- **groundedness**: whether every claim in the generated answer is supported by
  the evidence that generation actually used. 1.0 = all claims are supported by
  the provided evidence (or the answer is an explicit insufficiency statement
  with no usable evidence, which is grounded by construction); 0.0 = claims are
  invented or unsupported by the evidence.
- **citation_correctness**: whether the cited evidence (the evidence list the
  answer was generated from) actually contains the sources that support the
  answer's claims, i.e. whether the citations shown match evidence that
  supports the claims rather than unrelated or non-supporting chunks. 1.0 =
  every claim is attributable to one of the cited evidence items, 0.0 = claims
  cite evidence that does not support them.

A failed, timed-out, or schema-invalid judge result is recorded as
**unavailable** (never zero, never silently dropped), so a poor score and a
broken judgment remain distinguishable.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3-flash-preview")
JUDGE_TIMEOUT_SECONDS = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "30"))

JUDGE_AVAILABLE_KEYS = ("correctness", "groundedness", "citation_correctness")

EVIDENCE_CHAR_LIMIT = 1200

# Structured-output contract requested from the judge.
JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correctness": {"type": "number"},
        "groundedness": {"type": "number"},
        "citation_correctness": {"type": "number"},
    },
    "required": ["correctness", "groundedness", "citation_correctness"],
}


class MetricError(Exception):
    """Base class for evaluation-metric failures."""


class JudgeError(Exception):
    """Base class for LLM-judge failures."""


class JudgeTimeoutError(JudgeError):
    """Raised when the judge call exceeds its configured timeout."""


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def source_id_sequence(candidates: list[dict[str, Any]]) -> list[str]:
    """Map an ordered chunk candidate list to a deduplicated ordered source-id
    sequence.

    Order is preserved from the first time a source id appears; later chunks
    from the same source are dropped. This is the ranking the retrieval
    metrics score against (see module docstring for the matching rule).
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates or []:
        source_id = candidate.get("source_id")
        if not source_id:
            continue
        if source_id in seen:
            continue
        seen.add(source_id)
        ordered.append(source_id)
    return ordered


def recall_at_k(
    expected_source_ids: list[str], retrieved_source_ids: list[str], k: int
) -> float:
    """Fraction of ``expected_source_ids`` found among the top ``k`` retrieved
    source ids.

    Standard Recall@K = (# relevant retrieved up to rank K) / (# relevant
    total), where "relevant" = a matching expected source id. 0.0 when there
    are no expected sources or k <= 0. Empty retrieval is handled naturally as
    zero relevant results found.
    """
    if k <= 0:
        return 0.0
    expected = set(expected_source_ids)
    if not expected:
        return 0.0
    top_k = retrieved_source_ids[:k]
    hits = sum(1 for source_id in top_k if source_id in expected)
    return round(hits / len(expected), 4)


def mrr(expected_source_ids: list[str], retrieved_source_ids: list[str]) -> float:
    """Mean-Reciprocal-Rank contribution for one question.

    Returns 1 / rank of the first retrieved source id that matches an expected
    source, or 0.0 when no relevant result appears anywhere in the ranking
    (including an entirely empty retrieval).
    """
    expected = set(expected_source_ids)
    for rank, source_id in enumerate(retrieved_source_ids, start=1):
        if source_id in expected:
            return round(1.0 / rank, 4)
    return 0.0


def precision_at_k(
    expected_source_ids: list[str], retrieved_source_ids: list[str], k: int
) -> float:
    """Precision@K for one question: fraction of the top ``k`` retrieved source
    ids that are relevant.

    Denominator is len(top_k) so a question with, say, only 6 retrieved sources
    is scored against those 6 rather than being penalized for K's absence.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_source_ids[:k]
    if not top_k:
        return 0.0
    expected = set(expected_source_ids)
    hits = sum(1 for source_id in top_k if source_id in expected)
    return round(hits / len(top_k), 4)


# ---------------------------------------------------------------------------
# Aggregation over answerable questions
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def aggregate_retrieval_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question retrieval metrics across answerable questions.

    ``records`` are the per-question result records produced by the harness
    (each with ``expected_sources`` and a parallel list ``retrieved_source_ids``
    — the deduplicated retrieved source-id ranking). Unanswerable questions are
    excluded from all aggregation here; callers aggregate unanswerable
    handling separately (see runner).
    """
    answerable = []
    for record in records:
        if not record.get("expected_sources"):
            continue
        expected = list(record.get("expected_sources") or [])
        retrieved = list(record.get("retrieved_source_ids") or [])
        answerable.append(
            {
                "recall@5": recall_at_k(expected, retrieved, 5),
                "recall@10": recall_at_k(expected, retrieved, 10),
                "mrr": mrr(expected, retrieved),
                "precision@5": precision_at_k(expected, retrieved, 5),
                "precision@10": precision_at_k(expected, retrieved, 10),
            }
        )
    if not answerable:
        return {
            "answerable_count": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "precision_at_5": 0.0,
            "precision_at_10": 0.0,
        }
    return {
        "answerable_count": len(answerable),
        "recall_at_5": _mean([item["recall@5"] for item in answerable]),
        "recall_at_10": _mean([item["recall@10"] for item in answerable]),
        "mrr": _mean([item["mrr"] for item in answerable]),
        "precision_at_5": _mean([item["precision@5"] for item in answerable]),
        "precision_at_10": _mean([item["precision@10"] for item in answerable]),
    }


# ---------------------------------------------------------------------------
# LLM-judge generation metrics
# ---------------------------------------------------------------------------


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "".join(parts)


def _extract_json(text: str) -> Any:
    """Parse the judge's text output as JSON, tolerating code fences.

    Returns None when the text does not contain valid JSON so callers treat it
    as a validation failure. Mirrors the tolerant parser used by the router,
    reranker, and verifier.
    """
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def validate_judge_output(parsed: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate raw judge output against the fixed contract.

    Rules: object; all three keys present; each a number (not a bool) in
    [0.0, 1.0]; out-of-range or non-numeric values invalidate the whole output
    (returned as None + notes), so a broken judgment is never silently clamped
    into a fake score or dropped.

    Returns (cleaned_result, notes); ``cleaned_result`` is None when the
    output is unusable.
    """
    notes: list[str] = []
    if not isinstance(parsed, dict):
        return None, [f"Judge output was not an object: {parsed!r}."]
    missing = [key for key in JUDGE_AVAILABLE_KEYS if key not in parsed]
    if missing:
        return None, [f"Judge output is missing required fields: {missing}."]

    cleaned: dict[str, Any] = {}
    for key in JUDGE_AVAILABLE_KEYS:
        value = parsed[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, [f"Judge score {key!r} is not numeric: {value!r}."]
        if not (0.0 <= value <= 1.0):
            return None, [
                f"Judge score {key!r} is out of range [0, 1]: {value!r} "
                "(rejected rather than clamped so a broken judgment is never "
                "scored as a real number)."
            ]
        cleaned[key] = round(float(value), 4)
    return cleaned, notes


def build_judge_prompt(
    question: str,
    expected_answer: str,
    answer: str,
    evidence: list[dict[str, Any]],
    insufficient_evidence: bool,
) -> str:
    """Build the structured judge prompt.

    ``evidence`` is the final evidence set generation actually used (each item
    carrying at least id/title/text); ``insufficient_evidence`` tells the judge
    the answer may legitimately be a stated insufficiency instead of a factual
    response.
    """
    blocks = []
    for index, item in enumerate(evidence, start=1):
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()[:EVIDENCE_CHAR_LIMIT]
        title = str(item.get("title") or "")
        blocks.append(f"[{index}] id={item.get('id')} | title: {title}\n{text}")
    cited_ids = ", ".join(str(item.get("id") or "") for item in evidence)

    if insufficient_evidence:
        answer_note = (
            "The system determined the retrieved evidence was insufficient to "
            "answer the question, so the answer is expected to be (or resemble) "
            "an explicit statement of insufficiency rather than a factual answer."
        )
    else:
        answer_note = (
            "The system considered the retrieved evidence sufficient and "
            "answered from it."
        )

    return (
        "You are an evaluator judging the output of a retrieval-augmented "
        "generation system. Score three aspects on a 0.0 to 1.0 scale. Return "
        "ONLY a JSON object with exactly these three number fields, each in "
        "[0.0, 1.0]:\n"
        '{"correctness": <0.0 to 1.0>, "groundedness": <0.0 to 1.0>, '
        '"citation_correctness": <0.0 to 1.0>}\n\n'
        "DEFINITIONS:\n"
        "- correctness: How much of the information in the EXPECTED ANSWER the "
        "GENERATED ANSWER conveys. 1.0 = it conveys all the required "
        "information with no contradiction; 0.0 = none of it or outright "
        "contradiction; intermediate values for partial coverage.\n"
        "- groundedness: Whether every claim in the GENERATED ANSWER is "
        "supported by the EVIDENCE below (the evidence the answer was "
        "generated from). 1.0 = all claims supported (or the answer is an "
        "explicit insufficiency statement and the evidence is "
        "empty/below-threshold, which is grounded by construction); 0.0 = "
        "claims invented or unsupported by this evidence.\n"
        "- citation_correctness: Whether the CITATIONS (the evidence chunk ids "
        "the answer cites) actually contain the evidence that supports the "
        "answer's claims. 1.0 = every claim is attributable to one of the "
        "cited evidence chunks; 0.0 = the cited chunks do not support the "
        "claims.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED ANSWER (ground truth):\n{expected_answer}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        f"CITATIONS (evidence chunk ids used for generation):\n{cited_ids}\n\n"
        f"EVIDENCE ({len(evidence)} chunks):\n"
        + "\n\n".join(blocks)
        + "\n\n"
        f"ANSWER CONTEXT:\n{answer_note}\n\n"
        "Score each aspect against the DEFINITIONS only. Do not use outside "
        "knowledge to assign or withhold points."
    )


class LLMJudge:
    """Structured Gemini judge scoring correctness/groundedness/citations.

    Follows the same client-plus-threaded-timeout pattern used by the reranker
    and verifier. ``judge`` never raises; any failure, timeout, empty response,
    malformed JSON, or schema-invalid output is returned as
    ``{"status": "unavailable", ...}`` so the harness can record the question's
    generation metrics as unavailable.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or JUDGE_MODEL
        self.timeout = timeout if timeout is not None else JUDGE_TIMEOUT_SECONDS
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if client is not None:
            self.client = client
        elif api_key:
            from google import genai
            from google.genai import types

            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
        else:
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def judge(
        self,
        *,
        question: str,
        expected_answer: str,
        answer: str,
        evidence: list[dict[str, Any]],
        insufficient_evidence: bool = False,
    ) -> dict[str, Any]:
        """Score one question. Never raises.

        Returns:
          status              - "ok" or "unavailable"
          reason              - explainer ("" on success)
          correctness         - float 0-1 or None
          groundedness        - float 0-1 or None
          citation_correctness - float 0-1 or None
          errors              - validation notes / failure details
        """
        default: dict[str, Any] = {
            "status": "unavailable",
            "reason": "",
            "correctness": None,
            "groundedness": None,
            "citation_correctness": None,
            "errors": [],
        }
        if not self.available:
            return {
                **default,
                "reason": f"Judge unavailable ({self.model}); judgment not run.",
            }
        if not answer or not answer.strip():
            return {
                **default,
                "reason": "Judge skipped: no generated answer to score.",
            }

        try:
            text = self._generate_with_timeout(
                build_judge_prompt(
                    question=question,
                    expected_answer=expected_answer,
                    answer=answer,
                    evidence=evidence,
                    insufficient_evidence=insufficient_evidence,
                )
            )
        except JudgeTimeoutError as exc:
            return {
                **default,
                "reason": f"Judge call timed out after {self.timeout}s.",
                "errors": [str(exc)],
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            return {
                **default,
                "reason": f"Judge call failed ({exc}).",
                "errors": [str(exc)],
            }

        if not text.strip():
            return {
                **default,
                "reason": "Judge output was empty.",
                "errors": ["Judge returned an empty response."],
            }
        parsed = _extract_json(text)
        if parsed is None:
            return {
                **default,
                "reason": "Judge output was not valid JSON.",
                "errors": ["Judge output was not valid JSON."],
            }
        cleaned, notes = validate_judge_output(parsed)
        if cleaned is None:
            return {
                **default,
                "reason": "Judge output failed schema validation.",
                "errors": notes,
            }
        return {
            "status": "ok",
            "reason": "",
            "correctness": cleaned["correctness"],
            "groundedness": cleaned["groundedness"],
            "citation_correctness": cleaned["citation_correctness"],
            "errors": notes,
        }

    def _generate_with_timeout(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("No Gemini client configured for judgment.")
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["value"] = self._client_generate(prompt)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                result["error"] = exc

        worker = threading.Thread(target=run, name="judge-generate", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise JudgeTimeoutError(f"Judge request exceeded {self.timeout}s timeout.")
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _client_generate(self, prompt: str) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=JUDGE_RESPONSE_SCHEMA,
            ),
        )
        return _extract_response_text(response).strip()