"""Gemini-based reranking stage for the multimodal RAG pipeline.

Phase 2 turns the retrieval flow into: Vector Search -> Reranking -> Generation.

This module owns the reranking call, the validation rules applied to its
structured output, and the final_k selection. It never re-embeds or re-queries
Chroma - it only scores the candidate set it is handed.

Configuration (matching the repo's os.getenv-at-module-level pattern):
  RERANK_MODEL            model used for reranking (default gemini-3-flash-preview)
  RERANK_CANDIDATE_K      how many candidates Chroma returns (default 12)
  RERANK_FINAL_K          how many evidence items generation receives (default 5)
  RERANK_TIMEOUT_SECONDS  hard timeout for the reranking call (default 30)
"""

import json
import os
import re
import threading
from typing import Any

from google import genai
from google.genai import types

RERANK_MODEL = os.getenv("RERANK_MODEL", "gemini-3-flash-preview")
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "12"))
RERANK_FINAL_K = int(os.getenv("RERANK_FINAL_K", "5"))
RERANK_TIMEOUT_SECONDS = float(os.getenv("RERANK_TIMEOUT_SECONDS", "30"))

RELEVANCE_MIN = 0.0
RELEVANCE_MAX = 1.0

# Structured-output contract requested from the reranker.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "relevance": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "relevance"],
            },
        }
    },
    "required": ["results"],
}


class RerankerError(Exception):
    """Base class for reranker failures."""


class RerankTimeoutError(RerankerError):
    """Raised when the reranking call exceeds its configured timeout."""


def _build_prompt(query: str, candidates: list[dict[str, Any]]) -> str:
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        text = re.sub(r"\s+", " ", str(candidate.get("text") or "")).strip()[:1000]
        blocks.append(f"[{index}] id={candidate['id']}\n{text}")
    return (
        "You are a relevance reranker for a retrieval-augmented generation system.\n"
        f"QUESTION:\n{query}\n\n"
        f"CANDIDATE CHUNKS ({len(candidates)}):\n"
        + "\n\n".join(blocks)
        + "\n\n"
        "Score how relevant each candidate chunk is to the question.\n"
        "Return ONLY a JSON object matching this exact structure:\n"
        '{"results": [{"id": "<candidate id>", "relevance": <0.0 to 1.0>, "reason": "<short explanation>"}]}\n'
        "Rules:\n"
        "- Include exactly one entry per candidate id; use the ids exactly as shown.\n"
        "- relevance: a number from 0.0 (not relevant) to 1.0 (directly answers the question).\n"
        "- reason: one short sentence.\n"
    )


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    parts = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "".join(parts)


def _extract_json(text: str) -> Any:
    """Parse the reranker's text output as JSON, tolerating code fences.

    Returns None when the text does not contain valid JSON so callers treat it
    as a validation failure.
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


def _select_final(
    candidates: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    final_k: int,
) -> list[dict[str, Any]]:
    """Order candidates and pick the top `final_k` for evidence.

    Scored candidates are ordered by relevance (desc), then similarity (desc),
    then id. Candidates the reranker did not score are excluded from the final
    set unless fewer than final_k scored candidates exist - in that case the
    remaining slots are filled from unscored candidates by similarity (desc),
    with relevance explicitly set to None.
    """
    scored: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    for candidate in candidates:
        entry = scores.get(candidate["id"])
        if entry is not None:
            scored.append(
                {
                    **candidate,
                    "relevance": entry["relevance"],
                    "reason": entry["reason"],
                }
            )
        else:
            unscored.append({**candidate, "relevance": None, "reason": None})

    scored.sort(key=lambda item: (-item["relevance"], -item["similarity"], item["id"]))
    selected = scored[:final_k]
    if len(selected) < final_k and unscored:
        unscored.sort(key=lambda item: (-item["similarity"], item["id"]))
        selected.extend(unscored[: final_k - len(selected)])
    return selected


def rerank_and_select(
    reranker: Any,
    query: str,
    candidates: list[dict[str, Any]],
    final_k: int,
) -> dict[str, Any]:
    """Rerank `candidates` against `query` and select the top `final_k`.

    Returns a dict with:
      evidence       - ordered selected evidence items, each with the candidate
                       fields plus "relevance" (float or None) and "reason"
      used_reranking - True when reranking succeeded (no fallback)
      fallback       - True when the reranking path was skipped or invalidated
      reason         - human-readable fallback explanation ("" when reranked)
      errors         - validation notes produced while cleaning reranker output
    """
    if not candidates:
        return {
            "evidence": [],
            "used_reranking": False,
            "fallback": False,
            "reason": "No candidates to rerank (empty corpus).",
            "errors": [],
        }

    fallback_reason = ""
    errors: list[str] = []
    scores: dict[str, dict[str, Any]] = {}

    rerank_attempted = reranker is not None and reranker.available
    if not rerank_attempted:
        fallback_reason = f"Reranker unavailable ({RERANK_MODEL}); using similarity ranking."
    else:
        try:
            outcome = reranker.score_candidates(query, candidates)
        except RerankTimeoutError:
            fallback_reason = f"Reranking timed out after {reranker.timeout}s; using similarity ranking."
        except Exception as exc:
            fallback_reason = f"Reranking failed ({exc}); using similarity ranking."
        else:
            scores = outcome.get("scores") or {}
            errors = list(outcome.get("errors") or [])
            if not scores:
                fallback_reason = "Reranking validation produced no usable scores; using similarity ranking."

    used_reranking = not fallback_reason
    evidence = _select_final(candidates, scores, final_k)
    return {
        "evidence": evidence,
        "used_reranking": used_reranking,
        "fallback": bool(fallback_reason),
        "reason": fallback_reason,
        "errors": errors,
    }


class GeminiReranker:
    """Reranks a fixed candidate set against a query using Gemini."""

    def __init__(
        self,
        client: Any = None,
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or RERANK_MODEL
        self.timeout = timeout if timeout is not None else RERANK_TIMEOUT_SECONDS
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if client is not None:
            self.client = client
        elif api_key:
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
        else:
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def score_candidates(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Call Gemini, validate its structured output, and return usable scores.

        Returns {"scores": {id: {"relevance": float, "reason": str|None}}, "errors": [...]}.

        Validation rules:
          - Output must be valid JSON shaped as {"results": [...]}.
          - Entries referencing IDs outside this candidate set are discarded.
          - Non-numeric relevance discards the entry.
          - Relevance outside [RELEVANCE_MIN, RELEVANCE_MAX] is clamped to the
            nearest boundary (never trusted as-is).
          - Duplicate entries for the same id keep the FIRST occurrence.
        Candidates the reranker omits are simply not scored here; final_k
        selection fills from them only when fewer than final_k entries qualify.
        """
        if not candidates:
            return {"scores": {}, "errors": []}
        valid_ids = {candidate["id"] for candidate in candidates}
        errors: list[str] = []

        prompt = _build_prompt(query, candidates)
        text = self._generate_with_timeout(prompt)
        if not text.strip():
            errors.append("Reranker returned an empty response.")
            return {"scores": {}, "errors": errors}

        parsed = _extract_json(text)
        if parsed is None:
            errors.append("Reranker output was not valid JSON.")
            return {"scores": {}, "errors": errors}
        if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
            errors.append("Reranker JSON did not match the expected schema.")
            return {"scores": {}, "errors": errors}

        scores: dict[str, dict[str, Any]] = {}
        for item in parsed["results"]:
            if not isinstance(item, dict):
                errors.append("Ignored a reranking result that was not an object.")
                continue
            candidate_id = item.get("id")
            if not isinstance(candidate_id, str) or candidate_id not in valid_ids:
                errors.append(f"Ignored reranking entry for unknown candidate id: {candidate_id!r}")
                continue
            relevance = item.get("relevance")
            if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
                errors.append(
                    f"Discarded reranking entry for {candidate_id!r}: relevance is not numeric: {relevance!r}"
                )
                continue
            relevance = float(relevance)
            if not (RELEVANCE_MIN <= relevance <= RELEVANCE_MAX):
                errors.append(
                    f"Clamped out-of-range relevance {relevance!r} for {candidate_id!r} "
                    f"to [{RELEVANCE_MIN}, {RELEVANCE_MAX}]."
                )
                relevance = min(RELEVANCE_MAX, max(RELEVANCE_MIN, relevance))
            if candidate_id in scores:
                errors.append(
                    f"Ignored duplicate reranking entry for {candidate_id!r}; kept the first occurrence."
                )
                continue
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                reason = None
            scores[candidate_id] = {"relevance": relevance, "reason": reason}

        if not scores:
            errors.append("Reranker returned no usable scores.")
        return {"scores": scores, "errors": errors}

    def _generate_with_timeout(self, prompt: str) -> str:
        """Run the Gemini call with a hard wall-clock timeout.

        A daemon worker thread enforces the timeout; the client is additionally
        created with an SDK-level HttpOptions timeout so a real request cannot
        hang indefinitely in the background after we give up on it.
        """
        if self.client is None:
            raise RuntimeError("No Gemini client configured for reranking.")
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["value"] = self._client_generate(prompt)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                result["error"] = exc

        worker = threading.Thread(target=run, name="rerank-generate", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise RerankTimeoutError(f"Reranking request exceeded {self.timeout}s timeout.")
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _client_generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        return _extract_response_text(response).strip()