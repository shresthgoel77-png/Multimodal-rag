"""Bounded adaptive query router for the multimodal RAG pipeline (Phase 3).

Before retrieval begins, one Gemini call classifies the incoming question into
exactly one of three fixed strategies - STANDARD, COMPLEX, or MULTI_HOP - using
a constrained prompt plus a structured-output schema. The router is deliberately
small: it can never invent new strategies, propose new tools, or change the
Retrieve -> Rerank -> Generate structure at runtime.

This module owns:
  - the router call, its timeout, and its structured-output validation
  - the candidate merge/dedup rule used by the MULTI_HOP strategy
  - the insufficient-evidence predicate applied after reranking

Configuration (same os.getenv-at-module-level pattern as reranker.py):
  ROUTER_MODEL                 model used for classification (default gemini-3-flash-preview)
  ROUTER_TIMEOUT_SECONDS       hard timeout for the router call (default 15)
  ROUTER_COMPLEX_CANDIDATE_K   candidate_k used by the COMPLEX strategy (default 24)
  ROUTER_MULTIHOP_CANDIDATE_K  candidate_k used per subquery in MULTI_HOP (default 12)
  ROUTER_RELEVANCE_THRESHOLD   minimum effective evidence score to avoid the
                               insufficient-evidence path (default 0.15)
"""

import json
import os
import re
import threading
from typing import Any

from google import genai
from google.genai import types

from reranker import RERANK_CANDIDATE_K

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gemini-3-flash-preview")
ROUTER_TIMEOUT_SECONDS = float(os.getenv("ROUTER_TIMEOUT_SECONDS", "15"))
ROUTER_COMPLEX_CANDIDATE_K = int(os.getenv("ROUTER_COMPLEX_CANDIDATE_K", "24"))
ROUTER_MULTIHOP_CANDIDATE_K = int(
    os.getenv("ROUTER_MULTIHOP_CANDIDATE_K", str(RERANK_CANDIDATE_K))
)
ROUTER_RELEVANCE_THRESHOLD = float(os.getenv("ROUTER_RELEVANCE_THRESHOLD", "0.15"))
ROUTER_MIN_SUBQUERIES = 2
ROUTER_MAX_SUBQUERIES = 3

STANDARD = "STANDARD"
COMPLEX = "COMPLEX"
MULTI_HOP = "MULTI_HOP"
ROUTER_STRATEGIES = (STANDARD, COMPLEX, MULTI_HOP)

# Structured-output contract requested from the router.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": list(ROUTER_STRATEGIES)},
        "reason": {"type": "string"},
        "subqueries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["strategy", "reason"],
}

ROUTER_PROMPT = (
    "You are the query router of a retrieval-augmented generation system.\n"
    "Your only job is to classify the user's question into exactly one of three "
    "predefined retrieval strategies. You must not invent new strategies, propose "
    "new tools, or alter the retrieval pipeline.\n\n"
    "The only allowed strategies are:\n"
    '- "STANDARD": a simple, single-fact question answerable from one retrieval pass.\n'
    '- "COMPLEX": a broad, open-ended, or multi-aspect question that benefits from retrieving a larger candidate set.\n'
    '- "MULTI_HOP": a question that genuinely requires combining two or three separate pieces of ingested information that are unlikely to appear in a single chunk. For this strategy you must provide 2-3 short, self-contained sub-questions.\n\n'
    "Classification rules:\n"
    "- Choose exactly one strategy. Prefer STANDARD unless the question is clearly broad (COMPLEX) or clearly requires combining evidence across multiple separate retrievals (MULTI_HOP).\n"
    "- For MULTI_HOP, provide exactly 2 or 3 sub-questions; each must be a short standalone retrieval query able to find one piece of evidence on its own.\n"
    "- For STANDARD and COMPLEX, the subqueries list must be empty.\n"
    '- "reason": one concise sentence explaining the classification.\n\n'
    "Return ONLY a JSON object matching this exact structure:\n"
    '{"strategy": "STANDARD" | "COMPLEX" | "MULTI_HOP", "reason": "...", "subqueries": []}\n'
)


class RouterError(Exception):
    """Base class for router failures."""


class RouterTimeoutError(RouterError):
    """Raised when the router call exceeds its configured timeout."""


def _build_prompt(query: str) -> str:
    return ROUTER_PROMPT + f"\nUSER QUESTION:\n{query}\n"


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
    """Parse the router's text output as JSON, tolerating code fences.

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


def validate_router_output(parsed: Any) -> dict[str, Any] | None:
    """Validate raw router output against the fixed contract.

    Rules:
      - Output must be an object; "strategy" must be exactly one of the three
        allowed values (case-sensitive, no fuzzy acceptance).
      - For MULTI_HOP, "subqueries" must be a list of non-empty strings; more
        than 3 non-empty entries are truncated to the first 3, fewer than 2
        invalidate the output.
      - For STANDARD / COMPLEX any present "subqueries" are discarded.

    Returns a cleaned result dict on success, or None when the output is
    invalid (which makes the caller fall back to STANDARD).
    """
    if not isinstance(parsed, dict):
        return None
    strategy = parsed.get("strategy")
    if strategy not in ROUTER_STRATEGIES:
        return None
    reason = parsed.get("reason")
    if not isinstance(reason, str):
        reason = ""
    if strategy == MULTI_HOP:
        raw_subqueries = parsed.get("subqueries")
        if not isinstance(raw_subqueries, list):
            return None
        subqueries: list[str] = []
        for subquery in raw_subqueries:
            if isinstance(subquery, str) and subquery.strip():
                subqueries.append(subquery.strip())
        if len(subqueries) < ROUTER_MIN_SUBQUERIES:
            return None
        subqueries = subqueries[:ROUTER_MAX_SUBQUERIES]
    else:
        subqueries = []
    return {"strategy": strategy, "reason": reason, "subqueries": subqueries}


def merge_candidates_keep_best(candidate_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge candidate sets from independent retrievals, deduplicating by chunk id.

    Dedup rule: a chunk id retrieved by more than one subquery appears exactly
    once in the merged set, keeping the occurrence with the highest
    "similarity"; on a tie the first-seen occurrence wins. Order is preserved
    from the first time an id was seen.
    """
    merged: dict[str, dict[str, Any]] = {}
    for candidates in candidate_sets:
        for candidate in candidates or []:
            candidate_id = candidate.get("id")
            current = merged.get(candidate_id)
            candidate_score = candidate.get("similarity", 0.0)
            if current is None or candidate_score > current.get("similarity", 0.0):
                merged[candidate_id] = candidate
    return list(merged.values())


def evidence_is_sufficient(evidence: list[dict[str, Any]], threshold: float) -> bool:
    """Decide whether the final evidence set is strong enough to answer.

    Insufficient when the set is empty, or when every item's effective score is
    below `threshold`. An item's effective score is its reranker "relevance"
    when present and numeric, otherwise its retrieval "similarity" (this covers
    the reranking-fallback path where relevance is None).
    """
    if not evidence:
        return False
    for item in evidence:
        relevance = item.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
            score = item.get("similarity", 0.0)
        else:
            score = relevance
        if score >= threshold:
            return True
    return False


class QueryRouter:
    """Classifies an incoming query into one of the three fixed strategies."""

    def __init__(
        self,
        client: Any = None,
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or ROUTER_MODEL
        self.timeout = timeout if timeout is not None else ROUTER_TIMEOUT_SECONDS
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

    def classify(self, query: str) -> dict[str, Any]:
        """Classify `query` into a strategy; never raises, never blocks a query.

        Always returns a dict with:
          strategy     - "STANDARD" / "COMPLEX" / "MULTI_HOP" (the one to use)
          reason       - the router's stated reason, or the fallback explanation
          subqueries   - list of subqueries actually used (empty unless MULTI_HOP)
          fell_back    - True when a failure/invalid output forced STANDARD
          error        - detail of the failure ("" on success)

        Any call failure, timeout, malformed JSON, or invalid strategy/subquery
        output falls back to STANDARD unconditionally with fell_back=True.
        """
        default = {
            "strategy": STANDARD,
            "reason": "",
            "subqueries": [],
            "fell_back": False,
            "error": "",
        }
        if not query or not query.strip():
            return {
                **default,
                "reason": "Empty question; using STANDARD.",
                "fell_back": True,
                "error": "Empty question.",
            }
        if not self.available:
            return {
                **default,
                "reason": f"Router unavailable ({ROUTER_MODEL}); using STANDARD.",
                "fell_back": True,
                "error": "No Gemini client configured for routing.",
            }

        try:
            text = self._generate_with_timeout(_build_prompt(query))
        except RouterTimeoutError as exc:
            return {
                **default,
                "reason": f"Router timed out after {self.timeout}s; using STANDARD.",
                "fell_back": True,
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - surfaced in the response
            return {
                **default,
                "reason": f"Router call failed ({exc}); using STANDARD.",
                "fell_back": True,
                "error": str(exc),
            }

        parsed = _extract_json(text)
        if parsed is None:
            return {
                **default,
                "reason": "Router output was not valid JSON; using STANDARD.",
                "fell_back": True,
                "error": "Router output was not valid JSON.",
            }
        validated = validate_router_output(parsed)
        if validated is None:
            return {
                **default,
                "reason": "Router output failed validation; using STANDARD.",
                "fell_back": True,
                "error": "Router output failed validation.",
            }
        return {**default, **validated}

    def _generate_with_timeout(self, prompt: str) -> str:
        """Run the Gemini call with a hard wall-clock timeout.

        A daemon worker thread enforces the timeout; the client is additionally
        created with an SDK-level HttpOptions timeout so a real request cannot
        hang indefinitely in the background after we give up on it.
        """
        if self.client is None:
            raise RuntimeError("No Gemini client configured for routing.")
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["value"] = self._client_generate(prompt)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                result["error"] = exc

        worker = threading.Thread(target=run, name="router-generate", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise RouterTimeoutError(f"Router request exceeded {self.timeout}s timeout.")
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _client_generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        return _extract_response_text(response).strip()