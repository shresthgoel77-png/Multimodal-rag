"""Heuristic grounding and citation verifier for the multimodal RAG pipeline.

Phase 4 runs a best-effort verification step after generation completes. It
checks that the important claims in the generated answer are supported by the
final evidence set that generation actually used (not the full corpus), and
that the citations shown alongside the answer correspond to real evidence ids
from that same set.

This is explicitly a heuristic, best-effort check - not a hallucination
detector and not a guarantee of correctness. It never blocks or withholds an
answer: if the verification call fails, times out, or produces output that
fails validation, the original answer is returned unchanged and the response
marks verification as unavailable.

Configuration (same os.getenv-at-module-level pattern as reranker.py and
query_router.py):
  VERIFIER_MODEL            model used for verification (default gemini-3-flash-preview)
  VERIFIER_TIMEOUT_SECONDS  hard timeout for the verification call (default 30)

Groundedness scale (the 0-1 score):
  score = (# claims supported) / (# claims) over the validated claim list, so
  1.0 means every important claim the verifier identified is supported by the
  final evidence set and 0.0 means none are. An answer with no claims at all is
  vacuously grounded (score 1.0, grounded True). "grounded" is True exactly
  when there are no unsupported claims, so grounded, score, claims, and
  unsupported_claims stay mutually consistent. The verifier model's per-claim
  supported verdicts are used; its aggregate grounded/score are validated for
  type/range but recomputed from the claims so the surfaced numbers always
  follow the defined scale consistently.

Validation rules applied before any verifier output is surfaced:
  - Output must be valid JSON shaped as
    {"grounded": bool, "score": number, "claims": [{"claim": str,
    "supported": bool, "evidence_ids": [...]}], "unsupported_claims": [...]}.
  - "grounded" must be a boolean and "score" a number; missing/typed-wrong
    required top-level fields make the output unusable.
  - A numeric "score" outside [SCORE_MIN, SCORE_MAX] is clamped to the nearest
    boundary (never trusted as-is) and recorded as a validation note.
  - Each claim must be an object with a non-empty string "claim", a boolean
    "supported", and a list "evidence_ids". Malformed entries are dropped.
  - Every "evidence_ids" entry must exist in the final evidence set used for
    this answer; unknown ids are discarded and flagged, never trusted.
  - A claim marked supported=True that ends up with zero valid evidence ids
    after filtering is demoted to unsupported: its support cannot be attributed
    to real evidence from the final set, so it must not count toward the
    passing score.
  - "unsupported_claims" is recomputed from the validated claims so it always
    matches the claims list instead of trusting a separate model field.
"""

import json
import os
import re
import threading
from typing import Any

from google import genai
from google.genai import types

VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "gemini-3-flash-preview")
VERIFIER_TIMEOUT_SECONDS = float(os.getenv("VERIFIER_TIMEOUT_SECONDS", "30"))

SCORE_MIN = 0.0
SCORE_MAX = 1.0

EVIDENCE_CHAR_LIMIT = 1400

# Structured-output contract requested from the verifier.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "score": {"type": "number"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "supported", "evidence_ids"],
            },
        },
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded", "score", "claims"],
}


class VerifierError(Exception):
    """Base class for verifier failures."""


class VerifierTimeoutError(VerifierError):
    """Raised when the verification call exceeds its configured timeout."""


def _build_prompt(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    citations: list[str] | None,
) -> str:
    blocks = []
    for index, item in enumerate(evidence, start=1):
        text = str(item.get("text") or item.get("evidence") or "")
        text = re.sub(r"\s+", " ", text).strip()[:EVIDENCE_CHAR_LIMIT]
        title = str(item.get("title") or "")
        blocks.append(f"[{index}] id={item.get('id')} | title: {title}\n{text}")

    if citations is None:
        citations = [str(item.get("id") or "") for item in evidence]
    citation_list = ", ".join(citations)

    return (
        "You are a grounding verifier for a retrieval-augmented generation "
        "system. Your only job is to check whether the generated ANSWER is "
        "grounded in the FINAL EVIDENCE SET below - the exact evidence that "
        "was used to generate the answer. This is a heuristic check, not a "
        "guarantee of correctness.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"FINAL EVIDENCE SET ({len(evidence)} chunks):\n"
        + "\n\n".join(blocks)
        + "\n\n"
        "CITATIONS SHOWN WITH THE ANSWER:\n"
        f"{citation_list}\n\n"
        "TASKS:\n"
        "1. Extract the important factual claims the ANSWER asserts. Do not "
        "list trivial restatements.\n"
        "2. For each claim, decide whether it is supported ONLY by the FINAL "
        "EVIDENCE SET above (never by outside knowledge). A claim is "
        '"supported" only when the evidence clearly backs it.\n'
        "3. For supported claims, cite the exact evidence ids from the FINAL "
        "EVIDENCE SET that support them. Never invent ids.\n"
        "4. Verify the CITATIONS SHOWN WITH THE ANSWER: each must be one of "
        "the evidence ids above; flag nothing else.\n"
        "5. unsupported_claims must list every claim you marked unsupported.\n"
        "Return ONLY a JSON object matching this exact structure:\n"
        '{"grounded": <true iff no unsupported claims>, "score": <proportion '
        'of claims supported, 0.0 to 1.0>, "claims": [{"claim": "...", '
        '"supported": <bool>, "evidence_ids": ["<exact id>"]}], '
        '"unsupported_claims": ["<claim text>"]}\n'
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
    """Parse the verifier's text output as JSON, tolerating code fences.

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


def validate_verifier_output(parsed: Any, valid_ids: set[str]) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate raw verifier output against the fixed contract.

    Returns (cleaned_result, notes). On an unusable output returns (None,
    notes) so callers mark verification unavailable. Notes carry every
    validation decision (dropped ids, clamped score, demoted claims) so the
    raw model output is never silently trusted.
    """
    notes: list[str] = []
    if not isinstance(parsed, dict):
        return None, ["Verifier output was not an object."]

    grounded = parsed.get("grounded")
    if not isinstance(grounded, bool):
        return None, [f"Verifier output 'grounded' is not a boolean: {grounded!r}."]

    score = parsed.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None, [f"Verifier output 'score' is not numeric: {score!r}."]
    if not (SCORE_MIN <= score <= SCORE_MAX):
        notes.append(
            f"Clamped out-of-range verification score {score!r} to [{SCORE_MIN}, {SCORE_MAX}]."
        )
        score = min(SCORE_MAX, max(SCORE_MIN, score))

    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        return None, ["Verifier output 'claims' is not a list."]

    raw_unsupported = parsed.get("unsupported_claims")
    if raw_unsupported is not None and not isinstance(raw_unsupported, list):
        return None, ["Verifier output 'unsupported_claims' is not a list."]

    claims: list[dict[str, Any]] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            notes.append("Ignored a verification claim that was not an object.")
            continue
        claim_text = item.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            notes.append("Ignored a verification claim without a non-empty 'claim' string.")
            continue
        supported = item.get("supported")
        if not isinstance(supported, bool):
            notes.append(
                f"Ignored a verification claim without a boolean 'supported': {claim_text!r}"
            )
            continue
        raw_ids = item.get("evidence_ids")
        if not isinstance(raw_ids, list):
            notes.append(
                f"Ignored a verification claim without a list 'evidence_ids': {claim_text!r}"
            )
            continue
        filtered: list[str] = []
        for evidence_id in raw_ids:
            if isinstance(evidence_id, str) and evidence_id in valid_ids:
                filtered.append(evidence_id)
            else:
                notes.append(
                    f"Discarded evidence_id not in the final evidence set: {evidence_id!r}"
                )
        if supported and not filtered:
            notes.append(
                "Demoted claim to unsupported because it cites no real evidence "
                f"from the final evidence set: {claim_text!r}"
            )
            supported = False
        claims.append(
            {"claim": claim_text.strip(), "supported": supported, "evidence_ids": filtered}
        )

    if raw_claims and not claims:
        notes.append("All verification claims were malformed; output is unusable.")
        return None, notes

    supported_count = sum(1 for claim in claims if claim["supported"])
    total = len(claims)
    if total == 0:
        derived_score = SCORE_MAX
        derived_grounded = True
    else:
        derived_score = round(supported_count / total, 4)
        derived_grounded = supported_count == total
    derived_unsupported = [claim["claim"] for claim in claims if not claim["supported"]]

    from_schema_unsupported = [u for u in (raw_unsupported or []) if isinstance(u, str) and u.strip()]
    if set(from_schema_unsupported) != set(derived_unsupported):
        notes.append(
            "Verifier 'unsupported_claims' did not match its claims list; "
            "unsupported_claims is recomputed from the validated claims."
        )

    return (
        {
            "grounded": derived_grounded,
            "score": derived_score,
            "claims": claims,
            "unsupported_claims": derived_unsupported,
        },
        notes,
    )


class GeminiVerifier:
    """Verifies a generated answer against the final evidence set using Gemini."""

    def __init__(
        self,
        client: Any = None,
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or VERIFIER_MODEL
        self.timeout = timeout if timeout is not None else VERIFIER_TIMEOUT_SECONDS
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

    def verify(
        self,
        question: str,
        answer: str,
        evidence: list[dict[str, Any]],
        citations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Verify `answer` against the final `evidence` set; never raises.

        Always returns a dict with:
          unavailable        - True when verification did not produce a result
          reason             - why ("" on success)
          grounded           - bool (None when unavailable)
          score              - float (None when unavailable)
          claims             - validated claim list
          unsupported_claims - list of unsupported claim texts
          errors             - validation notes / failure details

        Any call failure, timeout, malformed JSON, or schema-invalid output
        results in unavailable=True with the reason, so the caller can return
        the original answer unchanged. An empty evidence set is skipped (there
        is nothing to ground the answer in) and also reported as unavailable.
        """
        default = {
            "unavailable": True,
            "reason": "",
            "grounded": None,
            "score": None,
            "claims": [],
            "unsupported_claims": [],
            "errors": [],
        }
        if not answer or not answer.strip():
            return {**default, "reason": "Verification skipped: the answer is empty."}
        if not evidence:
            return {
                **default,
                "reason": "Verification skipped: no evidence was available to ground or verify this answer.",
            }
        valid_ids = {str(item.get("id") or "") for item in evidence}
        if not self.available:
            return {
                **default,
                "reason": f"Verifier unavailable ({self.model}); verification not run.",
            }

        citations_errors: list[str] = []
        unknown_citations = [c for c in (citations or []) if c not in valid_ids]
        if unknown_citations:
            citations_errors.append(
                f"Citations not present in the final evidence set: {unknown_citations}"
            )

        try:
            text = self._generate_with_timeout(_build_prompt(question, answer, evidence, citations))
        except VerifierTimeoutError as exc:
            return {
                **default,
                "reason": f"Verification timed out after {self.timeout}s.",
                "errors": citations_errors + [str(exc)],
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            return {
                **default,
                "reason": f"Verification call failed ({exc}).",
                "errors": citations_errors + [str(exc)],
            }

        if not text.strip():
            return {
                **default,
                "reason": "Verification output was empty.",
                "errors": citations_errors + ["Verifier returned an empty response."],
            }
        parsed = _extract_json(text)
        if parsed is None:
            return {
                **default,
                "reason": "Verification output was not valid JSON.",
                "errors": citations_errors + ["Verification output was not valid JSON."],
            }
        cleaned, notes = validate_verifier_output(parsed, valid_ids)
        if cleaned is None:
            return {
                **default,
                "reason": "Verification output failed schema validation.",
                "errors": citations_errors + notes,
            }
        errors = citations_errors + notes
        return {
            "unavailable": False,
            "reason": "",
            "grounded": cleaned["grounded"],
            "score": cleaned["score"],
            "claims": cleaned["claims"],
            "unsupported_claims": cleaned["unsupported_claims"],
            "errors": errors,
        }

    def _generate_with_timeout(self, prompt: str) -> str:
        """Run the Gemini call with a hard wall-clock timeout.

        A daemon worker thread enforces the timeout; the client is additionally
        created with an SDK-level HttpOptions timeout so a real request cannot
        hang indefinitely in the background after we give up on it.
        """
        if self.client is None:
            raise RuntimeError("No Gemini client configured for verification.")
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["value"] = self._client_generate(prompt)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                result["error"] = exc

        worker = threading.Thread(target=run, name="verifier-generate", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise VerifierTimeoutError(f"Verification request exceeded {self.timeout}s timeout.")
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


def run_verification(
    verifier: Any,
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    citations: list[str] | None = None,
) -> dict[str, Any]:
    """Top-level verification entry point used by the /ask endpoint.

    Never raises. Delegates to the verifier object; if the verifier itself is
    absent (None), verification is reported as unavailable so the endpoint can
    still return the original answer unchanged.
    """
    if verifier is None:
        return {
            "unavailable": True,
            "reason": "Verifier not configured; verification unavailable.",
            "grounded": None,
            "score": None,
            "claims": [],
            "unsupported_claims": [],
            "errors": [],
        }
    try:
        return verifier.verify(question, answer, evidence, citations=citations)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        return {
            "unavailable": True,
            "reason": f"Verification unexpectedly failed ({exc}).",
            "grounded": None,
            "score": None,
            "claims": [],
            "unsupported_claims": [],
            "errors": [str(exc)],
        }