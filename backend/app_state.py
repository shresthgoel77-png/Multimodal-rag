import os

from google import genai
from google.genai import types

from query_router import QueryRouter
from rag_store import MultimodalRagStore
from reranker import GeminiReranker

# The router and the reranker share one Gemini client instead of each building
# its own. Module-level wall-clock timeouts (RERANK_TIMEOUT_SECONDS and
# ROUTER_TIMEOUT_SECONDS) still fire first and are smaller than this SDK-level
# HTTP timeout; this only stops an underlying request from hanging forever in a
# background thread after a fallback has already been returned.
_SHARED_HTTP_TIMEOUT_MS = int(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "60000"))


def _shared_gemini_client() -> genai.Client | None:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return None
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=_SHARED_HTTP_TIMEOUT_MS),
    )


_gemini_client = _shared_gemini_client()

RAG_STORE = MultimodalRagStore()
RERANKER = GeminiReranker(client=_gemini_client)
ROUTER = QueryRouter(client=_gemini_client)