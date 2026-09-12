from query_router import QueryRouter
from rag_store import MultimodalRagStore
from reranker import GeminiReranker


RAG_STORE = MultimodalRagStore()
RERANKER = GeminiReranker()
ROUTER = QueryRouter()
