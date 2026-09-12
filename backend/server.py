import ipaddress
import os
import socket
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types as genai_types
from pydantic import BaseModel, Field, HttpUrl
from starlette.concurrency import run_in_threadpool

from app_state import RAG_STORE, RERANKER, ROUTER, VERIFIER
from query_router import (
    COMPLEX,
    MULTI_HOP,
    ROUTER_COMPLEX_CANDIDATE_K,
    ROUTER_MULTIHOP_CANDIDATE_K,
    ROUTER_RELEVANCE_THRESHOLD,
    STANDARD,
    evidence_is_sufficient,
    merge_candidates_keep_best,
)
from reranker import RERANK_CANDIDATE_K, RERANK_FINAL_K, rerank_and_select
from verifier import run_verification

SETUP_ERROR = ""

try:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from agentic_rag_agent.agent import build_agent

    ADK_AVAILABLE = bool(os.getenv("GOOGLE_API_KEY"))
except Exception:
    Runner = None
    InMemorySessionService = None
    build_agent = None
    ADK_AVAILABLE = False
    SETUP_ERROR = "Google ADK could not be imported. Install backend requirements and set GOOGLE_API_KEY."

if not os.getenv("GOOGLE_API_KEY"):
    SETUP_ERROR = "GOOGLE_API_KEY is required for Gemini Embedding 2 and the ADK answer flow."


APP_NAME = "multimodal_agentic_rag"
USER_ID = "demo-user"

app = FastAPI(title="Multimodal Agentic RAG ADK")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5177,http://127.0.0.1:5177").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_service = InMemorySessionService() if ADK_AVAILABLE else None


class TextSourceRequest(BaseModel):
    title: str
    text: str
    modality: Literal["text"] = "text"


class UrlSourceRequest(BaseModel):
    url: HttpUrl
    title: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(max(1, min(RERANK_FINAL_K, RERANK_CANDIDATE_K)), ge=1, le=12)


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


def _validate_fetch_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if os.getenv("ALLOW_PRIVATE_URLS", "").lower() == "true":
        return

    if parsed.hostname.lower() == "localhost":
        raise ValueError("Private and localhost URLs are disabled for URL ingestion.")

    try:
        address_info = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve URL host: {parsed.hostname}") from exc

    for item in address_info:
        address = ipaddress.ip_address(item[4][0])
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise ValueError("Private and localhost URLs are disabled for URL ingestion.")


def _event_text(event: Any) -> str:
    if not getattr(event, "content", None) or not event.content.parts:
        return ""
    fragments = []
    for part in event.content.parts:
        text = getattr(part, "text", None)
        if text:
            fragments.append(text)
    return "".join(fragments)


async def _run_adk_agent(
    question: str,
    retrieval: dict[str, Any],
    insufficient_evidence: bool = False,
) -> str:
    if not ADK_AVAILABLE:
        raise HTTPException(503, SETUP_ERROR or "Google ADK is unavailable.")

    def retrieve_relevant_context(query: str, top_k: int = 6) -> dict:
        """Return the exact retrieval packet already embedded for this request."""
        return retrieval

    request_agent = build_agent(retrieve_relevant_context)
    request_runner = Runner(agent=request_agent, app_name=APP_NAME, session_service=session_service)
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
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
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=instruction.format(question=question))],
    )
    final_text = ""
    async for event in request_runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=content,
    ):
        text = _event_text(event)
        if text:
            final_text = text
    return final_text


@app.get("/health")
async def health():
    try:
        space_info = await run_in_threadpool(RAG_STORE.space_tool)
    except Exception as exc:
        space_info = {"storage": "unavailable", "storage_error": str(exc)}
    else:
        space_info["storage"] = "ok" if not RAG_STORE.chroma_error else "unavailable"
        space_info["persist_directory"] = RAG_STORE.persist_directory
    return {
        "status": "ok" if ADK_AVAILABLE and not SETUP_ERROR else "setup_required",
        "adk": ADK_AVAILABLE,
        "setup_error": SETUP_ERROR,
        **space_info,
    }


@app.get("/space")
async def space():
    return await run_in_threadpool(RAG_STORE.snapshot)


@app.post("/sources/text")
async def add_text_source(req: TextSourceRequest):
    try:
        source = await run_in_threadpool(RAG_STORE.add_text_source, req.title, req.text, req.modality)
        snapshot = await run_in_threadpool(RAG_STORE.snapshot)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"source": source.__dict__, "space": snapshot}


@app.post("/sources/url")
async def add_url_source(req: UrlSourceRequest):
    try:
        url = str(req.url)
        await run_in_threadpool(_validate_fetch_url, url)
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        text = _extract_text_from_html(response.text)
        title = req.title or url.replace("https://", "").replace("http://", "")[:80]
        source = await run_in_threadpool(RAG_STORE.add_text_source, title, text[:12000], "url")
        snapshot = await run_in_threadpool(RAG_STORE.snapshot)
    except Exception as exc:
        raise HTTPException(400, f"Could not ingest URL: {exc}") from exc
    return {"source": source.__dict__, "space": snapshot}


@app.post("/sources/file")
async def add_file_source(
    file: UploadFile = File(...),
    title: str = Form(""),
    notes: str = Form(""),
):
    data = await file.read()
    if len(data) > 120 * 1024 * 1024:
        raise HTTPException(400, "File too large for this demo. Keep uploads under 120 MB.")
    try:
        source = await run_in_threadpool(
            RAG_STORE.add_file_source,
            title=title or file.filename or "Uploaded source",
            data=data,
            mime_type=file.content_type or "application/octet-stream",
            notes=notes,
        )
        snapshot = await run_in_threadpool(RAG_STORE.snapshot)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"source": source.__dict__, "space": snapshot}


@app.delete("/sources/{source_id}")
async def delete_source(source_id: str):
    try:
        removed = await run_in_threadpool(RAG_STORE.remove_source, source_id)
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc
    if not removed:
        raise HTTPException(404, "Source not found.")
    return {"deleted": source_id, "space": await run_in_threadpool(RAG_STORE.snapshot)}


def _evidence_payload(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Shape the reranked evidence for the ADK generator.

    Mirrors MultimodalRagStore.retrieval_payload exactly (citation, source,
    modality, similarity, evidence) so generation's input conventions are
    unchanged; only the evidence items and their order change.
    """
    return {
        "provider": RAG_STORE.embedding_provider,
        "matches": [
            {
                "citation": item["id"],
                "source": item["title"],
                "modality": item["modality"],
                "similarity": item["similarity"],
                "evidence": item["text"],
            }
            for item in evidence
        ],
    }


async def _retrieve_multihop(
    subqueries: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run independent candidate retrieval per subquery and merge/dedup.

    A subquery that errors or retrieves nothing contributes zero candidates
    rather than failing the whole query. Returns (merged_candidates, space,
    query_point). `space` and `query_point` come from the first subquery that
    produced a usable retrieval (fallback empty placeholders if none did).
    """
    candidate_sets: list[list[dict[str, Any]]] = []
    space: dict[str, Any] | None = None
    query_point: dict[str, Any] | None = None
    multihop_k = max(1, ROUTER_MULTIHOP_CANDIDATE_K)
    for subquery in subqueries:
        try:
            retrieval = await run_in_threadpool(
                RAG_STORE.retrieve_candidates, subquery, multihop_k
            )
        except Exception:
            retrieval = None
        if retrieval is None:
            candidate_sets.append([])
            continue
        candidate_sets.append(retrieval.get("candidates") or [])
        if space is None:
            space = retrieval.get("space")
            query_point = retrieval.get("query_point")
    merged = merge_candidates_keep_best(candidate_sets)
    if space is None:
        space = await run_in_threadpool(RAG_STORE.snapshot)
    if query_point is None:
        query_point = {
            "id": "query-router-fallback",
            "source_id": "query",
            "title": "",
            "modality": "query",
            "projection": {"x": 0.0, "y": 0.0, "z": 0.0},
            "color": "#f54e00",
            "score": 1,
            "preview": "No subquery produced a usable embedding.",
        }
    return merged, space, query_point


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is required.")

    route = await run_in_threadpool(ROUTER.classify, req.question)
    strategy = route["strategy"]
    router_reason = route["reason"] or ""
    subqueries = list(route["subqueries"] or [])
    router_fell_back = bool(route["fell_back"])

    trace = [
        {
            "agent": "query_router",
            "status": "fallback" if router_fell_back else "complete",
            "detail": router_reason or f"Strategy {strategy} selected.",
        }
    ]

    if strategy == MULTI_HOP:
        candidate_k = max(1, ROUTER_MULTIHOP_CANDIDATE_K)
        candidates, space, query_point = await _retrieve_multihop(subqueries)
    else:
        candidate_k = max(
            1, ROUTER_COMPLEX_CANDIDATE_K if strategy == COMPLEX else RERANK_CANDIDATE_K
        )
        try:
            retrieval = await run_in_threadpool(
                RAG_STORE.retrieve_candidates, req.question, candidate_k
            )
        except Exception as exc:
            raise HTTPException(503, f"Retrieval failed: {exc}") from exc
        candidates = retrieval["candidates"]
        space = retrieval["space"]
        query_point = retrieval["query_point"]

    final_k = max(1, min(req.top_k, candidate_k))
    outcome = await run_in_threadpool(rerank_and_select, RERANKER, req.question, candidates, final_k)
    evidence = outcome["evidence"]

    insufficient = not evidence_is_sufficient(evidence, ROUTER_RELEVANCE_THRESHOLD)
    payload_evidence = [] if insufficient else evidence
    retrieval_payload = _evidence_payload(payload_evidence)
    answer = await _run_adk_agent(
        req.question, retrieval_payload, insufficient_evidence=insufficient
    )

    verification = await run_in_threadpool(
        run_verification,
        VERIFIER,
        req.question,
        answer,
        payload_evidence,
        [item["id"] for item in payload_evidence],
    )

    projection_by_source = {
        point["source_id"]: point.get("projection", {"x": 0.0, "y": 0.0, "z": 0.0})
        for point in space["points"]
    }
    matches = [
        {
            "id": item["id"],
            "source_id": item["source_id"],
            "title": item["title"],
            "modality": item["modality"],
            "text": item["text"],
            "score": item["similarity"],
            "similarity": item["similarity"],
            "relevance": item["relevance"],
            "reason": item["reason"],
            "projection": projection_by_source.get(
                item["source_id"], {"x": 0.0, "y": 0.0, "z": 0.0}
            ),
            "metadata": item["metadata"],
        }
        for item in evidence
    ]
    if strategy == MULTI_HOP:
        retrieval_detail = (
            f"Embedded {len(subqueries)} subqueries and retrieved "
            f"{len(candidates)} unique candidates after merge/dedup"
        )
    else:
        retrieval_detail = f"Embedded query and retrieved {len(candidates)} candidate chunks"
    trace.extend(
        [
            {
                "agent": "space_inspector",
                "status": "complete",
                "detail": f"{len(RAG_STORE.sources)} sources, {len(RAG_STORE.chunks)} chunks, {RAG_STORE.dimensions} dimensions",
            },
            {
                "agent": "retrieval_tool",
                "status": "complete",
                "detail": retrieval_detail,
            },
            {
                "agent": "reranker",
                "status": "complete" if outcome["used_reranking"] else "fallback",
                "detail": (
                    (outcome["reason"] or "Gemini reranking selected final evidence.")
                    + (f" Validation notes: {len(outcome['errors'])}." if outcome["errors"] else "")
                ),
            },
            {
                "agent": "answer_synthesizer",
                "status": "complete",
                "detail": (
                    "Stated that evidence is insufficient to answer the question."
                    if insufficient
                    else "Generated grounded answer; citations are shown separately"
                ),
            },
        ]
    )
    trace.append(
        {
            "agent": "grounding_verifier",
            "status": "unavailable" if verification["unavailable"] else "complete",
            "detail": (
                verification["reason"]
                if verification["unavailable"]
                else (
                    f"Heuristic check: grounded={verification['grounded']} "
                    f"score={verification['score']}; not a guarantee of correctness."
                )
            ),
        }
    )

    response = {
        "answer": answer,
        "matches": matches,
        "reranked": outcome["used_reranking"],
        "strategy": strategy,
        "router_reason": router_reason,
        "subqueries": subqueries,
        "router_fell_back": router_fell_back,
        "insufficient_evidence": insufficient,
        "query_point": query_point,
        "trace": trace,
        "space": space,
    }
    if verification["unavailable"]:
        response["verification_unavailable"] = verification["reason"]
    else:
        response["grounded"] = verification["grounded"]
        response["score"] = verification["score"]
        response["claims"] = verification["claims"]
        response["unsupported_claims"] = verification["unsupported_claims"]
    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8897"))
    uvicorn.run(app, host="0.0.0.0", port=port)
