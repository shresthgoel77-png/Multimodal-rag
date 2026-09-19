# Aether

A multimodal RAG app built with Gemini Embedding 2 and Google ADK. Add text, URLs, PDFs, images, audio, or video; ask a question; and get a grounded answer with citations, a pipeline trace, and a 3D embedding view.

The UI includes a 3D embedding view for inspecting the search space. Each source appears as one point. When you ask a question, the query is projected into the same space and the cited sources are highlighted.

## Tech Stack

- **Python (FastAPI)**: High-performance backend orchestration and API layer.
- **TypeScript (React + Vite)**: Modern, responsive frontend featuring recent UI improvements and a consistent header navigation bar.
- **ChromaDB**: Local, persistent vector database for storing and querying multimodal chunks seamlessly across restarts.
- **Gemini Embedding 2**: 768-dimensional embeddings for sources and queries.
- **Google ADK**: Agentic framework orchestrating grounded generative answers based on retrieved evidence.

## What It Does

- Adds and removes multimodal sources from a fully integrated, local persistent ChromaDB index.
- Offers an enhanced frontend with significant UI improvements, including a new header navigation bar allowing seamless switching between Q&A, citations, traces, 3D embeddings, and evaluations.
- Uses Gemini Embedding 2 for source and query embeddings (768 dimensions).
- Requires `GOOGLE_API_KEY`; there are no local embedding or answer fallbacks.
- Classifies each question (STANDARD / COMPLEX / MULTI_HOP), retrieves candidates, reranks with Gemini, generates a grounded answer via a Google ADK agent, and runs a heuristic grounding check.
- Every Gemini-dependent stage degrades explicitly: reranker/router/verifier failures fall back or are marked unavailable, never silently.
- Shows citations separately from the answer text so citation IDs do not clutter the response.
- Projects source and query vectors into a 3D PCA view for inspection.
- Ships a robust, verified evaluation pipeline and a fixed 25-question benchmark with retrieval metrics, an LLM judge, and a baseline-vs-improved comparison, all viewable in the UI.

## Architecture

Real request path for `POST /ask` (see `backend/server.py`):

```text
question
  → query_router.classify ........... STANDARD / COMPLEX / MULTI_HOP (or STANDARD fallback)
  → retrieve_candidates ............. chunk-level cosine-similarity candidates from Chroma
      STANDARD: 12 candidates · COMPLEX: 24 · MULTI_HOP: 12 per subquery, merged/deduped
  → rerank_and_select ............... Gemini rerank, top final_k (default 5); similarity fallback
  → insufficient-evidence check ..... threshold 0.15; weak/empty evidence → explicit statement
  → ADK answer_synthesizer .......... grounded generation from the final evidence only
  → run_verification ................ heuristic grounding check; never blocks the answer
  → response: answer + matches + trace + latencies + space + verification
```

| Layer | Role |
| --- | --- |
| React + Vite frontend | Source manager, Q&A panel, citations, pipeline trace, evaluation view, 3D embedding view |
| FastAPI backend (`backend/server.py`) | Ingestion, routing/retrieval/rerank orchestration, answer API, embedding-space snapshots, read-only evaluation summary |
| `MultimodalRagStore` (`backend/rag_store.py`) | ChromaDB-persisted sources/chunks/embeddings, search, PCA projection |
| `QueryRouter` (`backend/query_router.py`) | One bounded Gemini call → exactly one of STANDARD / COMPLEX / MULTI_HOP |
| `GeminiReranker` (`backend/reranker.py`) | Scores the fixed candidate set, selects top `final_k` |
| `GeminiVerifier` (`backend/verifier.py`) | Heuristic claim-vs-evidence check on the final evidence set |
| Google ADK agent (`backend/agentic_rag_agent/agent.py`) | Answer coordinator; receives the same retrieval packet shown in the UI |
| Evaluation (`evaluation/`) | Fixed corpus + `benchmark_v1.json`, `metrics.py` (Recall@K/MRR/judge), `run_evaluation.py`, `run_comparison.py` |

The important implementation detail is that `/ask` performs retrieval (plus reranking) once and passes that same final evidence into the ADK answer flow. The answer and the citation panel are therefore based on the same ranked evidence. For MULTI_HOP, each subquery is retrieved independently and merged (dedup by chunk id, keeping the highest similarity) before reranking.

## Project Structure

```text
./
|-- README.md
|-- assets/
|-- backend/
|   |-- server.py               # all endpoints + /ask orchestration
|   |-- app_state.py            # shared singletons (store, reranker, router, verifier)
|   |-- rag_store.py            # Chroma-backed store, ingestion, search, PCA
|   |-- query_router.py         # adaptive routing (STANDARD/COMPLEX/MULTI_HOP)
|   |-- reranker.py             # Gemini reranking + similarity fallback
|   |-- verifier.py             # heuristic grounding verification
|   |-- agentic_rag_agent/agent.py
|   |-- requirements.txt
|   └-- tests/                  # backend test suite (incl. test_phase9_closure.py)
|-- evaluation/
|   |-- benchmark/benchmark_v1.json  # v1.0.0, 25 questions
|   |-- corpus/                 # fixed corpus text + ingestion manifest (11 sources)
|   |-- metrics.py              # Recall@K / MRR / Precision@K + LLM judge
|   |-- run_evaluation.py       # full benchmark runner → results/run_*.json
|   |-- run_comparison.py       # baseline-vs-improved → results/comparison_*.json
|   |-- results/                # persisted run + comparison files
|   |-- scripts/                # ingest_corpus.py, verify_benchmark.py
|   └-- tests/                  # benchmark / metrics / comparison tests
└-- frontend/
    |-- src/App.tsx             # whole UI (Q&A, trace, citations, evaluation, 3D view)
    |-- src/main.tsx, src/styles.css
    └-- package.json, vite.config.ts, tsconfig.json
```

## Run Locally

Backend (tested):

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your-google-ai-studio-key"
python server.py
```

The backend runs at `http://localhost:8897` (override with `PORT`).

Frontend (tested, in another terminal):

```bash
cd frontend
npm install
npm run dev -- --port 5177
```

The frontend runs at `http://localhost:5177`. If the backend is on a different port:

```bash
VITE_API_URL=http://localhost:8897 npm run dev -- --port 5177
```

## Try It

1. Open `http://localhost:5177`.
2. Add a text, URL, PDF, image, audio, or video source.
3. Ask a question in the Q&A panel.
4. Review the answer, citations, and the Pipeline panel (strategy, rerank state, latencies, verification).
5. Inspect the source and query points in the embedding view.
6. Click “Load evaluation results” in the Evaluation panel to see stored benchmark numbers.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Status, ADK availability, provider, dimensions, source/chunk counts, storage state |
| `GET` | `/space` | Current sources, projected points, event trail, projection metadata |
| `GET` | `/evaluation/summary` | Read-only summary of stored benchmark + latest run + latest comparison (never recomputes; `null` fields with notes when data is missing/malformed) |
| `POST` | `/sources/text` | Add a text source. Body: `{title, text, modality: "text"}` |
| `POST` | `/sources/url` | Fetch and index a public URL. Body: `{url, title?}` |
| `POST` | `/sources/file` | Upload a PDF, image, audio, or video (multipart `file`, `title`, `notes`; 120 MB limit) |
| `DELETE` | `/sources/{source_id}` | Remove a source and all its chunks |
| `POST` | `/ask` | Full pipeline. Body: `{question, top_k? (default 5, 1–12)}`. Returns `answer`, `matches` (final evidence with similarity/relevance/projection), `candidates`, `strategy`, `router_reason`, `subqueries`, `router_fell_back`, `reranked`, `rerank_fallback`, `rerank_reason`, `insufficient_evidence`, `query_point`, `trace`, `space`, per-stage `latencies`, plus `grounded`/`score`/`claims`/`unsupported_claims` — or `verification_unavailable` when verification could not run |

## Environment Variables

Every variable the system actually reads (confirmed by inspection):

| Variable | Default | Used by |
| --- | --- | --- |
| `GOOGLE_API_KEY` | (none — required) | Embeddings, ADK answers, router, reranker, verifier, judge. Without it the backend reports `setup_required` and ingestion/answering fail with an explicit error |
| `CHROMA_PERSIST_DIRECTORY` | `backend/chroma_db` | Persistent Chroma storage; survives restarts |
| `PORT` | `8897` | Backend listen port (`python server.py`) |
| `ALLOWED_ORIGINS` | `http://localhost:5177,http://127.0.0.1:5177` | CORS allowlist |
| `ALLOW_PRIVATE_URLS` | (unset) | Set to `true` to allow URL ingestion of localhost/private IPs |
| `GEMINI_HTTP_TIMEOUT_MS` | `60000` | Shared SDK-level HTTP timeout (backstop; stage timeouts below fire first) |
| `ROUTER_MODEL` | `gemini-3-flash-preview` | Router model |
| `ROUTER_TIMEOUT_SECONDS` | `15` | Router wall-clock timeout → STANDARD fallback |
| `ROUTER_COMPLEX_CANDIDATE_K` | `24` | Candidates retrieved for COMPLEX |
| `ROUTER_MULTIHOP_CANDIDATE_K` | `12` | Candidates retrieved per subquery for MULTI_HOP |
| `ROUTER_RELEVANCE_THRESHOLD` | `0.15` | Minimum effective evidence score; below it → insufficient-evidence path |
| `RERANK_MODEL` | `gemini-3-flash-preview` | Reranker model |
| `RERANK_CANDIDATE_K` | `12` | Candidates retrieved for STANDARD |
| `RERANK_FINAL_K` | `5` | Final evidence items passed to generation; also the default `top_k` for `/ask` |
| `RERANK_TIMEOUT_SECONDS` | `30` | Rerank wall-clock timeout → similarity fallback |
| `VERIFIER_MODEL` | `gemini-3-flash-preview` | Verifier model |
| `VERIFIER_TIMEOUT_SECONDS` | `30` | Verification wall-clock timeout → `verification_unavailable` |
| `JUDGE_MODEL` | `gemini-3-flash-preview` | Eval LLM-judge model |
| `JUDGE_TIMEOUT_SECONDS` | `30` | Judge wall-clock timeout → `unavailable` (never zero) |
| `VITE_API_URL` | `http://localhost:8897` | Frontend → backend URL |

Fixed (not configurable): embedding model `gemini-embedding-2`, 768 dimensions, text chunking 170 words with 35-word overlap, ADK agent model `gemini-3-flash-preview` (temperature 0.25, 900 max output tokens).

## Ingestion

- **Text** (`POST /sources/text`): chunked (170 words / 35 overlap), each chunk embedded with the retrieval-document prefix. Empty text is rejected.
- **URL** (`POST /sources/url`): fetched server-side (20 s timeout), scripts/nav stripped, first 12,000 chars embedded as a `url`-modality source. Localhost/private IPs are blocked unless `ALLOW_PRIVATE_URLS=true`.
- **File** (`POST /sources/file`): modality derived from MIME type (pdf/image/audio/video). Small files embed inline; large/video/audio go through the Gemini File API (polled up to 90 s) and the uploaded file is deleted afterwards. The media embedding is blended with a text-annotation embedding. Uploads over 120 MB are rejected. Media processing runs in a threadpool so the event loop is not blocked.
- Source identity is deterministic (SHA-256 of kind + title + content), so re-ingesting the same source **upserts** (replaces chunks, keeps `created_at`) instead of creating duplicates.
- Embeddings persist in `CHROMA_PERSIST_DIRECTORY` and are reloaded on restart — no re-ingestion needed.

## Retrieval

- `retrieve_candidates(query, candidate_k)` embeds the query and returns up to `candidate_k` **chunk-level** candidates with `similarity = 1 − cosine_distance`, plus the query point and space snapshot for the 3D view.
- `candidate_k` depends on the routed strategy: 12 (STANDARD), 24 (COMPLEX), 12 per subquery (MULTI_HOP). `final_k = min(top_k, candidate_k)`.
- `search()` (used by the ADK agent's `retrieve_relevant_context` tool) is the same search collapsed to one best chunk per source.

## Reranking

- The reranker scores only the candidate set it is handed (never re-queries Chroma) via one structured Gemini call (`{results: [{id, relevance 0–1, reason}]}`).
- Validation: unknown chunk ids discarded, non-numeric relevance discarded, out-of-range relevance clamped to [0, 1] (recorded), duplicate ids keep the first occurrence. Unscored candidates fill remaining `final_k` slots by similarity with `relevance: None`.
- Ordering: relevance desc, then similarity desc, then id.
- **Fallback** (used whenever the reranker is unavailable, times out, errors, or produces no usable scores): candidates ordered by similarity, `reranked: false`, `rerank_fallback: true` with a human-readable `rerank_reason`. Generation still proceeds. An empty candidate set is not a fallback — it yields empty evidence with an explanatory reason.

## Adaptive RAG

Three fixed strategies; the router can never invent new ones or change pipeline structure:

- **STANDARD**: single-fact question, one retrieval pass (12 candidates).
- **COMPLEX**: broad/multi-aspect question, larger candidate set (24).
- **MULTI_HOP**: question needing 2–3 separate facts; the router must supply 2–3 standalone subqueries, each retrieved independently (12 each) and merged/deduped by chunk id keeping the highest similarity (ties keep first-seen). A failing/empty subquery contributes nothing instead of failing the whole query.
- **Fallback**: any router failure — no client, timeout, exception, non-JSON output, unknown strategy, or too few subqueries — falls back to STANDARD with `router_fell_back: true` and an explanatory reason. Empty questions also fall back.
- **Insufficient evidence**: after reranking, evidence is insufficient when empty or when every item's effective score (reranker `relevance`, else retrieval `similarity`) is below `ROUTER_RELEVANCE_THRESHOLD` (0.15). The generator is then instructed to state insufficiency explicitly instead of answering, and no weak evidence is forwarded to it.

## Grounding Verification

- After generation, the verifier checks the answer's important claims against the **final evidence set** (the exact chunks generation used) and confirms cited ids are real evidence ids.
- It is explicitly a **heuristic, best-effort** check — not a hallucination detector or correctness guarantee.
- Score = supported claims / total claims (no claims → vacuously grounded, 1.0). Supported-with-no-real-evidence claims are demoted to unsupported; unknown evidence ids are discarded; `unsupported_claims` is recomputed from the validated claims.
- It **never blocks an answer**: on failure, timeout, empty/malformed output, or empty evidence, the original answer is returned unchanged with `verification_unavailable` set and no `grounded`/`score` fields fabricated.

## Evaluation Methodology

- **Fixed corpus**: 11 text sources (coastal/ocean ecosystem science) in `evaluation/corpus/corpus.json`, ingested once via the real `add_text_source` path; ids recorded in `evaluation/corpus/manifest.json`. Seed it with `python -m evaluation.scripts.ingest_corpus`.
- **Benchmark** (`evaluation/benchmark/benchmark_v1.json`, v1.0.0, `multimodal-rag-benchmark`): 25 questions — 5 factual, 5 semantic, 4 difficult, 3 comparison, 4 multi-hop, 2 evidence_specific, 2 unanswerable. Every answerable question grounds its answer tokens in its listed sources; every multi-hop question needs all its sources (no single source covers the full answer); unanswerable subjects (penguins; anglerfish bioluminescence) are absent from the corpus.
- **Matching rule**: expected ids are source-level; retrieved chunk candidates are deduped by `source_id` (first-seen order) before scoring.
- **Retrieval metrics**: Recall@K = relevant retrieved in top K / total relevant; MRR = 1 / rank of first relevant hit (0 when none); Precision@K also recorded. Aggregates are means over **answerable questions only** (23 of 25).
- **LLM judge**: per-question `correctness` / `groundedness` / `citation_correctness` on a 0–1 scale. Failed/timed-out/invalid judgments are recorded as `unavailable` — never zero, never silently dropped. Out-of-range judge scores invalidate the whole judgment (rejected, not clamped).
- **Unanswerable questions** (q24, q25): excluded from Recall@K/MRR aggregates and scored separately as `correctly_handled` (insufficient evidence stated) vs `miscount`.
- **Runners**: `python -m evaluation.run_evaluation` writes `results/run_<timestamp>.json`; `python -m evaluation.run_comparison` writes `results/comparison_<timestamp>.json` with baseline (raw similarity top-K, K = final_k = 5) vs improved (full pipeline) aggregates plus computed `differences_improved_minus_baseline`. Without `GOOGLE_API_KEY`, deterministic offline embeddings are used and generation/judge record `unavailable`.

## Baseline vs. Improved Results

Real numbers from `evaluation/results/comparison_20260913T151835.json` (25 questions, offline deterministic embeddings, 23 answerable + 2 unanswerable; judge unavailable on both sides so generation quality is unmeasured):

| Metric (answerable, n=23) | Baseline | Improved | Δ (improved − baseline) |
| --- | --- | --- | --- |
| Recall@5 | 0.9565 | 0.9783 | +0.0218 |
| Recall@10 | 0.9565 | 1.0000 | +0.0435 |
| MRR | 0.9275 | 0.9275 | +0.0000 |
| Precision@5 | 0.2935 | 0.2522 | −0.0413 |
| Precision@10 | 0.2935 | 0.1567 | −0.1368 |

- Unanswerable handling: baseline 0/2 correctly handled, improved 1/2 (+1).
- Judge: 0/25 available on both sides (`correctness`/`groundedness`/`citation_correctness` all `null` diffs) — no API key at run time.
- No corpus drift (0 questions), no failed questions, no per-question errors.
- The latest full offline run (`run_20260913T141635.json`) agrees: Recall@5 0.9783, Recall@10 1.0, MRR 0.9275, unanswerable 1/2.

## Testing

Backend suite (117 tests, all passing, no API key needed):

```bash
cd backend
.venv/bin/python -m pytest tests -q
```

Evaluation unit tests (66 passing; `slow` integration tests need the ingested store):

```bash
cd backend
.venv/bin/python -m pytest ../evaluation/tests -q -m "not slow"
```

Frontend (both clean):

```bash
cd frontend
./node_modules/.bin/tsc -b
npm run build
```

Coverage highlights: normal retrieval, empty database, irrelevant query, restart persistence, source deletion, duplicate ingestion (store + API level), malformed reranker output, reranker timeout, router failure → STANDARD, genuine multi-hop merge/dedup, insufficient-evidence statement, verifier failure (answer still returned, marked unavailable), metric math on synthetic cases, unanswerable exclusion, benchmark schema/grounding, comparison diffs computed from aggregates, and `/evaluation/summary` full/empty/malformed states. No test was weakened to pass: the one failure found during this pass was a wrong expectation in the new test's own fake (it claimed verification succeeds on empty evidence; the real verifier correctly reports unavailable), fixed in the test.

## Limitations

- **Needs `GOOGLE_API_KEY` and network** for embeddings and answers — by design, with explicit errors, not silent fallbacks.
- **Generation quality unmeasured**: all persisted judge scores are `unavailable` (no API key at eval time), so baseline-vs-improved compares retrieval + unanswerable handling only.
- **Live-API eval run failed on quota**: `run_20260913T143139.json` (25 questions, API available) recorded 429 RESOURCE_EXHAUSTED on every question against the free tier (5 req/min, 20 req/day for `gemini-3-flash`) plus an async wiring error (`cannot unpack non-iterable coroutine object`) on some questions — its zeros reflect quota, not retrieval quality. Re-run with quota/backoff before trusting any live numbers.
- **`/evaluation/summary` latest-run pointer**: the newest `run_*.json` files are 1-question probes (`run_20260913T144939.json` and similar), so `latest_run` shows a 1-question snapshot. Full 25-question results are in `comparison_20260913T151835.json` and `run_20260913T141635.json` / `run_20260913T122137.json`.
- **No auth, single demo user, in-memory ADK sessions** — not production-ready (see Notes below).
- **URL ingestion** only handles HTML text extraction; JS-rendered pages and paywalls yield little or nothing.
- **Media files** must be processable by the Gemini File API; uploads are capped at 120 MB.
- Frontend production bundle emits a >500 kB chunk-size warning (three.js); informational only.

## Notes

- Embeddings live in local persistent ChromaDB (`CHROMA_PERSIST_DIRECTORY`); the persist directory survives backend restarts.
- Re-ingesting identical content upserts rather than duplicating.
- URL ingestion blocks localhost/private ranges unless `ALLOW_PRIVATE_URLS=true`.
- Uploaded Gemini File API media is cleaned up after embedding.
- Blocking media work runs in a threadpool so the FastAPI event loop is not held.
- For production, add authentication, background ingestion, evals in CI, observability, and a managed vector database.
