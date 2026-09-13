# Phase 5 — Fixed Evaluation Benchmark

This directory holds the **fixed evaluation corpus** and the **versioned
benchmark dataset** that Phase 6 (metrics) and Phase 7 (baseline-vs-improved)
must run against. Phase 5 implements no metrics, scoring, or comparison logic —
it only pins down *what* the later phases will measure and scores nothing.

> **Phase 6** (metrics + result recording) is documented in §8 below; it adds
> `metrics.py`, `run_evaluation.py`, `results/`, and `tests/test_metrics.py`.

## Layout

```text
evaluation/
|-- README.md                        # this file
|-- eval_embeddings.py               # deterministic, API-free embeddings (shared by ingest/verify/tests and Phase 6-7)
|-- metrics.py                       # Phase 6: metric definitions + LLM judge (importable, no side effects)
|-- run_evaluation.py                # Phase 6: full benchmark runner, persists results/ files
|-- corpus/
|   |-- corpus.json                  # the canonical fixed corpus text (immutable for Phases 6-7)
|   `-- manifest.json                # record of the real ingestion: source_id (hex) <-> corpus_id (name)
|-- benchmark/
|   `-- benchmark_v1.json            # v1.0.0 benchmark: 25 questions with grounded ground truth
|-- results/                         # Phase 6: per-question + aggregate results files (Phase 7 input)
|-- scripts/
|   |-- ingest_corpus.py             # seeds the persistent Chroma store with the corpus
|   `-- verify_benchmark.py          # retrieval-based verification harness (run before trusting the file)
`-- tests/
    |-- test_benchmark.py            # Phase 5 hermetic schema/grounding/distribution tests
    `-- test_metrics.py              # Phase 6 unit + integration tests for the metrics harness
```

## 1. The fixed corpus

The corpus is 11 text sources covering coastal/ocean ecosystem science
(mangroves, seagrass, ocean carbon cycle, coral reefs, ocean acidification,
kelp forests, marine protected areas, coastal fisheries, freshwater peatlands,
tropical rainforests, salt marshes). Full text lives in
`evaluation/corpus/corpus.json` and must be reused **unchanged** in Phases 6–7.

Ingestion uses the exact repository ingestion path —
`MultimodalRagStore.add_text_source`, the same call behind `POST /sources/text`
and `POST /sources/url` — into the persistent Chroma store at
`backend/chroma_db` (default `CHROMA_PERSIST_DIRECTORY`), so no Phase 1–4 code
was modified. The store held **zero** sources before this phase; the corpus was
deliberately created and ingested for the benchmark.

Real source ids (deterministic SHA-256 of `text::{modality}::{title}::{clean text}`, truncated to 16 hex):
recorded in `evaluation/corpus/manifest.json`.

> **Deterministic embeddings.** No `GOOGLE_API_KEY` is present in the
> environment, so this phase ingests with a deterministic, feature-hash
> embedding function (`eval_embeddings.py`) — the same API-free technique the
> repository's own unit tests use when they monkeypatch `_embed_text`.
> Chunk vectors live in Chroma and query vectors are produced by the same
> function, so retrieval is reproducible with no network/API. If a later phase
> wants real embeddings, it can re-ingest the same `corpus.json` text with the
> Gemini provider; `expected_sources` refer to ids derived from content, so the
> corpus identity is unchanged even if the ids change value.
> **Phase 6/7 must install the same embedder when querying** (see §4).

## 2. Question distribution

The benchmark has exactly **25 questions**:

| Category | Count |
| --- | --- |
| factual | 5 |
| semantic | 5 |
| difficult | 4 |
| comparison | 3 |
| multi_hop | 4 |
| evidence_specific | 2 |
| unanswerable | 2 |
| **Total** | **25** |

`category` uses the normalized labels `factual | semantic | difficult |
comparison | multi_hop | evidence_specific | unanswerable` (normalized from the
phase spec's "Multi-hop" → `multi_hop`, "Evidence-specific" →
`evidence_specific`). No deviation from the suggested distribution was needed;
the corpus was authored with enough distinct, overlapping sources to support
every category count.

`difficulty` is a real, per-question assessment (`easy|medium|hard`) based on
retrieval specificity and answer precision, not a placeholder. All 4 difficult
and all 4 multi-hop questions are rated `hard`; numeric-recall, multi-species,
and synthesis questions are `hard`; simple single-fact lookups are `easy`;
category-number recall and paraphrase questions are `medium`.

## 3. How expected_answer / expected_sources were verified

For every question the following was actually performed, not assumed:

1. **Content grounding.** Every `expected_answer` content token (≥5 chars) is
   asserted, by test and by inspection, to appear in the full text of the
   question's `expected_sources`. This guarantees no invented or approximate
   ground truth — if the corpus is messy, the answer reflects the corpus
   wording (which is why several answers quote the source text closely).
2. **Retrieval verification.** `scripts/verify_benchmark.py` runs every
   question through the real retrieval path used by `/ask`
   (`store.retrieve_candidates`, same Chroma query + same query embedding) and
   confirms every expected source appears in the top-8 candidates. Result: 27/27
   checks pass (see §6).
3. **Multi-hop genuineness.** Feedback test
   `test_multi_hop_requires_all_sources` proves each multi-hop question needs
   ≥2 sources: every listed source contributes at least one answer token found
   in no *other* expected source, and no single source alone covers the full
   answer (checked against the actual corpus text).
4. **Unanswerable genuineness.** q24 (penguins) and q25 (bioluminescent
   anglerfish) target entities that are provably absent from the entire corpus
   (checked by token search over all source texts; no source mentions
   penguins, anglerfish, bioluminescence/photophores, or any recorded
   anglerfish depth). Both have `"expected_sources": []`.
5. **Live evidence spot-check.** A sample of factual/semantic questions run
   through the live retrieval path returned the expected source as top-1 with
   the expected answer fact present in the retrieved chunk text (see §6).

## 4. Reproducibility contract for Phases 6/7

- Do **not** modify `evaluation/corpus/corpus.json`. That file *is* the fixed
  corpus; re-ingestion of changed text would change `source_id`s and invalidate
  the benchmark.
- Load the benchmark from `evaluation/benchmark/benchmark_v1.json`
  (`json.load`); it self-describes its corpus paths and category labels.
- When querying the store, install the deterministic embedder first:
  ```python
  from eval_embeddings import install_deterministic_embeddings
  store._embed_text = lambda text, task_prefix="": embed_text(text)
  ```
  (the ingest script already installed it at chunk time; queries must use the
  same space). `install_deterministic_embeddings(store)` does this for you.
- For multi-hop questions, the phase-3 router would issue subqueries; for
  metric runs you may simply score retrieval of the whole question, or split
  per expected source using the per-source evidence in the multi-hop answers.

## 5. Tests

```bash
python -m pytest evaluation/tests/test_benchmark.py -v   # hermetic: no API key, no Chroma needed
```

Covers: file parses/loads; per-question schema (exact required keys);
category ∈ 7 allowed labels; `expected_sources` ⊆ real ingested source ids;
unanswerable ⇒ `expected_sources == []`; no duplicate question ids; total count
= 25; per-category distribution = targets; answer grounding against corpus
text; multi-hop necessity; unanswerable subject absence.

Result: **14/14 pass** in both fresh and repeated runs.

Verification harness (requires the ingested store, no API key):

```bash
python -m evaluation.scripts.verify_benchmark --top-n 8   # 27 checks, 0 failures
```

Existing backend suite still passes unchanged (106 tests), confirming no
Phase 1–4 modification: run from `backend/` with `python -m pytest tests/ -q`.

## 6. Verification results (recorded this phase)

- Corpus ingested: 11 sources / 18 chunks into `backend/chroma_db`; re-running
  the ingest is idempotent and leaves the manifest byte-identical.
- Source ids: see `evaluation/corpus/manifest.json` (e.g. Mangrove Forests →
  `6bdc91f0a7a20e3c`).
- Retrieval verification: all 25 questions have every expected source inside
  the top-8 candidates; all 16 single-source questions retrieve their expected
  source at rank 0, and every comparison/multi-hop question retrieves all of
  its expected sources inside the window.
- Unanswerable checks: both subjects absent from the corpus text.
- Live evidence spot-check (factual + semantic sample):
  - "…algae that live inside coral polyps…" → top-1 Coral Reef Biology, chunk
    text contains the zooxanthellae sentence.
  - "…fringe of mangroves reduce the height of an incoming wave…" → top-1
    Mangrove Forests (wave/two-thirds content in source).
  - "…fully submerged blue carbon ecosystem… horizontal stems…" → top-1
    Seagrass Meadows.
  - "…giant algae that have no true roots…" → top-1 Kelp Forests.
- Full test suites: benchmark tests 14/14; backend suite 106/106.

## 7. Known limitations

- **Embeddings are deterministic, not semantic.** Retrieval quality is
  lexical (word unigrams + bigrams hashed to 768-d). This makes the benchmark
  fully reproducible and offline, but an improved retriever could reasonably
  beat it with genuinely semantic embeddings. Treat Phases 6–7 comparisons as
  relative improvements under a fixed, controlled retrieval function.
- **Corpus is text-only.** No PDF/image/audio/video sources are included, so
  multimodal retrieval is not exercised by this benchmark. The rare exception
  would be a future, separate multimodal benchmark over real media files.
- **`/ask` end-to-end was not executed.** The endpoint requires `GOOGLE_API_KEY`
  for its Gemini router/reranker/ADK stages. Ground truth was instead verified
  through the exact retrieval machinery `/ask` uses (store-level candidate
  retrieval + chunk text inspection) plus strict content-grounding tests.
- **Category semantics** (e.g. what separates a hard "semantic" question from a
  "difficult" one) are human-labeled. The automated tests enforce structure and
  grounding but not category semantics.
- **Multi-hop is 2-hop.** Every multi-hop question combines exactly two sources;
  none requires three sources. This matches the router's 2–3 subquery design but
  does not exercise the deepest case.

---

## 8. Phase 6 — Metrics and result recording

Phase 6 computes **retrieval metrics** (Recall@K, MRR) and **LLM-judge
generation metrics** (correctness, groundedness, citation_correctness) over the
fixed benchmark and **persists every result** for Phase 7. It changed no Phase
1–5 library code, no corpus, and no benchmark data.

### 8.1 Files

- `metrics.py` — pure metric definitions + the `LLMJudge` class (importable by
  tests with zero side effects).
- `run_evaluation.py` — CLI runner that drives the real pipeline
  (router → retrieve → rerank → insufficient gate → generate → verify → judge)
  for every question programmatically (not through the HTTP API).
- `results/run_<UTC>.json` — per-question + aggregate results (Phase 7 input).
- `tests/test_metrics.py` — 50 hermetic + integration tests (see §8.6).

Run full:

```bash
python -m evaluation.run_evaluation                    # full 25 questions
python -m evaluation.run_evaluation --limit 3          # smoke subset
python -m evaluation.run_evaluation --skip-judge       # no Gemini judge client
```

### 8.2 The Recall@K matching rule (documented, applied consistently)

`expected_sources` are **source-level** ids; retrieval returns **chunk-level**
candidates carrying `source_id`. The harness therefore **deduplicates the
retrieved chunk ranking by `source_id`, preserving first-seen order**, and
scores that source-id ranking against expected source ids. Recall@K =
(# expected sources present in the top K deduped source ids) / (total expected
sources). MRR = 1/(rank of the first expected source in that deduped ranking),
else 0. This is exactly the level Phase 5's `verify_benchmark.py` compared at,
so the metrics are comparable and immune to chunk-counting artifacts.

- Per-question Recall@K/MRR are computed from the **retrieval candidate
  ranking** (Chroma similarity order), not from the reranked evidence, so they
  measure retrieval quality directly and independently of the reranker and of
  the insufficient-evidence gate.
- Empty retrieval → Recall@K = 0 and MRR = 0, never an error.

### 8.3 Unanswerable and corpus-drift handling

- **Unanswerable questions** (`expected_sources == []`) are **excluded from
  Recall@K/MRR aggregation** (`answerable_count` = 23 of 25). They are scored
  separately: a pass/fail signal records whether the system hit Phase 3's
  insufficient-evidence gate (`evidence_is_sufficient` with
  `ROUTER_RELEVANCE_THRESHOLD`), plus the retrieved source ids and final
  evidence count for context.
- **Corpus drift** (an `expected_sources` id missing from the live store) is
  detected per question and flagged in `corpus_drift` — never silently scored
  as an ordinary miss. None was present in this run.
- **Insufficient-evidence questions** (incl. unanswerable ones caught by the
  gate) are judged on their **stated insufficiency** answer; the judge is told
  the evidence was insufficient so "grounded by construction" applies. Offline
  (no API key) the insufficiency-only proxy
  `"The available evidence is insufficient to answer the question."` is
  recorded when the gate trips, so the answer slot is never silently empty.

### 8.4 LLM-judge scoring definitions (consistent 0–1 scale)

The judge is a structured Gemini call (same
`os.getenv`/threaded-timeout/JSON-extraction pattern as the reranker/verifier)
returning `{"correctness", "groundedness", "citation_correctness"}`.

- **correctness** — how much of the info in `expected_answer` the generated
  answer conveys: 1.0 = all required info, no contradiction; 0.0 = none or
  outright contradiction; intermediates for partial coverage.
- **groundedness** — whether every claim in the generated answer is supported
  by the evidence actually used: 1.0 = all claims supported (or a stated
  insufficiency with empty/below-threshold evidence); 0.0 = invented/
  unsupported claims.
- **citation_correctness** — whether the cited evidence (the chunk ids the
  answer was generated from) actually supports the answer's claims: 1.0 =
  every claim attributable to a cited chunk; 0.0 = citations do not support
  the claims.

Every judge output is validated (valid JSON, all three fields present, each a
number in [0,1]); a failure, timeout, malformed JSON, or out-of-range score is
recorded as **unavailable** (never 0, never silently dropped).

### 8.5 Result file schema

```jsonc
{
  "run":   { "timestamp", "benchmark_version", "question_count", "api_available",
             "judge_model", "embedding", "k_candidates", "final_k" },
  "aggregates": {
    "retrieval": { "answerable_count", "recall_at_5", "recall_at_10", "mrr",
                   "precision_at_5", "precision_at_10" },
    "unanswerable": { "count", "correctly_handled", "miscount" },
    "judge":        { "total", "available", "unavailable", "correctness_mean",
                      "groundedness_mean", "citation_correctness_mean" },
    "corpus_drift": { "count", "questions" }, "errors": [...]
  },
  "per_question": [
    { "question_id", "question", ..., "strategy", "retrieved_ids",
      "retrieved_source_ids", "final_evidence_ids", "insufficient_evidence",
      "answer", "verification", "metrics": {"retrieval": {...}, "judge": {...}},
      "latency_ms": {"router/retrieval/rerank/generation/verification/judge/total"},
      "corpus_drift", "generation_status",
      "outcome": "complete|generation_unavailable|generation_skipped|"
                 "unanswerable_handled|unanswerable_missed|verification_unavailable|error",
      "unanswerable_result": {...}     // unanswerable only
    }
  ]
}
```

Phase 7 loads this file (and a second one for the improved pipeline) to compare
Recall@5/@10 and MRR plus judge means between runs.

### 8.6 Phase 6 test results

```bash
python -m pytest evaluation/tests/test_metrics.py -v   # 50 passed (incl. 3 slow integration)
python -m pytest evaluation/tests/test_benchmark.py -q # 14 passed (Phase 5 unchanged)
cd backend && python -m pytest tests/ -q               # 106 passed (Phases 1–4 unchanged)
```

Covered: synthetic Recall@K for K=5/10; MRR incl. no-relevant→0; unanswerable
exclusion + separate scoring; judge well-formed/out-of-range/malformed/timeout
behavior (via a fake Gemini client); full-run integration over the real
benchmark producing a results file; corpus-drift flagging.

### 8.7 Recorded aggregate results (offline run, no GOOGLE_API_KEY)

The environment has no `GOOGLE_API_KEY`, so generation/judge stages degrade
gracefully (recorded as unavailable) while retrieval metrics are fully
computed. From `evaluation/results/run_20260913T122137.json`:

- **Recall@5 = 0.9783, Recall@10 = 1.0000, MRR = 0.9275** over the 23
  answerable questions. The single sub-perfect Recall@5 is q17 (comparison):
  Coral Reef Biology ranks 1st but Kelp Forests ranks 6th, giving recall@5 =
  0.5 (recall@10 = 1.0, MRR = 1.0).
- **Unanswerable**: 2 questions, 1 correctly handled (q25 flagged
  insufficient; q24 missed because its mangrove-related vocabulary retrieves
  mangroves lexically above the Phase 3 relevance threshold). Excluded from
  Recall@K/MRR.
- **Precision@5 = 0.2522, Precision@10 = 0.1561** — precision was included
  (negligible extra code over Recall@K), but the corpus is small so chunk-level
  retrieval over 11 sources keeps source-level precision low.
- **Judge**: 25/25 unavailable (no API key) — the distinguishing behavior
  required by the spec, not zeros.
- 5 answerable questions trip Phase 3's insufficient gate offline (max
  deterministic similarity below 0.15, e.g. kelp retrieval peaks at ~0.12), so
  their generation slots record an insufficiency proxy rather than a fabricated
  answer.

### 8.8 Known limitations

- **Judge metrics require GOOGLE_API_KEY.** This run records them as
  unavailable; with a key set, `run_evaluation` automatically runs the real
  Gemini judge and records per-question 0–1 scores plus means.
- **Structured-output judgement is model-dependent**; judge scores on the same
  question can vary across models/temps. The judge uses `temperature=0` and
  exact numeric anchors in the prompt to reduce this.
- **Small benchmark (25 questions)**: aggregate retrieval values are driven by
  one or two questions; treat deltas in Phase 7 with that in mind.
- **Offline insufficiency gate**: without a reranker client the gate falls back
  to Chroma similarity (Phase 3's documented fallback path); a phase with real
  reranking may reach different insufficiency outcomes.