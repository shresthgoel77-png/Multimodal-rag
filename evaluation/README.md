# Phase 5 — Fixed Evaluation Benchmark

This directory holds the **fixed evaluation corpus** and the **versioned
benchmark dataset** that Phase 6 (metrics) and Phase 7 (baseline-vs-improved)
must run against. Phase 5 implements no metrics, scoring, or comparison logic —
it only pins down *what* the later phases will measure and scores nothing.

## Layout

```text
evaluation/
|-- README.md                        # this file
|-- eval_embeddings.py               # deterministic, API-free embeddings (shared by ingest/verify/tests and Phase 6-7)
|-- corpus/
|   |-- corpus.json                  # the canonical fixed corpus text (immutable for Phases 6-7)
|   `-- manifest.json                # record of the real ingestion: source_id (hex) <-> corpus_id (name)
|-- benchmark/
|   `-- benchmark_v1.json            # v1.0.0 benchmark: 25 questions with grounded ground truth
|-- scripts/
|   |-- ingest_corpus.py             # seeds the persistent Chroma store with the corpus
|   `-- verify_benchmark.py          # retrieval-based verification harness (run before trusting the file)
`-- tests/
    `-- test_benchmark.py            # hermetic schema/grounding/distribution tests
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