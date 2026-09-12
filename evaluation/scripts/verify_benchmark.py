"""Manual-verification harness for the Phase 5 benchmark.

For every question this:
  1. Runs the real deterministic retrieval path
     (store.retrieve_candidates -> Chroma, same code the /ask endpoint calls
     for its candidate stage) against the persistent backend/chroma_db store.
  2. Checks that every expected_source appears among the top `top_n`
     retrieved candidates and prints its similarity.
  3. For unanswerable questions, checks that a set of distinctive topic tokens
     appears nowhere in the corpus text (the corpus genuinely does not
     contain the subject matter the question asks about).
  4. Exits non-zero if any expected source is missing from the top-N
     candidates or an unanswerable check fails.

Run:
    python -m evaluation.scripts.verify_benchmark [--top-n 8]

This is a verification aid for building the benchmark; it is not the Phase 6
metric harness. Expected answers are validated by the tests in
evaluation/tests/test_benchmark.py (grounding against corpus text) plus the
token checks printed here for every question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
EVAL_DIR = REPO_ROOT / "evaluation"
CORPUS_PATH = EVAL_DIR / "corpus" / "corpus.json"
MANIFEST_PATH = EVAL_DIR / "corpus" / "manifest.json"
BENCHMARK_PATH = EVAL_DIR / "benchmark" / "benchmark_v1.json"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

from rag_store import MultimodalRagStore  # noqa: E402
from eval_embeddings import embed_text  # noqa: E402


# Distinctive subject terms that an answerable version of each unanswerable
# question would have to mention. Verified manually to be absent from every
# source of the fixed corpus; these are the entities the question is really
# about, not the generic vocabulary (mangrove, roots, ...) shared with the rest
# of the corpus. Kept as an explicit map so the absence check is auditable.
UNANSWERABLE_SUBJECT_TERMS = {
    "q24": ["penguin", "penguins"],
    "q25": ["anglerfish", "bioluminescence", "bioluminescent", "photophore"],
}


def join_corpus_text(corpus: dict) -> str:
    return " ".join(source["text"] for source in corpus["sources"])


def check_unanswerable(question: dict, corpus: dict) -> tuple[bool, list[str]]:
    """Return (ok, matched_terms).

    ok == True only when NONE of the question's distinctive subject terms
    appear anywhere in the corpus text, i.e. the corpus genuinely contains
    nothing about the entity the question asks about.
    """
    terms = UNANSWERABLE_SUBJECT_TERMS.get(question["id"], [])
    all_text = join_corpus_text(corpus).lower()
    matched = sorted(term for term in terms if term in all_text)
    return len(matched) == 0, matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=8, help="candidate window to check expected sources inside")
    args = parser.parse_args()

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    corpus_by_id = {source["id"]: source for source in corpus["sources"]}
    id_by_corpus_id = {source["corpus_id"]: source["id"] for source in manifest["sources"]}
    title_by_id = {source["id"]: source["title"] for source in manifest["sources"]}

    store = MultimodalRagStore()
    if store.chroma_error:
        sys.exit(f"ChromaDB unavailable: {store.chroma_error}")
    store._embed_text = lambda text, task_prefix="": embed_text(text)  # type: ignore[method-assign]
    print(f"Store: {len(store.sources)} sources / {len(store.chunks)} chunks @ {store.persist_directory}\n")

    failures = 0
    checks = 0
    for q in benchmark["questions"]:
        qid = q["id"]
        category = q["category"]
        expected = list(q["expected_sources"])
        retrieval = store.retrieve_candidates(q["question"], candidate_k=max(24, args.top_n))
        candidates = retrieval.get("candidates") or []
        top = candidates[: args.top_n]

        seen = {c["source_id"] for c in top}
        found = [sid for sid in expected if sid in seen]
        missing = [sid for sid in expected if sid not in seen]

        similarity_map = {}
        for c in candidates:
            similarity_map.setdefault(c["source_id"], c["similarity"])

        ok = not missing
        if not ok:
            failures += 1
        checks += 1

        print(f"[{qid}] {category:<16} {q['question'][:88]}")
        print(f"      expected: {', '.join(f'{title_by_id.get(sid, sid)} ({sid})' for sid in expected) or '(none)'}"
              f"  -> top-{args.top_n}: {'OK' if ok else 'MISSING'}  (found {len(found)}/{len(expected)})")
        for sid in expected:
            score = similarity_map.get(sid)
            rank = next((i for i, c in enumerate(top) if c["source_id"] == sid), None)
            print(f"        {title_by_id.get(sid, sid)} sim={score if score is not None else '--'} rank={rank if rank is not None else '--'}")
        if category == "unanswerable":
            u_ok, matched = check_unanswerable(q, corpus)
            checks += 1
            if not u_ok:
                failures += 1
            print(f"      unanswerable check: {'OK (subject absent from corpus)' if u_ok else f'FAIL matched={matched}'}")
        print()

    print(f"Verification done: {checks} checks, {failures} failure(s).")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()