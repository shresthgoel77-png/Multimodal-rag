"""Ingest the fixed Phase 5 evaluation corpus into the persistent Chroma store.

Usage:
    python -m evaluation.scripts.ingest_corpus [--persist <dir>] [--dry-run]

Default behavior (idempotent):
  - Loads evaluation/corpus/corpus.json, the canonical fixed corpus.
  - Installs the deterministic embedding override (eval_embeddings.py) so the
    run needs no Google API key, exactly like the repo's own unit tests.
  - Adds every source through MultimodalRagStore.add_text_source — the same
    ingestion path used by POST /sources/text and /sources/url.
  - Writes evaluation/corpus/manifest.json recording the REAL source ids
    (hex SHA-256 derived from title+content) that the benchmark file and the
    Phase 6/7 harnesses must reference.
  - Re-running replaces chunks only for sources that already exist (upsert),
    so the corpus content and ids are stable.

This script changes no Phase 1-4 library code; it only supplies embeddings to
the existing store, first-class data, and a manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
EVAL_DIR = REPO_ROOT / "evaluation"
CORPUS_PATH = EVAL_DIR / "corpus" / "corpus.json"
MANIFEST_PATH = EVAL_DIR / "corpus" / "manifest.json"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

from rag_store import MultimodalRagStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the fixed evaluation corpus.")
    parser.add_argument(
        "--persist",
        default=None,
        help="Chroma persist directory. Defaults to the store default (backend/chroma_db).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load corpus and print planned adds without writing.")
    args = parser.parse_args()

    from eval_embeddings import embed_text  # noqa: E402

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    sources = corpus["sources"]
    print(f"Corpus '{corpus['name']}' v{corpus['version']}: {len(sources)} sources\n")

    if args.persist:
        os.environ["CHROMA_PERSIST_DIRECTORY"] = args.persist

    for sp in sources:
        print(f"  would add  id={sp['id']:<24} title={sp['title']!r:<24} words={len(sp['text'].split())}")
    if args.dry_run:
        return

    store = MultimodalRagStore()
    if store.chroma_error:
        sys.exit(f"ChromaDB unavailable: {store.chroma_error}")

    # The same deterministic embedder is used at chunk time and at query time.
    store._embed_text = lambda text, task_prefix="": embed_text(text)  # type: ignore[method-assign]

    print(f"Persisting to: {store.persist_directory}\n")
    manifest_sources = []
    for sp in sources:
        source = store.add_text_source(sp["title"], sp["text"], sp.get("modality", "text"))
        manifest_sources.append(
            {
                "id": source.id,
                "corpus_id": sp["id"],
                "title": source.title,
                "modality": source.modality,
                "chunks": source.chunks,
                "summary": source.summary,
            }
        )
        print(f"  added  id={source.id}  corpus_id={sp['id']:<24} title={source.title!r:<24} chunks={source.chunks}")

    manifest_sources.sort(key=lambda item: item["corpus_id"])
    manifest = {
        "name": corpus["name"],
        "version": corpus["version"],
        "persist_directory": store.persist_directory,
        "embedding_provider": "deterministic-feature-hash",
        "dimensions": 768,
        "source_count": len(manifest_sources),
        "sources": manifest_sources,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nWrote manifest: {MANIFEST_PATH}")
    print(f"Store now holds {len(store.sources)} sources / {len(store.chunks)} chunks.")


if __name__ == "__main__":
    main()