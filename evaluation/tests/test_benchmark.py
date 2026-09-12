"""Tests for the Phase 5 fixed evaluation benchmark.

These tests are intentionally hermetic: they load the versioned benchmark file
and validate it against the fixed corpus files (corpus.json + the manifest
produced by the real ingestion run), requiring no Google API key, no Chroma,
and no network.

Covered requirements:
  - benchmark file parses and loads
  - every question matches the required object schema
  - every category value is one of the seven allowed labels
  - every expected_sources entry is a source_id that exists in the fixed corpus
  - every unanswerable question has expected_sources == []
  - no duplicate question ids
  - question count and per-category distribution match the documented targets
  - expected_answer grounding: each answer's content tokens are traceable to
    the text of its expected sources (on the actual corpus content)
  - multi-hop genuineness: no single expected source alone contains the full
    answer, and every listed source contributes at least one unique fact token
  - unanswerable questions ask about subject matter demonstrably absent from
    the corpus
"""

import json
import re
from pathlib import Path

import pytest

from eval_embeddings import tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"

BENCHMARK_PATH = EVAL_DIR / "benchmark" / "benchmark_v1.json"
CORPUS_PATH = EVAL_DIR / "corpus" / "corpus.json"
MANIFEST_PATH = EVAL_DIR / "corpus" / "manifest.json"

ALLOWED_CATEGORIES = {"factual", "semantic", "difficult", "comparison", "multi_hop", "evidence_specific", "unanswerable"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_KEYS = {"id", "question", "expected_answer", "expected_sources", "category", "difficulty"}

TARGET_COUNTS = {
    "factual": 5,
    "semantic": 5,
    "difficult": 4,
    "comparison": 3,
    "multi_hop": 4,
    "evidence_specific": 2,
    "unanswerable": 2,
}
TOTAL_TARGET = sum(TARGET_COUNTS.values())

UNANSWERABLE_SUBJECT_TERMS = {
    "q24": ["penguin", "penguins"],
    "q25": ["anglerfish", "bioluminescence", "bioluminescent", "photophore"],
}


@pytest.fixture(scope="module")
def benchmark():
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def questions(benchmark):
    return benchmark["questions"]


@pytest.fixture(scope="module")
def source_id_set(manifest):
    return {source["id"] for source in manifest["sources"]}


@pytest.fixture(scope="module")
def text_by_source_id(corpus, manifest):
    by_corpus_id = {source["id"]: source["text"] for source in corpus["sources"]}
    mapping = {}
    for source in manifest["sources"]:
        mapping[source["id"]] = by_corpus_id[source["corpus_id"]]
    return mapping


def content_tokens(text: str) -> set[str]:
    """Lowercased tokens of length >= 5 (drops stopwords/numbers/short words).

    Length >= 5 keeps the ground-check about distinctive content words rather
    than function words or tiny numerals.
    """
    return {token for token in tokenize(text) if len(token) >= 5}


# --------------------------------------------------------------------------- schema


def test_benchmark_file_loads_and_parses():
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["version"]
    assert isinstance(data["questions"], list)


def test_every_question_matches_schema(questions):
    for item in questions:
        assert set(item.keys()) == REQUIRED_KEYS, f"{item['id']}: unexpected keys {set(item) ^ REQUIRED_KEYS}"
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["question"], str) and item["question"]
        assert isinstance(item["expected_answer"], str) and item["expected_answer"]
        assert isinstance(item["expected_sources"], list)
        assert all(isinstance(s, str) and s for s in item["expected_sources"])
        assert isinstance(item["category"], str)
        assert isinstance(item["difficulty"], str)


def test_every_category_is_allowed(questions):
    for item in questions:
        assert item["category"] in ALLOWED_CATEGORIES, f"{item['id']}: bad category {item['category']!r}"


def test_every_difficulty_is_allowed(questions):
    for item in questions:
        assert item["difficulty"] in ALLOWED_DIFFICULTIES, f"{item['id']}: bad difficulty {item['difficulty']!r}"


# --------------------------------------------------------------------------- ids + sources


def test_no_duplicate_question_ids(questions):
    ids = [item["id"] for item in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids present"


def test_expected_sources_exist_in_fixed_corpus(questions, source_id_set):
    for item in questions:
        for source_id in item["expected_sources"]:
            assert source_id in source_id_set, (
                f"{item['id']}: expected_source {source_id} is not an ingested source id"
            )


def test_unanswerable_have_empty_expected_sources(questions):
    for item in questions:
        if item["category"] == "unanswerable":
            assert item["expected_sources"] == [], f"{item['id']}: unanswerable must have no expected sources"


def test_non_unanswerable_have_sources(questions):
    for item in questions:
        if item["category"] != "unanswerable":
            assert item["expected_sources"], f"{item['id']}: expected_sources must be non-empty"


# --------------------------------------------------------------------------- distribution


def test_total_question_count_matches_target(questions):
    assert len(questions) == TOTAL_TARGET


def test_category_distribution_matches_target(benchmark, questions):
    counts = {}
    for item in questions:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    assert counts == TARGET_COUNTS
    assert benchmark["counts_by_category"] == TARGET_COUNTS


# --------------------------------------------------------------------------- grounding


def test_expected_answer_grounded_in_expected_sources(questions, text_by_source_id):
    for item in questions:
        if item["category"] == "unanswerable":
            continue
        combined = " ".join(text_by_source_id[sid] for sid in item["expected_sources"]).lower()
        tokens = content_tokens(item["expected_answer"])
        missing = sorted(t for t in tokens if t not in combined)
        assert not missing, (
            f"{item['id']}: answer tokens not found in expected-source text: {missing}"
        )


def test_multi_hop_requires_all_sources(questions, text_by_source_id):
    """No single expected source alone contains the full answer, and each
    listed source contributes at least one answer token unique to it."""
    for item in questions:
        if item["category"] != "multi_hop":
            continue
        sources = item["expected_sources"]
        assert len(sources) >= 2, f"{item['id']}: multi_hop must cite >= 2 sources"
        tokens = content_tokens(item["expected_answer"])
        per_source = {sid: tokens & content_tokens(text_by_source_id[sid]) for sid in sources}
        for sid in sources:
            assert per_source[sid], f"{item['id']}: source {sid} contributes no answer tokens"
            other_tokens = set().union(*(per_source[other] for other in sources if other != sid))
            unique_to_this = sorted(per_source[sid] - other_tokens)
            assert unique_to_this, f"{item['id']}: source {sid} adds no unique fact; full answer lives elsewhere"
        single_coverage = max(len(per_source[sid]) for sid in sources)
        assert single_coverage < len(tokens), f"{item['id']}: one source alone covers the whole answer"


# --------------------------------------------------------------------------- unanswerable


def test_unanswerable_subject_absent_from_corpus(questions, corpus):
    all_text = " ".join(source["text"] for source in corpus["sources"]).lower()
    for item in questions:
        if item["category"] != "unanswerable":
            continue
        for term in UNANSWERABLE_SUBJECT_TERMS[item["id"]]:
            assert term not in all_text, f"{item['id']}: subject term {term!r} IS present in the corpus"


def test_unanswerable_questions_count(questions):
    assert sum(1 for item in questions if item["category"] == "unanswerable") == TARGET_COUNTS["unanswerable"]