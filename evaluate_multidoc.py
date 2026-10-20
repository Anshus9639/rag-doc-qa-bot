"""Evaluate RAG retrieval and grounded answers across a synthetic corpus.

The benchmark builds an in-memory VectorStore from evals/fixtures. It does not
read, overwrite, or delete the application's persistent data/index directory.
Run from the repository root:
    python evaluate_multidoc.py
    python evaluate_multidoc.py --retrieval-only
"""

import argparse
import json
import re
from pathlib import Path

from src.chunking import chunk_pages
from src.config import TOP_K
from src.embeddings import embed_texts
from src.loaders import load_document
from src.pipeline import (
    RAGPipeline,
    SECTION_QUERY_RE,
    _keyword_chunks,
    _named_heading_chunks,
    _section_chunks,
)
from src.prompts import NOT_FOUND_MESSAGE
from src.retriever import retrieve
from src.vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BASE_DIR / "evals" / "fixtures"
CASES_PATH = BASE_DIR / "evals" / "multidoc_cases.json"
BENCHMARK_CHUNK_SIZE = 700
BENCHMARK_CHUNK_OVERLAP = 100


def normalize(text: str) -> str:
    """Normalize whitespace and casing for labels and text checks."""
    return re.sub(r"\s+", " ", text).casefold().strip()


def matches_required_groups(text: str, groups: list[list[str]]) -> bool:
    """Every group must contain at least one matching alternative."""
    normalized = normalize(text)
    return all(
        any(normalize(term) in normalized for term in group)
        for group in groups
    )


def build_benchmark_pipeline() -> RAGPipeline:
    """Build an isolated in-memory index from synthetic TXT fixtures."""
    fixture_paths = sorted(FIXTURES_DIR.glob("*.txt"))
    if len(fixture_paths) < 2:
        raise SystemExit(
            f"Expected at least 2 synthetic TXT fixtures in {FIXTURES_DIR}; "
            f"found {len(fixture_paths)}."
        )

    chunks = []
    for path in fixture_paths:
        pages = load_document(path)
        chunks.extend(
            chunk_pages(
                pages,
                chunk_size=BENCHMARK_CHUNK_SIZE,
                overlap=BENCHMARK_CHUNK_OVERLAP,
            )
        )

    if not chunks:
        raise SystemExit("No chunks were created from the synthetic fixtures.")

    vectors = embed_texts([chunk.text for chunk in chunks])
    store = VectorStore()
    store.add_chunks(chunks, vectors)

    # Bypass RAGPipeline.__init__, which loads the user's persistent index.
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = store
    print("Using an isolated in-memory benchmark index; data/index is untouched.")
    print(f"Synthetic fixture documents: {len(fixture_paths)}")
    print(f"Synthetic chunks: {store.total}")
    print(f"Indexed fixture sources: {store.list_documents()}\n")
    return pipeline


def retrieve_hybrid(pipeline: RAGPipeline, question: str):
    """Reproduce the application's retrieval order without source filtering."""
    store = pipeline.store

    named = _named_heading_chunks(store, question, sources=None)
    if named:
        return [(chunk, 1.0) for chunk in named], "named-heading"

    match = SECTION_QUERY_RE.search(question)
    if match:
        numbers = re.findall(r"\d+(?:\.\d+)*", match.group(1))
        selected = []
        seen = set()
        for number in numbers:
            for chunk in _section_chunks(store, number, sources=None):
                key = (chunk.source, chunk.page, chunk.section, chunk.text)
                if key not in seen:
                    seen.add(key)
                    selected.append(chunk)
        if selected:
            return [(chunk, 1.0) for chunk in selected], "numbered-clause"

    keyword_hits = _keyword_chunks(store, question, sources=None)
    if keyword_hits:
        return keyword_hits, "keyword"

    result = retrieve(store, question, k=TOP_K, sources=None)
    if not result.answerable:
        return [], "semantic-below-threshold"
    return result.candidates, "semantic"


def target_chunks(pipeline: RAGPipeline, case: dict):
    """Find the annotated target chunks by expected source and section."""
    source = case["expected_source"]
    expected_section = normalize(case["expected_section_contains"])
    return [
        chunk
        for chunk in pipeline.store.chunks
        if chunk.source == source
        and expected_section in normalize(chunk.section or "")
    ]


def evaluate_retrieval(pipeline: RAGPipeline, cases: list[dict]) -> tuple[int, int, float, int]:
    """Return Hit@1 count, Hit@3 count, reciprocal-rank sum, and case count."""
    answerable_cases = [case for case in cases if case.get("should_answer", False)]
    hit_at_1 = hit_at_3 = 0
    reciprocal_ranks = []
    evaluated = 0

    print("MULTI-DOCUMENT RETRIEVAL RESULTS")
    for case in answerable_cases:
        gold = target_chunks(pipeline, case)
        if not gold:
            print(
                f"ANNOTATION ERROR {case['id']}: target source/section did not "
                "match any indexed chunk. Check the fixture heading and case label."
            )
            reciprocal_ranks.append(0.0)
            evaluated += 1
            continue

        hits, route = retrieve_hybrid(pipeline, case["question"])
        gold_ids = {chunk.chunk_id for chunk in gold}
        ranks = [
            rank
            for rank, (chunk, _score) in enumerate(hits, start=1)
            if chunk.chunk_id in gold_ids
        ]
        first_rank = min(ranks) if ranks else None
        hit_at_1 += int(first_rank == 1)
        hit_at_3 += int(first_rank is not None and first_rank <= 3)
        reciprocal_ranks.append(1 / first_rank if first_rank is not None else 0.0)
        evaluated += 1

        print(
            f"{'HIT' if first_rank else 'MISS'} {case['id']} | route={route} | "
            f"target={case['expected_source']} / {case['expected_section_contains']} | "
            f"rank={first_rank if first_rank is not None else 'not retrieved'}"
        )
        for rank, (chunk, score) in enumerate(hits[:3], start=1):
            marker = "*" if chunk.chunk_id in gold_ids else " "
            print(
                f"  {marker} {rank}. {chunk.source} | {chunk.section} | "
                f"score={score:.3f}"
            )

    print("-" * 64)
    print(f"Answerable retrieval cases: {evaluated}")
    if evaluated:
        print(f"Hit@1: {hit_at_1 / evaluated:.1%}")
        print(f"Hit@3: {hit_at_3 / evaluated:.1%}")
        print(f"MRR: {sum(reciprocal_ranks) / evaluated:.3f}")
    print()
    return hit_at_1, hit_at_3, sum(reciprocal_ranks), evaluated


def evaluate_answers(pipeline: RAGPipeline, cases: list[dict]) -> tuple[int, int, int, int]:
    """Run generation and verify answer, citation location, and evidence text."""
    answer_passed = citation_passed = evidence_passed = total_passed = 0
    evaluated = 0

    print("MULTI-DOCUMENT ANSWER / CITATION / EVIDENCE RESULTS")
    for case in cases:
        evaluated += 1
        case_id = case["id"]
        should_answer = bool(case.get("should_answer", False))

        try:
            result = pipeline.ask(case["question"])
        except Exception as exc:
            print(f"FAIL {case_id}: pipeline error: {exc}\n")
            continue

        if should_answer:
            answer_ok = result.answered and matches_required_groups(
                result.text, case.get("required_groups", [])
            )
            target_source = case["expected_source"]
            expected_section = normalize(case["expected_section_contains"])
            target_citations = [
                citation
                for citation in result.citations
                if citation.source == target_source
                and expected_section in normalize(citation.section or "")
            ]
            citation_ok = bool(target_citations)
            evidence_text = " ".join(citation.snippet for citation in target_citations)
            evidence_ok = matches_required_groups(
                evidence_text, case.get("evidence_groups", [])
            )
        else:
            answer_ok = (
                not result.answered
                and result.text.strip() == NOT_FOUND_MESSAGE
            )
            citation_ok = not result.citations
            evidence_ok = True

        answer_passed += int(answer_ok)
        citation_passed += int(citation_ok)
        evidence_passed += int(evidence_ok)
        case_passed = answer_ok and citation_ok and evidence_ok
        total_passed += int(case_passed)

        print(f"{'PASS' if case_passed else 'FAIL'} {case_id}")
        print(f"  Answer check:   {'PASS' if answer_ok else 'FAIL'}")
        print(f"  Citation check: {'PASS' if citation_ok else 'FAIL'}")
        print(f"  Evidence check: {'PASS' if evidence_ok else 'FAIL'}")
        print(f"  Answer: {result.text[:220]}")
        for citation in result.citations:
            print(
                f"  Citation: {citation.source}, page={citation.page}, "
                f"section={citation.section}"
            )
            print(f"  Evidence: {citation.snippet}")
        print()

    print("-" * 64)
    print(f"Answer cases evaluated: {evaluated}")
    print(f"Cases passed: {total_passed}/{evaluated}")
    print(f"Answer checks: {answer_passed}/{evaluated}")
    print(f"Citation checks: {citation_passed}/{evaluated}")
    print(f"Evidence checks: {evidence_passed}/{evaluated}")
    print()
    return total_passed, answer_passed, citation_passed, evidence_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip Ollama generation and measure cross-document retrieval only.",
    )
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    pipeline = build_benchmark_pipeline()
    expected_sources = {
        case["expected_source"]
        for case in cases
        if case.get("should_answer", False)
    }
    missing_sources = expected_sources - set(pipeline.documents())
    if missing_sources:
        raise SystemExit(f"Benchmark cases reference missing fixtures: {sorted(missing_sources)}")

    hit_at_1, _hit_at_3, _mrr_sum, retrieval_count = evaluate_retrieval(pipeline, cases)
    retrieval_hit_at_1 = hit_at_1
    retrieval_cases = retrieval_count

    answer_failed = False
    if not args.retrieval_only:
        passed, _answers, _citations, _evidence = evaluate_answers(pipeline, cases)
        answer_failed = passed != len(cases)
    else:
        print("Skipped answer generation because --retrieval-only was supplied.")
        print("For answer/citation/refusal checks, run with Ollama available: python evaluate_multidoc.py")

    retrieval_failed = retrieval_count == 0 or retrieval_hit_at_1 != retrieval_cases
    if retrieval_failed or answer_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
