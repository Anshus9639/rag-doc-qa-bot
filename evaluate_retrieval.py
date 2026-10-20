import json
import re
from pathlib import Path

from src.config import TOP_K
from src.pipeline import (
    RAGPipeline,
    SECTION_QUERY_RE,
    _keyword_chunks,
    _named_heading_chunks,
    _section_chunks,
)
from src.retriever import retrieve


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "evals" / "cases.json"


def normalize(text: str) -> str:
    """Normalize whitespace and case for reliable metadata matching."""
    return re.sub(r"\s+", " ", text).casefold().strip()


def retrieve_hybrid(pipeline, question, source):
    """Reproduce the pipeline's retrieval order without calling Ollama."""
    store = pipeline.store
    sources = {source}

    # 1. Named-heading lookup
    named = _named_heading_chunks(store, question, sources)
    if named:
        return named, "named-heading"

    # 2. Numbered section or clause lookup
    match = SECTION_QUERY_RE.search(question)

    if match:
        numbers = re.findall(r"\d+(?:\.\d+)*", match.group(1))
        selected = []
        seen = set()

        for number in numbers:
            for chunk in _section_chunks(store, number, sources):
                key = (
                    chunk.source,
                    chunk.page,
                    chunk.section,
                    chunk.text,
                )
                if key not in seen:
                    seen.add(key)
                    selected.append(chunk)

        if selected:
            return selected, "numbered-clause"

    # 3. Keyword retrieval
    keyword_hits = _keyword_chunks(store, question, sources)
    if keyword_hits:
        return [chunk for chunk, _score in keyword_hits], "keyword"

    # 4. Semantic retrieval
    result = retrieve(
        store,
        question,
        k=TOP_K,
        sources=sources,
    )

    if not result.answerable:
        return [], "semantic-below-threshold"

    return [chunk for chunk, _score in result.candidates], "semantic"


def main():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()
    documents = pipeline.documents()

    evaluated = 0
    skipped = 0
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks = []

    print("RAG Retrieval Evaluation — Page/Section Targets")
    print(f"Indexed documents: {documents}\n")

    for case in cases:
        if not case.get("should_answer", False):
            continue

        case_id = case["id"]
        source = case["source"]
        expected_page = case.get("expected_page")
        expected_section = case.get("expected_section_contains")

        if source not in documents:
            print(f"SKIP {case_id}: source is not indexed")
            skipped += 1
            continue

        if expected_page is None or not expected_section:
            print(f"SKIP {case_id}: expected page/section label is missing")
            skipped += 1
            continue

        # Ground-truth labels are based on annotated page and section,
        # not on matching the question's keywords against chunk text.
        gold = [
            chunk
            for chunk in pipeline.store.chunks
            if chunk.source == source
            and chunk.page == expected_page
            and normalize(expected_section)
            in normalize(chunk.section or "")
        ]

        if not gold:
            print(
                f"ANNOTATION ERROR {case_id}: no indexed chunk matches "
                f"page {expected_page}, section '{expected_section}'. "
                "Verify the PDF page numbering and section metadata."
            )
            skipped += 1
            continue

        gold_ids = {chunk.chunk_id for chunk in gold}

        try:
            hits, route = retrieve_hybrid(
                pipeline,
                case["question"],
                source,
            )
        except Exception as exc:
            print(f"ERROR {case_id}: {exc}")
            skipped += 1
            continue

        ranks = [
            rank
            for rank, chunk in enumerate(hits, start=1)
            if chunk.chunk_id in gold_ids
        ]
        first_rank = min(ranks) if ranks else None

        evaluated += 1
        hit_at_1 += int(first_rank == 1)
        hit_at_3 += int(first_rank is not None and first_rank <= 3)
        reciprocal_ranks.append(
            1 / first_rank if first_rank is not None else 0.0
        )

        print(f"{'HIT' if first_rank else 'MISS'} {case_id}")
        print(f"  Retrieval route: {route}")
        print(f"  Expected page: {expected_page}")
        print(f"  Expected section: {expected_section}")
        print(f"  Matching target chunks: {len(gold)}")
        print(f"  Retrieved chunks: {len(hits)}")
        print(f"  First target rank: {first_rank or 'not retrieved'}")

        for rank, chunk in enumerate(hits[:3], start=1):
            marker = "*" if chunk.chunk_id in gold_ids else " "
            print(
                f"  {marker} Rank {rank}: page={chunk.page}, "
                f"section={chunk.section}"
            )

        print()

    print("=" * 48)
    print(f"Answerable cases configured: {sum(bool(c.get('should_answer')) for c in cases)}")
    print(f"Cases evaluated: {evaluated}")
    print(f"Cases skipped: {skipped}")

    if evaluated:
        print(f"Hit@1: {hit_at_1 / evaluated:.1%}")
        print(f"Hit@3: {hit_at_3 / evaluated:.1%}")
        print(f"MRR: {sum(reciprocal_ranks) / evaluated:.3f}")

    if evaluated == 0 or skipped > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()