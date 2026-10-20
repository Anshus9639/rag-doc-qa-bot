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
    """Normalize whitespace and casing for phrase matching."""
    return re.sub(r"\s+", " ", text).casefold().strip()


def matches_groups(text: str, groups: list[list[str]]) -> bool:
    """Each group must have at least one matching phrase."""
    text = normalize(text)

    return all(
        any(normalize(term) in text for term in group)
        for group in groups
    )


def retrieve_hybrid(pipeline, question, source):
    """Reproduce the pipeline's retrieval precedence without calling Ollama."""
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
    keywords = _keyword_chunks(store, question, sources)
    if keywords:
        return [chunk for chunk, _score in keywords], "keyword"

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

    print("RAG Retrieval Evaluation")
    print(f"Indexed documents: {documents}\n")

    # Positive cases need annotated evidence. Refusal cases are evaluated
    # separately by evaluate_rag.py.
    for case in cases:
        if not case.get("should_answer", False):
            continue

        case_id = case["id"]
        source = case["source"]
        evidence_groups = case.get("evidence_groups", [])

        if source not in documents:
            print(f"SKIP {case_id}: source is not indexed")
            skipped += 1
            continue

        if not evidence_groups:
            print(f"SKIP {case_id}: no evidence labels defined")
            skipped += 1
            continue

        # The "gold" chunks are annotated by matching the expected
        # evidence phrases against the actual indexed document chunks.
        gold = [
            chunk
            for chunk in pipeline.store.chunks
            if chunk.source == source
            and matches_groups(chunk.text, evidence_groups)
        ]

        if not gold:
            print(
                f"ANNOTATION ERROR {case_id}: no indexed chunk contains "
                "all expected evidence groups. Review the labels or "
                "document extraction before interpreting retrieval metrics."
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

        # Rank is 1-based. A missing relevant chunk has reciprocal rank 0.
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

        status = "HIT" if first_rank is not None else "MISS"

        print(f"{status} {case_id}")
        print(f"  Retrieval route: {route}")
        print(f"  Relevant gold chunks: {len(gold)}")
        print(f"  Retrieved chunks: {len(hits)}")
        print(f"  First relevant rank: {first_rank or 'not retrieved'}")

        if hits:
            for rank, chunk in enumerate(hits[:3], start=1):
                marker = "*" if chunk.chunk_id in gold_ids else " "
                print(
                    f"  {marker} Rank {rank}: page={chunk.page}, "
                    f"section={chunk.section}"
                )

        print()

    print("=" * 48)
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