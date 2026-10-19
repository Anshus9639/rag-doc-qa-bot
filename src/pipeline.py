import re
from dataclasses import dataclass, field
from pathlib import Path

from src.chunking import Chunk, chunk_pages
from src.embeddings import embed_texts
from src.llm import generate_answer
from src.loaders import load_document
from src.logger import get_logger
from src.prompts import NOT_FOUND_MESSAGE, build_messages
from src.retriever import retrieve
from src.vector_store import VectorStore

logger = get_logger(__name__)

MAX_SECTION_CHUNKS = 6

# Supports singular/plural references and multiple numbered sections, e.g.:
# "Article 5.9", "Sections 3 and 4", "Articles 4.6 and 7.1".
SECTION_QUERY_RE = re.compile(
    r"\b(?:sections?|articles?|clauses?|chapters?|parts?)\s+"
    r"(\d+(?:\.\d+)*(?:\s*(?:,|and|&)\s*\d+(?:\.\d+)*)*)",
    re.IGNORECASE,
)

CITATION_RE = re.compile(r"\[(\d+)\]")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
WORD_RE = re.compile(r"[a-z0-9]{2,}")

MAX_KEYWORD_CHUNKS = 4
MIN_KEYWORD_OVERLAP = 0.50

# Common words that do not help identify a passage. Keep important negations
# such as "not" because they can change the meaning of a contract clause.
SEARCH_STOPWORDS = {
    "a", "about", "according", "after", "all", "also", "an", "and",
    "any", "are", "as", "at", "be", "been", "before", "both", "by",
    "can", "clause", "contract", "does", "for", "from", "give", "how",
    "in", "into", "is", "it", "its", "of", "on", "or", "party",
    "parties", "please", "should", "that", "the", "their", "them",
    "there", "this", "to", "under", "what", "when", "where", "which",
    "who", "why", "with", "would", "you", "your", "first", "second",
    "agreement", "provide", "provided", "provision", "provisions",
}


def _search_terms(text: str) -> set[str]:
    """Return informative words for lightweight lexical retrieval."""
    return set(WORD_RE.findall(text.lower())) - SEARCH_STOPWORDS


def _named_heading_chunks(
    store: VectorStore,
    question: str,
    sources: set[str] | None,
) -> list[Chunk]:
    """Find chunks whose section heading matches a named clause in the question."""
    query_terms = _search_terms(question)
    if len(query_terms) < 2:
        return []

    ranked: list[tuple[float, Chunk]] = []
    for chunk in store.chunks:
        if sources and chunk.source not in sources:
            continue

        # Prefer metadata; fall back to the opening text when no heading exists.
        heading = chunk.section or chunk.text[:140]
        heading_terms = _search_terms(heading)
        matched = query_terms & heading_terms

        # Require at least two meaningful heading words so generic questions
        # do not accidentally select a section just because one word overlaps.
        if len(matched) >= 2:
            score = len(matched) / max(len(query_terms), 1)
            ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    result = [chunk for _score, chunk in ranked[:MAX_SECTION_CHUNKS]]
    if result:
        logger.info(
            "Named-heading lookup for question found %d chunk(s): %s",
            len(result),
            "; ".join((chunk.section or "unknown section") for chunk in result),
        )
    return result


def _keyword_chunks(
    store: VectorStore,
    question: str,
    sources: set[str] | None,
) -> list[tuple[Chunk, float]]:
    """Find strong lexical matches when semantic top-k misses the exact passage."""
    query_terms = _search_terms(question)
    if len(query_terms) < 2:
        return []

    ranked: list[tuple[float, Chunk]] = []
    for chunk in store.chunks:
        if sources and chunk.source not in sources:
            continue

        chunk_terms = _search_terms(f"{chunk.section or ''}\n{chunk.text}")
        matched = query_terms & chunk_terms
        if len(matched) < 2:
            continue

        overlap = len(matched) / max(len(query_terms), 1)
        if overlap >= MIN_KEYWORD_OVERLAP:
            ranked.append((overlap, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    results = [(chunk, score) for score, chunk in ranked[:MAX_KEYWORD_CHUNKS]]
    if results:
        logger.info(
            "Keyword retrieval found %d chunk(s); best lexical overlap %.3f",
            len(results),
            results[0][1],
        )
    return results


@dataclass
class Citation:
    number: int
    source: str
    page: int | None
    section: str | None
    score: float
    snippet: str


@dataclass
class Answer:
    text: str
    answered: bool
    citations: list[Citation] = field(default_factory=list)
    best_score: float = 0.0
    note: str = ""


def _section_chunks(
    store: VectorStore,
    number: str,
    sources: set[str] | None,
) -> list[Chunk]:
    """Find chunks for a numbered section or clause.

    For dotted references such as 5.9, check both the chunk's section
    metadata and the chunk text. For broader references such as section 5,
    return chunks whose section headings begin with 5 or one of its clauses.
    """
    eligible = [
        chunk
        for chunk in store.chunks
        if not sources or chunk.source in sources
    ]

    # Match the requested clause and any nested subsection, for example
    # 4.6 can match metadata headed "4.6 ..." or "4.6.1 ...".
    clause_heading_pattern = re.compile(
        rf"^\s*(?:(?:section|article|clause|chapter|part)\s+)?"
        rf"{re.escape(number)}(?:\.\d+)*(?:[.)])?(?:\s|$)",
        re.IGNORECASE,
    )
    clause_text_pattern = re.compile(
        rf"(?m)^\s*{re.escape(number)}(?:\.\d+)*(?:[.)])?\s+",
        re.IGNORECASE,
    )

    if "." in number:
        clause_chunks = [
            chunk
            for chunk in eligible
            if (
                (chunk.section and clause_heading_pattern.match(chunk.section))
                or clause_text_pattern.search(chunk.text)
            )
        ]

        if clause_chunks:
            logger.info(
                "Exact clause lookup '%s' found %d chunk(s)",
                number,
                len(clause_chunks),
            )
            return clause_chunks[:MAX_SECTION_CHUNKS]

    # Also supports broader requests such as "Summarize section 5".
    heading_pattern = re.compile(
        rf"^(?:(?:section|article|clause|chapter|part)\s+)?"
        rf"{re.escape(number)}(?:\.\d+)*(?:[.)])?(?:\s|$)",
        re.IGNORECASE,
    )

    heading_chunks = [
        chunk
        for chunk in eligible
        if chunk.section and heading_pattern.match(chunk.section)
    ]

    if heading_chunks:
        logger.info(
            "Section lookup '%s' found %d chunk(s)",
            number,
            len(heading_chunks),
        )

    return heading_chunks[:MAX_SECTION_CHUNKS]


def _unsupported_numbers(
    answer: str,
    hits: list[tuple[Chunk, float]],
) -> list[str]:
    """Find numbers in the answer that appear nowhere in retrieved excerpts."""
    haystack = " ".join(
        f"{chunk.text} {chunk.section or ''}"
        for chunk, _score in hits
    )

    # Citation labels such as [1] are not factual numbers.
    cleaned = CITATION_RE.sub(" ", answer)
    missing = []

    for raw_number in NUMBER_RE.findall(cleaned):
        number = raw_number.rstrip(".,")
        if not re.search(rf"(?<!\d){re.escape(number)}(?!\d)", haystack):
            missing.append(number)

    return missing


def _make_citation(
    number: int,
    chunk: Chunk,
    score: float,
    answer_text: str = "",
) -> Citation:
    """Build a citation with a snippet focused on the answer's evidence."""
    normalized = re.sub(r"\s+", " ", chunk.text).strip()
    limit = 400
    start = 0

    if len(normalized) > limit:
        answer_terms = {
            word
            for word in WORD_RE.findall(
                CITATION_RE.sub(" ", answer_text).lower()
            )
            if len(word) >= 3 and word not in SEARCH_STOPWORDS
        }

        lower_text = normalized.lower()
        positions = []

        for term in answer_terms:
            positions.extend(
                match.start()
                for match in re.finditer(
                    rf"\b{re.escape(term)}\b",
                    lower_text,
                )
            )

        if positions:
            max_start = len(normalized) - limit
            candidate_starts = {
                max(0, min(position - 100, max_start))
                for position in positions
            }

            def window_score(window_start: int) -> tuple[int, int]:
                window = lower_text[window_start:window_start + limit]
                terms_found = sum(
                    bool(re.search(rf"\b{re.escape(term)}\b", window))
                    for term in answer_terms
                )
                occurrences = sum(
                    window_start <= position < window_start + limit
                    for position in positions
                )
                return terms_found, occurrences

            start = max(candidate_starts, key=window_score)

    end = min(start + limit, len(normalized))
    snippet = normalized[start:end]

    if start > 0:
        snippet = "…" + snippet
    if end < len(normalized):
        snippet += "…"

    return Citation(
        number=number,
        source=chunk.source,
        page=chunk.page,
        section=chunk.section,
        score=score,
        snippet=snippet,
    )


def _build_citations(
    text: str,
    hits: list[tuple[Chunk, float]],
) -> list[Citation]:
    """Map citations to retrieved chunks and prefer relevant evidence."""
    if not hits:
        return []

    # Honor explicit, valid answer markers first.
    cited_numbers = sorted(
        {
            int(number)
            for number in CITATION_RE.findall(text)
            if 1 <= int(number) <= len(hits)
        }
    )

    if cited_numbers:
        return [
            _make_citation(
                number,
                hits[number - 1][0],
                hits[number - 1][1],
                answer_text=text,
            )
            for number in cited_numbers
        ]

    # Without a valid marker, select chunks by lexical overlap.
    answer_words = set(
        WORD_RE.findall(CITATION_RE.sub(" ", text).lower())
    )
    scored = []

    for number, (chunk, score) in enumerate(hits, start=1):
        chunk_words = set(WORD_RE.findall(chunk.text.lower()))
        overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
        scored.append((overlap, number, chunk, score))

    best_overlap = max(item[0] for item in scored)

    if best_overlap > 0:
        cutoff = max(0.5 * best_overlap, 0.2)
        picked = [
            item
            for item in sorted(scored, key=lambda item: -item[0])
            if item[0] >= cutoff
        ]

        return [
            _make_citation(
                number,
                chunk,
                score,
                answer_text=text,
            )
            for _overlap, number, chunk, score in picked
        ]

    # Last resort: cite the top retrieved chunk, without claiming certainty.
    return [
        _make_citation(
            1,
            hits[0][0],
            hits[0][1],
            answer_text=text,
        )
    ]

class RAGPipeline:
    def __init__(self) -> None:
        self.store = VectorStore.load()

    def documents(self) -> dict[str, int]:
        """Return indexed documents and their chunk counts."""
        return self.store.list_documents()

    def ingest_file(self, path: str | Path) -> int:
        """Load, chunk, embed, index, and persist a document."""
        pages = load_document(path)
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError(f"No text chunks could be extracted from: {path}")

        embeddings = embed_texts([chunk.text for chunk in chunks])
        self.store.add_chunks(chunks, embeddings)
        self.store.save()
        return len(chunks)

    def remove_document(self, name: str) -> int:
        """Remove a document from the index and persist the change."""
        removed = self.store.remove_document(name)
        if removed:
            self.store.save()
        return removed

    def ask(
        self,
        question: str,
        sources: set[str] | None = None,
    ) -> Answer:
        """Retrieve evidence and generate an answer grounded in that evidence."""
        if not question or not question.strip():
            return Answer(
                "Please enter a question.",
                False,
                note="empty question",
            )

        if self.store.total == 0:
            return Answer(
                "No documents are indexed yet. Add a document first.",
                False,
                note="empty store",
            )

        question = question.strip()
        hits: list[tuple[Chunk, float]] | None = None
        best = 0.0

        # 1. First, try a named heading such as "offer and acceptance clause".
        # This catches heading-based questions that contain no clause number.
        named_chunks = _named_heading_chunks(self.store, question, sources)
        if named_chunks:
            hits = [(chunk, 1.0) for chunk in named_chunks]
            best = 1.0

        # 2. Direct lookup for one or more numbered references. Examples:
        # "Summarize section 3", "Explain article 5.9", and
        # "According to Articles 4.6 and 7.1, ...".
        match = SECTION_QUERY_RE.search(question)

        if hits is None and match:
            section_numbers = re.findall(r"\d+(?:\.\d+)*", match.group(1))
            selected_chunks: list[Chunk] = []
            seen_chunks: set[tuple[str, int | None, str | None, str]] = set()

            for section_number in section_numbers:
                chunks = _section_chunks(
                    self.store,
                    section_number,
                    sources,
                )

                for chunk in chunks:
                    key = (
                        chunk.source,
                        chunk.page,
                        chunk.section,
                        chunk.text,
                    )
                    if key not in seen_chunks:
                        seen_chunks.add(key)
                        selected_chunks.append(chunk)

            if selected_chunks:
                hits = [(chunk, 1.0) for chunk in selected_chunks]
                best = 1.0
                logger.info(
                    "Direct section/clause lookup %s found %d unique chunk(s)",
                    ", ".join(section_numbers),
                    len(selected_chunks),
                )
            else:
                logger.info(
                    "Direct section/clause lookup found no chunks for: %s",
                    ", ".join(section_numbers),
                )

        # 3. When no heading or numbered clause matched, try lexical retrieval.
        # This catches key phrases such as "minimum volume of work" that may
        # fall across chunk boundaries and be missed by semantic top-k alone.
        if hits is None:
            keyword_hits = _keyword_chunks(self.store, question, sources)
            if keyword_hits:
                hits = keyword_hits
                best = keyword_hits[0][1]
                logger.info(
                    "Using keyword retrieval for question (best overlap %.3f)",
                    best,
                )

        # 4. If direct and keyword lookup found nothing strong, use semantic search.
        if hits is None:
            result = retrieve(
                self.store,
                question,
                sources=sources,
            )
            best = result.best_score

            logger.info(
                "Question retrieval score: %.3f; answerable: %s",
                best,
                result.answerable,
            )

            if not result.answerable:
                logger.warning(
                    "Refusing question because the best chunk is below "
                    "the similarity threshold."
                )
                return Answer(
                    NOT_FOUND_MESSAGE,
                    False,
                    best_score=best,
                    note="best chunk is below the similarity threshold",
                )

            hits = result.candidates

        # 5. Build messages and log the context actually shown to the LLM.
        messages = build_messages(question, hits)
       

        # 6. Generate an answer using the existing grounded prompt.
        raw = generate_answer(messages).strip()

        if NOT_FOUND_MESSAGE.rstrip(".").lower() in raw.lower():
            logger.info("Question refused by the model: %s", question)
            return Answer(
                NOT_FOUND_MESSAGE,
                False,
                best_score=best,
                note="model said not found",
            )

        # 7. Strip citation markers that refer to nonexistent excerpts.
        raw = CITATION_RE.sub(
            lambda citation: (
                citation.group(0)
                if 1 <= int(citation.group(1)) <= len(hits)
                else ""
            ),
            raw,
        )
        raw = re.sub(r"[ \t]{2,}", " ", raw).strip()

        # 8. Reject numeric claims whose numeric tokens are absent from context.
        missing = _unsupported_numbers(raw, hits)
        if missing:
            logger.warning(
                "Rejected answer with unsupported numbers: %s",
                missing,
            )
            return Answer(
                NOT_FOUND_MESSAGE,
                False,
                best_score=best,
                note=(
                    "answer contained numbers not found in the documents: "
                    f"{missing}"
                ),
            )

        # 9. Build citations from retrieved metadata, not from model claims alone.
        citations = _build_citations(raw, hits)
        return Answer(
            text=raw,
            answered=True,
            citations=citations,
            best_score=best,
        )
