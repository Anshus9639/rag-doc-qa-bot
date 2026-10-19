from dataclasses import dataclass

from src.chunking import Chunk
from src.config import MIN_SCORE, TOP_K
from src.embeddings import embed_query
from src.logger import get_logger
from src.vector_store import VectorStore

logger = get_logger(__name__)


class RetrievalError(Exception):
    """Raised for invalid retrieval requests."""


@dataclass
class RetrievalResult:
    question: str
    candidates: list[tuple[Chunk, float]]   # top-k chunks with scores, best first
    min_score: float

    @property
    def hits(self) -> list[tuple[Chunk, float]]:
        """Only the candidates that clear the similarity threshold."""
        return [(c, s) for c, s in self.candidates if s >= self.min_score]

    @property
    def best_score(self) -> float:
        return self.candidates[0][1] if self.candidates else 0.0

    @property
    def answerable(self) -> bool:
        """True if at least one chunk is relevant enough to answer from."""
        return len(self.hits) > 0


def retrieve(
    store: VectorStore,
    question: str,
    k: int = TOP_K,
    min_score: float = MIN_SCORE,
    sources: set[str] | None = None,
) -> RetrievalResult:
    """Find the chunks most relevant to `question`.

    sources: optional set of file names to restrict the search to.
    """
    if not question or not question.strip():
        raise RetrievalError("Question is empty.")
    if k < 1:
        raise RetrievalError("k must be at least 1.")

    if store.total == 0:
        logger.warning("Retrieval attempted on an empty store.")
        return RetrievalResult(question, [], min_score)

    query_vector = embed_query(question)

    if sources:
        # Search everything, then filter by file (fine at our scale).
        found = store.search(query_vector, store.total)
        candidates = [(c, s) for c, s in found if c.source in sources][:k]
    else:
        candidates = store.search(query_vector, k)

    result = RetrievalResult(question.strip(), candidates, min_score)
    logger.info(
        "Retrieved %d candidate(s), best score %.3f, %d above threshold %.2f",
        len(candidates), result.best_score, len(result.hits), min_score,
    )
    return result