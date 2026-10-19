import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.logger import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


class EmbeddingError(Exception):
    """Raised when the embedding model can't be loaded or run."""


def _get_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading takes seconds, so we don't repeat it)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{EMBEDDING_MODEL}'. "
                f"Check the name in .env and your internet connection (first run downloads it). "
                f"Details: {exc}"
            ) from exc
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Turn a list of texts into a (len(texts), 384) float32 array of unit-length vectors."""
    if not texts:
        raise EmbeddingError("No texts to embed.")
    model = _get_model()
    try:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # unit length -> dot product == cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc
    return vectors.astype("float32")  # FAISS expects float32


def embed_query(question: str) -> np.ndarray:
    """Embed a single question. Returns shape (1, 384)."""
    if not question or not question.strip():
        raise EmbeddingError("Question is empty.")
    return embed_texts([question.strip()])