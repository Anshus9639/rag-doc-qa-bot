import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from src.chunking import Chunk
from src.config import INDEX_DIR, ensure_dirs
from src.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised for invalid input or a corrupt/mismatched saved index."""


class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None   # created on first add (needs vector size)
        self.chunks: list[Chunk] = []           # chunks[i] describes row i of the index

    @property
    def total(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their vectors. Re-adding a document replaces its old chunks."""
        if not chunks:
            raise VectorStoreError("No chunks to add.")
        if vectors.ndim != 2 or len(chunks) != vectors.shape[0]:
            raise VectorStoreError(
                f"Got {len(chunks)} chunks but vectors of shape {vectors.shape}."
            )
        if self.index is not None and vectors.shape[1] != self.index.d:
            raise VectorStoreError(
                f"Vector size {vectors.shape[1]} doesn't match the index ({self.index.d}). "
                "Did the embedding model change? Delete data/index/* and re-index."
            )

        for source in {c.source for c in chunks}:  # replace instead of duplicating
            self.remove_document(source)

        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.chunks.extend(chunks)
        logger.info("Added %d chunk(s). Store now has %d.", len(chunks), self.total)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, similarity score) pairs, best first."""
        if self.index is None or self.total == 0:
            return []
        scores, ids = self.index.search(
            np.ascontiguousarray(query_vector, dtype="float32"), min(k, self.total)
        )
        return [
            (self.chunks[i], float(score))
            for score, i in zip(scores[0], ids[0])
            if i != -1                      # -1 means "no result"
        ]

    def list_documents(self) -> dict[str, int]:
        """Map each indexed file name to its number of chunks."""
        return dict(Counter(c.source for c in self.chunks))

    def remove_document(self, source: str) -> int:
        """Remove all chunks of one file. Returns how many were removed."""
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = self.total - len(keep)
        if removed == 0:
            return 0
        if not keep:
            self.index, self.chunks = None, []
        else:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            rebuilt = faiss.IndexFlatIP(self.index.d)
            rebuilt.add(np.ascontiguousarray(all_vectors[keep]))
            self.index = rebuilt
            self.chunks = [self.chunks[i] for i in keep]
        logger.info("Removed %d chunk(s) of %s", removed, source)
        return removed

    def save(self, directory: Path = INDEX_DIR) -> None:
        """Write index.faiss + metadata.json to disk."""
        ensure_dirs()
        directory.mkdir(parents=True, exist_ok=True)
        index_file, meta_file = directory / "index.faiss", directory / "metadata.json"
        if self.index is None:              # empty store: clear any old files
            index_file.unlink(missing_ok=True)
            meta_file.unlink(missing_ok=True)
            return
        faiss.write_index(self.index, str(index_file))
        meta_file.write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d chunk(s) to %s", self.total, directory)

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "VectorStore":
        """Load a saved store, or return an empty one if nothing is saved yet."""
        index_file, meta_file = directory / "index.faiss", directory / "metadata.json"
        store = cls()
        if not (index_file.exists() and meta_file.exists()):
            logger.info("No saved index found in %s. Starting empty.", directory)
            return store
        try:
            store.index = faiss.read_index(str(index_file))
            store.chunks = [Chunk(**d) for d in json.loads(meta_file.read_text(encoding="utf-8"))]
        except Exception as exc:
            raise VectorStoreError(f"Saved index is unreadable: {exc}") from exc
        if store.index.ntotal != len(store.chunks):
            raise VectorStoreError(
                "Index and metadata are out of sync. Delete data/index/* and re-index."
            )
        logger.info("Loaded %d chunk(s) from %s", store.total, directory)
        return store