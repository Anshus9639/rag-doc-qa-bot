import re
from dataclasses import dataclass

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.loaders import DocumentPage
from src.logger import get_logger

logger = get_logger(__name__)

# Try to split at the biggest natural boundary first, then smaller ones.
SEPARATORS = ["\n\n", "\n", ". ", " "]

# Matches headings such as "Section 3 ...", "ARTICLE 5", "3. Termination", "3.1 Payment Terms".
HEADING_RE = re.compile(
    r"^((?i:section|article|clause|chapter|part)\s+\d+[^\n]{0,70}"
    r"|\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{2,70})$"
)


@dataclass
class Chunk:
    text: str
    source: str           # file name
    page: int | None      # page number (None for TXT/DOCX)
    chunk_id: str         # unique per chunk, e.g. "contract.pdf#7"
    section: str | None   # nearest heading, e.g. "3. Termination"


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively break text into pieces no longer than `size` (separators are kept)."""
    if len(text) <= size:
        return [text]
    if not separators:  # no natural boundary left: hard cut
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    parts = [p + sep for p in parts[:-1]] + [parts[-1]]  # keep the separator
    pieces = []
    for part in parts:
        pieces.extend(_split(part, size, rest))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Combine small pieces into chunks of up to `size`, carrying `overlap` chars forward."""
    chunks, current, length = [], [], 0
    for piece in pieces:
        if current and length + len(piece) > size:
            chunks.append("".join(current).strip())
            # drop pieces from the front until only the overlap remains (and the new piece fits)
            while current and (length > overlap or length + len(piece) > size):
                length -= len(current.pop(0))
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def _is_heading(line: str) -> bool:
    return (
        bool(line)
        and len(line) <= 80
        and (not line.endswith(".") or line.isupper())
        and bool(HEADING_RE.match(line))
    )


def _split_at_headings(text: str) -> list[tuple[str | None, str]]:
    """Split page text into (heading, body) segments; a new segment starts at each heading."""
    segments: list[list] = []
    for line in text.splitlines():
        if _is_heading(line.strip()):
            segments.append([line.strip(), [line]])
        elif segments:
            segments[-1][1].append(line)
        else:
            segments.append([None, [line]])
    return [(heading, "\n".join(lines).strip()) for heading, lines in segments]


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Turn the pages of ONE document into chunks; a chunk never spans two headings."""
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks: list[Chunk] = []
    current_section = None  # heading carried forward across chunks and pages

    for page in pages:
        for heading, body in _split_at_headings(page.text):
            if heading:
                current_section = heading
            for text in _merge(_split(body, chunk_size, SEPARATORS), chunk_size, overlap):
                chunks.append(
                    Chunk(
                        text=text,
                        source=page.source,
                        page=page.page,
                        chunk_id=f"{page.source}#{len(chunks)}",
                        section=current_section,
                    )
                )

    logger.info("Created %d chunk(s) (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)
    return chunks