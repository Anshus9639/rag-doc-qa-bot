import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from src.config import MAX_UPLOAD_MB
from src.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


class DocumentLoadError(Exception):
    """Raised when a document can't be loaded."""


@dataclass
class DocumentPage:
    text: str
    source: str
    page: int | None


def _normalize_pdf_text(text: str) -> str:
    """Repair confirmed PDF extraction artefacts conservatively.

    Keep the default pypdf extraction mode. Do not globally remove spaces,
    because layout-dependent extraction can merge separate words.
    """
    text = text.replace("\u00a0", " ")

    def repair_guarantee(match: re.Match) -> str:
        original = match.group(0)

        if original.isupper():
            return "GUARANTEE"
        if original[0].isupper():
            return "Guarantee"
        return "guarantee"

    def repair_the(match: re.Match) -> str:
        return "The" if match.group(0)[0].isupper() else "the"

    # Confirmed issue in the labour contractor agreement.
    text = re.sub(
        r"\bguarant\s+ee\b",
        repair_guarantee,
        text,
        flags=re.IGNORECASE,
    )

    # Repair "T he" while preserving standalone words such as "A party".
    text = re.sub(
        r"\bT\s+he\b",
        repair_the,
        text,
        flags=re.IGNORECASE,
    )

    return text


def _load_pdf(path: Path) -> list[DocumentPage]:
    reader = PdfReader(path)

    if reader.is_encrypted and reader.decrypt("") == 0:
        raise DocumentLoadError(
            f"{path.name} is password-protected."
        )

    pages = []

    for number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        text = _normalize_pdf_text(extracted).strip()

        if text:
            pages.append(
                DocumentPage(
                    text=text,
                    source=path.name,
                    page=number,
                )
            )

    return pages


def _load_txt(path: Path) -> list[DocumentPage]:
    text = ""

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    text = text.strip()

    return [
        DocumentPage(text, path.name, None)
    ] if text else []


def _load_docx(path: Path) -> list[DocumentPage]:
    doc = Document(str(path))
    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    )

    return [
        DocumentPage(text, path.name, None)
    ] if text else []


_LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".docx": _load_docx,
}


def load_document(file_path: str | Path) -> list[DocumentPage]:
    """Validate a file, then load it into DocumentPage objects."""
    path = Path(file_path)

    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{extension}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)

    if size_mb > MAX_UPLOAD_MB:
        raise DocumentLoadError(
            f"{path.name} is {size_mb:.1f} MB "
            f"(limit {MAX_UPLOAD_MB} MB)."
        )

    try:
        pages = _LOADERS[extension](path)
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(
            f"Could not read {path.name}: {exc}"
        ) from exc

    if not pages:
        raise DocumentLoadError(
            f"No extractable text in {path.name}. "
            "It may be a scanned/image-only document."
        )

    logger.info(
        "Loaded %s: %d page(s)/section(s)",
        path.name,
        len(pages),
    )

    return pages