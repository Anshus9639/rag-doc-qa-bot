from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from src.config import MAX_UPLOAD_MB
from src.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


class DocumentLoadError(Exception):
    """Raised when a document can't be loaded (bad type, too big, unreadable...)."""


@dataclass
class DocumentPage:
    text: str
    source: str          # file name, used later for citations
    page: int | None     # 1-based page number; None if the format has no pages


def _load_pdf(path: Path) -> list[DocumentPage]:
    reader = PdfReader(path)
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise DocumentLoadError(f"{path.name} is password-protected.")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:  # skip blank pages
            pages.append(DocumentPage(text, path.name, number))
    return pages


def _load_txt(path: Path) -> list[DocumentPage]:
    text = ""
    for encoding in ("utf-8-sig", "latin-1"):  # latin-1 never fails, so it's the fallback
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    text = text.strip()
    return [DocumentPage(text, path.name, None)] if text else []


def _load_docx(path: Path) -> list[DocumentPage]:
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [DocumentPage(text, path.name, None)] if text else []


_LOADERS = {".pdf": _load_pdf, ".txt": _load_txt, ".docx": _load_docx}


def load_document(file_path: str | Path) -> list[DocumentPage]:
    """Validate a file, then load it into a list of DocumentPage objects."""
    path = Path(file_path)

    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{extension}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise DocumentLoadError(f"{path.name} is {size_mb:.1f} MB (limit {MAX_UPLOAD_MB} MB).")

    try:
        pages = _LOADERS[extension](path)
    except DocumentLoadError:
        raise
    except Exception as exc:  # corrupt file, parser bug, etc.
        raise DocumentLoadError(f"Could not read {path.name}: {exc}") from exc

    if not pages:
        raise DocumentLoadError(
            f"No extractable text in {path.name}. It may be a scanned/image-only document."
        )

    logger.info("Loaded %s: %d page(s)/section(s)", path.name, len(pages))
    return pages