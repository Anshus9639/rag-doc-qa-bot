import re
import sys
from pathlib import Path

from pypdf import PdfReader


PATTERN = re.compile(
    r"guarant|volume of work|T\s+he|subjectto|subject\s+to",
    re.IGNORECASE,
)


def print_matches(text: str, limit: int = 8) -> None:
    """Print short text windows around terms worth inspecting."""
    matches = list(PATTERN.finditer(text))

    if not matches:
        print("  No target phrases found.")
        return

    for match in matches[:limit]:
        start = max(0, match.start() - 90)
        end = min(len(text), match.end() + 150)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        print(f"  ...{snippet}...")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python inspect_pdf_extraction.py "path\\to\\file.pdf"'
        )

    path = Path(sys.argv[1])

    if not path.is_file():
        raise SystemExit(f"PDF not found: {path}")

    reader = PdfReader(path)

    print(f"PDF: {path.name}")
    print(f"Pages: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages, start=1):
        print(f"\n{'=' * 60}")
        print(f"PAGE {page_number}")

        for mode in ("plain", "layout"):
            try:
                if mode == "plain":
                    text = page.extract_text() or ""
                else:
                    text = page.extract_text(
                        extraction_mode="layout"
                    ) or ""
            except (TypeError, ValueError, NotImplementedError) as exc:
                print(f"\n[{mode.upper()}] Not available: {exc}")
                continue

            print(f"\n[{mode.upper()}] Characters: {len(text)}")
            print_matches(text)


if __name__ == "__main__":
    main()