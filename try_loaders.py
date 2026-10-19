import sys

from src.loaders import DocumentLoadError, load_document

if len(sys.argv) != 2:
    print("Usage: python try_loaders.py <path-to-file>")
    sys.exit(1)

try:
    pages = load_document(sys.argv[1])
except DocumentLoadError as error:
    print(f"ERROR: {error}")
    sys.exit(1)

print(f"\nLoaded {len(pages)} page(s) from {pages[0].source}\n")
for p in pages[:2]:  # preview only the first two
    label = f"page {p.page}" if p.page else "no page info"
    print(f"--- {label} ({len(p.text)} chars) ---")
    print(p.text[:300], "\n")