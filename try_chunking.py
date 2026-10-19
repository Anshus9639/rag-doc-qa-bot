import sys

from src.chunking import chunk_pages
from src.loaders import DocumentLoadError, DocumentPage, load_document

if len(sys.argv) == 2:
    try:
        pages = load_document(sys.argv[1])
    except DocumentLoadError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
else:  # no file given: use a tiny built-in sample
    sample = (
        "SERVICE AGREEMENT\n\nThis Agreement is made between Acme Ltd and Beta LLC.\n\n"
        "1. Payment\nBeta LLC shall pay Acme Ltd within 30 days of each invoice.\n\n"
        "2. Termination\nEither party may terminate with 60 days written notice. "
        "Termination for breach is immediate if the breach is not cured within 14 days.\n\n"
        "3. Confidentiality\nBoth parties must keep all shared information confidential."
    )
    pages = [DocumentPage(sample, "sample.txt", None)]

chunks = chunk_pages(pages)

lengths = [len(c.text) for c in chunks]
print(f"\n{len(chunks)} chunks | avg {sum(lengths)//len(lengths)} chars | "
      f"min {min(lengths)} | max {max(lengths)}\n")

for c in chunks[:3]:
    print(f"[{c.chunk_id}] page={c.page} section={c.section!r} ({len(c.text)} chars)")
    print(c.text[:200].replace("\n", " "), "...\n")