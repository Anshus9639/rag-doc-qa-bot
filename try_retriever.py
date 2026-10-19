import sys

from src.chunking import chunk_pages
from src.config import MIN_SCORE
from src.embeddings import embed_texts
from src.loaders import DocumentLoadError, DocumentPage, load_document
from src.retriever import retrieve
from src.vector_store import VectorStore

CONTRACT = [
    DocumentPage("1. Payment\nBeta LLC shall pay Acme Ltd within 30 days of each invoice.", "contract.txt", 1),
    DocumentPage("2. Termination\nEither party may terminate with 60 days written notice.", "contract.txt", 2),
    DocumentPage("3. Confidentiality\nBoth parties must keep all shared information confidential.", "contract.txt", 3),
]
RECIPE = [
    DocumentPage("Tomato sauce: simmer fresh tomatoes with garlic and basil for 20 minutes.", "recipe.txt", 1),
]

if len(sys.argv) >= 2:
    try:
        docs = [load_document(sys.argv[1])]
    except DocumentLoadError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
else:
    docs = [CONTRACT, RECIPE]

store = VectorStore()
for pages in docs:
    chunks = chunk_pages(pages)
    store.add_chunks(chunks, embed_texts([c.text for c in chunks]))

print(f"\nIndexed: {store.list_documents()}")
print(f"Threshold MIN_SCORE = {MIN_SCORE}. Type a question (empty line to quit).")

while True:
    question = input("\nQuestion> ").strip()
    if not question:
        break
    result = retrieve(store, question)
    verdict = "ANSWERABLE" if result.answerable else "REFUSE (not in documents)"
    print(f"best score {result.best_score:+.3f} -> {verdict}")
    for chunk, score in result.candidates:
        mark = "[+]" if score >= result.min_score else "[ ]"
        print(f"  {mark} {score:+.3f}  {chunk.source} p.{chunk.page}: {chunk.text[:50]!r}")