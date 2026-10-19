import sys
import tempfile
from pathlib import Path

from src.chunking import chunk_pages
from src.embeddings import embed_query, embed_texts
from src.loaders import DocumentPage, load_document
from src.vector_store import VectorStore

CONTRACT = [
    DocumentPage("1. Payment\nBeta LLC shall pay Acme Ltd within 30 days of each invoice.", "contract.txt", 1),
    DocumentPage("2. Termination\nEither party may terminate with 60 days written notice.", "contract.txt", 2),
    DocumentPage("3. Confidentiality\nBoth parties must keep all shared information confidential.", "contract.txt", 3),
]
RECIPE = [
    DocumentPage("Tomato sauce: simmer fresh tomatoes with garlic and basil for 20 minutes.", "recipe.txt", 1),
    DocumentPage("Pasta: boil in salted water for 10 minutes, then drain and mix with the sauce.", "recipe.txt", 2),
]


def ingest(store, pages):
    chunks = chunk_pages(pages)
    store.add_chunks(chunks, embed_texts([c.text for c in chunks]))


def show(store, question, k=3):
    print(f"\nQ: {question}")
    for chunk, score in store.search(embed_query(question), k):
        print(f"  {score:+.3f}  {chunk.source} p.{chunk.page}: {chunk.text[:55]!r}")


if len(sys.argv) >= 2:   # real-file mode
    docs = [load_document(sys.argv[1])]
    questions = [sys.argv[2] if len(sys.argv) > 2 else "What are the termination conditions?"]
else:                    # sample mode
    docs = [CONTRACT, RECIPE]
    questions = ["How can the contract be ended?", "How do I make tomato sauce?"]

store = VectorStore()
for pages in docs:
    ingest(store, pages)
print("\nDocuments:", store.list_documents())
for q in questions:
    show(store, q)

# Re-adding the same document must NOT create duplicates
before = store.total
ingest(store, docs[0])
print(f"\nRe-added first document: {before} chunks before, {store.total} after (should match)")

# Save to a temporary folder, reload, and search again
with tempfile.TemporaryDirectory() as tmp:
    store.save(Path(tmp))
    reloaded = VectorStore.load(Path(tmp))
    print(f"Saved and reloaded: {reloaded.total} chunks (original: {store.total})")
    show(reloaded, questions[0], k=1)

# Remove a document
name = next(iter(store.list_documents()))
print(f"\nRemoved {store.remove_document(name)} chunk(s) of {name}. Remaining:", store.list_documents())