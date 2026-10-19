import sys

from src.embeddings import EmbeddingError
from src.llm import LLMError
from src.loaders import DocumentLoadError
from src.pipeline import Answer, RAGPipeline
from src.retriever import RetrievalError
from src.vector_store import VectorStoreError

HELP = "Commands: /add <path>   /docs   /remove <file name>   /help   /quit"


def add_file(pipeline: RAGPipeline, path: str) -> None:
    try:
        count = pipeline.ingest_file(path.strip().strip('"'))
        print(f"Indexed {count} chunk(s).")
    except (DocumentLoadError, EmbeddingError, VectorStoreError) as error:
        print(f"ERROR: {error}")


def print_answer(answer: Answer) -> None:
    print(f"\n{answer.text}")
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            where = f"page {c.page}" if c.page else "no page info"
            section = f", {c.section}" if c.section else ""
            print(f"  [{c.number}] {c.source}, {where}{section} (score {c.score:.2f})")
            print(f"      {c.snippet}...")
    elif answer.note:
        print(f"\n(refused because: {answer.note})")


def main() -> None:
    try:
        pipeline = RAGPipeline()
    except VectorStoreError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    for path in sys.argv[1:]:
        add_file(pipeline, path)

    print(f"\nDocuments: {pipeline.documents() or 'none yet'}")
    print(HELP)

    while True:
        line = input("\nYou> ").strip()
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/help":
            print(HELP)
        elif line == "/docs":
            print(pipeline.documents() or "none yet")
        elif line.startswith("/add "):
            add_file(pipeline, line[5:])
        elif line.startswith("/remove "):
            name = line[8:].strip()
            print(f"Removed {pipeline.remove_document(name)} chunk(s) of {name}.")
        else:
            try:
                print_answer(pipeline.ask(line))
            except (RetrievalError, EmbeddingError, LLMError) as error:
                print(f"ERROR: {error}")


if __name__ == "__main__":
    main()