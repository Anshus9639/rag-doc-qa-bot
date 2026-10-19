from src.chunking import Chunk
from src.llm import LLMError, stream_answer
from src.prompts import build_messages


def chunk(i, text, page, section):
    return Chunk(text, "contract.pdf", page, f"contract.pdf#{i}", section)


HITS = [
    (chunk(0, "Beta LLC shall pay Acme Ltd within 30 days of each invoice.", 1, "1. Payment"), 0.5),
    (chunk(1, "Either party may terminate this agreement with 60 days written notice. "
              "Termination for breach is immediate if the breach is not cured within 14 days.",
           2, "2. Termination"), 0.6),
]

QUESTIONS = [
    "How much notice is needed to terminate?",     # answerable from the excerpts
    "What is the penalty for late payment?",       # related, but NOT in the excerpts
    "What is the capital of France?",              # unrelated
]

try:
    for q in QUESTIONS:
        print(f"\nQ: {q}\nA: ", end="", flush=True)
        for piece in stream_answer(build_messages(q, HITS)):
            print(piece, end="", flush=True)
        print()
except LLMError as error:
    print(f"\nERROR: {error}")