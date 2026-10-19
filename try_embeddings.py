from src.embeddings import embed_query, embed_texts

chunks = [
    "Either party may terminate this agreement with 60 days written notice.",
    "Beta LLC shall pay Acme Ltd within 30 days of each invoice.",
    "Both parties must keep all shared information confidential.",
    "The best pasta sauce uses fresh tomatoes and basil.",
]
questions = [
    "When can the agreement be ended?",
    "How long do we have to pay?",
    "What is the recipe for tomato sauce?",
]

chunk_vectors = embed_texts(chunks)
print(f"Chunk vectors shape: {chunk_vectors.shape}  (chunks x dimensions)")
print(f"Vector length (should be ~1.0): {(chunk_vectors[0] ** 2).sum() ** 0.5:.3f}\n")

for q in questions:
    scores = (embed_query(q) @ chunk_vectors.T)[0]   # dot product == cosine similarity
    best = int(scores.argmax())
    print(f"Q: {q}")
    for i, s in enumerate(scores):
        marker = "  <-- best" if i == best else ""
        print(f"   {s:+.3f}  {chunks[i][:55]}{marker}")
    print()