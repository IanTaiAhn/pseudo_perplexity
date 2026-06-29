# Architecture decisions

> Record every non-obvious decision here as you build. This becomes your interview answer to "walk me through how you built this."

| Date | Decision | Alternatives considered | Reason |
|---|---|---|---|
| 2026-06-29 | Fixed-size chunking (512 tokens, 50 overlap) | Semantic chunking, recursive | Simplest to implement and debug; spec says measure before switching |
| 2026-06-29 | BAAI/bge-small-en-v1.5 for embeddings | OpenAI embeddings, all-MiniLM-L6-v2 | Free, fast, strong MTEB scores; no API cost during dev |
| 2026-06-29 | ChromaDB with cosine distance | FAISS, pgvector, Qdrant | Zero-config local dev; spec mandates Qdrant for prod migration |
| 2026-06-29 | Claude claude-sonnet-4-6 as LLM | GPT-4o | Spec-mandated primary; OpenAI GPT-4o kept as fallback |
| 2026-06-29 | filter_cited_chunks over extract_citations in response | return all retrieved chunks as citations | Only returning chunks the LLM actually cited reduces noise and false citations |
| 2026-06-29 | Confidence gate at reranker score < 0.3 | Always call LLM | Avoids hallucination when retrieval finds nothing relevant; per spec section 10 |
