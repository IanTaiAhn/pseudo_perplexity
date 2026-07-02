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
| 2026-07-01 | Score web chunks via cosine similarity (query embedding · chunk embedding) instead of an in-memory Chroma collection | Ephemeral Chroma collection per query, as literally described in spec Layer 2 step 4 | Same embedding model + normalized vectors make dot product mathematically identical to Chroma's cosine score, so doc and web scores stay comparable; skips the overhead of standing up a collection per query and is easier to unit test |
| 2026-07-02 | Layer 3: hybrid (dense+BM25) retrieval replaces dense-only inside `retrieval_agent.retrieve_from_documents`; cross-encoder reranker now runs in `orchestrator.retrieve` on the merged doc+web pool, replacing the plain score sort | Rerank only the document path, before merging with web results | Web results and doc results were never on a comparable scale even after hybrid fusion (hybrid_score vs. raw cosine); reranking the merged pool gives one consistent, joint-relevance ranking across both sources instead of two differently-scaled sorts |
| 2026-07-02 | `Chunk.score` stays as the single "authoritative current-stage" field (unchanged consumers: confidence threshold, citations, sorting); added `dense_score`/`bm25_score`/`hybrid_score`/`rerank_score` as parallel debug fields instead of replacing `score` | Rename `score` to something stage-specific everywhere it's used | Additive-only change — no existing call site (`query.py` confidence gate, `citation_tracker.py`, tests) needed to change, while still exposing per-stage scores for retrieval-failure debugging |
