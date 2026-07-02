import os
import httpx
from api.schemas import Chunk
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_query, embed_texts


_TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
_TAVILY_URL = "https://api.tavily.com/search"
_DEFAULT_TOP_K = 10


def _tavily_results_to_chunks(results: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    for result in results:
        content = result.get("content", "").strip()
        url = result.get("url", "unknown")
        if not content:
            continue
        for chunk in chunk_text(content, source=url, source_type="web"):
            chunks.append(chunk.model_copy(update={"chunk_index": chunk_index}))
            chunk_index += 1
    return chunks


def _score_chunks(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Score web chunks by cosine similarity to the query, using the same
    embedding model as document retrieval so scores are on a comparable scale."""
    if not chunks:
        return []

    query_embedding = embed_query(query)
    chunk_embeddings = embed_texts([c.text for c in chunks])

    scored = []
    for chunk, embedding in zip(chunks, chunk_embeddings):
        # Both embeddings are normalized, so the dot product equals cosine similarity
        # (the same score dense.py derives from ChromaDB's cosine distance).
        score = sum(a * b for a, b in zip(query_embedding, embedding))
        scored.append(chunk.model_copy(update={"score": score, "dense_score": score}))

    return sorted(scored, key=lambda c: c.score, reverse=True)


async def search(query: str, top_k: int = _DEFAULT_TOP_K) -> list[Chunk]:
    """Search Tavily and return results as scored Chunks ready for reranking."""
    if not _TAVILY_API_KEY:
        return []

    payload = {
        "api_key": _TAVILY_API_KEY,
        "query": query,
        "max_results": top_k,
        "include_raw_content": False,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(_TAVILY_URL, json=payload)
        response.raise_for_status()

    data = response.json()
    results = data.get("results", [])
    chunks = _tavily_results_to_chunks(results)
    scored_chunks = _score_chunks(query, chunks)
    return scored_chunks[:top_k]
