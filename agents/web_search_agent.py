import os
import httpx
from api.schemas import Chunk
from ingestion.chunker import chunk_text


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


async def search(query: str, top_k: int = _DEFAULT_TOP_K) -> list[Chunk]:
    """Search Tavily and return results as Chunks ready for reranking."""
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
    return _tavily_results_to_chunks(results)
