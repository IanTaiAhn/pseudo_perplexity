from api.schemas import Chunk
from retrieval.hybrid import retrieve as hybrid_retrieve


async def retrieve_from_documents(query: str, top_k: int = 10) -> list[Chunk]:
    """Async wrapper around hybrid (dense + BM25) retrieval for use in asyncio.gather."""
    return hybrid_retrieve(query, top_k=top_k)
