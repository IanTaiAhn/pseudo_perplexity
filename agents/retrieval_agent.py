from api.schemas import Chunk
from retrieval.dense import retrieve


async def retrieve_from_documents(query: str, top_k: int = 10) -> list[Chunk]:
    """Async wrapper around dense retrieval for use in asyncio.gather."""
    return retrieve(query, top_k=top_k)
