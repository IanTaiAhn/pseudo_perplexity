import asyncio
from api.schemas import Chunk, QueryRequest
from agents.retrieval_agent import retrieve_from_documents
from agents.web_search_agent import search


async def retrieve(request: QueryRequest) -> list[Chunk]:
    """
    Run document and web retrieval in parallel, merge results.
    Each path is optional based on request flags and data availability.
    """
    tasks = []

    if request.use_documents:
        tasks.append(retrieve_from_documents(request.query, top_k=request.top_k * 2))
    else:
        tasks.append(_empty())

    if request.use_web_search:
        tasks.append(search(request.query, top_k=request.top_k))
    else:
        tasks.append(_empty())

    doc_results, web_results = await asyncio.gather(*tasks)

    return doc_results + web_results


async def _empty() -> list[Chunk]:
    return []
