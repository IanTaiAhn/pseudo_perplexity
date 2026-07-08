import asyncio
import time

from api.schemas import Chunk, QueryRequest
from agents.retrieval_agent import retrieve_from_documents
from agents.web_search_agent import search
from monitoring.logger import get_logger
from monitoring.metrics import retrieval_latency_seconds
from retrieval.reranker import rerank

logger = get_logger(__name__)


async def retrieve(request: QueryRequest) -> list[Chunk]:
    """
    Run document and web retrieval in parallel, merge, then rerank with the
    cross-encoder so the final ranking reflects joint query+chunk relevance
    rather than each source's own (differently-scaled) similarity score.
    Each path is optional based on request flags and data availability.
    """
    start = time.time()

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

    merged = doc_results + web_results
    reranked = rerank(request.query, merged, top_k=request.top_k)

    retrieval_latency_seconds.observe(time.time() - start)
    logger.info(
        "retrieval_completed",
        doc_chunks=len(doc_results),
        web_chunks=len(web_results),
        returned_chunks=len(reranked),
        top_score=reranked[0].score if reranked else None,
    )

    return reranked


async def _empty() -> list[Chunk]:
    return []
