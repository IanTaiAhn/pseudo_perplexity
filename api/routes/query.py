import time
import uuid

from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse, Citation
from agents.orchestrator import retrieve
from ingestion.embedder import embed_query
from monitoring import drift_detector
from monitoring.logger import bind_query_context, clear_query_context, get_logger
from monitoring.metrics import (
    error_total,
    query_latency_seconds,
    query_total,
    record_retrieval_hit,
)
from synthesis.generator import generate

router = APIRouter()
logger = get_logger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 0.3

_LOW_CONFIDENCE_RESPONSE = (
    "I don't have enough information in the available sources to answer this question confidently."
)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    query_id = str(uuid.uuid4())
    bind_query_context(query_id=query_id)
    start = time.time()
    query_total.inc()

    try:
        logger.info(
            "query_received",
            query=request.query,
            use_documents=request.use_documents,
            use_web_search=request.use_web_search,
        )

        # Logged for drift_detector.check_drift() to compare against the
        # baseline distribution later — not used for retrieval itself.
        drift_detector.log_query_embedding(embed_query(request.query), query=request.query)

        chunks = await retrieve(request)

        # Trim to top_k after merging doc + web results
        chunks = chunks[: request.top_k]

        top_score = chunks[0].score if chunks and chunks[0].score is not None else None
        is_hit = top_score is not None and top_score >= _LOW_CONFIDENCE_THRESHOLD
        record_retrieval_hit(is_hit)

        if not chunks or (top_score is not None and top_score < _LOW_CONFIDENCE_THRESHOLD):
            logger.warning("low_retrieval_confidence", top_score=top_score, num_chunks=len(chunks))
            return QueryResponse(
                answer=_LOW_CONFIDENCE_RESPONSE,
                citations=[],
                query_id=query_id,
                latency_ms=0.0,
                model_used="",
                estimated_cost_usd=0.0,
            )

        response = generate(query=request.query, chunks=chunks, query_id=query_id)
        return response
    except Exception as e:
        error_total.labels(error_type=type(e).__name__).inc()
        logger.error("query_failed", error=str(e), error_type=type(e).__name__)
        raise
    finally:
        query_latency_seconds.observe(time.time() - start)
        clear_query_context()
