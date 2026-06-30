import uuid
from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse, Citation
from agents.orchestrator import retrieve
from synthesis.generator import generate

router = APIRouter()

_LOW_CONFIDENCE_THRESHOLD = 0.3

_LOW_CONFIDENCE_RESPONSE = (
    "I don't have enough information in the available sources to answer this question confidently."
)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    query_id = str(uuid.uuid4())

    chunks = await retrieve(request)

    # Trim to top_k after merging doc + web results
    chunks = chunks[: request.top_k]

    if not chunks or (chunks[0].score is not None and chunks[0].score < _LOW_CONFIDENCE_THRESHOLD):
        return QueryResponse(
            answer=_LOW_CONFIDENCE_RESPONSE,
            citations=[],
            query_id=query_id,
            latency_ms=0.0,
            model_used="",
            estimated_cost_usd=0.0,
        )

    return generate(query=request.query, chunks=chunks, query_id=query_id)
