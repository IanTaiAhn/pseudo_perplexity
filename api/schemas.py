from pydantic import BaseModel
from typing import Optional


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    source_type: str  # "document" or "web"
    chunk_index: int
    score: Optional[float] = None  # authoritative score at the current pipeline stage

    # Per-stage scores, kept alongside `score` for debugging retrieval quality —
    # e.g. distinguishing "BM25 missed this" from "reranker downgraded this".
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_web_search: bool = True
    use_documents: bool = True


class Citation(BaseModel):
    citation_number: int
    source: str
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    query_id: str
    latency_ms: float
    model_used: str
    estimated_cost_usd: float


class IngestRequest(BaseModel):
    source_type: str  # "file" or "url"
    content: Optional[str] = None
    url: Optional[str] = None


class IngestResponse(BaseModel):
    chunks_indexed: int
    source: str
    status: str
