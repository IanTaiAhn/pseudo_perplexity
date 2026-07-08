from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Response
from api.routes import ingest, query
from monitoring.logger import configure_logging
from monitoring.metrics import metrics_response

configure_logging()

app = FastAPI(
    title="Pseudo-Perplexity",
    description="AI research assistant with grounded answers and inline citations.",
    version="0.1.0",
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


@app.get("/debug/collection")
async def debug_collection() -> dict:
    """Shows how many chunks are in ChromaDB and a sample of them."""
    from ingestion.indexer import get_collection
    collection = get_collection()
    count = collection.count()
    sample = collection.peek(limit=3)  # first 3 chunks
    return {
        "chunks_in_db": count,
        "sample_ids": sample["ids"],
        "sample_sources": [m["source"] for m in sample["metadatas"]],
        "sample_text_preview": [d[:200] for d in sample["documents"]],
    }


@app.get("/debug/query")
async def debug_query(q: str, top_k: int = 5) -> dict:
    """Shows raw retrieval scores for a query — use this to diagnose the confidence gate."""
    from ingestion.embedder import embed_query
    from ingestion.indexer import get_collection
    collection = get_collection()
    if collection.count() == 0:
        return {"error": "No chunks in DB. Did you ingest a document?"}
    query_embedding = embed_query(q)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        hits.append({
            "chunk_id": doc_id,
            "score": round(1.0 - distance, 4),
            "source": results["metadatas"][0][i]["source"],
            "text_preview": results["documents"][0][i][:200],
        })
    return {"query": q, "hits": hits}
