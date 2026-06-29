from api.schemas import Chunk
from ingestion.embedder import embed_query
from ingestion.indexer import get_collection


def retrieve(query: str, top_k: int = 10) -> list[Chunk]:
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[Chunk] = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        # ChromaDB cosine distance: score = 1 - distance (higher = more similar)
        score = 1.0 - distance

        chunks.append(Chunk(
            chunk_id=doc_id,
            text=results["documents"][0][i],
            source=meta["source"],
            source_type=meta["source_type"],
            chunk_index=meta["chunk_index"],
            score=score,
        ))

    return chunks
