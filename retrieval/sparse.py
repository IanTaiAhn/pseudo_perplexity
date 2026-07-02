# BM25 keyword search
import re
from rank_bm25 import BM25Okapi
from api.schemas import Chunk
from ingestion.indexer import get_collection


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def retrieve(query: str, top_k: int = 10) -> list[Chunk]:
    collection = get_collection()

    if collection.count() == 0:
        return []

    corpus = collection.get(include=["documents", "metadatas"])
    documents = corpus["documents"]
    metadatas = corpus["metadatas"]
    ids = corpus["ids"]

    # BM25Okapi needs the whole corpus tokenized to compute term/document
    # frequency statistics — there's no persistent BM25 index like ChromaDB
    # gives us for dense vectors, so this gets rebuilt on every call.
    bm25 = BM25Okapi([_tokenize(doc) for doc in documents])
    scores = bm25.get_scores(_tokenize(query))

    ranked_indices = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    ranked_indices = ranked_indices[:top_k]

    chunks: list[Chunk] = []
    for i in ranked_indices:
        meta = metadatas[i]
        bm25_score = float(scores[i])
        chunks.append(Chunk(
            chunk_id=ids[i],
            text=documents[i],
            source=meta["source"],
            source_type=meta["source_type"],
            chunk_index=meta["chunk_index"],
            score=bm25_score,
            bm25_score=bm25_score,
        ))

    return chunks
