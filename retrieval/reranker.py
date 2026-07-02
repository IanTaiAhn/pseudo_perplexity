# Cross-encoder reranking
import os
from api.schemas import Chunk

_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    if not chunks:
        return []

    model = _get_model()
    # Cross-encoder: query and chunk text go in together as one input pair
    # per candidate, unlike dense/BM25 which score query and chunk independently.
    pairs = [[query, chunk.text] for chunk in chunks]
    scores = model.predict(pairs)

    reranked = [
        chunk.model_copy(update={"rerank_score": float(score), "score": float(score)})
        for chunk, score in zip(chunks, scores)
    ]
    reranked.sort(key=lambda c: c.score if c.score is not None else float("-inf"), reverse=True)
    return reranked[:top_k]
