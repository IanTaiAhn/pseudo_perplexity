# Merges dense + sparse results
from api.schemas import Chunk
from retrieval import dense, sparse

_DENSE_WEIGHT = 0.5
_BM25_WEIGHT = 0.5

# How many candidates each retriever contributes before fusion. Wider than
# the final top_k so weak-on-one-signal-but-strong-on-the-other chunks (the
# whole point of hybrid search) have a chance to surface.
_CANDIDATE_K = 20


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    values = list(scores.values())
    lo, hi = min(values), max(values)

    if hi - lo < 1e-9:
        # All candidates tied (or only one candidate) — no signal to
        # differentiate them on this axis, so contribute nothing to the blend
        # rather than dividing by zero.
        return {chunk_id: 0.0 for chunk_id in scores}

    return {chunk_id: (v - lo) / (hi - lo) for chunk_id, v in scores.items()}


def retrieve(query: str, top_k: int = 10, candidate_k: int = _CANDIDATE_K) -> list[Chunk]:
    dense_chunks = dense.retrieve(query, top_k=candidate_k)
    sparse_chunks = sparse.retrieve(query, top_k=candidate_k)

    # Union of both candidate sets, deduplicated by chunk_id — the same
    # chunk can (and often does) appear in both the dense and BM25 top-k.
    chunks_by_id: dict[str, Chunk] = {}
    dense_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}

    for chunk in dense_chunks:
        chunks_by_id[chunk.chunk_id] = chunk
        dense_scores[chunk.chunk_id] = chunk.score or 0.0

    for chunk in sparse_chunks:
        chunks_by_id.setdefault(chunk.chunk_id, chunk)
        bm25_scores[chunk.chunk_id] = chunk.score or 0.0

    norm_dense = _min_max_normalize(dense_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)

    merged: list[Chunk] = []
    for chunk_id, chunk in chunks_by_id.items():
        # A chunk retrieved by only one method (e.g. top-20 dense but not
        # top-20 BM25) gets 0.0 on the missing axis — a candidate-generation
        # approximation, not that method's true score for this chunk.
        d = norm_dense.get(chunk_id, 0.0)
        b = norm_bm25.get(chunk_id, 0.0)
        hybrid_score = _DENSE_WEIGHT * d + _BM25_WEIGHT * b

        merged.append(chunk.model_copy(update={
            "dense_score": dense_scores.get(chunk_id),
            "bm25_score": bm25_scores.get(chunk_id),
            "hybrid_score": hybrid_score,
            "score": hybrid_score,
        }))

    merged.sort(key=lambda c: c.score if c.score is not None else float("-inf"), reverse=True)
    return merged[:top_k]
