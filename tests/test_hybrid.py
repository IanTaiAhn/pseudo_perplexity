import pytest
from unittest.mock import patch
from api.schemas import Chunk
from retrieval.hybrid import retrieve, _min_max_normalize


def _chunk(chunk_id: str, score: float) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="t",
        source="s",
        source_type="document",
        chunk_index=0,
        score=score,
    )


def test_min_max_normalize_scales_to_zero_one():
    normalized = _min_max_normalize({"a": 10.0, "b": 5.0, "c": 0.0})
    assert normalized["a"] == 1.0
    assert normalized["c"] == 0.0
    assert normalized["b"] == 0.5


def test_min_max_normalize_handles_tied_scores_without_division_by_zero():
    normalized = _min_max_normalize({"a": 3.0, "b": 3.0})
    assert normalized == {"a": 0.0, "b": 0.0}


def test_min_max_normalize_handles_empty():
    assert _min_max_normalize({}) == {}


def test_retrieve_combines_dense_and_bm25_with_equal_weight():
    dense_chunks = [_chunk("shared", 1.0), _chunk("dense-only", 0.5)]
    sparse_chunks = [_chunk("shared", 10.0), _chunk("bm25-only", 2.0)]

    with patch("retrieval.hybrid.dense.retrieve", return_value=dense_chunks), \
         patch("retrieval.hybrid.sparse.retrieve", return_value=sparse_chunks):
        results = retrieve("test query", top_k=10)

    by_id = {c.chunk_id: c for c in results}

    # "shared" is top of both lists -> normalized 1.0 on both axes -> hybrid 1.0
    assert by_id["shared"].hybrid_score == pytest.approx(1.0)
    assert by_id["shared"].dense_score == 1.0
    assert by_id["shared"].bm25_score == 10.0

    # "dense-only" never appears in the BM25 candidate list, so its bm25_score
    # is genuinely unknown (not zero) — that distinction matters for debugging.
    assert by_id["dense-only"].bm25_score is None
    assert by_id["dense-only"].dense_score == 0.5

    # deduplicated: "shared" only appears once even though both retrievers found it
    assert len(results) == 3

    # final ranking is by hybrid_score, descending
    assert results[0].chunk_id == "shared"


def test_retrieve_respects_top_k():
    dense_chunks = [_chunk(f"d{i}", float(10 - i)) for i in range(5)]

    with patch("retrieval.hybrid.dense.retrieve", return_value=dense_chunks), \
         patch("retrieval.hybrid.sparse.retrieve", return_value=[]):
        results = retrieve("test query", top_k=2)

    assert len(results) == 2
