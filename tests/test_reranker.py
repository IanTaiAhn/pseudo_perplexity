import pytest
from unittest.mock import MagicMock, patch
from api.schemas import Chunk
from retrieval.reranker import rerank


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source="s", source_type="document", chunk_index=0)


def test_rerank_returns_empty_for_no_chunks():
    assert rerank("query", []) == []


def test_rerank_scores_query_chunk_pairs_and_sorts_descending():
    chunks = [_chunk("a", "irrelevant text"), _chunk("b", "highly relevant text")]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]

    with patch("retrieval.reranker._get_model", return_value=mock_model):
        results = rerank("query", chunks, top_k=5)

    # Cross-encoder scores (query, chunk) jointly, unlike dense/BM25 which
    # score each independently — verify it's fed pairs, not separate texts.
    mock_model.predict.assert_called_once_with(
        [["query", "irrelevant text"], ["query", "highly relevant text"]]
    )
    assert [c.chunk_id for c in results] == ["b", "a"]
    assert results[0].rerank_score == pytest.approx(0.9)
    assert results[0].score == pytest.approx(0.9)


def test_rerank_respects_top_k():
    chunks = [_chunk(str(i), f"text {i}") for i in range(5)]

    mock_model = MagicMock()
    mock_model.predict.return_value = [float(i) for i in range(5)]

    with patch("retrieval.reranker._get_model", return_value=mock_model):
        results = rerank("query", chunks, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "4"
