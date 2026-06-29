import pytest
from unittest.mock import patch, MagicMock
from api.schemas import Chunk
from retrieval.dense import retrieve


def _make_chunk(chunk_id: str, text: str, score: float) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source="test.txt",
        source_type="document",
        chunk_index=0,
        score=score,
    )


def test_retrieve_returns_empty_when_collection_empty():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch("retrieval.dense.get_collection", return_value=mock_collection):
        results = retrieve("what is RAG?")

    assert results == []


def test_retrieve_returns_chunks_sorted_by_score():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        "ids": [["id1", "id2", "id3"]],
        "documents": [["doc1", "doc2", "doc3"]],
        "metadatas": [[
            {"source": "a.txt", "source_type": "document", "chunk_index": 0},
            {"source": "b.txt", "source_type": "document", "chunk_index": 1},
            {"source": "c.txt", "source_type": "document", "chunk_index": 2},
        ]],
        "distances": [[0.1, 0.3, 0.5]],
    }

    with patch("retrieval.dense.get_collection", return_value=mock_collection):
        with patch("retrieval.dense.embed_query", return_value=[0.1] * 384):
            results = retrieve("test query", top_k=3)

    assert len(results) == 3
    # Lower distance = higher score; first result should have highest score
    assert results[0].score > results[1].score > results[2].score


def test_retrieve_respects_top_k():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[
            {"source": "a.txt", "source_type": "document", "chunk_index": 0},
            {"source": "b.txt", "source_type": "document", "chunk_index": 1},
        ]],
        "distances": [[0.2, 0.4]],
    }

    with patch("retrieval.dense.get_collection", return_value=mock_collection):
        with patch("retrieval.dense.embed_query", return_value=[0.1] * 384):
            results = retrieve("test query", top_k=2)

    assert len(results) == 2
