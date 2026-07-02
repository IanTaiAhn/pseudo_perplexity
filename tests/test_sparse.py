from unittest.mock import patch, MagicMock
from retrieval.sparse import retrieve, _tokenize


def test_tokenize_lowercases_and_splits_on_non_word_chars():
    assert _tokenize("Hello, World! RAG-2024") == ["hello", "world", "rag", "2024"]


def test_retrieve_returns_empty_when_collection_empty():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch("retrieval.sparse.get_collection", return_value=mock_collection):
        results = retrieve("what is RAG?")

    assert results == []


def test_retrieve_ranks_exact_keyword_match_highest():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.get.return_value = {
        "ids": ["id1", "id2", "id3"],
        "documents": [
            "the quick brown fox jumps over the lazy dog",
            "completely unrelated text about cooking recipes",
            "another unrelated passage about gardening tips",
        ],
        "metadatas": [
            {"source": "a.txt", "source_type": "document", "chunk_index": 0},
            {"source": "b.txt", "source_type": "document", "chunk_index": 1},
            {"source": "c.txt", "source_type": "document", "chunk_index": 2},
        ],
    }

    with patch("retrieval.sparse.get_collection", return_value=mock_collection):
        results = retrieve("quick brown fox", top_k=3)

    # Exact term overlap should dominate — BM25 has no notion of "fox" being
    # semantically close to "cooking," it only counts literal matches.
    assert results[0].chunk_id == "id1"
    assert results[0].bm25_score > 0
    assert results[0].score == results[0].bm25_score


def test_retrieve_respects_top_k():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.get.return_value = {
        "ids": ["id1", "id2", "id3"],
        "documents": ["fox fox fox", "dog dog", "cat"],
        "metadatas": [
            {"source": "a.txt", "source_type": "document", "chunk_index": 0},
            {"source": "b.txt", "source_type": "document", "chunk_index": 1},
            {"source": "c.txt", "source_type": "document", "chunk_index": 2},
        ],
    }

    with patch("retrieval.sparse.get_collection", return_value=mock_collection):
        results = retrieve("fox", top_k=1)

    assert len(results) == 1
