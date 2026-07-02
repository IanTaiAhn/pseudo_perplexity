from api.schemas import Chunk
from agents.web_search_agent import _score_chunks


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source="https://example.com",
        source_type="web",
        chunk_index=0,
    )


def test_score_chunks_returns_empty_for_no_chunks():
    assert _score_chunks("query", []) == []


def test_score_chunks_assigns_cosine_similarity_and_sorts_descending(monkeypatch):
    chunks = [_chunk("a", "low match"), _chunk("b", "high match")]

    def fake_embed_query(query: str) -> list[float]:
        return [1.0, 0.0]

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        # "low match" is nearly orthogonal to the query; "high match" is aligned.
        return [[0.1, 0.99], [0.9, 0.1]]

    monkeypatch.setattr("agents.web_search_agent.embed_query", fake_embed_query)
    monkeypatch.setattr("agents.web_search_agent.embed_texts", fake_embed_texts)

    scored = _score_chunks("query", chunks)

    assert scored[0].chunk_id == "b"
    assert scored[1].chunk_id == "a"
    assert scored[0].score > scored[1].score
