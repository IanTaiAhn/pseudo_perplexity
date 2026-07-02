import pytest
from api.schemas import Chunk, QueryRequest
from agents import orchestrator


def _chunk(chunk_id: str, source_type: str, score: float) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="text",
        source="source",
        source_type=source_type,
        chunk_index=0,
        score=score,
    )


@pytest.mark.asyncio
async def test_retrieve_merges_and_sorts_doc_and_web_results_by_score(monkeypatch):
    doc_chunks = [_chunk("doc-low", "document", 0.2), _chunk("doc-high", "document", 0.9)]
    web_chunks = [_chunk("web-mid", "web", 0.5)]

    async def fake_retrieve_from_documents(query, top_k):
        return doc_chunks

    async def fake_search(query, top_k):
        return web_chunks

    monkeypatch.setattr(orchestrator, "retrieve_from_documents", fake_retrieve_from_documents)
    monkeypatch.setattr(orchestrator, "search", fake_search)

    request = QueryRequest(query="test", top_k=5)
    results = await orchestrator.retrieve(request)

    assert [c.chunk_id for c in results] == ["doc-high", "web-mid", "doc-low"]


@pytest.mark.asyncio
async def test_retrieve_skips_web_search_when_disabled(monkeypatch):
    doc_chunks = [_chunk("doc-1", "document", 0.7)]

    async def fake_retrieve_from_documents(query, top_k):
        return doc_chunks

    async def fake_search(query, top_k):
        raise AssertionError("web search should not be called")

    monkeypatch.setattr(orchestrator, "retrieve_from_documents", fake_retrieve_from_documents)
    monkeypatch.setattr(orchestrator, "search", fake_search)

    request = QueryRequest(query="test", top_k=5, use_web_search=False)
    results = await orchestrator.retrieve(request)

    assert [c.chunk_id for c in results] == ["doc-1"]
