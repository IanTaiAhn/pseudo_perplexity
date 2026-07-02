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


def _passthrough_rerank(query, chunks, top_k=5):
    """Stand-in for the real cross-encoder reranker so orchestrator tests
    don't need to load a model — just enforces the top_k cutoff and
    preserves whatever order the caller passed in."""
    return chunks[:top_k]


@pytest.mark.asyncio
async def test_retrieve_merges_doc_and_web_results_and_hands_them_to_reranker(monkeypatch):
    doc_chunks = [_chunk("doc-low", "document", 0.2), _chunk("doc-high", "document", 0.9)]
    web_chunks = [_chunk("web-mid", "web", 0.5)]

    async def fake_retrieve_from_documents(query, top_k):
        return doc_chunks

    async def fake_search(query, top_k):
        return web_chunks

    received = {}

    def fake_rerank(query, chunks, top_k=5):
        received["query"] = query
        received["chunk_ids"] = [c.chunk_id for c in chunks]
        return chunks[:top_k]

    monkeypatch.setattr(orchestrator, "retrieve_from_documents", fake_retrieve_from_documents)
    monkeypatch.setattr(orchestrator, "search", fake_search)
    monkeypatch.setattr(orchestrator, "rerank", fake_rerank)

    request = QueryRequest(query="test", top_k=5)
    results = await orchestrator.retrieve(request)

    # The reranker — not a raw score sort — decides final ranking, so the
    # orchestrator's job is just to hand it the full merged candidate pool.
    assert received["query"] == "test"
    assert set(received["chunk_ids"]) == {"doc-low", "doc-high", "web-mid"}
    assert [c.chunk_id for c in results] == ["doc-low", "doc-high", "web-mid"]


@pytest.mark.asyncio
async def test_retrieve_skips_web_search_when_disabled(monkeypatch):
    doc_chunks = [_chunk("doc-1", "document", 0.7)]

    async def fake_retrieve_from_documents(query, top_k):
        return doc_chunks

    async def fake_search(query, top_k):
        raise AssertionError("web search should not be called")

    monkeypatch.setattr(orchestrator, "retrieve_from_documents", fake_retrieve_from_documents)
    monkeypatch.setattr(orchestrator, "search", fake_search)
    monkeypatch.setattr(orchestrator, "rerank", _passthrough_rerank)

    request = QueryRequest(query="test", top_k=5, use_web_search=False)
    results = await orchestrator.retrieve(request)

    assert [c.chunk_id for c in results] == ["doc-1"]


@pytest.mark.asyncio
async def test_retrieve_respects_top_k_after_reranking(monkeypatch):
    doc_chunks = [_chunk(f"doc-{i}", "document", 1.0 - i * 0.1) for i in range(5)]

    async def fake_retrieve_from_documents(query, top_k):
        return doc_chunks

    async def fake_search(query, top_k):
        return []

    monkeypatch.setattr(orchestrator, "retrieve_from_documents", fake_retrieve_from_documents)
    monkeypatch.setattr(orchestrator, "search", fake_search)
    monkeypatch.setattr(orchestrator, "rerank", _passthrough_rerank)

    request = QueryRequest(query="test", top_k=2, use_web_search=False)
    results = await orchestrator.retrieve(request)

    assert len(results) == 2
