import pytest
from api.schemas import Chunk
from synthesis.citation_tracker import (
    build_context_block,
    extract_citations,
    filter_cited_chunks,
)


def _chunk(chunk_id: str, text: str, source: str = "doc.txt", score: float = 0.9) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        source_type="document",
        chunk_index=0,
        score=score,
    )


def test_build_context_block_numbers_chunks():
    chunks = [_chunk("1", "First chunk."), _chunk("2", "Second chunk.")]
    block = build_context_block(chunks)
    assert "[1]" in block
    assert "[2]" in block
    assert "First chunk." in block
    assert "Second chunk." in block


def test_build_context_block_includes_source():
    chunks = [_chunk("1", "text", source="myfile.pdf")]
    block = build_context_block(chunks)
    assert "myfile.pdf" in block


def test_extract_citations_returns_all_chunks():
    chunks = [_chunk("a", "alpha"), _chunk("b", "beta")]
    citations = extract_citations(chunks)
    assert len(citations) == 2
    assert citations[0].citation_number == 1
    assert citations[1].citation_number == 2


def test_filter_cited_chunks_only_returns_referenced():
    chunks = [_chunk("1", "alpha"), _chunk("2", "beta"), _chunk("3", "gamma")]
    answer = "According to [1] and [3], the answer is clear."
    citations = filter_cited_chunks(answer, chunks)
    numbers = {c.citation_number for c in citations}
    assert numbers == {1, 3}
    assert 2 not in numbers


def test_filter_cited_chunks_empty_when_no_markers():
    chunks = [_chunk("1", "alpha"), _chunk("2", "beta")]
    answer = "The answer has no citation markers."
    citations = filter_cited_chunks(answer, chunks)
    assert citations == []
