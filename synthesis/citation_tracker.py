import re
from api.schemas import Chunk, Citation


def build_context_block(chunks: list[Chunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {chunk.source}) {chunk.text}")
    return "\n\n".join(lines)


def extract_citations(chunks: list[Chunk]) -> list[Citation]:
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        citations.append(Citation(
            citation_number=i,
            source=chunk.source,
            chunk_text=chunk.text,
            score=chunk.score or 0.0,
        ))
    return citations


def filter_cited_chunks(answer: str, chunks: list[Chunk]) -> list[Citation]:
    """Return only citations that are actually referenced in the answer."""
    cited_numbers: set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        cited_numbers.add(int(match.group(1)))

    citations = []
    for i, chunk in enumerate(chunks, start=1):
        if i in cited_numbers:
            citations.append(Citation(
                citation_number=i,
                source=chunk.source,
                chunk_text=chunk.text,
                score=chunk.score or 0.0,
            ))
    return citations
