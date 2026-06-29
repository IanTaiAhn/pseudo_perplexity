import os
import time
import anthropic
from api.schemas import Chunk, QueryResponse, Citation
from synthesis.citation_tracker import build_context_block, filter_cited_chunks
import uuid

_LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# Pricing as of mid-2026
_INPUT_COST_PER_TOKEN = 3.00 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000

_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the context "
    "provided below. For every factual claim in your answer, add a citation marker "
    "[1], [2], etc. that corresponds to the source chunk number. "
    "If the context does not contain enough information to answer, say so clearly."
)


def _build_user_prompt(query: str, context_block: str) -> str:
    return f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer (with citation markers):"


def generate(query: str, chunks: list[Chunk], query_id: str | None = None) -> QueryResponse:
    if query_id is None:
        query_id = str(uuid.uuid4())

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    context_block = build_context_block(chunks)
    user_prompt = _build_user_prompt(query, context_block)

    start = time.time()
    response = client.messages.create(
        model=_LLM_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency_ms = (time.time() - start) * 1000

    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    estimated_cost = (
        input_tokens * _INPUT_COST_PER_TOKEN
        + output_tokens * _OUTPUT_COST_PER_TOKEN
    )

    citations = filter_cited_chunks(answer, chunks)

    return QueryResponse(
        answer=answer,
        citations=citations,
        query_id=query_id,
        latency_ms=round(latency_ms, 2),
        model_used=_LLM_MODEL,
        estimated_cost_usd=round(estimated_cost, 6),
    )
