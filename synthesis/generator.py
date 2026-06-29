import os
import time
import uuid
import anthropic
from api.schemas import Chunk, QueryResponse
from synthesis.citation_tracker import build_context_block, filter_cited_chunks

_LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
# When set, use a local HuggingFace model instead of Claude.
# Example: LOCAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
_LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "")

# Pricing as of mid-2026 (Claude claude-sonnet-4-6)
_INPUT_COST_PER_TOKEN = 3.00 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000

_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the context "
    "provided below. For every factual claim in your answer, add a citation marker "
    "[1], [2], etc. that corresponds to the source chunk number. "
    "If the context does not contain enough information to answer, say so clearly."
)

# Lazy-loaded local pipeline (only initialised when LOCAL_LLM_MODEL is set)
_local_pipeline = None


def _get_local_pipeline():
    global _local_pipeline
    if _local_pipeline is None:
        from transformers import pipeline
        _local_pipeline = pipeline(
            "text-generation",
            model=_LOCAL_LLM_MODEL,
            # keep GPU memory optional — falls back to CPU if no CUDA
            device_map="auto",
        )
    return _local_pipeline


def _build_user_prompt(query: str, context_block: str) -> str:
    return f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer (with citation markers):"


def _generate_local(user_prompt: str) -> tuple[str, float]:
    """Call the local HuggingFace model. Returns (answer_text, estimated_cost_usd)."""
    pipe = _get_local_pipeline()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    output = pipe(messages, max_new_tokens=1024, do_sample=False)
    answer = output[0]["generated_text"][-1]["content"]
    return answer, 0.0  # local inference has no API cost


def _generate_claude(user_prompt: str) -> tuple[str, float, int, int]:
    """Call Claude. Returns (answer_text, estimated_cost_usd, input_tokens, output_tokens)."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=_LLM_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = input_tokens * _INPUT_COST_PER_TOKEN + output_tokens * _OUTPUT_COST_PER_TOKEN
    return answer, cost, input_tokens, output_tokens


def generate(query: str, chunks: list[Chunk], query_id: str | None = None) -> QueryResponse:
    if query_id is None:
        query_id = str(uuid.uuid4())

    context_block = build_context_block(chunks)
    user_prompt = _build_user_prompt(query, context_block)

    start = time.time()

    if _LOCAL_LLM_MODEL:
        answer, estimated_cost = _generate_local(user_prompt)
        model_used = _LOCAL_LLM_MODEL
    else:
        answer, estimated_cost, _, _ = _generate_claude(user_prompt)
        model_used = _LLM_MODEL

    latency_ms = (time.time() - start) * 1000
    citations = filter_cited_chunks(answer, chunks)

    return QueryResponse(
        answer=answer,
        citations=citations,
        query_id=query_id,
        latency_ms=round(latency_ms, 2),
        model_used=model_used,
        estimated_cost_usd=round(estimated_cost, 6),
    )
