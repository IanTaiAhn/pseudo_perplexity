import os
import time
import uuid

import anthropic
from api.schemas import Chunk, QueryResponse
from monitoring.logger import get_logger
from monitoring.metrics import llm_cost_usd_total, llm_latency_seconds, llm_tokens_total
from synthesis.citation_tracker import build_context_block, filter_cited_chunks, extract_citations

logger = get_logger(__name__)
_SLOW_LLM_CALL_MS = 5000

# Langfuse is optional: without keys set, tracing is a no-op rather than a
# hard dependency, so local dev (LOCAL_LLM_MODEL, no API keys at all) still
# works without signing up for anything.
_LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
_LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

_langfuse_client = None
if _LANGFUSE_PUBLIC_KEY and _LANGFUSE_SECRET_KEY:
    from langfuse import Langfuse
    _langfuse_client = Langfuse(
        public_key=_LANGFUSE_PUBLIC_KEY,
        secret_key=_LANGFUSE_SECRET_KEY,
        host=_LANGFUSE_HOST,
    )

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


def _call_llm(user_prompt: str) -> tuple[str, float, int, int, str]:
    """Returns (answer, estimated_cost_usd, input_tokens, output_tokens, model_used)."""
    if _LOCAL_LLM_MODEL:
        answer, estimated_cost = _generate_local(user_prompt)
        return answer, estimated_cost, 0, 0, _LOCAL_LLM_MODEL
    answer, estimated_cost, input_tokens, output_tokens = _generate_claude(user_prompt)
    return answer, estimated_cost, input_tokens, output_tokens, _LLM_MODEL


def generate(query: str, chunks: list[Chunk], query_id: str | None = None) -> QueryResponse:
    if query_id is None:
        query_id = str(uuid.uuid4())

    context_block = build_context_block(chunks)
    user_prompt = _build_user_prompt(query, context_block)
    model_used = _LOCAL_LLM_MODEL if _LOCAL_LLM_MODEL else _LLM_MODEL

    start = time.time()

    if _langfuse_client is not None:
        # Derive the Langfuse trace_id deterministically from query_id (via
        # create_trace_id's seed hashing) so this generation lands under a
        # trace_id that can be recomputed from the query_id in our structured
        # logs — no need to persist the Langfuse-generated ID anywhere.
        # Wrapping the actual call (not creating the span after the fact)
        # keeps the span's start/end timestamps accurate in the Langfuse UI.
        trace_id = _langfuse_client.create_trace_id(seed=query_id)
        with _langfuse_client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name="synthesis",
            as_type="generation",
            model=model_used,
            input=user_prompt,
        ) as generation:
            answer, estimated_cost, input_tokens, output_tokens, model_used = _call_llm(user_prompt)
            generation.update(
                output=answer,
                usage_details={"input": input_tokens, "output": output_tokens},
                cost_details={"total": estimated_cost},
                metadata={"retrieval_sources": [c.chunk_id for c in chunks]},
            )
    else:
        answer, estimated_cost, input_tokens, output_tokens, model_used = _call_llm(user_prompt)

    latency_ms = (time.time() - start) * 1000

    llm_latency_seconds.observe(latency_ms / 1000)
    llm_tokens_total.labels(direction="input").inc(input_tokens)
    llm_tokens_total.labels(direction="output").inc(output_tokens)
    llm_cost_usd_total.inc(estimated_cost)

    log_event = logger.warning if latency_ms > _SLOW_LLM_CALL_MS else logger.info
    log_event(
        "llm_call",
        query_id=query_id,
        model=model_used,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency_ms, 2),
        estimated_cost_usd=round(estimated_cost, 6),
        retrieval_sources=[c.chunk_id for c in chunks],
        faithfulness_score=None,
    )

    citations = filter_cited_chunks(answer, chunks)
    if not citations:
        citations = extract_citations(chunks)  # already imported in citation_tracker

    return QueryResponse(
        answer=answer,
        citations=citations,
        query_id=query_id,
        latency_ms=round(latency_ms, 2),
        model_used=model_used,
        estimated_cost_usd=round(estimated_cost, 6),
    )
