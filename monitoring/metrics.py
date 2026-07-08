# Prometheus metric definitions
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

query_total = Counter(
    "query_total",
    "Total queries received",
)

query_latency_seconds = Histogram(
    "query_latency_seconds",
    "End-to-end query latency",
)

retrieval_latency_seconds = Histogram(
    "retrieval_latency_seconds",
    "Retrieval step latency (doc + web + rerank)",
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM call latency",
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Tokens used (input + output)",
    ["direction"],  # "input" or "output"
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Estimated cost in USD",
)

retrieval_hit_rate = Gauge(
    "retrieval_hit_rate",
    "Rolling % of queries where retrieval found relevant chunks (top score >= confidence threshold)",
)

error_total = Counter(
    "error_total",
    "Total errors by type",
    ["error_type"],
)

# Rolling window backing retrieval_hit_rate — a Gauge only stores the latest
# value, so we keep a small in-process window to compute "rolling %" from.
_HIT_WINDOW_SIZE = 100
_hit_window: list[bool] = []


def record_retrieval_hit(hit: bool) -> None:
    """Update the rolling retrieval_hit_rate gauge with one query's outcome."""
    _hit_window.append(hit)
    if len(_hit_window) > _HIT_WINDOW_SIZE:
        _hit_window.pop(0)
    retrieval_hit_rate.set(sum(_hit_window) / len(_hit_window))


def metrics_response() -> tuple[bytes, str]:
    """Returns (body, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
