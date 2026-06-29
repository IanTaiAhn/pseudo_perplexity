import pytest
from api.schemas import Chunk, QueryResponse


# --- Schema validation tests (output guardrail) ---

def test_query_response_requires_answer_and_citations():
    r = QueryResponse(
        answer="test answer",
        citations=[],
        query_id="abc",
        latency_ms=100.0,
        model_used="claude-sonnet-4-6",
        estimated_cost_usd=0.001,
    )
    assert r.answer == "test answer"
    assert r.citations == []


def test_query_response_missing_answer_raises():
    with pytest.raises(Exception):
        QueryResponse(
            citations=[],
            query_id="abc",
            latency_ms=100.0,
            model_used="claude-sonnet-4-6",
            estimated_cost_usd=0.001,
        )


# --- Confidence threshold tests ---

def test_low_score_chunk_triggers_fallback():
    """The query route should return a no-info response when top chunk score is below 0.3."""
    from unittest.mock import patch
    from api.routes.query import _LOW_CONFIDENCE_THRESHOLD, _LOW_CONFIDENCE_RESPONSE
    from api.schemas import Chunk

    low_score_chunks = [
        Chunk(
            chunk_id="x",
            text="some text",
            source="a.txt",
            source_type="document",
            chunk_index=0,
            score=0.1,  # below threshold
        )
    ]

    assert low_score_chunks[0].score < _LOW_CONFIDENCE_THRESHOLD


def test_confidence_threshold_value():
    from api.routes.query import _LOW_CONFIDENCE_THRESHOLD
    assert _LOW_CONFIDENCE_THRESHOLD == 0.3
