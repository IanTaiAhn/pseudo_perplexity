import pytest
from monitoring.metrics import _hit_window, metrics_response, record_retrieval_hit, retrieval_hit_rate


def test_record_retrieval_hit_rolling_average():
    _hit_window.clear()
    record_retrieval_hit(True)
    record_retrieval_hit(True)
    record_retrieval_hit(False)
    assert retrieval_hit_rate._value.get() == pytest.approx(2 / 3)


def test_record_retrieval_hit_window_caps_at_100():
    _hit_window.clear()
    for _ in range(150):
        record_retrieval_hit(True)
    assert len(_hit_window) == 100
    assert retrieval_hit_rate._value.get() == 1.0


def test_metrics_response_exposes_registered_metrics():
    _hit_window.clear()
    record_retrieval_hit(True)
    body, content_type = metrics_response()
    text = body.decode()
    assert "text/plain" in content_type
    assert "query_total" in text
    assert "llm_cost_usd_total" in text
    assert "retrieval_hit_rate" in text
