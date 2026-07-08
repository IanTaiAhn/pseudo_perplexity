import numpy as np
import pytest

from monitoring import drift_detector


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_detector, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(drift_detector, "_LOG_PATH", tmp_path / "query_embeddings.jsonl")
    monkeypatch.setattr(drift_detector, "_BASELINE_PATH", tmp_path / "baseline_embeddings.json")


def _unit_vectors(mean_direction, n, noise=0.1, seed=0):
    """Synthetic embeddings clustered around a direction, L2-normalized like embedder.py's output."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(loc=mean_direction, scale=noise, size=(n, len(mean_direction)))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).tolist()


def test_check_drift_without_baseline_reports_no_drift():
    result = drift_detector.check_drift()
    assert result.drifted is False
    assert "baseline" in result.reason.lower()


def test_snapshot_baseline_requires_minimum_samples():
    for emb in _unit_vectors([1, 0, 0], 5):
        drift_detector.log_query_embedding(emb)
    with pytest.raises(ValueError):
        drift_detector.snapshot_baseline(min_samples=30)


def test_check_drift_flags_shifted_query_distribution():
    # Baseline: queries clustered tightly around one direction in embedding space.
    for emb in _unit_vectors([1.0, 0.0, 0.0], 60, seed=1):
        drift_detector.log_query_embedding(emb)
    drift_detector.snapshot_baseline(min_samples=30)

    # "Recent" queries clustered around an unrelated direction — simulates
    # users asking about a topic the system was never tuned to retrieve for.
    for emb in _unit_vectors([0.0, 1.0, 0.0], 60, seed=2):
        drift_detector.log_query_embedding(emb)

    result = drift_detector.check_drift(recent_window=60)

    assert result.drifted is True
    assert result.p_value < 0.05
    assert result.baseline_size == 60
    assert result.recent_size == 60


def test_check_drift_does_not_flag_stable_distribution():
    for emb in _unit_vectors([1.0, 0.0, 0.0], 80, seed=3):
        drift_detector.log_query_embedding(emb)
    drift_detector.snapshot_baseline(min_samples=30)

    # Same direction, different random draw — the normal case in production.
    for emb in _unit_vectors([1.0, 0.0, 0.0], 80, seed=4):
        drift_detector.log_query_embedding(emb)

    result = drift_detector.check_drift(recent_window=80)

    assert result.drifted is False
    assert result.p_value >= 0.05


def test_check_drift_reports_insufficient_recent_samples():
    # Only 20 queries logged total (all consumed into the baseline itself),
    # so there aren't enough *subsequent* queries yet to test drift against.
    for emb in _unit_vectors([1.0, 0.0, 0.0], 20, seed=5):
        drift_detector.log_query_embedding(emb)
    drift_detector.snapshot_baseline(min_samples=15)

    result = drift_detector.check_drift(recent_window=200)

    assert result.drifted is False
    assert "recent" in result.reason.lower()
