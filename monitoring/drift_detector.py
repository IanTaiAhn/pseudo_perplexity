# Input embedding drift detection
"""
Detects whether incoming query embeddings have drifted from a baseline
distribution captured when the system launched (or was last re-baselined).

Why a scalar summary instead of testing all 384 embedding dimensions:
a KS test is a 1-D test. Running it independently per dimension and
declaring "drift" if any one dimension trips is both noisy (5% of
dimensions will look significant at p<0.05 by chance alone) and doesn't
answer the question we actually care about — has the *topic mix* of
queries shifted? Instead we reduce each embedding to a single number,
its cosine similarity to the baseline centroid (the average direction of
"normal" queries), and KS-test the *distribution of that number* between
the baseline window and the recent window. A shift in that distribution
means recent queries are, on average, pointing in different directions
in embedding space than the queries the system was tuned against.
"""
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

_DATA_DIR = Path(os.getenv("MONITORING_DATA_DIR", "monitoring/data"))
_LOG_PATH = _DATA_DIR / "query_embeddings.jsonl"
_BASELINE_PATH = _DATA_DIR / "baseline_embeddings.json"

_DRIFT_P_VALUE_THRESHOLD = 0.05
_MIN_BASELINE_SIZE = 30
_MIN_RECENT_SIZE = 30


@dataclass
class DriftResult:
    drifted: bool
    p_value: float
    statistic: float
    baseline_size: int
    recent_size: int
    reason: str = ""


def log_query_embedding(embedding: list[float], query: str = "") -> None:
    """Append one query embedding to the rolling log. Call this on every query."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "embedding": embedding,
    }
    with open(_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read_log() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    with open(_LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def snapshot_baseline(min_samples: int = _MIN_BASELINE_SIZE) -> int:
    """
    Freeze the current query log as the drift baseline. Run this once after
    the system has processed a healthy, representative batch of queries
    (e.g. after the eval suite, or after the first week in production).

    Returns the number of embeddings captured in the baseline.
    Raises ValueError if fewer than min_samples queries have been logged —
    a baseline built from too few points is itself noisy and will produce
    false-positive drift alerts.
    """
    records = _read_log()
    if len(records) < min_samples:
        raise ValueError(
            f"Only {len(records)} queries logged; need at least {min_samples} "
            "to build a stable baseline."
        )
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BASELINE_PATH, "w") as f:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embeddings": [r["embedding"] for r in records],
            },
            f,
        )
    return len(records)


def _centroid_similarities(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row in `embeddings` to `centroid`.

    Embeddings from embedder.py are already L2-normalized, so a plain dot
    product against a normalized centroid *is* cosine similarity.
    """
    centroid_norm = centroid / np.linalg.norm(centroid)
    return embeddings @ centroid_norm


def check_drift(recent_window: int = 200) -> DriftResult:
    """
    Compare the most recent `recent_window` logged queries against the
    stored baseline using a KS test on distance-to-centroid.

    Intended to run on a schedule (spec: weekly) via a cron job or
    scheduled task, not on the request path — this is a monitoring signal,
    not a per-query gate.
    """
    if not _BASELINE_PATH.exists():
        return DriftResult(
            drifted=False, p_value=1.0, statistic=0.0,
            baseline_size=0, recent_size=0,
            reason="No baseline set. Call snapshot_baseline() first.",
        )

    with open(_BASELINE_PATH) as f:
        baseline_embeddings = np.array(json.load(f)["embeddings"])

    records = _read_log()[-recent_window:]
    if len(records) < _MIN_RECENT_SIZE:
        return DriftResult(
            drifted=False, p_value=1.0, statistic=0.0,
            baseline_size=len(baseline_embeddings), recent_size=len(records),
            reason=f"Only {len(records)} recent queries logged; need at least {_MIN_RECENT_SIZE}.",
        )
    recent_embeddings = np.array([r["embedding"] for r in records])

    centroid = baseline_embeddings.mean(axis=0)
    baseline_scores = _centroid_similarities(baseline_embeddings, centroid)
    recent_scores = _centroid_similarities(recent_embeddings, centroid)

    statistic, p_value = ks_2samp(baseline_scores, recent_scores)

    return DriftResult(
        drifted=bool(p_value < _DRIFT_P_VALUE_THRESHOLD),
        p_value=float(p_value),
        statistic=float(statistic),
        baseline_size=len(baseline_embeddings),
        recent_size=len(recent_embeddings),
    )
