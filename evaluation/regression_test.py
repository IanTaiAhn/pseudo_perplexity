"""
CI regression gate — runs the eval dataset through RAGAS and fails with a
non-zero exit code if any metric drops below its block threshold.
Thresholds are defined in perplexity_clone_spec.md section 8.
"""
import sys

from evaluation.ragas_runner import run

_BLOCK_THRESHOLDS = {
    "faithfulness": 0.75,
    "context_relevance": 0.70,
    "answer_relevance": 0.75,
    "context_recall": 0.70,
}
_WARN_THRESHOLDS = {
    "faithfulness": 0.85,
    "context_relevance": 0.80,
    "answer_relevance": 0.85,
    "context_recall": 0.80,
}


def main() -> int:
    results = run()
    aggregate = results["aggregate"]
    per_question = results["per_question"]

    failed = False
    for metric, block_threshold in _BLOCK_THRESHOLDS.items():
        score = aggregate.get(metric)
        if score is None:
            print(f"MISSING metric '{metric}' in results — treating as failure")
            failed = True
            continue

        warn_threshold = _WARN_THRESHOLDS[metric]
        if score < block_threshold:
            print(f"FAIL {metric}: {score:.3f} < block threshold {block_threshold}")
            failed = True
        elif score < warn_threshold:
            print(f"WARN {metric}: {score:.3f} < warn threshold {warn_threshold}")
        else:
            print(f"PASS {metric}: {score:.3f}")

    if failed:
        print("\nRegressed questions (below block threshold on at least one metric):")
        for row in per_question:
            regressed = [
                m for m in _BLOCK_THRESHOLDS
                if row.get(m) is not None and row[m] < _BLOCK_THRESHOLDS[m]
            ]
            if regressed:
                print(f"  - {row['user_input'][:80]!r} — low: {regressed}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
