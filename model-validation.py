"""
model_validation.py
Reads the latest training metrics and enforces quality thresholds.
Exits with code 1 (fails CI) if any threshold is not met.

Usage:
    python model_validation.py
    python model_validation.py --min-accuracy 0.90 --min-f1 0.88 --min-auc 0.95
    python model_validation.py --metrics-file metrics_output/latest_metrics.json
"""

import argparse
import json
import os
import sys

DEFAULT_MIN_ACCURACY = 0.85
DEFAULT_MIN_F1 = 0.82
DEFAULT_MIN_AUC = 0.90

METRICS_FILE = os.environ.get("METRICS_FILE", "metrics_output/latest_metrics.json")


def validate(
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
    min_f1: float = DEFAULT_MIN_F1,
    min_auc: float = DEFAULT_MIN_AUC,
    metrics_file: str = METRICS_FILE,
) -> bool:
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found: {metrics_file}")
        print("Make sure train.py ran successfully before validation.")
        sys.exit(1)

    with open(metrics_file) as f:
        metrics = json.load(f)

    # Fail loudly if schema changes
    required = ["accuracy", "f1_score", "auc"]
    missing = [k for k in required if k not in metrics]
    if missing:
        print(f"ERROR: Metrics file missing keys: {missing}")
        print(f"Found keys: {sorted(metrics.keys())}")
        sys.exit(1)

    accuracy = float(metrics["accuracy"])
    f1 = float(metrics["f1_score"])
    auc = float(metrics["auc"])
    run_id = metrics.get("run_id", "unknown")

    print("=" * 55)
    print("  Model Validation Report")
    print("=" * 55)
    print(f"  MLflow run_id : {run_id}")
    print(f"  accuracy      : {accuracy:.4f}  (threshold >= {min_accuracy})")
    print(f"  f1_score      : {f1:.4f}  (threshold >= {min_f1})")
    print(f"  auc           : {auc:.4f}  (threshold >= {min_auc})")
    print("=" * 55)

    failures = []
    if accuracy < min_accuracy:
        failures.append(f"FAILED accuracy={accuracy:.4f} < threshold={min_accuracy}")
    if f1 < min_f1:
        failures.append(f"FAILED f1_score={f1:.4f} < threshold={min_f1}")
    if auc < min_auc:
        failures.append(f"FAILED auc={auc:.4f} < threshold={min_auc}")

    if failures:
        print("\n  ✗ Quality gate FAILED:")
        for msg in failures:
            print(f"    • {msg}")
        print("\n  Model is NOT approved for promotion.")
        print("=" * 55)
        sys.exit(1)

    print("\n  ✓ All quality gates PASSED.")
    print("  Model is approved for Staging promotion.")
    print("=" * 55)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model quality gates")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1)
    parser.add_argument("--min-auc", type=float, default=DEFAULT_MIN_AUC)
    parser.add_argument("--metrics-file", type=str, default=METRICS_FILE)
    args = parser.parse_args()

    validate(
        min_accuracy=args.min_accuracy,
        min_f1=args.min_f1,
        min_auc=args.min_auc,
        metrics_file=args.metrics_file,
    )