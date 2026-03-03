"""
run_experiments.py
Runs multiple training experiments and logs them to MLflow.

This script is compatible with:
- preprocess.py (writes data/*.npy + meta.json)
- train.py (logs to MLflow and writes metrics_output/latest_metrics.json)
- model_validation.py (reads metrics_output/latest_metrics.json)

Usage:
  python run_experiments.py
  python run_experiments.py --register-best   (only if registry backend exists)

Notes:
- If MLFLOW_TRACKING_URI is file-based (file:./mlruns), Model Registry operations
  are typically not supported. This script will skip registration unless you pass
  --register-best AND MLflow registry is reachable.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Default envs (can be overridden by your shell/CI)
os.environ.setdefault("MLFLOW_EXPERIMENT", "wine-classification")
os.environ.setdefault("DATA_DIR", "data")
os.environ.setdefault("METRICS_FILE", "metrics_output/latest_metrics.json")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")  # may be None
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT"]
DATA_DIR = os.environ["DATA_DIR"]
METRICS_FILE = os.environ["METRICS_FILE"]

MODEL_NAME = "wine-classifier"

EXPERIMENTS = [
    # (n_estimators, max_depth, min_samples_split, min_samples_leaf, label)
    (50,  3,    2, 1, "shallow-fast"),
    (100, 5,    2, 1, "baseline"),
    (200, 10,   2, 1, "deeper-forest"),
    (150, 7,    4, 2, "regularized"),
    (300, None, 2, 1, "large-no-depth-limit"),
]


def run(cmd: list[str]) -> None:
    """Run a command and stream output. Fail fast on error."""
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def read_latest_metrics(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path) as f:
        return json.load(f)


def can_use_registry() -> bool:
    """
    Heuristic: file-based tracking stores usually don't support registry operations.
    If you're using a tracking server with a backend DB, registry may work.
    """
    uri = (MLFLOW_TRACKING_URI or "").lower()
    if uri.startswith("file:") or uri == "" or uri.startswith("./") or uri.startswith("/"):
        return False
    return True


def main(register_best: bool):
    # Make MLflow deterministic in this script too
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Ensure output dirs exist
    Path("metrics_output").mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Running preprocessing once")
    print("=" * 70)

    # Run preprocessing (single source of truth)
    run([sys.executable, "preprocess.py"])

    print("\n" + "=" * 70)
    print("Running experiments (train.py called 5 times)")
    print("=" * 70)

    results = []
    for n_est, max_depth, min_split, min_leaf, label in EXPERIMENTS:
        # We rely on train.py to:
        # - log to MLflow
        # - write metrics_output/latest_metrics.json with run_id, accuracy, f1_score, auc
        cmd = [
            sys.executable,
            "train.py",
            "--n-estimators", str(n_est),
            "--min-samples-split", str(min_split),
            "--min-samples-leaf", str(min_leaf),
            "--metrics-file", METRICS_FILE,
        ]
        if max_depth is not None:
            cmd += ["--max-depth", str(max_depth)]

        # Tag run name via env (train.py doesn't currently set run_name; that's fine)
        # If you want run names, we can add it to train.py later.
        run(cmd)

        metrics = read_latest_metrics(METRICS_FILE)
        metrics["tag"] = label
        metrics["n_estimators"] = n_est
        metrics["max_depth"] = max_depth
        metrics["min_samples_split"] = min_split
        metrics["min_samples_leaf"] = min_leaf
        results.append(metrics)

        print(
            f"  [{label:25s}] "
            f"acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1_score']:.4f} "
            f"auc={metrics['auc']:.4f} "
            f"run_id={metrics.get('run_id','')[:8]}..."
        )

    best = max(results, key=lambda r: r["accuracy"])

    print("\n" + "=" * 70)
    print("Run Comparison Table (best by accuracy)")
    print("=" * 70)
    print(f"  {'Tag':<25} {'n_est':>6} {'depth':>6} {'acc':>7} {'f1':>7} {'auc':>7}")
    print("  " + "-" * 60)
    for r in results:
        marker = " ← BEST" if r.get("run_id") == best.get("run_id") else ""
        depth = str(r["max_depth"]) if r["max_depth"] is not None else "None"
        print(
            f"  {r['tag']:<25} {r['n_estimators']:>6} {depth:>6} "
            f"{r['accuracy']:>7.4f} {r['f1_score']:>7.4f} {r['auc']:>7.4f}{marker}"
        )

    # Save summary for lineage/reporting
    with open("metrics_output/experiment_summary.json", "w") as f:
        json.dump({"best": best, "all_runs": results}, f, indent=2)

    # Ensure latest_metrics.json reflects best (for CI)
    with open("metrics_output/latest_metrics.json", "w") as f:
        json.dump(
            {
                "accuracy": best["accuracy"],
                "f1_score": best["f1_score"],
                "auc": best["auc"],
                "run_id": best.get("run_id", "unknown"),
            },
            f,
            indent=2,
        )

    print("\n✓ Wrote metrics_output/experiment_summary.json")
    print("✓ Updated metrics_output/latest_metrics.json to best run")

    # Optional: register best model (only if registry backend exists)
    if register_best:
        if not can_use_registry():
            print("\n[SKIP] Registry operations not supported with current MLFLOW_TRACKING_URI.")
            print("       Use a tracking server with a DB backend to enable Model Registry.")
            return

        run_id = best.get("run_id")
        if not run_id:
            raise RuntimeError("Best run has no run_id; cannot register model.")

        print(f"\nRegistering best run (run_id={run_id[:8]}...) as '{MODEL_NAME}'")
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

        print(f"Transitioning '{MODEL_NAME}' v{mv.version}: None → Staging → Production")
        client.transition_model_version_stage(
            name=MODEL_NAME, version=mv.version, stage="Staging", archive_existing_versions=True
        )
        client.transition_model_version_stage(
            name=MODEL_NAME, version=mv.version, stage="Production", archive_existing_versions=True
        )
        print(f"✓ Model '{MODEL_NAME}' v{mv.version} is now in Production")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple MLflow experiments for wine classifier")
    parser.add_argument("--register-best", action="store_true", help="Register/promote best model (requires registry backend)")
    args = parser.parse_args()
    main(register_best=args.register_best)