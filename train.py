"""
train.py
Trains a RandomForest classifier on the preprocessed Wine dataset and logs
everything to MLflow: hyperparameters, metrics, model artifact, and hashes.

Usage:
    python train.py
    python train.py --n-estimators 200 --max-depth 10
    python train.py --metrics-file metrics_output/latest_metrics.json
"""

import argparse
import hashlib
import json
import os
import pickle
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_DIR = os.environ.get("DATA_DIR", "data")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "wine-classification")

# CI-friendly, configurable outputs
DEFAULT_METRICS_FILE = os.environ.get("METRICS_FILE", "metrics_output/latest_metrics.json")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")  # may be None


def load_data(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    return X_train, X_test, y_train, y_test, meta


def compute_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_training(
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    data_dir: str = DATA_DIR,
    register: bool = False,
    metrics_file: str = DEFAULT_METRICS_FILE,
):
    # Ensure MLflow uses the same backend in local + CI
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, X_test, y_train, y_test, meta = load_data(data_dir)

    with mlflow.start_run() as run:
        # --- Log hyperparameters ---
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth if max_depth is not None else "None",
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": 42,
            "data_version": meta.get("data_version", "unknown"),
            "n_train": meta.get("n_train", int(len(y_train))),
            "n_test": meta.get("n_test", int(len(y_test))),
        }
        mlflow.log_params(params)
        mlflow.set_tag("dataset", "wine")
        mlflow.set_tag("model_type", "RandomForest")

        # --- Train ---
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

        metrics = {"accuracy": accuracy, "f1_score": f1, "auc": auc}
        mlflow.log_metrics(metrics)

        print(f"[train] accuracy={accuracy:.4f}  f1={f1:.4f}  auc={auc:.4f}")

        # --- Save model artifact + hash of serialized bytes ---
        with tempfile.TemporaryDirectory() as tmp:
            model_pkl_path = os.path.join(tmp, "model.pkl")
            with open(model_pkl_path, "wb") as f:
                pickle.dump(clf, f)

            model_hash = compute_file_hash(model_pkl_path)
            mlflow.set_tag("model_hash", model_hash)
            mlflow.set_tag("model_sha256", model_hash)

            # Log model in two ways: MLflow flavor + raw pickle
            mlflow.sklearn.log_model(clf, artifact_path="model")
            mlflow.log_artifact(model_pkl_path, artifact_path="model_pkl")

        run_id = run.info.run_id
        print(f"[train] MLflow run_id={run_id}  model_hash={model_hash[:16]}...")

        # --- Log metrics JSON as MLflow artifact ---
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as mf:
            json.dump(
                {**metrics, "run_id": run_id, "model_hash": model_hash},
                mf,
                indent=2,
            )
            metrics_artifact_path = mf.name

        mlflow.log_artifact(metrics_artifact_path, artifact_path="metrics")

        # --- Optional register (only works if registry backend exists) ---
        if register:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, "wine-classifier")
            print(f"[train] Registered model version={mv.version}")

        # --- Write metrics to local file so CI can validate ---
        os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump({**metrics, "run_id": run_id}, f, indent=2)

        print(f"[train] Wrote metrics file: {metrics_file}")

        return run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a wine classifier and log to MLflow")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--metrics-file", type=str, default=DEFAULT_METRICS_FILE)
    args = parser.parse_args()

    run_training(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        register=args.register,
        metrics_file=args.metrics_file,
    )
