"""
train.py
Trains a RandomForest classifier on the preprocessed Wine dataset and logs
everything to MLflow: hyperparameters, metrics, model artifact, and hashes.

Usage:
    python train.py                          # default hyperparams
    python train.py --n-estimators 200 --max-depth 10
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_DIR = os.environ.get("DATA_DIR", "data")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "wine-classification")


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
):
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
            "data_version": meta["data_version"],
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
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

        # --- Save model artifact with hash ---
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(clf, f)

            model_hash = compute_file_hash(model_path)
            mlflow.set_tag("model_hash", model_hash)
            mlflow.set_tag("model_sha256", model_hash)

            # Log model
            mlflow.sklearn.log_model(clf, artifact_path="model")
            mlflow.log_artifact(model_path, artifact_path="model_pkl")

        # --- Log metrics to file artifact for CI validation ---
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as mf:
            json.dump(
                {**metrics, "run_id": run.info.run_id, "model_hash": model_hash},
                mf,
                indent=2,
            )
            mlflow.log_artifact(mf.name, artifact_path="metrics")

        run_id = run.info.run_id
        print(f"[train] MLflow run_id={run_id}  model_hash={model_hash[:16]}...")

        # --- Optionally register ---
        if register:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, "wine-classifier")
            print(f"[train] Registered model version={mv.version}")

        # Write metrics to local file so CI can read them without MLflow client
        os.makedirs("metrics_output", exist_ok=True)
        with open("metrics_output/latest_metrics.json", "w") as f:
            json.dump({**metrics, "run_id": run_id}, f, indent=2)

        return run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    run_training(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        register=args.register,
    )