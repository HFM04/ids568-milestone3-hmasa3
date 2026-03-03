"""
preprocess.py
Loads the Wine dataset, applies standard scaling, and saves train/test splits
to disk. Designed to be idempotent: re-running always overwrites with the same
deterministic output (fixed random_state).
"""

import os
import hashlib
import pickle
import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.environ.get("DATA_DIR", "data")
RANDOM_STATE = 42


def compute_hash(obj) -> str:
    raw = pickle.dumps(obj)
    return hashlib.sha256(raw).hexdigest()


def run_preprocessing(data_dir: str = DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    # Load raw dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # Train/test split (deterministic)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Persist numpy arrays
    np.save(os.path.join(data_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    # Save metadata artifacts
    with open(os.path.join(data_dir, "feature_names.json"), "w") as f:
        json.dump(wine.feature_names, f, indent=2)

    with open(os.path.join(data_dir, "target_names.json"), "w") as f:
        json.dump(wine.target_names.tolist(), f, indent=2)

    # Save scaler
    with open(os.path.join(data_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Compute and save data version hash
    data_hash = compute_hash((X_train_scaled, y_train))
    meta = {
        "data_version": data_hash[:16],
        "data_hash": data_hash,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train_scaled.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "random_state": RANDOM_STATE,
    }

    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[preprocess] Done. "
        f"train={meta['n_train']} test={meta['n_test']} "
        f"data_version={meta['data_version']}"
    )

    return meta


if __name__ == "__main__":
    run_preprocessing()