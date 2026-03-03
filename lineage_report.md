# Lineage Report — Wine Classifier  
IDS 568 — Milestone 3

---

## 1. Experiment Overview

Five RandomForest models were trained on the scikit-learn Wine dataset:

- 178 samples  
- 13 numeric features  
- 3 balanced classes  

Objective: classify wine cultivars using chemical composition features.

---

## 2. Data Lineage

| Property | Value |
|----------|--------|
| Dataset | sklearn.datasets.load_wine |
| Random state | 42 |
| Train/Test split | 80% / 20% (stratified) |
| Preprocessing | StandardScaler (fit on train, applied to test) |
| Feature count | 13 |
| Classes | 3 |
| Data version | SHA-256 hash of (scaled X_train, y_train) |
| Metadata file | data/meta.json |

The preprocessing pipeline is fully deterministic. Re-running `preprocess.py` produces identical splits and identical data hashes.

No artificial noise augmentation was applied.

---

## 3. Run Comparison Table

| Run | n_estimators | max_depth | min_split | min_leaf | Accuracy | F1 (weighted) | AUC (OvR) |
|-----|-------------|-----------|-----------|----------|----------|---------------|-----------|
| shallow-fast ⭐ | 50 | 3 | 2 | 1 | 0.9259 | 0.9254 | 0.9898 |
| baseline | 100 | 5 | 2 | 1 | 0.9259 | 0.9251 | 0.9920 |
| deeper-forest | 200 | 10 | 2 | 1 | 0.9074 | 0.9070 | 0.9920 |
| regularized | 150 | 7 | 4 | 2 | 0.9259 | 0.9254 | 0.9930 |
| large-no-depth-limit | 300 | None | 2 | 1 | 0.8889 | 0.8889 | 0.9929 |

Full SHA-256 model hashes are stored as MLflow tags (`model_hash`) for artifact integrity verification.

---

## 4. Analysis

### Bias–Variance Tradeoff

Increasing model complexity beyond moderate levels reduced generalization performance:

- Models with 200+ trees showed decreased accuracy.
- Unlimited depth increased overfitting risk.
- Shallow ensembles generalized best on this small tabular dataset.

This confirms the bias–variance tradeoff in ensemble learning.

### Metric Stability

Accuracy, F1, and AUC remain tightly aligned across configurations, indicating:

- Balanced class distribution
- Stable multiclass discrimination
- Limited sensitivity to moderate hyperparameter variation

---

## 5. Production Candidate

**Selected model:** shallow-fast  

Configuration:

- n_estimators = 50  
- max_depth = 3  

### Justification

1. Equal highest accuracy (0.9259)  
2. Lowest computational complexity  
3. Fastest inference latency  
4. Minimal memory footprint  
5. Equivalent predictive performance to larger models  

When predictive performance is equal, the simplest model is preferred.

---

## 6. Governance & Monitoring Controls

### CI Quality Gates

Automatically enforced via GitHub Actions:

- Accuracy ≥ 0.85  
- F1 ≥ 0.82  
- AUC ≥ 0.90  

Models below threshold fail CI and cannot be promoted.

### Drift Monitoring (Recommended for Production)

- Weekly feature mean and standard deviation comparison  
- Alert if deviation exceeds 2σ  
- Monitor prediction class distribution  

### Model Rollback Strategy

The MLflow Model Registry retains all versions.

To revert to a previous version:

```bash
mlflow models transition-version \
  --model-name wine-classifier \
  --version <previous_version> \
  --stage Production
```

---

## 7. Reproducibility Guarantee

To reproduce the selected production model:

```bash
pip install -r requirements.txt
python preprocess.py
python train.py --n-estimators 50 --max-depth 3
```

The resulting model hash should match the `model_hash` tag stored in MLflow.

Reproducibility is guaranteed by:

- Pinned dependency versions  
- Deterministic preprocessing (`random_state=42`)  
- Logged hyperparameters  
- Stored model artifacts  
- Recorded data version hash  