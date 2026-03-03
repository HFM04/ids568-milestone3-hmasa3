# IDS 568 Milestone 3 — MLOps Pipeline
[![Train and Validate](https://github.com/HFM04/ids568-milestone3-hmasa3/actions/workflows/train_and_validate.yml/badge.svg)](https://github.com/HFM04/ids568-milestone3-hmasa3/actions/workflows/train_and_validate.yml)

End-to-end workflow automation and model governance using:

- Apache Airflow (Docker-based)
- MLflow experiment tracking
- GitHub Actions CI/CD
- Deterministic preprocessing and reproducible training

---

## Repository Structure

```
├── .github/workflows/train_and_validate.yml
├── dags/train_pipeline.py
├── preprocess.py
├── train.py
├── model_validation.py
├── run_experiments.py
├── metrics_output/
├── lineage_report.md
├── requirements.txt
└── README.md
```

---

## Quick Start (Local ML + Validation)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing

```bash
python preprocess.py
```

- Loads Wine dataset  
- Stratified 80/20 split  
- Applies StandardScaler  
- Saves data artifacts to `data/`  
- Writes `meta.json` with SHA-256 data version hash  

### 3. Train a model

```bash
python train.py --n-estimators 100 --max-depth 5
```

This will:

- Train RandomForest  
- Log parameters and metrics to MLflow  
- Log model artifact and model hash  
- Write `metrics_output/latest_metrics.json`  

### 4. Validate quality gates

```bash
python model_validation.py
```

Quality thresholds:

- Accuracy ≥ 0.85  
- F1 ≥ 0.82  
- AUC ≥ 0.90  

---

## Running All Experiments

```bash
python run_experiments.py
```

This:

- Runs preprocessing once  
- Trains 5 hyperparameter configurations  
- Logs all runs to MLflow  
- Writes `experiment_summary.json`  
- Updates `latest_metrics.json` with best run  

Optional registry promotion (only if registry backend exists):

```bash
python run_experiments.py --register-best
```

---

## MLflow UI

```bash
mlflow ui --port 5000
```

Open:

```
http://localhost:5000
```

View:

- Experiment runs  
- Parameters  
- Metrics  
- Artifacts  
- Model hashes  

---

## Airflow Orchestration (Docker-Based)

Airflow runs via Docker (`apache/airflow:3.1.7`).

Start services:

```bash
docker compose up -d
```

DAG:

```
train_pipeline
```

Pipeline stages:

```
preprocess_data → train_model → register_model
```

The `register_model` step enforces quality gates before promotion.

---

## CI/CD Workflow

GitHub Actions triggers on:

- Push to `main`  
- Pull request to `main`  
- Manual workflow dispatch  

Pipeline steps:

1. Install dependencies  
2. Preprocess data  
3. Train model  
4. Validate metrics  
5. Upload artifacts  

CI fails automatically if quality thresholds are not met.

Artifacts uploaded:

- `mlruns/`  
- `metrics_output/`  

---

## Idempotency Design

| Stage | Strategy |
|--------|-----------|
| Preprocess | Deterministic split (`random_state=42`) |
| Train | Creates new MLflow run (additive) |
| Register | Registry versioning is additive |

---

## Lineage Tracking

Each model run records:

- Data version (SHA-256)  
- Hyperparameters  
- Metrics  
- Model artifact  
- Model hash  
- Run ID  

See `lineage_report.md` for full analysis and production candidate selection.
