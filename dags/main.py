"""
Airflow DAG — Wine Classifier Training Pipeline
IDS 568 Milestone 3

Demonstrates:
  - DAG definition with default_args (retries, retry_delay, on_failure_callback)
  - PythonOperator for each pipeline stage
  - XCom for passing data between tasks
  - Task dependencies with the >> (bitshift) operator
  - Idempotent task design (safe to re-run)

Run with:
  airflow dags test train_pipeline 2024-01-01
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# Failure callback — called automatically by Airflow when a task fails after
# all retries are exhausted. In production you would send a Slack/email alert.
# ─────────────────────────────────────────────────────────────────────────────
def on_failure_callback(context):
    """Print a clear error summary when a task fails permanently."""
    dag_id  = context['dag'].dag_id
    task_id = context['task_instance'].task_id
    exec_dt = context['execution_date']
    print(f"[ALERT] Task failed — DAG: {dag_id} | Task: {task_id} | Date: {exec_dt}")
    # Replace the print with a Slack/email call in a real production system.


# ─────────────────────────────────────────────────────────────────────────────
# Default arguments applied to every task in this DAG.
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    'owner': 'mlops',
    # Don't wait for a previous run to succeed before scheduling the next one.
    'depends_on_past': False,
    # Retry a failed task up to 2 times before giving up.
    'retries': 2,
    # Wait 1 minute between retry attempts (prevents hammering a flaky service).
    'retry_delay': timedelta(minutes=1),
    # Call our alert function once all retries are exhausted.
    'on_failure_callback': on_failure_callback,
}


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — preprocess_data
# Loads the Wine dataset, scales features, and "saves" the result.
# Metadata is passed downstream via XCom (Airflow's key-value store).
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(**context):
    """Task 1: Simulate data preprocessing for the Wine dataset."""
    print("=" * 50)
    print("PREPROCESS_DATA: Starting data preprocessing...")

    # Simulate loading and cleaning the Wine dataset (178 samples, 13 features).
    n_samples    = 178
    n_features   = 13
    data_path    = "/tmp/wine_processed.csv"
    data_version = "wine_v1"   # version tag logged to MLflow for lineage

    print(f"  - Loaded {n_samples} samples with {n_features} features")
    print(f"  - Applied StandardScaler (zero mean, unit variance)")
    print(f"  - Split into 70% train / 30% test (stratified)")
    print(f"  - Data version tag: {data_version}")
    print(f"  - Saved processed data to {data_path}")
    print("PREPROCESS_DATA: Complete!")
    print("=" * 50)

    # Returning a dict automatically pushes it to XCom under 'return_value'.
    # Downstream tasks retrieve it with ti.xcom_pull(task_ids='preprocess_data').
    # Idempotency: re-running always produces the same output because the
    # real preprocess.py uses a fixed random_state=42.
    return {
        'data_path':    data_path,
        'n_samples':    n_samples,
        'data_version': data_version,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — train_model
# Reads preprocessing metadata from XCom, simulates training a RandomForest,
# and logs hyperparameters, metrics, and model hash to MLflow.
# ─────────────────────────────────────────────────────────────────────────────
def train_model(**context):
    """Task 2: Simulate model training and MLflow experiment logging."""
    print("=" * 50)
    print("TRAIN_MODEL: Starting model training...")

    # ── Pull upstream results from XCom ──────────────────────────────────────
    ti = context['ti']                               # TaskInstance shorthand
    preprocess_result = ti.xcom_pull(task_ids='preprocess_data')

    if preprocess_result:
        print(f"  - Received from preprocess_data: {preprocess_result}")
        data_path    = preprocess_result['data_path']
        n_samples    = preprocess_result['n_samples']
        data_version = preprocess_result['data_version']
    else:
        # Fallback values used when the task is run in isolation for testing.
        data_path    = "/tmp/wine_processed.csv"
        n_samples    = 178
        data_version = "wine_v1"

    # ── Hyperparameters ───────────────────────────────────────────────────────
    # In the real pipeline these come from Airflow Variables so they can be
    # changed in the UI without touching code.
    hyperparams = {
        'n_estimators': 100,
        'max_depth':    5,
        'random_state': 42,
    }

    # ── Simulate training results ─────────────────────────────────────────────
    model_path    = "/tmp/wine_model.pkl"
    mlflow_run_id = "abc123def456"    # In reality returned by mlflow.start_run()
    accuracy      = 0.9259
    f1_score      = 0.9254
    auc           = 0.9920
    model_hash    = "70718efe4435ff1d"  # SHA-256 of serialised model (first 16 chars)

    print(f"  - Training RandomForest on {n_samples} samples from {data_path}")
    print(f"  - Hyperparameters: {hyperparams}")
    print(f"  - data_version logged to MLflow: {data_version}")
    print(f"  - MLflow run_id: {mlflow_run_id}")
    print(f"  - Accuracy:  {accuracy}")
    print(f"  - F1 score:  {f1_score}")
    print(f"  - AUC:       {auc}")
    print(f"  - Model hash (SHA-256): {model_hash}")
    print(f"  - Saved model to {model_path}")
    print("TRAIN_MODEL: Complete!")
    print("=" * 50)

    # Idempotency: each run creates a NEW MLflow run (additive versioning).
    # Re-running never overwrites a previous run — it just adds another entry.
    return {
        'model_path':    model_path,
        'mlflow_run_id': mlflow_run_id,
        'accuracy':      accuracy,
        'f1_score':      f1_score,
        'auc':           auc,
        'model_hash':    model_hash,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — register_model
# Pulls training results from XCom, enforces quality thresholds (quality gate),
# then registers the model to the MLflow Model Registry: None → Staging.
# ─────────────────────────────────────────────────────────────────────────────
def register_model(**context):
    """Task 3: Validate quality thresholds and register model to MLflow registry."""
    print("=" * 50)
    print("REGISTER_MODEL: Starting model registration...")

    # ── Pull upstream results from XCom ──────────────────────────────────────
    ti = context['ti']
    train_result = ti.xcom_pull(task_ids='train_model')

    if train_result:
        print(f"  - Received from train_model: {train_result}")
        accuracy      = train_result['accuracy']
        f1_score      = train_result['f1_score']
        mlflow_run_id = train_result['mlflow_run_id']
        model_hash    = train_result['model_hash']
    else:
        # Fallback values for isolated testing.
        accuracy      = 0.9259
        f1_score      = 0.9254
        mlflow_run_id = "abc123def456"
        model_hash    = "70718efe4435ff1d"

    # ── Quality gate ──────────────────────────────────────────────────────────
    # Raising a ValueError causes Airflow to mark the task as failed,
    # which triggers retries and eventually on_failure_callback.
    MIN_ACCURACY = 0.85
    MIN_F1       = 0.82
    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Quality gate FAILED: accuracy={accuracy:.4f} < threshold={MIN_ACCURACY}"
        )
    if f1_score < MIN_F1:
        raise ValueError(
            f"Quality gate FAILED: f1_score={f1_score:.4f} < threshold={MIN_F1}"
        )

    # ── Simulate MLflow Model Registry steps ─────────────────────────────────
    model_name    = "wine-classifier"
    model_version = "v1.0"
    stage         = "Staging"   # Full progression: None → Staging → Production

    print(f"  - Quality gate PASSED: accuracy={accuracy} >= {MIN_ACCURACY} ✓")
    print(f"  - Quality gate PASSED: f1_score={f1_score}  >= {MIN_F1} ✓")
    print(f"  - Registering '{model_name}' from MLflow run {mlflow_run_id}")
    print(f"  - Model hash logged for reproducibility: {model_hash}")
    print(f"  - Version assigned: {model_version}")
    print(f"  - Stage transition: None → {stage}")
    print("REGISTER_MODEL: Complete!")
    print("=" * 50)

    # Idempotency: MLflow registry versioning is additive — re-running creates
    # a new version number but never deletes old versions, so rollback is
    # always possible by transitioning a previous version back to Production.
    return {
        'model_name':    model_name,
        'model_version': model_version,
        'stage':         stage,
        'registered':    True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# Everything inside the `with DAG(...)` block belongs to this DAG.
# ─────────────────────────────────────────────────────────────────────────────
with DAG(
    # Unique identifier shown in the Airflow UI and used by the CLI.
    dag_id='train_pipeline',

    # Apply the shared retry / callback settings to every task.
    default_args=default_args,

    description='Wine classifier: preprocess → train → register (IDS 568 M3)',

    # Run once a week automatically; can also be triggered manually from the UI.
    schedule='@weekly',

    # Airflow uses start_date to calculate the first scheduled run interval.
    start_date=datetime(2024, 1, 1),

    # Don't backfill historical runs between start_date and today.
    catchup=False,

    tags=['mlops', 'wine', 'training'],
) as dag:

    # ── Define tasks ──────────────────────────────────────────────────────────
    # PythonOperator runs a plain Python function as an Airflow task.
    # task_id must be unique within the DAG — it is also the key used by
    # xcom_pull() to retrieve the return value of a specific task.

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
    )

    # ── Set task dependencies ─────────────────────────────────────────────────
    # The >> operator sets execution order.
    # Airflow will NOT start train until preprocess succeeds, and will NOT
    # start register until train succeeds.
    #
    #   preprocess_data  →  train_model  →  register_model
    #
    preprocess >> train >> register