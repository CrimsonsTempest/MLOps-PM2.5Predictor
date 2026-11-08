#!/usr/bin/env python3
"""Register a successfully evaluated training run and transition its version to Staging.

This script is the final stage of the automated training pipeline (CI/CD job
"train-evaluate-register"). It reads the model metadata JSON produced by
src/train/train.py and the evaluation report JSON produced by
src/evaluation/evaluate_model.py, then:

    1. Verifies the evaluation gate passed (metrics.passed == True).
    2. Extracts the MLflow run ID from the model metadata.
    3. Registers the model artifact (runs:/<run_id>/model) to MLflow Model Registry.
    4. Transitions the registered model version to the "Staging" stage.
    5. Archives existing Staging versions so only one version is in Staging at a time.

Expected layout in the target repo:
    <root>/models/model_meta.json                 (produced by src/train/train.py)
    <root>/models/evaluation_report.json          (produced by src/evaluation/evaluate_model.py)

MLflow integration:
    * Uses MLFLOW_TRACKING_URI to connect to the tracking server.
    * Registers the model from the specific training run (runs:/<run_id>/model).
    * Transitions the version to "Staging" with archive_existing_versions=True.

Arguments:
    --tracking-uri    MLflow tracking URI (default: env MLFLOW_TRACKING_URI)
    --model-name      Registered model name (default: env MLFLOW_MODEL_NAME or "pm25-lag24-xgb")
    --model-meta      Path to model_meta.json (default: models/model_meta.json)
    --evaluation-report  Path to evaluation_report.json (default: models/evaluation_report.json)

Examples:
    python src/registry/register_model.py
    python src/registry/register_model.py --model-name xgboost_24compact
    python src/registry/register_model.py --tracking-uri sqlite:///mlflow.db --model-name my-model

Exit codes:
    0  Registration succeeded and model transitioned to Staging.
    1  Registration failed (missing files, evaluation gate not passed, MLflow error, etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def register(
    tracking_uri: str,
    model_name: str,
    model_meta_path: Path,
    evaluation_report_path: Path,
) -> dict:
    """Register the trained model to MLflow Model Registry and transition to Staging.

    Args:
        tracking_uri: MLflow tracking server URI (e.g. "sqlite:///mlflow.db" or
            "http://mlflow-server:5000").
        model_name: Name to register the model under in the Model Registry.
        model_meta_path: Path to model_meta.json produced by train.py. Must contain
            an "mlflow_run_id" field.
        evaluation_report_path: Path to evaluation_report.json produced by
            evaluate_model.py. Must have metrics.passed == True.

    Returns:
        A dict with registration details:
            registered_model_name (str), model_version (str), stage ("Staging"),
            run_id (str), model_uri (str), evaluation_skill (float),
            evaluation_model_mae (float), evaluation_persistence_mae (float).

    Raises:
        ValueError: If tracking_uri or model_name is empty.
        FileNotFoundError: If model_meta.json or evaluation_report.json does not exist.
        RuntimeError: If the evaluation gate did not pass, or if mlflow_run_id is missing
            from the model metadata.
    """
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is required")
    if not model_name:
        raise ValueError("MLFLOW_MODEL_NAME is required")
    if not model_meta_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {model_meta_path}")
    if not evaluation_report_path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {evaluation_report_path}")

    model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_report_path.read_text(encoding="utf-8"))

    if not evaluation.get("metrics", {}).get("passed", False):
        raise RuntimeError("Refusing to register: evaluation gate did not pass")

    run_id = model_meta.get("mlflow_run_id")
    if not run_id:
        raise RuntimeError("Model metadata does not contain mlflow_run_id")

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Stages are deprecated in newer MLflow versions, but this project explicitly
    # uses the Staging stage for its assignment workflow.
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Staging",
        archive_existing_versions=True,
    )

    return {
        "registered_model_name": model_name,
        "model_version": str(registered.version),
        "stage": "Staging",
        "run_id": run_id,
        "model_uri": model_uri,
        "evaluation_skill": evaluation["metrics"]["skill"],
        "evaluation_model_mae": evaluation["metrics"]["model_mae"],
        "evaluation_persistence_mae": evaluation["metrics"]["persistence_mae"],
    }


def main() -> int:
    """CLI entry point for model registration.

    Parses command-line arguments, calls register(), and prints the result as JSON.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Register a trained model to MLflow Model Registry."
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--model-name", default=os.environ.get("MLFLOW_MODEL_NAME", "pm25-lag24-xgb"))
    parser.add_argument("--model-meta", default="models/model_meta.json")
    parser.add_argument("--evaluation-report", default="models/evaluation_report.json")
    args = parser.parse_args()

    try:
        result = register(
            args.tracking_uri,
            args.model_name,
            Path(args.model_meta),
            Path(args.evaluation_report),
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"Registration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
