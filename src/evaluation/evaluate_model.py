#!/usr/bin/env python3
"""Evaluate the trained 24-horizon delta model against a persistence baseline.

This script is the evaluation stage of the MLOps pipeline. It loads a trained model
and a processed dataset, runs a chronological 70/15/15 train/validation/test split,
computes model MAE vs. persistence (zero-delta) baseline MAE, calculates a skill
score, and writes a JSON evaluation report. Optionally logs metrics to MLflow.

The feature-engineering contract is:
    target_delta_h = PM2.5(t+h) - PM2.5(t)

Therefore persistence predicts a zero delta for every horizon. This script uses
the same chronological 70/15/15 split and row filtering as src/train/train.py.

Expected layout in the target repo:
    <root>/data/processed/processed_data.csv              (processed dataset)
    <root>/data/processed/processed_data_meta.json        (feature/target lists)
    <root>/models/model.joblib                            (trained model artifact)
    <root>/models/evaluation_report.json                  (output evaluation report)

MLflow integration:
    * Loads an MLflow model when --model-uri is provided.
    * Optionally attaches local-model evaluation metrics to the run recorded in model_meta.json.
    * Uses --tracking-uri or MLFLOW_TRACKING_URI when connecting to MLflow.

Arguments:
    --input         Processed dataset CSV (default: data/processed/processed_data.csv)
    --meta          Feature meta JSON (default: data/processed/processed_data_meta.json)
    --model         Trained model artifact (default: models/model.joblib)
    --model-version Registered model version to evaluate (uses --model-name)
    --model-name    Registered model name (default: MLFLOW_MODEL_NAME or xgboost_24compact)
    --tracking-uri  MLflow tracking server URI (optional; uses MLflow defaults if omitted)
    --output        Evaluation report output path (default: models/evaluation_report.json)
    --min-skill     Minimum skill threshold (default: 0.20)
    --log-level     Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

Examples:
    python src/evaluation/evaluate_model.py
    python src/evaluation/evaluate_model.py --min-skill 0.25 --log-level DEBUG
    python src/evaluation/evaluate_model.py --model-version 2 --tracking-uri http://localhost:5000

Exit codes:
    0  Evaluation passed (model beats persistence and skill >= threshold).
    1  Evaluation failed (skill below threshold or model does not beat persistence).
    2  Evaluation error (missing files, invalid data, etc.).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

LOG = logging.getLogger("evaluate_model")
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
MIN_SKILL = 0.20
DEFAULT_REGISTERED_MODEL_NAME = "xgboost_24compact"


class EvaluationError(RuntimeError):
    """Raised when the model or evaluation inputs are invalid."""


def load_contract(meta_path: Path) -> tuple[list[str], list[str], int]:
    """Load and validate the feature/target contract from a metadata JSON file.

    Args:
        meta_path: Path to processed_data_meta.json. Must contain "features",
            "targets", and "horizon" keys.

    Returns:
        A tuple of (features, targets, horizon) where features and targets are
        lists of strings and horizon is an integer.

    Raises:
        EvaluationError: If the metadata file is missing, missing required keys,
            has a target contract mismatch, horizon != 24, or empty feature list.
    """
    if not meta_path.is_file():
        raise EvaluationError(f"Feature metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("features", "targets", "horizon"):
        if key not in meta:
            raise EvaluationError(f"{meta_path.name} is missing '{key}'")

    features = list(meta["features"])
    targets = list(meta["targets"])
    horizon = int(meta["horizon"])

    expected_targets = [f"target_delta_{h}" for h in range(1, horizon + 1)]
    if targets != expected_targets:
        raise EvaluationError(
            "Target contract mismatch. Expected ordered targets "
            f"{expected_targets[:3]} ... {expected_targets[-3:]}, got "
            f"{targets[:3]} ... {targets[-3:]}"
        )
    if horizon != 24:
        raise EvaluationError(f"This pipeline expects a 24-hour horizon, got {horizon}")
    if not features:
        raise EvaluationError("Feature list is empty")
    return features, targets, horizon


def load_dataset(
    csv_path: Path, features: list[str], targets: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a processed dataset CSV and return features, targets, and datetimes.

    Args:
        csv_path: Path to processed_data.csv.
        features: List of feature column names.
        targets: List of target column names.

    Returns:
        A tuple of (X, y, datetimes) as float64 numpy arrays, sorted by datetime
        ascending and with NaN-containing rows dropped.

    Raises:
        EvaluationError: If the CSV file is missing, empty, missing required columns,
            or has fewer than 30 valid rows.
    """
    if not csv_path.is_file():
        raise EvaluationError(f"Processed dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise EvaluationError(f"{csv_path.name} contains no rows")

    missing = [column for column in [*features, *targets] if column not in frame.columns]
    if missing:
        raise EvaluationError(f"{csv_path.name} is missing columns: {missing}")

    datetime_col = "datetime" if "datetime" in frame.columns else frame.columns[0]
    datetimes = pd.to_datetime(frame[datetime_col], utc=True).to_numpy()
    X = frame[features].to_numpy(dtype="float64")
    y = frame[targets].to_numpy(dtype="float64")

    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    if not valid.all():
        LOG.warning("Dropping %s rows containing NaN values", int((~valid).sum()))
        X, y, datetimes = X[valid], y[valid], datetimes[valid]

    order = np.argsort(datetimes, kind="stable")
    X, y, datetimes = X[order], y[order], datetimes[order]

    if len(X) < 30:
        raise EvaluationError(f"Not enough valid rows for chronological split: {len(X)}")
    return X, y, datetimes


def chronological_test_split(
    X: np.ndarray,
    y: np.ndarray,
    datetimes: np.ndarray,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the test partition using train.py's exact split arithmetic.

    Splits data chronologically (no shuffle) into train / validation / test using
    the configured fractions. Returns only the test partition.

    Args:
        X: Feature array of shape (n_rows, n_features).
        y: Target array of shape (n_rows, n_targets).
        datetimes: Datetime array of length n_rows, used only for ordering context.
        train_frac: Fraction of rows for training (default: 0.70).
        val_frac: Fraction of rows for validation (default: 0.15).
            Test receives the remainder (default: 0.15).

    Returns:
        A tuple of (X_test, y_test, datetimes_test) containing the test partition.

    Raises:
        EvaluationError: If any split partition has fewer than 10 rows.
    """
    n_rows = len(X)
    n_train = int(n_rows * train_frac)
    n_val = int(n_rows * val_frac)
    n_test = n_rows - n_train - n_val
    if min(n_train, n_val, n_test) < 10:
        raise EvaluationError(
            f"Not enough rows for split: total={n_rows}, train={n_train}, "
            f"validation={n_val}, test={n_test}"
        )
    start = n_train + n_val
    return X[start:], y[start:], datetimes[start:]


def load_model(
    model_path: Path,
    model_version: str | None = None,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    tracking_uri: str | None = None,
):
    """Load a local joblib model or a registered MLflow model version."""
    if model_version is not None:
        try:
            import mlflow
        except ImportError as exc:
            raise EvaluationError(
                "MLflow is required to load --model-version; install mlflow or use --model"
            ) from exc

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}/{model_version}"
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as exc:
            raise EvaluationError(
                f"Could not load registered model {model_name} version {model_version}: {exc}"
            ) from exc

    if not model_path.is_file():
        raise EvaluationError(f"Trained model not found: {model_path}")
    return joblib.load(model_path)


def log_evaluation_to_mlflow(
    metrics: dict[str, Any],
    report_path: Path,
    *,
    run_id: str | None = None,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    model_version: str | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Log evaluation results to a local model's run or a registered model's source run."""
    effective_tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if model_version is None and not (run_id and effective_tracking_uri):
        return

    try:
        import mlflow

        if effective_tracking_uri:
            mlflow.set_tracking_uri(effective_tracking_uri)

        if model_version is not None:
            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=effective_tracking_uri)
            registered_version = client.get_model_version(
                name=model_name, version=model_version
            )
            run_id = registered_version.run_id
            if not run_id:
                raise EvaluationError(
                    f"Registered model {model_name} version {model_version} "
                    "does not have a source run ID"
                )

        if run_id:
            mlflow.log_metrics({
                "test_persistence_mae": metrics["persistence_mae"],
                "test_skill_vs_persistence": metrics["skill"],
                "test_model_beats_persistence": float(metrics["model_beats_persistence"]),
                "test_skill_threshold_passed": float(metrics["skill_threshold_passed"]),
            }, run_id=run_id)
            mlflow.log_artifact(
                str(report_path), artifact_path="evaluation", run_id=run_id
            )
    except Exception:
        LOG.exception("Could not attach evaluation metrics to MLflow; local report remains valid")


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, min_skill: float = MIN_SKILL
) -> dict[str, Any]:
    """Compute evaluation metrics comparing model predictions to a persistence baseline.

    Skill is defined as 1 - (model_mae / persistence_mae), where persistence
    predicts a zero delta for every horizon (equivalent to predicting PM2.5(t)).

    Args:
        y_true: Ground truth target values, shape (n_rows, n_horizons).
        y_pred: Model predictions, same shape as y_true.
        min_skill: Minimum required skill score to pass the gate (default: 0.20).

    Returns:
        A dict with keys:
            model_mae (float), persistence_mae (float), skill (float),
            minimum_required_skill (float), model_beats_persistence (bool),
            skill_threshold_passed (bool), passed (bool),
            mae_reference_band_14_to_15 (bool),
            reference_band_is_informational_only (bool),
            per_horizon (list[dict]).

    Raises:
        EvaluationError: If inputs are not matching 2D arrays or contain
            non-finite values, or if baseline MAE is zero.
    """
    if y_true.ndim != 2 or y_pred.shape != y_true.shape:
        raise EvaluationError(
            f"Expected matching 2D arrays, got y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise EvaluationError("Targets or predictions contain non-finite values")

    # target_delta_h = PM2.5(t+h) - PM2.5(t).
    # Persistence predicts PM2.5(t+h) = PM2.5(t), hence delta prediction = 0.
    baseline_pred = np.zeros_like(y_true, dtype="float64")
    model_mae = float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average"))
    baseline_mae = float(
        mean_absolute_error(y_true, baseline_pred, multioutput="uniform_average")
    )
    if baseline_mae <= 0:
        raise EvaluationError("Baseline MAE is zero; skill score is undefined")

    skill = float(1.0 - (model_mae / baseline_mae))
    model_beats_baseline = model_mae < baseline_mae
    skill_passed = skill >= min_skill

    per_horizon = []
    for index in range(y_true.shape[1]):
        horizon_model_mae = float(mean_absolute_error(y_true[:, index], y_pred[:, index]))
        horizon_baseline_mae = float(
            mean_absolute_error(y_true[:, index], baseline_pred[:, index])
        )
        per_horizon.append({
            "horizon": index + 1,
            "model_mae": horizon_model_mae,
            "persistence_mae": horizon_baseline_mae,
            "skill": (
                float(1.0 - horizon_model_mae / horizon_baseline_mae)
                if horizon_baseline_mae > 0 else None
            ),
        })

    return {
        "model_mae": model_mae,
        "persistence_mae": baseline_mae,
        "skill": skill,
        "minimum_required_skill": min_skill,
        "model_beats_persistence": model_beats_baseline,
        "skill_threshold_passed": skill_passed,
        "passed": bool(model_beats_baseline and skill_passed),
        "mae_reference_band_14_to_15": bool(14.0 <= model_mae <= 15.0),
        "reference_band_is_informational_only": True,
        "per_horizon": per_horizon,
    }


def evaluate(
    dataset_path: Path,
    feature_meta_path: Path,
    model_path: Path,
    output_path: Path,
    min_skill: float = MIN_SKILL,
    *,
    model_version: str | None = None,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Run the full evaluation pipeline and write a JSON report.

    Loads the feature/target contract, dataset, and trained model; performs a
    chronological test split; computes metrics; writes the report atomically;
    and optionally logs metrics to MLflow.

    Args:
        dataset_path: Path to processed_data.csv.
        feature_meta_path: Path to processed_data_meta.json.
        model_path: Path to the trained model.joblib artifact (ignored when
            model_version is provided).
        output_path: Path where the evaluation report JSON will be written.
        min_skill: Minimum required skill score to pass the gate (default: 0.20).
        model_version: Optional version of the registered model to evaluate.
        model_name: Registered model name (default: xgboost_24compact).
        tracking_uri: Optional MLflow tracking server URI used to resolve the model.

    Returns:
        A report dict with keys: evaluation, target_definition, baseline_prediction,
        horizon, features_count, test_rows, test_datetime_min, test_datetime_max,
        split, metrics.

    Raises:
        EvaluationError: If contract validation fails, dataset is invalid,
            the selected model cannot be loaded, or prediction shape mismatches targets.
    """
    features, targets, horizon = load_contract(feature_meta_path)
    X, y, datetimes = load_dataset(dataset_path, features, targets)
    X_test, y_test, test_datetimes = chronological_test_split(X, y, datetimes)

    model = load_model(
        model_path,
        model_version=model_version,
        model_name=model_name,
        tracking_uri=tracking_uri,
    )
    predictions = np.asarray(model.predict(X_test), dtype="float64")

    if predictions.shape != y_test.shape:
        raise EvaluationError(
            f"Model prediction shape {predictions.shape} does not match "
            f"test target shape {y_test.shape}"
        )

    metrics = calculate_metrics(y_test, predictions, min_skill=min_skill)
    report: dict[str, Any] = {
        "evaluation": "persistence_baseline_skill",
        "target_definition": "target_delta_h = PM2.5(t+h) - PM2.5(t)",
        "baseline_prediction": "zero_delta (equivalent to predicting current PM2.5)",
        "horizon": horizon,
        "features_count": len(features),
        "test_rows": int(len(y_test)),
        "test_datetime_min": str(pd.Timestamp(test_datetimes[0])),
        "test_datetime_max": str(pd.Timestamp(test_datetimes[-1])),
        "split": {"train_fraction": TRAIN_FRAC, "validation_fraction": VAL_FRAC,
                  "test_fraction": 1.0 - TRAIN_FRAC - VAL_FRAC},
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".part")
    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)

    LOG.info("Test rows: %s", len(y_test))
    LOG.info("Model MAE: %.6f", metrics["model_mae"])
    LOG.info("Persistence MAE: %.6f", metrics["persistence_mae"])
    LOG.info("Skill: %.2f%% (required: %.2f%%)",
             metrics["skill"] * 100, min_skill * 100)
    LOG.info("Evaluation gate: %s", "PASSED" if metrics["passed"] else "FAILED")
    LOG.info("Report written to %s", output_path)

    # Attach evaluation metrics to the source run for the evaluated model.
    run_id = None
    if model_version is None:
        model_meta_path = model_path.with_name(model_path.stem + "_meta.json")
        if model_meta_path.is_file():
            model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
            run_id = model_meta.get("mlflow_run_id")

    log_evaluation_to_mlflow(
        metrics,
        output_path,
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        tracking_uri=tracking_uri,
    )

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script.

    Args:
        argv: Optional list of argument strings (defaults to sys.argv).

    Returns:
        An argparse.Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained model against persistence baseline."
    )
    parser.add_argument("--input", default="data/processed/processed_data.csv")
    parser.add_argument("--meta", default="data/processed/processed_data_meta.json")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--model", default="models/model.joblib")
    model_source.add_argument(
        "--model-version",
        help="Registered MLflow model version number (e.g. 2)",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MLFLOW_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME),
        help=f"registered model name (default: MLFLOW_MODEL_NAME or {DEFAULT_REGISTERED_MODEL_NAME})",
    )
    parser.add_argument("--tracking-uri", help="MLflow tracking server URI")
    parser.add_argument("--output", default="models/evaluation_report.json")
    parser.add_argument("--min-skill", type=float, default=MIN_SKILL)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the evaluation script.

    Parses arguments, configures logging, runs evaluation, and returns an
    exit code based on whether the evaluation gate passed.

    Args:
        argv: Optional list of argument strings (defaults to sys.argv).

    Returns:
        0 if the evaluation gate passed, 1 if it failed, 2 on error.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )
    try:
        report = evaluate(
            Path(args.input), Path(args.meta), Path(args.model),
            Path(args.output), min_skill=args.min_skill,
            model_version=args.model_version, model_name=args.model_name,
            tracking_uri=args.tracking_uri,
        )
        return 0 if report["metrics"]["passed"] else 1
    except Exception as exc:
        LOG.exception("Evaluation failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
