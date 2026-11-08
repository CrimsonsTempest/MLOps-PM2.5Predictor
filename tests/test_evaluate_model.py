#!/usr/bin/env python3
"""Unit tests for src/evaluation/evaluate_model.py.

These tests verify the evaluation helper functions (calculate_metrics and
chronological_test_split) using synthetic arrays. They do not train a model,
do not touch MLflow, and do not require real data.

The tests are designed to run quickly (< 2 seconds) and serve as a guardrail
in the CI pipeline: if the evaluation math changes, these tests fail before
any model training or registration occurs.

Arguments (pytest):
    None. Run with: pytest -q

Examples:
    python -m pytest -q
    python -m pytest tests/test_evaluate_model.py -q
    python -m pytest tests/test_evaluate_model.py::test_chronological_split_matches_training_arithmetic -q
"""

import numpy as np
import pytest
from types import SimpleNamespace

from src.evaluation.evaluate_model import (
    calculate_metrics,
    chronological_test_split,
    log_evaluation_to_mlflow,
    load_model,
    parse_args,
)


def test_persistence_skill_uses_zero_delta_baseline():
    """A perfect persistence model (zero delta prediction) should score 0% skill.

    Args:
        None (uses synthetic y_true and y_pred arrays).

    Returns:
        None. Asserts that skill == 0.0 and passed is False.
    """
    # A perfect persistence model has zero delta prediction and should score 0% skill.
    y_true = np.array([[2.0, -4.0], [1.0, 3.0]])
    y_pred = np.zeros_like(y_true)

    result = calculate_metrics(y_true, y_pred)

    assert result["persistence_mae"] == pytest.approx(2.5)
    assert result["model_mae"] == pytest.approx(2.5)
    assert result["skill"] == pytest.approx(0.0)
    assert result["model_beats_persistence"] is False
    assert result["passed"] is False


def test_model_with_25_percent_skill_passes():
    """A model with 25% skill should pass the evaluation gate (min_skill = 0.20).

    Args:
        None (uses synthetic y_true and y_pred arrays).

    Returns:
        None. Asserts model_mae=3.0, persistence_mae=4.0, skill=0.25, and passed=True.
    """
    y_true = np.array([[4.0, 4.0], [4.0, 4.0]])
    y_pred = np.array([[1.0, 1.0], [1.0, 1.0]])

    result = calculate_metrics(y_true, y_pred)

    assert result["model_mae"] == pytest.approx(3.0)
    assert result["persistence_mae"] == pytest.approx(4.0)
    assert result["skill"] == pytest.approx(0.25)
    assert result["model_beats_persistence"] is True
    assert result["skill_threshold_passed"] is True
    assert result["passed"] is True


def test_skill_below_threshold_fails():
    """A model with 10% skill (below the 0.20 threshold) should fail the gate.

    Args:
        None (uses synthetic y_true and y_pred arrays).

    Returns:
        None. Asserts skill=0.10, skill_threshold_passed=False, and passed=False.
    """
    y_true = np.full((5, 3), 10.0)
    y_pred = np.full((5, 3), 1.0)

    result = calculate_metrics(y_true, y_pred)

    assert result["model_mae"] == pytest.approx(9.0)
    assert result["persistence_mae"] == pytest.approx(10.0)
    assert result["skill"] == pytest.approx(0.10)
    assert result["model_beats_persistence"] is True
    assert result["skill_threshold_passed"] is False
    assert result["passed"] is False


def test_chronological_split_matches_training_arithmetic():
    """chronological_test_split should return exactly 15 test rows (last 15 of 100).

    Args:
        None (uses synthetic X, y, and datetimes arrays of length 100).

    Returns:
        None. Asserts test set has 15 rows and dates span indices 85..99.
    """
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100).reshape(100, 1)
    datetimes = np.arange(100)

    X_test, y_test, dates_test = chronological_test_split(X, y, datetimes)

    # train=70, validation=15, test=15
    assert len(X_test) == 15
    assert len(y_test) == 15
    assert dates_test[0] == 85
    assert dates_test[-1] == 99


def test_cli_accepts_registered_model_version_and_tracking_uri():
    """The CLI should accept a model version instead of requiring a full MLflow URI."""
    args = parse_args([
        "--model-version", "2",
        "--tracking-uri", "http://localhost:5000",
    ])

    assert args.model_version == "2"
    assert args.model_name == "xgboost_24compact"
    assert args.tracking_uri == "http://localhost:5000"


def test_load_model_uses_registered_model_name_and_version(monkeypatch, tmp_path):
    """A model name and version should form the MLflow registry URI."""
    loaded = object()
    calls = []

    class Pyfunc:
        @staticmethod
        def load_model(model_uri):
            calls.append(("load_model", model_uri))
            return loaded

    class Mlflow:
        pyfunc = Pyfunc()

        @staticmethod
        def set_tracking_uri(tracking_uri):
            calls.append(("set_tracking_uri", tracking_uri))

    monkeypatch.setitem(__import__("sys").modules, "mlflow", Mlflow)

    result = load_model(
        tmp_path / "does-not-exist.joblib",
        model_version="2",
        model_name="pm25-lag24-xgb",
        tracking_uri="http://localhost:5000",
    )

    assert result is loaded
    assert calls == [
        ("set_tracking_uri", "http://localhost:5000"),
        ("load_model", "models:/pm25-lag24-xgb/2"),
    ]


def test_evaluation_metrics_are_logged_to_registered_version_source_run(
    monkeypatch, tmp_path
):
    """Evaluation adds scores to the run that produced the selected registry version."""
    calls = []

    class Client:
        def __init__(self, tracking_uri):
            calls.append(("client", tracking_uri))

        def get_model_version(self, name, version):
            calls.append(("get_model_version", name, version))
            return type("Version", (), {"run_id": "source-run-123"})()

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(("set_tracking_uri", uri)),
        log_metrics=lambda metrics, run_id: calls.append(
            ("log_metrics", metrics, run_id)
        ),
        log_artifact=lambda path, artifact_path, run_id: calls.append(
            ("log_artifact", path, artifact_path, run_id)
        ),
        MlflowClient=Client,
    )
    monkeypatch.setitem(
        __import__("sys").modules, "mlflow", fake_mlflow
    )

    metrics = {
        "persistence_mae": 10.0,
        "skill": 0.25,
        "model_beats_persistence": True,
        "skill_threshold_passed": True,
    }
    report_path = tmp_path / "evaluation_report.json"
    log_evaluation_to_mlflow(
        metrics,
        report_path,
        model_name="pm25-lag24-xgb",
        model_version="2",
        tracking_uri="http://localhost:5000",
    )

    assert ("get_model_version", "pm25-lag24-xgb", "2") in calls
    metric_call = next(call for call in calls if call[0] == "log_metrics")
    assert metric_call[1]["test_skill_vs_persistence"] == 0.25
    assert metric_call[2] == "source-run-123"
    assert ("log_artifact", str(report_path), "evaluation", "source-run-123") in calls
