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

from src.evaluation.evaluate_model import (
    calculate_metrics,
    chronological_test_split,
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
