#!/usr/bin/env python3
"""
========================================================================
PM2.5 Prediction Model Inference Script
========================================================================

Formal inference script for the trained XGBoost PM2.5 prediction model.

Usage
-----
Run directly from terminal to output results to terminal:

    python inference.py

Or with custom input:

    python inference.py --input data/new_measurements.csv

Or save results to file:

    python inference.py --output predictions.txt

Model Loading Priority
--------------------
1. MLflow model registry (models:/model_name/Production)
2. Joblib file fallback (src/../models/model.joblib) with warning

Features
--------
- 24-hour ahead predictions (target_delta_1 through target_delta_24)
- Feature validation against model metadata
- Human-readable terminal output
- Optional file output
- Consistent with training pipeline features

========================================================================
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow

# ---------------------------------------------------------------------------
# Path configuration
# From src/inference/inference.py -> go up 3 levels to project root
#   __file__          : src/inference/inference.py
#   .parent           : src/inference/
#   .parent.parent    : src/
#   .parent.parent.parent : project root (contains models/, data/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "model_meta.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"


def load_model_mlflow(model_name: str = "xgboost_24compact") -> object:
    """
    Load model from MLflow model registry.

    Parameters
    ----------
    model_name : str
        Name of the model in MLflow registry (default: "xgboost_24compact")

    Returns
    -------
    object
        Loaded MLflow model, or None if MLflow loading fails
    """
    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        print(f"Loaded model from MLflow registry: {model_uri}", file=sys.stdout)
        return model
    except Exception as e:
        print(
            f"Warning: Could not load from MLflow registry ({e}), "
            "falling back to joblib",
            file=sys.stderr,
        )
        return None


def load_model_joblib(model_path: Path) -> object:
    """
    Load model from joblib file (fallback method).

    Parameters
    ----------
    model_path : Path
        Path to the model.joblib file

    Returns
    -------
    object
        Loaded model via joblib

    Raises
    ------
    FileNotFoundError
        If model file does not exist at the specified path
    """
    if not model_path.is_file():
        print(
            f"Error: Model file not found at {model_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    model = joblib.load(model_path)
    print(
        f"Loaded model from joblib fallback: {model_path}", file=sys.stdout,
    )
    return model


def load_meta(meta_path: Path) -> dict:
    """
    Load model metadata containing feature and target names.

    Parameters
    ----------
    meta_path : Path
        Path to model_meta.json

    Returns
    -------
    dict
        Metadata dictionary with "features", "targets", "horizon", and "params"
    """
    if not meta_path.is_file():
        print(f"Error: Meta file not found at {meta_path}", file=sys.stderr)
        sys.exit(1)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta


def load_data(input_path: Path, meta: dict) -> pd.DataFrame:
    """
    Load and validate input data for inference.

    Reads the input CSV, parses the datetime column, and selects only the
    feature columns that the model was trained on (as specified in meta).

    Parameters
    ----------
    input_path : Path
        Path to input CSV file
    meta : dict
        Model metadata containing the expected "features" list

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the required features, in the correct order

    Raises
    ------
    SystemExit
        If the input file is missing or required features are absent
    """
    if not input_path.is_file():
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path, parse_dates=["datetime"])

    # Select only the feature columns that the model expects
    feature_cols = meta["features"]

    # Check if all required features are present in the input data
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(
            f"Error: Missing feature(s) in input data: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use only the features the model was trained on, in the correct order
    df = df[feature_cols]

    return df


def run_inference(model, df: pd.DataFrame) -> np.ndarray:
    """
    Run inference using the loaded model.

    Parameters
    ----------
    model : object
        Trained model with a .predict() method (e.g., XGBoost / sklearn)
    df : pd.DataFrame
        Input features for prediction

    Returns
    -------
    np.ndarray
        Predictions array, shape (1, 24) for a single row input
    """
    predictions = model.predict(df)
    return predictions
def format_output(results: dict, target_cols: list) -> str:
    """
    Format prediction results as a human-readable string.

    Parameters
    ----------
    results : dict
        Dictionary mapping target names to predicted values
    target_cols : list
        List of target column names (e.g., ['target_delta_1', ..., 'target_delta_24'])

    Returns
    -------
    str
        Formatted output string suitable for terminal or file writing
    """
    lines = [
        "PM2.5 Prediction Results",
        "=" * 40,
        f"Features: 1 row",
        f"Horizon: 24 hour-ahead predictions",
        "",
        "Predicted target deltas (target_delta_1 through target_delta_24):",
    ]

    for target in target_cols:
        value = results.get(target, 0.0)
        lines.append(f"  {target}: {value:.4f}")

    lines.append("")
    lines.append("=" * 40)

    return "\n".join(lines)

def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(
        description="PM2.5 Prediction Model Inference",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV file "
        "(default: uses last row from processed_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file (results written to file instead of terminal)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_24compact",
        help="MLflow model name for registry lookup (default: xgboost_24compact)",
    )
    args = parser.parse_args()

    # Load model: try MLflow first, then joblib fallback
    model = load_model_mlflow(args.model_name)
    if model is None:
        model = load_model_joblib(DEFAULT_MODEL_PATH)

    # Load model metadata
    meta = load_meta(DEFAULT_META_PATH)

    # Determine input data
    if args.input:
        input_path = Path(args.input).expanduser()
        # Resolve relative paths relative to project root
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        df = load_data(input_path, meta)
    else:
        # Default: use last row from the processed training data
        print(
            "No input file provided, using last row from processed_data.csv...",
            file=sys.stdout,
        )
        df = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["datetime"])
        feature_cols = meta["features"]
        df = df[feature_cols].iloc[[-1]]  # Select last row as single sample

    # Run inference
    predictions = run_inference(model, df)

    # Format and output results
    target_cols = meta["targets"]
    results = {}
    for i, target in enumerate(target_cols):
        results[target] = float(predictions[0][i])

    output_text = format_output(results, target_cols)

    if args.output:
        output_path = Path(args.output)
        # Resolve output path relative to project root if relative
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        with open(output_path, "w") as f:
            f.write(output_text + "\n")
        print(f"Results written to {output_path}", file=sys.stdout)
    else:
        print(output_text, file=sys.stdout)


if __name__ == "__main__":
    main()
