#!/usr/bin/env python3
"""Model-training stage: processed_data.csv -> XGBoost multi-horizon model.

Expected layout in the target repo:
    <root>/.env
    <root>/data/processed/processed_data.csv                (published by src/feature/feature_engineer.py)
    <root>/data/processed/processed_data_meta.json          (feature/target lists for training)
    <root>/models/xgb_lag24_final.joblib                    (published by this script)
    <root>/models/xgb_lag24_final_meta.json                 (model metadata for serving)
    <root>/models/xgb_lag24_best_params.json                (optional tuned hyperparameters)
    <root>/mlflow.db                                        (MLflow SQLite tracking store, default)

Trains ONE XGBoost model (MultiOutputRegressor wrapping XGBRegressor) that predicts all
24 horizons (target_delta_1 .. target_delta_24) in a single predict() call, so serving can
call model.predict(features) once and get all 24 hour-ahead deltas.

The feature/target contract is read from processed_data_meta.json (produced by
feature_engineer.py), NOT hardcoded. Training and serving therefore consume exactly the same
features in the same order, which prevents training/serving skew.

A time-based (no-shuffle) train/validation/test split is used because the data is a regular
hourly series; rows are never shuffled across time.

MLflow integration (experiment management):
    * mlflow.log_param()   -> all XGBoost hyperparameters + split ratios + seed + source
    * mlflow.log_metric()  -> MAE / RMSE / R2 for train / val / test
    * mlflow.log_model()   -> the trained MultiOutputRegressor (sklearn flavor)
    * mlflow.register_model() -> optional promotion to the Model Registry

Output is written atomically (.part -> replace) and is deterministic given the same input and
hyperparameters, so retraining is reproducible and DVC-friendly.

Arguments:
    --input                 feature CSV (default: <root>/data/processed/processed_data.csv)
    --meta                  feature meta JSON (default: <root>/data/processed/processed_data_meta.json)
    --output-dir            model output directory (default: <root>/models)
    --model-name            model artifact file name (default: xgb_lag24_final.joblib)
    --params                JSON file with tuned XGB hyperparameters (default: <root>/models/xgb_lag24_best_params.json)
    --params-key            top-level key inside --params to use (default: limited)
    --train-frac            fraction of rows used for training (default: 0.70)
    --val-frac              fraction of rows used for validation (default: 0.15; test gets the remainder)
    --seed                  random seed (default: 42)
    --n-estimators          XGBoost n_estimators override (default: from --params or DEFAULT_XGB_PARAMS)
    --max-depth             XGBoost max_depth override
    --learning-rate         XGBoost learning_rate override
    --subsample             XGBoost subsample override
    --colsample-bytree      XGBoost colsample_bytree override
    --min-child-weight      XGBoost min_child_weight override
    --reg-lambda            XGBoost reg_lambda override
    --no-mlflow             disable MLflow logging
    --experiment-name       MLflow experiment name (default: pm25-lag24)
    --tracking-uri          MLflow tracking URI (default: sqlite:///mlflow.db)
    --run-name              MLflow run name
    --register-model-name   register model to MLflow Model Registry under this name
    --no-overwrite          refuse to replace an existing model artifact
    --dry-run               validate and report without publishing
    --log-level             logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

Run parameter variations example:
    python src/train/train.py  --n-estimators 100 --max-depth 4
    python src/train/train.py  --n-estimators 500 --max-depth 6 --learning-rate 0.01
    python src/train/train.py  --n-estimators 1000 --max-depth 8 --learning-rate 0.05 --subsample 0.7

MLflow UI run :
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Other Examples:
    python src/train/train.py
    python src/train/train.py --experiment-name pm25-lag24-v2 --n-estimators 500 --max-depth 6
    python src/train/train.py --register-model-name pm25-lag24-xgb --dry-run
    python src/train/train.py --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - mlflow is optional at import time
    mlflow = None  # type: ignore
    _MLFLOW_AVAILABLE = False

# --------------------------------------------------------------------------- config
SNAPSHOT_FEATURES = "processed_data.csv"
SNAPSHOT_META = "processed_data_meta.json"
DEFAULT_INPUT = ("data", "processed", SNAPSHOT_FEATURES)
DEFAULT_META = ("data", "processed", SNAPSHOT_META)
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_MODEL_NAME = "xgb_lag24_final.joblib"
DEFAULT_PARAMS_FILE = ("models", "xgb_lag24_best_params.json")
DEFAULT_PARAMS_KEY = "limited"
DEFAULT_EXPERIMENT_NAME = "pm25-lag24"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test receives the remainder (~0.15)

DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

LOG = logging.getLogger("train")


class TrainError(RuntimeError):
    """Raised when training cannot proceed."""


@dataclass(frozen=True)
class Config:
    input_path: Path
    meta_path: Path
    output_dir: Path
    model_name: str
    params_file: Path | None
    params_key: str
    train_frac: float
    val_frac: float
    seed: int
    dry_run: bool
    overwrite: bool
    # MLflow
    no_mlflow: bool
    experiment_name: str
    tracking_uri: str | None
    run_name: str | None
    register_model_name: str | None
    _cli_params: dict


# --------------------------------------------------------------------------- paths
def find_project_root(start: Path) -> Path | None:
    """Walk up from *start* until a folder containing .env is found (same as feature_engineer)."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".env").is_file():
            return candidate
    return None


def resolve_project_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        root = find_project_root(start)
        if root is not None:
            return root
    return Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ load data
def load_meta(meta_path: Path) -> dict:
    if not meta_path.is_file():
        raise TrainError(f"feature meta not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    for key in ("features", "targets", "horizon"):
        if key not in meta:
            raise TrainError(f"{meta_path.name} is missing '{key}'")
    return meta


def load_xy(input_path: Path, features: list[str], targets: list[str]):
    if not input_path.is_file():
        raise TrainError(f"feature input not found: {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        raise TrainError(f"{input_path.name} contains no rows")

    missing = [c for c in [*features, *targets] if c not in df.columns]
    if missing:
        raise TrainError(f"{input_path.name} missing columns: {missing}")

    datetime_col = "datetime" if "datetime" in df.columns else df.columns[0]
    # numpy array -> all later indexing (incl. [0] / [-1]) is positional, not label-based.
    datetimes = pd.to_datetime(df[datetime_col], utc=True).to_numpy()

    X = df[features].to_numpy(dtype="float64")
    y = df[targets].to_numpy(dtype="float64")

    # Defensive: drop any row with NaN features/targets (feature_engineer already drops NaNs).
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    if not valid.all():
        dropped = int((~valid).sum())
        LOG.warning("dropping %s rows with NaN features/targets", dropped)
        X, y, datetimes = X[valid], y[valid], datetimes[valid]

    # Guarantee ascending temporal order (defensive; input should already be ordered).
    order = np.argsort(datetimes, kind="stable")
    return X[order], y[order], datetimes[order]


def time_series_split(X, y, datetimes, train_frac: float, val_frac: float):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    if n_train < 10 or n_val < 10 or n_test < 10:
        raise TrainError(
            f"not enough rows ({n}) for train_frac={train_frac}, val_frac={val_frac} "
            f"(train={n_train}, val={n_val}, test={n_test})"
        )
    slices = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }
    return {name: (X[s], y[s], datetimes[s]) for name, s in slices.items()}


# ------------------------------------------------------------------- modeling
def rmse(a, b) -> float:
    return float(mean_squared_error(a, b) ** 0.5)


def build_model(params: dict, seed: int) -> MultiOutputRegressor:
    xgb_params = dict(DEFAULT_XGB_PARAMS)
    xgb_params.update(params)
    xgb_params["random_state"] = seed
    base = XGBRegressor(**xgb_params)
    return MultiOutputRegressor(base)


def evaluate(model: MultiOutputRegressor, X, y, targets: list[str]) -> dict:
    """Overall metrics plus per-horizon MAE."""
    pred = model.predict(X)
    metrics = {
        "mae": float(mean_absolute_error(y, pred, multioutput="uniform_average")),
        "rmse": rmse(y, pred),
        "r2": float(r2_score(y, pred, multioutput="uniform_average")),
        "mae_per_horizon": [
            float(mean_absolute_error(y[:, i], pred[:, i])) for i in range(y.shape[1])
        ],
        "rmse_per_horizon": [
            float(rmse(y[:, i], pred[:, i])) for i in range(y.shape[1])
        ],
    }
    return metrics


def average_feature_importance(model: MultiOutputRegressor, features: list[str]):
    """Average gain-based importance across the per-horizon estimators."""
    importances = []
    for est in model.estimators_:
        fi = getattr(est, "feature_importances_", None)
        if fi is None:
            return None
        importances.append(np.asarray(fi, dtype="float64"))
    if not importances:
        return None
    avg = np.mean(importances, axis=0)
    order = np.argsort(-avg, kind="stable")
    return [(features[i], float(avg[i])) for i in order]


# ------------------------------------------------------------------ output IO
def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def load_best_params(params_file: Path | None, params_key: str) -> dict:
    if params_file is None or not params_file.is_file():
        return {}
    with params_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if params_key in data and isinstance(data[params_key], dict):
        return data[params_key]
    if isinstance(data, dict) and all(
        isinstance(v, (int, float, str, bool)) or v is None for v in data.values()
    ):
        return data
    return {}


# ---------------------------------------------------------------------- pipeline
def run(cfg: Config) -> int:
    meta = load_meta(cfg.meta_path)
    features = list(meta["features"])
    targets = list(meta["targets"])
    horizon = int(meta["horizon"])
    if len(targets) != horizon:
        raise TrainError(f"meta horizon={horizon} but {len(targets)} targets listed")

    X, y, datetimes = load_xy(cfg.input_path, features, targets)
    LOG.info(
        "loaded %s rows x %s features | %s targets | %s .. %s",
        len(X), len(features), horizon, datetimes[0], datetimes[-1],
    )

    splits = time_series_split(X, y, datetimes, cfg.train_frac, cfg.val_frac)
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, datetimes_test = splits["test"]
    LOG.info("split train=%s val=%s test=%s", len(X_train), len(X_val), len(X_test))

    if cfg.dry_run:
        LOG.info(
            "[dry-run] nothing written; would train on %s rows and evaluate on val + test",
            len(X_train),
        )
        return 0

    params = load_best_params(cfg.params_file, cfg.params_key)
    params.update(cfg._cli_params)
    LOG.info("xgb params: %s", params)
    model = build_model(params, cfg.seed)
    model.fit(X_train, y_train)

    metrics_train = evaluate(model, X_train, y_train, targets)
    metrics_val = evaluate(model, X_val, y_val, targets)
    metrics_test = evaluate(model, X_test, y_test, targets)
    LOG.info(
        "val  MAE=%.4f RMSE=%.4f R2=%.4f",
        metrics_val["mae"], metrics_val["rmse"], metrics_val["r2"],
    )
    LOG.info(
        "test MAE=%.4f RMSE=%.4f R2=%.4f",
        metrics_test["mae"], metrics_test["rmse"], metrics_test["r2"],
    )

    # ------------------------------------------------------------------ MLflow
    run_id = None
    if not cfg.no_mlflow:
        if not _MLFLOW_AVAILABLE:
            raise TrainError("mlflow is not installed; pip install mlflow or use --no-mlflow")
        if cfg.tracking_uri:
            mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run(run_name=cfg.run_name) as run:
            run_id = run.info.run_id
            LOG.info("mlflow run_id=%s | experiment=%s", run_id, cfg.experiment_name)

            # log parameters
            mlflow.log_param("source_file", cfg.input_path.name)
            mlflow.log_param("n_rows", int(len(X)))
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("n_targets", len(targets))
            mlflow.log_param("horizon", horizon)
            mlflow.log_param("train_frac", cfg.train_frac)
            mlflow.log_param("val_frac", cfg.val_frac)
            mlflow.log_param("seed", cfg.seed)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # log metrics (train / val / test)
            for split_name, m in (("train", metrics_train), ("val", metrics_val), ("test", metrics_test)):
                mlflow.log_metric(f"{split_name}_mae", m["mae"])
                mlflow.log_metric(f"{split_name}_rmse", m["rmse"])
                mlflow.log_metric(f"{split_name}_r2", m["r2"])

            # log the trained model artifact
            # XGBoost booster is not a native sklearn type, so mark it trusted for skops signing.
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"],
            )

            # optional model registry promotion
            if cfg.register_model_name:
                mlflow.register_model(f"runs:/{run_id}/model", cfg.register_model_name)
                LOG.info("mlflow model registered: %s", cfg.register_model_name)

    # ------------------------------------------------------------------ file outputs
    model_path = cfg.output_dir / cfg.model_name
    existed = model_path.exists()
    if existed and not cfg.overwrite:
        raise TrainError(f"{model_path} already exists and --no-overwrite is set")

    joblib.dump(model, model_path)
    LOG.info("%s model -> %s", "overwritten" if existed else "published", model_path)

    importance = average_feature_importance(model, features)
    if importance:
        fi_path = cfg.output_dir / (cfg.model_name.removesuffix(".joblib") + "_feature_importance.csv")
        write_csv_atomic(pd.DataFrame(importance, columns=["feature", "importance"]), fi_path)
        LOG.info("published feature importance -> %s", fi_path)
        if run_id is not None and _MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(fi_path), artifact_path="feature_importance",run_id=run_id)

    meta_path = cfg.output_dir / (cfg.model_name.removesuffix(".joblib") + "_meta.json")
    model_meta = {
        "model": cfg.model_name,
        "features": features,
        "targets": targets,
        "horizon": horizon,
        "params": params,
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
        "source_file": cfg.input_path.name,
        "n_rows": int(len(X)),
        "datetime_min": str(pd.Timestamp(datetimes[0])),
        "datetime_max": str(pd.Timestamp(datetimes[-1])),
        "test_datetime_min": str(pd.Timestamp(datetimes_test[0])),
        "test_datetime_max": str(pd.Timestamp(datetimes_test[-1])),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": run_id,
    }
    write_json_atomic(model_meta, meta_path)
    LOG.info("published model meta -> %s", meta_path)
    return 0


# --------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the compact PM2.5 XGBoost model from processed features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=None,
                        help="feature CSV (default: <root>/data/processed/processed_data.csv)")
    parser.add_argument("--meta", default=None,
                        help="feature meta JSON (default: <root>/data/processed/processed_data_meta.json)")
    parser.add_argument("--output-dir", default=None,
                        help="model output directory (default: <root>/models)")
    parser.add_argument("--model-name", default=None,
                        help="model artifact file name")
    parser.add_argument("--params", default=None,
                        help="JSON file with tuned XGB hyperparameters")
    parser.add_argument("--params-key", default=DEFAULT_PARAMS_KEY,
                        help="top-level key inside --params to use")
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_XGB_PARAMS["random_state"])
    # XGBoost hyperparameter overrides (for easy experiment variation)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument("--min-child-weight", type=int, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    # MLflow
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME,
                        help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                        help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--register-model-name", default=None,
                        help="register model to MLflow Model Registry under this name")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="refuse to replace an existing model artifact")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and report without publishing")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    root = resolve_project_root()

    input_path = Path(args.input).expanduser() if args.input else root.joinpath(*DEFAULT_INPUT)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    meta_path = Path(args.meta).expanduser() if args.meta else root.joinpath(*DEFAULT_META)
    if not meta_path.is_absolute():
        meta_path = Path.cwd() / meta_path

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else root / DEFAULT_OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    model_name = args.model_name or DEFAULT_MODEL_NAME

    params_file = Path(args.params).expanduser() if args.params else root.joinpath(*DEFAULT_PARAMS_FILE)
    if params_file is not None and not params_file.is_absolute():
        params_file = Path.cwd() / params_file
    if params_file is not None and not params_file.is_file():
        LOG.info("params file not found, using defaults: %s", params_file)
        params_file = None

    # CLI hyperparameter overrides -> params dict
    cli_params = {}
    if args.n_estimators is not None:
        cli_params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        cli_params["max_depth"] = args.max_depth
    if args.learning_rate is not None:
        cli_params["learning_rate"] = args.learning_rate
    if args.subsample is not None:
        cli_params["subsample"] = args.subsample
    if args.colsample_bytree is not None:
        cli_params["colsample_bytree"] = args.colsample_bytree
    if args.min_child_weight is not None:
        cli_params["min_child_weight"] = args.min_child_weight
    if args.reg_lambda is not None:
        cli_params["reg_lambda"] = args.reg_lambda

    return Config(
        input_path=input_path.resolve(),
        meta_path=meta_path.resolve(),
        output_dir=output_dir.resolve(),
        model_name=model_name,
        params_file=params_file,
        params_key=args.params_key,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=not args.no_overwrite,
        no_mlflow=args.no_mlflow,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        run_name=args.run_name,
        register_model_name=args.register_model_name,
        _cli_params=cli_params,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        cfg = build_config(args)
    except Exception as exc:
        LOG.error("configuration error: %s", exc)
        return 2

    LOG.info("input=%s | meta=%s | output_dir=%s", cfg.input_path, cfg.meta_path, cfg.output_dir)
    try:
        return run(cfg)
    except TrainError as exc:
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.warning("interrupted by user")
        return 130
    except Exception:
        LOG.exception("training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
