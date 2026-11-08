#!/usr/bin/env python3
"""Experimental model + hyperparameter search (Optuna) for the PM2.5 multi-horizon model.

Suggested location: <root>/src/train/hyperparameter_tune.py

Expected layout in the target repo (same as train.py):
    <root>/.env
    <root>/data/processed/processed_data.csv
    <root>/data/processed/processed_data_meta.json
    <root>/config/model_default_param.json                  (read only: used as the baseline)
    <root>/config/model_hyperparameter_tune.json            (WRITTEN by this script)
    <root>/models/model_hyperparameter.joblib               (WRITTEN by this script)

What is searched (one Optuna study, conditional search space)
    family = xgboost       wide space, optionally with the loss itself searched (L1 / pseudo-Huber / L2)
             extra_trees   ONE multi-output model for all 24 horizons (shared splits -> lower variance)
             random_forest same idea, bootstrap
             ridge         standardised linear model (strong when rows are scarce)
             elastic_net   standardised sparse linear model
    All families are scikit-learn compatible multi-output regressors, so the saved joblib is a
    drop-in replacement for model.joblib (model.predict(X) -> (n, 24)). Only sklearn / xgboost
    classes are pickled, so serving does not need this script.

How candidates are judged
    * Same data contract and time-ordered split as train.py. The last rows are the TEST set;
      the search never sees them. Selection uses CV only.
    * Score = mean MAE over all horizons, expanding-window time-series CV on the dev (train+val) rows.
    * Naive reference predictors are scored the same way (predict 0 change / predict train median).
      If a model cannot clearly beat those, the features carry little signal and no amount of
      tuning will fix that.
    * Sampler: multivariate TPE with 20 random start-up trials (handles interacting / conditional
      parameters better than plain TPE), or pure random search via --sampler random.
    * The first trial is your current baseline params (warm start).

Output JSON. The FIRST block is the best XGBoost params in the same key/format as
model_default_param.json, so it can be copy-pasted straight into that file (manually):
    {
      "limited":      { best XGBoost params, flat dict, floats rounded to 6 significant digits },
      "_how_to_use":  "...",
      "_summary":     { cv / test MAE: naive vs baseline vs this block, plus a plain-language verdict },
      "model_type":   "<winning family>",
      "tuned":        { params of the winning model (may be a non-XGBoost family) },
      "meta":         { cv scores, baseline vs tuned vs naive, top trials, search space, ... }
    }
    The top-level key is --params-key (default "limited"), so train.py can also read this file
    directly:  python src/train/train.py --params config/model_hyperparameter_tune.json
    Nothing is written into model_default_param.json automatically.

Arguments:
    --input / --meta        same defaults as train.py
    --output-dir            model output directory (default: <root>/models)
    --model-name            default: model_hyperparameter.joblib
    --config-out            default: <root>/config/model_hyperparameter_tune.json
    --params / --params-key baseline params file / key (default: config/model_default_param.json / limited)
    --families              comma list (default: xgboost,extra_trees,random_forest,ridge,elastic_net)
    --search-loss           also search the XGBoost objective (absoluteerror / pseudohuber / squarederror)
    --sampler               tpe | random (default: tpe)
    --n-trials              Optuna trials (default: 120)
    --timeout               stop the search after this many seconds (default: none)
    --cv-splits             expanding-window CV folds on the dev set (default: 3)
    --cv-gap                rows dropped between CV train and validation blocks (default: 0)
    --train-frac/--val-frac same meaning as train.py (default: 0.70 / 0.15)
    --loss                  default XGBoost objective (default: reg:absoluteerror)
    --seed                  random seed (default: 42)
    --log-level             DEBUG, INFO, WARNING, ERROR (default: INFO)

Examples:
    python src/train/hyperparameter_tune.py
    python src/train/hyperparameter_tune.py --n-trials 250 --timeout 1800 --search-loss
    python src/train/hyperparameter_tune.py --families extra_trees,ridge --n-trials 60
    python src/train/hyperparameter_tune.py --cv-splits 4 --cv-gap 24
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# --------------------------------------------------------------------------- config
SNAPSHOT_FEATURES = "processed_data.csv"
SNAPSHOT_META = "processed_data_meta.json"
DEFAULT_LOSS = "reg:absoluteerror"
DEFAULT_MODEL_NAME = "model_hyperparameter.joblib"
DEFAULT_CONFIG_NAME = "model_hyperparameter_tune.json"
DEFAULT_PARAMS_KEY = "limited"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
ALL_FAMILIES = ["xgboost", "extra_trees", "random_forest", "ridge", "elastic_net"]

FALLBACK_BASELINE = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
}

XGB_LOSS_CHOICES = ["reg:absoluteerror", "reg:pseudohubererror", "reg:squarederror"]

# Single source of truth: used to sample trials AND to document the search in the output JSON.
SEARCH_SPACES: dict[str, dict[str, dict]] = {
    "xgboost": {
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "log": True},
        "max_depth": {"type": "int", "low": 1, "high": 10},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.3, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.2, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 30},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
        "reg_alpha": {"type": "float", "low": 0.001, "high": 50.0, "log": True},
        "gamma": {"type": "float", "low": 0.0, "high": 10.0},
    },
    "extra_trees": {
        "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 24},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {"type": "float", "low": 0.2, "high": 1.0},
        "bootstrap": {"type": "cat", "choices": [True, False]},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 24},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {"type": "float", "low": 0.2, "high": 1.0},
        "max_samples": {"type": "float", "low": 0.4, "high": 1.0},
    },
    "ridge": {
        "alpha": {"type": "float", "low": 0.01, "high": 10000.0, "log": True},
    },
    "elastic_net": {
        "alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
        "l1_ratio": {"type": "float", "low": 0.05, "high": 1.0},
    },
}
XGB_HUBER_SLOPE = {"type": "float", "low": 1.0, "high": 30.0, "log": True}

LOG = logging.getLogger("tune")


class TuneError(RuntimeError):
    """Raised when tuning cannot proceed."""


# --------------------------------------------------------------------------- paths
def find_project_root(start: Path) -> Path | None:
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


# ------------------------------------------------------------------ data
def load_meta(meta_path: Path) -> dict:
    if not meta_path.is_file():
        raise TuneError(f"feature meta not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    for key in ("features", "targets", "horizon"):
        if key not in meta:
            raise TuneError(f"{meta_path.name} is missing '{key}'")
    return meta


def load_xy(input_path: Path, features: list[str], targets: list[str]):
    if not input_path.is_file():
        raise TuneError(f"feature input not found: {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        raise TuneError(f"{input_path.name} contains no rows")
    missing = [c for c in [*features, *targets] if c not in df.columns]
    if missing:
        raise TuneError(f"{input_path.name} missing columns: {missing}")

    datetime_col = "datetime" if "datetime" in df.columns else df.columns[0]
    datetimes = pd.to_datetime(df[datetime_col], utc=True).to_numpy()
    X = df[features].to_numpy(dtype="float64")
    y = df[targets].to_numpy(dtype="float64")

    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    if not valid.all():
        LOG.warning("dropping %s rows with NaN features/targets", int((~valid).sum()))
        X, y, datetimes = X[valid], y[valid], datetimes[valid]

    order = np.argsort(datetimes, kind="stable")
    return X[order], y[order], datetimes[order]


def split_dev_test(X, y, train_frac: float, val_frac: float):
    """Dev = train + val rows, test = the remainder. Identical boundaries to train.py."""
    n = len(X)
    n_dev = int(n * train_frac) + int(n * val_frac)
    n_test = n - n_dev
    if n_dev < 30 or n_test < 10:
        raise TuneError(
            f"not enough rows ({n}) for train_frac={train_frac}, val_frac={val_frac} "
            f"(dev={n_dev}, test={n_test})"
        )
    return X[:n_dev], y[:n_dev], X[n_dev:], y[n_dev:]


# ------------------------------------------------------------------ modeling
def build_model(family: str, params: dict, loss: str, seed: int):
    """Return an unfitted multi-output regressor: predict(X) -> (n_rows, n_targets)."""
    if family == "xgboost":
        p = {"objective": loss}
        p.update(params)
        p.update({"random_state": seed, "n_jobs": -1, "tree_method": "hist"})
        return MultiOutputRegressor(XGBRegressor(**p))
    if family == "extra_trees":
        return ExtraTreesRegressor(**params, random_state=seed, n_jobs=-1)
    if family == "random_forest":
        return RandomForestRegressor(**params, bootstrap=True, random_state=seed, n_jobs=-1)
    if family == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(**params))])
    if family == "elastic_net":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(**params, max_iter=10000, random_state=seed)),
        ])
    raise TuneError(f"unknown family: {family}")


def metrics_for(pred, y) -> dict:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "mae_per_horizon": [
            float(mean_absolute_error(y[:, i], pred[:, i])) for i in range(y.shape[1])
        ],
    }


def make_folds(n_dev: int, n_splits: int, gap: int):
    folds = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.arange(n_dev)))
    if not folds or min(len(tr) for tr, _ in folds) < 10:
        raise TuneError(
            f"cv-splits={n_splits}, cv-gap={gap} leaves fewer than 10 training rows in the "
            f"first fold ({n_dev} dev rows); lower --cv-splits or --cv-gap"
        )
    return folds


def _suggest(trial: optuna.Trial, family: str, spec: dict) -> dict:
    out = {}
    for name, s in spec.items():
        key = f"{family}.{name}"
        if s["type"] == "int":
            out[name] = trial.suggest_int(key, s["low"], s["high"], step=s.get("step", 1), log=s.get("log", False))
        elif s["type"] == "float":
            out[name] = trial.suggest_float(key, s["low"], s["high"], log=s.get("log", False))
        else:
            out[name] = trial.suggest_categorical(key, s["choices"])
    return out


def suggest_candidate(trial: optuna.Trial, families: list[str], search_loss: bool):
    family = trial.suggest_categorical("family", families)
    params = _suggest(trial, family, SEARCH_SPACES[family])
    if family == "xgboost" and search_loss:
        params["objective"] = trial.suggest_categorical("xgboost.objective", XGB_LOSS_CHOICES)
        if params["objective"] == "reg:pseudohubererror":
            s = XGB_HUBER_SLOPE
            params["huber_slope"] = trial.suggest_float("xgboost.huber_slope", s["low"], s["high"], log=True)
    return family, params


def cv_score(family, params, X, y, folds, loss, seed, trial: optuna.Trial | None = None):
    """Mean MAE across folds; reports per fold so the pruner can stop weak trials early."""
    maes = []
    for step, (tr, va) in enumerate(folds):
        model = build_model(family, params, loss, seed)
        model.fit(X[tr], y[tr])
        maes.append(float(mean_absolute_error(y[va], model.predict(X[va]))))
        if trial is not None:
            trial.report(float(np.mean(maes)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(maes)), maes


def naive_scores(X, y, folds, X_dev, y_dev, y_test) -> dict:
    """Reference predictors that use no features."""
    zero_cv, med_cv = [], []
    for tr, va in folds:
        zero_cv.append(float(np.mean(np.abs(y[va]))))
        med = np.median(y[tr], axis=0)
        med_cv.append(float(np.mean(np.abs(y[va] - med))))
    med_dev = np.median(y_dev, axis=0)
    return {
        "predict_zero_change": {"cv_mae": float(np.mean(zero_cv)), "test_mae": float(np.mean(np.abs(y_test)))},
        "predict_train_median": {"cv_mae": float(np.mean(med_cv)), "test_mae": float(np.mean(np.abs(y_test - med_dev)))},
    }


def load_baseline_params(params_file: Path | None, params_key: str) -> tuple[dict, str]:
    """Baseline = the XGBoost params train.py would use today (unknown keys such as 'loss' dropped)."""
    keep = set(SEARCH_SPACES["xgboost"]) | {"objective"}
    if params_file is not None and params_file.is_file():
        with params_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        block = data.get(params_key) if isinstance(data, dict) else None
        if isinstance(block, dict):
            return {k: v for k, v in block.items() if k in keep}, f"{params_file.name}:{params_key}"
    return dict(FALLBACK_BASELINE), "built-in defaults"


def enqueue_baseline(study: optuna.Study, baseline: dict, search_loss: bool) -> None:
    """Warm start: make trial 0 the current baseline (only keys that exist in the xgboost space)."""
    try:
        # Params missing from the baseline get neutral values (not random ones), so trial 0
        # replays the baseline as closely as the search space allows.
        neutral = {**FALLBACK_BASELINE, "gamma": 0.0, "reg_alpha": 0.001}
        params = {"family": "xgboost"}
        for name in SEARCH_SPACES["xgboost"]:
            value = baseline.get(name, neutral.get(name))
            if value is not None:
                params[f"xgboost.{name}"] = value
        if search_loss and baseline.get("objective") in XGB_LOSS_CHOICES:
            params["xgboost.objective"] = baseline["objective"]
        study.enqueue_trial(params, skip_if_exists=True)
    except Exception as exc:  # warm start is a nicety, never fatal
        LOG.warning("could not enqueue baseline as trial 0: %s", exc)


# ------------------------------------------------------------------ output IO
def _py(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    return value


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(_py(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def dump_joblib_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    joblib.dump(obj, temp)
    temp.replace(path)


# ---------------------------------------------------------------------- pipeline
def run(args: argparse.Namespace) -> int:
    root = resolve_project_root()
    input_path = Path(args.input).resolve() if args.input else root / "data" / "processed" / SNAPSHOT_FEATURES
    meta_path = Path(args.meta).resolve() if args.meta else root / "data" / "processed" / SNAPSHOT_META
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "models"
    config_out = Path(args.config_out).resolve() if args.config_out else root / "config" / DEFAULT_CONFIG_NAME
    params_file = Path(args.params).resolve() if args.params else root / "config" / "model_default_param.json"

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    bad = [f for f in families if f not in ALL_FAMILIES]
    if bad or not families:
        raise TuneError(f"unknown --families {bad}; choose from {ALL_FAMILIES}")

    LOG.info("input=%s | meta=%s", input_path, meta_path)
    meta = load_meta(meta_path)
    features, targets, horizon = list(meta["features"]), list(meta["targets"]), int(meta["horizon"])
    if len(targets) != horizon:
        raise TuneError(f"meta horizon={horizon} but {len(targets)} targets listed")

    X, y, datetimes = load_xy(input_path, features, targets)
    X_dev, y_dev, X_test, y_test = split_dev_test(X, y, args.train_frac, args.val_frac)
    LOG.info(
        "loaded %s rows x %s features | %s targets | dev=%s test=%s",
        len(X), len(features), horizon, len(X_dev), len(X_test),
    )

    folds = make_folds(len(X_dev), args.cv_splits, args.cv_gap)
    LOG.info(
        "cv: %s expanding folds | train sizes=%s | val sizes=%s | gap=%s",
        len(folds), [len(t) for t, _ in folds], [len(v) for _, v in folds], args.cv_gap,
    )

    # ---- reference points: naive predictors and the current baseline
    naive = naive_scores(X_dev, y_dev, folds, X_dev, y_dev, y_test)
    for name, s in naive.items():
        LOG.info("naive %-22s cv MAE=%.4f | test MAE=%.4f", name, s["cv_mae"], s["test_mae"])

    baseline_params, baseline_source = load_baseline_params(params_file, args.params_key)
    baseline_cv, baseline_folds = cv_score("xgboost", baseline_params, X_dev, y_dev, folds, args.loss, args.seed)
    LOG.info("baseline xgboost (%s) cv MAE=%.4f folds=%s", baseline_source, baseline_cv, [round(m, 3) for m in baseline_folds])

    # ---- search
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(
            seed=args.seed, multivariate=True, group=True, n_startup_trials=20
        )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name="pm25-model-search",
    )
    if "xgboost" in families:
        enqueue_baseline(study, baseline_params, args.search_loss)

    def objective(trial: optuna.Trial) -> float:
        family, params = suggest_candidate(trial, families, args.search_loss)
        trial.set_user_attr("family", family)
        trial.set_user_attr("model_params", params)
        score, fold_mae = cv_score(family, params, X_dev, y_dev, folds, args.loss, args.seed, trial)
        trial.set_user_attr("fold_mae", fold_mae)
        return score

    def log_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        fam = trial.user_attrs.get("family", "?")
        if trial.state == optuna.trial.TrialState.COMPLETE:
            LOG.info("trial %3d  %-13s cv MAE=%.4f  best=%.4f", trial.number, fam, trial.value, study_.best_value)
        elif trial.state == optuna.trial.TrialState.PRUNED:
            LOG.info("trial %3d  %-13s pruned", trial.number, fam)

    started = time.time()
    LOG.info("starting search: families=%s n_trials=%s timeout=%s sampler=%s", families, args.n_trials, args.timeout, args.sampler)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[log_trial], gc_after_trial=True)
    elapsed = time.time() - started

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if not completed:
        raise TuneError("no trial completed; increase --n-trials / --timeout")
    best = study.best_trial
    best_family = best.user_attrs["family"]
    best_params = dict(best.user_attrs["model_params"])
    LOG.info(
        "search done in %.1fs | %s complete, %s pruned | best cv MAE=%.4f (%s) vs baseline %.4f",
        elapsed, len(completed), len(pruned), study.best_value, best_family, baseline_cv,
    )

    ranked = sorted(completed, key=lambda t: t.value)
    top = [
        {"trial": t.number, "family": t.user_attrs["family"], "cv_mae": t.value, "params": t.user_attrs["model_params"]}
        for t in ranked[:10]
    ]
    best_per_family = {}
    for t in ranked:
        best_per_family.setdefault(t.user_attrs["family"], t.value)
    for fam, val in best_per_family.items():
        LOG.info("best cv MAE for %-13s %.4f", fam, val)

    # ---- refit on the whole dev set, evaluate once on the untouched test set
    LOG.info("refitting winner (%s) on dev set (%s rows)", best_family, len(X_dev))
    best_model = build_model(best_family, best_params, args.loss, args.seed)
    best_model.fit(X_dev, y_dev)
    best_dev = metrics_for(best_model.predict(X_dev), y_dev)
    best_test = metrics_for(best_model.predict(X_test), y_test)

    base_model = build_model("xgboost", baseline_params, args.loss, args.seed)
    base_model.fit(X_dev, y_dev)
    base_test = metrics_for(base_model.predict(X_test), y_test)

    LOG.info("tuned    test MAE=%.4f RMSE=%.4f R2=%.4f (%s)", best_test["mae"], best_test["rmse"], best_test["r2"], best_family)
    LOG.info("baseline test MAE=%.4f RMSE=%.4f R2=%.4f (xgboost)", base_test["mae"], base_test["rmse"], base_test["r2"])
    best_naive = min(naive.values(), key=lambda s: s["test_mae"])
    if best_test["mae"] >= best_naive["test_mae"]:
        LOG.warning(
            "the tuned model does NOT beat the best feature-free predictor on test (%.4f vs %.4f): "
            "the features carry little usable signal for these targets",
            best_test["mae"], best_naive["test_mae"],
        )
    if best_test["mae"] > base_test["mae"]:
        LOG.warning(
            "tuned model is WORSE than the baseline on the held-out test set (%.4f vs %.4f); "
            "CV gains did not carry over (small-sample noise / CV overfitting)",
            best_test["mae"], base_test["mae"],
        )
    LOG.info(
        "note: test has only %s rows; differences of a few tenths of MAE are within noise. "
        "Do not keep re-running until test MAE hits a target, or test stops being a fair estimate.",
        len(X_test),
    )

    # ---- best XGBoost trial -> flat dict that goes FIRST in the JSON (copy-paste into the defaults)
    xgb_trials = [t for t in ranked if t.user_attrs["family"] == "xgboost"]
    paste_block = None
    xgb_cv = xgb_test_mae = None
    if xgb_trials:
        xgb_params = dict(xgb_trials[0].user_attrs["model_params"])
        xgb_cv = xgb_trials[0].value
        if best_family == "xgboost":
            xgb_test_mae = best_test["mae"]
        else:  # the winner is another family; still evaluate the XGBoost block that gets pasted
            xgb_model = build_model("xgboost", xgb_params, args.loss, args.seed)
            xgb_model.fit(X_dev, y_dev)
            xgb_test_mae = metrics_for(xgb_model.predict(X_test), y_test)["mae"]
        paste_block = {"objective": args.loss, **xgb_params}
        paste_block.update({"random_state": args.seed, "n_jobs": -1, "tree_method": "hist"})
        # round floats to 6 significant digits so the block is readable when pasted
        paste_block = {
            k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in paste_block.items()
        }

    summary = {
        "cv_mae": {
            "naive_best": min(s["cv_mae"] for s in naive.values()),
            "baseline": baseline_cv,
            "this_block": xgb_cv,
        },
        "test_mae": {
            "naive_best": best_naive["test_mae"],
            "baseline": base_test["mae"],
            "this_block": xgb_test_mae,
        },
        "test_rows": int(len(X_test)),
    }
    if xgb_cv is None:
        summary["verdict"] = "no XGBoost trial was run (see --families); nothing to paste into the defaults"
    elif xgb_test_mae >= base_test["mae"]:
        summary["verdict"] = (
            "NO reliable gain: this block is not better than the current baseline on the held-out test "
            f"({xgb_test_mae:.3f} vs {base_test['mae']:.3f}); keep the current defaults unless you have a reason"
        )
    else:
        summary["verdict"] = (
            f"better than baseline on CV ({xgb_cv:.3f} vs {baseline_cv:.3f}) and test "
            f"({xgb_test_mae:.3f} vs {base_test['mae']:.3f}), but test has only {len(X_test)} rows: treat as tentative"
        )

    payload = {}
    if paste_block is not None:
        payload[args.params_key] = paste_block   # same top-level key/format as model_default_param.json
        payload["_how_to_use"] = (
            f"Copy the '{args.params_key}' block above into config/model_default_param.json "
            f"(replace the existing '{args.params_key}' value). train.py reads it as-is."
        )
        if paste_block.get("objective") != args.loss:
            payload["_note"] = "objective differs from --loss; train.py overrides it with --loss, pass --loss to match"
    payload["_summary"] = summary
    payload.update({
        "model_type": best_family,
        "tuned": {**best_params, **({"random_state": args.seed} if best_family != "ridge" else {})},
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "optuna_version": optuna.__version__,
            "sampler": args.sampler + (" (multivariate, group, 20 startup)" if args.sampler == "tpe" else ""),
            "pruner": "MedianPruner",
            "families": families,
            "search_loss": args.search_loss,
            "objective_metric": "mean MAE over all horizons (expanding-window time-series CV)",
            "n_trials_requested": args.n_trials,
            "n_trials_complete": len(completed),
            "n_trials_pruned": len(pruned),
            "timeout_seconds": args.timeout,
            "elapsed_seconds": round(elapsed, 1),
            "seed": args.seed,
            "cv": {
                "splits": len(folds),
                "gap": args.cv_gap,
                "train_sizes": [len(t) for t, _ in folds],
                "val_sizes": [len(v) for _, v in folds],
            },
            "data": {
                "input": input_path.name,
                "rows": int(len(X)),
                "dev_rows": int(len(X_dev)),
                "test_rows": int(len(X_test)),
                "n_features": len(features),
                "n_targets": horizon,
                "start": str(datetimes[0]),
                "end": str(datetimes[-1]),
            },
            "best_cv_mae": study.best_value,
            "best_cv_fold_mae": best.user_attrs.get("fold_mae"),
            "best_cv_mae_per_family": best_per_family,
            "naive": naive,
            "baseline": {
                "family": "xgboost",
                "source": baseline_source,
                "params": baseline_params,
                "cv_mae": baseline_cv,
                "test": {k: base_test[k] for k in ("mae", "rmse", "r2")},
            },
            "winner": {
                "family": best_family,
                "dev": {k: best_dev[k] for k in ("mae", "rmse", "r2")},
                "test": {k: best_test[k] for k in ("mae", "rmse", "r2")},
                "test_mae_per_horizon": best_test["mae_per_horizon"],
            },
            "top_trials": top,
            "search_space": {f: SEARCH_SPACES[f] for f in families},
            "model_file": args.model_name,
        },
    })

    model_path = output_dir / args.model_name
    dump_joblib_atomic(best_model, model_path)
    write_json_atomic(payload, config_out)
    LOG.info("wrote model  -> %s", model_path)
    LOG.info("wrote config -> %s", config_out)
    return 0


# ---------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental model + hyperparameter search for the PM2.5 model")
    p.add_argument("--input", default=None, help="feature CSV (default: <root>/data/processed/processed_data.csv)")
    p.add_argument("--meta", default=None, help="feature meta JSON (default: <root>/data/processed/processed_data_meta.json)")
    p.add_argument("--output-dir", default=None, help="model output directory (default: <root>/models)")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"default: {DEFAULT_MODEL_NAME}")
    p.add_argument("--config-out", default=None, help=f"default: <root>/config/{DEFAULT_CONFIG_NAME}")
    p.add_argument("--params", default=None, help="baseline params JSON (default: <root>/config/model_default_param.json)")
    p.add_argument("--params-key", default=DEFAULT_PARAMS_KEY, help="key inside --params (default: limited)")
    p.add_argument("--families", default="xgboost", help=f"comma list from {ALL_FAMILIES}")
    p.add_argument("--search-loss", action="store_true", help="also search the XGBoost objective")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--n-trials", type=int, default=120, help="Optuna trials (default: 120)")
    p.add_argument("--timeout", type=float, default=None, help="stop search after N seconds")
    p.add_argument("--cv-splits", type=int, default=3, help="expanding-window CV folds (default: 3)")
    p.add_argument("--cv-gap", type=int, default=0, help="rows skipped between CV train and val blocks (default: 0)")
    p.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--loss", default=DEFAULT_LOSS, help=f"default XGBoost objective (default: {DEFAULT_LOSS})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        return run(args)
    except TuneError as exc:
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.warning("interrupted by user")
        return 130
    except Exception:
        LOG.exception("tuning failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
