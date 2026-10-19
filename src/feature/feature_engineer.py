#!/usr/bin/env python3
"""Feature-engineering stage (v1, compact set): cleaned snapshot -> model-ready features.

Expected layout in the target repo:
    <root>/.env
    <root>/data/preprocessed/cleandata_YYYYmmdd_HHMMSS.csv  (published by src/data/preprocess.py)
    <root>/data/processed/processed_data.csv                (published by this script)
    <root>/data/processed/processed_data_meta.json          (feature/target lists for training)
    <root>/data/processed/processed_data_YYYYmmdd_HHMMSS.csv (optional timestamped copy, --double / --timestamp-only)
    <root>/data/processed/processed_data_YYYYmmdd_HHMMSS_meta.json (meta for the timestamped copy)

The project root is discovered the same way the preprocessing stage does it.

Input: the long-format cleaned snapshot (sensor_id, parameter, units, datetime_utc,
datetime_local, value, ...). A wide hourly CSV (datetime, pm25, relativehumidity,
temperature) is also accepted. Long format is reduced to one row per hour:
  * only pm25 / relativehumidity / temperature are used
  * a parameter reported in several units keeps its most frequent unit (warning logged)
  * several readings in the same hour (extra sensors / sub-hourly) are averaged

Output: 22 features + 24 targets (target_delta_1 .. target_delta_24), one row per hour.
All lags/rolling windows are computed on a full hourly grid, so they are based on TIME and
a missing hour never shifts a lag. Rows whose windows touch a gap are dropped (and counted).

Output modes (mutually exclusive):
    * default (no flag): overwrite only `processed_data.csv` (+ `processed_data_meta.json`).
    * --double: overwrite `processed_data.csv` AND write a timestamped copy
      `processed_data_YYYYmmdd_HHMMSS.csv` (+ its meta).
    * --timestamp-only: write ONLY the timestamped copy
      `processed_data_YYYYmmdd_HHMMSS.csv` (+ its meta); `processed_data.csv` is untouched.

Overwrite policy: the main output `processed_data.csv` is ALWAYS overwritten (DVC removes
`outs` before it runs a stage anyway). Writes are atomic (.part -> replace) and deterministic
(no timestamps inside the files), so the same input yields byte-identical output and a stable
DVC hash. Use --no-overwrite to refuse replacing an existing `processed_data.csv`.
Timestamped copies always carry a unique timestamp, so they never collide.

ASSUMPTIONS to confirm (the lag24 notebook definitions were not visible when this was written):
    hour_sin / hour_cos    = sin/cos(2*pi*hour/24), hour taken in --hour-tz (default UTC)
    velocity_now           = pm25(t) - pm25(t-1)
    velocity_1h            = pm25(t-1) - pm25(t-2)
    pm25_rolling_std_6h    = std of the last 6 hours incl. now (ddof=1)
    target_delta_h         = pm25(t+h) - pm25(t)

Dependencies:
    pip install "pandas>=3.0" numpy

Examples:
    python src/feature/feature_engineer.py
    python src/feature/feature_engineer.py --input data/preprocessed/cleandata_20261019_120000.csv --dry-run
    python src/feature/feature_engineer.py --double
    python src/feature/feature_engineer.py --timestamp-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- config
SNAPSHOT_GLOB = "cleandata_*.csv"
DEFAULT_INPUT_DIR = ("data", "preprocessed")
DEFAULT_OUTPUT = ("data", "processed", "processed_data.csv")

# long-format columns (preprocess output)
TIMESTAMP_COLUMN = "datetime_utc"
PARAMETER_COLUMN = "parameter"
VALUE_COLUMN = "value"
UNITS_COLUMN = "units"

# wide / feature-frame columns
DATETIME_COL = "datetime"
RAW_COLS = ["pm25", "relativehumidity", "temperature"]

HORIZON = 24
SPARSE_LAGS = (1, 2, 3, 6, 12, 24)
MEAN_WINDOWS = (3, 6, 12, 24)

# Same order as `compact` in the training notebook (the final model was fitted in this order).
FEATURES = [
    "pm25", "relativehumidity", "temperature", "hour_sin", "hour_cos",
    "velocity_now", "velocity_1h", "pm25_rolling_std_6h",
    *[f"pm25_lag_{lag}" for lag in SPARSE_LAGS],
    *[f"pm25_mean_{n}h" for n in MEAN_WINDOWS],
    "pm25_max_6h", "pm25_max_24h", "pm25_min_24h", "pm25_anom_6h",
]
TARGETS = [f"target_delta_{h}" for h in range(1, HORIZON + 1)]

LOG = logging.getLogger("feature_engineer")


class FeatureError(RuntimeError):
    """Raised when the input cannot be turned into a feature snapshot."""


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    hour_tz: str = "UTC"
    overwrite: bool = True
    dry_run: bool = False
    double: bool = False
    timestamp_only: bool = False
    timestamp: str = ""


# ------------------------------------------------------------------------- paths
def find_project_root(start: Path) -> Path | None:
    """Walk up from *start* until a folder containing .env is found."""
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


def latest_snapshot(directory: Path) -> Path | None:
    """Newest cleandata_*.csv; the fixed-width name makes name order == time order."""
    if not directory.is_dir():
        return None
    snapshots = sorted(directory.glob(SNAPSHOT_GLOB), key=lambda path: path.name)
    return snapshots[-1] if snapshots else None


# ------------------------------------------------------------------------- input
def drop_minority_units(df: pd.DataFrame) -> pd.DataFrame:
    """If a parameter has several units, keep only its most frequent unit."""
    if UNITS_COLUMN not in df.columns:
        return df
    keep_mask = pd.Series(True, index=df.index)
    for parameter, group in df.groupby(PARAMETER_COLUMN):
        counts = group[UNITS_COLUMN].astype("string").fillna("unknown").value_counts()
        if len(counts) > 1:
            keep = counts.idxmax()
            LOG.warning("%s reported in several units %s; keeping %r only",
                        parameter, counts.to_dict(), keep)
            keep_mask &= ~((df[PARAMETER_COLUMN] == parameter)
                           & (df[UNITS_COLUMN].astype("string").fillna("unknown") != keep))
    return df[keep_mask]


def long_to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    """Cleaned long snapshot -> one row per hour with pm25 / relativehumidity / temperature."""
    df = frame.copy()
    df[PARAMETER_COLUMN] = df[PARAMETER_COLUMN].astype("string").str.strip()
    df = df[df[PARAMETER_COLUMN].isin(RAW_COLS)]

    missing = [p for p in RAW_COLS if p not in set(df[PARAMETER_COLUMN])]
    if missing:
        raise FeatureError(f"parameter(s) {missing} not present in the input snapshot")

    df = drop_minority_units(df)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True).dt.floor("h")
    df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors="coerce")

    multi = int((df.groupby([TIMESTAMP_COLUMN, PARAMETER_COLUMN]).size() > 1).sum())
    if multi:
        LOG.info("%s (hour, parameter) cells had several readings; averaged", multi)

    wide = df.pivot_table(index=TIMESTAMP_COLUMN, columns=PARAMETER_COLUMN,
                          values=VALUE_COLUMN, aggfunc="mean")
    wide = wide[RAW_COLS].reset_index().rename(columns={TIMESTAMP_COLUMN: DATETIME_COL})
    wide.columns.name = None
    return wide


def load_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FeatureError(f"input not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise FeatureError(f"{path.name} contains no rows")

    if {TIMESTAMP_COLUMN, PARAMETER_COLUMN, VALUE_COLUMN}.issubset(frame.columns):
        LOG.info("input %s: long format, %s rows", path.name, len(frame))
        return long_to_wide(frame)
    if {DATETIME_COL, *RAW_COLS}.issubset(frame.columns):
        LOG.info("input %s: wide hourly format, %s rows", path.name, len(frame))
        return frame[[DATETIME_COL, *RAW_COLS]]
    raise FeatureError(
        f"{path.name} is neither the long snapshot ({TIMESTAMP_COLUMN}, {PARAMETER_COLUMN}, "
        f"{VALUE_COLUMN}) nor a wide hourly file ({DATETIME_COL}, {RAW_COLS}); "
        f"columns found: {list(frame.columns)}"
    )


# ----------------------------------------------------------------------- features
def build_features(raw: pd.DataFrame, hour_tz: str = "UTC", with_targets: bool = True) -> pd.DataFrame:
    """Build the compact feature set from an hourly wide frame.

    with_targets=False is for serving: no targets, so the newest rows are kept.
    Use this same function for training and serving to avoid training/serving skew.
    """
    df = raw[[DATETIME_COL, *RAW_COLS]].copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], utc=True).dt.floor("h")

    n_dup = int(df[DATETIME_COL].duplicated().sum())
    if n_dup:
        LOG.warning("%s duplicate hours; keeping the last row of each", n_dup)
    df = df.drop_duplicates(DATETIME_COL, keep="last").sort_values(DATETIME_COL)

    # Full hourly grid -> shifts and rolling windows are TIME based; missing hours become NaN
    grid = pd.date_range(df[DATETIME_COL].min(), df[DATETIME_COL].max(), freq="h")
    s = df.set_index(DATETIME_COL).reindex(grid)
    s.index.name = DATETIME_COL
    pm = s["pm25"]
    LOG.info("hourly grid: %s hours | pm25 missing: %s", len(s), int(pm.isna().sum()))

    out = pd.DataFrame(index=s.index)
    out["pm25"] = pm
    out["relativehumidity"] = s["relativehumidity"]
    out["temperature"] = s["temperature"]

    hour = s.index.tz_convert(hour_tz).hour.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    out["velocity_now"] = pm - pm.shift(1)
    out["velocity_1h"] = pm.shift(1) - pm.shift(2)
    out["pm25_rolling_std_6h"] = pm.rolling(6, min_periods=6).std()

    for lag in SPARSE_LAGS:
        out[f"pm25_lag_{lag}"] = pm.shift(lag)
    for n in MEAN_WINDOWS:
        out[f"pm25_mean_{n}h"] = pm.rolling(n, min_periods=n).mean()
    out["pm25_max_6h"] = pm.rolling(6, min_periods=6).max()
    out["pm25_max_24h"] = pm.rolling(24, min_periods=24).max()
    out["pm25_min_24h"] = pm.rolling(24, min_periods=24).min()
    out["pm25_anom_6h"] = pm - out["pm25_mean_6h"]

    columns = list(FEATURES)
    if with_targets:
        for h in range(1, HORIZON + 1):
            out[f"target_delta_{h}"] = pm.shift(-h) - pm
        columns += TARGETS

    out = out[columns]
    before = len(out)
    out = out.dropna().reset_index()
    LOG.info("rows: %s -> %s after dropping NaN (first 24h, last 24h, windows touching gaps)",
             before, len(out))
    return out



# --------------------------------------------------------------------- output IO
def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".part")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".part")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def meta_payload(cfg: Config, frame: pd.DataFrame) -> dict[str, object]:
    """Deterministic metadata (no run timestamps) so the DVC hash only changes with the data."""
    return {
        "features": FEATURES,
        "targets": TARGETS,
        "horizon": HORIZON,
        "hour_tz": cfg.hour_tz,
        "source_file": cfg.input_path.name,
        "n_rows": int(len(frame)),
        "datetime_min": str(frame[DATETIME_COL].min()),
        "datetime_max": str(frame[DATETIME_COL].max()),
    }


# ---------------------------------------------------------------------- pipeline
def run(cfg: Config) -> int:
    wide = load_input(cfg.input_path)
    feats = build_features(wide, hour_tz=cfg.hour_tz, with_targets=True)
    if feats.empty:
        raise FeatureError("no rows left after feature building; need continuous hourly data "
                           "with at least 24h of history and 24h of lookahead")


    stamp = cfg.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ts_path = cfg.output_path.with_name(
        f"{cfg.output_path.stem}_{stamp}{cfg.output_path.suffix}"
    )
    ts_meta_path = ts_path.with_name(ts_path.stem + "_meta.json")
    main_meta_path = cfg.output_path.with_name(cfg.output_path.stem + "_meta.json")

    if cfg.dry_run:
        if cfg.timestamp_only:
            targets = [ts_path]
        elif cfg.double:
            targets = [cfg.output_path, ts_path]
        else:
            targets = [cfg.output_path]
        LOG.info(
            "[dry-run] nothing written; would have published %s (%s rows x %s columns)",
            " & ".join(str(p) for p in targets), len(feats), feats.shape[1],
        )
        return 0

    # Default / --double: write (or refuse) the main processed_data.csv.
    write_main = not cfg.timestamp_only
    if write_main:
        existed = cfg.output_path.exists()
        if existed and not cfg.overwrite:
            raise FeatureError(f"{cfg.output_path} already exists and --no-overwrite is set")
        write_csv_atomic(feats, cfg.output_path)
        LOG.info("%s: %s (%s rows x %s columns)",
                 "overwritten" if existed else "published",
                 cfg.output_path, len(feats), feats.shape[1])
        write_json_atomic(meta_payload(cfg, feats), main_meta_path)
        LOG.info("published: %s", main_meta_path)

    # --double / --timestamp-only: also (or only) write the timestamped copy.
    if cfg.double or cfg.timestamp_only:
        write_csv_atomic(feats, ts_path)
        write_json_atomic(meta_payload(cfg, feats), ts_meta_path)
        LOG.info("published: %s (%s rows x %s columns)", ts_path, len(feats), feats.shape[1])
        LOG.info("published: %s", ts_meta_path)

    return 0


# --------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the compact PM2.5 feature set from a cleaned snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=None,
                        help="cleaned snapshot to read (default: newest cleandata_*.csv)")
    parser.add_argument("--input-dir", default=None,
                        help="snapshot directory (default: <root>/data/preprocessed)")
    parser.add_argument("--output", default=None,
                        help="destination CSV (default: <root>/data/processed/processed_data.csv)")
    parser.add_argument("--hour-tz", default="UTC",
                        help="time zone for hour_sin/hour_cos")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--double", action="store_true",
        help="overwrite processed_data.csv and also write a timestamped copy "
             "processed_data_YYYYmmdd_HHMMSS.csv",
    )
    output_group.add_argument(
        "--timestamp-only", action="store_true",
        help="write ONLY to a timestamped copy processed_data_YYYYmmdd_HHMMSS.csv; "
             "leave processed_data.csv untouched",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="refuse to replace an existing output")
    parser.add_argument("--dry-run", action="store_true", help="build and report without publishing")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    root = resolve_project_root()
    input_dir = Path(args.input_dir).expanduser() if args.input_dir else root.joinpath(*DEFAULT_INPUT_DIR)

    if args.input:
        candidate = Path(args.input).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.is_file():
            raise FeatureError(f"input not found: {candidate}")
        input_path = candidate.resolve()
    else:
        found = latest_snapshot(input_dir)
        if found is None:
            raise FeatureError(f"no {SNAPSHOT_GLOB} snapshot inside {input_dir}")
        input_path = found.resolve()

    output_path = Path(args.output).expanduser() if args.output else root.joinpath(*DEFAULT_OUTPUT)

    try:
        pd.Timestamp("2026-01-01", tz="UTC").tz_convert(args.hour_tz)
    except Exception as exc:
        raise FeatureError(f"invalid --hour-tz {args.hour_tz!r}: {exc}") from exc


    return Config(
        input_path=input_path,
        output_path=output_path.resolve(),
        hour_tz=args.hour_tz,
        overwrite=not args.no_overwrite,
        dry_run=args.dry_run,
        double=args.double,
        timestamp_only=args.timestamp_only,
        timestamp="",
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

    LOG.info("input=%s | output=%s", cfg.input_path, cfg.output_path)
    try:
        return run(cfg)
    except FeatureError as exc:
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.warning("interrupted by user")
        return 130
    except Exception:
        LOG.exception("feature engineering failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())