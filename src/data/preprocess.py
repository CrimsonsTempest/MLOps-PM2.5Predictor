#!/usr/bin/env python3
"""Cleaning-only stage: raw OpenAQ snapshots into versioned cleaned snapshots.

Expected layout in the target repo:
    <root>/.env
    <root>/data/raw/rawdata_YYYYmmdd_HHMMSS.csv            (produced by src/data/ingest_data.py)
    <root>/data/preprocessed/cleandata_YYYYmmdd_HHMMSS.csv (published by this script)

The project root is discovered the same way the ingestion stage does it: walk up from
the script directory (then from the current directory) until a folder containing `.env`
is found; `../../.env` and `.env` are also tried.

This script performs only the cleaning step: it normalises timestamps, coerces
dtypes, drops unusable rows and de-duplicates on the ingestion key, and nothing
else. No pivoting, no hourly grid, no interpolation and no lag/cyclical/target
features are computed here; the cleaned long-format snapshot published below is
the input for the later feature-extraction stage.

Input contract (long format, one row per sensor per reported interval):
    sensor_id        int64     OpenAQ sensor id (provenance; -1 when unknown)
    parameter        str       pm25 / pm1 / relativehumidity / temperature / um003
    units            str       unit as reported by the API ("unknown" when absent)
    datetime_utc     str       ISO-8601 UTC, canonical YYYY-MM-DDTHH:MM:SSZ
    datetime_local   str       station local time as reported (here +07:00)
    value            float64   measurement value
    ingested_at_utc  str       when the ingestion run that produced the row ran

Only `datetime_utc`, `parameter` and `value` are mandatory. The notebook-era cache CSV
(notebooks/openaq_raw_6144741.csv holds exactly those three columns) is therefore
accepted too; the absent provenance columns are synthesised and logged as a warning.

Output contract (long format, like the raw input; `datetime_utc` stays a plain
column, not an index):
    sensor_id           int64     OpenAQ sensor id (-1 when the input had none)
    parameter           str       parameter name, stripped
    units               str       unit as reported ("unknown" when absent)
    datetime_utc        str       ISO-8601 UTC, canonical YYYY-MM-DDTHH:MM:SSZ
    datetime_local      str       station local time as reported ("" when unknown)
    value               float64   measurement value
    ingested_at_utc     str       carried over from the input ("unknown" when absent)
    preprocessed_at_utc str       when the cleaning run that produced the row ran

Flow:
    1. read the newest data/raw/rawdata_*.csv (or the --input given)
    2. normalise timestamps, drop unusable rows, de-duplicate on
       (sensor_id, parameter, datetime_utc) keeping the newest occurrence
    3. report per-parameter counts and the timeline span; every dropped row is
       accounted for by reason (bad timestamp, blank parameter, bad value)
    4. verify and publish a NEW timestamped snapshot atomically
       (cleandata_<ts>.csv.part -> cleandata_<ts>.csv) together with a
       *.meta.json sidecar holding provenance and drop counts

The fixed-width timestamp in the filename makes name order == time order, exactly
like the raw layer, so the newest cleaned version is always the last file and the
history stays versioned on disk (and under DVC, like the raw snapshots).

Dependencies:
    pip install "pandas>=3.0"

Examples:
    python src/data/preprocess.py
    python src/data/preprocess.py --input notebooks/openaq_raw_6144741.csv
    python src/data/preprocess.py --dry-run --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- config
SNAPSHOT_GLOB = "rawdata_*.csv"
DEFAULT_OUTPUT_PREFIX = "cleandata"
DEFAULT_RAW_DIR = ("data", "raw")
DEFAULT_OUTPUT_DIR = ("data", "preprocessed")

TIMESTAMP_COLUMN = "datetime_utc"
LOCAL_TIMESTAMP_COLUMN = "datetime_local"
PARAMETER_COLUMN = "parameter"
VALUE_COLUMN = "value"
SENSOR_COLUMN = "sensor_id"
UNITS_COLUMN = "units"
INGESTED_AT_COLUMN = "ingested_at_utc"
PREPROCESSED_AT_COLUMN = "preprocessed_at_utc"

KEY_COLUMNS = (SENSOR_COLUMN, PARAMETER_COLUMN, TIMESTAMP_COLUMN)
REQUIRED_COLUMNS = (TIMESTAMP_COLUMN, PARAMETER_COLUMN, VALUE_COLUMN)
UNKNOWN_SENSOR_ID = -1
UNKNOWN_UNITS = "unknown"
UNKNOWN_INGESTED_AT = "unknown"

#: Column order of the published CSV (the raw contract, plus the cleaning stamp).
OUTPUT_COLUMNS = (
    SENSOR_COLUMN,
    PARAMETER_COLUMN,
    UNITS_COLUMN,
    TIMESTAMP_COLUMN,
    LOCAL_TIMESTAMP_COLUMN,
    VALUE_COLUMN,
    INGESTED_AT_COLUMN,
    PREPROCESSED_AT_COLUMN,
)

LOG = logging.getLogger("preprocess")


class PreprocessError(RuntimeError):
    """Raised when the raw input cannot be turned into a cleaned snapshot."""


# ------------------------------------------------------------------ paths & secrets
@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    dry_run: bool = False


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    """Canonical UTC timestamp, identical to the one ingestion writes."""
    aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_project_root(start: Path) -> Path | None:
    """Walk up from *start* until a folder containing .env is found."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".env").is_file():
            return candidate
    return None


def resolve_project_root() -> Path:
    """Project root as seen by the ingestion stage, with a fallback for exotic cwd."""
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        root = find_project_root(start)
        if root is not None:
            return root
    return Path(__file__).resolve().parents[2]


def latest_snapshot(raw_dir: Path) -> Path | None:
    """Newest rawdata_*.csv; the fixed-width name makes name order == time order."""
    if not raw_dir.is_dir():
        return None
    snapshots = sorted(raw_dir.glob(SNAPSHOT_GLOB), key=lambda path: path.name)
    return snapshots[-1] if snapshots else None


# ------------------------------------------------------------- timestamp normalising
def normalize_timestamps(values: pd.Series) -> pd.Series:
    """Turn ISO-8601 text (Z or offset suffixed, T or space separated) into UTC.

    Mirrors the tolerant extraction used by the ingestion stage: the date/hour part is
    captured with a regex first, so extra precision or a trailing offset never breaks
    the parse. The result is timezone-aware UTC (``datetime64[ns, UTC]``).
    """
    text = values.astype("string").str.strip()
    captured = text.str.extract(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", expand=False)
    normalized = captured.str.replace(" ", "T", regex=False)
    return pd.to_datetime(
        normalized, format="%Y-%m-%dT%H:%M:%S", utc=True, errors="coerce"
    )


# ------------------------------------------------------------------------ raw input
def read_raw(path: Path) -> pd.DataFrame:
    """Read a long-format snapshot, filling the provenance columns when absent."""
    if not path.is_file():
        raise PreprocessError(f"raw input not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise PreprocessError(f"{path.name} contains no rows")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PreprocessError(
            f"{path.name} lacks the mandatory column(s) {missing}; "
            f"columns found: {list(frame.columns)}"
        )

    for column, default in (
        (SENSOR_COLUMN, UNKNOWN_SENSOR_ID),
        (UNITS_COLUMN, UNKNOWN_UNITS),
    ):
        if column not in frame.columns:
            LOG.warning(
                "%s has no %s column; filling it with %r", path.name, column, default
            )
            frame[column] = default
    if LOCAL_TIMESTAMP_COLUMN not in frame.columns:
        LOG.warning(
            "%s has no %s column; station local time will be unknown",
            path.name,
            LOCAL_TIMESTAMP_COLUMN,
        )
        frame[LOCAL_TIMESTAMP_COLUMN] = pd.NA
    if INGESTED_AT_COLUMN not in frame.columns:
        LOG.warning(
            "%s has no %s column; provenance stamp will be %r",
            path.name,
            INGESTED_AT_COLUMN,
            UNKNOWN_INGESTED_AT,
        )
        frame[INGESTED_AT_COLUMN] = UNKNOWN_INGESTED_AT

    stamped = normalize_timestamps(frame[INGESTED_AT_COLUMN].astype("string")).max()
    if pd.notna(stamped):
        LOG.info("source %s was ingested at %s", path.name, iso(stamped.to_pydatetime()))
    LOG.info("raw input: %s (%s rows)", path.name, len(frame))
    return frame


def unit_conflicts(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Parameters reported with more than one unit, i.e. a mixing risk downstream."""
    conflicts: dict[str, list[str]] = {}
    for parameter, units in frame.groupby(PARAMETER_COLUMN)[UNITS_COLUMN]:
        unique = sorted({str(unit) for unit in units})
        if len(unique) > 1:
            conflicts[str(parameter)] = unique
    return conflicts


# -------------------------------------------------------------------------- cleaning
def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalise dtypes, drop unusable rows and enforce the ingestion key.

    Same cleaning the feature-extraction stage applies to a raw input, minus any
    pivoting. Returns the cleaned frame plus the per-reason drop counts.
    """
    rows_in = len(frame)
    cleaned = frame.copy()
    drops: dict[str, int] = {}

    cleaned[TIMESTAMP_COLUMN] = normalize_timestamps(cleaned[TIMESTAMP_COLUMN])
    drops["bad_timestamp"] = int(cleaned[TIMESTAMP_COLUMN].isna().sum())
    if drops["bad_timestamp"]:
        LOG.warning("dropping %s rows with an unusable timestamp", drops["bad_timestamp"])
    cleaned = cleaned[cleaned[TIMESTAMP_COLUMN].notna()]

    cleaned[PARAMETER_COLUMN] = cleaned[PARAMETER_COLUMN].astype("string").str.strip()
    drops["blank_parameter"] = int(
        (cleaned[PARAMETER_COLUMN].isna() | (cleaned[PARAMETER_COLUMN] == "")).sum()
    )
    if drops["blank_parameter"]:
        LOG.warning("dropping %s rows without a parameter name", drops["blank_parameter"])
    cleaned = cleaned[
        cleaned[PARAMETER_COLUMN].notna() & (cleaned[PARAMETER_COLUMN] != "")
    ]

    cleaned[VALUE_COLUMN] = pd.to_numeric(cleaned[VALUE_COLUMN], errors="coerce")
    drops["bad_value"] = int(cleaned[VALUE_COLUMN].isna().sum())
    if drops["bad_value"]:
        LOG.warning("dropping %s rows with a non-numeric value", drops["bad_value"])
    cleaned = cleaned[cleaned[VALUE_COLUMN].notna()]

    cleaned[SENSOR_COLUMN] = (
        pd.to_numeric(cleaned[SENSOR_COLUMN], errors="coerce")
        .fillna(UNKNOWN_SENSOR_ID)
        .astype("int64")
    )
    cleaned[UNITS_COLUMN] = cleaned[UNITS_COLUMN].astype("string").fillna(UNKNOWN_UNITS)
    cleaned[LOCAL_TIMESTAMP_COLUMN] = cleaned[LOCAL_TIMESTAMP_COLUMN].astype("string")
    cleaned[INGESTED_AT_COLUMN] = cleaned[INGESTED_AT_COLUMN].astype("string")

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
    drops["duplicates"] = before - len(cleaned)
    if drops["duplicates"]:
        LOG.info("dropped %s duplicate rows on %s", drops["duplicates"], list(KEY_COLUMNS))

    conflicts = unit_conflicts(cleaned)
    for parameter, units in conflicts.items():
        LOG.warning("parameter %s is reported in several units: %s", parameter, units)

    cleaned = cleaned.sort_values([TIMESTAMP_COLUMN, PARAMETER_COLUMN]).reset_index(drop=True)
    drops["cleaned_rows"] = int(len(cleaned))
    LOG.info("clean: %s rows in -> %s rows out", rows_in, len(cleaned))
    return cleaned, drops


# ----------------------------------------------------------------------- reporting
def log_report(cleaned: pd.DataFrame, source_name: str) -> None:
    """Per-parameter row counts plus the cleaned timeline span."""
    first = cleaned[TIMESTAMP_COLUMN].min().to_pydatetime()
    last = cleaned[TIMESTAMP_COLUMN].max().to_pydatetime()
    LOG.info(
        "timeline: %s -> %s | %s rows", iso(first), iso(last), len(cleaned)
    )
    for parameter, group in cleaned.groupby(PARAMETER_COLUMN, sort=True):
        span_hours = int(
            (group[TIMESTAMP_COLUMN].max() - group[TIMESTAMP_COLUMN].min())
            .total_seconds() // 3600
        ) + 1
        LOG.info(
            "  %-16s %6s rows  missing_hours %s  (source %s)",
            parameter,
            len(group),
            max(span_hours - len(group), 0),
            source_name,
        )


# ----------------------------------------------------------------------- output IO
def stamp_frame(cleaned: pd.DataFrame, stamped_at: str) -> pd.DataFrame:
    """Render the cleaned frame back into the long-format snapshot contract."""
    frame = cleaned.copy()
    frame[TIMESTAMP_COLUMN] = (
        frame[TIMESTAMP_COLUMN].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    frame[PARAMETER_COLUMN] = frame[PARAMETER_COLUMN].astype("string")
    frame[SENSOR_COLUMN] = frame[SENSOR_COLUMN].astype("int64")
    frame[VALUE_COLUMN] = frame[VALUE_COLUMN].astype("float64")
    frame[UNITS_COLUMN] = frame[UNITS_COLUMN].astype("string")
    frame[LOCAL_TIMESTAMP_COLUMN] = (
        frame[LOCAL_TIMESTAMP_COLUMN].astype("string").fillna("")
    )
    frame[INGESTED_AT_COLUMN] = frame[INGESTED_AT_COLUMN].astype("string")
    frame[PREPROCESSED_AT_COLUMN] = stamped_at
    return frame[list(OUTPUT_COLUMNS)]


def verify_output(frame: pd.DataFrame) -> list[str]:
    """Return the list of problems; empty means the snapshot is safe to publish."""
    problems: list[str] = []
    if frame.empty:
        return ["the cleaned snapshot is empty"]
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    unparsed = int(
        normalize_timestamps(frame[TIMESTAMP_COLUMN]).isna().sum()
    )
    if unparsed:
        problems.append(f"{unparsed} rows with an unusable timestamp")
    LOG.info(
        "verification: cleaned snapshot holds %s rows x %s columns",
        len(frame),
        frame.shape[1],
    )
    return problems


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Publish through a .part file so a half-written CSV is never visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".part")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)
    LOG.info("published: %s", path)


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".part")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    LOG.info("published: %s", path)


def meta_payload(
    cfg: Config,
    counts: dict[str, int],
    frame: pd.DataFrame,
    conflicts: dict[str, list[str]],
) -> dict[str, object]:
    """Provenance + drop counts published next to the CSV for the next stage."""
    first = normalize_timestamps(frame[TIMESTAMP_COLUMN]).min()
    last = normalize_timestamps(frame[TIMESTAMP_COLUMN]).max()
    return {
        "generated_at_utc": iso(utcnow()),
        "source_file": cfg.input_path.name,
        "source_path": str(cfg.input_path),
        "output_file": cfg.output_path.name,
        "rows": counts,
        "first_hour_utc": iso(first.to_pydatetime()),
        "last_hour_utc": iso(last.to_pydatetime()),
        "parameters": sorted(frame[PARAMETER_COLUMN].unique().tolist()),
        "unit_conflicts": conflicts,
        "output_columns": list(frame.columns),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(cfg).items()
        },
        "library_versions": {"pandas": pd.__version__},
    }


# --------------------------------------------------------------------------- pipeline
def run(cfg: Config) -> int:
    raw = read_raw(cfg.input_path)
    counts = {"raw_rows": int(len(raw))}

    cleaned, drops = clean(raw)
    counts.update({key: int(value) for key, value in drops.items()})

    log_report(cleaned, cfg.input_path.name)

    stamped = stamp_frame(cleaned, iso(utcnow()))
    counts["published_rows"] = int(len(stamped))

    problems = verify_output(stamped)
    if problems:
        for problem in problems:
            LOG.error("output rejected: %s", problem)
        return 1

    if cfg.dry_run:
        LOG.info("[dry-run] nothing written; would have published %s", cfg.output_path)
        return 0

    write_csv_atomic(stamped, cfg.output_path)
    write_json_atomic(
        meta_payload(cfg, counts, stamped, unit_conflicts(cleaned)),
        cfg.output_path.with_suffix(".meta.json"),
    )
    return 0


# -------------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a raw OpenAQ snapshot into a versioned cleaned snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default=None, help="raw snapshot to read (default: newest rawdata_*.csv)"
    )
    parser.add_argument(
        "--raw-dir", default=None, help="raw snapshot directory (default: <root>/data/raw)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="destination CSV (default: <root>/data/preprocessed/cleandata_<ts>.csv)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="clean and report without publishing"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    root = resolve_project_root()
    raw_dir = (
        Path(args.raw_dir).expanduser()
        if args.raw_dir
        else root.joinpath(*DEFAULT_RAW_DIR)
    )

    if args.input:
        candidate = Path(args.input).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.is_file():
            raise PreprocessError(f"raw input not found: {candidate}")
        input_path = candidate.resolve()
    else:
        found = latest_snapshot(raw_dir)
        if found is None:
            raise PreprocessError(f"no {SNAPSHOT_GLOB} snapshot inside {raw_dir}")
        input_path = found

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        stamp = utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = root.joinpath(
            *DEFAULT_OUTPUT_DIR, f"{DEFAULT_OUTPUT_PREFIX}_{stamp}.csv"
        )
    return Config(
        input_path=input_path.resolve() if input_path.exists() else input_path,
        output_path=output_path.resolve(),
        dry_run=args.dry_run,
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
    except PreprocessError as exc:
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.warning("interrupted by user")
        return 130
    except Exception:
        LOG.exception("cleaning failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

    # __APPEND_NEXT__

