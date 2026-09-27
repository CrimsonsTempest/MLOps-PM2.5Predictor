#!/usr/bin/env python3
"""Incremental OpenAQ v3 ingestion into an append-only raw snapshot layer.

Expected layout in the target repo:
    <root>/.env                              API_KEY=...
    <root>/data/raw/rawdata_YYYYmmdd_HHMMSS.csv

The script works from any depth: the project root is found by walking up from the
script directory (then from the current directory) until a folder containing
`.env` is found; `../../.env` and `.env` are also tried.

Raw snapshot contract (long format, one row per sensor per reported interval):
    sensor_id        int64     OpenAQ sensor id (provenance)
    parameter        str       pm25 / pm1 / relativehumidity / temperature / ...
    units            str       unit as reported by the API
    datetime_utc     str       ISO-8601 UTC, canonical YYYY-MM-DDTHH:MM:SSZ
    datetime_local   str       station local time, exactly as the API returned it
    value            float64   measurement value
    ingested_at_utc  str       when the run that produced the row pulled it

Flow:
    1. the newest data/raw/rawdata_*.csv is the state (fixed-width name = time order)
    2. read its max datetime_utc, subtract --overlap-minutes (late/back-filled values)
    3. pull only newer rows per sensor from /v3/sensors/{id}/measurements
       (datetime_from + limit/page, retry with backoff on throttling/5xx)
    4. concat old + new, de-duplicate on (sensor_id, parameter, datetime_utc) keeping
       the freshly pulled row, sort chronologically
    5. write a NEW snapshot atomically (rawdata_<ts>.csv.part -> rawdata_<ts>.csv) and
       verify it; a snapshot that fails verification is never published

Dependencies:
    pip install "polars>=1.0" "requests>=2.31" "python-dotenv>=1.0"

Examples:
    python src/data/ingest_data.py                   # incremental, full history on cold start
    python src/data/ingest_data.py --days 90         # bound a cold start to the last 90 days
    python src/data/ingest_data.py --overlap-minutes 180
    python src/data/ingest_data.py --force-full --dry-run
    python src/data/ingest_data.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

# --------------------------------------------------------------------------- config
API_BASE = "https://api.openaq.org/v3"
DEFAULT_LOCATION_ID = 6144741
PAGE_LIMIT = 1000             # OpenAQ clamps `limit` at 1000 per page
MAX_PAGES = 200               # safety valve for the pagination loop
REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
BACKOFF_SECONDS = 2.0
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

SNAPSHOT_GLOB = "rawdata_*.csv"
TIMESTAMP_COLUMN = "datetime_utc"
PERIOD_ANCHOR = "datetimeFrom"   # hourly values are stamped at the interval start
KEY_COLUMNS = ("sensor_id", "parameter", TIMESTAMP_COLUMN)
MEASUREMENT_SCHEMA = {
    "sensor_id": pl.Int64,
    "parameter": pl.Utf8,
    "units": pl.Utf8,
    TIMESTAMP_COLUMN: pl.Utf8,
    "datetime_local": pl.Utf8,
    "value": pl.Float64,
}
RAW_COLUMNS = list(MEASUREMENT_SCHEMA) + ["ingested_at_utc"]
API_KEY_ENV_VARS = ("API_KEY", "OPENAQ_API_KEY", "OPENAQ_KEY")

LOG = logging.getLogger("ingestion")


@dataclass(frozen=True)
class Config:
    env_file: Path
    api_key: str
    raw_dir: Path
    location_id: int = DEFAULT_LOCATION_ID
    overlap_minutes: int = 120
    max_pages: int = MAX_PAGES
    since_days: int | None = None
    force_full: bool = False
    dry_run: bool = False


# ------------------------------------------------------------------ paths & secrets
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    """Canonical UTC timestamp used on the wire and inside the snapshots."""
    aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_project_root(start: Path) -> Path | None:
    """Walk up from *start* until a folder containing .env is found."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".env").is_file():
            return candidate
    return None


def resolve_env_file() -> tuple[Path, Path]:
    """Return (project_root, env_file) for <root>/.env."""
    tried: list[str] = []
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        root = find_project_root(start)
        if root is not None:
            return root, root / ".env"
        tried.append(str(start))
    for relative in ("../../.env", ".env"):
        candidate = Path(relative).resolve()
        if candidate.is_file():
            return candidate.parent, candidate
        tried.append(str(candidate))
    raise FileNotFoundError(
        ".env not found; searched upward from: " + ", ".join(tried)
    )


def build_config(args: argparse.Namespace) -> Config:
    root, env_file = resolve_env_file()
    load_dotenv(env_file, override=False)

    api_key = next(
        (value for name in API_KEY_ENV_VARS if (value := os.getenv(name))), None
    )
    if not api_key:
        raise ValueError(
            f"no API key found in {env_file} (expected {API_KEY_ENV_VARS[0]}=...)"
        )

    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else root / "data" / "raw"
    return Config(
        env_file=env_file,
        api_key=api_key.strip(),
        raw_dir=raw_dir.resolve(),
        location_id=args.location_id,
        overlap_minutes=args.overlap_minutes,
        max_pages=args.max_pages,
        since_days=args.days,
        force_full=args.force_full,
        dry_run=args.dry_run,
    )


# ------------------------------------------------------------------------ HTTP layer
def build_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"accept": "application/json", "X-API-Key": api_key})
    return session


def api_get(session: requests.Session, path: str, params: dict | None = None) -> dict:
    """GET a v3 endpoint, retrying on throttling and transient server errors."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_SECONDS * attempt
            LOG.warning("%s failed (%s); retrying in %.0fs", path, exc, wait)
            time.sleep(wait)
            continue
        if response.status_code in RETRY_STATUS:
            if attempt == MAX_RETRIES:
                response.raise_for_status()
            wait = float(response.headers.get("Retry-After") or BACKOFF_SECONDS * attempt)
            wait = min(wait, 60.0)
            LOG.warning("HTTP %s from %s; retrying in %.0fs", response.status_code, path, wait)
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"unreachable retry loop for {path}")


def fetch_sensors(session: requests.Session, location_id: int) -> list[dict]:
    """Map every sensor of a location to (sensor_id, parameter, units)."""
    payload = api_get(session, f"/locations/{location_id}/sensors", {"limit": 100})
    sensors: list[dict] = []
    for item in payload.get("results") or []:
        parameter = item.get("parameter") or {}
        if item.get("id") is None or not parameter.get("name"):
            continue
        sensors.append(
            {
                "sensor_id": int(item["id"]),
                "parameter": parameter["name"],
                "units": parameter.get("units"),
            }
        )
    if not sensors:
        raise RuntimeError(f"location {location_id} returned no sensors")
    return sorted(sensors, key=lambda sensor: sensor["sensor_id"])


def pull_sensor(
    session: requests.Session,
    cfg: Config,
    sensor: dict,
    datetime_from: str | None,
) -> list[tuple]:
    """Page through one sensor's measurements, optionally bounded by datetime_from."""
    rows: list[tuple] = []
    for page in range(1, cfg.max_pages + 1):
        params: dict[str, object] = {"limit": PAGE_LIMIT, "page": page}
        if datetime_from:
            params["datetime_from"] = datetime_from
        payload = api_get(
            session, f"/sensors/{sensor['sensor_id']}/measurements", params=params
        )
        results = payload.get("results") or []
        if not results:
            break
        for item in results:
            period = item.get("period") or {}
            anchor = period.get(PERIOD_ANCHOR) or period.get("datetimeTo") or {}
            value = item.get("value")
            if not anchor.get("utc") or value is None:
                continue
            rows.append(
                (
                    sensor["sensor_id"],
                    sensor["parameter"],
                    sensor["units"],
                    anchor["utc"],
                    anchor.get("local"),
                    float(value),
                )
            )
        LOG.debug("sensor %s page %s -> %s results", sensor["sensor_id"], page, len(results))
        if len(results) < PAGE_LIMIT:
            break
    else:
        LOG.warning(
            "sensor %s stopped at the --max-pages cap (%s)",
            sensor["sensor_id"],
            cfg.max_pages,
        )
    return rows


# --------------------------------------------------------------------- polars helpers
def rows_to_frame(rows: list[tuple]) -> pl.DataFrame:
    """Materialise API rows with an explicit schema (nothing inferred)."""
    columns = list(MEASUREMENT_SCHEMA)
    if not rows:
        return pl.DataFrame(schema=MEASUREMENT_SCHEMA)
    return pl.DataFrame(
        {name: [row[index] for row in rows] for index, name in enumerate(columns)},
        schema=MEASUREMENT_SCHEMA,
    )


def normalize_timestamps(frame: pl.DataFrame, column: str = TIMESTAMP_COLUMN) -> pl.DataFrame:
    """Turn ISO-8601 text (Z or offset suffixed, T or space separated) into naive UTC."""
    if frame.schema.get(column) != pl.Utf8:
        return frame
    text = pl.col(column).cast(pl.Utf8).str.strip_chars()
    parsed = pl.coalesce(
        text.str.extract(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", 1).str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S", strict=False
        ),
        text.str.extract(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", 1).str.to_datetime(
            format="%Y-%m-%d %H:%M:%S", strict=False
        ),
    )
    return frame.with_columns(parsed.alias(column))


def format_timestamps(frame: pl.DataFrame, column: str = TIMESTAMP_COLUMN) -> pl.DataFrame:
    """Render datetimes back into the canonical YYYY-MM-DDTHH:MM:SSZ form."""
    if frame.schema.get(column) == pl.Utf8:
        return frame
    return frame.with_columns(
        pl.concat_str(
            [pl.col(column).dt.strftime("%Y-%m-%dT%H:%M:%S"), pl.lit("Z", dtype=pl.Utf8)]
        ).alias(column)
    )


def last_timestamp(frame: pl.DataFrame) -> datetime:
    """Newest stored timestamp, i.e. the incremental cursor."""
    parsed = normalize_timestamps(frame)
    value = parsed.select(pl.col(TIMESTAMP_COLUMN).max()).item()
    if value is None:
        raise ValueError("base snapshot has no usable timestamps")
    return value


# ----------------------------------------------------------------------- snapshot IO
def latest_snapshot(raw_dir: Path) -> Path | None:
    """Newest rawdata_*.csv; the fixed-width name makes name order == time order."""
    if not raw_dir.is_dir():
        return None
    snapshots = sorted(raw_dir.glob(SNAPSHOT_GLOB), key=lambda path: path.name)
    return snapshots[-1] if snapshots else None


def read_snapshot(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=10000)


def load_base_state(raw_dir: Path) -> tuple[Path | None, pl.DataFrame | None]:
    """Read the newest snapshot; degrade to a cold start whenever it is unusable."""
    path = latest_snapshot(raw_dir)
    if path is None:
        LOG.info("no snapshot in %s yet -> cold start", raw_dir)
        return None, None
    try:
        frame = read_snapshot(path)
    except Exception as exc:
        LOG.warning("cannot read %s (%s) -> cold start", path.name, exc)
        return path, None
    if frame.height == 0:
        LOG.warning("%s is empty -> cold start", path.name)
        return path, None
    for column in RAW_COLUMNS:
        if column not in frame.columns:
            LOG.warning("%s lacks column %s; filling it with nulls", path.name, column)
            frame = frame.with_columns(
                pl.lit(None).cast(MEASUREMENT_SCHEMA.get(column, pl.Utf8)).alias(column)
            )
    return path, frame.select(RAW_COLUMNS)


def coverage_lines(frame: pl.DataFrame) -> list[str]:
    """Per-sensor row count, time span and missing-hour estimate."""
    parsed = normalize_timestamps(frame)
    lines: list[str] = []
    for group in parsed.partition_by(["sensor_id", "parameter"], maintain_order=True):
        first = group[TIMESTAMP_COLUMN].min()
        last = group[TIMESTAMP_COLUMN].max()
        span_hours = int((last - first).total_seconds() // 3600) + 1
        lines.append(
            "  {:.<18} sensor {:<10} {:>6} rows  {} -> {}  missing_hours {}".format(
                group["parameter"][0],
                group["sensor_id"][0],
                group.height,
                iso(first),
                iso(last),
                max(span_hours - group.height, 0),
            )
        )
    return lines


def verify_snapshot(path: Path) -> list[str]:
    """Return the list of problems; empty means the snapshot is safe to publish."""
    frame = normalize_timestamps(read_snapshot(path))
    problems: list[str] = []
    if frame.height == 0:
        problems.append("snapshot contains no rows")
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    duplicates = frame.height - frame.unique(subset=list(KEY_COLUMNS)).height
    if duplicates:
        problems.append(
            f"{duplicates} rows violate the (sensor_id, parameter, datetime_utc) key"
        )
    unparsed = frame.filter(pl.col(TIMESTAMP_COLUMN).is_null()).height
    if unparsed:
        problems.append(f"{unparsed} rows with an unusable timestamp")
    LOG.info("verification: %s holds %s rows x %s columns", path.name, frame.height, frame.width)
    for line in coverage_lines(frame):
        LOG.info(line)
    return problems


# --------------------------------------------------------------------------- pipeline
def run(cfg: Config) -> int:
    session = build_session(cfg.api_key)

    sensors = fetch_sensors(session, cfg.location_id)
    LOG.info(
        "location %s exposes %s sensors: %s",
        cfg.location_id,
        len(sensors),
        ", ".join(sensor["parameter"] for sensor in sensors),
    )

    if cfg.force_full:
        LOG.info("--force-full: existing snapshots are ignored")
        base_path, base_frame = None, None
    else:
        base_path, base_frame = load_base_state(cfg.raw_dir)

    datetime_from: str | None = None
    if base_frame is not None and base_path is not None:
        previous = last_timestamp(base_frame)
        datetime_from = iso(previous - timedelta(minutes=cfg.overlap_minutes))
        LOG.info(
            "state %s ends at %s -> pulling from %s (overlap %s min)",
            base_path.name,
            iso(previous),
            datetime_from,
            cfg.overlap_minutes,
        )
    elif cfg.since_days:
        datetime_from = iso(utcnow() - timedelta(days=cfg.since_days))
        LOG.info(
            "cold start bounded to the last %s days (from %s)", cfg.since_days, datetime_from
        )
    else:
        LOG.info("cold start: pulling the full available history of every sensor")

    pulled: list[tuple] = []
    for sensor in sensors:
        rows = pull_sensor(session, cfg, sensor, datetime_from)
        LOG.info("  %-18s %6s rows pulled", sensor["parameter"], len(rows))
        pulled.extend(rows)

    new_frame = rows_to_frame(pulled).with_columns(
        pl.lit(iso(utcnow())).alias("ingested_at_utc")
    )
    LOG.info("fetched %s rows from the API", new_frame.height)

    frames = [
        frame.select(RAW_COLUMNS) for frame in (base_frame, new_frame) if frame is not None
    ]
    merged = pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]

    merged = normalize_timestamps(merged)
    unparsed = merged.filter(pl.col(TIMESTAMP_COLUMN).is_null()).height
    if unparsed:
        LOG.warning("dropping %s rows with an unusable timestamp", unparsed)
        merged = merged.filter(pl.col(TIMESTAMP_COLUMN).is_not_null())

    merged = merged.unique(
        subset=list(KEY_COLUMNS), keep="last", maintain_order=False
    ).sort([TIMESTAMP_COLUMN, "sensor_id"])

    if base_frame is None:
        LOG.info("merged snapshot: %s rows (cold start)", merged.height)
    else:
        LOG.info(
            "merged snapshot: %s rows (%+d versus the %s rows of %s)",
            merged.height,
            merged.height - base_frame.height,
            base_frame.height,
            base_path.name,
        )

    output_path = cfg.raw_dir / f"rawdata_{utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    if cfg.dry_run:
        LOG.info("[dry-run] nothing written; would have created %s", output_path)
        for line in coverage_lines(merged):
            LOG.info(line)
        return 0

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".part")
    format_timestamps(merged).write_csv(temp_path)

    problems = verify_snapshot(temp_path)
    if problems:
        for problem in problems:
            LOG.error("snapshot rejected: %s", problem)
        LOG.error("the rejected file was kept for inspection: %s", temp_path)
        return 1

    temp_path.replace(output_path)
    LOG.info("raw snapshot published: %s", output_path)
    return 0


# -------------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental OpenAQ v3 ingestion into rawdata_*.csv snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--location-id", type=int, default=DEFAULT_LOCATION_ID, help="OpenAQ location id"
    )
    parser.add_argument(
        "--raw-dir", default=None, help="snapshot directory (default: <root>/data/raw)"
    )
    parser.add_argument(
        "--overlap-minutes",
        type=int,
        default=120,
        help="re-fetch window behind the last stored timestamp",
    )
    parser.add_argument(
        "--days", type=int, default=None, help="bound a cold-start pull to the last N days"
    )
    parser.add_argument(
        "--max-pages", type=int, default=MAX_PAGES, help="pagination cap per sensor"
    )
    parser.add_argument(
        "--force-full", action="store_true", help="ignore existing snapshots and pull everything"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch and report without writing a snapshot"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        cfg = build_config(args)
    except Exception as exc:
        LOG.error("configuration error: %s", exc)
        return 2

    LOG.info("env=%s | raw_dir=%s | location=%s", cfg.env_file, cfg.raw_dir, cfg.location_id)
    try:
        return run(cfg)
    except requests.HTTPError as exc:
        LOG.error("OpenAQ rejected the request: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOG.warning("interrupted by user")
        return 130
    except Exception:
        LOG.exception("ingestion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
