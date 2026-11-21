#!/usr/bin/env python3
"""
PM2.5 Predictor Inference API (FastAPI)

Layanan ini:
1. Menjalankan pipeline data (ingest -> preprocess -> feature) setiap jam dan saat startup.
2. Melakukan inference model PM2.5 dari MLflow Model Registry.
3. Menyimpan hasil inference ke SQLite cache (satu baris terbaru).
4. Hanya melayani hasil cache melalui endpoint /predict (tidak recompute per request).

Endpoints:
- GET  /          -> Info service
- GET  /health    -> Kesehatan MLflow + ketersediaan cache
- GET  /predict   -> Ambil hasil inference terbaru dari SQLite
- POST /inference -> Paksa jalankan pipeline + inference segera
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

try:
    import mlflow
    import mlflow.pyfunc
    import mlflow.tracking
except Exception:
    mlflow = None

import sqlite3

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "xgboost_24compact")
CACHE_DB_PATH = Path(os.getenv("CACHE_DB_PATH", "/app/cache/inference_cache.db"))

DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
MODEL_META_PATH = PROJECT_ROOT / "models" / "model_meta.json"

PIPELINE_STEPS = [
    ["python", str(PROJECT_ROOT / "src" / "data" / "ingest_data.py")],
    ["python", str(PROJECT_ROOT / "src" / "data" / "preprocess.py")],
    ["python", str(PROJECT_ROOT / "src" / "feature" / "feature_engineer.py")],
]
from src.feature.feature_engineer import get_latest_features

app = FastAPI(title="PM2.5 Predictor Inference API", version="1.0.0")


def get_db():
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS latest_inference (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            inferenced_at TEXT NOT NULL,
            base_datetime TEXT NOT NULL,
            predictions_json TEXT NOT NULL,
            horizon INTEGER DEFAULT 24,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_model_meta():
    if MODEL_META_PATH.exists():
        with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_pipeline_step(cmd):
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline step failed: {' '.join(cmd)}\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return result.stdout


def run_data_pipeline():
    for step in PIPELINE_STEPS:
        run_pipeline_step(step)

def get_latest_row():
    import pandas as pd

    df = pd.read_csv(DATA_PROCESSED_PATH, parse_dates=["datetime"])
    if df.empty:
        raise RuntimeError("processed_data.csv is empty")
    return df.iloc[[-1]]


def convert_delta_to_absolute(latest_pm25: float, deltas):
    return [float(latest_pm25 + d) for d in deltas]


def run_inference():
    run_data_pipeline()

    latest_feats = get_latest_features(lookback_hours=48)
    meta = load_model_meta()

    feature_cols = meta.get("features", [])
    target_cols = meta.get("targets", [f"target_delta_{i}" for i in range(1, 25)])

    if feature_cols:
        X = latest_feats[feature_cols]
    else:
        X = latest_feats

    latest_pm25 = float(latest_feats["pm25"].iloc[0]) if "pm25" in latest_feats.columns else 0.0
    base_datetime = (
        str(latest_feats["datetime"].iloc[0])
        if "datetime" in latest_feats.columns
        else datetime.now(timezone.utc).isoformat()
    )

    if mlflow is None:
        raise RuntimeError("mlflow package not installed in this container")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}/Production"
    try:
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
    except Exception:
        # Fallback: muat artefak model lokal yang di-mount dari host
        local_model_path = PROJECT_ROOT / "models" / "registered" / "xgboost_24compact" / "v1"
        if not local_model_path.exists():
            raise RuntimeError(
                f"MLflow registry gagal dan model lokal tidak ditemukan di {local_model_path}"
            )
        model = mlflow.pyfunc.load_model(str(local_model_path))

    predictions = model.predict(X)
    if hasattr(predictions, "tolist"):
        deltas = predictions.tolist()[0]
    else:
        deltas = list(predictions)[0]

    absolute_predictions = convert_delta_to_absolute(latest_pm25, deltas)

    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    conn.execute("DELETE FROM latest_inference")
    conn.execute(
        """
        INSERT INTO latest_inference
            (id, inferenced_at, base_datetime, predictions_json, horizon, updated_at)
        VALUES (1, ?, ?, ?, ?, ?)
        """,
        (now, base_datetime, json.dumps(absolute_predictions), len(absolute_predictions), now),
    )
    conn.commit()
    conn.close()

    return {
        "inferenced_at": now,
        "base_datetime": base_datetime,
        "horizon": len(absolute_predictions),
        "predictions": absolute_predictions,
        "unit": "absolute_pm25",
    }


async def run_inference_async():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_inference)


async def hourly_scheduler():
    while True:
        await asyncio.sleep(3600)
        try:
            await run_inference_async()
        except Exception as exc:
            print(f"Hourly inference failed: {exc}", file=sys.stderr)


@app.on_event("startup")
async def startup():
    asyncio.create_task(run_inference_async())
    asyncio.create_task(hourly_scheduler())

@app.get("/")
def root():
    return {
        "service": "pm25-predictor-api",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/inference"],
    }


@app.get("/health")
def health():
    mlflow_ok = False
    try:
        if mlflow is not None:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            client.search_experiments(max_results=1)
            mlflow_ok = True
    except Exception:
        mlflow_ok = False

    cache_ok = False
    latest = None
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT inferenced_at, base_datetime FROM latest_inference WHERE id=1"
        ).fetchone()
        conn.close()
        if row:
            cache_ok = True
            latest = {"inferenced_at": row[0], "base_datetime": row[1]}
    except Exception:
        pass

    return {
        "status": "ok",
        "mlflow_reachable": mlflow_ok,
        "cache_available": cache_ok,
        "latest_inference": latest,
    }


@app.get("/predict")
def predict():
    conn = get_db()
    row = conn.execute(
        "SELECT inferenced_at, base_datetime, predictions_json, horizon "
        "FROM latest_inference WHERE id=1"
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="No inference cache available. Call POST /inference to generate.",
        )

    return {
        "inferenced_at": row[0],
        "base_datetime": row[1],
        "horizon": row[3],
        "predictions": json.loads(row[2]),
        "unit": "absolute_pm25",
    }


@app.post("/inference")
def inference():
    try:
        result = run_inference()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
