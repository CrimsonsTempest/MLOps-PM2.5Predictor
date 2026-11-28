#!/usr/bin/env python3
"""
Wrapper for MLflow model serving that exposes Prometheus metrics.

Runs MLflow `models serve` as a subprocess on internal port 8081,
and exposes a FastAPI proxy on port 8080 with /metrics endpoint.
"""

import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

MLFLOW_INTERNAL_PORT = 8081
EXTERNAL_PORT = 8080
MODEL_URI = os.getenv("MODEL_URI", "/opt/ml/model")
MLFLOW_CMD = [
    "mlflow",
    "models",
    "serve",
    "--model-uri",
    MODEL_URI,
    "--host",
    "0.0.0.0",
    "--port",
    str(MLFLOW_INTERNAL_PORT),
    "--env-manager",
    "local",
]

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUEST_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["method", "endpoint"],
)

PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of model prediction values (PM2.5 absolute)",
    buckets=[0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of predictions made",
)

MLFLOW_HEALTH = Gauge(
    "mlflow_serve_healthy",
    "Whether MLflow model serve subprocess is healthy (1) or not (0)",
)

mlflow_process: Optional[subprocess.Popen] = None
http_client: Optional[httpx.AsyncClient] = None


def start_mlflow_serve() -> subprocess.Popen:
    """Start MLflow model serve as subprocess."""
    print(f"Starting MLflow model serve on port {MLFLOW_INTERNAL_PORT}...", flush=True)
    proc = subprocess.Popen(
        MLFLOW_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def wait_for_mlflow_ready(timeout: int = 60) -> bool:
    """Wait for MLflow serve to be ready by polling /ping."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://localhost:{MLFLOW_INTERNAL_PORT}/ping", timeout=2
            ) as resp:
                if resp.status == 200:
                    print("MLflow model serve is ready", flush=True)
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def stop_mlflow_serve(proc: subprocess.Popen):
    """Stop MLflow serve subprocess gracefully."""
    print("Stopping MLflow model serve...", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mlflow_process, http_client

    # Startup
    mlflow_process = start_mlflow_serve()

    # Log MLflow output in background
    import threading

    def log_output():
        for line in mlflow_process.stdout:
            print(f"[MLflow] {line.rstrip()}", flush=True)

    threading.Thread(target=log_output, daemon=True).start()

    # Wait for MLflow to be ready
    if not wait_for_mlflow_ready():
        print("ERROR: MLflow model serve failed to start", flush=True)
        MLFLOW_HEALTH.set(0)
    else:
        MLFLOW_HEALTH.set(1)

    http_client = httpx.AsyncClient(base_url=f"http://localhost:{MLFLOW_INTERNAL_PORT}", timeout=30.0)

    yield

    # Shutdown
    if http_client:
        await http_client.aclose()
    if mlflow_process:
        stop_mlflow_serve(mlflow_process)
    MLFLOW_HEALTH.set(0)


app = FastAPI(title="PM2.5 Model Serving with Metrics", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path
    endpoint = path if path in ["/invocations", "/ping", "/metrics", "/health"] else "other"

    REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
    start_time = time.time()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:
        status = 500
        raise
    finally:
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

    return response


async def proxy_request(request: Request, path: str):
    """Proxy request to MLflow serve."""
    if not http_client:
        return Response(content="MLflow not ready", status_code=503)

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        resp = await http_client.request(
            method=request.method,
            url=path,
            content=body,
            headers=headers,
        )

        # Extract predictions from response for metrics
        if path == "/invocations" and resp.status_code == 200:
            try:
                import json

                data = json.loads(resp.content)
                # MLflow returns {"predictions": [[...]]} or similar
                predictions = data.get("predictions", data)
                if isinstance(predictions, list):
                    for pred in predictions:
                        if isinstance(pred, list):
                            for p in pred:
                                PREDICTION_VALUE.observe(float(p))
                                PREDICTION_COUNT.inc()
                        else:
                            PREDICTION_VALUE.observe(float(pred))
                            PREDICTION_COUNT.inc()
            except Exception:
                pass

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type"),
        )
    except httpx.RequestError as e:
        return Response(content=f"Proxy error: {e}", status_code=502)


@app.post("/invocations")
async def invocations(request: Request):
    return await proxy_request(request, "/invocations")


@app.get("/ping")
async def ping(request: Request):
    return await proxy_request(request, "/ping")


@app.get("/health")
async def health(request: Request):
    return await proxy_request(request, "/health")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {
        "service": "pm25-model-serving",
        "version": "1.0.0",
        "mlflow_internal_port": MLFLOW_INTERNAL_PORT,
        "endpoints": ["/invocations", "/ping", "/health", "/metrics"],
    }


def handle_signal(signum, frame):
    print(f"Received signal {signum}, shutting down...", flush=True)
    if mlflow_process:
        stop_mlflow_serve(mlflow_process)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=EXTERNAL_PORT)