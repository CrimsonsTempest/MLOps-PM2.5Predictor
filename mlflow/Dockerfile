FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==3.17.2

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////app/mlflow.db", "--default-artifact-root", "/app/mlruns", "--allowed-hosts", "*"]