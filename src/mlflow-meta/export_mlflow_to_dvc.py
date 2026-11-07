#!/usr/bin/env python3
"""
Ekspor artefak model dari MLflow Model Registry ke folder DVC-tracked
dan generate/update model_version.yml.

Skrip ini bersifat generik: bekerja untuk nama model apa pun yang sudah
terdaftar di MLflow Model Registry. Tidak melakukan promoti atau transisi
stage apa pun — hanya menyalin artefak yang sudah ada dan mencatatnya
ke dalam metadata YAML untuk dokumentasi serta reproducibility.

Contoh penggunaan:
    python src/mlflow-meta/export_mlflow_to_dvc.py --model-name xgboost_24compact

Setelah menjalankan skrip ini, jalankan:
    dvc add models/registered/<model_name>/v1/
    dvc add models/registered/<model_name>/v2/
    ...
"""

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
REGISTERED_DIR = PROJECT_ROOT / "models" / "registered"


def get_model_versions(client: MlflowClient, model_name: str):
    """Ambil semua versi model dari MLflow Model Registry, diurutkan versi ascending."""
    versions = client.search_model_versions(f"name = '{model_name}'")
    return sorted(versions, key=lambda mv: int(mv.version))


def extract_model_id_from_source(source: str) -> str | None:
    """
    Ekstrak model ID dari source MLflow.

    Source biasanya berformat: models:/m-3b5e7753086345648694d82c206665b1
    Kembalikan bagian setelah 'models:/', yaitu 'm-3b5e7753086345648694d82c206665b1'.
    """
    if source and source.startswith("models:/"):
        return source[len("models:/") :]
    return None


def find_artifact_source(model_name: str, version: int, mlruns_models_dir: Path) -> Path | None:
    """
    Cari folder artefak asli di mlruns/ untuk versi model tertentu.

    MLflow menyimpan model registered di mlruns/1/models/m-<model_id>/artifacts/.
    Model ID diambil dari mv.source (format: models:/m-<id>).
    """
    if not mlruns_models_dir.exists():
        return None

    # Langsung coba berdasarkan model ID yang di-parse dari source
    for cand in sorted(mlruns_models_dir.glob("m-*")):
        artifacts_dir = cand / "artifacts"
        if not artifacts_dir.is_dir():
            continue

        # Verifikasi dengan registered_model_meta jika ada
        meta_file = artifacts_dir / "registered_model_meta"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if (
                    meta.get("model_name") == model_name
                    and str(meta.get("model_version")) == str(version)
                ):
                    return artifacts_dir
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback: cocokkan nama folder m-* dengan model ID
        if cand.name == model_name or cand.name.endswith(model_name):
            return artifacts_dir

    return None


def copy_artifacts(source_dir: Path, dest_dir: Path) -> None:
    """Salin semua file artefak dari source ke dest."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dest_dir / item.name)


def fetch_run_metrics(run_id: str, db_path: Path) -> dict:
    """Ambil metrik run dari MLflow SQLite DB."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM runs WHERE run_uuid = ?", (run_id,))
        row = cur.fetchone()
        run_meta = dict(row) if row else {}
        conn.close()
        return run_meta
    except sqlite3.Error:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MLflow model versions to DVC-tracked folder and generate model_version.yml"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Nama model di MLflow Model Registry (contoh: xgboost_24compact)",
    )
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DB}")
    versions = get_model_versions(client, args.model_name)

    if not versions:
        print(f"Tidak ada versi model yang ditemukan untuk '{args.model_name}' di MLflow Model Registry.")
        return

    REGISTERED_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = REGISTERED_DIR / args.model_name
    mlruns_models_dir = PROJECT_ROOT / "mlruns" / "1" / "models"

    manifest = {
        "model_name": args.model_name,
        "mlflow_tracking_uri": f"sqlite:///{MLFLOW_DB}",
        "inference_uri": f"models:/{args.model_name}/Production",
        "model_flavor": "sklearn/skops",
        "versions": [],
    }

    for mv in versions:
        v = int(mv.version)
        run_id = mv.run_id
        stage = mv.status or "None"

        # Ekstrak model ID dari mv.source (format: models:/m-<id>)
        model_id = extract_model_id_from_source(mv.source)
        if model_id is None:
            print(f"Warning: tidak bisa ekstrak model ID dari source versi {v}, dilewati.")
            continue

        source_path = mlruns_models_dir / model_id / "artifacts"
        if not source_path.is_dir():
            print(f"Warning: folder artefak tidak ditemukan untuk versi {v} di {source_path}, dilewati.")
            continue

        dest_dir = model_dir / f"v{v}"
        copy_artifacts(source_path, dest_dir)
        print(f"Exported versi {v} (stage={stage}) -> {dest_dir}")

        run_meta = fetch_run_metrics(run_id, MLFLOW_DB)

        manifest["versions"].append({
            "version": v,
            "stage": stage,
            "run_id": run_id,
            "model_id": model_id,
            "artifact_uri": str(source_path),
            "exported_dir": str(dest_dir.relative_to(PROJECT_ROOT)),
            "dvc_file": str(dest_dir.parent.relative_to(PROJECT_ROOT)) + "/" + dest_dir.name + ".dvc",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "metrics": run_meta,
        })

    # Tulis model_version.yml
    yml_path = REGISTERED_DIR / "model_version.yml"
    yml_path.write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"\nManifest ditulis ke {yml_path}")
    print("\nSelanjutnya jalankan DVC pada setiap versi:")
    for v_info in manifest["versions"]:
        print(f"  dvc add {v_info['exported_dir']}")


if __name__ == "__main__":
    main()