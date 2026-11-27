[![Docker Smoke Test](https://github.com/CrimsonsTempest/MLOps-PM2.5Predictor/actions/workflows/docker-smoke.yml/badge.svg)](https://github.com/CrimsonsTempest/MLOps-PM2.5Predictor/actions/workflows/docker-smoke.yml)
# MLOps-PM2.5Predictor

Proyek ini bertujuan untuk membuat sebuah sistem machine learning untuk melakukan prediksi pada nilai pm25 di udara. Dilengkapi dengan proses retraining setiap terjadinya drift yang akan di detect oleh software monitoring, dengan menggunakan apache airflow untuk melakukan fetching data.

Berikut adalah flow dari system yang akan dibuat


1. **Ingestion**: fetch data dengan apache airflow, versioning dengan Data Versioning Control ( DVC )
2. **Preprocessing**: Pembersihan data mentah (normalisasi timestamp, penanganan missing values, dan deduplikasi) dilakukan dengan pandas dan numpy
3. **Model Training** : training dengan model XGBoost 
4. **Model Registry & Evaluation:** : model dilakukan comparison antara model yang running dan model yang baru saja di train dengan logika champion vs challanger, model disimpan artefaknya dengan registry MlFlow
5. **API Serving:** Menyajikan atau serving prediksi dari pm2.5 dengan model yang dipilih melalui REST API menggunakan FastAPI.
6. **Continuous Monitoring:** Mengawasi performa metrik seperti MAE dan mendeteksi data drift untuk memicu proses retraining secara otomatis jika diperlukan.

## Tech Stack
* **Orchestration:** Apache Airflow
* **Data Versioning:** DVC
* **Machine Learning:** XGBoost, Scikit-Learn
* **Model Tracking:** MLflow
* **API Serving:** FastAPI
* **Monitoring:** Prometheus & Grafana

## Struktur Direktori
```text
.
├── .devcontainer/          # konfigurasi development container
├── .dvc/                   # konfigurasi DVC
├── .github/                # konfigurasi GitHub
├── config/                 # metadata run MLflow
├── data/
│   ├── raw/                # data mentah hasil ingestion dan pointer DVC
│   ├── preprocessed/       # data bersih dan metadata preprocessing
│   └── processed/          # data hasil feature engineering dan pointer DVC
├── models/
│   ├── model.joblib        # model legacy (fallback inferensi)
│   ├── model_meta.json     # metadata model legacy
│   └── registered/         # model yang diekspor dari MLflow Model Registry, di-track DVC
│       ├── .gitignore
│       ├── model_version.yml   # metadata versi model (MLflow version ↔ DVC hash)
│       └── xgboost_24compact/
│           ├── v1.dvc            # pointer DVC versi 1
│           ├── v2.dvc            # pointer DVC versi 2
│           ├── v1/               # artefak MLflow versi 1 (DVC-tracked)
│           └── v2/               # artefak MLflow versi 2 (DVC-tracked)
├── notebooks/              # notebook eksplorasi
├── src/
│   ├── data/
│   │   ├── ingest_data.py  # mengambil data mentah dari OpenAQ API
│   │   └── preprocess.py   # membersihkan dan menyiapkan data
│   ├── feature/
│   │   └── feature_engineer.py # membuat fitur untuk training
│   ├── inference/
│   │   └── inference.py    # load model dari MLflow registry (models:/xgboost_24compact/Production)
│   ├── mlflow-meta/
│   │   ├── extract_mlflow_run.py # mengekstrak metadata run MLflow dari SQLite
│   │   └── export_mlflow_to_dvc.py # ekspor artefak MLflow ke folder DVC-tracked + generate model_version.yml
│   └── train/
│       └── train.py        # melatih dan mengevaluasi model
├── requirements.txt        # dependensi Python
└── README.md
```

##  Menjalankan Pipeline Data

```bash
# 1. Ingestion: tarik data mentah dari OpenAQ (butuh API_KEY di .env)
python src/data/ingest_data.py

# 2. Preprocessing: bersihkan snapshot mentah terbaru
python src/data/preprocess.py

# 3. Feature Engineering : susun fitur yang akan digunakan untuk model training
python src/feature/feature_engineer.py

4. Mlflow server dan UI : menyalakan server penyimpan model dan melakukan cek hasil pembuatan model

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlruns \
  --serve-artifacts \
  --allowed-hosts "*"


  mlflow ui --backend-store-uri sqlite:///mlflow.db

# 5. Model Training : melatih model berdasarkan data hasil feature engineering
python src/train/train.py



# 6 Infrence delta pm.25 dari mlflow, jika mlflow tidak ada, akan menggunakan file joblib di models
python src/inference/inference.py --model-name xgboost_24compact


#buka localhost:5000
```

## DVC

DVC berguna untuk melakukan tracking pada versi data yang digunakan untuk membuat sebuah model. secara konsekuen ini membantu reproducibility pembuatan dari suatu model yang sudah ada karena data adalah penentu utama hasil kualitas dari model. sehingga dengan memiliki data yang digunakan sebelumnya untuk membuat model, dapat dilakukan cleaning, engineering, dan training ulang.


### 1. Set Up DVC Remote

 a. Tambah remote storage
```
dvc remote add -d <remote-name> s3://<bucket-name>
```

b. Set endpoint & kredensial lokal

```bash
dvc remote modify <remote-name> endpointurl <ENDPOINT_URL>
dvc remote modify --local <remote-name> access_key_id <ACCESS_KEY_ID>
dvc remote modify --local <remote-name> secret_access_key <SECRET_ACCESS_KEY>
```

### 2. Data Version Tracking 

Ingestion secara default menulis snapshot dengan nama konstan `data/raw/rawdata.csv` (tanpa timestamp di dalamnya). DVC butuh nama file yang tetap untuk di-track, jadi langsung track `rawdata.csv` — tidak perlu langkah copy untuk running default. `dataset.csv` (lama) telah di-deprecate dan diganti oleh `rawdata.csv`.

```bash
dvc add data/raw/rawdata.csv
git add data/raw/rawdata.csv.dvc
git commit -m "Track raw dataset v<VERSION>"
git tag v<VERSION>
git push && git push --tags
```

> **Deprecasi:** `data/raw/dataset.csv` telah di-deprecate. Snapshot kanan yang kanonikal sekarang adalah `data/raw/rawdata.csv`. `dataset.csv` tidak lagi dibaca atau ditulis oleh stage pipeline mana pun dan akan dihapus di rilis mendatang.

### 3. Data Version Switching

```bash
git tag --list                              # untuk melihat versi apa saja yang terdaftar 
git checkout v<VERSION> && dvc checkout     # pindah ke versi terpilih
```

---

## Model Registry & Versioning (MLflow + DVC)

Proyek ini menggunakan **MLflow Model Registry** untuk mengelola siklus hidup model (dari eksperimen ke produksi) dan **DVC** untuk versioning artefak biner model yang telah diekspor dari registry.

### Alur Kerja

1. **Registrasi Model** — Model terbaik dari eksperimen LK-06 didaftarkan ke MLflow Model Registry dengan nama `xgboost_24compact`.
2. **Versioning Artefak** — Model baru dengan parameter berbeda didaftarkan sebagai versi berikutnya (v2, v3, dst). MLflow secara otomatis melacak riwayat perubahan.
3. **Transisi Stage** — Model dipindahkan dari stage `None` → `Staging` → `Production` melalui MLflow UI atau API. Hanya versi yang berstatus `Production` yang dapat diakses oleh layanan inferensi.
4. **Sinkronisasi Metadata dengan DVC** — Artefak model yang dipilih diekspor dari `mlruns/` ke `models/registered/<nama_model>/v<N>/`, lalu di-track dengan `dvc add`. Metadata disimpan di `models/registered/model_version.yml` yang menghubungkan MLflow version ↔ DVC hash → lokasi artefak.
5. **Verifikasi Inferensi** — Model Production dimuat secara programatik menggunakan `mlflow.pyfunc.load_model("models:/xgboost_24compact/Production")` melalui `src/inference/inference.py`.

### Model Aktif untuk Inferensi

| Versi | Stage | Run ID | Test MAE | Test RMSE | Test R2 | Catatan |
|-------|-------|--------|----------|-----------|---------|---------|
| v1 | Production | `a7fd25dcba614697ba700e1e00ae8c20` | 13.88 | 19.37 | 0.452 | Versi aktif untuk inferensi. Generalisasi lebih baik pada data uji. |
| v2 | Staging | `93c966cea7b24b17b943ce1313ae92e6` | 14.05 | 19.58 | 0.440 | Versi challenger dengan n_estimators 750, performa test sedikit lebih rendah dari v1. |

File `models/registered/model_version.yml` adalah sumber kebenaran untuk versi model mana yang saat ini aktif. File `.dvc` di `models/registered/xgboost_24compact/v1.dvc` dan `v2.dvc` menyimpan checksum artefak biner untuk reproducibility.

### Perintah Export Model ke DVC

```bash
# Ekspor semua versi model dari MLflow Model Registry ke folder DVC-tracked
python src/mlflow-meta/export_mlflow_to_dvc.py --model-name xgboost_24compact

# Track artefak biner dengan DVC (data biner disimpan di DVC remote)
dvc add models/registered/xgboost_24compact/v1/
dvc add models/registered/xgboost_24compact/v2/

# Commit metadata dan pointer ke Git
git add models/registered/
git commit -m "Track model version metadata with DVC"
git push
```

### Perintah Inferensi dari Registry

```bash
# Muat model Production dari MLflow Model Registry
python src/inference/inference.py --model-name xgboost_24compact
```

Kode inferensi mencoba memuat dari `models:/xgboost_24compact/Production` terlebih dahulu. Jika gagal, fallback ke `models/model.joblib`.

### Struktur Artefak

- `models/registered/<model>/v<N>/` — artefak MLflow yang diekspor dari Model Registry, di-track oleh DVC. Hanya pointer `.dvc` dan metadata YAML yang disimpan di Git; artefak biner besar disimpan di DVC remote.

## Model Serving & Horizontal Scaling (LK-10)

`docker-compose.yaml` kini mengorkestrasi stack serving model MLflow dengan 3 replika:

1. **model-serving** — image hasil `mlflow models build-docker` (model dibundel di dalam image), `deploy: replicas: 3`, port 8080 internal (`command` menimpa entrypoint: `mlflow models serve --host 0.0.0.0 --port 8080`)
2. **serving-proxy** — nginx reverse proxy (port 8080), satu pintu masuk yang mendistribusikan request secara round-robin ke 3 replika

### Persiapan

**Persyaratan:**
- Docker dan Docker Compose (v2) terinstal.
- Model `xgboost_24compact` berstatus **Production** di MLflow Model Registry.

**Langkah 1 — Build image original (hasil `mlflow models build-docker`):**

```bash
mlflow models build-docker -m models:/xgboost_24compact/Production -n pm25-model-serving:latest
```

Image original berisi model + entrypoint mlflow, tetapi memakai Python 3.11 dan memasang dependensi saat runtime (mode `virtualenv`) — tidak cocok untuk 3 replika (pip install ×3 saat start → crash-loop).

**Langkah 2 — Build image turunan:**

```bash
docker compose build model-serving
```

Image turunan (`mlflow/Dockerfile.serving`) menyalin model dari image original ke `python:3.14-slim` (sesuai environment training) dan memasang dependensi saat build — start cepat tanpa pip runtime.

### Menjalankan Sistem

```bash
# 1. Jalankan seluruh stack (3 replika model-serving + proxy)
docker compose up -d

# 2. Lihat status kontainer (harus tampil 3 replika model-serving)
docker compose ps

# 3. Matikan semua layanan
docker compose down
```

### Mengakses Endpoint API

| Endpoint | Alamat | Fungsi |
|----------|--------|--------|
| Inference | `http://localhost:8080/invocations` (POST) | Prediksi PM2.5 24 jam (input 22 fitur, output 24 delta) |

```bash
# Uji prediksi via proxy (didistribusikan ke salah satu dari 3 replika)
curl -sS -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

### Mengubah Jumlah Replika Secara Dinamis

```bash
# Tambah replika menjadi 5 (tanpa ubah file compose)
docker compose up -d --scale model-serving=5

# Cek replika yang berjalan
docker compose ps

# Kembalikan ke 3 replika
docker compose up -d --scale model-serving=3
```

### Stack Lama (LK-09)

Stack `api-service` (FastAPI cache) dipindah ke `docker-compose.lk09.yaml` dan dijalankan terpisah bila diperlukan:

```bash
docker compose -f docker-compose.lk09.yaml up -d
```

### Deprecated

- **mlflow-server (Tracking Server + Registry UI, port 5000)** — dihapus dari `docker-compose.yaml` pada LK-10. Replika serving memakai model yang sudah bundel di dalam image dan tidak memerlukan tracking server saat runtime. Bila perlu menjalankan registry (mis. untuk registrasi atau verifikasi model), jalankan manual:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlruns \
  --serve-artifacts \
  --allowed-hosts "*"
```

