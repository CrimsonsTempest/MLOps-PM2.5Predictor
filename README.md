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

Karena ingestion memberikan timestamp otomatis , dan dvc membutuhkan 1 nama file konstan, pertama masukan/copy isi data yang ingin di version ke dataset.csv , lalu track.

```bash
cp data/raw/rawdata_<TIMESTAMP>.csv data/raw/dataset.csv
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Track dataset version <TIMESTAMP>"
git tag v<TIMESTAMP>
git push && git push --tags
```

### 3. Data Version Switching

```bash
git tag --list                              # untuk melihat versi apa saja yang terdaftar 
git checkout v<TIMESTAMP> && dvc checkout   # pindah ke versi timestamp terpilih
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
