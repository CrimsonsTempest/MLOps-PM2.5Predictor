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
├── .dvc/
├── config/
├── data/
│   ├── raw/              # rawdata_*.csv dari ingestion (append-only, versioned)
│   │   └── dataset.csv.dvc   # pointer DVC ke dataset aktif
│   └── preprocessed/     # cleandata_*.csv dari preprocessing (versioned)
├── models/
├── notebooks/
├── src/
│   └── data/
│       ├── ingest_data.py  # fetch data mentah dari OpenAQ API
│       └── preprocess.py   # cleaning: timestamp, missing values, deduplikasi
└── README.md
```

##  Menjalankan Pipeline Data

```bash
# 1. Ingestion: tarik data mentah dari OpenAQ (butuh API_KEY di .env)
python src/data/ingest_data.py

# 2. Preprocessing: bersihkan snapshot mentah terbaru
python src/data/preprocess.py

# Cek dulu tanpa menulis file:
python src/data/preprocess.py --dry-run
```

## DVC

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

Karena ingestion memberikan timestamp otomatis , dan dvc membutuhkan 1 nama file konstan, ubah versi yang ingin di version menjadi dataset.csv , lalu

```bash
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
