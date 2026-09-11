# MLOps-PM2.5Predictor

Proyek ini bertujuan untuk membuat sebuah sistem machine learning untuk melakukan prediksi pada nilai pm25 di udara. Dilengkapi dengan proses retraining setiap terjadinya drift yang akan di detect oleh software monitoring, dengan menggunakan apache airflow untuk melakukan fetching data.

berikut adalah flow dari system yang akan dibuat


1. **Ingestion**: fetch data dengan apache airflow, versioning dengan Data Versioning Control ( DVC )
2. **Preprocessing**: Transformasi data,Missing Values handling, Feature Engineering dilakukan dengan pandas dan numpy
3. **Model Training** : training dengan model yang akan digunakan (pending penentuan antara xgboost dan randomforest)
4. **Model Registry & Evaluation:** : model dilakukan comparison antara model yang running dan model yang baru saja di train dengan logika cham[pion vs challanger, model disimpan artefaknya dengan registry MlFlow
5. **API Serving:** Menyajikan atau serving prediksi dari pm2.5 dengan model yang dipilih melalui REST API menggunakan FastAPI.
6. **Continuous Monitoring:** Mengawasi performa metrik seperti MAE dan mendeteksi data drift untuk memicu proses retraining secara otomatis jika diperlukan.

## 🛠️ Tech Stack
* **Orchestration:** Apache Airflow
* **Data Versioning:** DVC
* **Machine Learning:** XGBoost, Scikit-Learn
* **Model Tracking:** MLflow
* **API Serving:** FastAPI
* **Monitoring:** Prometheus & Grafana 

## 📂 Struktur Direktori
```text
.
├── config/                  
├── data/   
├── models/  
├── notebooks/            
├── src/                            
└── README.md