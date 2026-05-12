# Fintech Credit Risk Engine

Büyük Veri Analizine Giriş dersi dönem projesi.

**Veri Seti:** Lending Club Loan Data (~2.2M kayıt)  
**Problem:** Binary Classification — Kredi temerrüt tahmini (loan_status)

## Mimari

```
Kafka Producer → Kafka Broker → Spark Structured Streaming → Delta Lake (Bronze/Silver/Gold) → ML (MLflow) → Dashboard
```

## Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Konteynerizasyon | Docker, Docker Compose |
| Veri Üretimi | Apache Kafka + Python Producer |
| Veri İşleme | Apache Spark (PySpark) |
| Depolama | Delta Lake |
| Makine Öğrenmesi | Spark MLlib + MLflow |
| Görselleştirme | Matplotlib |

---

## Kurulum

### Gereksinimler
- Docker Desktop
- Python 3.11+
- GNU Make (`choco install make` veya Git Bash ile gelir)

### 1. Repoyu klonla
```bash
git clone <repo-url>
cd fintech-credit-risk-engine
```

### 2. Python kütüphanelerini kur (yerel geliştirme için)
```bash
make install
# veya
pip install -r requirements.txt
```

### 3. Veri setini hazırla
Kaggle'dan indir: https://www.kaggle.com/datasets/wordsforthewise/lending-club

Zip'i çıkardıktan sonra `data/` klasörü şöyle görünmeli:
```
data/
  accepted_2007_to_2018q4.csv/
    accepted_2007_to_2018Q4.csv   ← bu dosya kullanılıyor
  rejected_2007_to_2018q4.csv/
    rejected_2007_to_2018Q4.csv
```

### 4. Servisleri başlat
```bash
make start
# veya adım adım:
make build
make up
```

### 5. Spark Streaming job'ı çalıştır
```bash
make streaming
```

---

## Servis Adresleri

| Servis | URL |
|--------|-----|
| Kafka UI | http://localhost:8080 |
| Spark UI | http://localhost:4040 |
| MLflow UI | http://localhost:5000 |

---

## Makefile Komutları

```bash
make help          # tüm komutları listele
make install       # Python kütüphanelerini kur
make build         # Docker image'larını build et
make up            # servisleri arka planda başlat
make up-logs       # servisleri loglarla başlat
make down          # servisleri durdur
make restart       # servisleri yeniden başlat
make streaming     # Spark Streaming job'ı çalıştır
make logs          # tüm logları izle
make logs-producer # sadece producer logları
make clean-data    # delta_lake ve mlruns klasörlerini sil
make clean-docker  # Docker container/image'larını temizle
```

---

## Proje Adımları

| Adım | Branch | İçerik |
|------|--------|--------|
| 1-2-3 | feature/data-ingestion-pipeline | Docker + Kafka + Spark + Delta Lake |
| 4 | feature/eda | Keşifsel Veri Analizi |
| 5 | feature/feature-engineering | Özellik Mühendisliği |
| 6 | feature/ml-mlflow | 5 ML Modeli + MLflow |
| 7 | feature/dashboard | Görselleştirme |

## Notebooklar

- `notebooks/step3_spark_streaming.ipynb` — Spark Streaming + Delta Lake (Bronze/Silver/Gold)
- `notebooks/step4_eda.ipynb` — EDA
- `notebooks/step5_feature_engineering.ipynb` — Feature Engineering
- `notebooks/step6_ml_mlflow.ipynb` — ML Modelleri
- `notebooks/step7_dashboard.ipynb` — Dashboard
- `README_STEP4_STEP5.md` — Step 4 ve Step 5 icin detayli calistirma ve teslim rehberi

## Klasör Yapısı

```
fintech-credit-risk-engine/
├── Makefile                          # Kısayol komutları
├── requirements.txt                  # Tüm Python bağımlılıkları
├── docker-compose.yml                # Servis tanımları
├── dockerfiles/
│   ├── Dockerfile.producer           # Producer container
│   ├── Dockerfile.spark              # Spark container
│   ├── requirements-producer.txt     # Producer bağımlılıkları
│   └── requirements-spark.txt        # Spark bağımlılıkları
├── producer/
│   └── kafka_producer.py             # CSV → Kafka producer
├── spark/
│   └── streaming_job.py              # Kafka → Delta Lake streaming
├── notebooks/                        # Adım adım notebook'lar
├── data/                             # Veri seti (git'e eklenmez)
├── delta_lake/                       # Delta Lake depolama (git'e eklenmez)
├── mlruns/                           # MLflow deneyleri (git'e eklenmez)
└── output/                           # Görseller ve raporlar
```
