import os
import json
import time
import logging
import pandas as pd
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "loan-events")
MSG_PER_SECOND = int(os.getenv("MSG_PER_SECOND", "50"))
DATA_FILE = os.getenv("DATA_FILE", "data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv")

COLUMNS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose", "dti",
    "delinq_2yrs", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"
]


def wait_for_kafka(retries=10, delay=5):
    for attempt in range(retries):
        try:
            producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
            logger.info("Kafka bağlantısı başarılı.")
            return producer
        except NoBrokersAvailable:
            logger.warning(f"Kafka hazır değil, bekleniyor... ({attempt + 1}/{retries})")
            time.sleep(delay)
    raise RuntimeError("Kafka'ya bağlanılamadı.")


def build_message(row: pd.Series) -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "loan_id": str(row.get("id", "")),
        "event_type": "loan_application",
        "loan_amnt": float(row.get("loan_amnt", 0) or 0),
        "funded_amnt": float(row.get("funded_amnt", 0) or 0),
        "term": str(row.get("term", "")),
        "int_rate": str(row.get("int_rate", "")),
        "installment": float(row.get("installment", 0) or 0),
        "grade": str(row.get("grade", "")),
        "sub_grade": str(row.get("sub_grade", "")),
        "emp_length": str(row.get("emp_length", "")),
        "home_ownership": str(row.get("home_ownership", "")),
        "annual_inc": float(row.get("annual_inc", 0) or 0),
        "verification_status": str(row.get("verification_status", "")),
        "issue_d": str(row.get("issue_d", "")),
        "loan_status": str(row.get("loan_status", "")),
        "purpose": str(row.get("purpose", "")),
        "dti": float(row.get("dti", 0) or 0),
        "delinq_2yrs": float(row.get("delinq_2yrs", 0) or 0),
        "open_acc": float(row.get("open_acc", 0) or 0),
        "pub_rec": float(row.get("pub_rec", 0) or 0),
        "revol_bal": float(row.get("revol_bal", 0) or 0),
        "revol_util": str(row.get("revol_util", "")),
        "total_acc": float(row.get("total_acc", 0) or 0),
    }


def main():
    logger.info(f"Veri dosyası okunuyor: {DATA_FILE}")

    try:
        df = pd.read_csv(DATA_FILE, usecols=lambda c: c in COLUMNS, low_memory=False)
    except FileNotFoundError:
        logger.error(f"Veri dosyası bulunamadı: {DATA_FILE}")
        raise

    logger.info(f"Toplam kayıt sayısı: {len(df):,}")

    producer = wait_for_kafka()
    delay = 1.0 / MSG_PER_SECOND

    sent = 0
    for _, row in df.iterrows():
        message = build_message(row)
        producer.send(
            KAFKA_TOPIC,
            value=json.dumps(message).encode("utf-8"),
            key=str(message["loan_id"]).encode("utf-8")
        )
        sent += 1

        if sent % 1000 == 0:
            producer.flush()
            logger.info(f"Gönderilen mesaj: {sent:,} / {len(df):,}")

        time.sleep(delay)

    producer.flush()
    logger.info(f"Tamamlandı. Toplam gönderilen: {sent:,} mesaj.")


if __name__ == "__main__":
    main()
