import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp, when, regexp_replace, trim
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "loan-events")
DELTA_BASE = os.getenv("DELTA_BASE", "/app/delta_lake")

BRONZE = f"{DELTA_BASE}/bronze/loans"
SILVER = f"{DELTA_BASE}/silver/loans"
GOLD   = f"{DELTA_BASE}/gold/loans"

CHECKPOINT_BRONZE = f"{DELTA_BASE}/checkpoints/bronze"
CHECKPOINT_SILVER = f"{DELTA_BASE}/checkpoints/silver"

LOAN_SCHEMA = StructType([
    StructField("timestamp",           StringType()),
    StructField("loan_id",             StringType()),
    StructField("event_type",          StringType()),
    StructField("loan_amnt",           DoubleType()),
    StructField("funded_amnt",         DoubleType()),
    StructField("term",                StringType()),
    StructField("int_rate",            StringType()),
    StructField("installment",         DoubleType()),
    StructField("grade",               StringType()),
    StructField("sub_grade",           StringType()),
    StructField("emp_length",          StringType()),
    StructField("home_ownership",      StringType()),
    StructField("annual_inc",          DoubleType()),
    StructField("verification_status", StringType()),
    StructField("issue_d",             StringType()),
    StructField("loan_status",         StringType()),
    StructField("purpose",             StringType()),
    StructField("dti",                 DoubleType()),
    StructField("delinq_2yrs",         DoubleType()),
    StructField("open_acc",            DoubleType()),
    StructField("pub_rec",             DoubleType()),
    StructField("revol_bal",           DoubleType()),
    StructField("revol_util",          StringType()),
    StructField("total_acc",           DoubleType()),
])


JARS = ",".join([
    "/app/jars/delta-spark.jar",
    "/app/jars/delta-storage.jar",
    "/app/jars/antlr4-runtime.jar",
    "/app/jars/spark-sql-kafka.jar",
    "/app/jars/spark-token-provider-kafka.jar",
    "/app/jars/kafka-clients.jar",
    "/app/jars/lz4-java.jar",
    "/app/jars/snappy-java.jar",
    "/app/jars/slf4j-api.jar",
    "/app/jars/hadoop-client-runtime.jar",
    "/app/jars/hadoop-client-api.jar",
    "/app/jars/commons-logging.jar",
    "/app/jars/jsr305.jar",
    "/app/jars/commons-pool2.jar",
])


def create_spark_session():
    return (
        SparkSession.builder
        .appName("CreditRiskStreaming")
        .master("local[*]")
        .config("spark.jars", JARS)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def read_kafka(spark):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
        .selectExpr("CAST(value AS STRING) as raw_json")
        .select(from_json(col("raw_json"), LOAN_SCHEMA).alias("data"))
        .select("data.*")
    )


def write_bronze(df):
    return (
        df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_BRONZE)
        .start(BRONZE)
    )


def clean_silver(df):
    int_rate_clean = regexp_replace(trim(col("int_rate")), "%", "").cast("double")
    revol_util_clean = regexp_replace(trim(col("revol_util")), "%", "").cast("double")

    is_default = when(
        col("loan_status").isin("Charged Off", "Default", "Late (31-120 days)"), 1
    ).otherwise(0)

    return (
        df
        .filter(col("loan_id").isNotNull())
        .filter(col("loan_amnt") > 0)
        .filter(col("annual_inc") > 0)
        .filter(col("loan_status").isNotNull())
        .withColumn("event_time", to_timestamp(col("timestamp")))
        .withWatermark("event_time", "1 hour")
        .dropDuplicates(["loan_id", "event_time"])
        .withColumn("int_rate_pct",   int_rate_clean)
        .withColumn("revol_util_pct", revol_util_clean)
        .withColumn("is_default",     is_default)
        .drop("int_rate", "revol_util", "timestamp")
    )


def write_silver(df):
    return (
        df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_SILVER)
        .start(SILVER)
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    raw_df = read_kafka(spark)

    bronze_query = write_bronze(raw_df)

    silver_df = clean_silver(raw_df)
    silver_query = write_silver(silver_df)

    bronze_query.awaitTermination()
    silver_query.awaitTermination()


if __name__ == "__main__":
    main()
