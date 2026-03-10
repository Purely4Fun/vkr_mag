import uuid
import numpy as np
from datetime import datetime, timezone
from pyspark.sql import SparkSession
from xgb_model.model import (
    add_time_features_df,
    add_behavioral_features,
    get_model
)
from batch_processing.db_client import insert_async
from pyspark.sql.functions import col, broadcast
from batch_processing.s3_client import save_result
from batch_processing.config import (
    MODE,
    MODEL_DIR,
    S3_TRANSACTIONS_BUCKET,
    BATCH_SIZE
)
from pyspark.ml import PipelineModel

PIPELINE_PATH = "./xgb_model/pipeline"
_executor_pipeline = None
BATCH_S3_SIZE = 1000  

'''
export PYTHONPATH=/home/user/vkr/vkr_mag

spark-submit \
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,com.datastax.spark:spark-cassandra-connector_2.12:3.1.0 \
  --conf spark.hadoop.fs.s3a.access.key=minio \
  --conf spark.hadoop.fs.s3a.secret.key=minio123 \
  --conf spark.hadoop.fs.s3a.endpoint=http://localhost:9000 \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
  --conf spark.hadoop.fs.s3a.fast.upload=true \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.cassandra.connection.host=127.0.0.1 \
  --conf spark.cassandra.connection.port=9042 \
  batch_processing/run_batch.py

'''
def get_spark():
    return (
        SparkSession.builder
        .appName("FraudBatchProcessor")
        .config(
            "spark.jars.packages",
            ",".join([
                "org.apache.hadoop:hadoop-aws:3.3.4",
                "com.amazonaws:aws-java-sdk-bundle:1.12.262",
                "com.datastax.spark:spark-cassandra-connector_2.12:3.1.0"
            ])
        )
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minio")
        .config("spark.hadoop.fs.s3a.secret.key", "minio123")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.cassandra.connection.host", "127.0.0.1")
        .config("spark.cassandra.connection.port", "9042")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .getOrCreate()
    )

def get_executor_pipeline():

    global _executor_pipeline

    if _executor_pipeline is None:
        _executor_pipeline = PipelineModel.load(PIPELINE_PATH)

    return _executor_pipeline

def build_payload(tx):
    return {
        "SENDER_ACCOUNT_ID": tx["SENDER_ACCOUNT_ID"],
        "RECEIVER_ACCOUNT_ID": tx["RECEIVER_ACCOUNT_ID"],
        "TX_TYPE": tx["TX_TYPE"],
        "TX_AMOUNT": tx["tx_amount"],
        "TIMESTAMP": tx["timestamp"],
        "INIT_BALANCE": tx["init_balance"],
        "COUNTRY": tx["COUNTRY"],
        "ACCOUNT_TYPE": tx["ACCOUNT_TYPE"],
        "score": tx["score"],
    }


def load_accounts(spark):
    return (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="accounts", keyspace="batch_layer")
        .load()
        .select("account_id", "init_balance", "country", "account_type")
    )


def predict_partition(payloads, model, pipeline):
    if not payloads:
        return []

    spark = SparkSession.getActiveSession()

    df = spark.createDataFrame(payloads)
    df = add_time_features_df(df)

    df_transformed = pipeline.transform(df)

    X = np.array(
        df_transformed.select("features").rdd.map(lambda r: r[0].toArray()).collect(),
        dtype=np.float32
    )

    scores = model.predict_proba(X)[:, 1]

    return [
        {"fraud_probability": float(s), "is_fraud": bool(s > 0.5)}
        for s in scores
    ]

def predict_and_save(features, meta, model):

    X = np.array(features, dtype=np.float32)
    scores = model.predict_proba(X)[:, 1]

    payloads = []
    results = []
    s3_batch = []

    for row, score in zip(meta, scores):

        payload = build_payload(row)

        result = {
            "fraud_probability": float(score),
            "is_fraud": bool(score > 0.5)
        }

        s3_item = {
            "tx_id": row["tx_id"],
            "sender": row["SENDER_ACCOUNT_ID"],
            "receiver": row["RECEIVER_ACCOUNT_ID"],
            "amount": row["tx_amount"]
        }

        payloads.append(payload)
        results.append(result)
        s3_batch.append(s3_item)

        print({
            "tx_id": row["tx_id"],
            "fraud_probability": float(score),
            "is_fraud": bool(score > 0.5)
        })

    save_results(payloads, meta, s3_batch, results)

def process_partition(rows, bc_model):

    model = bc_model.value

    batch_features = []
    meta = []

    for row in rows:

        meta.append(row)

        batch_features.append(
            row["features"].toArray()
        )

        if len(batch_features) >= BATCH_SIZE:

            predict_and_save(batch_features, meta, model)

            batch_features.clear()
            meta.clear()

    if batch_features:
        predict_and_save(batch_features, meta, model)


def save_results(payloads, meta, s3_batch, results):

    cassandra_rows = []

    for payload, tx, result, s3_item in zip(payloads, meta, results, s3_batch):

        ts = datetime.now(timezone.utc)
        tx_id = uuid.UUID(tx["tx_id"])

        cassandra_row = {
            "tx": (
                tx_id, ts, payload["SENDER_ACCOUNT_ID"], payload["RECEIVER_ACCOUNT_ID"],
                payload["TX_TYPE"], payload["TX_AMOUNT"], result["is_fraud"],
                tx["score"], tx["threshold"], None
            ),
            "sender": (
                payload["SENDER_ACCOUNT_ID"], ts, tx_id, payload["RECEIVER_ACCOUNT_ID"],
                payload["TX_TYPE"], payload["TX_AMOUNT"], result["is_fraud"],
                tx["score"], tx["threshold"], None
            ),
            "receiver": (
                payload["RECEIVER_ACCOUNT_ID"], ts, tx_id, payload["SENDER_ACCOUNT_ID"],
                payload["TX_TYPE"], payload["TX_AMOUNT"], result["is_fraud"],
                tx["score"], tx["threshold"], None
            )
        }

        cassandra_rows.append(cassandra_row)

        s3_item.update({
            "fraud_probability": result["fraud_probability"],
            "is_fraud": result["is_fraud"],
            "timestamp": ts.isoformat()
        })

    insert_async(cassandra_rows)

    for batch_start in range(0, len(s3_batch), BATCH_S3_SIZE):

        batch_to_save = s3_batch[batch_start:batch_start + BATCH_S3_SIZE]

        object_name = f"{uuid.uuid4()}.json"

        save_result(object_name, batch_to_save)

        print(f"S3: saved batch {object_name} ({len(batch_to_save)} transactions)")


def run_s3_mode(spark):

    model = get_model()
    bc_model = spark.sparkContext.broadcast(model)

    pipeline = PipelineModel.load(PIPELINE_PATH)

    tx_df = spark.read.json(f"s3a://{S3_TRANSACTIONS_BUCKET}/")

    accounts_df = load_accounts(spark)

    tx_df = tx_df.select(
        col("tx_id"), col("score"), col("threshold"),
        col("transaction.SENDER_ACCOUNT_ID").alias("sender_id"),
        col("transaction.RECEIVER_ACCOUNT_ID").alias("receiver_id"),
        col("transaction.TX_TYPE").alias("tx_type"),
        col("transaction.TX_AMOUNT").alias("tx_amount"),
        col("transaction.TIMESTAMP").alias("timestamp")
    )

    df = tx_df.join(
        broadcast(accounts_df),
        tx_df.sender_id == accounts_df.account_id,
        "left"
    )

    df = df.select(
        "tx_id","score","threshold","sender_id","receiver_id",
        "tx_type","tx_amount","timestamp",
        "init_balance","country","account_type"
    )

    df = df.withColumnRenamed("sender_id", "SENDER_ACCOUNT_ID")
    df = df.withColumnRenamed("receiver_id", "RECEIVER_ACCOUNT_ID")
    df = df.withColumnRenamed("tx_type", "TX_TYPE")
    df = df.withColumnRenamed("country", "COUNTRY")
    df = df.withColumnRenamed("account_type", "ACCOUNT_TYPE")

    df = add_time_features_df(df)

    df = add_behavioral_features(df)

    df = pipeline.transform(df)

    df = df.repartition(100)

    df.foreachPartition(
        lambda rows: process_partition(rows, bc_model)
    )


def run_local_mode():

    model = get_model()

    X = np.load(f"{MODEL_DIR}/dataframes/main_X.npy")

    for i in range(0, len(X), BATCH_SIZE):

        batch = X[i:i + BATCH_SIZE]

        scores = model.predict_proba(batch)[:, 1]

        for s in scores:
            print({
                "fraud_probability": float(s),
                "is_fraud": bool(s > 0.5)
            })


def main():

    spark = get_spark()

    if MODE == "s3":
        run_s3_mode(spark)
    else:
        run_local_mode()


if __name__ == "__main__":
    main()
