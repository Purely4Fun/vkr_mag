import os
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, abs as spark_abs, log1p, sin, cos, lit,
    lag, count, avg, stddev, when
)
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

SEED = 1212
MODEL_DIR = "./xgb_model"
AMOUNT_EPS = 1e-6

def get_spark():
    return (
        SparkSession.builder
        .appName("FraudBatchPreprocess")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

def load_data(spark, predictions_path, transactions_path, accounts_path):
    pred = spark.read.csv(predictions_path, header=True, inferSchema=True)
    tx = spark.read.csv(transactions_path, header=True, inferSchema=True)
    acc = spark.read.csv(accounts_path, header=True, inferSchema=True)
    return pred, tx, acc

def match_transactions(pred, tx):
    pred = pred.withColumnRenamed("transaction.SENDER_ACCOUNT_ID", "SENDER_ACCOUNT_ID") \
               .withColumnRenamed("transaction.RECEIVER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID") \
               .withColumnRenamed("transaction.TX_TYPE", "TX_TYPE") \
               .withColumnRenamed("transaction.TX_AMOUNT", "TX_AMOUNT") \
               .withColumnRenamed("transaction.TIMESTAMP", "TIMESTAMP")

    p = pred.alias("p")
    t = tx.alias("t")

    joined = p.join(
        t,
        on=(
            (col("p.SENDER_ACCOUNT_ID") == col("t.SENDER_ACCOUNT_ID")) &
            (col("p.RECEIVER_ACCOUNT_ID") == col("t.RECEIVER_ACCOUNT_ID")) &
            (col("p.TX_TYPE") == col("t.TX_TYPE")) &
            (spark_abs(col("p.TX_AMOUNT") - col("t.TX_AMOUNT")) < lit(AMOUNT_EPS)) &
            (col("p.TIMESTAMP") == col("t.TIMESTAMP"))
        ),
        how="inner"
    )

    joined = joined.select(
        *[col(f"p.{c}").alias(c) for c in pred.columns],
        col("t.IS_FRAUD").alias("TX_IS_FRAUD")
    )

    return joined

def add_time_features(df):
    seconds_in_day = 24 * 60 * 60

    return df.withColumn(
        "TX_TIME_SIN",
        sin((col("TIMESTAMP") % seconds_in_day) * (2 * np.pi / seconds_in_day))
    ).withColumn(
        "TX_TIME_COS",
        cos((col("TIMESTAMP") % seconds_in_day) * (2 * np.pi / seconds_in_day))
    ).withColumn(
        "TX_AMOUNT_LOG",
        log1p(col("TX_AMOUNT"))
    ).withColumn(
        "INIT_BALANCE_LOG",
        log1p(col("INIT_BALANCE"))
    )


def add_behavioral_features(df):
    w_sender = Window.partitionBy("SENDER_ACCOUNT_ID").orderBy("TIMESTAMP")
    w_sender_all = Window.partitionBy("SENDER_ACCOUNT_ID")

    df = df.withColumn("prev_ts", lag("TIMESTAMP").over(w_sender))

    df = df.withColumn(
        "time_since_last_tx",
        when(col("prev_ts").isNull(), 999999)
        .otherwise(col("TIMESTAMP") - col("prev_ts"))
    )

    w24 = Window.partitionBy("SENDER_ACCOUNT_ID") \
        .orderBy("TIMESTAMP") \
        .rangeBetween(-86400, 0)

    df = df.withColumn("cnt_last_24h", count("*").over(w24))

    df = df.withColumn(
        "sender_avg_amount",
        avg("TX_AMOUNT").over(w_sender_all)
    )

    df = df.withColumn(
        "sender_std_amount",
        stddev("TX_AMOUNT").over(w_sender_all)
    )

    return df

def build_pipeline(df, cat_cols, num_cols):
    stages = []

    for c in cat_cols:
        stages.append(
            StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        )

    stages.append(
        VectorAssembler(
            inputCols=num_cols + [f"{c}_idx" for c in cat_cols],
            outputCol="features"
        )
    )

    return Pipeline(stages=stages)

def oversample(X, y):
    print("\nBefore oversampling:")
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    smote = SMOTE(
        sampling_strategy=0.7, 
        random_state=SEED,
        k_neighbors=5
    )

    X_res, y_res = smote.fit_resample(X, y)

    print("\nAfter oversampling:")
    unique, counts = np.unique(y_res, return_counts=True)
    print(dict(zip(unique, counts)))

    return X_res, y_res

def main():
    spark = get_spark()

    pred, tx, acc = load_data(
        spark,
        "./dataset/predictions.csv",
        "./dataset/transactions.csv",
        "./dataset/accounts.csv",
    )

    df = match_transactions(pred, tx)

    a = acc.alias("a")

    df = df.alias("d").join(
        a,
        col("d.SENDER_ACCOUNT_ID") == col("a.ACCOUNT_ID"),
        how="left"
    )

    df = df.select(
        "d.*",
        col("a.INIT_BALANCE"),
        col("a.COUNTRY"),
        col("a.ACCOUNT_TYPE"),
        col("a.TX_BEHAVIOR_ID")
    )

    df = add_time_features(df)
    df = add_behavioral_features(df)

    df = df.withColumn("label", col("TX_IS_FRAUD").cast(IntegerType()))
    
    num_cols = [
        "TX_AMOUNT_LOG",
        "INIT_BALANCE_LOG",
        "TX_TIME_SIN",
        "TX_TIME_COS",
        "score",
        "time_since_last_tx",
        "cnt_last_24h",
        "sender_avg_amount",
    ]

    cat_cols = [
        "TX_TYPE",
        "ACCOUNT_TYPE",
        "COUNTRY",
    ]

    pipeline = build_pipeline(df, cat_cols, num_cols)
    model = pipeline.fit(df)
    model.write().overwrite().save(f"{MODEL_DIR}/pipeline")
    df_transformed = model.transform(df)

    X = np.array(
        df_transformed.select("features")
        .rdd.map(lambda r: r[0].toArray())
        .collect(),
        dtype=np.float32,
    )

    y = np.array(
        df_transformed.select("label")
        .rdd.map(lambda r: r[0])
        .collect(),
        dtype=np.int32,
    )

    X, y = oversample(X, y)

    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        X,
        y,
        train_size=0.9,
        stratify=y,
        random_state=SEED,
    )

    X_dev, X_test, y_dev, y_test = train_test_split(
        X_dev_test,
        y_dev_test,
        train_size=0.5,
        stratify=y_dev_test,
        random_state=SEED,
    )

    os.makedirs(f"{MODEL_DIR}/dataframes", exist_ok=True)
    np.save(f"{MODEL_DIR}/dataframes/X_train.npy", X_train)
    np.save(f"{MODEL_DIR}/dataframes/X_dev.npy", X_dev)
    np.save(f"{MODEL_DIR}/dataframes/X_test.npy", X_test)

    np.save(f"{MODEL_DIR}/dataframes/y_train.npy", y_train)
    np.save(f"{MODEL_DIR}/dataframes/y_dev.npy", y_dev)
    np.save(f"{MODEL_DIR}/dataframes/y_test.npy", y_test)

    print("Preprocessing finished")


if __name__ == "__main__":
    main()