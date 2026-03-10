import pickle
import numpy as np
from pyspark.sql.functions import col, log1p, sin, cos
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, count, avg, stddev, when

MODEL_DIR = "./xgb_model"
MODEL_PATH = f"{MODEL_DIR}/xgb_model.pickle"
PIPELINE_PATH = f"{MODEL_DIR}/pipeline"
SECONDS_IN_DAY = 24 * 60 * 60

_model = None
_pipeline = None

def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = PipelineModel.load(PIPELINE_PATH)
    return _pipeline

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

    df = df.withColumn(
        "cnt_last_24h",
        count("*").over(w24)
    )

    df = df.withColumn(
        "sender_avg_amount",
        avg("TX_AMOUNT").over(w_sender_all)
    )

    df = df.withColumn(
        "sender_std_amount",
        stddev("TX_AMOUNT").over(w_sender_all)
    )

    return df

def add_time_features_df(df):
    return (
        df.withColumn(
            "TX_TIME_SIN",
            sin((col("TIMESTAMP") % SECONDS_IN_DAY) * (2 * np.pi / SECONDS_IN_DAY))
        )
        .withColumn(
            "TX_TIME_COS",
            cos((col("TIMESTAMP") % SECONDS_IN_DAY) * (2 * np.pi / SECONDS_IN_DAY))
        )
        .withColumn("TX_AMOUNT_LOG", log1p(col("TX_AMOUNT")))
        .withColumn("INIT_BALANCE_LOG", log1p(col("INIT_BALANCE")))
    )

def preprocess_numpy(payloads, pipeline):
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(payloads)
    df = add_time_features_df(df)
    df_transformed = pipeline.transform(df)
    X = np.array(df_transformed.select("features").rdd.map(lambda r: r[0].toArray()).collect(), dtype=np.float32)
    return X

def predict_numpy(payloads, model, pipeline):
    if not payloads:
        return []
    X = preprocess_numpy(payloads, pipeline)
    scores = model.predict_proba(X)[:, 1]
    return [{"fraud_probability": float(s), "is_fraud": bool(s > 0.5)} for s in scores]