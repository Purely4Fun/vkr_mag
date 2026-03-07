import pickle
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, sin, cos
from pyspark.ml import PipelineModel

MODEL_DIR = "./xgb_model"

MODEL_PATH = f"{MODEL_DIR}/xgb_model.pickle"
PIPELINE_PATH = f"{MODEL_DIR}/pipeline"

SECONDS_IN_DAY = 24 * 60 * 60


def get_spark():
    return (
        SparkSession.builder
        .appName("FraudInference")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


spark = get_spark()


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_pipeline():
    return PipelineModel.load(PIPELINE_PATH)


model = load_model()
pipeline = load_pipeline()


def add_time_features(df):

    return (
        df.withColumn(
            "TX_TIME_SIN",
            sin((col("TIMESTAMP") % SECONDS_IN_DAY) * (2 * np.pi / SECONDS_IN_DAY)),
        )
        .withColumn(
            "TX_TIME_COS",
            cos((col("TIMESTAMP") % SECONDS_IN_DAY) * (2 * np.pi / SECONDS_IN_DAY)),
        )
        .withColumn("TX_AMOUNT_LOG", log1p(col("TX_AMOUNT")))
        .withColumn("INIT_BALANCE_LOG", log1p(col("INIT_BALANCE")))
    )


def preprocess(df):

    df = add_time_features(df)

    df_transformed = pipeline.transform(df)

    X = np.array(
        df_transformed.select("features")
        .rdd.map(lambda r: r[0].toArray())
        .collect(),
        dtype=np.float32,
    )

    return X


def predict(payload):

    df = spark.createDataFrame([payload])

    X = preprocess(df)

    score = model.predict_proba(X)[0, 1]

    return {
        "fraud_probability": float(score),
        "is_fraud": bool(score > 0.5),
    }


def predict_batch(payloads):

    df = spark.createDataFrame(payloads)

    X = preprocess(df)

    scores = model.predict_proba(X)[:, 1]

    results = []

    for s in scores:
        results.append(
            {
                "fraud_probability": float(s),
                "is_fraud": bool(s > 0.5),
            }
        )

    return results