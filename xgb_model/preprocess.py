import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs as spark_abs, log1p, sin, cos, lit
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType

MODEL_DIR = "./xgb_model"
RANDOM_STATE = 42
TOP_K_COUNTRIES = 20
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

def deduplicate_columns(df):
    seen = {}
    new_names = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_names.append(c)
        else:
            seen[c] += 1
            new_names.append(f"{c}_dup{seen[c]}")
    return df.toDF(*new_names)

def match_transactions(pred, tx):
    pred = pred.withColumnRenamed("transaction.TX_ID", "TX_ID_pred") \
               .withColumnRenamed("transaction.SENDER_ACCOUNT_ID", "SENDER_ACCOUNT_ID") \
               .withColumnRenamed("transaction.RECEIVER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID") \
               .withColumnRenamed("transaction.TX_TYPE", "TX_TYPE") \
               .withColumnRenamed("transaction.TX_AMOUNT", "TX_AMOUNT") \
               .withColumnRenamed("transaction.TIMESTAMP", "TIMESTAMP")

    tx = tx.withColumnRenamed("TX_ID", "TX_ID_tx")
    print("pred count:", pred.count())
    print("tx count:", tx.count())

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

    t_unique_cols = [c for c in t.columns if c not in p.columns]

    joined = joined.select(
        *[col(f"p.{c}") for c in p.columns],
        *[col(f"t.{c}") for c in t_unique_cols],
    )
    print("joined count:", joined.count())

    return joined

def add_time_features(df):
    seconds_in_day = 24 * 60 * 60
    return df.withColumn("TX_TIME_SIN", sin((col("TIMESTAMP") % seconds_in_day) * (2 * np.pi / seconds_in_day))) \
             .withColumn("TX_TIME_COS", cos((col("TIMESTAMP") % seconds_in_day) * (2 * np.pi / seconds_in_day))) \
             .withColumn("TX_AMOUNT_LOG", log1p(col("TX_AMOUNT"))) \
             .withColumn("INIT_BALANCE_LOG", log1p(col("INIT_BALANCE")))

def build_pipeline(df, cat_cols, num_cols):
    stages = []

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols]

    stages.extend(indexers)
    stages.extend(encoders)

    assembler = VectorAssembler(
        inputCols=num_cols + [f"{c}_ohe" for c in cat_cols],
        outputCol="features"
    )
    stages.append(assembler)

    pipeline = Pipeline(stages=stages)
    return pipeline

def main():
    spark = get_spark()

    pred, tx, acc = load_data(
        spark,
        "./dataset/predictions.csv",
        "./dataset/transactions.csv",
        "./dataset/accounts.csv",
    )
    pred = pred.withColumnRenamed("IS_FRAUD", "IS_FRAUD_pred")
    tx = tx.withColumnRenamed("IS_FRAUD", "IS_FRAUD_tx")
    acc = acc.withColumnRenamed("IS_FRAUD", "IS_FRAUD_acc")

    print("Matching transactions...")
    df = match_transactions(pred, tx) 

    print("Joining accounts...") 
    df = df.join( acc.alias("a"), 
                 col("SENDER_ACCOUNT_ID") == col("a.ACCOUNT_ID"), 
                 how="left" ).drop("ACCOUNT_ID") 
    
    df = add_time_features(df) 
    df = df.withColumn(
        "fraud_prediction", 
        col("fraud_prediction").cast(IntegerType())
    )
    num_cols = ["TX_AMOUNT_LOG", "INIT_BALANCE_LOG", "TX_TIME_SIN", "TX_TIME_COS", "score"] 
    cat_cols = ["TX_TYPE", "ACCOUNT_TYPE", "COUNTRY", "TX_BEHAVIOR_ID", "fraud_prediction"] 
    pipeline = build_pipeline(df, cat_cols, num_cols) 
    model = pipeline.fit(df)

    df_transformed = model.transform(df) 

    df_transformed = deduplicate_columns(df_transformed)
    df_transformed = df_transformed.drop("alert_id")
    print("joined columns:", df_transformed.columns)
    
    os.makedirs(f"{MODEL_DIR}/dataframes", exist_ok=True) 
    y = df_transformed.select("IS_FRAUD_tx").rdd.map(lambda r: r[0]).collect() 
    np.save(f"{MODEL_DIR}/dataframes/y.npy", np.array(y, dtype=np.int32)) 
    features = df_transformed.select("features").rdd.map(lambda r: r[0].toArray()).collect() 
    
    np.save(f"{MODEL_DIR}/dataframes/X.npy", np.array(features, dtype=np.float32)) 
    
    df_transformed.write.mode("overwrite").save(f"{MODEL_DIR}/transformed_df") 
    
    pipeline_path = f"{MODEL_DIR}/pipeline_spark" 
    model.write().overwrite().save(pipeline_path) 
    print("Batch preprocessing finished")

if __name__ == "__main__":
    main()