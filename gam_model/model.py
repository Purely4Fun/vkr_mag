import numpy as np
import joblib
from datetime import datetime, timezone

SECONDS_IN_DAY = 24 * 60 * 60
TWO_PI_OVER_DAY = 2 * np.pi / SECONDS_IN_DAY

MODEL_NAME = "GAM_v1"
MODEL_DIR = "./gam_model/joblibs"

gam = joblib.load(f"{MODEL_DIR}/gam_model.joblib")
threshold = joblib.load(f"{MODEL_DIR}/decision_threshold.joblib")
cat_mapping = joblib.load(f"{MODEL_DIR}/cat_mapping.joblib")

NUM_FEATURES_DIM = 4
CAT_FEATURES_ORDER = [
    "TX_TYPE",
    "ACCOUNT_TYPE",
    "COUNTRY",
    "TX_BEHAVIOR_ID",
]

CAT_DIMS = [len(cat_mapping[f]) for f in CAT_FEATURES_ORDER]
CAT_TOTAL_DIM = sum(CAT_DIMS)

TOTAL_DIM = NUM_FEATURES_DIM + CAT_TOTAL_DIM

def build_features(payload):

    X = np.zeros((1, TOTAL_DIM), dtype=np.float32)

    account = payload["account"]
    tx = payload["transaction"]

    X[0, 0] = np.log1p(float(tx["TX_AMOUNT"]))
    X[0, 1] = np.log1p(float(account["INIT_BALANCE"]))

    t = int(tx["TIMESTAMP"]) % SECONDS_IN_DAY
    X[0, 2] = np.sin(t * TWO_PI_OVER_DAY)
    X[0, 3] = np.cos(t * TWO_PI_OVER_DAY)

    offset = NUM_FEATURES_DIM

    values = [
        tx["TX_TYPE"],
        account["ACCOUNT_TYPE"],
        account["COUNTRY"],
        account["TX_BEHAVIOR_ID"],
    ]

    for feature, value, dim in zip(CAT_FEATURES_ORDER, values, CAT_DIMS):
        mapping = cat_mapping[feature]
        if value in mapping:
            X[0, offset + mapping[value]] = 1.0
        offset += dim

    return X

def predict_proba(payload):
    X = build_features(payload)
    return float(gam.predict_proba(X)[0])


def predict(payload):
    score = predict_proba(payload)
    is_fraud = int(score >= threshold)

    return {
        "is_fraud": is_fraud,
        "fraud_score": round(score, 6),
        "threshold": round(float(threshold), 6),
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }