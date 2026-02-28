import numpy as np
import joblib

MODEL_DIR = "./batch_xgb/joblibs"

model = joblib.load(f"{MODEL_DIR}/xgb_model.joblib")
threshold = joblib.load(f"{MODEL_DIR}/decision_threshold.joblib")
encoder = joblib.load(f"{MODEL_DIR}/encoder.joblib")

SECONDS_IN_DAY = 24 * 60 * 60
TWO_PI_OVER_DAY = 2 * np.pi / SECONDS_IN_DAY


def build_features(payload):
    account = payload["account"]
    tx = payload["transaction"]
    gam_score = payload["gam_score"]

    t = int(tx["TIMESTAMP"]) % SECONDS_IN_DAY

    X_num = np.array([[
        np.log1p(float(tx["TX_AMOUNT"])),
        np.log1p(float(account["INIT_BALANCE"])),
        np.sin(t * TWO_PI_OVER_DAY),
        np.cos(t * TWO_PI_OVER_DAY),
        gam_score,
    ]])
    
    cat = [[
        tx["TX_TYPE"],
        account["ACCOUNT_TYPE"],
        account["COUNTRY"],
        account["TX_BEHAVIOR_ID"],
    ]]

    X_cat = encoder.transform(cat)

    return np.hstack([X_num, X_cat])


def predict(payload):
    X = build_features(payload)
    score = float(model.predict_proba(X)[0, 1])
    return score >= threshold, score, threshold