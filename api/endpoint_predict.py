from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
from api.entity import get_account, Transaction
from gam_model.model import predict, build_features  

app = FastAPI(title="Financial Monitoring Speed Layer")

@app.post("/predict")
def score_transaction(tx: Transaction):
    account = get_account(tx.SENDER_ACCOUNT_ID)
    if not account:
        raise HTTPException(status_code=400, detail="Account not found")

    enriched_payload = {
        "account": account,
        "transaction": tx.model_dump()
    }

    enriched_payload = build_features(enriched_payload)

    is_fraud, score, threshold = predict(enriched_payload)
    response_time = datetime.now(timezone.utc).isoformat()

    return {
        "fraud_prediction": bool(is_fraud),
        "score": float(score),
        "threshold": float(threshold),
        "prediction_timestamp": response_time
    }
