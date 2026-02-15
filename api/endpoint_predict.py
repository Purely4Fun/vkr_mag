'''
Request example:
uvicorn api.endpoint_predict:app --reload --loop asyncio

    curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
            "SENDER_ACCOUNT_ID": 1234,
            "RECEIVER_ACCOUNT_ID": 5678,
            "TX_TYPE": "TRANSFER",
            "TX_AMOUNT": 123.45,
            "TIMESTAMP": 1708001234
        }'
'''
from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
from api.entity import get_account, Transaction, save_transaction
from gam_model.model import predict, build_features  
from s3.storage import save_transaction_to_s3

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
    tx_id, alert_id = save_transaction(tx, is_fraud, score)

    s3_payload = {
        "tx_id": str(tx_id),
        "alert_id": str(alert_id) if alert_id else None,
        "transaction": tx.model_dump(),
        "fraud_prediction": bool(is_fraud),
        "score": float(score),
        "threshold": float(threshold),
        "prediction_timestamp": datetime.now(timezone.utc).isoformat()
    }

    save_transaction_to_s3(str(tx_id), s3_payload)

    return s3_payload