'''
message example:
{"SENDER_ACCOUNT_ID":9876,"RECEIVER_ACCOUNT_ID":6789,"TX_TYPE":"TRANSFER","TX_AMOUNT":333.44,"TIMESTAMP":1708001234}

docker exec -it kafka bash -c "export PATH=\$PATH:/opt/kafka/bin && kafka-console-producer.sh --topic transactions --bootstrap-server kafka:9092"

'''
import json
from datetime import datetime, timezone
from kafka import KafkaConsumer
from api.entity import get_account, Transaction, save_transaction
from gam_model.model import predict, build_features
from s3.storage import save_transaction_to_s3
from kafka_adapters.producer import send_prediction
import time

KAFKA_BROKER = "localhost:9094"
INPUT_TOPIC = "transactions"
CONSUMER_GROUP = "fraud-speed-layer"

consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id=CONSUMER_GROUP
)


def process_transaction(message: dict) -> dict:
    tx = Transaction(**message)

    account = get_account(tx.SENDER_ACCOUNT_ID)
    if not account:
        raise ValueError(f"Account {tx.SENDER_ACCOUNT_ID} not found")

    enriched_payload = {
        "account": account,
        "transaction": tx.model_dump()
    }
    enriched_payload = build_features(enriched_payload)
    is_fraud, score, threshold = predict(enriched_payload)

    tx_id, alert_id = save_transaction(tx, is_fraud, score)
    result_payload = {
        "tx_id": str(tx_id),
        "alert_id": str(alert_id) if alert_id else None,
        "transaction": tx.model_dump(),
        "fraud_prediction": bool(is_fraud),
        "score": float(score),
        "threshold": float(threshold),
        "prediction_timestamp": datetime.now(timezone.utc).isoformat()
    }
    save_transaction_to_s3(str(tx_id), result_payload)

    return result_payload


def start():
    print("Init")
    for msg in consumer:
        try:
            message = msg.value
            start_time = time.time()
            result = process_transaction(message)
            latency = time.time() - start_time
            print(f"LATENCY: {latency:.6f}")
            send_prediction(result)

        except Exception as e:
            print(f"Error processing message: {e}")

if __name__ == "__main__":
    start()
