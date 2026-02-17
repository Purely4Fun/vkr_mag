import json
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9094"
OUTPUT_TOPIC = "results"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


def send_prediction(payload: dict):
    producer.send(OUTPUT_TOPIC, value=payload)
    producer.flush()