import json
import time
import random
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9094",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def generate_tx():
    return {
        "SENDER_ACCOUNT_ID": random.randint(1, 10000),
        "RECEIVER_ACCOUNT_ID": random.randint(1, 10000),
        "TX_TYPE": "TRANSFER",
        "TX_AMOUNT": random.random() * 1000,
        "TIMESTAMP": int(time.time())
    }

RATE = 500       
DURATION = 60    

interval = 1 / RATE

start = time.time()
count = 0

while time.time() - start < DURATION:
    producer.send("transactions", generate_tx())
    count += 1
    time.sleep(interval)

producer.flush()
print(f"Sent {count} messages")
