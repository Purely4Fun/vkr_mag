# ВКР Карамушка А. А. КТмо2-14 

# Интеллектуальная система выявления мошеннических финансовых операций

---

## Getting Started

### Prerequisites

* Docker & Docker Compose
* Python
* Java 
* Apache Spark (for local usage)

---

## Run Infrastructure

Start all services:

```bash
docker-compose up -d
```

---

## Services & Ports

| Service         | Host      | Port | Description             |
| --------------- | --------- | ---- | ----------------------- |
| Kafka           | localhost | 9094 | Kafka external listener |
| Spark UI        | localhost | 8080 | Spark master UI         |
| Spark Worker UI | localhost | 8081 | Spark worker UI         |
| Cassandra       | localhost | 9042 | Cassandra Database      |
| MinIO API       | localhost | 9000 | S3 storage              |
| MinIO UI        | localhost | 9001 | S3 Web console          |

### Credentials

**MinIO:**

* Access Key: `minio`
* Secret Key: `minio123`

---

## API

### Run API

From the root directory:

```bash
uvicorn api.endpoint_predict:app --reload --loop asyncio
```

API will be available at:

```
http://localhost:8000
```

---

### Request Example

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
            "SENDER_ACCOUNT_ID": 1234,
            "RECEIVER_ACCOUNT_ID": 5678,
            "TX_TYPE": "TRANSFER",
            "TX_AMOUNT": 123.45,
            "TIMESTAMP": 1708001234
        }'
```

---

## Database Setup

### Create Cassandra Schema

```bash
python3 -m database/schema.py
```

---

### Import from CSV Data to CassandraDB

Loads data into both:

* `speed_layer`
* `batch_layer`

```bash
python3 -m database/import.py
```

---

## Streaming Layer (Kafka + GAM Model)

### Kafka Consumer

Run:

```bash
python3 -m kafka_adapters/consumer.py
```

---

### Send Message to Kafka

Example message:

```json
{"SENDER_ACCOUNT_ID":9876,"RECEIVER_ACCOUNT_ID":6789,"TX_TYPE":"TRANSFER","TX_AMOUNT":333.44,"TIMESTAMP":1708001234}
```

Send via Kafka CLI:

```bash
docker exec -it kafka bash -c "export PATH=\$PATH:/opt/kafka/bin && kafka-console-producer.sh --topic transactions --bootstrap-server kafka:9092"
```

---

### GAM Model (Streaming)

Directory: `/gam_model`

Run in sequence (process data -> train model):

```bash
python preprocess.py
python train.py
```

Main interface:

```
model.py
```

---

## Batch Layer (Spark + XGBoost)

### Run Batch Processing

```bash
export PYTHONPATH=/home/user/vkr/vkr_mag

spark-submit \
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,com.datastax.spark:spark-cassandra-connector_2.12:3.1.0 \
  --conf spark.hadoop.fs.s3a.access.key=minio \
  --conf spark.hadoop.fs.s3a.secret.key=minio123 \
  --conf spark.hadoop.fs.s3a.endpoint=http://localhost:9000 \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
  --conf spark.hadoop.fs.s3a.fast.upload=true \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.cassandra.connection.host=127.0.0.1 \
  --conf spark.cassandra.connection.port=9042 \
  batch_processing/run_batch.py
```

---

### XGBoost Model (Batch)

Directory: `/xgb_model`

Run in sequence (process data -> train model):

```bash
python preprocess.py
python train.py
```

Main interface:

```
model.py
```

---

## Load Testing

### API Load Test

```bash
node load_tests/test_api.js
```

---

### Kafka Load Test

```bash
python3 -m load_tests/test_kafka.py
```

---

## Docker Compose Services

The system includes:

* Kafka (KRaft mode)
* Spark Master & Worker
* Cassandra
* MinIO

To stop:

```bash
docker-compose stop
```

---

## Notes

* Ensure Cassandra is fully initialized before running schema scripts
* Import data from csv-files to Cassandra before running api and adapters
* Kafka topic (`transactions`) must exist or be auto-created
* MinIO must be accessible before running batch jobs
* Spark jobs require correct `PYTHONPATH`
* сделать **более “enterprise” README (с диаграммой архитектуры)**
* или **добавить badges + CI/CD + Makefile**
