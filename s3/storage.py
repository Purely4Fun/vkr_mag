from minio import Minio
import json
import io

MINIO_ENDPOINT = "127.0.0.1:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
BUCKET_NAME = "transactions-bucket"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)


def save_transaction_to_s3(tx_id: str, payload: dict):
    object_name = f"{tx_id}.json"

    data = json.dumps(payload).encode("utf-8")
    data_stream = io.BytesIO(data)

    minio_client.put_object(
        BUCKET_NAME,
        object_name,
        data_stream,
        length=len(data),
        content_type="application/json"
    )
