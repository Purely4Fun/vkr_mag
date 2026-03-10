import json
from minio import Minio
import io

from batch_processing.config import (
    S3_ENDPOINT,
    S3_ACCESS_KEY,
    S3_SECRET_KEY,
    S3_BATCH_BUCKET,
)

client = Minio(
    S3_ENDPOINT,
    access_key=S3_ACCESS_KEY,
    secret_key=S3_SECRET_KEY,
    secure=False,
)


def save_result(object_name, payload):

    data = json.dumps(payload).encode()

    client.put_object(
        S3_BATCH_BUCKET,
        object_name,
        data=io.BytesIO(data),
        length=len(data),
        content_type="application/json",
    )
