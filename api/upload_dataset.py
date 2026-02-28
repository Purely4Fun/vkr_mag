import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import os
import sys
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor

from api.entity import get_account, save_transaction, init_cassandra
from gam_model.model import predict, build_features
from s3.storage import save_transaction_to_s3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_PATH = "./dataset/transactions.csv"
CHUNK_SIZE = 5000
MAX_PROCESSES = mp.cpu_count()
MAX_S3_THREADS = 4

account_cache = {}
queue = None


# -------------------- INIT WORKER --------------------
def init_process(progress_queue):
    global account_cache, queue
    account_cache = {}
    queue = progress_queue
    init_cassandra()


# -------------------- PROCESS CHUNK --------------------
def process_chunk(chunk_df):
    global account_cache, queue

    try:
        print(f"[PID {os.getpid()}] start chunk {len(chunk_df)}", flush=True)

        results = []
        records = chunk_df.to_dict("records")

        with ThreadPoolExecutor(max_workers=MAX_S3_THREADS) as s3_executor:
            futures = []
            processed_in_chunk = 0

            for record in records:
                try:
                    sender_id = int(record["SENDER_ACCOUNT_ID"])

                    if sender_id not in account_cache:
                        account_cache[sender_id] = get_account(sender_id)
                    account = account_cache[sender_id]

                    if not account:
                        if queue:
                            queue.put(("progress", 1))
                        continue

                    tx_dict = {
                        "SENDER_ACCOUNT_ID": sender_id,
                        "RECEIVER_ACCOUNT_ID": int(record["RECEIVER_ACCOUNT_ID"]),
                        "TX_TYPE": record["TX_TYPE"],
                        "TX_AMOUNT": float(record["TX_AMOUNT"]),
                        "TIMESTAMP": int(record["TIMESTAMP"]),
                    }

                    enriched = build_features(
                        {"account": account, "transaction": tx_dict}
                    )
                    is_fraud, score, threshold = predict(enriched)

                    class TransactionStub:
                        def __init__(self, data):
                            self.__dict__.update(data)

                    tx_id, alert_id = save_transaction(
                        TransactionStub(tx_dict),
                        is_fraud,
                    )

                    s3_payload = {
                        "tx_id": str(tx_id),
                        "alert_id": str(alert_id) if alert_id else None,
                        "transaction": tx_dict,
                        "fraud_prediction": bool(is_fraud),
                        "score": float(score),
                        "threshold": float(threshold),
                        "prediction_timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }

                    futures.append(
                        s3_executor.submit(
                            save_transaction_to_s3, str(tx_id), s3_payload
                        )
                    )

                    results.append(s3_payload)
                    processed_in_chunk += 1

                    if queue and processed_in_chunk % 100 == 0:
                        queue.put(("progress", 100))

                except Exception as e:
                    print(f"Ошибка транзакции: {e}", flush=True)
                    if queue:
                        queue.put(("progress", 1))
                    continue

            for future in futures:
                future.result()

            if queue and processed_in_chunk % 100 != 0:
                queue.put(("progress", processed_in_chunk % 100))

        return results

    except Exception as e:
        print(f"FATAL in worker: {e}", flush=True)
        return []  # 🔥 ВАЖНО — никогда не падаем наружу
    
# -------------------- CHUNK GENERATOR --------------------
def chunk_generator():
    for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
        yield chunk


# -------------------- PROGRESS LISTENER --------------------
def progress_listener(progress_queue, total_rows):
    with tqdm(total=total_rows, desc="Обработка", unit="тр.") as pbar:
        while True:
            msg = progress_queue.get()
            if msg == "DONE":
                break

            msg_type, value = msg
            if msg_type == "progress":
                pbar.update(value)


# -------------------- MAIN --------------------
def main_parallel():
    start_time = time.time()

    # считаем строки
    with open(CSV_PATH, "r") as f:
        next(f)
        total_rows = sum(1 for _ in f)

    print(f"Всего транзакций для обработки: {total_rows}")

    progress_queue = mp.Queue()

    progress_process = mp.Process(
        target=progress_listener, args=(progress_queue, total_rows)
    )
    progress_process.start()

    output_csv = "./dataset/predictions.csv"
    first_chunk = True
    processed_total = 0

    with mp.Pool(
        processes=MAX_PROCESSES,
        initializer=init_process,
        initargs=(progress_queue,),
    ) as pool:

        # ⚡ потоковая обработка (без list!)
        for chunk_results in pool.imap_unordered(
            process_chunk, chunk_generator()
        ):
            if chunk_results:
                df_chunk = pd.json_normalize(chunk_results)
                df_chunk.to_csv(
                    output_csv, mode="a", header=first_chunk, index=False
                )
                first_chunk = False
                processed_total += len(chunk_results)

    progress_queue.put("DONE")
    progress_process.join()

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("Обработка завершена!")
    print(f"Обработано транзакций: {processed_total}")
    print(f"Затраченное время: {elapsed:.2f} с")
    print(f"Средняя скорость: {processed_total/elapsed:.2f} тр./с")
    print(f"Результаты сохранены в {output_csv}")


if __name__ == "__main__":
    mp.freeze_support()
    main_parallel()