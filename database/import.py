import csv
from cassandra.cluster import Cluster

CSV_FILE = "./dataset/accounts.csv"

def connect():
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect()
    return cluster, session


def count_rows_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        return sum(1 for _ in reader)

def migrate_accounts(session):
    insert_query = """
        INSERT INTO accounts (
            account_id,
            customer_id,
            init_balance,
            country,
            account_type,
            is_fraud,
            tx_behavior_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    prepared = session.prepare(insert_query)

    inserted_rows = 0

    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            session.execute(prepared, (
                int(row["ACCOUNT_ID"]),
                row["CUSTOMER_ID"],
                float(row["INIT_BALANCE"]),
                row["COUNTRY"],
                row["ACCOUNT_TYPE"],
                None,
                int(row["TX_BEHAVIOR_ID"])
            ))
            inserted_rows += 1

    return inserted_rows


def main():
    cluster, session = connect()

    total_csv_rows = count_rows_csv(CSV_FILE)
    print(f"Total rows in CSV: {total_csv_rows}")

    session.set_keyspace("speed_layer")
    inserted_speed = migrate_accounts(session)

    print("\n--- SPEED LAYER ---")
    print(f"Inserted rows: {inserted_speed}")

    session.set_keyspace("batch_layer")
    inserted_batch = migrate_accounts(session)

    print("\n--- BATCH LAYER ---")
    print(f"Inserted rows: {inserted_batch}")

    cluster.shutdown()

if __name__ == "__main__":
    main()
