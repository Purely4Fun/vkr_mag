from cassandra.cluster import Cluster
from batch_processing.config import CASSANDRA_HOSTS, CASSANDRA_KEYSPACE

cluster = Cluster(CASSANDRA_HOSTS)
session = cluster.connect()

session.set_keyspace(CASSANDRA_KEYSPACE)

insert_tx = session.prepare("""
INSERT INTO transactions (
tx_id,
tx_timestamp,
sender_account_id,
receiver_account_id,
tx_type,
tx_amount,
is_fraud,
speed_layer_score,
speed_layer_treshold,
alert_id
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""")

insert_sender = session.prepare("""
INSERT INTO transactions_by_sender (
sender_account_id,
tx_timestamp,
tx_id,
receiver_account_id,
tx_type,
tx_amount,
is_fraud,
speed_layer_score,
speed_layer_treshold,
alert_id
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""")

insert_receiver = session.prepare("""
INSERT INTO transactions_by_receiver (
receiver_account_id,
tx_timestamp,
tx_id,
sender_account_id,
tx_type,
tx_amount,
is_fraud,
speed_layer_score,
speed_layer_treshold,
alert_id
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""")


def get_account(account_id):

    q = """
    SELECT init_balance, country, account_type, tx_behavior_id
    FROM accounts
    WHERE account_id=%s
    """

    return session.execute(q, [account_id]).one()


def insert_async(rows):

    futures = []

    for r in rows:

        futures.append(session.execute_async(insert_tx, r["tx"]))
        futures.append(session.execute_async(insert_sender, r["sender"]))
        futures.append(session.execute_async(insert_receiver, r["receiver"]))

    for f in futures:
        f.result()