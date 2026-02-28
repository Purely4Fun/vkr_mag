from pydantic import BaseModel
from cassandra.cluster import Cluster
from uuid import uuid4
from datetime import datetime, timezone

cluster = None
session = None

insert_transaction_stmt = None
insert_transaction_by_sender_stmt = None
insert_transaction_by_receiver_stmt = None
insert_alert_stmt = None
select_account_stmt = None

def init_cassandra():
    global cluster, session
    global insert_transaction_stmt
    global insert_transaction_by_sender_stmt
    global insert_transaction_by_receiver_stmt
    global insert_alert_stmt
    global select_account_stmt

    if session is not None:
        return

    cluster = Cluster(["127.0.0.1"])
    session = cluster.connect("speed_layer")

    insert_transaction_stmt = session.prepare("""
        INSERT INTO transactions (
            tx_id, tx_timestamp, sender_account_id,
            receiver_account_id, tx_type, tx_amount,
            is_fraud, alert_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)

    insert_transaction_by_sender_stmt = session.prepare("""
        INSERT INTO transactions_by_sender (
            sender_account_id, tx_timestamp, tx_id,
            receiver_account_id, tx_type, tx_amount,
            is_fraud, alert_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)

    insert_transaction_by_receiver_stmt = session.prepare("""
        INSERT INTO transactions_by_receiver (
            receiver_account_id, tx_timestamp, tx_id,
            sender_account_id, tx_type, tx_amount,
            is_fraud, alert_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)

    insert_alert_stmt = session.prepare("""
        INSERT INTO alerts (
            alert_id, alert_timestamp, alert_type,
            is_fraud, tx_id, sender_account_id,
            receiver_account_id, tx_type, tx_amount
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)

    select_account_stmt = session.prepare("""
        SELECT account_id, customer_id, init_balance,
               country, account_type, tx_behavior_id
        FROM accounts
        WHERE account_id = ?
    """)

class Transaction(BaseModel):
    SENDER_ACCOUNT_ID: int
    RECEIVER_ACCOUNT_ID: int
    TX_TYPE: str
    TX_AMOUNT: float
    TIMESTAMP: int

def get_account(account_id: int):
    if session is None:
        init_cassandra()

    future = session.execute_async(select_account_stmt, (account_id,))
    rows = future.result()  
    row = rows.one()

    if not row:
        return None

    return {
        "ACCOUNT_ID": row.account_id,
        "CUSTOMER_ID": row.customer_id,
        "INIT_BALANCE": row.init_balance,
        "COUNTRY": row.country,
        "ACCOUNT_TYPE": row.account_type,
        "TX_BEHAVIOR_ID": row.tx_behavior_id,
    }

def save_transaction(tx, is_fraud: bool):
    if session is None:
        init_cassandra()

    tx_id = uuid4()
    timestamp = datetime.now(timezone.utc)

    futures = []
    alert_id = None

    if is_fraud:
        alert_id = uuid4()

        futures.append(
            session.execute_async(insert_alert_stmt, (
                alert_id,
                timestamp,
                "FRAUD",
                True,
                tx_id,
                tx.SENDER_ACCOUNT_ID,
                tx.RECEIVER_ACCOUNT_ID,
                tx.TX_TYPE,
                tx.TX_AMOUNT,
            ))
        )

    futures.append(
        session.execute_async(insert_transaction_stmt, (
            tx_id,
            timestamp,
            tx.SENDER_ACCOUNT_ID,
            tx.RECEIVER_ACCOUNT_ID,
            tx.TX_TYPE,
            tx.TX_AMOUNT,
            is_fraud,
            alert_id,
        ))
    )

    futures.append(
        session.execute_async(insert_transaction_by_sender_stmt, (
            tx.SENDER_ACCOUNT_ID,
            timestamp,
            tx_id,
            tx.RECEIVER_ACCOUNT_ID,
            tx.TX_TYPE,
            tx.TX_AMOUNT,
            is_fraud,
            alert_id,
        ))
    )

    futures.append(
        session.execute_async(insert_transaction_by_receiver_stmt, (
            tx.RECEIVER_ACCOUNT_ID,
            timestamp,
            tx_id,
            tx.SENDER_ACCOUNT_ID,
            tx.TX_TYPE,
            tx.TX_AMOUNT,
            is_fraud,
            alert_id,
        ))
    )
    for f in futures:
        f.result()

    return tx_id, alert_id