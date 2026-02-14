from pydantic import BaseModel
from cassandra.cluster import Cluster

cluster = Cluster(["127.0.0.1"])
session = cluster.connect()
session.set_keyspace("speed_layer")

class Transaction(BaseModel):
    SENDER_ACCOUNT_ID: int
    RECEIVER_ACCOUNT_ID: int
    TX_TYPE: str
    TX_AMOUNT: float
    TIMESTAMP: int

def get_account(account_id: int):
    query = """
        SELECT account_id, customer_id, init_balance,
               country, account_type, tx_behavior_id
        FROM accounts
        WHERE account_id = %s
    """
    rows = session.execute(query, (account_id,))
    row = rows.one()

    if not row:
        return None

    return {
        "ACCOUNT_ID": row.account_id,
        "CUSTOMER_ID": row.customer_id,
        "INIT_BALANCE": row.init_balance,
        "COUNTRY": row.country,
        "ACCOUNT_TYPE": row.account_type,
        "TX_BEHAVIOR_ID": row.tx_behavior_id
    }
