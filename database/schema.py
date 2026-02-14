from cassandra.cluster import Cluster

cluster = Cluster(['127.0.0.1'], port=9042)
session = cluster.connect()

session.execute("""
CREATE KEYSPACE IF NOT EXISTS speed_layer
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};
""")

session.set_keyspace("speed_layer")

session.execute("""
CREATE TABLE IF NOT EXISTS accounts (
    account_id INT,
    customer_id TEXT,
    init_balance DOUBLE,
    country TEXT,
    account_type TEXT,
    is_fraud BOOLEAN,
    tx_behavior_id INT,
    PRIMARY KEY ((account_id), customer_id)
) WITH CLUSTERING ORDER BY (customer_id ASC);
""")

session.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    alert_type TEXT,
    alert_timestamp TIMESTAMP,
    alert_id INT,
    is_fraud BOOLEAN,
    tx_id INT,
    sender_account_id INT,
    receiver_account_id INT,
    tx_type TEXT,
    tx_amount DOUBLE,
    PRIMARY KEY ((alert_id), alert_timestamp, alert_type)
) WITH CLUSTERING ORDER BY (alert_timestamp DESC, alert_type ASC);
""")

session.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    tx_id INT,
    tx_timestamp TIMESTAMP,
    sender_account_id INT,
    receiver_account_id INT,
    tx_type TEXT,
    tx_amount DOUBLE,
    is_fraud BOOLEAN,
    alert_id INT,
    PRIMARY KEY ((tx_id), tx_timestamp)
) WITH CLUSTERING ORDER BY (tx_timestamp DESC);
""")

session.execute("""
CREATE TABLE IF NOT EXISTS transactions_by_sender (
    sender_account_id uuid,
    tx_timestamp timestamp,
    tx_id uuid,
    receiver_account_id uuid,
    tx_type text,
    tx_amount decimal,
    is_fraud boolean,
    alert_id uuid,
    PRIMARY KEY ((sender_account_id), tx_timestamp, tx_id)
) WITH CLUSTERING ORDER BY (tx_timestamp DESC, tx_id ASC);
""")

session.execute("""
CREATE TABLE IF NOT EXISTS transactions_by_receiver (
    receiver_account_id uuid,
    tx_timestamp timestamp,
    tx_id uuid,
    sender_account_id uuid,
    tx_type text,
    tx_amount decimal,
    is_fraud boolean,
    alert_id uuid,
    PRIMARY KEY ((receiver_account_id), tx_timestamp, tx_id)
) WITH CLUSTERING ORDER BY (tx_timestamp DESC, tx_id ASC);
""")

session.execute("""
CREATE KEYSPACE IF NOT EXISTS batch_layer
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};
""")

session.set_keyspace("batch_layer")

session.execute("""
CREATE TABLE IF NOT EXISTS accounts (
    account_id INT,
    customer_id TEXT,
    init_balance DOUBLE,
    country TEXT,
    account_type TEXT,
    is_fraud BOOLEAN,
    tx_behavior_id INT,
    PRIMARY KEY ((account_id), customer_id)
) WITH CLUSTERING ORDER BY (customer_id ASC);
""")

session.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    alert_type TEXT,
    alert_timestamp TIMESTAMP,
    alert_id INT,
    is_fraud BOOLEAN,
    tx_id INT,
    sender_account_id INT,
    receiver_account_id INT,
    tx_type TEXT,
    tx_amount DOUBLE,
    PRIMARY KEY ((alert_id), alert_timestamp, alert_type)
) WITH CLUSTERING ORDER BY (alert_timestamp DESC, alert_type ASC);
""")

session.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    tx_id INT,
    tx_timestamp TIMESTAMP,
    sender_account_id INT,
    receiver_account_id INT,
    tx_type TEXT,
    tx_amount DOUBLE,
    is_fraud BOOLEAN,
    is_fraud_speed BOOLEAN,
    speed_layer_score DOUBLE,
    speed_layer_treshold DOUBLE,
    alert_id INT,
    PRIMARY KEY ((tx_id), tx_timestamp)
) WITH CLUSTERING ORDER BY (tx_timestamp DESC);
""")

session.execute("""
CREATE TABLE IF NOT EXISTS transactions_by_sender (
    sender_account_id uuid,
    tx_timestamp timestamp,
    tx_id uuid,
    receiver_account_id uuid,
    tx_type text,
    tx_amount decimal,
    is_fraud boolean,
    is_fraud_speed BOOLEAN,
    speed_layer_score DOUBLE,
    speed_layer_treshold DOUBLE,
    alert_id uuid,
    PRIMARY KEY ((sender_account_id), tx_timestamp, tx_id)
) WITH CLUSTERING ORDER BY (tx_timestamp DESC, tx_id ASC);
""")

session.execute("""
CREATE TABLE IF NOT EXISTS transactions_by_receiver (
    receiver_account_id uuid,
    tx_timestamp timestamp,
    tx_id uuid,
    sender_account_id uuid,
    tx_type text,
    tx_amount decimal,
    is_fraud boolean,
    is_fraud_speed BOOLEAN,
    speed_layer_score DOUBLE,
    speed_layer_treshold DOUBLE,
    alert_id uuid,
    PRIMARY KEY ((receiver_account_id), tx_timestamp, tx_id)
) WITH CLUSTERING ORDER BY (tx_timestamp DESC, tx_id ASC);
""")

