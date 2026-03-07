import model

def build_payload(i=0):
    return {
        "SENDER_ACCOUNT_ID": 1000 + i,
        "RECEIVER_ACCOUNT_ID": 2000 + i,
        "TX_TYPE": "TRANSFER",
        "TX_AMOUNT": 100 + i * 50,
        "TIMESTAMP": 1710000000 + i * 60,

        "INIT_BALANCE": 10000,
        "ACCOUNT_TYPE": "PERSONAL",
        "COUNTRY": "US",

        "score": 0.2,
        "time_since_last_tx": 120,
        "cnt_last_24h": 2,
        "sender_avg_amount": 150,
    }


def test_single_transaction():

    payload = build_payload()

    print("\n=== SINGLE TRANSACTION TEST ===")

    result = model.predict(payload)

    print("Payload:")
    print(payload)

    print("\nPrediction:")
    print(result)


def test_batch_transactions():
    payloads = [build_payload(i) for i in range(5)]

    print("\n=== BATCH TEST (5 TRANSACTIONS) ===")

    results = model.predict_batch(payloads)
    for i, (p, r) in enumerate(zip(payloads, results)):
        print(f"\nTransaction {i+1}")
        print("Payload:", p)
        print("Prediction:", r)


def main():

    test_single_transaction()
    test_batch_transactions()

if __name__ == "__main__":
    main()