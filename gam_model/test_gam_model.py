import json
import time
import model

MODEL_DIR = "./gam_model/"

def main():
    with open(f"{MODEL_DIR}example.json", "r") as f:
        payload = json.load(f)
    sum_latency = 0
    payload = model.build_features(payload)
    for i in range(1000):
        start_ts = time.perf_counter()
        result = model.predict(payload)
        end_ts = time.perf_counter()

        latency_ms = (end_ts - start_ts) * 1000


        sum_latency+=latency_ms
    print(f"AVG latency:{sum_latency/1000}")

if __name__ == "__main__":
    main()
