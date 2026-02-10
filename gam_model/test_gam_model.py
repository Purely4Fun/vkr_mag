import json
import time
import model

MODEL_DIR = "./gam_model/"

def main():
    with open(f"{MODEL_DIR}example.json", "r") as f:
        payload = json.load(f)

    print("=== Input payload ===")
    print(json.dumps(payload, indent=2))
    start_ts = time.perf_counter()
    result = model.predict(payload)
    end_ts = time.perf_counter()

    latency_ms = (end_ts - start_ts) * 1000

    print("\n=== Model output ===")
    print(json.dumps(result, indent=2))

    print("\n=== Performance ===")
    print(f"Inference latency: {latency_ms:.3f} ms")


if __name__ == "__main__":
    main()
