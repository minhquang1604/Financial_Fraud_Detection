import os
import json
import time
import pandas as pd
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError


BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = "transaction_events"
API_URL = os.environ.get("API_URL", "http://localhost:8000")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "realtime")
BATCH_SIZE = 1000

FEATURE_COLS = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount", "Time"
]


def create_consumer():
    return KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='fraud-detection-consumer'
    )


def call_prediction_api(features: dict) -> dict:
    url = f"{API_URL}/predict"
    payload = {"features": features}
    try:
        import requests
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API call failed: {e}")
        return {"error": str(e)}


def save_to_parquet(records: list, batch_id: int):
    os.makedirs(STAGING_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"staging_consumer_{timestamp}_batch{batch_id}.parquet"
    filepath = os.path.join(STAGING_DIR, filename)
    
    df = pd.DataFrame(records)
    df.to_parquet(filepath, index=False)
    print(f"[SAVED] {len(records)} records -> {filepath}")
    return filepath


def run_consumer():
    consumer = create_consumer()
    print(f"Consumer connected to {BOOTSTRAP_SERVERS}")
    print(f"Listening to topic: {TOPIC_NAME}")
    print(f"Saving to: {STAGING_DIR}")
    print("-" * 60)
    
    buffer = []
    batch_id = 0
    
    try:
        for message in consumer:
            try:
                features = message.value
                transaction_time = features.get("Time", "N/A")
                transaction_key = message.key.decode('utf-8') if message.key else "N/A"
                
                record = {col: features.get(col, None) for col in FEATURE_COLS}
                record["_transaction_key"] = transaction_key
                buffer.append(record)
                
                result = call_prediction_api(features)
                
                if "error" in result:
                    print(f"[{transaction_key}] Time: {transaction_time} | Error: {result['error']}")
                else:
                    pred = result.get("prediction", "N/A")
                    prob = result.get("fraud_probability", "N/A")
                    msg = result.get("message", "N/A")
                    print(f"[{transaction_key}] Time: {transaction_time} | Prediction: {pred} | Probability: {prob:.4f} | {msg}")
                
                if len(buffer) >= BATCH_SIZE:
                    batch_id += 1
                    save_to_parquet(buffer, batch_id)
                    buffer = []
                    
            except Exception as e:
                print(f"Error processing message: {e}")
                
    except KeyboardInterrupt:
        print("\nConsumer stopped by user")
    finally:
        if buffer:
            batch_id += 1
            save_to_parquet(buffer, batch_id)
        consumer.close()


if __name__ == "__main__":
    run_consumer()
