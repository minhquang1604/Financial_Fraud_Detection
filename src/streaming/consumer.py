import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mlops"))
from s3_manager import S3DataManager


BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = "transaction_events"
API_URL = os.environ.get("API_URL", "http://localhost:8000")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_SIZE = 1000

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "fraud-detection-data")
USE_S3 = os.environ.get("USE_S3", "true").lower() == "true"

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


s3_manager = S3DataManager() if USE_S3 else None


def save_to_parquet(records: list, batch_id: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"realtime_{timestamp}_batch{batch_id}.parquet"

    df = pd.DataFrame(records)

    if USE_S3 and s3_manager:
        s3_path = s3_manager.upload_dataframe(df, "realtime", filename, "parquet")
        print(f"[S3 SAVED] {len(records)} records -> s3://{S3_BUCKET_NAME}/{s3_path}")
        return s3_path
    else:
        os.makedirs(PROJECT_ROOT + "/data/realtime", exist_ok=True)
        filepath = os.path.join(PROJECT_ROOT, "data", "realtime", filename)
        df.to_parquet(filepath, index=False)
        print(f"[LOCAL SAVED] {len(records)} records -> {filepath}")
        return filepath


def run_consumer():
    consumer = create_consumer()
    print(f"Consumer connected to {BOOTSTRAP_SERVERS}")
    print(f"Listening to topic: {TOPIC_NAME}")
    if USE_S3:
        print(f"Storage: S3 ({S3_BUCKET_NAME})")
    else:
        print(f"Storage: Local")
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
                    prob_pct = prob * 100
                    msg = result.get("message", "N/A")
                    print(f"[{transaction_key}] Time: {transaction_time} | Prediction: {pred} | Fraud Probability: {prob_pct:.2f}% | {msg}")
                
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
