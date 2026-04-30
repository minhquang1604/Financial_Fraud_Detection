import os
import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
KAFKA_DATA_PATH = os.environ.get("KAFKA_DATA_PATH", os.path.join(STAGING_DIR, "staging_batch_v1.parquet"))

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = "transaction_events"
DELAY_SECONDS = 0.1


def get_data_path():
    if os.path.exists(KAFKA_DATA_PATH):
        return KAFKA_DATA_PATH
    files = [f for f in os.listdir(STAGING_DIR) if f.endswith('.parquet')]
    if files:
        return os.path.join(STAGING_DIR, files[0])
    raise FileNotFoundError(f"No data found in {STAGING_DIR}")


def create_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None,
        acks='all',
        retries=3
    )


def prepare_payload(row):
    payload = {
        "V1": float(row["V1"]),
        "V2": float(row["V2"]),
        "V3": float(row["V3"]),
        "V4": float(row["V4"]),
        "V5": float(row["V5"]),
        "V6": float(row["V6"]),
        "V7": float(row["V7"]),
        "V8": float(row["V8"]),
        "V9": float(row["V9"]),
        "V10": float(row["V10"]),
        "V11": float(row["V11"]),
        "V12": float(row["V12"]),
        "V13": float(row["V13"]),
        "V14": float(row["V14"]),
        "V15": float(row["V15"]),
        "V16": float(row["V16"]),
        "V17": float(row["V17"]),
        "V18": float(row["V18"]),
        "V19": float(row["V19"]),
        "V20": float(row["V20"]),
        "V21": float(row["V21"]),
        "V22": float(row["V22"]),
        "V23": float(row["V23"]),
        "V24": float(row["V24"]),
        "V25": float(row["V25"]),
        "V26": float(row["V26"]),
        "V27": float(row["V27"]),
        "V28": float(row["V28"]),
        "Amount": float(row["Amount"]),
        "Time": float(row["Time"]),
    }
    return payload


def run_producer():
    try:
        data_path = get_data_path()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records from {data_path}")
    
    producer = create_producer()
    print(f"Producer connected to {BOOTSTRAP_SERVERS}")
    
    print(f"Publishing messages to topic '{TOPIC_NAME}'...")
    
    for idx, row in df.iterrows():
        payload = prepare_payload(row)
        transaction_key = f"txn_{idx}"
        
        try:
            future = producer.send(TOPIC_NAME, key=transaction_key, value=payload)
            record_metadata = future.get(timeout=10)
            print(f"Sent: {transaction_key} -> Partition: {record_metadata.partition}, Offset: {record_metadata.offset}")
        except KafkaError as e:
            print(f"Error sending message: {e}")
        
        time.sleep(DELAY_SECONDS)
    
    producer.flush()
    producer.close()
    print("Producer finished")


if __name__ == "__main__":
    run_producer()