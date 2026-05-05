import os
import json
import requests
from kafka import KafkaConsumer
from kafka.errors import KafkaError


BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = "transaction_events"
API_URL = os.environ.get("API_URL", "http://localhost:8000")


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
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return {"error": str(e)}


def run_consumer():
    consumer = create_consumer()
    print(f"Consumer connected to {BOOTSTRAP_SERVERS}")
    print(f"Listening to topic: {TOPIC_NAME}")
    print("-" * 60)
    
    try:
        for message in consumer:
            try:
                features = message.value
                transaction_time = features.get("Time", "N/A")
                transaction_key = message.key.decode('utf-8') if message.key else "N/A"
                
                result = call_prediction_api(features)
                
                if "error" in result:
                    print(f"[{transaction_key}] Time: {transaction_time} | Error: {result['error']}")
                else:
                    pred = result.get("prediction", "N/A")
                    prob = result.get("fraud_probability", "N/A")
                    msg = result.get("message", "N/A")
                    print(f"[{transaction_key}] Time: {transaction_time} | Prediction: {pred} | Probability: {prob:.4f} | {msg}")
                    
            except Exception as e:
                print(f"Error processing message: {e}")
                
    except KeyboardInterrupt:
        print("\nConsumer stopped by user")
    finally:
        consumer.close()


if __name__ == "__main__":
    run_consumer()