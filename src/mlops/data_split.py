import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "creditcard.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

def load_and_split(train_ratio=0.8, random_state=42):
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df)} total records")
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    kafka_df = df.iloc[split_idx:].copy()
    logger.info(f"Split at index {split_idx}")
    logger.info(f"  Train: {len(train_df)}")
    logger.info(f"  Kafka (no label): {len(kafka_df)}")
    return train_df, kafka_df

def remove_labels(df):
    df = df.copy()
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    return df

def save_train_data(train_df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, "cleaned_fraud_data.parquet")
    train_df.to_parquet(filepath, index=False)
    logger.info(f"Saved train data to {filepath}")
    return filepath

def save_kafka_data(kafka_df):
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    filepath = os.path.join(TEST_DATA_DIR, "kafka_stream.parquet")
    kafka_df = remove_labels(kafka_df)
    kafka_df.to_parquet(filepath, index=False)
    logger.info(f"Saved Kafka data to {filepath}")
    return filepath

def check_class_distribution(df):
    if "Class" not in df.columns:
        return {"total": len(df)}
    return {"total": len(df), "fraud": int((df["Class"]==1).sum()), "normal": int((df["Class"]==0).sum())}

def run_split(train_ratio=0.8):
    logger.info("INITIAL DATA SPLIT")
    train_df, kafka_df = load_and_split(train_ratio)
    train_stats = check_class_distribution(train_df)
    train_path = save_train_data(train_df)
    kafka_path = save_kafka_data(kafka_df)
    logger.info(f"SPLIT COMPLETE")
    return {"success": True, "train_file": train_path, "kafka_file": kafka_path}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    run_split(args.train_ratio)
