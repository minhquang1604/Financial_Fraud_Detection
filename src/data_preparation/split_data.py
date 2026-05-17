#!/usr/bin/env python3
"""
Split raw data:
- 80% → train/ (full data with Class)
- 20% → staging/ (remove Class for producer)
"""
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "creditcard.csv")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")

def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    total = len(df)
    print(f"Total records: {total}")
    
    # Shuffle with seed for reproducibility
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split 80/20
    train_size = int(0.8 * total)
    train_df = df.iloc[:train_size]
    staging_df = df.iloc[train_size:]
    
    print(f"Train: {len(train_df)} records (80%)")
    print(f"Staging: {len(staging_df)} records (20%)")
    
    # Save train data (full with Class)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    train_path = os.path.join(TRAIN_DIR, "train_full.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"Saved to: {train_path}")
    
    # Save staging data (remove Class)
    os.makedirs(STAGING_DIR, exist_ok=True)
    staging_path = os.path.join(STAGING_DIR, "staging_batch_v1.parquet")
    staging_df_no_class = staging_df.drop(columns=['Class'])
    staging_df_no_class.to_parquet(staging_path, index=False)
    print(f"Saved to: {staging_path}")
    
    # Class distribution
    print(f"\nTrain class distribution:")
    print(f"  Class 0: {(train_df['Class']==0).sum()}")
    print(f"  Class 1: {(train_df['Class']==1).sum()}")
    
    print(f"\nStaging class distribution (before removing):")
    print(f"  Class 0: {(staging_df['Class']==0).sum()}")
    print(f"  Class 1: {(staging_df['Class']==1).sum()}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()