import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_drift_data(
    source_path: str = None,
    output_dir: str = None,
    n_samples: int = 5000,
    drift_type: str = "feature",
    drift_severity: float = 0.3
):
    """
    Generate synthetic drift data for testing.
    
    drift_type: "feature" (feature distribution shift) or "label" (label ratio shift)
    drift_severity: 0.0 - 1.0 (higher = more drift)
    """
    
    if source_path is None:
        source_path = os.path.join(PROJECT_ROOT, "data", "staging", "staging_batch_v1.parquet")
    
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "staging")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading source data from: {source_path}")
    df = pd.read_parquet(source_path)
    
    print(f"Original shape: {df.shape}")
    
    has_class = "Class" in df.columns
    if has_class:
        print(f"Original Class distribution: {df['Class'].value_counts().to_dict()}")
    
    if drift_type == "feature":
        feature_cols = [f"V{i}" for i in range(1, 29)]
        
        np.random.seed(42)
        
        for col in feature_cols[:10]:
            shift = np.random.uniform(-drift_severity, drift_severity)
            df[col] = df[col] + shift * df[col].std()
        
        df["Amount"] = df["Amount"] * (1 + drift_severity * np.random.uniform(-1, 1, len(df)))
        
        print(f"\nApplied feature drift: severity={drift_severity}")
        
    elif drift_type == "label":
        if not has_class:
            print("WARNING: No Class column in data, skipping label drift")
            return None
        
        fraud_mask = df["Class"] == 1
        n_fraud = fraud_mask.sum()
        
        flip_ratio = drift_severity * 0.5
        n_flip = int(n_fraud * flip_ratio)
        
        fraud_indices = df[fraud_mask].index
        flip_indices = np.random.choice(fraud_indices, n_flip, replace=False)
        
        df.loc[flip_indices, "Class"] = 0
        
        normal_indices = df[df["Class"] == 0].index[:n_flip]
        df.loc[normal_indices, "Class"] = 1
        
        print(f"\nApplied label drift: flipped {n_flip} labels")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drift_test_{drift_type}_{int(drift_severity*100)}_{timestamp}.parquet"
    output_path = os.path.join(output_dir, filename)
    
    df.to_parquet(output_path, index=False)
    
    print(f"\nDrift data saved to: {output_path}")
    print(f"New shape: {df.shape}")
    if has_class:
        print(f"New Class distribution: {df['Class'].value_counts().to_dict()}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic drift data for testing")
    parser.add_argument("--source", type=str, help="Source data path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--type", type=str, default="feature", choices=["feature", "label"], help="Drift type")
    parser.add_argument("--severity", type=float, default=0.3, help="Drift severity (0.0-1.0)")
    
    args = parser.parse_args()
    
    generate_drift_data(
        source_path=args.source,
        output_dir=args.output,
        n_samples=args.samples,
        drift_type=args.type,
        drift_severity=args.severity
    )