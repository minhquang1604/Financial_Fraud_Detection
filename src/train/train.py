import os
import duckdb
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, f1_score

from utils import engineer_features, get_feature_columns


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "creditcard.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


def load_and_clean_data() -> pd.DataFrame:
    print("--- Phase 1: Data Loading & Cleaning ---")
    
    con = duckdb.connect(database=':memory:')
    
    try:
        con.execute(f"CREATE TABLE raw_transactions AS SELECT * FROM read_csv_auto('{RAW_DATA_PATH}')")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    
    con.execute("""
        CREATE TABLE processed_transactions AS 
        SELECT 
            CAST(Time AS DOUBLE) as Time,
            CAST(Amount AS DOUBLE) as Amount,
            CAST(Class AS INTEGER) as Class,
            V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, 
            V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
            V21, V22, V23, V24, V25, V26, V27, V28
        FROM raw_transactions
        WHERE Amount IS NOT NULL AND Class IS NOT NULL
    """)
    
    df_final = con.execute("SELECT * FROM processed_transactions").df()
    con.close()
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    df_final.to_parquet(f'{PROCESSED_DATA_PATH}/cleaned_fraud_data.parquet', index=False)
    print(f"Data cleaning completed. Total records: {len(df_final)}")
    
    return df_final


def train_model(df: pd.DataFrame) -> None:
    print("\n--- Phase 2: Feature Engineering ---")
    df_features = engineer_features(df)
    print("Feature engineering completed.")
    
    print("\n--- Phase 3: Model Training & Evaluation ---")
    
    feature_cols = get_feature_columns()
    X = df_features[feature_cols]
    y = df_features['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    os.makedirs(TEST_DATA_PATH, exist_ok=True)
    test_df = X_test.copy()
    test_df['Class'] = y_test.values
    test_df.to_parquet(f'{TEST_DATA_PATH}/test_data.parquet', index=False)
    print(f"Test data saved to {TEST_DATA_PATH}/test_data.parquet")
    
    counter = y_train.value_counts()
    imbalance_ratio = counter[0] / counter[1]
    print(f"Calculated scale_pos_weight: {imbalance_ratio:.2f}")
    
    mlflow.set_experiment("FraudGuard_XGBoost_Baseline")
    
    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=imbalance_ratio,
            eval_metric='logloss',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        auprc = average_precision_score(y_test, y_probs)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("scale_pos_weight", imbalance_ratio)
        mlflow.log_metric("AUPRC", auprc)
        mlflow.log_metric("F1", f1)
        mlflow.sklearn.log_model(model, "fraud_model_xgboost")
        
        print(f"Model successfully trained. AUPRC Score: {auprc:.4f}, F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def main():
    df = load_and_clean_data()
    train_model(df)


if __name__ == "__main__":
    main()