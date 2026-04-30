import os
import duckdb
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    f1_score,
    precision_recall_curve
)

from utils import engineer_features, get_feature_columns


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "creditcard.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


# =========================
# 1. LOAD & CLEAN DATA
# =========================
def load_and_clean_data() -> pd.DataFrame:
    print("--- Phase 1: Data Loading & Cleaning ---")

    con = duckdb.connect(database=':memory:')

    con.execute(f"""
        CREATE TABLE raw_transactions 
        AS SELECT * FROM read_csv_auto('{RAW_DATA_PATH}')
    """)

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

    df = con.execute("SELECT * FROM processed_transactions").df()
    con.close()

    # 🔥 sort theo Time (QUAN TRỌNG)
    df = df.sort_values("Time").reset_index(drop=True)

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    df.to_parquet(f'{PROCESSED_DATA_PATH}/cleaned_fraud_data.parquet', index=False)

    print(f"Data cleaned. Records: {len(df)}")
    return df


# =========================
# 2. FEATURE ENGINEERING (use utils)
# =========================
def get_train_stats(df: pd.DataFrame) -> dict:
    return {
        "mean_amt": df['Amount'].mean(),
        "median_amt": df['Amount'].median(),
        "threshold_95": df['Amount'].quantile(0.95)
    }


# =========================
# 3. THRESHOLD TUNING
# =========================
def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    return thresholds[best_idx], f1_scores[best_idx]


# =========================
# 4. TRAIN MODEL
# =========================
def train_model(df: pd.DataFrame):
    print("\n--- Phase 2: Train / Validation Split ---")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['Class'],
        random_state=42
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df['Class'],
        random_state=42
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df, reference_df=train_df)
    test_df = engineer_features(test_df, reference_df=train_df)

    train_stats = get_train_stats(train_df)

    feature_cols = get_feature_columns()

    X_train, y_train = train_df[feature_cols], train_df['Class']
    X_val, y_val = val_df[feature_cols], val_df['Class']
    X_test, y_test = test_df[feature_cols], test_df['Class']

    # =========================
    # Handle Imbalance
    # =========================
    counter = y_train.value_counts()
    scale_pos_weight = counter[0] / counter[1]

    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # =========================
    # MLflow
    # =========================
    mlflow.set_experiment("FraudGuard_XGBoost_Improved")

    with mlflow.start_run():

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        )

        model.fit(X_train, y_train)

        # =========================
        # VALIDATION
        # =========================
        val_probs = model.predict_proba(X_val)[:, 1]

        best_threshold, best_f1 = find_best_threshold(y_val, val_probs)

        print(f"Best threshold: {best_threshold:.4f}, Best F1: {best_f1:.4f}")

        # =========================
        # TEST EVALUATION
        # =========================
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs > best_threshold).astype(int)

        auprc = average_precision_score(y_test, test_probs)
        f1 = f1_score(y_test, test_preds)

        print(f"\nAUPRC: {auprc:.4f}, F1: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, test_preds))

        # =========================
        # LOGGING
        # =========================
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("threshold", best_threshold)
        mlflow.log_param("mean_amt", train_stats["mean_amt"])
        mlflow.log_param("median_amt", train_stats["median_amt"])
        mlflow.log_param("threshold_95", train_stats["threshold_95"])

        mlflow.log_metric("AUPRC", auprc)
        mlflow.log_metric("F1", f1)

        mlflow.sklearn.log_model(model, "fraud_model_xgboost")

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump({
        "model": model,
        "threshold": best_threshold,
        "features": feature_cols,
        "reference_stats": train_stats
    }, os.path.join(MODEL_DIR, "fraud_model.pkl"))

    print(f"\nModel saved to {MODEL_DIR}/fraud_model.pkl")


# =========================
# MAIN
# =========================
def main():
    df = load_and_clean_data()
    train_model(df)


if __name__ == "__main__":
    main()