import os
import sys
from dotenv import load_dotenv

load_dotenv()

import duckdb
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.train.utils import engineer_features, get_feature_columns


# =========================
# PATH CONFIG
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train", "train_full.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://13.250.11.23:5000")
EXPERIMENT_NAME = "FraudGuard_XGBoost"
MODEL_NAME = "FraudDetectionModel"


# =========================
# 1. LOAD DATA
# =========================
def load_and_clean_data():
    print(f"Loading training data from: {TRAIN_DATA_PATH}")
    df = pd.read_parquet(TRAIN_DATA_PATH)
    
    df = df.sort_values("Time").reset_index(drop=True)
    
    print(f"Loaded {len(df)} records")
    
    return df


# =========================
# 2. THRESHOLD
# =========================
def find_best_threshold(y_true, y_probs):
    p, r, t = precision_recall_curve(y_true, y_probs)
    f1 = 2 * (p * r) / (p + r + 1e-8)
    idx = np.argmax(f1)
    return t[idx], f1[idx], t, f1


# =========================
# 3. TRAIN
# =========================
def train():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_and_clean_data()

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Class'], random_state=42)

    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df, reference_df=train_df)
    test_df = engineer_features(test_df, reference_df=train_df)

    features = get_feature_columns()

    X_train, y_train = train_df[features], train_df['Class']
    X_val, y_val = val_df[features], val_df['Class']
    X_test, y_test = test_df[features], test_df['Class']

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    with mlflow.start_run() as run:

        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )

        # =====================
        # VALIDATION
        # =====================
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold, best_f1, thresholds, f1_scores = find_best_threshold(y_val, val_probs)

        # =====================
        # TEST
        # =====================
        test_probs = model.predict_proba(X_test)[:, 1]
        preds = (test_probs > threshold).astype(int)

        auprc = average_precision_score(y_test, test_probs)
        f1 = f1_score(y_test, preds)

        print("F1:", f1, "AUPRC:", auprc)

        # =====================
        # LOG PARAM
        # =====================
        mlflow.log_param("n_estimators", 500)
        mlflow.log_param("threshold", threshold)

        # =====================
        # LOG METRIC
        # =====================
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUPRC", auprc)

        # =====================
        # SIGNATURE
        # =====================
        signature = infer_signature(X_train, model.predict(X_train))

        # =====================
        # LOG MODEL + REGISTER
        # =====================
        import tempfile
        
        client = MlflowClient()

        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            
            # Use sklearn.save_model which doesn't trigger new API
            mlflow.sklearn.save_model(
                sk_model=model,
                path=model_path,
                signature=signature
            )
            
            mlflow.log_artifacts(tmpdir, artifact_path="model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )

        # =====================
        # ARTIFACTS
        # =====================

        # Confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_test, preds)
        plt.savefig("cm.png")
        mlflow.log_artifact("cm.png")
        plt.clf()

        # PR curve
        plt.plot(thresholds, f1_scores[:-1])
        plt.title("Threshold vs F1")
        plt.savefig("threshold.png")
        mlflow.log_artifact("threshold.png")
        plt.clf()

        # Feature importance
        plot_importance(model)
        plt.savefig("importance.png")
        mlflow.log_artifact("importance.png")
        plt.clf()


        # =====================
        # AUTO PROMOTE
        # =====================

        latest_versions = client.get_latest_versions(MODEL_NAME, stages=None)

        if not latest_versions:
            print("No model versions found, skipping promotion")
        else:
            new_version = latest_versions[-1].version

            if f1 > 0.85:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=new_version,
                    stage="Production"
                )
                print("Promoted to Production")

    # =====================
    # SAVE LOCAL BACKUP
    # =====================
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "backup.pkl"))


if __name__ == "__main__":
    train()