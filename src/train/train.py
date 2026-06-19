import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.train.utils import engineer_features, get_feature_columns


# =====================================================
# CONFIG
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATA_PATH = os.path.join(
    DATA_DIR,
    "train",
    "train_full.parquet"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

ALB_DNS = os.environ.get("ALB_DNS", "")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"http://{ALB_DNS}:5000" if ALB_DNS else "http://localhost:5000"
)

EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "FraudGuard_XGBoost")
MODEL_NAME = os.environ.get("MODEL_NAME", "FraudDetectionModel")


# =====================================================
# LOAD DATA
# =====================================================

def load_and_clean_data():

    print(f"Loading training data from: {TRAIN_DATA_PATH}")

    df = pd.read_parquet(TRAIN_DATA_PATH)

    df = df.sort_values("Time").reset_index(drop=True)

    print(f"Loaded {len(df)} records")

    return df


# =====================================================
# FIND BEST THRESHOLD
# =====================================================

def find_best_threshold(y_true, y_probs):

    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        y_probs
    )

    f1_scores = (
        2 * precisions * recalls
        / (precisions + recalls + 1e-8)
    )

    best_idx = np.argmax(f1_scores)

    return (
        thresholds[best_idx],
        f1_scores[best_idx],
        thresholds,
        f1_scores
    )


# =====================================================
# TRAIN
# =====================================================

def train():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment:
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
    else:
        client.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_and_clean_data()

    # =========================
    # SPLIT
    # =========================

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Class"],
        random_state=42
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["Class"],
        random_state=42
    )

    # =========================
    # FEATURE ENGINEERING
    # =========================

    train_df = engineer_features(train_df)

    val_df = engineer_features(
        val_df,
        reference_df=train_df
    )

    test_df = engineer_features(
        test_df,
        reference_df=train_df
    )

    features = get_feature_columns()

    X_train = train_df[features]
    y_train = train_df["Class"]

    X_val = val_df[features]
    y_val = val_df["Class"]

    X_test = test_df[features]
    y_test = test_df["Class"]

    scale_pos_weight = (
        y_train.value_counts()[0]
        / y_train.value_counts()[1]
    )

    # =================================================
    # MLFLOW RUN
    # =================================================

    with mlflow.start_run() as run:

        # =============================================
        # MODEL
        # =============================================

        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=30
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # =============================================
        # VALIDATION
        # =============================================

        val_probs = model.predict_proba(X_val)[:, 1]

        (
            threshold,
            best_f1,
            thresholds,
            f1_scores
        ) = find_best_threshold(
            y_val,
            val_probs
        )

        # =============================================
        # TEST
        # =============================================

        test_probs = model.predict_proba(X_test)[:, 1]

        preds = (
            test_probs > threshold
        ).astype(int)

        auprc = average_precision_score(
            y_test,
            test_probs
        )

        f1 = f1_score(
            y_test,
            preds
        )

        print(f"F1: {f1:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        # =============================================
        # LOG PARAMS
        # =============================================

        mlflow.log_param(
            "n_estimators",
            500
        )

        mlflow.log_param(
            "learning_rate",
            0.05
        )

        mlflow.log_param(
            "threshold",
            float(threshold)
        )

        # =============================================
        # LOG METRICS
        # =============================================

        mlflow.log_metric(
            "F1",
            float(f1)
        )

        mlflow.log_metric(
            "AUPRC",
            float(auprc)
        )

        # =============================================
        # SIGNATURE
        # =============================================

        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        # =============================================
        # LOG MODEL
        # =============================================

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=None
        )

        # =============================================
        # REGISTER MODEL
        # =============================================

        client = MlflowClient()

        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass

        run_id = run.info.run_id

        # IMPORTANT FIX
        model_uri = f"runs:/{run_id}/model"

        mv = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )

        print(f"Registered model version: {mv.version}")

        # =============================================
        # CONFUSION MATRIX
        # =============================================

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            preds
        )

        plt.savefig("cm.png")

        mlflow.log_artifact("cm.png")

        plt.clf()

        # =============================================
        # THRESHOLD GRAPH
        # =============================================

        plt.plot(
            thresholds,
            f1_scores[:-1]
        )

        plt.title("Threshold vs F1")

        plt.savefig("threshold.png")

        mlflow.log_artifact("threshold.png")

        plt.clf()

        # =============================================
        # FEATURE IMPORTANCE
        # =============================================

        plot_importance(model)

        plt.savefig("importance.png")

        mlflow.log_artifact("importance.png")

        plt.clf()

        # =============================================
        # AUTO PROMOTE
        # =============================================

        if f1 > 0.85:

            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=mv.version,
                stage="Production"
            )

            print(
                f"Promoted version {mv.version} to Production"
            )

    # =================================================
    # LOCAL BACKUP
    # =================================================

    os.makedirs(MODEL_DIR, exist_ok=True)

    backup_path = os.path.join(
        MODEL_DIR,
        "backup.pkl"
    )

    joblib.dump(model, backup_path)

    print(f"Backup saved: {backup_path}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    train()