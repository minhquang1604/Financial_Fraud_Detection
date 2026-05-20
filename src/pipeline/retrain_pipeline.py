import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix
)

from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


# =====================================================
# IMPORTS
# =====================================================

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "train")
)

from utils import engineer_features, get_feature_columns

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "mlops")
)

from data_version import DataVersionManager
from s3_manager import S3DataManager


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================================
# CONFIG
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

MODEL_NAME = "FraudDetectionModel"

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

USE_S3 = os.environ.get(
    "USE_S3",
    "true"
).lower() == "true"


# =====================================================
# RETRAIN PIPELINE
# =====================================================

class RetrainPipeline:

    def __init__(
        self,
        version: Optional[str] = None,
        model_dir: str = MODEL_DIR,
        test_size: float = 0.2,
        val_size: float = 0.2
    ):

        self.version = version
        self.model_dir = model_dir
        self.test_size = test_size
        self.val_size = val_size

        self.data_manager = DataVersionManager()

        self.s3_manager = (
            S3DataManager()
            if USE_S3
            else None
        )

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )

    # =================================================
    # LOAD DATA
    # =================================================

    def load_training_data(self) -> pd.DataFrame:

        logger.info(
            f"Loading training data version: "
            f"{self.version or 'latest'}"
        )

        if USE_S3 and self.s3_manager:

            train_files = self.s3_manager.list_files(
                "train"
            )

            if train_files:

                latest_file = train_files[-1]

                logger.info(
                    f"Downloading latest train file: "
                    f"{latest_file}"
                )

                df = self.s3_manager.download_dataframe(
                    latest_file,
                    "parquet"
                )

                logger.info(
                    f"Loaded {len(df)} rows from S3"
                )

                return df

        # fallback local
        train_path = os.path.join(
            PROJECT_ROOT,
            "data",
            "train",
            "train_full.parquet"
        )

        if os.path.exists(train_path):

            df = pd.read_parquet(train_path)

            logger.info(
                f"Loaded local train data: "
                f"{len(df)} rows"
            )

            return df

        raise ValueError(
            "No training data found"
        )

    # =================================================
    # FIND BEST THRESHOLD
    # =================================================

    def find_best_threshold(
        self,
        y_true,
        y_probs
    ):

        precisions, recalls, thresholds = (
            precision_recall_curve(
                y_true,
                y_probs
            )
        )

        f1_scores = (
            2 * precisions * recalls
            / (precisions + recalls + 1e-8)
        )

        best_idx = np.argmax(f1_scores)

        best_threshold = thresholds[best_idx]

        best_f1 = f1_scores[best_idx]

        logger.info(
            f"Best threshold={best_threshold:.6f}"
        )

        logger.info(
            f"Best validation F1={best_f1:.4f}"
        )

        return best_threshold, best_f1

    # =================================================
    # PREPARE DATA
    # =================================================

    def prepare_data(self, df: pd.DataFrame):

        logger.info("Preparing data...")

        # =============================================
        # SORT
        # =============================================

        df = df.sort_values(
            "Time"
        ).reset_index(drop=True)

        # =============================================
        # REMOVE DUPLICATES
        # =============================================

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        logger.info(
            f"Removed duplicates: "
            f"{before - after}"
        )

        # =============================================
        # SPLIT FIRST
        # =============================================

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df["Class"],
            random_state=42
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_size,
            stratify=train_df["Class"],
            random_state=42
        )

        logger.info(
            f"Train={len(train_df)}, "
            f"Val={len(val_df)}, "
            f"Test={len(test_df)}"
        )

        # =============================================
        # FEATURE ENGINEERING
        # IMPORTANT:
        # FIT ONLY ON TRAIN
        # =============================================

        train_df = engineer_features(
            train_df
        )

        val_df = engineer_features(
            val_df,
            reference_df=train_df
        )

        test_df = engineer_features(
            test_df,
            reference_df=train_df
        )

        feature_cols = get_feature_columns()

        X_train = train_df[feature_cols]
        y_train = train_df["Class"]

        X_val = val_df[feature_cols]
        y_val = val_df["Class"]

        X_test = test_df[feature_cols]
        y_test = test_df["Class"]

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols
        )

    # =================================================
    # TRAIN
    # =================================================

    def train(self) -> Dict[str, Any]:

        logger.info(
            "Starting retrain pipeline..."
        )

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            "FraudGuard_XGBoost_Retrain"
        )

        start_time = datetime.now()

        # =============================================
        # LOAD DATA
        # =============================================

        df = self.load_training_data()

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols
        ) = self.prepare_data(df)

        # =============================================
        # SCALE POS WEIGHT
        # =============================================

        scale_pos_weight = (
            y_train.value_counts()[0]
            / y_train.value_counts()[1]
        )

        logger.info(
            f"scale_pos_weight="
            f"{scale_pos_weight:.2f}"
        )

        # =============================================
        # START RUN
        # =============================================

        with mlflow.start_run() as run:

            # =========================================
            # MODEL
            # =========================================

            model = XGBClassifier(
                n_estimators=200,
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

            # =========================================
            # VALIDATION
            # =========================================

            val_probs = model.predict_proba(
                X_val
            )[:, 1]

            threshold, best_f1 = (
                self.find_best_threshold(
                    y_val,
                    val_probs
                )
            )

            # =========================================
            # TEST
            # =========================================

            test_probs = model.predict_proba(
                X_test
            )[:, 1]

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

            logger.info(
                f"AUPRC={auprc:.4f}"
            )

            logger.info(
                f"F1={f1:.4f}"
            )

            # =========================================
            # CONFUSION MATRIX
            # =========================================

            cm = confusion_matrix(
                y_test,
                preds
            )

            logger.info(
                f"\nConfusion Matrix:\n{cm}"
            )

            logger.info(
                "\n"
                + classification_report(
                    y_test,
                    preds
                )
            )

            # =========================================
            # LOG PARAMS
            # =========================================

            mlflow.log_param(
                "n_estimators",
                500
            )

            mlflow.log_param(
                "learning_rate",
                0.05
            )

            mlflow.log_param(
                "scale_pos_weight",
                float(scale_pos_weight)
            )

            mlflow.log_param(
                "threshold",
                float(threshold)
            )

            mlflow.log_param(
                "data_version",
                self.version or "latest"
            )

            # =========================================
            # LOG METRICS
            # =========================================

            mlflow.log_metric(
                "AUPRC",
                float(auprc)
            )

            mlflow.log_metric(
                "F1",
                float(f1)
            )

            mlflow.log_metric(
                "best_f1_val",
                float(best_f1)
            )

            # =========================================
            # SIGNATURE
            # =========================================

            signature = infer_signature(
                X_train,
                model.predict(X_train)
            )

            # =========================================
            # LOG MODEL
            # =========================================

            logger.info("Logging model to MLflow...")

            model_path = os.path.join(self.model_dir, "fraud_model_retrain")
            mlflow.sklearn.save_model(
                sk_model=model,
                path=model_path
            )
            
            mlflow.log_artifact(model_path)

            logger.info("Model saved and logged successfully")

            # =========================================
            # REGISTER MODEL
            # =========================================

            client = MlflowClient()

            try:
                client.create_registered_model(MODEL_NAME)
                logger.info(f"Created registered model: {MODEL_NAME}")
            except Exception as e:
                logger.info(f"Registered model may already exist: {e}")

            run_id = run.info.run_id
            
            model_uri = f"runs:/{run_id}/model"

            mv = client.create_model_version(
                name=MODEL_NAME,
                source=model_uri,
                run_id=run_id
            )

            logger.info(f"Created model version: {mv.version}")

            try:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=mv.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info(f"Set version {mv.version} as Production")
            except Exception as e:
                logger.warning(f"Could not set Production stage: {e}")

            logger.info(
                f"Registered model "
                f"version={mv.version}"
            )

        # =============================================
        # SAVE LOCAL BACKUP
        # =============================================

        model_data = {
            "model": model,
            "threshold": float(threshold),
            "features": feature_cols,
            "version": self.version or "latest",
            "trained_at": datetime.now().isoformat(),
            "metrics": {
                "AUPRC": float(auprc),
                "F1": float(f1),
                "best_f1_val": float(best_f1)
            }
        }

        local_model_path = os.path.join(
            self.model_dir,
            "fraud_model_retrain.pkl"
        )

        joblib.dump(
            model_data,
            local_model_path
        )

        logger.info(
            f"Saved local backup: "
            f"{local_model_path}"
        )

        duration = (
            datetime.now() - start_time
        ).total_seconds()

        return {
            "success": True,
            "version": self.version,
            "model_path": local_model_path,
            "threshold": float(threshold),
            "metrics": {
                "AUPRC": float(auprc),
                "F1": float(f1),
                "best_f1_val": float(best_f1)
            },
            "training_duration_seconds": duration,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }


# =====================================================
# RUN RETRAIN
# =====================================================

def run_retrain(
    version: Optional[str] = None
):

    pipeline = RetrainPipeline(
        version=version
    )

    result = pipeline.train()

    print("\n" + "=" * 60)
    print("RETRAIN RESULT")
    print("=" * 60)

    print(f"Success: {result['success']}")
    print(f"Version: {result['version']}")
    print(f"Threshold: {result['threshold']:.6f}")

    print("\nMetrics:")
    print(f"AUPRC: {result['metrics']['AUPRC']:.4f}")
    print(f"F1: {result['metrics']['F1']:.4f}")

    print(
        f"\nTraining Duration: "
        f"{result['training_duration_seconds']:.2f}s"
    )

    print("=" * 60)

    return result


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Training data version"
    )

    args = parser.parse_args()

    run_retrain(args.version)