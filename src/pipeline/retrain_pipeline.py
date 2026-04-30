import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    f1_score,
    precision_recall_curve
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from utils import engineer_features, get_feature_columns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlops"))
from data_version import DataVersionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


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
        
        os.makedirs(model_dir, exist_ok=True)
    
    def load_training_data(self) -> pd.DataFrame:
        logger.info(f"Loading training data version: {self.version or 'latest'}")
        
        df = self.data_manager.load_version(self.version)
        
        if df.empty:
            raise ValueError(f"No training data found for version: {self.version}")
        
        logger.info(f"Loaded {len(df)} training records")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        df = df.sort_values("Time").reset_index(drop=True)
        
        df = engineer_features(df)
        
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df['Class'],
            random_state=42
        )
        
        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_size,
            stratify=train_df['Class'],
            random_state=42
        )
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        feature_cols = get_feature_columns()
        
        X_train, y_train = train_df[feature_cols], train_df['Class']
        X_val, y_val = val_df[feature_cols], val_df['Class']
        X_test, y_test = test_df[feature_cols], test_df['Class']
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def find_best_threshold(self, y_true, y_probs):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        return thresholds[best_idx], f1_scores[best_idx]
    
    def train(self) -> Dict[str, Any]:
        logger.info("Starting retrain pipeline...")
        
        start_time = datetime.now()
        
        df = self.load_training_data()
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = self.prepare_data(df)
        
        logger.info("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE: {len(X_train_resampled)} samples "
                  f"(Class 0: {(y_train_resampled==0).sum()}, "
                  f"Class 1: {(y_train_resampled==1).sum()})")
        
        counter = y_train_resampled.value_counts()
        scale_pos_weight = counter[0] / counter[1]
        
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")
        
        mlflow.set_experiment("FraudGuard_XGBoost_Retrain")
        
        with mlflow.start_run():
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=42
            )
            
            model.fit(X_train_resampled, y_train_resampled)
            
            mlflow.log_param("smote_applied", True)
            
            val_probs = model.predict_proba(X_val)[:, 1]
            best_threshold, best_f1 = self.find_best_threshold(y_val, val_probs)
            
            logger.info(f"Best threshold: {best_threshold:.4f}, Best F1: {best_f1:.4f}")
            
            test_probs = model.predict_proba(X_test)[:, 1]
            test_preds = (test_probs > best_threshold).astype(int)
            
            auprc = average_precision_score(y_test, test_probs)
            f1 = f1_score(y_test, test_preds)
            
            logger.info(f"\nAUPRC: {auprc:.4f}, F1: {f1:.4f}")
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(y_test, test_preds))
            
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            mlflow.log_param("threshold", best_threshold)
            mlflow.log_param("version", self.version or "latest")
            
            mlflow.log_metric("AUPRC", auprc)
            mlflow.log_metric("F1", f1)
            
            mlflow.sklearn.log_model(model, "fraud_model_retrain")
        
        model_data = {
            "model": model,
            "threshold": best_threshold,
            "features": feature_cols,
            "version": self.version or "latest",
            "trained_at": datetime.now().isoformat(),
            "smote_applied": True,
            "metrics": {
                "AUPRC": float(auprc),
                "F1": float(best_f1)
            }
        }
        
        model_path = os.path.join(self.model_dir, "fraud_model_retrain.pkl")
        joblib.dump(model_data, model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "success": True,
            "version": self.version,
            "model_path": model_path,
            "threshold": float(best_threshold),
            "metrics": {
                "AUPRC": float(auprc),
                "F1": float(f1),
                "best_f1_val": float(best_f1)
            },
            "training_duration_seconds": duration,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
    
    def evaluate_against_baseline(
        self,
        new_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        improvement_threshold: float = 0.05
    ) -> Dict[str, Any]:
        logger.info("Evaluating against baseline...")
        
        f1_improvement = new_metrics.get("F1", 0) - baseline_metrics.get("F1", 0)
        auprc_improvement = new_metrics.get("AUPRC", 0) - baseline_metrics.get("AUPRC", 0)
        
        should_deploy = (
            f1_improvement > improvement_threshold or
            new_metrics.get("F1", 0) > baseline_metrics.get("F1", 0)
        )
        
        return {
            "should_deploy": should_deploy,
            "f1_improvement": f1_improvement,
            "auprc_improvement": auprc_improvement,
            "new_metrics": new_metrics,
            "baseline_metrics": baseline_metrics
        }


def run_retrain(version: Optional[str] = None) -> Dict[str, Any]:
    pipeline = RetrainPipeline(version=version)
    
    result = pipeline.train()
    
    print("\n" + "=" * 60)
    print("RETRAIN RESULT")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Version: {result['version']}")
    print(f"Model Path: {result['model_path']}")
    print(f"Threshold: {result['threshold']:.4f}")
    print(f"Metrics:")
    print(f"  AUPRC: {result['metrics']['AUPRC']:.4f}")
    print(f"  F1: {result['metrics']['F1']:.4f}")
    print(f"Training Duration: {result['training_duration_seconds']:.2f}s")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None, help="Data version to use")
    args = parser.parse_args()
    
    run_retrain(args.version)