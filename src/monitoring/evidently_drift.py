import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EvidentlyDriftMonitor:
    def __init__(
        self,
        reference_data: pd.DataFrame,
        model,
        feature_columns: List[str],
        drift_threshold: float = 0.1,
        output_dir: str = "monitoring/reports",
    ):
        self.reference_data = reference_data
        self.model = model
        self.feature_columns = feature_columns
        self.drift_threshold = drift_threshold
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self._prepare_column_mapping()
    
    def _prepare_column_mapping(self):
        # Try multiple import paths for different Evidently versions
        ColumnMapping = None
        
        try:
            from evidently.pipeline.column_mapping import ColumnMapping
        except ImportError:
            try:
                from evidently.legacy.pipeline.column_mapping import ColumnMapping
            except ImportError:
                try:
                    from evidently.core.pipeline.column_mapping import ColumnMapping
                except ImportError:
                    logger.warning("Could not find ColumnMapping, using basic dict")
                    ColumnMapping = None
        
        if ColumnMapping is None:
            # Fallback: use simple dict-based mapping
            self.ColumnMapping = None
            logger.warning("Using simple column mapping (no ColumnMapping class)")
            return
        
        self.ColumnMapping = ColumnMapping
        
        numerical_features = []
        categorical_features = []
        
        for col in self.feature_columns:
            if col in self.reference_data.columns:
                if self.reference_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numerical_features.append(col)
                else:
                    categorical_features.append(col)
        
        self.column_mapping = self.ColumnMapping()
        self.column_mapping.numerical_features = numerical_features
        self.column_mapping.categorical_features = categorical_features
        
        logger.info(f"Column mapping: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Detecting data drift with Evidently AI...")
        
        try:
            # Try legacy preset first (v0.4.x compatible)
            try:
                from evidently.legacy.pipeline import Pipeline
                from evidently.legacy.tabs import DataDriftTab
                
                # Use legacy API
                ref_subset = self.reference_data[self.feature_columns].head(10000)
                curr_subset = current_data[self.feature_columns].head(10000)
                
                dashboard = Pipeline(steps=[DataDriftTab()])
                result = dashboard.calculate(ref_subset, curr_subset, self.column_mapping)
                
                # Parse results from legacy format
                drift_result = result.get('options', {}).get('data_drift', {}).get('drift_detected', False)
                drift_share = result.get('options', {}).get('data_drift', {}).get('drift_share', 0)
                
            except (ImportError, AttributeError):
                # Fallback: simple statistical drift detection
                logger.warning("Using fallback drift detection (simple statistics)")
                drift_result, drift_share = self._simple_drift_detection(current_data)
            
            return {
                "drift_detected": bool(drift_result),
                "drift_share": float(drift_share) if drift_share else 0.0,
                "feature_drifts": {},
                "threshold": self.drift_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evidently data drift detection error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "drift_detected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _simple_drift_detection(self, current_data: pd.DataFrame) -> tuple:
        """Fallback: Simple statistical drift detection using KS test"""
        from scipy import stats
        
        ref_subset = self.reference_data[self.feature_columns].head(5000)
        curr_subset = current_data[self.feature_columns].head(5000)
        
        drift_count = 0
        for col in ref_subset.columns:
            if col in curr_subset.columns:
                try:
                    ks_stat, ks_pval = stats.ks_2samp(ref_subset[col].dropna(), curr_subset[col].dropna())
                    if ks_pval < 0.05:  # Significant drift
                        drift_count += 1
                except:
                    pass
        
        drift_share = drift_count / len(ref_subset.columns)
        drift_detected = drift_share > self.drift_threshold
        
        return drift_detected, drift_share
    
    def detect_prediction_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Detecting prediction drift with Evidently AI...")
        
        try:
            current_predictions = self.model.predict_proba(current_data[self.feature_columns])[:, 1]
            reference_predictions = self.model.predict_proba(self.reference_data[self.feature_columns])[:, 1]
            
            # Simple statistical test for prediction drift
            from scipy import stats
            ks_stat, ks_pval = stats.ks_2samp(reference_predictions, current_predictions)
            
            pred_drift = ks_pval < 0.05
            
            return {
                "drift_detected": bool(pred_drift),
                "prediction_drift_share": float(ks_stat),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction drift detection error: {e}")
            return {
                "drift_detected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def detect_concept_drift(
        self,
        current_data: pd.DataFrame,
        current_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        logger.info("Detecting concept drift...")
        
        if current_labels is not None:
            return self._detect_with_labels(current_data, current_labels)
        else:
            return self.detect_prediction_drift(current_data)
    
    def _detect_with_labels(self, current_data: pd.DataFrame, current_labels: np.ndarray) -> Dict[str, Any]:
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            current_predictions = self.model.predict_proba(current_data[self.feature_columns])[:, 1]
            current_preds_binary = (current_predictions > 0.5).astype(int)
            
            precision = precision_score(current_labels, current_preds_binary, zero_division=0)
            recall = recall_score(current_labels, current_preds_binary, zero_division=0)
            f1 = f1_score(current_labels, current_preds_binary, zero_division=0)
            
            return {
                "concept_drift_detected": f1 < 0.5,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Concept drift detection error: {e}")
            return {
                "concept_drift_detected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_full_report(
        self,
        current_data: pd.DataFrame,
        current_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        logger.info("Generating full monitoring report with Evidently AI...")
        
        data_drift = self.detect_data_drift(current_data)
        concept_drift = self.detect_concept_drift(current_data, current_labels)
        prediction_drift = self.detect_prediction_drift(current_data)
        
        return {
            "data_drift": data_drift,
            "concept_drift": concept_drift,
            "prediction_drift": prediction_drift,
            "alert_triggered": (
                data_drift.get("drift_detected", False) or
                concept_drift.get("concept_drift_detected", False) or
                prediction_drift.get("drift_detected", False)
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        if filename is None:
            filename = f"evidently_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
        return filepath


def create_drift_monitor_from_paths(
    reference_data_path: str,
    model_path: str,
    feature_columns: List[str] = None,
    drift_threshold: float = 0.1,
    output_dir: str = "monitoring/reports"
) -> EvidentlyDriftMonitor:
    reference_data = pd.read_parquet(reference_data_path)
    
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "train"))
    from utils import engineer_features
    
    # Engineer features on reference data
    reference_data = engineer_features(reference_data)
    
    if feature_columns is None:
        from utils import get_feature_columns
        feature_columns = get_feature_columns()
    
    # Filter to only features that exist in reference data
    available_features = [f for f in feature_columns if f in reference_data.columns]
    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        logger.warning(f"Missing features in reference data: {missing}. Using {len(available_features)} features.")
    
    model_data = joblib.load(model_path)
    model = model_data["model"]
    
    return EvidentlyDriftMonitor(
        reference_data=reference_data,
        model=model,
        feature_columns=available_features,  # Use only available features
        drift_threshold=drift_threshold,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evidently AI Drift Monitor")
    parser.add_argument("--reference", type=str, required=True, help="Reference data path")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--current", type=str, required=True, help="Current data path")
    parser.add_argument("--output", type=str, default="monitoring/reports", help="Output directory")
    args = parser.parse_args()
    
    monitor = create_drift_monitor_from_paths(
        reference_data_path=args.reference,
        model_path=args.model,
        output_dir=args.output
    )
    
    current_data = pd.read_parquet(args.current)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "train"))
    from utils import engineer_features
    current_data = engineer_features(current_data)
    
    report = monitor.generate_full_report(current_data)
    
    print(json.dumps(report, indent=2))
    
    monitor.save_report(report)