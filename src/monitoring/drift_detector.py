import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import joblib
from evidently.dashboard import Dashboard
from evidently.tabs import (
    DataDriftTab,
    CatTargetDriftTab,
    RegressionPerformanceTab,
    ClassificationPerformanceTab,
)
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DriftMonitor:
    def __init__(
        self,
        reference_data: pd.DataFrame,
        model,
        feature_columns: List[str],
        drift_threshold: float = 0.05,
        output_dir: str = "monitoring/reports",
    ):
        self.reference_data = reference_data
        self.model = model
        self.feature_columns = feature_columns
        self.drift_threshold = drift_threshold
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.reference_predictions = self.model.predict_proba(
            reference_data[feature_columns]
        )[:, 1]
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Detecting data drift...")
        
        data_dift_report = Dashboard(
            tabs=[DataDriftTab()])
        
        data_dift_report.calculate(
            reference_data=self.reference_data[self.feature_columns],
            current_data=current_data[self.feature_columns],
        )
        
        report_path = os.path.join(
            self.output_dir, 
            f"data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        data_dift_report.save_html(report_path)
        
        drift_score = self._calculate_drift_score(
            self.reference_data[self.feature_columns],
            current_data[self.feature_columns]
        )
        
        psi_results = {}
        key_cols = ["V4", "Amount", "V17", "V12", "V14"]
        for col in key_cols:
            if col in self.reference_data.columns and col in current_data.columns:
                psi_result = self.calculate_psi(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                psi_results[col] = psi_result
        
        max_psi = 0.0
        max_psi_col = None
        for col, result in psi_results.items():
            if result["psi"] > max_psi:
                max_psi = result["psi"]
                max_psi_col = col
        
        return {
            "drift_detected": drift_score < self.drift_threshold or max_psi > 0.1,
            "drift_score": drift_score,
            "psi": psi_results,
            "max_psi": max_psi,
            "max_psi_column": max_psi_col,
            "threshold": self.drift_threshold,
            "report_path": report_path,
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_concept_drift(
        self, 
        current_data: pd.DataFrame, 
        current_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        logger.info("Detecting concept drift...")
        
        current_predictions = self.model.predict_proba(
            current_data[self.feature_columns]
        )[:, 1]
        
        if current_labels is not None:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            current_preds_binary = (current_predictions > 0.5).astype(int)
            
            precision = precision_score(current_labels, current_preds_binary, zero_division=0)
            recall = recall_score(current_labels, current_preds_binary, zero_division=0)
            f1 = f1_score(current_labels, current_preds_binary, zero_division=0)
            
            return {
                "concept_drift_detected": f1 < 0.5,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            from scipy import stats
            
            ks_stat, ks_pvalue = stats.ks_2samp(
                self.reference_predictions, 
                current_predictions
            )
            
            prediction_drift = float(ks_stat > 0.1)
            
            return {
                "concept_drift_detected": prediction_drift > self.drift_threshold,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_drift_score(
        self, 
        reference: pd.DataFrame, 
        current: pd.DataFrame
    ) -> float:
        drift_scores = []
        
        for col in reference.columns:
            if col in current.columns:
                stat, pvalue = scipy_stats.ks_2samp(
                    reference[col].dropna(), 
                    current[col].dropna()
                )
                drift_scores.append(pvalue)
        
        return float(np.mean(drift_scores)) if drift_scores else 1.0
    
    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        buckets: int = 10
    ) -> Dict[str, float]:
        try:
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            if min_val == max_val:
                return {"psi": 0.0, "status": "no_variation"}
            
            bins = np.linspace(min_val, max_val, buckets + 1)
            
            ref_counts, _ = np.histogram(reference, bins=bins)
            curr_counts, _ = np.histogram(current, bins=bins)
            
            ref_pcts = (ref_counts + 1) / (ref_counts.sum() + buckets)
            curr_pcts = (curr_counts + 1) / (curr_counts.sum() + buckets)
            
            ref_pcts = ref_pcts / ref_pcts.sum()
            curr_pcts = curr_pcts / curr_pcts.sum()
            
            ref_pcts = np.where(ref_pcts == 0, 0.0001, ref_pcts)
            curr_pcts = np.where(curr_pcts == 0, 0.0001, curr_pcts)
            
            psi = np.sum((curr_pcts - ref_pcts) * np.log(curr_pcts / ref_pcts))
            
            status = "stable"
            if psi > 0.25:
                status = "high_drift"
            elif psi > 0.1:
                status = "moderate_drift"
            elif psi > 0.05:
                status = "low_drift"
            
            return {
                "psi": float(psi),
                "status": status,
                "threshold": 0.1
            }
        except Exception as e:
            logger.error(f"PSI calculation error: {e}")
            return {"psi": 0.0, "status": "error", "error": str(e)}
    
    def get_prediction_distribution(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        predictions = self.model.predict_proba(data[self.feature_columns])[:, 1]
        
        return {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "q25": float(np.percentile(predictions, 25)),
            "q50": float(np.percentile(predictions, 50)),
            "q75": float(np.percentile(predictions, 75)),
            "fraud_ratio": float(np.mean(predictions > 0.5)),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_full_report(
        self,
        current_data: pd.DataFrame,
        current_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        logger.info("Generating full monitoring report...")
        
        data_drift = self.detect_data_drift(current_data)
        concept_drift = self.detect_concept_drift(current_data, current_labels)
        pred_dist = self.get_prediction_distribution(current_data)
        
        return {
            "data_drift": data_drift,
            "concept_drift": concept_drift,
            "prediction_distribution": pred_dist,
            "alert_triggered": (
                data_drift["drift_detected"] or 
                concept_drift["concept_drift_detected"]
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        if filename is None:
            filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
        return filepath