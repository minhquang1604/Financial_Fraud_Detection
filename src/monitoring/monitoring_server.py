import os
import sys
import logging
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from utils import engineer_features, get_feature_columns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "monitoring"))
from drift_detector import DriftMonitor
from metrics_exporter import PrometheusMetricsExporter, MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MONITORING_REPORT_DIR = os.path.join(PROJECT_ROOT, "monitoring", "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


class MonitoringServer:
    def __init__(
        self,
        model_path: str = None,
        reference_data_path: str = None,
        prometheus_port: int = 8001,
        check_interval: int = 300
    ):
        self.model_path = model_path or os.path.join(MODEL_DIR, "fraud_model.pkl")
        self.reference_data_path = reference_data_path or os.path.join(
            DATA_DIR, "processed", "cleaned_fraud_data.parquet"
        )
        self.prometheus_port = prometheus_port
        self.check_interval = check_interval
        
        self.model = None
        self.model_data = None
        self.feature_columns = None
        self.reference_data = None
        self.drift_monitor = None
        self.metrics_exporter = None
        self.metrics_collector = MetricsCollector()
        
        os.makedirs(MONITORING_REPORT_DIR, exist_ok=True)
    
    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}")
            return False
        
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data.get("model") or self.model_data.get("metadata", {}).get("model")
        
        if self.model is None:
            logger.error("Model not found in model file")
            return False
        
        self.feature_columns = self.model_data.get(
            "features", 
            get_feature_columns()
        )
        
        logger.info(f"Model loaded with {len(self.feature_columns)} features")
        
        return True
    
    def load_reference_data(self):
        logger.info(f"Loading reference data from {self.reference_data_path}")
        
        if not os.path.exists(self.reference_data_path):
            logger.warning(f"Reference data not found at {self.reference_data_path}")
            return False
        
        self.reference_data = pd.read_parquet(self.reference_data_path)
        
        logger.info(f"Reference data loaded: {len(self.reference_data)} records")
        
        return True
    
    def initialize(self):
        logger.info("Initializing monitoring server...")
        
        if not self.load_model():
            raise RuntimeError("Failed to load model")
        
        if not self.load_reference_data():
            raise RuntimeError("Failed to load reference data")
        
        self.drift_monitor = DriftMonitor(
            reference_data=self.reference_data,
            model=self.model,
            feature_columns=self.feature_columns,
            drift_threshold=0.05,
            output_dir=MONITORING_REPORT_DIR
        )
        
        self.metrics_exporter = PrometheusMetricsExporter(
            port=self.prometheus_port
        )
        
        self.metrics_exporter.start_server()
        
        logger.info(f"Monitoring server initialized on port {self.prometheus_port}")
        
        return True
    
    def check_and_export_metrics(self):
        logger.info("Running scheduled drift check...")
        
        try:
            current_data = self.reference_data.sample(
                min(1000, len(self.reference_data)), 
                random_state=42
            )
            
            report = self.drift_monitor.generate_full_report(current_data)
            
            if report.get("alert_triggered"):
                logger.warning(f"ALERT: Drift detected - {report}")
                
                self.metrics_exporter.export_drift_metrics(
                    data_drift_score=report["data_drift"]["drift_score"],
                    concept_drift_score=report["concept_drift"].get("ks_statistic", 0),
                    alert_triggered=True
                )
                
                filepath = self.drift_monitor.save_report(
                    report, 
                    f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                
                self._trigger_retrainWebhook(report)
            else:
                logger.info("No drift detected")
                
                self.metrics_exporter.export_drift_metrics(
                    data_drift_score=report["data_drift"]["drift_score"],
                    concept_drift_score=report["concept_drift"].get("ks_statistic", 0),
                    alert_triggered=False
                )
            
            pred_dist = report.get("prediction_distribution", {})
            self.metrics_exporter.export_data_stats(len(self.reference_data))
            
            logger.info(f"Drift check completed: {report['data_drift']['drift_score']:.4f}")
            
            return report
        
        except Exception as e:
            logger.error(f"Error during drift check: {e}")
            return {}
    
    def _trigger_retrain_webhook(self, report: Dict[str, Any]):
        webhook_url = os.environ.get("RETRAIN_WEBHOOK_URL")
        
        if not webhook_url:
            logger.warning("RETRAIN_WEBHOOK_URL not set, skipping webhook trigger")
            return
        
        try:
            import requests
            
            payload = {
                "trigger_type": "drift_alert",
                "timestamp": datetime.now().isoformat(),
                "report": report,
                "model_path": self.model_path
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            
            logger.info(f"Webhook triggered: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error triggering webhook: {e}")
    
    def run(self):
        logger.info("Starting monitoring server...")
        
        self.initialize()
        
        self.check_and_export_metrics()
        
        logger.info(f"Running drift checks every {self.check_interval} seconds...")
        
        while True:
            try:
                time.sleep(self.check_interval)
                self.check_and_export_metrics()
            
            except KeyboardInterrupt:
                logger.info("Monitoring server stopped")
                break
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)


def run_monitoring_server():
    model_path = os.environ.get("MODEL_PATH")
    prometheus_port = int(os.environ.get("PROMETHEUS_PORT", "8001"))
    check_interval = int(os.environ.get("CHECK_INTERVAL", "300"))
    
    server = MonitoringServer(
        model_path=model_path,
        prometheus_port=prometheus_port,
        check_interval=check_interval
    )
    
    server.run()


if __name__ == "__main__":
    run_monitoring_server()