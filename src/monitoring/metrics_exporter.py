import os
import sys
import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "monitoring"))

from evidently_drift import EvidentlyDriftMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrometheusMetricsExporter:
    def __init__(
        self,
        reference_data_path: str,
        model_path: str,
        feature_columns: list,
        check_interval: int = 60,
        metrics_port: int = 8001
    ):
        self.reference_data_path = reference_data_path
        self.model_path = model_path
        self.feature_columns = feature_columns
        self.check_interval = check_interval
        self.metrics_port = metrics_port
        
        self.metrics = {
            "data_drift_score": 1.0,
            "concept_drift_score": 0.0,
            "model_f1_score": 0.0,
            "model_auprc": 0.0,
            "last_check_timestamp": 0,
            "drift_alert": 0
        }
        
        self.running = False
    
    def _load_reference_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.reference_data_path)
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "train"))
        from utils import engineer_features
        df = engineer_features(df)
        return df
    
    def _load_model(self):
        import joblib
        return joblib.load(self.model_path)
    
    def _run_drift_check(self):
        try:
            reference_data = self._load_reference_data()
            model_data = self._load_model()
            model = model_data["model"]
            
            drift_monitor = DriftMonitor(
                reference_data=reference_data,
                model=model,
                feature_columns=self.feature_columns,
                drift_threshold=0.1
            )
            
            staging_dir = os.path.join(PROJECT_ROOT, "data", "staging")
            files = [f for f in os.listdir(staging_dir) if f.endswith('.parquet')]
            if files:
                staging_df = pd.concat([
                    pd.read_parquet(os.path.join(staging_dir, f)) 
                    for f in files
                ], ignore_index=True)
                
                sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "train"))
                from utils import engineer_features
                staging_df = engineer_features(staging_df)
                
                report = drift_monitor.generate_full_report(staging_df)
                
                data_drift = report.get("data_drift", {})
                concept_drift = report.get("concept_drift", {})
                
                self.metrics["data_drift_score"] = data_drift.get("max_psi", 0.0)
                self.metrics["concept_drift_score"] = concept_drift.get("drift_score", 0.0)
                self.metrics["drift_alert"] = 1 if report.get("alert_triggered") else 0
                
                logger.info(f"Drift check: PSI={self.metrics['data_drift_score']:.4f}, "
                          f"Concept={self.metrics['concept_drift_score']:.4f}")
            
            self.metrics["last_check_timestamp"] = time.time()
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
    
    def _generate_prometheus_metrics(self) -> str:
        lines = []
        for name, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            elif isinstance(value, int):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"
    
    def _run_http_server(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    exporter = getattr(self.server, "exporter")
                    self.wfile.write(exporter._generate_prometheus_metrics().encode())
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status": "healthy"}')
                else:
                    self.send_response(404)
                    self.end_headers()
        
        server = HTTPServer(("0.0.0.0", self.metrics_port), MetricsHandler)
        server.exporter = self
        logger.info(f"Metrics server running on port {self.metrics_port}")
        server.serve_forever()
    
    def run(self):
        logger.info("=" * 60)
        logger.info("PROMETHEUS METRICS EXPORTER")
        logger.info(f"Metrics port: {self.metrics_port}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info("=" * 60)
        
        self.running = True
        
        metrics_thread = threading.Thread(target=self._run_http_server, daemon=True)
        metrics_thread.start()
        
        while self.running:
            self._run_drift_check()
            time.sleep(self.check_interval)
    
    def stop(self):
        self.running = False


def run_metrics_exporter(
    reference_data_path: str,
    model_path: str,
    feature_columns: list,
    check_interval: int = 60,
    metrics_port: int = 8001
):
    exporter = PrometheusMetricsExporter(
        reference_data_path=reference_data_path,
        model_path=model_path,
        feature_columns=feature_columns,
        check_interval=check_interval,
        metrics_port=metrics_port
    )
    
    try:
        exporter.run()
    except KeyboardInterrupt:
        logger.info("Stopping metrics exporter...")
        exporter.stop()


if __name__ == "__main__":
    import argparse
    import joblib
    
    parser = argparse.ArgumentParser(description="Prometheus Metrics Exporter")
    parser.add_argument("--reference", type=str, required=True, help="Reference data path")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--port", type=int, default=8001, help="Metrics port")
    args = parser.parse_args()
    
    model_data = joblib.load(args.model)
    features = model_data.get("features", [])
    
    run_metrics_exporter(
        reference_data_path=args.reference,
        model_path=args.model,
        feature_columns=features,
        check_interval=args.interval,
        metrics_port=args.port
    )