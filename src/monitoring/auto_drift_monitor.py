import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "monitoring"))
from evidently_drift import EvidentlyDriftMonitor

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "mlops"))
from data_version import DataVersionManager

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))
from retrain_pipeline import RetrainPipeline

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "mlops"))
from s3_manager import S3DataManager

from dotenv import load_dotenv
load_dotenv()


class AutoRetrainTrigger:
    DRIFT_THRESHOLD = 0.05
    F1_THRESHOLD = 0.5
    MIN_NEW_DATA = 1000
    RETRAIN_COOLDOWN_HOURS = 24  # Min time between retrains
    STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
    LABELED_DIR = os.path.join(PROJECT_ROOT, "data", "labeled")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
    
    def __init__(
        self,
        reference_data_path: str,
        model_path: str,
        feature_columns: list,
        interval: int = 300,
        auto_retrain: bool = True
    ):
        self.reference_data_path = reference_data_path
        self.model_path = model_path
        self.feature_columns = feature_columns
        self.interval = interval
        self.auto_retrain = auto_retrain
        self.running = False
        self.drift_monitor = None
        self.data_manager = DataVersionManager()
        self.s3_manager = S3DataManager() if os.environ.get("USE_S3", "true").lower() == "true" else None
        self.webhook_url = os.environ.get("DRIFT_WEBHOOK_URL")
        self.last_retrain_time = None
    
    def _load_reference_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.reference_data_path)
        
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "train"))
        from utils import engineer_features
        
        df = engineer_features(df)
        
        return df
    
    def _load_model(self):
        if self.model_path.startswith("mlflow:"):
            stage_or_version = self.model_path.split(":", 1)[1]
            
            if stage_or_version.isdigit():
                stage = None
                version = int(stage_or_version)
            else:
                stage = stage_or_version
                version = None
            
            logger.info(f"Loading model from MLflow: stage={stage}, version={version}")
            
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "api"))
            from model_loader import load_model_from_mlflow, get_model_info
            
            model_data = load_model_from_mlflow(stage=stage, version=version)
            
            model_info = get_model_info(stage=stage, version=version)
            model_data["threshold"] = model_info["threshold"]
            model_data["metrics"] = model_info["metrics"]
            model_data["mlflow_version"] = model_info["version"]
            
            logger.info(f"MLflow model loaded: version={model_info['version']}, "
                       f"threshold={model_info['threshold']:.4f}, F1={model_info['metrics'].get('F1', 0):.4f}")
            
            self.model_data = model_data
            return model_data
        else:
            import joblib
            return joblib.load(self.model_path)
    
    def _load_staging_data(self) -> pd.DataFrame:
        USE_S3 = os.environ.get("USE_S3", "true").lower() == "true"

        if USE_S3 and self.s3_manager:
            try:
                files = self.s3_manager.list_files("realtime")
                if not files:
                    return pd.DataFrame()
                
                dfs = []
                for s3_path in files[-3:]:
                    try:
                        df = self.s3_manager.download_dataframe(s3_path, "parquet")
                        dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to download {s3_path}: {e}")
                
                if not dfs:
                    return pd.DataFrame()
                
                combined = pd.concat(dfs, ignore_index=True)
            except Exception as e:
                logger.warning(f"S3 staging load failed: {e}")
                combined = pd.DataFrame()
        else:
            if not os.path.exists(self.STAGING_DIR):
                return pd.DataFrame()
            
            files = [f for f in os.listdir(self.STAGING_DIR) if f.endswith('.parquet')]
            if not files:
                return pd.DataFrame()
            
            dfs = []
            for f in files:
                df = pd.read_parquet(os.path.join(self.STAGING_DIR, f))
                dfs.append(df)
            
            if not dfs:
                return pd.DataFrame()
            
            combined = pd.concat(dfs, ignore_index=True)
        
        if combined.empty:
            return pd.DataFrame()
        
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "train"))
        from utils import engineer_features
        
        combined = engineer_features(combined)
        
        return combined
    
    def _load_labeled_data(self) -> pd.DataFrame:
        USE_S3 = os.environ.get("USE_S3", "true").lower() == "true"

        if USE_S3 and self.s3_manager:
            try:
                files = self.s3_manager.list_files("labeled")
                if not files:
                    return pd.DataFrame()
                
                latest = files[-1]
                df = self.s3_manager.download_dataframe(latest, "parquet")
            except Exception as e:
                logger.warning(f"S3 labeled load failed: {e}")
                return pd.DataFrame()
        else:
            if not os.path.exists(self.LABELED_DIR):
                return pd.DataFrame()
            
            files = sorted([f for f in os.listdir(self.LABELED_DIR) if f.endswith('.parquet')])
            if not files:
                return pd.DataFrame()
            
            latest = os.path.join(self.LABELED_DIR, files[-1])
            df = pd.read_parquet(latest)
        
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "train"))
        from utils import engineer_features
        
        df = engineer_features(df)
        
        return df
    
    def _check_data_available(self) -> bool:
        staging_df = self._load_staging_data()
        labeled_df = self._load_labeled_data()
        
        total_records = len(staging_df) + len(labeled_df)
        
        if total_records < self.MIN_NEW_DATA:
            logger.info(f"Not enough data for retrain: {total_records} < {self.MIN_NEW_DATA}")
            return False
        return True
    
    def _run_retrain(self) -> Dict[str, Any]:
        logger.info("Starting auto retrain...")
        
        try:
            pipeline = RetrainPipeline(version=None)
            result = pipeline.train()
            
            if result.get('success'):
                logger.info(f"Retrain completed: F1={result['metrics']['F1']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _compare_and_deploy(
        self, 
        new_metrics: Dict[str, float]
    ) -> bool:
        old_f1 = 0
        current_version = None
        
        if self.model_path.startswith("mlflow:"):
            stage_or_version = self.model_path.split(":", 1)[1]
            
            if stage_or_version.isdigit():
                stage = None
                version = int(stage_or_version)
            else:
                stage = stage_or_version
                version = None
            
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "api"))
            from model_loader import get_model_info
            
            try:
                model_info = get_model_info(stage=stage, version=version)
                old_f1 = model_info.get("metrics", {}).get("F1", 0)
                current_version = model_info.get("version")
                logger.info(f"Current MLflow model version: {current_version}, F1={old_f1:.4f}")
            except Exception as e:
                logger.warning(f"Could not get current model info from MLflow: {e}")
        elif os.path.exists(self.model_path):
            import joblib
            old_model_data = joblib.load(self.model_path)
            old_f1 = old_model_data.get("metrics", {}).get("F1", 0)
        else:
            logger.warning(f"Model path not found: {self.model_path}")
            return False
        
        new_f1 = new_metrics.get("F1", 0)
        
        logger.info(f"Model comparison: old F1={old_f1:.4f}, new F1={new_f1:.4f}")
        
        improvement = new_f1 - old_f1
        should_deploy = improvement > 0 or new_f1 > old_f1
        
        if should_deploy:
            logger.info(f"New model has better F1 (improvement: {improvement:.4f}). "
                       f"Will be deployed via GitHub Actions workflow.")
        else:
            logger.info(f"New model NOT deployed (no improvement)")
        
        return should_deploy
    
    def _run_drift_check(self) -> Dict[str, Any]:
        logger.info("Running drift check...")
        
        try:
            reference_data = self._load_reference_data()
            model_data = self._load_model()
            model = model_data["model"]
            threshold = model_data.get("threshold", 0.5)
            
            self.drift_monitor = EvidentlyDriftMonitor(
                reference_data=reference_data,
                model=model,
                feature_columns=self.feature_columns,
                drift_threshold=self.DRIFT_THRESHOLD
            )
            
            staging_df = self._load_staging_data()
            
            if staging_df.empty:
                return {"status": "no_data", "drift_detected": False}
            
            report = self.drift_monitor.generate_full_report(staging_df)
            
            logger.info(f"Drift report: alert={report['alert_triggered']}")
            
            return report
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _process_drift_alert(self, report: Dict[str, Any]) -> bool:
        should_trigger = False
        
        if report.get("alert_triggered"):
            data_drift = report.get("data_drift", {})
            concept_drift = report.get("concept_drift", {})
            
            logger.info(f"Draft alert: data_drift={data_drift.get('drift_detected')}, "
                      f"concept_drift={concept_drift.get('concept_drift_detected')}")
            
            max_psi = data_drift.get("max_psi", 0)
            if max_psi > 0.1:
                logger.warning(f"PSI drift detected: {max_psi:.2f} > 0.1 threshold!")
            
            should_trigger = True
        
        if should_trigger:
            logger.info("Committing and pushing data to Git...")
            if self._commit_and_push_data():
                logger.info("Data pushed to Git, triggering retrain workflow...")
            else:
                logger.warning("Failed to push data, still triggering webhook...")
            
            self._send_webhook_alert(report)
        
        if should_trigger and self.auto_retrain:
            # Check cooldown to avoid too frequent retrain
            if self.last_retrain_time:
                hours_since = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
                if hours_since < self.RETRAIN_COOLDOWN_HOURS:
                    logger.info(f"Retrain on cooldown. {self.RETRAIN_COOLDOWN_HOURS - hours_since:.1f}h remaining. Skipping...")
                    return False
            
            if self._check_data_available():
                retrain_result = self._run_retrain()
                
                if retrain_result.get('success'):
                    self.last_retrain_time = datetime.now()
                    self._compare_and_deploy(retrain_result.get('metrics', {}))
                    return True
            else:
                logger.info("Not enough data for retrain, skipping...")
        
        return False
    
    def _commit_and_push_data(self) -> bool:
        import subprocess
        import os
        
        data_dirs = ["data/staging", "data/labeled", "data/realtime"]
        
        try:
            os.chdir(PROJECT_ROOT)
            
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    for root, dirs, files in os.walk(data_dir):
                        for file in files:
                            if file.endswith('.parquet'):
                                file_path = os.path.join(root, file)
                                result = subprocess.run(
                                    ["git", "add", file_path],
                                    capture_output=True
                                )
            
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if not result.stdout.strip():
                logger.info("No new data files to commit")
                return False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            commit_msg = f"Auto retrain: adding new data for drift retrain {timestamp}"
            
            subprocess.run(["git", "config", "user.email", "auto@drift.ml"], check=True)
            subprocess.run(["git", "config", "user.name", "Auto Drift Monitor"], check=True)
            
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            
            logger.info("Pushing data to Git...")
            subprocess.run(["git", "push"], check=True, capture_output=True)
            
            logger.info("Data pushed to Git successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to commit/push data: {e}")
            return False
    
    def _send_webhook_alert(self, report: Dict[str, Any]) -> bool:
        if not self.webhook_url:
            logger.warning("DRIFT_WEBHOOK_URL not set, skipping webhook")
            return False
        
        try:
            import requests
            
            token = os.environ.get("PAT_TOKEN", "")
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            payload = {
                "event_type": "drift_alert",
                "client_payload": {
                    "trigger_type": "drift_alert",
                    "psi": report.get("data_drift", {}).get("max_psi", 0),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"Sending webhook to: {self.webhook_url}")
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            logger.info(f"Webhook response: {response.status_code} - {response.text[:200]}")
            
            if response.status_code in [200, 201, 202, 204]:
                logger.info(f"Webhook sent successfully: {response.status_code}")
                return True
            else:
                logger.error(f"Webhook failed: {response.status_code}")
                return False
                
        except ImportError:
            logger.warning("requests not installed, skipping webhook")
            return False
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False
    
    def run(self):
        logger.info("=" * 60)
        logger.info("AUTO DRIFT MONITOR + RETRAIN")
        logger.info(f"Interval: {self.interval}s")
        logger.info(f"Auto retrain: {self.auto_retrain}")
        logger.info("=" * 60)
        
        self.running = True
        
        while self.running:
            try:
                report = self._run_drift_check()
                
                if report.get("alert_triggered"):
                    self._process_drift_alert(report)
                else:
                    logger.info("No drift detected")
                
            except Exception as e:
                logger.error(f"Error in drift monitor: {e}")
            
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
        
        logger.info("Drift monitor stopped")
    
    def stop(self):
        self.running = False


def run_auto_drift_monitor(
    reference_data_path: str,
    model_path: str,
    feature_columns: list,
    interval: int = 300,
    auto_retrain: bool = True
):
    trigger = AutoRetrainTrigger(
        reference_data_path=reference_data_path,
        model_path=model_path,
        feature_columns=feature_columns,
        interval=interval,
        auto_retrain=auto_retrain
    )
    
    try:
        trigger.run()
    except KeyboardInterrupt:
        logger.info("Stopping drift monitor...")
        trigger.stop()


if __name__ == "__main__":
    import argparse
    import joblib
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_REFERENCE = os.path.join(PROJECT_ROOT, "data", "train", "train_full.parquet")
    
    parser = argparse.ArgumentParser(description="Auto Drift Monitor with Retrain")
    parser.add_argument("--reference", type=str, default=DEFAULT_REFERENCE, 
                        help=f"Reference data path (default: {DEFAULT_REFERENCE})")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model path or 'mlflow:Production' / 'mlflow:latest' / 'mlflow:<version>'")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--no-auto-retrain", action="store_true", help="Disable auto retrain")
    parser.add_argument("--webhook-url", type=str, default=None, help="Webhook URL for GitHub Actions")
    parser.add_argument("--pat-token", type=str, default=None, help="Personal Access Token for webhook")
    args = parser.parse_args()
    
    if args.webhook_url:
        os.environ["DRIFT_WEBHOOK_URL"] = args.webhook_url
    if args.pat_token:
        os.environ["PAT_TOKEN"] = args.pat_token
    
    if args.model.startswith("mlflow:"):
        stage_or_version = args.model.split(":", 1)[1]
        
        if stage_or_version.isdigit():
            stage = None
            version = int(stage_or_version)
        else:
            stage = stage_or_version
            version = None
        
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "api"))
        from model_loader import load_model_from_mlflow, get_model_info
        
        model_data = load_model_from_mlflow(stage=stage, version=version)
        model_info = get_model_info(stage=stage, version=version)
        features = model_data.get("features", [])
        
        print(f"Loaded MLflow model: version={model_info['version']}, "
              f"threshold={model_info['threshold']:.4f}, F1={model_info['metrics'].get('F1', 0):.4f}")
    else:
        model_data = joblib.load(args.model)
        features = model_data.get("features", [])
    
    run_auto_drift_monitor(
        reference_data_path=args.reference,
        model_path=args.model,
        feature_columns=features,
        interval=args.interval,
        auto_retrain=not args.no_auto_retrain
    )