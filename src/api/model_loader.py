import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import mlflow
import mlflow.sklearn
import joblib
from src.train.utils import get_feature_columns


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "fraud_model_xgboost"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model")


def load_model_from_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = mlflow.tracking.MlflowClient()
    
    experiments = client.search_experiments(filter_string="name='FraudGuard_XGBoost_Improved'")
    if not experiments:
        experiments = client.search_experiments(filter_string="name='Default'")
    
    if not experiments:
        raise ValueError("No experiment found in MLflow")
    
    experiment_id = experiments[0].experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status='FINISHED'",
        max_results=1,
        order_by=["metrics.AUPRC DESC"]
    )
    
    if not runs:
        raise ValueError("No finished runs found in MLflow")
    
    best_run = runs[0]
    
    import joblib
    import tempfile
    import os
    
    artifact_path = client.download_artifacts(best_run.info.run_id, "fraud_model.pkl", dst_dir=tempfile.mkdtemp())
    model_data = joblib.load(artifact_path)
    
    threshold = best_run.data.metrics.get("threshold", 0.5)
    
    print(f"Model loaded from MLflow run: {best_run.info.run_id}, threshold: {threshold}")
    return {
        "model": model_data["model"],
        "threshold": threshold,
        "features": get_feature_columns(),
        "reference_stats": None
    }


def load_model_local():
    model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model_data = joblib.load(model_path)
    print(f"Model loaded from local: {model_path}")
    return model_data


def get_model():
    try:
        return load_model_from_mlflow()
    except Exception as e:
        print(f"Could not load from MLflow: {e}")
        print("Falling back to local model...")
        return load_model_local()