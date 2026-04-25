import os
import mlflow
import mlflow.sklearn
import joblib


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "fraud_model_xgboost"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model")


def load_model_from_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_names=["FraudGuard_XGBoost_Baseline"],
        filter_string="status='FINISHED'",
        max_results=1,
        order_by=["metrics.AUPRC DESC"]
    )
    
    if not runs:
        raise ValueError("No finished runs found in MLflow")
    
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/fraud_model_xgboost"
    model = mlflow.sklearn.load_model(model_uri)
    
    print(f"Model loaded from run: {best_run.info.run_id}")
    return model


def load_model_local():
    model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from local: {model_path}")
    return model


def get_model():
    try:
        return load_model_from_mlflow()
    except Exception as e:
        print(f"Could not load from MLflow: {e}")
        print("Falling back to local model...")
        return load_model_local()