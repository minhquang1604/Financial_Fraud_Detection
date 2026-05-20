import os
import sys
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import mlflow
import mlflow.sklearn
from src.train.utils import get_feature_columns


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://13.215.163.1:5000")
MODEL_NAME = "FraudDetectionModel"


def load_model_from_mlflow(stage: str = None, version: int = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = mlflow.tracking.MlflowClient()
    
    try:
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        if not all_versions:
            raise ValueError(f"No versions found for model: {MODEL_NAME}")
        
        if version:
            target_version = next((v for v in all_versions if v.version == version), None)
        elif stage:
            target_version = next((v for v in all_versions if v.current_stage == stage), None)
        else:
            target_version = next((v for v in all_versions if v.current_stage == "Production"), None)
            if not target_version:
                target_version = sorted(all_versions, key=lambda x: x.version, reverse=True)[0]
        
        if not target_version:
            raise ValueError(f"No version found for stage={stage}, version={version}")
        
        run_id = target_version.run_id
        run = client.get_run(run_id)
        
        threshold = run.data.metrics.get("threshold") or run.data.params.get("threshold", 0.5)
        threshold = float(threshold)
        
        model_uri = f"models:/{MODEL_NAME}/{target_version.version}"
        
        try:
            model = mlflow.sklearn.load_model(model_uri=model_uri)
            print(f"Model loaded from MLflow Registry: {MODEL_NAME}/{target_version.version}")
            return {
                "model": model,
                "threshold": float(threshold),
                "features": get_feature_columns(),
                "reference_stats": None
            }
        except Exception as load_err:
            if "MLmodel" in str(load_err) or "No such artifact" in str(load_err):
                print(f"Registry artifact path incorrect, using direct artifact URI")
                artifact_uri = run.info.artifact_uri.rstrip('/')
                model = mlflow.sklearn.load_model(f"{artifact_uri}/model/model")
                print(f"Model loaded from artifact: {MODEL_NAME}/{target_version.version}")
                return {
                    "model": model,
                    "threshold": float(threshold),
                    "features": get_feature_columns(),
                    "reference_stats": None
                }
            raise
    
    except Exception as e:
        raise ValueError(f"Failed to load from MLflow: {e}")


def get_model():
    return load_model_from_mlflow()


def get_model_info(stage: str = None, version: int = None) -> dict:
    """
    Get model info including threshold and metrics from MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = mlflow.tracking.MlflowClient()
    
    try:
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        if not all_versions:
            raise ValueError(f"No versions found for model: {MODEL_NAME}")
        
        if version:
            target_version = next((v for v in all_versions if v.version == version), None)
        elif stage:
            target_version = next((v for v in all_versions if v.current_stage == stage), None)
        else:
            target_version = next((v for v in all_versions if v.current_stage == "Production"), None)
            if not target_version:
                target_version = sorted(all_versions, key=lambda x: x.version, reverse=True)[0]
        
        if not target_version:
            raise ValueError(f"No version found for stage={stage}, version={version}")
        
        run_id = target_version.run_id
        run = client.get_run(run_id)
        
        metrics = run.data.metrics
        params = run.data.params
        
        threshold = metrics.get("threshold") or params.get("threshold", 0.5)
        threshold = float(threshold)
        f1_score = metrics.get("F1", 0.0)
        auprc = metrics.get("AUPRC", 0.0)
        
        return {
            "version": target_version.version,
            "stage": target_version.current_stage,
            "run_id": run_id,
            "threshold": float(threshold),
            "metrics": {
                "F1": float(f1_score),
                "AUPRC": float(auprc)
            },
            "created_at": run.info.end_time
        }
    
    except Exception as e:
        raise ValueError(f"Failed to get model info from MLflow: {e}")


def predict_with_booster(model_data, X):
    import pandas as pd
    import xgboost as xgb
    import numpy as np
    
    model = model_data["model"]
    features = model_data.get("features", None)
    booster = model.get_booster()
    
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        if features is None and hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_.tolist()
    else:
        X_arr = X
    
    if features is not None:
        dtest = xgb.DMatrix(X_arr, feature_names=features)
    else:
        dtest = xgb.DMatrix(X_arr)
    
    raw_pred = booster.predict(dtest, output_margin=False)
    raw_pred = np.asarray(raw_pred).flatten()
    
    prob = np.clip(raw_pred, 0, 1)
    
    if len(prob) == 1:
        return np.array([prob[0]])
    return prob


def predict_proba_safe(model_data, X):
    import pandas as pd
    import numpy as np
    
    model = model_data["model"]
    
    try:
        if isinstance(X, pd.DataFrame):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"predict_proba error: {e}")
        proba = model.predict(X)
        if len(proba.shape) > 1 and proba.shape[1] > 1:
            proba = proba[:, 1]
        else:
            proba = proba
    
    proba = np.asarray(proba).flatten()
    
    proba = np.clip(proba, 0.0001, 0.9999)
    
    print(f"Raw proba: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}")
    
    return proba