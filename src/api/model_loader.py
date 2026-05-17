import os
import sys
import pickle
from dotenv import load_dotenv

load_dotenv()

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
    
    try:
        all_versions = client.search_model_versions("name='FraudDetectionModel'")
        
        if not all_versions:
            raise ValueError("No model versions found in MLflow")
        
        latest_version = sorted(all_versions, key=lambda x: x.version, reverse=True)[0]
        
        run_id = latest_version.run_id
        threshold = 0.5
        
        try:
            run = client.get_run(run_id)
            if "threshold" in run.data.params:
                threshold = float(run.data.params["threshold"])
        except Exception:
            pass
        
        # Try loading from registered model first
        try:
            model = mlflow.sklearn.load_model(
                model_uri=f"models:/FraudDetectionModel/{latest_version.version}"
            )
            print(f"Model loaded from MLflow: FraudDetectionModel v{latest_version.version}, threshold: {threshold}")
            return {
                "model": model,
                "threshold": threshold,
                "features": get_feature_columns(),
                "reference_stats": None
            }
        except Exception as load_err:
            # Fallback: load directly from run's artifact path
            print(f"Registered model load failed: {load_err}, trying run artifact...")
            model = mlflow.sklearn.load_model(
                model_uri=f"runs:/{run_id}/model/model"
            )
            print(f"Model loaded from run: {run_id}, threshold: {threshold}")
            return {
                "model": model,
                "threshold": threshold,
                "features": get_feature_columns(),
                "reference_stats": None
            }
        
    except Exception as e:
        raise ValueError(f"Failed to load from MLflow: {e}")


def load_model_local():
    model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    if hasattr(model, '__dict__'):
        for key in list(model.__dict__.keys()):
            if 'label_encoder' in key.lower() or key == 'use_label_encoder':
                del model.__dict__[key]
    
    if hasattr(model, 'set_params'):
        try:
            params = model.get_params()
            if 'use_label_encoder' in params:
                model.set_params(**{k: v for k, v in params.items() if k != 'use_label_encoder'})
        except:
            pass
    
    print(f"Model loaded from local: {model_path}")
    return model_data


def get_model():
    try:
        return load_model_from_mlflow()
    except Exception as e:
        print(f"Could not load from MLflow: {e}")
        print("Falling back to local model...")
        return load_model_local()


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
    
    raw_pred = booster.predict(dtest)
    raw_pred = np.asarray(raw_pred).flatten()
    prob = 1 / (1 + np.exp(-raw_pred))
    
    if len(prob) == 1:
        return np.array([prob[0]])
    return prob


def predict_proba_safe(model_data, X):
    import pandas as pd
    import numpy as np
    
    model = model_data["model"]
    
    try:
        if isinstance(X, pd.DataFrame):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict_proba(X)[:, 1]
    except (AttributeError, TypeError) as e:
        if 'use_label_encoder' in str(e) or 'gpu_id' in str(e) or 'predictor' in str(e):
            print(f"XGBoost version mismatch, using booster fallback")
            return predict_with_booster(model_data, X)
        raise