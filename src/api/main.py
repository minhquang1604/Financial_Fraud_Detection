import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.train.utils import engineer_features, get_feature_columns
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.model_loader import get_model


model = None
model_data = None
threshold = 0.5
reference_stats = None

LIVE_DATA_DIR = os.path.join(os.path.dirname(\
    os.path.dirname(\
    os.path.dirname(\
    os.path.abspath(__file__)))), "data", "live")

os.makedirs(LIVE_DATA_DIR, exist_ok=True)


class RunningStats:
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.amounts = []
        self.mean_amt = 0.0
        self.median_amt = 0.0
        self.threshold_95 = 0.0
        self.alpha = 0.01

    def update(self, amount: float):
        self.amounts.append(amount)
        if len(self.amounts) > self.window_size:
            self.amounts.pop(0)

        if len(self.amounts) >= 100:
            self.mean_amt = np.mean(self.amounts)
            self.median_amt = np.median(self.amounts)
            self.threshold_95 = np.percentile(self.amounts, 95)

    def get_stats(self) -> dict:
        return {
            "mean_amt": self.mean_amt if self.mean_amt > 0 else 1.0,
            "median_amt": self.median_amt if self.median_amt > 0 else 1.0,
            "threshold_95": self.threshold_95 if self.threshold_95 > 0 else 1000.0
        }


running_stats = RunningStats()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_data, threshold, reference_stats
    try:
        model_data = get_model()
        model = model_data["model"]
        threshold = model_data.get("threshold", 0.5)
        reference_stats = model_data.get("reference_stats")
        print(f"Model loaded. Threshold: {threshold:.4f}")
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        model = None
        model_data = None
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time Credit Card Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        raw_features = request.features.model_dump()
        df = pd.DataFrame([raw_features])

        amount = raw_features.get("Amount", 0)
        running_stats.update(amount)

        stats = running_stats.get_stats()
        fake_ref_df = pd.DataFrame({
            "Amount": [stats["mean_amt"], stats["median_amt"], stats["threshold_95"]]
        })
        
        df = engineer_features(df, reference_df=fake_ref_df)
        
        feature_cols = get_feature_columns()
        X = df[feature_cols]
        
        prob = model.predict_proba(X)[0, 1]
        pred = int(prob >= threshold)
        
        # Luu vao live_data (B1: Luu predictions)
        live_record = raw_features.copy()
        live_record["prediction"] = pred
        live_record["fraud_probability"] = float(prob)
        live_record["prediction_time"] = pd.Timestamp.now().isoformat()
        
        live_file = os.path.join(LIVE_DATA_DIR, "live_predictions.parquet")
        live_df = pd.DataFrame([live_record])
        
        if os.path.exists(live_file):
            existing_df = pd.read_parquet(live_file)
            live_df = pd.concat([existing_df, live_df], ignore_index=True)
        
        live_df.to_parquet(live_file, index=False)
        
        return PredictionResponse(
            transaction_time=raw_features['Time'],
            fraud_probability=float(prob),
            prediction=pred,
            message="Fraud detected" if pred == 1 else "Normal transaction"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)