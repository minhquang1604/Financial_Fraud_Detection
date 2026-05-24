import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.train.utils import engineer_features, get_feature_columns
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.model_loader import get_model, predict_proba_safe


PREDICTIONS_TOTAL = Counter(
    'fraud_predictions_total', 'Total predictions by class',
    ['prediction']
)
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds', 'Inference latency in seconds',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
FRAUD_PROBABILITY = Gauge(
    'fraud_prediction_probability', 'Current fraud prediction probability'
)


model = None
model_data = None
threshold = 0.5
reference_stats = None


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


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
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
        
        prob = predict_proba_safe(model_data, X)[0]
        pred = int(prob >= threshold)
        
        PREDICTIONS_TOTAL.labels(prediction=str(pred)).inc()
        FRAUD_PROBABILITY.set(float(prob))
        
        return PredictionResponse(
            transaction_time=raw_features['Time'],
            fraud_probability=float(prob),
            prediction=pred,
            message="🚨🚨 FRAUD DETECTED 🚨🚨" if pred == 1 else "✅ Normal transaction"
        )
    except Exception as e:
        import traceback
        print(f"Predict error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        INFERENCE_LATENCY.observe(time.time() - start)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)