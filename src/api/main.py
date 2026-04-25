from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from schemas import PredictionRequest, PredictionResponse
from model_loader import get_model


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = get_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        model = None
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
        features = request.features.model_dump()
        
        import pandas as pd
        X = pd.DataFrame([features])
        
        prob = model.predict_proba(X)[0, 1]
        pred = int(model.predict(X)[0])
        
        return PredictionResponse(
            transaction_time=features['Time'],
            fraud_probability=float(prob),
            prediction=pred,
            message="Fraud detected" if pred == 1 else "Normal transaction"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)