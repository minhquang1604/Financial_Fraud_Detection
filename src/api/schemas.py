from pydantic import BaseModel, Field
from typing import Optional


class TransactionFeatures(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(gt=0)
    Time: float
    hour_of_day: int = Field(ge=0, le=23)
    is_night_transaction: int = Field(ge=0, le=1)
    amt_to_mean_ratio: float
    is_high_amount: int = Field(ge=0, le=1)
    log_amount: float


class PredictionRequest(BaseModel):
    features: TransactionFeatures


class PredictionResponse(BaseModel):
    transaction_time: float
    fraud_probability: float
    prediction: int
    message: str