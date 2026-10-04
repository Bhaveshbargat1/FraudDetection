"""
Digital Payment Fraud Detection System — FastAPI Application
=============================================================
Main entry point serving the REST API and interactive dashboard.
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from model import train, predict_single, get_model_info, encode_hour, FEATURE_NAMES
from data_generator import generate_training_data, generate_sample_transaction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# In-memory transaction history (capped)
# ---------------------------------------------------------------------------
_transaction_history: list[dict] = []
MAX_HISTORY = 500


# ---------------------------------------------------------------------------
# Lifespan — train model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Generate synthetic data and train the model at startup."""
    logger.info("=== Fraud Detection System starting up ===")
    training_df = generate_training_data(n_samples=10_000, fraud_ratio=0.05)
    train(training_df, contamination=0.05)
    logger.info("=== Model ready — serving requests ===")
    yield
    logger.info("=== Shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Payment Fraud Detection API",
    description="Anomaly-detection-based fraud scoring for digital payments.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PaymentRequest(BaseModel):
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour_of_day: int = Field(0, ge=0, le=23, description="Hour of day (0-23)")
    velocity_1h: int = Field(0, ge=0, description="Transactions by this card in last 1 hour")
    velocity_24h: int = Field(0, ge=0, description="Transactions by this card in last 24 hours")
    location_risk: float = Field(0.0, ge=0.0, le=1.0, description="Merchant location risk score (0-1)")
    amount_to_avg_ratio: Optional[float] = Field(None, ge=0, description="Amount / user's average amount")
    weekend: int = Field(0, ge=0, le=1, description="1 if weekend, 0 if weekday")
    merchant_name: Optional[str] = Field(None, description="Merchant name (for display)")
    cardholder_id: Optional[str] = Field(None, description="Cardholder identifier")


class PredictionResponse(BaseModel):
    risk_score: float
    label: str
    is_fraud: bool
    anomaly_score: float
    warnings: list[str]
    contributions: list[dict]
    timestamp: str
    transaction: dict


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_payment(payment: PaymentRequest):
    """Score a payment transaction for fraud risk."""

    # Cyclical encoding
    hour_sin, hour_cos = encode_hour(payment.hour_of_day)

    # Default amount_to_avg_ratio if not provided
    avg_ratio = payment.amount_to_avg_ratio if payment.amount_to_avg_ratio is not None else 1.0

    features = {
        "transaction_amount": payment.transaction_amount,
        "hour_of_day_sin": hour_sin,
        "hour_of_day_cos": hour_cos,
        "velocity_1h": payment.velocity_1h,
        "velocity_24h": payment.velocity_24h,
        "location_risk": payment.location_risk,
        "amount_to_avg_ratio": avg_ratio,
        "weekend": payment.weekend,
    }

    result = predict_single(features)

    transaction_summary = {
        "amount": payment.transaction_amount,
        "hour": payment.hour_of_day,
        "merchant": payment.merchant_name or "Unknown",
        "cardholder": payment.cardholder_id or "Anonymous",
        "velocity_1h": payment.velocity_1h,
        "location_risk": payment.location_risk,
    }

    response = PredictionResponse(
        risk_score=result["risk_score"],
        label=result["label"],
        is_fraud=result["is_fraud"],
        anomaly_score=result["anomaly_score"],
        warnings=result["warnings"],
        contributions=result["contributions"],
        timestamp=datetime.utcnow().isoformat() + "Z",
        transaction=transaction_summary,
    )

    # Store in history
    _transaction_history.insert(0, {
        **response.model_dump(),
        "raw_features": features,
    })
    if len(_transaction_history) > MAX_HISTORY:
        _transaction_history.pop()

    return response


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history(limit: int = 50):
    """Return recent transaction history."""
    return _transaction_history[:limit]


# ---------------------------------------------------------------------------
# GET /api/sample
# ---------------------------------------------------------------------------
@app.get("/api/sample")
async def get_sample(fraud: int = 0):
    """Generate a random sample transaction for testing."""
    return generate_sample_transaction(fraudulent=bool(fraud))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_info = get_model_info()
    return {
        "status": "healthy",
        "model": model_info,
        "transactions_scored": len(_transaction_history),
        "uptime": "running",
    }


# ---------------------------------------------------------------------------
# GET / — Dashboard UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the interactive fraud-detection dashboard."""
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        log_level="info",
    )
