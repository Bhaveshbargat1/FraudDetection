"""
Fraud Detection ML Model — Isolation Forest Anomaly Detection
=============================================================
Provides training, inference, and feature contribution extraction
for real-time digital payment fraud scoring.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names used by the model (order matters for inference)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "transaction_amount",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "velocity_1h",          # transactions in last 1 hour (simulated)
    "velocity_24h",         # transactions in last 24 hours (simulated)
    "location_risk",        # 0-1 risk score for the merchant location
    "amount_to_avg_ratio",  # current amount / user's rolling average
    "weekend",              # 1 if Saturday/Sunday
]


# ---------------------------------------------------------------------------
# Global model artefacts (singleton-style)
# ---------------------------------------------------------------------------
_model: IsolationForest | None = None
_scaler: StandardScaler | None = None
_normalisation_stats: dict | None = None  # for synthetic training data
_score_min: float = 0.0  # training data min anomaly score
_score_max: float = 0.0  # training data max anomaly score


def _build_scaler_and_stats(X: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler and store column statistics."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def train(training_df: pd.DataFrame, contamination: float = 0.05) -> None:
    """
    Train the Isolation Forest on the synthetic payment dataset.

    Parameters
    ----------
    training_df : pd.DataFrame
        Must contain all columns listed in FEATURE_NAMES.
    contamination : float
        Expected proportion of anomalies in the training data.
    """
    global _model, _scaler, _score_min, _score_max

    logger.info("Training Isolation Forest on %d samples …", len(training_df))

    X = training_df[FEATURE_NAMES].values.astype(np.float64)

    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )
    _model.fit(X_scaled)

    # Compute score range on training data for normalisation
    train_scores = _model.decision_function(X_scaled)
    _score_min = float(train_scores.min())
    _score_max = float(train_scores.max())

    # Store column means / stds for reference
    global _normalisation_stats
    _normalisation_stats = {
        "mean": _scaler.mean_.tolist(),
        "std": _scaler.scale_.tolist(),
    }

    logger.info(
        "Model training complete. Score range: [%.4f, %.4f]. Contamination: %.2f",
        _score_min, _score_max, contamination,
    )


def predict_single(features: dict) -> dict:
    """
    Score a single transaction and return a rich result dict.

    Parameters
    ----------
    features : dict
        Keys must match FEATURE_NAMES (cyclical encoding already applied).

    Returns
    -------
    dict  with keys:
        anomaly_score   – raw continuous score (lower = more anomalous)
        risk_score      – 0-100 normalised risk score
        is_fraud        – bool (True if predicted anomaly)
        label           – "Fraud" | "Legit"
        warnings        – list[str] human-readable warnings
        contributions   – list[dict] top feature contributions to anomaly
    """
    if _model is None or _scaler is None:
        raise RuntimeError("Model has not been trained yet. Call train() first.")

    raw = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]], dtype=np.float64)
    scaled = _scaler.transform(raw)

    # anomaly_score: lower is more anomalous (sklearn convention)
    anomaly_score = float(_model.decision_function(scaled)[0])
    prediction = int(_model.predict(scaled)[0])  # -1 = anomaly, 1 = normal
    is_fraud = prediction == -1

    # ------------------------------------------------------------------
    # Normalise to 0-100 risk score using training-data score range
    # Maps: score_min (most anomalous) → 100,  score_max (most normal) → 0
    # ------------------------------------------------------------------
    score_range = _score_max - _score_min
    if score_range > 1e-8:
        risk_score = float(np.clip(
            (_score_max - anomaly_score) / score_range * 100, 0, 100
        ))
    else:
        risk_score = 50.0  # fallback if all scores are identical

    # ------------------------------------------------------------------
    # Feature contributions via per-feature isolation (approximation)
    # We use a single-tree decomposition approach
    # ------------------------------------------------------------------
    contributions = _compute_feature_contributions(raw, scaled)

    # ------------------------------------------------------------------
    # Build warning flags
    # ------------------------------------------------------------------
    warnings: list[str] = []
    if features.get("transaction_amount", 0) > 5000:
        warnings.append("High transaction amount (>$5,000)")
    if features.get("velocity_1h", 0) > 5:
        warnings.append("High transaction velocity (>5 txns in 1 hour)")
    if features.get("location_risk", 0) > 0.7:
        warnings.append("High-risk merchant location")
    hour_val = features.get("hour_of_day_sin", 0)
    # Night hours roughly correspond to sin/cos in certain ranges
    hour_raw = _reverse_hour_cyclical(
        features.get("hour_of_day_sin", 0),
        features.get("hour_of_day_cos", 0),
    )
    if hour_raw is not None and (hour_raw < 5 or hour_raw > 23):
        warnings.append("Unusual transaction hour (late night)")
    if features.get("amount_to_avg_ratio", 1) > 3:
        warnings.append("Amount is >3× user's average")
    if features.get("weekend", 0) == 1 and features.get("transaction_amount", 0) > 2000:
        warnings.append("Large weekend transaction")
    if is_fraud and not warnings:
        warnings.append("Anomalous pattern detected by model")

    return {
        "anomaly_score": round(anomaly_score, 6),
        "risk_score": round(risk_score, 2),
        "is_fraud": is_fraud,
        "label": "Fraud" if is_fraud else "Legit",
        "warnings": warnings,
        "contributions": contributions,
    }


def _compute_feature_contributions(raw: np.ndarray, scaled: np.ndarray) -> list[dict]:
    """
    Approximate per-feature contribution to the anomaly score by
    measuring how much each feature deviates from the training mean
    in standard-deviation units (signed z-score of each feature).
    """
    if _scaler is None:
        return []

    z_scores = (raw[0] - _scaler.mean_) / (_scaler.scale_ + 1e-8)
    contribs = []
    for i, fname in enumerate(FEATURE_NAMES):
        contribs.append({
            "feature": _pretty_name(fname),
            "raw_value": round(float(raw[0][i]), 4),
            "z_score": round(float(z_scores[i]), 4),
            "abs_deviation": round(abs(float(z_scores[i])), 4),
        })

    # Sort by absolute deviation descending, return top features
    contribs.sort(key=lambda c: c["abs_deviation"], reverse=True)
    return contribs


def _reverse_hour_cyclical(sin_val: float, cos_val: float):
    """Attempt to recover hour from cyclical encoding (approximate)."""
    try:
        angle = np.arctan2(sin_val, cos_val)
        hour = (angle / (2 * np.pi)) * 24
        if hour < 0:
            hour += 24
        return int(round(hour))
    except Exception:
        return None


def _pretty_name(feature: str) -> str:
    mapping = {
        "transaction_amount": "Transaction Amount",
        "hour_of_day_sin": "Hour of Day (cyclical)",
        "hour_of_day_cos": "Hour of Day (cyclical)",
        "velocity_1h": "Velocity (1h)",
        "velocity_24h": "Velocity (24h)",
        "location_risk": "Location Risk",
        "amount_to_avg_ratio": "Amount / Avg Ratio",
        "weekend": "Weekend Flag",
    }
    return mapping.get(feature, feature)


# ---------------------------------------------------------------------------
# Helpers used by data_generator and main
# ---------------------------------------------------------------------------

def encode_hour(hour: int) -> Tuple[float, float]:
    """Cyclical encoding of hour (0-23) → sin / cos pair."""
    angle = 2 * np.pi * hour / 24
    return round(np.sin(angle), 6), round(np.cos(angle), 6)


def get_model_info() -> dict:
    """Return model metadata for the health endpoint."""
    if _model is None:
        return {"status": "not_trained"}
    return {
        "status": "ready",
        "n_estimators": _model.n_estimators,
        "contamination": _model.contamination,
        "features": FEATURE_NAMES,
    }
