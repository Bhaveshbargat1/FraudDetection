"""
Synthetic Payment Data Generator
=================================
Creates realistic synthetic payment transaction data for training
the Isolation Forest fraud-detection model at app startup.
"""

import numpy as np
import pandas as pd
from model import FEATURE_NAMES, encode_hour
import logging

logger = logging.getLogger(__name__)


def generate_training_data(n_samples: int = 10_000, fraud_ratio: float = 0.05) -> pd.DataFrame:
    """
    Generate a synthetic payment dataset with realistic fraud patterns.

    Parameters
    ----------
    n_samples : int
        Total number of transactions to generate.
    fraud_ratio : float
        Proportion of fraudulent transactions.

    Returns
    -------
    pd.DataFrame with all FEATURE_NAMES columns.
    """
    rng = np.random.default_rng(seed=42)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # ---------- LEGITIMATE transactions ----------
    legit_amounts = np.abs(rng.lognormal(mean=4.5, sigma=1.0, size=n_legit))
    legit_amounts = np.clip(legit_amounts, 1, 10_000)

    # Legit transactions tend to happen during business hours (9-21)
    legit_hours = rng.normal(loc=14, scale=4, size=n_legit).clip(0, 23).astype(int)

    legit_velocity_1h = rng.poisson(lam=1.5, size=n_legit)
    legit_velocity_24h = rng.poisson(lam=6, size=n_legit)

    legit_location_risk = rng.beta(a=2, b=5, size=n_legit)  # low risk bias

    legit_avg_amounts = legit_amounts * rng.uniform(0.5, 2.0, size=n_legit)
    legit_avg_ratio = legit_amounts / (legit_avg_amounts + 1e-6)
    legit_avg_ratio = np.clip(legit_avg_ratio, 0.1, 3.0)

    legit_weekend = rng.choice([0, 1], size=n_legit, p=[5/7, 2/7])

    # ---------- FRAUDULENT transactions ----------
    # Fraud: higher amounts, unusual hours, higher velocity, riskier locations
    fraud_amounts = np.abs(rng.lognormal(mean=7.5, sigma=1.5, size=n_fraud))
    fraud_amounts = np.clip(fraud_amounts, 500, 50_000)

    # Fraud tends to happen at unusual hours (0-5 AM)
    fraud_hours = rng.choice(
        np.arange(24), size=n_fraud,
        p=_make_night_heavy_distribution()
    )

    fraud_velocity_1h = rng.poisson(lam=8, size=n_fraud)
    fraud_velocity_24h = rng.poisson(lam=20, size=n_fraud)

    fraud_location_risk = rng.beta(a=5, b=2, size=n_fraud)  # high risk bias

    fraud_avg_amounts = rng.uniform(200, 800, size=n_fraud)  # user's usual avg is low
    fraud_avg_ratio = fraud_amounts / (fraud_avg_amounts + 1e-6)
    fraud_avg_ratio = np.clip(fraud_avg_ratio, 2.0, 30.0)

    fraud_weekend = rng.choice([0, 1], size=n_fraud, p=[0.4, 0.6])

    # ---------- Combine ----------
    amounts = np.concatenate([legit_amounts, fraud_amounts])
    hours = np.concatenate([legit_hours, fraud_hours])
    velocity_1h = np.concatenate([legit_velocity_1h, fraud_velocity_1h])
    velocity_24h = np.concatenate([legit_velocity_24h, fraud_velocity_24h])
    location_risk = np.concatenate([legit_location_risk, fraud_location_risk])
    amount_ratio = np.concatenate([legit_avg_ratio, fraud_avg_ratio])
    weekends = np.concatenate([legit_weekend, fraud_weekend])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Cyclical encoding of hour
    hour_sin, hour_cos = np.vectorize(encode_hour)(hours)

    df = pd.DataFrame({
        "transaction_amount": amounts,
        "hour_of_day_sin": hour_sin,
        "hour_of_day_cos": hour_cos,
        "velocity_1h": velocity_1h,
        "velocity_24h": velocity_24h,
        "location_risk": location_risk,
        "amount_to_avg_ratio": amount_ratio,
        "weekend": weekends,
        "is_fraud": labels.astype(int),
    })

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "Generated %d synthetic transactions (%d fraud, %.1f%% fraud rate)",
        len(df), n_fraud, fraud_ratio * 100,
    )
    return df


def _make_night_heavy_distribution() -> np.ndarray:
    """Create a 24-element probability array that peaks around 1-4 AM."""
    probs = np.zeros(24)
    for h in range(24):
        # Base uniform probability
        probs[h] = 0.5
        # Boost for late night / early morning hours (0-5)
        if 0 <= h <= 5:
            probs[h] = 3.0
        # Small boost for 22-23
        if 22 <= h <= 23:
            probs[h] = 1.5
    return probs / probs.sum()


def generate_sample_transaction(fraudulent: bool = False) -> dict:
    """
    Generate a single sample transaction for testing / quick-fill presets.

    Parameters
    ----------
    fraudulent : bool
        If True, generate a transaction with fraud-like characteristics.

    Returns
    -------
    dict with keys matching the API input schema.
    """
    rng = np.random.default_rng()

    if fraudulent:
        amount = float(rng.lognormal(mean=7.5, sigma=1.2))
        amount = min(amount, 50_000)
        hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
        velocity_1h = int(rng.poisson(8))
        velocity_24h = int(rng.poisson(20))
        location_risk = float(rng.beta(5, 2))
        avg_amount = float(rng.uniform(100, 500))
        weekend = int(rng.choice([0, 1], p=[0.4, 0.6]))
        merchant = str(rng.choice([
            "Offshore Electronics Ltd", "Crypto Exchange XYZ",
            "Unknown Merchant #4821", "High-Risk Gaming Site",
        ]))
    else:
        amount = float(rng.lognormal(mean=4.5, sigma=0.8))
        amount = min(amount, 5_000)
        hour = int(rng.normal(loc=14, scale=3))
        hour = max(0, min(23, hour))
        velocity_1h = int(rng.poisson(1.5))
        velocity_24h = int(rng.poisson(5))
        location_risk = float(rng.beta(2, 5))
        avg_amount = float(rng.uniform(amount * 0.5, amount * 1.5))
        weekend = int(rng.choice([0, 1], p=[5/7, 2/7]))
        merchant = str(rng.choice([
            "Amazon.com", "Starbucks #1247", "Uber Trip",
            "Whole Foods Market", "Netflix Subscription",
            "Local Grocery Store", "Shell Gas Station",
        ]))

    hour_sin, hour_cos = encode_hour(hour)
    avg_ratio = amount / (avg_amount + 1e-6)

    return {
        "transaction_amount": round(amount, 2),
        "hour_of_day": hour,
        "hour_of_day_sin": hour_sin,
        "hour_of_day_cos": hour_cos,
        "velocity_1h": velocity_1h,
        "velocity_24h": velocity_24h,
        "location_risk": round(location_risk, 4),
        "amount_to_avg_ratio": round(avg_ratio, 4),
        "weekend": weekend,
        "merchant_name": merchant,
        "cardholder_id": f"CH-{rng.integers(10000, 99999)}",
    }
