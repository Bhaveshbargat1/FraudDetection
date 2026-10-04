# 🛡️ Digital Payment Fraud Detection System

An end-to-end fraud detection platform using **Isolation Forest** anomaly detection, built with **FastAPI** and served with a **Tailwind CSS** interactive dashboard.

---

## Architecture

```
fraud-detection/
├── main.py              # FastAPI app, routes, lifespan
├── model.py             # Isolation Forest model, training & inference
├── data_generator.py    # Synthetic payment data generator
├── templates/
│   └── index.html       # Interactive dashboard (Tailwind CSS)
├── requirements.txt     # Python dependencies
├── Procfile             # Heroku / Render process declaration
├── render.yaml          # Render.com deployment configuration
└── README.md
```

## Features

### Machine Learning
- **Isolation Forest** (200 estimators) trained on 10,000 synthetic payment transactions
- Detects anomalies based on 8 engineered features:
  - Transaction amount, Hour of day (cyclical encoding), Transaction velocity (1h & 24h), Location risk, Amount-to-average ratio, Weekend flag
- Returns **risk score (0–100)**, binary **Fraud/Legit** label, and **per-feature contribution** analysis

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/predict` | Score a payment transaction → risk score, label, warnings |
| `GET` | `/api/sample` | Generate random test transaction (query `?fraud=1` for fraud) |
| `GET` | `/api/history` | Recent transaction scoring history |
| `GET` | `/health` | Health check with model status |
| `GET` | `/` | Interactive dashboard UI |

### Frontend Dashboard
- **Transaction input form** with all model features
- **Quick-fill presets** — generate Normal or Fraud sample transactions
- **Animated risk gauge** with colour-coded severity
- **Feature contribution bars** — see which features drove the anomaly score
- **Live transaction history** — fraud alerts highlighted in red

---

## Quick Start (Local)

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
# → http://localhost:8000

# Or via uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Deploy to Render

1. Push this directory to a GitHub repository.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New → Web Service**.
3. Connect your repo — Render auto-detects `render.yaml`.
4. Click **Deploy**. The service will:
   - `pip install -r requirements.txt`
   - Start with `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Monitor health via `/health`

## API Example

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 12500.00,
    "hour_of_day": 3,
    "velocity_1h": 8,
    "velocity_24h": 22,
    "location_risk": 0.85,
    "amount_to_avg_ratio": 15.0,
    "weekend": 1,
    "merchant_name": "Offshore Electronics Ltd",
    "cardholder_id": "CH-54321"
  }'
```

Response:
```json
{
  "risk_score": 94.72,
  "label": "Fraud",
  "is_fraud": true,
  "anomaly_score": -0.4736,
  "warnings": [
    "High transaction amount (>$5,000)",
    "High transaction velocity (>5 txns in 1 hour)",
    "High-risk merchant location",
    "Unusual transaction hour (late night)",
    "Amount is >3× user's average",
    "Large weekend transaction"
  ],
  "contributions": [...],
  "timestamp": "2025-07-24T10:30:00Z",
  "transaction": {...}
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn (Isolation Forest), Pandas, NumPy |
| Frontend | HTML5, Tailwind CSS (CDN), Vanilla JS |
| Templates | Jinja2 |
| Deploy | Render.com, Procfile, render.yaml |

---

## 👨‍💻 Author

**Bhavesh Bargat**  
GitHub: [@Bhaveshbargat1](https://github.com/Bhaveshbargat1)

---

*Built as a Principal Full-Stack ML Engineering reference implementation.*
