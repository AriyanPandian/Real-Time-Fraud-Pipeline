"""FastAPI real-time fraud scoring service."""

import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import FEATURE_COLUMNS, compute_realtime_features
from src.simulator import DB_PATH, generate_live_batch, insert_transactions, seed_database
from src.train import train_and_evaluate


DB_PATH = Path(os.getenv("FRAUD_DB_PATH", DB_PATH))
MODEL_PATH = Path(os.getenv("FRAUD_MODEL_PATH", "models/fraud_model.pkl"))
METADATA_PATH = Path(os.getenv("FRAUD_METADATA_PATH", "models/model_metadata.json"))


class Transaction(BaseModel):
    user_id: int = Field(ge=1001, le=1100)
    amount: float = Field(gt=0)
    merchant_category: str
    location: str
    device_type: str
    timestamp: Optional[str] = None
    mins_since_prev_tx: Optional[float] = None


ARTIFACT = None
METADATA = {"optimal_decision_threshold": 0.5}


def ensure_artifacts() -> None:
    global ARTIFACT, METADATA
    if not DB_PATH.exists():
        seed_database(db_path=DB_PATH)
    if not MODEL_PATH.exists():
        train_and_evaluate(DB_PATH, MODEL_PATH.parent)
    ARTIFACT = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        METADATA = json.loads(METADATA_PATH.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_artifacts()
    yield


app = FastAPI(title="Sentinel Fraud Scoring API", version="1.0", lifespan=lifespan)


def hybrid_score(features: dict) -> float:
    frame = pd.DataFrame([features])[FEATURE_COLUMNS]
    preprocessor = ARTIFACT["preprocessor"]
    transformed = preprocessor.transform(frame)
    supervised = float(ARTIFACT["classifier"].predict_proba(transformed)[0, 1])

    # Reproducible normalization based on the model training distribution is stored
    # implicitly in the isolation model; use its score scale directly for live traffic.
    raw = -float(ARTIFACT["isolation_forest"].score_samples(transformed)[0])
    anomaly = float(np.clip((raw + 0.05) / 0.20, 0.0, 1.0))
    risk = 0.80 * supervised + 0.20 * anomaly
    if features["amount"] >= 900 and features["is_foreign_location"] and features["is_high_risk_merchant"]:
        risk += 0.35
    if features["amt_to_user_avg_ratio"] >= 6.0:
        risk += 0.20
    if features["rapid_burst_flag"]:
        risk += 0.08
    return float(np.clip(risk, 0.0, 1.0))


@app.get("/health")
def health():
    ensure_artifacts()
    return {"status": "healthy", "model_loaded": ARTIFACT is not None}


@app.post("/score")
def score(transaction: Transaction):
    ensure_artifacts()
    tx = transaction.model_dump()
    tx["timestamp"] = tx.get("timestamp") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    tx.setdefault("tx_id", str(uuid.uuid4()))
    started = time.perf_counter()
    features, reasons = compute_realtime_features(tx, DB_PATH)
    risk_score = hybrid_score(features)
    threshold = float(METADATA.get("optimal_decision_threshold", 0.5))
    is_flagged = risk_score >= threshold
    latency_ms = (time.perf_counter() - started) * 1000.0

    row = {
        **tx,
        "predicted_risk_score": risk_score,
        "is_flagged": int(is_flagged),
        "risk_reasons": ", ".join(reasons) if reasons else "No high-risk indicators",
    }
    insert_transactions([row], DB_PATH)
    return {
        "tx_id": row["tx_id"],
        "risk_score": risk_score,
        "is_flagged": bool(is_flagged),
        "latency_ms": round(latency_ms, 3),
        "engineered_features": features,
        "risk_reasons": reasons or ["No high-risk indicators"],
    }


@app.post("/simulate")
def simulate(n: int = 15, fraud_ratio: float = 0.20):
    ensure_artifacts()
    n = max(1, min(n, 500))
    fraud_ratio = min(max(fraud_ratio, 0.0), 1.0)
    threshold = float(METADATA.get("optimal_decision_threshold", 0.5))
    scored = []
    for tx in generate_live_batch(n=n, fraud_ratio=fraud_ratio):
        features, reasons = compute_realtime_features(tx, DB_PATH)
        risk_score = hybrid_score(features)
        is_flagged = risk_score >= threshold
        tx.update({
            "predicted_risk_score": risk_score,
            "is_flagged": int(is_flagged),
            "risk_reasons": ", ".join(reasons) if reasons else "No high-risk indicators",
        })
        scored.append(tx)
    insert_transactions(scored, DB_PATH)
    flagged = sum(int(tx["is_flagged"]) for tx in scored)
    volume = sum(float(tx["amount"]) for tx in scored if tx["is_flagged"])
    return {
        "count": len(scored),
        "flagged_count": flagged,
        "average_risk_score": round(float(np.mean([tx["predicted_risk_score"] for tx in scored])), 4),
        "blocked_volume_usd": round(volume, 2),
    }


@app.get("/metrics")
def metrics():
    ensure_artifacts()
    threshold = float(METADATA.get("optimal_decision_threshold", 0.5))
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        flagged = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE is_flagged = 1"
        ).fetchone()[0]
        fraud_count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE is_fraud = 1"
        ).fetchone()[0]
        blocked_volume = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE is_flagged = 1"
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT tx_id, user_id, timestamp, amount, merchant_category, location,
                   predicted_risk_score, risk_reasons
            FROM transactions
            WHERE is_flagged = 1
            ORDER BY timestamp DESC
            LIMIT 50
            """
        ).fetchall()
    alerts = [
        {
            "tx_id": row[0],
            "user_id": row[1],
            "timestamp": row[2],
            "amount": row[3],
            "merchant_category": row[4],
            "location": row[5],
            "risk_score": row[6],
            "risk_reasons": row[7],
        }
        for row in rows
    ]
    return {
        "total_tx": int(total),
        "flagged_count": int(flagged),
        "fraud_rate": round(fraud_count / max(total, 1), 4),
        "blocked_volume_usd": round(float(blocked_volume), 2),
        "holdout_model_metrics": METADATA,
        "recent_flagged_alerts": alerts,
    }
