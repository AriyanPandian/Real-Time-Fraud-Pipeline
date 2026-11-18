import importlib
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features import compute_realtime_features, extract_training_features
from src.simulator import seed_database
from src.train import train_and_evaluate


def test_database_seeding(tmp_path):
    db_path = tmp_path / "transactions.db"
    stats = seed_database(n_rows=1000, seed=42, db_path=db_path)
    assert stats["seeded_rows"] == 1000
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(transactions)")}
        users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM transactions").fetchone()[0]
        fraud_ratio = conn.execute("SELECT AVG(is_fraud) FROM transactions").fetchone()[0]
    assert {"tx_id", "user_id", "timestamp", "amount", "is_fraud"} <= columns
    assert users <= 100
    assert 0.025 <= fraud_ratio <= 0.060


def test_sql_window_features(tmp_path):
    db_path = tmp_path / "transactions.db"
    seed_database(n_rows=1200, seed=42, db_path=db_path)
    frame = extract_training_features(db_path)
    assert not frame[
        ["amt_to_user_avg_ratio", "mins_since_prev_tx", "location_changed"]
    ].isna().any().any()
    assert (frame["amt_to_user_avg_ratio"] >= 0).all()
    assert (frame["mins_since_prev_tx"] >= 0).all()
    first_by_user = frame.groupby("user_id").head(1)
    assert ((first_by_user["mins_since_prev_tx"] == 1440.0)).all()
    assert ((first_by_user["location_changed"] == 0)).all()


def test_model_training_and_artifacts(tmp_path):
    db_path = tmp_path / "transactions.db"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    seed_database(n_rows=3000, seed=42, db_path=db_path)
    metadata = train_and_evaluate(db_path=db_path, model_dir=model_dir)
    assert (model_dir / "fraud_model.pkl").exists()
    assert (model_dir / "model_metadata.json").exists()
    assert metadata["roc_auc"] > 0.85
    assert metadata["pr_auc"] > 0.60


def test_realtime_feature_parity(tmp_path):
    db_path = tmp_path / "transactions.db"
    seed_database(n_rows=500, seed=42, db_path=db_path)
    tx = {
        "user_id": 1001,
        "amount": 4500.0,
        "merchant_category": "wire_transfer",
        "location": "NG-LOS",
        "device_type": "mobile",
        "timestamp": "2026-11-18 12:00:00",
    }
    features, reasons = compute_realtime_features(tx, db_path)
    expected = {
        "user_hist_avg_amt", "amt_to_user_avg_ratio", "tx_count_prev_5",
        "mins_since_prev_tx", "location_changed", "is_high_risk_merchant",
        "is_foreign_location", "rapid_burst_flag",
    }
    assert expected <= set(features)
    assert reasons
    assert any("above user average" in reason for reason in reasons)


def test_fastapi_endpoints(tmp_path, monkeypatch):
    db_path = tmp_path / "transactions.db"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    seed_database(n_rows=3000, seed=42, db_path=db_path)
    train_and_evaluate(db_path=db_path, model_dir=model_dir)

    api = importlib.import_module("src.api")
    monkeypatch.setattr(api, "DB_PATH", db_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_dir / "fraud_model.pkl")
    monkeypatch.setattr(api, "METADATA_PATH", model_dir / "model_metadata.json")
    api.ARTIFACT = None
    api.METADATA = {"optimal_decision_threshold": 0.5}

    with TestClient(api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        normal = client.post("/score", json={
            "user_id": 1001,
            "amount": 25,
            "merchant_category": "groceries",
            "location": "US-IN",
            "device_type": "mobile",
        })
        assert normal.status_code == 200
        normal_payload = normal.json()

        foreign = client.post("/score", json={
            "user_id": 1001,
            "amount": 4500,
            "merchant_category": "wire_transfer",
            "location": "NG-LOS",
            "device_type": "web",
            "mins_since_prev_tx": 5,
        })
        assert foreign.status_code == 200
        foreign_payload = foreign.json()

    assert normal_payload["risk_score"] < 0.50
    assert foreign_payload["risk_score"] > 0.50
