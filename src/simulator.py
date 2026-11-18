"""Synthetic transaction generation and SQLite persistence."""

import argparse
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


DB_PATH = Path(os.getenv("FRAUD_DB_PATH", "data/transactions.db"))
CATEGORIES = [
    "groceries", "restaurants", "travel", "electronics", "digital_goods",
    "wire_transfer", "luxury_retail",
]
LOCATIONS = ["US-IN", "US-PA", "US-NY", "US-CA", "GB-LON", "NG-LOS", "SG-SIN"]
DEVICES = ["mobile", "web", "pos_terminal"]


def create_database(db_path=DB_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant_category TEXT NOT NULL,
                location TEXT NOT NULL,
                device_type TEXT NOT NULL,
                is_fraud INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
                predicted_risk_score REAL,
                is_flagged INTEGER CHECK (is_flagged IN (0, 1)),
                risk_reasons TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_timestamp ON transactions(user_id, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON transactions(timestamp)"
        )
        conn.commit()


def _baseline_profiles(rng: np.random.Generator) -> Dict[int, Dict[str, float | str]]:
    profiles: Dict[int, Dict[str, float | str]] = {}
    for user_id in range(1001, 1101):
        profiles[user_id] = {
            "mean": float(rng.uniform(25, 180)),
            "home": "US-IN" if user_id % 2 else "US-PA",
        }
    return profiles


def seed_database(n_rows: int = 10000, seed: int = 42, db_path=DB_PATH) -> Dict[str, int]:
    """Seed a chronologically ordered synthetic transaction dataset."""
    rng = np.random.default_rng(seed)
    profiles = _baseline_profiles(rng)
    create_database(db_path)
    end = datetime.now().replace(microsecond=0)
    start = end - timedelta(days=30)
    seconds = int((end - start).total_seconds())
    records: List[Dict[str, object]] = []

    for _ in range(n_rows):
        user_id = int(rng.integers(1001, 1101))
        profile = profiles[user_id]
        mean = float(profile["mean"])
        timestamp = start + timedelta(seconds=int(rng.integers(0, seconds + 1)))
        amount = max(1.0, float(rng.lognormal(np.log(mean), 0.55)))
        location = str(profile["home"]) if rng.random() < 0.82 else str(rng.choice(LOCATIONS))
        merchant = str(rng.choice(CATEGORIES, p=[0.30, 0.18, 0.10, 0.12, 0.13, 0.08, 0.09]))
        device = str(rng.choice(DEVICES, p=[0.48, 0.37, 0.15]))
        records.append({
            "tx_id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(amount, 2),
            "merchant_category": merchant,
            "location": location,
            "device_type": device,
            "is_fraud": 0,
            "predicted_risk_score": None,
            "is_flagged": None,
            "risk_reasons": None,
        })

    records.sort(key=lambda row: str(row["timestamp"]))

    # Inject patterns across ~3.5% of the records.
    target_fraud = max(1, round(n_rows * 0.035))
    used = set()
    pattern_counts = {"velocity": 0, "travel": 0, "takeover": 0}

    # Velocity bursts
    for _ in range(target_fraud // 3):
        choices = [i for i in range(len(records) - 3) if i not in used]
        if not choices:
            break
        i = int(rng.choice(choices))
        base = datetime.strptime(str(records[i]["timestamp"]), "%Y-%m-%d %H:%M:%S")
        user_id = int(records[i]["user_id"])
        mean = float(profiles[user_id]["mean"])
        for offset in range(4):
            j = i + offset
            if j >= len(records):
                break
            records[j]["user_id"] = user_id
            records[j]["timestamp"] = (base + timedelta(minutes=2 + offset * 3)).strftime("%Y-%m-%d %H:%M:%S")
            records[j]["amount"] = round(rng.uniform(mean * 0.8, mean * 2.2), 2)
            records[j]["merchant_category"] = "digital_goods" if offset % 2 == 0 else "electronics"
            records[j]["is_fraud"] = 1
            used.add(j)
            pattern_counts["velocity"] += 1

    # Impossible travel plus high-ticket purchases
    for i in range(len(records)):
        if len(used) >= target_fraud:
            break
        if i not in used and rng.random() < 0.012:
            user_id = int(records[i]["user_id"])
            mean = float(profiles[user_id]["mean"])
            records[i]["location"] = str(rng.choice(["GB-LON", "NG-LOS", "SG-SIN"]))
            records[i]["amount"] = round(rng.uniform(max(900, mean * 5), max(1800, mean * 12)), 2)
            records[i]["merchant_category"] = str(rng.choice(["electronics", "luxury_retail", "travel"]))
            records[i]["is_fraud"] = 1
            used.add(i)
            pattern_counts["travel"] += 1

    # Account takeover drains
    for i in range(len(records)):
        if len(used) >= target_fraud:
            break
        if i not in used:
            user_id = int(records[i]["user_id"])
            mean = float(profiles[user_id]["mean"])
            if rng.random() < 0.012:
                records[i]["amount"] = round(rng.uniform(mean * 6.5, mean * 12), 2)
                records[i]["merchant_category"] = str(rng.choice(["wire_transfer", "luxury_retail"]))
                records[i]["is_fraud"] = 1
                used.add(i)
                pattern_counts["takeover"] += 1

    # Fill remaining fraud labels if pattern injection stopped early.
    for i in range(len(records)):
        if len(used) >= target_fraud:
            break
        records[i]["is_fraud"] = 1
        records[i]["merchant_category"] = str(rng.choice(["wire_transfer", "luxury_retail"]))
        used.add(i)

    records.sort(key=lambda row: str(row["timestamp"]))
    create_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM transactions")
        conn.executemany(
            """
            INSERT INTO transactions (
                tx_id, user_id, timestamp, amount, merchant_category, location,
                device_type, is_fraud, predicted_risk_score, is_flagged, risk_reasons
            ) VALUES (
                :tx_id, :user_id, :timestamp, :amount, :merchant_category, :location,
                :device_type, :is_fraud, :predicted_risk_score, :is_flagged, :risk_reasons
            )
            """,
            records,
        )
        conn.commit()
    return {"seeded_rows": len(records), "fraud_rows": sum(r["is_fraud"] for r in records), **pattern_counts}


def generate_live_batch(n: int = 15, fraud_ratio: float = 0.20, seed: int | None = None) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    now = datetime.now().replace(microsecond=0)
    batch: List[Dict[str, object]] = []
    profiles = _baseline_profiles(rng)
    fraud_count = int(round(n * fraud_ratio))
    for i in range(n):
        is_fraud = i < fraud_count
        user_id = int(rng.integers(1001, 1101))
        profile = profiles[user_id]
        mean = float(profile["mean"])
        timestamp = (now - timedelta(seconds=n - i)).strftime("%Y-%m-%d %H:%M:%S")
        if is_fraud:
            pattern = int(rng.integers(0, 3))
            if pattern == 0:
                amount = float(rng.uniform(mean * 0.8, mean * 2.0))
                merchant = str(rng.choice(["digital_goods", "electronics"]))
                location = str(profile["home"])
            elif pattern == 1:
                amount = float(rng.uniform(900, 3500))
                merchant = str(rng.choice(["electronics", "travel", "luxury_retail"]))
                location = str(rng.choice(["GB-LON", "NG-LOS", "SG-SIN"]))
            else:
                amount = float(rng.uniform(mean * 6.5, mean * 12))
                merchant = str(rng.choice(["wire_transfer", "luxury_retail"]))
                location = str(profile["home"]) if rng.random() < 0.7 else str(rng.choice(["GB-LON", "NG-LOS"]))
            device = str(rng.choice(DEVICES))
        else:
            amount = max(1.0, float(rng.lognormal(np.log(mean), 0.5)))
            merchant = str(rng.choice(CATEGORIES, p=[0.30, 0.18, 0.10, 0.12, 0.13, 0.08, 0.09]))
            location = str(profile["home"]) if rng.random() < 0.85 else str(rng.choice(LOCATIONS))
            device = str(rng.choice(DEVICES, p=[0.48, 0.37, 0.15]))
        batch.append({
            "tx_id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": timestamp,
            "amount": round(amount, 2),
            "merchant_category": merchant,
            "location": location,
            "device_type": device,
            "is_fraud": int(is_fraud),
        })
    return batch


def insert_transactions(rows: Iterable[Dict[str, object]], db_path=DB_PATH) -> None:
    rows = list(rows)
    if not rows:
        return
    create_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO transactions (
                tx_id, user_id, timestamp, amount, merchant_category, location,
                device_type, is_fraud, predicted_risk_score, is_flagged, risk_reasons
            ) VALUES (
                :tx_id, :user_id, :timestamp, :amount, :merchant_category,
                :location, :device_type, :is_fraud,
                :predicted_risk_score, :is_flagged, :risk_reasons
            )
            """,
            [
                {
                    **row,
                    "predicted_risk_score": row.get("predicted_risk_score"),
                    "is_flagged": row.get("is_flagged"),
                    "risk_reasons": row.get("risk_reasons"),
                    "is_fraud": int(row.get("is_fraud", 0)),
                }
                for row in rows
            ],
        )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the fraud transaction database")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()
    print(seed_database(args.rows, args.seed, args.db))


if __name__ == "__main__":
    main()
