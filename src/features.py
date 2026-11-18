"""SQL and real-time feature engineering for fraud detection."""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_DB_PATH = Path("data/transactions.db")

FEATURE_COLUMNS = [
    "amount", "merchant_category", "device_type", "location",
    "user_hist_avg_amt", "amt_to_user_avg_ratio", "tx_count_prev_5",
    "mins_since_prev_tx", "location_changed", "is_high_risk_merchant",
    "is_foreign_location", "rapid_burst_flag",
]

WINDOW_FEATURE_SQL = """
WITH ordered AS (
  SELECT tx_id, user_id, timestamp, amount, merchant_category, location, device_type, is_fraud,
    AVG(amount) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS user_hist_avg_amt,
    COUNT(*) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS tx_count_prev_5,
    LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_timestamp,
    LAG(location) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_location
  FROM transactions
)
SELECT tx_id, user_id, timestamp, amount, merchant_category, location, device_type, is_fraud,
  COALESCE(user_hist_avg_amt, 50.0) AS user_hist_avg_amt,
  amount / COALESCE(NULLIF(user_hist_avg_amt, 0), 50.0) AS amt_to_user_avg_ratio,
  COALESCE(tx_count_prev_5, 0) AS tx_count_prev_5,
  COALESCE((julianday(timestamp) - julianday(prev_timestamp)) * 1440.0, 1440.0) AS mins_since_prev_tx,
  CASE WHEN prev_location IS NULL THEN 0 WHEN location != prev_location THEN 1 ELSE 0 END AS location_changed,
  CASE WHEN merchant_category IN ('wire_transfer', 'digital_goods', 'luxury_retail') THEN 1 ELSE 0 END AS is_high_risk_merchant,
  CASE WHEN location NOT LIKE 'US-%' THEN 1 ELSE 0 END AS is_foreign_location,
  CASE WHEN COALESCE((julianday(timestamp) - julianday(prev_timestamp)) * 1440.0, 1440.0) <= 10.0 THEN 1 ELSE 0 END AS rapid_burst_flag
FROM ordered
ORDER BY timestamp, tx_id
"""


def extract_training_features(db_path=DEFAULT_DB_PATH) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction database not found: {path}")
    with sqlite3.connect(path) as conn:
        frame = pd.read_sql_query(WINDOW_FEATURE_SQL, conn)
    numeric = [
        "amount", "user_hist_avg_amt", "amt_to_user_avg_ratio",
        "tx_count_prev_5", "mins_since_prev_tx", "location_changed",
        "is_high_risk_merchant", "is_foreign_location", "rapid_burst_flag",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame[numeric] = frame[numeric].fillna(0.0)
    return frame


def compute_realtime_features(tx_dict: Dict[str, object], db_path=DEFAULT_DB_PATH) -> Tuple[Dict[str, float | int | str], List[str]]:
    amount = float(tx_dict["amount"])
    user_id = int(tx_dict["user_id"])
    timestamp = str(tx_dict.get("timestamp") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    category = str(tx_dict["merchant_category"])
    location = str(tx_dict["location"])
    device = str(tx_dict["device_type"])

    with sqlite3.connect(db_path) as conn:
        history = pd.read_sql_query(
            "SELECT timestamp, amount, location FROM transactions WHERE user_id = ? "
            "ORDER BY timestamp DESC LIMIT 20",
            conn,
            params=(user_id,),
        )

    hist_average = float(history["amount"].mean()) if not history.empty else 50.0
    ratio = amount / (hist_average if hist_average != 0 else 50.0)
    if history.empty:
        mins_since_prev = 1440.0
        prev_location = None
    else:
        previous = history.iloc[0]
        prev_location = str(previous["location"])
        if tx_dict.get("mins_since_prev_tx") is not None:
            mins_since_prev = float(tx_dict["mins_since_prev_tx"])
        else:
            try:
                delta = pd.Timestamp(timestamp) - pd.Timestamp(previous["timestamp"])
                mins_since_prev = max(0.0, float(delta.total_seconds() / 60.0))
            except Exception:
                mins_since_prev = 1440.0

    location_changed = int(bool(prev_location and location != prev_location))
    is_high_risk = int(category in ("wire_transfer", "digital_goods", "luxury_retail"))
    is_foreign = int(not location.startswith("US-"))
    rapid_burst = int(mins_since_prev <= 10.0)
    features = {
        "amount": amount,
        "merchant_category": category,
        "device_type": device,
        "location": location,
        "user_hist_avg_amt": hist_average,
        "amt_to_user_avg_ratio": ratio,
        "tx_count_prev_5": float(min(len(history), 5)),
        "mins_since_prev_tx": mins_since_prev,
        "location_changed": location_changed,
        "is_high_risk_merchant": is_high_risk,
        "is_foreign_location": is_foreign,
        "rapid_burst_flag": rapid_burst,
    }
    reasons: List[str] = []
    if ratio >= 6.0:
        reasons.append(f"Spend {ratio:.1f}x above user average")
    if mins_since_prev <= 10.0:
        reasons.append(f"Rapid velocity ({mins_since_prev:.0f} mins since last tx)")
    if prev_location and location_changed:
        reasons.append(f"Cross-border location jump ({prev_location} -> {location})")
    if is_high_risk:
        reasons.append(f"High-risk merchant: {category}")
    if is_foreign:
        reasons.append(f"Foreign location: {location}")
    return features, reasons
