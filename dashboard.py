"""Streamlit live operations dashboard for Sentinel."""

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.features import FEATURE_COLUMNS, WINDOW_FEATURE_SQL, compute_realtime_features
from src.simulator import DB_PATH, generate_live_batch, seed_database
from src.train import train_and_evaluate


st.set_page_config(
    page_title="Sentinel | Real-Time Fraud & Anomaly Pipeline",
    layout="wide",
)


@st.cache_resource
def initialize():
    db_path = Path(DB_PATH)
    model_path = Path("models/fraud_model.pkl")
    if not db_path.exists() or not model_path.exists():
        with st.spinner("Initializing database and training fraud model..."):
            if not db_path.exists():
                seed_database(db_path=db_path)
            train_and_evaluate(db_path=db_path, model_dir="models")
    metadata_path = Path("models/model_metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    artifact = joblib.load(model_path)
    return db_path, artifact, metadata


db_path, artifact, metadata = initialize()


def api_available() -> bool:
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=0.35)
        return response.status_ok if hasattr(response, "status_ok") else response.status_code == 200
    except requests.RequestException:
        return False


def direct_score(tx):
    features, reasons = compute_realtime_features(tx, db_path)
    frame = pd.DataFrame([features])[FEATURE_COLUMNS]
    transformed = artifact["preprocessor"].transform(frame)
    supervised = float(artifact["classifier"].predict_proba(transformed)[0, 1])
    raw = -float(artifact["isolation_forest"].score_samples(transformed)[0])
    anomaly = float(np.clip((raw + 0.05) / 0.20, 0.0, 1.0))
    score = float(np.clip(0.80 * supervised + 0.20 * anomaly, 0.0, 1.0))
    if features["amount"] >= 900 and features["is_foreign_location"] and features["is_high_risk_merchant"]:
        score = min(1.0, score + 0.35)
    if features["amt_to_user_avg_ratio"] >= 6.0:
        score = min(1.0, score + 0.20)
    if features["rapid_burst_flag"]:
        score = min(1.0, score + 0.08)
    return score, features, reasons


API_CONNECTED = api_available()
default_threshold = float(metadata.get("optimal_decision_threshold", 0.50))

st.sidebar.title("Sentinel Controls")
st.sidebar.caption("Real-time fraud operations")
threshold = st.sidebar.slider(
    "Alert risk threshold", 0.30, 0.90, default_threshold, 0.01
)

if st.sidebar.button("⚡ Simulate Live Traffic Burst (20 Tx)", use_container_width=True):
    if API_CONNECTED:
        try:
            requests.post(
                "http://127.0.0.1:8000/simulate",
                params={"n": 20, "fraud_ratio": 0.20},
                timeout=30,
            )
        except requests.RequestException:
            st.sidebar.error("The API became unavailable.")
    else:
        from src.simulator import insert_transactions
        rows = generate_live_batch(n=20, fraud_ratio=0.20)
        for tx in rows:
            score, _, reasons = direct_score(tx)
            tx["predicted_risk_score"] = score
            tx["is_flagged"] = int(score >= threshold)
            tx["risk_reasons"] = ", ".join(reasons) if reasons else "No high-risk indicators"
        insert_transactions(rows, db_path)
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Manual Transaction Sandbox")
with st.sidebar.form("sandbox"):
    user_id = st.number_input("User ID", 1001, 1100, 1001)
    amount = st.number_input("Amount (USD)", min_value=1.0, value=25.0, step=10.0)
    merchant = st.selectbox(
        "Merchant category",
        ["groceries", "restaurants", "travel", "electronics", "digital_goods",
         "wire_transfer", "luxury_retail"],
    )
    location = st.selectbox(
        "Location",
        ["US-IN", "US-PA", "US-NY", "US-CA", "GB-LON", "NG-LOS", "SG-SIN"],
    )
    device = st.selectbox("Device", ["mobile", "web", "pos_terminal"])
    mins_override = st.number_input(
        "Mins since previous tx override",
        min_value=0.0, max_value=1440.0, value=1440.0,
    )
    score_submit = st.form_submit_button("Score Transaction Now", use_container_width=True)

st.title("Sentinel")
if API_CONNECTED:
    st.success("API Connected (FastAPI)")
else:
    st.info("Direct Engine Mode")

if score_submit:
    tx = {
        "user_id": int(user_id),
        "amount": float(amount),
        "merchant_category": merchant,
        "location": location,
        "device_type": device,
        "mins_since_prev_tx": float(mins_override),
    }
    if API_CONNECTED:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/score", json=tx, timeout=10
            )
            payload = response.json()
            score = float(payload["risk_score"])
            features = payload["engineered_features"]
            reasons = payload["risk_reasons"]
            latency = float(payload["latency_ms"])
        except Exception:
            score, features, reasons = direct_score(tx)
            latency = 0.0
    else:
        score, features, reasons = direct_score(tx)
        latency = 0.0
    c1, c2 = st.columns([1, 3])
    c1.metric("Risk Score", f"{score:.3f}")
    c2.metric("Inference Latency", f"{latency:.2f} ms")
    st.progress(min(max(score, 0.0), 1.0))
    st.write("**Risk explanations:**")
    for reason in reasons:
        st.write(f"- {reason}")
    st.json(features)

tab1, tab2, tab3 = st.tabs([
    "Live Fraud Operations Center",
    "SQL Feature Engineering & Behavioral Diagnostics",
    "Model Performance & MLOps Metrics",
])

with sqlite3.connect(db_path) as conn:
    transactions = pd.read_sql_query(
        "SELECT * FROM transactions ORDER BY timestamp DESC",
        conn,
    )

transactions["timestamp_dt"] = pd.to_datetime(transactions["timestamp"], errors="coerce")
transactions["predicted_risk_score"] = pd.to_numeric(
    transactions["predicted_risk_score"], errors="coerce"
).fillna(0.0)

with tab1:
    blocked = float(transactions.loc[
        transactions["is_flagged"].fillna(0).astype(int) == 1, "amount"
    ].sum())
    flagged_rate = float(transactions["is_flagged"].fillna(0).astype(int).mean()) * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Transactions Processed", f"{len(transactions):,}")
    k2.metric("Flagged Anomaly Rate", f"{flagged_rate:.2f}%")
    k3.metric("Total Fraud Volume Blocked", f"${blocked:,.2f}")
    k4.metric(
        "Holdout ROC-AUC / PR-AUC",
        f"{metadata.get('roc_auc', 0):.3f} / {metadata.get('pr_auc', 0):.3f}",
    )

    timeline = (
        transactions.assign(hour=transactions["timestamp_dt"].dt.floor("h"))
        .groupby("hour")
        .agg(volume=("tx_id", "count"), flagged=("is_flagged", "sum"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_bar(x=timeline["hour"], y=timeline["volume"], name="Transaction Volume")
    fig.add_scatter(
        x=timeline["hour"], y=timeline["flagged"],
        name="Flagged Spikes", yaxis="y2", mode="lines+markers",
    )
    fig.update_layout(
        template="plotly_dark",
        barmode="overlay",
        yaxis=dict(title="Volume"),
        yaxis2=dict(title="Flagged", overlaying="y", side="right"),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    alerts = transactions[
        transactions["is_flagged"].fillna(0).astype(int) == 1
    ].sort_values("timestamp", ascending=False).head(50)
    st.subheader("High-Risk Alert Triage")
    st.dataframe(
        alerts[[ "tx_id", "user_id", "amount", "merchant_category",
                 "location", "predicted_risk_score", "risk_reasons"]].rename(
            columns={"predicted_risk_score": "risk_score"}
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    try:
        from src.features import extract_training_features
        diagnostic = extract_training_features(db_path)
    except Exception:
        diagnostic = pd.DataFrame()
    if not diagnostic.empty:
        diagnostic["tx_type"] = np.where(diagnostic["is_fraud"] == 1, "Fraud", "Legit")
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.box(diagnostic, x="tx_type", y="amt_to_user_avg_ratio", color="tx_type",
                   template="plotly_dark", log_y=True),
            use_container_width=True,
        )
        c2.plotly_chart(
            px.scatter(diagnostic.sample(min(3000, len(diagnostic)), random_state=42),
                       x="mins_since_prev_tx", y="amt_to_user_avg_ratio",
                       color="tx_type", template="plotly_dark", opacity=0.7),
            use_container_width=True,
        )
    with st.expander("View Production SQL Window Function Query"):
        st.code(WINDOW_FEATURE_SQL, language="sql")

with tab3:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{metadata.get('roc_auc', 0):.3f}")
    c2.metric("PR-AUC", f"{metadata.get('pr_auc', 0):.3f}")
    c3.metric("F1", f"{metadata.get('f1_score', 0):.3f}")
    c4.metric("Precision", f"{metadata.get('precision', 0):.3f}")
    c5.metric("Recall", f"{metadata.get('recall', 0):.3f}")

    importances = pd.DataFrame(metadata.get("feature_importances", []))
    if not importances.empty:
        fig = px.bar(
            importances.head(12).sort_values("importance"),
            x="importance", y="feature", orientation="h",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    cm = metadata.get("confusion_matrix", {})
    matrix = [[cm.get("tn", 0), cm.get("fp", 0)],
              [cm.get("fn", 0), cm.get("tp", 0)]]
    cm_fig = go.Figure(go.Heatmap(
        z=matrix, x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"], text=matrix, texttemplate="%{text}",
        colorscale="Blues",
    ))
    cm_fig.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(cm_fig, use_container_width=True)
