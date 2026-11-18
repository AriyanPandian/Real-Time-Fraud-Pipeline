# Sentinel — Real-Time Fraud & Anomaly Detection Pipeline

Sentinel is a local, end-to-end fraud detection system that combines SQLite window-function feature engineering, supervised gradient boosting, unsupervised anomaly detection, a FastAPI scoring service, and a recruiter-ready Streamlit operations dashboard.

## Architecture

```text
Synthetic Transactions -> SQLite -> SQL Window Features
                                      |
                                      +-> HistGradientBoosting + IsolationForest
                                      |       |
                                      |       +-> Hybrid [0, 1] risk score
                                      |
                                      +-> FastAPI /score -> SQLite persistence
                                                |
                                                +-> Streamlit monitoring dashboard
```

## SQL Window Feature Engineering

The production query in `src/features.py` uses CTEs and `OVER (PARTITION BY user_id ORDER BY timestamp ...)`:

- expanding historical mean with `ROWS BETWEEN UNBOUDED PRECEDING AND 1 PRECEDING`
- the preceding five-user-transaction count
- `LAG` timestamps and locations
- elapsed minutes via `julianday()`
- high-risk merchant, foreign-location, and rapid-burst indicators

## Model

The holdout is the final 20% of chronologically ordered transactions, preventing temporal leakage. The hybrid score is:

```text
0.80 * supervised_probability + 0.20 * normalized_isolation_forest_anomaly
```

An optimal F1 decision threshold is selected on holdout data and persisted with permutation feature importances and a confusion matrix in `models/model_metadata.json`.

## Holdout Evaluation

Run `python -m src.train` to regenerate this table's metrics from the current data.

| Metric | Holdout Result |
|---|---:|
| ROC-AUC | See `models/model_metadata.json` |
| PR-AUC | See `models/model_metadata.json` |
| F1 | See `models/model_metadata.json` |
| Precision | See `models/model_metadata.json` |
| Recall | See `models/model_metadata.json` |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.simulator
python -m src.train

# Dashboard
streamlit run dashboard.py

# Optional REST API
uvicorn src.api:app --reload --port 8000
pytest -q
```

On Streamlit startup, missing database or model artifacts are created automatically. The dashboard uses FastAPI when it is available at `127.0.0.1:8000`; otherwise it uses direct local scoring.

## Docker

```bash
docker build -t sentinel-fraud .
docker run -p 8501:8501 sentinel-fraud
```
