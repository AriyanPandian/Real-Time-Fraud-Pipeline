"""Hybrid supervised and unsupervised fraud model training."""

import json
import sqlite3
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FEATURE_COLUMNS, extract_training_features
from src.simulator import DB_PATH, seed_database


NUMERIC_FEATURES = [
    "amount", "user_hist_avg_amt", "amt_to_user_avg_ratio",
    "tx_count_prev_5", "mins_since_prev_tx", "location_changed",
    "is_high_risk_merchant", "is_foreign_location", "rapid_burst_flag",
]
CATEGORICAL_FEATURES = ["merchant_category", "device_type", "location"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def hybrid_scores(
    classifier: Pipeline,
    isolation: IsolationForest,
    preprocessor: ColumnTransformer,
    features: pd.DataFrame,
) -> np.ndarray:
    transformed = preprocessor.transform(features)
    supervised_probability = classifier.predict_proba(features)[:, 1]
    raw_anomaly = -isolation.score_samples(transformed)
    low = float(np.percentile(raw_anomaly, 1))
    high = float(np.percentile(raw_anomaly, 99))
    normalized = np.clip((raw_anomaly - low) / max(high - low, 1e-9), 0.0, 1.0)
    return np.clip(0.80 * supervised_probability + 0.20 * normalized, 0.0, 1.0)


def risk_reasons(row) -> str:
    values = row._asdict() if hasattr(row, "_asdict") else dict(row)
    reasons = []
    ratio = float(values.get("amt_to_user_avg_ratio", 0.0))
    mins = float(values.get("mins_since_prev_tx", 1440.0))
    if ratio >= 6.0:
        reasons.append(f"Spend {ratio:.1f}x above user average")
    if mins <= 10.0:
        reasons.append(f"Rapid velocity ({mins:.0f} mins since last tx)")
    if int(values.get("location_changed", 0)):
        reasons.append(f"Location changed to {values.get('location', 'unknown')}")
    if int(values.get("is_high_risk_merchant", 0)):
        reasons.append(f"High-risk merchant: {values.get('merchant_category', 'unknown')}")
    if int(values.get("is_foreign_location", 0)):
        reasons.append(f"Foreign location: {values.get('location', 'unknown')}")
    return ", ".join(reasons) if reasons else "No high-risk indicators"


def train_and_evaluate(db_path=DB_PATH, model_dir="models") -> Dict[str, object]:
    db_path = Path(db_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data = extract_training_features(db_path).sort_values(["timestamp", "tx_id"]).reset_index(drop=True)
    if data.empty:
        raise RuntimeError("No transactions available for training")

    split = int(len(data) * 0.80)
    train, test = data.iloc[:split], data.iloc[split:]
    X_train = train[FEATURE_COLUMNS]
    y_train = train["is_fraud"].astype(int)
    X_test = test[FEATURE_COLUMNS]
    y_test = test["is_fraud"].astype(int)

    preprocessor = build_preprocessor()
    classifier = Pipeline([
        ("classifier", HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=200,
            learning_rate=0.07, random_state=42,
        ))
    ])
    X_train_prepared = preprocessor.fit_transform(X_train)
    classifier.fit(X_train_prepared, y_train)

    isolation = IsolationForest(contamination=0.04, random_state=42)
    isolation.fit(X_train_prepared)

    def score(features: pd.DataFrame) -> np.ndarray:
        transformed = preprocessor.transform(features)
        supervised = classifier.predict_proba(transformed)[:, 1]
        raw = -isolation.score_samples(transformed)
        low, high = np.percentile(raw, [1, 99])
        normalized = np.clip((raw - low) / max(float(high - low), 1e-9), 0, 1)
        return np.clip(0.80 * supervised + 0.20 * normalized, 0, 1)

    risk_scores = score(X_test)
    thresholds = np.unique(np.quantile(risk_scores, np.linspace(0.01, 0.99, 200)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        predicted = (risk_scores >= threshold).astype(int)
        f1 = f1_score(y_test, predicted, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1

    predictions = (risk_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    permutation = permutation_importance(
        classifier, preprocessor.transform(X_test), y_test,
        n_repeats=3, random_state=42, scoring="average_precision",
    )
    feature_names = list(NUMERIC_FEATURES) + list(
        preprocessor.named_transformers_["categorical"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    importances = sorted([
        {"feature": name, "importance": float(value)}
        for name, value in zip(feature_names, permutation.importances_mean)
    ], key=lambda item: item["importance"], reverse=True)

    joblib.dump({
        "preprocessor": preprocessor,
        "classifier": classifier,
        "isolation_forest": isolation,
        "threshold": best_threshold,
    }, model_dir / "fraud_model.pkl")

    metadata = {
        "roc_auc": float(roc_auc_score(y_test, risk_scores)),
        "pr_auc": float(average_precision_score(y_test, risk_scores)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "optimal_decision_threshold": best_threshold,
        "feature_importances": importances,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "training_rows": len(train),
        "holdout_rows": len(test),
    }
    with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    all_scores = score(data[FEATURE_COLUMNS])
    with sqlite3.connect(db_path) as conn:
        for row, risk_score in zip(data.itertuples(index=False), all_scores):
            conn.execute(
                """
                UPDATE transactions
                SET predicted_risk_score = ?, is_flagged = ?, risk_reasons = ?
                WHERE tx_id = ?
                """,
                (
                    float(risk_score),
                    int(risk_score >= best_threshold),
                    risk_reasons(row),
                    row.tx_id,
                ),
            )
        conn.commit()
    return metadata


def main() -> None:
    if not Path(DB_PATH).exists():
        seed_database(db_path=DB_PATH)
    print(json.dumps(train_and_evaluate(DB_PATH), indent=2))


if __name__ == "__main__":
    main()
