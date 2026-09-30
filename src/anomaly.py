"""Unsupervised anomaly detection for window-level audio features."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_COLUMNS = {
    "source_file",
    "file_name",
    "start_sec",
    "end_sec",
    "sample_rate",
}


def build_model(random_state: int = 42) -> Pipeline:
    """Build a reproducible preprocessing + Isolation Forest pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination="auto",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def score_anomalies(
    features: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Pipeline]:
    """Fit Isolation Forest and return relative anomaly scores.

    Higher anomaly_score values represent windows that are more unusual
    relative to the population used for fitting. This is not a probability.
    """
    feature_columns = [
        column for column in features.columns if column not in META_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No feature columns available for anomaly detection.")

    x = features[feature_columns]

    model = build_model(random_state=random_state)
    model.fit(x)

    result = features.copy()
    result["anomaly_score"] = (-model.decision_function(x)).astype(float)
    result["is_anomaly"] = model.predict(x) == -1

    return (
        result.sort_values("anomaly_score", ascending=False).reset_index(drop=True),
        model,
    )


def save_model(model: Pipeline, path: str | Path) -> None:
    """Persist a fitted scikit-learn pipeline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
