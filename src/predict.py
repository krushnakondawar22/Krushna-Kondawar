"""Batch and single-student prediction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from src.data_pipeline import load_and_clean_data
from src.train_model import (
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    load_model,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH: Final[Path] = PROJECT_ROOT / "data" / "placement_predictions.csv"

FEATURE_COLUMNS: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _assign_risk_level(probability_percent: float) -> str:
    """Map placement probability to actionable risk tiers."""
    if probability_percent >= 70:
        return "Placement Ready"
    if probability_percent >= 45:
        return "Moderate"
    return "High Risk"


def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Run model inference and append prediction columns."""
    pipeline = model or load_model()
    features = df[FEATURE_COLUMNS].copy()

    predicted_status = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    class_labels = list(pipeline.named_steps["classifier"].classes_)
    placed_index = class_labels.index("Placed")
    placement_probability = probabilities[:, placed_index] * 100

    result = df.copy()
    result["Predicted_Placement_Status"] = predicted_status
    result["Placement_Probability_Percent"] = placement_probability.round(2)
    result["Risk_Level"] = result["Placement_Probability_Percent"].apply(_assign_risk_level)
    return result


def export_for_powerbi(
    csv_path: Path | None = None,
    model_path: Path | None = None,
) -> Path:
    """Generate enriched predictions for Power BI dashboard refresh."""
    df = load_and_clean_data()
    pipeline = load_model(model_path or MODEL_PATH)
    enriched = predict_batch(df, model=pipeline)

    output_path = csv_path or PREDICTIONS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    return output_path
