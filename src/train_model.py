"""Scikit-learn model training, evaluation, and serialization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_pipeline import load_and_clean_data

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "placement_model.joblib"

NUMERIC_FEATURES: Final[list[str]] = [
    "Diploma_Percentage",
    "CGPA",
    "Attendance_Percent",
    "Aptitude_Score",
    "Technical_Score",
    "Projects_Count",
]
CATEGORICAL_FEATURES: Final[list[str]] = [
    "Branch",
    "Communication_Skill",
    "Internship_Completed",
]
TARGET_COLUMN: Final[str] = "Placement_Status"


def build_pipeline() -> Pipeline:
    """Create preprocessing + Random Forest classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate(df: pd.DataFrame | None = None) -> tuple[Pipeline, dict[str, float]]:
    """Train the model and return metrics on the held-out test set."""
    data = df if df is not None else load_and_clean_data()

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = data[feature_cols]
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="Placed"),
        "recall": recall_score(y_test, predictions, pos_label="Placed"),
    }

    logger.info("Model evaluation — accuracy: %.3f", metrics["accuracy"])
    logger.info("Model evaluation — precision: %.3f", metrics["precision"])
    logger.info("Model evaluation — recall: %.3f", metrics["recall"])

    return pipeline, metrics


def save_model(pipeline: Pipeline, model_path: Path | None = None) -> Path:
    """Serialize trained pipeline to disk."""
    path = model_path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(model_path: Path | None = None) -> Pipeline:
    """Load a serialized model pipeline."""
    path = model_path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return joblib.load(path)


def run_training(df: pd.DataFrame | None = None) -> tuple[Pipeline, dict[str, float]]:
    """Train, evaluate, and persist the placement prediction model."""
    pipeline, metrics = train_and_evaluate(df)
    save_model(pipeline)
    return pipeline, metrics
