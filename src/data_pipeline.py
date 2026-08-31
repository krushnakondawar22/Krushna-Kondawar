"""Pandas data generation, cleaning, and feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw_placement_data.csv"

BRANCHES: Final[list[str]] = ["Computer", "Mechanical", "Electrical", "Civil", "E&TC"]
COMMUNICATION_LEVELS: Final[list[str]] = ["Poor", "Average", "Good", "Excellent"]
INTERNSHIP_OPTIONS: Final[list[str]] = ["Yes", "No"]

COMMUNICATION_SCORES: Final[dict[str, float]] = {
    "Poor": 0.25,
    "Average": 0.5,
    "Good": 0.75,
    "Excellent": 1.0,
}


def generate_synthetic_dataset(n_students: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic diploma student records with realistic metrics."""
    rng = np.random.default_rng(random_state)

    diploma_pct = rng.uniform(40.0, 95.0, n_students).round(1)
    cgpa = (diploma_pct / 9.5).round(2)
    attendance = rng.uniform(50.0, 98.0, n_students).round(1)
    aptitude = rng.integers(0, 101, n_students)
    technical = rng.integers(0, 101, n_students)
    projects = rng.integers(0, 6, n_students)

    branch = rng.choice(BRANCHES, n_students)
    communication = rng.choice(
        COMMUNICATION_LEVELS,
        n_students,
        p=[0.15, 0.35, 0.35, 0.15],
    )
    internship = rng.choice(INTERNSHIP_OPTIONS, n_students, p=[0.55, 0.45])

    df = pd.DataFrame(
        {
            "Student_ID": [f"STU{i:04d}" for i in range(1, n_students + 1)],
            "Branch": branch,
            "Diploma_Percentage": diploma_pct,
            "CGPA": cgpa,
            "Attendance_Percent": attendance,
            "Aptitude_Score": aptitude,
            "Technical_Score": technical,
            "Communication_Skill": communication,
            "Internship_Completed": internship,
            "Projects_Count": projects,
        }
    )

    df["Placement_Status"] = _derive_placement_status(df)
    return df


def _derive_placement_status(df: pd.DataFrame) -> pd.Series:
    """Derive placement outcome from weighted academic and skill metrics."""
    comm_score = df["Communication_Skill"].map(COMMUNICATION_SCORES).fillna(0.5)
    internship_score = (df["Internship_Completed"] == "Yes").astype(float)
    project_score = (df["Projects_Count"] / 5.0).clip(0, 1)

    weighted_score = (
        (df["Diploma_Percentage"] / 100.0) * 0.22
        + (df["CGPA"] / 10.0) * 0.10
        + (df["Attendance_Percent"] / 100.0) * 0.12
        + (df["Aptitude_Score"] / 100.0) * 0.18
        + (df["Technical_Score"] / 100.0) * 0.18
        + comm_score * 0.10
        + internship_score * 0.05
        + project_score * 0.05
    )

    noise = np.random.default_rng(42).normal(0, 0.04, len(df))
    return np.where(weighted_score + noise >= 0.58, "Placed", "Not Placed")


def load_and_clean_data(csv_path: Path | None = None) -> pd.DataFrame:
    """Load raw data, handle missing values, and validate feature ranges."""
    path = csv_path or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)

    numeric_cols = [
        "Diploma_Percentage",
        "CGPA",
        "Attendance_Percent",
        "Aptitude_Score",
        "Technical_Score",
        "Projects_Count",
    ]
    categorical_cols = ["Branch", "Communication_Skill", "Internship_Completed", "Placement_Status"]

    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[categorical_cols] = df[categorical_cols].astype("string")

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(
        df[categorical_cols].mode().iloc[0].to_dict()
    )

    _validate_ranges(df)
    return df


def _validate_ranges(df: pd.DataFrame) -> None:
    """Ensure generated or loaded data stays within expected bounds."""
    checks = {
        "Diploma_Percentage": (40.0, 95.0),
        "CGPA": (4.0, 10.0),
        "Attendance_Percent": (50.0, 98.0),
        "Aptitude_Score": (0, 100),
        "Technical_Score": (0, 100),
        "Projects_Count": (0, 5),
    }
    for column, (low, high) in checks.items():
        if not df[column].between(low, high).all():
            raise ValueError(f"Column '{column}' contains out-of-range values.")

    invalid_branch = ~df["Branch"].isin(BRANCHES)
    invalid_comm = ~df["Communication_Skill"].isin(COMMUNICATION_LEVELS)
    invalid_intern = ~df["Internship_Completed"].isin(INTERNSHIP_OPTIONS)
    if invalid_branch.any() or invalid_comm.any() or invalid_intern.any():
        raise ValueError("Invalid categorical values detected in raw dataset.")


def save_raw_data(df: pd.DataFrame, csv_path: Path | None = None) -> Path:
    """Persist generated dataset to CSV."""
    path = csv_path or RAW_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_data_pipeline(n_students: int = 500) -> pd.DataFrame:
    """Generate, clean, and save the raw placement dataset."""
    df = generate_synthetic_dataset(n_students=n_students)
    save_raw_data(df)
    return load_and_clean_data()
