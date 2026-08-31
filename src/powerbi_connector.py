# Copy this script into Power BI Desktop: Get Data -> Python script
# Ensure the project root path below matches your local installation.

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\HP\smart_placement_predictor")
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_placement_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "placement_model.joblib"

FEATURE_COLUMNS = [
    "Diploma_Percentage",
    "CGPA",
    "Attendance_Percent",
    "Aptitude_Score",
    "Technical_Score",
    "Projects_Count",
    "Branch",
    "Communication_Skill",
    "Internship_Completed",
]


def assign_risk_level(probability_percent: float) -> str:
    if probability_percent >= 70:
        return "Placement Ready"
    if probability_percent >= 45:
        return "Moderate"
    return "High Risk"


df = pd.read_csv(RAW_DATA_PATH)
pipeline = joblib.load(MODEL_PATH)
*-
features = df[FEATURE_COLUMNS]
predicted_status = pipeline.predict(features)
probabilities = pipeline.predict_proba(features)
class_labels = list(pipeline.named_steps["classifier"].classes_)
placed_index = class_labels.index("Placed")
placement_probability = probabilities[:, placed_index] * 100

df["Predicted_Placement_Status"] = predicted_status
df["Placement_Probability_Percent"] = placement_probability.round(2)
df["Risk_Level"] = df["Placement_Probability_Percent"].apply(assign_risk_level)

dataset = df
