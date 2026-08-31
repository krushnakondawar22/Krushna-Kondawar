
"""Lightweight Flask REST API for real-time placement inference."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request

from src.predict import FEATURE_COLUMNS, predict_batch
from src.train_model import load_model

app = Flask(__name__)
_model = None


def get_model():
    """Lazy-load the trained model pipeline."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _build_recommendations(student: dict[str, Any], probability: float) -> list[str]:
    """Generate tailored skill-improvement recommendations."""

    recommendations: list[str] = []

    if student.get("Attendance_Percent", 100) < 75:
        recommendations.append("Improve class attendance to at least 75% for better placement readiness.")

    if student.get("Aptitude_Score", 100) < 60:
        recommendations.append("Practice aptitude tests weekly to strengthen logical and quantitative skills.")

    if student.get("Technical_Score", 100) < 65:
        recommendations.append("Complete branch-specific technical modules and coding/problem-solving drills.")

    if student.get("Communication_Skill") in {"Poor", "Average"}:
        recommendations.append("Join communication workshops and mock interview sessions.")

    if student.get("Internship_Completed") == "No":
        recommendations.append("Pursue an industry internship to gain practical exposure.")

    if student.get("Projects_Count", 0) < 2:
        recommendations.append("Build at least two portfolio projects demonstrating core domain skills.")

    if student.get("Diploma_Percentage", 100) < 60:
        recommendations.append("Focus on improving diploma academic performance in upcoming assessments.")

    if not recommendations:
        if probability >= 70:
            recommendations.append("Maintain current performance and apply to companies aligned with your branch.")
        else:
            recommendations.append("Continue balanced improvement across aptitude, technical, and communication skills.")

    return recommendations


@app.route("/health", methods=["GET"])
def health() -> tuple[Any, int]:
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Any, int]:
    """Predict placement outcome for a single student profile."""
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    missing = [col for col in FEATURE_COLUMNS if col not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        import pandas as pd

        student_df = pd.DataFrame([payload])
        result = predict_batch(student_df, model=get_model()).iloc[0]

        probability = float(result["Placement_Probability_Percent"])
        response = {
            "predicted_placement_status": result["Predicted_Placement_Status"],
            "placement_probability_percent": probability,
            "risk_level": result["Risk_Level"],
            "recommendations": _build_recommendations(payload, probability),
        }
        return jsonify(response), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
