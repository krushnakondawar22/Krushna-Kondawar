# Smart College Placement Predictor

A modular Python + Scikit-learn + Power BI solution for predicting diploma engineering student placement outcomes.

## Tech Stack

- **Python 3.10+** — data processing and ML pipeline
- **Pandas / NumPy** — synthetic data generation and feature engineering
- **Scikit-learn** — Random Forest classifier with preprocessing pipeline
- **Flask** — real-time inference REST API
- **Power BI** — analytics dashboard connected via CSV or Python script

## Project Structure

```
smart_placement_predictor/
├── data/
│   ├── raw_placement_data.csv
│   └── placement_predictions.csv
├── models/
│   └── placement_model.joblib
├── src/
│   ├── data_pipeline.py
│   ├── train_model.py
│   ├── predict.py
│   └── powerbi_connector.py
├── powerbi/
│   ├── powerbi_theme.json
│   └── dax_measures.dax
├── app_api.py
├── main.py
└── requirements.txt
```

## Setup

```bash
cd smart_placement_predictor
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run the Full Pipeline

```bash
python main.py
```

This will:

1. Generate 500 synthetic diploma student records
2. Train the Random Forest model and save it to `models/placement_model.joblib`
3. Export enriched predictions to `data/placement_predictions.csv`

## Flask API

Start the API:

```bash
python app_api.py
```

POST to `http://localhost:5000/predict` with JSON body:

```json
{
  "Diploma_Percentage": 72.5,
  "CGPA": 7.63,
  "Attendance_Percent": 85.0,
  "Aptitude_Score": 68,
  "Technical_Score": 74,
  "Projects_Count": 2,
  "Branch": "Computer",
  "Communication_Skill": "Good",
  "Internship_Completed": "Yes"
}
```

## Power BI Setup

### Option A — CSV Import (Recommended)

1. Run `python main.py`
2. Open Power BI Desktop
3. **Get Data → Text/CSV** → select `data/placement_predictions.csv`

### Option B — Python Script

1. Update `PROJECT_ROOT` in `src/powerbi_connector.py` if needed
2. **Get Data → Python script** → paste contents of `src/powerbi_connector.py`

### Theme

1. **View → Themes → Browse for themes**
2. Select `powerbi/powerbi_theme.json`

### DAX Measures

Open `powerbi/dax_measures.dax`, copy measures into Power BI (**Modeling → New measure**), and use them in metric cards:

- Placement Rate %
- Avg Attendance %
- High Risk Count
- Avg Aptitude Score

## Suggested Dashboard Visuals

- **KPI cards** — Placement Rate %, Total Students, High Risk Count
- **Donut chart** — Predicted_Placement_Status
- **Bar chart** — Risk_Level by Branch
- **Scatter plot** — Aptitude_Score vs Technical_Score (colored by Risk_Level)
- **Table** — student-level details with filters for Branch and Risk_Level

## License

MIT
