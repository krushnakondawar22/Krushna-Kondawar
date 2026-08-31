"""Single-entry execution script for the Smart College Placement Predictor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import RAW_DATA_PATH, run_data_pipeline
from src.predict import PREDICTIONS_PATH, export_for_powerbi
from src.train_model import MODEL_PATH, run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run data generation, model training, and Power BI export."""
    logger.info("Starting Smart College Placement Predictor pipeline...")

    logger.info("Step 1/3: Generating and cleaning synthetic student data...")
    run_data_pipeline(n_students=500)
    logger.info("Raw data saved to %s", RAW_DATA_PATH)

    logger.info("Step 2/3: Training Random Forest placement model...")
    _, metrics = run_training()
    logger.info(
        "Model saved to %s (accuracy=%.3f, precision=%.3f, recall=%.3f)",
        MODEL_PATH,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
    )

    logger.info("Step 3/3: Exporting enriched predictions for Power BI...")
    export_for_powerbi()
    logger.info("Predictions exported to %s", PREDICTIONS_PATH)

    logger.info("Pipeline complete. Data is ready for Power BI dashboard refresh.")


if __name__ == "__main__":
    main()
