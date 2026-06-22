"""
run.py
------
RiskLens

Entry point for the RiskLens pipeline.

Usage
-----
    python run.py                          # uses default dataset path
    python run.py --data path/to/data.xls  # custom dataset location
    python run.py --no-show                # save plots without displaying them
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from data_preprocess import preprocess_data
from train import train_models
from plot import plot_results, plot_all_confusion_matrices

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = "dataset.xls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit Card Client Default Risk — ML Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the raw Excel dataset (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Display plots without saving them to disk",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    """Load raw Excel file; exit gracefully if the file is missing."""
    data_path = Path(path)
    if not data_path.exists():
        logger.error(f"Dataset not found at '{data_path}'. "
                     "Download it from: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")
        sys.exit(1)

    logger.info(f"Loading dataset from '{data_path}' ...")
    return pd.read_excel(data_path, header=1)


def main() -> None:
    args = parse_args()
    show_plots = not args.no_show

    # 1. Load raw data
    data = load_data(args.data)

    # 2. Preprocess + SMOTE
    logger.info("Preprocessing data ...")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # 3. Train & evaluate all models
    logger.info("Training models ...")
    results = train_models(X_train, X_test, y_train, y_test)

    # 4. Visualise results
    logger.info("Generating plots ...")
    plot_results(results, save_path=None, show=show_plots)
    plot_all_confusion_matrices(results, X_test, y_test, save_dir=None, show=show_plots)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
