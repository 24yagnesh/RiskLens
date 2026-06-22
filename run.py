"""
run.py
RiskLens
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
logger = logging.getLogger(**name**)

DEFAULT_DATA_PATH = "dataset.xls"

def parse_args():
parser = argparse.ArgumentParser(
description="RiskLens Pipeline"
)

```
parser.add_argument(
    "--data",
    default=DEFAULT_DATA_PATH,
    help="Path to dataset",
)

parser.add_argument(
    "--no-show",
    action="store_true",
    help="Run without displaying plots",
)

return parser.parse_args()
```

def load_data(path):
"""Load dataset."""

```
data_path = Path(path)

if not data_path.exists():
    logger.error(f"Dataset not found: {data_path}")
    sys.exit(1)

logger.info(f"Loading dataset from {data_path}")

return pd.read_excel(data_path, header=1)
```

def main():
args = parse_args()

```
data = load_data(args.data)

logger.info("Preprocessing data...")
X_train, X_test, y_train, y_test = preprocess_data(data)

logger.info("Training models...")
results = train_models(
    X_train,
    X_test,
    y_train,
    y_test,
)

logger.info("Generating plots...")
plot_results(
    results,
    show=not args.no_show,
)

plot_all_confusion_matrices(
    results,
    X_test,
    y_test,
    show=not args.no_show,
)

logger.info("Done.")
```

if **name** == "**main**":
main()
