# RiskLens 💳

A modular machine learning pipeline to predict credit card payment defaults using real-world client data.

> **Yagnesh Bonnada · BS Undergraduate · Indian Institute of Technology Kanpur**

## Overview

RiskLens builds and benchmarks five classifiers on the [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset (30,000 clients, Taiwan, 2005). The pipeline handles data cleaning, mixed-type feature encoding, class imbalance via SMOTE, model training, and evaluation — all in a clean, modular codebase.

## Key Features

- **Mixed preprocessing pipeline** — StandardScaler for numeric features, OneHotEncoder for nominal categoricals, OrdinalEncoder for payment-status columns, composed in a single `ColumnTransformer`
- **SMOTE oversampling** — addresses the ~78/22 non-defaulter/defaulter imbalance in the training set without touching the test set
- **Five classifiers benchmarked** — Logistic Regression, Random Forest, SVM, XGBoost, LightGBM
- **F1-Score as primary metric** — appropriate for imbalanced classification; accuracy alone is misleading here
- **Modular codebase** — separate files for preprocessing, training, plotting, and the entry point; easy to extend

## Project Structure

```
RiskLens/
├── data_preprocess.py   # Cleaning, encoding, train-test split, SMOTE
├── train.py             # Model registry and training logic
├── plot.py              # Model comparison bar chart + confusion matrices
├── run.py               # Entry point (CLI with --data and --no-show flags)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/24yagnesh/RiskLens.git
cd RiskLens
pip install -r requirements.txt
```

## Dataset

Download the dataset from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and place it in the project root as `dataset.xls`.

| Feature group  | Columns |
|---|---|
| Client info    | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Payment status | `PAY_1` … `PAY_6` |
| Bill amounts   | `BILL_AMT1` … `BILL_AMT6` |
| Payment amounts| `PAY_AMT1` … `PAY_AMT6` |
| Target         | `Default` (1 = defaulted, 0 = did not default) |

## Usage

```bash
# Run the full pipeline with default dataset path
python run.py

# Custom dataset path
python run.py --data path/to/dataset.xls

# Save plots without displaying them interactively
python run.py --no-show
```

## Methods

| Step | Technique |
|---|---|
| Numeric scaling    | StandardScaler |
| Nominal encoding   | OneHotEncoder (drop first) |
| Ordinal encoding   | OrdinalEncoder |
| Class imbalance    | SMOTE (training set only) |
| Models             | Logistic Regression, Random Forest, SVM, XGBoost, LightGBM |
| Primary metric     | Macro F1-Score |

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
matplotlib
seaborn
openpyxl
```

Install all at once:

```bash
pip install -r requirements.txt
```

## Author

Yagnesh Bonnada · BS Undergraduate · Indian Institute of Technology Kanpur
