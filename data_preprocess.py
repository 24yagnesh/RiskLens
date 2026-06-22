"""
data_preprocess.py
------------------
RiskLens

Handles data cleaning, feature engineering, preprocessing pipeline,
and SMOTE-based oversampling.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as imPipeline
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
    "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6",
]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
ORDINAL_COLS     = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

TARGET_COL = "Default"
DROP_COLS  = ["ID"]

# Invalid codes to remove (per dataset documentation)
INVALID_EDUCATION = {0, 5, 6}
INVALID_MARRIAGE  = {0}


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with undocumented / invalid category codes and
    cast categorical columns to string for encoding.

    Parameters
    ----------
    data : pd.DataFrame
        Raw dataframe loaded from the Excel source file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    initial_len = len(data)

    data = data[~data["EDUCATION"].isin(INVALID_EDUCATION)]
    data = data[~data["MARRIAGE"].isin(INVALID_MARRIAGE)]

    removed = initial_len - len(data)
    logger.info(f"Removed {removed} rows with invalid category codes.")

    for col in CATEGORICAL_COLS:
        data[col] = data[col].astype(str)

    return data.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies:
      - StandardScaler     → numeric billing / payment amount columns
      - OneHotEncoder      → nominal categorical columns (SEX, EDUCATION, MARRIAGE)
      - OrdinalEncoder     → ordinal payment-status columns (PAY_1 … PAY_6)

    Returns
    -------
    ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_COLS),
            ("ord", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ), ORDINAL_COLS),
        ],
        remainder="drop",
    )


def preprocess_data(
    data: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline:
      1. Clean raw data.
      2. Split into train / test (stratified by target).
      3. Fit preprocessing transformers on training set only.
      4. Apply SMOTE oversampling to balance the training set.

    Parameters
    ----------
    data         : Raw dataframe.
    test_size    : Fraction of data reserved for testing (default 0.25).
    random_state : Seed for reproducibility (default 42).

    Returns
    -------
    X_train_resampled : np.ndarray  — SMOTE-oversampled training features
    X_test_processed  : np.ndarray  — Scaled / encoded test features
    y_train_resampled : np.ndarray  — Oversampled training labels
    y_test            : np.ndarray  — Original test labels
    """
    data = clean_data(data)

    X = data.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    y = data[TARGET_COL]

    logger.info(f"Class distribution before SMOTE:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    preprocessor = build_preprocessor()
    pipeline = imPipeline(steps=[("preprocessor", preprocessor)])
    pipeline.fit(X_train)

    X_train_processed = pipeline.transform(X_train)
    X_test_processed  = pipeline.transform(X_test)

    smote = SMOTE(sampling_strategy="auto", random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed, y_train
    )
    logger.info(
        f"Class distribution after SMOTE:\n"
        f"{pd.Series(y_train_resampled).value_counts()}"
    )

    return X_train_resampled, X_test_processed, y_train_resampled, y_test
