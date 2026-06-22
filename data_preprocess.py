"""
data_preprocess.py
RiskLens
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

NUMERIC_COLS = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
ORDINAL_COLS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

TARGET_COL = "Default"
DROP_COLS = ["ID"]

INVALID_EDUCATION = {0, 5, 6}
INVALID_MARRIAGE = {0}


def clean_data(data):
    """Remove invalid category codes."""

    data = data[~data["EDUCATION"].isin(INVALID_EDUCATION)]
    data = data[~data["MARRIAGE"].isin(INVALID_MARRIAGE)]

    for col in CATEGORICAL_COLS:
        data[col] = data[col].astype(str)

    return data.reset_index(drop=True)


def build_preprocessor():
    """Create preprocessing pipeline."""

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_COLS),
            (
                "ord",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_COLS,
            ),
        ]
    )


def preprocess_data(data, test_size=0.25, random_state=42):
    """Clean, preprocess, and balance the dataset."""

    data = clean_data(data)

    X = data.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    y = data[TARGET_COL]

    logger.info(f"Class distribution before SMOTE:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    pipeline = imPipeline(
        steps=[
            ("preprocessor", preprocessor)
        ]
    )

    pipeline.fit(X_train)

    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    smote = SMOTE(random_state=random_state)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed,
        y_train,
    )

    logger.info(
        f"Class distribution after SMOTE:\n"
        f"{pd.Series(y_train_resampled).value_counts()}"
    )

    return (
        X_train_resampled,
        X_test_processed,
        y_train_resampled,
        y_test,
    )
