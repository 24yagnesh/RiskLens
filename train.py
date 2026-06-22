"""
train.py
--------
RiskLens

Trains multiple classifiers on the preprocessed credit-card default dataset
and returns a summary of evaluation metrics for each model.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — add or remove models here without touching training logic
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=0.01, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "SVM": SVC(
        C=1, kernel="rbf", random_state=42
    ),
    "XGBoost": XGBClassifier(
        random_state=42, eval_metric="logloss", verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        random_state=42, verbose=-1
    ),
}


def train_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    models: dict | None = None,
) -> dict[str, dict]:
    """
    Train each classifier in the model registry and evaluate on the test set.

    Evaluation focuses on F1-Score (macro) as the primary metric because
    the raw dataset is heavily imbalanced (defaulters are the minority class).

    Parameters
    ----------
    X_train : Training features (post-SMOTE).
    X_test  : Test features.
    y_train : Training labels (post-SMOTE).
    y_test  : Test labels.
    models  : Optional custom model dict; falls back to MODEL_REGISTRY.

    Returns
    -------
    dict
        Mapping of model name → {"Accuracy": float, "F1-Score": float,
        "model": fitted estimator}
    """
    if models is None:
        models = MODEL_REGISTRY

    results: dict[str, dict] = {}

    for name, model in models.items():
        logger.info(f"Training {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")

        results[name] = {
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "model":    model,
        }

        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Non-Defaulter', 'Defaulter'])}")

    # Print ranked summary
    summary = (
        pd.DataFrame(
            {k: {"Accuracy": v["Accuracy"], "F1-Score": v["F1-Score"]}
             for k, v in results.items()}
        )
        .T
        .sort_values("F1-Score", ascending=False)
    )
    logger.info(f"\nModel Comparison (sorted by F1-Score):\n{summary.to_string()}")

    return results
