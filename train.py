"""
train.py
RiskLens
"""

import logging
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=0.01,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    ),
    "SVM": SVC(
        C=1,
        kernel="rbf",
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    ),
    "LightGBM": LGBMClassifier(
        random_state=42,
        verbose=-1,
    ),
}


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    models=None,
):
    """Train and evaluate all models."""

    if models is None:
        models = MODEL_REGISTRY

    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        results[name] = {
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "model": model,
        }

        logger.info(
            "\n"
            + classification_report(
                y_test,
                y_pred,
                target_names=["Non-Defaulter", "Defaulter"],
            )
        )

    summary = (
        pd.DataFrame(
            {
                k: {
                    "Accuracy": v["Accuracy"],
                    "F1-Score": v["F1-Score"],
                }
                for k, v in results.items()
            }
        )
        .T
        .sort_values("F1-Score", ascending=False)
    )

    logger.info(f"\nModel Comparison:\n{summary}")

    return results
