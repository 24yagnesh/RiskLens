"""
plot.py
RiskLens
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="Blues_d")


def plot_results(
    results,
    save_path=None,
    show=True,
):
    """Compare model performance."""

    metrics_df = (
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

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_df[["Accuracy", "F1-Score"]].plot(
        kind="bar",
        ax=ax,
        colormap="Blues",
        edgecolor="black",
        width=0.6,
    )

    ax.set_title("Model Comparison", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    _save_and_show(fig, save_path, show)


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name="Model",
    save_path=None,
    show=True,
):
    """Plot confusion matrix."""

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Defaulter", "Defaulter"],
        yticklabels=["Non-Defaulter", "Defaulter"],
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()

    _save_and_show(fig, save_path, show)


def plot_all_confusion_matrices(
    results,
    X_test,
    y_test,
    save_dir="images",
    show=True,
):
    """Generate confusion matrices for all models."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, info in results.items():
        save_path = save_dir / f"cm_{name.replace(' ', '_').lower()}.png"

        plot_confusion_matrix(
            model=info["model"],
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            save_path=save_path,
            show=show,
        )


def _save_and_show(fig, save_path, show):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        logger.info(f"Figure saved to {path}")

    if show:
        plt.show()

    plt.close(fig)
