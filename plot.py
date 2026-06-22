"""
plot.py
-------
RiskLens

Visualization utilities for model comparison and evaluation.
All plots can optionally be saved to disk instead of (or in addition to)
being displayed interactively.
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

# Consistent style across all plots
sns.set_theme(style="whitegrid", palette="Blues_d")


def plot_results(
    results: dict[str, dict],
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Bar chart comparing Accuracy and F1-Score across all trained models,
    sorted by F1-Score descending.

    Parameters
    ----------
    results   : Output dict from train_models() (name → metrics dict).
    save_path : File path to save the figure; None to skip saving.
    show      : Whether to display the plot interactively.
    """
    metrics_df = (
        pd.DataFrame(
            {k: {"Accuracy": v["Accuracy"], "F1-Score": v["F1-Score"]}
             for k, v in results.items()}
        )
        .T
        .sort_values("F1-Score", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df[["Accuracy", "F1-Score"]].plot(
        kind="bar", ax=ax, colormap="Blues", edgecolor="black", width=0.6
    )
    ax.set_title("Model Comparison — Credit Card Default Prediction", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    _save_and_show(fig, save_path, show)


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Heatmap of the confusion matrix for a single fitted model.

    Parameters
    ----------
    model      : Fitted sklearn-compatible estimator.
    X_test     : Test feature array.
    y_test     : True test labels.
    model_name : Display name used in the plot title.
    save_path  : File path to save the figure; None to skip saving.
    show       : Whether to display the plot interactively.
    """
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
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    _save_and_show(fig, save_path, show)


def plot_all_confusion_matrices(
    results: dict[str, dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str | Path = "images",
    show: bool = True,
) -> None:
    """
    Convenience wrapper: plots a confusion matrix for every model in results.

    Parameters
    ----------
    results  : Output dict from train_models().
    X_test   : Test feature array.
    y_test   : True test labels.
    save_dir : Directory where individual confusion matrix PNGs are saved.
    show     : Whether to display plots interactively.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, info in results.items():
        safe_name = name.replace(" ", "_").lower()
        save_path = save_dir / f"cm_{safe_name}.png"
        plot_confusion_matrix(
            model=info["model"],
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            save_path=save_path,
            show=show,
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def _save_and_show(
    fig: plt.Figure,
    save_path: str | Path | None,
    show: bool,
) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {path}")
    if show:
        plt.show()
    plt.close(fig)
