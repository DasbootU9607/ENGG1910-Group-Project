from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


PROJECT_DIR = Path(__file__).resolve().parent
BASELINE_PREDICTIONS = PROJECT_DIR / "outputs" / "predictions.csv"
TRANSFORMER_PREDICTIONS = PROJECT_DIR / "transformer_outputs" / "transformer_predictions.csv"
BASELINE_METRICS = PROJECT_DIR / "outputs" / "evaluation_metrics.json"
TRANSFORMER_METRICS = PROJECT_DIR / "transformer_outputs" / "transformer_metrics.json"
FIGURE_DIR = PROJECT_DIR / "paper" / "figures"
DOCS_ASSET_DIR = PROJECT_DIR / "docs" / "assets"


COLORS = {
    "ink": "#1f2937",
    "muted": "#6b7280",
    "blue": "#2563eb",
    "green": "#15803d",
    "amber": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "grid": "#e5e7eb",
}


def load_json(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for target_dir in [FIGURE_DIR, DOCS_ASSET_DIR]:
        fig.savefig(target_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(df: pd.DataFrame) -> None:
    order = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    counts = df["original_label"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    bars = ax.bar(counts.index, counts.values, color=[COLORS["red"], COLORS["red"], COLORS["amber"], COLORS["amber"], COLORS["green"], COLORS["green"]])
    ax.set_title("LIAR Test-Set Label Distribution")
    ax.set_xlabel("Original LIAR label")
    ax.set_ylabel("Number of claims")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f"{int(bar.get_height())}", ha="center", fontsize=9)
    save(fig, "label_distribution.png")


def plot_risk_distribution(df: pd.DataFrame) -> None:
    order = ["Low", "Medium", "High"]
    counts = df["risk_level"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(order, counts.values, color=[COLORS["green"], COLORS["amber"], COLORS["red"]])
    ax.set_title("Predicted Risk-Level Distribution")
    ax.set_xlabel("Predicted risk level")
    ax.set_ylabel("Number of claims")
    style_axes(ax)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6, f"{int(bar.get_height())}", ha="center", fontsize=9)
    save(fig, "risk_distribution.png")


def plot_confusion_matrix(df: pd.DataFrame, score_column: str, name: str, title: str) -> None:
    y_true = df["label"].astype(int).to_numpy()
    y_pred = (df[score_column].to_numpy() >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1], labels=["Lower risk", "Higher risk"])
    ax.set_yticks([0, 1], labels=["Lower risk", "Higher risk"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=COLORS["ink"], fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, name)


def plot_roc_curves(baseline: pd.DataFrame, transformer: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for df, score_column, label, color in [
        (baseline, "risk_score", "TF-IDF + Logistic Regression", COLORS["blue"]),
        (transformer, "transformer_risk_score", "BERT-tiny Transformer", COLORS["purple"]),
    ]:
        fpr, tpr, _ = roc_curve(df["label"].astype(int), df[score_column])
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["muted"], linewidth=1)
    ax.set_title("ROC Curves on LIAR Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    save(fig, "roc_curves.png")


def plot_model_comparison(baseline_metrics: dict[str, float], transformer_metrics: dict[str, float]) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metric_names))
    width = 0.36
    baseline_values = [baseline_metrics[name] for name in metric_names]
    transformer_values = [transformer_metrics[name] for name in metric_names]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - width / 2, baseline_values, width, label="TF-IDF + LR", color=COLORS["blue"])
    ax.bar(x + width / 2, transformer_values, width, label="BERT-tiny", color=COLORS["purple"])
    ax.set_xticks(x, labels=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(frameon=False)
    style_axes(ax)
    save(fig, "model_comparison.png")


def plot_score_density(baseline: pd.DataFrame, transformer: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.hist(baseline["risk_score"], bins=24, alpha=0.68, label="TF-IDF + LR", color=COLORS["blue"], density=True)
    ax.hist(transformer["transformer_risk_score"], bins=24, alpha=0.58, label="BERT-tiny", color=COLORS["purple"], density=True)
    ax.axvline(0.4, color=COLORS["amber"], linestyle="--", linewidth=1)
    ax.axvline(0.7, color=COLORS["red"], linestyle="--", linewidth=1)
    ax.set_title("Risk-Score Distribution")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    style_axes(ax)
    save(fig, "score_distribution.png")


def main() -> None:
    baseline = pd.read_csv(BASELINE_PREDICTIONS)
    transformer = pd.read_csv(TRANSFORMER_PREDICTIONS)
    baseline_metrics = load_json(BASELINE_METRICS)
    transformer_metrics = load_json(TRANSFORMER_METRICS)

    plot_label_distribution(baseline)
    plot_risk_distribution(baseline)
    plot_confusion_matrix(baseline, "risk_score", "baseline_confusion_matrix.png", "TF-IDF + LR Confusion Matrix")
    plot_confusion_matrix(transformer, "transformer_risk_score", "transformer_confusion_matrix.png", "BERT-tiny Confusion Matrix")
    plot_roc_curves(baseline, transformer)
    plot_model_comparison(baseline_metrics, transformer_metrics)
    plot_score_density(baseline, transformer)
    print(f"Saved figures to {FIGURE_DIR} and {DOCS_ASSET_DIR}")


if __name__ == "__main__":
    main()
