from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


PROJECT_DIR = Path(__file__).resolve().parent
BASELINE_PREDICTIONS = PROJECT_DIR / "outputs" / "predictions.csv"
TRANSFORMER_PREDICTIONS = PROJECT_DIR / "transformer_outputs" / "transformer_predictions.csv"
BASELINE_METRICS = PROJECT_DIR / "outputs" / "evaluation_metrics.json"
TRANSFORMER_METRICS = PROJECT_DIR / "transformer_outputs" / "transformer_metrics.json"
FAIR_OUTPUTS = PROJECT_DIR / "fair_outputs"
TFIDF_TEXT_PREDICTIONS = FAIR_OUTPUTS / "tfidf_text_only_predictions.csv"
TRANSFORMER_TEXT_PREDICTIONS = FAIR_OUTPUTS / "transformer_text_only_predictions.csv"
FAIR_METRICS = FAIR_OUTPUTS / "metrics.json"
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
    order = [0, 1, 2]
    counts = df["prediction"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(["Low", "Medium", "High"], counts.values, color=[COLORS["green"], COLORS["amber"], COLORS["red"]])
    ax.set_title("Predicted Three-Level Risk Distribution")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Number of claims")
    style_axes(ax)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6, f"{int(bar.get_height())}", ha="center", fontsize=9)
    save(fig, "risk_distribution.png")


def plot_confusion_matrix(df: pd.DataFrame, score_column: str, name: str, title: str) -> None:
    y_true = df["label"].astype(int).to_numpy()
    if "prediction" in df.columns:
        y_pred = df["prediction"].astype(int).to_numpy()
    else:
        probability_columns = ["prob_low", "prob_medium", "prob_high"]
        if score_column.startswith("transformer"):
            probability_columns = ["transformer_prob_low", "transformer_prob_medium", "transformer_prob_high"]
        y_pred = np.argmax(df[probability_columns].to_numpy(), axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1, 2], labels=["Low", "Medium", "High"])
    ax.set_yticks([0, 1, 2], labels=["Low", "Medium", "High"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=COLORS["ink"], fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, name)


def plot_roc_curves(tfidf_text: pd.DataFrame, transformer: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for df, probability_columns, label, color in [
        (tfidf_text, ["prob_low", "prob_medium", "prob_high"], "TF-IDF text-only", COLORS["blue"]),
        (transformer, ["transformer_prob_low", "transformer_prob_medium", "transformer_prob_high"], "BERT-base text-only", COLORS["purple"]),
    ]:
        y_true = label_binarize(df["label"].astype(int), classes=[0, 1, 2])
        probabilities = df[probability_columns].to_numpy()
        mean_fpr = np.linspace(0, 1, 101)
        tprs = []
        aucs = []
        for class_index in range(3):
            fpr, tpr, _ = roc_curve(y_true[:, class_index], probabilities[:, class_index])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        ax.plot(mean_fpr, mean_tpr, label=f"{label} (macro AUC={np.mean(aucs):.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["muted"], linewidth=1)
    ax.set_title("One-vs-Rest ROC Curves on LIAR Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    save(fig, "roc_curves.png")


def plot_model_comparison(metrics: dict[str, dict[str, float]]) -> None:
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]
    x = np.arange(len(metric_names))
    width = 0.32
    models = [
        ("tfidf_text_only", "TF-IDF text-only", COLORS["blue"]),
        ("transformer_text_only", "BERT-base text-only", COLORS["purple"]),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    offsets = [-width / 2, width / 2]
    for offset, (key, label, color) in zip(offsets, models):
        values = [metrics[key][name] for name in metric_names]
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_xticks(x, labels=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro AUC"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(frameon=False)
    style_axes(ax)
    save(fig, "model_comparison.png")


def plot_score_density(tfidf_text: pd.DataFrame, transformer: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.hist(tfidf_text["risk_score"], bins=24, alpha=0.58, label="TF-IDF text-only", color=COLORS["blue"], density=True)
    ax.hist(transformer["transformer_risk_score"], bins=24, alpha=0.50, label="BERT-base text-only", color=COLORS["purple"], density=True)
    ax.set_title("Risk-Score Distribution")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    style_axes(ax)
    save(fig, "score_distribution.png")


def main() -> None:
    tfidf_text = pd.read_csv(TFIDF_TEXT_PREDICTIONS)
    transformer = pd.read_csv(TRANSFORMER_TEXT_PREDICTIONS)
    metrics = load_json(FAIR_METRICS)

    plot_label_distribution(tfidf_text)
    plot_risk_distribution(transformer)
    plot_confusion_matrix(tfidf_text, "risk_score", "baseline_confusion_matrix.png", "TF-IDF Text-Only Confusion Matrix")
    plot_confusion_matrix(transformer, "transformer_risk_score", "transformer_confusion_matrix.png", "BERT-Base Text-Only Confusion Matrix")
    plot_roc_curves(tfidf_text, transformer)
    plot_model_comparison(metrics)
    plot_score_density(tfidf_text, transformer)
    print(f"Saved figures to {FIGURE_DIR} and {DOCS_ASSET_DIR}")


if __name__ == "__main__":
    main()
