from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from run_demo import DEFAULT_LIAR_ZIP, ensure_liar_zip  # noqa: E402
from src.misinformation_risk_assistant import RiskAssistant, load_liar_dataset  # noqa: E402


class ClaimDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fair LIAR experiments with matched splits and features.")
    parser.add_argument("--liar-zip", type=Path, default=DEFAULT_LIAR_ZIP, help="LIAR dataset zip path.")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "fair_outputs", help="Output folder.")
    parser.add_argument("--skip-transformer", action="store_true", help="Only run the logistic-regression baselines.")
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Transformer checkpoint for the fair text-only model.",
    )
    parser.add_argument("--max-train", type=int, default=0, help="Maximum transformer training rows; 0 uses full train.")
    parser.add_argument("--max-valid", type=int, default=0, help="Maximum validation rows for checkpoint selection; 0 uses full valid.")
    parser.add_argument("--max-test", type=int, default=0, help="Maximum test rows; 0 uses full test.")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum transformer training epochs.")
    parser.add_argument(
        "--selection-metric",
        choices=["f1_macro", "f1_weighted", "roc_auc_ovr_macro"],
        default="f1_macro",
        help="Validation metric used to select the transformer checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Transformer batch size.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max sequence length.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. Use cuda after installing a CUDA-enabled PyTorch build.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_frame(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    sampled_parts = []
    for _, part in df.groupby("label"):
        n_rows = max(1, round(max_rows * len(part) / len(df)))
        sampled_parts.append(part.sample(n=n_rows, random_state=seed))
    return pd.concat(sampled_parts).sample(frac=1.0, random_state=seed).head(max_rows).reset_index(drop=True)


def texts(df: pd.DataFrame) -> list[str]:
    return df["text"].fillna("").astype(str).tolist()


CLASS_LABELS = [0, 1, 2]
RISK_LEVELS = {0: "low", 1: "medium", 2: "high"}


def metrics_from_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=CLASS_LABELS).tolist(),
        "test_size": int(len(y_true)),
    }


def add_probability_columns(df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["prob_low"] = probabilities[:, 0]
    out["prob_medium"] = probabilities[:, 1]
    out["prob_high"] = probabilities[:, 2]
    out["risk_score"] = out["prob_medium"] * 0.5 + out["prob_high"]
    out["prediction"] = np.argmax(probabilities, axis=1)
    out["predicted_risk_level"] = out["prediction"].map({0: "Low", 1: "Medium", 2: "High"})
    return out


def run_logistic_model(
    name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_metadata: bool,
    output_dir: Path,
) -> dict[str, float]:
    assistant, _, _ = RiskAssistant.train_with_split(train_df, valid_df, include_metadata=include_metadata)
    test_predictions = assistant.predict_dataframe(test_df)
    probabilities = test_predictions[["prob_low", "prob_medium", "prob_high"]].to_numpy()
    metrics = metrics_from_probabilities(
        test_predictions["label"].astype(int).to_numpy(),
        probabilities,
    )
    metrics.update(
        {
            "model": name,
            "feature_set": "text_plus_metadata" if include_metadata else "text_only",
            "train_size": int(len(train_df)),
            "valid_size": int(len(valid_df)),
        }
    )
    test_predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has CPU-only PyTorch. "
                "Install a CUDA-enabled torch build, then rerun with --device cuda."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weights(labels: pd.Series, device: torch.device) -> torch.Tensor:
    counts = labels.astype(int).value_counts().reindex(CLASS_LABELS).fillna(1).to_numpy(dtype=np.float32)
    weights = counts.sum() / (len(CLASS_LABELS) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_transformer_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def transformer_probabilities(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    probabilities: list[list[float]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_probabilities = torch.softmax(outputs.logits, dim=1)
        labels.extend(batch["labels"].numpy().tolist())
        probabilities.extend(batch_probabilities.cpu().numpy().tolist())
    return np.array(labels), np.array(probabilities)


def run_transformer_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float]:
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.to(device)

    train_dataset = ClaimDataset(texts(train_df), train_df["label"].astype(int).tolist(), tokenizer, args.max_length)
    valid_dataset = ClaimDataset(texts(valid_df), valid_df["label"].astype(int).tolist(), tokenizer, args.max_length)
    test_dataset = ClaimDataset(texts(test_df), test_df["label"].astype(int).tolist(), tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train_df["label"], device))

    training_log = []
    best_state = copy.deepcopy(model.state_dict())
    best_valid_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_score = -1.0
    for epoch in range(args.epochs):
        loss = train_transformer_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_labels, valid_probabilities = transformer_probabilities(model, valid_loader, device)
        valid_metrics = metrics_from_probabilities(valid_labels, valid_probabilities)
        selected_score = valid_metrics[args.selection_metric]
        training_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss,
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_precision_macro": valid_metrics["precision_macro"],
                "valid_recall_macro": valid_metrics["recall_macro"],
                "valid_f1_macro": valid_metrics["f1_macro"],
                "valid_f1_weighted": valid_metrics["f1_weighted"],
                "valid_roc_auc_ovr_macro": valid_metrics["roc_auc_ovr_macro"],
            }
        )
        if selected_score > best_score:
            best_score = selected_score
            best_epoch = epoch + 1
            best_valid_metrics = valid_metrics
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(
            f"Transformer epoch {epoch + 1}/{args.epochs} | "
            f"loss={loss:.4f} | valid_macro_f1={valid_metrics['f1_macro']:.4f} | "
            f"valid_auc={valid_metrics['roc_auc_ovr_macro']:.4f}"
        )

    model.load_state_dict(best_state)
    model.to(device)
    if best_valid_metrics is None:
        raise RuntimeError("Transformer training did not produce validation metrics.")
    test_labels, test_probabilities = transformer_probabilities(model, test_loader, device)
    metrics = metrics_from_probabilities(test_labels, test_probabilities)
    metrics.update(
        {
            "model": args.model,
            "feature_set": "text_only",
            "train_size": int(len(train_df)),
            "valid_size": int(len(valid_df)),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "learning_rate": float(args.learning_rate),
            "device": str(device),
            "best_epoch": int(best_epoch),
            "selection_metric": args.selection_metric,
            "best_validation_metrics": best_valid_metrics,
            "training_log": training_log,
            "class_weight": "balanced",
        }
    )

    predictions = add_probability_columns(test_df, test_probabilities)
    predictions = predictions.rename(
        columns={
            "prob_low": "transformer_prob_low",
            "prob_medium": "transformer_prob_medium",
            "prob_high": "transformer_prob_high",
            "risk_score": "transformer_risk_score",
        }
    )
    predictions.to_csv(output_dir / "transformer_text_only_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(training_log).to_csv(output_dir / "transformer_training_log.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_liar_zip(args.liar_zip)
    if not args.skip_transformer:
        resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = load_liar_dataset(args.liar_zip)
    transformer_train = sample_frame(train_df, args.max_train, args.seed)
    transformer_valid = sample_frame(valid_df, args.max_valid, args.seed)
    transformer_test = sample_frame(test_df, args.max_test, args.seed)

    results: dict[str, dict[str, float]] = {
        "tfidf_text_only": run_logistic_model(
            "tfidf_text_only",
            train_df,
            valid_df,
            test_df,
            include_metadata=False,
            output_dir=args.output,
        )
    }

    if not args.skip_transformer:
        results["transformer_text_only"] = run_transformer_model(
            transformer_train,
            transformer_valid,
            transformer_test,
            args,
            args.output,
        )

    summary = pd.DataFrame(results).T
    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
        "train_size",
        "valid_size",
        "best_epoch",
    ]
    summary_view = summary.reindex(columns=metric_columns)
    summary_view.to_csv(args.output / "summary.csv", encoding="utf-8-sig")
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Fair LIAR experiment")
    print(summary_view.round(4).to_string())
    print(f"Saved metrics: {args.output / 'metrics.json'}")
    print(f"Saved summary: {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
