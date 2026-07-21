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
    parser.add_argument("--max-valid", type=int, default=0, help="Maximum validation rows for threshold tuning; 0 uses full valid.")
    parser.add_argument("--max-test", type=int, default=0, help="Maximum test rows; 0 uses full test.")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum transformer training epochs.")
    parser.add_argument(
        "--selection-metric",
        choices=["f1", "roc_auc"],
        default="f1",
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


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "test_size": int(len(y_true)),
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y_true, (scores >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def run_logistic_model(
    name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_metadata: bool,
    output_dir: Path,
) -> dict[str, float]:
    assistant, _, _ = RiskAssistant.train_with_split(train_df, valid_df, include_metadata=include_metadata)
    valid_predictions = assistant.predict_dataframe(valid_df)
    test_predictions = assistant.predict_dataframe(test_df)
    threshold = choose_threshold(
        valid_predictions["label"].astype(int).to_numpy(),
        valid_predictions["risk_score"].to_numpy(),
    )
    metrics = metrics_from_scores(
        test_predictions["label"].astype(int).to_numpy(),
        test_predictions["risk_score"].to_numpy(),
        threshold,
    )
    metrics.update(
        {
            "model": name,
            "feature_set": "text_plus_metadata" if include_metadata else "text_only",
            "train_size": int(len(train_df)),
            "valid_size": int(len(valid_df)),
        }
    )
    test_predictions["prediction_threshold"] = threshold
    test_predictions["prediction"] = (test_predictions["risk_score"] >= threshold).astype(int)
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
    counts = labels.astype(int).value_counts().reindex([0, 1]).fillna(1).to_numpy(dtype=np.float32)
    weights = counts.sum() / (2.0 * counts)
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
def transformer_scores(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)[:, 1]
        labels.extend(batch["labels"].numpy().tolist())
        scores.extend(probabilities.cpu().numpy().tolist())
    return np.array(labels), np.array(scores)


def run_transformer_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float]:
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
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
        valid_labels, valid_scores = transformer_scores(model, valid_loader, device)
        threshold = choose_threshold(valid_labels, valid_scores)
        valid_metrics = metrics_from_scores(valid_labels, valid_scores, threshold)
        selected_score = valid_metrics[args.selection_metric]
        training_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss,
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_precision": valid_metrics["precision"],
                "valid_recall": valid_metrics["recall"],
                "valid_f1": valid_metrics["f1"],
                "valid_roc_auc": valid_metrics["roc_auc"],
                "valid_threshold": valid_metrics["threshold"],
            }
        )
        if selected_score > best_score:
            best_score = selected_score
            best_epoch = epoch + 1
            best_valid_metrics = valid_metrics
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(
            f"Transformer epoch {epoch + 1}/{args.epochs} | "
            f"loss={loss:.4f} | valid_f1={valid_metrics['f1']:.4f} | "
            f"valid_auc={valid_metrics['roc_auc']:.4f}"
        )

    model.load_state_dict(best_state)
    model.to(device)
    if best_valid_metrics is None:
        raise RuntimeError("Transformer training did not produce validation metrics.")
    threshold = best_valid_metrics["threshold"]
    test_labels, test_scores = transformer_scores(model, test_loader, device)
    metrics = metrics_from_scores(test_labels, test_scores, threshold)
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

    predictions = test_df.copy()
    predictions["transformer_risk_score"] = test_scores
    predictions["prediction_threshold"] = threshold
    predictions["prediction"] = (test_scores >= threshold).astype(int)
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
        ),
        "tfidf_metadata_enhanced": run_logistic_model(
            "tfidf_metadata_enhanced",
            train_df,
            valid_df,
            test_df,
            include_metadata=True,
            output_dir=args.output,
        ),
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
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "threshold",
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
