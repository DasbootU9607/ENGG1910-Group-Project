from __future__ import annotations

import argparse
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

from src.misinformation_risk_assistant import load_liar_dataset  # noqa: E402
from run_demo import DEFAULT_LIAR_ZIP, ensure_liar_zip  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Fine-tune a transformer on LIAR risk labels.")
    parser.add_argument("--liar-zip", type=Path, default=DEFAULT_LIAR_ZIP, help="LIAR dataset zip path.")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "transformer_outputs", help="Output folder.")
    parser.add_argument(
        "--model",
        type=str,
        default="prajjwal1/bert-tiny",
        help="Hugging Face transformer checkpoint. Use distilbert-base-uncased for a larger model.",
    )
    parser.add_argument("--max-train", type=int, default=2000, help="Maximum training examples for CPU-friendly demo.")
    parser.add_argument("--max-test", type=int, default=1283, help="Maximum test examples.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max-length", type=int, default=96, help="Tokenizer max sequence length.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_frame(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    sampled_parts = []
    for _, part in df.groupby("label"):
        n_rows = max(1, round(max_rows * len(part) / len(df)))
        sampled_parts.append(part.sample(n=n_rows, random_state=seed))
    return pd.concat(sampled_parts).sample(frac=1.0, random_state=seed).head(max_rows).reset_index(drop=True)


def build_texts(df: pd.DataFrame) -> list[str]:
    return df["text"].fillna("").astype(str).tolist()


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        probabilities = torch.softmax(outputs.logits, dim=1)[:, 1]
        scores.extend(probabilities.cpu().numpy().tolist())
        labels.extend(batch["labels"].cpu().numpy().tolist())

    y_true = np.array(labels)
    y_score = np.array(scores)
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "test_size": int(len(y_true)),
    }
    return metrics, y_score


def risk_level(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_liar_zip(args.liar_zip)

    train_df, valid_df, test_df = load_liar_dataset(args.liar_zip)
    train_full = pd.concat([train_df, valid_df], ignore_index=True)
    train_sample = sample_frame(train_full, args.max_train, args.seed)
    test_sample = sample_frame(test_df, args.max_test, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = ClaimDataset(
        build_texts(train_sample),
        train_sample["label"].astype(int).tolist(),
        tokenizer,
        args.max_length,
    )
    test_dataset = ClaimDataset(
        build_texts(test_sample),
        test_sample["label"].astype(int).tolist(),
        tokenizer,
        args.max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={loss:.4f}")

    metrics, scores = evaluate(model, test_loader, device)
    metrics.update(
        {
            "model": args.model,
            "train_size": int(len(train_sample)),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "device": str(device),
            "training_loss": losses,
        }
    )

    predictions = test_sample.copy()
    predictions["transformer_risk_score"] = scores
    predictions["transformer_risk_level"] = [risk_level(score) for score in scores]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transformer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(args.output / "transformer_predictions.csv", index=False, encoding="utf-8-sig")

    print("Transformer misinformation risk demo")
    print(f"Model: {args.model}")
    print(f"Train size: {metrics['train_size']} | Test size: {metrics['test_size']} | Device: {device}")
    print(
        "Metrics: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, "
        f"f1={metrics['f1']:.3f}, "
        f"roc_auc={metrics['roc_auc']:.3f}"
    )
    print(f"Saved metrics: {args.output / 'transformer_metrics.json'}")
    print(f"Saved predictions: {args.output / 'transformer_predictions.csv'}")


if __name__ == "__main__":
    main()
