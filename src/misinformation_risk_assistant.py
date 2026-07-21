from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


EMOTIONAL_TERMS = {
    "breaking",
    "shocking",
    "secret",
    "urgent",
    "panic",
    "wake up",
    "toxic",
    "miracle",
    "cure",
    "guarantees",
    "guaranteed",
    "deleted",
    "suppressed",
    "hiding",
    "hate",
    "share now",
}

UNSUPPORTED_MARKERS = {
    "everyone knows",
    "anonymous",
    "insider",
    "they are deleting",
    "mainstream media refuses",
    "doctors do not want",
    "doctors don't want",
    "no one wants you to see",
    "proves",
    "secret document",
    "forwarded message",
    "screenshot says",
}

OFFICIAL_DOMAINS = {
    "health.gov",
    "nasa.gov",
    "police.gov",
    "transport.gov",
    "weather.gov",
    "education.gov",
    "centralbank.gov",
    "fire.gov",
    "legislature.gov",
    "environment.gov",
}

RELIABLE_DOMAINS = {
    "university.edu",
    "hospital.org",
    "library.org",
    "manufacturer.com",
    "school.edu",
    "consumer-council.org",
    "medicaljournal.org",
    "supermarket.com",
}

LOW_CREDIBILITY_SUFFIXES = (".info", ".biz")
LOW_CREDIBILITY_KEYWORDS = ("rumor", "viral", "secret", "panic", "conspiracy", "leak", "truth")


@dataclass
class RiskAssistant:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    numeric_columns: list[str]
    use_metadata: bool = True

    @classmethod
    def train(cls, df: pd.DataFrame) -> tuple["RiskAssistant", dict[str, float], pd.DataFrame]:
        required = {"text", "label"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        work = df.copy()
        work["label"] = work["label"].astype(int)

        train_df, test_df = train_test_split(
            work,
            test_size=0.28,
            random_state=42,
            stratify=work["label"],
        )
        return cls.train_with_split(train_df, test_df)

    @classmethod
    def train_with_split(
        cls, train_df: pd.DataFrame, test_df: pd.DataFrame, include_metadata: bool = True
    ) -> tuple["RiskAssistant", dict[str, float], pd.DataFrame]:
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["label"] = train_df["label"].astype(int)
        test_df["label"] = test_df["label"].astype(int)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=500,
            stop_words="english",
        )
        x_train_text = vectorizer.fit_transform(train_df["text"])
        x_test_text = vectorizer.transform(test_df["text"])

        numeric_columns: list[str] = []
        if include_metadata:
            numeric_train = build_numeric_features(train_df)
            numeric_test = build_numeric_features(test_df)
            numeric_columns = numeric_train.columns.tolist()
            x_train = sparse.hstack([x_train_text, sparse.csr_matrix(numeric_train.values)])
            x_test = sparse.hstack([x_test_text, sparse.csr_matrix(numeric_test.values)])
        else:
            x_train = x_train_text
            x_test = x_test_text

        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        classifier.fit(x_train, train_df["label"])

        prob = classifier.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_df["label"], pred, labels=[0, 1]).ravel()
        metrics = {
            "accuracy": float(accuracy_score(test_df["label"], pred)),
            "precision": float(precision_score(test_df["label"], pred, zero_division=0)),
            "recall": float(recall_score(test_df["label"], pred, zero_division=0)),
            "f1": float(f1_score(test_df["label"], pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_df["label"], prob)),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "test_size": int(len(test_df)),
            "train_size": int(len(train_df)),
            "feature_set": "text_plus_metadata" if include_metadata else "text_only",
        }

        assistant = cls(vectorizer, classifier, numeric_columns, include_metadata)
        test_results = test_df.copy()
        test_results["risk_score"] = prob
        test_results["prediction"] = pred
        return assistant, metrics, test_results

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        text_features = self.vectorizer.transform(df["text"].fillna(""))
        if self.use_metadata:
            numeric_features = build_numeric_features(df)[self.numeric_columns]
            x = sparse.hstack([text_features, sparse.csr_matrix(numeric_features.values)])
        else:
            x = text_features
        scores = self.classifier.predict_proba(x)[:, 1]

        out = df.copy()
        out["risk_score"] = scores
        out["risk_level"] = [risk_level(score) for score in scores]
        out["explanation"] = [explain_row(row, score) for (_, row), score in zip(df.iterrows(), scores)]
        return out

    def predict_single(
        self,
        text: str,
        source_domain: str = "unknown",
        account_age_days: int = 30,
        follower_count: int = 100,
        following_count: int = 100,
        repost_count_1h: int = 0,
        distinct_accounts_1h: int = 1,
        has_link: int = 1,
    ) -> dict[str, object]:
        row = pd.DataFrame(
            [
                {
                    "post_id": "custom",
                    "text": text,
                    "source_domain": source_domain,
                    "account_age_days": account_age_days,
                    "follower_count": follower_count,
                    "following_count": following_count,
                    "repost_count_1h": repost_count_1h,
                    "distinct_accounts_1h": distinct_accounts_1h,
                    "has_link": has_link,
                }
            ]
        )
        result = self.predict_dataframe(row).iloc[0]
        return {
            "risk_score": float(result["risk_score"]),
            "risk_level": str(result["risk_level"]),
            "explanation": str(result["explanation"]),
        }


def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        domain = str(row.get("source_domain", "unknown")).lower().strip()
        account_age = float(row.get("account_age_days", 30) or 0)
        followers = float(row.get("follower_count", 0) or 0)
        following = float(row.get("following_count", 0) or 0)
        reposts = float(row.get("repost_count_1h", 0) or 0)
        distinct = float(row.get("distinct_accounts_1h", 1) or 1)
        has_link = float(row.get("has_link", 1) or 0)
        barely_true_counts = float(row.get("barely_true_counts", 0) or 0)
        false_counts = float(row.get("false_counts", 0) or 0)
        half_true_counts = float(row.get("half_true_counts", 0) or 0)
        mostly_true_counts = float(row.get("mostly_true_counts", 0) or 0)
        pants_on_fire_counts = float(row.get("pants_on_fire_counts", 0) or 0)
        speaker_history_total = (
            barely_true_counts
            + false_counts
            + half_true_counts
            + mostly_true_counts
            + pants_on_fire_counts
        )
        speaker_false_ratio = (barely_true_counts + false_counts + pants_on_fire_counts) / max(
            speaker_history_total, 1.0
        )

        emotional_count = count_terms(text, EMOTIONAL_TERMS)
        unsupported_count = count_terms(text, UNSUPPORTED_MARKERS)
        exclamation_count = text.count("!")
        question_count = text.count("?")
        uppercase_ratio = uppercase_letter_ratio(text)
        source_risk = source_risk_score(domain)
        repost_velocity = math.log1p(reposts)
        coordination_ratio = reposts / max(distinct, 1.0)
        new_account = 1.0 if account_age < 30 else 0.0
        follower_following_ratio = math.log1p((following + 1.0) / (followers + 1.0))

        records.append(
            {
                "emotional_count": emotional_count,
                "unsupported_count": unsupported_count,
                "exclamation_count": exclamation_count,
                "question_count": question_count,
                "uppercase_ratio": uppercase_ratio,
                "source_risk": source_risk,
                "repost_velocity": repost_velocity,
                "coordination_ratio": coordination_ratio,
                "new_account": new_account,
                "follower_following_ratio": follower_following_ratio,
                "missing_link": 1.0 - has_link,
                "speaker_history_total": math.log1p(speaker_history_total),
                "speaker_false_ratio": speaker_false_ratio,
            }
        )
    return pd.DataFrame(records, index=df.index)


def count_terms(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def uppercase_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.isupper()) / len(letters)


def source_risk_score(domain: str) -> float:
    if domain in OFFICIAL_DOMAINS:
        return 0.05
    if domain in RELIABLE_DOMAINS:
        return 0.18
    if domain == "unknown" or any(domain.endswith(suffix) for suffix in LOW_CREDIBILITY_SUFFIXES):
        return 0.90
    if any(keyword in domain for keyword in LOW_CREDIBILITY_KEYWORDS):
        return 0.78
    return 0.45


def risk_level(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def explain_row(row: pd.Series, score: float) -> str:
    text = str(row.get("text", ""))
    domain = str(row.get("source_domain", "")).lower().strip()
    reasons: list[str] = []

    emotional_hits = matching_terms(text, EMOTIONAL_TERMS)
    unsupported_hits = matching_terms(text, UNSUPPORTED_MARKERS)
    if emotional_hits:
        reasons.append("emotional wording: " + ", ".join(emotional_hits[:4]))
    if unsupported_hits:
        reasons.append("unsupported claim marker: " + ", ".join(unsupported_hits[:3]))
    if domain and source_risk_score(domain) >= 0.70:
        reasons.append(f"weak or unknown source: {domain}")
    speaker_history_total = sum(
        float(row.get(column, 0) or 0)
        for column in [
            "barely_true_counts",
            "false_counts",
            "half_true_counts",
            "mostly_true_counts",
            "pants_on_fire_counts",
        ]
    )
    speaker_false_history = sum(
        float(row.get(column, 0) or 0)
        for column in ["barely_true_counts", "false_counts", "pants_on_fire_counts"]
    )
    if speaker_history_total >= 5 and speaker_false_history / speaker_history_total >= 0.55:
        reasons.append("speaker has a high-risk fact-checking history")
    if float(row.get("account_age_days", 30) or 0) < 30:
        reasons.append("very new account")
    reposts = float(row.get("repost_count_1h", 0) or 0)
    distinct = float(row.get("distinct_accounts_1h", 1) or 1)
    if reposts >= 400 and reposts / max(distinct, 1.0) >= 10:
        reasons.append("fast reposting with possible coordination")
    if text.count("!") >= 2:
        reasons.append("heavy punctuation")

    if not reasons:
        reasons.append("no strong warning signal in the available metadata")

    return f"{risk_level(score)} risk ({score:.2f}). " + "; ".join(reasons)


def matching_terms(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return sorted(term for term in terms if term in lowered)


def save_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_risk_chart(predictions: pd.DataFrame, path: Path, title: str = "Risk Level Distribution") -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = predictions["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    ax.bar(counts.index, counts.values, color=["#2e7d32", "#f9a825", "#c62828"])
    ax.set_title(title)
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Number of posts")
    ax.set_ylim(0, max(counts.values) + 3)
    for index, value in enumerate(counts.values):
        ax.text(index, value + 0.2, str(int(value)), ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


LIAR_COLUMNS = [
    "id",
    "original_label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

HIGH_RISK_LIAR_LABELS = {"false", "barely-true", "pants-fire"}


def read_liar_split(zip_path: Path, split: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{split}.tsv") as handle:
            df = pd.read_csv(handle, sep="\t", names=LIAR_COLUMNS, quoting=3, encoding="utf-8")
    return normalize_liar_dataframe(df, split)


def normalize_liar_dataframe(df: pd.DataFrame, split: str) -> pd.DataFrame:
    work = df.copy()
    for column in [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
    ]:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)

    work["post_id"] = split + "_" + work["id"].astype(str)
    work["text"] = (
        work["statement"].fillna("")
        + " Subject: "
        + work["subject"].fillna("")
        + " Context: "
        + work["context"].fillna("")
    )
    work["source_domain"] = ""
    work["account_age_days"] = 365
    work["follower_count"] = 0
    work["following_count"] = 0
    work["repost_count_1h"] = 0
    work["distinct_accounts_1h"] = 1
    work["has_link"] = 0
    work["label"] = work["original_label"].isin(HIGH_RISK_LIAR_LABELS).astype(int)
    return work


def load_liar_dataset(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = read_liar_split(zip_path, "train")
    valid = read_liar_split(zip_path, "valid")
    test = read_liar_split(zip_path, "test")
    return train, valid, test
