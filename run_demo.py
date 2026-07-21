from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from src.misinformation_risk_assistant import RiskAssistant, load_liar_dataset, save_json, save_risk_chart


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "demo_posts.csv"
DEFAULT_LIAR_ZIP = PROJECT_DIR / "data" / "liar_dataset.zip"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs"
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the misinformation risk assistant demo.")
    parser.add_argument(
        "--dataset",
        choices=["demo", "liar"],
        default="demo",
        help="Use the synthetic demo CSV or the LIAR dataset zip.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA, help="CSV dataset path.")
    parser.add_argument("--liar-zip", type=Path, default=DEFAULT_LIAR_ZIP, help="LIAR dataset zip path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder.")
    parser.add_argument("--post", type=str, default=None, help="Optional custom post text to score.")
    parser.add_argument("--source", type=str, default="unknown", help="Source domain for custom post.")
    parser.add_argument("--account-age", type=int, default=30, help="Account age in days for custom post.")
    parser.add_argument("--followers", type=int, default=100, help="Follower count for custom post.")
    parser.add_argument("--following", type=int, default=100, help="Following count for custom post.")
    parser.add_argument("--reposts", type=int, default=0, help="Reposts in the first hour for custom post.")
    parser.add_argument("--distinct-reposters", type=int, default=1, help="Distinct reposting accounts in first hour.")
    parser.add_argument("--no-link", action="store_true", help="Set custom post has_link to 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "liar":
        ensure_liar_zip(args.liar_zip)
        train_df, valid_df, test_df = load_liar_dataset(args.liar_zip)
        train_full = pd.concat([train_df, valid_df], ignore_index=True)
        assistant, metrics, test_results = RiskAssistant.train_with_split(train_full, test_df)
        predictions = assistant.predict_dataframe(test_df)
        dataset_label = f"LIAR dataset: {args.liar_zip}"
        chart_title = "Risk Level Distribution on LIAR Test Set"
    else:
        df = pd.read_csv(args.input)
        assistant, metrics, test_results = RiskAssistant.train(df)
        predictions = assistant.predict_dataframe(df)
        dataset_label = f"Demo dataset: {args.input}"
        chart_title = "Risk Level Distribution on Demo Posts"

    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "evaluation_metrics.json", metrics)
    predictions.to_csv(args.output / "predictions.csv", index=False, encoding="utf-8-sig")
    chart_saved = save_risk_chart(predictions, args.output / "risk_distribution.png", title=chart_title)

    print("Misinformation risk assistant demo")
    print(f"Dataset: {dataset_label}")
    print(f"Train size: {metrics['train_size']} | Test size: {metrics['test_size']}")
    print(
        "Metrics: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, "
        f"f1={metrics['f1']:.3f}, "
        f"roc_auc={metrics['roc_auc']:.3f}"
    )
    print(f"Saved predictions: {args.output / 'predictions.csv'}")
    print(f"Saved metrics: {args.output / 'evaluation_metrics.json'}")
    if chart_saved:
        print(f"Saved chart: {args.output / 'risk_distribution.png'}")
    else:
        print("Chart skipped because matplotlib is not installed.")

    print("\nExample explanations:")
    examples = predictions.sort_values("risk_score", ascending=False).head(3)
    for _, row in examples.iterrows():
        label_text = f" | original_label={row['original_label']}" if "original_label" in row else ""
        print(f"- post {row['post_id']}{label_text}: {row['explanation']}")

    if args.post:
        custom = assistant.predict_single(
            text=args.post,
            source_domain=args.source,
            account_age_days=args.account_age,
            follower_count=args.followers,
            following_count=args.following,
            repost_count_1h=args.reposts,
            distinct_accounts_1h=args.distinct_reposters,
            has_link=0 if args.no_link else 1,
        )
        print("\nCustom post result:")
        print(f"Risk score: {custom['risk_score']:.3f}")
        print(f"Risk level: {custom['risk_level']}")
        print(f"Explanation: {custom['explanation']}")


def ensure_liar_zip(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LIAR dataset to {path}")
    urllib.request.urlretrieve(LIAR_URL, path)


if __name__ == "__main__":
    main()
