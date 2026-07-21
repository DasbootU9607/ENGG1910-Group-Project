# AI-Powered Misinformation Risk Assistant

This folder contains a runnable ENGG1910 course project prototype.

The system estimates whether a social media post has high misinformation risk. It does not directly decide whether a post is true or false. It gives a risk score and short explanations, such as emotional wording, unsupported claims, weak source information, or suspicious repost patterns.

## How to Run

From this folder:

```powershell
python run_demo.py
```

Or from the course root:

```powershell
python "group project\misinformation_risk_assistant\run_demo.py"
```

The script will create:

- `outputs/evaluation_metrics.json`
- `outputs/predictions.csv`
- `outputs/risk_distribution.png`

## Dependencies

The current environment already has the required packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- datasets, only if loading Hugging Face datasets

If another computer is missing packages:

```powershell
pip install -r requirements.txt
```

## Notes for the Report

The demo dataset is synthetic and is meant to prove the system pipeline. In the report, describe it as a prototype experiment rather than a real-world performance claim.

For a stronger final version, replace `data/demo_posts.csv` with a public misinformation dataset such as LIAR, FakeNewsNet, or CoAID, then keep the same input columns.

## LIAR Dataset Experiment

The project also supports the LIAR political fact-checking dataset. This repository version uses the original LIAR TSV files downloaded from the URL referenced by the Hugging Face dataset script.

```powershell
python run_demo.py --dataset liar
```

If `data/liar_dataset.zip` is missing, the script downloads it automatically from the original LIAR dataset URL.

The script maps LIAR labels into binary risk labels:

- High risk: `false`, `barely-true`, `pants-fire`
- Lower risk: `half-true`, `mostly-true`, `true`

## Transformer Experiment

The baseline model uses TF-IDF and logistic regression. A transformer-based experiment is also included:

```powershell
python run_transformer_demo.py
```

By default, it fine-tunes `prajjwal1/bert-tiny` for a CPU-friendly demonstration. To try a larger DistilBERT model:

```powershell
python run_transformer_demo.py --model distilbert-base-uncased --max-train 4000 --epochs 1
```

Transformer outputs are saved under `transformer_outputs/`.

## Optional Single Post Test

```powershell
python run_demo.py --post "SHOCKING secret cure that doctors do not want you to know!!!" --source "unknown-blog.info" --account-age 3 --followers 12 --following 900 --reposts 420 --distinct-reposters 35
```
