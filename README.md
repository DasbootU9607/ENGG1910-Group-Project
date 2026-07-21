# AI-Powered Misinformation Risk Assistant

This repository contains the core experimental code for an AI misinformation risk assistant. The system estimates whether a short social-media-style claim deserves further checking. It outputs a risk score and warning level; it does not claim to be a final truth judge.

## Repository Structure

```text
.
├── data/
│   └── demo_posts.csv
├── src/
│   └── misinformation_risk_assistant.py
├── run_demo.py
├── run_transformer_demo.py
├── requirements.txt
└── README.md
```

Generated outputs, downloaded datasets, model caches, reports, and rendered figures are intentionally excluded from version control.

## Methods

The repository includes two reproducible model paths:

- `run_demo.py`: TF-IDF text features plus logistic regression, with additional metadata features when available.
- `run_transformer_demo.py`: a compact BERT-style transformer fine-tuning experiment for the same binary risk task.

The LIAR labels are mapped into a binary risk setting:

- Higher risk: `false`, `barely-true`, `pants-fire`
- Lower risk: `half-true`, `mostly-true`, `true`

## Installation

```powershell
pip install -r requirements.txt
```

## Baseline Experiment

Run the synthetic demo data:

```powershell
python run_demo.py
```

Run the LIAR experiment:

```powershell
python run_demo.py --dataset liar
```

If `data/liar_dataset.zip` is missing, the script downloads it automatically from the original LIAR dataset URL.

Expected output files are written to `outputs/`:

```text
evaluation_metrics.json
predictions.csv
risk_distribution.png
```

## Transformer Experiment

Run the CPU-friendly transformer experiment:

```powershell
python run_transformer_demo.py --epochs 5
```

The default checkpoint is `prajjwal1/bert-tiny`. A larger model can be tested with:

```powershell
python run_transformer_demo.py --model distilbert-base-uncased --max-train 4000 --epochs 1
```

Transformer outputs are written to `transformer_outputs/`.

## Data

The core public experiment uses the LIAR dataset:

W. Y. Wang, "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection," ACL 2017.

The repository does not commit the LIAR zip file or Hugging Face cache. This keeps the repository lightweight and reproducible.
