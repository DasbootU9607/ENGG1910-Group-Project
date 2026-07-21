# AI-Powered Misinformation Risk Assistant

This repository contains the core experimental code for an AI misinformation risk assistant. The system estimates whether a short social-media-style claim deserves further checking. It outputs a risk score and warning level; it does not claim to be a final truth judge.

Project page: https://dasbootu9607.github.io/ENGG1910-Group-Project/

Paper: [`paper/report.pdf`](paper/report.pdf)

## Repository Structure

```text
.
├── docs/
│   └── index.html
├── paper/
│   ├── figures/
│   ├── report.md
│   ├── report.pdf
│   └── report.tex
├── data/
│   └── demo_posts.csv
├── src/
│   └── misinformation_risk_assistant.py
├── run_demo.py
├── run_transformer_demo.py
├── requirements.txt
└── README.md
```

Generated outputs, downloaded datasets, model caches, and render scratch directories are intentionally excluded from version control.

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

For NVIDIA GPU training on Windows, install the CUDA PyTorch wheel after the base requirements:

```powershell
pip install --upgrade --force-reinstall -r requirements-cuda.txt
```

The CUDA requirements file uses `--no-deps`, so it replaces only `torch`, `torchvision`, and `torchaudio`; install `requirements.txt` first.

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

## Legacy Transformer Demo

Run the old CPU-friendly transformer demo:

```powershell
python run_transformer_demo.py --epochs 5
```

The default checkpoint is `prajjwal1/bert-tiny`. A larger model can be tested with:

```powershell
python run_transformer_demo.py --model distilbert-base-uncased --max-train 4000 --epochs 1
```

Transformer outputs are written to `transformer_outputs/`. This script is kept for a lightweight demo; use `run_fair_experiment.py` for final results.

## Fair LIAR Experiment

For a fair model comparison, use the official LIAR train, validation, and test splits with the same binary labels and the same feature access. The fair runner trains:

- `tfidf_text_only`: TF-IDF plus logistic regression using only claim text.
- `transformer_text_only`: transformer fine-tuning using only claim text.
- `tfidf_metadata_enhanced`: TF-IDF plus logistic regression with speaker-history metadata, reported separately because it uses extra information.

Quick baseline-only check:

```powershell
python run_fair_experiment.py --skip-transformer
```

Full GPU transformer comparison:

```powershell
python run_fair_experiment.py --device cuda --model bert-base-uncased --epochs 5 --batch-size 8 --max-train 0
```

If `--device cuda` fails, verify that the current Python environment has a CUDA-enabled PyTorch build:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Outputs are written to `fair_outputs/`, including `metrics.json`, `summary.csv`, per-model prediction files, and `transformer_training_log.csv`. For the fair text-only comparison, both TF-IDF and transformer train on the official LIAR training split only, tune thresholds on the official validation split, and report once on the official test split. The metadata-enhanced model uses the same rows but has extra speaker-history features, so it should be reported separately rather than treated as a direct model-vs-model comparison.

The transformer runner treats `--epochs` as a training budget. After every epoch, it evaluates the validation split and keeps the checkpoint with the best validation F1 by default.

## Data

The core public experiment uses the LIAR dataset:

W. Y. Wang, "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection," ACL 2017.

The repository does not commit the LIAR zip file or Hugging Face cache. This keeps the repository lightweight and reproducible.
