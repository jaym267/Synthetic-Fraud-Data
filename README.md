# Synthetic Fraud Data, Proven Useful

A solo project that generates synthetic credit-card fraud transactions with a **CTGAN**, then
**rigorously proves** whether that synthetic data is (a) actually *useful* for a downstream fraud
classifier and (b) actually *private* — i.e. it doesn't leak which real transactions were used to
train the generator.

The point isn't the generator. Generating plausible-looking tabular data is easy. The point is the
**proof**: honest, statistical evidence of usefulness and privacy, including reporting where the
synthetic data did *not* help.

## Why this matters

Synthetic data is pitched as a fix for two real problems: **class imbalance** (fraud is ~0.17% of
transactions) and **privacy** (you can't freely share real financial records). But "it looks real"
is not the same as "it helps a model" or "it's safe to share." This project measures both claims
instead of assuming them.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle,
`mlg-ulb/creditcardfraud`): ~284,807 European card transactions over 2 days, 492 frauds (~0.172%).
Features `V1`–`V28` are PCA-transformed (anonymized); `Time` and `Amount` are raw.

## Roadmap

| Milestone | Goal |
|-----------|------|
| **1** ✅ | Environment setup + exploratory data analysis (EDA) |
| 2 | Train a CTGAN on the fraud-only rows to generate synthetic fraud |
| 3 | Fidelity checks — real vs. synthetic distributions & correlations, via statistical tests |
| 4 | Utility test — classifier on real vs. real+synthetic; precision/recall on the rare class |
| 5 | Membership inference attack against the generator — does it leak training rows? |
| 6 | Honest technical write-up, including where synthetic data did *not* help |

## Setup

```powershell
# 1. Virtual environment
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Dependencies
pip install -r requirements.txt

# 3. Kaggle credentials (one-time)
#    Kaggle -> Account -> "Create New API Token" downloads kaggle.json
#    Move it to:  %USERPROFILE%\.kaggle\kaggle.json

# 4. Download the dataset
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

# 5. Run the EDA
python src/explore_data.py
```

Outputs: annotated stats to the console, figures in `reports/figures/`, and a written summary in
`reports/eda_summary.md`.

## Project layout

```
src/            analysis + (later) generator and evaluation code
data/raw/       creditcard.csv (git-ignored — download it yourself)
reports/        eda_summary.md and figures/
notebooks/      ad-hoc exploration
```

## Design choices

- **Standalone `ctgan`**, not a batteries-included framework — the fidelity, utility, and privacy
  tests are written by hand so every metric is understood and defensible, not a black box.
- **Precision/recall on the positive (fraud) class**, never accuracy — at 0.17% positives a model
  that predicts "never fraud" is 99.83% accurate and useless.
