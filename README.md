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

## Results at a glance

**➡ Full write-up: [reports/TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md)**

| Milestone | Question | Verdict |
|-----------|----------|---------|
| **1** ✅ | What does the real data look like? | 0.173% fraud; signal concentrated in V17/V14/V12 |
| **2** ✅ | Train CTGAN on fraud-only rows | 378 training rows, 95 held out *before* training for M5 |
| **3** ✅ | Is synthetic data statistically faithful? | **No** — 28/29 columns fail KS vs a sample-size-scaled real-vs-real noise floor; correlations halved |
| **4** ✅ | Does it help a fraud classifier? | **No** — dose-dependent AP decline; loses to naive oversampling |
| **5** ✅ | Does it leak training records? | **No** — membership attack AUC 0.51 (coin flip), zero memorized rows |
| **6** ✅ | Honest write-up | One mechanism, three results: *you cannot leak what you never learned* |

## Experiment B — does more training close the gap?

**➡ [reports/experiment_b_report.md](reports/experiment_b_report.md)**

Follow-up: swept 4 stronger CTGAN configs (up to 4,000 epochs, doubled network capacity) against
the unmodified fidelity and privacy tests. Result: fidelity improved modestly then **plateaued**
(correlation error stuck at ~0.16 regardless of training time or network size), the best config
still failed utility the same way the original did, and **privacy stayed flat at coin-flip across
all five generators tested** — the predicted fidelity-privacy tradeoff never engaged within this
search space. A shadow-model attack (the stronger privacy test) was scoped and started but honestly
stopped partway when the compute cost stopped being justified by the five-generator DCR consensus
already in hand — documented as unfinished future work, not a completed result.

## Generator bake-off — the twist ending

**➡ [reports/generator_bakeoff.md](reports/generator_bakeoff.md) · [reports/utility_v2_report.md](reports/utility_v2_report.md)**

Prompted by external review, two more generator families went through the unchanged harness:

| generator | fidelity (of 29) | corr error | attack AUC | fit time |
|---|---|---|---|---|
| CTGAN (best of 5 configs) | 1 pass | 0.160 | 0.503 | 31 min |
| **Gaussian copula (~40 lines)** | **29 pass** | 0.090 | 0.498 | **<1 s** |
| TVAE | 0 pass | **0.080** | 0.544 | 8 min |

The copula — a closed-form model fitting in under a second — swept the per-column fidelity test
that 31 minutes of adversarial training couldn't crack (partly by construction: it resamples
empirical marginals, which is itself evidence that per-column KS is gameable). TVAE learned the
best *joint* structure and was the only generator whose attack score nudged upward (not
statistically significant, but the first data point in the predicted fidelity→leakage direction).

The properly-powered utility test (10 seeds, paired deltas, bootstrap CIs, data-ablation arm) then
delivered the project's deepest finding: **even the 29/29-fidelity copula does not help the
classifier** (−0.004 to −0.005 AP, 7–8 of 10 seeds below baseline), while the ablation curve shows
each ~95 additional *real* fraud rows buy ~+0.007 AP. Synthetic rows — even statistically
near-perfect ones — are interpolations of information the model already has; they don't substitute
for new real data. **Fidelity metrics, including a clean 29/29 sweep, do not predict utility.**

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
