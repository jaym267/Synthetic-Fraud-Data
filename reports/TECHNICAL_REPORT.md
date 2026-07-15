# Synthetic Fraud Data, Proven Useless (and Therefore Private)

**A technical report on rigorously evaluating a CTGAN fraud-data generator — fidelity, downstream
utility, and privacy — with honest negative results.**

*Solo project. All evaluation code hand-written (scipy/scikit-learn); generator is the standalone
`ctgan` package. Full pipeline reproducible from this repo: see [Reproducing](#reproducing).*

---

## TL;DR

I trained a CTGAN on 378 real credit-card fraud records and asked three questions most synthetic-data
demos never ask, each with a pre-registered measurement design:

| Question | Method | Result |
|---|---|---|
| Is the synthetic data statistically faithful? | Per-column KS tests vs a real-vs-real noise floor; correlation-matrix comparison | **No.** 28/29 columns fail (1 borderline, 0 pass); strongest correlations halved |
| Does it improve a fraud classifier? | Leakage-safe RandomForest experiment, dose-response + oversampling control, 3 seeds | **No.** Dose-dependent *degradation*; loses to photocopying rows |
| Does it leak the training records? | Membership inference (DCR attack), members vs holdout ground truth | **No.** Attack AUC 0.51 ≈ coin flip; zero memorized rows |

The three results are causally one result: **the generator never captured the real joint
distribution, so its output could neither help a classifier nor betray its training data.**
Low fidelity bought privacy for free — by producing data that wasn't worth protecting.
The regime that matters for real deployments is the opposite corner: a generator good enough to be
useful is good enough to start leaking. This pipeline measures exactly where a generator sits on
that tradeoff, and that measurement — not the GAN — is the deliverable.

---

## 1. The problem

Fraud datasets are doubly constrained: the positive class is vanishingly rare, and the records are
too sensitive to share. Synthetic data is marketed as the fix for both — generate more fraud
examples for training, publish synthetic rows instead of real ones. Both claims are usually
supported by "the histograms look similar."

This project tests both claims properly on the standard Kaggle credit-card dataset (284,807
transactions, **492 frauds = 0.173%**, features `V1–V28` PCA-anonymized plus `Amount`/`Time`).

## 2. Setup decisions that paid off later

- **The member/holdout split came *before* training.** Fraud rows were deduplicated (492 → 473;
  exact duplicates would blur both the utility split and the privacy ground truth), then split
  80/20 with a fixed seed: **378 members** the GAN trained on, **95 holdout** rows it never saw.
  Milestone 5's attack is only measurable because this split predates the generator.
- **`Time` was dropped** (seconds since collection start — an artifact, not a fraud property).
- **Accuracy is never reported.** At 0.173% prevalence a "never fraud" model is 99.83% accurate.
  All classifier results are fraud-class precision/recall and average precision (PR-AUC).
- EDA flagged that within the fraud subset the PCA features are *strongly correlated* (V17–V18
  r = 0.97) even though PCA orthogonalizes the full population — conditioning on a subpopulation
  reintroduces correlation. That joint structure is what a generator must learn, and (it turned
  out) what this one lost.

Generator: CTGAN (`ctgan` 0.12.1), 500 epochs, batch 40, seed 42, 3.6 min CPU; 2,000 synthetic
fraud rows sampled. Training config in `models/training_metadata.json`.

## 3. Fidelity: failed 28 of 29 columns

**Design.** Two-sample Kolmogorov–Smirnov per column, judged by the KS *statistic* as an effect
size — at n=2000 vs 378, p-values flag even trivial gaps (Holm-corrected p-values reported as
support). To make the statistic interpretable I computed a **noise floor**: the same KS between the
two *real* samples (members vs holdout), i.e. what pure sampling noise looks like — then rescaled it
to the synthetic comparison's sample sizes, since KS null values scale like √(1/n + 1/m). Median
scaled floor: **0.037**. At/below floor = indistinguishable from real; above 2× floor = fail.
*(An earlier version of this analysis skipped the rescaling, used a floor of 0.075 — about 2× too
generous to the generator — and reported 23/29 FAIL. Caught in code review; the corrected
calibration makes the fidelity verdict strictly worse.)*

**Results** ([fidelity_report.md](fidelity_report.md), figures 05–07):

- **28 FAIL / 1 borderline / 0 PASS** — nothing passes; V12, the best column, is merely borderline.
- **V17 — the dataset's strongest fraud signal — is the second-worst column** (KS 0.46, ~12× the floor).
  Real fraud sits at V17 ≈ −6.4; the synthetic rows sit at +1.0, where *legitimate* transactions
  live. `Amount` (long right tail, a known GAN weakness flagged in the EDA) fails at KS 0.39.
- **Correlation structure collapsed:** V17~V18 real r = 0.974 → synthetic 0.478; V7~V10 0.880 →
  0.134. Mean |Δcorr| = 0.211 across all pairs, max 0.808. The generator learned the parts,
  not the whole.

A methodological aside: in the post-training sanity check V14's *mean* looked fine (−7.95 vs −6.88
real) — the KS test still fails it. Matching a mean is not matching a distribution;
this is why the pipeline never trusts summary statistics or eyeballed histograms.

## 4. Utility: synthetic data lost to copy-paste

**Design.** RandomForest (200 trees, 3 seeds per condition, mean ± std), trained on ~220k deduped
legit rows + fraud under five conditions: real-only (378 frauds), real + {500, 1000, 2000}
synthetic (a dose-response curve), and real + 1000 *random duplicates* of the real frauds — the
control that asks whether the GAN beats literal photocopying. Two leakage traps closed: **test
frauds are exactly the 95 GAN-holdout rows** (synthetic rows derive from members, so nothing in the
test set was ever visible to the generator), and legit rows were deduplicated before splitting
(the EDA's 1,081 duplicate rows would otherwise straddle the train/test line).

**Results** ([utility_report.md](utility_report.md), figures 08–09):

| Condition | Average precision |
|---|---|
| real_only | **0.8424 ± 0.0057** |
| real+500synth | 0.8337 ± 0.0042 |
| real+1000synth | 0.8333 ± 0.0051 |
| real+2000synth | 0.8308 ± 0.0036 |
| oversample+1000 | **0.8435 ± 0.0070** |

- AP declines **monotonically with synthetic dose** — three conditions in strict dose order, each
  gap ~2× the seed spread. Not one unlucky run; a pattern with a mechanism: the synthetic rows put
  "fraud" mass where legit transactions live (V17 ≈ +1), blurring the classifier's sharpest boundary.
- **The photocopy control matched baseline and beat every synthetic dose.** Two lines of pandas
  outperformed the entire deep generative stack.
- All augmented conditions (including photocopies) nudged recall@0.5 from 0.789 → 0.800 — exactly
  one extra fraud caught out of 95, a threshold effect, not new information; the threshold-free AP
  shows the ranking got worse with synthetic.

**Claim, calibrated:** synthetic data from *this* generator provided no benefit over free baselines,
with a consistent dose-dependent degradation (~1% relative AP). Not catastrophic — just strictly
worse than doing nothing, at positive cost.

## 5. Privacy: the attack found nothing to steal

**Design.** Distance-to-closest-record membership inference: an attacker holding the released
synthetic set scores each candidate real record by its distance to the nearest synthetic row(s)
(closer ⇒ "was probably in training"). Ground truth: 378 members vs 95 non-member holdout frauds.
All attacker preprocessing (standardization) derives from the synthetic data alone — the attacker
doesn't have the training set. Attack quality = ROC AUC; 0.5 is a coin flip.

**Results** ([privacy_report.md](privacy_report.md), figures 10–11):

- Attack AUC **0.5096** (nearest-neighbor score) and **0.5047** (mean of 5 nearest);
  Mann-Whitney p = 0.39 / 0.44. Member and non-member distance distributions overlap almost
  entirely (medians 3.43 vs 3.40).
- **Memorization scan: zero near-copies.** The closest synthetic row to *any* training row sits
  2.16 standardized units away — nothing resembling a copied record.

## 6. The one-sentence synthesis

> **The generator failed fidelity, therefore failed utility, therefore passed privacy — one
> mechanism, three measurements: you cannot leak what you never learned.**

This is the fidelity–privacy tradeoff observed end-to-end on my own artifacts. The corollary I take
seriously: "our synthetic data passed a privacy audit" is, on its own, compatible with "our
synthetic data is worthless." The audit that matters reports *both* axes — and vendors rarely
volunteer the second one.

## 7. Limitations — stated, not buried

1. **One generator, one configuration.** 500 epochs on 378 rows, default architecture. These
   results characterize *this* run, not CTGANs in general; an undertrained generator was always the
   likely outcome at this data size.
2. **95 holdout frauds bound the resolution** of both the utility test (each missed fraud ≈ 1 point
   of recall) and the attack AUC. The split couldn't be bigger without starving the generator.
3. **One attack.** The DCR attack is the standard first test, not the strongest possible adversary
   (shadow models, density estimation). A null here means "resists the standard attack," not
   "privacy proven."
4. **Small effects in utility.** The dose-response trend is consistent, but ~1% relative AP; with
   3 seeds I report it as a trend with a mechanism, not a law.
5. **Friendly data.** All-continuous PCA features are CTGAN's easy mode. Real tables with
   high-cardinality categoricals are harder — failures here likely understate failures there.

## 8. What I'd do next

- **Experiment B — trade privacy for utility, and watch the dial:** retrain at 2,000+ epochs and
  larger capacity, then rerun this exact pipeline unchanged. Hypotheses: fidelity improves, utility
  approaches (maybe passes) baseline, attack AUC rises above 0.5. The interesting number is *where*
  it lands — the pipeline turns the tradeoff from a slogan into a curve.
- Try TVAE and a Gaussian copula as alternative generators under the identical evaluation.
- Strengthen the attacker (shadow-model attack) so a future "no leak" claim survives a better adversary.
- Differential-privacy training (DP-SGD) once utility exists — provable bounds instead of empirical nulls.

## Reproducing

```powershell
python -m venv venv; venv\Scripts\Activate.ps1
pip install -r requirements.txt
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
python src/explore_data.py          # M1: EDA -> eda_summary.md, figures 01-04
python src/train_ctgan.py           # M2: split + train + sample (~4 min CPU)
python src/evaluate_fidelity.py     # M3: KS + correlations -> fidelity_report.md
python src/evaluate_utility.py      # M4: classifier experiment (~20 min CPU)
python src/membership_inference.py  # M5: privacy attack -> privacy_report.md
```

Every run is seeded; the member/holdout split is persisted in `data/splits/` and reused verbatim
by every downstream script.
