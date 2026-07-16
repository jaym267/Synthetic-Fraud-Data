# Your Synthetic Data Passed Every Statistical Test. It Still Didn't Work.

*A solo project trained fraud-data generators, evaluated them the way a skeptical buyer should,
and found that the metrics the industry leans on can be perfect while the product is useless.
Everything below is reproducible: [github.com/jaym267/Synthetic-Fraud-Data](https://github.com/jaym267/Synthetic-Fraud-Data).*

---

## The claim being sold

Synthetic data vendors make a two-part pitch to banks: *our generator produces fraud
transactions so realistic they'll boost your detection models* (fixing the fact that fraud is
~0.17% of transactions), *and so private you can share them freely* (fixing the fact that you
can't pass real financial records around). The evidence offered is usually a fidelity report:
per-column statistical tests showing the synthetic data is indistinguishable from real.

This project bought the pitch, built the product, and then tested it the way a buyer should.
The result is a finding I haven't seen stated this directly:

> **Marginal fidelity is easy. Joint structure is the bottleneck. And the standard metrics
> measure the easy part.**

## The setup, in one paragraph

Kaggle's credit-card fraud dataset: 284,807 transactions, 473 unique frauds. Before training
anything, 95 frauds were locked away — never shown to any generator — so that both the utility
test and the privacy attacks would have untouched ground truth. Generators were trained on the
other 378 and judged by three hand-written evaluations that never changed across the whole
project: per-column KS tests against a sample-size-scaled real-vs-real noise floor (fidelity),
a leakage-safe RandomForest augmentation experiment with a random-oversampling control
(utility), and membership-inference attacks (privacy).

## Act 1 — The deep learning generator fails everything except privacy

A CTGAN (the standard neural generator for tabular data) failed fidelity 28/29 columns, made
the fraud classifier slightly *worse* at every dose, and showed zero privacy leak. Those three
results are one mechanism: **it never learned the real distribution, so it had nothing useful
to add and nothing memorized to leak.** "You cannot leak what you never learned."

## Act 2 — Compute doesn't fix it

Four stronger configurations: 4× and 8× the training, doubled network capacity, different batch
sizes. Per-column (marginal) fidelity improved steadily — median KS distance halved. But the
error in the *correlation structure* — how features move together, which is where fraud actually
lives (two features correlate at r = 0.97 in real fraud) — hit a wall at ~0.16 and did not move
in any configuration. Marginals kept improving; the joint structure stayed broken. Utility
still got worse with every synthetic row added.

Notice what this means for evaluation practice: **a sweep that only tracked per-column metrics
would have reported steady progress while the thing that matters didn't improve at all.**

## Act 3 — A 40-line model embarrasses the neural network

Was the wall CTGAN's fault, or is 378 rows just too few to learn from? The cheap way to answer:
hold the data fixed and swap the generator family. A hand-rolled Gaussian copula — a closed-form
statistical model, ~40 lines, fitting in under one second — swept the fidelity suite **29/29**
with *half* the correlation error that 31 minutes of adversarial training couldn't beat. A TVAE
landed the mirror image: best joint structure of all, worst marginals.

So the data was never the bottleneck. But there's a sharper lesson in *how* the copula passed:
it resamples the training data's empirical marginals, so per-column KS tests are nearly
guaranteed-passable **by construction**. A metric a trivial model can saturate without learning
anything new is not evidence of generative quality. Vendor fidelity reports are built on exactly
these metrics.

## Act 4 — Perfect fidelity, still useless

The final experiment is the one that should worry anyone buying synthetic data on the strength
of a fidelity report. The copula's output — 29/29 on the very tests the industry quotes — went
into the properly-powered utility experiment (10 seeds, paired statistics, bootstrap confidence
intervals). It did not help the fraud classifier. Neither did anything else, in a specific and
telling pattern:

| training data | average precision |
|---|---|
| 378 real frauds | 0.842 |
| + 2000 CTGAN rows | 0.830 |
| + 2000 copula rows (29/29 fidelity) | 0.838 |
| + 1000 literal photocopies of the real rows | 0.842 |
| 283 real frauds instead of 378 | 0.835 |
| 189 real frauds instead of 378 | 0.828 |

Read the last two rows against the first. The classifier is visibly data-starved — each ~95
additional *real* frauds buys about +0.007 AP. Yet 2000 statistically-near-perfect synthetic
rows can't outperform doing nothing. **Synthetic rows are interpolations of information the
model already has. Real rows carry new information. No fidelity metric measures the
difference — you can max out the metric and add zero information.**

## What about privacy?

The distance-based membership attack read coin-flip (AUC 0.49–0.51) against every generator
tested. Where an empirical "we found no leak" isn't enough, the project also includes a
differentially private copula with a provable (ε, δ) bound and an honest accounting of what the
proof costs in fidelity. And a stronger LiRA-style shadow-model attack is included — because a
privacy claim tested with only the weakest attack in the literature isn't a claim, it's a hope.

## The takeaways, ranked by who needs them

1. **If you're buying synthetic data:** demand a *utility* benchmark on your downstream task
   with a naive-oversampling control, and a membership-inference result. A fidelity table —
   however green — predicts nothing. A 40-line copula can turn it entirely green for free.
2. **If you're evaluating generators:** report joint-structure error alongside per-column
   tests. In this project it was the only fidelity number that tracked the real failure, and
   the only one deep generators couldn't move.
3. **If you're data-starved:** the ablation curve says the money goes into collecting more real
   minority-class examples, not into generating fake ones.

*Every number above regenerates from seed with the code in the repo — including the negative
results, the corrected bug that made our own fidelity finding stricter, and the attack we
abandoned honestly before finishing it the right way.*
