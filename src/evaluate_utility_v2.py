"""
Utility evaluation v2 — same leakage-safe design, properly powered.

Run:  python src/evaluate_utility_v2.py

Upgrades over evaluate_utility.py (v1, 3 seeds), following external review:

1. TEN SEEDS per condition (v1 used 3). RandomForest seed variance was the
   softest part of the original claim.
2. PAIRED PER-SEED DELTAS. For each seed s, delta_s = AP(condition, s) -
   AP(real_only, s). Pairing removes shared seed variance; we report the mean
   paired delta with its std and a sign count across seeds.
3. STRATIFIED BOOTSTRAP CIs. 1000 bootstrap resamples of the test set
   (legit and fraud resampled separately — with only 95 positives, a plain
   bootstrap can produce degenerate resamples). Baseline and condition are
   evaluated on the SAME bootstrap samples, so the CI on the delta is paired.
4. DATA-ABLATION ARM. real-only conditions at 189 / 283 / 378 member frauds.
   This is the CLEAN version of "what would more real fraud buy?" — the
   reviewer-suggested alternative (train on all 473) would put the 95
   holdout test frauds inside the training set: train-on-test contamination.
   The ablation slope answers the data-scarcity question without touching
   the test set.

Synthetic doses use the Experiment B winner (B3, 4000 epochs) — the strongest
available generator, i.e. the best case for augmentation.

Outputs:
   reports/utility_v2_report.md
   reports/figures/12_utility_v2.png
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SYNTH_PATH = PROJECT_ROOT / "data" / "synthetic" / "expB_B3_4000ep.csv"
COPULA_PATH = PROJECT_ROOT / "data" / "synthetic" / "bakeoff_copula.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "utility_v2_report.md"

SEED = 42
N_SEEDS = 10
N_BOOTSTRAP = 1000
LEGIT_TEST_FRACTION = 0.20
RF_PARAMS = dict(n_estimators=200, n_jobs=-1)


def build_base():
    df = pd.read_csv(DATA_PATH)
    synthetic = pd.read_csv(SYNTH_PATH)
    cols = list(synthetic.columns)
    members = df.loc[pd.read_csv(SPLIT_DIR / "member_indices.csv")["index"], cols]
    holdout = df.loc[pd.read_csv(SPLIT_DIR / "holdout_indices.csv")["index"], cols]
    legit = df[df["Class"] == 0].drop_duplicates(subset=cols)
    legit_train, legit_test = train_test_split(
        legit[cols], test_size=LEGIT_TEST_FRACTION, random_state=SEED)
    X_test = pd.concat([legit_test, holdout], ignore_index=True)
    y_test = np.concatenate([np.zeros(len(legit_test)), np.ones(len(holdout))])
    print(f"train legit: {len(legit_train):,}   members: {len(members)}   "
          f"test: {len(legit_test):,} legit + {len(holdout)} fraud", flush=True)
    return legit_train, members, synthetic, X_test, y_test


def condition_probas(legit_train, fraud_rows, X_test) -> list[np.ndarray]:
    """Fit N_SEEDS forests on legit_train + fraud_rows; return test probas."""
    X_tr = pd.concat([legit_train, fraud_rows], ignore_index=True)
    y_tr = np.concatenate([np.zeros(len(legit_train)), np.ones(len(fraud_rows))])
    probas = []
    for s in range(N_SEEDS):
        clf = RandomForestClassifier(random_state=SEED + s, **RF_PARAMS)
        clf.fit(X_tr, y_tr)
        probas.append(clf.predict_proba(X_test)[:, 1])
    return probas


def bootstrap_delta_ci(proba_cond: np.ndarray, proba_base: np.ndarray,
                       y_test: np.ndarray, rng: np.random.Generator):
    """Paired stratified bootstrap CI on AP(cond) - AP(base), seed-0 probas."""
    legit_idx = np.where(y_test == 0)[0]
    fraud_idx = np.where(y_test == 1)[0]
    deltas = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = np.concatenate([rng.choice(legit_idx, len(legit_idx), replace=True),
                              rng.choice(fraud_idx, len(fraud_idx), replace=True)])
        yb = y_test[idx]
        deltas[b] = (average_precision_score(yb, proba_cond[idx])
                     - average_precision_score(yb, proba_base[idx]))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    start = time.time()
    legit_train, members, synthetic, X_test, y_test = build_base()
    rng = np.random.default_rng(SEED)

    # Ablation subsets: seeded, nested (189 ⊂ 283 ⊂ 378) so the curve is clean.
    shuffled = members.sample(frac=1.0, random_state=SEED)
    copula = pd.read_csv(COPULA_PATH)
    conditions = [
        ("real_189 (ablation)", shuffled.iloc[:189]),
        ("real_283 (ablation)", shuffled.iloc[:283]),
        ("real_only (378)", members),
        ("real+1000 B3synth", pd.concat([members, synthetic.iloc[:1000]], ignore_index=True)),
        ("real+2000 B3synth", pd.concat([members, synthetic.iloc[:2000]], ignore_index=True)),
        ("real+1000 copula", pd.concat([members, copula.iloc[:1000]], ignore_index=True)),
        ("real+2000 copula", pd.concat([members, copula.iloc[:2000]], ignore_index=True)),
        ("oversample+1000", pd.concat(
            [members, members.sample(1000, replace=True, random_state=SEED)],
            ignore_index=True)),
    ]

    print(f"\n{len(conditions)} conditions x {N_SEEDS} seeds:", flush=True)
    probas: dict[str, list[np.ndarray]] = {}
    aps: dict[str, np.ndarray] = {}
    for name, fraud_rows in conditions:
        probas[name] = condition_probas(legit_train, fraud_rows, X_test)
        aps[name] = np.array([average_precision_score(y_test, p)
                              for p in probas[name]])
        print(f"  {name:<22} AP {aps[name].mean():.4f} ± {aps[name].std():.4f}",
              flush=True)

    # Paired analysis + bootstrap CIs vs the real_only(378) baseline.
    base = "real_only (378)"
    rows = []
    for name, _ in conditions:
        paired = aps[name] - aps[base]
        lo, hi = bootstrap_delta_ci(probas[name][0], probas[base][0], y_test, rng)
        rows.append({
            "condition": name,
            "ap_mean": aps[name].mean(), "ap_std": aps[name].std(),
            "paired_delta_mean": paired.mean(), "paired_delta_std": paired.std(),
            "seeds_below_baseline": int((paired < 0).sum()) if name != base else 0,
            "bootstrap_delta_lo": lo, "bootstrap_delta_hi": hi,
        })
    res = pd.DataFrame(rows).set_index("condition")
    print("\n" + res.round(4).to_string(), flush=True)

    # Figure: ablation curve + augmentation deltas.
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    abl = [("real_189 (ablation)", 189), ("real_283 (ablation)", 283),
           (base, 378)]
    axes[0].errorbar([n for _, n in abl],
                     [aps[c].mean() for c, _ in abl],
                     yerr=[aps[c].std() for c, _ in abl],
                     marker="o", capsize=4, color="#4C72B0")
    axes[0].set_xlabel("real fraud rows in training")
    axes[0].set_ylabel("average precision")
    axes[0].set_title("Data-ablation curve (is 378 rows still data-starved?)")
    aug = [c for c, _ in conditions if c not in dict(abl)]
    deltas = [aps[c].mean() - aps[base].mean() for c in aug]
    stds = [np.std(aps[c] - aps[base]) for c in aug]
    colors = ["#C44E52"] * 3 + ["#937860"]
    axes[1].bar(aug, deltas, yerr=stds, capsize=4, color=colors)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("paired ΔAP vs real_only")
    axes[1].set_title(f"Augmentation effect (paired over {N_SEEDS} seeds)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "12_utility_v2.png", dpi=120)
    plt.close(fig)

    elapsed = (time.time() - start) / 60
    REPORT_PATH.write_text(f"""# Utility Report v2 — Properly Powered (10 seeds, bootstrap CIs, ablation)

*Generated by `src/evaluate_utility_v2.py` in {elapsed:.1f} min. Synthetic doses from BOTH the
Experiment B winner (CTGAN B3, 4000 epochs) and the Gaussian copula (29/29 fidelity pass in the
bake-off) — the decisive head-to-head. Same leakage-safe design as v1: test frauds are the 95
GAN-holdout rows, legit deduplicated before splitting.*

## Results

{res.round(4).to_markdown()}

- `paired_delta_mean/std`: per-seed AP difference vs `real_only (378)` — pairing removes shared
  seed variance.
- `seeds_below_baseline`: how many of the {N_SEEDS} seeds individually scored below baseline.
- `bootstrap_delta_lo/hi`: 95% CI on the AP delta from {N_BOOTSTRAP} paired stratified bootstrap
  resamples of the test set (seed-0 models). If the CI spans 0, the per-dose effect is not
  individually significant at this test-set size — the honest statement is then about the
  *consistent direction across doses and seeds*, not any single number.

## Ablation note

The 189/283/378 curve holds the test set fixed and subsamples only the training frauds. The
reviewer-suggested "train on all 473" condition was rejected: the 95 holdout frauds ARE the test
set, so that condition would train on test data.

## Figure

- `figures/12_utility_v2.png` — ablation curve + paired augmentation deltas.
""", encoding="utf-8")
    print(f"\nDone in {elapsed:.1f} min -> {REPORT_PATH.relative_to(PROJECT_ROOT)}",
          flush=True)


if __name__ == "__main__":
    main()
