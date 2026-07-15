"""
Milestone 4 — Utility test: does synthetic fraud actually help a classifier?

Run:  python src/evaluate_utility.py

Experimental design, and the three leakage traps it dodges:

1. SYNTHETIC-TO-TEST LEAKAGE. The synthetic rows were generated from the 378
   "member" fraud rows. If members appeared in the classifier's test set, the
   augmented conditions would be evaluated partly on near-copies of their own
   training data. So: test frauds = the 95 HOLDOUT rows the GAN never saw;
   training frauds = the 378 members. The Milestone 2 split is reused as-is.

2. DUPLICATE LEAKAGE IN LEGIT ROWS. The EDA found ~1,081 exact-duplicate rows
   (mostly legit). A random split would put identical rows on both sides of
   the train/test line. Legit rows are deduplicated before splitting.

3. SEED LUCK. RandomForest has run-to-run variance. Every condition is run
   with N_SEEDS seeds; results are reported as mean +/- std, and differences
   smaller than the spread are treated as noise.

Conditions:
   real_only          378 real frauds (baseline)
   real+{500,1000,2000}  baseline + N synthetic rows (dose-response)
   oversample+1000    baseline + 1000 random duplicates of the real frauds —
                      the control that answers "does the GAN beat photocopying?"

Metrics (fraud class only): precision / recall / F1 at threshold 0.5, and
average precision (PR-AUC) as the threshold-independent headline number.

Outputs:
   reports/utility_report.md
   reports/figures/08_utility_dose_response.png
   reports/figures/09_pr_curves.png
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
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SYNTH_PATH = PROJECT_ROOT / "data" / "synthetic" / "synthetic_fraud.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "utility_report.md"

SEED = 42
N_SEEDS = 3                     # classifier seeds per condition
LEGIT_TEST_FRACTION = 0.20      # of deduped legit rows
SYNTH_DOSES = [500, 1000, 2000]
OVERSAMPLE_DOSE = 1000
RF_PARAMS = dict(n_estimators=200, n_jobs=-1)  # seed set per run


def build_datasets():
    """Assemble leakage-safe train/test sets. Returns (X_tr_base, y_tr_base,
    X_test, y_test, real_fraud_train, synthetic)."""
    df = pd.read_csv(DATA_PATH)
    synthetic = pd.read_csv(SYNTH_PATH)
    feature_cols = list(synthetic.columns)          # V1..V28 + Amount (no Time)

    member_idx = pd.read_csv(SPLIT_DIR / "member_indices.csv")["index"]
    holdout_idx = pd.read_csv(SPLIT_DIR / "holdout_indices.csv")["index"]

    fraud_train = df.loc[member_idx, feature_cols]      # 378 — GAN members
    fraud_test = df.loc[holdout_idx, feature_cols]      # 95  — GAN never saw

    # Legit: dedupe on the feature columns, then a seeded split.
    legit = df[df["Class"] == 0]
    legit = legit.drop_duplicates(subset=feature_cols)
    legit_train, legit_test = train_test_split(
        legit[feature_cols], test_size=LEGIT_TEST_FRACTION, random_state=SEED
    )

    X_tr_base = pd.concat([legit_train, fraud_train], ignore_index=True)
    y_tr_base = np.concatenate([np.zeros(len(legit_train)), np.ones(len(fraud_train))])
    X_test = pd.concat([legit_test, fraud_test], ignore_index=True)
    y_test = np.concatenate([np.zeros(len(legit_test)), np.ones(len(fraud_test))])

    print(f"train: {len(legit_train):,} legit + {len(fraud_train)} real fraud")
    print(f"test:  {len(legit_test):,} legit + {len(fraud_test)} fraud "
          f"(all GAN-holdout -> no synthetic-to-test leakage)")
    return X_tr_base, y_tr_base, X_test, y_test, fraud_train, synthetic


def augment(X_base: pd.DataFrame, y_base: np.ndarray, extra: pd.DataFrame):
    X = pd.concat([X_base, extra], ignore_index=True)
    y = np.concatenate([y_base, np.ones(len(extra))])
    return X, y


def run_condition(name: str, X_tr: pd.DataFrame, y_tr: np.ndarray,
                  X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Train/evaluate one condition across N_SEEDS seeds."""
    per_seed = []
    curves = None
    for s in range(N_SEEDS):
        clf = RandomForestClassifier(random_state=SEED + s, **RF_PARAMS)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        per_seed.append({
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "ap": average_precision_score(y_test, proba),
        })
        if s == 0:  # keep one PR curve per condition for the figure
            curves = precision_recall_curve(y_test, proba)
    seed_df = pd.DataFrame(per_seed)
    result = {"condition": name, "n_train_fraud": int(y_tr.sum())}
    for metric in ["precision", "recall", "f1", "ap"]:
        result[f"{metric}_mean"] = seed_df[metric].mean()
        result[f"{metric}_std"] = seed_df[metric].std()
    result["_pr_curve"] = curves
    print(f"  {name:<18} AP={result['ap_mean']:.4f}±{result['ap_std']:.4f}  "
          f"P={result['precision_mean']:.3f}±{result['precision_std']:.3f}  "
          f"R={result['recall_mean']:.3f}±{result['recall_std']:.3f}")
    return result


def make_figures(results: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    res = pd.DataFrame([{k: v for k, v in r.items() if k != "_pr_curve"}
                        for r in results]).set_index("condition")

    # Fig 8: dose response for AP, precision, recall.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric, label in zip(
        axes, ["ap", "precision", "recall"],
        ["Average precision (PR-AUC)", "Precision @0.5", "Recall @0.5"],
    ):
        ax.bar(res.index, res[f"{metric}_mean"],
               yerr=res[f"{metric}_std"], capsize=4,
               color=["#4C72B0"] + ["#C44E52"] * len(SYNTH_DOSES) + ["#937860"])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(bottom=max(0.0, res[f"{metric}_mean"].min() - 0.15))
    fig.suptitle("Fraud-class utility by training condition (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_utility_dose_response.png", dpi=120)
    plt.close(fig)

    # Fig 9: PR curves (seed 0 of each condition).
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        precision, recall, _ = r["_pr_curve"]
        ax.plot(recall, precision, label=f"{r['condition']} (AP={r['ap_mean']:.3f})")
    ax.set_xlabel("Recall (fraud)")
    ax.set_ylabel("Precision (fraud)")
    ax.set_title("Precision-recall curves on the 95 held-out real frauds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_pr_curves.png", dpi=120)
    plt.close(fig)


def write_report(results: list[dict], elapsed_min: float) -> None:
    res = pd.DataFrame([{k: v for k, v in r.items() if k != "_pr_curve"}
                        for r in results]).set_index("condition")
    baseline_ap = res.loc["real_only", "ap_mean"]
    res["ap_delta_vs_real"] = res["ap_mean"] - baseline_ap
    table_md = res.round(4).to_markdown()

    content = f"""# Utility Report — Does Synthetic Fraud Help a Classifier? (Milestone 4)

*Generated by `src/evaluate_utility.py` in {elapsed_min:.1f} min.
RandomForest ({RF_PARAMS['n_estimators']} trees), {N_SEEDS} seeds per condition, metrics on the
fraud class only. Test frauds are the 95 GAN-holdout rows — the generator never saw them, so
synthetic augmentation cannot leak into the test set. Legit rows deduplicated before splitting.*

## Results

{table_md}

`ap_delta_vs_real` is the change in average precision relative to the real-only baseline —
the single number that answers the milestone's question.

## How to read this

- **average precision (ap)** is the area under the precision-recall curve: threshold-independent,
  and the honest headline metric at 0.17% prevalence.
- The **oversample+{OVERSAMPLE_DOSE}** row is the control: it adds exact photocopies of the real
  frauds instead of GAN output. Synthetic data must beat it to justify the GAN's existence.
- Differences smaller than the seed-to-seed std are noise.

## Figures

- `figures/08_utility_dose_response.png` — AP / precision / recall by condition
- `figures/09_pr_curves.png` — full precision-recall curves

*Interpretation recorded separately after inspection; this file records the measurements.*
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    start = time.time()
    X_tr_base, y_tr_base, X_test, y_test, fraud_train, synthetic = build_datasets()

    conditions: list[tuple[str, pd.DataFrame | None]] = [("real_only", None)]
    conditions += [(f"real+{n}synth", synthetic.iloc[:n]) for n in SYNTH_DOSES]
    dup_rows = fraud_train.sample(OVERSAMPLE_DOSE, replace=True, random_state=SEED)
    conditions.append((f"oversample+{OVERSAMPLE_DOSE}", dup_rows))

    print(f"\nRunning {len(conditions)} conditions x {N_SEEDS} seeds:")
    results = []
    for name, extra in conditions:
        if extra is None:
            X_tr, y_tr = X_tr_base, y_tr_base
        else:
            X_tr, y_tr = augment(X_tr_base, y_tr_base, extra)
        results.append(run_condition(name, X_tr, y_tr, X_test, y_test))

    elapsed_min = (time.time() - start) / 60
    make_figures(results)
    write_report(results, elapsed_min)
    print(f"\nDone in {elapsed_min:.1f} min")
    print(f"Report -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("Figures -> reports/figures/08..09_*.png")


if __name__ == "__main__":
    main()
