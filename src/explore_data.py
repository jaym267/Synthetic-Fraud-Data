"""
Milestone 1 — Exploratory Data Analysis for the Credit Card Fraud dataset.

Run:  python src/explore_data.py

What this does, and WHY (the "why" is the point of the project):

We are about to build a CTGAN that generates synthetic *fraud* transactions, and then prove whether
that synthetic data is useful and private. Before any of that, we have to deeply understand the real
data — its shape, its imbalance, its quirks — because every later decision (how we train the GAN,
which metric we trust, how we attack our own generator) depends on facts we establish here.

This script prints an annotated report to the console, saves four figures to reports/figures/, and
writes a plain-language summary to reports/eda_summary.md.
"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: we save PNGs, we don't pop up windows
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths. We resolve everything relative to this file so the script runs the
# same way no matter what directory you launch it from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "eda_summary.md"


def rule(title: str) -> None:
    """Print a titled section separator so the console output is readable."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def explain(text: str) -> None:
    """Print a wrapped, indented explanation block (the teaching narration)."""
    print(textwrap.indent(textwrap.fill(text, width=74), "    "))


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(
            f"ERROR: could not find the dataset at:\n  {DATA_PATH}\n\n"
            "Download it first (see README):\n"
            "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip\n"
        )
        sys.exit(1)
    return pd.read_csv(DATA_PATH)


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # Collected facts we reuse for the written summary at the end.
    facts: dict[str, object] = {}

    # -----------------------------------------------------------------------
    # 1. Shape & dtypes
    # -----------------------------------------------------------------------
    rule("1. SHAPE & COLUMN TYPES")
    n_rows, n_cols = df.shape
    facts["n_rows"] = n_rows
    facts["n_cols"] = n_cols
    print(f"Rows:    {n_rows:,}")
    print(f"Columns: {n_cols}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    explain(
        "We expect ~284,807 rows and 31 columns: Time, V1..V28, Amount, and Class. "
        "V1..V28 are the outputs of a PCA the dataset authors ran to anonymize the "
        "original features, so they carry no human-readable meaning — but their "
        "statistical structure is exactly what our CTGAN will have to reproduce. "
        "Everything is numeric (float), which keeps modeling simple: there are no "
        "categorical columns to encode."
    )

    # -----------------------------------------------------------------------
    # 2. Data quality: missing values + duplicates
    # -----------------------------------------------------------------------
    rule("2. DATA QUALITY — MISSING VALUES & DUPLICATES")
    total_missing = int(df.isna().sum().sum())
    facts["total_missing"] = total_missing
    print(f"Total missing values across all cells: {total_missing}")

    n_dupes = int(df.duplicated().sum())
    facts["n_duplicates"] = n_dupes
    dupe_fraud = int(df[df.duplicated(keep=False)]["Class"].sum()) if n_dupes else 0
    print(f"Fully-duplicated rows: {n_dupes:,}")
    if n_dupes:
        print(f"  ...of which fraud rows involved in any duplication: {dupe_fraud}")
    explain(
        "No missing values means no imputation decisions to defend — good. Duplicates "
        "are more interesting: this dataset has a known set of exact-duplicate rows. "
        "They matter for TWO later milestones. (a) Utility: duplicate rows leak between "
        "train and test if we split naively, inflating scores. (b) Privacy: a membership "
        "inference attack (Milestone 5) asks 'was this exact row in the generator's "
        "training set?' — duplicated rows make that question ambiguous and can distort "
        "the attack. We only FLAG them here; we decide how to handle them when we build "
        "each of those steps, and we document the choice."
    )

    # -----------------------------------------------------------------------
    # 3. Class distribution — THE central fact
    # -----------------------------------------------------------------------
    rule("3. CLASS DISTRIBUTION (THE CORE PROBLEM)")
    class_counts = df["Class"].value_counts().sort_index()
    n_legit = int(class_counts.get(0, 0))
    n_fraud = int(class_counts.get(1, 0))
    fraud_pct = 100.0 * n_fraud / n_rows
    facts["n_legit"] = n_legit
    facts["n_fraud"] = n_fraud
    facts["fraud_pct"] = fraud_pct
    print(f"Legitimate (Class=0): {n_legit:,}")
    print(f"Fraud      (Class=1): {n_fraud:,}")
    print(f"Fraud proportion:     {fraud_pct:.3f}%")
    print(f"Imbalance ratio:      1 fraud per {n_legit / max(n_fraud, 1):,.0f} legit")
    explain(
        "This single number — ~0.17% fraud — shapes the whole project. A model that "
        "blindly predicts 'never fraud' is ~99.83% ACCURATE and catches zero fraud, "
        "which is why we will never report accuracy. From Milestone 4 on we report "
        "PRECISION and RECALL on the fraud class specifically. The imbalance is also the "
        "core motivation for synthetic data: there are only ~492 fraud examples to learn "
        "from, so we ask whether generating more of them helps a classifier — and whether "
        "it does so without simply memorizing and regurgitating those 492 real people."
    )

    # -----------------------------------------------------------------------
    # 4. Amount & Time — the two non-PCA columns
    # -----------------------------------------------------------------------
    rule("4. AMOUNT & TIME (THE TWO RAW, NON-PCA COLUMNS)")
    print("Amount summary (all transactions):")
    print(df["Amount"].describe().to_string())
    print(f"\nAmount — median: {df['Amount'].median():.2f}, "
          f"max: {df['Amount'].max():,.2f}, zero-amount rows: {(df['Amount'] == 0).sum():,}")
    print("\nAmount summary by class (mean / median):")
    amt_by_class = df.groupby("Class")["Amount"].agg(["mean", "median", "max"])
    print(amt_by_class.to_string())

    time_hours = df["Time"].max() / 3600.0
    facts["time_span_hours"] = time_hours
    print(f"\nTime spans {df['Time'].min():.0f} to {df['Time'].max():,.0f} seconds "
          f"(~{time_hours:.1f} hours / {time_hours/24:.1f} days).")
    explain(
        "Amount and Time were NOT put through the PCA, so they live on their own scales. "
        "Amount is heavily right-skewed: most transactions are small, with a long tail of "
        "large ones. That skew matters because (a) it can dominate distance-based models "
        "unless scaled, and (b) a GAN often struggles with long tails, so Amount is a "
        "column we'll watch closely in the fidelity checks. Time is just seconds elapsed "
        "from the first transaction over ~2 days — it's not a real timestamp, so we treat "
        "it as a weak feature, not a calendar."
    )

    # -----------------------------------------------------------------------
    # 5. V1..V28 — confirm PCA character
    # -----------------------------------------------------------------------
    rule("5. V1..V28 FEATURE STATISTICS (PCA COMPONENTS)")
    v_cols = [c for c in df.columns if c.startswith("V")]
    v_stats = df[v_cols].agg(["mean", "std", "min", "max"]).T
    pd.set_option("display.float_format", lambda x: f"{x:8.3f}")
    print(v_stats.to_string())
    pd.reset_option("display.float_format")
    explain(
        "PCA outputs are centered, so every V-feature has a mean very close to 0. Their "
        "standard deviations differ (earlier components capture more variance -> larger "
        "spread). This is the 'shape' the CTGAN must learn: not just each column's "
        "distribution, but how they relate. Because they're PCA components on the FULL "
        "data they're near-uncorrelated with each other overall — so in the next section "
        "the correlations worth reading are feature -> Class, not V_i -> V_j."
    )

    # -----------------------------------------------------------------------
    # 6. Correlation with the target
    # -----------------------------------------------------------------------
    rule("6. WHICH FEATURES CARRY FRAUD SIGNAL?")
    corr_with_class = (
        df.corr(numeric_only=True)["Class"].drop("Class").sort_values(key=np.abs, ascending=False)
    )
    top_signal = corr_with_class.head(8)
    facts["top_signal"] = top_signal
    print("Top 8 features by |correlation with Class|:")
    print(top_signal.to_string())
    explain(
        "Correlation is linear and crude, but it's a fast first look at which anonymized "
        "components separate fraud from legit. Features like V17, V14, V12, V10 typically "
        "top this list. These are the columns whose fraud-vs-legit distributions differ "
        "most — so they're both what makes fraud DETECTABLE and what our synthetic fraud "
        "must get right. If the GAN reproduces Amount but flattens V14, a classifier won't "
        "benefit. We'll test exactly that in Milestones 3 and 4."
    )

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    rule("SAVING FIGURES")

    # Fig 1: class balance (log scale so the tiny fraud bar is visible)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["Legit (0)", "Fraud (1)"], y=[n_legit, n_fraud], ax=ax,
                palette=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("count (log scale)")
    ax.set_title(f"Class balance — fraud is {fraud_pct:.3f}% of rows")
    for i, v in enumerate([n_legit, n_fraud]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_balance.png", dpi=120)
    plt.close(fig)

    # Fig 2: Amount distribution (log1p) split by class
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, color, label in [(0, "#4C72B0", "Legit"), (1, "#C44E52", "Fraud")]:
        sns.histplot(np.log1p(df.loc[df["Class"] == cls, "Amount"]), bins=60,
                     stat="density", color=color, alpha=0.5, label=label, ax=ax)
    ax.set_xlabel("log(1 + Amount)")
    ax.set_title("Transaction amount distribution by class (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_amount_distribution.png", dpi=120)
    plt.close(fig)

    # Fig 3: the two most-separating V-features, fraud vs legit overlay
    top_two = list(top_signal.index[:2])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, top_two):
        for cls, color, label in [(0, "#4C72B0", "Legit"), (1, "#C44E52", "Fraud")]:
            sns.kdeplot(df.loc[df["Class"] == cls, col], color=color, fill=True,
                        alpha=0.4, label=label, ax=ax, warn_singular=False)
        ax.set_title(f"{col} — fraud vs legit")
        ax.legend()
    fig.suptitle("Most fraud-separating PCA components")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_top_features_by_class.png", dpi=120)
    plt.close(fig)

    # Fig 4: correlation heatmap (full matrix — shows V_i near-independence + Class row)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=120)
    plt.close(fig)

    for name in ["01_class_balance", "02_amount_distribution",
                 "03_top_features_by_class", "04_correlation_heatmap"]:
        print(f"  saved reports/figures/{name}.png")

    # -----------------------------------------------------------------------
    # Written summary
    # -----------------------------------------------------------------------
    write_summary(facts, top_two)
    rule("DONE")
    print(f"Wrote summary -> {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print("Review the four figures in reports/figures/ alongside the console output above.")


def write_summary(facts: dict, top_two: list[str]) -> None:
    top_signal: pd.Series = facts["top_signal"]
    top_lines = "\n".join(f"| {k} | {v:+.3f} |" for k, v in top_signal.items())
    content = f"""# EDA Summary — Credit Card Fraud Dataset (Milestone 1)

*Auto-generated by `src/explore_data.py`. Numbers are computed from the real data;
the interpretation is what I'd say about them in an interview.*

## Headline numbers

- **Rows:** {facts['n_rows']:,}  |  **Columns:** {facts['n_cols']}
- **Fraud:** {facts['n_fraud']:,} rows = **{facts['fraud_pct']:.3f}%** of all transactions
  (≈ 1 fraud per {facts['n_legit'] / max(facts['n_fraud'],1):,.0f} legit)
- **Missing values:** {facts['total_missing']}
- **Fully-duplicated rows:** {facts['n_duplicates']:,}
- **Time span:** ~{facts['time_span_hours']/24:.1f} days of transactions

## What matters and why

**1. Extreme class imbalance is the whole ballgame.** At {facts['fraud_pct']:.3f}% positives,
accuracy is meaningless — a "never fraud" model scores ~{100 - facts['fraud_pct']:.2f}% accuracy and
catches nothing. Every downstream result is reported as **precision/recall on the fraud class**.
The scarcity of fraud (~{facts['n_fraud']} examples) is also the reason synthetic data might help,
and the thing this project sets out to actually verify.

**2. Data quality is clean except for duplicates.** No missing values. But {facts['n_duplicates']:,}
exact-duplicate rows exist. These are flagged, not yet removed: they can leak across a naive
train/test split (inflating utility scores) and can blur the membership-inference question in
Milestone 5. The handling decision will be made and documented at each of those steps.

**3. Amount and Time are the only raw columns.** `V1`–`V28` are PCA components (centered near zero,
anonymized). `Amount` is heavily right-skewed — a long tail that GANs often reproduce poorly, so it
gets special attention in the fidelity checks. `Time` is just seconds elapsed, not a real clock; a
weak feature.

**4. Signal is concentrated in a handful of PCA components.** Ranked by |correlation with Class|:

| feature | corr with Class |
|---------|-----------------|
{top_lines}

The strongest separators (here **{top_two[0]}** and **{top_two[1]}**) are simultaneously what makes
fraud *detectable* and what the synthetic fraud *must* reproduce. If the CTGAN gets `Amount` right
but flattens these, synthetic data won't help a classifier — exactly the failure mode Milestones 3–4
are built to catch.

## Figures (`reports/figures/`)

1. `01_class_balance.png` — the imbalance, on a log scale.
2. `02_amount_distribution.png` — right-skewed Amount, fraud vs legit.
3. `03_top_features_by_class.png` — the two most fraud-separating components.
4. `04_correlation_heatmap.png` — near-independence of the V-components + the Class row.

## Next (Milestone 2)

Train a CTGAN on the fraud-only rows to generate synthetic fraud, then (Milestone 3) test whether
the synthetic distributions and correlations match the real ones — with statistical tests, not eyeballing.
"""
    SUMMARY_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
