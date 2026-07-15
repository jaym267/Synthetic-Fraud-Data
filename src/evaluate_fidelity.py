"""
Milestone 3 — Fidelity evaluation: does the synthetic fraud statistically match real fraud?

Run:  python src/evaluate_fidelity.py

Three-part design (each part exists for a reason):

1. PER-COLUMN KS TESTS, JUDGED BY EFFECT SIZE. The two-sample Kolmogorov-
   Smirnov statistic is the max vertical gap between two empirical CDFs
   (0 = identical, 1 = disjoint). With n=2000 vs n=378 the test is powerful
   enough that trivial differences reach p < 0.05, so p-values alone would
   flag everything. We therefore treat the KS statistic as an effect size and
   apply a Holm correction to the 29 p-values as a supporting signal only.

2. A REAL-VS-REAL NOISE FLOOR. Comparing the GAN's training rows (members,
   n=378) against the held-out real fraud (holdout, n=95) shows how large a
   KS statistic arises from pure sampling noise between two samples of
   GENUINE fraud. A synthetic column near that floor is effectively
   indistinguishable from real; far above it is genuinely wrong. This
   calibration is what makes the numbers interpretable.

3. CORRELATION STRUCTURE. Matching 29 marginal histograms is not enough — a
   classifier consumes JOINT structure. We compare the full Pearson
   correlation matrices (real members vs synthetic) and summarize the
   element-wise absolute differences.

Outputs:
   reports/fidelity_report.md        generated findings (committed)
   reports/figures/05..07_*.png      KS bar chart, worst/best overlays, corr-diff heatmap
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SYNTH_PATH = PROJECT_ROOT / "data" / "synthetic" / "synthetic_fraud.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "fidelity_report.md"

# Same convention as training: KS near the real-vs-real floor is "pass",
# within 2x the floor is "borderline", beyond that is "fail". The floor is
# computed, not assumed.
BORDERLINE_MULTIPLIER = 2.0


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (members, holdout, synthetic) with identical column sets."""
    df = pd.read_csv(DATA_PATH)
    member_idx = pd.read_csv(SPLIT_DIR / "member_indices.csv")["index"]
    holdout_idx = pd.read_csv(SPLIT_DIR / "holdout_indices.csv")["index"]
    synthetic = pd.read_csv(SYNTH_PATH)
    cols = list(synthetic.columns)  # V1..V28 + Amount, the training feature set
    members = df.loc[member_idx, cols]
    holdout = df.loc[holdout_idx, cols]
    return members, holdout, synthetic


def ks_table(members: pd.DataFrame, holdout: pd.DataFrame,
             synthetic: pd.DataFrame) -> pd.DataFrame:
    """Per-column KS: synthetic-vs-members, plus members-vs-holdout noise floor."""
    rows = []
    for col in synthetic.columns:
        ks_synth = stats.ks_2samp(members[col], synthetic[col])
        ks_floor = stats.ks_2samp(members[col], holdout[col])
        rows.append({
            "column": col,
            "ks_synth": ks_synth.statistic,
            "p_synth": ks_synth.pvalue,
            "ks_floor": ks_floor.statistic,  # real-vs-real sampling noise
        })
    table = pd.DataFrame(rows).set_index("column")

    # The floor is measured between samples of (n_members, n_holdout), but the
    # judged comparison runs at (n_members, n_synthetic). Under the KS null the
    # statistic scales like sqrt(1/n + 1/m), so the measured floor must be
    # rescaled to the synthetic pairing's sample sizes; otherwise the floor is
    # ~2x too generous to the synthetic data (it was, before this fix).
    scale = np.sqrt((1 / len(members) + 1 / len(synthetic))
                    / (1 / len(members) + 1 / len(holdout)))
    table["ks_floor_scaled"] = table["ks_floor"] * scale

    # Holm correction across the 29 synthetic-vs-real tests (supporting signal).
    order = np.argsort(table["p_synth"].values)
    m = len(table)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * table["p_synth"].values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    table["p_holm"] = adjusted

    floor = table["ks_floor_scaled"].median()  # robust summary of sampling noise
    def verdict(ks: float) -> str:
        if ks <= floor:
            return "PASS"
        if ks <= BORDERLINE_MULTIPLIER * floor:
            return "borderline"
        return "FAIL"
    table["verdict"] = table["ks_synth"].map(verdict)
    table.attrs["noise_floor"] = floor
    return table.sort_values("ks_synth", ascending=False)


def correlation_comparison(members: pd.DataFrame,
                           synthetic: pd.DataFrame) -> dict:
    """Compare Pearson correlation matrices; return summary + diff matrix."""
    corr_real = members.corr()
    corr_synth = synthetic.corr()
    diff = (corr_real - corr_synth).abs()
    # Off-diagonal entries only (diagonal is trivially 1 in both).
    mask = ~np.eye(len(diff), dtype=bool)
    off_diag = diff.values[mask]
    # The pairs that matter most: strongest real correlations.
    strongest = (
        corr_real.where(mask).abs().unstack().dropna().sort_values(ascending=False)
    )
    top_pairs = []
    seen = set()
    for (a, b), r in strongest.items():
        if (b, a) in seen:
            continue
        seen.add((a, b))
        top_pairs.append({
            "pair": f"{a}~{b}",
            "real_corr": corr_real.loc[a, b],
            "synth_corr": corr_synth.loc[a, b],
        })
        if len(top_pairs) == 8:
            break
    return {
        "mean_abs_diff": float(off_diag.mean()),
        "max_abs_diff": float(off_diag.max()),
        "diff_matrix": diff,
        "top_pairs": pd.DataFrame(top_pairs).set_index("pair"),
    }


def make_figures(table: pd.DataFrame, members: pd.DataFrame,
                 synthetic: pd.DataFrame, corr: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    floor = table.attrs["noise_floor"]

    # Fig 5: KS per column vs the noise floor.
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"PASS": "#55A868", "borderline": "#DD8452", "FAIL": "#C44E52"}
    t = table.sort_values("ks_synth")
    ax.barh(t.index, t["ks_synth"], color=[colors[v] for v in t["verdict"]])
    ax.axvline(floor, color="black", linestyle="--",
               label=f"real-vs-real noise floor ({floor:.3f})")
    ax.axvline(BORDERLINE_MULTIPLIER * floor, color="gray", linestyle=":",
               label=f"{BORDERLINE_MULTIPLIER:.0f}x floor")
    ax.set_xlabel("KS statistic (synthetic vs real members)")
    ax.set_title("Per-column fidelity: KS effect size against the sampling-noise floor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_ks_by_column.png", dpi=120)
    plt.close(fig)

    # Fig 6: overlays for the 3 worst and 3 best columns.
    worst = list(table.index[:3])
    best = list(table.index[-3:])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, worst + best):
        sns.kdeplot(members[col], ax=ax, color="#4C72B0", fill=True,
                    alpha=0.4, label="real", warn_singular=False)
        sns.kdeplot(synthetic[col], ax=ax, color="#C44E52", fill=True,
                    alpha=0.4, label="synthetic", warn_singular=False)
        ax.set_title(f"{col} (KS={table.loc[col, 'ks_synth']:.3f}, "
                     f"{table.loc[col, 'verdict']})")
        ax.legend(fontsize=8)
    fig.suptitle("Worst three (top row) and best three (bottom row) columns")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_worst_best_overlays.png", dpi=120)
    plt.close(fig)

    # Fig 7: correlation difference heatmap.
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr["diff_matrix"], cmap="Reds", vmin=0, vmax=0.5,
                square=True, cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_title("|real corr - synthetic corr| (0 = structure preserved)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_correlation_diff.png", dpi=120)
    plt.close(fig)


def write_report(table: pd.DataFrame, corr: dict,
                 n_members: int, n_holdout: int, n_synth: int) -> None:
    floor = table.attrs["noise_floor"]
    counts = table["verdict"].value_counts()
    n_pass = int(counts.get("PASS", 0))
    n_border = int(counts.get("borderline", 0))
    n_fail = int(counts.get("FAIL", 0))

    ks_md = table.round(4).to_markdown()
    pairs_md = corr["top_pairs"].round(3).to_markdown()

    content = f"""# Fidelity Report — Synthetic vs Real Fraud (Milestone 3)

*Generated by `src/evaluate_fidelity.py`. Samples: {n_members} real members (GAN training set),
{n_holdout} real holdout rows (never seen by the GAN), {n_synth} synthetic rows.*

## Method in one paragraph

Each of the 29 columns is tested with a two-sample Kolmogorov–Smirnov test (synthetic vs real
members), judged primarily by the KS **statistic** (an effect size: max CDF gap, 0–1) because at
these sample sizes p-values flag even trivial differences (Holm-corrected p-values included as a
supporting column). To make the statistics interpretable, a **noise floor** is computed by running
the same KS test between two samples of *genuinely real* fraud — members vs holdout — then rescaled
to the synthetic comparison's sample sizes (KS nulls scale like sqrt(1/n + 1/m); without rescaling
the floor would be ~2x too generous to the synthetic data). Median scaled floor: **{floor:.3f}**. Columns at or below the floor are indistinguishable from real ("PASS");
within {BORDERLINE_MULTIPLIER:.0f}× the floor "borderline"; beyond that "FAIL". Joint structure is
assessed by comparing full Pearson correlation matrices.

## Headline result

**{n_pass} PASS / {n_border} borderline / {n_fail} FAIL** out of {len(table)} columns.

## Per-column KS results (sorted worst-first)

{ks_md}

## Correlation structure

- Mean |Δcorr| across all off-diagonal pairs: **{corr['mean_abs_diff']:.3f}**
- Max |Δcorr|: **{corr['max_abs_diff']:.3f}**

Strongest real-fraud correlations and what the generator produced:

{pairs_md}

## Figures

- `figures/05_ks_by_column.png` — every column's KS vs the noise floor
- `figures/06_worst_best_overlays.png` — density overlays, worst and best columns
- `figures/07_correlation_diff.png` — where joint structure was lost

*Interpretation is written up separately after inspection — this file records the measurements.*
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    members, holdout, synthetic = load_frames()
    print(f"members={len(members)}, holdout={len(holdout)}, synthetic={len(synthetic)}")

    table = ks_table(members, holdout, synthetic)
    floor = table.attrs["noise_floor"]
    print(f"\nReal-vs-real KS noise floor (median): {floor:.3f}")
    print(f"Verdicts: {table['verdict'].value_counts().to_dict()}")
    print("\nPer-column KS (worst 10):")
    print(table.head(10).round(4).to_string())
    print("\nBest 5:")
    print(table.tail(5).round(4).to_string())

    corr = correlation_comparison(members, synthetic)
    print(f"\nCorrelation structure: mean |diff| = {corr['mean_abs_diff']:.3f}, "
          f"max |diff| = {corr['max_abs_diff']:.3f}")
    print("\nStrongest real correlations vs synthetic:")
    print(corr["top_pairs"].round(3).to_string())

    make_figures(table, members, synthetic, corr)
    write_report(table, corr, len(members), len(holdout), len(synthetic))
    print(f"\nReport -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("Figures -> reports/figures/05..07_*.png")


if __name__ == "__main__":
    main()
