"""
Milestone 5 — Membership inference attack against our own CTGAN.

Run:  python src/membership_inference.py

Threat model: an attacker holds the released synthetic dataset plus some real
candidate records, and wants to decide which candidates were in the GAN's
training set. Membership itself is sensitive here — it means "this transaction
sits in a confirmed-fraud database."

Attack (distance-to-closest-record, DCR): generators gravitate toward their
training data. If the GAN memorized — even partially — synthetic rows will sit
closer to actual training rows (members) than to same-distribution rows it
never saw (non-members). Score every candidate by distance to the nearest
synthetic row(s); small distance => guess "member".

Ground truth we can evaluate against, thanks to the Milestone 2 split:
   members     = 378 real frauds the GAN trained on
   non-members = 95 held-out real frauds it never saw

Design details:
- All preprocessing derives from the SYNTHETIC data only (standardization uses
  synthetic means/stds): the attacker doesn't have the real training set, so
  nothing in the attack pipeline may depend on it. It also prevents Amount's
  scale (~800x V28's) from dominating the distance metric for free.
- Two attack scores: DCR (nearest synthetic neighbor) and mean distance to the
  5 nearest (robust to a single lucky synthetic point).
- Attack quality = ROC AUC over member/non-member labels. 0.5 = coin flip =
  no leak. Significance via Mann-Whitney U (the test behind the AUC).
- Separate memorization scan: minimum member<->synthetic distances, flagging
  near-exact copies — the catastrophic failure mode, distinct from
  statistical leakage.

Outputs:
   reports/privacy_report.md
   reports/figures/10_dcr_distributions.png
   reports/figures/11_attack_roc.png
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SYNTH_PATH = PROJECT_ROOT / "data" / "synthetic" / "synthetic_fraud.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "privacy_report.md"

K_NEIGHBORS = 5
# A synthetic row within this standardized distance of a training row is
# treated as a near-copy in the memorization scan. In 29 standardized
# dimensions, two independent same-distribution points are typically several
# units apart, so 0.5 is a conservative "suspiciously close" line.
NEAR_COPY_THRESHOLD = 0.5


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    synthetic = pd.read_csv(SYNTH_PATH)
    cols = list(synthetic.columns)
    members = df.loc[pd.read_csv(SPLIT_DIR / "member_indices.csv")["index"], cols]
    holdout = df.loc[pd.read_csv(SPLIT_DIR / "holdout_indices.csv")["index"], cols]
    return members, holdout, synthetic


def main() -> None:
    members, holdout, synthetic = load_frames()
    print(f"members={len(members)}, non-members={len(holdout)}, "
          f"synthetic={len(synthetic)}")

    # Attacker-side preprocessing: standardize with synthetic stats only.
    mu = synthetic.mean()
    sigma = synthetic.std().replace(0, 1.0)
    synth_z = (synthetic - mu) / sigma
    members_z = (members - mu) / sigma
    holdout_z = (holdout - mu) / sigma

    # k nearest synthetic neighbors for every candidate row.
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(synth_z.values)
    d_members, _ = nn.kneighbors(members_z.values)
    d_holdout, _ = nn.kneighbors(holdout_z.values)

    labels = np.concatenate([np.ones(len(members)), np.zeros(len(holdout))])

    results = {}
    for name, dm, dh in [
        ("DCR (nearest)", d_members[:, 0], d_holdout[:, 0]),
        (f"mean of {K_NEIGHBORS} nearest", d_members.mean(axis=1),
         d_holdout.mean(axis=1)),
    ]:
        scores = -np.concatenate([dm, dh])  # closer => higher member score
        auc = roc_auc_score(labels, scores)
        u = stats.mannwhitneyu(dm, dh, alternative="less")  # members closer?
        results[name] = {"auc": auc, "p": u.pvalue,
                         "member_median": float(np.median(dm)),
                         "nonmember_median": float(np.median(dh))}
        print(f"\n{name}:")
        print(f"  attack AUC:        {auc:.4f}   (0.5 = no leak)")
        print(f"  Mann-Whitney p:    {u.pvalue:.4f}   (H1: members closer)")
        print(f"  median distance:   members {np.median(dm):.3f} | "
              f"non-members {np.median(dh):.3f}")

    # Memorization scan: closest synthetic row to ANY training row.
    min_d = d_members[:, 0]
    n_near_copies = int((min_d < NEAR_COPY_THRESHOLD).sum())
    print(f"\nMemorization scan (standardized space):")
    print(f"  closest member<->synthetic distance: {min_d.min():.3f}")
    print(f"  members with a synthetic near-copy (<{NEAR_COPY_THRESHOLD}): "
          f"{n_near_copies} / {len(members)}")

    # ------------------------------------------------------------------ figures
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(d_members[:, 0], ax=ax, color="#C44E52", fill=True, alpha=0.4,
                label=f"members (n={len(members)})")
    sns.kdeplot(d_holdout[:, 0], ax=ax, color="#4C72B0", fill=True, alpha=0.4,
                label=f"non-members (n={len(holdout)})")
    ax.set_xlabel("distance to closest synthetic record (standardized)")
    ax.set_title("DCR distributions — overlap means the attack cannot separate them")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_dcr_distributions.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    scores = -np.concatenate([d_members[:, 0], d_holdout[:, 0]])
    fpr, tpr, _ = roc_curve(labels, scores)
    ax.plot(fpr, tpr, color="#C44E52",
            label=f"DCR attack (AUC={results['DCR (nearest)']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="coin flip (AUC=0.5)")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Membership inference attack ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_attack_roc.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------ report
    rows = pd.DataFrame(results).T.round(4)
    content = f"""# Privacy Report — Membership Inference Attack (Milestone 5)

*Generated by `src/membership_inference.py`. Attack: distance-to-closest-record against the
released synthetic set. Candidates: {len(members)} members (GAN training rows) vs {len(holdout)}
non-members (held-out real fraud from the same distribution). All attacker preprocessing derives
from the synthetic data only.*

## Results

{rows.to_markdown()}

- **attack AUC**: probability the attack ranks a random member as "more member-like" than a random
  non-member. 0.5 = no information leaked; 1.0 = perfect de-anonymization.
- **Mann-Whitney p**: significance of members being closer to synthetic rows than non-members.

## Memorization scan

- Closest member↔synthetic distance (standardized): **{min_d.min():.3f}**
- Members with a synthetic near-copy (distance < {NEAR_COPY_THRESHOLD}): **{n_near_copies} / {len(members)}**

## Figures

- `figures/10_dcr_distributions.png` — member vs non-member distance distributions
- `figures/11_attack_roc.png` — attack ROC vs the coin-flip diagonal

## Caveats

- 95 non-members bounds the resolution of the AUC estimate; the p-value carries that uncertainty.
- This is ONE attack (the standard distance-based one). A stronger attacker could train shadow
  models or use density estimates; a null result here means "this generator resists the standard
  attack," not "privacy is proven in general."

*Interpretation recorded separately after inspection; this file records the measurements.*
"""
    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"\nReport -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("Figures -> reports/figures/10..11_*.png")


if __name__ == "__main__":
    main()
