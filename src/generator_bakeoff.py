"""
Generator bake-off — is the fidelity ceiling CTGAN's fault, or the data's?

Run:  python src/generator_bakeoff.py

Experiment B found CTGAN's correlation-structure error plateaued at ~0.16 no
matter how much compute we threw at it, while per-column (marginal) fidelity
kept improving. That leaves the project's key open question unresolved:
architecture ceiling, or data-scarcity ceiling (378 rows)?

This script runs the missing arm: hold the data fixed, vary the GENERATOR
FAMILY, and push every candidate through the exact same unchanged evaluation
harness (strict scaled-floor KS verdicts + correlation comparison + DCR attack).

Candidates:
  1. Gaussian copula (hand-rolled, ~40 lines, fits in seconds). The perfect
     foil: it models the correlation matrix DIRECTLY (the thing CTGAN failed
     at) with empirical marginals (the thing CTGAN slowly got better at).
     If the copula nails dcorr but CTGAN couldn't, the failure is localized
     to CTGAN's joint-structure learning, not the data.
  2. TVAE (variational autoencoder from the same ctgan package) — a second
     neural family with a completely different training objective (ELBO, not
     adversarial). If TVAE plateaus at the same dcorr as CTGAN, that points
     back toward the data; if it breaks through, architecture mattered.

Reference rows: CTGAN A (500 ep) and CTGAN B3 (4000 ep, Experiment B winner).

Outputs:
  data/synthetic/bakeoff_<name>.csv        each generator's 2000 rows
  models/bakeoff_results.json              all scores
  reports/generator_bakeoff.md             the comparison table
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from evaluate_fidelity import ks_table, correlation_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
REPORT_PATH = PROJECT_ROOT / "reports" / "generator_bakeoff.md"
RESULTS_PATH = PROJECT_ROOT / "models" / "bakeoff_results.json"

SEED = 42
N_SYNTHETIC = 2000

# Measured reference rows from Experiment A / Experiment B.
REFERENCE_ROWS = [
    {"name": "CTGAN_A_500ep", "family": "GAN", "pass": 0, "borderline": 1,
     "fail": 28, "median_ks": None, "mean_abs_dcorr": 0.211,
     "attack_auc": 0.5096, "fit_seconds": 216},
    {"name": "CTGAN_B3_4000ep", "family": "GAN", "pass": 1, "borderline": 4,
     "fail": 24, "median_ks": 0.1135, "mean_abs_dcorr": 0.1604,
     "attack_auc": 0.5028, "fit_seconds": 1872},
]


class GaussianCopula:
    """Gaussian copula with empirical marginals, hand-rolled per project ethos.

    Fit: map each column through its empirical CDF to uniforms, then through
    the standard normal PPF to z-space; estimate the z-space correlation
    matrix. Sample: draw multivariate normal, map back z -> uniform ->
    empirical quantile. Marginals are reproduced (up to interpolation) by
    construction; the interesting question is whether the Gaussian dependence
    structure captures real fraud's joint behavior.
    """

    def fit(self, data: pd.DataFrame) -> "GaussianCopula":
        self.columns = list(data.columns)
        self.train_values = {c: np.sort(data[c].values) for c in self.columns}
        n = len(data)
        # Empirical CDF ranks -> (rank - 0.5)/n keeps u strictly inside (0,1).
        z = np.column_stack([
            stats.norm.ppf((stats.rankdata(data[c]) - 0.5) / n)
            for c in self.columns
        ])
        self.corr = np.corrcoef(z, rowvar=False)
        return self

    def sample(self, n: int, seed: int = SEED) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(np.zeros(len(self.columns)), self.corr,
                                    size=n, method="cholesky")
        u = stats.norm.cdf(z)
        out = {}
        for j, c in enumerate(self.columns):
            # Inverse empirical CDF with linear interpolation between order stats.
            out[c] = np.quantile(self.train_values[c], u[:, j],
                                 method="linear")
        return pd.DataFrame(out)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    members = df.loc[pd.read_csv(SPLIT_DIR / "member_indices.csv")["index"], cols]
    holdout = df.loc[pd.read_csv(SPLIT_DIR / "holdout_indices.csv")["index"], cols]
    return members, holdout


def attack_auc(members, holdout, synthetic) -> float:
    mu = synthetic.mean()
    sigma = synthetic.std().replace(0, 1.0)
    nn = NearestNeighbors(n_neighbors=1).fit(((synthetic - mu) / sigma).values)
    dm, _ = nn.kneighbors(((members - mu) / sigma).values)
    dh, _ = nn.kneighbors(((holdout - mu) / sigma).values)
    labels = np.concatenate([np.ones(len(dm)), np.zeros(len(dh))])
    return float(roc_auc_score(labels, -np.concatenate([dm[:, 0], dh[:, 0]])))


def score(name: str, family: str, members, holdout, synthetic,
          fit_seconds: float) -> dict:
    table = ks_table(members, holdout, synthetic)
    counts = table["verdict"].value_counts()
    corr = correlation_comparison(members, synthetic)
    row = {
        "name": name, "family": family,
        "pass": int(counts.get("PASS", 0)),
        "borderline": int(counts.get("borderline", 0)),
        "fail": int(counts.get("FAIL", 0)),
        "median_ks": float(table["ks_synth"].median()),
        "mean_abs_dcorr": corr["mean_abs_diff"],
        "attack_auc": attack_auc(members, holdout, synthetic),
        "fit_seconds": round(fit_seconds, 1),
    }
    print(f"  {name:<22} PASS {row['pass']:>2} / bord {row['borderline']:>2} / "
          f"FAIL {row['fail']:>2}   medKS {row['median_ks']:.3f}   "
          f"|dcorr| {row['mean_abs_dcorr']:.3f}   AUC {row['attack_auc']:.4f}   "
          f"({fit_seconds:.0f}s fit)", flush=True)
    return row


def main() -> None:
    members, holdout = load_real()
    print(f"members={len(members)}, holdout={len(holdout)}\n", flush=True)
    rows = list(REFERENCE_ROWS)

    # ---- Gaussian copula --------------------------------------------------
    start = time.time()
    cop = GaussianCopula().fit(members)
    synth_cop = cop.sample(N_SYNTHETIC)
    cop_seconds = time.time() - start
    synth_cop.to_csv(SYNTH_DIR / "bakeoff_copula.csv", index=False)
    rows.append(score("GaussianCopula", "copula", members, holdout,
                      synth_cop, cop_seconds))

    # ---- TVAE -------------------------------------------------------------
    from ctgan import TVAE
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.time()
    tvae = TVAE(epochs=2000, batch_size=40, cuda=False)
    tvae.fit(members, discrete_columns=[])
    synth_tvae = tvae.sample(N_SYNTHETIC)
    tvae_seconds = time.time() - start
    synth_tvae.to_csv(SYNTH_DIR / "bakeoff_tvae.csv", index=False)
    rows.append(score("TVAE_2000ep", "VAE", members, holdout,
                      synth_tvae, tvae_seconds))

    # ---- Results ----------------------------------------------------------
    results = pd.DataFrame(rows).set_index("name")
    RESULTS_PATH.write_text(json.dumps(rows, indent=2))
    print("\n" + results.round(4).to_string(), flush=True)

    REPORT_PATH.write_text(f"""# Generator Bake-off — Architecture vs. Data Scarcity

*Generated by `src/generator_bakeoff.py`. Same 378-member split, same strict scaled-floor KS
verdicts, same correlation comparison, same DCR attack as every prior evaluation — the harness is
unchanged; only the generator family varies.*

{results.round(4).to_markdown()}

## Why these candidates

- **GaussianCopula** models the correlation matrix directly with empirical marginals — the exact
  inverse of CTGAN's failure profile (CTGAN slowly improved marginals while its correlation error
  plateaued at ~0.16). If the copula fixes dcorr on the same 378 rows, the plateau was CTGAN's
  joint-structure learning, not the data.
- **TVAE** is a second neural family (variational objective instead of adversarial). Same plateau
  as CTGAN would point at the data; breaking through would point at architecture.

*Interpretation recorded separately after inspection; this file records the measurements.*
""", encoding="utf-8")
    print(f"\nReport -> {REPORT_PATH.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
