"""
LiRA-style shadow-model membership inference attack (the strong privacy test).

Run:  python src/lira_attack.py

Replaces the naive design in shadow_model_attack.py (100 full trainings,
honestly abandoned at 1/3 — see experiment_b_report.md Part 3). Following the
external review's pointer to Carlini et al.'s LiRA: per-example calibration
gets real attack power from a few dozen shadows, and the shadows can be CHEAP
(300 epochs) because we only need the membership signal's distributional
shape, not full fidelity.

Design (offline LiRA, adapted to generators):
  1. Train N_SHADOWS CTGANs, each on a random half (189) of the 378 members.
     Every candidate row (378 members + 95 holdout) is thus OUT of ~half the
     shadows (holdout rows are out of all of them).
  2. Statistic s(row, synth) = negative standardized distance to the nearest
     synthetic row — same attacker knowledge as the Milestone 5 DCR attack
     (preprocessing derived from synthetic data only).
  3. For each candidate row, fit a Gaussian to its OUT-shadow statistics:
     (mu_out, sigma_out). This is the row's personal "I wasn't in training"
     baseline — per-example calibration is what makes LiRA stronger than
     global DCR (a row in a naturally dense region always looks "close";
     LiRA asks whether it looks closer THAN IT SHOULD).
  4. Attack each target generator: z(row) = (s(row, target) - mu_out) / sigma_out.
     Rows far above their own baseline are flagged as members. AUC over
     members (1) vs holdout (0); one-sided Mann-Whitney p-value.

Targets attacked: CTGAN-B3 (Experiment B winner), GaussianCopula (29/29
fidelity), TVAE (best joint structure AND the only DCR AUC that nudged up,
0.544 — the one theory says to worry about).

Resumable: each shadow's synthetic sample + membership mask are cached in
models/lira_shadows/; rerunning skips finished shadows.

Outputs:
  models/lira_shadows/shadow_XX_{synth.csv,mask.npy}
  reports/lira_attack_report.md
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
SHADOW_DIR = PROJECT_ROOT / "models" / "lira_shadows"
REPORT_PATH = PROJECT_ROOT / "reports" / "lira_attack_report.md"

SEED = 42
N_SHADOWS = 24
SHADOW_EPOCHS = 300
SHADOW_BATCH = 40
N_SHADOW_SAMPLES = 1000
MIN_OUT_SHADOWS = 6  # a row needs at least this many out-shadows to be scored

TARGETS = {
    "CTGAN_B3_4000ep": SYNTH_DIR / "expB_B3_4000ep.csv",
    "GaussianCopula": SYNTH_DIR / "bakeoff_copula.csv",
    "TVAE_2000ep": SYNTH_DIR / "bakeoff_tvae.csv",
}


def load_candidates():
    df = pd.read_csv(DATA_PATH)
    cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    members = df.loc[pd.read_csv(SPLIT_DIR / "member_indices.csv")["index"], cols]
    holdout = df.loc[pd.read_csv(SPLIT_DIR / "holdout_indices.csv")["index"], cols]
    candidates = pd.concat([members, holdout], ignore_index=True)
    is_member = np.concatenate([np.ones(len(members)), np.zeros(len(holdout))])
    return members, candidates, is_member


def statistic(candidates: pd.DataFrame, synthetic: pd.DataFrame) -> np.ndarray:
    """s = -standardized NN distance; higher = closer to the synthetic set.
    Standardization uses synthetic-side statistics only (attacker-realistic,
    same as the M5 DCR attack)."""
    mu = synthetic.mean()
    sigma = synthetic.std().replace(0, 1.0)
    nn = NearestNeighbors(n_neighbors=1).fit(((synthetic - mu) / sigma).values)
    d, _ = nn.kneighbors(((candidates - mu) / sigma).values)
    return -d[:, 0]


def train_or_load_shadow(shadow_id: int, members: pd.DataFrame):
    """Returns (synthetic_sample, in_mask over the 378 members)."""
    synth_path = SHADOW_DIR / f"shadow_{shadow_id:02d}_synth.csv"
    mask_path = SHADOW_DIR / f"shadow_{shadow_id:02d}_mask.npy"
    rng = np.random.default_rng(SEED + 1000 + shadow_id)
    in_idx = rng.choice(len(members), size=len(members) // 2, replace=False)
    in_mask = np.zeros(len(members), dtype=bool)
    in_mask[in_idx] = True
    if synth_path.exists() and mask_path.exists():
        print(f"  shadow {shadow_id:02d}: cached, skipping", flush=True)
        return pd.read_csv(synth_path), np.load(mask_path)
    t0 = time.time()
    torch.manual_seed(SEED + 1000 + shadow_id)
    np.random.seed(SEED + 1000 + shadow_id)
    model = CTGAN(epochs=SHADOW_EPOCHS, batch_size=SHADOW_BATCH,
                  verbose=False, cuda=False)
    model.fit(members.iloc[in_mask.nonzero()[0]], discrete_columns=[])
    synth = model.sample(N_SHADOW_SAMPLES)
    synth.to_csv(synth_path, index=False)
    np.save(mask_path, in_mask)
    print(f"  shadow {shadow_id:02d}: trained in {time.time()-t0:.0f}s", flush=True)
    return synth, in_mask


def main() -> None:
    start = time.time()
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    members, candidates, is_member = load_candidates()
    n_cand = len(candidates)
    print(f"{N_SHADOWS} shadows x {SHADOW_EPOCHS} epochs on random halves of "
          f"{len(members)} members; {n_cand} candidate rows\n", flush=True)

    # Per-shadow statistics for every candidate, plus out-membership bookkeeping.
    stats = np.zeros((N_SHADOWS, n_cand))
    out_of_shadow = np.zeros((N_SHADOWS, n_cand), dtype=bool)
    for sid in range(N_SHADOWS):
        synth, in_mask = train_or_load_shadow(sid, members)
        stats[sid] = statistic(candidates, synth)
        out_of_shadow[sid, :len(members)] = ~in_mask   # members out of this shadow
        out_of_shadow[sid, len(members):] = True       # holdout out of all shadows

    # Per-candidate out-distribution Gaussians.
    mu_out = np.full(n_cand, np.nan)
    sigma_out = np.full(n_cand, np.nan)
    n_out = out_of_shadow.sum(axis=0)
    for i in range(n_cand):
        s = stats[out_of_shadow[:, i], i]
        if len(s) >= MIN_OUT_SHADOWS:
            mu_out[i], sigma_out[i] = s.mean(), max(s.std(), 1e-6)
    scorable = ~np.isnan(mu_out)
    print(f"\nscorable candidates: {scorable.sum()}/{n_cand} "
          f"(min out-shadows per row: {n_out.min()})", flush=True)

    # Attack each target.
    results = []
    for name, path in TARGETS.items():
        target_synth = pd.read_csv(path)
        s_target = statistic(candidates, target_synth)
        z = (s_target - mu_out) / sigma_out
        zs, ys = z[scorable], is_member[scorable]
        auc = roc_auc_score(ys, zs)
        # One-sided: are member z-scores stochastically larger?
        p = mannwhitneyu(zs[ys == 1], zs[ys == 0], alternative="greater").pvalue
        # TPR at low FPR — the metric LiRA is actually judged on.
        order = np.argsort(-zs)
        fpr_grid = {}
        for target_fpr in (0.01, 0.05):
            n_fp_allowed = int(target_fpr * (ys == 0).sum())
            fp = tp = 0
            tpr = 0.0
            for idx in order:
                if ys[idx] == 1:
                    tp += 1
                else:
                    fp += 1
                    if fp > n_fp_allowed:
                        break
                tpr = tp / (ys == 1).sum()
            fpr_grid[target_fpr] = tpr
        results.append({"target": name, "lira_auc": auc, "p_one_sided": p,
                        "tpr_at_1pct_fpr": fpr_grid[0.01],
                        "tpr_at_5pct_fpr": fpr_grid[0.05]})
        print(f"  {name:<18} LiRA AUC {auc:.4f}   p {p:.3f}   "
              f"TPR@1%FPR {fpr_grid[0.01]:.3f}   TPR@5%FPR {fpr_grid[0.05]:.3f}",
              flush=True)

    res = pd.DataFrame(results).set_index("target")
    elapsed = (time.time() - start) / 60
    REPORT_PATH.write_text(f"""# LiRA-Style Shadow-Model Attack — The Strong Privacy Test

*Generated by `src/lira_attack.py` in {elapsed:.1f} min. {N_SHADOWS} shadow CTGANs
({SHADOW_EPOCHS} epochs each) trained on random halves of the 378 members; offline LiRA with
per-example Gaussian calibration on each candidate's out-shadow statistics. Statistic and
attacker knowledge identical to the Milestone 5 DCR attack — only the calibration is stronger.*

## Results

{res.round(4).to_markdown()}

- **lira_auc**: 0.5 = coin flip. Per-example calibration removes the "naturally dense region"
  confound that limits global DCR.
- **tpr_at_X%_fpr**: the metric LiRA is properly judged on — how many members can the attacker
  confidently identify while accusing almost no non-members? At 1% FPR, chance level is 0.01.
- Reference DCR AUCs (global attack): CTGAN-B3 0.503, copula 0.498, TVAE 0.544.

## Honest scope

- {N_SHADOWS} shadows at {SHADOW_EPOCHS} epochs (cheap shadows per the LiRA insight that the
  membership signal's shape, not full fidelity, is what matters). Members have on average
  ~{N_SHADOWS//2} out-shadows each; holdout rows have {N_SHADOWS}.
- Shadows model the 500-epoch training regime imperfectly for the 4000-epoch B3 target — a
  standard LiRA assumption violation, noted rather than hidden.
- This completes the work abandoned in `shadow_model_attack.py` (kept for the record).
""", encoding="utf-8")
    print(f"\nDone in {elapsed:.1f} min -> {REPORT_PATH.relative_to(PROJECT_ROOT)}",
          flush=True)


if __name__ == "__main__":
    main()
