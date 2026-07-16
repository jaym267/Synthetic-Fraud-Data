"""
Experiment B — Can a better-trained CTGAN reach high fidelity, and what does
that do to privacy?

Run:  python src/experiment_b.py

Experiment A (500 epochs, default architecture) produced synthetic fraud that
failed 28/29 fidelity columns but leaked nothing (attack AUC 0.51). This sweep
trains stronger configurations on the SAME 378-row member split and scores
every one of them on BOTH axes:

  - fidelity: per-column KS pass count at the strict sample-size-scaled noise
    floor (identical logic to evaluate_fidelity.ks_table — the test is not
    loosened for this experiment)
  - privacy:  DCR membership inference attack AUC (identical logic to
    membership_inference.py)

so the sweep output IS the fidelity-privacy tradeoff curve. The best config by
fidelity is saved as the challenger for the full evaluation gauntlet
(fidelity report + utility experiment + privacy report).

Integrity rules:
  - Same member/holdout split as Experiment A (data/splits/*.csv, untouched).
  - Same 2000-sample size, same seeds discipline (one fixed seed per config).
  - Model selection uses members + synthetic only (fidelity vs the training
    rows); the holdout stays out of selection except as the pre-existing
    noise-floor ingredient, and the utility test on the winner remains
    leakage-safe exactly as in Experiment A.

Outputs:
  models/experiment_b/<config>.pkl            each trained generator
  data/synthetic/expB_<config>.csv            each generator's 2000 rows
  models/experiment_b/sweep_results.json      all scores
  reports/experiment_b_sweep.md               the tradeoff table
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# Reuse the EXACT fidelity scoring from Milestone 3 (strict scaled floor).
from evaluate_fidelity import ks_table, correlation_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "experiment_b"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
REPORT_PATH = PROJECT_ROOT / "reports" / "experiment_b_sweep.md"

SEED = 42
N_SYNTHETIC = 2000

# The challengers. Experiment A (500 epochs, dims 256x256, batch 40) is the
# baseline row in the results table, not retrained here.
CONFIGS = [
    {"name": "B1_2000ep",        "epochs": 2000, "batch_size": 40,
     "generator_dim": (256, 256), "discriminator_dim": (256, 256)},
    {"name": "B2_2000ep_big",    "epochs": 2000, "batch_size": 40,
     "generator_dim": (512, 512), "discriminator_dim": (512, 512)},
    {"name": "B3_4000ep",        "epochs": 4000, "batch_size": 40,
     "generator_dim": (256, 256), "discriminator_dim": (256, 256)},
    {"name": "B4_2000ep_batch80","epochs": 2000, "batch_size": 80,
     "generator_dim": (256, 256), "discriminator_dim": (256, 256)},
]

# Experiment A's measured results, for the tradeoff table.
BASELINE_ROW = {"name": "A_500ep (baseline)", "epochs": 500, "pass": 0,
                "borderline": 1, "fail": 28, "median_ks": None,
                "mean_abs_dcorr": 0.211, "attack_auc": 0.5096,
                "minutes": 3.6}


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    synth_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    members = df.loc[pd.read_csv(SPLIT_DIR / "member_indices.csv")["index"], synth_cols]
    holdout = df.loc[pd.read_csv(SPLIT_DIR / "holdout_indices.csv")["index"], synth_cols]
    return members, holdout


def attack_auc(members: pd.DataFrame, holdout: pd.DataFrame,
               synthetic: pd.DataFrame) -> float:
    """DCR membership inference attack — identical to membership_inference.py."""
    mu = synthetic.mean()
    sigma = synthetic.std().replace(0, 1.0)
    nn = NearestNeighbors(n_neighbors=1).fit(((synthetic - mu) / sigma).values)
    dm, _ = nn.kneighbors(((members - mu) / sigma).values)
    dh, _ = nn.kneighbors(((holdout - mu) / sigma).values)
    labels = np.concatenate([np.ones(len(dm)), np.zeros(len(dh))])
    return float(roc_auc_score(labels, -np.concatenate([dm[:, 0], dh[:, 0]])))


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    members, holdout = load_real()
    print(f"members={len(members)}, holdout={len(holdout)}; "
          f"{len(CONFIGS)} configs to train\n")

    rows = [dict(BASELINE_ROW)]
    for cfg in CONFIGS:
        name = cfg["name"]
        csv_path = SYNTH_DIR / f"expB_{name}.csv"
        # Resumable: a completed config's samples are already on disk — score
        # them without retraining (a died/interrupted sweep loses nothing).
        if csv_path.exists():
            print(f"=== {name}: found existing {csv_path.name}, skipping training ===")
            synthetic = pd.read_csv(csv_path)
            minutes = float("nan")
        else:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            print(f"=== {name}: epochs={cfg['epochs']}, batch={cfg['batch_size']}, "
                  f"gen={cfg['generator_dim']} ===", flush=True)
            start = time.time()
            model = CTGAN(
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                generator_dim=cfg["generator_dim"],
                discriminator_dim=cfg["discriminator_dim"],
                verbose=False,
                cuda=False,
            )
            model.fit(members, discrete_columns=[])
            minutes = (time.time() - start) / 60

            synthetic = model.sample(N_SYNTHETIC)
            synthetic.to_csv(csv_path, index=False)
            model.save(str(MODEL_DIR / f"{name}.pkl"))

        # Score on both axes with the unmodified Milestone-3/5 machinery.
        table = ks_table(members, holdout, synthetic)
        counts = table["verdict"].value_counts()
        corr = correlation_comparison(members, synthetic)
        auc = attack_auc(members, holdout, synthetic)

        row = {
            "name": name, "epochs": cfg["epochs"],
            "pass": int(counts.get("PASS", 0)),
            "borderline": int(counts.get("borderline", 0)),
            "fail": int(counts.get("FAIL", 0)),
            "median_ks": float(table["ks_synth"].median()),
            "mean_abs_dcorr": corr["mean_abs_diff"],
            "attack_auc": auc,
            "minutes": round(minutes, 1),
        }
        rows.append(row)
        print(f"  -> PASS {row['pass']} / borderline {row['borderline']} / "
              f"FAIL {row['fail']}   median KS {row['median_ks']:.3f}   "
              f"mean|dcorr| {row['mean_abs_dcorr']:.3f}   "
              f"attack AUC {auc:.4f}   ({minutes:.1f} min)\n", flush=True)

    results = pd.DataFrame(rows).set_index("name")
    (MODEL_DIR / "sweep_results.json").write_text(json.dumps(rows, indent=2))

    # Winner = most PASSes, tiebreak lowest median KS.
    challengers = results.iloc[1:]
    winner = challengers.sort_values(["pass", "median_ks"],
                                     ascending=[False, True]).index[0]
    print("=" * 70)
    print(results.round(4).to_string())
    print(f"\nWINNER (fidelity): {winner}")

    REPORT_PATH.write_text(f"""# Experiment B — Generator Sweep (fidelity vs privacy)

*Generated by `src/experiment_b.py`. Same member split, same strict scaled noise floor and DCR
attack as Experiments A's evaluations — the tests were not modified for this sweep.*

{results.round(4).to_markdown()}

**Winner on fidelity: `{winner}`** — promoted to the full evaluation gauntlet
(fidelity report, leakage-safe utility experiment, full privacy report).

Reading the table: `pass`/`borderline`/`fail` are per-column KS verdicts out of 29 against the
sample-size-scaled real-vs-real noise floor; `attack_auc` is the DCR membership inference attack
(0.5 = no leak). The two columns TOGETHER are the fidelity-privacy tradeoff.
""", encoding="utf-8")
    print(f"Report -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
