"""
Shadow-model membership inference attack (the harder privacy test).

Run:  python src/shadow_model_attack.py

Why this matters: The DCR (distance-based) attack we ran in Milestone 5 is the
baseline test. But the research community uses shadow models as the standard
strength test — if a generator passes this, privacy is harder to break.

Attack idea: Train N "shadow" generators on random resamples of the 378-member
fraud set (with and without each row). A classifier trained to distinguish
"trained with row X" vs "trained without row X" from synthetic output behavior
is a much more sophisticated attacker than "which synthetic is closest."

Here: 5 shadow models, with/without each held-out row. Simple: does a logistic
regression trained on synthetic-sample statistics distinguish member-inclusion?

STATUS: incomplete, stopped deliberately (see reports/experiment_b_report.md, Part 3).
Each shadow requires a full CTGAN training run; the full sweep (10 members x 5
shadows x 2 conditions = 100 trainings) was projected at 3-4 hours for a
10/378-member subsample. It was killed at ~33/100 trainings once the DCR attack
had already agreed across five independently-trained generators (0.488-0.511,
all coin-flip) -- that convergent evidence made the marginal value of finishing
this run, on a small subsample, not worth the compute. Left here as a documented
next step, not a finished result: do not cite a score from this script without
re-running it to completion first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
REPORT_PATH = PROJECT_ROOT / "reports" / "privacy_shadow_report.md"

SEED = 42
N_SHADOWS = 5  # few shadow models (compute-limited); even one is a real test
N_SAMPLES_PER_SHADOW = 1000  # synthetic samples from each shadow

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    synth_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    members = df.loc[pd.read_csv(SPLIT_DIR / "member_indices.csv")["index"], synth_cols]
    holdout = df.loc[pd.read_csv(SPLIT_DIR / "holdout_indices.csv")["index"], synth_cols]
    return members, holdout

def train_shadow(data: pd.DataFrame, shadow_id: int, exclude_idx: int | None = None) -> pd.DataFrame:
    """Train one shadow generator on data, optionally excluding one row."""
    if exclude_idx is not None:
        training = data.drop(exclude_idx, errors='ignore')
    else:
        training = data
    torch.manual_seed(SEED + shadow_id)
    np.random.seed(SEED + shadow_id)
    model = CTGAN(epochs=500, batch_size=40, verbose=False, cuda=False)
    model.fit(training, discrete_columns=[])
    return model.sample(N_SAMPLES_PER_SHADOW)

def extract_stats(synthetic: pd.DataFrame) -> np.ndarray:
    """Summary statistics of synthetic data: mean, std, min, max per column."""
    return np.concatenate([
        synthetic.mean().values,
        synthetic.std().values,
        synthetic.min().values,
        synthetic.max().values,
    ])

def main() -> None:
    members, holdout = load_data()
    print(f"Training shadow models (this will take a while)...")
    print(f"Members: {len(members)}, Holdout: {len(holdout)}\n")

    # Collect training data for the membership classifier.
    # X: statistics from shadow synthetic output
    # y: 1 if shadow was trained WITH member[i], 0 if WITHOUT
    X_train, y_train = [], []

    for member_idx in members.index[:10]:  # small subset for speed (full would be len(members))
        print(f"  Member {member_idx}: training shadows with/without...")
        for shadow_id in range(N_SHADOWS):
            synth_with = train_shadow(members, shadow_id, exclude_idx=None)
            synth_without = train_shadow(members, shadow_id, exclude_idx=member_idx)
            X_train.append(extract_stats(synth_with))
            X_train.append(extract_stats(synth_without))
            y_train.append(1)  # trained WITH the member
            y_train.append(0)  # trained WITHOUT the member

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train membership classifier.
    print(f"\nTraining membership classifier on {len(X_train)} shadow samples...")
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])

    # Now: can this classifier tell member vs holdout from our target generator?
    print(f"Membership classifier train AUC: {train_auc:.4f}\n")

    # Load the best real generator (B3).
    target_synth = pd.read_csv(SYNTH_DIR / "expB_B3_4000ep.csv")
    target_stats = extract_stats(target_synth)

    member_score = clf.predict_proba(target_stats.reshape(1, -1))[0, 1]
    holdout_score = clf.predict_proba(target_stats.reshape(1, -1))[0, 1]

    print(f"Target generator membership score: {member_score:.4f}")
    print(f"(0.5 = can't tell member vs non-member, 0.0 = clearly non-member, 1.0 = clearly member)")
    print(f"\nInterpretation:")
    if abs(member_score - 0.5) < 0.1:
        print(f"  -> Classifier is confused (≈ 0.5): generator doesn't leak membership")
    elif member_score > 0.6:
        print(f"  -> Classifier thinks it's member-trained (>{0.6}): potential leak")
    else:
        print(f"  -> Classifier thinks it's non-member-trained (<0.4): generator obscured member")

    (REPORT_PATH).write_text(f"""# Shadow-Model Privacy Attack Results

Membership classifier trained on {len(X_train)} shadow-model comparisons (with/without each member).
Classifier AUC on shadow data: {train_auc:.4f}.

Target generator (B3_4000ep) membership score: {member_score:.4f}

Scores near 0.5 indicate no detectable difference between member and non-member training,
i.e., the generator does not leak membership information.
""", encoding="utf-8")
    print(f"\nReport -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
