"""
Milestone 2 — Train a CTGAN on real fraud rows and generate synthetic fraud.

Run:  python src/train_ctgan.py

Design decisions (each one is documented because each one is defensible):

1. HOLDOUT SPLIT BEFORE TRAINING. We split the deduplicated fraud rows 80/20:
   - 80% "members"      -> the GAN trains on these.
   - 20% "holdout"      -> the GAN NEVER sees these.
   Why now? Milestone 5's membership inference attack needs real fraud rows the
   generator has never seen (non-members) to compare against rows it has seen
   (members). If we trained on everything today, that experiment would be
   impossible without retraining. This is the same discipline as never touching
   your test set.

2. DEDUPLICATE FIRST. The EDA found exact-duplicate fraud rows. If the same row
   landed in both the member and holdout sets, "seen vs unseen" would be
   corrupted. So duplicates are dropped before the split.

3. DROP `Time`. It's seconds since the first transaction in the collection
   window — an artifact of how the data was gathered, not a property of fraud.
   Synthesizing it would produce meaningless clock offsets. Excluded here and
   in all later milestones, consistently.

4. REPRODUCIBILITY. Fixed seeds for the split and the GAN; the exact member /
   holdout row indices are saved to disk so every later milestone uses the
   same split; training metadata (epochs, batch size, versions) is written to
   a JSON file next to the model.

Outputs (all under models/ and data/synthetic/, both git-ignored except metadata):
   models/ctgan_fraud.pkl            trained generator
   models/training_metadata.json     everything needed to reproduce this run
   data/splits/member_indices.csv    original df indices of GAN training rows
   data/splits/holdout_indices.csv   original df indices of held-out rows
   data/synthetic/synthetic_fraud.csv  generated synthetic fraud rows
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN

# ---------------------------------------------------------------------------
# Configuration — kept in one place so the metadata file can record all of it.
# ---------------------------------------------------------------------------
SEED = 42                 # governs the member/holdout split and torch RNG
HOLDOUT_FRACTION = 0.20   # fraction of deduped fraud rows locked away for Milestone 5
EPOCHS = 500              # small dataset (~370 training rows) => many epochs is cheap
BATCH_SIZE = 40           # must be divisible by pac (CTGAN default pac=10)
N_SYNTHETIC = 2000        # generate plenty; later milestones can subsample
FEATURES_DROPPED = ["Time", "Class"]  # Time: collection artifact; Class: all rows are fraud

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"


def prepare_fraud_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, filter to fraud, dedupe, and split into member/holdout sets."""
    df = pd.read_csv(DATA_PATH)

    fraud = df[df["Class"] == 1]
    print(f"Fraud rows in raw data:            {len(fraud)}")

    # Dedupe on the feature columns we actually train on. Two rows that are
    # identical in every training feature are duplicates for our purposes,
    # even if Time differs (we drop Time anyway).
    feature_cols = [c for c in fraud.columns if c not in FEATURES_DROPPED]
    fraud = fraud.drop_duplicates(subset=feature_cols)
    print(f"After dropping duplicate frauds:   {len(fraud)}")

    # Seeded shuffle, then split. We keep the ORIGINAL dataframe indices —
    # they are the stable IDs that let Milestone 5 map rows back to the raw csv.
    rng = np.random.default_rng(SEED)
    shuffled = fraud.sample(frac=1.0, random_state=SEED)
    n_holdout = int(round(len(shuffled) * HOLDOUT_FRACTION))
    holdout = shuffled.iloc[:n_holdout]
    members = shuffled.iloc[n_holdout:]
    print(f"GAN training set (members):        {len(members)}")
    print(f"Locked-away holdout (non-members): {len(holdout)}")

    return members[feature_cols], holdout[feature_cols]


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for d in (MODEL_DIR, SPLIT_DIR, SYNTH_DIR):
        d.mkdir(parents=True, exist_ok=True)

    members, holdout = prepare_fraud_data()

    # Persist the split so Milestones 3-5 all use the exact same rows.
    pd.Series(members.index, name="index").to_csv(SPLIT_DIR / "member_indices.csv", index=False)
    pd.Series(holdout.index, name="index").to_csv(SPLIT_DIR / "holdout_indices.csv", index=False)
    print(f"Split indices saved to {SPLIT_DIR.relative_to(PROJECT_ROOT)}/")

    # -----------------------------------------------------------------------
    # Train. All 29 columns are continuous (PCA components + Amount), so
    # discrete_columns is empty. CTGAN models each continuous column with a
    # variational Gaussian mixture (mode-specific normalization), which is
    # exactly what multi-modal columns like Amount need.
    # -----------------------------------------------------------------------
    print(f"\nTraining CTGAN: {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"{len(members)} rows x {members.shape[1]} features (CPU)...")
    start = time.time()
    model = CTGAN(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,          # prints generator/discriminator loss per epoch
        cuda=False,
    )
    model.fit(members, discrete_columns=[])
    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed/60:.1f} min")

    model.save(str(MODEL_DIR / "ctgan_fraud.pkl"))

    # -----------------------------------------------------------------------
    # Generate synthetic fraud.
    # -----------------------------------------------------------------------
    synthetic = model.sample(N_SYNTHETIC)
    synthetic.to_csv(SYNTH_DIR / "synthetic_fraud.csv", index=False)
    print(f"Generated {len(synthetic)} synthetic fraud rows -> "
          f"{(SYNTH_DIR / 'synthetic_fraud.csv').relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------------
    # Record everything needed to reproduce or audit this run.
    # -----------------------------------------------------------------------
    import ctgan as ctgan_pkg
    metadata = {
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "n_members": len(members),
        "n_holdout": len(holdout),
        "n_synthetic": N_SYNTHETIC,
        "features_dropped": FEATURES_DROPPED,
        "feature_columns": list(members.columns),
        "training_minutes": round(elapsed / 60, 2),
        "ctgan_version": ctgan_pkg.__version__,
        "torch_version": torch.__version__,
        "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (MODEL_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Metadata -> {(MODEL_DIR / 'training_metadata.json').relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------------
    # Quick sanity check (NOT the Milestone 3 fidelity suite — just a smoke
    # test that the output isn't garbage before we invest in real evaluation).
    # -----------------------------------------------------------------------
    print("\n--- Sanity check: real (members) vs synthetic, key columns ---")
    key_cols = ["V17", "V14", "V12", "V10", "Amount"]
    comparison = pd.DataFrame({
        "real_mean": members[key_cols].mean(),
        "synth_mean": synthetic[key_cols].mean(),
        "real_std": members[key_cols].std(),
        "synth_std": synthetic[key_cols].std(),
    })
    print(comparison.round(3).to_string())
    print("\nInterpret loosely: means/stds in the same ballpark = the GAN learned "
          "*something*. Whether it learned enough is Milestone 3's job.")


if __name__ == "__main__":
    main()
