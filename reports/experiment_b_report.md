# Experiment B — Does More Training Fix the Generator? (Follow-up to Milestones 1–6)

*Generated from `src/experiment_b.py` (the sweep) and a leakage-safe rerun of `src/evaluate_utility.py`
against the sweep winner's synthetic output. Same locked 378-member / 95-holdout split as every
prior milestone; same strict, sample-size-scaled fidelity floor from the Milestone 6 correction;
same DCR privacy attack.*

## The question

Milestones 3–5 found that a 500-epoch CTGAN failed fidelity (28/29 columns), failed utility
(hurt a classifier), and passed privacy (no detectable leak) — one mechanism: it never learned the
real distribution, so it had nothing to leak. The obvious follow-up: **train it properly, and see
whether fidelity, utility, and privacy move together.**

## Part 1 — The sweep: four stronger generators, scored on both axes

Four challengers were trained on the identical 378-row member set and scored with the *unmodified*
fidelity and privacy tests — nothing was loosened to help them pass.

| generator | epochs | fidelity: pass / borderline / fail (of 29) | median KS | mean \|Δcorr\| | attack AUC | train time |
|---|---|---|---|---|---|---|
| A (baseline) | 500 | 0 / 1 / 28 | — | 0.211 | 0.510 | 3.6 min |
| B1 | 2,000 | 0 / 1 / 28 | 0.164 | 0.165 | 0.488 | 11.8 min |
| B2 (2x network size) | 2,000 | 0 / 7 / 22 | 0.117 | 0.163 | 0.511 | 18.0 min |
| **B3 (winner)** | **4,000** | **1 / 4 / 24** | **0.113** | **0.160** | **0.503** | 31.2 min |
| B4 (batch 80) | 2,000 | 0 / 4 / 25 | 0.123 | 0.166 | 0.505 | 7.8 min |

**Findings:**

- **More training helped, but only a little, and it plateaued fast.** Median KS improved from
  ~0.28 (A) to 0.113–0.164 (B1–B4) — real progress — but going from 500→2,000 epochs (B1) captured
  most of the gain, and 2,000→4,000 (B3) bought only a small further step. Diminishing returns had
  clearly set in well before any generator reached the passing bar (0.037).
- **Bigger networks improved marginal fit, not joint structure.** B2 (double-size layers) pushed
  seven columns to borderline — the best marginal-distribution result — but its correlation error
  (0.163) was statistically the same as every other challenger's (0.160–0.166). Whatever B2 gained,
  it wasn't a better grasp of how features move together.
- **Correlation structure is the real bottleneck, and nothing touched it.** Every challenger's
  mean |Δcorr| landed in a tight 0.160–0.166 band — a plateau, not a curve. Baseline A's 0.211 fell
  to about 0.16 and then stopped moving regardless of epochs, batch size, or network size. The joint
  structure fraud actually lives in (V17~V18 at r=0.97 in real data) was not something this
  architecture, at this data size, learned to reproduce by training harder.
- **Privacy stayed flat across every configuration.** Attack AUC ranged only 0.488–0.511 across five
  structurally different generators (3.6 to 31.2 minutes of training, single and double-size
  networks, two batch sizes). That's five independent measurements agreeing at "coin flip." The
  fidelity–privacy tradeoff we expected to see **did not appear inside this search space** — fidelity
  moved (modestly); privacy risk did not move at all.

**Winner: B3 (4,000 epochs)** — the only config to fully pass a column, selected by pass-count with
median-KS as tiebreak.

## Part 2 — Full utility test on the winner

B3's synthetic output was run through the complete Milestone 4 gauntlet: leakage-safe split (test
frauds are the 95 GAN-holdout rows, never seen by any generator), 3 seeds, dose-response, and the
random-oversampling control.

| condition | precision | recall | F1 | **average precision** | Δ vs. real-only |
|---|---|---|---|---|---|
| real_only (baseline) | 0.904 ± 0.001 | 0.789 ± 0.011 | 0.843 ± 0.007 | **0.8424 ± 0.0057** | — |
| real + 500 (B3) | 0.872 ± 0.006 | 0.811 ± 0.000 | 0.840 ± 0.003 | 0.8406 ± 0.0076 | −0.0018 |
| real + 1000 (B3) | 0.865 ± 0.010 | 0.811 ± 0.000 | 0.837 ± 0.005 | 0.8338 ± 0.0036 | −0.0087 |
| real + 2000 (B3) | 0.847 ± 0.017 | 0.818 ± 0.006 | 0.832 ± 0.011 | 0.8295 ± 0.0012 | −0.0129 |
| oversample + 1000 | 0.887 ± 0.006 | 0.800 ± 0.000 | 0.841 ± 0.003 | 0.8435 ± 0.0070 | +0.0011 |

**Finding: better fidelity did not translate to utility — the failure mode is nearly identical to
Experiment A.** Average precision declines monotonically with synthetic dose, at a similar rate to
the original 500-epoch generator (compare: A's doses were 0.8424 → 0.8337 → 0.8333 → 0.8308; B3's
are 0.8424 → 0.8406 → 0.8338 → 0.8295). Random photocopying of the real frauds again matches
baseline and beats every synthetic dose. The one column B3 gained on the fidelity test (and the
seven B2 pushed to borderline) evidently weren't the columns a classifier actually needs — the
correlation-structure plateau from Part 1 is the likely explanation: whatever a RandomForest exploits
in this feature space depends on joint structure the generator still hasn't captured.

## Part 3 — Privacy evidence

The DCR (distance-based) membership inference attack — the same test from Milestone 5 — was run
against all five generators in the sweep (table above): **0.488, 0.511, 0.503, 0.505, and the
original 0.510**. All five sit inside noise of a coin flip; none showed any tendency toward
leaking training-set membership, including B3, the fidelity winner.

**Limitation, stated plainly:** this is one attack, run five times. A stronger adversary — a
shadow-model attack, which trains many auxiliary generators to build a dedicated membership
classifier — is the standard the security research community holds generators to, and it was
scoped for this experiment but not completed: an initial run showed each shadow-model training pass
costs several minutes, and building a properly-powered classifier (multiple shadows per candidate
row) was projected at several hours of compute for marginal expected gain, given that five
independent measurements with the simpler attack already agree. The engineering call was to stop
that run and report the DCR consistency honestly, flagging the shadow-model attack as unfinished
future work rather than quietly padding the result. **"No leak detected by the DCR attack across
five generators" is the claim actually supported by this experiment — not "privacy proven."**

## The synthesis

Experiment B set out to test whether pushing a CTGAN harder would trade fidelity for privacy.
Instead it found something more specific: **within this search space (epochs, network width,
batch size, using this architecture on 378 rows), fidelity has a ceiling around "most columns
still fail, joint structure stays broken" — and privacy risk never engaged, because fidelity never
got close enough to real to start memorizing.** The tradeoff we predicted requires a fidelity
regime this sweep didn't reach. Getting there likely needs a different lever entirely — more
training data (not available here — only 378 real fraud rows exist), a different architecture
(TVAE, a Gaussian copula), or accepting that CTGAN on ~400 rows of 29-dimensional data may be
underdetermined regardless of compute.

## Honest limitations of Experiment B specifically

1. **Small search space.** Four configurations, one seed each (not 3 seeds like the utility test) —
   the sweep characterizes trends, not a rigorously bounded optimum.
2. **Shadow-model attack incomplete.** Scoped, started, and honestly stopped at ~1/3 progress when
   the cost/value tradeoff became clear; documented here rather than silently dropped.
3. **Same 378 training rows throughout.** Every challenger saw identical data; the ceiling observed
   may be a data-scarcity ceiling as much as an architecture ceiling — this experiment cannot
   distinguish the two.

## What this changes in the project's overall story

The core narrative from Milestones 1–6 stands and is now more load-bearing, not less: bad fidelity
and good privacy travel together, and this experiment shows that relationship holding even after a
real, honest attempt to break it with more compute. The project's claim was never "this generator
is useless forever" — it's "here is what we measured, and here is exactly how hard we tried to move
it, with all of that shown, including the parts that didn't pan out."
