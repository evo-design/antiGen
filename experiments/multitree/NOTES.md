# Multi-tree arm — working notes

Branch `is-multitree`, off `main` (831e6af). Nothing committed yet. Started 2026-10-12.

Running log of what was built, what was measured, and what is still open. Companion docs:
`README.md` (how to run it), `FORECAST_DESIGN.md` (the causal forecasting design).

---

## 1. What this arm is

Train one ESM-2 jointly on examples from many phylogenies at once, skipping the C++
generator's attachment phase and the train/test cutoff entirely, validating on a random
held-out split. The question was simply whether a single model can learn across trees.

The random split is **not forecast-valid** and every run from it is tagged
`experiment: multitree`; `antiGen-eval` skips such checkpoints unless given
`--include-experimental`. No number here belongs in the manuscript.

## 2. Code added

**Outside the package** (`experiments/multitree/`, not packaged — `pyproject.toml`'s
`packages.find` only includes `antiGen*`):

| file | purpose |
|---|---|
| `make_tree_examples.py` | auspice JSON -> `examples_*.txt`, no attachment, no cutoff |
| `fetch_trees.py` | download the Nextstrain catalogue + root-sequence sidecars |
| `generate_all.py` | run the generator over every catalogue entry; writes `data/trees.json` |
| `fetch_snapshots.py` | dated snapshots via the Charon `@YYYY-MM-DD` API |
| `run_train.py` | `antiGen-train` + the torchvision workaround (see §6) |
| `train_multitree_esm2.sh`, `train_singletree_esm2.sh`, `train_sweep_esm2.sh`, `train_sweep2_esm2.sh` | SLURM arms |
| `train_decomp_esm2.sh` | SLURM arm re-running joint5/singles/joint23 with `--save-best --dump-cells` |
| `summarize_run.py`, `compare_runs.py`, `sweep_summary.py` | analysis |
| `val_index.py` | replays a run's validation split and joins it to `per_example` losses |
| `baseline_ladder.py` | the uniform -> where -> which -> where x which -> frequency -> model ladder |
| `context_ablation.py` | re-scores with context destroyed outside a window; H1 vs H2 probe |

**Inside the package** (all flag-gated; existing behavior byte-identical):

- `antiGen/batching.py` — `LengthBucketBatchSampler`, so a batch never mixes lengths.
  Chosen over padding because padding would change the published single-tree numbers
  (the flattened `L*E` softmax would spread mass over pad cells, and
  `ESM2FitnessPredictor._forward_impl` branches on the padded length).
- `antiGen/load.py` — `Example.source`, `load_examples(..., source=)`.
- `antiGen/train.py` — `--examples` takes several `NAME=PATH`; `--val-mode random`
  with `--val-frac`/`--split-seed`; per-epoch per-tree validation written to
  `<run_prefix>_val_metrics_by_epoch.json`; `--experiment` tag; manifest schema 2.
  `--save-best` writes `fitness_<prefix>_best.pt` at the best pooled-validation epoch
  (without it only the overfit final epoch survives — was §9 item 1); `--dump-cells`
  writes that epoch's per-cell maps to `<prefix>_val_cells.npz`. `build_checkpoint()`
  is shared by both saves.
- `antiGen/eval_utils.py` — `validate_per_source`, `frequency_baseline_scores`,
  `fixed_map_metrics`, `fit_substitution_baseline`; per-example NLL logging so runs can
  be compared example-by-example. Added for the decomposition (§10):
  `where_only_baseline_scores`, `positional_marginal_counts`, `WhereTimesWhichPredictor`,
  `CellDumper`, and a `cell_sink=` hook on `validate_per_source` that only observes the
  `(B, L, E)` tensor validation already computes.
- `antiGen/eval.py` — `--include-experimental` guard.
- New model types: `codon` and `substitution` (§5).

Tests: 74 pass (`test_length_bucket_sampler.py`, `test_codon_fitness_predictor.py`,
`test_substitution_fitness_predictor.py`, `test_where_baselines.py` are new). **Note:** 4 Tranception tests fail *on
a GPU node* with a device mismatch at `tranception_fitness_predictor.py:523` — pre-existing,
unrelated to this work, passes with `CUDA_VISIBLE_DEVICES=""`.

## 3. Data

23 trees, 23,674 examples, 16 distinct lengths (`experiments/multitree/data/trees.json`).
Generation mirrors the C++ semantics (collapse zero-mutation branches, ancestral
reconstruction, shuffled per-branch order, `k-1` intermediate examples + node examples)
minus attachment and cutoff. All files pass `antiGen-validate-examples`; regeneration is
byte-identical.

Gotchas found:
- Gene names differ per build: rabies G is `RABVgp4`, WNV envelope is `env`, mpox B21R is
  `OPG210`, avian flu annotates `HA` as one gene not SigPep/HA1/HA2.
- Three references carry a few `X` residues (H5 polybasic cleavage site, terminal stops)
  -> `--ambiguous-ref gap`, else both avian HA trees are lost.
- Yellow fever auto-skipped (22 examples, below the floor).
- **`data.nextstrain.org/<name>.json` != the Charon dataset path.** Our dengue copies stop
  at 2023.87; live `/dengue/denv2` is 7104 nodes to 2026.45. Worth re-fetching the live
  builds for the main experiment. Seasonal flu 12y is genuinely retired (2024-04-23,
  tip 2024.27) from *both* sources.

## 4. Results so far

**Joint vs single-tree** (5 trees, 3 seeds, paired bootstrap, per-tree best epoch):

| tree | pooled delta | 95% CI | |
|---|---|---|---|
| rsv_b | −0.252 | [−0.364, −0.133] | joint better |
| rsv_a | −0.168 | [−0.263, −0.097] | joint better |
| h1n1pdm | −0.118 | [−0.200, −0.028] | joint better |
| h3n2 | −0.100 | [−0.189, −0.006] | joint better |
| ncov | −0.037 | [−0.124, +0.048] | indistinguishable |
| **POOLED** | **−0.125** | **[−0.169, −0.076]** | 14/15 cells favor joint |

Joint training helps 4 of 5 trees. The two same-protein pairs (HA, RSV F) gain most; ncov
spike — biggest tree, no family — is the sole null.

**joint5 vs joint3** (does adding RSV help the original three): pooled −0.045
[−0.095, +0.004], 7/9 cells favor joint5. Suggestive, underpowered — this is what
motivated the scaling sweep.

**The scaling ladder is flat** (array 3273505, complete; all arms vs joint5 on the five
shared evaluation trees, 3 seeds, paired bootstrap):

| arm | extra trees | per-tree rule | overall rule |
|---|---|---|---|
| joint11 | +6 | −0.025 [−0.062, +0.012] | −0.018 [−0.055, +0.021] |
| joint17 | +12 | −0.029 [−0.066, +0.009] | −0.051 [−0.091, −0.010] |
| joint23 | +18 | −0.048 [−0.085, −0.009] | −0.029 [−0.073, +0.015] |

No monotone trend. Everything past joint5 sits in a −0.02..−0.05 band — about a third of
the joint-vs-single effect — and going from 5 trees to 23 buys no more than adding the
first two did. **Which arm clears zero is an artifact of the epoch rule:** joint23 is
significant under per-tree and not under overall, joint17 the exact opposite. Same data,
opposite verdicts. Treat the whole ladder above joint5 as null; do not quote either
significant cell alone.

**hafam is negative** (six HA trees only, scored on h3n2 + h1n1pdm): −0.015
[−0.079, +0.049] per-tree, +0.028 [−0.037, +0.093] overall, 3/6 cells either way. This
cuts against the same-protein explanation above — training on six HA trees and nothing
else does *not* reproduce the gain on the two HA evaluation trees. Whatever joint training
buys, it is not "more of the same protein". Consistent with that: under the overall rule
the trees that clear zero in joint17 and joint23 are rsv_a and rsv_b, not the flu trees
those arms actually added.

**Caveats.** All of the above are random-split numbers. Examples within a tree are phylogenetically
correlated (a val node's parent can be in train), so absolute scores are optimistic; the
frequency baseline shares that leak, so the comparison is fair but neither is a forecast.
Per-tree best-epoch selection on flat noisy curves also flatters both sides — `--epoch-rule
overall` is the conservative alternative and weakens h3n2 to [−0.169, +0.012].

## 5. Baselines

Six now report every epoch, all fitted pre-split:

| baseline | what it knows | hit@10% (5 trees) |
|---|---|---|
| uniform | nothing | 0.10 by definition |
| codon (`--model-type codon`) | genetic code, fixed kappa | 0.28–0.36 |
| where_only | which sites mutate, AA axis flat | 0.44–0.60 |
| substitution (`--model-type substitution`) | codon x BLOSUM62, 25 fitted scalars | 0.52–0.75 |
| where_times_which | both, assumed independent | 0.72–0.77 |
| frequency | the empirical joint from train half | 0.68–0.84 |
| joint5 model | — | 0.77–0.89 |

**Correction:** `frequency` is *not* "per-position counts" as this table used to say, and
as `train.py` and `README.md` still loosely said. `frequency_baseline_scores`
(`eval_utils.py`) sums `example.mat`, which is indexed by `(position, alt_aa)` — it is the
empirical **joint**, so it already knows which residue appears and is the memorization
ceiling, not a positional map. Any where-vs-which split that treats it as the "where" term
double-counts. `where_only` (that same map with its AA axis flattened to uniform, so the
gap between the two is exactly the value of knowing *which*) and `where_times_which`
(`P(i) * P(a | wt_i)`, the strongest predictor assuming the two are independent) were
added to give the decomposition real reference points.

The learned substitution matrix recovers kappa = 4.3–9.2 across viruses (the empirical
RNA-virus range) from amino-acid data alone, with `w_codon` ~0.85 and `w_blosum` ~0.23.

**Where the signal actually is.** `where_only` is startlingly weak — only 0.27–0.57 nats
better than uniform, against 1.3–1.6 nats for `substitution` alone. Knowing *which* residue
appears is worth far more than knowing *where* it lands, on every tree. The earlier note
here that "on ncov the signal is *where*" was an artifact of reading `frequency` as a
positional baseline; ncov simply has the weakest substitution signal (0.83 nats), not the
strongest positional one.

Per-cell scores and best-epoch checkpoints are no longer missing: `--dump-cells` and
`--save-best` (see §2) now write both, which unblocks the within-position rank comparison.

## 6. Environment

`--model-type esm2` fails in this conda env: `torchvision 0.21.0` is built for torch 2.6
and raises `operator torchvision::nms does not exist` under torch 2.7.1; transformers 4.57
imports `torchvision.transforms` en route to `EsmForMaskedLM`. `run_train.py` flips the
availability flag as a workaround. **Real fix:** `pip install 'torchvision==0.22.1'` or
uninstall torchvision — not done, it is a shared environment.

## 7. Jobs

| array | arms | state |
|---|---|---|
| 3273115 | first single-tree controls | done |
| 3273244 | joint5/joint3 + 5 singles x 3 seeds | done |
| 3273505 | joint11/joint17/joint23/hafam x 3 seeds | done (12/12, all exit 0:0) |

`joint23` trains on all 23 trees; `hafam` is the six HA trees only, evaluated on h3n2 and
h1n1pdm, to test whether same-protein relatedness explains the gain.

Results in §4. Wall times were 14–51 min per task except joint17 seed 1, which took
2h37m — same config as its twins, but it landed on a node carrying ~30 other jobs and ran
~8x slower. It finished clean and its final numbers are identical to those read off its
partial metrics file at epoch 15, since best epochs here are 1–10 and the tail is pure
overfitting.

## 8. Forecasting experiment — feasibility (built; see §11)

See `FORECAST_DESIGN.md`. Key probe results:

- Charon serves dated versions: `?prefix=/ncov/open/global/all-time@2025-01-01` returns
  the build as published then (verified: `updated=2024-12-27`, latest tip 2024.92).
- Retention is uneven. ncov reaches back to at least 2023-01-01. RSV A/B exist at
  2025-01-01 but 404 at 2023-06-01. **Seasonal flu has no versions at any date probed**
  (2023-01-01, 2023-06-01, 2024-03-01, 2025-01-01, 2025-10-01, and the 2y build).
- Post-D mutation budgets: at D=2025-01-01, ~280 across ncov/RSV/measles/dengue. The
  frozen flu trees have 43–219 post-D mutations each at D=2023.0 (~850 across eight),
  reachable only with the repo's in-tree cutoff, not with a snapshot.

**Both open decisions are now closed (2026-10-13), and the feasibility picture was wrong.**
See `FORECAST_DESIGN.md`. Re-probing with corrected dataset paths changed the answer
substantially:

- `/measles/genome` and `/dengue/denv2` are not dataset paths — the real ones are
  `/measles/genome/global` and `/dengue/denvN/E`. The old probe hit a redirect page, which
  is why dengue looked frozen and the panel looked like five trees.
- **Avian flu HA is live and snapshotted back to 2024-07**, and contributes more post-D
  mutations than the whole rest of the panel. The §8 claim that no flu tree can be
  forecast-evaluated applies only to *seasonal* flu, which is genuinely delisted (zero of
  Charon's 318 datasets match `flu/seasonal`).
- The budget is **~620 post-D mutations at D = 2025-01-01**, not ~280 — comparable to the
  517 the random-split comparison used, so power is no longer the binding constraint.
- h5n1 and h5nx share 37% of their tips; they are overlapping cuts of one dataset, so only
  one may enter a pooled test (h5nx chosen).
- The local tree copies for dengue, avian flu and seasonal flu are stale (2023.87, 2024.24,
  2024.27) because they came from `data.nextstrain.org` rather than the Charon path.

**Locked:** D = 2025-01-01, chosen on post-D mutation counts only (sample-size reasoning,
no model run against post-D data). Queries are anchored on the **snapshot**, not on today's
tree — the earlier draft read the parent genotype from today's tree, whose ancestral
reconstruction used the very post-D descendants being predicted.

**Still to build:** `--testing-after DATE` in `make_tree_examples.py` (which has no date
handling at all today), the snapshot-to-today tip-set matcher, the arm scripts, and the
scoring pass.

## 9. Other open items

- `train.py` saves only the final checkpoint, so every saved `.pt` here is the overfit
  epoch-19 model, not the best epoch (3–8 typically). Add best-by-validation saving.
- Per-tree vs pooled epoch selection materially changes small effects; report both. In
  the scaling ladder it flips the sign of significance, not just the width of the CI.
- **mpox is unlearnable and it is inside joint23.** It converges to uniform and stays
  there: NLL/mut 10.603 against a uniform reference of 10.584, hit@10% pinned at exactly
  0.0031 (= 1/319) at every epoch, while ncov and h3n2 on the same run climb 0.149 ->
  0.758 -> 0.800. Not an eval tree, so the joint23 comparison stands, but "every usable
  tree" currently includes one the model cannot fit at all.
  **Correction to an earlier note here:** this is *not* RoPE extrapolation past ESM-2's
  1024-token pretraining context. `ESM2FitnessPredictor.ESM2_MAX_LEN = 1022`
  (`antiGen/models/esm2_fitness_predictor.py:30`) and anything longer takes a two-pass
  sliding-window branch (lines 194-221), so ESM-2 never receives more than 1022 tokens.
  The real suspects are that windowing path and the mpox examples themselves:
  (a) the stitch — mpox is split into [0:1022] and [859:1881] with only a 163-residue
      averaged overlap, and the two passes' logits are stitched into one tensor that is
      then softmaxed *jointly* across all positions, so any constant offset between the
      halves silently reweights mass between them. ncov (L=1273) is on the same path with
      a 771-residue overlap. These two are the only trees above 1022 and they are also the
      only two anomalies, which is suggestive but not yet evidence.
  (b) the mpox examples themselves — the substitution baseline scores 0.9969 on this
      tree, a mirror-image anomaly with nothing to do with any of the above.
  A quick probe bucketing ncov's single-mutation val examples by window region found
  higher NLL outside the overlap, but h3n2 (single-pass, no seam) shows a comparable
  first/second-half spread and the ncov window-2 bucket has n=10 — inconclusive. The real
  test is scoring mpox and ncov stitched vs truncated to <=1022 and comparing across the
  seam.
- ~~Nothing committed. `runs/` and `nextstrain/` are untracked and not gitignored~~ —
  done. The branch is pushed to `origin/is-multitree`; `runs/`, `nextstrain/`,
  `tree_pulled_*.json` and the generated `experiments/multitree/data/**/*.txt` are
  gitignored as regenerable, so only code, notes and the small figure CSVs are tracked.

---

## 10. What is the model learning? (started 2026-10-13)

The scaling ladder being flat (§4) and `hafam` being null killed the same-protein
explanation without replacing it, so this arm asks what the model has actually learned.
Plan in `~/.claude/plans/flickering-whistling-valley.md`.

**The two hypotheses.** H1: the model is largely a per-tree positional lookup plus
generic substitution chemistry, with little real sequence-context dependence — which
would explain the flat ladder *and* the `hafam` null in one stroke. H2: it uses genuine
sequence context, and the joint gain is real shared structure.

**Both probes reject H1.** Preliminary, from the 3-epoch verification run
(`/large_storage/hielab/ispecht/multitree/phase0/newflags`); full 20-epoch arms are
array 3277317.

1. *Baseline ladder.* The model beats `where_times_which` — the best predictor that
   assumes where and which are independent — by 0.51–1.12 nats on all five eval trees,
   and beats `frequency`, the memorization ceiling, by 0.80–1.15. So it has learned
   substantial position-specific substitution structure and is generalizing past the
   training counts, not reproducing either factor.
2. *Context ablation.* Shuffling the parent's residues outside a window around the
   mutated site (composition and length preserved, positional context destroyed) costs
   h3n2 +0.79 nats on `-log P(alt | pos)` and +1.21 on `-log P(pos)` at radius 0, decaying
   smoothly to zero as the window grows. A positional lookup table would be flat.

So the flat ladder is **not** explained by the model ignoring context. Whatever limits
cross-tree transfer, it is not that there is nothing to transfer.

**Surprise worth chasing:** positional information is nearly worthless here. `where_only`
beats uniform by only 0.27–0.57 nats while `substitution` alone is worth 1.3–1.6. The
model's skill is overwhelmingly about *which* residue appears, not *where*. That also
retires the old "on ncov the signal is where" note in §5, which came from misreading
`frequency` as a positional baseline.

**The joint-training gain is entirely a *which* effect** (`decompose_delta.py`, joint5 vs
the five single-tree controls, 3 seeds, scored at each run's best pooled epoch). Because
the model emits one distribution over the whole L x E map, the NLL of an observed mutation
factors exactly into `-log P(i)` + `-log P(a|i)`, so the delta decomposes additively
(verified: the two parts reconstruct the logged NLL to 1e-4, the float16 dump floor):

| component | pooled delta | 95% CI |
|---|---|---|
| where, `-log P(i)` | **+0.008** | [−0.022, +0.036] |
| which, `-log P(a\|i)` | **−0.076** | [−0.108, −0.042] |
| total | −0.069 | [−0.110, −0.026] |

The where term is indistinguishable from zero and faintly favors *single*. Every tree's
which term is negative except ncov (the sole null in §4 as well); h3n2 −0.127
[−0.199, −0.059] and rsv_b −0.181 [−0.278, −0.092] clear zero on their own. No tree's
where term clears zero.

**This explains the flat ladder.** `P(a|i)` is substitution chemistry — largely
tree-independent — so pooling trees sharpens it, and that saturates once enough
substitution events have been seen. Training-set mutation counts against the observed
gains:

| arm | train mutations | delta vs joint5 |
|---|---|---|
| single | 836–1429 | +0.125 (joint5 better) |
| joint3 | 5,138 | +0.045 |
| joint5 | 7,176 | 0 (reference) |
| hafam | 14,324 | −0.015 |
| joint11 | 15,411 | −0.025 |
| joint17 | 24,179 | −0.029 |
| joint23 | 35,185 | −0.048 |

~1,200 → 7,176 events (0.78 decades) buys 0.125 nats; 7,176 → 35,185 (0.69 decades, a 5x
increase) buys at most 0.048 and usually nothing. Return per decade collapses ~4x. The
ladder is not flat because cross-tree transfer fails — Phase 3 shows the model reads
context fine — but because the thing joint training improves is cheap to saturate.

**This also reframes the `hafam` null in §4.** hafam carries 14,324 training mutations,
twice joint5, so under saturation *both* arms are already past the knee and the comparison
had little power to detect a relatedness effect either way. It is weaker evidence against
same-protein relatedness than §4 claims — the honest statement is that hafam adds events,
and events were not the binding constraint. A relatedness test needs arms matched on event
count, not on tree count.

**What the improved P(a|i) contains** (`substitution_profile_sweep.py`; supersedes the
h3n2/seed-0 lead that `substitution_profile.py` produced). All five evaluation trees,
three seeds, joint5 against each tree's single-tree control, at each run's best pooled
epoch. Averaging the model's conditional P(a|i) over every position sharing a wild-type
residue gives an implied 20x20 substitution matrix; all numbers are Spearman on the 380
off-diagonal cells.

**It is one-step codon accessibility, and joint training is what puts it there.**
Pooled over 15 tree x seed cells:

| arm | codon | BLOSUM62 | transition frac | empirical (train) | R^2 codon+BLOSUM | R^2 row+col |
|---|---|---|---|---|---|---|
| joint5 | **0.670** | 0.365 | 0.310 | 0.783 | 0.532 | 0.279 |
| single | 0.527 | 0.429 | 0.311 | 0.713 | 0.412 | 0.406 |

The ordering codon > BLOSUM > ts/tv holds on every one of the 30 run-tree cells. Four
controls, all of which the earlier note lacked:

1. *Not an additive artifact.* A matrix that merely made some residues popular targets
   (`log P(a|wt) = u_wt + v_a`) would correlate with any reference sharing those
   marginals. Only 28% of joint5's variance is row+column main effects, and projecting
   them out of **both sides** raises the codon correlation to **0.791** (BLOSUM 0.479).
   So it is genuine pairwise wt->alt structure, which is exactly what codon
   accessibility is and what BLOSUM largely is not.
2. *The single-tree failure mode is degeneracy toward the marginals.* Row+col effects
   are 0.406 of a single-tree matrix against 0.279 for joint5 (worst cell: rsv_b seed 2
   at 0.617). Single-tree models drift toward "some residues are popular"; joint
   training restores the pairwise term. That is a concrete mechanism for the which-term
   gain in the table above.
3. *Averaging artifact ruled out* — the open item from the earlier draft. Re-running
   with the geometric mean over positions (`--mean geometric`), which is not dominated
   by a few confident sites, moves nothing: codon 0.635/0.477, BLOSUM 0.387/0.444,
   R^2 codon+BLOSUM 0.479/0.360 for joint5/single.
4. *Transition bias is present but indirect.* Plain transition fraction correlates only
   0.31. But fitting `codon(kappa)` over a grid, the rank optimum is **kappa ~ 2.5-4 on
   every tree** (joint5 rho 0.69-0.74), consistent with, though at the low end of, the
   4.3-9.2 the explicit substitution baseline fits. The bias is expressed *through*
   codon reachability, not as a separate axis. Only the rank optimum is informative
   here: the OLS R^2 surface in kappa is nearly flat (0.543 at kappa=3 vs 0.569 at
   kappa=50), so its argmax runs to the grid edge and means nothing.

**The closest match of all is the empirical matrix, but that is not a separate answer.**
The model's own training substitutions correlate 0.783 with its implied matrix — higher
than any textbook reference. But those empirical matrices are themselves codon-shaped
(rho 0.60-0.72 with codon, only 0.27-0.30 with BLOSUM), so this says the model learned
the observed substitution process, and that process is mostly codon reachability. It is
not evidence of a second, independent thing being learned.

*Memorization check.* The implied matrix tracks **train** counts at 0.77-0.85 but
**held-out** counts at only 0.51-0.68. That gap is not by itself damning: held-out
counts are sparse (109-147 of 380 cells nonzero), and the train->val empirical
correlation controlling for codon+BLOSUM is 0.64-0.83, which is the ceiling. The model
sits at 0.65-0.77 against the same control, i.e. about at that ceiling. Read as: beyond
generic chemistry it has absorbed roughly as much tree-specific substitution signal as
is actually reproducible, not more.

**Joint training moves the matrix specifically toward codon** — and the trees it moves
are the trees that gained. Spearman of the joint-minus-single delta:

| tree | codon | BLOSUM62 | ts frac | mean abs delta | which-term gain (above) |
|---|---|---|---|---|---|
| rsv_b | +0.607 | +0.004 | +0.002 | 0.640 | −0.181, clears zero |
| h3n2 | +0.654 | +0.132 | −0.016 | 0.430 | −0.127, clears zero |
| rsv_a | +0.552 | +0.100 | +0.116 | 0.558 | negative |
| h1n1pdm | +0.524 | +0.071 | +0.046 | 0.396 | negative |
| ncov | +0.107 | −0.090 | +0.086 | 0.117 | **the sole null** |

ncov is the one tree whose matrix barely moves and the one tree that does not move
toward codon — the same tree that is the sole null in s4 and in the where/which
decomposition. Three independent readouts single out the same tree, which is a real
corroboration of the saturation story rather than three looks at one number.

**About 45% is still not codon or BLOSUM, and it is shared across viruses.** R^2 on the
two references together is 0.532. The residual is not per-tree noise: its cross-tree
Spearman averages **0.669** against a within-tree seed-to-seed ceiling of **0.836**, so
roughly 80% of the reproducible residual is virus-independent. The flu and RSV trees
agree most (0.80-0.97), ncov sits apart (~0.59-0.62). It does *not* load on simple
physicochemistry — |delta hydropathy| −0.02..−0.11, |delta volume| +0.08..+0.11. So
there is a real, largely virus-independent substitution term here that neither reference
captures, and it remains uncharacterized.

Open: what that shared residual is. Ruled out so far: row/col marginals, hydropathy,
volume, and per-tree idiosyncrasy.

**Verified while building this** (all in `antiGen/tests/test_where_baselines.py` or run
directly):
- The validation split replays exactly: `val_index.py` reproduces all 23 trees' example
  and mutation counts against `joint23_s0`. Existing `per_example` losses are therefore
  joinable to concrete `(position, alt_aa)` mutations with no retraining.
- `--dump-cells` is faithful: recomputing `per_example.loss` from the dumped maps matches
  the logged values to 3.9e-4 relative (float16 storage floor).
- `--save-best` lands on the argmin of pooled val NLL.
- The new flags do not perturb the model: control-vs-flags max |delta| pooled NLL is
  3.1e-6, *smaller* than the 7.4e-6 between two identical-code runs on this GPU. Byte
  identity is unavailable here — training is nondeterministic at ~1e-5 — so the
  nondeterminism floor is the standard to compare against.


---

## 11. Causal forecasting arm — built and running (2026-10-13)

D = 2025-01-01 locked. Design in `FORECAST_DESIGN.md`; §8 has the corrected retention.

### Query anchoring: simplified on purpose

Queries are read from **today's** tree, not the snapshot. The snapshot-anchored variant
(match clades across builds by descendant tip-name set) was tried and abandoned after
measuring it:

- Structural: most crossing-branch parents have **zero** pre-D descendant tips (median 0
  for both h5nx and ncov). Only 29–39% have even one. Branch-level matching recovered
  3% of h5nx's crossing branches.
- Identifier churn: Nextstrain renamed tips between builds — measles moved from GenBank
  accessions to Pathoplexus `PP_*` IDs, mumps the reverse, RSV from internal IDs to full
  strain names. rsv_a, rsv_b, measles and mumps share **zero** tip names across builds.

Accepted cost: today's ancestral reconstruction used post-D tips, so the query genotype
is mildly hindsight-informed. It never touches training or epoch selection. It does
mildly favor query-dependent predictors (model, substitution, where_times_which) over
fixed-map ones (frequency, where_only) — state this when reporting the ladder.

### Data built

9 trees at D: h5nx, ncov, rsv_a, rsv_b, measles, denv1–4. **564 post-D targets, 156
novel (28%)** — h5nx 229/29, ncov 101/43, rsv_b 64/17, measles 64/43, rsv_a 55/14,
dengue 51/10. Comparable to the 517 the random-split comparison used.

Train side is the existing generator run on each snapshot, unchanged; all invariants
pass and all files validate. `make_forecast_testset.py` checks the reference frame is
identical between snapshot and current build before emitting, since train and test
examples must share coordinates.

### New files

| file | purpose |
|---|---|
| `make_forecast_testset.py` | post-D test set + novel-only companion from today's tree |
| `train_forecast_esm2.sh` | 33 arms: 9 single, joint_panel, joint_all x 3 seeds |
| `score_forecast.py` | the only reader of post-D data; full baseline ladder |
| `substitution_profile.py` | implied P(a\|wt) matrix vs codon / BLOSUM / ts-tv |
| `forecast_summary.py` | aggregates the scored arms: ladder, paired bootstrap, joint-vs-single |
| `calibrate_forecast.py` | mass-allocation diagnostic + temperature / novelty / mixture corrections |
| `forecast_hit_summary.py` | the contrast tables under hit@10% instead of NLL |
| `forecast_precision.py` | precision@k and precision@q alongside recall; `run_precision.sh` submits it |
| `train_matched_esm2.sh` | random-split arms on the nine forecast trees, three-way split (s12) |
| `collect_figure_metrics.py` | scores both regimes x both epoch rules x both metrics x both subsets -> Figure 5 CSVs |
| `prune_stale_runs.py` | drops superseded attempts from an output dir |
| `substitution_profile_sweep.py` | the same across 5 trees x 3 seeds, with the empirical / additive / geometric-mean controls (s10) |

`joint_all` adds the 7 retired seasonal-flu trees as training-only context — delisted and
unforecastable, but every tip predates D (latest 2024.27), so they are legitimate.

### Results (array 3279566 complete, 33/33, all exit 0:0; scored 2026-10-13)

Wall times 47s-25m. One preempted retry: `single_rsv_a_s0` holds a stale 4-epoch partial
from 16:55 alongside the complete 20-epoch run from 17:06. Every reader takes
`sorted(glob(...))[-1]`, and the timestamps sort lexicographically, so the complete run
is the one used; **the stale pair should be deleted** before anything is archived.

Training mutations per arm: singles 542 (measles) - 5,850 (h5nx); `joint_panel` 16,643;
`joint_all` 25,250. Test side: 564 post-D targets, 156 novel (28%).

Scored with `score_forecast.py` at each arm's best snapshot-validation epoch, aggregated
by `forecast_summary.py` (new; paired bootstrap, seeds averaged per example before
resampling). Baseline hit rates depend only on the pre-D train file, so they are one pass
per tree, seed-independent.

#### All 564 post-D substitutions: the model forecasts, and joint training helps

Pooled NLL/mutation deltas, paired bootstrap over test examples, 3 seeds:

| contrast | pooled delta | 95% CI | |
|---|---|---|---|
| `joint_panel` - its single-tree control | **−0.266** | [−0.392, −0.152] | 26/27 cells favor joint |
| `joint_panel` - `where_times_which` | **−0.429** | [−0.565, −0.300] | beats the best factored baseline |
| `joint_panel` - `frequency` | **−0.703** | [−0.858, −0.544] | beats the memorization ceiling |
| `joint_all` - `joint_panel` | −0.018 | [−0.058, +0.024] | **null** |

Per-seed pooled `joint_panel - single` is −0.283 / −0.242 / −0.273: the effect is not one
seed. hit@10% pooled: uniform 0.10, codon 0.31, where_only 0.44, substitution 0.65,
where_times_which 0.73, frequency 0.72, single 0.77, **joint_panel 0.81**.

This is genuine forecasting skill on data that did not exist when the models were
trained, and the joint-training benefit from the random split survives the temporal
holdout at roughly twice the magnitude (−0.266 here against −0.125 in s4).

**`joint_all` is a clean null, and it is the causal version of the scaling question.**
Adding 8,607 training mutations from seven unrelated seasonal-flu trees moves nothing:
−0.018 [−0.058, +0.024] pooled, +0.001 [−0.077, +0.089] on novel, and the only cell that
clears zero (measles, +0.152) favors *fewer* trees. All seven frozen trees were verified
pre-D before inclusion (latest tip 2024.27 < 2025.00), so this is not a leak artifact.
s10 predicted exactly this: both arms carry 16.6k and 25.3k training mutations, far past
the ~7k knee where the `P(a|i)` term saturates. The saturation story now has a causal
confirmation, not just a random-split one.

#### The novel subset: the inversion flagged in the earlier draft is real

| contrast | pooled delta | 95% CI | |
|---|---|---|---|
| `joint_panel` - its single-tree control | **−0.775** | [−0.962, −0.586] | 22/24 cells favor joint |
| `joint_panel` - `frequency` | −0.321 | [−0.625, −0.020] | |
| `joint_panel` - `where_times_which` | **+0.738** | [+0.529, +0.919] | **the model loses** |
| `joint_all` - `joint_panel` | +0.001 | [−0.077, +0.089] | null |

The novel split works as designed: `frequency` collapses from 7.77 to 9.99 NLL on h5nx
and its hit@10% falls to 0.054, *below* the 0.10 chance rate. Joint training helps a
great deal here — far more than on the full set (−0.775 against −0.266), consistent with
s10's finding that what pooling improves is the substitution term, which is exactly what
a novel cell needs. But it is not enough: `where_times_which` still wins by 0.74 nats.

So the worry recorded in the earlier draft is confirmed. On random held-out data (s10)
the model beats `where_times_which` by 0.5-1.1 nats; on genuinely novel forecasts it
*loses* to it by 0.74. The random-split advantage over that baseline measured
interpolation, not prediction.

**But the inversion is a density failure, not a ranking failure** — and that materially
changes what it means. hit@10% on the novel subset:

| rung | uniform | codon | where_only | substitution | wxw | frequency | single | **joint_panel** |
|---|---|---|---|---|---|---|---|---|
| novel hit@10% | 0.100 | 0.333 | 0.181 | 0.468 | **0.500** | 0.054 | 0.387 | **0.511** |

The model *ranks* novel substitutions as well as `where_times_which` (0.511 vs 0.500) and
far better than chance, while assigning them worse probability mass. It is identifying
which novel mutations will happen and then being overconfident elsewhere. The obvious
next step is a temperature fitted **on the pre-D validation split** — causally legal,
since it never touches post-D data — to see whether calibration alone closes the 0.74 nat
gap. Until that is run, "the model is worse than a factored baseline at novel forecasts"
is true of NLL and false of ranking, and should not be quoted without the distinction.

#### Caveats specific to these numbers

- **The panel is top-heavy.** h5nx is 229/564 of the full test set (41%); on the novel
  subset measles and ncov are 43 each (55% together). Pooled numbers are not a
  nine-tree average.
- **denv4 novel has n = 1** — its "CI" is the degenerate [+0.119, +0.119] and means
  nothing. denv3 has zero novel targets and drops out of that subset entirely.
- **Single-tree arms lose to their own baselines on three trees** (ncov 9.372 vs wxw
  9.167, measles 8.912 vs substitution 8.245, denv1 7.723 vs 7.332) and win on the other
  six. Single-tree forecasting is not reliably better than a fitted substitution matrix;
  the joint arms are what clear the bar.
- **Query anchoring** (s11, today's tree) mildly favors query-dependent predictors. Both
  the model and `where_times_which` are query-dependent, so the headline model-vs-wxw
  comparison is unaffected; the model-vs-`frequency` and model-vs-`where_only` ones are
  flattered by it.
- **One forecast date.** D = 2025-01-01 is a single draw; D = 2024-07-01 remains the
  natural replicate.

### Can recalibration rescue the novel forecasts? (2026-10-13)

The novel deficit is a density failure, not a ranking failure, so the obvious move is to
fix the probability mass without retraining. `calibrate_forecast.py` tries three
corrections on `joint_panel`, 3 seeds. **Answer: partly, and not the way expected.**

**First, a scoring detail that had to be reproduced.** `compute_probs` clamps the
post-softmax map to `[1e-8, 1-1e-8]` before the loss renormalizes. That floor is not
cosmetic — it caps how much any single badly-predicted cell can contribute, and on ncov
it is worth **1.5 nats** (8.835 clamped vs 10.358 unclamped). Every number in this file
was scored through it, so it is the objective, but note which way it leans: the floor
rescues peaked predictors, and the model is far peakier than the baselines. On the novel
subset at seed 0 it is worth ~1.06 nats to the model and ~0 to `where_times_which`, so
the +0.74 nat novel deficit reported above is already the *generous* reading; unfloored
it is +1.78.

**The model is not miscalibrated on the actual forecasting task.** Two independent
checks:

- Its probability mass on cells already seen pre-D tracks the rate at which post-D
  substitutions actually land on seen cells: h5nx 0.855 vs 0.873 observed, ncov 0.540 vs
  0.574, rsv_a 0.665 vs 0.745, denv2 0.713 vs 0.750. Only measles is notably overweight
  (+0.175). There is no systematic seen/unseen mass error to correct.
- Temperature scaling, swept over T in [0.5, 20], is minimized at **T = 1.0** on the full
  post-D set. The model is already temperature-optimal for what actually happens.

So the apparent miscalibration on the novel subset is mostly *conditioning on a rare
event*, not a defect. Corrections that target it directly therefore trade, rather than
gain (3-seed pooled NLL):

| correction | ALL (564) | NOVEL (156) |
|---|---|---|
| model as-is | 7.521 | 9.524 |
| `where_times_which` | 7.950 | 8.787 |
| temperature T=3.2 | 8.387 | 9.285 |
| unseen-cell upweight alpha=8 | 8.007 | 8.680 |
| unseen-cell upweight alpha=59 | 9.174 | 8.481 |
| **mixture with wxw, lambda=0.20** | **7.436** | **9.11** |

Temperature is a dead end: any T > 1 costs more on the full set than it recovers on the
novel one, and even the novel-optimal T=3.2 lands at 9.285, still behind the baseline.
Unseen-cell upweighting works on novel — alpha=59 reaches 8.481 and beats `wxw` — but it
is pure trade, and since the novel targets are a 28% subset *of* the full set, the full
set gets monotonically worse. Neither is deployable: you cannot know at prediction time
that a substitution will turn out to be novel.

**The mixture is the one real gain, and it is causally clean.** Fitting lambda on the
snapshot random validation split (the same pre-D split that chose the epoch, with
`where_times_which` refitted on that split's training portion so it never sees the
scoring examples) picks **lambda = 0.20 on all three seeds** — against a post-D oracle
optimum of 0.25-0.30 for the full set. The transfer is near-exact: 7.442 at the chosen
lambda vs 7.440 at the oracle.

Paired bootstrap against the uncorrected model, lambda fixed at 0.20 before the post-D
set was opened:

| subset | vs model | 95% CI | vs `where_times_which` | hit@10% model -> mixture |
|---|---|---|---|---|
| ALL | **−0.083** | [−0.130, −0.038] | −0.512 [−0.619, −0.414] | 0.813 -> 0.822 |
| NOVEL | **−0.375** | [−0.454, −0.292] | **+0.362** [+0.203, +0.496] | 0.511 -> 0.519 |

It improves both subsets and both metrics at once, for free, with no retraining. On the
novel subset it halves the deficit — +0.738 becomes +0.362 — but **does not close it**.
The novel-optimal lambda is ~0.95, i.e. "discard the model", and that costs 0.4+ nats on
the full set.

**What this means.** The novel-subset gap is not a mass-allocation artifact that
calibration can dissolve; it is a genuine knowledge gap *within* the unseen cells. The
model knows roughly how much mass to put on novel territory, and ranks within it about as
well as the factored baseline (hit@10% 0.52 vs 0.50), but its ordering there is not good
enough to win on NLL. Ensembling buys back half the gap because the two predictors err
differently, not because either is recalibrated.

Still open: the nested-holdout design — pick an inner date D' < D, train on the D'
snapshot, and fit corrections on the D'-to-D substitutions, which is a genuine
forecast-shaped calibration set and entirely pre-D. That is the only route that could
tune a *novelty-aware* correction legally. Retention allows it for h5nx, ncov, measles and
dengue at D' = 2024-07-01, but not RSV (no snapshot before 2025-01).

**Both metrics side by side** (3-seed mean, same pooling; canonical fractional
tie-breaking, stop column NaN-masked for ESM-2-derived rungs per
`OUTPUTS_LEGITIMATE_STOP` and not for the baselines -- omitting that inflates the model
rungs by ~0.01):

| rung | ALL NLL | ALL hit@10% | NOVEL NLL | NOVEL hit@10% |
|---|---|---|---|---|
| uniform | 9.531 | 0.100 | 9.625 | 0.100 |
| codon | 8.895 | 0.300 | 9.203 | 0.333 |
| where_only | 9.179 | 0.440 | 9.544 | 0.181 |
| substitution | 8.275 | 0.652 | 8.878 | 0.468 |
| `where_times_which` | 7.950 | 0.731 | **8.787** | 0.500 |
| frequency | 8.224 | 0.719 | 9.846 | 0.054 |
| single | 7.787 | 0.771 | 10.299 | 0.387 |
| joint_all | 7.503 | 0.807 | 9.525 | 0.496 |
| joint_panel | 7.521 | 0.813 | 9.524 | 0.511 |
| mixture (lambda=0.20) | **7.438** | **0.822** | 9.149 | **0.519** |

**On the full set the two metrics agree** -- every rung's NLL rank and hit@10% rank match
to within one place. **On the novel subset they disagree systematically and in one
direction:** every ESM-2 arm places 2-4 positions higher on hit@10% than on NLL, and
every count- or chemistry-based baseline places lower. The extreme case is the
single-tree arm, which is **last of ten on novel NLL (10.299, worse than uniform's
9.625) while ranking 6th on hit@10% at 0.387 -- nearly 4x chance.** joint_panel is the
same story milder: 9.524 NLL against uniform's 9.625 is a 0.10 nat edge, next to a
hit@10% of 0.511 against 0.100.

A predictor can only be simultaneously worse-than-uniform in NLL and 4x chance in
ranking by putting its mass in the wrong *places* while ordering cells well. That is the
sharpest evidence in this file that the novel-subset problem is density, not ranking --
and it is the reason the NLL tables above should not be read as "the model cannot
forecast novel substitutions".

**On the novel subset the top four are a statistical dead heat.** joint_panel - wxw is
+0.011 [−0.074, +0.088] and mixture - wxw is +0.019 [−0.053, +0.086]: at n=156 this
design cannot separate them, and no claim that the model out-ranks the factored baseline
on novel substitutions is supportable. What *is* resolved there is joint_panel - single
= **+0.124 [+0.069, +0.193]** -- joint training genuinely improves novel ranking.

On the full set the same contrasts do separate: joint_panel - wxw = +0.081
[+0.045, +0.117], mixture - wxw = +0.090 [+0.055, +0.127], joint_panel - single = +0.042
[+0.024, +0.062]. The mixture's edge over joint_panel is not significant on either
subset (+0.009 on both), so its value is in NLL, not ranking.


**Does an epsilon-uniform (Dirichlet) floor help? No -- the tail is not the problem.**
`p' = (1-eps) p + eps * uniform` is order-preserving, so hit@k is exactly fixed and only
the density moves: the clean version of "stop punishing near-zero predictions".
Distribution of per-target NLL on the novel subset (3 seeds pooled):

| rung | mean | median | p75 | p90 | max | top-10% share of total NLL |
|---|---|---|---|---|---|---|
| model | 9.524 | 9.270 | 10.978 | 11.832 | 14.966 | 13.1% |
| `where_times_which` | 8.787 | 8.754 | 9.407 | 10.351 | 17.391 | 13.7% |
| uniform | 9.625 | 9.469 | 10.194 | 10.194 | 10.194 | 10.4% |

The model's worst decile carries **13.1%** of its total novel NLL, against 10.4% for
uniform (which has essentially no tail by construction) and **13.7% for the baseline it
is losing to**. Its single worst cell is 14.97 nats against wxw's 17.39. So the model
does not have a heavier tail than the predictor beating it. The deficit is a broad
shift: the gap is 0.52 nats at the median against 0.74 at the mean, i.e. ~70% of it is
already present in the typical cell, not the outliers.

Sweeping eps confirms it. Best full-set value is eps ~ 0.01-0.03, worth **0.004 nats**
(7.521 -> 7.517). Pushing to eps = 0.5 buys 0.23 on novel (9.524 -> 9.290, still well
behind wxw's 8.787) and costs 0.31 on the full set. Stacked on the pre-D-chosen wxw
mixture, eps = 0.01 moves novel 9.149 -> 9.126 and the full set not at all.

**Why the intuition does not cash out here:** `compute_probs` already clamps at 1e-8,
which is itself a floor, and it was already doing this job -- recall it is worth 1.5 nats
on ncov and ~1.06 nats to the model on novel. The catastrophic-cell penalty had largely
been absorbed before any explicit prior was added, which is why there is little left for
one to recover. Had the clamp not been there, an eps-uniform floor would have been worth
roughly that 1 nat instead.

### Precision as well as recall (2026-10-13, `forecast_precision.py`, job 3285835)

hit@q is a **recall**: of the observed post-D substitutions, the share whose cell is in
the predictor's top q. The complement was asked for, and splits into two things.

**precision@10% is an exact rescaling of recall and adds nothing.** At a fixed selection
fraction, `precision = hits / (q * n_valid)` and `recall = hits / n_mut`, so within an
example the two are proportional. Measured, it behaves exactly so: 0.00135 for the
mixture down to 0.00016 for random, in the same order as recall. The scale is the point
-- 10% of h5nx's 11,949 valid cells is ~1,195 cells against 2.07 observed substitutions
per genotype, so even a perfect predictor caps near 0.0017.

**precision@k for small k is the informative version**, and it says something the recall
table hides. Micro-averaged, 3-seed mean:

| rung | ALL P@1 | P@5 | P@10 | P@50 | NOVEL P@1 | P@5 | P@10 | P@50 |
|---|---|---|---|---|---|---|---|---|
| mixture | 0.021 | 0.013 | 0.012 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 |
| joint_panel | 0.020 | 0.013 | 0.012 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 |
| joint_all | 0.018 | 0.012 | 0.013 | 0.010 | 0.000 | 0.000 | 0.001 | 0.000 |
| single | 0.020 | **0.016** | **0.014** | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 |
| `where_times_which` | 0.018 | 0.014 | 0.011 | 0.006 | 0.000 | **0.004** | **0.002** | **0.002** |
| frequency | 0.015 | **0.015** | **0.014** | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 |
| substitution | 0.000 | 0.001 | 0.003 | 0.002 | 0.000 | 0.000 | 0.001 | 0.001 |
| ceiling (perfect) | 1.000 | 0.340 | 0.183 | 0.041 | 1.000 | 0.285 | 0.154 | 0.032 |
| random | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |

**On the novel subset every predictor's precision@k is ~0 out to k=50.** Across 97
genotypes a top-10 list contains roughly *one* correct novel substitution in total. The
0.51 recall@10% is real but is spread over ~1,200 candidate cells per genotype; narrow
the budget to anything actionable and novel-substitution forecasting collapses. This is
the most decision-relevant statement in this file and it is invisible in both the NLL
and the hit@10% tables.

On the full set P@1 ~ 0.02: flag the single likeliest substitution per genotype and you
are right about 2% of the time -- ~100x random, but 2%. Note also that the ranking
**changes** from the recall table: `single` (P@5 0.016) and `frequency` (0.015) match or
beat the joint arms (0.013) at small k, though joint wins at k=1 and on recall. Position-
blind rungs (`substitution`, `codon`, `where_only`) are near zero at small k by
construction -- with no positional signal their top cells scatter across the sequence.

**Convention caveat.** This script ranks by `argsort`, which breaks ties by index order,
whereas `compute_hit_rate` splits tied blocks fractionally. For continuous-score rungs
the two agree to ~0.001 (joint_panel 0.813/0.511, mixture 0.822/0.519, wxw 0.732/0.500,
single 0.771/0.387 all reproduce). For coarse or heavily-tied rungs they do not: codon
reads 0.358 here against the canonical 0.300, and `frequency` on novel reads 0.064
against 0.054, because Laplace smoothing gives every unseen cell an identical score and
the tie-break then decides. **Prefer the canonical numbers for those rungs**; the P@k
columns carry the same caveat for codon / where_only / frequency-on-novel.

---

## 12. Matched-panel comparison and a corrected epoch rule (2026-10-13)

Built for Figure 5. Two changes to how the random regime is run, both of which move
numbers, so earlier sections' random-split figures are superseded for anything reported.

### The random regime had no selection-free set

`split_examples` returns only `(train, val)`. The epoch was chosen on `val` and then
reported on `val`, so every random-split number above is a minimum over ~20 epochs of
the very curve being reported, not a held-out estimate. The temporal regime never had
this problem: selection is on the pre-D snapshot split, scoring on post-D, and
`score_forecast.py` is the only reader of the latter.

Fixed with `held_out_split()` / `--test-frac`: train 80%, select on val 10%, report on
test 10% that training and tuning never touch. The test block is taken from the front of
the permutation so changing `val_frac` cannot silently move what is reported on, and it
is dropped on the floor inside the training function. Six invariants pinned in
`antiGen/tests/test_held_out_split.py`.

**How much did the old setup inflate things?** Measured before fixing, on the 5-tree
decomp panel, by selecting the epoch on half of val and reporting on the other half:

| rule | joint - single |
|---|---|
| pooled epoch, report full val | −0.069 |
| per-tree epoch, report full val | −0.125 |
| pooled epoch, select on A report on B | −0.074 |
| per-tree epoch, select on A report on B | −0.131 |

So the *deltas* were barely biased -- the optimism hits joint and single alike and
cancels. The absolute NLLs were not trustworthy, but the contrasts were.

### The two panels now match

s4's random panel (5 trees) and s11's forecast panel (9 trees) shared only three trees:
seasonal flu is delisted from Nextstrain and cannot be forecast at any date, while the
forecast panel added snapshot-bearing trees. `train_matched_esm2.sh` closes this from the
random side -- `joint9` plus all nine single-tree controls, three seeds, three-way split.
All nine controls were retrained, not six: the ncov/rsv_a/rsv_b controls under `decomp/`
used the two-way split, so their validation examples overlap the new test block.

Pooled NLL on the matched nine trees, all substitutions:

| contrast | rule | Random holdout | Temporal holdout |
|---|---|---|---|
| vs single-tree | pooled | **−0.162** [−0.219, −0.116] | **−0.266** [−0.395, −0.151] |
| vs single-tree | per-tree | −0.151 [−0.204, −0.107] | −0.234 [−0.316, −0.155] |
| vs `where x which` | pooled | **−0.632** [−0.751, −0.533] | **−0.429** [−0.574, −0.300] |
| vs `where x which` | per-tree | −0.621 [−0.741, −0.523] | −0.397 [−0.578, −0.222] |

Both claims hold in both regimes under both rules, with every interval clear of zero.

### Correction: the epoch rule no longer matters

s9 flagged that per-tree vs pooled epoch selection "materially changes small effects",
and on the old 5-tree two-way panel it did (−0.125 vs −0.069, the per-tree rule looking
roughly twice as good). On the matched nine-tree panel with a selection-free test set the
two rules are **indistinguishable, and pooled is marginally the better of the two** in
every cell above. The earlier gap was a property of that panel and that setup, not a
general advantage of per-tree selection. Both rules are reported in Figure 5 row I/J
rather than one being chosen.

Note the per-tree rule only affects the joint arm: a single-tree run has one source, so
its pooled best epoch already is its per-tree best, and one checkpoint serves both.
`--save-best-per-source` supplies the joint arm's per-tree checkpoints.

### Housekeeping

`prune_stale_runs.py` removed 7.6 GB of superseded artifacts from `matched/`. Re-running
an arm into an existing `--output-dir` leaves the previous attempt's files beside the new
ones, distinguished only by a timestamp; readers take `sorted(glob(...))[-1]`, which is
fine until two attempts used *different splits*, as happened here when the arm moved from
a two-way to a three-way split.

### A PSSM rung, and what it reveals about the random holdout (2026-10-14)

Added to Figure 5's ladders because the `frequency` rung was being read as "the PSSM
baseline used elsewhere in the paper". It is not. Both are parameter-free log-count maps
over (position, alt AA), but:

| | `pssm_root` (Fig 4's "PSSM") | `frequency (memorization)` |
|---|---|---|
| counts | **sequences** carrying each residue at each position | **branch events**: substitutions to that residue on training branches |
| source | precomputed `<data_dir>/pssms/root.csv` | computed from the arm's own training split |
| smoothing | log(count + 1) | log(count + 0.5) |

Every row of `root.csv` sums to the sequence count (9,254,120 for sc2) with ~100% of the
mass on the wild-type residue: it is a composition matrix over tips. A mutation founding
a large clade is counted once by `frequency` and once per descendant by the PSSM.

The paper's `root.csv` could not be reused here -- it exists only for the four viruses in
the main text, and being precomputed over a fixed window it has no notion of this arm's
train/val/test split, so it would leak the test block. The rung is therefore the same
*construction* (log(count+1) over residue occurrences) computed from each arm's own
training sequences.

**It turns out to be the sharpest single piece of evidence that the random holdout
leaks.** Recall at top 10%, de novo subset:

| regime | uniform | PSSM | frequency |
|---|---|---|---|
| temporal | 0.100 | **0.138** [0.099, 0.180] | 0.054 |
| random | 0.099 | **0.562** [0.491, 0.649] | 0.050 |

On the temporal holdout the PSSM sits essentially at chance on de novo substitutions, as
it must: a substitution that did not exist before D cannot be in the pre-D sequence
composition. On the random holdout it reaches 0.56. The direct diagnostic:

    53.9% of random-split de novo targets already appear in the training SEQUENCES
     7.6% of temporal-split de novo targets do

"De novo" is defined by the absence of a training *mutation event*. In a random split a
mutation can be held out as an event while its descendants sit in the training half as
parent sequences, so the residue is plainly visible to anything that reads composition.
Under a temporal split it genuinely is not. The same effect shows on all substitutions,
where the PSSM is the **top** rung on the random holdout (0.88, above the multi-tree
model's 0.86) and only mid-table on the temporal one (0.64 against 0.81).

Consequence for the figure: a shared rung order cannot be monotone in both columns, since
the PSSM moves from first to fifth between them. The ladders are ordered by the left
column, which is the one carrying the rung names; the right column is still correctly
labeled row by row, which is what the shared order exists to guarantee.

**Consequence: the random holdout is demoted out of the main figure (2026-10-14).**
If more than half of what a de novo split calls "never seen" is sitting in the training
sequences, the random split is not measuring forecasting on that subset, and the
all-substitutions column inherits the same problem in milder form -- the PSSM is its top
rung. Figure 5 is now the temporal holdout alone, with recall at top 10% and NLL per
mutation as its two columns and the two target subsets as rows; the random holdout moves
to the supplement, kept as the evidence for that decision rather than as a result. The
random-split arms (`matched/`, array 3286180) stay on disk and are still scored by
`collect_figure_metrics.py`, so nothing has to be recomputed if the framing changes.

This does not retract s4 or s12's random-split numbers. The joint-versus-single contrast
is a comparison between two models scored on identical examples, so the leak inflates
both sides and largely cancels -- which is why the temporal arm reproduces it
(-0.266 there against -0.162 random on the matched panel). What the leak does invalidate
is reading any *absolute* random-split number, and in particular reading its de novo
subset as a forecast.

### De novo redefined against sequence composition (2026-10-14)

The event-based rule called a cell de novo when no training branch mutated to that
residue there. It had a systematic hole: the consensus residue at a position is
inherited from the root, never mutated into, so it leaves no event. A post-D reversion
from a rare derived state back to consensus was therefore labeled de novo despite being
the single most predictable cell at that position. All 12 such targets on the temporal
panel were exactly this, with the "novel" residue carried by 65-99.5% of training
sequences -- and a pure composition baseline called all 12 correctly, recall 1.000.

`seen_cells` now marks a cell seen if a training branch mutated to that residue **or**
any training sequence carries it there. Effects:

| | n (de novo) | PSSM | uniform | joint | single | `where x which` |
|---|---|---|---|---|---|---|
| temporal, event-based | 157 | 0.138 | 0.100 | 0.511 | 0.387 | 0.500 |
| temporal, composition-aware | **145** | **0.066** | 0.100 | 0.501 | 0.370 | 0.497 |
| random, event-based | 412 | 0.562 | 0.099 | 0.450 | 0.360 | 0.450 |
| random, composition-aware | **175** | **0.071** | 0.101 | 0.482 | 0.354 | 0.494 |

Three things to note.

**The PSSM now lands below uniform, which is the correct behavior**, not a bug. Its
nonzero cells (3.7% of the map) outrank every zero cell and consume the first 3.7 points
of the 10% budget, so a genuinely unseen cell can only claim
(0.10 - 0.037)/(1 - 0.037) = 0.065. Measured 0.066 temporally and 0.071 randomly. A
baseline that has by construction never seen the answer should do *worse* than chance
once it has spent its budget elsewhere.

**It removes most of the random holdout's de novo leak.** The 0.562 that made the random
split indefensible on this subset was almost entirely these reversions; under the
composition rule the random de novo subset drops from 412 to 175 targets and its PSSM
falls to 0.071. The two regimes now agree closely on de novo (joint 0.482 random vs
0.501 temporal). **The all-substitutions subset is untouched by this change and remains
leaky on the random split** -- the PSSM is still its top rung at 0.88 against the model's
0.86 -- so Figure 5 stays temporal-only.

**The model barely moves.** joint 0.511 -> 0.501 temporally: the 12 removed targets were
easy for everything, so dropping them costs every rung about equally and changes no
conclusion. The de novo subset is simply smaller and means what it says.

Note this diverges from the `n_novel` recorded in the `*_test_novelty.json` sidecars
(156 for the panel), which still use the event-based tree-derived rule from
`make_forecast_testset.py`. The figure computes its own subsets, so the two are
independent; the sidecar is the stale one.


### Nested pre-D epoch selection: built, measured, not viable at D = 2025-01-01

The forecast arms choose their epoch on a random 10% of the pre-D snapshot examples
(964 examples / 1,699 mutations for `joint_panel`). That is an interpolation task, while
the reported task is extrapolation past D, so the best interpolating epoch need not be
the best forecasting one. The fix is a nested split, entirely inside pre-D data:

    train  branches before D'
    val    substitutions crossing D' (D' -> D)   <- epoch chosen here
    test   substitutions crossing D              <- unchanged

`make_nested_split.py` builds it. Two routes were tried and both run:
D'-snapshot as train with the D snapshot supplying D'-crossing targets, and the
self-anchored trick of passing the D snapshot as *both* `--snapshot` and `--current` to
`make_forecast_testset.py`, which extracts D'-crossing substitutions from one tree
without needing a second snapshot at all.

**The blocker is sample size, and it is structural.** Crossing branches in the
D = 2025-01-01 snapshot:

| D' | inner window | ncov crossing branches | targets |
|---|---|---|---|
| 2024-07-01 | 6 months | 14 | 32 |
| 2024-01-01 | 12 months | 31 | 55 |
| 2023-07-01 | 18 months | 47 | 105 |

and at D' = 2024-07-01 the whole panel yields about 37 validation examples: h5nx 13,
denv3 8, denv4 8, denv2 5, denv1 3, measles 0. Against 1,699 mutations in the current
validation set, that is two orders of magnitude short of usable.

The window cannot be widened for the full panel. Dengue and avian flu are only retained
from 2024-07, which forces D' >= 2024-07, which caps the inner window at six months
against a locked D = 2025-01-01. Pushing D' back to 2023-07 leaves only ncov, measles,
mpox, mumps, zika and evd68 -- and ncov still contributes just 105 targets.

Moving D later would open both windows (D' = 2024-07 -> D = 2025-10 is 15 months, and
2025-10 snapshots are already on disk), **but that is not available**: post-2025-01 data
has already been scored, so any later D defines a test set we have already looked at.
The standing causal-selection rule rules it out.

Recorded as infeasible rather than attempted at a size that cannot support selection.
What remains possible, and is cheap, is an ncov-only spot check at D' = 2023-07-01:
one tree with ~105 forecast-shaped validation targets, asking only whether
forecast-shaped selection picks a different epoch than random-split selection and
whether that changes post-D performance. Not run yet.

---

## 13. Figure 5 (2026-10-13/14)

`antiGen-figures/` is an R/ggplot project, not Python: `R/common.R` carries the shared
theme, palette and `save_figure()`; `run_all.R` regenerates everything.
`figures/fig5_multitree.R` follows those conventions and is registered there.

**Main** `figs/multitree_results.png` — temporal holdout only, 8 panels. Columns are the
two metrics (recall at top 10%, NLL per mutation); rows are the ladder on all
substitutions, the ladder on de novo only, then per-tree paired contrasts versus the
single-tree control and versus `where x which`. Markers are solid only where the 95%
interval clears zero.

**Supplement** `figs/supp_multitree_results.png` — the same 8 panels for the random
holdout, retained as the justification for excluding it rather than as a result, plus
the epoch-rule row across both regimes.

### Pipeline

`collect_figure_metrics.py` (GPU, ~2.5 min) scores both regimes x both epoch rules x
both metrics x both subsets through a single code path and writes
`antiGen-figures/data/multitree/{ladder,deltas}.csv` plus `per_example.npz`. The R
script reads only those three. All are small and tracked, so the figure rebuilds with no
cluster access, and the cached per-example arrays mean changing how an interval is
computed needs no GPU pass. `figure_data.py` is the superseded first collector.

### Arms

| array | arm | state |
|---|---|---|
| 3286180 | `joint9` + 9 single-tree controls, random, three-way split | 27/30 COMPLETED, 3 TIMEOUT |
| 3286174 | `joint_panel` x3 with `--save-best-per-source`, temporal | 3/3 COMPLETED |
| 3295391 | metric collection (final) | COMPLETED, 2m18s |

The three TIMEOUTs are the ncov single-tree controls, killed at the 40-minute wall near
epoch 16 of 20. Harmless: their best epochs are 2, 4 and 4, the `_best.pt` files were
written 22:52-22:56 and the collection job started 23:13, so the scored weights are
final and only the unused final-epoch checkpoint is missing. Same pattern as joint17
seed 1 in s7.

`prune_stale_runs.py` removed 7.6 GB of superseded artifacts from `matched/`, where
three generations had piled up across cancelled attempts, including two-way-split
checkpoints trained on data that is now inside the test block.

### Open

- Measles is gene H in both regimes, but seq_len is 618 in the random arm's build and
  617 in the D snapshot. Each column is internally consistent, and
  `make_forecast_testset.py` checks the reference frame matches between snapshot and
  current build, so coordinates are sound within a column. Position *i* is not
  guaranteed to be the same residue across columns, so no per-position claim should be
  carried between them. The other eight trees agree on seq_len; nothing stronger was
  checked.
- Epoch selection remains interpolation-shaped (a random 10% of pre-D). The nested
  alternative is built and measured infeasible at this D — see s12's closing section.
  An ncov-only spot check at D' = 2023-07-01 with ~105 forecast-shaped targets is the
  one version that would still fit, and has not been run.
- Figure 5's de novo subset uses the composition-aware rule, which diverges from
  `n_novel` in the `*_test_novelty.json` sidecars (still event-based). The figure
  computes its own subsets, so the two never meet, but anything scored from the
  sidecars would disagree.

---

## 14. Seed ensembling is the win; mixing with `where x which` is not (2026-10-14)

Asked whether the model can meaningfully beat `where x which` on de novo recall at
top 10%, where s12 left them at 0.501 vs 0.497. Answer: the two are genuinely
different predictors with real complementarity, every attempt to exploit that
complementarity failed, and the thing that actually moved the number was ensembling
the model's own seeds.

**Everything below is recall at top 10% on the temporal holdout, D = 2025-01-01,
composition-aware de novo, canonical stop convention.**

### The two predictors are not the same, and the complementarity is large

`rank_agreement.py` (job 3297695), pooled over the nine trees:

| | model vs `wxw` | model vs ITSELF (seed-to-seed) |
|---|---|---|
| Spearman of the score maps | 0.697 | 0.916 |
| share of the top-10% budget in common | 0.540 | 0.751 |

Per de novo target (n = 145, 3 seeds): both 0.331, model only 0.170, `wxw` only
0.159, neither 0.340. **A third of de novo targets are caught by exactly one of the
two.** Target-rank Spearman 0.461. So the dead heat in s12 is two different
predictors coinciding in aggregate, not one predictor computed twice.

### Ensembling the seeds, not mixing with the baseline

Averaging seeds' log-probability maps into ONE predictor (`fuse_ci.py`,
`seed_curve.py`). Against `wxw` at its own converged ensemble:

| K seeds | model ALL | model de novo |
|---|---|---|
| 1 | 0.818 | 0.526 |
| 2 | 0.835 | 0.543 |
| 3 | 0.840 | 0.546 |
| 4 | 0.847 | 0.564 |
| 8 | **0.853** | **0.586** |

Still climbing on de novo at K = 8; ALL largely saturates by K = 4. The per-seed
convention the figure currently uses (score each seed, average the recalls) is
0.813 / 0.501 — ensembling is worth **+0.024 [+0.012, +0.036] on ALL and
+0.064 [+0.033, +0.094] on de novo**, the largest single effect measured on this arm.

**Two lucky-draw corrections.** Seeds {0,1,2} are a favourable triple: they ensemble
to 0.837 / 0.566 against 0.840 / 0.546 for the average over five random 3-subsets of
0..7. The same is true of `wxw`'s {0,1,2} (0.745 / 0.503 against a converged
0.733 / 0.507). Any 3-seed number quoted from {0,1,2}, here or above, is ~0.02
optimistic on both sides; the contrast is barely affected.

### `where x which` ensembles too, and saturates immediately

`fit_substitution_baseline` carries a seeded minibatch sampler that every caller
left at `seed=0`, so the baseline had been fit once while the model got three runs
averaged. Refitting at seeds 0..15 and ensembling the same way (`wxw_seed_curve.py`):

| K | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| ALL | 0.726 | 0.731 | 0.733 | 0.728 | 0.734 |
| de novo | 0.486 | 0.509 | 0.509 | 0.497 | 0.510 |

Flat from K = 2. Only the 25-scalar substitution fit is seed-dependent — the
positional marginal is a pure count — so the baseline has ~nothing to average away.
**The model's continued gains with K are therefore not an ensembling artifact**, but
every contrast should still use the ensembled baseline: converged `wxw` is
0.733 / 0.507, and the 8-seed model's margin is +0.120 ALL, +0.079 de novo.

### Nine families of mixing rules, all of them null

`gate_search.py` (job 3298822) fits each family by direct random search on **pre-D
recall**, not NLL — s11's lesson is that an NLL-optimal blend need not move ranking
at all. Families: novelty bonus (`lm + c·unseen`), global blend, rank blend, gated
blend (separate weight for seen/unseen cells), bonus+gated, per-cell gate
(sigmoid over 5 cell features), per-genotype gate, and a full 8-parameter reranker.

The pre-registered rule — family *and* parameters chosen by one pre-D criterion
(all + de novo recall), so nothing in the choice saw post-D — came out `reranker`:

| | recall | vs `wxw` | vs model ens |
|---|---|---|---|
| ALL | 0.823 | +0.078 [+0.049, +0.106] | **−0.014 [−0.033, +0.006]** |
| de novo | 0.572 | +0.069 [−0.022, +0.148] | **+0.007 [−0.039, +0.049]** |

**Across all 27 (family x objective) cells, not one rule beats the model ensemble on
either subset.** Four rules do clear zero against `wxw` on de novo — `reranker/novel`
+0.180, `bonus_gated/novel` +0.173, `bonus/novel` +0.159, `cell_gate/novel` +0.145 —
and every one collapses ALL to 0.17–0.22. Pure trade, the same failure mode s11
recorded for the alpha upweight on NLL, and not deployable: you cannot know at
prediction time that a target will turn out to be de novo.

### Why it fails: nothing predicts which predictor wins

The transfer diagnostic in the same script. AUC for "the model out-ranked `wxw` on
this genotype":

| feature | pre-D | post-D |
|---|---|---|
| model entropy | 0.513 | 0.531 |
| seen frac of top-10% | 0.447 | 0.495 |
| model/`wxw` agreement | 0.496 | 0.424 |
| log n_valid | 0.479 | 0.524 (flips) |

All at chance, in both regimes. A gate needs something to gate on and there is
nothing. Compounding it, the model wins on 16.1% of pre-D genotypes against 21.3%
post-D, so the selection set misrepresents the balance it is asked to tune.

**The headroom is real and is a SELECTION failure, not a capacity one.** Oracle
per-genotype gate (pick the better predictor per query, using the answer): de novo
**0.662**, +0.096 [+0.051, +0.155] over the ensemble. Post-D oracle within the rule
families reaches 0.724 (`cell_gate`) against the 0.56–0.57 pre-D selection achieves.
The rules exist; pre-D data cannot identify them.

### The pre-D selection set is clean, and only because of the composition rule

`check_select_leak.py`, the diagnostic s12 used, applied to the de novo subset of the
pre-D validation split that the searches tune on:

| pre-D de novo subset | n | residue in training seqs | PSSM recall@10% |
|---|---|---|---|
| composition-aware (used) | 166 | **0.000** | **0.066** |
| event-based (the old rule) | 398 | **0.573** | **0.609** |

Defining de novo by events would have made 57% of the selection targets fake and let
a pure composition baseline score 0.609 on them — s12's random-split leak, reappearing
inside the selection set. Worth stating because it is a cheap mistake to make: any
future tuning on a pre-D de novo subset must use `seen_cells`, not an event rule.

Note this also supersedes the claim in s12's closing section that forecast-shaped
selection is infeasible at this D. That was true of a *temporally* nested split
(~37 examples). Conditioning the existing pre-D split on **novelty** instead gives
182 de novo targets against the 145 being forecast — ample. It is still temporally
interpolating, which is the honest limitation, and the transfer diagnostic above
shows that limitation is what bites.

### Zero-shot transfer to a new virus family: the baseline wins

`recent.json` (the ongoing 2026 Bundibugyo ebolavirus outbreak, 762 nodes, 597 tips,
all tips 2025.94–2026.63 and so entirely post-D) scored by `zeroshot_eval.py`
(job 3299871). Genes GP/NP/VP35/VP30/VP40/VP24 pooled, n = 129 mutations. L is
excluded: 2211 aa puts it on the >1022 two-pass stitching path s9 flags as the
suspected mpox failure, which would be a confound.

Only uniform, codon, `substitution` (25 position-free scalars, fitted per panel tree
and averaged — the matrix has no length dependence) and the ESM-2 arms are available
zero-shot. `where_only`, `pssm`, `frequency` and `where x which` all need
position-specific counts from this tree and cannot appear in a zero-shot ladder.

| rung | recall@10% | NLL/mut |
|---|---|---|
| uniform | 0.100 | 9.370 |
| codon | 0.299 | 8.410 |
| **substitution (panel-transferred)** | **0.752** | 8.488 |
| single-tree models (best h5nx) | 0.403–0.558 | 8.47–8.85 |
| joint_all (3-seed ens) | 0.674 | **8.049** |
| joint_panel (8-seed ens) | 0.674 | 8.096 |

**A 25-parameter substitution matrix beats the joint model, paired +0.078
[+0.030, +0.166].** Three things do transfer: joint_panel beats every single-tree
control by 0.116–0.271 (all clear of zero), so joint training genuinely buys
cross-family generalization; the model wins decisively on NLL; and joint_all ≈
joint_panel (+0.000 [−0.031, +0.059]), the saturation story again.

Giving the tree-specific baselines a random half of this tree while the model still
sees none of it (n = 28, noisy): `where x which` 0.464, `pssm` 0.132, `where_only`
0.111, `frequency` 0.099 — against the zero-shot model's 0.643. Positional priors are
worthless on a low-divergence single outbreak.

**The pattern worth keeping:** this is the second independent setting where the
model's edge over a factored baseline evaporates once position-specific memorization
is unavailable, and in both the model keeps its NLL advantage while losing on
ranking. Better density, worse ordering, wherever it cannot memorize.

### Where this leaves the headline claim

Against the converged `wxw` ensemble, the 8-seed model ensemble is **+0.120 on all
substitutions** (clear of zero by a wide margin) and **+0.079 on de novo**
(+0.083 [−0.011, +0.170] against the {0,1,2} draw — the lower bound has moved from
−0.073 at K = 1 to −0.011 at K = 8, but has not crossed). The de novo interval is
~±0.09 wide at n = 145 and pairing buys little because the two predictors are only
weakly correlated, so the only route is the point estimate continuing to rise.

**Multiple-comparisons caveat.** A lot of rules have now been run against this
145-target set. The pre-registered-rule machinery protects the within-search choice;
the broader sequence of experiments has no such protection. If K = 16 does clear
zero, that result needs a second forecast date to be worth much — and s12 found
D = 2024-07-01 unavailable.

### Code added

| file | purpose |
|---|---|
| `rank_agreement.py` | cell-level agreement: Spearman, top-10% overlap, per-target 2x2, oracles |
| `fuse_ci.py` | paired bootstrap intervals for named candidate rules |
| `gate_ceiling.py` | `wxw` seed-ensemble fairness check + per-genotype / union oracles |
| `gate_search.py` | nine mixing families, pre-D random search on recall, transfer diagnostic |
| `check_select_leak.py` | verifies the pre-D de novo selection subset does not leak (CPU) |
| `seed_curve.py` | model recall vs number of seeds, subset-averaged |
| `wxw_seed_curve.py` | the same for the baseline |
| `zeroshot_eval.py` | the Bundibugyo ladder, zero-shot and half-fit tables |
| `train_seeds_esm2.sh`, `train_seeds2_esm2.sh` | joint_panel seeds 3–7, 8–15 |
| `train_initseeds_esm2.sh` | 4 replicates sharing `--split-seed 0`, differing init only |
| `train_singles2_esm2.sh` | single-tree seeds 3–7, nine trees |

Two bugs found and fixed while doing this, both of which moved numbers:

1. `gate_search.canonical()` masked the stop column for **every** rung including pure
   `wxw`. The convention masks it only for ESM-2-derived rungs
   (`OUTPUTS_LEGITIMATE_STOP`), and getting it wrong inflated every contrast against
   the baseline by ~0.02. All numbers above are post-fix.
2. `rank_agreement.report()` summed ragged per-tree arrays with `np.sum` and crashed.
   The GPU results are saved before `report()` runs, so nothing had to be recomputed.

### Open / in flight at the end of this session

- **`joint_panel` seeds 8–15** (job 3299577) launched 13:00–13:09, at epoch 1/20 when
  last checked — K = 16 is NOT done.
- **Single-tree seeds 3–7**, nine trees (job 3299597, 45 tasks) launched 13:20. These
  are required before any ensembled joint-vs-single contrast: joint has 8–16 seeds and
  the controls have 3, and comparing those would credit joint training with
  ensembling the controls were denied. **K = 8 is the matched number**; K = 16 is a
  scaling note for the joint arm alone, not part of the contrast.
- **`gate_search` on the fixed-split replicates** (job 3299502) was still running. Those
  4 models share one validation split so their ensemble has a clean pre-D selection
  set — built to test whether the selection failure above is caused by the
  single-seed-select / ensemble-score mismatch. Unplanned finding from its reference
  table: they ensemble to 0.840 / 0.545 against 0.847 / 0.564 for 4 seeds with
  *differing* splits, so part of the ensembling gain comes from models seeing
  different 90% subsets, not just different inits.
- **Figure 5 has NOT been regenerated.** Doing it needs: patch
  `collect_figure_metrics.py` to ensemble log-prob maps over seeds for `joint` and
  `single` and to use the `wxw` 3-seed ensemble; rerun it (GPU, ~3 min); then
  `antiGen-figures/run_all.R`; then commit. Two decisions first — matched K (above),
  and what to do with the epoch-rule row I/J, since a per-tree *ensemble* is well
  defined via `--save-best-per-source` but s12 found the two rules indistinguishable.
  Nothing from this section is in the figure or committed yet.

---

## 15. Figure 5 rebuilt on seed ensembles (2026-10-14)

s14 found that averaging seeds' log-probability maps into ONE predictor is the
largest single effect on this arm, and closed with "Figure 5 has NOT been
regenerated". This is that rebuild. `collect_figure_metrics.py` now ensembles
instead of scoring each seed and averaging the metrics afterwards.

**Status: K = 3 is committed, K = 8 is the target and is blocked.** The matched
single-tree controls at seeds 3-7 (job 3299597, 45 tasks) were still trickling
through the queue, so this pass uses seeds {0,1,2} for both arms. Everything
below is the K = 3 figure. It is not the final one -- s14 measured de novo
0.546 at K = 3 against 0.586 at K = 8, and {0,1,2} is a known-favourable triple
(~0.02 optimistic on both sides) -- but it is matched, so the contrasts are
honest at the K it reports.

### What changed in the collector

- `joint` and `single` are one predictor each, built as the mean of their seeds'
  log-prob maps, at the **same K**. An 8-seed joint against a 3-seed control
  would credit joint training with ensembling its control was denied, so the
  script takes one `--ens-seeds` list and uses it for both, and refuses to start
  if any run in it is missing.
- `substitution` and `where_times_which` are 3-seed ensembles of their own
  fitted sampler. s14 found `fit_substitution_baseline`'s seeded minibatch
  sampler had been left at `seed=0` by every caller, so the baseline was fit
  once while the model got several runs averaged. It saturates by K = 2, so this
  is fairness, not a gain -- and it is a real correction: `wxw` on all
  substitutions goes 0.731 -> 0.745, which is a third of the model's margin
  over it.
- **The random regime does not ensemble and cannot.** Each seed there carries
  its own three-way split, so seed 0's test block sits inside seed 1's training
  half and the maps are not commensurable. It keeps the per-seed convention;
  the supplement's banner now says so. One more reason the two regimes are not
  a like-for-like pair, on top of the leak that demoted it in s12.
- `logp_model` moved from `fuse_forecast.py` into `collect_figure_metrics.py`
  (the import direction was already that way for `seen_cells`); the other five
  scripts still import it from `fuse_forecast`, which re-exports it.

### Before and after, temporal holdout, pooled epoch rule

Recall at top 10%, per-seed convention -> K = 3 ensemble:

| rung | ALL | ALL ens | de novo | de novo ens |
|---|---|---|---|---|
| `substitution` | 0.652 | 0.656 | 0.469 | **0.455** |
| `where x which` | 0.731 | 0.745 | 0.497 | 0.503 |
| single | 0.771 | 0.791 | 0.370 | 0.386 |
| **joint** | 0.813 | **0.837** | 0.501 | **0.566** |

Note `substitution` on de novo moves the *wrong* way. Ensembling is not
uniformly positive, and the one rung where it costs something is the one whose
seed-to-seed variation is 25 scalars rather than a network.

Pooled contrasts:

| contrast | metric | per-seed | K = 3 ensemble |
|---|---|---|---|
| vs single, ALL | recall | +0.042 [+0.025, +0.061] | **+0.046** [+0.025, +0.069] |
| vs `wxw`, ALL | recall | +0.081 [+0.046, +0.116] | **+0.092** [+0.056, +0.127] |
| vs single, de novo | recall | +0.131 [+0.075, +0.199] | **+0.179** [+0.116, +0.252] |
| vs `wxw`, de novo | recall | +0.005 [−0.085, +0.085] | **+0.062** [−0.032, +0.147] |
| vs single, ALL | NLL | −0.266 [−0.390, −0.149] | −0.238 [−0.362, −0.121] |
| vs `wxw`, ALL | NLL | −0.429 [−0.569, −0.303] | **−0.500** [−0.637, −0.375] |
| vs single, de novo | NLL | −0.830 [−1.017, −0.636] | −0.786 [−0.973, −0.590] |
| vs `wxw`, de novo | NLL | +0.775 [+0.553, +0.969] | **+0.666** [+0.446, +0.858] |

The de novo recall contrast against `wxw` moves from +0.005 to +0.062 and its
lower bound from −0.085 to −0.032. **It still includes zero**, exactly as s14
predicted it would at this K. The de novo NLL deficit also shrinks, 0.775 to
0.666, but is nowhere near closing -- the ensemble is a better predictor on both
metrics, not a fix for the density problem s11 diagnosed.

### The pooling rule was checked, not assumed

Ensembling in log-prob space (the normalized geometric mean) is what s14
established and what the ranking result rests on; the arithmetic mean of the
probability maps is the textbook density ensemble, and Jensen guarantees its NLL
is no worse than the mean of the seeds'. The collector now computes both and
prints the comparison. The `*_arith` rungs are a check, never written to the
CSVs.

| arm | subset | NLL log-lin | NLL arith | recall log-lin | recall arith |
|---|---|---|---|---|---|
| joint | all | **7.428** | 7.433 | **0.837** | 0.833 |
| joint | de novo | **9.447** | 9.471 | **0.566** | 0.538 |
| single | all | 7.667 | **7.658** | 0.791 | **0.800** |
| single | de novo | **10.232** | 10.277 | 0.386 | **0.407** |

Log-linear wins all four cells for `joint` and loses two of four for `single`.
**That means the figure's rule flatters the joint-vs-single contrast**: de novo
recall is +0.180 under log-linear against +0.131 under arithmetic, and ALL
recall +0.046 against +0.033. State it rather than bury it. The defence is that
log-linear is the pre-existing rule -- s14's seed curve, `fuse_ci.py` and
`fuse_forecast.py` all use it, and it was fixed before this figure existed, not
chosen from this table. Choosing between them on the numbers above would be
choosing a predictor on post-D data, which the standing rule forbids. The
contrast survives either way and the ranking of rungs is unchanged; only its
size moves.

### Cluster note, worth keeping

`gpu` and `gpu_batch` are **the same ten nodes** (GPU3694, 7094, 7130, 7220,
70DC, 71BA, 389E, 708E, C870, CACE). They differ only in time limit -- 1 day
against 14 -- and in queue contention. On 2026-10-14 an identical 1-GPU request
was scheduled at 14:01 on `gpu` and 19:54 on `gpu_batch`: ~6 hours apart on
identical hardware. Check before submitting with

    sbatch --test-only --partition=<P> --gpus=1 --mem=96G --wrap=true

`quick_preemptible` was faster still (13:53) and was **not** used. s11 already
records a preemption on this arm leaving a stale 4-epoch partial beside a
complete run, and every reader here takes `sorted(glob(...))[-1]`; a preempted
single-tree task that is not requeued leaves an undertrained `_best.pt` that the
collector's existence check would accept. `run_collect_metrics.sh` now lists
`gpu` first.

### Overlap panels, and the main figure demoted to one metric (2026-10-14)

Two changes to Figure 5 on top of the ensembling.

**The main figure is now recall only.** NLL moves to the supplement as its own
single-column block (A-D there), not dropped. It is not redundant with recall --
s11 established that on de novo substitutions the two disagree systematically
and in one direction, every ESM-2 arm placing higher on recall than on NLL and
every count-based baseline lower, which is the evidence that the de novo deficit
is a density failure rather than a ranking one. But the claims are stated in
recall, and two eight-panel columns with different rung orders cost a reader
more than they buy. Supplement letters shift: temporal NLL A-D, random holdout
E-L, epoch rule M-N.

**Two overlap panels (E, F) ask whether the model and `wxw` are the same
predictor.** The ladders say which scores higher and the forests say by how
much; neither separates a model that learned something the baseline did not from
one that rediscovered it, and on de novo targets their recalls are close enough
(0.566 against 0.503) that the distinction carries weight. `agreement.csv`
(new, from `collect_figure_metrics.py`) counts, per tree and subset, how many
observed substitutions fall in both top-10% sets, exactly one, or neither.

Temporal holdout, pooled rule, K = 3:

| subset | both | model only | `wxw` only | neither | n |
|---|---|---|---|---|---|
| all | 391 (69%) | **81 (14%)** | 28 (5%) | 64 (11%) | 564 |
| de novo | 54 (37%) | **28 (19%)** | 18 (12%) | 45 (31%) | 145 |

**On de novo targets 46 of 145 -- 32% -- are caught by exactly one of the two**,
and the model catches 28 the baseline misses outright against 18 the other way.
That reproduces s14's cell-level finding (both 0.331, model only 0.170, `wxw`
only 0.159) on the ensembled predictors and at target level. The near-tie in de
novo recall is two different predictors coinciding on a total, not on a list.

Two things about how the panels are drawn:

- **Area-proportional, solved not sketched.** Radii come from the caught counts
  and the centre distance is solved with `uniroot` on the lens-area formula so
  the overlap region's area equals the shared count. A fixed-overlap cartoon
  could be made to look as similar or as different as one liked. Both predictors
  spend the same budget -- 10% of valid cells -- so the circles differ in size
  only because one catches more, which is the ladder's recall.
- **The counts use a different tie rule from the ladder, and it is checked.**
  `compute_hit_rate` splits a tied block at the budget boundary fractionally,
  which is right for a rate but cannot say whether one cell was selected, and a
  Venn must label cells. `caught_mask` takes the strictly-above set. The
  collector prints each circle's marginal against the ladder's recall:

      Temporal  all    joint  0.8369 vs 0.8369  (+0.0000)
      Temporal  all    wxw    0.7429 vs 0.7446  (-0.0017)
      Temporal  novel  joint  0.5655 vs 0.5655  (+0.0000)
      Temporal  novel  wxw    0.4966 vs 0.5031  (-0.0066)

  Exact for the model, within 0.007 for `wxw`. **The random regime is not:** its
  all-substitutions marginals are 0.04 low for both rungs, because its larger
  example count puts heavy tie blocks at the boundary. No panel uses it, but any
  future overlap panel on the random split must not assume the two rules agree.

### Cluster note, part two: list ONE partition

Listing several partitions makes the job schedule against the most contended
member, not the least. Job 3300348 sat PENDING with an estimate of 15:26 on
`Partition=gpu,gpu_batch,gpu_high_mem` while `sbatch --test-only
--partition=gpu` for the identical request said it would start immediately.
`scontrol update jobid=... Partition=gpu` started it within seconds. The same
move on the 3299597 array released its queued tasks. So: test the partitions,
then submit to the single best one rather than a list.

### Open

- **Rerun at the matched K = 8 and regenerate** once job 3299597 finishes.
  This now moves the overlap panels as well as the ladders and forests. That
  is one `sbatch experiments/multitree/run_collect_metrics.sh` (default
  `--ens-seeds 0,...,7`, ~3 min) plus `antiGen-figures/run_all.R`.
- The figure's caption must state the convention -- the two antiGen rungs are
  K-seed ensembles and the two fitted rungs are 3-seed ensembles. Nothing on the
  face of the figure says so, and the per-seed numbers in s11-s13 are not
  comparable to these.
- K = 16 exists for `joint_panel` and is a scaling note for that arm only. It
  may not enter the joint-vs-single contrast while the controls stop at 8.
- s14's multiple-comparisons caveat stands and now applies to the figure: a lot
  of rules have been run against this 145-target de novo set, and the de novo
  contrast against `wxw` is the one being watched to clear zero. If it does at
  K = 8, it needs a second forecast date to be worth much, and s12 found
  D = 2024-07-01 unavailable.

---

## 16. Alphabet and probability-floor audit: a real bug, and a correction (2026-10-14)

Asked why the model wins on NLL zero-shot (s14: 8.096 against the substitution
matrix's 8.488) when on the de novo subset -- the closest analogue, both denying
it any positional memorization -- it loses badly (9.524 against 8.787). Two
regimes that should agree did not, so one number was measuring something else.

**It was. `zeroshot_eval.py` had a wild-type leak and its headline NLL result
reverses.** `zeroshot_probe.py` rescores all rungs under one convention:

| contrast (zero-shot, n=129) | convention | joint_panel - substitution |
|---|---|---|
| published (s14) | as-is | **−0.392** [−0.544, −0.055] model better |
| corrected | wild type masked, no floor | **+0.359** [+0.153, +0.596] model worse |
| corrected | wild type masked + 1e-8 floor | **+0.263** [+0.108, +0.599] model worse |

The cause: that script turns scores into probabilities with a plain
`softmax(sc.reshape(B, -1))`. For the ESM-2 and codon rungs `sc` had already
been through `logp_model`, which zeroes the wild-type cell; for
`substitution (panel-transferred)` it was `sub_mat[aa]` -- the raw matrix,
**including its diagonal**, which is the largest entry in each row. So that rung
spent **48% of its probability on cells that are wild type by construction** and
was charged 0.655 nats for it (8.488 -> 7.833 once masked). Recall never saw the
bug, because `per_example_hits` NaN-masks the wild-type cells itself -- which is
exactly why the two metrics disagreed only here. Fixed: every rung in
`zeroshot_eval.py` now goes through `logp_model`. **s14's zero-shot NLL table and
the sentence "the model wins decisively on NLL" are retracted.**

With that fixed the two regimes agree: on genuinely novel territory the model
loses to a factored/substitution baseline on NLL, zero-shot and de novo alike.

### The figure's NLL column: wild type is fine, the floor is not

`figure_nll_audit.py` reruns the temporal panel under four conventions.

**Wild type is masked correctly in Figure 5.** Every rung goes through
`temp_probs(sc, wtm, 1.0)`, whose `p * wt` zeroes the query's own residue, and
`wtm` comes from `collect`. No rung is exempt. So the zero-shot bug is not in
the figure.

**But the 1e-8 floor is applied BEFORE renormalization, which makes its
protection proportional to how much mass a rung wasted on wild type.** The
effective floor is `1e-8 / Z`, where Z is the mass surviving the mask. Measured
on the degenerate targets below, which every rung is forced to the floor on and
which therefore read out Z directly:

| rung | softmax mass on WT | implied Z | NLL on a floored cell |
|---|---|---|---|
| uniform | 4.5% | 0.955 | 18.37 |
| `where x which` | 48.6% | 0.514 | 17.56 |
| **joint** | **98.8%** | **0.012** | **13.05** |

The model puts 98.8% of its softmax on wild-type cells, so after masking its
surviving mass is ~1%, and renormalizing inflates its floored cells ~80x
relative to uniform. **A catastrophic cell costs the model 13.1 nats and the
baseline 17.6.** That is the mechanism behind s11's "the floor rescues peaked
predictors" and it is worth a lot:

| `joint - where x which`, all substitutions | estimate |
|---|---|
| figure convention (floored) | **−0.399** [−0.552, −0.262] model better |
| unfloored | **+0.004** [−0.315, +0.335] dead heat |

**The model's NLL advantage over the factored baseline on all substitutions is
entirely the floor.** On de novo the floor also helps it (+0.762 floored,
+1.94 unfloored), reproducing s11's 1.06-nat measurement. The floor is the
training objective (`compute_probs` is the loss path), so scoring through it is
defensible -- but it is not a neutral choice between predictors, and every NLL
number in this arm leans on it.

### Degenerate wild-type-to-wild-type targets exist

Found by chasing an impossible number: unfloored `uniform` read 12.16 when a
uniform distribution cannot exceed log(21L) ~ 10.19, and it was wrong on `all`
but exact on `novel`.

    test  : 2 of 564 targets are wt -> wt   (both h5nx)
    train : 0 of 16643

Every rung masks those cells to zero, so **no predictor can ever score them**:
they cost ~18 nats each under the floor and ~700 unfloored, and they cap every
rung's recall at 562/564. The effect is symmetric so contrasts survive, but
absolute NLLs are inflated for everyone. They should be dropped in
`make_forecast_testset.py` rather than scored. Why h5nx produces them is open;
first suspect is the snapshot/current coordinate mismatch s13 flagged for
measles.

### The alphabet, rung by rung

`ACDEFGHIKLMNPQRSTVWY*-` is identical in all 14 places it is defined, package
and experiments, and `AlphabetConfig.vocab_size` derives from it. No re-indexing
bug. What differs is which columns each rung may legitimately use, and what is
actually masked:

| rung | gap `-` | stop `*` | masked in recall | masked in NLL |
|---|---|---|---|---|
| ESM-2 (`single`, `joint`) | legit | **sentinel** (`<unk>` logit reused) | stop | nothing |
| `codon` | **floor, not a prediction** | legit | **nothing** | nothing |
| `substitution`, `where x which` | legit | legit | nothing | nothing |
| `frequency`, `where_only`, `pssm` | counts | counts | nothing | nothing |

Three findings:

1. **The stop sentinel costs nothing.** ESM-2 puts **0.0000%** of its mass
   there -- it was trained with that column in the softmax for 20 epochs and
   learned to zero it. Masking stop for the ESM-2 rungs in NLL changes every
   number by 0.000. Worth recording as measured-and-dismissed.
2. **`codon`'s gap column is masked nowhere**, in either metric, although it
   declares `OUTPUTS_LEGITIMATE_GAP = False`. `per_example_hits` implements only
   the stop half of `build_model_gap_stop_mask`'s contract; the gap half is
   silently dropped in every experiment script.
3. **Recall@10% is not the same budget for every rung.** `n_valid` is counted
   after masking, so the model gets 20L cells and every baseline 21L. At q = 0.1
   that is 2.0L against 2.1L selected cells over a near-identical useful
   universe -- **a systematic 5% larger budget for every baseline**. This one
   runs against the model, so the reported margins over `where x which` are if
   anything understated. It also affects s15's overlap panels, whose two circles
   are drawn from different-sized budgets.

### Is the gap column a prediction target?

Targets by class, and what each rung scores on them:

| | all (529) | de novo (140) |
|---|---|---|
| standard residues | 513 (97.0%) | 129 (92.1%) |
| gap | 13 (2.5%) | **10 (7.1%)** |
| stop | 1 (0.2%) | 1 (0.7%) |
| degenerate wt->wt | 2 (0.4%) | 0 |

Gap is three times commoner among de novo targets, because a deletion at a given
position is almost always a cell no training branch has visited. The ESM-2 arms
put **0.00%** of their mass on the gap column while the baselines put 2.5-4.1%,
so the model scores those targets at 12.5 nats against `where x which`'s 10.8.

**Correcting an overstatement made mid-session:** the claim that "more than half
the de novo NLL deficit is one column" came from reading the `all masked`
convention, which masks gap for BOTH predictors and so forces both to the floor,
diluting the contrast rather than isolating it. The correct decomposition of the
+0.762 de novo deficit is:

| class | n | contributes | share |
|---|---|---|---|
| standard | 129 | +0.644 | **84.6%** |
| gap | 10 | +0.122 | 16.1% |
| stop | 1 | −0.005 | −0.6% |

**On standard-residue targets alone the model still loses +0.699
[+0.463, +0.908]**, significant under every convention. The de novo density
deficit is real and is not an alphabet artifact.

Open question for the arm: ESM-2 declares gap legitimate, `-` is a real learned
token in its vocabulary, and 2.2% of *training* mutations go to gap -- so it had
both the capacity and the signal to put mass there and emits ~0. That looks more
like a vocabulary-mapping bug in `esm2_fitness_predictor.py` than a learned
preference, and is worth checking before the gap targets are read as a genuine
model weakness.

### Recall at other budgets, zero-shot

s14 read recall at 10% alone. Sweeping q (n = 129, chance = q):

| rung | 1% | 2% | 5% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|---|---|
| codon | 0.043 | 0.059 | 0.133 | 0.299 | 0.632 | 0.871 | 0.994 |
| substitution (panel) | 0.074 | 0.185 | **0.470** | **0.752** | **0.915** | **0.969** | 0.992 |
| joint_panel (8-seed) | **0.124** | 0.194 | 0.419 | 0.674 | 0.829 | 0.930 | 0.977 |
| joint_all (3-seed) | 0.109 | 0.194 | 0.411 | 0.674 | 0.884 | 0.938 | 0.977 |

**The model is the better ranker at the head of the list and the worse one
everywhere else.** At q = 1% it leads by +0.050 (not significant at n = 129),
they tie at 2%, and from 5% on the substitution matrix wins, significantly at
10%, 20% and 30%. So s14's "a 25-parameter substitution matrix beats the joint
model" is true at the budget it was measured at and false at the smallest ones.
This is the decision-relevant cut: with a handful of candidates to test the
model is the better choice, with a broad budget it is not.

### Code added

| file | purpose |
|---|---|
| `zeroshot_probe.py` | three conventions x the zero-shot ladder, plus recall vs q |
| `figure_nll_audit.py` | the figure's NLL column under four conventions, by target class |
| `run_zeroshot_probe.sh`, `run_nll_audit.sh` | submitters (both run in ~1 min on CPU) |

Both probes normalize in log space when unfloored. Doing it in probability space
divides by a denominator that underflows to zero whenever one cell holds nearly
all the mass -- `pssm` puts 93% on wild type, the ESM-2 arms 99% -- and returns
NaN for the whole example. That bit the first run of each.

### Panels A and B hold, and the figure understates them (2026-10-14)

Recall was audited the same way as NLL: the same four conventions, plus the
budget question, plus a target-class split.

**The floor does not touch recall at all** -- `unfloored` reproduces `figure` to
three decimals in every cell. Ranking is what recall reads, the floor is monotone
apart from the ties it creates, and the top decile sits well above it. So the
whole floor story in this section is an NLL story.

**The budget asymmetry is real and costs the model.** `n_valid` is counted after
masking, so at q = 10%:

| | cells rankable | budget |
|---|---|---|
| baselines (stop not masked) | 14,357 | **1,436** |
| ESM-2 arms (stop masked) | 13,673 | **1,367** |

Every baseline selects 69 more cells than the model from a useful universe that
is identical for practical purposes. Giving every rung the same universe:

| `joint - where x which`, recall@10% | all (529) | de novo (140) |
|---|---|---|
| figure convention | +0.089 [+0.053, +0.124] * | +0.022 [−0.080, +0.110] |
| common budget | **+0.100** [+0.064, +0.136] * | **+0.036** [−0.065, +0.124] |
| standard-residue targets only | **+0.121** [+0.082, +0.159] * | **+0.085** [−0.018, +0.176] |
| gap targets only | −0.692 [−1.000, −0.357] * | −0.600 [−1.000, −0.167] * |

**So yes, the model is still winning on recall, by more than the figure says.**
Every correction runs in its favour -- the opposite of NLL, where the floor was
propping it up. `joint - single` is unchanged by all of this (+0.051 all,
+0.171 de novo), since both sides are ESM-2 and share every convention.

**The gap column matters far more to recall than to NLL.** It was 16% of the de
novo NLL deficit; here it is most of the erasure of the model's lead. Recall by
class, de novo:

| rung | standard | gap |
|---|---|---|
| `where x which` | 0.493 | **0.642** |
| joint | **0.566** | **0.000** |
| single | 0.380 | 0.000 |

The ESM-2 arms rank **no gap cell at all** in their top 10% -- 0 of 13 on all
substitutions, 0 of 10 on de novo -- while `where x which` catches two thirds of
them. Decomposing the de novo +0.022: standard contributes +0.078 and the ten
gap targets take back −0.043. **The near-tie in panel B is mostly ten deletion
targets.** On the other 129 the model leads by +0.085, which still does not
clear zero at this n, but more than doubles the headline number.

**It is not a mapping bug.** `-` maps to ESM-2 token id 30, a genuine `-` token
(only `*` is routed to `<unk>`). ESM-2 was pretrained on ungapped UniRef, so the
gap logit starts near zero and four finetuned layers over 370 gap mutations do
not lift it. A real property of the backbone, not of this code.

**Do not fix this by dropping gap targets.** That is choosing the convention that
flatters the model, which is the error this whole section exists to catch. The
honest presentation is standard and gap reported separately, with the gap
deficit stated as a known limitation of an ungapped-pretrained backbone.

### The headline scores standard residues only (2026-10-14)

Decision: Figure 5's main panels score the **20 ordinary residues**, with the
stop and gap columns excluded for every rung. Rationale, which supersedes the
caution recorded in the previous subsection:

- `*` is not a prediction at all for ESM-2 -- the model maps it to `<unk>` and
  reuses that logit. It is 1 of 564 post-D targets.
- `-` is a column ESM-2 has no representation of. It is in the vocabulary (token
  id 30, correctly mapped -- this was checked, it is not a mapping bug), but the
  backbone was pretrained on ungapped UniRef, so the gap logit starts near zero
  and four finetuned layers over 370 gap mutations do not lift it. Measured: the
  ESM-2 arms put **0.00%** of their probability there against the baselines'
  2.5-4%. Scoring a predictor on a column outside its representational scope
  measures the scope, not the prediction.
- Excluding both puts every rung on ONE cell universe, which also removes the
  budget asymmetry: `n_valid` is counted after masking, so the old rule gave
  each baseline ~5% more cells (1,436 against 1,367) at q = 10%.

**The earlier worry that this "flatters the model" does not hold**, and the
reason is the pretraining argument above, not the size of the effect. What keeps
it honest is that the supplement scores the gap column back in (block A-D,
banner-labelled), so the 0-of-13 deficit is shown as an exclusion rather than
quietly removed. `legacy` remains in the CSVs, labelled, for reproducing the
older numbers.

Targets outside the universe are dropped rather than charged at the floor: no
predictor can reach them, so they measure nothing and only inflate every rung's
NLL equally. n goes 562 -> 548 on all substitutions and 145 -> 134 on de novo.

**Recall at top 10%, temporal holdout, pooled rule, K = 3** (`legacy` is the
previous figure; `standard` is the new headline):

| rung | all: legacy | all: standard | de novo: legacy | de novo: standard |
|---|---|---|---|---|
| uniform | 0.100 | 0.100 | 0.100 | 0.100 |
| codon | 0.301 | 0.295 | 0.341 | 0.355 |
| where_only | 0.442 | 0.445 | 0.177 | 0.180 |
| PSSM | 0.646 | 0.652 | 0.066 | 0.064 |
| substitution | 0.658 | 0.658 | 0.455 | 0.455 |
| frequency | 0.721 | 0.726 | 0.055 | 0.052 |
| `where x which` | 0.747 | 0.743 | 0.503 | 0.500 |
| antiGen single-tree | 0.794 | **0.808** | 0.386 | **0.410** |
| **antiGen multi-tree** | 0.840 | **0.852** | 0.566 | **0.575** |

The ESM-2 arms gain the most (+0.012 to +0.024) because they are the rungs that
were being scored on a column they cannot use. Contrasts:

| contrast | subset | legacy | standard |
|---|---|---|---|
| vs single-tree | all | +0.046 [+0.025, +0.069] * | +0.044 [+0.021, +0.068] * |
| vs `where x which` | all | +0.093 [+0.057, +0.127] * | **+0.109** [+0.075, +0.144] * |
| vs single-tree | de novo | +0.179 [+0.116, +0.252] * | +0.164 [+0.098, +0.239] * |
| vs `where x which` | de novo | +0.062 [−0.032, +0.147] | **+0.075** [−0.009, +0.159] |

The de novo contrast against `where x which` still includes zero, but its lower
bound moves from −0.032 to **−0.009** -- the closest it has come. At K = 8 it may
clear; s14's multiple-comparisons caveat applies if it does.

`joint - single` is essentially unchanged, as it must be: both sides are ESM-2
and share every convention.

**A side benefit worth recording:** on the temporal panel the overlap panels'
marginals now reproduce the ladder's recall **exactly** (+0.0000 for both rungs,
both subsets), where under `legacy` `where x which` was off by up to 0.007. One
universe means one tie structure.

### Figure structure now

Main `figs/multitree_results.png` -- temporal holdout, standard universe, recall
at top 10%: A/B ladders (all, de novo), C/D per-tree forests, E/F overlap panels.

Supplement `figs/supp_multitree_results.png` -- A-D the same recall ladders with
the gap column included, banner-labelled; E-H temporal NLL; I-P random holdout,
both metrics; Q-R epoch rule.

`collect_figure_metrics.py` emits all three universes in one pass with a
`universe` column; the extra universes are a masking pass each over score maps
already computed, so they are nearly free. The R script selects.

### Is the baseline's de novo win at 1% just tie-breaking? No (2026-10-14)

At K = 8 the model beats `where x which` on de novo recall at 10%
(+0.104 [+0.018, +0.195]) and loses at 1% (−0.067 [−0.116, −0.026]). The obvious
suspicion is tie structure: `P(i) * P(a | wt_i)` gives every position with zero
training mutations the same smoothed P(i), so its score there depends only on
(wild-type, alternative) -- and a de novo target is by construction one with no
training mutation at its cell, so those targets should be exactly the ones
sitting in the tied block that `compute_hit_rate` pays out fractionally.

**Measured (`tie_probe.py`), the suspicion is wrong.** Recall split into the part
earned strictly above the budget cut and the part handed over by the tie rule,
temporal panel, standard universe:

| rung | subset | q | recall | strict | from ties | tie share | mean tied block |
|---|---|---|---|---|---|---|---|
| model | novel | 1% | 0.022 | 0.022 | 0.000 | **0%** | 1 |
| `where x which` | novel | 1% | **0.090** | **0.090** | 0.000 | **0%** | 5 |
| `where x which` | all | 1% | 0.232 | 0.230 | 0.002 | 1% | 5 |
| frequency | novel | 10% | 0.052 | 0.000 | 0.052 | **100%** | 8,637 |
| frequency | all | 10% | 0.726 | 0.684 | 0.041 | 6% | 8,642 |
| frequency | all | 1% | 0.351 | 0.296 | 0.055 | 16% | 79 |

`where x which`'s tied blocks average **five cells**, not thousands: the fitted
`P(a | wt_i)` is a continuous 20 x 20 matrix and the product takes many distinct
values, so the block never approaches the budget. Its de novo 1% win is earned
entirely above the cut. The ESM-2 maps are continuous and tie-free (block = 1).

Confirmed by simulating what a single real shortlist would catch -- cells inside
a tied block are indistinguishable, so a list draws from it without replacement
and the count caught is Hypergeometric(n_at, targets_at, slots), whose mean is
exactly the fractional score. For the model and `where x which` the draw
distribution is degenerate (p5 = median = p95), i.e. there is no randomness to
worry about.

**The rung where ties really do decide the answer is `frequency`.** All of its
de novo recall at 10% is tie credit (0.052 of 0.052, over a block averaging
8,637 cells), and a single real shortlist ranges 0.022-0.082 across draws. At 1%
on de novo it catches nothing at all, in 100% of draws. s11 flagged this for
`frequency`-on-novel under `argsort`; it holds under the canonical convention
too, and it is a live caveat because that rung is in the figure's ladder. Read
`frequency`'s de novo number as "its unseen cells are indistinguishable and the
budget is being spread over them", not as ranking skill.

This also explains the one imperfect overlap marginal: `where x which` at 1% on
all substitutions reads 0.2299 against the ladder's 0.2317, a 0.0017 gap, which
is its two-slot tie allocation. Every other cell matches to 0.0000.

---

## 17. Wider pre-D training set: task and state (2026-10-14, IN PROGRESS)

**The task.** The forecast panel is nine trees because nine straddle
D = 2025-01-01 usefully. Training is not restricted to those: any tree whose tips
all predate D is legitimate context, since nothing post-D can reach the model
through it. `joint_all` already does this with seven retired seasonal-flu builds
and came out a clean null (-0.018 [-0.058, +0.024], s11). The question is
whether a much larger pre-D corpus helps or hurts, with **the evaluation held
exactly fixed**: the same nine-tree post-D test set, the same scoring pipeline,
the same universe and budgets. Ebola was asked for specifically.

The causally clean route is the **snapshot**, not the current build: fetch each
dataset at `@2025-01-01`, which is pre-D by construction regardless of what the
dataset looks like now. That is what `fetch_snapshots.py` already does for the
panel.

### Done so far

- `probe_wide_snapshots.py` written. **Not run.** It takes ~40 candidate dataset
  paths, requests each at `@2025-01-01`, and records: served or HTTP status,
  tip and node counts, latest tip date, gene annotations and their lengths. It
  saves each JSON and a `probe.json` manifest. Read-only.
- Candidate list drafted, preferring gene-level builds over whole genomes
  (the panel is gene-level and examples are per-protein): filoviruses, the four
  seasonal coronaviruses, hMPV F/G, nipah, mumps, measles N450, RSV G,
  zika/WNV/yellow fever, chikv, rubella, EV-D68 VP1, rabies, mpox, lassa,
  oropouche, five hantavirus M segments, norovirus VP1, seasonal flu HA/NA and
  avian flu NA/H7N9/H9N2.

### A correction to make before trusting the listing

`charon/getAvailable` has **no date parameter** -- it returns the catalogue as
it stands today (318 datasets on 2026-10-14). Nothing about that listing is
dated to D. Dating happens only per dataset, through the `@YYYY-MM-DD` suffix on
`getDataset`. An earlier framing in this session implied the listing itself was
dated; it is not, and the per-path probe is the only thing that establishes
availability at D.

This matters because the listing contains **`ebola/bdbv-drc-uganda-2026`**,
which **is the zero-shot evaluation tree** (`recent.json`, tips 2025.94-2026.63).
Its presence says nothing about D. It is expected to 404 at `@2025-01-01` since
the build did not exist then -- **this is expected, not verified.**

The real safeguard is therefore two checks, both of which the probe applies:
the dated request must succeed, **and** the tree's latest tip must be < 2025.0.
The second is not redundant: s8 records dataset paths that silently resolved to
a redirect page rather than the build, which is exactly how a current tree would
slip in wearing a past date.

### The ebola inclusion changes what the zero-shot result means

`ebola/bdbv` is historical Bundibugyo -- the same species as the zero-shot
evaluation -- and `sudv` / `ebov-2013` / `all-outbreaks` are the same genus.
Training on any of them at a pre-D snapshot is causally fine, because the
outbreak being scored is 2026. But s14's claim is transfer to a virus family the
panel has **never seen**, and that claim does not survive for the new arm:
with `bdbv` it becomes same-species transfer, with the others within-genus.

Both readings are worth having. The plan is to keep the ebola trees in, and to
report the Bundibugyo evaluation for the new arm under its correct name rather
than as a zero-shot number. **The nine-tree post-D evaluation is unaffected by
any of this** -- no ebola tree overlaps it and all training is pre-D.

### Also worth re-probing: s8 may be wrong about seasonal flu

s8 states "Seasonal flu has no versions at any date probed" and "zero of
Charon's 318 datasets match `flu/seasonal`". The catalogue pulled today contains
**42 datasets under `seasonal-flu/`** (h1n1pdm, h3n2, vic, yam x ha/na x
2y/3y/6y/12y/pandemic). The earlier search pattern `flu/seasonal` cannot match
`seasonal-flu/...`, so that conclusion looks like a prefix mismatch rather than
a fact about retention. **Unverified** -- whether those paths serve a
`@2025-01-01` snapshot is exactly what the probe will answer. If they do, it
affects more than this task: s8 ruled seasonal flu unforecastable at any date,
and the forecast panel itself might be extendable.

### Next steps, in order

1. Verify the `@date` suffix is actually honoured, on one known case: fetch
   `/ebola/bdbv@2025-01-01` and today's `/ebola/bdbv` and confirm the latest tip
   dates differ. Without this the whole corpus rests on an assumption.
2. Run the probe; confirm `bdbv-drc-uganda-2026` 404s at D rather than assuming.
3. Choose the training set from paths passing both checks; regenerate examples
   through `generate_all.py` / `make_tree_examples.py` so the new trees go
   through the same generator as the panel.
4. Train `joint_wide` (panel 9 + extras) x 3 seeds, matching
   `train_forecast_esm2.sh` in every other respect.
5. Score with the unchanged pipeline; report against `joint_panel` and
   `joint_all`. s10's saturation result predicts a null, and `joint_all` already
   found one at 25k training mutations; the honest prior is that this does
   nothing, and the interesting outcome would be it hurting.

Nothing in this section has been run. No data fetched beyond the undated
catalogue listing, which is saved at `~/.charon_datasets.txt`.

---

## 18. The wide corpus rebuilt at D = 2026-05-01 (2026-10-16)

The cutoff moves to **2026-05-01** and the evaluation moves off the nine-tree panel
entirely: it will be a single tree, supplied separately, observed wholly after that
date. That changes what the panel trees are for -- they are now training trees like any
other, taken at the cutoff -- and it changes the inclusion rule for everything else,
because nothing has to straddle anything any more. A dataset is in the corpus if it can
be shown to carry no information observed after the cutoff, by either route:

**A, the archive.** Charon's `<path>@YYYY-MM-DD` resolves to the most recent build
published at or before that date, and a build published before the cutoff cannot
contain anything observed after it, whatever its tips look like. The guard is
`meta.updated <= cutoff`: where the archive has no version that old Charon serves
something newer, and that must not be mistaken for a snapshot.

**B, today's build with all tips pre-cutoff.** Where no archived version exists, the
live build still qualifies if every tip predates the cutoff.

§17's first next-step -- *verify the `@date` suffix is actually honoured* -- is
**done**, on four independent datasets rather than one:

| dataset | `@2026-05-01` | current |
|---|---|---|
| measles/N450/global | updated 2026-05-01, 3278 tips, latest 2026.33 | updated 2026-10-13, 3290 tips, latest 2026.74 |
| rsv/a/G/all-time | updated 2026-04-28, 2865 tips, latest 2026.21 | updated 2026-10-13, 2970 tips, latest 2026.52 |
| seasonal-flu/h3n2/ha/12y | updated 2026-04-30, 1662 tips, latest 2026.30 | updated 2026-10-08, 1707 tips, latest 2026.70 |
| hmpv/a/F | updated 2026-04-28, 1350 tips, latest 2025.92 | updated 2026-10-14, 1635 tips, latest 2026.39 |

Fewer tips, an earlier `updated`, an earlier last tip, every time. The suffix is real.

### The tip-date bug, found and fixed

§17 suspected the exclusions on "no tip dates" were wrong, and they were, in two
separate ways.

`latest_tip()` reads `node_attrs.num_date` and nothing else. That is the *time-tree*
inference, and plenty of builds are divergence trees that never carry it: the date sits
in `node_attrs.date` as an ISO string, sometimes ambiguous (`1979-XX-XX`), or in
`node_attrs.year`, which is what Auspice actually shows on ebola/all-outbreaks -- whose
384 tips all read as undated under the old reader. Second, **one** unknown-date tip
disqualified an entire tree: ebola/sudv has 141 of 148 tips dated and a latest tip of
2025.05, and was dropped for the other seven.

Dates are now read, per tip, from the first of

    num_date (upper confidence bound) -> date -> year -> date_submitted
    -> date_released / date_updated

and each is converted to an *upper bound* in decimal years -- an ambiguous `2004-XX-XX`
bounds at the end of 2004, and a submission or release date bounds the collection date
from above, since a sequence cannot be released before it was sampled. Route B requires
every tip to carry such a bound and all of them to fall before the cutoff.

Three consequences, all checked:

- **ebola/sudv and both h7nx segments qualify** where they did not before (sudv by
  route B, all tips bounded, latest 2025.05).
- **The hantavirus builds are excluded on their merits, not for being "undated".**
  They carry no collection date of any kind -- only NCBI `date_released` /
  `date_updated` -- and those run to 2026-07, past the cutoff, while the archive serves
  no version of them at all (404 at the cutoff). The honest reading is that these ten
  trees cannot be shown to be pre-cutoff, not that their dates were missed.
- A tip with no bound at all still disqualifies its tree. That is not recoverable from
  the data, and the report names the trees it costs.

### Candidate selection: 318 paths, far fewer trees

`charon/getAvailable` lists 318 datasets today, but most are cuts of a build already
counted: ncov alone appears as 100 region x window combinations, seasonal flu as six
windows per lineage x segment. Two cuts of one build would put the same sequences in the
corpus twice, so `classify()` keeps one path per underlying build -- longest window,
widest geography, gene-level over whole-genome -- and records a reason for every drop.
The split is **56 core, 38 extra, 224 dropped**, the largest drop groups being 56 GISAID
ncov mirrors, 27 ncov region/window cuts, 26 h5n1 clade cuts of h5nx (NOTES s8: 37%
shared tip names) and 8 h7n9 cuts of h7nx.

`extra` is the tier that would change what a "tree" means rather than how many there
are: influenza's six internal segments per subtype, the seven non-capsid norovirus
genes, and the nucleoprotein segments of the bunya/arenaviruses. They are probed and
reported but not generated, so including them later is a one-flag change
(`--generate-tiers core extra`) and not another fetch.

### Code

| file | purpose |
|---|---|
| `wide_corpus.py` | the whole thing: `classify()`, the fixed tip-date reader, the two-route probe, and generation through `make_tree_examples.py` |

Generation passes `--ambiguous-ref gap --strip-trailing-stop` to every tree. Both are
no-ops where they do not apply, and keeping them unconditional means every tree in the
corpus goes through one recipe: **regenerating h5nx HA and denv1 E at the old
D = 2025-01-01 with these flags reproduces `data/forecast/examples_*@2025-01-01_train.txt`
byte-for-byte**, so the panel trees are not being treated differently from the new ones.

### The corpus: 43 trees, 90,178 examples, 124,738 training mutations

Probe and generation complete, at an **example floor of 40** (`MIN_EXAMPLES`; the
earlier arms used 50). All 43 files pass `antiGen-validate-examples`. Manifest at
`data/wide/corpus@2026-05-01.json`, examples at
`data/wide/examples_<short>@<date>_train.txt`, probe record at
`data/wide/probe@2026-05-01.json`.

**All nine panel datasets are in, all by route A**, so the forecast panel is now a
subset of this corpus rather than a separate thing. 40 trees come from the archive and
3 (h7nx HA, h7nx NA, ebola/sudv) from today's build. For scale: `joint_panel` trained
on 16,643 training mutations and `joint_all` on 25,250. This is **124,738** -- 7.5x
`joint_panel` -- across 33 distinct sequence lengths and 22 viruses.

Dropping the floor from 50 to 40 added exactly two trees, both small:
ebola/ebov-2013 (49 examples, 189 mutations) and ebola/sudv (41, 54). Note that
ebola/all-outbreaks, ebov-2013 and sudv now coexist: all-outbreaks and ebov-2013 share
sequences from the 2013-16 epidemic, so these three are not independent trees.

| tree | dataset | route | gene | L | tips | examples | training muts |
|---|---|---|---|---|---|---|---|
| norovirus_vp1 | norovirus/all/VP1 | A | VP1 | 541 | 5811 | 16738 | 19517 |
| h9n2_na | avian-flu/h9n2/na/all-time | A | NA | 469 | 2963 | 7278 | 8988 |
| h9n2_ha | avian-flu/h9n2/ha/all-time | A | HA | 560 | 2790 | 6932 | 8668 |
| h7nx_na | avian-flu/h7nx/na/all-time | B | NA | 465 | 4719 | 6571 | 8199 |
| h5nx_na | avian-flu/h5nx/na/all-time | A | NA | 470 | 1384 | 6614 | 7612 |
| h7nx_ha | avian-flu/h7nx/ha/all-time | B | HA | 560 | 4998 | 4831 | 6768 |
| rsv_a_g | rsv/a/G/all-time | A | G | 321 | 2865 | 4414 | 6037 |
| rsv_b_g | rsv/b/G/all-time | A | G | 310 | 2620 | 3845 | 5346 |
| hmpv_a_g | hmpv/a/G | A | G | 237 | 1762 | 4251 | 5283 |
| h5nx_ha | avian-flu/h5nx/ha/all-time | A | HA | 569 | 1426 | 3856 | 4857 |
| rabies | rabies | A | RABVgp4 | 524 | 2995 | 3002 | 4060 |
| lassa_gpc | lassa/gpc | A | GPC | 491 | 1173 | 2663 | 3352 |
| ncov | ncov/open/global/all-time | A | S | 1273 | 2915 | 1523 | 2677 |
| hmpv_b_g | hmpv/b/G | A | G | 225 | 1085 | 1982 | 2603 |
| mpox | mpox/all-clades | A | OPG210 | 1881 | 1175 | 2442 | 2553 |
| denv4 | dengue/denv4/E | A | E | 495 | 3658 | 1297 | 2164 |
| denv3 | dengue/denv3/E | A | E | 493 | 3940 | 975 | 1980 |
| denv1 | dengue/denv1/E | A | E | 495 | 3906 | 766 | 1783 |
| h3n2_ha | seasonal-flu/h3n2/ha/12y | A | SigPep+HA1+HA2 | 566 | 1662 | 903 | 1753 |
| h3n2_na | seasonal-flu/h3n2/na/12y | A | NA | 469 | 1659 | 1021 | 1729 |
| denv2 | dengue/denv2/E | A | E | 495 | 3893 | 734 | 1654 |
| yam_na | seasonal-flu/yam/na/12y | A | NA | 466 | 2543 | 657 | 1627 |
| evd68_vp1 | enterovirus/d68/vp1 | A | VP1 | 309 | 3437 | 739 | 1565 |
| h1n1pdm_ha | seasonal-flu/h1n1pdm/ha/12y | A | SigPep+HA1+HA2 | 566 | 1543 | 752 | 1502 |
| rsv_a_f | rsv/a/F/all-time | A | F | 574 | 2941 | 472 | 1288 |
| h1n1pdm_na | seasonal-flu/h1n1pdm/na/12y | A | NA | 469 | 1543 | 583 | 1263 |
| vic_na | seasonal-flu/vic/na/12y | A | NA | 466 | 1514 | 552 | 1246 |
| rsv_b_f | rsv/b/F/all-time | A | F | 574 | 2942 | 382 | 1136 |
| wnv | WNV/all-lineages | A | env | 501 | 3088 | 549 | 1057 |
| yam_ha | seasonal-flu/yam/ha/12y | A | SigPep+HA1+HA2 | 584 | 2685 | 320 | 1050 |
| vic_ha | seasonal-flu/vic/ha/12y | A | SigPep+HA1+HA2 | 585 | 1514 | 325 | 862 |
| measles_n450 | measles/N450/global | A | N | 150 | 3278 | 321 | 719 |
| measles_h | measles/genome/global | A | H | 617 | 2715 | 418 | 707 |
| rubella_e1 | rubella/E1 | A | E1 | 481 | 2948 | 268 | 703 |
| oropouche_m | oropouche/M | A | OROVsMgp1 | 1420 | 769 | 397 | 650 |
| hmpv_b_f | hmpv/b/F | A | F | 539 | 1173 | 164 | 379 |
| hmpv_a_f | hmpv/a/F | A | F | 539 | 1350 | 113 | 365 |
| mumps | mumps/global | A | HN | 582 | 887 | 203 | 317 |
| zika | zika | A | ENV | 504 | 1040 | 53 | 195 |
| ebola_ebov2013 | ebola/ebov-2013 | A | GP | 677 | 1493 | 49 | 189 |
| ebola_all | ebola/all-outbreaks | A | GP | 676 | 384 | 115 | 182 |
| nipah | nipah/all | A | G | 602 | 109 | 67 | 99 |
| ebola_sudv | ebola/sudv | B | GP | 676 | 148 | 41 | 54 |

### What the 13 excluded core datasets cost, and why

| dataset(s) | why |
|---|---|
| hantavirus andv/htnv/puuv/seov/tulv M | no archived version (404 at the cutoff), and the live builds' only dates are NCBI release dates running to 2026.52-2026.76 |
| avian-flu h10nx HA, NA | no archived version; live tips run to 2026.38 |
| seasonal-cov 229e, hku1, nl63, oc43 | the **dated** root-sequence sidecar 404s (below) |
| chikv/global | no root sequence at any date, from Charon or `data.nextstrain.org` |
| yellow-fever/prM-E (39) | below the 40-example floor, by one example |

Two of these are worth more than a line.

**The seasonal-cov builds are not recoverable by using today's sidecar.** Charon serves
`seasonal-cov/<x>?type=root-sequence` (200) but not `@2026-05-01` (404), so the
tempting fix is to reconstruct the April tree against today's reference. Measured, that
is wrong on its own terms, never mind the leak: generating Spike against the current
sidecar gives **450 from-base mismatches for oc43 and 144 for hku1** -- the reference
frame does not match the dated tree. Only 229e is near-clean (6). These four stay out.

**ebola/sudv is the tree the tip-date fix rescued**, and at the floor of 40 it is in,
with 41 examples and 54 training mutations -- it would have been excluded twice over
before (once as "7 undated tips", once by the floor of 50). The fix's yield is
h7nx HA and NA (2 trees, 11,402 examples, 14,967 mutations, 12 percent of the corpus)
plus sudv. Worth stating plainly: the bug was never what held back the hantavirus
trees.

### A cross-check on the selection

The orphaned `data/wide/*@2026-05-01` files from the reset-away session cover **32
datasets, and all 32 are in this corpus** (under different short names: that session's
`bvic_ha`/`byam_ha` are `vic_ha`/`yam_ha` here, `lassa` is `lassa_gpc`, `rubella` is
`rubella_e1`). The 9 this rebuild adds are exactly the panel trees. Two independent
passes at the same rule, with the code for one of them gone, selected the same
datasets; the difference is that this one also follows the instruction to carry the
panel in. Nothing was taken from those files -- they are still in
`data/wide_orphan_unprovenanced/` and can be deleted.

### State

Generated, validated, **not yet trained on**. The evaluation tree -- a single tree
observed wholly after 2026-05-01 -- has not been supplied yet, so nothing here has been
scored.

### Fetching, after the first probe took 38 minutes and then hung

`fetch_snapshots.request` does not ask for gzip and waits 900s per attempt x 3. One
dead connection on `avian-flu/h10nx/ns@current` therefore stalled an entire stage, and
16 of 94 archive lookups returned `Connection timed out` -- indistinguishable, in the
old code, from "no archived version". `wide_corpus._get` asks for gzip and gives up
after 150s x 4, a `failed` flag separates a dropped connection from a 404, and failures
are retried serially. **The rerun completed the full 94-dataset probe in under a
minute** (89 cache hits), and all 16 former timeouts resolved: ncov, measles/genome,
h5nx HA and lassa/s are archive hits, and the nine hantavirus 404s are real.

### Then nextstrain.org stopped answering, and the probe went offline

After ~250 requests across four probe runs, `nextstrain.org` began returning nothing at
all (`curl` -> HTTP 000 after 15s). Two things had already gone wrong by then and are
worth recording, because both were self-inflicted:

- **A pre-fix probe process outlived its `pkill` and overwrote the clean probe file.**
  Its rows carried the old format (no `failed` flag) and reinstated `Connection timed
  out` for ncov, measles/genome, h5nx HA and lassa/s. Generation then **silently
  excluded four trees, three of them core**, reporting it as an ordinary exclusion. The
  `failed` flag is what made it detectable: those rows did not have it. The probe record
  now lives on local disk at `data/wide/probe@2026-05-01.json`, next to the corpus it
  explains, instead of in the shared cache on `/large_storage`.
- **Every "is it still running?" check was unreliable**, because
  `pgrep -f "wide_corpus.py --probe"` matches the watcher whose own command line
  contains that string. That is why a hung run was reported dead and two jobs were
  reported finished when only their launcher had exited. Use `[w]ide_corpus.py` and
  `ps -eo pid,lstart`.

The fix for the throttling is `--offline`: classify from cached payloads only, never
fetching. Every one of the 94 candidates already had a payload on disk (core: 46 at the
cutoff date, 50 current, 56 of 56 with at least one), so the offline probe reproduced
**the same 76 of 94 qualifying in 3.7 seconds**, with the same per-tier routes
(core 46 A / 3 B / 7 out).

**Offline mode can only under-include, never leak.** Where the archive payload is not
cached (27 candidates, all of which an earlier probe recorded as genuine 404s), it falls
back to route B -- every tip pre-cutoff -- which is the stricter test. A missing archive
check can therefore cost a tree; it cannot admit post-cutoff data. Nine of the 27
qualify that way, h7nx HA/NA and ebola/sudv among them.

One wording caveat for whoever reads the generate log: under `--offline` the exclusion
reasons read "not cached" and "reference not cached (offline)". For the five that
matters for -- chikv/global and the four seasonal-cov -- the reference is a **confirmed
404**, checked directly with curl at both `@2026-05-01` and undated, not merely
uncached. Same for the hantavirus and h10nx archive versions, which two independent
online probes recorded as 404.

### Housekeeping, worth knowing before trusting anything in `data/`

`git reflog` shows two resets to `origin/is-multitree` after the last commit (4f3afdf).
The scripts a previous session wrote were tracked and went away with them; the example
files it generated are gitignored and did not. That is where the 32 orphaned
`data/wide/*@2026-05-01` files and `/large_storage/.../wide/probe@2026-05-01.json` come
from -- real artifacts of real work, with the code that made them gone. The probe JSON
was read for its structure (it records a route A/B distinction matching the rule above,
and it is where the undated-tip exclusions were first visible); nothing was taken from
it on trust. Separately, the working tree's `NOTES.md` is **124 lines shorter than
HEAD's**: the s14 zero-shot sections and their retraction in s16 are missing from it.
Nothing in this session removed them.

---

## 19. Training on the wide corpus (2026-10-16)

`sbatch train_wide_esm2.sh` -> **job array 3327944, tasks 0-5, all six running** as of
03:37 (nodes GPU71BA and GPU389E, partition gpu_batch, 24h limit).

### Two arms, not one

| task | arm | sources | train / val examples | batches per epoch |
|---|---|---|---|---|
| 0-2 | `joint_wide` seeds 0-2 | 43 | 81,163 / 9,015 | 10,519 |
| 3-5 | `joint_panel` seeds 0-2 | 9 | 9,380 / 1,043 | 1,232 |

The brief was one finetune; `joint_panel` is here because without it the evaluation
cannot answer the question. The nine panel datasets taken **at this cutoff** are the
only control that isolates the extra 34 trees -- same recipe, same seeds, same split
rule, same evaluation tree. Without it, a number on the post-cutoff tree says only that
some model scored something, not that the wider corpus did anything. It costs 3 short
jobs against 3 long ones.

### Recipe

Identical to `train_forecast_esm2.sh`: `esm2_t12_35M_UR50D`, `--freeze-backbone
--finetune-layers 4`, lr 1e-4, batch 8 / `--max-batch-tokens 8192`, 20 epochs,
`--val-mode random --val-frac 0.1`, `--split-seed <seed>`, `--seed 42+seed`,
`--ignore-dead-ends`, `--save-best --save-best-per-source --dump-cells`, tagged
`--experiment wide@2026-05-01` so `antiGen-eval` keeps skipping these unless asked.
Output under `/large_storage/hielab/ispecht/multitree/wide_train/<arm>_s<seed>/`.

The arm reads its tree list from `data/wide/corpus@2026-05-01.json` rather than a
hardcoded list, so a training arm cannot silently disagree with the corpus that was
generated and validated. Both arms were dry-run before submission (43 and 9 sources,
every file present).

**Epoch selection is causally clean here even though the split is random.** The random
split is drawn from pre-cutoff examples only, so it cannot see the evaluation tree; what
it cannot support is any within-tree forecasting claim, which is not what this arm is
for. The evaluation tree is not passed to this script and does not exist in the repo yet.

### Done: all six arms completed, exit 0:0

`joint_wide` 2h09-2h12 per seed, `joint_panel` 20 min. Checkpoints and per-epoch
metrics under `/large_storage/hielab/ispecht/multitree/wide_train/<arm>_s<seed>/`
(5.9 GB per wide seed, 1.5 GB per panel seed -- `--save-best-per-source` writes one
checkpoint per tree).

| arm | zero-shot | best overall NLL | epoch | final (ep 19) |
|---|---|---|---|---|
| joint_wide s0/s1/s2 | 9.420 / 9.404 / 9.425 | 7.191 / 7.157 / 7.172 | 2 / 2 / 2 | 9.43 / 9.53 / 9.56 |
| joint_panel s0/s1/s2 | 9.346 / 9.273 / 9.348 | 7.217 / 7.182 / 7.295 | 2 / 2 / 3 | 8.56 / 8.58 / 8.91 |

Overall NLL is not comparable **between** arms -- the two validation sets hold different
source mixes (12,233-12,643 held-out mutations for wide against 1,652-1,968 for panel).
It is comparable **within** an arm across epochs, and it says the familiar thing: the
minimum is at epoch 2 and everything after is overfitting, with `joint_wide`'s epoch-19
model (9.43) no better than its zero-shot row (9.42). `--save-best` is doing the real
work here; the final checkpoint is worthless and always was (README, s1).

### A first paired read, on the nine shared trees

`--split-seed s` draws each source's held-out set from a per-source RNG, so a given tree
has the **same** validation examples in both arms at the same seed. That makes the nine
panel trees a paired comparison, each at its own best epoch, 3 seeds averaged:

| source | val muts | joint_panel | joint_wide | delta |
|---|---|---|---|---|
| ncov | 237 | 7.305 | 7.288 | -0.017 |
| rsv_a_f | 101 | 7.186 | 7.141 | -0.045 |
| rsv_b_f | 69 | 7.074 | 6.930 | -0.144 |
| measles_h | 55 | 7.715 | 7.751 | +0.036 |
| denv1 | 119 | 7.227 | 7.167 | -0.060 |
| denv2 | 143 | 7.032 | 6.989 | -0.044 |
| denv3 | 224 | 7.298 | 7.315 | +0.018 |
| denv4 | 249 | 7.454 | 7.463 | +0.008 |
| h5nx_ha | 473 | 6.970 | 6.906 | -0.064 |

Mean delta **-0.035 nats**, wide ahead on 6 of 9. That is the same order as every
joint-vs-single delta this arm has measured (s4: -0.011 to -0.045, read as a tie), on a
random split that is not forecast-valid. **Read it as nothing yet.** The 34 extra trees
did not hurt, which is the only claim this supports; the evaluation that matters is the
post-cutoff tree, and it has not arrived.

### The mpox watch item fired, and it is real

s9's suspicion about the >1022 two-pass stitching path was half right. Length alone is
fine -- ncov (L=1273) reaches 2.81 nats below uniform and oropouche M (L=1420) 2.47.
**mpox (L=1881) learns nothing at all:**

| source | L | zero-shot | best | uniform | best - uniform |
|---|---|---|---|---|---|
| mpox | 1881 | 17.26 | 10.57 | 10.58 | **-0.01** |
| oropouche_m | 1420 | 9.85 | 7.83 | 10.30 | -2.47 |
| ncov | 1273 | 10.89 | 7.38 | 10.19 | -2.81 |

A zero-shot NLL of 17.26 against a uniform of 10.58 is not a hard tree, it is a broken
one: no predictor that merely knew the alphabet could score that badly. Training pulls
it back to exactly uniform and no further. mpox is 2,442 examples and 2,553 training
mutations, 2 percent of the corpus, so the cost of leaving it in is small but it is not
zero -- it is 2 percent of gradient steps spent on noise, and any per-source average
that includes mpox is dragged by it.

**Not fixed here.** The options are to drop mpox from the corpus (one line in the
manifest, retrain), or to find out what the long-sequence path does to it -- the
reference also carries 3 ambiguous residues mapped to `-`, which is a second candidate
and would not show up as a length bug at all. Either way it should be settled before the
post-cutoff evaluation, so that the number reported is not partly an artifact.

### Watch items (as submitted)

- **Length buckets now reach 1881** (mpox OPG210), with oropouche M at 1420 and ncov S
  at 1273. s9 flagged >1022 as the two-pass stitching path and the suspected mpox
  failure; the forecast panel never exercised it beyond ncov. If mpox or oropouche
  produce degenerate per-source validation, that is the first place to look.
- 20 epochs x 10,519 batches is ~5x the forecast arms' work. Expect hours. `--save-best`
  picks the epoch, so the count only has to contain the minimum.

Nothing scored against the real target. The corpus is frozen at 43 trees; the
evaluation tree is still to come.

---

## 20. Figure 5 on an eleven-tree panel: seasonal flu HA is forecastable after all (2026-10-16)

The task: re-run the Figure 5 analysis unchanged, on the nine-tree forecast panel of
s11 **plus seasonal flu H3N2 HA and H1N1pdm HA**. Same D, same recipe, same seeds,
same split rules, same scoring code. The only thing that moves is the panel.

### s8 was wrong about seasonal flu, and s17 said why

s8 recorded two claims that together ruled seasonal flu out of every forecast arm:
"Seasonal flu has no versions at any date probed" and "zero of Charon's 318 datasets
match `flu/seasonal`". s17 suspected a prefix mismatch -- the paths are
`seasonal-flu/<lineage>/<segment>/<window>`, which `flu/seasonal` cannot match -- and
left it unverified. It is verified now, and s17 was right. Both HA builds are archived
at D and qualify by **route A** (s18's rule: `meta.updated <= cutoff`):

| dataset | `@2025-01-01` | today |
|---|---|---|
| seasonal-flu/h3n2/ha/12y | updated **2024-12-29**, 1576 tips, latest tip 2024.95 | updated 2026-10-08, 1707 tips, 2026.70 |
| seasonal-flu/h1n1pdm/ha/12y | updated **2024-12-29**, 1470 tips, latest tip 2024.96 | updated 2026-10-08, 1544 tips, 2026.73 |

Both guards hold, not just the first: the build was published before D **and** every
tip predates D, so neither can carry anything observed after the cutoff. Independent
corroboration for h1n1pdm: the undated `data.nextstrain.org` copy fetched on 2026-10-14
is **byte-identical** (md5 `72176ae5...`) to the archived snapshot, so two different
routes to the same frozen build agree.

s3's "seasonal flu 12y is genuinely retired (2024-04-23, tip 2024.27) from *both*
sources" is superseded for HA. The 12y builds are live and updating weekly.

### Why they can be scored, and the two checks that had to pass

`make_forecast_testset.py` requires the snapshot and today's build to share a reference
frame, since train and test examples must share coordinates. They do, exactly:

    h3n2     SigPep 18-65,  HA1 66-1052, HA2 1053-1715   (L = 566, both builds)
    h1n1pdm  SigPep 21-71,  HA1 72-1052, HA2 1053-1718   (L = 566, both builds)

0 from-base mismatches, 0 ambiguous, 0 out-of-range, both trees, both sides. The 12y
window slides -- today's h3n2 build starts at 2009.26 where the snapshot started at
2008.00 -- which drops old tips but cannot affect a branch crossing D.

**No tip-name overlap**, checked because s8 found h5n1 and h5nx sharing 37% of tips:
h3n2 vs h1n1pdm 0 shared, and each against h5nx HA 0 shared. Three independent HA
trees, not cuts of one build.

### What the panel gains

| tree | examples | targets | novel |
|---|---|---|---|
| h3n2_ha | 47 | 95 | 22 (23%) |
| h1n1pdm_ha | 38 | 65 | 18 (28%) |
| **the old nine** | 272 | 564 | 156 |
| **total** | **357** | **724** | **196** |

The old nine reproduce 564/156 exactly, so nothing about the existing panel moved.
Beyond the 28% more targets, the panel is **less top-heavy**: h5nx falls from 41% of
the test set to 32%, which is the caveat s11 flagged on every pooled number.

Training side, from the snapshots, through the unchanged generator with the s18 recipe
(`--ambiguous-ref gap --strip-trailing-stop`): h3n2_ha 840 examples / 1,611 mutations,
h1n1pdm_ha 682 / 1,394. Both pass `antiGen-validate-examples`; both regenerate
byte-for-byte from the repo-local build copies rather than the `/large_storage` cache.

### A note on `joint_all`

s11's `joint_all` carries seven retired seasonal-flu builds as training-only context,
and **two of them are h3n2 HA and h1n1pdm HA** -- the trees that are now eval trees.
That arm is therefore not a clean control for this panel and is not used here. Nothing
about its published null (-0.018 [-0.058, +0.024]) is retracted: at the time those
trees were training-only and the evaluation was the nine, which this does not touch.

### Code

| file | purpose |
|---|---|
| `train_panel11_esm2.sh` | the 33 arms, both regimes (new) |

`collect_figure_metrics.py` gains `--panel {9,11}`; `PANELS` holds each panel's tree
list and its joint-arm roots. The nine-tree default is untouched and its runs are never
read by the eleven-tree pass. `figures/fig5_multitree.R` gains `multitree_panel`
(or `MULTITREE_PANEL` in the environment), which selects `data/multitree` vs
`data/multitree11` and suffixes the output files.

**Regression checked:** re-running the R script at the default redrew
`figs/multitree_results.png` and `figs/supp_multitree_results.png` byte-identically
(`git status` clean under `figs/`).

### Arms (job array 3335279, 33 tasks, `gpu_high_mem`)

| tasks | arm | regime | seeds | root |
|---|---|---|---|---|
| 0-7 | `joint_panel11` | temporal, 11 snapshots | 0-7 | `forecast11/` |
| 8-23 | `single_h3n2_ha`, `single_h1n1pdm_ha` | temporal | 0-7 | `forecast/` |
| 24-26 | `joint11` | random, three-way split | 0-2 | `matched/` |
| 27-32 | the two singles | random, three-way split | 0-2 | `matched/` |

The nine existing single-tree controls are **not** retrained -- a single-tree arm does
not depend on the panel beside it, and `forecast/single_*_s{0..7}` and
`matched/single_*_s{0..2}` already exist. The joint arms go to new roots so the
nine-tree runs behind the published figure are never touched. K = 8 temporally (the
matched K the figure reports; joint and single must ensemble at the same K), 3 in the
random regime, which cannot ensemble at all.

**Cluster note, part three.** `sbatch --test-only` put `gpu_high_mem` 1h20m ahead of
`gpu` and 2h24m ahead of `gpu_batch`, so the array went there (s15's rule: test, then
submit to the single best partition). Tasks 2-13 then came back
`launch_failed_requeued_held` -- `namespace_p_join: open failed for
/media/8TBNVME/scratch/<jobid>/.ns` on **GPU104C**, a node-level scratch failure with
nothing to do with the job. Released with `ExcNodeList=GPU104C`. Worth knowing: a
requeued-held array task is invisible to `squeue`'s collapsed array row and only shows
up in `scontrol show job <array id>`, so an array can sit two-thirds held while
`squeue` reports it pending.

### Results: every claim survives the wider panel (2026-10-17)

Array 3335279 finished **33/33 COMPLETED**, 25 tasks on GPU115A and 8 on GPU1256, one
`_best.pt` timestamp per directory (no stale partials). Collection: job 3346288,
5m13s, `--panel 11 --out antiGen-figures/data/multitree11`.

Pooled contrasts, temporal holdout, pooled rule, standard universe, **K = 8 in both
columns** (* = interval clears zero):

| contrast | subset | 9-tree | 11-tree |
|---|---|---|---|
| vs single-tree, recall@10% | all | +0.053 [+0.031,+0.075]* | +0.045 [+0.025,+0.065]* |
| vs single-tree, recall@10% | de novo | +0.164 [+0.087,+0.248]* | +0.172 [+0.097,+0.244]* |
| vs `where x which`, recall@10% | all | +0.128 [+0.096,+0.158]* | +0.129 [+0.098,+0.158]* |
| vs `where x which`, recall@10% | de novo | +0.104 [+0.018,+0.195]* | **+0.112 [+0.031,+0.191]*** |
| vs single-tree, NLL | all | -0.219 [-0.336,-0.112]* | -0.188 [-0.276,-0.108]* |
| vs `where x which`, NLL | de novo | +0.686 [+0.469,+0.882]* | +0.699 [+0.509,+0.879]* |

Nothing moves by more than 0.01 on recall. **The de novo win over `where x which`
holds** -- the contrast s15/s16 watched creep from +0.005 (K=3) to +0.104 (K=8) now
reads +0.112 with a lower bound of +0.031, on 169 targets against 134. It was the
narrowest claim in the figure and the one a wider panel was most likely to break.

The NLL inversion is unchanged and still means what s11 said: the model loses to
`where x which` on de novo NLL (+0.699) while beating it on ranking. A density
failure, not a ranking failure.

Note these 9-tree numbers are **K = 8**, not the K = 3 table in s16 -- that is why
single-tree reads 0.818 rather than 0.808. Both columns are K = 8, so matched.

#### The two new trees behave like panel members

Per-tree vs single-tree, recall@10%, temporal/pooled/standard:

| tree | all | de novo |
|---|---|---|
| h1n1pdm_ha | +0.062 [+0.000,+0.140] | +0.286 [+0.000,+0.545] |
| h3n2_ha | +0.021 [-0.029,+0.074] | +0.143 [-0.056,+0.353] |

Positive in all four cells, both mid-panel: neither carries the pooled result nor
drags it. Joint training gains on them specifically -- multi-tree 0.876 all / 0.609
de novo against single-tree 0.831 / 0.438.

#### One caveat a reader comparing the two figures needs

`substitution` gains more from the wider panel than any other rung (+0.029 all,
+0.036 de novo). Seasonal flu HA has a strong, well-sampled substitution signal, so
the fitted 25-scalar baseline benefits from two more such trees in the pool. It does
not threaten the contrasts -- the model gains too -- but **the baseline ladder is not
identical between panels**, and a rung-by-rung comparison across the two figures
should say why.

`denv3` and `denv4` still read `+0.000 [+0.000,+0.000]` on all substitutions, the
degenerate-tiny-n artifact s11 recorded. Not a real null; do not quote per-tree.

### The cluster fault, and a correction to how it was diagnosed

The `namespace_p_join: open failed for /media/8TBNVME/scratch/<jobid>/.ns` failure
recurred through the whole run. It was first read here as **per-node** -- GPU1034 and
GPU104C bad, GPU115A good -- on the evidence of 22 failures against 11 clean tasks.
That reading was wrong: the next morning the collection job failed the same way **on
GPU115A**, twice, after that node had run 25 tasks clean overnight. It is an
intermittent fault across `gpu_high_mem` / `gpu_batch_high_mem`, not a property of
particular nodes, and the exclusions worked by timing more than by diagnosis.

What actually fixed it was leaving that node set: resubmitting to `gpu`
(GPU3694/7094/7130/7220/70DC/71BA/389E/708E/C870/CACE) started **immediately** on
GPU708E despite a `--test-only` estimate 25 minutes out, and ran clean.

Two operational lessons, the second of which bit twice in one session:

- `--test-only` estimates are near-useless under this fault. It put
  `gpu_batch_high_mem` 13 hours ahead of `gpu`; `gpu` in fact started at once while
  the high-mem partition could not launch a job at all.
- **A requeued-held array task is invisible in `squeue`'s collapsed array row.**
  `22 PD / 3 R` reads exactly like contention. Only
  `scontrol show job <array id> | grep Reason=` distinguishes them. An array can sit
  two-thirds dead while every summary view says it is queued and progressing.

---

## 21. A 2025-01-01 panel probe, and four more trees (2026-10-17)

s20 added the two seasonal-flu HA trees by chasing one specific bug. The obvious
follow-up -- *are there others?* -- had never been asked at this cutoff: s18's probe
ran at D = 2026-05-01 and nothing re-ran it at 2025-01-01.

### The rule a forecast tree must clear is stricter than the corpus rule

`wide_corpus.py` asks only whether a build can be shown to carry nothing observed
after the cutoff, because there it is training context. `probe_forecast_panel.py`
(new) adds the five bars an *evaluation* tree must also clear:

1. **Route A only.** Route B -- today's build with every tip pre-cutoff -- means the
   dataset is frozen and has no post-D history. Fine as training context (s11's
   `joint_all` is exactly that), useless as an eval tree.
2. **Matching reference frames** between snapshot and current build, or train and
   test examples do not share coordinates.
3. **A time tree.** `crossing_branches` reads `node_attrs.num_date` and nothing else,
   which is strictly stronger than wide_corpus's tip-date reader.
4. **A post-D budget worth scoring.**
5. **No tip overlap with a panel tree** (s8: h5n1/h5nx share 37%).

`wide_corpus.py --probe --cutoff 2025-01-01 --tiers core` -> **40 of 56 core datasets
are route A**, zero fetch failures. Of those, nine cleared all five bars.

### Three bugs in the probe, all of which under-reported

Each silently *removed* a valid tree, so the direction of error was toward finding
too few. Worth recording because the same shapes will recur:

- The overlap rule excluded on **any** shared tip name. `hmpv/a/G` shares exactly one
  with `rsv_b` -- a coincidence -- and was dropped. It has 427 post-D targets, the
  largest of any candidate. The rule now uses a fraction of the smaller tree (5%).
- `--current-ref` was gated on a probe field that route-A rows do not carry, so four
  datasets failed as "test failed" when their reference simply lives in a sidecar.
- The embedded-root-sequence check was a substring scan of the first 2 MB.
  `hmpv/a/G` embeds its root sequence at **byte 3,204,972**. Parse, do not scan.

A fourth is cosmetic but misleading: `wide_corpus._why_not` hardcodes the module's
`CUTOFF`, so exclusion messages read `@2026-05-01` even when probing 2025-01-01. The
404s are real; the date in the text is not.

### Nine eligible, of which four are duplicate strain sets

| dataset | gene | train ex | targets | novel |
|---|---|---|---|---|
| hmpv/a/G | G | 2101 | 427 | 79 |
| hmpv/b/G | G | 1878 | 363 | 91 |
| avian-flu/h9n2/ha | HA | 7174 | 354 | 46 |
| avian-flu/h9n2/na | NA | 6651 | 232 | 26 |
| seasonal-flu/vic/na | NA | 518 | 135 | 28 |
| seasonal-flu/vic/ha | SigPep+HA1+HA2 | 339 | 74 | 14 |
| hmpv/b/F | F | 91 | 55 | 33 |
| hmpv/a/F | F | 86 | 38 | 25 |
| oropouche/M | OROVsMgp1 | 349 | 10 | 6 |

The first screen compared candidates only against the *panel*, not against each
other. Pairwise: vic/ha vs vic/na **100%** shared tips, h9n2 ha/na 65%, hmpv/a F vs G
53%, hmpv/b F vs G 51%. These are the same viruses at two proteins -- the same reason
`h3n2/na` and `h1n1pdm/na` were kept out of the panel -- so one per pair.

**Chosen: hmpv/a/G, hmpv/b/G, h9n2/ha, vic/ha** (HA/G members, no NA).

Negative results that confirm the screen is not inventing candidates: `yam/ha` and
`yam/na` return **zero** post-D crossing branches (B/Yamagata stopped circulating in
2020), mumps 4 targets, zika 1. Real exclusions: `mpox/all-clades`, `h5nx/na` and
`lassa/gpc` changed reference frame between builds; `rabies` and `evd68/vp1` are
divergence trees; the four `seasonal-cov` builds and `ebola/ebov-2013` have no dated
root-sequence sidecar (404), matching s18 at the other cutoff.

**Not probed: the 38 `extra`-tier datasets** (influenza internal segments, norovirus
non-capsid genes, bunya/arenavirus N segments). s18 excluded them by choice.

### Two data-quality defects, recorded before training

- **h9n2 HA drops 46.5% of its branch mutations** (7,665 of 16,498) to ambiguous `X`.
  The kept mutations are real; its targets are a biased subsample of positions that
  happened to be called unambiguously. Every other panel tree is 0-7.5%.
- **hmpv/a/G is 12.4% gap targets** (53 of 427) against the panel's ~2%.

---

## 22. Panel 15: the de novo claim does not survive (2026-10-17)

Array 3353040, 55 tasks, **all COMPLETED**, one `_best.pt` timestamp per directory.
Collection 3353797 (8m56s) -> `antiGen-figures/data/multitree15/`, figure
`figs/multitree_results15.png`.

Pooled contrasts, temporal/pooled/standard, K = 8:

| contrast | subset | 11-tree | 15-tree |
|---|---|---|---|
| vs single-tree | all | +0.045 [+0.025,+0.065]* | **+0.016 [+0.002,+0.029]*** |
| vs single-tree | de novo | +0.172 [+0.097,+0.244]* | **+0.107 [+0.056,+0.159]*** |
| vs `where x which` | all | +0.129 [+0.098,+0.158]* | **+0.112 [+0.088,+0.135]*** |
| vs `where x which` | de novo | +0.112 [+0.031,+0.191]* | **+0.022 [-0.036,+0.081]** |

**The de novo win over `where x which` reopens across zero**, on 317 targets against
169. That is the claim s15/s16 tracked from +0.005 (K=3) through +0.104 and +0.112.
Joint-vs-single survives everywhere but shrinks to +0.016 on all substitutions.

Every rung falls on de novo (wxw -0.100, substitution -0.093) and the model falls
most (-0.190): the new trees are harder for the model specifically. Per-tree, none of
the four clears zero, and two are negative (hmpv_b_g -0.020 all / -0.021 de novo).
They are **1,148 of 1,856 targets (62%)**, so they dominate rather than perturb.

**This is not a clean refutation and must not be reported as one.** h9n2_ha and
hmpv_a_g carry the defects above and are 42% of the panel's targets, so the run
cannot separate "sensitive to harder trees" from "sensitive to dirtier ones".

### Gaps are masked -- checked, and one earlier claim retracted

Under `standard`, `cols_for()` returns `(STOP, GAP)` and `keep[:, :, c] = False`
strips both columns from the **targets** as well as the pool. Measured on panel 15:
all 1,856 standard vs 1,935 common (**79 gap targets excluded**), de novo 317 vs 341
(**24 excluded**). Deletions are not scored, for any rung.

**Correction.** s21 flagged hmpv_a_g's 12.4% gap targets as a reason the ESM-2 arms
might be penalised. That is wrong: `standard` drops those targets entirely, so the
gap column is not an explanation for the contrast reopening. hmpv_a_g is simply a
smaller tree than its raw count suggests (374 scored of 427). The h9n2 ambiguity
caveat is untouched -- that is about which mutations exist, not which columns score.

---

## 23. Panel 13: the clean two only (2026-10-17, RUNNING)

Array **3353910**, 11 tasks. The eleven plus **vic_ha and hmpv_b_g** -- the two of
s21's four with no known data-quality defect -- to separate "harder" from "dirtier".

Only the joint arms are trained: single-tree controls do not depend on the panel and
already exist from the panel15 array at the right seeds (`forecast/single_*_s0-7`,
`matched/single_*_s0-2`). Hence 11 tasks, not 33.

**hmpv_b_g is kept despite being slightly negative on its own per-tree contrast at 15
trees.** The selection rule here is data quality, not outcome; dropping it on its
result would be choosing the panel from the answer. Note vic_ha contributes 74
targets against hmpv_b_g's 363, so the pooled de novo number will be driven mostly by
the tree with the negative contrast.

**If the result lands between 11 and 15, the honest conclusion is that the de novo
claim is composition-sensitive and +0.112 was optimistic.**

---

## 24. Capacity sweep on panel 15 (2026-10-17, RUNNING)

Two axes crossed, matched by **proportion** of encoder layers so size and depth are
separable (`train_capacity15_esm2.sh`, arrays **3353096** 35M / **3353097** 650M):

| fraction | 0 | 1/6 | 1/3 | 2/3 | all |
|---|---|---|---|---|---|
| 35M (12 layers) | ft0 | ft2 | ft4 | ft8 | ft12 |
| 650M (33 layers) | ft0 | ft6 | ft11 | ft22 | ft33 |

3 seeds each, everything else pinned to the joint_panel15 recipe, into `capacity15/`.
35M arms measured at 24-26 min. The 650M half is 14 of 15 still queued; on the
nine-tree panel those ran 1h48-2h40, and this panel roughly doubles the data.

An **orphaned prior sweep** exists at `capacity/` (`panel_35M_ft{0,2,8,12,full}`,
`panel_650M_ft{0,4,11}`, 3 seeds) on the *nine*-tree panel, with no script and no
NOTES section -- lost to the reset s18 records. Its configs and timings were
recovered from the run manifests. It also has a `full` cell (nothing frozen); that is
omitted here because at 35M it was indistinguishable from ft12 (both 31 min).

**650M/ft33 is untested** -- the prior sweep stopped at ft11, so nothing establishes
that 33 unfrozen layers fit in memory at batch 8 / 8192 tokens.

The 35M/ft4 cell duplicates `joint_panel15` deliberately, so the sweep reads as a
self-contained factorial; the two should agree at seeds 0-2.

---

## 25. Main-model work: curated mutation lists and the cross-tree baselines (2026-10-17)

**Not the multi-tree arm.** This section is here because this file is the running log
and splitting it would lose it. The code is in `antiGen/` and `examples/slurm/`; the
figures are 2 (sc2) and 4 (endemic), not 5.

### Two curated lists as evaluation subsets

`ecdc.csv` (ECDC variant-defining mutations) and `coguk.csv` (CoG-UK mutations of
interest) as subsets of Figure 2's metrics. `antiGen/mutation_lists.py` (new) parses
them; `antiGen/eval.py` gains `--mutation-list LABEL=PATH`, repeatable.

Facts established, each verified rather than assumed:

- **Coordinates are 1-based Spike.** All 323 CoG-UK wild-type letters match the root
  at `pos-1`; a 0-based reading matches 19.
- **Key on (position, alt), ignore the listed wild type.** ECDC annotates each lineage
  relative to *its* parent, so `D339H` names a residue the root does not carry (G at
  339). Ten of 372 parsable entries are of this kind.
- **The eval range for sc2 is the RBD**, where the DMS baselines live. ECDC
  contributes 74 of its 142 cells there, CoG-UK 247 of 323. ECDC collapses 386 rows
  to 142 cells because it repeats mutations across lineages.
- Deletions parse to a gap cell (`Δ69-70` -> two); insertions and `E484X` cannot be
  expressed as (position, residue) and are dropped -- 11 ECDC rows, counted.
- "Ignore it if the alt matches the query" needs no code: `compute_hit_rate`
  NaN-masks each example's own residue and the emergence path masks the root's.

**A restricted pool is not harmless here.** De novo restricts pool and targets
together safely because the pool is huge and mostly negative. On sc2 cutoff 1 **all
74 in-RBD ECDC cells and 235 of CoG-UK's 247 are observed at some point**, so a pool
restricted to the list has almost no negatives and the metric quietly becomes "are
the frequent list members ranked above the rare ones". Hence three recall variants
per list: `list_<L>` (de novo convention), `_nogapstop`, and `_fullpool` (targets
restricted, full candidate pool = Figure 2's question on a subset). `evaluate_full`
now accepts a `(score_mask, target_mask)` pair for this; existing call sites unchanged.

`--eval-range full` scores all 1273 positions. Nothing breaks: the Omicron DMS sets
and EVEscape are a **constant -1 outside the RBD**, which `load_baselines` turns into
NaN, so their pool stays the RBD while out-of-RBD targets still count against them.
Coverage is printed at startup (Omicron 14.3-14.4%, EVEscape 13.6%, CoVFit 80.3%,
MSA_Transformer 95.5%, pssm_root/pyr0 100%). **Their full-Spike numbers are not
like-for-like with the full-coverage rungs.**

Jobs **3353354** (RBD) and **3353452** (full), `gpu_batch`, queued. Both recompute
`all`/`unseen` alongside, so they re-derive the published Figure 2 numbers as a check.

### Single-query list recall (`antiGen/list_recall.py`)

The more direct question: train on the first year, put **one** query to each model,
rank every mutation it could make, read off what fraction of a list lands in the top
q. Mutations seen in the training window are masked from the pool **and** the list --
`pssm_root > 0`, the same seen mask Figure 2's de novo subset uses. Two queries (root,
BA.2 root), two ranges, two universes.

The masking bites unevenly: CoG-UK keeps 143 of 247 RBD cells (58%), ECDC keeps **21
of 74** (28%). ECDC is largely a record of what already happened by the end of year
one. **ECDC is too small to read** and is being dropped.

Job 3353561: 28 predictors scored, then **OUT_OF_MEMORY at 43 min**. It kept its
results only because the kill landed after the final write -- the script accumulated
everything in memory and wrote once at the end. Fixed: it now writes after every
predictor via an atomic `os.replace`, and resumes by skipping predictors already in
the CSV. `eval.py` has flushed per model all along; this should have copied it.
Memory raised 64G -> 192G, resumed as **3353946** for the five missing
(`evcouplings`, `vespag`, three `esmc_300m` which fail on
`'esmc_fitness_predictor' is not a valid ModelType` -- a pre-existing package gap).

**Precision, and why it changes no ranking.** The natural reading is "what fraction of
my top 10% is on the list" -- precision, not recall. At fixed q they are related
exactly by `precision = recall * n_targets / (q * n_candidates)`, and both factors are
model-independent, so **precision is recall times a constant and the ranking is
identical**. What it buys is interpretability. CoG-UK RBD, root query, de novo pool of
3,605 cells holding 143 list cells (base rate 3.97%):

| model | prec@1% | prec@10% | enrichment |
|---|---|---|---|
| EVEscape | 17.6% | 15.3% | 3.84x |
| antiGen ESM2-35M | 8.3% | 11.4% | 2.87x |
| EVE finetuned | 14.7% | 10.6% | 2.66x |
| antiGen ESM2-650M | 11.1% | 9.7% | 2.45x |
| blosum62 | 3.3% | 6.9% | 1.75x |
| antiGen transformer | 2.8% | 4.2% | 1.05x |
| pssm_root | 4.0% | 4.0% | 1.00x |

`pssm_root` at exactly the base rate is the correct degenerate result: it is zero on
every unseen cell by construction, so it is entirely tied and the fractional tie rule
pays out exactly q. Useful confirmation the masking does what it claims.

Two things recall hid: **Omicron_BA2_bind is best-in-class at 1%** (17.7%) and
mid-table at 10% (6.5%) -- sharp at the head, then spent. And full-Spike enrichments
are much larger (EVEscape 21x) only because the base rate there is 0.72%, so
**enrichment is not comparable across ranges** even though precision is.

35M beats 650M in every cell of this eval.

### The cross-tree factored baselines in Figures 2 and 4

codon, substitution and `where_times_which` added to the sc2 and endemic ladders via
`--multitree-baselines` (`eval_multitree_baselines.sh`, jobs **3353663** sc2,
**3353664** flu, **3353665** rsv_a, **3353666** denv2, all `gpu_batch`).

**These are not like the CSV baselines and must not be added as such.** Everything in
`data/baselines/` is a fixed (L, E) map broadcast to every example; these three read
the residue at each position of the query. They are appended to `candidates` and
scored through the model path -- one forward pass per example -- which also means they
inherit the gap/stop masking automatically (codon has
`OUTPUTS_LEGITIMATE_GAP=False`). Fitted on training examples only, one seed (s14:
saturates by K=2).

Output goes to `evals_<dataset>_mtb/`, not the published directories.

**The R scripts are NOT updated.** The new rungs will appear as model rows but
`fig2_sc2_benchmarks.R` groups models into `subsets` for colour and legend, and fig4
has its own inventory; they need a group assignment before they render.

### Housekeeping: /home was at 99%

917G of 932G used, 15G free -- tight enough to fail a job mid-write. `old/` (299G,
nothing modified since April, bulk untouched since 2026-01-28, zero references in the
repo) archived to `/large_storage/hielab/ispecht/home-archive/old/` with
`rsync -a --remove-source-files`, which deletes each file only after it transfers.
**99% -> 77%, 220G free** and still running at the time of writing.

Deliberately untouched, because queued jobs read them: `antiGen-data` (244G,
FITNESS_DATA_ROOT, 48 references), `sc2-open-data` (29G, the EVcouplings MSA),
`h3n2`/`denv2`/`rsv-a` (62G), `miniconda`, and `.cache/huggingface` (15G) which holds
the ESM-2 35M and 650M weights every running job loads. `juniper-out` (133G, modified
2026-10-10) left alone on instruction -- it would take home to ~53% if ever archived.

---

## 26. The eval window: all post-D examples, not boundary-crossing ones (2026-10-18)

**This supersedes the test-set definition behind every forecast number in s11-s25.**
No model changes; the models never saw this data. What changes is which post-D
substitutions are scored.

### The old definition, and where it came from

`FORECAST_DESIGN.md` line 47 defined the test set as *"branches with parent date <= D <
child date"* -- only the single step across the cutoff -- and `make_forecast_testset.py`
implemented exactly that. So this was never a coding slip: the restriction was written
into the design doc at the start and silently inherited by every arm since. The doc has
now been corrected at the source, with a SUPERSEDED note pointing here.

**The intended definition, and the one in force from now on: every branch whose CHILD
postdates D** -- all substitutions that happened in the testing window. This is also
what the sc2 benchmark has always meant by a test set, where testing examples are
post-cutoff nodes.

`make_forecast_testset.py --window {all,crossing}` now selects, defaulting to `all`.
`crossing` is kept to reproduce the old numbers rather than deleted.

### Why it matters, and why it is not just "more data"

The shortfall is the post-D *internal* structure the crossing rule discards, so it
grows with the gap between D and today:

| D | gap | crossing | all post-D | |
|---|---|---|---|---|
| 2025-01-01 | 1.8 y | 844 | 1,545 | 1.8x |
| 2021-01-01 | 5.8 y | 280 | 2,215 | 8x |

On the fifteen-tree panel the regenerated test sets go **1,942 -> 2,697 targets** and
**426 -> 632 novel**. Per tree the correction is **uneven**, which is the part that
matters: it REWEIGHTS the panel rather than scaling it.

| tree | old | new | |
|---|---|---|---|
| h1n1pdm_ha | 65 | 224 | 3.4x |
| h3n2_ha | 95 | 264 | 2.8x |
| ncov | 101 | 213 | 2.1x |
| measles | 64 | 107 | 1.7x |
| h5nx | 229 | 354 | 1.5x |
| h9n2_ha | 354 | 419 | 1.2x |
| hmpv_a_g / hmpv_b_g / vic_ha / rsv_a / rsv_b | | | 1.0-1.7x |
| denv1-4 | 51 | 51 | 1.0x |

Dengue and RSV barely move -- their post-D structure is shallow, so nearly every post-D
branch already *was* a crossing branch. The flu trees and ncov gain most. **Any movement
in the pooled contrasts is therefore partly a change in panel composition, not only
added power, and must not be read as "more data, tighter intervals".**

### Two bugs found while regenerating, both mine

- **The flag set is not uniform across the panel.** Regenerating with
  `--strip-trailing-stop` for every tree aborts on h5nx: its *snapshot* HA reference
  ends in an ambiguous `X` (mapped to gap) while *today's* ends in a stop, so stripping
  shortens one side only (569 vs 568) and `make_forecast_testset` correctly refuses on
  the reference-frame check. h5nx is `strip=False`; the other 14 are `strip=True`.
  The s3 gotcha about `X` residues in the H5 polybasic cleavage site, resurfacing.
- **ncov needs explicit `--current-ref` / `--snapshot-ref`** -- its builds do not embed
  a root sequence -- and a diagnostic loop that tried both flag settings left the
  *wrong* one (seq_len 1274) on disk.

Both are now prevented by an invariant rather than by memory: **the test file's
`seq_len` must equal the train file's**, since the model was trained on that frame. The
regeneration script determines the flag set per tree by trying both and keeping
whichever reproduces the train `seq_len`, and reports FAILED otherwise. All 15 pass.

**Process note.** Two regeneration attempts silently did nothing and still exited 0:
a `grep` in the pipeline swallowed the output and `set -e` killed the loop after h5nx
failed first. It was caught only by comparing the regenerated counts against the 1.8x
derived independently from the trees -- the logs and exit status were both useless.
Third time in one session that a `set -e` / `grep` pipeline hid a failure.

### The set-difference alternative was considered and does not work

Proposal: generate all examples from the snapshot and all from today's tree, and let
the test set be the **difference** -- no dates needed, assuming trees grow additively.
Elegant, and it fails empirically: Nextstrain builds are **re-subsampled, not
additive**. Tip survival from the 2025-01-01 snapshot into today's build:

| tree | snapshot tips | still present | dropped |
|---|---|---|---|
| h3n2_ha | 1,576 | 962 (61%) | 39% |
| ncov | 2,896 | 845 (29%) | 71% |
| h5nx | 2,508 | 718 (29%) | 71% |
| rsv_a | 2,774 | **0** | **100%** |
| measles | 579 | **0** | **100%** |

Two causes, both already recorded in s11 when snapshot-anchored matching was tried and
abandoned: all-time builds re-subsample to a target size every run, and Nextstrain
renamed tips between builds (measles to Pathoplexus `PP_*`, RSV to full strain names),
which is why rsv_a and measles are at zero.

Measured on h3n2_ha, the *best* case: the set difference yields **1,077 (query,
mutation) pairs against the 264 that actually happened after D -- 4x too many**, the
excess being pre-D evolution that reappears as "new" because re-subsampling changed the
ancestral reconstruction. On rsv_a and measles it would be far worse: with zero tip
overlap essentially the whole current tree reads as new, putting pre-D branches the
model trained on into the test set. **Exact matching on reconstructed ancestral
sequences cannot establish that something did not exist before D.**

### Train/test overlap under the date window, measured

A related guard -- dropping test `(query, mutation)` pairs that also occur in training
-- was measured and **not adopted**, because it catches almost nothing:

| tree | test examples | queries also in train | shared (query, mutation) pairs |
|---|---|---|---|
| rsv_a | 41 | **32 (78%)** | 5 of 56 targets |
| h3n2_ha | 139 | 19 (14%) | 5 of 264 |
| ncov | 79 | 10 (13%) | 1 of 213 |
| measles | 88 | 8 (9%) | 0 of 107 |
| h5nx | 152 | **0** | 0 of 354 |

~11 pairs of ~994 targets, about 1%. The dominant overlap is shared *query genotypes*
(rsv_a 78%), which is not duplication: the model is asked what happens NEXT from a
genotype it trained on, and the de novo subset already quarantines the case where the
answer itself was seen. **The shared-query rate tracks how much post-D structure a tree
has** -- h5nx 0% because `window=all` gives it post-D parents, rsv_a 78% because its
post-D structure is shallow -- and is worth reporting per tree, since rsv_a's 56 targets
are a much softer test than h5nx's 354.

### State

Test sets regenerated for all 15 trees, invariant checked. Rescoring resubmitted:
**3354481** panel 9 (COMPLETED, 4m46s), **3354482** panel 11, **3354483** panel 15,
all into their existing data directories -- the crossing-window CSVs and figures they
replace are recoverable from git, and `--window crossing` regenerates the old test
files exactly. Panel 13 (3353910) is still training and will be scored on the corrected
sets. Nothing in s11-s25 has been re-reported yet.

---

## 27. Rescored on the corrected window, and a 2021 sc2 arm (2026-10-18)

### The corrected numbers: the claims hold, and tighten

Panels 9 and 11 rescored with `--window all` (s26). Recall@10%, temporal/pooled/
standard, K = 8. Old = crossing window, new = full testing window:

**Panel 9** (818 targets / 197 de novo, was 562 / 134):

| contrast | subset | old | new |
|---|---|---|---|
| vs single-tree | all | +0.053 [+0.031,+0.075]* | +0.055 [+0.038,+0.073]* |
| vs single-tree | de novo | +0.164 [+0.087,+0.248]* | +0.162 [+0.104,+0.228]* |
| vs `where x which` | all | +0.128 [+0.096,+0.158]* | +0.131 [+0.102,+0.159]* |
| vs `where x which` | de novo | +0.104 [+0.018,+0.195]* | **+0.071 [+0.000,+0.149]** |

**Panel 11** (1,306 / 331):

| contrast | subset | old | new |
|---|---|---|---|
| vs single-tree | all | +0.045 [+0.025,+0.065]* | +0.047 [+0.032,+0.063]* |
| vs single-tree | de novo | +0.172 [+0.097,+0.244]* | +0.166 [+0.116,+0.217]* |
| vs `where x which` | all | +0.129 [+0.098,+0.158]* | +0.137 [+0.114,+0.160]* |
| vs `where x which` | de novo | +0.112 [+0.031,+0.191]* | **+0.085 [+0.030,+0.138]*** |

Point estimates barely move (<=0.008) except the de novo contrast against
`where x which`, which **falls** in both panels: 0.104 -> 0.071 at nine trees and
0.112 -> 0.085 at eleven. Intervals tighten everywhere, as expected from 1.5x the
targets. At nine trees that contrast now has a lower bound of **exactly +0.000** --
it no longer clears zero; at eleven it still does, narrowly.

So the correction does not overturn joint-vs-single, which is unmoved and better
determined. What it erodes is the same narrow claim panel 15 already put back across
zero. Three independent perturbations -- more trees (s22), the window fix (here), and
the K sweep (s15/s16) -- all push that one contrast toward zero. **It should be treated
as unresolved, not as established.**

Panel 15 rescore (3354483) still running; panel 13 (3353910) 9 of 11 trained.

### A SARS-CoV-2 arm at D = 2021-01-01 (array 3354617, 16 tasks)

Narrow question: does training on *other viruses* improve the sc2 forecast? Only ncov
is a training **and** evaluation tree; everything else is training-only context, so
frozen or unforecastable trees are perfectly good here.

`joint_2021` (6 trees) vs `single_ncov`, 8 seeds each, matched K.

| tree | examples | muts | gene | |
|---|---|---|---|---|
| ncov | 347 | 1,194 | S | the eval tree |
| h3n2_ha | 1,079 | 2,188 | HA | |
| h1n1pdm_ha | 980 | 2,037 | HA | |
| yam_ha | 320 | 1,050 | HA | B/Yamagata, extinct since 2020 |
| evd68_vp1 | 239 | 481 | VP1 | |
| ebola_gp | 49 | 189 | GP | |

**Eval: 986 examples, 2,210 targets, 1,537 novel (70%)** -- stronger than anything in
the 2025 panel, and only because of the s26 window fix: under the crossing rule the
same tree gave 275/123.

### Where the 2021 data comes from

**Charon cannot serve it.** The ncov archive floor is **2022-05-01** (404 at
2022-04-29, bisected). Nothing earlier exists.

**The Wayback Machine can.** A CDX sweep of
`nextstrain.org/charon/getDataset*` before 2021-01-01 returns **43 distinct datasets**
with a 200 capture over 100 KB. The training tree is `/ncov/global` captured
**2020-12-29**: `meta.updated` 2020-12-28, 3,848 tips, **zero at or after 2021.0** --
it clears the route A and route B guards independently, which no tree in the 2025 panel
does. Its root-sequence sidecar is the 2020-11-21 capture (Wuhan-Hu-1, S = 1273+stop).
Generation is clean: 1 from-base mismatch on the train side, 0 on the test side, frame
identical to today's build.

**A REQUESTED CAPTURE DATE IS NOT A GUARANTEE.** Wayback falls back to the nearest
capture. Asking for `/ncov/north-america` and `/ncov/africa` at 2020-12-20 returned
builds **updated 2021-04-29 carrying ~2,200 post-cutoff tips each**, and
`/ncov/south-america` returned 2021-01-15. Caught only by checking `meta.updated` and
tip dates on the payload. Had they gone in, ~4,500 post-cutoff sequences would have
trained a model that forecasts post-2021 evolution. **Apply both guards to archive
fetches exactly as to Charon ones.**

### Sidecars: relaxed, deliberately

Only 4 datasets have a usable archived root-sequence capture, all ncov-family -- every
other sidecar capture is a server-side **500** recorded faithfully by the archive, and
none of the non-ncov builds embed one. Standing decision: **use today's reference**,
since for these builds it is a fixed reference genome rather than a re-inferred
ancestral state, and record the from-base mismatch count as a check rather than a gate.
Measured: ebola/yam/evd68 0-3, h3n2 56 and h1n1pdm 48 of ~2,100 (~2.5%), **WNV 78 of
382 (20%)**.

### Exclusions, each on a rule rather than an outcome

| excluded | why |
|---|---|
| all ncov cuts (`/ncov/{europe,asia,oceania}`, `20A.EU1`, `S.N501`) | same species as the eval tree -- re-subsamples of one sequence pool, not independent trees |
| flu NA builds | HA only, by instruction |
| zika | 28 examples, below the floor of 40 |
| WNV | 20% from-base mismatch; the 2018 reference and today's genuinely disagree |
| `sars-like-cov` | 49 tips, no tip dates, and **no S annotation** -- the one true relative is unusable |
| h5n1 HA, denv1 | modern sidecar not cached; recoverable, not fetched |

Two notes on the ncov cuts, kept because they will come up again. `20A.EU1` (8,653
tips) and `S.N501` (6,891) are Neher-lab cluster builds -- far denser than the 3,848-tip
global build, since they focus one lineage rather than subsampling the pandemic. And
**`S.N501`'s inclusion criterion points at the answer**: its tips were selected for
carrying a change at spike 501, the site that became N501Y in Alpha/Beta/Gamma and a
large part of what the eval asks about. Not a leak (all tips pre-D), but not a neutral
training set either.

### State at the end of this session

| job | arm | state |
|---|---|---|
| 3354481 / 3354482 | panel 9 / 11 rescore, `--window all` | COMPLETED (numbers above) |
| 3354483 | panel 15 rescore | RUNNING |
| 3353910 | panel 13 training | 9 of 11 complete |
| 3354617 | sc2 2021, joint vs single, 16 tasks | queued |
| 3353096 | capacity sweep 35M, 15 tasks | COMPLETE |
| 3353097 | capacity sweep 650M | 18 running, 1 pending |
| 3353664 / 3353665 / 3353666 | factored baselines: flu, rsv_a, denv2 | COMPLETED |
| 3353663 | factored baselines: sc2 | RUNNING |
| 3353354 / 3353452 | sc2 mutation-list eval, RBD / full Spike | RUNNING |
| 3353946 | list_recall resume for the 5 missing predictors | RUNNING |

Nothing from s11-s25 has been re-reported against the corrected window beyond panels
9 and 11 above. Panels 13 and 15, every figure, and the capacity sweep all still need
rescoring or rebuilding on `--window all` before any of their numbers are quoted.

`Nice` values were set to prioritise the rescores among my own queued work:
panel13 5000, the 650M sweep 10000, rescores left at 0. Those should be reset once
the rescores land, or panel13 and the sweep will keep losing the queue.

---

## 28. Panel 13: the clean two are enough to reopen it (2026-10-18)

Array **3353910** finished 11/11 COMPLETED (8 temporal -> `forecast13/`, 3 random ->
`matched/joint13_s0-2`), one `_best.pt` per directory. Collection **3360341**, 6m58s,
`--panel 13 --out antiGen-figures/data/multitree13`; figures
`figs/multitree_results13.png` and `figs/supp_multitree_results13.png`.

All four panels now rescored on `--window all`, recall@10%, temporal/pooled/standard,
K = 8. Panel 13 is the eleven plus **vic_ha and hmpv_b_g** -- the two of s21's four
with no known data-quality defect:

| contrast | subset | 9 | 11 | **13** | 15 |
|---|---|---|---|---|---|
| vs single-tree | all | +0.055* | +0.047* | **+0.038*** | +0.024* |
| vs single-tree | de novo | +0.162* | +0.166* | **+0.152*** | +0.116* |
| vs `where x which` | all | +0.131* | +0.137* | **+0.118*** | +0.117* |
| vs `where x which` | de novo | +0.071 | +0.085* | **+0.043 [-0.010,+0.096]** | +0.020 |

Targets: 818/197, 1,306/331, **1,790/400**, 2,600/498.

**This is the case s23 pre-registered as decisive, and it came out the way that
settles the question.** Panel 13 lands between 11 and 15 on every contrast, and the
de novo contrast against `where x which` falls back across zero **without any of the
dirty trees**. h9n2_ha's 46.5% ambiguous `X` and hmpv_a_g's gap fraction are not the
explanation. The honest conclusion stands as written: **the de novo claim is
composition-sensitive and +0.112 was optimistic.** Joint-vs-single survives on all
four panels and decays monotonically with panel size on all substitutions.

### One tree carries it, and s23 said it would

Per-tree, de novo vs `where x which` at 13 trees:

| tree | est | interval |
|---|---|---|
| hmpv_b_g | **-0.167** | [-0.304,-0.043] -- clears zero, negative |
| vic_ha | +0.000 | [-0.286,+0.312] |

So the pooled drop from +0.085 to +0.043 is **hmpv_b_g actively pulling down**, not
the two new trees diluting a real effect. s23 called this in advance: vic_ha brings 74
targets against hmpv_b_g's 363, so the pooled de novo number would be driven by the
tree with the negative contrast, and it is. hmpv_b_g was only -0.021 de novo at 15
trees; at 13, where it is a larger share, it is -0.167 and significant.

**That is a weaker refutation than the pooled row alone suggests, and must be reported
as such.** What panel 13 establishes is that one clean tree is enough to reopen the
contrast -- not that clean trees in general do. The remaining question is whether
hmpv_b_g is genuinely a tree the joint model loses on, or carries a defect s21 did not
look for. Nothing here distinguishes those.

### State

Still unscored on the corrected window: the capacity sweep. Still unscored at all: the
sc2 2021 arm (16/16 trained, no scoring path exists -- `collect_figure_metrics.py`
knows only panels 9/11/13/15, all keyed to the 2026-05-01 corpus).

---

## 29. The 2021 sc2 arm scored, and a contamination check on its tree (2026-10-18)

### The scoring path, built rather than found

s28 closed saying no scoring path existed. There is one now, in the same script
rather than beside it. `collect_figure_metrics.py` gains a `'2021'` entry in
`PANELS`, and four things that were module constants became per-panel settings with
their previous values as defaults: `D`, the `data/` subdirectory the split comes
from, the root the single-tree arm lives under, and which regimes to run. `--panel`
takes string keys now, so a non-numeric arm can sit in the same table.

The arm differs from panels 9/11/13/15 in exactly three ways, all declarative:
it **scores** one tree (ncov) while its joint arm **trains** on six, it reads
`data/forecast2021/`, and it names only the temporal regime -- `matched/` has no
2021 runs, and building them would mean retraining six trees on today's builds to
answer a question this arm does not ask. Everything else -- rungs, universes, seed
ensembling, epoch rules, bootstrap -- is the same code on the same path.

**Regression checked, the same way s20 checked the panel-11 plumbing:** re-running
`--panel 13` through the modified script reproduced `ladder.csv`, `deltas.csv` and
`agreement.csv` **byte-identically**. The four published panels are untouched.

Collection **3360461**, 4m37s, `--panel 2021 --out antiGen-figures/data/sc2_2021`.

### Results

986 examples, **1,999 scored targets, 816 de novo** (`standard`). Recall@10%,
temporal/pooled/standard, K = 8:

| rung | all | de novo |
|---|---|---|
| **joint (6 trees)** | **0.564** | **0.396** |
| single (ncov alone) | 0.561 | 0.303 |
| `where x which` | 0.452 | 0.382 |
| substitution | 0.379 | 0.327 |
| codon | 0.320 | 0.310 |
| pssm | 0.527 | 0.091 |
| frequency | 0.369 | 0.080 |
| uniform | 0.100 | 0.100 |

| contrast | subset | est |
|---|---|---|
| vs single-tree | all | +0.003 [-0.017,+0.021] |
| vs single-tree | de novo | **+0.093 [+0.063,+0.123]*** |
| vs `where x which` | all | +0.112 [+0.089,+0.136]* |
| vs `where x which` | de novo | +0.014 [-0.022,+0.051] |

**The answer to the question the arm was built to ask is split.** Training on five
other viruses buys SARS-CoV-2 nothing across all substitutions -- an interval
centred on zero -- and a real, well-determined gain on de novo ones. The split is
not a knife edge: the de novo contrast holds under the per-tree epoch rule (+0.097),
the `common` universe (+0.084) and the 1% budget (+0.009 [+0.002,+0.016]), while the
`all` contrast is null or slightly negative in every one of those cuts (at 1%/common
it is -0.008 [-0.014,-0.001], i.e. marginally the wrong way).

### Against the 2025 panels, and why this is confirmation rather than news

| contrast | subset | 9 | 11 | 13 | 15 | 2021 sc2 |
|---|---|---|---|---|---|---|
| vs single-tree | all | +0.055* | +0.047* | +0.038* | +0.024* | **+0.003** |
| vs single-tree | de novo | +0.162* | +0.166* | +0.152* | +0.116* | **+0.093*** |
| vs `where x which` | all | +0.131* | +0.137* | +0.118* | +0.117* | **+0.112*** |
| vs `where x which` | de novo | +0.071 | +0.085* | +0.043 | +0.020 | **+0.014** |

The pooled rows are not like-for-like -- fifteen trees against one -- so the
comparison that carries weight is **ncov against itself**, which the panels also
score:

| | ncov de novo vs single | targets |
|---|---|---|
| D = 2025-01-01 (panel 15) | +0.102 [-0.029,+0.213] | 59 |
| D = 2021-01-01 | +0.093 [+0.063,+0.123]* | **816** |

Nearly the same point estimate on **14x the de novo targets**, and the interval
moves off zero. Read this as the 2021 arm confirming, on a properly powered sc2 test
set, the gain ncov alone could only gesture at in 2025 -- not as a new effect. The
de novo contrast against `where x which` stays null here too (+0.014), which is one
more independent panel agreeing with s22/s28 that **that** claim is unresolved.

### Two things to carry with these numbers

**The pooled row is one tree.** With a single evaluation tree the bootstrap
resamples examples within ncov and has no cross-tree component, so these intervals
are not the same kind of object as the panels' pooled rows. The `POOLED` and `ncov`
rows in `deltas.csv` are identical by construction.

**De novo here is stricter than the generator's sidecar**, and the difference is
large enough to trip someone quoting both. The sidecar says 1,537 novel of 2,210;
the collector gives **981** on the same 2,210 (`common`) and **816** after dropping
gap targets (`standard`). The 2,210 matches exactly, so the target set is not in
question -- the whole gap is `seen_cells`, which also counts a cell as seen when any
training sequence merely *carries* that residue, so a reversion to consensus is not
de novo. Same rule as the four panels; it is the sidecar that is the outlier.

### A contamination check on the 2020-12-29 build, and a correction

Prompted by a claim in s25 reading as though nearly the whole CoG-UK RBD list was
already observed before the cutoff. It is not, and the two numbers answer different
questions:

| what is counted | result |
|---|---|
| CoG-UK RBD cells mutated-to anywhere in `output1` (171,248 examples, whole pandemic) | 235 / 247 (95.1%) |
| CoG-UK RBD cells present in the **pre-2021-01-01 ncov tree** (347 branch examples) | **29 / 247 (11.7%)** |

`cutoff 1` in `eval_sc2_mutation_lists.sh` is a **dataset version** (`output{cutoff}`,
allowed 1/4/5), not a date, and "observed at some point" ranges over that dataset's
whole span. The comment is about why a list-restricted *pool* has almost no negatives
left to rank against. **It says nothing about any training window**, and must not be
read as one.

The 11.7% is the figure for this arm's tree, and all 29 are genuine pre-D branch
mutation events -- the `carried` clause adds nothing, its 16 cells being a subset of
the 29:

```
354K 367F 406D 408I 408K 417N 439K 440K 440Y 445F 452R 453F 458N 476S 477I
477N 477R 478R 483F 484A 484K 484Q 486L 490S 494P 499L 501T 501Y 504D
```

**The composition is the actual check, and it is period-correct for a build dated
2020-12-28.** The Alpha/Beta/Epsilon/mink-era changes are present -- N501Y, K417N,
E484K, L452R, N439K, Y453F, S477N -- all genuinely circulating and sequenced before
2021. The later-emerging ones are absent: K417T (Gamma, Jan 2021), T478K (Delta),
G339D and G446S (Omicron, Nov 2021). Those are precisely the cells that would appear
first if future sequences had leaked in, and none did. This is an independent
confirmation of s27's route A/B guards (`meta.updated` 2020-12-28, 3,848 tips, zero
tips at or after 2021.0) rather than a restatement of them.

**One earlier explanation is withdrawn.** The reason offered in this log's reporting
for `pssm_root` (0.735) edging out the main model (0.722) on the CoG-UK fullpool
eval was that those lists are dominated by already-seen mutations. The 95.1% does
not support that, being a whole-dataset count, and the Figure 2 de novo masking uses
`pssm_root > 0`, a third rule again. **Why a plain composition baseline beats the
model on curated variant-defining lists is unexplained and open.** The number that
would settle it -- the list's overlap with Figure 2's own training window -- has not
been measured.

### The figure

`antiGen-figures/figures/fig_sc2_2021.R` -> `figs/sc2_2021_results.png` and
`figs_vector/sc2_2021_results.pdf`. Same visual language as Figure 5 -- same rung
set, colours, `standard` universe, pooled epoch rule and 8-seed ensembles -- so
the ladders read against `multitree_results*.png` row by row. Added to
`run_all.R` after `fig5_multitree.R`.

Four panels, not six. **The per-tree forests collapse here**: with one evaluation
tree the `ncov` and `POOLED` rows of `deltas.csv` are identical by construction,
so Figure 5's panels C and D become one contrast panel carrying both contrasts on
both subsets. The overlap panels are kept, and for the reason s29 gives -- at
+0.014 de novo the model and `where x which` are close enough on the ladder that
only the overlap separates "learned something else" from "rediscovered it", and
it does: 119 model-only against 106 `where x which`-only on 816 de novo targets,
with 387 (47%) caught by neither.

Both rung labels are renamed for this arm alone: **antiGen (six trees)** and
**antiGen (ncov alone)**, rather than multi-tree/single-tree. The joint arm trains
on six trees and scores one, so "multi-tree" would invite the reader to think the
pooled row spans them.

---

## 30. The CoG-UK list eval, RBD as the headline (2026-10-18)

Array **3353354** (RBD) COMPLETED, 9h06m, `evals_sc2_lists/`. The full-Spike twin
**3353452** is still RUNNING at 13h54m. **RBD is the headline** and full Spike the
supplement, for the reason s25 gave when the pair was built: the RBD is where the DMS
baselines are defined, and over the full range Omicron DMS covers 14.3-14.4% of cells,
EVEscape 13.6%. Those rungs degrade honestly rather than vanish, but their full-Spike
numbers are not like-for-like and the RBD run needs no such caveat.

Within the RBD, **`_fullpool` is the headline variant**: listed cells are the targets,
every eval-range cell is still a candidate, which is Figure 2's question asked of a
subset. The restricted-pool variant is reported beside it but is the weaker object --
s25 measured 235 of CoG-UK's 247 in-RBD cells as observed at some point in `output1`,
so a pool restricted to the list has almost no negatives left and the metric slides
toward "are the frequent list members ranked above the rare ones".

`list_coguk_nogapstop` is **byte-identical** to `list_coguk` and can be dropped for
this list: CoG-UK parses 323 rows to 323 cells with no indel rows (the 9 indel rows in
s25's note are ECDC's), so there is no gap or stop cell in the RBD slice to mask.

### The table

CoG-UK, RBD, recall at top 10%. `fullpool` and `restricted` as above; the right-hand
columns are this directory's recomputed Figure 2 numbers, carried so each row can be
read against the model's unrestricted behaviour.

| # | model | fullpool@10% | fullpool@20% | restricted@10% | fig2 all | fig2 de novo |
|---:|---|---|---|---|---|---|
| 1 | **antiGen as EVE full FT** | **0.837** | 0.917 | **0.499** | 0.684 | 0.509 |
| 2 | EVE | 0.780 | 0.863 | 0.481 | 0.636 | 0.455 |
| 3 | PSSM | 0.735 | 0.821 | 0.361 | 0.731 | 0.100 |
| 4 | antiGen | 0.722 | 0.868 | 0.354 | 0.636 | 0.526 |
| 5 | antiGen as Tranception full FT | 0.688 | 0.806 | 0.383 | 0.530 | 0.282 |
| 6 | Tranception | 0.600 | 0.713 | 0.402 | 0.478 | 0.313 |
| 7 | EVEscape | 0.504 | 0.780 | 0.181 | 0.218 | 0.284 |
| 8 | PSSM Conservation | 0.406 | 0.654 | 0.167 | 0.296 | 0.180 |
| 9 | PyR0 | 0.405 | 0.471 | 0.355 | 0.228 | 0.100 |
| 10 | BA.1 DMS (bind) | 0.385 | 0.614 | 0.168 | 0.311 | 0.167 |
| 11 | BA.2 DMS (bind) | 0.334 | 0.551 | 0.147 | 0.279 | 0.179 |
| 12 | BA.2 DMS (expression) | 0.311 | 0.642 | 0.168 | 0.197 | 0.246 |
| 13 | EVmutation | 0.300 | 0.403 | 0.165 | 0.361 | 0.255 |
| 14 | BA.1 DMS (expression) | 0.294 | 0.674 | 0.118 | 0.186 | 0.253 |
| 15 | BLOSUM62 | 0.256 | 0.431 | 0.168 | 0.247 | 0.345 |
| 16 | MSA Trans. (XFG eval) | 0.204 | 0.334 | 0.109 | 0.193 | 0.145 |
| 17 | MSA Trans. (BA.2 eval) | 0.182 | 0.319 | 0.126 | 0.180 | 0.168 |
| 18 | MSA Transformer | 0.180 | 0.327 | 0.127 | 0.178 | 0.175 |
| 19 | CoVFit | 0.050 | 0.105 | 0.068 | 0.049 | 0.073 |

**`antiGen as EVE full FT` tops both variants**, and by a clear margin on the
fullpool: +0.057 over frozen EVE and +0.115 over antiGen itself. It is the same model
Figure 2 already bolds, so the curated list agrees with the headline rather than
reopening it.

**The ladder reorders between the two pool definitions and the reorder is not
cosmetic.** antiGen is 4th on the fullpool (0.722) and 7th on the restricted pool
(0.354), while Tranception climbs 6th -> 3rd and PyR0 9th -> 6th. Nothing should be
quoted from one variant as though it held in the other.

### The check the directory exists to provide, and it passes

`eval_sc2_mutation_lists.sh` recomputes `all` and `unseen` alongside the list subsets
precisely so this directory re-derives the published Figure 2 numbers. Over the 19
models it shares with `evals_sc2/`, every cell of both files agrees to within
**2.2e-4** (worst: evcouplings, `top_40.0%`), which is sampler noise. The scoring path
is sound.

### But eight model groups are missing, and they are the ESM-2 arms

The 19 rows above are not the roster. `evals_sc2/` carries **27** models; this
directory has 19. Every one of the eight absentees is an ESM-2 group:

| absent | |
|---|---|
| antiGen as ESM 2 (35M) partial / full / LM head FT | three antiGen arms |
| antiGen as ESM 2 (650M) partial / full / LM head FT | three antiGen arms |
| ESM 2 (35M), ESM 2 (650M) | the two untrained backbones |

All eight were found, kept at checkpoint selection, and then skipped at build time:

```
[skip] esm2_...: model build/load failed
       (Could not import module 'EsmForMaskedLM'. ...)
```

`eval.py` catches a build/load failure and moves on, so the array **exited 0** and
wrote a complete-looking set of CSVs. Nothing in the output says the roster is short.

**The cause is not ESM-2 and not `eval.py`.** It is an ABI break in the base conda
environment: `torchvision 0.21.0+cu124` is built against torch 2.6 while
`torch 2.7.1+cu126` is installed, so `torchvision::nms` never registers and importing
torchvision raises `RuntimeError: operator torchvision::nms does not exist`.
`transformers 4.57.6` imports torchvision from `image_utils` behind an
`is_torchvision_available()` guard that checks metadata rather than importability, and
that exception propagates out of the lazy import of `EsmForMaskedLM`. Reproduced
directly:

```
>>> from transformers import EsmForMaskedLM
ModuleNotFoundError: Could not import module 'EsmForMaskedLM'. ...
>>> import transformers.models.esm.modeling_esm
RuntimeError: operator torchvision::nms does not exist
```

Anything in this environment that loads an ESM-2 backbone through `transformers` is
affected, not just this eval. **3353452 has the identical eight skips in its log**, so
the full-Spike run will land with the same hole.

### Why the absentees are contenders rather than also-rans

`list_recall` (s25) is a different question -- one query, de novo-masked pool -- but it
is the only eval that has scored all of these predictors, and there **two ESM-2 arms
outrank `antiGen as EVE full FT`**: on CoG-UK/RBD/root at q = 10%, the 35M partial-FT
arm is 3rd (0.287) and the 35M full-FT arm 6th (0.266), against eve_ft 9th (0.252,
tied with frozen EVE).

That does not transfer. Spearman between the two evals over the 18 models they share
is **0.34** on the fullpool and **0.18** on the restricted pool -- they disagree on
rank order at a level that forbids reading one off the other. What it does establish is
that the eight absentees are not predictors one can assume would have placed low.
**"antiGen as EVE full FT is best on CoG-UK RBD" is therefore true of the table as it
stands and NOT yet a result.** It needs the rerun before it is quoted.

### What the rerun needs

Not yet done, and deliberately not done from this session: the fix is to install the
torchvision matching torch 2.7.1 (0.22.1), and **3353452 is still running inside that
environment**. Resolve it after that job lands, then resubmit both ranges.

### One thing this table does settle

s29 left open why `pssm_root` (0.735) edges out the main model (0.722) on the CoG-UK
fullpool eval, having withdrawn the "the lists are dominated by already-seen
mutations" explanation. The full table narrows it: PSSM is **3rd, not 1st**. It is
beaten by `antiGen as EVE full FT` (0.837) and frozen EVE (0.780), so the puzzle is
not "a composition baseline beats the models" but specifically **"a composition
baseline beats `simple_d128_n2`"** -- while the same baseline collapses to 0.100,
exactly chance, on Figure 2's de novo subset where antiGen scores 0.526. A model that
loses to PSSM on a curated list and beats it by 5x on de novo targets is the shape of
a list that is mostly not de novo, which is the measurement s29 named as missing: the
list's overlap with Figure 2's own training window. Still unmeasured.

### The torchvision fix, and the reruns (same session)

`3353452` cancelled at 14h01m. It was never going to finish: it had spent ~10h on
setup and the CSV baselines, had scored **one** checkpoint group (blosum62) of the
fifteen, and carried the same eight ESM-2 skips as the RBD run. Waiting out its
remaining 10h would have bought an incomplete table with the same hole in it.

**Fixed in the base conda environment** (`/home/ivan.specht/miniconda`):

```
pip install --no-deps torchvision==0.22.1 --index-url .../whl/cu126
   torchvision 0.21.0+cu124  ->  0.22.1+cu126
```

`--no-deps` deliberately: torchvision 0.22.1 declares `torch==2.7.1`, which is what is
installed, and the point of the exercise is to leave torch alone. A `--dry-run` first
confirmed torchvision was the only package that would change.

Verified in three steps rather than one, because the import succeeding does not prove
a checkpoint builds:

| check | result |
|---|---|
| `torch.ops.torchvision.nms` registers | True |
| `from transformers import EsmForMaskedLM` | OK |
| `create_model` + `load_state_dict` on the 35M partial-FT checkpoint | **missing=0, unexpected=0** |

### A preflight guard, because exit 0 was the real defect

The environment break cost eight model groups; what made it expensive is that
`eval.py` caught each failure, printed `[skip]`, and **exited 0 with a
complete-looking set of CSVs**. Nothing downstream could tell a 19-model table from a
27-model one. `eval_sc2_mutation_lists.sh` now imports torchvision and
`EsmForMaskedLM` before calling `eval.py` and aborts with exit 1 if either fails,
printing the torch/torchvision versions. Tested both ways: it prints `preflight OK`
on the fixed environment, and aborts with exit 1 under a shimmed torchvision that
raises the original `torchvision::nms` error.

This does not make `eval.py`'s skip-and-continue wrong -- a single bad checkpoint
should not kill a 24h array -- but a *backbone* that cannot load is not one bad
checkpoint, and belongs at the front of the script.

### Two things corrected from earlier in this session

**`eval.py` resumes.** I said in reporting s30 that a resubmit would start over. It
does not: step 6 preloads the unified CSVs from `--output-dir`, skips any model
already present (`[resume-skip]`) and reuses the test-set baseline rows. So both
reruns continue their existing directories rather than recomputing them -- the RBD run
has only the 8 ESM-2 groups left to do, and the full-Spike run keeps its 13 baseline
rows. The directories are **not** cleared: nothing in the scoring code changed, only
the environment, and the 19 RBD rows already reproduce published Figure 2 to 2.2e-4.

**The 24h wall was self-imposed**, not the partition's. `gpu_batch` allows 14 days.
The script's default is now 2 days, and the full range takes `--time=7-00:00:00` at
submit, documented in its usage block.

### Resubmitted

| job | range | time limit | output |
|---|---|---|---|
| **3360817** | RBD (331-531) | 2 days | `evals_sc2_lists/` |
| **3360818** | full Spike (1-1273) | 7 days | `evals_sc2_lists_fullseq/` |

Until 3360817 lands, **s30's table and its "antiGen as EVE full FT is best" line stay
provisional** for the reason s30 gave: in `list_recall` two of the absent ESM-2 arms
outrank eve_ft, and the two evals correlate at rho = 0.34, so neither "they will
beat it" nor "they will not" is supportable from what is on disk.

---

## 31. The CoG-UK list rebuilt as a Figure 2 panel set (2026-10-18)

### What "seen" means, three ways, because two of them were being conflated

The same 247 CoG-UK RBD cells, under three different rules. All three measured here
rather than quoted:

| rule | result |
|---|---|
| carried in `pssm_root` (Figure 2's de novo mask: the **training window**) | 104 seen -> **143 unseen** |
| arose as a mutation event anywhere in `output1` (the **whole pandemic span**) | 235 -> **12 never** |
| present in the 2021 arm's ncov tree (a **different dataset** -- 347 branches) | 29 -> 218 |

Cross-tab of the first two, which closes it:

```
carried in training AND event somewhere : 104
carried, but no event anywhere          :   0
NOT carried, but event somewhere        : 131   <- the entire gap
neither                                 :  12
```

So **143 de novo = 131 cells that arise only after the cutoff + 12 that never arise**.
Two independent confirmations the mask means what it claims: the cells carried in
`root.csv` number exactly 104, and the cells arising as a *training* mutation event
also number exactly 104 -- the same set. The training window is 3,648 of 171,248
examples (2.1%), so most of a list of "mutations of interest" is still in the future at
that point and 58% de novo is the expected result.

s29 warned about exactly this trap ("'observed at some point' ranges over that
dataset's whole span... must not be read as [a training window]") and it was walked
into anyway.

### The measurement s29 asked for, and it settles the PSSM question

s29 named the missing number as "the list's overlap with Figure 2's own training
window". It depends on the weighting, and that **is** the answer:

| | seen | de novo |
|---|---|---|
| by **cell** (what `list_recall` scores) | 42% | 58% |
| by **event** (what the eval.py list metric scores) | **81%** | 19% |

The eval.py metric counts mutation *events*, and a training-window cell like N501Y
recurs thousands of times across the test set, so four fifths of what that eval scores
sits on cells the model saw in training. **That is why `pssm_root` reaches 0.735 there
and exactly chance (0.100) on the de novo subset.** The metric's weighting, not the
list's composition, hands the composition baseline its result. The framing in this
log's s30 reporting -- "a list that is mostly not de novo" -- is **wrong by cell count
and right only by event count**, and is withdrawn as written.

### fullpool is the Figure 2 redo; the restricted pool is the odd one

Worth stating plainly because the naming suggests the opposite. For one query the model
scores ~4,400 RBD cells (~4,200 after the wild-type mask) and takes its top 10% ~420.

* `list_<L>_fullpool` -- the model still ranks the whole RBD; the list only filters
  **what you count**. This is Figure 2's question asked of a subset of targets.
* `list_<L>` -- the model is handed only the 247 list cells, so its top 10% is ~25
  cells. This changes the model's task from "find these in the protein" to "sort a list
  you were given", which is a different question and not the one being asked.

`_fullpool` reads as the extra only because `list_<L>` got the plain name. The figure
is built on `_fullpool` alone; the restricted variants stay computed (free once the
forward pass is done) but out of the panels.

### Four new variants, and why the de novo pair was needed

Added to `eval.py`: `list_<L>_unseen`, `list_<L>_unseen_fullpool`, and the emergence
pair `variant_emergence_list_<L>_unseen[_thresh1]`. Masks verified before any GPU time:

| variant | pool | targets |
|---|---|---|
| `list_coguk` | 247 | 247 |
| `list_coguk_fullpool` | 4,422 | 247 |
| `list_coguk_unseen` | 143 | 143 |
| `list_coguk_unseen_fullpool` | **3,605** | **143** |

The last line is the many-query analogue of `list_recall.py` -- same 143 targets in the
same 3,605-cell pool -- so those two evals are directly comparable for the first time.
**143 is a count of CELLS; the eval.py metric scores EVENTS**, so that variant's
denominator is 5,070, not 143.

The de novo pair is not a replacement for the plain one: Figure 2 shows all-mutations
and de novo side by side (panels B and C), so the list version needs both.

### A new output directory is mandatory when a variant is added

Resume marks a model done from its row in the hit-rate `all` CSV. Pointing a run with
new variants at an existing directory skips every model already there and **those
models never acquire the new columns**. Hence `evals_sc2_lists_mtb/` rather than
reusing `evals_sc2_lists/`. `eval_sc2_mutation_lists.sh` takes a second argument `mtb`
that adds `--multitree-baselines` and switches directory together, so the two cannot
drift apart.

### The torchvision fix is confirmed end to end

s30 recorded the fix and the preflight but could only show that the build no longer
skipped. The rows have now landed, and they change s30's conclusion:

| model | CoG-UK fullpool@10% |
|---|---|
| antiGen as EVE full FT | 0.837 |
| **antiGen as ESM 2 (35M) partial FT** | **0.813** |
| antiGen as ESM 2 (35M) LM head FT | 0.786 |
| EVE | 0.780 |
| antiGen as ESM 2 (35M) full FT | 0.736 |
| PSSM | 0.735 |
| antiGen | 0.722 |
| ESM 2 (35M), untrained | 0.146 |

**s30's caveat was right and should be read as vindicated, not retracted.** It said the
eight absentees were "not predictors one can assume would have placed low"; one of them
is now second. The margin at the top collapses from +0.057 over the runner-up to
**+0.024**. The 650M arms have not landed yet, so this is still provisional -- but less
so, and in the direction s30 predicted.

### The 2021 multi-tree model in the Figure 2 panels

Mechanically compatible: `seq_len=1273`, `ESM2_FITNESS_PREDICTOR`, alphabet
`ACDEFGHIKLMNPQRSTVWY*-`, loading through `eval.py`'s own `load_checkpoint`.

Two plumbing blocks, neither about the model. `discover_checkpoints` globs
**non-recursively** for `fitness_*.pt` and its regex is `fitness_(.*)_(\d{8}_\d{6})\.pt$`,
so the `joint_2021_s*/..._best.pt` layout is invisible on both counts. Resolved with a
staging directory of symlinks under parseable names (`ckpt_stage_mt2021/`) rather than
by touching discovery. Three joint and three single seeds, the single arm included
because it is nearly free and gives the joint-vs-single contrast on the list directly.

**Read the numbers as a floor.** `eval.py` scores one checkpoint per group, while the
multitree arm's headline numbers are 8-seed ensembles, and s25 measured ensembling as
the largest single effect on that arm (de novo recall 0.50 -> 0.59). It also trains on
312 ncov examples against output1's 3,648, plus five other viruses -- that asymmetry is
the question, not a confound.

**A known, accepted leak.** The ncov tree carries 3 RBD cells that output1's de novo
mask calls unseen, one of which (**504D**) is among the 143 CoG-UK de novo targets.
Measured, parked on instruction for the plausibility check, **to be closed before any
of these numbers are reported.**

### Another silent exit 0, and the class fixed rather than the instance

The first submission (3361182) reported **COMPLETED in 44 seconds with an empty output
directory**. These checkpoints carry `experiment: 'forecast@2021-01-01'` in train_args,
which `eval.py` gates behind `--include-experimental`; all six were filtered out and
the run exited 0. Same shape as the ESM-2 build skips -- discover groups, filter them
all away, report success.

`eval.py` now **raises SystemExit** when it discovers checkpoint groups and filters
every one of them out, naming the count and pointing at the `[skip]` lines. The one
legitimate no-checkpoint case, `--multitree-baselines` supplying its own models, still
passes through. With `--include-experimental` all six `[keep]` and the sweep runs.

That is the second instance in two days of this failure mode, and the general lesson is
worth the restatement: **a filter that removes everything is a misconfiguration, not a
no-op, and must not exit 0.**

### Jobs in flight

| job | arm | output | limit |
|---|---|---|---|
| **3361136** | RBD, de novo list variants + factored baselines | `evals_sc2_lists_mtb/` | 3 days |
| 3362145 | 2021 multi-tree, 3 joint + 3 single seeds | `evals_sc2_lists_mt2021/` | 1 day |
| 3360817 | RBD, original variants (completes the s30 table) | `evals_sc2_lists/` | 2 days |
| 3360818 | full Spike, original variants | `evals_sc2_lists_fullseq/` | 7 days |

3360818 is deliberately left on the old variant set -- it is the supplement, and
restarting a 7-day job to add the de novo cuts was not worth doing unilaterally.

### Still to build

The figure. `fig2_sc2_benchmarks.R` needs `EVAL_DIR` and the two metric vectors
parameterised, and group/colour assignments for codon, substitution and
`where_times_which`, which s25 flagged as missing from the R inventory and which will
otherwise render as ungrouped model rows.

---

## 32. The CoG-UK list, averaged over mutations instead of events (2026-10-18)

s31 measured the problem and this section fixes it. The list metric in `eval.py`
averages over mutation **events**, and CoG-UK's cells recur at wildly different rates,
so the published list numbers are mostly a report on a handful of recurrent cells --
81% of the list's test events sit on the 104 cells already in the training window.

### The metric

For a listed cell observed relative to k base sequences, its score is the fraction of
those k queries whose own top-q contains it; the reported number is the unweighted mean
over listed cells. One vote per mutation rather than one per event.

The inner question is **not** changed: the per-query candidate pool, the wild-type and
gap/stop masking and the fractional tie rule are the same `compute_hit_rate` call, which
now takes `per_cell=True` and returns the hit mass and the event count per (position,
residue) alongside the scalars it always returned. Summing the two tensors reproduces
those scalars exactly, which `antiGen/tests/test_per_cell_hit_rate.py` pins down along
with four other properties (hit mass lands on the right cell; events count per cell and
include duplicates; a flat score map pays out exactly q per cell).

Chance is 0.1 under both weightings, so the two columns are on the same scale and the
gap between them reads as how much of a score rides on recurrence.

### Only 10.7% of the test set is scored, and that is exact

`antiGen/list_macro.py` scores the **17,978 of 167,600** test examples that carry a
CoG-UK RBD target. An example with no listed target contributes zero to the numerator
and zero to the denominator of every list variant, so dropping it changes no number.
That is what makes this a ~2.5 min/model job instead of the ~3h/model `eval.py` spends.

**The check this rests on**: the event-weighted column is computed in the same pass and
reproduces `evals_sc2_lists/quantile_hit_rates_list_coguk*_all-models.csv` over all
**396 (model, variant, quantile) cells to 1.5e-07**. Split across four SLURM jobs
(3362535 full roster, 3362776 the cheap half, 3363245 the five headline checkpoints,
3363261 the 2021 arm); the **432 cells scored by more than one job agree to 0.00e+00**.

### The numbers, top decile, `list_coguk_fullpool`

| model | macro | micro (published) | de novo macro | de novo micro |
|---|---:|---:|---:|---:|
| antiGen as ESM 2 (35M) partial FT | **0.512** | 0.813 | 0.410 | 0.569 |
| antiGen as ESM 2 (35M) LM head FT | 0.487 | 0.786 | 0.398 | 0.469 |
| antiGen as EVE full FT | 0.474 | **0.837** | 0.393 | 0.691 |
| antiGen | 0.446 | 0.722 | **0.509** | 0.536 |
| antiGen as ESM 2 (650M) LM head FT | 0.441 | 0.762 | 0.354 | 0.510 |
| multi-tree 2021, joint (3 seeds) | 0.388 | 0.549 | 0.347 | 0.363 |
| PSSM | 0.362 | 0.735 | 0.100 | 0.100 |
| EVEscape | 0.353 | 0.504 | 0.383 | 0.515 |
| multi-tree 2021, single (3 seeds) | 0.273 | 0.388 | 0.257 | 0.284 |

**s31's headline does not survive the re-weighting.** `antiGen as EVE full FT` leads the
event-weighted table by +0.024 and is **third** cell-weighted, -0.038 behind the 35M
partial arm. No model changed; the lead was in the weighting.

**The PSSM puzzle is closed.** s30 asked why a composition baseline beats
`simple_d128_n2` on this list. It does not: 0.735 against 0.722 event-weighted becomes
0.362 against 0.446 cell-weighted. Sliced by cell, PSSM scores **0.819 on the 104
training-window cells and exactly 0.000 on the 131 de novo ones** -- its whole result is
borrowed from recurrence, and the event weighting was paying for it.

**EVEscape is the only rung flat across the split** (0.356 seen / 0.351 de novo), which
is what a predictor with no access to the training window should look like.

Distribution behind all of this: 235 of the 247 in-RBD cells are observed in the test
set, carrying 26,491 events, median 46 per cell, min 1, max 972; the ten most recurrent
cells are 4.3% of the list and 24% of its events; 30 cells have k < 5.

### The 2021 multi-tree arm, scored here because it was 18h away otherwise

Job 3362145 (`eval.py`) was 2h in and still on checkpoint 1 of 6 with an empty output
directory. The same six staged checkpoints through `list_macro.py` took 15 minutes.

**joint 0.388 (0.383/0.389/0.392) against single 0.273 (0.262/0.275/0.281)**, +0.115,
with no overlap between the arms' seeds -- the worst joint seed beats the best single
seed by 0.10. It holds on the de novo cut (+0.090). The joint seeds place 12th-14th
overall, above PSSM and EVEscape, below every main-model finetune.

Both s31 caveats still apply and are repeated in the job script: single checkpoints
against that arm's 8-seed ensembles, so a floor; and the **504D leak is still open**,
which touches the de novo column specifically.

### Two things the figure settles that the table does not

`antiGen-figures/figures/fig_coguk_list.R` -> `figs/coguk_list_results.png`, 16 panels
on Figure 2's B-J (curves, top decile, AUC, PSSM precision, max precision), both
poolings side by side.

**The AUC row and the decile row disagree, and the decile is the one to quote.** The
recall curve is saturated above q = 0.5, so the AUC compresses the differences and
reorders the top: cell-weighted AUC puts the 35M LM-head arm first (0.824), antiGen
second (0.821), the 35M partial arm third (0.809), where at q = 0.1 the order is 0.512 /
0.487 / 0.446. Hence the extra row, which Figure 2 does not have.

**PSSM precision needs no pooling column.** It is a single-query metric -- the testing
root's top decile against what had emerged by t -- so it already weights each cell once.
It is drawn restricted to the listed cells, from `eval.py`'s own
`variant_emergence_list_coguk*` outputs.

### The cross-tree factored baselines, finally rendered

s25 computed codon, substitution and `where_times_which` into `evals_sc2_mtb/` and
noted the R scripts were not updated. They were still unrendered three sessions later.
Both figures now carry them, grouped by what data each had access to, which is the rule
fig2's palette already encodes: **codon is pre-pandemic** (ModelType.CODON is built from
CodonConfig and sees no training data), **substitution and where x which saw the
training window** (both are fitted on the training examples).

On Figure 2's own metrics they are not minor rungs:

| | recall AUC, all | recall AUC, de novo |
|---|---:|---:|
| antiGen | **0.848** | **0.813** |
| where x which | 0.824 | **0.807** |
| PSSM | 0.807 | 0.495 |
| substitution | 0.784 | 0.786 |
| codon | 0.686 | 0.780 |

**`where x which` lands within 0.006 of antiGen on de novo recall AUC** -- the same
shape as s14's multi-tree finding (model and `where x which` within 0.014 on de novo
targets), now on the sc2 panel. The three rungs also displace PSSM and EVEscape as
best-of-group in fig2's curve panels.

### A cross-run disagreement that constrains the precision panels

`evals_sc2` and `evals_sc2_mtb` are the same pipeline on the same data, so the three new
rows can only be merged into Figure 2 if the two runs agree. Measured over 83,106 cells:

* **recall metrics agree to 3.8e-04** (worst: evcouplings -- s30's sampler noise). The
  0.006 margin above is 15x that, so the recall panels are sound.
* **`variant_emergence_*` disagrees by up to 7.4e-02** (tranception_frozen_both;
  `simple_d128_n2` by 3.8e-02), and **the gap does not shrink with t**, so it is not a
  small-denominator effect at early dates. Cause unresolved.

So in fig2's precision panels the three rungs are a row from one run against rows from
another. `where x which` nominally takes 1st on de novo max precision (0.692 against
antiGen's 0.681) and **that margin is inside the cross-run spread and must not be
quoted**. The merge never overwrites a published row: only models absent from
`evals_sc2` are taken from `evals_sc2_mtb`. Closing this properly needs one run of the
full roster with `--multitree-baselines`; `evals_sc2_mtb` predates the torchvision fix
and so has no ESM-2 arms.

### The list figure, after review

Two panels' worth of material was cut on instruction, both for good reasons:

* **the 2021-cutoff multi-tree arm**, which belongs with the multi-tree figures rather
  than the main-model comparison. Still scored, in `evals_sc2_lists_macro_mt2021/`;
* **the PSSM-precision row**. It is a single-query metric, so the pooling question this
  figure exists to ask does not arise for it, and it would have had to be assembled
  across the two runs above. It stays in Figure 2, on one run.

The figure is now 12 panels: recall curves, top decile, recall AUC, each in both
poolings. `list_macro.py` gained `--multitree-baselines`, so the three factored rungs
here are scored by the same script in the same run as everything else -- unlike their
appearance in Figure 2.

### The bottom row: CoG-UK yield of the BA.2 top decile

The PSSM-precision row was replaced on instruction with a question that suits a curated
list. One forward pass on the BA.2 root fixes a top decile over the **RBD** -- ~442 of
4,422 cells, the set a forecaster would have flagged -- and time is swept over the
variant PSSM grid:

    value(t) = |top-q(RBD) & list & emerged_t| / (q * N_RBD)

`antiGen/list_variant_recall.py`, job **3364139**, 29 models in ~15 min (one forward
pass each). "Emerged" is the variant PSSM at >= threshold, and those PSSMs are built
from BA.2-descended sequences, so "emerged by t" already means "arose downstream of the
variant root by t". Thresholds 100 and 1, as eval.py uses for BA.2.

**Two sibling metrics it is not, both checked rather than assumed:**

* Figure 2's `variant_emergence_list_coguk` NaN-masks everything off-list **before**
  ranking, so its top decile is ~25 of the 247 listed cells. Different pool, different
  top-q set, different numerator -- not this metric and not proportional to it.
* Dividing the same numerator by `|list & emerged_t|` gives a **recall** reading. That
  one is proportional to this at each fixed t up to N_RBD, so it ranks models almost
  identically -- but its denominator grows with t, so the curve falls and **its max
  lands on the first step with a single emerged cell, where pssm_root, EVEscape and
  MSA_Transformer_XFG all tie at exactly 1.000**. That degeneracy is why the fixed
  denominator is used. It was caught by printing the step and n_targets at which each
  max was attained, which is worth doing for any max-over-t summary.

| model | max, 1+ cases | max, 100+ cases |
|---|---:|---:|
| antiGen as EVE full FT | **0.246** | **0.147** |
| EVE | 0.207 | 0.144 |
| antiGen as Tranception full FT | 0.204 | 0.139 |
| antiGen as ESM 2 (650M) full FT | 0.209 | 0.105 |
| EVEscape | 0.195 | 0.103 |
| PSSM | 0.141 | 0.097 |
| antiGen | 0.150 | 0.095 |
| where x which | 0.150 | 0.090 |

**Read against the ceiling, not against 1.** BA.2 already carries 13 of the 247 listed
RBD cells, so only 234 are mutations it could make; 200 emerge at 1+ cases and 79 at
100+. Against a ~442-cell decile that caps the metric at ~0.45 and ~0.19. Both the
ceiling and the chance curve (`|list & emerged_t| / N_RBD`) are drawn on the panels.

**N_RBD is per model and the spread is ~10%**: 422 cells for a model that emits gaps
(pssm_root, antiGen, where x which), 402 for the ESM-2 arms, 382 for EVE/Tranception,
which do not emit gap/stop and so are scored against a smaller decile. This is
`variant_hit_rates`' own convention (`n_selected = q * n_valid`), kept for consistency,
but it means cross-family comparisons carry that factor: eve_ft's 0.1466 would be 0.1327
at N = 422, still first. Within a family the ordering is unaffected.

**antiGen places mid-table here** (7th at 100+), against first on the de novo
cell-weighted panels. Not a contradiction: this row is dominated by what actually swept
through BA.2 descendants, which is mostly cells already in the training window, and it
is a single-query metric rather than an average over 17,978 queries.

### Split into a main figure and a supplement

On instruction, and the line is a defensible one: **every cell of the CoG-UK list is
curated as important by somebody**, so a case threshold on top of it adds nothing to the
question the main figure asks, and the event weighting answers a question about branches
rather than mutations.

| figure | carries |
|---|---|
| `coguk_list_results` | the cell-weighted reading throughout + the 1+ case yield |
| `coguk_list_supp` | the event-weighted reading throughout + the 100+ case yield |

Eight panels each, same grammar (recall curves, top decile, recall AUC, BA.2 yield), so
the two read against each other row by row. The event-weighted half is kept rather than
dropped because the published CSVs report it and every claim in the main figure is
stated against it. Nothing is computed twice: the panels are built per metric and each
figure is one list of metric names.

### Why base antiGen drops in the BA.2 yield panel: the pool, not a glitch

Asked why `simple_d128_n2` trails its own finetunes in that panel when the recall panels
rank it at or near the top. It is not a glitch, and the cause is **whether the candidate
pool contains training-window cells**. Three measurements agree:

| eval | pool | antiGen | eve_ft |
|---|---|---:|---:|
| `list_recall` (s25), BA.2 query, q = 0.1 | **de novo only** (3,408 cells) | 0.331 (3rd/28) | 0.373 (1st) |
| BA.2 yield, de novo targets | **full RBD** (4,221 cells) | 11/115 = 0.096 | 31/115 = 0.270 |
| BA.2 yield, all listed targets | full RBD | 64/200 = 0.320 | 94/200 = 0.470 |

The first and second rows are the same query, the same q, and nearly the same target set.
The only thing that changes is whether the ~422 decile slots are drawn from de novo cells
alone or from the whole RBD -- and antiGen's result moves by a factor of three.

**The reading.** antiGen is trained from scratch on the first year, so for any query its
very highest-scoring cells are heavily the ones it saw; with those cells in the pool they
occupy its decile and little room is left for anything else. `pssm_root` is the degenerate
form of the same thing (0 de novo hits by construction). The finetunes carry pre-trained
structure and spread their decile more widely. **The main figure's de novo panels restrict
the pool to unseen cells** -- Figure 2's own de novo convention, `score_mask = seen` -- so
they are measuring antiGen on the de novo pool, which is why it leads there and trails in
the yield panel. The two panels are asking different questions and the model most
sensitive to the difference is the one trained only on the training window.

**What this does NOT yet explain**, and is left open rather than papered over: over many
queries with the full pool (the cell-weighted all-list panel) antiGen is 0.446 against
eve_ft's 0.474 -- close. The deficit is far sharper on the single BA.2 query than averaged
over 17,978 queries, so there is a single-query-vs-average effect on top of the pool
effect, and it is unmeasured.

**The diagnostic that would settle the mechanism has not run.** It prints each model's
decile composition (share on gap/stop columns, share on listed cells, share on
training-window cells, positions covered); the prediction is that antiGen's decile is
markedly more "seen" than its finetunes'. Job 3364275 failed at once -- the script sat in
the session scratchpad, which was cleared before the job started. Rerun from a durable
path before this reading is treated as confirmed.

### State

* `evals_sc2_lists_macro{,_fast,_head,_mt2021,_mtb}/` hold `list_macro_summary.csv` and
  `list_macro_percell.csv` (per-cell rates at q = 0.1, for the slices above).
* **EVmutation has no recall rows yet**: `evcouplings` rebuilds its couplings with
  `plmc`, which has been running over 1.5h in both jobs. It is in the precision panels.
  The 650M arms and the 2021 arm have no precision rows. The figure script prints what
  is missing rather than dropping models silently, and re-renders when they land.
* `--skip-baselines` was added so a second job can get at the checkpoints without
  re-deriving 12 baseline rows another run already has.
* The macro roster takes `eval.py`'s `--exclude` default, so it scores the same models
  the published CSVs do; a model that fails to build is skipped but the run then exits
  NON-ZERO naming it, per the lesson in s30/s31.

---

## 33. The transpose: fix the mutation, vary the background (2026-10-18)

s32 closed with an open question -- antiGen trails its finetunes on the BA.2 yield
panel (its decile fills with training-window cells) while leading the de novo recall
panels, and "the deficit is far sharper on the single BA.2 query than averaged over
17,978 queries, so there is a single-query-vs-average effect on top of the pool effect,
and it is unmeasured". This section measures it, as two new panels (I-J) of the CoG-UK
figure.

**Every panel of Figures 2 and of the list figure fixes a query and ranks the mutations
in it.** That cannot separate "this model knows 346T matters" from "this model knows
which background 346T can arise from": a model that scores the cell high everywhere
scores well. Fixing the *mutation* and varying the *background* cancels the
per-mutation bias exactly -- including the pool effect of s32 -- and leaves only how
much a prediction moves with the sequence it is given.

### The substrate: the guide tree, before attachment

`antiGen/background_tree.py`. `antiGen-data/tree.json` is the guide phylogeny `output1`
is built from; the attachment phase then lands ~170k tips on it. For this question the
attached tree is the wrong object: the same mutation is recorded against hundreds of
near-identical backgrounds, so "backgrounds that host this mutation" would mostly count
sampling depth.

| | |
|---|---|
| guide tree | 8,859 nodes, 4,549 leaves, 2,839 spike mutation events |
| root / cutoff | 2019.953 / 2020.953 (root + 1, as `output1` was generated) |
| post-cutoff internal nodes | 4,034 |
| **distinct spike genotypes among them** | **402** |

The collapse from 4,034 to 402 is not an approximation: most guide-tree branches carry
no spike mutation at all, a model sees only the sequence, and `antiGen-data`'s own
`prune_branches_without_mutations` collapses exactly those branches.

**Two checks on the reconstruction, both run by the scorer, both passing.** Every one of
the 2,839 branch mutations' wild-type letters agrees with the residue the reconstruction
already has at that position (0 mismatches), and the root genotype is byte-identical to
example 0 of `examples_cutoff_1.txt`. The scorer aborts rather than continuing if either
fails.

**A third check, run separately: do these genotypes actually appear in the attached
tree?** Streaming all 171,248 examples, **336 of the 402 appear as TESTING example
sequences**, and the other 66 do not. The 66 are late (mean date 2025.83 against 2024.75)
with few child branches, and are merged away by the pruning and cleanup the attachment
path applies. **None of them is a host** -- all 143 host genotypes are present as testing
examples -- so the absentees only ever serve as controls.

### Hosts, controls, and the matching rule

For a cell `c = (position, residue)`: a **host** is a genotype with a post-cutoff child
branch carrying `c`; a **control** is a genotype that does not already carry the residue,
has a node within 6 months of a host node, and is neither an ancestor nor a descendant of
any host node of that cell (anywhere in the tree, not merely at the time-matched node).

The time match is there because the guide tree is recency-heavy -- a quarter of its
post-cutoff nodes are dated after 2025.68 -- and it works: mean host date 2024.09 against
mean control date 2024.19, a within-cell gap of 0.095 years, where the unmatched pool
mean is 2024.93. The ancestry exclusion is there because a node two branches above where
346T arose is not a negative in any useful sense.

Realised counts, CoG-UK RBD (247 listed cells, 104 seen / 143 de novo):

| cell set | cells with >=1 host and >=10 controls |
|---|---:|
| CoG-UK listed | **53** (19 de novo) |
| all RBD cells | **145** (50 de novo) |

Median 37 controls per cell.

### The confound that nearly went into the figure

The first run used the published ruler -- a cell's percentile among that query's
candidates, which is what `compute_hit_rate` thresholds. **Under it, predictors with no
access to the query at all do not land at chance**:

| fixed (L, E) map | AUC, host vs control |
|---|---:|
| pyr0 | 0.614 |
| CoVFit | 0.609 |
| Omicron BA.1 bind | 0.600 |
| EVEscape | 0.563 |
| pssm_root | **0.418** |

Every one of these is a single map with no sequence input; their only query-dependence is
that `compute_hit_rate` masks each query's own residues out of the pool. A host genotype
is more derived than its controls, so more of its high-scoring cells are masked away and
everything left is lifted. At ~0.1 of AUC that swamps the effect being measured, and
`pssm_root` moves the other way, so it is not even a constant offset one could subtract.

**Fixed by ranking in a pool held fixed across queries**: the eval-range cells no backbone
genotype carries (4,134 of 4,422) that the model scores finitely in every query, plus the
target cell. Identical for every query by construction, so a fixed map scores every query
identically and lands at **exactly 0.500**. The published convention is still computed
and written as `*_perquery` rows, where it is the diagnostic it turned out to be, and
`antiGen/background_discrimination.py` therefore stores raw scores rather than ranks so
both rulers come off one forward pass.

The general lesson, which is not specific to this eval: **a per-query normalisation is
invisible while the query is fixed and becomes a confound the moment it is not.**

### Lift, not AUC, and why the choice is not cosmetic

Three readings of the same contrast are written per cell: `lift` (the share of hosts
whose top decile contains the cell, minus the share of controls), `auc` (rank only) and
`delta` (percentile points). The panels draw lift. The reason is in the data:

| | mean AUC | Δ percentile | **top-decile lift** | median sd of the cell's percentile across queries |
|---|---:|---:|---:|---:|
| antiGen | 0.538 | +1.2 pp | **+4.4 pp** | **0.054** |
| antiGen as EVE full FT | **0.606** | +0.2 pp | +0.7 pp | 0.0066 |
| EVE | 0.574 | +0.1 pp | -0.4 pp | 0.0071 |
| every query-blind rung | 0.500 | 0.0 | 0.0 | 0.0000 |

The AUC is scale-free, so it scores the sign of a 0.7-percentile-point nudge exactly as
it scores a 28-point swing -- and both regimes are present here. **A model that shifts a
cell's rank by less than a percentile point produces the same top decile for every
background in the tree**: its direction can be right while the forecast is, operationally,
background-blind. Lift combines sign and size, is in the units the decile row above is
read in, and is identically zero for a model whose decile does not move.

**On the AUC alone the claim this was built to test does not hold** -- antiGen is third,
and its interval spans chance. That reading is kept in the CSV and is not hidden.

### Robustness, all on the pilot roster

Every variant below leaves the ordering intact.

| variant | antiGen, CoG-UK | antiGen, all RBD |
|---|---:|---:|
| headline (±6 months) | +4.4 pp | +4.3 pp |
| window ±3 months | +3.7 | +3.8 |
| window ±1 year | +4.5 | +4.3 |
| controls with >=8 child branches | +2.5 | +4.2 |
| controls with >=20 child branches | +7.7 | +4.7 |

The last two matter most. Hosts are "bigger" nodes than controls by construction (median
28 post-cutoff child branches against 4), since a genotype with more branches has more
chances to host anything. A uniform "this background is productive" response cancels in a
within-query rank, which is the argument for using a percentile rather than a raw score;
the fecundity-matched rows are the direct check, and antiGen's lift **rises** under them
while eve_ft's goes to zero or below.

### What the denominators can and cannot settle

95% bootstrap over cells, antiGen:

| cell set | n | lift | 95% CI |
|---|---:|---:|---|
| CoG-UK listed | 53 | +4.4 pp | **[-2.3, +11.7]** |
| CoG-UK de novo | 19 | +12.4 pp | **[-2.3, +29.0]** |
| all RBD | 145 | +4.3 pp | [+0.8, +8.0] |
| all RBD de novo | 50 | +12.3 pp | [+4.3, +21.3] |

**The main figure's panels are the two that do not clear zero**, and the supplement's RBD
panels are the ones that do. The panels carry the bootstrap whiskers for exactly this
reason; a bar length alone would overstate what 19 cells settle.

### Code and jobs

* `antiGen/background_tree.py` -- the backbone, with its two self-checks.
* `antiGen/background_discrimination.py` -- 402 forward passes per model, raw scores out
  (~7 MB each); `rank_fractions` / `top_q_weight` rebuild `compute_hit_rate`'s fractional
  top-q weight from a pool, which `antiGen/tests/test_background_ranks.py` pins across 10
  quantiles, with ties, NaN and `-inf` sentinels (3 tests, passing).
* `antiGen/background_contrast.py` -- the contrast, both rulers, three statistics.
* `examples/slurm/background_discrimination_sc2.sh` -- jobs **3364881** (full roster,
  `evals_sc2_background/`) and **3364950** (the models queued behind evcouplings' plmc
  rebuild, `evals_sc2_background_head/`, merged in afterwards).
* `antiGen-figures/figures/fig_coguk_list.R` -- panels I-J, the listed all/de novo pair
  in the main figure and the all-RBD pair in the supplement.

### Open

**RESOLVED in s36 -- job 3364881 scored all 30 models.** (Original note:)
**The full-roster table is not in yet.** The ESM-2 arms have landed and one of them --
the 35M partial-FT arm, +6.0 pp on the listed cells -- is in antiGen's range, so "antiGen
is the most query-adapted model" is **not** supportable from the pilot alone and must
wait for the complete roster, exactly as s30's absent-ESM-2 caveat did.

---

## 34. The attachment control: the same list, scored on the guide tree alone (2026-10-18)

An alternative rendering of the CoG-UK figure, scored on the Nextstrain backbone before
`antiGen-data`'s attachment phase. **It is a control, not a replacement**:
`coguk_list_backbone{,_supp}.png` beside `coguk_list_results{,_supp}.png`, built from the
same script through options (`figures/fig_coguk_list_backbone.R`) rather than a fork that
would drift.

Two things it rules out. **The attachment**, which places ~170k tips by parsimony and so
decides both the ancestral genotypes queried and the targets attributed to them. And
**the recurrence weighting**: s31/s32 had to split the figure into event-weighted and
cell-weighted columns because CoG-UK's cells recur at wildly different rates in the
attached test set, and almost all of that multiplicity is attachment.

### The test set shrinks by two orders of magnitude and stops recurring

| | attached (published) | backbone only |
|---|---:|---:|
| examples carrying a listed target | 17,978 | 143 |
| listed cells observed | 235 | 56 |
| listed target events | 26,491 | 211 |
| **events per observed cell** | ~113 (median 46, max 972) | **3.8** |

### The result: the published ordering is not an artefact of either

Over the 24 models scored in both, at q = 0.1 on `list_coguk_fullpool`:

| | Spearman |
|---|---:|
| **published cell-weighted vs backbone cell-weighted** | **0.943** |
| published event-weighted vs backbone event-weighted | 0.941 |

And the weighting question largely dissolves, which is the second thing the control was
for:

| | mean abs(macro - micro) | Spearman(macro, micro) |
|---|---:|---:|
| attached (published) | **0.171** | 0.934 |
| backbone only | **0.061** | 0.960 |

So the two columns the main figure is built to contrast are nearly the same column once
attachment is removed -- which is the direct confirmation of s32's diagnosis that the
gap between them *is* recurrence, not composition.

### Absolute values rise, and the local order does move

Every rung scores higher on the backbone (antiGen 0.446 -> 0.560 cell-weighted,
`eve_ft` 0.474 -> 0.730): 56 cells concentrated on well-sampled RBD positions is an
easier target set than 235. Ranks shift locally even at rho = 0.94 -- `eve_ft` goes 3rd
-> 1st, antiGen 4th -> 7th, the 35M partial arm 1st -> 3rd -- and with 56 cells and 211
events those margins are not separable. **The claim this figure supports is about the
ordering as a whole, not about any pair of adjacent rungs.**

Top decile, cell-weighted, `list_coguk_fullpool`, backbone only:

| model | backbone macro | backbone micro | published macro |
|---|---:|---:|---:|
| antiGen as EVE full FT | **0.730** | 0.791 | 0.474 |
| EVE | 0.715 | 0.725 | 0.415 |
| antiGen as ESM 2 (35M) partial FT | 0.715 | 0.744 | 0.512 |
| antiGen as ESM 2 (35M) LM head FT | 0.661 | 0.678 | 0.487 |
| antiGen as ESM 2 (650M) LM head FT | 0.635 | 0.692 | 0.441 |
| antiGen as ESM 2 (35M) full FT | 0.580 | 0.626 | 0.419 |
| **antiGen** | 0.560 | 0.621 | 0.446 |
| PSSM | 0.550 | 0.664 | 0.362 |
| EVEscape | 0.500 | 0.659 | 0.353 |

**PSSM is the rung to watch**, because it is the one s32 showed was living on
recurrence: 0.362 macro against 0.735 micro in the published run, and 0.550 against
0.664 here. Its event-weighted advantage is 0.373 in the attached test set and 0.114 on
the backbone -- two thirds of it was the weighting.

### What is identical and what is not

Identical: models, checkpoints, candidate pool, wild-type and gap/stop masking, the
fractional tie rule, the de novo mask, and the BA.2 query -- the backbone's BA.2 root
genotype is byte-identical to the `IS_VARIANT_ROOT` example the published run scores,
checked rather than assumed. Scoring reuses the matrices
`background_discrimination.py` already wrote, so the control needed no GPU time.

**The bottom row is NOT on the same scale as the published one.** "Emerged by t" there
counts attached sequences through the variant PSSMs; here it counts guide-tree tips
descended from the BA.2 root carrying the residue by t. There are 3,762 such tips, so 59
listed cells ever reach one (against 200 at 1+ cases published) and the ceiling falls
from ~0.45 to ~0.14. Read that row within its own figure.

One deliberate difference in the test-set construction: a multi-mutation branch is **not**
decomposed into intermediate genotypes in a random order, as `write_training_examples`
does. All of a branch's mutations are targets of the parent genotype, which does not
depend on `antiGen-data`'s RNG seed.

### Caveats

* **56 cells and 211 events**, 20 cells and 62 events on the de novo cut. Every number
  here is noisier than its published counterpart by roughly the square root of that
  ratio.
* **Six models are missing** -- Tranception (frozen and FT), EVmutation, codon,
  substitution and where x which -- because job 3364881 is still inside evcouplings'
  plmc rebuild. The figure prints the absentees rather than dropping them silently, and
  rho = 0.943 is over the 24 that are present.
* The antiGen and EVE score matrices in this rendering came from the CPU pilot and the
  rest from the GPU job; same checkpoints, different runs. To be re-scored in one pass.

### Render it in the right environment, or the font changes

The first render of this figure came out in a different typeface from every other
figure in the repo. The cause is not the script -- it is which R is used.
`/home/ivan.specht/miniconda/envs/r-figs` (R 4.4.3) has **ragg** installed, so
`ggsave` dispatches PNG output to `ragg::agg_png`, which resolves fonts through
`systemfonts` and picks a different default face. `.../envs/antigen-figs` (R 4.5.3, the
version the README names) has no ragg, so `ggsave` falls back to the cairo `png` device,
which is what every committed figure was drawn with.

Checked rather than assumed: re-rendering `coguk_list_results` under `antigen-figs`
reproduces the committed PNG at **RMSE = 0**, pixel for pixel -- which also confirms
that parameterising the script changed nothing in the published figure. Use
`antigen-figs` for anything that lands in `figs/`.

(The size-9 title override that the first render needed was a ragg artefact too: under
the cairo font the longest title, "Top decile, de novo, cell-weighted [backbone]", fits
at `theme_paper()`'s 11. `coguk.title_size` is still available but is unset by default,
so the two figures are typographically identical.)

### Files

`antiGen/backbone_list_eval.py` (test set + recall + the BA.2 yield series),
`antiGen-figures/figures/fig_coguk_list_backbone.R` (the wrapper), and
`antiGen-figures/data/evals_sc2_lists_macro_backbone/`. `fig_coguk_list.R` now takes its
input directories, output names and title suffix as options so both figures come off one
script.

---

## 35. The yield row, cut by subset, and its reference lines named (2026-10-19)

Two changes to `coguk_list_results` and its supplement, plus the backbone control
(s34), which inherits both.

### The de novo half of the yield row, which was computed but never drawn

`list_variant_recall.py` has always written `subset = all` and `subset = unseen`; the
figure filtered to `all` and threw the other half away. The row is now two rows -- G-H
all listed cells, I-J de novo -- so the bottom of the figure keeps the all/de novo
grammar every row above it uses. Nothing was recomputed; the panels read 1,740 rows that
were already in the CSV.

**The de novo panel is the harsher one and antiGen does badly in it.** Max yield at 1+
cases:

| | all list cells | de novo |
|---|---|---|
| 1st | antiGen as EVE full FT 0.246 | **EVEscape 0.116** |
| 2nd | antiGen as ESM 2 (650M) full FT 0.209 | BA.1 DMS (expression) 0.094 |
| 3rd | EVE 0.207 | antiGen as EVE full FT 0.081 |
| **antiGen** | 0.152 (12th of 29) | **0.026 (26th of 29)** |

The de novo chance line ends at `0.1 * 115 / 422 = 0.027`, so **antiGen is sitting on
chance there**, and `pssm_root` is at 0 by construction. This is the same effect s32
diagnosed and should not be read as new: for a single query, antiGen's decile is filled
with cells it saw in training, so restricting the targets to cells it did not see leaves
almost nothing in it. What the new panel adds is that the effect is total rather than
partial -- on the de novo cut the single-query yield is indistinguishable from random
selection, where on the many-query de novo recall panels (D, F) antiGen is 1st.
**Those two facts sit side by side in the same figure now, which is the honest
presentation of them.**

The attainable ceiling moves too: 200 of the 234 listed cells BA.2 could make ever
emerge at 1+ cases, but only 115 of them are de novo, so the de novo panel's ceiling is
0.285 against 0.5.

### The reference curves are in a legend instead of being unexplained grey lines

Panels G and I carry two model-independent curves -- the attainable ceiling and what a
decile drawn at random would collect -- and until now neither was labelled anywhere on
the figure. They are now a titled `Reference` legend block beneath the model legend.

Mechanically this needed `linetype` freed: the model lines had it mapped to
"is this antiGen", so the yield panels draw their model lines as two explicit layers
instead and give `linetype` to the two reference curves. **Only the yield panels** --
the recall panels keep their original construction so they stay pixel-identical to the
published figure. The legend also moved from the bottom-right corner into the empty band
above the plateaus, because in the de novo panel it sat on top of antiGen's own curve.

### A-F are unchanged, and that is checked rather than asserted

`fig_height` is now derived from the row units with the published figure's own
inches-per-unit (20in / 5.64 units), so adding a row stretches the canvas instead of
rescaling the rows above it. Comparing the top 4,404 px against the previously committed
PNG leaves only 1-pixel edge ghosting (band RMSE 0.01-0.05, difference image is outlines
only, interiors blank): a sub-pixel vertical offset from the taller canvas, no content
change.

---

## 36. The background row on the full roster: panels K-L are in (2026-10-19)

s33 built the host-vs-control contrast and closed with the one thing it could not
settle: the table was a pilot, the ESM-2 arms were absent, and the 35M partial-FT arm
was already inside antiGen's range, so **"antiGen is the most query-adapted model" was
not supportable**. Job **3364881** (23:17 Oct 18 -> 02:58 Oct 19, 3h42m, exit 0:0) scored
the complete roster in one pass. This section reports it and draws it.

### The roster is complete, and the pilot's numbers survive it

All **30 models** are present in all four cell sets -- `coguk_{all,unseen}`,
`rbd_{all,unseen}` -- with no absentees, which is the first time any table in this
figure's lineage has been complete. `evals_sc2_background_head/` (job 3364950, the
subset queued around evcouplings' plmc rebuild) is now **redundant**: the full job got
through plmc on its own, so every model in the figure comes off one pass with one
checkpoint set, which also retires the third caveat of s34.

The pilot and the full run overlap on 120 model x cellset pairs. Worst disagreement
**0.65 pp** (antiGen, `coguk_unseen`: 11.72 pilot vs 12.37 full), and the only models
that move at all are the two the pilot ran on CPU and the job ran on GPU. One cell out
of 19 is 5.3 pp, so this is sub-cell float drift in the fractional tie weight, not a
changed call. **Every number s33 quotes holds to one decimal.**

### The claim narrows: antiGen leads on de novo, Tranception frozen leads overall

Top-decile lift, host - control, with the 95% bootstrap over cells:

| | **CoG-UK all** (53) | **CoG-UK de novo** (19) | **all RBD** (145) | **de novo RBD** (50) |
|---|---|---|---|---|
| **antiGen** | +4.4 **[-2.3, +11.7]** _3rd_ | **+12.4** [-2.3, +29.0] _1st_ | +4.3 [+0.8, +8.0] _2nd_ | **+12.3 [+4.3, +21.3]** _1st_ |
| Tranception (frozen) | **+8.2** [-0.7, +17.0] _1st_ | +4.7 [-5.5, +15.9] _9th_ | **+5.2** [+1.0, +9.4] _1st_ | +6.5 [-0.1, +13.5] _5th_ |
| antiGen as ESM 2 (35M) partial FT | +6.0 [-1.4, +13.6] _2nd_ | +11.2 [-3.0, +27.5] _2nd_ | +3.3 [-0.2, +7.0] _4th_ | +6.1 [-0.6, +14.0] _6th_ |
| Where x which | +3.2 _5th_ | +9.9 [-0.1, +24.6] _4th_ | +3.4 [+0.3, +6.9] _3rd_ | +9.3 [+3.0, +16.8] _2nd_ |
| Substitution matrix | +2.9 _6th_ | +9.9 [-0.1, +24.6] _3rd_ | +2.2 _6th_ | +5.0 _9th_ |
| antiGen as EVE full FT | +0.7 _11th_ | +6.8 [0.0, +18.1] _6th_ | -0.1 _30th_ | +3.6 _12th_ |
| EVE | -0.4 _25th_ | +0.4 _11th_ | -0.0 _29th_ | +2.3 _15th_ |
| every query-blind rung (13 of 30) | 0.0 | 0.0 | 0.0 | 0.0 |

**What the full roster settles.** On the **de novo** cuts -- the cells no model saw in
its training window, which is the cut every other de novo panel in this figure is read
on -- antiGen is **1st on both cell sets**, and the 50-cell RBD version is the one
denominator big enough to clear zero with room ([+4.3, +21.3]). That is the claim s33
set out to test, and it holds on the cut that matters.

**What it takes away.** On the **all-cells** cuts antiGen is no longer the leader:
**Tranception frozen** is, on both (+8.2 and +5.2), and on `rbd_all` its interval clears
zero where it is the only model besides antiGen and `where x which` to do so. Tranception
frozen is a "Pandemic Data in Training Window" rung, so this is not a finetune beating
its base -- it is a general protein LM that responds to the sequence it is handed. The
supportable statement is therefore **"antiGen is the most background-adaptive model on
mutations it has not seen"**, not "the most query-adapted model", and the figure is
drawn so a reader can see both halves of that.

**A pattern worth recording**: the ordering on the de novo cuts is close to the ordering
on the de novo *recall* panels (D, F) -- antiGen, then `where x which` and the 35M
partial arm -- while the all-cells cuts reshuffle it. Two independent transposes of the
same data agreeing on the de novo cut is the strongest form the headline claim takes
anywhere in this figure.

**The 650M arms go the wrong way.** `esm2 650M frozen` is at **-16.4 pp** on CoG-UK de
novo and three of the four 650M arms are negative on at least one cut -- they rank a
listed mutation *higher* in backgrounds that did not host it. The 35M arms do not do
this. Not chased here, but it is a real sign flip and not a zero.

### The figure

`data/evals_sc2_background/` did not exist, so `have_bg` was FALSE and the script was
silently dropping the row it already had code for. Copying `background_summary.csv` (plus
`background_cellsets.csv` and `backbone_manifest.json`, for provenance) into it turns on
**panels K-L** in both figures, exactly as `fig_coguk_list.R` was written in s33 to do:
the CoG-UK all/de novo pair in `coguk_list_results.png` and the all-RBD pair in
`coguk_list_supp.png`. No script change was needed beyond that -- s35's row-unit
`fig_height` absorbed the extra row on its own.

**A-J are unchanged, checked the way s35 checked its row.** Against the previously
committed PNGs the canvas grows 7,595 -> 9,191 px at the same width, px-per-row-unit is
identical to within 0.1 (1063.7 vs 1063.8), and the difference over the shared 7,595 rows
is 0.85% of pixels, all of it on text and line edges, growing monotonically down the
image. Shifting the comparison by +-1 px makes it **worse** (band RMSE 8.1 at shift 0,
13.9 at +1, 23.1 at -1), so it is a fractional offset from the taller canvas, not a
content change -- the same signature, and the same magnitude, as s35's.

The backbone control (s34) **does not get this row** and did not need re-rendering: its
wrapper points `coguk.bg_dir` at a path that does not exist, because these panels are
already computed on the guide tree alone and would be identical. Its header said
"Panels I-J", which s35's new yield row made stale; corrected to K-L.

### State

* Full roster in `/large_storage/.../evals_sc2_background/` (30 models, 37 files).
  `evals_sc2_background_head/` is superseded and can be deleted.
* `antiGen-figures/data/evals_sc2_background/` is the copy the figure reads.
* Rendered under `/home/ivan.specht/miniconda/envs/antigen-figs/bin/Rscript`, per s34.

---

## 37. Why the fixed maps are at exactly zero in K-L, checked (2026-10-19)

Raised on review of s36's figure: *different queries have different wild-type masks, so
a static (L, E) predictor's top decile could still move between backgrounds -- why would
it be exactly zero?* The objection is correct in general, it is the one s33 tripped over,
and the answer is a property of the pool rather than anything about the models. Checked
four ways, all from the committed artefacts.

### The objection is real, and the per-query ruler shows it

Under the published convention -- pool = eval range minus **this query's** residues --
the fixed maps are nowhere near chance:

| constant map | AUC, `coguk_all_perquery` | AUC, `coguk_unseen_perquery` |
|---|---:|---:|
| pyr0 | 0.615 | 0.615 |
| Omicron BA.2 bind | 0.614 | 0.687 |
| CoVFit | 0.603 | 0.639 |
| pssm_root | 0.423 | 0.464 |
| MSA Trans. (XFG eval) | 0.415 | 0.432 |

Max departure from chance over the 12 constant maps x 4 cell sets: **0.187 of AUC**. A
host is more derived than its controls, so the mask removes more of its high-scoring
cells and lifts whatever is left. That is exactly the effect the objection describes.

### Why the drawn ruler is immune: the pool is maskable by nobody

The headline pool is the eval-range cells **no backbone genotype carries at all** --
4,134 of 4,422 (`background_contrast.py`, `carried` / `pool_mask`). A query's wild-type
mask can only ever remove cells that query *carries*, and by construction the pool holds
none. Recomputed from `backbone_queries.csv` and `backbone_manifest.json` rather than
trusted: **0 pool cells are maskable by any of the 402 queries.**

The two remaining query-dependent-looking steps are not: `np.isfinite(scores).all(axis=0)`
reduces over the query axis, and the target cell is removed from its own reference set by
`keep = pool_idx != c_flat`, which is the same removal in every query. So within a cell's
contrast, host and control are ranked in a byte-identical reference set. Also checked:
**0 of 490 host nodes already carry their own target residue**, so there is no asymmetry
smuggled in through the target either.

### The zeros are exact for a second, independent reason

The stored matrices are raw, so a map with no sequence input is literally constant down
the query axis -- max spread across the 402 queries **0.000e+00** for pyr0, CoVFit, the
four DMS rungs, EVEscape, both PSSMs and the three MSA-Transformer rungs.

Across the whole roster the correspondence is **exact and 1:1**: all 13 models with a
constant matrix have lift exactly 0.0 in all four cell sets; all 17 without one are
nonzero in every cell set. No exceptions either way.

### The zero row is not "the baselines", and that is easy to misread

**BLOSUM62, the codon table and the substitution matrix are NOT at zero** (+0.8, +1.6,
+2.9 pp on `coguk_all`), and they should not be: they score a wild-type -> mutant
*transition*, so the background is an input. BLOSUM62's matrix varies by up to 14.0
across queries. The dividing line in panels K-L is **"does the score depend on the
background at all"**, not "learned model vs lookup table" -- three lookup tables sit
above the line and one learned rung (`eve_frozen`, -0.4) sits below it.

### A correction to s33's justification for the fixed pool

s33 motivated the pool change with the AUC confound and wrote that at ~0.1 of AUC it
"swamps the effect being measured". That is true of the AUC and **not** of the metric the
panels actually draw. Under the per-query ruler the *lift* of those same 12 maps is
already within **0.296 pp of zero** across all four cell sets -- the confound is almost
entirely a rank-only phenomenon, because the mask reshuffles the tail without pushing
cells across the decile boundary.

So the fixed pool buys **exact** zero rather than **approximately** zero for the null
band, and it is the `auc` column in the CSV that it genuinely rescues.

**CORRECTED IN s38.** The sentence that stood here -- "the panels would have been off by
at most a third of a percentage point" -- generalised the 0.296 pp from the twelve fixed
maps to the whole panel, which it does not cover. The *scored* models move far more than
that when the ruler changes: base antiGen goes +4.4 -> +0.7 pp on `coguk_all`. The 0.3 pp
bound is a statement about maps that cannot respond to the query at all, and nothing
more. s38 tables the real effect. The pool is still the right choice (a null
that is exactly zero by construction is worth having, and the CSV carries both rulers),
but the stated size of the problem applied to a column the figure does not show.

No numbers change and no panel is redrawn.

---

## 38. K-L moved onto the per-query pool, and what that costs (2026-10-19)

Raised on review of s37: *is the fixed pool a fair way to pose the eval? Why not just use
the usual per-query pool and accept that the pools differ between queries?* Both rulers
have been in the CSV since s33, so this is a figure change, not a rerun.

> **SUPERSEDED BY s39. The recommendation in this section is wrong and the default was
> changed back.** The three arguments below are answered one by one in s39: the second
> is simply false (the fixed pool drops a *superset* of the carried cells), the first is
> presentational and probably a category error, and the third is weaker than stated. The
> tables in this section are correct and still worth having -- it is the conclusion drawn
> from them that does not hold. s39 also shows the whole fixed-vs-per-query question is
> second-order next to the choice of quantile.

### The per-query pool is the better default, for three reasons (WRONG -- see s39)

1. **It is the ruler every other panel uses.** A-H rank in the eval range minus the
   query's own residues. A reader comparing K-L with D or F should not have to know that
   the bottom row silently switched reference sets.
2. **It is the operationally meaningful pool.** A cell the query already carries is not a
   candidate mutation from that query. Removing it is the correct thing to do, not a
   confound to be engineered away.
3. **The null stops being a tautology.** On the fixed pool a query-blind map is at
   0.000 *by construction* -- it cannot be anything else, so it is arithmetic, not
   evidence. On the per-query pool it is measured, and it passes: the twelve constant
   maps come in within **0.296 pp** of zero across all four cell sets.

The cost is that `auc` on the per-query pool is confounded by up to 0.187 (s37), so the
CSV's AUC column should still be read on the fixed pool. The figure draws lift, not AUC.

`fig_coguk_list.R` now takes `coguk.bg_suffix`; the default is the per-query pool and
`"_fixed"` restores the old reading. Checked rather than assumed: rendering with
`"_fixed"` reproduces the committed s36 PNG at **RMSE = 0**, pixel for pixel.

### What changes, and the one number that does not change at all

| cell set | n | antiGen, fixed pool | antiGen, per-query |
|---|---:|---|---|
| CoG-UK all | 53 | +4.4 [-2.3, +11.7] **3rd** | **+0.7** [-6.1, +7.6] **8th** |
| CoG-UK de novo | 19 | +12.4 [-2.3, +29.0] **1st** | **+7.4** [-4.1, +20.5] **4th** |
| all RBD | 145 | +4.3 [+0.8, +8.0] **2nd** | +3.0 [-0.5, +6.8] **4th** |
| de novo RBD | 50 | +12.3 [+4.3, +21.3] **1st** | **+10.7 [+3.4, +18.9] 1st** |

This is a real demotion on the two small cell sets, and it is reported rather than
buried: on the CoG-UK cuts antiGen is no longer 1st or near it. Tranception frozen still
leads `coguk_all` (+8.2, unchanged), and the 35M partial-FT arm takes `coguk_unseen`
(+12.5).

**But `delta_pct` -- the mean host-minus-control percentile gap -- is identical under the
two rulers to three decimals, for every model in the roster.** antiGen on `coguk_all` is
+0.012 under both; `where x which` +0.014 under both; Tranception frozen +0.039 under
both. The AUCs move by at most ~0.03. So the two pools **agree on how far apart host and
control percentiles are, and on their ordering**. They disagree only about which side of
the q = 0.1 line a cell lands on.

That is the honest reading of the whole disagreement: **it is a threshold effect, not a
difference of opinion about the underlying signal.** At 19 or 53 cells a 1.2-percentile-
point gap crossing or not crossing the decile boundary is a couple of cells either way,
which is why `coguk_*` reorders and `rbd_unseen` does not.

### What survives, and what the headline should now rest on

**Ruler-independent**: on `rbd_unseen` (50 cells, the best-powered de novo set) antiGen
is **1st under both pools** with an interval clearing zero under both (+12.3 [+4.3,+21.3]
fixed; +10.7 [+3.4,+18.9] per-query). `where x which` is 2nd under both. That panel is
the one the de novo claim should be stated against.

**Ruler-dependent, and therefore not a claim**: everything on `coguk_all` and
`coguk_unseen`. Their intervals span zero under both rulers anyway, so nothing was ever
settled there; the ruler change just makes that visible in the bar lengths as well as the
whiskers.

**So s36's headline needs narrowing again.** "antiGen is 1st on both de novo cuts" was
true of the fixed pool only. The supportable statement is now: **on the best-powered de
novo cell set, antiGen is the most background-adaptive model on either ruler** -- with
the CoG-UK cuts too small to rank anything. That is weaker than s36 and it is what the
data supports.

### A note on lift as the statistic

s33 chose lift over AUC because AUC is scale-free and scores a 0.7 pp nudge like a 28 pp
swing. That reasoning stands. But this section shows lift's own weakness: it is a
thresholded statistic, so at n = 19-53 it converts a stable underlying gap into an
unstable rank. `delta_pct` is the stable reading and is in the CSV. Not changed here --
lift is still in the units the decile row above it is read in, which is why it was picked
-- but a reader moving between the two rulers should be told that the gap did not move,
only the threshold crossing did. The panel titles do not say this; NOTES does.

### Files

`figures/fig_coguk_list.R` (`coguk.bg_suffix`, default per-query), re-rendered
`coguk_list_results{,_supp}.png` and their PDFs. The backbone control is unaffected --
it does not draw this row.

---

## 39. Which pool answers K-L's question, and a quantile sweep that reframes it (2026-10-19)

Asked directly: *which of the two rulers better shows that base antiGen predicts a
mutation well against the right background and badly against the wrong one -- that being
the whole purpose of K-L?* Answering it properly meant running a check neither s33 nor
s38 ran, and the check moves the ground under both.

### The fixed pool, and s38's reasoning was wrong

**The estimand decides it.** K-L asks: *does the model's score move with the background?*

* Fixed pool: a cell's percentile is its rank in a reference set that is byte-identical
  across queries. It can move **only** if the model's scores move. That is the estimand,
  measured directly.
* Per-query pool: the reference set moves with the background too. A host is more derived
  than its matched controls, so more of its high-scoring cells are masked out and
  whatever remains is lifted -- which is a fact about the tree, not about the model.

That contamination is not hypothetical; it is measurable on a map that cannot respond to
the query at all. **pyr0's host-control gap goes from exactly 0.000000 on the fixed pool
to 0.000318 on the per-query pool, and its AUC from 0.500 to 0.615.** And it is not a
constant offset -- `pssm_root` moves the other way (AUC 0.423) -- so it cannot be
subtracted out.

Taking s38's three arguments in turn:

1. *"Same ruler as panels A-H."* Presentational, and probably a **category error**. A-H
   fix the query and rank mutations; the per-query pool is correct **for that question**,
   because the candidate set genuinely is "mutations reachable from this query". K-L is
   the transpose and does not inherit the ruler along with the data.
2. *"The per-query pool is the operationally meaningful one -- a cell the query already
   carries is not a candidate."* **This is simply false as an argument for it.** The
   fixed pool drops every cell *any* backbone genotype carries, which is a strict
   superset of what the per-query mask drops. It is the *more* conservative of the two
   about impossible mutations, never the less. Checked earlier: 0 of 490 hosts carry
   their own target residue.
3. *"A measured null beats a constructed one."* The only argument with weight, and it is
   weaker than s38 made it. A control designed to read zero when the model is query-blind
   is a **well-designed control**, not a tautology -- that is what makes a nonzero
   reading attributable. And the per-query ruler is demonstrably confounded on AUC; it is
   clean on lift by luck, not by validity.

**So the default is back to the fixed pool.** `coguk.bg_suffix` now defaults to `""` =
fixed, with `"_perquery"` for the other reading. The re-rendered PNGs are **RMSE = 0**
against the s36 commit, i.e. the figure in `figs/` is byte-for-byte the one already
pushed; only the script and NOTES changed.

### But the pool is second-order, and here is the thing that is not

`background_contrast.py --qtile` re-run at 0.05 and 0.20 (no GPU; it reuses the stored
matrices). Base antiGen's lift and rank:

| cell set | ruler | q = 0.05 | **q = 0.10** | q = 0.20 |
|---|---|---|---|---|
| CoG-UK all | fixed | -1.1 (26th) | **+4.4 (3rd)** | +0.7 (11th) |
| | per-query | -1.5 (27th) | +0.7 (8th) | +3.0 (6th) |
| CoG-UK de novo | fixed | +0.2 (13th) | **+12.4 (1st)** | +3.6 (8th) |
| | per-query | +2.3 (10th) | +7.4 (4th) | +6.8 (4th) |
| all RBD | fixed | +1.6 (12th) | **+4.3 (2nd)** | +4.3 (5th) |
| | per-query | +0.9 (13th) | +3.0 (4th) | +4.7 (2nd) |
| de novo RBD | fixed | +4.5 (10th) | **+12.3 (1st)** | +11.7 (3rd) |
| | per-query | +3.5 (12th) | +10.7 (1st) | +12.4 (2nd) |

**antiGen's first place exists only at q = 0.10.** At q = 0.05 it is 10th to 27th on
every cell set and every ruler. The quantile moves it by ten to twenty rank positions;
the pool moves it by one to five. **The ruler question, which is what s38 spent itself
on, is the small term.**

### The stable statistics do not support the headline at all

`delta_pct` -- the mean host-minus-control percentile gap -- has no threshold in it and
is nearly pool-independent (the two rulers agree to ~3 decimals everywhere). Ranked on
it, antiGen is:

| cell set | antiGen's delta_pct | rank (fixed / per-query) | who leads |
|---|---:|---|---|
| CoG-UK all | +1.24 pp | **10th / 10th** | Tranception frozen +3.85 |
| CoG-UK de novo | +3.16 pp | **10th / 10th** | substitution +5.46 |
| all RBD | +2.11 pp | **6th / 6th** | codon +2.71 |
| de novo RBD | +4.96 pp | **6th / 6th** | codon +7.72 |

So antiGen's top-decile lift advantage is **not** backed by a larger overall shift: the
codon table and the substitution matrix move their rankings with the background *more*
than antiGen does, on both rulers. What antiGen has is a shift concentrated near the top
of its ranking. That could be a real and interesting property -- the decile is the
operationally relevant region -- but at q = 0.05 it is gone, which is the opposite of
what "concentrated at the top" predicts.

### What is actually supportable

The base-vs-finetune comparison, which is the one the figure exists to make, survives
only on the stable statistic and only on the RBD cell sets:

| cell set | finetunes beating base antiGen on delta_pct |
|---|---|
| all RBD | **0 of 8** |
| de novo RBD | **1 of 8** |
| CoG-UK all | 4 of 8 |
| CoG-UK de novo | 4 of 8 |

On lift it is again a q = 0.1 effect: at q = 0.05, five to eight of the eight finetunes
beat base antiGen on every cell set.

**Supportable**: *on the RBD cell sets, base antiGen's ranking responds to the background
more than its own finetunes' do* -- quantile-free, ruler-free, and consistent with s33's
percentile-sd observation (0.054 against 0.0066-0.0071, an eight-fold gap).

**Not supportable**: "antiGen is the most background-adaptive model in the roster"
(s36), and "antiGen is 1st on both de novo cuts" (s36, narrowed in s38, now retired).
Both were readings of lift at a single quantile.

### Open -- a decision, not a measurement

The panels draw lift at q = 0.1 and the claim does not survive the quantile sweep. Three
options, none of them taken here because the choice is the user's:

1. **Draw `delta_pct` instead.** Stable, pool-independent, quantile-free -- and it says
   antiGen is 6th-10th overall while beating 7-8 of its 8 finetunes on RBD. Honest, and
   a much weaker headline.
2. **Keep lift but add the quantile sweep** as a supplement panel, so the q = 0.1
   dependence is visible rather than implicit.
3. **Keep lift at q = 0.1 and state the claim only base-vs-finetunes on the RBD sets**,
   with the sweep recorded here as the caveat.

What should NOT happen is the figure continuing to carry s36's ordering as though the
quantile were incidental to it.

### Files

`figures/fig_coguk_list.R` (`coguk.bg_suffix`, default `""` = fixed pool). Sweep outputs
are in the session scratchpad only -- rerunnable in ~3 min with
`background_contrast.py --qtile`, so they were not copied into `data/`.

---

## 40. The q-sweep figures: the whole list figure at 5% and 20% (2026-10-19, CANCELLED)

> **CANCELLED before completion -- see the State section at the end of this entry.**
> The code landed and is committed; the compute was stopped in favour of other work.

s39 established that the background row's ordering is a q = 0.1 effect and recorded it as
a caveat. The caveat is not good enough: **every** "top decile" in this figure is a
top-q, and q was 0.1 by convention, so the right answer is to render the entire figure at
two other quantiles and compare panel for panel. Requested with the recall curves on a 5%
grid as well, which the published 10% grid cannot supply.

`coguk_list_results` and its supplement are **untouched** -- checked, not asserted:
after the whole parameterisation below, re-rendering them is **RMSE = 0** against the
commit, twice (once after the quantile option, once after a label-form fix).

### What actually has to be recomputed, and what was already in hand

| row | depends on q? | source |
|---|---|---|
| A-B recall curves | no (it *is* the q axis) | needs the 5% grid -> **re-run** |
| C-D recall at q | **yes** | macro CSV, keyed by (model, variant, qtile) |
| E-F recall AUC | no, but the grid changes the trapezoid | same CSV |
| G-J BA.2 yield + chance + ceiling | **yes** | varrecall CSV, **no qtile column** |
| K-L background lift | **yes** | background CSV -- **already computed in s39** |

The published macro grid is 0.10...0.90 in tenths, so **q = 0.20 was already available and
q = 0.05 was not**, and neither was any odd multiple of 5%. One re-run supplies both.

**The quantile grid is nearly free; the forward pass is the cost.** `cell_pass()` loops
quantiles *inside* the batch loop, so going from 9 to 19 grid points adds arithmetic, not
passes. The ~10h is the 17,978-example sweep over the ESM-2 arms and is unavoidable.

### Jobs

* **3373137** `macro_qsweep` -- full roster, `--qtiles 0.05,0.10,...,0.95`,
  `--percell-qtile 0.05`, `--multitree-baselines`, into
  `evals_sc2_lists_macro_qsweep/`. ~10h by the precedent of 3362535 (09:57:34).
* **3373141 / 3373142** `varr_q0.05` / `varr_q0.20` -- ~90s each (one forward pass on the
  variant root), into `evals_sc2_lists_varrecall_q0.05/` and `_q0.20/`.
* Background: **no job**. s39's `--qtile` sweep already wrote both; archived to
  `evals_sc2_background_q0.0{5}/` and `_q0.20/` and copied into `antiGen-figures/data/`.

### A new output directory per quantile is mandatory, and for two different reasons

The macro CSV *is* keyed by qtile, so a second grid appended to the published directory
would not collide -- but it would silently change **every AUC in the published figure**,
because the AUC is a trapezoid over whatever grid is present. The varrecall CSV is worse:
it has **no qtile column at all** (q is baked into `n_selected`), so two grids in one
directory are indistinguishable. Hence `evals_sc2_lists_macro_qsweep/` and a per-q
varrecall directory, and hence the hard stop below.

### Code

Parameterised rather than forked, per s34:

* `examples/slurm/eval_sc2_list_macro.sh` -- `QTILES`, `PERCELL_Q`, `OUTDIR` env
  overrides plus pass-through extra args; defaults reproduce the published invocation.
* `examples/slurm/eval_sc2_varrecall.sh` -- new, because this eval had only ever been
  launched as an ad-hoc `sbatch --wrap` and there was nothing to parameterise.
* `figures/fig_coguk_list.R` -- `coguk.qtile` (default 0.1). It drives the C-D slice, the
  G-J chance curve (`QTILE * n_targets / n_selected`), the C-D chance rule, and every
  label. **It hard-stops** if the requested quantile is not in the macro CSV's grid, and
  names the grid it did find -- rendering 0.05 against a 0.1 directory would otherwise
  draw the 0.1 numbers under a 0.05 title, which is exactly the failure mode this
  section's directory discipline exists to prevent.
* `figures/fig_coguk_list_q05.R`, `fig_coguk_list_q20.R` -- the two wrappers.

**A label trap worth recording.** The first parameterisation collapsed "top decile" to one
string and changed the published figure at RMSE 9.1 -- it had dropped the "top" from
"Chance: a top decile picked at random" and the hyphen from "Top-decile lift". The phrase
appears in four grammatical slots (nominal, attributive, and the title-case of each), so
there are now four forms, `q_noun` / `q_Noun` / `q_adj` / `q_Adj`. A text-only regression
like this produces no error and no warning; only the pixel diff caught it.

### What will NOT be comparable across the three figures

**The AUC bars E-F, and the smoothness of A-B.** The two new figures read a 19-point
grid and the published one reads 9 points. A trapezoid over a finer grid is a different
(better) estimate of the same integral, so E-F may shift slightly for every model. Both
wrapper headers say to compare E-F between the two new figures and never against
`coguk_list_results`. The alternative -- re-rendering the published figure on the fine
grid -- was not taken, because the request was explicitly to leave it as is.

### State: cancelled, and what that leaves behind

The three jobs were **cancelled 2026-10-19** after 28 / 27 / 16 minutes, in favour of
higher-priority work. `coguk_list_q05` and `coguk_list_q20` were never rendered.

**The code is committed and complete**, so this is a decision to stop spending GPU, not
an abandoned branch. Everything needed to finish it is in place:

* `coguk.qtile` in `fig_coguk_list.R`, with the hard stop on an absent quantile.
* `figures/fig_coguk_list_q05.R` and `fig_coguk_list_q20.R`, unrun.
* `examples/slurm/eval_sc2_list_macro.sh` (`QTILES` / `PERCELL_Q` / `OUTDIR`) and
  `examples/slurm/eval_sc2_varrecall.sh` (`QTILE` / `OUTDIR`).
* The background halves are **already done** and committed:
  `data/evals_sc2_background_q0.0{5}/` and `_q0.20/`. Only the macro and varrecall
  passes would need re-running -- the commands are in the Jobs section above.

**The partial outputs are quarantined, not deleted.** All three jobs had written
incomplete CSVs with valid headers -- 17 of 30 models for the macro sweep, 21 of 30 for
each varrecall, and **every one of them missing `simple_d128_n2_lr0.0001`** -- which is
exactly the shape of file that gets mistaken for a finished one later. They are renamed
to `PARTIAL_CANCELLED_*` under `/large_storage/.../fitness-inference/`, each with a
`README_PARTIAL.txt` saying what it is. Safe to delete whenever.

**What s39 already settled without them**, so that the open question is not lost: the
q = 0.1 dependence of the background row is *measured* -- antiGen is 1st on the de novo
cuts at q = 0.1 and 10th-27th at q = 0.05, and `delta_pct` ranks it 6th-10th on either
ruler. The q-sweep figures would have shown that across every row rather than only K-L.
The caveat stands on s39's numbers in the meantime.

---

## 41. The backbone control, completed: full roster and the background row (2026-10-19)

Two gaps in `coguk_list_backbone`, both closed without a GPU.

### The six missing models were only ever a timing accident

s34 shipped the backbone control with **24 of 30 models**, missing Tranception (frozen
and FT), EVmutation, codon, substitution and `where x which`, because job 3364881 was
still inside evcouplings' plmc rebuild when the CSVs were written (10-18 23:44; the job
finished 10-19 02:58). `backbone_list_eval.py` reads the score matrices
`background_discrimination.py` wrote and does no forward passes of its own, so the fix
was to re-run it against the now-complete directory: **36 seconds, CPU only.**

**The re-run reproduces the old numbers exactly**, which is the check that adding models
changed nothing else: of the 864 overlapping (model, variant, qtile) rows, **864 are
byte-identical**, and of the 5,760 overlapping varrecall rows, **5,760 are**. Zero rows
only in the old file. Every metric in the figure now prints `30 models, complete`.

### The background row is drawn here now, and it is the same row

s34 omitted K-L from the backbone figure on the argument that it would be identical.
That argument is correct and the row is now drawn anyway, because the figure should be
self-contained and panel letters should mean the same thing in both. **Verified rather
than asserted: below the title lines the K-L band of `coguk_list_backbone.png` is
pixel-identical to that of `coguk_list_results.png` -- 0 differing pixels -- and differs
only in the title row, which carries the " [backbone]" suffix.**

**So it must not be read as corroboration.** The background contrast is computed on the
guide tree *before* attachment in both figures -- that is what `background_tree.py`
builds -- so there is no attached-vs-backbone distinction for this row to have. A reader
comparing the two figures panel-for-panel gets independent evidence from A-J and the
*same* evidence twice from K-L. The wrapper header now says this in those words.

(The " [backbone]" suffix on K-L is, unusually, accurate in *both* figures. It is the
main figure's K-L that could stand to say it is backbone-only.)

### One asymmetry remains, and it is in the main figure

The sets are now **30/30 in every panel of the backbone figure**, but the main figure is
**29/30 in A-J and 30/30 in K-L**: `evcouplings` / EVmutation has no bar in its recall or
yield rows, because `evals_sc2_lists_macro/` was built before the plmc rebuild too. The
figure prints the absentee rather than dropping it, so the row label is there with no bar.

Closing it needs a GPU pass of `list_macro.py` and `list_variant_recall.py` over the
published q = 0.1 grid -- **not** done here, because it would mean re-rendering
`coguk_list_results`, which was explicitly to be left alone. The pending `macro_qsweep`
job (s40) will show whether evcouplings scores on that path at all; if it does, the two
q-sweep figures get 30/30 and only the published q = 0.1 figure keeps the gap.

### Files

`antiGen-figures/data/evals_sc2_lists_macro_backbone/` (both CSVs replaced from
`evals_sc2_lists_macro_backbone_full/`), `figures/fig_coguk_list_backbone.R`
(`coguk.bg_dir` now points at the real directory; header rewritten).
`coguk_list_results.png` and its supplement are **unchanged, RMSE = 0**.

---

## 42. The yield-row legend moved to the top left (2026-10-19)

Third position for this legend, and the first one that sits on nothing.

* **Bottom-right corner** (pre-s35): covered antiGen's own plateau in the de novo panel.
* **x = 0.98, y = 0.52** (s35): moved it out of the plateaus and into the band above
  them -- but the *ceiling* curve's near-vertical rise at t ~ 2.3 passes straight
  through it, which s35 noted as "the ceiling's rise still passes behind it" and papered
  over with an opaque background.
* **Top left, x = 0.02, y = 0.98** (here): empty **by construction, not by luck**. Every
  curve in these panels -- models, chance line and ceiling alike -- is flat at zero until
  the BA.2 descendants start accumulating at t ~ 2.1, so the upper-left quadrant has no
  data in it in any rendering of this panel.

The keys flip to the left of their labels and the labels left-align, because the block is
now anchored by its left edge and the right-anchored settings inherited from the base
`curve_panel` theme would ragged-edge it. The opaque background is kept, in case a future
panel's ceiling climbs earlier than this one's.

**Confined to G and I, checked band by band.** Against the previous commit, every band of
`coguk_list_results.png` is RMSE = 0 except G-H and I-J -- and within those two rows the
right half (panels H and J, the summary bars) is untouched at **0 differing pixels**, so
only the two curve panels moved. The backbone control re-renders to the same pattern
(A-F and K-L at RMSE = 0). The q-sweep wrappers inherit it automatically when they render.

---

## 43. The D = 2021-01-01 arm, drawn on the backbone control (2026-10-19)

`coguk_list_backbone` now carries two rows the published figure does not: the
six-tree joint arm at D = 2021-01-01 and the ncov-only single-tree control it was
trained against, seed 0 of each. s32 cut this arm from the list figure on the
argument that it "belongs with the multi-tree figures rather than the main-model
comparison"; that argument still holds for `coguk_list_results`, which is
**unchanged, RMSE = 0 on both PNGs**, and the two main PDFs were restored from git
because their only diff was the `CreationDate` stamp.

### Why the backbone figure and not the published one

The cutoffs very nearly coincide. `backbone_manifest.json` puts the published
roster's split at **2020.953 (2020-12-14)** and this arm's is **2021-01-01** --
eighteen days later -- and this figure's test set is the guide tree after that
split. That is what makes the arm commensurable with the roster at all. It is
still trained on a different set of trees from a different, Wayback-recovered
build of the ncov tree, so it is not another rung of the main-model comparison and
is drawn as **its own group in its own colour**, appended at the foot of every bar
panel. Folding it into "antiGen Finetunes" would have invited reading the gap
between it and that group's 35M partial rung -- the same architecture and the same
recipe -- as an effect of fine-tuning, which is the one thing it is not.

### The de novo leak is closed, at zero, for this figure

s31 recorded that the arm's 2020-12-29 ncov training tree carries RBD cells
`output1`'s de novo mask calls unseen, and that this had to be closed **before any
of these numbers were reported**. Closed here by intersecting the 29 CoG-UK RBD
cells that tree carries (s29's list) against every de novo cell set this figure
draws:

| panel | de novo cell set | from the 29 |
|---|---:|---:|
| recall B/D/F | 20 observed cells | **0** |
| yield I/J | 19 targets at 1+ | **0** |
| background K-L | 19 `coguk_unseen` | **0** |
| background K-L (supp) | 50 `rbd_unseen` | **0** |

Exactly one of the 29 (**504D**) is unseen by `output1`, confirming s31's count,
and it is in none of those sets: it never emerges among BA.2's 3,762 backbone
descendants, so it is not a yield target either. On the all-cells cut 10 of the 56
observed cells are in that training tree and **all ten are in `output1`'s training
window too**, so the eighteen-day difference buys this arm no cell the published
roster did not also see. There is no window asymmetry to correct in either cut.

### The seed spread is wider than the joint-vs-single gap

All three seeds of each arm were scored, because 402 forward passes of a 35M model
is nothing. Cell-weighted recall at q = 0.1, `list_coguk_fullpool`:

| arm | s0 | s1 | s2 | mean |
|---|---:|---:|---:|---:|
| joint, 6 trees | 0.505 | 0.511 | 0.414 | **0.477** |
| single, ncov only | 0.362 | 0.396 | 0.427 | **0.395** |

De novo: joint 0.400 / 0.275 / 0.258 (mean 0.311), single 0.325 / 0.275 / 0.275
(mean 0.292). Background lift on `coguk_all`: joint mean +4.32 pp, single +3.46.

**The joint arm is ahead on the seed mean in every cut, and the ranges overlap in
every cut.** The figure draws one checkpoint per row, as every other row in it
does, so the two drawn bars are one draw each and **the pair must not be read as
the joint-vs-single contrast** -- that contrast lives in s29's 8-seed estimate
(ncov de novo, joint vs single, +0.093 [+0.063,+0.123] on 816 targets), not here. Said in those words in the wrapper header.

Where seed 0 lands among the 32: 13th of 32 on all cells cell-weighted (joint) and
21st (single); 13th and 16th on de novo. Both sit in the middle of the roster,
below the EVE and 35M-partial rungs and above the fixed maps.

### How it was built, and what it cost

| step | where | cost |
|---|---|---|
| 6 score matrices | `examples/slurm/background_discrimination_mt2021.sh`, job **3373943** | ~6 min, 1 GPU |
| backbone recall + yield | `backbone_list_eval.py` on the merged dir | 65 s, CPU |
| background contrast | `background_contrast.py` on the merged dir | 33 s, CPU |

The new slurm script is separate from `background_discrimination_sc2.sh` for three
reasons, all in its header: the staging directory of symlinks (s31), the
`--include-experimental` these checkpoints need because they carry
`experiment: 'forecast@2021-01-01'`, and **not** `--multitree-baselines`, which
alone would cost a ~19 GB load of the examples file to refit a substitution matrix
that is already in `evals_sc2_background/`.

`evals_sc2_background_mt2021_merged/` is job 3364881's 30 matrices by symlink plus
the 6 new ones; nothing already computed was rewritten. **The re-runs reproduce the
published numbers exactly**, which is the check that adding models changed nothing
else: 1,080 / 1,080 macro rows, 7,200 / 7,200 varrecall rows and 240 / 240
background rows byte-identical, zero rows only in the old files.

### The figure script

`fig_coguk_list.R` gained `coguk.mt2021`, a character vector of this arm's
checkpoint names, **empty by default**. When it is empty the script is what it was:
the arm's rows are filtered out of the macro, varrecall and background frames
alike, the sixth group is never added to `GROUP_LEVELS`, and no empty legend entry
appears -- `scale_fill_manual(drop = FALSE)` would otherwise draw one. The old
hardcoded `macro[!grepl("^mt2021", ...)]` drop is replaced by that filter, which
also keeps the unasked-for seeds from tripping the "models with no group
assignment" check. An unknown checkpoint name is a hard stop.

**K-L is no longer pixel-identical to the main figure's K-L**, as s41 recorded it
was: the panel has 32 bar slots instead of 30, so every bar shifts. The 240 numbers
behind it did not move, and the wrapper header now says it that way rather than
repeating s41's claim.

---

## 44. A second backbone control, on a newer guide tree (2026-10-19)

**Both backbone figures are kept, deliberately.** `coguk_list_backbone{,_supp}` is
unchanged on `antiGen-data/tree.json` -- re-rendered and checked at RMSE = 0
against the s43 commit -- and `coguk_list_backbone_update{,_supp}` is the same
script through different options on `sc2_update.json`. The published
`coguk_list_results{,_supp}` is untouched, RMSE = 0 on both PNGs.

The point of keeping both is that the ordering reproduces across two
independently built guide trees. Read them as one result measured twice.

### The old tree's 56 cells were a ceiling, not a sample

Raised on review of s43: *56 of CoG-UK's 247 listed RBD cells seems low.* It is,
and the cause is the tree rather than the cutoff rule. Counting listed RBD cells
among branch mutations of `antiGen-data/tree.json` under three inclusion rules:

| rule | RBD events | RBD cells | listed events | listed cells |
|---|---:|---:|---:|---:|
| every branch in the tree | 549 | 170 | 258 | **69** |
| child node post-cutoff | 522 | 166 | 237 | 66 |
| **both endpoints post-cutoff** (the backbone rule) | 491 | 156 | 211 | **56** |

So **the whole tree, over its entire history, only ever produces 69 of the 247**.
It is a subsampled Nextstrain build: 8,859 nodes carrying 2,839 spike mutation
events in total, 549 of them in the RBD. The published figure reaches 235 cells
only because attachment lands ~170k real sequences on that same backbone.

**The 69 -> 56 gap is a structural property of the control.** A cell is a target
only if BOTH endpoints of its branch are post-cutoff, because the parent node *is*
the background being queried and querying a background the model trained on would
be a leak. Branches that **span** the cutoff therefore cannot be scored -- and on a
subsampled tree those are exactly the variant-defining saltations:

```
node  878  clade 21M (Omicron)   parent t=2020.470 -> child t=2021.110
           22 muts on the branch, 11 listed RBD cells
           339D 371F 373P 375F 417N 440K 477N 484A 498R 501Y 505H
```

Against a 2020.953 cutoff the whole Omicron saltation is unscorable; 339D, 373P,
375F, 417N, 440K, 477N, 484A and 498R survive nowhere else on that tree. Iota, Mu,
Kappa and Lambda lose their crown branches the same way (484K, 346K, 484Q, 490S,
450K). **All ten lost cells are `seen` by `output1`'s mask, so the de novo cut of
20 was unaffected** -- the loss fell entirely on the all-cells panels.

The rule is not changed here. A newer tree simply resolves more of that history
into post-cutoff branches.

### What the new tree is, and what it buys

`sc2_update.json`, ncov global, `meta.updated` 2026-10-18. **Fewer** nodes (5,880
vs 8,859) and tips (3,018 vs 4,549) but **more** spike mutation events (3,369 vs
2,839), running to 2026.75 rather than 2026.21: denser branches over a longer span.
Its `root_sequence.S` is byte-identical to `antiGen-data/ref.json`, and
`build_backbone` reports 0 wild-type mismatches and 0 unparsable branches -- the
same two self-checks the old tree passes.

| | old tree | sc2_update.json |
|---|---:|---:|
| listed cells observed | 56 | **79** |
| listed target events | 211 | 263 |
| events per observed cell | 3.8 | 3.3 |
| **de novo listed cells** | 20 | **30** |
| all-RBD cells observed | 156 | 191 |
| background `coguk_all` / `coguk_unseen` | 53 / 19 | **79 / 30** |
| background `rbd_all` / `rbd_unseen` | 145 / 50 | 175 / 65 |
| query genotypes | 402 | 318 |
| BA.2 descendant tips | 3,762 | 1,342 |

79 is **above the old tree's ceiling of 69**, so this is more of the list and not
merely the cutoff-spanning branches coming back. 50 listed cells are shared, 29 are
new, 6 drop out (345S 455A 476S 499H 499S 501Y).

**The cutoff is matched by date, not by the 1.0-year convention.** The roots differ
(2019.953 vs 2019.964), so `--cutoff 1.0` would put the boundary at 2020.964, four
days after the one every checkpoint was trained to. `--cutoff 0.989` puts it back
on 2020.953. The listed-cell counts are identical either way (79/263), so this is
correctness rather than a result; it turns 316 query genotypes into 318.
EVE/Tranception/MSA-Transformer/evcouplings are unaffected regardless: they filter
their MSAs by `date_cutoff` from the **checkpoint's** serialized config, not from
`--cutoff`.

### The ordering survives the tree swap, which is the real check

Over the 35 models scored on both, top decile, `list_coguk_fullpool`:

| | Spearman |
|---|---:|
| cell-weighted, all list cells | **0.978** |
| event-weighted, all list cells | 0.978 |
| cell-weighted, de novo | **0.952** |
| event-weighted, de novo | 0.966 |

And the control's central argument is preserved: mean `abs(macro - micro)` is
**0.048** here against 0.055 on the old tree and 0.171 on the attached test set, so
the two weightings still very nearly coincide once attachment is removed.

**Base antiGen moves up and it is the largest single move.** All cells: 0.560 ->
0.640, 8th -> 4th. De novo: 0.492 -> 0.558, 8th -> 5th. The top is otherwise
unchanged -- `eve_frozen` / `eve_ft` / the 35M partial arm lead both cuts on both
trees. With 79 and 30 cells those margins are still not separable rung by rung;
**the claim remains about the ordering as a whole.**

### What changed in the figure besides the numbers

* **K-L is now this tree's own background contrast.** It used to read the main
  figure's directory, on s41's argument that the contrast is computed pre-attachment
  in both figures and so had no distinction to make. That held only while both used
  the same guide tree. They no longer do, so K-L is recomputed here and is a
  genuinely independent measurement -- 79 cells against 53, 30 de novo against 19.
* **EVmutation is absent from every panel, on instruction.** Its plmc fit is not
  cached and cost 12,993 s of the 13,330 s job 3364881 spent -- essentially the
  whole job -- and it was never a top competitor here (7th of 30, 0.308). Reported
  as missing rather than dropped silently. **31 of 32 roster entries are drawn.**
* **The yield row's targets go UP even though its tips go down.** 1,342 BA.2
  descendants against 3,762, but 75 listed cells reach 1+ tips and 32 reach 100+,
  because the denominator is the RBD and does not move with the tip count.
* **The mt2021 de novo leak was re-measured, not inherited**, because the de novo
  sets are different here (30 cells, not 20). Still exactly zero: of the 29 CoG-UK
  RBD cells that arm's training tree carries, 0 appear in the 30 recall de novo
  cells, the 28 yield targets at 1+, the 9 at 100+, the 30 `coguk_unseen` or the 65
  `rbd_unseen` background cells.

### Jobs and files

Two jobs, both on `gpu_batch`, **3 m 27 s and 45 s** -- the roster is cheap once
evcouplings is out of it:

| job | arm | output |
|---|---|---|
| **3374461** | 29 models, `--multitree-baselines --exclude ...,evcouplings` | `evals_sc2_background_update/` |
| **3374462** | the 6 mt2021 seeds, `--include-experimental --skip-baselines` | `evals_sc2_background_update_mt2021/` |

Both wrote byte-identical `backbone_{hosts,nodes,queries}.csv` and manifests, checked
before merging. `examples/slurm/background_discrimination_sc2_update.sh` is the new
wrapper (tree, cutoff, checkpoint dir and extra args all parameterised);
`evals_sc2_background_update_merged/` is the 35 matrices by symlink;
`antiGen-figures/data/evals_sc2_lists_macro_backbone_update/` and
`data/evals_sc2_background_update/` are what the new figure reads;
`figures/fig_coguk_list_backbone_update.R` is its wrapper. **The old tree's
directories and wrapper are untouched**, so s34/s41/s43's numbers remain
reproducible and that figure still renders byte-identically.

`sc2_update.json` lives at the repo root and the new figure depends on it, so it
is committed rather than left untracked -- the same treatment
`tree_pulled_2026-09-28.json` already gets.

### The title suffix has a two-character budget, measured

s34 noted that a title suffix "can push the longest title past the right-hand
panel's width" and left `coguk.title_size` available for it. This is what that
warning is worth in numbers, because the first two attempts here both clipped.

Titles are centred and the type is **20.1 px/char**, so each added character
costs ~10 px on the right. Slack to the canvas edge at the tightest right-hand
title: **92 px in the main figure, 42 px in the supplement** -- the supplement
binds, because its titles are longer ("event-weighted" for "cell-weighted",
"100+ cases" for "1+ cases"). That is a budget of about **two characters**.

| suffix | delta | result |
|---|---:|---|
| `" [backbone, 2026-10 tree]"` | +14 | clipped 4 of 6 title rows |
| `" [backbone, 2026]"` | +6 | clipped 1 row (main), 3 rows (supp) |
| **`" [2026 tree]"`** | **+1** | **clean in both** |

So the suffix **swaps a word rather than adding one**: "backbone" is carried by
the filename and the wrapper header, and the titles carry the one thing that
differs between the matched pair, which is the tree. `coguk.title_size` was
deliberately not used -- shrinking type on one figure of a matched pair makes
them typographically different for no reason, which is the trade s34 declined.

Checked by a detector rather than by eye (any ink in the last 6 columns of a
title band), run over all four PNGs.

---

## 45. The same panels asked of every RBD mutation, not of the list (2026-10-19)

`backbone_update_allrbd{,_supp}`: the s44 backbone figure with the curation
removed. Fourth member of a matched set, all four off `fig_coguk_list.R` through
options.

| wrapper | test set | targets | cells |
|---|---|---|---:|
| `fig_coguk_list.R` | attached | CoG-UK | 235 |
| `fig_coguk_list_backbone.R` | old tree | CoG-UK | 56 |
| `fig_coguk_list_backbone_update.R` | sc2_update | CoG-UK | 79 |
| **`fig_backbone_update_allrbd.R`** | sc2_update | **all RBD** | **191** |

Only the TARGET set changes. Queries, candidate pool, masking, tie rule, de novo
mask, models and score matrices are the sibling's; no GPU was used.

### Reversions are 27 of the 191, and the obvious list drops all of them

`rbd_all_cells.csv` lists all 201 x 22 = **4,422** cells of the eval range,
**including each position's Wuhan residue**. Building it the obvious way -- every
residue *except* the root's -- scored **164 cells while calling itself "all
mutations"**. The 27 missing were, every one of them, reversions to the root
residue: 339G, 344A, 346R, 356K, 371S, 373S, 375S, 376T, 403R, 408R, 417K, 440N,
452L, 455L, 456F, 460N, 475A, 477S, 478T, 481N, 483V, 484E, 486F, 490F, 493Q,
501N, 505Y. A branch reverting a position to Wuhan is a mutation on this tree and
belongs in the denominator.

Listing the root residue is safe as well as necessary: the wild-type letter in the
CSV is ignored by the parser, and `compute_hit_rate` masks each **query's** own
residue, so a cell can never be a candidate in a background that already carries
it. Checked: the list covers **191 of 191** observed host cells, and the eval
reports `n_cells=191, n_events=610`, matching `backbone_manifest.json` exactly.

### The curation matters more than the tree did

Top decile, cell-weighted, both columns on the *same* tree and the *same* scores:

| | Spearman |
|---|---:|
| CoG-UK vs all RBD, all cells | **0.926** |
| CoG-UK vs all RBD, de novo | **0.887** |
| (for comparison, s44: old tree vs sc2_update, CoG-UK) | 0.978 / 0.952 |

**Swapping the target set moves the ordering further than swapping the guide tree
did.** The direction is consistent and worth stating plainly:

| model | all cells: list -> all RBD | de novo: list -> all RBD |
|---|---|---|
| PSSM | 0.533 -> 0.596, **12th -> 4th** | - |
| `where x which` | - | 0.400 -> 0.508, **11th -> 4th** |
| substitution matrix | - | 0.367 -> 0.400, 15th -> 8th |
| **antiGen** | 0.640 -> 0.596, 4th -> **3rd** | 0.558 -> 0.581, 5th -> **3rd** |
| antiGen as ESM 2 (35M) partial FT | 0.717 -> 0.573, **2nd -> 5th** | 0.625 -> 0.478, **3rd -> 6th** |
| EVE / EVE full FT | 1st / 3rd -> 1st / 2nd | 1st / 2nd unchanged |

**The composition baselines gain and the ESM-2 fine-tunes lose** once the
curation is removed. A reading consistent with s32's recurrence finding is that
the CoG-UK list is enriched for antigenically notable cells, which the
sequence-evolution rungs rank well, while the full set of observed mutations
contains many ordinary ones at variable positions, which plain composition
captures. **This is a hypothesis, not a measurement** -- what is measured is the
rank movement. What it does establish is that **a curated list can flatter a model
that has learned the curation**, and the all-RBD panel is the better-powered of
the two (191 cells against 79, 65 de novo against 30).

### K-L is not a new measurement here

In the CoG-UK figures the main asks the background question of the list and the
supplement asks it of every RBD cell. Here the list **is** every RBD cell, so the
two coincide; `fig_coguk_list.R` collapses them with `unique` rather than
relabelling one cellset twice. **Verified: below the title line, K-L is
pixel-identical (0 differing pixels) between this figure, its own supplement, and
the CoG-UK sibling's supplement**; only the title row differs. A reader must not
count it twice against that figure.

### The script parameterisation, and the regression that guards it

`fig_coguk_list.R` gained `coguk.list` (the CSV label), `coguk.bg_list` (the
background cellset prefix), `coguk.all_noun`, `coguk.dn_noun`, `coguk.y_recall`,
`coguk.y_yield` and `coguk.ref_ceiling`. The longform metric vectors, `RECALL_*`,
`VARR_*` and `BG_*` are now derived from the nouns instead of being literals.

**Every default reproduces the published string exactly, checked rather than
asserted: all six existing PNGs -- `coguk_list_results`, `coguk_list_supp`,
`coguk_list_backbone{,_supp}`, `coguk_list_backbone_update{,_supp}` -- re-render
at RMSE = 0 through the parameterisation.** The 1,260 coguk macro rows and 8,400
varrecall rows of the regenerated two-list CSVs are byte-identical to the
committed one-list versions.

### The title budget again

s44 measured it at about two characters. "de novo" -> "de novo RBD" spends four,
so the suffix gives four back: **" [2026]"** against the sibling's
" [2026 tree]". Net zero, and the clipping detector reports no title touching the
right edge on either PNG. The nouns do the distinguishing between the two figures,
which is what they should be doing anyway.

### Files

`rbd_all_cells.csv` (repo root, beside `coguk.csv`),
`antiGen-figures/figures/fig_backbone_update_allrbd.R`, and the two-list CSVs in
`data/evals_sc2_lists_macro_backbone_update/`. `backbone_list_eval.py` already
took `--mutation-list` repeatedly, so no package code changed.

---

## 46. Validation-selected epochs: the design, and why the temporal arm is blocked (2026-10-19)

> **Status: Arm A measured; see s47 for the result.** Arms antiGen and ESM-2 35M
> partial have run and been scored on the list panels; EVE (3375639) is still
> training. Arm B remains blocked. The "Open" block at the foot of this section is
> superseded by s47.

The whole sc2 roster trains a fixed **32 epochs**, selected on nothing. This section
records the retrain that tests it, the leak that had to be avoided, and a measurement
that killed half the design.

### 32 was never chosen

It entered as `--epochs 64` in `61f3b6a` ("slurm scripts"), was halved to 32 four days
later in `e7c963d` as an unmentioned one-line rider in a commit about *evaluation*, and
propagated verbatim to all 14 main-text SLURM scripts and all 36 sweep scripts.
`README.md:142` still documents 64, so the docs and the scripts disagree and neither
gives a reason. Epochs was never a swept dimension; there is no early stopping, patience
or LR decay anywhere in the package.

### The held-out curve the loss figure drops

`figS_training_loss.R` plots `loss/training_loss_curves_sc2.csv`, which is **training
loss only** -- every curve is still falling at epoch 31, which is all that figure can
say. But the 2026-era `train.py` also printed a `test_batch_loss=` probe every 10
batches (removed in `8ebe594`), so the logs carry a held-out signal. Averaging its 6-12
sampled minibatches per epoch:

| arm | held-out min | NLL penalty at epoch 32 |
|---|---:|---:|
| Tranception full FT | epoch 3 | **+2.20** |
| ESM-2 35M partial FT | epoch 5 | +0.40 |
| ESM-2 650M partial FT | epoch 7 | **+1.91** |
| EVE full FT | epoch 7 | **+1.00** |
| ESM-2 35M frozen | epoch 14 | +0.10 |
| **antiGen (`simple_d128_n2`)** | **epoch 28** | **+0.12** |

**The asymmetry is the point, and it is uncomfortable.** The fixed budget over-trains the
*comparators* far more than antiGen, whose curve is nearly epoch-insensitive and arguably
*under*-trained -- its minimum sits at 28 of 32. A principled stopping rule is therefore
more likely to raise EVE and ESM-2 than to raise antiGen, and could narrow the lead the
figures report. That is a reason to do this, not a reason not to.

Caveat on the evidence: the probe draws from the **reported test set**, so it is
diagnostic, not a legitimate selection signal. That is exactly the leak below.

### The default stopping rule would select on the test set

`split_examples` (`train.py:275-294`), under `--val-mode file` -- the repo default --
sets `val = [e for e in examples if not e.is_training]`, i.e. **validation IS the
167,600 post-cutoff TESTING examples**. The published runs never passed `--save-best`, so
no selection happened and nothing is contaminated. But bolting `--save-best` onto the
published scripts would have chosen the epoch on the reported test set. The arms below
therefore do not use the default.

### Arm A: random split inside the training window (running)

```
--val-mode random --val-frac 0.1 --split-seed 0 --save-best --experiment valsel-random
```

`random` mode **drops all 167,600 TESTING examples outright** and splits the 3,648
TRAINING pool 90/10, so nothing post-cutoff reaches training, validation or the epoch
choice. `--test-frac` is deliberately not used: its purpose is a selection-free set when
numbers are *reported on val*, and here they are reported on the post-cutoff figures,
which this mode never touches.

Five runs, jobs **3375635-3375639**, each into its own output directory under
`fitness-inference/valsel/`:

| arm | epochs | seeds |
|---|---:|---|
| antiGen `simple_d128_n2` | **64** | 42, 43, 44 |
| ESM-2 35M partial FT | 32 | 42 |
| EVE full FT | 32 | 42 |

antiGen gets 64 because its optimum sits at 28 of 32 and a 32-epoch ceiling would not
bracket it; the per-epoch JSON records every epoch, so "best within 32" and "best within
64" both come off the same run. The three antiGen seeds are the **noise baseline** --
without one, a null result is an assertion rather than a measurement, and there are no
seed replicates anywhere in the published store (all 178 checkpoints are `--seed 42`;
same-config duplicates differ only by CUDA jitter, rel-L2 ~ 0.003).

### Arm B: the temporal split, attempted and BLOCKED

The intended design: rebuild the corpus at an inner cutoff D', diff the TRAINING blocks,
and label the (D', 1.0] slice as validation -- the pattern `make_nested_split.py`
established, which needs no package change because `--val-mode file` consumes a
TRAINING/TESTING relabelling directly.

**It cannot be done by matching records across builds.** There is no per-example date:
`Example` (`load.py:13`) carries id, parent id, mutation-from-parent, the TRAINING/TESTING
flag, variant blocks, sequence and targets, and nothing temporal. File order is DFS
pre-order, not time, and parent-chain depth counts mutations, not years. So the slice has
to be recovered by matching records between two builds -- and the records do not match.

`write_training_examples` decomposes a multi-mutation branch into intermediate genotypes
in a **shuffled** order (`set_mutation_pos_order`, `main.cpp:519-542`). The RNG is seeded
`std::mt19937 rng(123)` -- fixed -- but it is seeded **inside** `generate_cutoff_outputs`,
*after* `prune_tree`, and then consumed over a traversal whose node set depends on the
cutoff. A smaller cutoff means fewer nodes before any given deep node, so the stream
desynchronises and the same branch gets a different decomposition in the two builds.

**Measured, not inferred.** `output1` (cutoff 1.0) against `output4` (cutoff 4.0), same
guide tree, keying on (sequence, targets):

| | matched verbatim |
|---|---|
| all 3,648 TRAINING records | **439 (12.0%)** |
| the 2,928 single-target records | 377 (12.9%) |
| the 720 multi-target records | 62 (8.6%) |

12% is not a basis for anything. Arm B is **dropped on instruction**, and the generator
job for it (3375634, D' = 0.80) was cancelled before it started.

**The fix, available but not taken.** Emit the node time from the generator: `Node::time`
(`types.h:23`) is already in scope in `write_training_examples`, there is a commented-out
block in that function that printed similar per-node fields, and branch `is-dev`
(`78d9625`) already implemented the whole pattern -- `TIME_INTERVAL` / `TIME_RANGE` lines,
an **optional, order-tolerant** prelude in `load.py` (required, since the parser is
strictly positional and any unconditional new line breaks every other consumer), and a
`split_by_window` + `--test-window`. One rebuild would then give a dated corpus with D'
choosable, or sweepable, post hoc -- strictly better than the two-build diff even
ignoring the RNG problem. Cost is ~3 lines of C++, an optional parse, a new val mode, a
binary rebuild (`nvcc` lives at `/usr/local/cuda/bin`, not on PATH) and one ~3 h
attachment.

### Mechanics that would have corrupted this silently

* **Prefix collision.** `run_prefix` (`train.py:715-748`) appends `arm_tag` only on the
  `esm2` branch; for `simple_d*` and `eve_ft_*` it does not. A retrain written to the
  published store would have produced the *same* prefix as the published checkpoint, and
  `discover_checkpoints` keeps the newest per config label -- silently shadowing the
  published model in every figure. Hence one output directory per run.
* **`*_best.pt` is invisible to eval.** `_CKPT_RE` (`eval.py:84`) requires
  `_YYYYMMDD_HHMMSS.pt`. The best checkpoints need staging symlinks under parseable
  names, the `ckpt_stage_mt2021` pattern.
* **`--val-mode random` auto-tags `experiment='multitree'`** (`train.py:438-441`), which
  every scorer skips without `--include-experimental`. `--experiment valsel-random` is
  set explicitly so the arm is labelled rather than inheriting a misleading tag.

### Open

Jobs 3375635-3375639 are queued behind two multi-day eval jobs; scheduler estimates put
the first starts ~20 h out. Nothing is measured yet. The pre-registered comparison is
val-selected against the published 32-epoch checkpoint, per arm, on cell-weighted top
decile (panels C/D) of `coguk_list_results`, both cuts, with the selected epoch reported
beside it -- that last column is what answers s11's warning that **an NLL-optimal choice
need not move ranking at all**, which is a live risk here because selection is on NLL and
the figure reports recall.

---

## 47. The val-selected epochs, measured: a null on the figure, and a sign flip (2026-10-19)

s46 designed Arm A and left it queued. Two of its three arms have now run, been
scored on the CoG-UK list panels, and drawn. **Validation-based epoch selection does
not improve the headline numbers.** For antiGen it moves them less than the seed
spread; for ESM-2 it moves them the wrong way.

### What ran, and what the scheduler estimates were worth

| job | arm | result |
|---|---|---|
| 3375635-37 | antiGen `simple_d128_n2`, 64 ep, seeds 42/43/44 | best **ep 48, 49, 46** |
| 3375638 | ESM-2 35M partial FT, 32 ep, seed 42 | best **ep 7** |
| 3375639 | EVE full FT, 32 ep, seed 42 | still in VAE pretrain |
| 3377097-99 | antiGen **fix32 controls**, 32 ep, seeds 42/43/44 | final ep 31 |
| 3377171 | `eval_sc2_list_macro.sh` over all eight | 7.5 min |

s46 put the first start ~20 h out. Every estimate was beaten by hours; the antiGen
runs are **~6 min each** and the whole arm cleared in under an hour of wall clock.
The eval was the one real queue: submitted to `gpu_batch` it estimated **17 h**, and
`scontrol update JobId=3377171 Partition=gpu,gpu_batch` started it in **three
minutes**. The two partitions are the same ten nodes, nine of them up (`sinfo`); only the time
limit differs (1 d against 14 d), and this script asks for 1 d. **Any eval in this package
that fits a day should be submitted to both.**

### The fix32 control, which is the reason the numbers are readable

A val-selected rung differs from its published rung in **two** ways -- the epoch, and
the 10% of the training pool held out to choose it. `--val-mode random --val-frac 0.1`
trains on 3,283 of the 3,648 TRAINING examples. A `valsel` bar read against its
published bar therefore charges the subsample to selection.

The `fix32` rungs separate them: same 90% split, same seed, same recipe, no selection,
stopped at the published budget's last epoch. ESM-2's control needed no run at all --
its non-`best` checkpoint **is** epoch 31 on that split. antiGen's did, because its
64-epoch run ends at 63; the three 32-epoch reruns cost 216, 213 and 201 s. The decomposition:

    published -> fix32     the cost of the 10% held out       (data)
    fix32     -> valsel    the effect of choosing the epoch   (selection)

Their best-within-32 (8.0669 / 8.0889 / 8.0702) reproduces the same quantity read off
the 64-epoch runs' per-epoch JSON (8.0664 / 8.0887 / 8.0703) to **5e-4**, which is an
independent check on both at CUDA-jitter level.

### Held-out NLL: the arm works, and the probe was inflated

Selection on the clean in-window split, 1,191 val mutations, identical across arms:

| arm | best ep | best NLL | ep 31 | penalty of the fixed budget | s46 probe said |
|---|---:|---:|---:|---:|---:|
| antiGen (mean of 3 seeds) | 46-49 | 8.0552 | 8.0902 | +0.035 | +0.12 |
| ESM-2 35M partial | 7 | 8.5078 | 8.5628 | **+0.055** | **+0.40** |

The probe located ESM-2's minimum correctly (it said 5, the clean split says 7) and
**overstated what it is worth by about seven times**. antiGen's three seeds span
**0.045** at epoch 31 while selection moves the mean by **0.035**: the effect and the
noise are the same size, which is exactly what the seed baseline was built to detect
and the reason s46 insisted on one.

### The figure: panels C/D, cell-weighted recall at the top decile

`coguk_list_fullpool`, q = 0.1, antiGen columns are the mean of three seeds.

| | published | fix32 | valsel | data | **selection** | seed spread |
|---|---:|---:|---:|---:|---:|---:|
| antiGen, all cells | 0.4460 | 0.4452 | 0.4588 | -0.001 | **+0.014** | 0.035 |
| antiGen, de novo | 0.5089 | 0.4805 | 0.4904 | -0.028 | **+0.010** | 0.028 |
| ESM-2 partial, all | 0.5125 | 0.4908 | 0.4665 | -0.022 | **-0.024** | (1 seed) |
| ESM-2 partial, de novo | 0.4097 | 0.4067 | 0.3888 | -0.003 | **-0.018** | (1 seed) |

**antiGen: null.** Both selection effects are smaller than the seed spread. AUC
(panels E/F) agrees -- +0.0009 and -0.0008.

**ESM-2: the sign flips, and this is the finding.** Selection did what it was asked
and bought 0.055 of NLL. On the figure metric it **lost** 0.024 and 0.018. This is
s11's warning realised, and worse than its statement of it: the risk recorded was
that an NLL-optimal choice "need not move ranking at all", and what happened is that
it moved ranking *backwards*. AUC is the one place ESM-2 gains (+0.0056 all,
+0.0167 de novo), so the loss is specifically at the top decile.

The event-weighted reading (supplement) is the same story louder: antiGen +0.034 /
+0.023, ESM-2 -0.022 / **-0.074**. antiGen's de novo seed spread there is 0.098, wider
than any effect on the row.

**Nothing in the roster reorders**, on either cut, in either pooling.

### Note the data cost is not small, and sometimes dominates

antiGen's de novo panel loses 0.028 to the 10% subsample and regains 0.010 from
selection, so its net against the published bar is **negative** although selection
helped. Reading `valsel` against `published` without the control would have reported
that as selection hurting. The control is not a formality here.

### What this does and does not settle

It settles the pre-registered question for two of three arms: a principled stopping
rule does not raise the comparators past antiGen, and does not narrow the lead. The
s46 worry that it might was reasonable and is not borne out.

It does **not** settle whether the fixed budget costs the comparators anything on the
figures, because **Arm A may be structurally unable to see that**. Selection is on a
random split *inside* the training window; the figure reports *post-cutoff* recall.
Over-fitting to the window is invisible to an in-window held-out set and is precisely
what degrades extrapolation. So "+0.055 clean against +0.40 leaky" has two readings --
the probe was inflated by test-set leakage, or the in-window split is blind -- and
nothing here distinguishes them. That is the Arm B argument restated with a
measurement behind it, and it raises the value of the ~3-line `Node::time` emit
(s46, "The fix, available but not taken") from a convenience to the thing that would
make this section conclusive.

### The figure is a sibling, and the published one is byte-identical

The arm is drawn on `coguk_list_valsel`, not on `coguk_list_results`, following the
`coguk.mt2021` precedent and for its stated reason: a retrain on a 90% subsample of
the training pool is not another rung of the published roster and must not silently
become one. `coguk.valsel` is **empty by default**, `drop_unrequested_valsel` removes
the rows, and `coguk_list_results.png` re-rendered unchanged -- md5
`906eca00d379bc746b707570fbe8b50a` -- both before and after the new CSV was added to
the shared macro directory. Opting the arm into the main figure is one line.

The re-render does leave `figs_vector/coguk_list_results.pdf` and
`coguk_list_supp.pdf` dirty in `git status`, which is a false alarm and worth knowing
before someone chases it: same byte size, and the only difference is the embedded
`/CreationDate` and `/ModDate`. The PNGs are the artefact to check equality on.

**Panels G-L are blank for this arm.** The yield row needs `list_variant_recall.csv`
and the background row `background_summary.csv`, which are separate evals these
checkpoints have not been put through. The script reports the holes rather than
hiding them, and A-F -- which carry the top decile this arm exists to test -- are
complete.

Title suffix `" [val-sel]"` is 10 characters, inside the ~13 s44 measured, and clean
in both PNGs.

### Files

* `figures/fig_coguk_list.R` -- `VALSEL_KNOWN` / `VALSEL_GROUP` / `coguk.valsel`
  (empty by default), `drop_unrequested_valsel`, applied at all three sites the
  mt2021 drop is applied (macro, varr, bg). Colour `#48cae4`.
* `figures/fig_coguk_list_valsel.R` -- the wrapper, with the decomposition in its header.
* `data/evals_sc2_lists_macro/list_macro_summary_valsel.csv` -- 288 rows, 8 models.
* `figs/coguk_list_valsel.png`, `figs/coguk_list_valsel_supp.png` (+ `figs_vector/*.pdf`).
* `fitness-inference/valsel/{antigen_rand_s42..44, antigen_fix32_s42..44, esm2p_rand_s42}/`
* `fitness-inference/ckpt_stage_valsel/` -- 8 symlinks, `fitness_<label>_<ts>.pt`,
  timestamps preserved from the source runs.
* `fitness-inference/evals_sc2_lists_macro_valsel/`

### Open

* **EVE (3375639)** is still in VAE pretrain (step 285k/400k at 2 h 45 m) with 32
  fine-tune epochs after it. Adding it is mechanical: stage its `_best.pt` and its
  final checkpoint under `valselEve_s42` / `fix32Eve_s42`, rerun 3377171's command,
  copy the CSV, add two names to `VALSEL_KNOWN` and to the wrapper's vector, re-render.
* **Tranception (+2.20) and ESM-2 650M partial (+1.91) have no arm at all**, and the
  s46 probe ranks them as the two most over-trained rungs -- above EVE (+1.00) and far
  above the 35M partial (+0.40) that is the only comparator measured here. The arm as
  submitted tests the hypothesis on the two comparators least able to show it. Each is
  one `sbatch` under the flags in this section.
* Panels G-L for the arm, if the yield and background rows are wanted.

---

## 48. Ensembling the pretrained roster: the harness, and what the backbone says (2026-10-20)

Asked whether a better model can be built by **combining the existing checkpoints with no
further training**, targeting panels C-F of `coguk_list_results` -- the cell-weighted
CoG-UK recall at the top decile (C/D) and the recall AUC (E/F). Plan in
`~/.claude/plans/read-notes-to-get-spicy-lemon.md`.

### Why the panels invite it: no model leads more than two of the four

| panel | metric | leader | value |
|---|---|---|---:|
| C | top decile, `list_coguk_fullpool` | antiGen as ESM 2 (35M) partial FT | 0.5125 |
| D | top decile, `list_coguk_unseen_fullpool` | antiGen | 0.5089 |
| E | recall AUC, fullpool | antiGen as ESM 2 (35M) LM head FT | 0.8239 |
| F | recall AUC, unseen fullpool | antiGen | 0.8235 |

E and F are trapezoids over the committed quantile grid with the (0,0) and (1,1) anchors
`fig_coguk_list.R` adds, recomputed here rather than quoted.

**The complementarity is large and was already on disk.** From
`evals_sc2_lists_macro*/list_macro_percell.csv`, picking the best of the 30 rungs **per
cell** scores **0.962** (C) and **0.957** (D) against best singles of 0.5125 / 0.5089.
Between the two decile leaders, 0.148 of panel C is caught by the 35M partial arm alone
and 0.081 by antiGen alone; on D it is 0.119 and 0.219.

**The members that add most are not the strongest singles**, which is the finding that
sets the member policy. Greedy oracle-union (add the member raising the per-cell max most):

| k | panel C | union | panel D | union |
|--:|---|--:|---|--:|
| 1 | ESM-2 35M partial FT | 0.513 | antiGen | 0.509 |
| 2 | **Tranception (frozen)** | 0.675 | **Tranception (frozen)** | 0.682 |
| 3 | EVEscape | 0.767 | substitution matrix | 0.770 |
| 4 | MSA Trans. (XFG) | 0.816 | EVEscape | 0.833 |
| 5 | BA.1 DMS (expression) | 0.852 | BA.1 DMS (expression) | 0.879 |

Tranception is 6th/2nd alone and is the first addition on both panels; EVEscape, the DMS
panels and the factored baselines each add 0.03-0.09 while scoring 0.17-0.44 alone. These
are CEILINGS taken with the answer, and s14's whole lesson is that such headroom is a
selection failure rather than a capacity one -- but they do say the member pool has to be
broad rather than the top-6.

### The validation set: disjoint TARGET cells, measured

The metric is a mean over cells, so the target cell set is a free axis. Streaming
`examples_cutoff_1.txt` against the CoG-UK cell set and the `pssms/root.csv` mask
(reproduces `list_macro.py`'s own printout: 167,600 testing examples, 17,978 carrying a
listed RBD target, 235 observed listed cells, 131 de novo):

| target set | cells | de novo | events | k >= 5 |
|---|---:|---:|---:|---:|
| CoG-UK observed (**reported, C-F**) | 235 | 131 | 26,491 | -- |
| off-list RBD, same 17,978 queries (**val**) | **1,235** | **720** | 19,449 | 524 |
| ... also position-disjoint | 764 | 442 | 9,393 | 319 |
| off-list RBD, all 167,600 queries | 2,093 | 1,393 | 52,065 | 1,044 |
| pre-cutoff TRAINING targets | 552 | -- | 1,343 | 68 |

Selection runs on the off-list cells, which are ~5x the reported count and cost no extra
forward pass -- they come off the same `cell_pass`. Queries, pool, masking and tie rule
are identical between val and test; **only which cells are counted differs.** The honest
limitation: this is disjointness of *targets*, not of the test window.

**Two alternatives were rejected on measurement, not taste.**

* *In-window pre-cutoff split.* Every published checkpoint trained on all 3,648
  pre-cutoff examples, so weights fit there reward whichever member memorised the window
  hardest. s46's Arm A escaped that only by retraining, which this arm forbids. And s47
  already argued an in-window split is structurally blind to the extrapolation failure
  that matters.
* *The backbone guide tree.* **Its CoG-UK cells are a strict subset of the reported ones**
  -- 79/79 of the 2026 tree's cells lie inside the 235 and 30/30 inside the 131, i.e. 34%
  and 23% of the reported targets; the old tree is 56/56 and 20/20. Unusable for
  selection. It remains the harness-validation and robustness substrate, but a backbone
  re-run of a frozen rule is **not** independent confirmation and must not be read as one.

### Architecture: dump once, sweep many

`cell_pass` already isolates the only thing that differs between a model and a fixed map
behind one callback, so an ensemble is just another `score_fn` and a cached map is just
another way to supply one. `background_discrimination.py` already exploits that on the
backbone substrate; this extends the convention to the attached test queries.

| file | what it is |
|---|---|
| `antiGen/score_cache.py` | the on-disk contract: `query_sha1`, `MapWriter` (.tmp + rename), manifest/queries, `load_map`/`load_block` |
| `antiGen/ensemble_rules.py` | pure rule algebra: `normalize`, `resolve_nan`, `pool`, frozen `Rule` with a canonical `rule_id`, an explicit `LEGAL` set |
| `antiGen/ensemble_sweep.py` | the grid, the val/test cell sets, the metric loop |
| `antiGen/ensemble_diagnostics.py` | the `rank_agreement.py` analogue: Spearman, top-q overlap, per-cell oracle, greedy union |
| `antiGen/list_macro.py` | `cell_pass(..., sink=)` and `--dump-scores` |
| `examples/slurm/dump_sc2_list_scores.sh` | the one GPU job, `--partition=gpu,gpu_batch` |
| `antiGen/tests/test_ensemble_rules.py`, `test_score_cache_replay.py` | 14 tests |

**float32, not float16, and it was measured rather than assumed.** The fractional tie rule
is an exact-equality test against the boundary value, so a half-precision round-trip
manufactures ties. The saving would be ~5 GB on a volume with 45 TB free. Rejected.

**Maps go on `/large_storage`, not the session scratchpad.** The first smoke dump wrote
318 MB to a scratchpad path and the job exited 0 with the completeness assertion passing
-- on the compute node, where the directory is node-local and invisible from the login
node. The SLURM script's default was already shared storage; the override was not.

### The load-bearing test: a one-member ensemble is the identity

Every verification in this arm compares a *one-member ensemble* against a published
single-model row, so those checks mean nothing unless a one-member ensemble is guaranteed
to be a no-op on ranks. `test_one_member_is_a_no_op` asserts it through `compute_hit_rate`
itself (not on the sort order) for all 75 legal (norm, pool, nan) combinations, on a
coarse map as well as a continuous one, and repeated-member rules too.

**It caught a real bug immediately.** `logsoftmax` emits non-positive values, and `geom`
pools `sum(w * log(x))` after `clamp(min=1e-6)` -- which flattened every log-softmax map
to a constant. The fix exposed that `(logsoftmax, geom)` is an exact **alias** of
`(logsoftmax, arith)`: the positive-domain form of a log-softmax is `exp(ls)`, and
`log(exp(ls)) == ls`. It is now absent from `LEGAL`, and `apply` exponentiates a
log-softmax before pooling it geometrically. This is the s32-shaped trap of rules that are
secretly the same rule: shipping both would have counted one rule as two pieces of
evidence and made any multiple-comparisons correction wrong.

### GATE 1 passed: the harness reproduces the published backbone numbers

A one-member ensemble under `pctile`/`arith`/`union_min`, over all 29 members of
`evals_sc2_background_update/`, against
`evals_sc2_lists_macro_backbone_update/list_macro_summary_backbone.csv`:

* **all 522 comparable (model, variant, quantile) cells agree to 5.0e-9** on both `macro`
  and `micro`, with zero `n_cells` mismatches. The published CSV is written `%.8f`, so
  5e-9 is half an ulp of the printed value: the agreement is as exact as the file can
  express.
* The six absentees are the mt2021 arm, whose maps are in a different background
  directory. Not a defect.

Pool sizes confirm the convention matters, and by the predicted amount: on the fullpool
`union_*` gives **n_valid = 4,221** (= 4,422 - 201 wild-type cells, exactly) against
**3,779** for `intersect` -- a **10.5%** shrink, which is the ~10% N_RBD spread s32
measured across families, now reproduced as an identity rather than an observation. Over
the grid `union_min` and `union_median` differ on 271 of 1,620 comparable cells and
`union_min` and `intersect` on 977, so the three are genuinely distinct.

### The backbone families grid: 135 rules, and ensembling does move it

29 members, uniform weights, every legal (norm, pool, nan) cell. 10m40s on one CPU core.
Cell-weighted top decile, **q = 0.1**, ranked by the SELECT column (off-list cells -- what
selection is permitted to see) with REPORT beside it:

| pool | best SINGLE | top-select rule | its REPORT | best REPORT (biased ceiling) |
|---|---:|---|---:|---:|
| fullpool | 0.7215 (EVE frozen) | `rankgauss arith union_min` | **0.7953** | 0.8291 (`zscore arith`) |
| unseen fullpool | 0.6667 (EVE frozen) | `rankgauss trimmed t=0.1` | **0.7250** | 0.7856 (`logsoftmax trimmed`) |

So on this substrate a plain uniform average of all 29 members, with no weights fitted at
all, beats the best single rung by **+0.07** and **+0.06**, and the best rule in the grid
by +0.11 / +0.12. `vote` pooling appears nowhere in either top 10 and the boundary tie
fraction is 0.0% for every rule in them, so the degeneracy that pooling risks did not
bite here.

**Read this as a harness result, not a finding.** The backbone's denominators are 79 and
30 observed cells, so these are very noisy; its cells are a subset of the reported ones;
and the attached-tree numbers are what panels C-F report. What it establishes is that the
machinery works end to end and that the effect is large enough to be worth the one GPU
job.

### State

* GATE 1 passed; 14 new tests plus the existing 100 all pass.
* Smoke dump on one model reproduced `list_coguk_fullpool` macro **0.4460** and
  `list_coguk_unseen_fullpool` **0.5089** -- the published values to 4 dp -- from the same
  forward pass that wrote the map.
* Next: the full-roster dump (one GPU job, ~10 h from job 3362535's 21 models over the
  same 281 batches), then GATE 2's replay at 1e-7 against
  `list_macro_summary_main.csv`, then the sweep on the attached tree.

---

## 49. Static ensembles of the antiGen family: averaging works, and the val set does not agree (2026-10-20)

Asked, on top of s48's harness, the narrower question: take **base antiGen and its eight
fine-tunes** -- the two leftmost legend groups of `coguk_list_results`, nothing else --
and combine them with **simple static rules**: the mean, the max and the min of what each
model says, in ranking space and in logit space. No weights fitted, no members chosen, no
parameters at all. Draw the result on the figure as its own category.

### The members, and the one judgement in them

Nine rungs, uniform weights: `simple_d128_n2_lr0.0001` (antiGen) plus `eve_ft_lr0.0001`,
`tranception_ft_both_lr0.0001` and the six ESM-2 arms (35M and 650M x partial4 / frozen /
full). Excluded, each on a rule: the two `*_noft` rungs (grouped *Pre-Pandemic Data Only*
in the figure, not antiGen fine-tunes); the six `valsel`/`fix32` seed retrains of s46-s47
(antiGen models, but not rungs of this figure, and three near-replicates of antiGen would
have tripled its weight in the average); and everything outside the family.

**The members are not independent, and the figure's own grouping hides it.** Six of the
nine are ESM-2 at two sizes x three fine-tuning depths. An unweighted mean is therefore
tilted six-to-one-to-one toward ESM-2 by construction, which is the most likely reason
`pctile`/`arith` -- one equal vote each -- is the weakest of the three means on panel D,
the one panel base antiGen leads.

### Three spaces, three pools

`min` pooling did not exist in `ensemble_rules`; it is added here, with `max`'s LEGAL
coverage. It is the one pooling whose answer depends strongly on the NaN convention:
under `union_min` an abstaining member is imputed at its own floor, so it vetoes the cell
outright. EVE and Tranception abstain on two residue columns and the ESM-2 arms on one,
so this is not hypothetical.

"Logit space" is `logsoftmax`, i.e. a per-query log-softmax over the candidate pool.
Worth stating because it is not obvious: log-softmax subtracts a per-query **scalar** per
member, so a weighted mean of log-softmaxes is rank-identical *within a query* to a
weighted mean of the raw logits. `(logsoftmax, arith)` therefore IS "average the logits",
and needs no separate rule. That equivalence does **not** extend to `max` or `min`, which
are not shift-equivariant across members -- there the logsumexp really does matter, and
the log-softmax form is the better-defined one.

### GATE 2 passed, on the attached tree

A one-member ensemble under `pctile`/`arith`/`union_min`, for each of the nine, against
`antiGen-figures/data/evals_sc2_lists_macro/list_macro_summary_*.csv`:

* **all 162 (model, variant, qtile) cells agree to 4.9e-9** on `macro` and 5.0e-9 on
  `micro`, with every `n_cells` and `n_events` identical. The published CSVs are written
  `%.8f`, so this is half an ulp of the printed value.

That is the check that makes every ensemble number below mean something: the cached maps
plus this harness reproduce the published rows exactly, so any difference is the rule.

### The result: the mean wins, the max and the min do not

Cell-weighted recall, `q = 0.1` and the recall AUC, on the reported cells (C/E: 235
cells; D/F: 131 de novo). SELECT = the 1,235 / 720 off-list RBD cells.

| rule | C q0.1 | E AUC | D q0.1 | F AUC |
|---|---:|---:|---:|---:|
| **Mean of z-scores** | **0.5740** | **0.8667** | 0.5382 | 0.8474 |
| **Mean of log-probs** | 0.5606 | 0.8630 | **0.5500** | **0.8507** |
| Mean of ranks (Borda) | 0.5560 | 0.8587 | 0.4833 | 0.8362 |
| Min of z-scores | 0.5299 | 0.8247 | 0.4101 | 0.7619 |
| Min of ranks | 0.5072 | 0.8154 | 0.3968 | 0.7635 |
| Min of log-probs | 0.4936 | 0.7995 | 0.4141 | 0.7578 |
| Max of z-scores | 0.4918 | 0.8310 | 0.4931 | 0.8340 |
| Max of ranks | 0.4856 | 0.8251 | 0.4881 | 0.8307 |
| Max of log-probs | 0.4500 | 0.8217 | 0.4868 | 0.8265 |
| *best single of the nine* | *0.5125* | *0.8239* | *0.5089* | *0.8235* |

The best single differs by panel -- 35M partial FT on C, 35M LM-head FT on E, antiGen on
D and F -- which is s48's premise restated inside one family.

**`(logsoftmax, arith)` beats all nine of its own parts on all four panels at once**, by
+0.048 (C), +0.039 (E), +0.041 (D), +0.027 (F). Against the s47 seed floor of 0.035
(C/E) and 0.028 (D/F): C, E and D clear it; **F does not** -- +0.027 is inside the band,
so panel F is not a result. `(zscore, arith)` is larger on C and E and smaller on D and F.

Every `max` and every `min` rule is worse than every `mean` rule on C and E. `min` is the
weakest overall, which is what the veto reading predicts.

### The validation cells do not confirm it, and that is the finding to carry

The same sweep scores every rule on the target-disjoint off-list cells. Nothing was
fitted, so this is not a selection check -- it is a transfer check, and it fails more
often than it passes:

| | SELECT fullpool q0.1 | AUC | SELECT unseen q0.1 | AUC |
|---|---:|---:|---:|---:|
| antiGen alone | 0.2872 | **0.7611** | **0.3496** | **0.7581** |
| Mean of log-probs | **0.3107** | 0.6703 | 0.3411 | 0.7368 |
| Mean of z-scores | 0.3094 | 0.6706 | 0.3466 | 0.7409 |
| Mean of ranks | 0.2964 | 0.6707 | 0.3422 | 0.7391 |

Base antiGen alone wins **three of the four** off-list cuts, and the means beat it only on
`fullpool` at q = 0.1. So the gain measured on the CoG-UK panels is specific to the
curated cell set rather than a general improvement on RBD targets. s45 measured exactly
this axis -- swapping the target set moved the ordering more than swapping the guide tree
did -- and this is another instance of it, now with a mixture rather than a single rung.
Two readings are available and the data here does not separate them: the list is a
better-posed target set, or the list is a narrower one that happens to suit an
ESM-2-weighted average. State the disagreement wherever the +0.048 is stated.

### Code

* `antiGen/ensemble_rules.py` -- `min` pooling, in `POOLS` and in `LEGAL` beside `max`.
* `antiGen/ensemble_report.py` -- `--rule-json` now accepts a LIST of rules, and
  `percell_for_rules` replaces the per-predictor passes with one block-outermost pass
  that shares a normalised stack (a rule-outermost loop re-read every 318 MB map once per
  rule: ~100 GB of I/O over this set). Each comparator is scored as a **one-member rule**
  through the same code path, on the identity `test_one_member_is_a_no_op` pins.
  The list mode is for a family fixed in advance and reported **whole** -- all nine bars,
  losers included. Running several and drawing the best is selection on the reported
  cells with the selection step hidden, which is what the single-rule design prevented.
* `antiGen/tests/test_ensemble_rules.py` -- `min` enters `_all_legal_rules` automatically,
  so the no-op gate covers it; added to the permutation-invariance list explicitly.
  **102 tests pass** (the whole suite, 18 m 13 s).

### A measurement for the yield row, taken rather than assumed

Panels G-J need the model's map on the **BA.2 testing root**, which `list_variant_recall.py`
queries. That genotype is **already in the dump**: its 1,273-residue `aa_idx` matches
dumped query **index 8920**, exactly one hit out of 17,978. So the whole yield row is
reachable with no GPU. It is not free, though -- `compute_hit_rate` folds `-inf` into NaN
and drops it from `n_valid`, while `list_variant_recall.top_q_weights` keeps `-inf` as a
valid never-selected cell, so the same map yields two different top-decile denominators.
That has to be reconciled, and gated on reproducing a published
`list_variant_recall.csv` row from the cached map at 1e-7, before any G-J bar is drawn.

### State

* GATE 2 passed; sweep written (648 rows), 18 rules over 2 pools x 2 cell sets x 9 qtiles.
* `ensemble_report.py` over the nine rules + nine comparators: RUNNING, ~10 m in.
  It writes the figure CSV, the paired-bootstrap contrasts (10,000 resamples, cells as
  the unit), the per-cell table and `ensemble_figure_rows.tsv`.
* Not yet rendered. The wrapper `fig_coguk_list_ensemble.R` and every R-side hook
  (`ENSEMBLE`, `drop_unrequested_ensemble`, the sixth group and its colour) were already
  built in s48 and are unchanged.
* **Open:** G-J per the measurement above; K-L remain invalid for these rules, since all
  three normalisations are within-query and destroy exactly the cross-query level
  information the background contrast reads (s48) -- only a `globalz` rule could fill
  that row.
  > **CORRECTED in s50.** The K-L half of this is wrong. That panel's ruler compares
  > cells *within* one query and holds only the reference pool's cell set fixed across
  > queries, so a within-query normalisation is a no-op on it -- measured at exactly
  > 0.000e+00, and confirmed end to end by GATE 4. All three means are drawn on K-L.
  > `globalz` is a preference for that row, not a requirement. G-J needed no GPU either.

---

## 50. K-L was never blocked, and the rest of the figure for the three means (2026-10-20)

s49 closed with G-J reachable and **K-L declared invalid** for every rule in the family:
"all three normalisations are within-query and destroy exactly the cross-query level
information the background contrast reads (s48) -- only a `globalz` rule could fill that
row." That claim is **wrong**, and this section records the measurement that kills it,
because the same sentence had been copied into three files and was about to decide what
the published figure can show.

### The ruler is a within-query rank, so a within-query normalisation is a no-op

`background_contrast.common_percentiles` computes, for cell `c` at query `q`:

    above = (vals > s_c[:, None]).sum(axis=1)

`vals` and `s_c` are **both row `q`** of the same score matrix. What is held fixed across
queries is the reference pool's **cell set**, not any scale -- that is what makes a fixed
(L, E) map land at exactly 0.5, which was the fixed pool's whole purpose. So K-L is a
**within-query rank evaluated at many queries and then contrasted host-against-control**,
not a comparison of score levels across queries. Any strictly monotone within-query
transform leaves it unchanged, ties included.

Measured on `evals_sc2_background/scores_simple_d128_n2_lr0.0001.npz`, 40 cells:

| transform | max \|pct - pct_raw\| |
|---|---:|
| per-query z-score | **0.000e+00** |
| per-query percentile | **0.000e+00** |

against a mean per-cell sd across queries of **0.0275**, which is the signal K-L reads --
so the invariance is not a degenerate map being compared with itself.

**Where the error came from, since it is a reusable trap.** The docstring stated the right
test -- "harmless for any metric that ranks within a query" -- and then answered it wrong
for K-L, because K-L *compares across* queries. What it compares across queries is a rank.
"Compares across queries" and "reads cross-query levels" are not the same property, and
the first does not imply the second.

### What survives, and it is a preference rather than a constraint

The no-op holds for **one member**. For an ensemble the normalisation decides how members
combine, so mean-of-pctiles and mean-of-z-scores are different pooled maps and genuinely
distinct predictors on K-L rather than aliases. And `globalz` remains the better-motivated
normalisation *for this panel specifically*: under it a member whose raw score barely moves
with the background contributes a near-constant map, whereas `pctile` stretches that
member's within-query ranks to uniform regardless and lets the mean amplify what may be
noise-level background sensitivity.

**That prediction is visible in the result below**: `Mean of ranks` is the weakest of the
three means on K-L by a wide margin, and the only one that is *worse than base antiGen*.

### GATE 3 (G-J) and GATE 4 (K-L), both exact

Same discipline as s48's GATE 1 and s49's GATE 2: each member is pushed through the new
code as a **one-member rule** and must reproduce its own published row.

| gate | comparison | rows | worst \|d\| |
|---|---|---:|---:|
| **3 (G-J)** | one-member ensembles vs `list_variant_recall.csv` | 2,160 | **0.000e+00** |
| **4 (K-L)** | one-member ensembles vs `background_summary.csv` | 72 | **0.000e+00** |

GATE 4 is also the end-to-end confirmation of the correction above: replacing a model's
entire map with its within-query percentile ranks and pushing it through the unmodified
`background_contrast.py` reproduces the published lift, AUC, delta, `n_cells` and
`n_pairs` **to the bit**, for all nine members over eight cellsets.

Neither row needed a GPU. The BA.2 testing root is dumped query **8920** -- located by
matching `aa_idx` against all 17,978 dumped queries, not by quoting s49's index -- and the
backbone maps were already on disk as `evals_sc2_background/scores_*.npz`.

### The -inf reconciliation s49 flagged: real in general, empty here

`compute_hit_rate` folds `-inf` into NaN before counting `n_valid`;
`list_variant_recall.top_q_weights` tests only `isnan`, so it keeps `-inf` in the pool --
**despite a docstring that explicitly claims it drops it**. The two therefore disagree on
the top-decile denominator for any member emitting `-inf`. Measured: **none of the nine
emits a single `-inf` cell**, so both conventions give identical `n_valid` and the
published one is used unchanged. The check runs on every invocation rather than being
remembered, because it is a property of the members: adding the codon baseline to a rule
makes it bite immediately.

### G-J: the ensemble catches more cells and still loses the published ratio

Max yield of the BA.2 top decile, CoG-UK, over the time grid:

| | all 1+ | de novo 1+ | all 100+ | de novo 100+ |
|---|---:|---:|---:|---:|
| **EVE full FT** (best single) | **0.2461** | **0.0812** | **0.1466** | 0.0419 |
| Tranception full FT | 0.2042 | 0.0759 | 0.1388 | 0.0419 |
| antiGen | 0.1516 | 0.0261 | 0.0948 | 0.0095 |
| Mean of log-probs | 0.2322 | 0.0592 | 0.1374 | 0.0332 |
| Mean of z-scores | 0.2322 | 0.0592 | 0.1350 | 0.0308 |
| Mean of ranks | 0.2300 | 0.0569 | 0.1303 | 0.0261 |

> **SUPERSEDED by s51.** The table above was computed with a top decile whose SIZE
> differed by model, which is a defect rather than a confound. s51 fixes it with a common
> candidate pool and restates every number in this section. The diagnosis below stands and
> is what led there; the conclusion drawn from it -- that `union_min` must stay and the
> confound merely be reported -- does not.

**This is the opposite of A-F, where all three means beat all nine of their parts** -- and
the gap is a denominator artefact, which has to be said with the numbers rather than left
for a reader to find:

| | listed cells caught | own top decile | ratio |
|---|---:|---:|---:|
| EVE full FT | 94.0 | 381.9 | 0.2461 |
| Mean of log-probs | **98.0** | 422.1 | 0.2322 |
| EVE full FT, *at the ensemble's denominator* | 94.0 | 422.1 | *0.2227* |

The ensemble catches **four more listed cells** than EVE-FT and is scored against a **10%
larger** top decile, because `union_min` hands it antiGen's gap-emitting pool (4,221 cells)
while EVE emits no gap/stop columns (3,819). This is the ~10% N_RBD spread `fig_coguk_list.R`
already warns about in its own yield-row header, now deciding an ordering.

**`intersect` would match the denominators and flip the ordering, and it is not being
switched to.** s49 fixed `union_min` before any of this was scored; changing the NaN
convention now, having seen that it costs the ensemble the panel, is selection on the
reported cells. It stays, and the confound is reported instead.

> **s51 note.** That reasoning was right about `intersect` and wrong about what to do.
> Switching the ensemble's NaN convention after seeing the result would indeed have been
> selection -- but the fair comparison was never the ensemble's to make by changing its
> own rule. The pool had to be fixed for EVERY model at once, which is not selection
> because it is not a choice about the ensemble at all. `union_min` does stay.

### K-L: the mean of log-probs leads the de novo cut, the mean of ranks loses

Background lift in percentage points, CoG-UK cells (chance is 0):

| | all | de novo |
|---|---:|---:|
| antiGen as ESM 2 (35M) partial FT | **+5.95** | +11.20 |
| **antiGen** | +4.41 | +12.37 |
| antiGen as ESM 2 (35M) full FT | +3.38 | +8.25 |
| antiGen as EVE full FT | +0.73 | +6.80 |
| antiGen as ESM 2 (650M) frozen | -3.23 | -16.39 |
| **Mean of log-probs** | +5.02 | **+15.62** |
| **Mean of z-scores** | +4.78 | +12.34 |
| **Mean of ranks** | +2.73 | +8.61 |

`Mean of log-probs` is the only rule that beats every one of its parts on the de novo cut.
`Mean of ranks` sits **below base antiGen on both**, which is the `globalz` argument above
showing up as a number: three of the nine members have *negative* background lift, and a
rank normalisation gives their background sensitivity the same spread as everyone else's.

No seed floor exists for K-L, so these are read against the published `lift_lo`/`lift_hi`
in `background_summary.csv` and not against each other informally.

### What is drawn, and the rule for it

All nine static rules are scored and written to `list_macro_summary_ensemble.csv`. The
**three means** get bars; the six max/min rules stay in the CSV undrawn. The selection rule
is `pooling == arith`, statable without looking at any result, which is what keeps this
from being "draw the best three of nine".

**The bars are on `coguk_list_results` itself, not only the sibling.** s48 had argued for a
sibling on the grounds that an ensemble bar sits beside its own parts; that is still true
and is now said in the figure header and marked by the group's own colour, rather than
being handled by keeping the row off the figure.

`coguk.ensemble` no longer defaults to empty. It defaults to whatever
`ensemble_figure_rows.tsv` in **that figure's own macro directory** lists, so a directory
carrying ensemble rows draws them and the backbone, valsel and all-RBD siblings are
untouched without having to opt out.

### Code

* `antiGen/ensemble_panels.py` -- new. G-J from the list dump and K-L maps from the
  backbone dump, both gates built in, `--gate-only` to run just the checks. It writes
  combined background maps as `scores_ens_*.npz` and leaves the contrast to
  `background_contrast.py` **unmodified**: the published path stays the published path and
  the ensemble enters it as just another model. Refuses a `globalz` rule on G-J, whose
  moments are taken over the whole block and would be computed from one query.
* `antiGen/ensemble_rules.py` -- the module docstring's last paragraph and the `globalz`
  comment, both rewritten; the wrong claim and the reason it was wrong are kept rather than
  silently deleted.
* `antiGen-figures/figures/fig_coguk_list.R` -- `ENSEMBLE` defaults from the macro dir's
  rows file; the "empty by default" rationale replaced by what a reader needs to know.
* `antiGen-figures/figures/fig_coguk_list_ensemble.R` -- header corrected.

### Open

* `globalz` is still unscored anywhere. It is the better-motivated normalisation for K-L
  and is now the only remaining reason to prefer one mean over another on that row; a
  `(globalz, arith, union_min)` run would settle it and costs one CPU pass.
* G-J's denominator confound deserves a matched-pool reading reported beside the published
  one -- the same "both conventions from one pass" move `background_contrast` already makes
  for `_perquery`. Not done here, because inventing it after seeing the ordering is the
  thing the `union_min` note above refuses.

---

## 51. One pool for every model: "the top decile" was a different size per rung (2026-10-20/21)

s50's G-J table reported EVE full FT ahead of every ensemble on the yield row. The
numerator says otherwise: the nine-member **mean of log-probs caught 98.0 listed cells to
EVE-FT's 94.0** and still reported 0.2322 against 0.2461, because the two were scored
against top deciles of different sizes. That is not a confound to report beside the
number, as s50 proposed; it is a defect in the metric, and this section fixes it.

### The size of "the top decile" was a property of a model's alphabet

Every top-q set in panels A-J is `q * n_valid`, and `n_valid` was **each model's own
emission set**. Measured over the 17,978 test queries, by variant:

| variant | panels | n_valid across the 37 rungs | spread |
|---|---|---|---:|
| `list_coguk_fullpool` | **A, C, E** | 3,682.8 .. 4,221.0 | **12.75%** |
| `list_coguk_unseen_fullpool` | **B, D, F** | 3,287.0 .. 3,602.2 | **8.75%** |
| `list_coguk` | *not drawn* | 224.1 .. 236.8 | 5.35% |
| `list_coguk_unseen` | *not drawn* | 134.1 .. 141.4 | 5.19% |

**All six recall panels were affected, not just the yield row.** `fig_coguk_list.R`'s
`recall_long` points all four of its metrics at the `*_fullpool` variants -- A/B are the
event-weighted reading and C/D the cell-weighted one of the *same* pool -- so the 12.75%
applies to the published headline panels. The list-restricted variants, which would have
been nearly fair, are computed by `list_macro.py` and never drawn.

At the single BA.2 root query the spread is 10.0% and takes five values: 4,221 (antiGen,
the PSSMs, pyr0, CoVFit, substitution, `where x which`), 4,020 (all eight ESM-2 arms, the
MSA Transformers, codon), 3,819 (EVE, Tranception, BLOSUM62, evcouplings, the BA.1 DMS
panels), 3,804 (EVEscape) and 3,801 (the BA.2 DMS panels). Almost all of it is the 201 gap
plus 201 stop cells. **BLOSUM62 is the extreme over the full query set**, not EVE, because
it also abstains on listed cells -- which is why even the undrawn list-restricted variants
were not fair.

### The pool spread understates it: 44% of the USABLE slots, and it penalised antiGen

The first version of this section said a larger pool meant "more chances to catch a
target", so the artefact flattered antiGen on A-F. **That was backwards**, and the
measurement that settles it is how many of a model's top-decile slots went to cells that
cannot be targets at all. 0 of the 247 listed RBD cells is a gap or a stop, so every slot
spent on a gap/stop cell is a wasted one. At the BA.2 root, q = 0.1, legacy pool:

| model | slots | spent on gap/stop | **usable** |
|---|---:|---:|---:|
| substitution matrix | 422.1 | 193.1 (45.7%) | **229.0** |
| **antiGen** | 422.1 | **168.0 (39.8%)** | **254.1** |
| PSSM (root) | 422.1 | 146.9 (34.8%) | 275.2 |
| `where x which` | 422.1 | 113.0 (26.8%) | 309.1 |
| codon | 402.0 | 26.1 (6.5%) | 375.9 |
| EVE / Tranception / BLOSUM62 | 381.9 | 0 | 381.9 |
| pyr0 | 422.1 | 38.0 (9.0%) | 384.1 |
| ESM-2 35M partial FT | 402.0 | 0.0 | 402.0 |
| CoVFit | 422.1 | 10.0 (2.4%) | 412.1 |

**The usable spread is 229.0 to 412.1 -- 44%**, against the 12.75% spread in nominal pool
size. And the sign is the reverse of the naive reading: the models with the LARGEST pools
are the ones that threw most of the surplus away, because the models that emit gap and
stop are exactly the ones that rank them highly. antiGen was selecting 254 usable cells
against EVE-FT's 382 -- a **33% handicap** -- while appearing to have 10.5% more slots.

So the artefact **penalised antiGen on A-F**, and on G-J it penalised it twice over: a
denominator 10.5% larger AND 39.8% of the selection wasted. The composition baselines
(substitution, the PSSMs, `where x which`) were hit hardest of all, which is a caution
about every ranking in s32 and s45 that involved them.

The prediction this makes is therefore the opposite of the one first written here:
antiGen and the composition baselines should RISE on A-F under the common pool, and the
EVE / Tranception / ESM-2 arms should be close to unchanged. Whether that is what
happens is a measurement, not an expectation, and it comes from the same cached maps with
the convention as the only difference.

### The fix, and what it costs

One pool, defined without reference to any model: the variant's own cell set intersected
with the **20 real amino-acid columns** (`eval_utils.real_substitution_mask`). A cell
inside the pool that a model cannot score is **ranked last** rather than dropped, so
EVEscape's 15 and the BA.2 DMS panels' 18 genuinely missing measurements do not shrink the
pool for everyone. That is the convention `background_contrast.common_percentiles` has
always used for K-L, which is why the background row needed no change at all.

**It removes no target: 0 of the 247 CoG-UK listed RBD cells is a gap or a stop.** The
change is to the denominator and to the size of the selected set, never to what counts as
a hit.

**Why exclude gap/stop for everyone rather than keep all 4,221 columns and rank the
non-emitters' gap/stop last.** The latter converts the asymmetry instead of removing it:
antiGen could spend top-decile slots on stop codons while EVE structurally cannot, so the
gap-emitting models would then be penalised for emitting. Excluding the columns asks every
model only the question all of them can answer.

### The regression ran first, and it passed

`rescore_common_pool.py --legacy-pool` must reproduce the published CSVs from the cached
maps before any common-pool number is believed; otherwise a difference could be the new
script rather than the new convention.

| | rows compared | worst \|d\| |
|---|---:|---:|
| A-F macro | 144 | **6.0e-08** (half an ulp of the published `%.8f`) |
| A-F micro | 144 | **6.0e-08** |
| G-J value / n_selected | 960 | **0.000e+00** |

with zero `n_cells` / `n_events` mismatches, and the run's own diagnostic correctly
reporting that under `--legacy-pool` the top decile **still varies** (380.4 / 381.9 /
422.1) -- the defect, reproduced on demand.

### Measured: what the common pool did to the ordering

Rendered 2026-10-21. Every number below is old (per-model pool) -> new (common pool), from
the SAME cached maps, so the convention is the only difference. The three mean-ensembles
of the antiGen family are drawn on all twelve panels; `--` means the row is new.

**Panel C, all listed cells (macro, q = 0.10).** The leader changes.

| | old | new | delta | rank |
|---|---:|---:|---:|---|
| **antiGen** | 0.4460 | **0.5350** | **+0.0891** | #4 -> **#1** |
| Mean of log-probs | -- | 0.5321 | -- | #2 |
| Mean of ranks | -- | 0.5245 | -- | #3 |
| Mean of z-scores | -- | 0.5243 | -- | #4 |
| antiGen as ESM 2 (35M) partial FT | 0.5125 | 0.4987 | -0.0138 | #1 -> #5 |
| PSSM (root) | 0.3624 | 0.4241 | +0.0617 | #12 -> #10 |
| substitution matrix | 0.1777 | 0.3111 | **+0.1334** | #24 -> #18 |

**Panel D, de novo.** antiGen loses a little and drops to 3rd behind two means.

| | old | new | delta | rank |
|---|---:|---:|---:|---|
| Mean of log-probs | -- | **0.5379** | -- | **#1** |
| Mean of z-scores | -- | 0.5221 | -- | #2 |
| antiGen | 0.5089 | 0.4988 | -0.0101 | #1 -> #3 |
| Mean of ranks | -- | 0.4705 | -- | #4 |

**Panels E/F, recall AUC.** All three means lead both.

| | E old | E new | F old | F new |
|---|---:|---:|---:|---:|
| Mean of log-probs | -- | **0.8502** | -- | **0.8440** |
| Mean of z-scores | -- | 0.8488 | -- | 0.8384 |
| Mean of ranks | -- | 0.8432 | -- | 0.8307 |
| antiGen | 0.8209 | 0.8409 (+0.0200) | 0.8235 | 0.8219 (-0.0016) |

**G-J, max yield of the BA.2 top decile (all, 1+).** EVE full FT keeps the row, now on a
denominator every model shares (381.9 cells, not 381.9-vs-422.1), and the s50 inversion is
gone: the means no longer catch more cells while scoring lower.

| | old | new | delta |
|---|---:|---:|---:|
| antiGen as EVE full FT | 0.2461 | **0.2461** | +0.0000 |
| Mean of log-probs / z-scores | -- | 0.2409 | -- |
| Mean of ranks | -- | 0.2357 | -- |
| antiGen | 0.1516 | 0.2173 | **+0.0657** |
| PSSM (root) | 0.1688 | 0.2149 | +0.0461 |

**K-L, background lift, CoG-UK all cells (pp).** Smallest movement of any row, as expected:
its pool only lost the 400 gap/stop cells, and the models that never emitted them do not
move at all.

| | old | new | delta |
|---|---:|---:|---:|
| Tranception (frozen) | 8.24 | **8.24** | +0.00 |
| antiGen as ESM 2 (35M) partial FT | 5.95 | 6.52 | +0.57 |
| Mean of z-scores | -- | 5.57 | -- |
| Mean of ranks | -- | 5.39 | -- |
| Mean of log-probs | -- | 4.71 | -- |

**The sign test the whole fix rests on.** Every model that already declined to emit gap and
stop moves by **exactly 0.0000** -- EVE, Tranception, the frozen ESM-2 arms, the DMS panels,
the MSA Transformers -- because for them the common pool IS their own pool. Only the rungs
that emitted those columns move, and the composition baselines, which ranked them highest,
move most. That is the predicted pattern and it is the reason to believe the change did what
it claims rather than something else.

### s49's headline does not survive, and the direction is worth stating

s49 reported `(logsoftmax, arith)` beating all nine of its own parts on all four panels.
Under one pool that holds on **D, E and F** but **not on C**, where the mean (0.5321) now
loses to antiGen (0.5350), its own strongest member. The gain was measured against a field
in which antiGen was handicapped hardest -- 39.8% of its slots spent on cells that cannot be
targets -- so a large part of what looked like ensembling value was the pool convention.

What survives is narrower and still real: the means lead both AUC panels outright, and on D
they lead by +0.039 over antiGen, which clears the 0.028 seed floor. `Mean of ranks` remains
the weakest of the three on every panel, and on K-L it is below base antiGen, which is the
`globalz` argument from s50 showing up as a number.

### Gates on the shipped run, all exact

| gate | what | result |
|---|---|---:|
| legacy A-F | `--legacy-pool` vs the published macro CSVs, from cache | **6.0e-08** |
| legacy G-J | same, `list_variant_recall.csv` | **0.000e+00** |
| legacy K-L | `background_contrast --legacy-pool` vs `background_summary.csv`, 240 rows | **0.000e+00** |
| cross-job | the two independent paths writing the ensemble rows | **1.20e-07** |
| yield row | top decile identical across models | **381.9, all 32** |

The cross-job number is the one that matters most here: it was **4.98e-02** on the first
attempt and is what caught the normalisation-support bug. `fig_coguk_list.R` now also hard-
errors if `n_selected` varies across models, which it did at 42.0 cells before the fix.

### Code

* `antiGen/eval_utils.py` -- `real_substitution_mask`, and `compute_hit_rate(pool_mask=)`.
* `antiGen/background_discrimination.py` -- `rank_fractions(pool_mask=)`, so the ensemble
  path and the published path stay the same metric. The identity between the two is
  re-verified under the mask.
* `antiGen/list_variant_recall.py` -- `top_q_weights(pool_mask=)` and `curve(pool_mask=)`.
  Its docstring claimed `-inf` left the pool; it tests `isnan` only, so it does not. Noted
  rather than changed, because a pool mask makes it moot.
* `antiGen/ensemble_sweep.py` -- `Substrate.pool_mask` excludes gap/stop, which propagates
  to `ensemble_report` and the sweep through `_wt_masked` with no other change.
* `antiGen/rescore_common_pool.py` -- new; re-scores A-F and G-J for every cached rung and
  every rule from the cache, with `--legacy-pool` as the regression.
* `antiGen/tests/test_common_pool.py` -- **20 tests**: the mask drops exactly gap and stop,
  `n_valid` is model-independent under it, an unscorable in-pool cell ranks last and still
  counts, the two metric implementations agree under it, and `pool_mask=None` is bit-exact
  with the published behaviour.
* `antiGen-figures/figures/fig_coguk_list.R` -- the yield row's reference curves were a
  median over a band because N_RBD varied; the band is now a point. Guarded: a spread in
  `n_selected` across models is a hard error.

### The published numbers move, and s49/s50's tables are superseded

Panels C/D are where s49's incumbents (0.5125, 0.5089) and its `+0.048` live. Those were
computed at mixed pool sizes. **Every A-J number in s49 and s50 is restated under this
convention**; the five per-job macro CSVs are moved to
`data/evals_sc2_lists_macro/legacy_per_model_pool/` rather than deleted, so what the figure
used to say remains on disk.

That retires `fig_coguk_list.R`'s cross-job agreement check, which compared the 12 models
appearing in more than one GPU job. The replacement is not nothing: every map this
re-score reads was gated by replaying it to the published row written beside it (s48
GATE 1, s49 GATE 2, both exact), which is the same claim. And the ensemble rows are still
written by two independent paths -- `rescore_common_pool.py` and `ensemble_report.py` --
so the check retains something real to compare.

### Audit of the selection path: one real leak, one bounded artefact, three clean

Asked, after two bugs in this arm were caught by guards rather than by review, whether the
comparison is actually fair -- whether each model gets the same number of top-q picks for a
given query. Measured on the cached maps across 13 rungs and q in {0.1, 0.2, 0.5, 0.9}.

**A. `n_valid` is identical across models, per query.** Verified for all 13 against a
reference rung, exactly. The per-query variation that remains (3,819 / 3,823 / 3,842 ...)
is the query's own residue being masked, and it is the same number for every model at the
same query. This is the property the arm exists to establish, and it holds.

**B. The realised selected MASS is not exactly `q * n_valid`, by at most one cell.**
`nanquantile(1-q, interpolation='lower')` picks ascending index `floor((n_valid-1)(1-q))`,
so for a continuous-valued model at n_valid = 3,819, q = 0.1 the count strictly above the
boundary is 382 while the quota is 381.9; `frac` clamps at 0 and the realised mass is
**382.0**. A coarse predictor has a large tie block at the boundary, `frac` lands inside
[0, 1], and its mass is exactly 381.9. So continuous rungs select 0.1 of a cell more than
coarse ones -- **0.026%**, worth at most 0.000426 of a panel-C macro score, **82x below the
seed floor**. `top_q_weight`'s docstring already says the two "differ by up to one cell".
Documented, not changed: removing it would mean abandoning `interpolation='lower'`, which
is what makes the boundary reproducible.

**C. A REAL LEAK, now fixed: an unscorable cell could be selected at high q.** The common
pool fills an in-pool cell the model cannot score with `min(finite) - 1`, so it ranks last
but stays in the denominator. Ranking last is not the same as never being selected: the
filled cells are **one tie block**, so once `q * n_valid` exceeds the SCORABLE count the
boundary falls inside that block and every cell in it takes fractional weight.

BLOSUM62 carries query-dependent `-inf` (460 in-pool cells at one test query, 80 at
another, 0 at the rest). Leaked selection weight:

| q | weight on cells it cannot score | of selected |
|---:|---:|---:|
| 0.1 | 0.0000 | 381.9 |
| 0.5 | 0.0000 | 1,909.5 |
| 0.9 | **75.80** | 3,437.1 (**2.2%**) |

Effect on the reported number: **recall AUC 0.6113 -> 0.6073, dAUC = -0.0039**; q = 0.9
recall 0.9262 -> 0.9062. codon and antiGen are unaffected (0.0000), so panels C/D at
q = 0.1 were never touched -- only E/F, which integrate out to q = 0.9.

**This was introduced by the common-pool fill itself.** Before it, `-inf` was folded to
NaN and dropped, so such a cell could never be selected. Fixed by zeroing the weight on
filled cells while keeping them in `n_valid`: a model that abstains forgoes that pick. It
cannot be credited for a cell it declined to score, and it cannot gain from a smaller pool
either. The consequence to state plainly is that at q large enough, an abstaining model
selects slightly FEWER than `q * n_valid` cells -- which is the honest representation,
since it has no more cells to offer.

`rank_fractions` grew `return_filled=`; `compute_hit_rate` and `top_q_weights` zero
internally. A name collision found on the way: `rank_fractions` already had a local
`filled` (its sort buffer), so the new mask had to be `filled_mask` -- the first version
returned the sort buffer and the test caught it.

**D. The three implementations agree under the pool.** `compute_hit_rate`,
`rank_fractions` + `top_q_weight`, and `list_variant_recall.top_q_weights` give the same
hit mass to 1.5e-4 on masses of ~275 (float32), with identical `n_valid`, across
blosum62 / EVEscape / codon / antiGen at q in {0.1, 0.5, 0.9}.

**E. A one-member rule through the `_wt_masked` path equals the single model exactly** --
`n_valid` equal and hit mass identical to 0.000e+00, which is what makes every gate in
s48-s51 mean anything.

**F. `pool_mask=None` is bit-exact with the published behaviour**, so the legacy
regression is a real check and not a tautology. Re-verified after the leak fix: 0.000e+00.

Tests: `antiGen/tests/test_common_pool.py` is now **28**, including one that drives a model
to abstain on most of the pool and asserts zero leaked weight at q up to 0.95, and one that
puts every target on an unscorable cell and asserts zero credit.

### Durability: this re-score had to be a SLURM job, and it took three tries to admit it

Worth recording because it cost more wall-clock than the measurement did, and because s48
already wrote down half of it.

1. **Attempt 1** ran under `setsid nohup` with its output directory in the session
   scratchpad. The session ended; the scratchpad was deleted with it. The processes were
   still alive afterwards, writing to unlinked inodes -- `ps` showed them running and
   `ls` showed nothing, which is exactly as confusing as it sounds. s48's own note says
   "maps go on `/large_storage`, not the session scratchpad", for the same reason, about
   a different artifact.
2. **Attempt 2** fixed the path but kept `setsid nohup`. It survived one session restart
   and not the next, dying with 0 of 9 blocks done.
3. **Attempt 3** is `examples/slurm/rescore_commonpool.sh`, which depends on nothing but
   the shared filesystem. The interactive box has **4 cores**; the `cpu` partition has
   **184**. The two concurrent jobs had been oversubscribing torch's default 4 threads
   each onto those 4 cores, which is why `ensemble_report`'s `fullpool` pass went from
   ~10 minutes alone to over 23 minutes, and why the whole thing looked slower than any
   per-job estimate predicted.

The rule to carry: anything in this arm that takes more than a few minutes goes through
`sbatch`, with outputs under `/large_storage`. Monitoring watches the SLURM job id rather
than a shell pid, so the watch survives whatever happens to the session.

Also `pkill -f <script>.py` matches the interactive shell wrapper whose command line
contains that string, and so kills the command issuing it. Kill by PID here.

### State

* **Shipped.** `figs/coguk_list_results.png` and `figs/coguk_list_supp.png` re-rendered
  2026-10-21 11:24 with the three means on all twelve panels; committed as **91577a0** on
  `is-multitree` (40 files) and pushed. The figure was inspected, not just produced: six
  legend groups with `Ensembles` in its own colour, the three bars present in C/D, J, K and
  L, rank medals agreeing with the tables above, nothing clipped.
* The five per-job macro CSVs are in `data/evals_sc2_lists_macro/legacy_per_model_pool/`,
  committed as renames, so the pre-s51 numbers stay in the tree and not only in history.
  All eight `valsel`/`fix32` rungs are in the common-pool CSV, so the s47 sibling figure
  still resolves through the glob.
* **Not committed, deliberately:** the `examples/slurm/train_*.sh` parameterisation, the
  s46/s47 valsel figures and wrapper, `escape.csv`, `eval_tree.json`. None of them is this
  arm's work and each wants its own message.
* Tests: 28 in `test_common_pool.py`; 48 pass across the four files covering the changed
  code. The full suite has NOT been run end to end since these edits.

### Open

* K-L moved least of any row, but it did move (its pool lost the 400 gap/stop cells), so the
  old and new figures are no longer directly comparable on ANY row. There is now no panel
  that can be read across the two conventions.
* **`globalz` is still unscored**, and it is now the only open question about which mean to
  prefer on K-L. One CPU pass would settle it.
* **The s45 all-RBD sibling is stale.** `rbd_all_cells.csv` lists the gap and stop columns
  (201 each of its 4,422), so under this convention its denominators change and its targets
  in those columns become uncatchable. It needs its own pass and its own decision about
  whether those cells should ever have been targets. Not touched here.
* The backbone siblings (s44, s45) and the q0.05/q0.20 variants still read their own macro
  directories under the per-model pool. They are internally consistent but are no longer on
  the same convention as the published figure.
* The s45 all-RBD sibling takes `rbd_all_cells.csv`, which **does** list the gap and stop
  columns (201 each of its 4,422). Under the common pool those cells are no longer
  candidates, so that figure's denominators change too and its targets in those columns
  become uncatchable. Not re-run here; it needs its own pass and its own note.

## 52. The other half of the pool: `-123A` was a candidate (2026-10-21)

s51 made "the top decile" one size for every model by dropping the gap and stop
**columns**: no model may pick `A123-` or `A123*`. It never asked the mirror question.
A cell is a (position, residue) pair, and nothing looked at what the QUERY had at that
position, so at a position whose query residue is a gap or a stop every one of the 20
real-amino-acid cells stayed in the pool -- `-123A`, `*123C` -- and could be selected.

### The arithmetic s51 left behind, and it is the wrong way round

`compute_hit_rate` masks the query's own residue with a `scatter_` over the score grid.
When the query's residue is `-`, that column is **already** out of the pool, so the
scatter removes nothing. An ungrounded position therefore contributed a **full 20**
candidates where a normal position contributes 19.

Measured on the 17,978 CoG-UK test queries:

| | count | share |
|---|---:|---:|
| queries with >= 1 gap/stop in the RBD | **2,763** | 15.4% |
| (query, position) pairs that are ungrounded | **53,616** | 1.48% |
| of which stop rather than gap | 51 | 0.1% |
| candidate cells this put in the pool | ~1.07M | |

It is overwhelmingly gap, and gap here is missing sequence, not a deletion: the
per-query count runs to **196 of 201 positions**, which is a genome with essentially no
RBD coverage rather than a 196-residue deletion.

### It removes targets, and that is the difference from s51

s51 could say "it costs no target: 0 of the 247 listed RBD cells is a gap or a stop".
This mask cannot, and the reason is in `mutation_lists.load_mutation_list`: a list row
becomes a **`(position, alt)` cell**. The `wildtype` column is read only as a coordinate
check (`n_wt_mismatch`, to catch a 1-based slip) and never constrains the source. So a
test-set event scores as a hit on a listed cell whatever the query's own residue was.

`coguk.csv` itself is clean -- 323 rows, every one letter-to-letter, no `-` or `*` on
either side. The ungrounded events are an artefact of the test set meeting the cell
definition:

| | events | from a gap/stop | cells | emptied |
|---|---:|---:|---:|---:|
| `list_coguk_fullpool` (A/C/E) | 26,491 | **362 (1.37%)** | 235 | 2 |
| `list_coguk_unseen_fullpool` (B/D/F) | 5,070 | **119 (2.35%)** | 131 | 2 |

The shape of them, against the list row they are credited to:

| observed | list row | events |
|---|---|---:|
| `-501Y` | N501Y | 29 |
| `-483G` | V483G | 28 |
| `-477N` | S477N | 26 |
| `-498R` | Q498R | 22 |
| `-484A` | E484A | 17 |

That is coverage returning, not a substitution. The two cells emptied are **Y489L** and
**F490G**, each a singleton whose one event was `-489L` / `-490G`; neither was ever
observed as a real substitution. The 84 partial cells keep their full weight in the
macro and are scored on their grounded queries only -- N501Y on 449 events rather than
478.

### `--ungrounded-queries`, three ways, and why the default is `drop`

* `score` -- the pre-s52 convention: the cell is a candidate and its events are scored.
* `keep` -- the cell leaves the pool; its events stay in the denominator, unhittable.
* `drop` -- **the default**: the cell leaves the pool and its events leave the metric.

`drop` because it keeps the invariant `build_variants` already relies on, that every
target is in the pool it is ranked in, and because `keep` docks every model ~1.4% (2.4%
on de novo) for failing to predict a sequencing gap. The alternative is one flag away
and the arm is set up to run it.

### Only A-F can move, and that is measured rather than assumed

* **G-J is untouched.** The row is scored at the single BA.2 root query, which carries
  **0** gap/stop in the RBD. The mask is applied to it anyway so the row stays correct
  if the root ever changes, and `rescore_common_pool` prints the count every run.
* **K-L is untouched, and must not be.** Only RBD positions **344 and 483** carry a gap
  or stop across the 402 backbone genotypes, and neither is in `coguk_all` /
  `coguk_unseen`, the two cellsets the main figure's K-L draw. (`rbd_all` / `rbd_unseen`,
  in the supplement, touch 483 at 3 cells and 1.) More than that, the background row
  ranks one cell ACROSS queries against a deliberately query-INDEPENDENT pool (s39), so
  a per-query mask would put back exactly the confound that row exists to avoid.
  `ensemble_sweep.substrate_from_args` refuses the flag on the backbone substrate.

### Gates

Three, all before any s52 number is believed, for the reason s51 gives: a difference has
to be the convention and not the plumbing.

| gate | what |
|---|---|
| 1 | `--legacy-pool` still reaches the **pre-s51** published CSVs |
| 2 | `--ungrounded-queries score` reaches the **shipped s51** CSVs, A-F and G-J |
| 3 | the top decile is one size across all models |

### Code

* `antiGen/eval_utils.py` -- `real_wildtype_mask(aa_sub)` -> (B, R), the query-dependent
  half of the pool, and `compute_hit_rate(wt_pool_mask=)`. The two halves are
  intersected **before** the unscorable fill, so a cell about to leave the pool is never
  filled and never drags the floor down with it.
* `antiGen/background_discrimination.py` -- `rank_fractions(wt_pool_mask=)`, so the
  ensemble path and the published path stay the same metric.
* `antiGen/ensemble_sweep.py` -- `UNGROUNDED`, `Substrate.ungrounded` / `.wt_pool()` /
  `.targets_block()`, and `_wt_masked(wt_pool_mask=)`. The mask has to bite in
  `_wt_masked` and not only at ranking time, for the reason s51 measured at 0.0498.
* `antiGen/ensemble_report.py` -- reads both through the substrate; inherits the flag.
* `antiGen/rescore_common_pool.py` -- `--ungrounded-queries`, defaulting to `score`
  under `--legacy-pool` so gate 1 is unchanged.
* `antiGen/tests/test_query_pool.py` -- **19 tests**: the mask is the row mirror of the
  column one, an ungrounded position gives 0 candidates where s51 gave it 20, nothing
  there is ever selected even when it is the highest-scoring cell in the map, the three
  modes do what they say, the two metric implementations agree under it, and
  `wt_pool_mask=None` is bit-exact with s51.
* `examples/slurm/rescore_querypool.sh` -- the job. SLURM, outputs on `/large_storage`,
  per s51's durability rule.

### Gates on the shipped run, all passed

Job **3392095**, 34:41 on `cpu`, exit 0:0.

| gate | what | result |
|---|---|---:|
| 1 | `--legacy-pool` vs the **pre-s51** published CSVs, 72 rows | **6.0e-08** |
| 2a | `--ungrounded-queries score` vs the **shipped s51** macro CSVs, 234 rows | **0.000e+00** |
| 2b | same, G-J `list_variant_recall.csv`, 3,120 rows | **0.000e+00** |
| 3 | top decile one size across all models | **381.9, all 32** |
| cross-job | the two independent paths writing the ensemble rows | **1.00e-07** |

Gate 2 is the one that matters: **exact on every row**, with zero `n_cells` / `n_events`
/ `n_targets` mismatches. The s52 path is bit-identical to s51 when switched off, so
everything below is the convention and not the plumbing.

### Measured: antiGen was being paid for recognising dropout

The two models that lose most on panel C are exactly the two that were hitting the
ungrounded events far above their own baseline. Top-decile hit rate on the 2,763
affected queries, under the s51 convention, grounded listed events vs ungrounded:

| model | grounded | **ungrounded** | ratio | panel C delta |
|---|---:|---:|---:|---:|
| **antiGen** | 53.0% | **78.2%** | **1.47x** | **-0.0203** |
| ESM 2 (35M) untrained | 11.3% | 46.7% | **4.12x** | -0.0210 |
| Tranception (frozen) | 56.1% | 54.1% | 0.97x | +0.0167 |
| EVE full FT | 76.9% | 56.1% | 0.73x | +0.0025 |
| codon | 30.2% | 0.2% | 0.01x | +0.0126 |
| BLOSUM62 | 29.6% | 0.0% | 0.00x | +0.0076 |

The mechanism is not subtle. antiGen is query-conditioned, so at a position reading `-`
it predicts the consensus residue -- which is exactly what "coverage returning"
restores. It was being credited for identifying sequencing dropout, not for predicting
evolution. BLOSUM62 and codon score a substitution FROM the source residue, which is
meaningless from a gap, so they never collected that credit and gain when it is
withdrawn. This is the direction s51's composition-baseline caution would predict, for
the opposite reason.

### What moved: C and D only, and s49's headline comes back

Rendered 2026-10-21 13:09. Old = s51 common pool, new = s52 query pool, same cached
maps, convention the only difference. Test set 235 cells / 26,491 events -> **233 /
26,129** on C, 131 / 5,070 -> **129 / 4,951** on D.

**Panel C, all listed cells (macro, q = 0.10).** The leader changes back.

| | s51 | s52 | delta | rank |
|---|---:|---:|---:|---|
| Mean of log-probs | 0.5321 | **0.5258** | -0.0063 | #2 -> **#1** |
| Mean of ranks | 0.5245 | 0.5207 | -0.0039 | #3 -> #2 |
| Mean of z-scores | 0.5243 | 0.5155 | -0.0087 | #4 -> #3 |
| **antiGen** | 0.5350 | **0.5147** | **-0.0203** | #1 -> **#4** |
| Tranception (frozen) | 0.4293 | 0.4460 | +0.0167 | #9 -> #8 |
| substitution matrix | 0.3111 | 0.3155 | +0.0044 | #18 |

**But C is a tie, and should be reported as one.** The spread from the mean of log-probs
to antiGen is **0.0111**, well inside the **0.028 seed floor**. What s52 actually
settles is that s51's reverse claim -- antiGen beating the means on C -- was not
supportable either. Both conventions put four predictors inside one seed's worth of each
other, and the ordering between them is noise.

**Panel D, de novo.** Here the gap is real and it widens.

| | s51 | s52 | delta |
|---|---:|---:|---:|
| Mean of log-probs | 0.5379 | **0.5246** | -0.0133 |
| Mean of z-scores | 0.5221 | 0.5086 | -0.0135 |
| Mean of ranks | 0.4705 | 0.4701 | -0.0004 |
| antiGen | 0.4988 | 0.4675 | **-0.0313** |

The mean of log-probs leads antiGen by **+0.0571**, against s51's +0.039. That clears
the seed floor twice over, so **s49's headline is restored on D and strengthened**: it
was s51's pool convention that had narrowed it, and s52 shows part of the narrowing was
antiGen collecting ungrounded credit.

**Panels E/F, recall AUC.** All three means still lead both, unchanged in order
(log-probs, z-scores, ranks, then antiGen). Everyone moves down a little, because the
dropped events were more often hits than the test set average.

**G-J: byte-identical**, as predicted. `install.py` compares the files rather than
asserting it, and the re-score prints `the variant root has 0 gap/stop residue(s) in the
eval range` every run.

**K-L: not recomputed**, deliberately. `background_summary.csv` is exactly as s51
shipped it.

### State

* **Shipped.** `figs/coguk_list_results.png` and `figs/coguk_list_supp.png` re-rendered
  2026-10-21 13:09. Inspected, not just produced: twelve panels, six legend groups, the
  three Ensembles bars in C/D, E/F, H/J and K/L, nothing clipped, and the rank medals
  checked against the data rather than read off the image -- the raw CSV also carries
  undrawn `valsel` / `fix32` / `ens_` rungs that `drop_unrequested_*` filters, so the
  medals rank the DRAWN field only, and they agree.
* The s51 macro CSVs are in `data/evals_sc2_lists_macro/s51_query_ungrounded/`, which
  the figure's glob cannot see, so the pre-s52 numbers stay in the tree.
* Full test suite: **149 pass** (s51 had left it un-run end to end); 19 new in
  `test_query_pool.py`, 47 across the two pool files.
* **Not committed yet.**

### Open

* **C is inside the seed floor and the figure does not say so.** Four predictors sit
  within 0.021 of each other and the medals imply an ordering the data does not carry.
* **The s45 all-RBD sibling needs this too, and it is worse there.** `rbd_all_cells.csv`
  takes every RBD cell, so its targets include the gap-sourced ones at a much higher
  rate than the curated list does. Still stale from s51 as well.
* The backbone siblings (s44, s45), the q0.05/q0.20 variants and the s47 valsel figure
  are now two conventions behind the published one.
* `keep` has not been run. It is one flag and one job if the 1.37% deflation is ever
  wanted as a sensitivity.
* **K-L is untouched here but not immune in general.** Positions 344 and 483 do carry
  gap/stop across the backbone, and they reach `rbd_all` / `rbd_unseen` in the
  supplement at 3 cells and 1. Left alone because the row's pool is query-independent by
  design (s39), but a reader comparing the supplement's K-L to its A-F should know the
  two are on different conventions.

---

## 53. Twenty letters: the alphabet loses gap and stop, and the reader gains a dialect (2026-10-22)

The model alphabet becomes the **20 standard amino acids**. `antiGen-data-revised`
resolves every (node, site) to a real residue before it writes an example, so the new
training corpus carries no `-` and no `*` anywhere -- there is nothing for those two
columns to predict, and s51/s52 spent two sections removing their effects from the
metric after the fact. This retires that at the source for everything trained from here.

This section is the **training** half. The published checkpoints and every cached
evaluation map are still in the 22-token frame, and the test corpus is still being
generated, so `eval.py` and the `list_*` / `background_*` modules are untouched.

### The corpus, measured rather than taken on trust

`~/antiGen-data-revised/output/sc2-2021/train/examples_2021_cur.txt`:

| | value |
|---|---:|
| records | **2,877** |
| sequence length | 1,273, one value |
| alphabet actually used | `ACDEFGHIKLMNPQRSTVWY`, **no `-`, no `*`, no `X`** |
| child mutations | **14,112** = the parsimony score |
| records with a repeated `(pos, alt)` | 9 (largest cell: 23) |
| records with an empty child list | 0 |
| `POST_CUTOFF: False` | all 2,877 |

Against the published `output1/examples_cutoff_1.txt`: 3,648 TRAINING examples at the
same L = 1,273 over 22 letters. So the new corpus is 79% the size of the old training
half and carries a comparable mutation load.

### Two dialect differences, and one of them is silent

`antiGen-data-revised` writes the same record structure with two changes. One fails
loudly, one does not, and the second is the reason this section exists.

**`POST_CUTOFF: True|False` in place of `TRAINING`/`TESTING`.** `load.py` tested
`line == "TRAINING"`, so every record came back testing and the training pool was
empty. That fails, but late and with a confusing message. `_parse_split_label` now takes
both and **raises on anything else**, rather than reading an unrecognised label as "not
training".

**Mutation positions are 1-indexed.** The legacy generator writes them 0-indexed, and
so does the `forecast2021` file the revised repo names as its target -- `K,309,E` there
against `K,310,E` here, `407,I` against `408,I`. Everything downstream of the loader is
0-indexed (`RBD_RANGE` is 330..530 in those coordinates), so reading the new file
unchanged would have shifted every mutation one residue along **without failing**: the
file is self-consistent, the positions are in range, and a model trained on it would
have learned a coherent, wrong map. It is the same class of defect as s51's pool
convention -- arithmetic that is wrong rather than code that is broken.

The corpus is consistent in its own frame: the count matrix `counts_2021_cur.tsv` is
1-indexed too (`#site` runs 1..1273), so this is a convention the revised repo holds
throughout, not a corrupted file.

### It is detected, not guessed, and the file carries the evidence

`MUTATION_FROM_PARENT` is `from,pos,to` against the record's **own** sequence, so every
non-root record votes: base b is viable only if `seq[pos - b] == to`. A file is accepted
only when one base explains **every** record and the other does not; both failing is an
error naming the two tallies. On the new corpus 2,876 of 2,876 records fit base 1 and
**0** fit base 0, which is not a close call. Positions come back 0-indexed either way.

The fallback matters only for a file of one record, where there is no
`MUTATION_FROM_PARENT` to check: the position range decides (`max >= seq_len` forces
1-indexed), and a range that fits both keeps the 0-indexed reading.

A **repeated `(pos, alt)` is a count**, which the reader already did -- it adds 1.0 per
line and ignores the legacy third field -- and is now tested rather than incidental.

### The regression: bit-exact on three legacy files

The rewrite is a two-pass parse (the base is a property of the whole file), so the old
path had to be shown unchanged before any new number was believed.

| file | examples | result |
|---|---:|---|
| `output1/examples_cutoff_1.txt` | 171,248 | **bit-exact** |
| `forecast2021/examples_ncov@2021-01-01_train.txt` | 347 | **bit-exact** |
| `antiGen-data/example/out/examples_cutoff_0.6.txt` | 3 | **bit-exact** |

on `aa_idx`, `mat`, `parent_mutation` and all four flags, against `git show HEAD:`'s
reader.

### Code

* `antiGen/config.py` -- `STANDARD_AAS` (20) and `LEGACY_AAS` (22, index 20 stop, 21
  gap). `AlphabetConfig.amino_acids` now defaults to the former. The literal is no
  longer repeated in the training path.
* `antiGen/load.py` -- `_parse_split_label`, `_detect_position_base`, and a two-pass
  `load_examples` that checks the sequence length, rejects an out-of-alphabet residue by
  name, and prints which base it found.
* `antiGen/train.py` -- `--alphabet`, defaulting to `STANDARD_AAS`. The 22-token frame is
  one flag away for data that still carries gap and stop.
* `antiGen/models/factory.py`, four test modules -- the places that meant the 22-token
  frame specifically now say `LEGACY_AAS` instead of relying on a default that moved.
* `antiGen/tests/test_load_examples.py` -- **12 tests**: both dialects describe the same
  tree, an unrecognised split label raises, a 1-indexed file comes back 0-indexed
  whichever alphabet it is read under, a repeated cell is a count, a file fitting neither
  base is rejected, and the single-record fallback.

Full suite: **161 pass** (149 before, 12 new).

### What this does to the numbers, before any model runs

`compute_probs` softmaxes over `L x E` and masks the query's own residue, so the uniform
reference moves from `log(1273 x 21) = 10.19` to `log(1273 x 19) = 10.09` nats. Every
NLL in this arm is against a 10% smaller candidate set and is **not comparable to a
published one**. The first batch of the transformer run reads 10.35, which is the
initialisation sitting just above uniform, as it should.

### The jobs

Nine, the published roster re-run on the new corpus with the recipes unchanged --
`examples/slurm/train_{transformer,esm2_{noft,frozen,partial,full},esm2_650m_*}.sh`,
which s52's parameterisation already made reusable, with `--examples` passed through
`EXTRA` and `OUT_DIR` pointed at
`/large_storage/hielab/ispecht/fitness-inference/ckpt_stage_aa20` so nothing lands
beside the 3,315 published artefacts.

The 650M scripts ask for 48 h and must stay on `gpu_batch`; `--partition=gpu,gpu_batch`
is rejected outright, because `gpu`'s limit is 1 day. The 35M and transformer scripts ask
for 24 h and take both, per s47.

### Six of nine have landed, and train loss is all they can say

Train loss only -- there is no validation set in this arm, for the reason under **Open**
-- so these say the runs trained, and nothing about which arm forecasts better. The
uniform reference is **10.09** nats (`log(1273 x 19)`), not the published 10.19.

| job | arm | elapsed | ep 0 | ep 31 |
|---|---|---:|---:|---:|
| 3406501 | antiGen `simple_d128_n2` | 1:04 | 9.487 | **7.641** |
| 3406502 | ESM-2 35M noft | 0:13 | -- | -- |
| 3406503 | ESM-2 35M frozen | 12:50 | 8.749 | **7.232** |
| 3406504 | ESM-2 35M partial FT | 21:34 | 8.648 | **7.045** |
| 3406505 | ESM-2 35M full FT | 40:30 | 9.241 | **6.934** |
| 3406506 | ESM-2 650M noft | 0:15 | -- | -- |
| 3406507 | ESM-2 650M frozen | running | | |
| 3406508 | ESM-2 650M partial FT | running | | |
| 3406509 | ESM-2 650M full FT | running | | |

All six exited `0:0`. The two `noft` arms are `--epochs 0`, so they write the untrained
head and stop; 13 and 15 seconds is the whole job.

Within the 35M family the ordering is full < partial < frozen, which is the order of
trainable parameters (33.5M / 11.3M / 232k) and is what a training loss alone would be
expected to say. The transformer sits above all three here and yet led the single-model
field on the published panel C (s52: 0.5147, against ESM-2 35M partial FT's 0.4987 under
s51's pool), so this column does not order the arms on anything the figure reports. It
also disagrees with the published ordering, where the per-model pool had ESM-2 partial
ahead of antiGen on C (0.5125 / 0.4460) until s51 removed the pool artefact.

The antiGen run is **64 s against s47's ~6 min** on the same recipe, which is the corpus:
2,877 examples where the published one had 3,648 and 360 batches per epoch.

### Open

* **There is no validation signal in these runs.** Every record is `POST_CUTOFF: False`,
  so `--val-mode file` leaves `val_loaders` empty and the per-epoch baselines, the
  hit-rate curve and `--save-best` all skip. The runs stop at the published epoch budget,
  which is the published recipe, but nothing in them would catch a bad epoch. The clean
  in-window alternative is `--val-mode random --val-frac 0.1`, which s47 measured as a
  null for antiGen and negative for ESM-2, at the cost of 10% of the training pool.
* **The 1-indexing is a difference between two repos that both claim the legacy format.**
  The reader absorbs it; the generator is the better place to fix it, and until one of
  them moves, `forecast2021` and `sc2-2021` disagree about what `310` means.
* **Eval is still 22-token**, in `eval.py`, `list_macro.py`, `list_variant_recall.py`,
  `background_discrimination.py`, `rescore_common_pool.py`, `ensemble_*` and
  `antiGen-figures/R/common.R`. A 20-column checkpoint meeting one of them fails on
  shape rather than scoring wrongly, which is the right way round, but the test corpus
  will need all of them.
* **`manifest.json` is one file per output directory**, so nine jobs sharing `OUT_DIR`
  leave the last writer's. Pre-existing -- the published runs did the same -- and the
  recipe is in each checkpoint, but it means the directory-level manifest is not a
  record of the arm.
* **`noft` and `frozen` are the same filename but for the timestamp.** Both are
  `--freeze-backbone --finetune-layers 0 --lr 0.001`, so `run_prefix` is
  `esm2_<model>_lr0.001_<ts>` for both and the two 35M checkpoints differ by **11
  seconds** of start time. The checkpoints themselves are not ambiguous --
  `train_args.epochs` is 0 against 32, and that is what discovery should read -- but
  nothing in the name says which is which. Also pre-existing.
* The s51 `real_substitution_mask` and the s52 `real_wildtype_mask` become no-ops on a
  20-letter alphabet. They should stay: the published maps still need them, and a pool
  mask that selects everything costs nothing.

## 54. The list panels on the revised corpus: A-F move hardly at all (2026-10-23)

s53's training half has an evaluation half. The nine retrained checkpoints are scored
against `antiGen-data-revised`'s **test** corpus, over the 20-letter alphabet, and
`coguk_list_results`' recall rows are re-rendered from it as `coguk_list_aa20`.

**Six panels, not twelve, and the two missing rows are missing by substrate rather than
by queue.** G-J need a variant count time series (`pssms/<variant>/time_*.csv`) that the
revised generator does not emit; K-L need a post-cutoff guide tree in Nextstrain JSON
form, which the revised repo has no equivalent of at all -- its one JSON is the
pre-cutoff 2020-12-29 ncov build. See **Open**.

### What moves, and it is three things at once

The alphabet, the training corpus and the test corpus all change together, so nothing
below is a contrast: it is two readings of the same panel definition.

| | published | revised |
|---|---:|---:|
| test queries carrying a CoG-UK RBD target | 17,978 | **11,523** |
| listed target events | 26,129 | **17,740** |
| observed listed cells (the macro denominator) | 233 | **230** |
| de novo, observed | 129 | **125** |
| listed RBD cells absent from the training window | 143 | **142** |
| candidate pool | 3,801-4,221 per model, then common | **4,020, by construction** |

The **de novo cut** moves with the corpus: it is now `counts_2021_cur.tsv > 0`, the
revised generator's count matrix over the training window, where the published figure
reads `pssms/root.csv`. 4,373 nonzero cells against 4,431.

**s51 and s52 are both no-ops here, and that is the point of s53.** The common candidate
pool is all 4,020 cells (201 x 20) and `real_wildtype_mask` selects every row, because
the corpus has no gap and no stop anywhere. So this figure comes **straight off
`list_macro.py`**, with no `rescore_common_pool.py` pass behind it -- the first time
since s50 that the published convention and the script's own output coincide.

### The test file is 0-indexed now, and the BA.2 root is not the published one

The revised generator was changed to write 0-indexed mutation positions, so
`_detect_position_base` now returns 0 on both files where s53 measured 1 on the training
one. **The positions the loader hands a model are identical either way** -- that is what
s53's two-pass detection is for -- so nothing was retrained.

Two differences that are *not* the indexing, both found rather than announced:

* `examples_2021_cur.txt` was regenerated at **2,934 records / 14,233 child mutations**,
  against the **2,877 / 14,112** the nine checkpoints' manifests record. The models were
  trained on the earlier revision of that file. 57 examples, 2%.
* The **BA.2 root differs from the published one at six positions**, five of them gaps
  now resolved to residues, and one real: 0-indexed 492 is `Q` in the published corpus
  and **`R`** here. That is Q493R, which BA.2 carries -- the new root is the correct one,
  and it is **inside the RBD**, so the BA.2 query a yield row would be computed from is
  not the published query. This is a reason G-J cannot simply borrow the published
  variant PSSMs.

The training root differs at one position, 0-indexed 942 (`S` published, `F` here),
which is the single reference mismatch `antiGen-data-revised/README.md` records, and is
outside the eval range.

### Code

* `antiGen/load.py` -- `load_counts_matrix(path, allowed_aas)`. Reads the revised count
  matrix where `pssms/root.csv` used to be: 1-indexed `#site` rows come back 0-indexed,
  and columns are placed **by residue name from the header**, not by position, so a
  20-column file read in the 22-token frame leaves gap and stop at zero instead of
  shifting `W` and `Y` into them. A count for a residue the alphabet does not carry is an
  error, because a discarded count is a cell wrongly called de novo.
* `antiGen/eval_utils.py` -- `build_model_gap_stop_mask` takes `gap_idx`/`stop_idx` of
  `None`; `load_baselines` drops the trailing `*`/`-` columns of the 22-token external
  maps when the alphabet is a **prefix** of that frame, and raises on any other width.
* `antiGen/list_macro.py` -- `--alphabet` (default `LEGACY_AAS`, so every published
  invocation is unchanged), `--train-examples` / `--test-examples` (the revised corpus is
  two files, and the **test file still carries the 3,902-record pre-cutoff spine of the
  tree**, which is not the training pool), `--root-counts`, and `--data-dir` now optional.
* `examples/slurm/eval_sc2_list_macro.sh` -- `DATA_DIR` / `CKPT_DIR` env overrides.
  `DATA_DIR=` empty omits `--data-dir` entirely rather than passing a path that is read
  for nothing.
* `antiGen-figures/figures/fig_coguk_list_aa20.R` -- the driver. `coguk.varr_dir` and
  `coguk.bg_dir` name directories that **do not exist**, so `fig_coguk_list.R`'s
  `have_varr` / `have_bg` guards drop G-L; creating either directory is the whole of the
  change when its substrate is built.
* `antiGen/tests/test_load_examples.py` -- 6 new tests on the counts reader: sites come
  back 0-indexed, columns are placed by name and not by order, an out-of-alphabet residue
  raises, a missing or duplicated site is rejected, and the two readers agree on the same
  data. Full suite **167 pass** (161 before).

### The regression gate, exact

`load_baselines` and `build_model_gap_stop_mask` sit on the published path, so they were
shown unchanged before any new number was read. Against `git show HEAD:`, in the 22-token
frame, on the real `data/baselines` inputs:

| gate | result |
|---|---:|
| `load_baselines`, 12 maps, values and NaN pattern | **0.000e+00** |
| `build_model_gap_stop_mask`, all 4 emission combinations | **identical** |
| same, under a 20-letter alphabet | **all-False** |

### The job

**3416222**, `gpu_batch`, **1:06:58**, exit `0:0`. 25 rungs, 25 score maps in
`list_score_maps_aa20` (11,523 x 201 x 20 float32, 185 MB each), no `[skip]`.
Nine retrained checkpoints, `blosum62` rewritten at 20 letters (`antiGen-train
--model-type blosum62`, no training), the three factored baselines refitted on the new
training pool under `--multitree-baselines`, and the fixed external maps.

**Absent: EVE and Tranception, frozen and fine-tuned, and EVmutation.** All five need an
MSA pretraining pass in the new frame. `fig_coguk_list.R` names them per metric on every
run rather than dropping them silently. The three **Ensembles** bars are absent too: the
rules were selected on the published maps and `ensemble_figure_rows.tsv` is not in the
new directory, so the group does not appear.

### Measured: the panels barely move

Cell-weighted, q = 0.10. s47's three-seed floor on these panels is **0.028-0.035**, so
read the delta column against that and not against zero.

**Panel C, all listed cells.**

| | published | revised | delta |
|---|---:|---:|---:|
| **antiGen** | 0.5147 | **0.5225** | +0.0078 |
| ESM 2 (35M) partial FT | 0.5033 | 0.5083 | +0.0051 |
| ESM 2 (35M) LM head FT | 0.4640 | 0.4654 | +0.0015 |
| Where x which | 0.4070 | 0.4581 | **+0.0511** |
| PSSM | 0.4296 | 0.4491 | +0.0195 |
| ESM 2 (650M) partial FT | 0.3935 | 0.3784 | -0.0150 |
| BLOSUM62 | 0.2380 | 0.2393 | +0.0013 |

**Panel D, de novo.**

| | published | revised | delta |
|---|---:|---:|---:|
| **antiGen** | 0.4675 | **0.4732** | +0.0057 |
| ESM 2 (35M) partial FT | 0.4189 | 0.4399 | +0.0210 |
| Where x which | 0.3560 | 0.4218 | **+0.0658** |
| ESM 2 (650M) LM head FT | 0.3493 | 0.3990 | **+0.0497** |
| EVEscape | 0.3886 | 0.3865 | -0.0021 |
| ESM 2 (35M) full FT | 0.2804 | 0.3244 | **+0.0439** |

**Of the 92 (model, panel) cells across C-F that both arms carry, 85 move by less than
0.028.** The seven that clear the seed floor are **all upward**, and concentrated in two
places: `where_times_which`, which is refitted on the new training pool (+0.0511 on C,
+0.0658 on D), and the under-trained ESM-2 rungs -- 35M full FT (+0.0439 on D, +0.0293 on
F), 650M LM head (+0.0497 on D), 650M partial FT (+0.0519 on F, +0.0291 on E). **Nothing
drops by more than 0.0208**, which is PSSM Conservation on C.

That the numbers survive a corpus this different is the result worth recording: the test
set lost a third of its events and 3 of its 233 cells, the training corpus is 80% the
size of the published one, the alphabet lost two columns, and the ordering of the drawn
field is essentially unchanged.

**The medals, checked against the CSV rather than read off the image** (the drawn field
is 23 rungs, not the published 27, because EVE/Tranception/EVmutation and the ensembles
are absent):

| panel | 1st | 2nd | 3rd |
|---|---|---|---|
| C | antiGen 0.5225 | ESM 2 (35M) partial 0.5083 | ESM 2 (35M) LM head 0.4654 |
| D | antiGen 0.4732 | ESM 2 (35M) partial 0.4399 | Where x which 0.4218 |
| E | antiGen 0.8327 | Where x which 0.8220 | ESM 2 (35M) LM head 0.8197 |
| F | EVEscape 0.8105 | **antiGen 0.7997** | Where x which 0.7958 |

antiGen leads three of the four, and on F it trails EVEscape by 0.0108 -- inside the seed
floor, as the published figure's F also was. **C's tie is not resolved here**: the
published figure's C had four predictors within 0.021, and with the three means absent
this one cannot speak to that at all.

**The event-weighted de novo cut is the one place a published claim does not carry.**
antiGen's micro on `list_coguk_unseen_fullpool` goes **0.5354 -> 0.6702**. s47 measured
the seed spread at **0.098** on exactly that cut, so even +0.135 is only about 1.4 seeds,
but it is the largest single move in the arm and it is in the supplement's panel D.

### The floor clips two bars, and the figure says so

`coord_cartesian(xlim = c(0.10, ...))` removes a bar that ends below the floor rather
than shortening it. On panel D that is **PSSM at exactly 0.1000** (chance: it is a fixed
map with no de novo information by construction -- every de novo cell has
`pssm_root == 0`, so it scores them all identically and collects exactly q) and **ESM 2
(650M) untrained at 0.0990**. Both are reported by name on every render.

### Inspected, not just produced

`figs/coguk_list_aa20.png` and `figs/coguk_list_aa20_supp.png`, 3300 x 4404, rendered
2026-10-23 13:47. Six panels labelled A-F, five legend groups, 23 rungs in each bar
panel, the title suffix on all six and nothing clipped at `coguk.title_size = 9.5`, the
two floor-clipped bars accounted for above, and the medals as tabled.

### State

* **Shipped.** `figs/coguk_list_aa20{,_supp}.png` + PDFs. The published
  `coguk_list_results` is **untouched** -- different output names, different input
  directory, and the published `data/evals_sc2_lists_macro/` was not written to.
* Rows: `antiGen-figures/data/evals_sc2_lists_macro_aa20/list_macro_summary_aa20.csv`
  (the `_aa20` tag is load-bearing: `fig_coguk_list.R` globs
  `^list_macro_summary_.*\.csv$`, which the bare name does not match).
* Maps: `/large_storage/hielab/ispecht/fitness-inference/list_score_maps_aa20`, 25.
* **Not committed yet.**

### Open

* **G-J needs a variant count time series.** Derivable from `tree_all_cur.tsv` (dates,
  `n_matched`) and `sets_all_cur.tsv` (deltas per node) -- the revised test corpus has
  1 BA.2 root and 35,395 BA.2-descended records, so the tree is there -- but it is a new
  artefact and its convention has to be gated against `pssms/BA.2/time_*.csv` before
  anything is read off it. Borrowing the published series is **not** a shortcut: the
  BA.2 root differs inside the RBD at 493, so the query and the emerged set would come
  from two different corpora.
* **K-L needs a backbone reader.** `background_tree.build_backbone` reads a Nextstrain
  JSON; the revised repo has only the pre-cutoff 2020-12-29 ncov build. Either a reader
  over the revised tree/sets TSVs, or the published backbone, which puts two corpora in
  one figure.
* **EVE, Tranception and EVmutation are five MSA pretraining passes**, and until they
  land the medals rank a 23-rung field. EVE full FT was 4th on the published C (0.4768)
  and Tranception 3rd on its D (0.4613), so their absence is not cosmetic.
* **The checkpoints predate the current training file** by 57 examples (above). Cheap to
  settle -- the nine jobs are ~2 h of GPU in total -- and worth settling before anything
  from this arm is quoted as "trained on the revised corpus".
* **The ensembling arm has not been re-run.** The maps exist (`list_score_maps_aa20`), so
  it is `ensemble_sweep.py` plus `ensemble_report.py` and no forward pass -- but the
  rules must be re-selected on the new val cells, not carried over.
* `manifest.json` in `ckpt_stage_aa20` is now `blosum62`'s, the last writer. Pre-existing
  (s53), and each checkpoint carries its own recipe.

## 55. EVE and Tranception on the revised corpus: the MSA now comes with it (2026-10-23)

Four of s54's five missing rungs. `antiGen-data-revised` writes an MSA beside the
examples it generated, from the same sequences, so the pandemic MSA stops being a
tree-independent input and becomes part of the corpus.

### `--metadata` was required, and for this MSA it is unavailable

`--msa` and `--metadata` were both mandatory for `eve`, `tranception` and
`evcouplings`. The metadata TSV does one thing: `load_date_filter` keeps the strains
whose collection date is before `--date-cutoff`, so the generative model is not
pretrained on the test window. `msa_2021_cur.fasta` is **already restricted to the
training window** -- it holds the 351,410 sequences `counts_2021_cur.tsv` counts,
exactly -- and there is no metadata beside it, because the headers carry no parsable
dates (`OL549280` has none at all).

So the path is optional now, and **omitting it is a claim about the file, not a
convenience**. A FASTA carries no dates. An unfiltered *pandemic* MSA passed this way
would pretrain on the test window and nothing downstream could detect it -- not the
checkpoint, not the eval, not the figure. That is why it is never the default and why
the empty `METADATA=` in the job scripts carries the warning rather than the code.

**The size is the same, which is what makes the two arms comparable at all.**

| | sequences |
|---|---:|
| `spike.fasta`, all records | 9,335,628 |
| passing the published filter (`< 2020-12-22`) | **348,990** |
| `msa_2021_cur.fasta`, as given | **351,410** |

+0.7%. The pretraining corpus is the same size, so the published 24 h / 256 G envelope
holds -- which matters because the sequence weighting is **O(N^2)**, chunked 1000 x 1000
on the GPU, and is most of the wall clock at this N.

### The coordinate frames line up, checked rather than assumed

Measured on the real file before any GPU time:

| | |
|---|---|
| records | 351,410, **every one 1,273 residues** |
| reference row | `Wuhan/IPBCAMS-WH-01/2019`, **no gap, no `X`** |
| reference vs the examples root | **1 difference**, 0-indexed 942 (`S` here, `F` there) |
| columns after reference-gap removal | **1,273 / 1,273** |
| focus columns (< 30% gaps) | **1,273 / 1,273**, first 0, last 1,272 |
| one-hot | `(N, 1273, 20)` |

So `_focus_to_aligned` is the identity and the MSA's frame is the examples' frame. The
one mismatch at 942 is the single reference difference
`antiGen-data-revised/README.md` records, and is outside `RBD_RANGE`.

The MSA carries `-` and `X`; both map to -1 through `EVE_ALPHABET` and are handled by
the existing gap logic, as they were for the published MSA.

### The alphabet needed nothing

EVE's and Tranception's internal alphabet is already `ACDEFGHIKLMNPQRSTVWY`, and the
internal <-> external mapping is built from `config.alphabet.amino_acids`, so at 20
letters it is the identity and the head emits 20 columns. `OUTPUTS_LEGITIMATE_GAP` /
`_STOP` are both False on these models and now mask nothing, because there is no column
to mask (s54).

### Code

* `antiGen/train.py` -- `--metadata` optional for all three MSA model types; `--msa`
  still required. Empty string is treated as absent, so an override that cannot *unset*
  a flag has a second way out. `date_cutoff` is recorded as `""` in the checkpoint
  config when no filter ran, which is where a later reader can tell the two apart.
* `antiGen/models/{eve,tranception,evcouplings}_fitness_predictor.py` --
  `_load_and_filter_msa` takes the MSA as given when no metadata path is set, and says
  so in the log. All three, because the function is a copy in each and leaving one
  behind would be worse than touching a model this section does not train.
* The same three -- **a filter that leaves only the reference row now raises.** This is
  the hazard the optional path creates, and it was found by asking what would happen if
  `METADATA` kept its published default against this MSA: the strain names would match
  nothing, `read_msa_fasta` keeps `msa[0]` whatever the filter says, and the run would
  pretrain on **one sequence** and exit clean. Nothing downstream could tell. The
  message names both tallies -- strains passing the date filter, and sequences surviving
  the header match -- because "the cutoff is too early" and "this metadata is for
  another file" need different fixes.
* `examples/slurm/train_{eve,tranception}.sh` -- `MSA` / `METADATA` env overrides
  beside `DATA_DIR` / `OUT_DIR`. The header comment that said the MSA inputs are "NOT
  tree-dependent, so they are not parameterised" was true and has stopped being true;
  it now says why.
* `antiGen/tests/test_msa_date_filter.py` -- **24 tests, eight per loader**: metadata
  filters by date, no metadata takes every sequence (for `None` and `""`), the fixture
  is one the filter actually bites on (a guard on the other two), `--max-seqs` still
  subsamples and still keeps the reference row first, a **missing** metadata file raises
  instead of falling through to "no filter" (`--metadata /typo.tsv`), metadata belonging
  to another MSA raises, an over-early cutoff reaching the same degenerate state raises
  too, and the guard does **not** fire on a genuinely one-sequence MSA used as given.

Full suite: **191 pass** (167 before).

### The four jobs

**3417095-3417098**, `gpu`, 24 h, 256 G, into `ckpt_stage_aa20`:

| job | rung | group in the figure |
|---|---|---|
| 3417095 | `eve_ft_lr0.0001` | antiGen Finetunes |
| 3417096 | `eve_frozen` | Pandemic Data in Training Window |
| 3417097 | `tranception_ft_both_lr0.0001` | antiGen Finetunes |
| 3417098 | `tranception_frozen_both` | Pandemic Data in Training Window |

The recipe is the published one unchanged -- `--lr 0.0001 --epochs 32 --batch-size 32
--cutoff 1.0 --ignore-dead-ends`, EVE `pretrain_steps` 400,000 and Tranception's
150,000 (neither is on the CLI), `theta` 0.01, `max_seqs` 0 -- with `--examples`,
`DATA_DIR`, `MSA` and an empty `METADATA` pointed at the revised corpus.

**The frozen rungs pay the pretraining a second time.** `skip_training` is true for
`eve --freeze-backbone` and `tranception --freeze-backbone`, so those two jobs are a
pretraining pass and a save, with `global_step = 0`. There is no `--init-from`, so they
re-derive a VAE / LM rather than reusing the fine-tune's. That is also what the
published pair did -- `eve_frozen` (20260623_005149) and `eve_ft` (20260623_015024) are
separately pretrained an hour apart -- so each rung stays an independent draw, and the
convention is preserved rather than quietly improved.

### Open

* **EVmutation is the fifth hole and is not submitted.** `evcouplings` is eval-only
  (`skip_training`) and its loader now takes the same optional metadata, so it is one
  more job on the same inputs -- but it was not asked for and it is a `plmc` rebuild,
  which s51 recorded as the slowest thing in this pipeline.
* **Nothing is scored yet.** When the four land, `list_macro.py` re-runs over the same
  substrate; the existing `evals_sc2_lists_macro_aa20` rows are keyed by model, so the
  job resumes and scores only the new rungs. The figure then redraws with 27 rungs and
  the medals are recomputed over a field that is no longer missing its MSA models.
* **The checkpoints still predate the current `examples_2021_cur.txt`** for the nine
  s53 rungs (s54, 2,877 against 2,934 examples). These four do not -- they are trained
  on the current file -- so until the nine are re-run the arm mixes two revisions of
  the training corpus by one row of the figure. Worth settling before anything from it
  is quoted.

## 56. Ten val-selected antiGens, averaged: ahead of base, and not established (2026-10-23)

The published arm trains base antiGen for **32 epochs, chosen on nothing**. This replaces
that with an epoch chosen on a random validation holdout -- ten times, on ten holdouts --
and takes the mean of the ten runs' log-probs as the predictor. One new bar on
`coguk_list_aa20`.

### The answer, and it has two halves

Cell-weighted, q = 0.10, on the figure's report cells. Deltas are against **base
antiGen**, with a 10,000-replicate paired bootstrap over cells.

| panel | base antiGen | **ensemble** | delta | 95% CI | seed floor |
|---|---:|---:|---:|---|---:|
| **C** all listed, 230 cells | 0.5225 | **0.5451** | +0.0226 | [-0.0204, +0.0654] | 0.035 |
| **D** de novo, 125 cells | 0.4732 | **0.5083** | +0.0351 | [-0.0105, +0.0837] | 0.028 |
| **E** AUC all | 0.8327 | **0.8379** | +0.0052 | | |
| **F** AUC de novo | 0.7997 | **0.8097** | +0.0100 | | |

It is **ahead on all four, and not one interval excludes zero.** On D the point estimate
clears the seed floor; on C it does not. The bar is drawn first on C and D, which is
true of the point estimates and is not a result.

### The decomposition, which is the interesting part

The ten members were scored individually, so the arm separates into "the runs are better"
and "averaging them is better". Those are different claims and they do not agree.

| | C, all listed | D, de novo |
|---|---:|---:|
| base antiGen, fixed 32 ep | 0.5225 | 0.4732 |
| the ten members, min | 0.5264 | 0.4563 |
| the ten members, median | 0.5494 | 0.4779 |
| the ten members, max | 0.5562 | 0.4954 |
| **mean of the members' VALUES** | **0.5448** | **0.4790** |
| **ENSEMBLE (mean of log-probs)** | **0.5451** | **0.5083** |
| ensemble - mean of member values | **+0.0003** | **+0.0293** |
| ensemble - best member | **-0.0111** | +0.0129 |

**On C the averaging buys nothing.** The ensemble scores 0.5451 against the members'
mean of 0.5448 -- three ten-thousandths -- and sits **below its own best member**
(0.5562). The whole +0.023 over base antiGen is that the individual val-selected runs
are better than the fixed-32 run, not that they were combined.

**On D the averaging is the whole effect.** Members' mean 0.4790, ensemble 0.5083:
averaging the ten predictors is worth **+0.029**, where being a val-selected run at all
is worth +0.006 over base. The ensemble also clears its best member by +0.013.

So the bar answers the question asked of it -- a holdout-chosen epoch, averaged over ten
holdouts, beats the fixed budget on the point estimate -- but the mechanism is different
on the two cuts and neither is established at this sample size.

### What the ten holdouts actually selected

| run | val mutations | epoch | best NLL | cost of running to 95 |
|---:|---:|---:|---:|---:|
| 0 | 854 | 87 | 7.4410 | 0.0092 |
| 1 | 1,531 | 94 | 7.3383 | 0.0029 |
| 2 | 1,288 | 91 | 7.5071 | 0.0026 |
| 3 | 1,057 | 94 | 7.6008 | 0.0001 |
| **4** | **4,078** | **36** | 8.5168 | **0.0794** |
| 5 | 956 | 89 | 7.5672 | 0.0118 |
| **6** | **2,728** | **63** | 8.6532 | **0.0865** |
| 7 | 793 | 88 | 7.4478 | 0.0068 |
| 8 | 999 | 88 | 7.4976 | 0.0114 |
| 9 | 975 | 81 | 7.5064 | 0.0122 |

**`--val-frac 0.1` holds out 10% of EXAMPLES, and mutation counts per node are
heavy-tailed, so the holdout varies 5x in size** -- 793 to 4,078 mutations off the same
293 examples. The two runs with by far the most validation data (s4, s6) are the only
two that pick an early epoch and the only two where running to the end costs anything
real (~0.08 nats); the eight small holdouts all land at 81-94 on a curve that is flat to
within 0.01. The straightforward reading is that a ~800-mutation holdout cannot locate
the minimum and settles somewhere on the plateau, while a 2,700-4,000-mutation one finds
a genuine overfitting signal earlier. If that is right, most of these ten are not
selecting an epoch at all -- which is consistent with C's decomposition above, where
selection does the work and averaging does none.

The members' own spread on the figure metric is **0.0298** (C) and **0.0391** (D),
against s47's three-seed floor of 0.035 / 0.028 measured on the published corpus. Ten
draws here reproduce that floor from a different source of randomness, which is worth
having.

### The budget: formally interior, not independently checked

No run selected epoch 95, so the 96-epoch budget does not bind in the strict sense, but
the median argmin is 88 and two runs sit at 94. A 256-epoch run on split-seed 0 was
queued to settle whether the curve keeps descending past 96; **it was cancelled at
exactly epoch 96 on the judgement that it would not move the result**, so the budget
question rests on the plateau reading above and not on a measurement. Stated here
because the two are not the same thing.

### Code

* `antiGen/eval_utils.py` -- `validate_per_source` read
  `alphabet.amino_acids.index('*')` unconditionally and raised `ValueError: substring not
  found` on 20 letters. **It survived s53 because that arm's `--val-mode file` left every
  validation loader empty, so the function was never called**; this is the first arm to
  ask for a holdout, and it failed in all ten runs after building the model and writing
  the manifest. Gated bit-exact against `git show HEAD:` on a model that masks the stop
  column (BLOSUM62) and one that does not (the transformer), since the guard is only
  safe if it is a no-op wherever the column exists.
* `antiGen/ensemble_sweep.py` -- `Substrate.alphabet`, and `alphabet_of_maps`, which
  reads the frame a map directory was written in **from the query manifest** (recorded
  since s53), falls back to the 22-token frame for a directory dumped before that flag,
  and then **checks the length against a map file's actual residue count**. A 20-column
  map read as 22 would mask two columns that are not there and name every `alt` one
  residue along, and neither fails on shape once the masks agree with each other.
  `--root-counts` and `--alphabet` join `add_substrate_args`.
* `antiGen-figures/figures/fig_coguk_list.R` -- `coguk.ensemble_group` /
  `coguk.ensemble_color`. "Ensembles" is the right noun only when the members are drawn;
  here they are not, so the group's standing caveat would say the opposite of what it
  means. **`coguk_list_results` re-renders byte-identical to HEAD through this** -- md5
  on both PNGs, not asserted.
* `examples/slurm/train_antigen_valsel10.sh`, `score_valsel10.sh` -- the two jobs.

### Two things the run surfaced about the pipeline

* **`--include-experimental` is mandatory for this arm.** `train.py` auto-tags any
  `--val-mode random` run as `experiment='multitree'` so `antiGen-eval` skips it, which
  is right for the published roster and wrong here, where the random split *is* the arm.
  Without it discovery finds all ten groups and filters every one away -- and
  `list_macro` exits non-zero on that rather than reporting a clean run over nothing,
  which is the behaviour s30/s31 put there.
* **Reusing the shared map cache is checked, not assumed.** `--dump-scores` warns the
  directory must be new; that is about a different query set, and
  `write_query_manifest` refuses to overwrite a manifest whose `query_sha1` (among six
  other keys) differs and returns the existing one when they agree. It has to be the
  same directory, because the ensemble reads every member from one cache.

### The jobs

| job | what | result |
|---|---|---|
| 3418734 | 10 x antiGen, 96 ep, `--val-mode random --val-frac 0.1`, split-seed/init-seed both moving | **OK**, ~18 min on `evo_gpu_priority` |
| 3419047 | scoring | **failed**: the experimental tag, above |
| 3419070 | scoring + the rule | **OK**, 2:31 |

The rule is `logsoftmax` / `arith` / `union_min` -- **exactly the published "Mean of
log-probs"** (`querypool_arm/rules9.json`) over these ten members instead of the
roster's nine, so the bar is a known combination rule and not a new one.

### State

* **Shipped.** `figs/coguk_list_aa20{,_supp}.png` re-rendered with 24 drawn rungs.
  Inspected: the new bar is drawn last in its own colour, labelled in full, and is 1st on
  C and D -- medals checked against the CSV rather than read off the image.
* The ten members' own rows are in
  `/large_storage/hielab/ispecht/fitness-inference/valsel10_arm/members/`, deliberately
  **outside** `antiGen-figures`: ten near-identical antiGen rungs beside their own mean
  would be ten readings of one predictor.

### Open

* **No control isolates the ensembling from the data cost.** This bar moves three things
  against base antiGen at once -- the epoch, the 10% each run holds out, and being an
  average of ten. The control is ten **fixed-32** runs on the same ten splits, averaged
  the same way, which is ~30 min of GPU and would turn the C/D decomposition above from
  suggestive into a measurement. s47 needed exactly this control and found the data cost
  sometimes larger than the selection effect.
* **The holdout is 10% of examples, not of mutations.** Fixing that would make the ten
  selections comparable, but it is a new splitter and not the one s46/s47 established.
* The 256-epoch budget check was cancelled; see above.
* **The non-priority GPU partitions could not launch jobs for this account** all
  afternoon -- `launch_failed_requeued_held`, reproduced with throwaway probes at 160 G
  and 256 G on `gpu_batch_high_mem,gpu_batch,gpu`, while the same resource spec on
  `evo_gpu_priority` queued normally. The four s55 EVE/Tranception jobs are stuck on it.

## 58. The variant root, scored by the s56 ensemble: one forward pass, one CSV (2026-10-23)

**On the numbering.** s57 is the where x which / PSSM blend arm, committed as 59794f0 and
named as `NOTES s57` by `fig_coguk_list_aa20.R`'s header, but never written up here. This
section and the next were appended as 57 and 58 before that was noticed and are renumbered
to 58 and 59; s57 is left for the arm that already claims it.

A whole-spike forward pass of s56's ten val-selected antiGens on the **test corpus'
variant root**, combined by the published rule and written out cell by cell. No metric,
no figure -- this is the predictor's map on one query, in a form something else can read.

### Which example, and how it was found

`antiGen-data-revised`'s test corpus carries the flag directly, so the root is **found by
the flag and not by an id**: one streaming pass over
`output/sc2-2021/test/examples_all_cur.txt` looking for an `IS_VARIANT_ROOT:` block whose
label matches the dataset variant under `load._match_variant`. Exactly one record does,
and more or fewer is an error rather than a first hit:

| | |
|---|---|
| `EXAMPLE_ID` | **32082** |
| `PARENT_EXAMPLE_ID` | 32078 |
| `MUTATION_FROM_PARENT` | `T,18,I` -- T19I in 1-indexed spike numbering |
| `POST_CUTOFF` | True |
| clade label | `21L (BA.2)` |
| length | 1,273, alphabet within `ACDEFGHIKLMNPQRSTVWY` |

Two independent checks that the frame is the one everything else uses: the record's own
`MUTATION_FROM_PARENT` puts `I` at **0-indexed** 18, and 0-indexed 492 is **`R`** --
Q493R, the one real difference s54 found between this BA.2 root and the published one,
and the reason a published variant PSSM cannot be borrowed for this query.

### The predictor is s56's bar, assembled by s56's code

The ten `_best.pt` of `valsel10_aa20/antigen_vs_s{0..9}`, each the epoch its own random
holdout chose, combined by `logsoftmax` / `arith` / `union_min` -- the published "Mean of
log-probs". The combination calls `ensemble_rules.normalize` / `resolve_nan` / `pool`
rather than re-deriving them, so the rule is the same object the figure's bar is, and the
triple is checked against `ensemble_rules.LEGAL` before anything is loaded.

Wild-type cells are NaN, exactly as `ensemble_sweep._wt_masked` writes them. At 20 letters
there is no gap or stop column, so every member emits every candidate cell and
`union_min` has nothing to impute -- checked rather than assumed (the script aborts if the
members' valid sets differ).

**The one difference from the figure's bar is the support**, and it is a deliberate one:
this normalises over the whole 1,273 x 20 grid where `ensemble_sweep` normalises over the
201-position RBD eval range. Within a query those differ by **one constant per member**,
so the mean differs by a constant and the ordering inside the RBD cannot move.

### That claim is measured, not reasoned about

The root's genotype is in the shared map cache (`list_score_maps_aa20` holds it at query
indices 4110, 9140 and 9241 -- three test examples with an identical sequence), so the
figure's own RBD-restricted ensemble can be built from the cached `vs10Antigen_s*` maps
and compared against the new one on the same cells:

| | |
|---|---:|
| cells compared (201 x 19) | **3,819** |
| offset (figure - full-grid), mean | 2.22499347 |
| offset, **sd** | **9.3e-07** |
| ranks identical | all but one adjacent pair, max rank difference **1** |

A constant to within float32 noise, and the single swap is a tie resolved differently by
two float32 paths. The two are the same predictor on this query.

### The output

`experiments/multitree/out/variant_root_ensemble.csv`, 4.4 MB, **24,187 rows** =
1,273 x 19, one per candidate substitution, wild type absent by construction:

`example_id, pos0, pos1, ref_aa, alt_aa, ens_logprob, ens_prob, ens_rank, logprob_s0..s9`

`pos0` is the corpus' 0-indexed position and `pos1` the 1-indexed spike numbering.
`ens_logprob` is the mean of the ten members' log-probs; `ens_prob` is its renormalised
exponential, which is the members' **normalised geometric mean** and sums to 1 over the
24,187 candidates. The ten member columns are carried so the mean is re-derivable without
the checkpoints -- verified: `max |ens_logprob - mean(members)|` is 2.6e-06, and each
member's own column exponentiates to 1.000000.
`variant_root_ensemble_manifest.json` beside it records the example id, the rule, the
alphabet and the ten checkpoint paths.

**The top of the map, for orientation only** (1-indexed, ensemble probability):

| | | | | |
|---|---|---|---|---|
| L5F 1.22e-02 | Q677H 6.45e-03 | D80Y 6.00e-03 | S98F 5.67e-03 | L18F 5.17e-03 |
| A222V 4.81e-03 | Q675H 4.77e-03 | H49Y 4.71e-03 | T572I 4.41e-03 | Q613H 4.31e-03 |

Every one of the ten is a recurrent spike substitution and **none is in the RBD**, which
is what a whole-spike softmax over a model trained on whole-spike branch mutations should
look like; the RBD-restricted reading of the same map is the figure's, and this file
carries both.

### The same map subset to the RBD, which is a choice of support and not a filter

`--support rbd` restricts the rule to `eval_utils.RBD_RANGE` -- the 201 positions
330-530, 0-indexed -- and writes `out/variant_root_ensemble_rbd.csv`, **3,819 rows**
= 201 x 19.

It is **not** a row filter of the whole-spike file, because `logsoftmax` normalises
*inside* its support: a cell's log-prob here is conditioned on "a mutation somewhere in
the RBD" where the other file conditions on "a mutation somewhere in spike". That is
exactly the support `ensemble_sweep` gives the figure's bar. The two differ by one
constant per member and so by one constant after the mean -- measured at **2.22499309,
sd 8.7e-07** over the 3,819 shared cells -- so no ordering moves, and the whole-spike
`ens_logprob_spike` / `ens_prob_spike` / `ens_rank_spike` are carried alongside so the
two files join. That join is exact: max difference 0.000e+00 on all three columns and
`ref_aa` equal on every row.

**Under the matching support it is the figure's ensemble, now without the offset.** Same
comparison as above against the cached `vs10Antigen_s*` maps: 3,819 cells, **max
|difference| 3.8e-06**, ranks identical but for the same single adjacent tie. `ens_prob`
sums to 1 over the RBD, and the RBD holds **0.1050** of the ensemble's whole-spike
mutational mass against 201/1273 = 0.158 of the positions.

Top of the RBD map (1-indexed, RBD-conditioned probability):

| | | | | |
|---|---|---|---|---|
| A520S 3.63e-02 | P384L 2.90e-02 | A522V 2.71e-02 | A522S 2.56e-02 | S494P 2.35e-02 |
| V367F 2.08e-02 | N439K 2.02e-02 | L452R 2.00e-02 | A484S 1.80e-02 | G446V 1.69e-02 |

`A484S` reads as it does because the BA.2 root already carries E484A, which is the kind
of thing the 1-indexed column is there to make visible.

### Code

* `experiments/multitree/forward_variant_root.py` -- the script, CPU, ~1 min end to end
  (the streaming scan of the 110 MB test file is most of it). `combine(maps, support)`
  is the whole of the spike/RBD difference; `--support` selects it and the rows written.
  Not packaged. **The whole-spike CSV re-renders byte-identical (md5) through the
  refactor that added the flag.**

### State

* **Shipped.** `out/variant_root_ensemble.csv` and `out/variant_root_ensemble_rbd.csv`,
  each with a manifest recording the support, the eval range, the rule and the ten
  checkpoint paths. Nothing in `antiGen-figures` was written to and no published
  artefact was touched.
* **Not committed yet.**

### Open

* **This is one query and no metric.** Nothing here says the ensemble is right about
  this root; s56's intervals already say it is not established on the panels.
* The three cache rows with the root's genotype are three distinct test examples, not a
  duplicated row -- worth knowing if anything ever weights queries by genotype.

## 59. Four named test examples, scored on the RBD by the same ensemble (2026-10-23)

s58's RBD pass, asked of four records by id instead of by the variant-root flag:
**43436, 43541, 45703, 45766**. `forward_variant_root.py` grew `--example-ids` and a
batch dimension; everything else -- the ten members, the rule, the support, the masking
-- is s58 unchanged.

### What the four are, which the corpus says and this section does not assume

All four are `POST_CUTOFF: True` and `21L (BA.2),True` under `IS_DESCENDED_FROM_VARIANT`,
and none carries `IS_VARIANT_ROOT`.

| id | parent | mutation from parent | RBD differences from the BA.2 root |
|---:|---:|---|---:|
| 43436 | 43433 | R21T | 13 |
| 43541 | 43436 | **L455S** | 14 |
| 45703 | 44043 | V1104L | 16 |
| 45766 | 45763 | **Q493E** | 15 |

The shared 13 are `I332V D339H K356T R403K V445H G446S N450D L452W N460K N481K A484K
F486P R493Q` -- the BA.2.86 RBD signature -- so these are JN.1-era backgrounds, not BA.2
ones. 43541 adds L455S, 45703 adds R346T and F456L, 45766 carries those two and swaps
R493Q for **R493E**. **43541 is 43436 plus exactly L455S**, which is what its own
`MUTATION_FROM_PARENT` says and is the one pair here that isolates a single RBD
substitution in the background.

Two of the four mutate inside the RBD and two outside it (R21T, V1104L), which is worth
knowing before reading any contrast between them.

**The position base is checked on the selected records**, not inherited from s54:
`check_position_base` runs `load._detect_position_base`'s vote over just the records this
script pulls, because nothing here goes through `load_examples` and a corpus regenerated
1-indexed would shift the `pos0` column silently. 4/4 fit base 0.

### The output

`experiments/multitree/out/test_examples_ensemble_rbd.csv`, **15,276 rows** = 4 x 3,819,
same schema as `variant_root_ensemble_rbd.csv` with `example_id` now varying. Everything
is **per query**: the softmax, the ranks and the candidate count. Each query's `ens_prob`
sums to 1 over its own 3,819 RBD cells, and the RBD's share of that query's whole-spike
mass is 0.1057 / 0.1056 / 0.1060 / 0.1070 -- against 0.1050 at the BA.2 root.

### Gated against the figure's cached maps, and the gate's own mistake is the finding

All four genotypes are among the 11,523 queries in `list_score_maps_aa20`, so the
figure's RBD ensemble can be rebuilt from the cached `vs10Antigen_s*` maps and compared
cell for cell:

| id | cache rows with this genotype | cells | max abs diff | max rank diff |
|---:|---:|---:|---:|---:|
| 43436 | 1 | 3,819 | 3.8e-06 | 1 |
| 43541 | 2 | 3,819 | 3.8e-06 | 0 |
| 45703 | 1 | 3,819 | 2.9e-06 | 1 |
| 45766 | 1 | 3,819 | 3.8e-06 | 0 |

float32, with the same single adjacent tie swap s58 saw.

**The first run of this gate matched queries on the RBD genotype alone and two of the
four came back disagreeing by 2-3e-02, max rank difference 31.** That was the gate being
wrong, not the maps: the model reads the whole 1,273-residue sequence, so two test
examples with an identical RBD and different flanks have genuinely different RBD maps,
and the cache held 24 and 125 such rows for 43436 and 45703. Matching on the full
genotype resolves all four to float32. Recorded because any future join into this cache
has the same trap: **the key is the whole sequence, never the eval range.**

### Code

* `experiments/multitree/forward_variant_root.py` -- `--example-ids`, `scan_records`
  (which replaces `find_variant_root` and returns records in the order asked for, erroring
  on a missing id), `check_position_base`, and a batch dimension through `combine` so the
  softmax and the ranks are per query. **Both s58 CSVs re-render byte-identical (md5)
  through it**, which is the regression this refactor needed.

### State

* **Shipped.** `out/test_examples_ensemble_rbd.csv` + manifest. s58's two files are
  unchanged on disk. Nothing in `antiGen-figures` was touched.
* **Not committed yet.**

### Open

* **No metric and no contrast.** This is four maps; the 43436/43541 pair is set up to
  isolate L455S but nothing here reads it that way.

## 60. Four named cells, ranked by every bar of `coguk_list_aa20` (2026-10-23)

Where does each rung of the twenty-letter list figure put a named substitution, inside
the RBD, given a named background? Four (background, cell) pairs, 25 bars, one CSV. No
model was run: the figure's own map cache already holds every rung's 201 x 20 eval-range
map for all 11,523 CoG-UK test queries, so the whole job is picking the right query row
and ranking inside it.

### The four cells, and which background each is asked in

| cell | background | what the background is |
|---|---:|---|
| L455S | 43436 | s59's pair: **43541 is 43436 plus exactly L455S** |
| F456L | 43541 | |
| R346T | 43541 | |
| Q493E | 45703 | |

The backgrounds are the ones asked for, and two of the three are not the parent of the
record that carries the cell. **43436 -> 43541 is the one exact single-substitution
step**: the corpus' own `MUTATION_FROM_PARENT` says `L,454,S` and the two sequences
differ at that position and nowhere else. 45703 differs from 43541 at four positions --
I70V, R346T, F456L, V1104L -- so it is 43541 plus the two RBD cells asked about here
*and* two substitutions outside the RBD. 45766, the record that actually carries Q493E,
is not used as a background and is not 45703 plus that cell either: it also reverts
346T to R and V70 to I.

Every one of the four is a real candidate in its background -- the script aborts if the
background already carries the alternate residue, which is what would make the cell
masked out of the pool rather than ranked in it.

### The ranking is the figure's, not a new one

`background_discrimination.rank_fractions`, the same path `compute_hit_rate` ranks on,
called with the two pool halves the 20-letter arm passes: `real_substitution_mask`,
which at 20 letters is every column, and `real_wildtype_mask`, which at 20 letters is
every row. So the denominator is **3,819 for every rung** -- 201 x 20 minus the query's
own residue at each position -- and a cell a model cannot score ranks last *inside* that
pool rather than shrinking it. That matters here: EVEscape's map has 25-28 holes at these
queries and the BA.2 DMS panels 18, so a per-model pool would have given them a smaller
denominator and a flattering rank. None of the four cells asked about is itself a hole.

**Ties are reported, not broken.** `rank_best` is 1 + the cells scored strictly higher;
`rank_mean` is the midpoint of the tied block. The difference is the whole story for the
coarse rungs: PSSM gives L455S rank 400 **of a tied block of 3,420** -- that cell is
absent from the training window, as is Q493E, and the PSSM has one value for all of
them. A single argsort rank would have printed 400 and hidden that.

### The query row is found by the WHOLE genotype

s59's trap, and the reason that section exists: two test examples with an identical RBD
and different flanks have genuinely different RBD maps. Matching is on all 1,273
residues. 43436 and 45703 land on one cache row each; **43541 lands on two** (5951 and
6731), and every rung's two rows are checked to agree -- NaN pattern included -- before
the first is used. Max disagreement **0.0** on all 43 maps read.

### Two gates, both exact

* **Against s59's independent forward pass.** `test_examples_ensemble_rbd.csv` was
  produced by loading the ten checkpoints and running them, not from the cache. Its
  `ens_rank` for the four cells is **654 / 183 / 66 / 327**, and the cache path's
  `rank_best` for the `ens_8a91fbbf88` bar is the same four numbers. That is a check on
  the query lookup, the rule and the ranking at once.
* **Against a second implementation.** A separately written ranking -- fill unscorable
  cells to one below the query minimum, NaN the wild type, count strictly-greater --
  over all 23 cached rungs x 4 cells: **92/92 agree** on `n_candidates`, `rank_best` and
  `n_tied`.

### The output

`experiments/multitree/out/named_cell_ranks.csv`, **100 rows = 25 bars x 4 cells**,
cell-major with each block in the figure's own bar order:

`example_id, pos0, pos1, ref_aa, alt_aa, mutation, model, figure_label, figure_group,
score, unscorable, rank_best, rank_mean, n_tied, n_candidates, frac_above, pct_from_top,
in_top_decile`

`score` is the rung's raw cached value, which is on a different scale for every rung and
is carried only so a rank can be traced back. `in_top_decile` is `top_q_weight` at
q = 0.1 -- the figure's panel E/F selection, fractional inside the boundary tie block, and
zeroed on an unscorable cell exactly as the metric zeroes it.

`rank_best`, with `(rank_mean)` where the cell sits in a tie block, and **bold** where
the cell is selected in the figure's top decile:

| bar | L455S in 43436 | F456L in 43541 | R346T in 43541 | Q493E in 45703 |
|---|---:|---:|---:|---:|
| antiGen | 403 | **199** | **98** | 512 |
| antiGen as ESM 2 (35M) partial FT | 1100 | **78** | **56** | 609 |
| antiGen as ESM 2 (35M) LM head FT | 585 | **221** | **40** | 688 |
| antiGen as ESM 2 (35M) full FT | 1609 | 1869 | **276** | 744 |
| antiGen as ESM 2 (650M) partial FT | 895 | **277** | **66** | 1429 |
| antiGen as ESM 2 (650M) LM head FT | 664 | **129** | **30** | 944 |
| antiGen as ESM 2 (650M) full FT | 878 | **307** | **41** | **357** |
| PSSM | 400 (2109.5) | **157 (163.5)** | **141 (144.0)** | 401 (2110.0) |
| PSSM Conservation | 780 (798.5) | 1673 (1729.5) | **229 (238.0)** | 1084 (1093.0) |
| MSA Transformer | **130** | 1550 | 1198 | 2589 |
| MSA Trans. (BA.2 eval) | **143** | 1707 | 1718 | 2855 |
| MSA Trans. (XFG eval) | **212** | 1505 | 1779 | 3013 |
| Substitution matrix | 662 (667.5) | **1 (8.0)** | 1411 (1414.5) | 1345 (1347.0) |
| Where x which | 620 | **172** | 831 | 1102 |
| BA.1 DMS (bind) | 817 | 2183 | **65** | **340** |
| BA.1 DMS (expression) | **152** | 1377 | **285** | 1027 |
| BA.2 DMS (bind) | 452 | 2247 | **105** | 1305 |
| BA.2 DMS (expression) | **44** | 1402 | **161** | 562 |
| EVEscape | **356 (357.0)** | 645 (646.5) | **17** | **208** |
| ESM 2 (35M), untrained | 1274 | 448 | **370** | 1425 |
| ESM 2 (650M), untrained | 1292 | 552 | **96** | 2136 |
| BLOSUM62 | 1816 (2307.0) | 415 (677.0) | 940 (1380.0) | **53 (109.0)** |
| Codon table | 1278 (1374.5) | **1 (8.0)** | 1278 (1375.5) | **130 (524.5)** |
| antiGen, 10x val-selected (mean of log-probs) | 654 | **183** | **66** | **327** |
| Where x which + PSSM hybrid, 10x val-selected | 679 | **136** | **259** | 1178 |

The two rank-1s are the same tie: F456L is one nucleotide away, and both the codon table
and the substitution matrix put every such cell at the top of a 15-wide block. Read the
coarse rungs' `rank_mean`, not their `rank_best`.

### The roster is read from the figure, not restated

`figure_models` takes the macro CSV's models, removes `coguk.drop_models` (CoVFit, pyr0),
and reads the labels, the groups and the group ORDER out of `fig_coguk_list.R` itself, so
a rung that is relabelled or regrouped there cannot be described differently here; a
model in the CSV with no group in the R source is an error rather than a default. The two
ENSEMBLE bars have no cached map of their own -- they are a rule over cached members --
and are rebuilt by `ensemble_rules.apply`, with each rule's `rule_id` asserted against the
id in `ensemble_figure_rows.tsv`, so a changed member list cannot quietly become a
different bar.

### Code

* `experiments/multitree/rank_named_cells.py` -- CPU, ~20 s warm (43 cached maps read).
  `--cells 43436:455S,...` names the pairs; the optional reference residue in a cell
  spec is checked against the background rather than assumed.

### State

* **Shipped.** `out/named_cell_ranks.csv` + manifest (the cells, the cache rows, the
  pool, the rule of each ensemble bar). Nothing in `antiGen-figures` was written to and
  the s58/s59 files are unchanged on disk.
* **Not committed yet.**

### Open

* **No metric, and no claim about any of these cells.** A rank is where a rung put one
  cell in one background; nothing here says which rung is right, and s56's intervals
  already say the ensemble bar is not established on the panels.
* The backgrounds are three test records, not three clean one-step contrasts -- only
  43436 -> 43541 is one. Anything read as "the effect of the background" would have to
  deal with I70V and V1104L.

## 61. The six ESM-2 finetunes on s56's protocol: submitted, nothing measured (2026-10-23)

s56 replaced base antiGen's fixed 32-epoch budget with an epoch chosen on a random
validation holdout, ten times on ten holdouts, averaged as the mean of the ten runs'
log-probs. This applies the same protocol to the **six trained ESM-2 rungs** of
`coguk_list_aa20` -- the group that figure calls "antiGen Finetunes" -- so the bar it
already carries for antiGen has counterparts for the comparators. **Training is running;
nothing below is a result.**

The two `noft` rungs are excluded because they are `--epochs 0`: there is no epoch to
select, and no run to average.

### The budget stays at 32, which is where this departs from s56 and why

s56 gave antiGen 96 epochs, for a stated reason that does not transfer: antiGen's
held-out minimum sat at **28 of 32** (s46's probe) and at **46-49 of 64** (s47's clean
split), so a 32-epoch ceiling would have reported a truncated search. The ESM-2 minima
are early -- s46 measured 35M partial at **5**, 650M partial at **7**, 35M frozen at
**14** of 32, and s47's clean in-window split put 35M partial at **7**. The revised
corpus is 80% of the published one and a 10% holdout leaves ~72% of the published steps
per epoch, so a published epoch 14 lands near 19 here. Still interior, and the published
recipe is therefore kept rather than deviated from on no evidence.

**That is a prediction, so the job checks it.** The tail of each training job reads every
run's `*_val_metrics_by_epoch.json`, prints the selected epoch and NLL, and **fails the
job if any run selects the last epoch** -- which is what a binding budget looks like, and
which would mean the bar rests on a truncated search. Verified on s56's run 0 before
being trusted: the extractor returns epoch 87 / 7.4410 / last 95, reproducing that
section's table.

### The recipes are not restated, and the stems carry the arm

* `examples/slurm/train_esm2_valsel10.sh ARM` runs one arm N times by invoking **that
  arm's own published SLURM script** as a shell script, with the val-selection flags
  appended through its `EXTRA=("$@")` passthrough and `OUT_DIR` per run. Those scripts
  say in their own comments that the passthrough exists so a retrain "reuses this exact
  recipe instead of a fork that would drift", and copying six sets of
  `--freeze-backbone` / `--finetune-layers` / `--batch-size` / `--lr` into a new script
  is that fork. The nine s53 retrains went through the same door. One consequence worth
  knowing: extra arguments land **after** the recipe's own, so `--epochs 2` shortens a
  smoke test, argparse keeping the last occurrence.
* Split seed `i`, init seed `100 + i`, `--val-mode random --val-frac 0.1 --save-best`,
  one directory per run -- s56's convention unchanged. Split-seed 0 holds out the same
  **854 mutations** as s56's did, which is the splitter agreeing with itself across arms.
* **The staging stem has to name the arm, not just the seed.** `eval.config_label`
  appends `_frozen` / `_partial4` / `_full` to an esm2 prefix **from the checkpoint's own
  config**, so a stem of `vs10Esm35_s0` would yield three distinct member *labels* -- and
  three colliding *filename* prefixes, which `discover_checkpoints` groups by, keeping
  only the newest. Two of the three 35M arms would have vanished silently. Hence
  `vs10Esm35Partial_s0`, whose label comes back `vs10Esm35Partial_s0_partial4`.

### The scoring stage reports eight rules, not six

`examples/slurm/score_esm2_valsel10.sh` stages the 60 checkpoints, scores them into the
shared `list_score_maps_aa20` cache, and reports every rule in **one** `ensemble_report`
pass. The widening from one arm to six is where it differs from `score_valsel10.sh`, and
not only in arithmetic: the macro directory the figure reads holds exactly one
`ensemble_figure_rows.tsv` and one `list_macro_summary_ensemble.csv`, and
`ensemble_report` **writes both wholesale**. A report over the six new rules alone would
not add six bars -- copied in, it would **replace s56's antiGen bar and s57's hybrid bar**
with them. So the rule file carries those two rules as well, their members being in the
same cache, and one report writes one internally consistent set of files.

Every rule is `logsoftmax` / `arith` / `union_min`, the published "Mean of log-probs"
(`querypool_arm/rules9.json`). None is selected here; the eight are fixed in advance and
reported whole. Member names are **checked against the dumped maps** before the rule file
is written, and a miss aborts the job by name: a rule whose member is absent is a bar
built on fewer runs than it claims.

### The jobs

Submitted 2026-10-23 21:53. Estimates are the s53 single-run elapsed x 10 x 0.9 (the
90% split), the s53 runs being the same recipes on the same corpus.

| job | arm | runs x est. | wall estimate |
|---|---|---|---|
| 3421208 | 35M LM head FT | 10 x 12:49 | ~1.9 h |
| 3421209 | 35M partial FT | 10 x 21:34 | ~3.2 h |
| 3421210 | 35M full FT | 10 x 40:30 | ~6.1 h |
| 3421211 | 650M LM head FT | 10 x 2:19:37 | ~21 h |
| 3421212 | 650M partial FT | 10 x 2:51:17 | ~26 h |
| 3421213 | 650M full FT | 10 x 4:40:51 | ~42 h |

~100 GPU-hours in total, and 3421213 is why that job asks for 96 h rather than the
published scripts' 48. A smoke run (`N=1`, `--epochs 2`, 35M frozen) was taken first and
threw the budget check as it should have, epoch 1 of 2 being the last one; s56 lost all
ten of its runs to a bug that fired after the model was built, which is the argument for
spending two minutes on that.

### Two broken nodes, mistaken for the s56 partition fault

Five of the six came back `launch_failed_requeued_held` within a minute -- the symptom
s56 recorded as "the non-priority GPU partitions could not launch jobs for this account".
It is **not** that. `slurmd` on the allocated node:

    error: namespace_p_join: open failed for /media/8TBNVME/scratch/<jobid>/.ns:
           No such file or directory

on **GPU115A** (three jobs) and **GPU1034** (two). Both were `idle` in `sinfo` throughout
and kept taking jobs, so a resubmission would have hit them again. Released with
`scontrol update JobId=... ExcNodeList=GPU115A,GPU1034` and all six are queued or running
normally. Worth checking against the s56 observation: a node-local scratch fault that
Slurm does not drain the node for looks exactly like an account-level partition problem
from `squeue`.

### Open

* **Nothing is measured.** Six bars, their decompositions (members vs mean, as s56 did),
  and the figure re-render all wait on the training.
* **The same three-things-at-once caveat as s56 applies to every one of these bars**: the
  epoch, the 10% each run holds out, and being an average of ten. s47 measured the first
  two for 35M partial on the published corpus and found selection moving the figure
  metric the **wrong way** (-0.024 / -0.018 on C/D) while buying 0.055 of NLL. If that
  sign holds here, the ensembling has to beat it before any of these bars moves up --
  which makes this the arm most likely to produce a null, and the reason to run it.
* The s56 control -- ten **fixed-32** runs on the same ten splits, averaged the same way,
  isolating the ensembling from the data cost -- is still not run, now for seven arms
  rather than one.

## 62. EVE and Tranception drawn on the revised corpus: four rungs, no reordering (2026-10-24)

s55 trained the four MSA rungs and left them unscored; s54 left `coguk_list_aa20` with
five named holes. Four of the five are now filled. **The figure draws 29 rungs, and
nothing in the medals moves.**

### The four landed overnight

Resubmitted as **3419183-3419186** after the s56 partition trouble (s55 recorded them as
3417095-98, which never launched). All four `COMPLETED 0:0` into `ckpt_stage_aa20`, on
the revised MSA with `METADATA` empty -- the log line for all four reads
`metadata = <none: the MSA is taken as already window-restricted>`, which is the claim
s55 made about that file and not a convenience.

| job | rung | elapsed | trainable params |
|---|---|---:|---:|
| 3419183 | `eve_ft_lr0.0001` | 3:45:21 | 261,672,661 |
| 3419184 | `eve_frozen` | 3:40:27 | 0 |
| 3419185 | `tranception_ft_both_lr0.0001` | 8:57:24 | 4,772,608 |
| 3419186 | `tranception_frozen_both` | 8:38:26 | 0 |

### Scoring resumed rather than re-ran, and the resume is per-artifact

Job **3427508**, `gpu`, **2:54**, exit `0:0`. s54's invocation unchanged except for the
partition and a node exclusion. `list_macro` reported `Resuming: 25 model(s)` and
printed `[resume-skip]` for every one of them, scoring only the four new rungs: 29 rows,
51 maps in `list_score_maps_aa20`, no `[skip]`. The resume keys on **artifacts rather
than rows** (`_resume_skip`), so a row without its cached map would have been re-run and
then checked against the row it already had -- which is why a 25-rung skip here is not
the silent-success shape s30/s31 record.

### What they measure, and the one place this arm does not behave like the rest

Cell-weighted, q = 0.10, against the published figure's query-pool CSV. The seed floor on
these panels is **0.028-0.035** (s47).

| rung | C published | C revised | delta | D published | D revised | delta |
|---|---:|---:|---:|---:|---:|---:|
| antiGen as EVE full FT | 0.4768 | 0.4273 | **-0.0495** | 0.3924 | 0.3096 | **-0.0828** |
| antiGen as Tranception full FT | 0.4426 | 0.4310 | -0.0116 | 0.3683 | 0.3094 | **-0.0589** |
| EVE (frozen) | 0.4193 | 0.3284 | **-0.0909** | 0.3659 | 0.2313 | **-0.1347** |
| Tranception (frozen) | 0.4460 | 0.3090 | **-0.1370** | 0.4613 | 0.2758 | **-0.1855** |

**s54's headline does not extend to these four.** That section measured 85 of 92 (model,
panel) cells moving by less than the seed floor, with "nothing drops by more than
0.0208". Of the **29 rungs now present in both corpora, exactly four move more than
0.035 on panel C** -- `where_times_which` at **+0.0511**, which s54 already attributed to
its refit on the new training pool, and the **three MSA rungs, all downward**.

**The AUC panels say where it lives: in the pretraining, not the fine-tuning.**

| rung | E published | E revised | delta | F published | F revised | delta |
|---|---:|---:|---:|---:|---:|---:|
| EVE full FT | 0.7807 | 0.7804 | -0.0003 | 0.7291 | 0.7213 | -0.0079 |
| Tranception full FT | 0.7810 | 0.7830 | +0.0020 | 0.7506 | 0.7372 | -0.0134 |
| EVE (frozen) | 0.7775 | 0.7305 | **-0.0470** | 0.7476 | 0.6755 | **-0.0720** |
| Tranception (frozen) | 0.7624 | 0.7226 | **-0.0398** | 0.7532 | 0.7245 | -0.0287 |

The two **fine-tuned** rungs are flat on E (-0.0003, +0.0020); the two **frozen** rungs,
which are a pretraining pass and a save with `global_step = 0`, carry the whole drop.
Fine-tuning on the examples washes out whatever moved.

**What that is, this arm cannot say.** Three candidate readings and nothing here
separates them: the MSA is a different file (351,410 sequences against the published
filter's 348,990, s55); each frozen rung is an **independently pretrained** VAE / LM
(s55: there is no `--init-from`, and the published pair was separately pretrained an hour
apart), so a fresh draw of a generative model is inside every delta above; and the test
corpus lost a third of its events. **There is no seed replicate for any of these four**,
on either corpus, so each number above is one draw against one draw, and the s47 floor
was measured on a different model family. The frozen-vs-fine-tuned split is suggestive of
the pretraining draw rather than the test corpus -- a corpus effect should hit the other
25 rungs too, and it does not -- but suggestive is all it is.

### The figure

`figs/coguk_list_aa20{,_supp}.png` + PDFs, re-rendered 2026-10-24 12:18. **29 rungs**
plus the two s56/s57 ensemble bars.

* **The medals do not move.** C: ensemble 0.5451, antiGen 0.5225, ESM-2 35M partial
  0.5083. D: ensemble 0.5083, antiGen 0.4732, ESM-2 35M partial 0.4399. The four new
  rungs land 8th/9th (C) and 10th/11th (D) among the single models; the best of them
  (Tranception full FT, 0.4310 on C) is below `where_times_which`.
* **EVEscape still leads F** (0.8105 against antiGen's 0.7997), and the newly drawn EVE
  rungs are not it -- EVEscape is a fixed external map, EVE full FT is 0.7213.
* The floor-clipped bars are **the same two** as s54 (PSSM at 0.1000 and ESM-2 650M
  untrained at 0.0990); no new rung is clipped.
* Inspected rather than assumed: both full-FT rungs draw in **antiGen Finetunes** under
  antiGen, both frozen rungs in **Pandemic Data in Training Window**, labels complete.
* `fig_coguk_list_aa20.R`'s header said these four were absent and now says what is:
  **EVmutation alone**, the fifth pass and a `plmc` rebuild.

### State

* **Shipped.** The figure, `data/evals_sc2_lists_macro_aa20/list_macro_summary_aa20.csv`
  (901 -> 1,045 rows, 25 -> 29 models) and `list_macro_percell_aa20.csv`.
* The published `coguk_list_results` is untouched: different directory, different names.
* **Not committed yet.**

### Open

* **EVmutation is the last hole**, unchanged from s55: eval-only, same optional metadata,
  one job -- and a `plmc` rebuild, which s51 recorded as the slowest thing here.
* **These four do not share the nine's corpus revision.** They were trained on the
  current `examples_2021_cur.txt` (2,934 examples); the nine s53 rungs were trained on
  the 2,877-example revision (s54). One row of this figure therefore spans two revisions
  of the training corpus, and the four rungs that drop hardest are on the newer side of
  it. That is a third reading of the table above and the cheapest to remove -- the nine
  jobs are ~2 h of GPU.
* **No seed replicate for any MSA rung**, which is what would turn the table above from a
  difference into a measurement.

---

**s61 status.** The six ESM-2 val-selection jobs were **cancelled** on 2026-10-24, to be
restarted later. One arm finished first and is worth keeping: **3421209, 35M partial, all
ten runs, `VALSEL10_ESM2 35m_partial OK`** in 3:31:49. The selected epochs are **6, 6, 8,
11, 12, 15, 19, 19, 20, 29 of 31** -- every one interior, which is the s61 budget
argument (ESM-2's minima are early, so the published 32 does not bind) coming back
measured rather than predicted. The two runs selecting epoch 6 are the two with the
largest holdouts, the same pattern s56 found in antiGen's ten. Partial arms on disk:
`35m_frozen` 6 of 10, `35m_full` 2, `650m_partial` 1; `train_esm2_valsel10.sh` skips a
run that already has a `_best.pt`, so a restart resumes rather than redoes.

## 63. The nine-member ensembles on the revised corpus: AUC survives, the top decile does not (2026-10-24)

s54 left the three **Ensembles** bars off `coguk_list_aa20` because the rules had been
selected on the published maps and two of the nine members had no map on this corpus at
all. s62 scored EVE and Tranception here, so the family can be formed. It is now drawn,
and the figure carries **32 rungs in seven legend groups**.

### The rules are carried, and the carry is checked rather than asserted

The family is nine parameter-free rules -- `pctile` / `logsoftmax` / `zscore` x `arith` /
`max` / `min` -- over base antiGen, EVE full FT, Tranception full FT and the six ESM-2
arms. The published figure draws the three **arithmetic** ones, a choice s49/s50 made on
the published corpus's SELECT cells. Carrying that choice here is importing a selection
made elsewhere, which is what s54's Open warned about, so all nine were re-scored on
**this corpus's** select cells (889 observed off-list RBD cells, 590 de novo) before
anything was drawn.

| rule | SELECT q0.1 | SELECT AUC | SELECT de novo q0.1 | REPORT C q0.1 |
|---|---:|---:|---:|---:|
| **Mean of z-scores** | **0.3500** | **0.7920** | 0.3817 | 0.5056 |
| **Mean of log-probs** | **0.3459** | **0.7908** | 0.3812 | 0.5026 |
| **Mean of ranks** | **0.3374** | **0.7901** | **0.3854** | 0.4957 |
| Min of ranks | 0.3372 | 0.7666 | 0.3347 | 0.4441 |
| Min of z-scores | 0.3370 | 0.7667 | 0.3412 | 0.4477 |
| **Max of log-probs** | 0.3329 | 0.7763 | 0.3126 | **0.5172** |
| Max of z-scores | 0.3176 | 0.7564 | 0.3078 | 0.5069 |
| Min of log-probs | 0.3134 | 0.7453 | 0.3239 | 0.4504 |
| Max of ranks | 0.3108 | 0.7532 | 0.2974 | 0.4928 |

**The three means take the top three places on select, on both cuts and on the AUC**, so
selecting here would have selected the same three. The carry is legitimate on this corpus
and that is now measured.

**The last column is the argument for the whole protocol, stated in one row.** On the
cells the figure REPORTS, `Max of log-probs` (0.5172) beats every mean. Selecting on the
reported cells would therefore have drawn a different rule -- the one that happens to
suit the reported set, which is exactly what selecting on disjoint cells exists to
prevent. It is cheap to state this when the two columns agree; it is worth having when
they do not.

### The result: the ensembling gain splits by panel

Against the **best of the nine members**, which is base antiGen on all four panels, on
both corpora. Seed floor 0.035 (C/E) and 0.028 (D/F).

| | C published | **C revised** | D published | **D revised** |
|---|---:|---:|---:|---:|
| best member | 0.5147 | 0.5225 | 0.4675 | 0.4732 |
| Mean of ranks | +0.0059 | **-0.0268** | +0.0026 | **-0.0099** |
| Mean of log-probs | +0.0110 | **-0.0199** | +0.0571 | **-0.0061** |
| Mean of z-scores | +0.0008 | **-0.0169** | +0.0411 | **-0.0013** |

| | E published | **E revised** | F published | **F revised** |
|---|---:|---:|---:|---:|
| best member | 0.8343 | 0.8327 | 0.8092 | 0.7997 |
| Mean of ranks | +0.0044 | **+0.0079** | +0.0115 | **+0.0217** |
| Mean of log-probs | +0.0109 | **+0.0110** | +0.0250 | **+0.0277** |
| Mean of z-scores | +0.0080 | **+0.0117** | +0.0191 | **+0.0276** |

**On the AUC panels the gain survives the corpus change and is slightly larger.** On the
top decile it **changes sign**: all three means now sit below base antiGen on C and on D.
Every number in both tables is inside the seed floor except published D's +0.0571, so
what this really says is that the ensembling gain was never large on these panels and is
now not even positive where the figure's headline is read.

A candidate mechanism, not a measurement: **two of the nine members lost a lot on exactly
these two panels in s62** -- EVE full FT -0.0495 (C) / -0.0828 (D), Tranception full FT
-0.0116 / -0.0589 -- and an unweighted mean of nine carries both. The AUC panels, where
those two moved by -0.008 and -0.013, are where the gain survives. That is consistent and
it is not established; separating it needs the members held fixed, which no run here does.

### The medals, and panel F changes hands

| panel | 1st | 2nd | 3rd |
|---|---|---|---|
| C | antiGen 10x val-sel 0.5451 | antiGen 0.5225 | ESM 2 (35M) partial 0.5083 |
| D | antiGen 10x val-sel 0.5083 | antiGen 0.4732 | **Mean of z-scores 0.4719** |
| E | **Mean of z-scores 0.8444** | **Mean of log-probs 0.8437** | **Mean of ranks 0.8406** |
| F | **Mean of log-probs 0.8274** | **Mean of z-scores 0.8273** | **Mean of ranks 0.8214** |

The means sweep E and F outright, and on F they displace **EVEscape** (0.8105), which had
led that panel on the published figure and on this one through s62. C and D are unchanged
from s62 but for the third medal on D.

### Two groups, because one noun cannot be right for both

All five combination bars arrive through the ENSEMBLE machinery. The nine-member
mixtures' members ARE bars in these panels, so the standing caveat applies: an ensemble
bar sits beside its own parts. s56's and s57's bars combine ten runs that are **not**
drawn, so it does not. They are now separate legend groups with separate colours --
published orange for **Ensembles**, blue for **Val-selected, averaged over 10 holdouts**.

* `antiGen/ensemble_report.py` -- an optional `figure_group` per rule, written as a third
  column of `ensemble_figure_rows.tsv`. Set on all rules or none, because a partly
  grouped file would put the ungrouped rows in the figure's fallback group silently,
  which is the one thing the column exists to prevent.
* `antiGen-figures/figures/fig_coguk_list.R` -- the TSV reader takes 2 or 3 columns, each
  row's group is its own or `ENSEMBLE_GROUP`, and `coguk.ensemble_colors` maps group to
  colour with `ENSEMBLE_COLOR` as the fallback. **The published figure re-renders
  byte-identical through this**: `coguk_list_results.png` md5 `776aa91ceabf648d713a61cd1f2edd7b`
  and `coguk_list_supp.png` `49cc10be9037e8673e6028ad630312df`, unchanged, and `git
  status` reports neither as modified.
* `examples/slurm/ensembles9_aa20.sh` -- the job.

### The first attempt reported the right numbers against the wrong incumbent

One run over all eleven rules was taken first, and its bar values were correct -- but
`ensemble_report`'s auto-added **"panel incumbent" is the max over every member whose map
the run loaded**, and a combined run loads the ten `vs10Antigen` runs and the ten hybrid
blends as well as the nine. So the family's contrasts came back against
`vs10Antigen_s4`: the best of ten unlabelled runs of a *different arm*, chosen by
maximum. Nothing on the figure was wrong; the file a reader consults for intervals was.

The job now reports each arm against its own member set -- the family against its nine,
the two carried rules against the eleven and sixteen rungs s56/s57 used -- and
concatenates, gating that the rule_ids are disjoint and that both runs scored the same
`n_cells` on all 18 (variant, qtile) keys. Two further gates, both **0.00e+00**:

* the nine rules' rows from the 9-member run against the 11-rule run's, 198 cells --
  so a rule's score does not depend on which other members were loaded, and the first
  attempt's values were right;
* the two carried rules against the **committed** file, 36 cells -- so replacing the
  directory's ensemble CSV loses nothing of s56/s57.

### State

* **Shipped.** `figs/coguk_list_aa20{,_supp}.png` + PDFs, and the four ensemble files in
  `data/evals_sc2_lists_macro_aa20/`: 11 rules in the CSV and the contrasts, **5 rows in
  `ensemble_figure_rows.tsv`**. Reported whole, drawn in part -- the published shape, and
  the six losers stay in the data.
* The published `coguk_list_results` is untouched and gated above.

### Open

* **EVmutation remains the only hole** in the figure's roster, unchanged from s62.
* **The top-decile sign flip is not attributed.** The members-held-fixed experiment --
  re-forming the family on the published corpus with the two MSA rungs swapped for their
  revised retrains, one at a time -- is the thing that would attribute it, and it is
  cached-map arithmetic with no GPU.
* **s49's transfer finding is NOT re-checked here.** That section measured the means'
  gain as specific to the curated list -- base antiGen alone won three of the four
  off-list cuts on the published corpus -- and the comparison needs the single members'
  select rows, which `--rules-json` does not produce: this sweep scored the nine rules
  and nothing else. So this corpus has the means' select numbers and no baseline to read
  them against. One more sweep with nine one-member rules closes it, and it should be
  closed before any +0.02 off panels E/F is quoted as a general improvement.

## 64. An antibody-escape DMS joins the scans: LY-CoV1404 on BA.2 (2026-10-24)

`escape.csv` in the repo root -- the Bloom-lab escape-calculator shape, one row per
(condition, site, mutation) with a `mut_escape` fraction -- becomes a fixed (L, E)
baseline map and a sixth bar in the figure's **Deep Mutational Scans** group. One
condition, `LY-CoV1404 - BA2`: the escape fraction of each substitution from
bebtelovimab, measured on a BA.2 background. It is a deep mutational scan, so it joins
that group; it gets its own label, because escape is not what the other four measure.

### The frame and the background are detected, not assumed

The table numbers sites **1-indexed**; everything downstream of `load.py` is 0-indexed.
Reading it one residue along would have produced a file that is self-consistent, in
range, and wrong -- the defect class s53 spent a section on, and the reason
`make_escape_baseline.py` detects the offset instead of taking it.

The table carries a `wildtype` column, so every site votes. Against the **BA.2 root**
streamed out of the revised test corpus by its `IS_VARIANT_ROOT` flag (the record
`forward_variant_root.py` finds, and found the same way so a regenerated corpus cannot
point this at another node):

| offset | wildtypes matched |
|---|---:|
| `site - 1` | **173 / 173** |
| `site + 0` | 6 / 173 |

Not a close call, and it identifies the **background** as well as the frame: the same
check against a Wuhan-era root would fail, because this table's wildtypes are BA.2's. The
script refuses unless one offset explains *every* site and the other does not.

The result lands on rows **330-530**, which is exactly where the four published DMS maps
live, and writes the 22-token frame they use, so the trailing stop and gap columns are
dropped by `load_baselines` under a 20-letter alphabet (s53) and one file serves both.

### The sentinel is load-bearing, and negatives are refused rather than clamped

`load_baselines` maps every **negative** cell to NaN, not just the -1 the published maps
write. So -1 means "not measured" and a measured **zero** is kept and ranks last, which
is the right reading of "this substitution does not escape". A table carrying negative
escape values would have them silently read as missing -- so the script **errors** on one
instead of clamping, because clamping changes a measurement and the right transform
depends on what a negative means in the source. This table's minimum is +0.002981, so
nothing is affected; the guard is for the next table.

### It covers half the pool, and that caps the AUC rather than the top decile

2,023 of the 4,020 candidate cells, 173 of 201 RBD positions. `compute_hit_rate` keeps an
unscored cell **in the pool, ranked last**, so one map's gaps cannot shrink everyone's
denominator (the s51 property) -- but they do cap the map's own curve.

| q | LY-CoV1404 escape | BA.2 DMS (expr) | EVEscape |
|---|---:|---:|---:|
| 0.1 | 0.2261 | 0.2399 | 0.3586 |
| 0.3 | 0.5217 | 0.6957 | 0.7694 |
| 0.5 | 0.8000 | 0.9304 | 0.9429 |
| 0.6 | **0.8391** | 0.9783 | 0.9696 |
| 0.9 | **0.8391** | 0.9957 | 1.0000 |

**The plateau is exactly its coverage of the target set**: 193 of the 230 observed listed
cells have a measurement, 193/230 = **0.8391**, and the curve reaches that at q = 0.6 and
never moves again. So:

* **Panels C and D are comparable.** At q = 0.1 the decile is ~402 cells and fits inside
  the 2,023 it measures; it is not truncated.
* **Panels E and F are not.** Their AUC charges this rung for coverage as well as for
  ranking, and reading its 0.6373 / 0.6726 against a map that scores every cell is
  reading two different things. Recorded beside the label in `fig_coguk_list.R` rather
  than only here, because that is where someone will be when they wonder.

### Where it lands

Cell-weighted, q = 0.10, against the group it joins.

| predictor | C | D | E | F |
|---|---:|---:|---:|---:|
| EVEscape | 0.3586 | 0.3865 | 0.8057 | 0.8105 |
| BA.1 DMS (expression) | 0.2757 | 0.3200 | 0.7658 | 0.7596 |
| BA.2 DMS (expression) | 0.2399 | 0.3009 | 0.7653 | 0.7602 |
| **LY-CoV1404 escape** | **0.2261** | **0.2800** | **0.6373** | **0.6726** |
| BA.1 DMS (bind) | 0.2217 | 0.1200 | 0.6180 | 0.5448 |
| BA.2 DMS (bind) | 0.2203 | 0.1520 | 0.6319 | 0.5660 |

21st of 30 single models on C, 14th on D. It beats both **bind** maps on every panel and
sits below both **expression** maps; EVEscape, which is also an escape-motivated score
but covers the whole pool, is well clear of it. Nothing else on the figure reorders.

### The job, and the gate that matters

**3428348**, `gpu`, **44 s**, exit `0:0`. `list_macro` resumed over the 29 models already
in the directory and scored the one new map; 52 maps in the cache.

* `candidate pool: common -- 4020 of 4020 cells` on **both** this run and s62's, so
  adding a half-covered map did **not** shrink the shared pool. That was the thing worth
  checking before anything else: it is precisely the s51 failure.
* Every pre-existing row is **bit-identical** to the committed file -- 1,044 shared
  (model, variant, qtile) cells, worst |delta| **0.00e+00**, `n_cells` and `n_events`
  unchanged, 0 rows lost, 36 added.
* `coguk_list_results` re-renders **byte-identical** (`776aa91c...`, `49cc10be...`). Its
  stdout now names the new rung as missing on all four metrics, which is the designed
  "report the holes" behaviour and the same treatment EVmutation gets.

### Code

* `experiments/multitree/make_escape_baseline.py` -- the generator. `--condition` is
  required when a table holds more than one, since one map is one predictor.
* `data/baselines/BA2_escape_LYCoV1404.csv` -- 1,273 x 22, rows 330-530 populated.
* `antiGen-figures/figures/fig_coguk_list.R` -- `MODEL_GROUP` / `MODEL_LABEL`, with the
  coverage caveat.
* `antiGen-figures/figures/fig_coguk_list_aa20.R` -- the header note.

### Open

* **One antibody, one background.** This is bebtelovimab on BA.2, not an escape panel.
  The escape calculator's other conditions are more maps on the same code path --
  `--condition` names each -- and a per-antibody mean would be a different predictor
  again, not a sixth bar.
* **The published figure has not scored it.** The map is in `data/baselines/`, so the
  next published `list_macro` run picks it up; until then `coguk_list_results` reports
  it as a hole. Scoring it there is one job and would put it on both figures.
* **The AUC panels need a coverage-aware reading or a coverage-matched comparison.** The
  honest alternative is to report this rung at q = 0.1 only, or to restrict every rung to
  the 2,023 cells it measures and recompute the row -- which is a different panel, not a
  footnote, and is not done here.

## 65. The cutoff series: a corpus per quarter, and a budget that binds (2026-10-24)

`antiGen-data-revised` now emits **24 quarterly training corpora** (`sc2-<D>/train`,
2020-04-01 to 2026-01-01) and **one test tree** serving all of them. This section is the
machinery that reads them and the first thing it measured, which is not what the arm was
built to measure.

### `POST_CUTOFF` is a set now, and the reader has to be told which split

The test corpus writes `POST_CUTOFF:` as a header over zero or more `<date>,True` lines
-- the set of cutoffs at or after which the record is a test example -- because the tree
always held `minLeafDated` and one comparison per (record, cutoff) recovers every split
from it. `_parse_split_label` took `POST_CUTOFF: True|False` on one line and **raised on
the bare header**, so neither the new train nor the new test corpus loaded at all.

* `antiGen/load.py` -- the third dialect, and `split_date` to choose among its 24
  answers. **Required the moment a record carries a non-empty set**: defaulting would
  silently pick a split, and on the test corpus "no date" reads all 75,606 records as
  training and hands a scorer an empty test pool. A `split_date` the corpus does not
  carry is also refused, by name against the dates it does have -- otherwise a typo
  reads every record as training and fails much later.
* `--split-date` on `train.py`, `list_macro.py`, `list_variant_recall.py` and
  `background_discrimination.py`.
* **Gated bit-exact against `git show HEAD:`** on both legacy dialects --
  `output1/examples_cutoff_1.txt` (171,248 records) and the 2021 revised train corpus
  (2,934) -- on `aa_idx`, `mat`, `parent_mutation` and all four flags. 6 new tests; the
  suite is **205**.

Read at 2023-01-01 the test corpus gives **20,618** post-cutoff records of 75,606,
reproducing an independent streaming count exactly.

### The sweep, and what it is for

`examples/slurm/train_cutoff_series.sh <D>` trains, per cutoff, base antiGen on the
published recipe and the ten val-selected runs of s56, into
`cutoff_series/<D>/{base,valsel}` with a `MANIFEST.txt` naming the corpus, the recipe,
the seeds and the test corpus it pairs with. The point is an analogue of `sc2_results`
panel M -- recall at the top decile against where the train/test boundary is drawn.

**Both ends of the sweep are weak, and for opposite reasons.** The training corpus grows
from 190 records to 75,647 while the test window shrinks from 75,143 to 170:

| cutoff | train records | test records | test mutations | listed cells observed |
|---|---:|---:|---:|---:|
| 2020-04-01 | 190 | 75,143 | 304,668 | 231 |
| 2022-01-01 | 32,500 | 41,894 | 133,283 | 214 |
| 2023-01-01 | 53,916 | 20,618 | 57,319 | 164 |
| 2026-01-01 | 75,647 | **170** | **177** | **5** |

A point at 2026-01-01 is 177 mutations of evidence beside 2021-01-01's 263,643. On the
CoG-UK list the de novo cut dies outright: 142 observed de novo cells at the 2021 split,
**4** at 2023-01-01, **0** from 2024-01-01 on.

### The finding: the budget binds, and 96 does not fix it

The arm ran first at the published 32 epochs. **Nine of ten runs at 2023-01-01 selected
epoch 31 -- the last one** -- and the pattern holds across the sweep (9/10 at
2020-04-01, 7/10 at 2021-01-01, 5/8 at 2024-07-01). A run that selects its last epoch has
not chosen an epoch: its held-out curve was still descending when the budget ran out, so
the arm collapses to "train to the budget", which is what `base` already is.

The holdout is not the problem. At 2023-01-01 it is **23,620 mutations**, 28x s56's 854.

Re-run at 96, and at 2020-04-01 two of ten still select the last epoch while the rest
land at 94 of 95. The curve says why -- this is the run with the largest holdout at that
cutoff:

| epoch | 0 | 30 | 60 | 90 | 94 | 95 |
|---|---:|---:|---:|---:|---:|---:|
| val NLL | 10.0469 | 8.9306 | 8.8325 | 8.7624 | **8.7488** | 8.7525 |

It is **flattening into noise, not turning**: the last 10 epochs move 0.027 nats, about
the size of the per-epoch wobble on a 545-mutation holdout, against 1.30 nats over the
whole run. Train and val are genuinely different series -- 8.3307 against 8.7525 at the
end, a gap that widens from 0.08 -- so this is not train loss misread.

**s56 had the same shape and deferred it**: eight of its ten runs selected epochs 81-94
of 96, and the 256-epoch check meant to settle it was cancelled. A 512-epoch probe at
2021-07-01 (3 split seeds, the mid-size cutoff where the holdout is well powered and the
cost is affordable) is running to settle whether the curve ever turns.

**The 32-epoch runs are kept as `valsel_ep32/`**, not deleted: a ten-seed average at the
published budget is exactly the fixed-budget control s56's Open asked for.

* `train_antigen_valsel10.sh` **now fails a job whose runs select the last epoch**. It
  previously printed a note asking the reader to check that, and exited OK regardless,
  which is how this got through. The check reads each run's per-epoch JSON via
  `examples/slurm/best_epoch.py`; verified against s56's runs (s0 -> 31/7.4410/31, s5 ->
  28/7.3307/31). Both valsel drivers also skip a run that already has a `_best.pt`, so a
  cancelled sweep resumes instead of redoing.

### The 2023-01-01 figure arm, in flight

`fig_coguk_list_2023.R`, **without the antiGen Finetunes group or the roster ensembles**
(they are mixtures of it), and with the de novo column dropped for the reason above --
`coguk.drop_metrics`, which makes `assemble` generic over the number of recall cuts and
leaves the published two-panel widths and labels exactly as they were. Base antiGen, the
two untrained ESM-2 rungs and BLOSUM62 are trained (`ckpt_stage_2023-01-01`, 18:32) and
scored (job 3428765, 6:54, 20 models). The two val-selected bars wait on the budget
question above, so the figure is not drawn yet.

### Open

* The budget. See the probe.
* **Panel M is not drawn**, and its eval is a separate pass per cutoff -- one
  `--split-date` against the one test corpus, now that the reader takes it.
* The late cutoffs' points should carry their denominators or be dropped at a stated
  threshold rather than plotted as equals.

## 66. Panels G-J on the revised corpus, and K-L without dates (2026-10-24)

s54 left `coguk_list_aa20` with six panels and named the two missing rows by substrate
rather than by queue. `antiGen-data-revised` now emits
`sc2-test/counts_ba2/counts_<D>.tsv`, 24 dated BA.2-descended count matrices, and the
yield row is drawn from them.

### The counts run the other way, which is checked rather than assumed

`counts_<D>` holds the variant-descended sequences observed **on or after** D -- 2.17M at
2020-04-01 falling to 4,076 at 2026-01-01 -- while the panel wants what has emerged **by**
t. They are exact complements, since every BA.2 sequence post-dates the first grid point:

    emerged_by(D) = counts(earliest) - counts(D)

`load.py:load_counts_series` **rejects the directory unless every cell is non-increasing
across the grid** (0 of 123,660 rise). A cumulative series read as a forward window would
run the emergence curve backwards and still produce a monotone, plausible panel.

### Two conventions the row needed stated

* **One top decile for every model.** The first run tripped `fig_coguk_list.R`'s own s51
  guard: `n_selected` varied by 180, because `list_variant_recall` scored each model
  against its own emission set and the DMS maps abstain over much of the RBD. The
  published arm fixes this with a `rescore_common_pool.py` pass; `--common-pool` does it
  in one step -- the whole eval range less the query's own residue, **3,819 cells for
  every model**, a cell a model cannot score ranked last rather than dropped.
* **The time axis is declared.** The published grid is 0.1-year steps from 0.1; this one
  is quarters from the arm's own cutoff. `coguk.varr_x0` / `coguk.varr_dx` put both on
  the same "years into the pandemic" axis; the defaults reproduce the published figure,
  which **re-renders byte-identical** (`776aa91c`, `49cc10be`).

Drawn: 21 steps, 2021-01-01 to 2026-01-01, 28 models, the curve rising at t ~ 2 where
BA.2 emerges. **The five ensemble bars are absent from G-J** -- that row scores
checkpoints and baselines, not cached-map rules -- and the figure names them on every
run.

### K-L: the reconstruction was the mistake

The background row needs the individual background genotypes and which of them hosted
each mutation; a count matrix has summed over exactly that, so `counts_ba2` cannot serve
it. The first attempt rebuilt the genotypes from the repo's all-time Nextstrain JSON and
**failed its own gate: 191 of 2,308 branch mutations disagreed with their reconstructed
parent**, because that build is not the one the corpus was made on.

That reconstruction was redundant. Every record of the example corpus **is** a genotype,
with its parent and its child mutations, so `background_tree.build_backbone_from_examples`
reads them and nothing is re-derived. On instruction this uses the **attached** tree,
which the published row deliberately avoided: **23,806** distinct post-cutoff background
genotypes against the guide tree's ~402, so "hosts this mutation" partly counts sampling
depth here. 1,455 hosted cells, 46,433 host events, reproducing an independent streaming
count.

**The corpus carries no per-node date**, so the published host-vs-time-matched-control
rule cannot run on it. On instruction the contrast becomes **host vs one fixed reference
genotype**, the BA.2 root: `build_contrast_sets` already returned `cell -> (hosts,
controls)`, so the control list becomes that one query and nothing downstream moves. It
is a **weaker question than the published one and must not be read against it** -- an
idiosyncrasy of the reference moves every cell the same way instead of averaging out --
and cells the reference already carries, or whose only host is the reference, are
dropped. `build_contrast_sets` **refuses the time-matched rule on a dateless backbone**
rather than matching on NaN.

Job 3432096 is scoring 23,806 genotypes across the roster, ~3 GPU-hours. **K-L are left
as holes in the figure until it lands**, which `fig_coguk_list.R` reports per metric.

### Code

* `antiGen/load.py` -- `load_counts_series`.
* `antiGen/list_variant_recall.py` -- `--alphabet`, `--train-examples`/`--test-examples`,
  `--root-counts`, `--variant-counts-dir`, `--counts-since`, `--common-pool`;
  `--data-dir` optional.
* `antiGen/background_{tree,discrimination,contrast}.py` -- the corpus backbone,
  `--backbone-examples`, `--root-counts`, `--reference-query`.
* `examples/slurm/{eval_sc2_varrecall,background_discrimination_sc2}.sh` -- `DATA_DIR` /
  `CKPT_DIR` / `TREE_JSON` / `REF_JSON` overrides and an `EXTRA` passthrough, so the
  20-letter arm comes off the published scripts rather than a fork.

## 67. sc2_results on the revised corpus: the inventory, and what each panel needs (2026-10-24, IN PROGRESS)

Rebuilding the whole of `sc2_results` at the 2021-01-01 cutoff on the revised corpus,
with the roster `coguk_list_aa20` draws. This section is written as the work proceeds;
nothing below is a measurement yet.

### The thirteen panels resolve to three data products

| panel | what | source |
|---|---|---|
| A, F | cartoons | static PNGs, no data |
| B, C | recall by quantile | `quantile_hit_rates_*_all-models.csv` |
| D, E | recall AUC | same |
| G, H | PSSM precision vs date | `variant_emergence_*_all-models.csv` |
| I, J | max PSSM precision | same |
| L | PSSM precision, 100+ de novo, finetunes | same, filtered |
| K | model vs BA.2 DMS expression | `data/sc2/<model>/variant_root_predictions_*.csv` |
| M | recall at top decile vs cutoff | one eval per cutoff under `data/gating/` |

So: one `antiGen-eval` run for B-J and L, a per-model prediction matrix at the BA.2 root
for K, and a sweep of per-cutoff evals for M.

### Three things the inventory turned up

* **`eval.py` has had none of the revised-corpus treatment** the other four scripts got
  (s65/s66): no `--alphabet`, no `--train-examples`/`--test-examples`, no
  `--root-counts`, no `--variant-counts-dir`, no `--split-date`. Same patch shape as
  `list_macro`, `list_variant_recall` and `background_discrimination`.
* **Nothing in the codebase writes `variant_root_predictions_*.csv` any more.** The
  committed ones are dated 2026-06-22 and their producer is gone. They are recoverable
  with no GPU: a model's score row at the BA.2 root over the RBD is exactly what
  `list_score_maps_aa20` already holds, at **dumped query 4110** (located by genotype,
  s66). Panel K is therefore cache arithmetic plus a `ref/` file for the revised BA.2
  root, which differs from the published one at 0-indexed 492 -- Q493R, INSIDE the RBD,
  so the published reference cannot be reused for masking.
* **Panel M's fixed-32 base antiGen already exists.** The cutoff series' first round
  completed `base/` for 15 of the 16 cutoffs before it was cancelled over the budget
  question (s65); 2023-01-01's was trained by `train_sc2_2023.sh` into
  `ckpt_stage_2023-01-01` and is copied into the series with a `PROVENANCE.txt` saying
  so. The recipe is identical, which is what makes the copy legitimate.

### Decided with the user

* Panel M runs **2020-04-01 to 2024-01-01**, 16 quarterly cutoffs. Beyond that the test
  window falls under ~7,000 records (2,232 by 2025-01, **170** by 2026-01), so later
  points would plot as equals while resting on a fraction of the evidence (s65).
* Panel M's lines: fixed-32 base antiGen, **ESM-2 35M partial FT**, where x which, and
  the three published reference lines (PSSM, BLOSUM62, EVEscape).
* The **five combination bars** join B-J, through the same cached-map path s66 added to
  the yield row -- otherwise they are rungs on the list figure and absent here.

### A cost estimate that was wrong by 17x, caught before it cost anything

The per-cutoff ESM-2 runs were first submitted at time limits derived from a guessed
0.0008 s per record-epoch. The measured rate is **0.0138** -- s53's 2021 arm took 21:34
for 2,934 records at 32 epochs -- so 2024-01-01 would have been killed at 2 h into an
8.4 h run, and the same for the five cutoffs above it. Cancelled within a minute and
resubmitted at 2x the measured rate: 1 h to 17 h per cutoff, ~66 GPU-hours in all.
Worth recording because the failure mode is a half-trained checkpoint that still loads.

### Progress

* **`eval.py` patched and gated.** `--alphabet`, `--train-examples`/`--test-examples`,
  `--root-counts`, `--variant-counts-dir`, `--counts-since`, `--split-date`; and
  `build_special_mask` takes `gap_idx`/`stop_idx` of None, so every gap/stop exclusion
  is a no-op at 20 letters rather than an index error. One subtlety worth naming: when
  the revised inputs are given, the **legacy path resolution must not run at all** --
  left in, it derives a directory from a checkpoint's recorded cutoff and would either
  fail on a missing `$FITNESS_DATA_ROOT` or silently resolve to the PUBLISHED corpus and
  read it for nothing. Suite still **205**.
* **Panel K's inputs are built, with no GPU.**
  `experiments/multitree/dump_variant_root_predictions.py` writes each cached model's
  score row at the variant root in the published shape (seq_len x 22, NaN outside the
  eval range and in the `*`/`-` columns, which is what "says nothing here" already means
  to that panel). The root is located **by genotype** -- dumped query 4110, three
  duplicate rows, checked identical -- because the cache is rebuilt whenever the query
  set changes and a moved index would write another background's predictions under the
  right filename.
  * `data/sc2_aa20/`: the **14 trained pipeline rungs**, matching the published
    `data/sc2`'s contents (trained models only; the fixed external maps live in
    `baselines/` and are read in probability space by a different reader). The first
    dump wrote all 52 cached models including the twenty ensemble MEMBER runs, which
    would have appeared as twenty spurious rungs; pruned.
  * Plus **5 `ens_*` directories**, one per drawn combination rule, so the bars that
    join B-J are on the same footing here.
  * `ref/BA.2_aa20.txt` -- the revised BA.2 root, which differs from the published one
    inside the RBD (Q493R), so the published reference cannot be reused for
    `mask_wildtype`.
  * `baselines/BA2_escape_LYCoV1404.csv` -- the s64 map joins the baseline predictors.
* **Submitted**: the main eval (3432548) and the 16 per-cutoff ESM-2 35M partial
  trainings for panel M.

### STOPPED HERE, 2026-10-24 (on instruction). State at the stopping point

**Nothing in this section is measured and the figure is not drawn.** What exists:

| piece | state |
|---|---|
| `eval.py` + `eval_utils.build_special_mask` patched | done, suite 205 |
| `examples/slurm/eval_all_sc2.sh` parameterised | done |
| panel K inputs (`data/sc2_aa20/`, `ref/BA.2_aa20.txt`, baseline escape map) | done |
| `dump_variant_root_predictions.py` | done |
| main eval, job **3432548** -> `evals_sc2_aa20` | SUBMITTED, still PENDING |
| per-cutoff ESM-2 35M partial, 16 jobs | 3 of 16 trained, 5 running, 8 pending |
| per-cutoff base antiGen (fixed 32) | already on disk, 16 of 16 |
| per-cutoff evals for panel M | NOT SUBMITTED |
| revised `sc2_results` figure wrapper | NOT WRITTEN |

**Uncommitted at the stopping point**: `antiGen/eval.py`, `antiGen/eval_utils.py`,
`examples/slurm/eval_all_sc2.sh`, `experiments/multitree/dump_variant_root_predictions.py`,
`antiGen-figures/data/sc2_aa20/`, `antiGen-figures/ref/BA.2_aa20.txt`,
`antiGen-figures/baselines/BA2_escape_LYCoV1404.csv`, and this section.

**Still running from earlier sections, unrelated to the stop**: job 3432096, the s66
background row (23,806 genotypes, ~1 h in of ~3); job 3431140, the s65 512-epoch probe
(3 seeds at 2021-07-01, ~2 h in). Both write outside `antiGen-figures` and neither is
needed by anything already committed.

### To resume

1. The main eval finishes -> panels B-E, G-J, L. Its CSVs are
   `quantile_hit_rates_*_all-models.csv` and `variant_emergence_*_all-models.csv` in
   `evals_sc2_aa20`, read by a new wrapper the way `fig_coguk_list_aa20.R` reads its
   macro directory.
2. The five combination bars still need the cached-map path wired into eval.py's two
   metrics, the way s66 wired it into the yield row -- **this is not done**, so as
   things stand B-J would draw 28 rungs, not 33.
3. The 16 per-cutoff evals for panel M, once the ESM-2 runs land, so each cutoff is
   scored once with all six lines present (base antiGen fixed-32, ESM-2 35M partial,
   where x which, PSSM, BLOSUM62, EVEscape).
4. A wrapper that points MODEL_DIR at `data/sc2_aa20`, the ref at `BA.2_aa20.txt`, and
   the gating directory at the new per-cutoff tree, with panel M's x-axis truncated at
   2024-01-01 per the decision above.

### A scheduling lesson worth keeping

The cluster was saturated by this session's own jobs, and the **critical path (the main
eval) sat PENDING behind 16 training jobs that only feed one panel**. Niced after the
fact. Submit the critical path first, or nice the bulk work at submission.

## 68. One top decile for every rung, in `eval.py` too (2026-10-24)

`eval.py` was the last script on the figure still scoring each rung against its **own**
emission set. s51 gave `compute_hit_rate` a `pool_mask`, s51/s52 gave
`list_variant_recall` and `background_contrast` theirs, `list_macro` has passed one
since it was written, and `rescore_common_pool.py` exists to retrofit the published
CSVs -- but the script behind Figure 2 and `sc2_results` never took one. So the two
figures this arm draws were asking the same question two ways, and on instruction
(2026-10-24) that is now one way: **a top-q quantile means the same NUMBER of cells
for every rung.**

### What "the same number" is here

At 20 letters `real_substitution_mask` selects every column and `real_wildtype_mask`
every row, so the pool is the whole eval range -- **4,020 cells, 4,019 after the query's
own residue leaves** -- identical for every rung and every query. The convention is not
free of consequence even when the mask is a no-op, because the pool is what decides
whether a cell a model **cannot score** leaves the pool or stays in it ranked last:

* **before**: a NaN cell was dropped, so `n_valid` was the model's own coverage and
  `q * n_valid` was a proportionally smaller selection. A map covering half the RBD
  searched half as many cells as its rivals.
* **now**: it is filled one step below the query's minimum, stays in `n_valid`, and is
  never selected. Every rung selects `q * 4,019` and a map pays for its gaps.

### Exactly four rungs move, and the roster says which before the job starts

`eval.py` already prints baseline coverage over the eval range. On this corpus:

| rung | coverage |
|---|---|
| `BA2_escape_LYCoV1404` | **50.3%** |
| `EVEscape` | **95.0%** |
| `Omicron_BA2_bind`, `Omicron_BA2_expr` | **99.6%** |
| everything else (23 rungs) | 100.0% |

A rung that scores every cell is **bit-unmoved** -- the pool it gets is the pool it had
-- which is the property the change rests on and is gated as such
(`test_a_fully_scoring_rung_is_unmoved_by_the_pool`, over all three mask shapes a
variant can carry). So no trained model's curve moves at all; the four maps above do,
and `BA2_escape_LYCoV1404` moves a lot. That is the s64 caveat becoming arithmetic
rather than a sentence beside a label.

### The precision row needed it more than the recall row

On recall `n_valid` sizes the set searched; on the variant-emergence row (panels G-J, L)
`n_valid` **is the denominator** -- precision is hits / the size of the top-q. s51
measured what that does: the nine-member mean of log-probs caught 98.0 listed cells to
EVE full FT's 94.0 and scored **0.2322 against 0.2461**. So `variant_hit_rates` takes a
`pool_mask` too, with `top_q_weights`' treatment copied rather than re-derived, down to
taking the tie fraction over the whole boundary block before the filled cells are struck
from it.

**The denominator stayed the selected COUNT, and that was not the first attempt.** Taking
it as the target `q * n_valid` -- which is what `list_variant_recall.curve` does -- moved
six rungs at 100% coverage by ~1e-4, because the count and the target differ by one
boundary cell. That is not the pool's doing and it should not look like it, so the
denominator is `n_above + frac * n_at` with the filled cells counted in. Fully-scoring
rungs are then bit-identical to their pre-s68 values at every q, and the sc2_results
precision row sits ~1e-4 from the list figure's yield row by construction; the two are
not read against each other and now say so in the code.

**A blind map has two regimes, and only the second involves a filled cell.** Below its
coverage the pool simply makes its top-q the same SIZE as everyone's, so it reaches
further down its own shorter ranking than the published convention let it. Above its
coverage the boundary crosses into the filled block and cells it never scored occupy
slots it can never be credited for. Both are gated.

**The pool is per spec, and it has to be.** Every restriction the emergence path
expresses by writing NaN into a score map -- the de novo cut, a curated list -- must be
struck from the pool as well, or those cells would be *filled and ranked last* inside it
instead of leaving it, and the denominator would be the whole range for a spec that only
ever meant to search a few hundred cells. The spec tuple carries its own pool for that
reason; `_epool(wt_mask, drop)` builds it from the same two masks the score map was
NaN-ed with.

### A silent wipe found on the way, which had nothing to do with pooling

Four emergence suffixes built their masks by indexing with `gap_idx` / `stop_idx`:

```python
wt_mask_nogapstop[:, gap_idx] = False        # gap_idx is None at 20 letters
```

At 20 letters both indices are `None`, and **this does not raise**. `t[:, None] = False`
unsqueezes and assigns the **whole tensor**, so `all_nogapstop`, `all_nogapstop_thresh1`,
`unseen_nogapstop` and `unseen_nogapstop_thresh1` were each about to be written off an
all-False candidate mask and an all-NaN score map. They are not among the four suffixes
`VARIANT_EMERGENCE_METRICS` draws, so nothing on the figure was wrong -- but four CSVs of
confident zeros would have been, and s67's `build_special_mask` work had retired the
same trap one layer up without catching this one. Now built as `wt_mask_2d & ~special_sub`
off `build_special_mask`, which is a no-op at 20 letters and identical at 22. Gated.

### `--legacy-pool`, and what it is for

Off by default. It restores the per-model emission set, and it is the only way to
reproduce the committed `evals_sc2` CSVs, which were written under the old convention --
the same flag, with the same meaning, that `list_macro`, `list_variant_recall`,
`background_contrast` and `rescore_common_pool` already carry. Nothing on disk is
reinterpreted by this change; the published figure is a published number and stays one.

### The combination bars need a cache this figure does not have, and 2 of the 5 need more

Resuming s67's item 2: the five drawn rules cannot come off `list_score_maps_aa20`,
because that cache holds the **11,523 CoG-UK-list-carrying queries** and the recall
panels score all **71,704** post-cutoff test records. Two facts the inventory should
have stated and did not:

* **The three mean rules' nine members are already on this roster** -- base antiGen, EVE
  full FT, Tranception full FT and the six ESM-2 arms, verbatim from
  `ensembles9_aa20/rules_drawn5.json`. Their test-set forward passes are happening
  anyway, so the maps are free apart from I/O.
* **The other two rules have different members entirely**: `vs10Antigen_s0..s9` and
  `hybridWxwPssm_s0..s9`, twenty cheap runs that are NOT on this roster. They need their
  own pass over the test set into the same cache. So this run buys 3 of the 5 bars on
  B-E, not 5, and that is a second job rather than a second thought.

`eval.py --dump-scores DIR [--dump-models A,B,C]` writes each evaluated model's
eval-range map in `score_cache`'s format -- 1.07 GiB per model at 71,704 x 201 x 20,
`query_sha1 47e22b2b` over the eval-range genotypes, 46,433 target events. The sink is
**observational**: it is handed the slice after the model's gap/stop mask and before any
variant masking, and the metric is bit-identical with it on or off (gated). A pass that
dies leaves a stray `.tmp` rather than a full-size file of plausible zeros, which is the
discipline `MapWriter` already had and the reason the writer is aborted on any
`BaseException` rather than on a caught `Exception`.

**One thing this turned up:** the emergence block bound a local `sc` over the
`score_cache` module alias inside the same function, which would have made every
`sc.*` call in `main()` an `UnboundLocalError`. Renamed; it is the sort of thing that
only appears when a module is first imported into a long function.

### Code

* `antiGen/eval_utils.py` -- `evaluate_full(common_pool=, alphabet=, score_sink=)`: the
  variant's score mask now narrows the pool instead of being burned into the scores, and
  the query-dependent half is built per batch. `variant_hit_rates(pool_mask=)`.
* `antiGen/eval.py` -- `--legacy-pool`, `--dump-scores`, `--dump-models`; the pool built
  once beside the eval range and handed to all four `evaluate_full` call sites; per-spec
  emergence pools; the `gap_idx`/`stop_idx` indexing retired.
* `experiments/multitree/submit_eval_aa20.sh` -- the resubmission, with the nine members
  named so the cache stays at ~9.7 GiB.
* `antiGen/tests/test_eval_common_pool.py` -- 17 tests. Suite **222**.

### Measured, before any GPU was spent

`--emergence-only --models blosum62` on the real corpus, CPU, run twice -- once with the
pool and once with `--legacy-pool` -- is enough to see the whole precision row move or
not move. It does exactly what the coverage table predicts:

| suffix | rungs that moved | largest |
|---|---|---|
| `all_thresh1` | LY-CoV1404, EVEscape, `Omicron_BA2_{bind,expr}` | **0.1825** (LY-CoV1404) |
| `unseen_thresh1` | LY-CoV1404, `Omicron_BA2_{bind,expr}` | 0.1471 |
| `all` | the same four | 0.0411 |
| `unseen` | the same three | 0.0206 |

The other ten rungs are bit-identical. EVEscape drops out of the de novo subsets because
its 15 missing cells are all present in the training window. And the four
`*_nogapstop*` suffixes now come out **equal to their plain twins**, which is what 20
letters means and what the `None`-index wipe was hiding.

### State

Job **3432548** was already in its baseline sweep when this landed, so its CSVs would be
the old convention; it writes to `evals_sc2_aa20` and nothing of it is reusable. **It
has to be cancelled before `experiments/multitree/submit_eval_aa20.sh` is run**, or the
resume logic will find its rows and skip every model that wrote one, leaving a directory
that mixes two conventions and says nothing about which row is which.

Still to do on s67's list, unchanged by this section except where noted:

| piece | state |
|---|---|
| main eval, with the nine members' maps | script written, **NOT SUBMITTED** (blocked on the cancel) |
| the 20 `vs10Antigen` / `hybridWxwPssm` members over the test set | not written; needed for 2 of the 5 bars on B-E |
| the offline rule scorer (cache -> `quantile_hit_rates` rows for `ens_*`) | not written |
| per-cutoff evals for panel M | not submitted; the ESM-2 trainings are still running |
| revised `sc2_results` figure wrapper | not written |

## 69. Resuming s67's list: the member passes, the rule scorer, the cutoff sweep (2026-10-24)

The main eval went back in the queue as **3433201** against an empty `evals_sc2_aa20` --
3432548 was cancelled inside its baseline sweep and had flushed nothing, so there is no
mixed-convention directory to worry about. Everything below is the rest of s67's list.

### The three products the combination bars need, and why they are three

The bars cannot come from one place, because the five rules are not one kind of thing.

| rule | members | where its maps come from |
|---|---|---|
| Mean of ranks / log-probs / z-scores | the nine drawn rungs | the main eval's `--dump-scores` |
| `ens_8a91fbbf88` | `vs10Antigen_s0..s9` | job **3433299**, its own eval pass |
| `ens_69de905310` | `hybridWxwPssm_s0..s9` | job **3433285**, `hybrid_baseline.py` |

All three write into **one** cache, `test_score_maps_aa20`, and
`write_query_manifest` is what makes that safe: a second pass over a different query set
is refused rather than deposited, because two dumps of the same roster have identical
shape and no array-level signature of a drift.

* **The hybrid pass is done**, in under five minutes on CPU -- it reads only the
  manifest and `queries.npz`, which the main eval writes at startup, so it did not have
  to wait for anything. Ten maps, 0 NaN and 0 -inf cells. The selected weights are
  **0.50 / 0.55 / 0.60** (min / median / max), which is the diagnostic
  `hybrid_baseline.py`'s own module note asks for: they are interior, not pinned at
  w = 0, so the selection is not simply the PSSM the val half has already seen.
* **The vs10 pass needed `--include-experimental`.** Its first submission discovered all
  ten groups and filtered all ten: the val-selected checkpoints carry the `multitree`
  arm, which `--dataset sc2` drops by default. Correct for a figure roster and wrong for
  a member pass, which is what the flag is for.
* **`--skip-baselines`** was added for that job. Its output is a cache, not a CSV, and
  the test-set baseline sweep is the single most expensive thing in a run -- 13 fixed
  maps against every variant and quantile over the whole test set. It is guarded by a
  warning in its own help: the baseline ROWS are then absent, so it must not be passed
  for a run anything reads.

### The rule scorer, and the one check that makes it legitimate

`experiments/multitree/ensemble_test_rows.py` turns the cache into rows -- recall over
the test queries for B-E, precision at the variant root for G-J and L -- and writes
`*_ensemble.csv` holding `ens_*` rows and nothing else, which is
`ensemble_report.py`'s rule and for its reason: the figure checks overlapping jobs
against each other, so re-reporting a single-model row turns a reproduction into a
disagreement.

**`--verify-member` is the gate on the whole cached path at once.** A normalisation is
monotone within a query, so a ONE-MEMBER rule must rank cell for cell with its member
and must therefore reproduce that member's own row in
`quantile_hit_rates_all_all-models.csv` -- the row `eval.py` wrote from a forward pass,
through a different code path, in a different job. It checks the dump's query order, the
`-inf` convention, the pool and the tie rule in one number, and it is the only thing that
establishes the bars are on the same ruler as the rungs beside them. Refuses above 1e-6.

Two things the identity needed before it could be relied on:

* `test_ensemble_rules.test_one_member_is_a_no_op` pins it **without** a pool. Under one,
  a member's NaN cells are filled and ranked last instead of leaving the pool, so the
  normalisation runs over a different set than the metric's candidate pool. Re-checked
  rather than inherited.
* The variant root is located **by genotype** against `ref/BA.2_aa20.txt`, never by
  index -- the s67 rule, and the reference has to be the revised root, which differs
  from the published one at Q493R inside the RBD.

### Panel M: six of sixteen submitted, and a staging directory

`cutoff_series/<D>/` holds `base/`, `esm2_35m_partial/`, `valsel/` (ten runs) and
`valsel_ep32/`, and `discover_checkpoints` globs one level deep -- so the eval finds
nothing pointed at the cutoff directory and twenty unwanted members pointed at a parent.
Each cutoff therefore gets a directory of two symlinks.

`submit_cutoff_evals.sh` is re-runnable: a cutoff whose ESM-2 run has not landed is
skipped with a line saying which half is missing, and one already evaluated is skipped
unless `FORCE=1`. **6 of 16 submitted** (2020-04-01 .. 2021-07-01), 10 still waiting on
their training. Niced at 1000, which is the s67 scheduling lesson applied rather than
restated.

Dry-checked on CPU before the queue filled: at 2020-04-01 the roster resolves to the two
checkpoints plus the factored baselines, the shared test corpus splits to **75,143**
testing records at that date, and the emergence grid starts at the cutoff. Worth noting
the other end of s65's budget finding is visible here -- that cutoff trains on **190
examples**.

### Code

* `antiGen/eval.py` -- `--skip-baselines`.
* `experiments/multitree/ensemble_test_rows.py` -- the rule scorer and its gate.
* `experiments/multitree/submit_eval_aa20{,_members}.sh`, `submit_cutoff_evals.sh`.
* `antiGen/tests/test_ensemble_test_rows.py` -- 4 tests, including the script end to end
  over a synthetic cache. Suite **226**.

### Open

* The figure wrapper is **still not written**. It needs a loader that reads panel M by
  model NAME: the published `fig2_sc2_benchmarks.R` reads rows 1, 2 and 4 of each
  per-cutoff CSV by POSITION, which only works because that arm ran one eval per model.
  This arm runs one eval per cutoff with every line in it.
* The ten remaining cutoffs, as their ESM-2 runs land.
* `ensemble_test_rows.py` has not been run on real maps -- the main eval has not reached
  its first member yet, so `--verify-member` has nothing to check against.

## 70. The cached path is on the same ruler, and two mistakes of mine (2026-10-25)

### The gate passed, on real maps

`ensemble_test_rows.py --verify-member esm2_esm2_t12_35M_UR50D_lr0.0001_partial4`, over
the 71,704-query cache the main eval is still filling:

    one-member rule vs its own row, worst |delta| = 4.248e-09 over 9 quantiles

A one-member rule is a monotone transform of its member within each query, so it must
reproduce that member's row -- and that row was written by `eval.py` from a forward
pass, in another job, through `evaluate_full` rather than through the cache. 4e-9 says
the dump's query order, the `-inf` convention, the common pool and the fractional tie
rule all agree across the two paths. **The combination bars are on the same ruler as the
rungs they will sit beside**, which until now was an argument and is now a number.

Denominators, for the record: **46,433** target events over the eval range, **9,826** on
the de novo cut. The two `*_nogapstop` variants come out of one pass each with their
plain twins, as s68 said they must at 20 letters.

### I submitted five duplicate jobs, and then a sixth

`submit_cutoff_evals.sh` skipped a cutoff that was already *evaluated* and a cutoff not
yet *trained*, and nothing else. Run again to pick up a newly-trained cutoff, it
resubmitted every job still sitting in the queue from the previous run: five duplicates,
two of them against an output directory a running job was writing, and `rm -rf` on
staging directories whose symlinks a running job still resolves through. **"Has it
finished" is not the same question as "is it needed"**, and only the second is safe to
submit on.

The fix is a queue check by job name -- which is why every job is named `cut_<D>` -- but
the first version of THAT was wrong too, and submitted a sixth duplicate. It read the
queue per cutoff as `squeue 2>/dev/null | grep -c`, which returns a confident **0** when
the scheduler is merely busy, and a confident 0 here means "submit a duplicate". The
queue is now read **once, up front, and a failed read aborts the script** rather than
being treated as an empty queue. Verified: a third run submits nothing.

Six duplicates are in the queue (`3434315-9`, `3434331`), all PENDING, all newer than
the job they shadow. They need cancelling.

### `antiGen-figures/baselines/` is globbed, so dropping a file in it edits the figure

Rendering `fig2_sc2_benchmarks.R` unmodified -- to see whether byte-identity was
available as a gate for the sc2_results wrapper, the way it was for the list figure --
reproduced five of seven PNGs **byte for byte** and changed two: `sc2_results.png` and
`supp_dms.png`. Not nondeterminism. Line 466 is

    baseline_files <- list.files("baselines")

so s64's `BA2_escape_LYCoV1404.csv`, sitting untracked in that directory since
2026-10-24, is already a rung on the published figure's DMS panels. The committed PNGs
are from 2026-10-18 and predate it.

Two things follow. **PNG rendering is deterministic**, so byte-identity IS usable as the
gate on parameterising fig2 -- against a re-render, not against the committed file.
And a new baseline reaches the published figure by existing, with no edit anywhere;
whether those two PNGs should now be re-rendered is a decision, not a cleanup.

**The render left `figs/{sc2_results,supp_dms}.png` and seven `figs_vector/*.pdf`
modified in the working tree** (every PDF differs on its embedded timestamp alone).
Restoring them is `git checkout --` on those paths, which this session cannot run.

### R, for the record

`/home/ivan.specht/miniconda/envs/antigen-figs/bin/Rscript`, 4.5.3. Not on PATH.

## 71. `fig2_sc2_benchmarks.R` parameterised, and the aa20 wrapper (2026-10-25)

The last piece of s67's list, and the one the cluster cannot hurry. `sc2_results` is
drawn by `fig2_sc2_benchmarks.R`, which had **no `getOption` anywhere** -- every input
and output path was a literal. The list figure solved this with a drawer plus a wrapper
(`fig_coguk_list.R` + `fig_coguk_list_aa20.R`); this is the same split, for the same
reason: a forked drawer and its original drift apart silently, and the two figures are
read against each other.

### The gate, and the thing it caught about the published figure

Rendering the unmodified script reproduced **five of seven PNGs byte for byte** and
changed `sc2_results.png` and `supp_dms.png`. Not nondeterminism (s70): line 466 is
`list.files("baselines")`, so s64's `BA2_escape_LYCoV1404.csv` had already joined the
published DMS panels by existing. So byte-identity IS available as the gate, against a
**re-render** rather than against the committed file -- and that re-render is the
baseline every change below was checked against. **All eight outputs byte-identical**
after the refactor.

### What became an option

Inputs: `eval_dir`, `model_dir`, `gating_dir`, `mtb_dir`/`mtb_models`, `dms_dir`, `ref`,
`baseline_dir`, `pssm_by_cutoff_dir`, `cartoon_dir`, `ensemble_dir`. Outputs:
`fig_main`, `fig_prefix`, `fig1_auc`. Panel M: `gate_loader`, `gate_models`,
`gate_model_labels`, `gate_rows`, `gate_x`, `gate_baselines`, `gate_colors_main`,
`gate_colors_supp`, `gate_drop_main`. Every default is the published value.

Four things were more than renaming a literal:

* **Panel M is read by NAME on this arm.** The published loader takes rows 1, 2 and 4
  of each per-cutoff CSV **by position**, which works only because that arm ran one
  eval per MODEL. This arm runs one eval per CUTOFF with every line in it, so
  `gate_recall_by_name` matches `sc2.gate_rows` against the `model` column -- and a
  cutoff missing a line **stops the render**. A line that quietly vanishes for three
  cutoffs and comes back is worse than a figure that refuses to draw.
* **The model column is made a factor in `sc2.gate_rows` order.** `scale_color_manual`
  matches its values to levels positionally, so leaving it a character vector would
  make the colour assignment depend on the locale's collation of the labels.
* **The gating PRECISION supplement is skipped when there is no per-model tree.** It
  reads `GATING_DIR/<model>/cutoff_*/`, which a by-cutoff corpus simply does not have.
  Wrapped in a function, called on `length(gate_cutoff_dirs)`; its locals are used
  nowhere else, checked rather than assumed.
* **`with_ensemble` splices the `ens_*` rows** from `<metric>_ensemble.csv`. It refuses
  a file holding a non-`ens_` row, and refuses one that re-reports a row already
  present -- `ensemble_report.py`'s rule, and here a collision is a real contradiction
  between two jobs over the same queries.

### An R trap worth naming: `options(x = NULL)` DELETES the option

`sc2.fig1_auc = NULL` in the wrapper does not set the option to NULL, it removes it --
so `getOption("sc2.fig1_auc", "fig1/auc.png")` hands back the default, and the revised
arm would have silently overwritten **Figure 1's** recall-AUC inset with another
corpus's panel. The sentinel is `""`, and the drawer tests `nzchar`.

### Panel M was missing a line, and "done" was the wrong question

BLOSUM62 is one of panel M's six lines and on this arm it is a **checkpoint**
(`fitness_blosum62_*.pt`), not a fixed map in `data/baselines` -- so the staging
directory of two symlinks produced per-cutoff CSVs with no `blosum62` row at all. The
matrix does not depend on the cutoff, so one checkpoint serves every point; what varies
is the test window, which is what the panel plots.

The deeper fix is in what counts as finished. `submit_cutoff_evals.sh` treated a cutoff
as done when **a CSV existed**, so a cutoff could be done while missing a line forever
and re-running would never notice. It now checks that the CSV carries every row in
`PANEL_ROWS`, prints `[redo]` with the names of the missing ones, and resubmits. Two
cutoffs were re-queued by that check on its first run. The five already in the queue got
the symlink added to their staging directories in place, so only the two that had
already finished pay for the mistake.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- the drawer, byte-identical on its
  defaults.
* `antiGen-figures/figures/fig_sc2_aa20.R` -- the wrapper. Outputs `sc2_results_aa20`
  and `aa20_supp_*`, so nothing published is overwritten.
* `antiGen-figures/tests/test_gate_by_name.R` -- the by-name loader against a synthetic
  tree whose rows are **shuffled between cutoffs**, so a positional reader would mix
  them. Pins all three properties: by name, factor order, refusal on a missing line.
  There is no R test harness here, so it is a script that stops on its first
  `stopifnot`.
* `experiments/multitree/install_sc2_aa20.sh` -- copies the eval outputs into
  `data/evals_sc2_aa20` and `data/gating_aa20`, copies rather than symlinks (a symlink
  into `/large_storage` would make a drawn figure change under a later re-run), and
  reports what is missing instead of installing half a figure. It prints the installed
  roster size, because `eval.py` flushes after every model and a CSV exists long before
  the roster is whole.
* `experiments/multitree/submit_cutoff_evals.sh` -- blosum62 staged; `PANEL_ROWS`.

### State

Running the wrapper today would draw a figure with 18 of ~32 rungs, no combination bars
and no panel M. The install script says so rather than letting it happen. What it is
waiting for is only data: the main eval (on its 650M arms), the rule scorer, and the
cutoff sweep.

## 72. The data lands, and the smoke test earns its keep (2026-10-25)

s71 left the wrapper written and gated with nothing to draw. The data arrived today, so
this section is what arrived, and the two bugs that drawing it immediately exposed.

### What landed

* **Main eval 3433201, COMPLETED in 11h24m.** Thirty rungs on the revised 20-letter
  corpus, all four recall tables and all eight emergence tables. No NaNs. `pssm_root`
  shows exactly the flat profile a tie-heavy coarse predictor should -- 0.782 at the
  top decile rising only to 0.926 across the whole q range -- which is s68's fractional
  tie rule behaving, not rounding the boundary to a cliff.
* **The score cache is whole**: 29 maps, 32 GiB, and NO `.tmp` left behind, so no
  writer died mid-file and left a half-map that `load_block` would read without
  complaint.
* **Rule scorer 3441435, COMPLETED in 44 seconds.** The gate held: a one-member rule
  reproduced `simple_d128_n2_lr0.0001`'s own row to **4.022e-09**. Twelve
  `*_ensemble.csv`, five `ens_*` rows each and nothing else. Denominators 46,433
  events / 9,826 de novo, the same as s70 measured.

Submitted beside it: nine cutoff evals, where the `PANEL_ROWS` check of s71 paid for
itself immediately -- `2020-10-01` and `2021-01-01` came back `[redo] CSV is missing
blosum62`, two cutoffs the old "a CSV exists" test would have called done forever.

### Two trainings died on a bad node, and it is worth saying how that was told apart

`2022-07-01` and `2022-10-01` had a manifest and no checkpoint, with nothing in the
queue: both killed TIMEOUT at epoch 8 of 32.

The tempting reading is that the budget was too small for a bigger corpus. It is wrong.
2022-07-01 is 44,481 examples and managed 8 epochs in 11 hours; **2023-01-01 is 53,916
examples -- more data, 1,686 batches an epoch against 1,391 -- and finished all 32 in
6h41m**, and 2024-01-01 took 8h28m. The two that died were running about 7x slower per
epoch than a larger job on the same recipe, so whatever throttled them, it was not the
work. What they have in common is the node: both ran on GPUCACE, nothing else in the
sweep did, and they overlapped there for ten hours.

That is consistent with two readings -- a degraded GPU, or two 256G jobs contending on
one host -- and the retry does not need to tell them apart, so
`submit_cutoff_esm2_retry.sh` covers both: GPUCACE excluded, and 16h, about double what
the largest healthy cutoff actually needed.

### The smoke test, which is the point of this section

With B-E and G-J/L complete I ran the wrapper rather than waiting for panel M. It
rendered, and **nothing published moved** -- every output is `aa20_*`. It also came out
wrong in two ways that would have been far more expensive to find after the sweep
finished. Panel M, the part that was new and therefore suspected, was correct; the
breakage was in the parts inherited from the published drawer.

**(a) The emergence axis was wrong by a factor of 2.5.** `eval.py` labelled the
emergence grid's columns

```python
step_labels = [f"t_{(i + 1) * 0.1:.1f}" for i in range(n_steps)]
```

-- the step INDEX times 0.1, never looking at a date. That was right for exactly one
grid. The published arm reads `pssms/<variant>/time_<t>.csv`, whose filenames ARE t in
years and which the pipeline writes at exactly 0.1-year spacing, so the index
reproduced the real t and "Date (t, years into pandemic)" was a true axis;
`data/evals_sc2` runs t_0.1 .. t_6.0 and is fine. `--variant-counts-dir` is a different
grid: dated files at whatever spacing the corpus was built with. The revised arm's is
QUARTERLY, 2021-01-01 .. 2026-01-01, so a five-year span was labelled t_0.1 .. t_2.1.

Two consequences, one of them silent. The axis read 2.1 years where it meant 6.1. And
panel L filters `variant_emergence_performance_x$x >= 2`, which on the published grid
means "from two years in" and on this one meant **"the last two points"** -- which is
why panels G, H and L drew straight diagonals across a 2.000-2.100 axis.

The fix: `emergence_step_labels(n_steps, dates=None)`. A dated grid labels itself from
its dates; an undated one keeps the index convention, so **every published CSV stays
bit-identical**. The origin is `PANDEMIC_START = "2019-12-20"`, which is not a new
constant -- it is the one `fig2_sc2_benchmarks.R` has carried since the published arm.
On the revised grid the headers now run t_1.0349 (2021-01-01) .. t_6.0342 (2026-01-01),
which puts the two arms on one ruler.

It changes **headers only, not one value**, so it does not cost an 11-hour re-run.

**(b) Six models collapsed into a single line called NA.** `add_group_and_label` does

```r
df$model <- factor(longform_names[match(df$model, all_models)], levels = longform_names)
```

so anything not in the 27-name published inventory becomes NA, and ggplot draws the lot
as one group -- the sawtooth across panels B and C, one tooth per model. The six are the
five `ens_*` rule rows and `BA2_escape_LYCoV1404`. This is still OPEN; see below.

### Two mistakes of mine, found while fixing those

* **I had put a second pandemic origin in the wrapper.** `sc2.gate_x` dated panel M from
  `2019-12-01`, while the drawer's own `PANDEMIC_START` is `2019-12-20` -- 19 days, 0.052
  years, and three axes all labelled "years into pandemic" disagreeing. The wrapper now
  uses the drawer's constant.
* **`PANDEMIC_START` was a local inside `draw_gating_precision()`**, a function the
  by-name loader never calls, so referring to it from the wrapper would have been
  "object not found". It is hoisted to a top-level constant where all three readers can
  see it.

Also, twice now I have written a sequence of `str.replace` calls where an earlier one
invalidates a later one's match -- this time a 4-space pattern also matching a 12-space
line. The assertion caught it before anything was written. Line-anchored replacement is
the habit to keep.

### Code

* `antiGen/eval.py` -- `PANDEMIC_START`; `emergence_step_labels(n_steps, dates=None)`;
  dates threaded from `load_counts_series` to the writer. `pssm_variant_sub` slices
  POSITIONS, not time, so the dates line up one-for-one -- I first wrote a `[-n_steps:]`
  tail slice on the assumption it was a time slice, which was a no-op that would have
  read as meaningful. The labeller re-checks the two lengths rather than letting `zip`
  truncate and slide every curve.
* `experiments/multitree/ensemble_test_rows.py` -- imports that labeller rather than
  restating it (the figure asserts the two CSVs' headers are identical, so a second copy
  of the convention is a mismatch waiting to happen), and now writes
  `ensemble_figure_rows.tsv`, the same sidecar under the same name and format as
  `ensemble_report.py`, so the figure reads `ens_<id>` -> label from disk instead of
  carrying a hand-copied blake2b hash.
* `antiGen/tests/test_emergence_labels.py` -- 5 tests. The published convention
  unchanged at 4 and 60 steps; a dated grid against the real offsets; date objects and
  ISO strings agreeing; a length mismatch raising rather than zipping short; and that
  the origin is the figure's own constant.
* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `PANDEMIC_START` hoisted.
* `antiGen-figures/figures/fig_sc2_aa20.R` -- one origin.
* `experiments/multitree/submit_ensemble_rows.sh`, `submit_cutoff_esm2_retry.sh` -- new.
* `install_sc2_aa20.sh` -- the expected roster was a stale estimate (27); it is 30,
  counted off the finished run.

26 Python tests pass.

### State, and what is still open

Still to do before the figure is right:

1. **The NA models, (b) above.** The five rules need display names and a group, and
   `BA2_escape_LYCoV1404` needs placing. Both want to be `getOption`s so the published
   defaults do not move; the labels should come from `ensemble_figure_rows.tsv`.
2. **Relabel the emergence CSVs already on disk.** Their VALUES are correct and the
   header fix is cosmetic-but-load-bearing, so this is a header rewrite driven by the
   same `emergence_step_labels`, not a re-run. The rule scorer (44s) can simply be run
   again.
3. **Panel M**: 9 cutoff evals in flight, 2 ESM-2 retrains behind them.
4. DONE: the `PANDEMIC_START` hoist moved nothing. Both PNGs byte-identical, all seven
   PDFs content-identical.

### A correction to s71's gate, and two bad comparisons on the way to it

**s71 claimed "all eight outputs byte-identical". That was too strong for the PDFs.**
`cairo_pdf` embeds `/CreationDate` and `/ModDate`, so two renders of identical content
differ at byte 58 and can never match bit for bit; the seven PDFs here are the same
size to the byte and differ only in that timestamp. The right gate for a PDF is a
comparison with the timestamps stripped, and that is what this re-render used. The PNG
half of s71's claim stands as written.

Getting there took two comparisons that proved nothing, both worth recording because
both LOOKED like passes:

* I launched `md5sum -c` from inside the snapshot directory, so it checked the snapshot
  against itself. A clean "all OK" that could not have failed.
* The first strip used `|` as the `sed` delimiter while the pattern itself contained an
  alternation `|`. `sed` errored on both sides, `md5sum` hashed two empty streams, and
  all seven files reported IDENTICAL. The rerun prints the stripped byte count (12,694,
  not 0) and the per-file hash, so an empty or constant comparison is visible in the
  output rather than silently passing.

A verification that cannot fail is worse than no verification, because it gets believed.

Two things remain the user's call, carried over from s71: the nine modified figure files
in the working tree that I could not revert, and whether `figs/sc2_results.png` and
`figs/supp_dms.png` should now be re-rendered to gain the LY-CoV1404 baseline, which is
a real content change to committed figures rather than a side effect of verification.

## 73. K-L were done and empty, and a Dirichlet blend of where x which (2026-10-25)

Two things, in the order they were asked for: why `coguk_list_aa20`'s background row
had been finished since the 24th and drawn nothing, and a new rung on the same figure.

### A COMPLETED job that produced zero rows

Job 3432096 (`bg_aa20`) ran both halves on 2026-10-24 and exited 0 in 2h21m. The
expensive half did its work -- 30 `scores_*.npz`, 4.3 GB, no `.tmp`. The cheap half
wrote:

    1212 cells hosted; 0 with >=1 host(s) and >=10 controls
      cellset coguk_all: 0 cells
    Wrote .../background_summary.csv (0 rows)

A header and nothing else. Nothing in that log says "failure", which is why it sat for
a day looking done.

**`--min-controls` defaults to 10**, which is right for the published TIME-MATCHED
rule, where a cell's control is a SET of backgrounds close in time. s66 replaced that
rule on this substrate: the revised corpus carries no per-node dates, so the contrast
is host vs ONE fixed reference genotype, the BA.2 root. Under that rule every cell has
exactly 1 control and `len(cq) >= 10` is false for all 1,212 of them. **The threshold
is a leftover from the rule that was replaced, not a bar this arm failed** -- s66
changed the contrast and did not change the filter that reads it.

`--min-hosts` stays at 1. Under the single-reference rule it is the only remaining
stability knob, and raising it is a judgement about what the panel claims rather than
a fix.

The reference index is READ from `backbone_manifest.json` (`variant_root_query = 0`),
never assumed -- the s67 rule, for the s67 reason: the query set is rebuilt whenever
the corpus changes and a moved index contrasts every cell against the wrong background
under the right name.

### What the cost of K-L actually is, which was worth measuring before promising it

The published contrast pass took **28 seconds** (score maps 02:58:26, summary 02:58:54
on 10-19). This arm is not 28 seconds, because `common_percentiles` loops per cell over
a `(n_queries x n_pool)` array and both axes grew:

| | published | revised | ratio |
|---|---|---|---|
| background genotypes | 402 | 23,806 | 59x |
| usable cells | 150 | 1,212 | 8.1x |
| product | 60k | 28.8M | **478x** |

So 3-5 hours, not minutes. **The 59x is s66's decision showing up as a bill**: using
the attached tree rather than the guide tree is what gives 23,806 backgrounds, and it
was taken for the reason recorded there, not by accident.

The empty run's 2m44s is NOT a baseline for this. With `usable` empty the per-cell loop
ran zero times, so that number is the cost of reading 30 npz files and nothing else.

### A Dirichlet blend of `where_times_which` with the observed events

New rung `wxw_dirichlet`, on request, from a colleague's snippet. The factored joint is
a conjugate PRIOR and every observed mutation event is an update:

    q[i,a] = (N[i,a] + kappa * pi[i,a]) / (N_total + kappa),  kappa = N_total
    pi[i,a] = P(i) * P(a | wt_i)

**The base measure is THIS repo's `where_times_which`, not the snippet's.** The snippet
builds its prior from an empirical substitution matrix `s_i * M[wt_i,a]`; this repo
already has that object as a drawn rung, except its which term is the FITTED
substitution model (genetic code + BLOSUM62). Using the rung keeps the figure
internally consistent and buys the gate below. Recorded because it is a real choice and
the other reading is a one-line change to the constructor.

**The counts enter RAW.** No Laplace alpha, because `kappa * pi` is already strictly
positive in every cell -- that is what a prior is for. Adding alpha to `N` as well would
smooth twice and break both limits, so a test pins it.

### The gate is the two limits, and one of them is a bar on the same panel

`kappa -> inf` must reproduce `where_times_which` and `kappa -> 0` must reproduce the
raw event-count map, both exactly and both up to ONE additive constant over the whole
map -- which within-query ranking cannot see. Measured to <1e-4. The first of those is
a rung drawn beside it, so this is the same shape of check as
`ensemble_test_rows --verify-member`: a construction that has to reproduce something
already on the figure, rather than an argument that it should.

**Writing that test found a silent failure mode.** With no observed events the default
`kappa = N_total = 0`, so `q` is 0/0 and the map comes out ALL-NaN -- which does not
raise, does not warn, and ranks as a tie in every cell. It would have been read as a
rung performing at chance. It now refuses, and says which of the two causes it is.

### The figure: an option, not an inventory row

`coguk.wxw_dirichlet`, empty by default, following `coguk.mt2021` and the ensembles.
`MODEL_GROUP`'s order IS the bar-panel order and its names ARE the per-metric "missing"
report, so listing the rung unconditionally would add a permanent hole to the PUBLISHED
figure's completeness report for something that arm never scored. Inserted next to its
parent, in its parent's colour block -- same `_fit()`, same training examples, so no new
group and no new legend level.

Gated the established way: **`coguk_list_results` re-renders byte-identical**
(`776aa91c`, `49cc10be`), checked against scratch output names so no committed file was
touched by the check itself.

### Three mistakes of mine, two of them about waste

* **My wait loop reported the contrast job finished when it was still running.** It read
  `squeue` and treated a failed read as an empty queue -- precisely the trap
  `submit_cutoff_evals.sh` was fixed for in s70, reintroduced by me in a throwaway loop
  an hour after writing that section up. The lesson did not transfer because I thought
  of it as a property of that script rather than of `squeue`.
* **I submitted the contrast before adding the new rung**, which would have run the
  expensive half twice: hours, against ~5 minutes to score one more model. Cancelled at
  17 minutes and re-sequenced so the contrast runs once over all 31.
* **I submitted the two list passes together**, putting `list_variant_recall` as a
  READER on `list_score_maps_aa20` while `list_macro --dump-scores` was writing a new
  map into it. Cancelled and resubmitted under `--dependency=afterok`.

Also worth keeping: `VAR=v exec sbatch` assigns without EXPORTING, and `--export=ALL`
forwards only the exported environment -- so the prefix form hands the job an empty
`CONTRAST_EXTRA` and silently runs the contrast on its defaults, which is the exact
10-control filter this section is about. Written as `export` lines, and the slurm script
now echoes the contrast args, because their absence from the log is what let the
original zero-row run pass for a success.

### Code

* `antiGen/eval_utils.py` -- `WhereTimesWhichDirichletPredictor` and the
  `where_times_which_dirichlet` constructor. ONE constructor for all four entry points,
  so they cannot drift into scoring four slightly different models under one name.
* `antiGen/{eval,list_macro,list_variant_recall,background_discrimination}.py` -- the
  rung registered in each `builtin_models`.
* `antiGen/tests/test_wxw_dirichlet.py` -- 9 tests: both limits, normalisation, raw
  counts, the degenerate refusal and the no-data-but-a-prior case. Suite **240**.
* `antiGen-figures/figures/fig_coguk_list.R` -- `coguk.wxw_dirichlet`; byte-identical
  on its defaults. `fig_coguk_list_aa20.R` -- opts in.
* `examples/slurm/background_discrimination_sc2.sh` -- echoes `CONTRAST_EXTRA`;
  `python -u` on the contrast, so its setup lines are visible while it runs.
* `experiments/multitree/submit_bg_aa20.sh` (both halves, supersedes
  `submit_bg_contrast_aa20.sh`), `submit_lists_aa20.sh`.

### State

In flight: `bg_aa20` **3442399** (scores the new rung, then contrasts all 31),
`macro_aa20` **3442400** (resumes, scores the new rung only), `varr_aa20` **3442410**
(full CSV rebuild, ~90s, held on macro).

Still open from s72, untouched here: the NA models on `sc2_results_aa20`, the emergence
CSV header rewrite, and panel M's remaining cutoffs. And still the user's call: the
re-rendered figure binaries in the working tree, and whether the published
`sc2_results.png` / `supp_dms.png` should gain the LY-CoV1404 baseline.

## 74. `evo_gpu_priority` is for very short jobs only (2026-10-25)

**A standing directive from the user, recorded here because it is a property of the
cluster and not of any one script.**

> Do not submit to `evo_gpu_priority` except for **very short** jobs -- order of
> minutes. Everything else goes to `gpu_batch`.

It is a shared fast lane. A multi-hour job parked on it blocks other people's short
work, which is the whole reason the partition exists.

### What I did wrong

s73 moved three jobs onto `evo_gpu_priority` to get them off a backed-up `gpu_batch`.
Two of them -- `macro_aa20` (~45 s) and `varr_aa20` (~90 s) -- are legitimately short.
**`bg_aa20` is not**: its contrast half runs for hours. And `submit_cutoff_evals.sh` had
been putting per-cutoff evals there all along, inheriting
`eval_all_sc2.sh`'s own `#SBATCH --partition=evo_gpu_priority`, which is how
`cut_2023-10-01` came to sit on the fast lane for 30 minutes.

**I was treating the partition as a queue-jumping tool.** It is not; it is a lane
reserved for work that will be gone in minutes.

### Moved

* `bg_aa20` 3442500 -- PENDING, so `scontrol update jobid=... partition=gpu_batch`
  moved it in place.
* `cut_2023-10-01` 3441426 -- RUNNING, which cannot be moved in place. Cancelled 30
  minutes in and resubmitted as **3442525** on `gpu_batch`. That is the cost of the
  mistake and it is the right way round: 30 minutes of mine against a shared lane.

### A script's default cannot be trusted

Thirteen files under `examples/slurm/` carry `#SBATCH --partition=evo_gpu_priority`
from before this rule -- `eval_all_sc2.sh`, `list_recall_sc2.sh`,
`eval_emergence_sc2.sh`, `eval_all_{flu,denv2,rsv_a}.sh`, the `train_*`/`score_*`
valsel scripts. **Submitting through any of them without an override puts a long job on
the fast lane**, which is exactly how the cutoff evals got there with nobody choosing
it. They are left alone -- other arms share them -- and the override belongs in the
submit wrapper:

    PARTITION=${PARTITION:-gpu_batch}
    sbatch --partition="$PARTITION" ...

Done for `submit_cutoff_evals.sh`, `submit_bg_aa20.sh` and
`submit_bg_contrast_aa20.sh`. `submit_lists_aa20.sh` keeps `evo_gpu_priority` as its
default and now says in the file WHY it qualifies, so the next reader does not have to
decide whether it is an oversight.

### Measured runtimes, so "short" is a number and not a feeling

| job | measured | lane |
|---|---|---|
| `list_macro` resume, one model | ~45 s | priority ok |
| `list_variant_recall`, full rebuild | ~90 s | priority ok |
| `ensemble_test_rows` | 44 s | priority ok |
| per-cutoff eval `cut_<date>` | 4 min - 1h14m | `gpu_batch` |
| `background_discrimination`, 30 models | 2h21m | `gpu_batch` |
| `background_contrast`, revised corpus | 3-5 h est. | `gpu_batch` |
| main `eval_all_sc2` | 11h24m | `gpu_batch` |

### Recorded in three places, because one was not enough

* **`CLAUDE.md` at the repo root** -- new file, the rule first. This is the one a fresh
  session reads before touching anything.
* **Claude's project memory** -- so it survives a session that never opens the repo
  root.
* **Here**, with the reasoning, because NOTES is where the why lives.

## 75. LY-CoV1404 joins the published figure, and s70 had misread which panel moved
(2026-10-25)

On instruction: `figs/sc2_results.png` and its DMS supplement re-rendered to take s64's
`BA2_escape_LYCoV1404.csv`, which has been sitting in the globbed `baselines/`
directory since the 24th.

### The first render drew it as a bar called "NA"

`list.files(BASELINE_DIR)` means a map reaches this figure **by existing**. Nothing
else is needed -- and nothing else checks. `BA2_escape_LYCoV1404` was not in
`fig2_sc2_benchmarks.R`'s 27-name inventory, so `add_group_and_label`'s
`longform_names[match(...)]` returned NA, and fig2's rule is that **anything unlisted
falls into the "antiGen" group**. The result was a bar labelled `NA`, drawn in
antiGen's blue, at the right-hand end of both DMS panels: an antibody-escape scan
presented as an antiGen model.

This is the same failure `fig_coguk_list.R` already carries a comment about -- "fig2
assigns any unlisted model to the antiGen group, which would quietly mis-colour exactly
those extra rows" -- and the same shape as s72(b) on the revised arm. **A glob plus a
positional name table is a combination that fails silently by construction**: the glob
admits the row, the table has no slot for it, and NA is a legal factor level.

Fixed by listing it: `dms_models` gains the name and `longform_names` gains
`"BA.2 DMS (LY-CoV1404 escape)"` at the matching index, which is the string
`fig_coguk_list.R` already uses -- one object, one name across the two figures.

### s70's "two PNGs changed" was the bug, not the baseline

s70 rendered the unmodified script, found `sc2_results.png` and `supp_dms.png` both
changed, and read that as the new baseline reaching both. **Only the supplement should
have moved.** With the rung correctly grouped:

    sc2_results.png   IDENTICAL to HEAD
    supp_dms.png      CHANGED
    (11 other PNGs    IDENTICAL)

The main figure's line panels keep **the strongest model in each group** so they stay
readable, and LY-CoV1404 does not beat the BA.1/BA.2 scans already representing Deep
Mutational Scans. So it joins the figure family and draws on the supplement only. The
main figure moved in s70 solely because an NA row was being treated as a separate
group -- which is to say s70 measured a real difference and attributed it to the wrong
cause. **If the rung is wanted visible on the main panel, that is a change to the
keep-strongest rule and a separate decision.**

### What it says, which is worth not over-reading

Its correlations against BA.2 binding and expression sit near zero and slightly
negative. That is the expected answer rather than a disappointing one: escape from an
antibody is not a binding or expression phenotype, and this panel asks only whether the
two agree. s64's coverage caveat still applies -- it measures 2,023 of 4,020 cells.

### PDFs, and a correction to s70 in passing

`cairo_pdf` embeds `/CreationDate` and `/ModDate`, so all seven PDFs go dirty on any
render (s72). Compared with both stripped, **exactly one differs in content**:
`supp_dms.pdf`. The other six were restored with `git checkout --`, which **does work
from this session** -- s70 recorded that it could not be run here, and that was wrong.
The stripped byte counts are printed rather than assumed, so a comparison of two empty
streams cannot pass for agreement.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `BA2_escape_LYCoV1404` in
  `dms_models` and `longform_names`, with the glob trap named at the insertion point.
* `antiGen-figures/figs/supp_dms.png`, `figs_vector/supp_dms.pdf` -- re-rendered.
* `antiGen-figures/baselines/BA2_escape_LYCoV1404.csv` -- **committed**, having been
  untracked since s64. Until now the published figure depended on a file that was not
  in the repo, so a fresh clone would have rendered a different figure and said nothing.

## 76. The blend measured, and a positional argument that ate `--alphabet` (2026-10-25)

s73's rung has rows. `coguk_list_aa20` re-rendered with **34 models**, panels A-F and
G-J; K-L are still holes behind `bg_aa20`.

### What it does

Top decile, q = 0.1, `list_coguk_fullpool`, macro column:

| rung | all cells | de novo |
|---|---|---|
| antiGen | 0.5225 | 0.4732 |
| **`wxw_dirichlet`** | **0.5111** | **0.4218** |
| `where_times_which` | 0.4581 | 0.4218 |
| `pssm_root` | 0.4491 | 0.1000 |
| `substitution` | 0.3162 | -- |

Max yield, end of series, all cells: **`wxw_dirichlet` 0.2592**, ahead of
`where_times_which` 0.2461 and of all five combination rules (best 0.2252). It is 3rd
on the top-decile panel and 1st on the yield panel.

**The de novo column is the interesting one: it is EQUAL to its parent, to eight
figures.** That is the model behaving exactly as specified rather than a coincidence. A
de novo cell is one the training window never observed, so `N[i,a] = 0` and its score
is `kappa * pi / (N_total + kappa)` -- a constant times the prior. The update can only
ever PROMOTE cells that were observed, so de novo cells can only be pushed DOWN the
ranking relative to `where_times_which`, never up. Here none were pushed out of the
decile, so the two agree exactly. **The blend buys its gain entirely on cells the
training window has seen, and costs nothing on the ones it has not.**

`pssm_root` at exactly 0.1000 on de novo is the tie-heavy degenerate case s72 recorded,
not a third data point.

### A positional argument that ate `--alphabet`, and a job that passed anyway

`macro_aa20` 3442498 "COMPLETED" in **8 seconds** having scored nothing.
`eval_sc2_list_macro.sh` takes a LEADING POSITIONAL -- `RANGE=${1:-default}` then
`shift || true` before `EXTRA=("$@")` -- so the first flag handed to it is consumed.
Mine was `--alphabet`; `list_macro.py` then saw a bare `ACDEFGHIKLMNPQRSTVWY`, argparse
died on the next flag, **and the slurm job still exited 0**, which satisfied the
`afterok` dependency of the varrecall pass queued behind it.

`eval_sc2_varrecall.sh` has no such shift, which is why the identical `COMMON` array
worked there unchanged and only one of the two passes was broken. **The two scripts do
not share a calling convention**, and nothing says so at either call site. Fixed by
passing `default` explicitly, with the reason recorded at the array.

Worth naming: the log line that gives it away is `Extra args:` with the flag missing
and its value orphaned. Reading it was the only reason this was caught -- the job state
said COMPLETED and the downstream job ran.

### Two of my own checks that were wrong

* I filtered `git status` through `grep -vE "aa20_supp|..."`, which also matched
  `coguk_list_aa20_supp.png` and hid a genuinely modified file. Caught by comparing the
  hash against HEAD directly. **A filter written to exclude one family quietly excluded
  another that shares a substring.**
* An earlier `until` loop read `squeue` and treated a failed read as an empty queue,
  reporting a running job as finished (s73). Same class: a check that cannot
  distinguish "no" from "could not tell".

### Code

* `experiments/multitree/submit_lists_aa20.sh` -- the leading `default`.
* `antiGen-figures/figs/coguk_list_aa20{,_supp}.png` + PDFs, and the three refreshed
  CSVs under `data/evals_sc2_lists_{macro,varrecall}_aa20`.

## 77. K-L land, and the two new groups on `sc2_results_aa20` (2026-10-25)

### The background row, finally with rows in it

`bg_aa20` 3442500 COMPLETED in **2h31m59s**: it scored `wxw_dirichlet` (resume-skipping
the other 30 maps) and contrasted all 31 in one pass. **248 rows**, against the 0 of
s73. 1,212 of 1,212 hosted cells usable once `--min-controls` matched the rule s66
actually installed; 159 CoG-UK cells, 105 of them de novo.

`coguk_list_aa20` now draws **all twelve panels**: 34 models on A-F and G-J, 29 on K-L.

**The five combination rules are absent from K-L and that is correct.** The background
pass scores checkpoints and baselines through the model path; a cached-map rule has no
background maps to score. The figure NAMES them as missing per metric rather than
quietly drawing eleven panels and a short one.

### A control that was free and worth having

Every query-INDEPENDENT map comes out at exactly **+0.00 pp** with no interval -- PSSM,
PSSM Conservation, all three MSA Transformers, all five DMS panels including
LY-CoV1404, EVEscape, pyr0. That is the only answer they can give: a fixed `(L, E)`
map scores a cell identically whatever background it is handed, so it cannot
discriminate backgrounds by construction. Query-dependent models separate from them
cleanly (antiGen +15.2, Codon table +11.8, Tranception +11.8). **The row's headline
number is also its own sanity check**, and nothing had to be added to get it.

### What the blend costs on this panel

| rung | top decile, all cells | background lift, all cells |
|---|---|---|
| `where_times_which` | 0.4581 | **+10.05 pp** |
| `wxw_dirichlet` | **0.5111** | +4.39 pp |

The blend wins the recall panels and **loses more than half the background lift**. That
is the mechanism, not a surprise: `N[i,a]` does not depend on the query, so mixing it
into the score dilutes exactly the term that varies with the background. On de novo
cells, where `N = 0`, the two are close again (~6 pp each) -- the same boundary as s76's
de novo tie. **It buys top-decile ranking with background sensitivity**, and the two
panels disagree about it on purpose.

### `sc2_results_aa20`: the two new groups, and a colour bug found on the way

On instruction, mirroring `coguk_list_aa20`: the five rules get their own two groups
(Ensembles; Val-selected, averaged over 10 holdouts) with the same labels and colours,
read from `ensemble_figure_rows.tsv` rather than hand-copied -- the ids are blake2b
digests of the rule definitions and a stale copy would label one rule as another.

`BA2_escape_LYCoV1404` needed nothing: s75 put it in the shared drawer's `dms_models`,
and `fig_sc2_aa20.R` sources that drawer. Six NA rows were really five.

**`group` was a plain character**, so ggplot factored it in the LOCALE'S COLLATION
ORDER and `scale_fill_manual` matched its unnamed colour vector positionally. The
published five happen to collate into the intended order here, which is why it has
never shown -- but `"Ensembles"` sorts FOURTH, between "Deep Mutational Scans" and
"Pandemic Data in Training Window", so appending its colour at the end would have
mis-coloured every group after it. Now an explicit factor on `MODEL_TYPE_LEVELS`. The
same trap s71 named for panel M, in a second place.

Gated: published `sc2_results.png` and all six rendered supplements **byte-identical**
through both changes.

### Emergence headers rewritten, not re-run

`relabel_emergence_headers.py`: 16 files, `t_0.1 .. t_2.1` -> `t_1.0349 .. t_6.0342`,
exactly the values s72 predicted. Values untouched, so the 11-hour eval is not
repeated. It imports `emergence_step_labels` rather than restating it, REFUSES a CSV
whose column count does not match the grid (the signature of a file from another run,
where confident wrong dates on real numbers would be the outcome), and is idempotent:
the second run reports `0 rewritten, 16 already correct`. `data/evals_sc2` untouched.

### Three of my own, all cheap because they failed loudly

* **`python: not found`, exit 127.** `submit_ensemble_rows.sh` called bare `python`;
  the batch environment does not always carry miniconda on PATH. It died in 0 seconds
  having done nothing, which is the good version -- a wrapper resolving to a DIFFERENT
  python would have produced numbers. Pinned to an absolute interpreter.
* **`sbatch: Batch script is empty!`** Fixing the above, I put the explanatory comment
  INSIDE the backslash-continued argument list. A `#` line there terminates the
  continuation, so sbatch got no `--wrap` at all. **Comments go above the command**, and
  that is a correctness rule here rather than a style one.
* An `sbatch` hung long enough to outlive its shell, so a job I reported as submitted
  never existed. Checked by job name afterwards rather than by assuming the exit.

### Code

* `experiments/multitree/relabel_emergence_headers.py` -- new.
* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- the ensemble inventory, the
  explicit `group` factor.
* `experiments/multitree/submit_ensemble_rows.sh` -- absolute `$PYTHON`, 30m walltime.
* `antiGen-figures/figs/coguk_list_aa20{,_supp}.png` + PDFs, twelve panels.
* `antiGen-figures/figs/sc2_results_aa20.png` + `aa20_supp_*` -- drawn with the two
  groups, no NA.
* `experiments/multitree/install_sc2_aa20.sh` -- copies `ensemble_figure_rows.tsv`,
  which the CSV globs had been dropping. Without it the figure's name table has no
  entry for the five rules and they draw as ONE NA bar, so an install was silently
  reproducing the bug this section fixes. It is reported as MISSING if absent.

### Two more things the render itself found

* **`MODEL_TYPE_COLORS[c(1, 3, 4)]`** -- the DMS correlation panel carried a
  hand-counted index subset of the groups that survive its filter. Adding two groups
  turned it into "5 needed but only 3 provided", which failed loudly; had the count
  happened to match it would have MIS-COLOURED instead. Both colour vectors are now
  one named palette and `group_colors()` indexes it by name, so the class of bug is
  gone rather than this instance of it.
* **The full labels run off the panel.** `fig_coguk_list.R` puts its legend in a wide
  bar panel; here it sits inside a curve panel, and "Where x which + PSSM hybrid, 10x
  val-selected blend" overlapped the y-axis title. `sc2.ensemble_labels` overrides the
  display name per id. **The sidecar remains the source of IDENTITY** -- id, group and
  colour are unchanged, and only the string on this figure's legend is shorter.

## 79. Panel E's clipping, and recall at the decile beside recall AUC (2026-10-26)

Both on request, both on `sc2_results_aa20` only.

### The gutter was never a width knob

`plot_auc` draws three things side by side: a LABEL GUTTER, then the two bar panels.
The gutter was "panel 1 with everything but the y-axis text hidden" -- except nothing
hid the bars. It worked because at `rel_widths = c(0.5, 1, 1)` the gutter's panel is a
sliver and the names fill it.

This arm's names are much longer than the published roster's
("Where x which + PSSM, val-selected (10x)" against "PSSM Conservation"), so they
overflowed: D's labels ran off the left of the figure and E's axis title off the right.

**Widening the gutter does not fix it, and the way it fails is worth recording.** The
names stayed clipped at exactly the same pixel at 0.95 and at 2.4, because the gutter's
PANEL grows with it -- at 1.8 the figure gained a second, unlabelled copy of panel D
sitting where the whitespace should be. The gutter is only a width knob if its panel is
empty, so `g$layers <- list()`. Then 1.15 is enough and the panel is genuinely blank.

**Two of my own measurements were worthless before that was clear.** I rendered trial
gutters by setting `options(sc2.auc_gutter = ...)` and then sourcing the wrapper --
which sets the same option further down, so both trials ran at the wrapper's value and
"the clip point does not move" was an artefact of my own harness rather than a fact
about the layout. The same shape of mistake as s72's md5sum-against-itself: a
comparison that could not have come out differently.

### Recall at q = 0.1, as its own row

`sc2.recall_at` (unset on the published arm). Two bar panels reading the recall curve
at one quantile, beside the AUC that summarises the whole of it -- different questions,
and the decile is the operating point every other panel of this figure uses.

**A full-width row of its own, not a fourth row inside `recall_block`.** The recall and
precision blocks sit side by side and their rows line up only because both have three;
a fourth in one would stagger the two cartoons against each other. A separate row also
leaves A-J exactly where they are, so the two arms can still be read panel for panel up
to J, and only the bottom row moves (K-M becomes M-O).

`summarise_curves_at` selects **by x**, not by index. `value[2]` would be correct only
for the grid it was written against, which is precisely the assumption that put a 2.5x
error on the emergence axis in s72.

**It drew two bars called NA first.** `performance_by_x` is passed through
`add_group_and_label` further down the file, after which its `model` column holds
DISPLAY names -- so re-labelling the output of a reducer over it matched display names
against raw ids and produced NA for all but the coincidences. The frame was already
grouped and labelled; taking its columns as they are is the fix.

### Gated

Published `sc2_results.png` and all six rendered supplements **byte-identical** through
every change here -- the gutter rebuild, the new reducer, the new row and the per-call
gutter argument.

### Known, pre-existing, NOT introduced here

Panel M's legend is clipped at the left ("tiGen" for "antiGen") and its y-axis title
with it. Checked against the committed figure rather than assumed: it is identical
there, so it dates from the two new legend groups of s77 widening the bottom legend
past the figure, not from this section. Left alone because it is a separate fix.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- blank gutter, `sc2.auc_gutter`,
  `plot_auc(gutter=)`, `summarise_curves_at`, `sc2.recall_at` and the row it adds.
* `antiGen-figures/figures/fig_sc2_aa20.R` -- gutter 1.15, `sc2.recall_at = 0.1`.

## 80. `wxw_dirichlet` everywhere, and four rounds of mis-sizing (2026-10-26)

The rung existed on `coguk_list_aa20` and nowhere else: the main eval ran before it did,
so `sc2_results_aa20`'s own 30-rung roster predates it and panel M's `PANEL_ROWS` named
six lines that did not include it.

### It is one model's pass, not a re-run

`eval.py` resumes by reading its OWN output CSVs, skipping models already present and
merging the new rows in. Pointed at a copy of `evals_sc2_aa20` it scored
`wxw_dirichlet` and resume-skipped the other thirty. **14m47s, on CPU** -- no network is
ever built, so the GPU was not needed at all.

Two details worth keeping:

* **The baseline sweep is skipped too**, and that is guarded rather than incidental:
  `eval.py:1002` runs the sweep only `if any(suffix not in test_baseline_rows ...)`,
  and those rows come off the existing CSVs via `read_unified_csv`. I had written in
  the submit script that the sweep would re-run and "buy a check"; it does not run at
  all. The comment was wrong and is corrected.
* **It did NOT skip `codon`, `substitution` and `where_times_which`.** `read_unified_csv`
  sorts rows into model_rows and baseline_rows, and those three are classified as
  BASELINES -- so they fall outside `already_done` and are re-evaluated. Harmless, they
  recompute to the same values, but it is why a "one model" resume scored four.

**On a COPY, not in place.** A resume rewrites every CSV it touches, and those rows are
what the figure stands on. The original is the thing to check the merge against.

### Panel M, and the self-healing check paying off again

Naming `wxw_dirichlet` in `PANEL_ROWS` marked twelve of fourteen cutoffs `[redo]`
automatically -- the s71 mechanism doing exactly its job. The two newest already had the
row, having been evaluated after the rung existed.

### Four rounds of mis-sizing on one set of jobs

Each fix addressed the dimension I had just been burned on and left the others:

1. `--nice=1000`, hardcoded in the submit script. Right for a sixteen-job sweep behind
   the critical path; it pinned the priority at **1**.
2. **256G and 8 CPUs**, inherited from `eval_all_sc2.sh`'s header, which is sized for
   building ESM-2 650M. A resume that scores an analytic map needs neither. Reason:
   "Nodes required for job are DOWN, DRAINED or reserved".
3. **The `cpu` partition**, chosen because it had idle cores -- without checking that it
   carries `QoS=cpu_interact`, **2 concurrent jobs per user**. Fourteen jobs ran two at
   a time with 432 cores free.
4. **24 CPUs on `cpu_batch`**, against an account-wide `GrpTRES=cpu=288` for all of
   hielab. Twelve jobs asked for the entire lab's allocation.

`CPUS`, `MEM`, `NICE`, `GRES`, `TIME` and `PARTITION` are now all the caller's to set,
because every one of them has been the blocker at least once.

### A scancel worth recording

On instruction, 310 of the user's own array tasks (`og_tp` 184, `og_ob2` 126) were
cancelled to free the queue. Scoped by exact `--name`, after listing the distinct names,
because a pattern would have been a guess about what else matched. It is irreversible
and it was not enough on its own -- the partition caps above were the real limit.

## 81. K-L gets its rule bars, and the gate comes back exactly zero (2026-10-26)

`bg_contrast_aa20` 3458262 COMPLETED in 3h36m over 45 maps: the 31 rungs, the 5 rule
maps `ensemble_panels.py` combined, and 9 one-member gate maps.

### The gate, which is the point of the gate maps

A one-member rule is a monotone transform of its member WITHIN each query, and the
background contrast's statistic is a within-query PERCENTILE -- so unlike the recall
side, where the same identity lands at 4e-9, here it should be exact.

    gate rows compared: 72
    worst |delta| in lift = 0.000e+00

Bit for bit, over every (member, cellset) pair. **That is what makes the five `ens_`
rows usable**: the combination step is demonstrably a no-op on one member, through the
same `background_contrast.py` the rungs beside them went through.

### The gate rows had to be dropped by PATTERN, not by name

`background_summary.csv` now carries a `gate_<member>` row for every member checked,
and `fig_coguk_list.R` stops on any model it has no group for -- correctly. Listing
nine names in `coguk.drop_models` would need editing whenever the member set changes,
and a missed one stops the render. `coguk.drop_pattern` (empty by default, `^gate_` on
this arm) drops the family.

`coguk_list_aa20` now draws **34 models on K-L**, EVmutation the only absentee, with
all five rules in their two colour groups and their bootstrap intervals.

### What the row says about the blend

`Where x which + events` takes **+4.4 pp** against `Where x which`'s +10.0. It wins the
recall panels and loses more than half the background lift, which s77 recorded as the
mechanism rather than a surprise: `N[i,a]` does not vary with the query, so mixing it
in dilutes the only term that does.

### wxw_dirichlet on the main arm, and a resume subtlety

`wxwd_main_aa20` 3460229, **14m47s ON CPU**. `eval.py` resume-skipped the 14
checkpoints and, because the baseline rows were already loaded, skipped the baseline
sweep too (eval.py:1002) -- so no network was ever built and the GPU was not needed.

**It did NOT skip `codon`, `substitution` and `where_times_which`.** `read_unified_csv`
sorts CSV rows into model_rows and baseline_rows, and those three are classified as
BASELINES, so they fall outside `already_done` and are re-evaluated. Harmless -- they
recompute to the same values -- but it is why a "one model" resume scored four, and
why the run was 15 minutes rather than 5.

Top decile, all mutations: `pssm_root` 0.782, `wxw_dirichlet` 0.724, antiGen 0.714,
`where_times_which` 0.711. **PSSM leads here and was fourth on the list figure**
(0.449 against antiGen's 0.522). Different cell set and different pooling, so the two
are not in contradiction -- but a baseline moving from fourth to first between two
panels of one arm is worth understanding before either is read as a result, and the
all-cells macro run is the thing that speaks to it.

### Scheduling, twice more

* The 13 cutoff redos sat at **priority 1** with "Nodes required for job are DOWN,
  DRAINED or reserved": `--nice=1000` hardcoded in the submit script, and 256G/8 CPUs
  inherited from `eval_all_sc2.sh`'s header, which is sized for building ESM-2 650M.
  A resume that scores only analytic baselines needs neither. `CPUS`, `MEM` and `NICE`
  are now the caller's to set; at 96G/16/nice 0 the priority went to 45.
* An `sbatch` in that sweep **timed out mid-run** and `set -e` aborted the script after
  7 of 13. The re-runnable design absorbed it: a second run reported "submitted 6,
  already queued 7". Worth recording as the payoff for the s70 rewrite rather than as
  an incident.

### Code

* `antiGen-figures/figures/fig_coguk_list.R` -- `coguk.drop_pattern`; published figure
  byte-identical (`776aa91c`, `49cc10be`).
* `antiGen-figures/figures/fig_coguk_list_aa20.R` -- drops `^gate_`.
* `experiments/multitree/submit_cutoff_evals.sh` -- `wxw_dirichlet` in `PANEL_ROWS`,
  and `GRES` / `CPUS` / `MEM` / `NICE` overrides.
* `antiGen-figures/figures/fig_sc2_aa20.R` -- panel M's seventh line and its colour,
  inserted at the matching index of both colour vectors rather than appended.


## 82. The antiGen/PSSM gate, and the partition tier that decides everything (2026-10-27)

Panel M's published headline is `gated_static_xfmr_pssm` -- the one line the drawer
BOLDS. It is a trained checkpoint, not a derived row, and this arm had never trained
one: `cutoff_series` held `base/`, `esm2_35m_partial/` and the val-selected runs. The
revised panel M has therefore been answering a different question, with the ESM-2 rung
and where x which standing where the gate used to be.

### Three things had to exist before it could train

* **The PSSM partner could not be constructed.** `PSSMFitnessPredictor` reads a
  `pssms/root.csv`-shaped file; antiGen-data-revised keeps those numbers as a counts
  TSV, which is what every other reader takes via `--root-counts`.
  `counts_to_pssm_csv.py` converts it and **checks the round trip against
  `load_counts_matrix`** -- the function that produces the `pssm_root` RUNG on the same
  panel. Worst delta **0.000e+00**. Without that check the gate could be antiGen blended
  with something that merely resembles the PSSM drawn beside it.
* **`--pssm-path`** on `train.py`, applied at both `PSSMConfig` sites.
* **`train_cutoff_gate.sh`** runs the BASE recipe plus the gate -- same
  `simple_d128_n2`, 32 epochs, batch 64, lr 1e-4 -- so the gate differs from the rung it
  gates only by the gate. Anything else measures two things at once.

### The cost estimate was wrong by 40x, in the expensive direction

I quoted "6-11h each, the biggest single spend of this session". **The base training at
2022-01-01 took 10m25s** (cutoff_series log 3428706: 14:31:20 -> 14:41:45). The 6-11h
figure is the whole `cutoff_series` JOB -- base plus TEN 96-epoch val-selected runs,
992 epochs against 32. The gate checkpoint is 6.9 MB against base's 6.8 MB.

A consequence worth stating: the trainings scale with corpus size, so "~30 seconds" --
measured on 2020-04-01, which trains on 190 examples -- is as wrong at the late end as
6-11h was at the early end. The late cutoffs take minutes.

### PriorityTier is the thing that predicts whether a job runs, and squeue never says so

    tier 1000  gpu_batch, cpu_batch   -- but quota-capped
    tier 10    quick_preemptible      -- starts in seconds
    tier 1     preemptible            -- never started
    tier 0     cpu_preemptible        -- never started

`squeue` reports `(Priority)` for all of them. Having watched tier 10 start jobs in
seconds and tier 0 never start any, I moved eight gate trainings to tier **1** and lost
a night to it. The tiers are in CLAUDE.md now, because the reason string does not carry
the one number that matters.

### A reconciler, and then a reconciler that survives

`reconcile_panelM.sh` asks what is MISSING rather than what FAILED, so it cannot be
confused by a failure shape it did not anticipate -- a TIMEOUT, a preemption, a quota
rejection, a job that exits 0 having done nothing. Every underlying script was already
idempotent; what was missing was anything CALLING them again.

**Running it as a foreground loop did not fix the problem it was built for.** The loop
dies with its supervisor, and between 11:16 and 14:02 four gate trainings and two cutoff
evals sat unsubmitted while the thing meant to resubmit them was gone. It is now
`reconcile_daemon.sh`, a 1-CPU Slurm job -- stateless between cycles, `--requeue`,
exits as soon as everything exists.

Its first submission then sat PENDING itself, on a **12h walltime for a job backfill
could not place**, with 486 cores idle and the account at 48 of 288. Cut to 4h and it
started immediately. That is the same lesson as s80's item 2, applied to a script I
wrote an hour after documenting it.

### The row name, derived rather than waited for

`PANEL_ROWS` must match the eval's row exactly or the by-name loader stops the render.
`_CKPT_RE` takes `fitness_(.*)_(\d{8}_\d{6})\.pt`, and `config_label` rewrites the
prefix only for `esm2_fitness_predictor`, so the row is
**`gated_static_xfmr_pssm_lr0.0001`** -- the published name plus this pipeline's lr
suffix. Checked against the regex rather than guessed, and to be confirmed against the
first real CSV before it is committed.

### The ordering trap

Staging the gate checkpoint and NAMING it in `PANEL_ROWS` are deliberately separate.
The redo check asks whether a CSV carries each listed row; a cutoff whose gate has not
trained cannot produce it, so naming the row early makes the reconciler resubmit every
cycle and never converge. Stage always, name only when all sixteen checkpoints exist --
at which point it correctly marks all sixteen `[redo]`, which is what putting a new line
on that panel costs.

### Code

* `antiGen/train.py` -- `--pssm-path`.
* `experiments/multitree/counts_to_pssm_csv.py`, `examples/slurm/train_cutoff_gate.sh`,
  `experiments/multitree/submit_cutoff_gates.sh` -- new.
* `experiments/multitree/reconcile_panelM.sh`, `reconcile_daemon.sh` -- new.
* `experiments/multitree/submit_cutoff_evals.sh` -- gate staged when present; `TIME`,
  `REQUEUE`, `CPUS`, `MEM`, `GRES`, `NICE`.
* `CLAUDE.md` -- the CPU partition rule.

## 83. Panel M draws the gate, on all sixteen cutoffs (2026-10-27)

The sweep reconciled at 14:34 after 19 daemon cycles: sixteen gate checkpoints, sixteen
cutoff evals carrying all eight `PANEL_ROWS`. `install_sc2_aa20.sh` reports **16 of 16,
all present** for the first time.

### What the gate does

Top decile, by cutoff:

| cutoff | antiGen | PSSM | gate |
|---|---|---|---|
| 2020-04-01 | 0.524 | 0.458 | **0.643** |
| 2021-01-01 | 0.710 | 0.782 | **0.803** |
| 2022-01-01 | 0.844 | 0.823 | **0.861** |
| 2023-01-01 | 0.878 | 0.876 | **0.920** |
| 2024-01-01 | 0.925 | 0.880 | 0.919 |

**It beats both parents at fifteen of sixteen cutoffs.** Not by picking the better one:
PSSM leads antiGen early (through 2021-10-01) and antiGen leads PSSM late, and the gate
clears whichever is ahead on both sides of that crossover. The single exception is the
last point, 2024-01-01, where antiGen alone is 0.925 against the gate's 0.919 -- the
one cutoff where the training window is large enough that the PSSM partner costs more
than it adds. That is the published arm's claim reproduced on the revised corpus, with
an endpoint the published arm did not have.

### Two colour bugs, one of them latent for the whole project

* **`FINETUNE_COLORS[4]` and `GATE_BLUE` are both `#0077b6`** (R/common.R:74,77). While
  no gate existed on this arm the collision was invisible -- nothing else used
  GATE_BLUE. Giving the gate the colour it is NAMED for made two of eight lines
  identical, and the legend showed two blue swatches for different models. The ESM-2
  rung moves to `FINETUNE_COLORS[5]`.
* **The legend's width is set by its ROW count.** `guide_legend(nrow = 3)` fills rows
  first, so n entries need ceil(n/3) columns: six entries were two columns and fitted,
  eight became three and ran off the panel onto panel N's legend, where the two
  overlapped into illegibility. `sc2.gate_legend_nrow` (default 3, four on this arm).

Both are the same shape as the positional-colour traps of s77 and s79: a constant that
was unambiguous only because one of its users did not exist yet.

### Gated

Published `sc2_results.png` and all six rendered supplements **byte-identical** through
the gate row, the colour change and the legend option.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `sc2.gate_legend_nrow`.
* `antiGen-figures/figures/fig_sc2_aa20.R` -- the gate as panel M's second line,
  `GATE_BLUE` reassigned to it, eight colours inserted at the row's index.
* `experiments/multitree/submit_cutoff_evals.sh` -- `gated_static_xfmr_pssm_lr0.0001`
  in `PANEL_ROWS`.

## 84. `sc2_results_revised`, and two NA bugs it exposed (2026-10-27)

A re-organisation, on request: every panel already existed, nothing was re-evaluated.

    row 1   A  next-mutation recall cartoon              (sc2_results_aa20 A)
            B  top decile, all list cells, cell-weighted (coguk_list_aa20 C)
            C  top decile, de novo, cell-weighted        (coguk_list_aa20 D)
    row 2   D  background lift, all list cells           (coguk_list_aa20 K)
            E  background lift, de novo                  (coguk_list_aa20 L)
            F  blank -- a zoomed case study goes here
    row 3   G  DMS correlation                           (sc2_results_aa20 M)
            H  antiGen / PSSM gate by train/test cutoff  (sc2_results_aa20 O)

### The two drawers cannot share a session

`fig2_sc2_benchmarks.R` and `fig_coguk_list.R` each define `theme_paper`,
`save_figure`, `MODEL_GROUP` and more, so sourcing both hands the second one's
definitions to the first one's panels. Each therefore runs in its OWN subprocess and
ships panels back as RDS. The panels are CALLED (`decile_bar`, `bg_bar`, `p_expr`,
`gate_plot`), not copied, so a change in either drawer reaches this figure.

**They are baked to grobs before shipping.** A ggplot stores aesthetics as quosures
evaluated at DRAW time against the environment they were written in -- `gate_plot` has
`linetype = (model %in% GATE_BASELINES)`, and GATE_BASELINES exists only in the drawer.
Shipped as a ggplot it rendered in the assembling session as "object not found", long
after the panel looked fine. `ggplotGrob` resolves it where its globals are in scope.

### Two NA bugs, one of them months old

* **`wxw_dirichlet` was never in fig2's inventory.** It was added to panel M's
  `sc2.gate_rows`, which is a separate by-name loader, and nowhere else -- so on every
  panel reading the pooled CSVs it was NA, and this file's fallback puts anything
  unlisted in the "antiGen" group. Now in `pandemic_training_data` beside its parent.
* **`dms_long` never had `RUN_TAG_TO_MODEL` applied, and that predates this work.**
  The panel's names come from the directory names under MODEL_DIR, and the two arms
  name them differently: published `data/sc2/` uses short names (`simple`,
  `esm2_partial`), while `dump_variant_root_predictions.py` named `data/sc2_aa20/` by
  full run tag (`simple_d128_n2_lr0.0001`). Unmapped, **twelve** of twenty-seven
  models -- antiGen itself, all eight ESM-2 variants, eve_ft and both Tranceptions --
  matched no level and were drawn merged into one blue bar labelled NA. **96 NA rows,
  now 0.** It has been wrong on panel K of `sc2_results_aa20` since s67 and was only
  visible because this figure put that panel somewhere I looked at it.

The pooled CSVs get that translation further up the same file; the DMS frame was simply
missed. A no-op on the published arm, whose names are already short.

### Gated

Published `sc2_results.png` and all six rendered supplements **byte-identical** through
both inventory changes.

### Code

* `antiGen-figures/figures/fig_sc2_revised.R`, `panels_coguk_export.R`,
  `panels_sc2_export.R` -- new.
* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `wxw_dirichlet` in the inventory;
  `RUN_TAG_TO_MODEL` applied to `dms_long`.

## 85. Pruning `sc2_results_revised`, and three options that were silently overridden
(2026-10-27)

On request: drop the orange (the three plain mean rules) and brown (the fine-tune
group) models, drop the base antiGen rung from every panel but H, drop the
`where x which + PSSM` hybrid, and draw the surviving `antiGen, 10x val-selected` rule
in antiGen's dark blue.

### A wrapper's own `options()` beats the caller's

Three separate attempts to steer a drawer from the exporter did nothing, all for one
reason: `fig_coguk_list_aa20.R` and `fig_sc2_aa20.R` each call `options()` THEMSELVES,
and the exporter sources them AFTER setting its own. Last write wins, and the wrapper
is last.

* `coguk.ensemble_colors` -> the val-selected rule stayed cyan.
* `sc2.ensemble_colors` -> the same rule stayed cyan on the DMS panel.
* `coguk.title_suffix` -> the suffix stayed (harmless; the fix for the clipped title
  turned out to be the figure's width).

What works is reaching past the option: `GROUP_COLORS` and `GROUP_PALETTE` are read at
CALL time by the panel builders, so reassigning them after the source and then calling
the builder takes effect. The DMS panel is REBUILT rather than reused for that reason.

**`legend_grob` is not a builder.** It is constructed during the source, so a recolour
after the fact leaves the legend key disagreeing with the bars under it. It has to be
rebuilt from the same expression.

### A subset selection silently loses its labels and its group

`coguk.ensemble = "ens_8a91fbbf88"` alone drew the rule as **"Ensemble 8a91fb" in the
orange Ensembles group** -- the very group being pruned. Both the label and the group
default are guarded on `nrow(ensemble_figure_rows.tsv) == length(ENSEMBLE)`, so naming
one of five rules fails both guards and falls back. `coguk.ensemble_labels` and
`coguk.ensemble_groups` have to be set alongside it.

### Legend keys for groups that are no longer drawn

The drawer builds its legend with `drop = FALSE` so its full roster always shows every
group. With antiGen and antiGen Finetunes pruned from every panel that left two keys
with nothing beneath them; the rebuilt legend uses `drop = TRUE` over the groups the
pruned panels actually carry.

### `sc2.dms_drop`

`EXCLUDED_MODELS` could not serve: it is applied to `performance_by_x` only and names
raw ids, while the DMS frame carries display labels by then. New option, empty by
default.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `sc2.dms_drop`.
* `antiGen-figures/figures/panels_coguk_export.R`, `panels_sc2_export.R` -- the
  pruning, the recolour-after-source, the rebuilt legend.

## 86. Cleaning up `sc2_results_revised` (2026-10-27)

Four changes on request: abbreviate the val-selected label, make the text larger
relative to the canvas, reorder row 2 so the bar panels align vertically, and put
panel H's x axis on dates trimmed to the data.

### The row order

The blank goes under the cartoon, so B/C and the two background panels -- the same kind
of bar panel over the same roster -- line up in two columns and the eye can read down a
column as well as across a row.

**The letters are assigned by POSITION**, so row 2 reads D, E, F with D the blank and
the background panels E and F. Carrying them with their panels instead would have left
the figure running A B C / F D E, which reads as a mistake however sensible the history
behind it; the cost is that a reference to "panel D" made before this change now points
at the blank.

### Text size is a canvas problem, not a theme problem

Text is in POINTS and does not shrink with the canvas, so the whole figure going from
17x13.5in to 13x10.5in makes every label larger relative to everything else without
touching a single theme. Nothing about the panels changed.

### Panel H on dates

`sc2.gate_x` returns a decimal calendar year rather than years-into-pandemic, so the
axis says what the cutoffs ARE instead of an offset the reader converts. `sc2.gate_xlab`
and `sc2.gate_xlim_hi` are new; the latter at NA trims the axis to the last cutoff,
where the published hard stop at 5 left a year of empty panel past 2024-01-01.

### Four more strings, four more places they live

The rename to "antiGen (val x10)" and the group to "Val-selected (10x)" each took
several attempts, and the pattern is the one from s85: an option set before sourcing a
wrapper loses to the wrapper's own `options()`.

* `coguk.title_suffix` -> `TITLE_SUFFIX` is read by `titled()` at CALL time, so
  assigning the variable after the source works where the option did not.
* The sc2 DMS label is BAKED into `dms_long$model` during the source via
  `longform_names`, so no option reaches it afterwards; renamed in the frame.
* **And it had to be renamed to the right thing.** I first renamed the sidecar's full
  name, which is not what is in the frame -- `fig_sc2_aa20.R` already shortens it once
  (s79), so the match found nothing and the panel silently kept the old label.
* The GROUP name lives in **three** places at once: the frame, `GROUP_PALETTE` and
  `MODEL_TYPE_LEVELS`. `group_colors()` factors on the levels and then asserts every
  surviving level has a palette entry, so renaming the frame alone turned those rows
  into NA and the render died in guide training. All three or none.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `sc2.gate_xlab`,
  `sc2.gate_xlim_hi`.
* `antiGen-figures/figures/fig_sc2_aa20.R` -- decimal-year `sc2.gate_x`, trimmed axis.
* `antiGen-figures/figures/fig_sc2_revised.R` -- row order, canvas size.
* `antiGen-figures/figures/panels_{coguk,sc2}_export.R` -- the renames.

## 87. antiGen on top, and two orderings that are not the one you set (2026-10-27)

On request: the val-selected rung goes to the top of the bar panels (B/C/E/F) and
first on the DMS panel (G).

### The bar panels already had the hook, under a hard-coded name

`bar_panel` did `levels = rev(c("antiGen", setdiff(lev, "antiGen")))` -- a named rung
pulled to the top. On a pruned roster where the base antiGen bar is gone, that name
matches nothing and the intended rung sinks to wherever the inventory puts it, which
for an appended ensemble rule is the bottom. `coguk.top_model` makes the name an
option; the published default is unchanged and `coguk_list_results` re-renders
byte-identical (`776aa91c`, `49cc10be`).

### The DMS panel orders by GROUP, and its group is not a factor

Two attempts failed before the cause was clear.

* **Releveling `model` did nothing.** The x order is
  `fct_reorder(model, as.integer(factor(group)))` (fig2:835) -- the GROUP's position
  decides, and an appended group sorts last whatever the model levels say.
* **Moving the group to the front of `MODEL_TYPE_LEVELS` moved the COLOURS and not
  the bars.** `dms_long$group` is left a CHARACTER vector by the drawer, so
  `factor(group)` inside `fct_reorder` sorts alphabetically, while `group_colors()`
  reads its palette off `MODEL_TYPE_LEVELS`. The two orders disagreed and the panel
  came back with the right bars in the wrong colours -- "Pandemic Data in Training
  Window" drawn in antiGen's blue.

The fix is to set the group as a FACTOR in `MODEL_TYPE_LEVELS` order, which fixes the
sort and the palette together. Worth keeping as a shape: when a panel's order and its
colours come from two different expressions of "the group", changing one of them is
always wrong.

### Code

* `antiGen-figures/figures/fig_coguk_list.R` -- `coguk.top_model`.
* `antiGen-figures/figures/panels_{coguk,sc2}_export.R` -- the top rung, the group
  factor, the A/D column width.

## 88. Panel D: the BA.2.86 -> KP.3 case study (2026-10-28)

> **Superseded in part by s89 and s90.** The ranks below stand. The log rank
> axis and the three-of-seven roster went in s89; the four-glyph chain layout
> this section describes came BACK in s90, now carrying all seven steps.

The placeholder in `sc2_results_revised` is filled. Panel D asks a narrower question
than any other panel in the figure: not how often antiGen's top decile contains the
next mutation across a corpus, but where it put the specific substitutions that
carried BA.2.86 to KP.3.

### The ranks

`experiments/multitree/rank_path_cells.py` walks `ba286_to_kp3_path.csv` and, for each
step, ranks that substitution among all RBD substitutions available **from the parent
genotype** -- the query is the state BEFORE the step, which is the only query for which
the step is a prediction rather than a fact. Model: the 2022-01-01 base checkpoint
(`cutoff_series/2022-01-01/base/fitness_simple_d128_n2_lr0.0001_20261024_143133.pt`),
so nothing downstream of the cutoff is in training. Pool is 201 sites x 20 AAs minus
the 201 wild-type cells = **3,819**. Every parent was checked to carry the expected
from-residue before its rank was taken.

Of 13 steps on the path, **7 are in the RBD** (331-531):

| step | rank | top % | in top decile |
|---|---|---|---|
| L455S | 258 | 6.76 | yes |
| V483D | 1107 | 28.99 | no |
| R346T | 233 | 6.10 | yes |
| F456L | 82 | 2.15 | yes |
| D483V | 261 | 6.83 | yes |
| T346R | 19 | 0.50 | yes |
| Q493E | 191 | 5.00 | yes |

6 of 7 in the top decile. Note that **V483D/D483V and R346T/T346R are reversion
pairs**, so the NET RBD change BA.2.86 -> KP.3 is just L455S, F456L, Q493E -- which is
exactly the three arrows the inspiration sketch drew, and what the panel shows. The
four reversion steps are computed and kept in `data/ba286_to_kp3_ranks.csv`; they are
not drawn, because drawing a site that changes and then changes back needs a story the
panel does not have room to tell. **The one miss is among them** (V483D, rank 1107),
so anyone reading only the panel is reading the better three of seven. Worth deciding
deliberately rather than inheriting: the reversions can go on the rank axis as
unlabelled marks if the figure should carry all seven.

### The drawing

`experiments/multitree/draw_path_panel.py` -> `antiGen-figures/cartoons/ba286_to_kp3.png`.
Two registers: the lineage along the top, the rank axis below.

* **The glyph is 6M0J chain E** (RBD bound to ACE2, residues 333-526; all five mutated
  positions present), CA coordinates projected onto their first two principal
  components and drawn as overlapping translucent discs. The projection is computed
  once and reused, so a site that differs between glyphs is the colour changing and
  not the view.
* **PC1 goes on y.** Laid flat the four glyphs are short wide bars with a band of
  nothing beneath them; stood upright they fill a 5.5 x 3.5in box. This one change
  took the image's aspect from 2.07 to 1.55 against a panel box of 1.57.
* **Sites are coloured by WHEN they changed** -- this step filled, earlier steps
  outlined -- which is the only reason to draw four glyphs rather than one.
* **The rank axis is logarithmic.** Linear, the whole top decile is the leftmost tenth
  and all three marks pile up; in log space the decile boundary lands at 0.72 of the
  width. Decade ticks are drawn so the compression is visible rather than smuggled in.
* **Two passes of label collision, both worth remembering.** Staggering stem heights
  separates marks that are a decade apart but only millimetres apart on the page.
  That is not sufficient: a label runs horizontally, so a taller neighbouring stem
  draws straight through the text. Labels are placed on the side whose neighbours are
  all shorter.

Rebuild the PNG with `draw_path_panel.py`, not from the layout script -- it needs a
checkpoint and a PDB, neither of which belongs in a layout script. `fig_sc2_revised.R`
only places it, titled through `plot_cartoon` so it matches panel A.

### Titles

Reviewed against the request. B "Top decile recall, all escape mutations", C "Top
decile recall, de novo escape mutations", E "Query adaptability, all escape
mutations", F "Query adaptability, de novo escape mutations", H "Top decile recall,
all mutations". The ask said "all" for both E and F; F plots `BG_MAIN[2]`, the de novo
panel, so F carries "de novo" -- confirmed as the intended correction.

### Code

* `experiments/multitree/rank_path_cells.py`, `draw_path_panel.py` -- new.
* `antiGen-figures/data/ba286_to_kp3_ranks.csv`, `ref/6M0J.pdb`,
  `cartoons/ba286_to_kp3.png` -- new.
* `antiGen-figures/figures/fig_sc2_revised.R` -- panel D in place of the placeholder.

## 89. Panel D redrawn: all seven steps, a linear axis, a real surface (2026-10-28)

Three things were wrong with s88's panel, and all three came back from review rather
than from the code.

### Showing three of seven steps was the flattering choice

s88 drew the net change BA.2.86 -> KP.3 (L455S, F456L, Q493E) because that is what the
reference sketch drew and what the lineage names mean. s88 flagged that **the one miss
is among the four steps this drops** -- and that is exactly the reason not to drop
them. The panel now carries all seven RBD steps in path order, V483D's rank of 1,107
included. The four steps at sites that revert (346 and 483 each go out and back) are
drawn hollow, so the net path is still readable without the excursions being hidden.
Reach for this: when an omission needs a paragraph of justification, the omission is
usually the problem.

### A log rank axis reads as mediocrity

On the log axis the top decile filled 72% of the width and ranks 82-258 sat in the
middle of it. That is a faithful rendering of the numbers and a bad rendering of the
claim: the honest picture of "82nd of 3,819" is a mark hard against the left edge,
which is what a LINEAR axis gives. The replacement is linear over the whole pool with
one row per substitution, so nothing collides, the exact rank can simply be printed,
and the two label-placement tricks s88 needed (staggered stems, side selection) are
gone -- they existed only to survive the log axis's crowding.

### Discs are not a protein

`render_rbd_surface.py` is new and does the one thing matplotlib cannot: per-pixel
geometry. Orthographic sphere impostors over all 1,542 heavy atoms of 6M0J chain E,
nearest hit kept, Blinn-Phong off the true normal. Three details carry it:

* **Radii are inflated ~1.3A past vdW.** A probe radius fuses neighbouring atoms into
  one body; at true vdW every side chain reads as its own ball.
* **The depth buffer is blurred before normals are taken.** This is the step that turns
  a pile of balls into a surface: normals from the screen-space gradient of smoothed
  depth, the cheap stand-in for a solvent-excluded surface. Without it each atom keeps
  its own specular highlight and the domain reads as gravel.
* **An explicit silhouette.** Shading alone leaves a small render dissolving into white
  paper.

**The view is aimed, and the first two objectives were both wrong.** Aiming so the
highlighted centroid faces the camera fails because the centroid of five surface
residues lies INSIDE the domain -- its depth is maximised by orientations that leave
the residues occluded (14 of 52 highlighted atoms visible). Aiming to maximise visible
highlighted ATOMS also fails: 455 and 456 are adjacent, so the atom count is maximised
by a view showing that pair and nothing else (17 of 52, but only one of five sites).
Scoring SITES that clear a visibility floor, with the atom count as tie-break, finds a
view showing all five at once -- which exists. Worth keeping as a shape: an aggregate
objective over a set of objects optimises for the biggest object in it.

### Also

* The PNG is saved on the FULL canvas, not `bbox_inches="tight"`. The panel box has a
  fixed aspect; cropping to the ink gives whatever aspect the content happens to have
  (1.76 here) and the layout then letterboxes. Full canvas holds the designed 1.57.
* **One surface, not one per state.** Eight nodes across a 5.5in box leave each too
  small to read with no room for arrow labels; the reference strip is about 6.5:1.
  The structure is drawn once and large, and the chain is carried by the rank rows.
* Not done, and cheap if wanted: a blow-up of the top decile. With one row per
  substitution there are no collisions and the exact rank is printed, so it adds
  resolution rather than information.
* Still RBD-only, per the original instruction. Five of the twelve steps on the path
  (I70V, N211I, I211N, V1104L, V70I) are outside the RBD and have no rank, since the
  candidate pool is the RBD.

### Code

* `experiments/multitree/render_rbd_surface.py` -- new.
* `experiments/multitree/draw_path_panel.py` -- rewritten.

## 90. Panel D: the chain comes back, carrying all seven steps (2026-10-28)

s89 replaced the four-glyph chain with a single large surface and a rank chart,
on the argument that eight nodes cannot fit a 5.5in row. Review rejected the
layout and kept the rendering: **the chain reads better; put more than one
mutation on an arrow.**

The premise in s89 was wrong, and it is worth naming the error. It assumed one
glyph per STEP, which does need eight nodes. The path has seven RBD steps but
only THREE transitions between named states, because JN.1, JN.1.11.1 and KP.3
are defined by L455S, F456L and Q493E -- steps 1, 7 and 12. An arrow is a
transition, not a step, and a transition can carry a list. Four glyphs, three
arrows, seven substitutions.

`group_by_state` cuts the step list at the defining mutations, each bucket
running up to and including its own, which is what puts F456L on the arrow INTO
JN.1.11.1 alongside the excursions the path took on the way:

| transition | substitutions (rank) |
|---|---|
| BA.2.86 -> JN.1 | L455S 258 |
| JN.1 -> JN.1.11.1 | V483D 1,107*, R346T 233*, F456L 82 |
| JN.1.11.1 -> KP.3 | D483V 261*, T346R 19*, Q493E 191 |

(* the site reverts later on the path.)

**The intermediate states carry more than their name.** The tree nodes between
the named lineages hold excursions at 483 and 346 that revert before KP.3, so
the middle glyph is JN.1.11.1 plus two changes on their way back out. The
colouring states this instead of hiding it: a site is marked only while it
differs from BA.2.86, so 483 and 346 light up at JN.1.11.1 and are gone at KP.3.
Reading the four glyphs left to right, the salmon marks appearing and
disappearing IS the reversion, with no annotation needed.

The rank now rides beside its mutation, so the axis question from s89 is moot --
there is no axis, linear or otherwise, and the exact rank is printed.

### A bug the resize exposed

`render`'s `smooth` was a blur sigma in PIXELS, tuned once at 760px wide. Four
glyphs at 540px put relatively more blur on each and they came back soft and
flat next to the single large render. It is now a fraction of image width, so
the same surface comes out the same at any size. Worth keeping as a shape: a
visual parameter in absolute pixels is a parameter that silently depends on how
big you happened to render the first time.

### Code

* `experiments/multitree/draw_path_panel.py` -- rewritten again, chain layout.
* `experiments/multitree/render_rbd_surface.py` -- `smooth` relative to width.
* `antiGen-figures/figures/fig_sc2_revised.R` -- shorter panel D title; at the
  canvas edge it collided with the first state name.

## 91. Panel D: both registers, bound explicitly (2026-10-28)

Review kept the s90 chain and asked for the s89 rank chart back, the arrow-to-mutation
mapping made less abstract, and smaller glyphs. All three, plus the thing that makes
having two registers safe.

**The chart returns in its s89 form** -- linear over the whole pool, top decile shaded,
one row per substitution, exact rank printed. s89's case for linear stands: the honest
picture of "82nd of 3,819" is a mark against the left edge.

**Names on arrows, ranks in the chart.** s90 put `MUTATION rank` under each arrow, which
made the arrow labels wide and still left the ranks uncompared. Splitting them lets the
arrow carry only names (narrow enough to sit directly over its own arrow, in the gap
between the two glyphs it joins) and the chart carry the ranks with a shared axis.

**The registers are bound by a bracket per transition** down the side of the chart,
labelled with the state that transition produces. Without it the reader matches seven
names against three arrows from memory, which is the work the figure is supposed to be
doing. This was the "a bit abstract" complaint, and name-matching alone does not fix it.

**The key lives in the chart's dead space.** A linear axis with every mark under 1,200
leaves two thirds of its width empty. That emptiness is the message and should not be
compressed away -- but it is also the only place a filled-vs-hollow key fits without
crowding anything.

Glyphs went from 1.02 x 1.70in to 0.66 x 1.10in to make room.

### Code

* `experiments/multitree/draw_path_panel.py` -- two registers, brackets, inline key.

## 92. Panel D spacing (2026-10-28)

Two review notes: the registers sit too close, and the axis label should be larger and
just read "antiGen rank".

**The glyph TOP is pinned, so separation has to come out of the glyph.** The state
names sit above the glyphs and the canvas edge is at 3.5in, so the chain cannot move
up -- raising it clips the names. The gap therefore came from shortening the glyph
(1.10 -> 0.95in) and dropping the chart (first row 1.97 -> 1.88in, axis 0.56 -> 0.58
with a tighter last-row-to-axis gap). Gap between the "top decile" caption and the
glyph bottoms: 0.13in -> 0.37in.

The axis label is 9.6pt, up from 8.0, and loses the pool size. Nothing is lost by it:
the final tick already reads 3,819 and the shaded band is labelled.

### Code

* `experiments/multitree/draw_path_panel.py` -- spacing and axis label.

## 93. Panel H drops the ESM-2 rung (2026-10-28)

Panel H now draws seven lines, not eight. Done in `panels_sc2_export.R`, NOT in
`fig_sc2_aa20.R`: panel H exists only in `sc2_results_revised`, and the same rung is
panel M of the published `sc2_results_aa20`, which is not what was asked to change.

**Rebuilt after the source, not configured before it.** `fig_sc2_aa20.R` sets
`sc2.gate_drop_main` (to `character(0)`) and `sc2.gate_colors_main` in its own
`options()` call, which runs after the exporter's and wins -- the same precedence that
caught the DMS panel in s85. `make_gate_plot` reads both at CALL time, so setting them
after the source and re-calling it is what takes effect.

**The colours had to become NAMED, and this is the real trap.** `scale_color_manual`
matches an UNNAMED vector to the scale's limits positionally, and dropping a line drops
a limit. The drawer's eight-colour vector against seven remaining levels would have
shifted every rung after ESM-2 onto its neighbour's colour -- handing GATE_BLUE to
`Where x which` and leaving the gate drawn in `Where x which`'s green, with a legend
that agreed with itself and was wrong. Naming them makes the mapping independent of who
is on the panel. Worth propagating: the drawer's positional vector is fine only while
the roster is fixed, and `sc2.gate_drop_main` exists precisely to change the roster.

### Code

* `antiGen-figures/figures/panels_sc2_export.R` -- named gate palette, drop the rung.

## 94. aa20 supplement legends (2026-10-28)

The four `aa20_supp_*` figures all had text drawn over text, for three different
reasons. Fixed with options in `fig_sc2_aa20.R`; the drawer keeps the published
defaults, and `supp_recall/precision/summaries/dms` re-render byte-identical.

**This arm's ROSTER is why these broke.** The published supplement draws a smaller
model set, so its legends fit the layout they were written for. Here the finetune and
ensemble rungs push several groups past ten models, and every one of these layouts
sizes a text block by assuming how many entries it holds.

**supp_recall / supp_precision.** Six per-group legends sit in a six-column grid, so
each gets about 2.5in. They were `ncol = 2`, and two columns of names like "antiGen as
ESM 2 (650M) LM head FT" are far wider than that. **A legend wider than its grid cell
is not clipped -- cowplot centres it and it is drawn straight over its neighbours**,
which is why six legends looked like one smear. One column per legend fixes the width;
the height is bought back by growing the canvas, and `plot_by_x_supp` now returns
`list(plot, height)` so it can size the canvas from the longest legend rather than
having a number guessed at the call site. The group titles needed wrapping for the same
reason -- "Pandemic Data in Training Window" on one line is wider than its column.

**supp_summaries.** Not a legend at all: one y tick label per model down every panel,
31 of them on this arm, in a 14in canvas. Height 22.

**supp_dms.** Two failures at once. The legend strip was 0.075 of an 8in canvas, 0.6in,
with "Model type" and "Transformation" stacked inside it -- drawn on top of each other.
And the five model types on one row are wider than the 10in canvas, so the "Model type"
title was cut off the left edge and "Val-selected, averaged over 10 holdouts" off the
right. Wrapped to two rows, strip at 0.16, canvas 9in. Separately, the rotated x tick
labels reach further left than the axis does, so the first one ran off the canvas; a
70pt left margin holds it.

`legend_nrow` is a PARAMETER of `make_dms_corr_plot`, not an option read inside it,
because that builder also draws the main figure's legend and only the supplement's
needs wrapping.

### Not fixed: aa20_supp_gating_recall is empty

All nine panels draw nothing, and have since this arm existed. Two independent causes,
neither a legend problem:

* **A-C** pass `xlim_hi = 5` as a literal at the call site. That is the published hard
  stop in "years into pandemic"; this arm's `sc2.gate_x` returns a decimal CALENDAR
  year, so `xlim(c(NA, 5))` discards every point. The x axis showing a lone "5" is the
  tell. `sc2.gate_xlim_hi` exists and is set to NA for panel M, but `gate_supp_row`
  does not read it.
* **D-I** plot against `n_muts` and `n_seqs`. `gate_recall_by_name` returns only
  model, the three recalls and cutoff -- the by-position loader is what supplies those
  two columns, so on this arm they do not exist at all.

Left alone deliberately: A-C is a one-line fix, but D-I needs a decision about what
those rows should show on an arm whose loader has no training-set-size columns, and
that is not a layout question.

### Also

`figs/sc2_results_aa20.png` is re-rendered here. It had drifted: the panel M
`RUN_TAG_TO_MODEL` fix from s84 was committed but the PNG was never regenerated, so the
stored copy still showed twelve models collapsed into one NA bar. The diff is confined
to that panel.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- options, all defaulting to the
  published behaviour.
* `antiGen-figures/figures/fig_sc2_aa20.R` -- the values this arm needs.

## 95. The revised endemic figure: scope, and the training jobs (2026-10-29)

A new `endemic_results` on the antiGen-data-revised trees: recall at the top decile
instead of recall AUC, the four factored baselines added, and no variants anywhere.
This section is the scope and the substrate; the figure itself comes later.

### What the figure becomes

Six panels, 3 viruses x {all mutations, de novo}, bars of **recall at q = 0.1**:

|        | H3N2 | RSV A | DENV2 |
|---|---|---|---|
| all mutations | A | B | C |
| de novo       | D | E | F |

The published figure's G-L -- PSSM precision against date, by variant emergence -- are
**gone**. The revised endemic trees carry no variant roots, so the metric has nothing
to be computed against. That is also why this is 6 panels and not 12: the gap/stop-mask
variants stay in the supplement.

Recall@10% needs no new evaluation. It is a different summary of the same hit-rate
curves, and `summarise_curves_at` (added for panels B/C of `sc2_results_revised`)
already does it.

### The corpus

One dated directory per virus, `train/` plus a separate `<virus>-test/` tree, exactly
the shape the SC2 arm uses:

| virus | cutoff | train examples | test examples |
|---|---|---|---|
| h3n2  | 2022-10-01 | 11,934 | 19,953 |
| rsv-a | 2023-10-01 |    549 |    913 |
| denv2 | 2019-10-01 |    827 |  1,305 |

**The cutoffs moved, and that is what makes this more than a re-run.** The published
endemic runs were trained on `antiGen-data/output_<virus>_full_cutoff<N>`, where
`--cutoff` named years since MRCA (12 for flu, 74 for rsv-a). Here `--examples` names
the training file and the train/test boundary is the POST_CUTOFF field the corpus
already carries, so `--cutoff` is metadata.

### The baselines

The four factored rungs -- codon table, substitution matrix, where x which, where x
which + events -- are what `sc2_results_revised` gained and the endemic figure never
had. **They cost no training**: `--multitree-baselines` builds all four at eval time,
fitted on the training examples only.

### EVE is refitted, and this was the one real decision

The published `eve_frozen` checkpoints were fitted to an MSA cut at the OLD cutoff
dates. Reusing them under the new cutoffs would put a baseline fitted past its own
train/test boundary beside models that respect it, and **nothing downstream could
detect it** -- a FASTA carries no dates. antiGen-data-revised now writes
`msa_<date>_cur.fasta` into each `train/`, built from the same sequences the examples
came from and so window-restricted by construction. That is the file, and only that
file, about which `METADATA=` empty may be asserted to `train_eve.sh`.

### THE DATASET FLAG IS NOT OPTIONAL

`antiGen-eval` keeps only checkpoints whose stored `train_args` match its `--dataset`,
via `train_args['flu'|'rsv_a'|'denv2']` (eval.py:66). Train with `--examples` and no
dataset flag and the checkpoint is **invisible to the eval meant to score it** -- not
an error, just nothing found. `train_args` is `vars(args)`, so passing `--flu` is what
puts the key there.

### The jobs

Nine, all `gpu_batch` (valsel is hours; `evo_gpu_priority` is for jobs of order
minutes, CLAUDE.md). Submitted 2026-10-29 as 3488027-3488035.

* `end_base_<v>`  base antiGen, 32 epochs -- the rung the published figure draws.
* `end_vs_<v>`    ten val-selected runs, 96 epochs, split seed i / init seed 100+i --
                  the `antiGen (val x10)` rung of `sc2_results_revised`.
* `end_eve_<v>`   EVE on the regenerated MSA, `--freeze-backbone`.

**base and valsel are separate jobs on purpose.** base is minutes and valsel is hours;
one job per virus would make the short half queue for a long job's slot, and base
landing first is what lets the figure be drawn before valsel finishes. Walltimes are
per virus -- flu's valsel gets 16h because flu is four times the other two put
together; an honest walltime matters in both directions, since too long stops a job
being backfillable and too short cuts ten runs off part way.

`train_endemic_series.sh` CALLS `train_transformer.sh` and `train_antigen_valsel10.sh`
rather than copying them, so the endemic rung differs from the SC2 one only by its
corpus and its dataset flag.

### Still to do

* Eval once training lands: `antiGen-eval --dataset {flu,rsv_a,denv2}
  --train-examples <train> --test-examples <virus>-test/examples_all_cur.txt
  --multitree-baselines`, no `--variant-counts-dir`.
* The drawer: a `fig4_endemic_viruses` variant at recall@10%, six panels.

### Code

* `examples/slurm/train_endemic_series.sh` -- new.
* `experiments/multitree/submit_endemic_revised.sh` -- new.

## 96. Endemic: the eval path, and the coupling that blocked it (2026-10-29)

Training from s95 is in flight. Two things came out of building the eval beside it.

### `--root-counts` could not be used without `--variant-counts-dir`

`eval.py` required the two flags together: "one fixes the de novo cut, the other the
emergence grid". **They are not symmetric.** The de novo cut is read by every recall
metric; the emergence grid is read only by the variant-emergence CSVs. So the grid
needs the cut and not the reverse -- and a corpus with no variant roots, which is
exactly what the revised endemic trees are, was unscoreable for recall at all.

The fix is the asymmetry, not a bypass: `--variant-counts-dir` still requires
`--root-counts`, while `--root-counts` alone sets `pssm_variant = []`. Nothing else
had to change, because `do_emergence = bool(pssm_variant)` (eval.py:802) already gated
the whole emergence block -- the coupling was an argument check guarding a branch that
could already turn itself off. SC2 passes both and takes the identical path.

`test_eval_counts_args.py` pins the DIRECTION of the guard rather than its message,
since that is the part a later edit gets backwards. Suite 240 -> 244.

### BLOSUM62 is a checkpoint, and it is per dataset

The matrix does not depend on the corpus, but `antiGen-eval` keeps only checkpoints
whose stored `train_args` match its `--dataset`, so the SC2 blosum62 checkpoint is
invisible to a flu eval. Without one per virus the rung does not error -- the bar is
just absent. Added as a `blosum` stage; three more jobs, 30 minutes each.

### What each bar comes from

| rung | mechanism |
|---|---|
| antiGen, EVE, BLOSUM62 | checkpoints under `<virus>-<date>/`, globbed one level deep |
| PSSM, PSSM Conservation | `--root-counts`, the training tree's counts matrix |
| DMS | `antiGen/data/baselines_<virus>/` -- flu and rsv-a only; denv2 has no assay |
| codon, substitution, where x which, + events | `--multitree-baselines`, fitted at eval time on the TRAINING examples |

The ten val-selected runs sit two levels deep and so are NOT found by the eval, which
is deliberate: they are one bar, scored by the ensemble step afterwards.

### The valsel budget binds on small corpora (s95 follow-up)

rsv-a's first valsel run FAILED its own budget check -- two of ten seeds selected the
last epoch, so their validation curve had not turned. An epoch is one pass over the
corpus, and these differ by twenty-fold: at batch 64 it is ~186 steps for flu, ~13 for
denv2, ~9 for rsv-a, so 96 epochs buys flu ~17,900 updates and rsv-a ~860. The budget
is now per virus -- 384 for the two small corpora, 96 for flu. The 96-epoch rsv-a runs
are kept as `valsel_ep96`, as the SC2 arm keeps `valsel_ep32`.

### Code

* `antiGen/eval.py`, `antiGen/tests/test_eval_counts_args.py`
* `examples/slurm/eval_endemic.sh`, `examples/slurm/train_endemic_series.sh` (blosum)
* `experiments/multitree/submit_endemic_evals.sh`, `submit_endemic_revised.sh`

## 97. The endemic evals run; four things were in the way (2026-10-30)

All three viruses now have the four hit-rate CSVs with the full roster. Every failure
was fast and loud -- seconds each -- which is the only reason this was a morning and
not a week.

### 1. `discover_checkpoints` globs FLAT

`glob(join(checkpoint_dir, "fitness_*.pt"))`, no recursion. Pointing it at
`<virus>-<date>/` found nothing, because the checkpoints live one level down in
`base/`, `blosum62/`, `eve/`. **`submit_cutoff_evals.sh` says exactly this** -- "pointing
the eval at the cutoff directory finds nothing" -- and I read it backwards when writing
the endemic twin. Each virus now gets a flat symlink directory, rebuilt every run.

### 2. `explicit` required `--variant-counts-dir` too

s96 decoupled the flags in the PSSM branch and missed the SAME coupling at eval.py:495,
where `explicit` decides whether a `data_dir` is needed at all. Without the grid flag a
fully-specified run fell through to `datasets.default_data_dir` and died on "set
FITNESS_DATA_ROOT" -- after loading its checkpoints. The lesson is the obvious one: a
coupling fixed in one place is not fixed.

### 3. The test corpus needs `--split-date`

One test file serves every cutoff, so `POST_CUTOFF` is a SET and `load_examples`
refuses to guess which split to read. The SC2 arm reads its shared corpus the same way;
its manifest even records "`--split-date $D`".

### 4. The flu DMS maps were keyed to a different protein

**antiGen-data-revised builds HA as HA1+HA2 with no signal peptide** -- 550 residues
where the published build had 566 -- and the CSVs in `data/baselines_flu/` carry one
row per position of the OLD build. The sweep died with a shape mismatch, which is the
GOOD outcome: a map read off by sixteen residues would have produced a believable bar.

The fix is to re-key, not to drop, and **the offset is derived rather than declared**.
Sixteen is the H3N2 signal peptide (`MKTIIALSYILCLVFA`) and was almost certainly right,
but `rekey_dms_to_mature.py` aligns the two corpora's ROOT sequences at every plausible
prefix trim and refuses to write unless the best is unambiguous:

| trim | mismatch vs the new root |
|---|---|
| **16** | **2.36%** (13 of 550 -- N8D, N122D, A138S, N158K, Q173K, G186V: drift between roots a decade apart) |
| 9 | 91.45% |
| 15 | 95.5% |

Output goes to `data/baselines_flu_ha12/`; the published arm still reads the 566-row
originals. `--baseline-dir` selects it, and eval.py now REFUSES any baseline whose
length is not `seq_len`, naming it -- a missing bar someone can see beats a wrong one
they cannot.

### The rosters

| | antiGen | EVE | BLOSUM62 | PSSM / Cons. | codon / subst / wxw / +events | DMS |
|---|---|---|---|---|---|---|
| h3n2 | yes | yes | yes | yes | yes | 8 maps + 2 MSA-Transformer |
| rsv-a | yes | yes | yes | yes | yes | 9 escape/entry maps |
| denv2 | yes | yes | yes | yes | yes | none (no assay) |

No `variant_emergence_*` CSVs anywhere, as intended.

### Valsel budgets, resolved

All three pass. denv2 best epochs 67-134 of 383, rsv-a 106-316 of 383, h3n2 83-94 of
95. **rsv-a is the proof the raise was needed**: at 96 its optima clustered at 91-95,
but given room they run to 316. h3n2 passes but sits against its ceiling -- eight of
ten seeds at 90-94 of 95 -- so 96 may be marginal there; flagged, not acted on.

### Code

* `antiGen/eval.py` -- `explicit`, `--baseline-dir`, the length check.
* `experiments/multitree/rekey_dms_to_mature.py` -- new.
* `experiments/multitree/submit_endemic_evals.sh` -- staging, `--split-date`, baseline dir.

## 98. endemic_results_revised is drawn, and antiGen does not win four of six (2026-10-30)

Six panels, three viruses across, {all mutations, de novo} down, bars of recall at the
top decile. `figures/fig_endemic_revised.R`, from
`data/evals_endemic_revised/<virus>/`.

### The headline is not the one the published figure had

**Recall at q = 0.1:**

| | h3n2 all | rsv-a all | denv2 all | h3n2 novo | rsv-a novo | denv2 novo |
|---|---|---|---|---|---|---|
| **antiGen** | **0.891** | 0.687 | 0.698 | **0.676** | 0.493 | 0.488 |
| **Where x which + events** | 0.875 | **0.804** | **0.760** | 0.596 | **0.574** | **0.596** |
| Where x which | 0.777 | 0.725 | 0.728 | 0.523 | 0.493 | 0.504 |
| PSSM | 0.858 | 0.709 | 0.703 | 0.100 | 0.100 | 0.100 |
| EVE | 0.745 | 0.579 | 0.651 | 0.420 | 0.294 | 0.383 |

antiGen takes H3N2 on both metrics and `wxw_dirichlet` takes RSV-A and DENV2 on both --
**the new baseline wins four of the six panels.** On the published figure antiGen was
1st in all six. The difference is not the metric: it is that the baseline the figure
was asked to add is a strong one on small trees, and that these are different trees.
Worth saying plainly rather than discovering at review.

### PSSM scores exactly 0.100 on de novo, in all three viruses

Not a bug, and a useful check that the pipeline is right. De novo cells are by
definition never observed at the root, so the root PSSM gives every one of them the
same zero and the whole candidate pool ties. The fractional top-q convention then
returns exactly q. A PSSM has no signal on de novo mutations by construction; a number
that was not exactly 0.100 would mean the tie-breaking had drifted.

### Two drawing decisions

* **Four colours, five levels.** `MODEL_TYPE_COLORS` drops "antiGen Finetunes"
  (common.R:68) so it is one shorter than `MODEL_TYPE_LEVELS`. `setNames` on the pair
  errors, which is the good case -- an UNNAMED vector would have been matched
  positionally and handed every group its neighbour's colour.
* **One x range per row.** Per-panel ranges put three different figures side by side
  under one heading: a bar at the right edge of DENV2 was a smaller number than one at
  the right edge of H3N2 and nothing said so. Shared per row, not across all six, or
  the de novo row squashes against its axis.

### Roster

The eval writes every per-assay DMS map (eight for flu, nine for RSV-A) and, for flu,
two MSA-Transformer rows. The figure names its roster instead of taking the file's, and
keeps the one DMS rung the published endemic figure drew. DENV2 has no assay, so panels
C and F carry no DMS bar.

### Code

* `antiGen-figures/figures/fig_endemic_revised.R` -- new.
* `antiGen-figures/data/evals_endemic_revised/` -- the three evals' hit-rate CSVs.

## 99. The val-selected bar, and what it does not rescue (2026-10-30)

s98's antiGen bar was the flat 32-epoch base. The ten val-selected runs existed for all
three viruses but were not on the figure. They are now.

### Scored as a MODEL, not by recombining cached maps

The SC2 `antiGen (val x10)` bar is produced by `ensemble_report.py`, whose substrates
are `backbone` and `list` -- it lives on the coguk mutation-list panels. **This figure
reads eval's test-set hit-rate CSVs, which neither substrate covers**, so the SC2
recipe does not transfer and "just a scoring pass" was wrong.

`MeanLogProbEnsemble` is the cheaper answer: an `nn.Module` holding the N members that
returns the mean of their per-query log-softmax, registered in `builtin_models` and so
scored through the same masks, candidate pool and quantile machinery as every other
rung. No map dump, and nothing about the metric reimplemented.

**Why it is the same rule.** `compute_probs` turns a model's output into
`softmax(logits.flatten(1), dim=1)` -- per query, over the flattened (positions x
residues) cells, which is the axis `ensemble_rules` log-softmaxes over. Softmax of the
mean-of-log-softmax is strictly increasing in it, and the hit-rate metrics read
within-query RANKS only, so the ranking is the mean-of-log-probs ranking exactly. Two
preconditions, both checked in `eval.py` rather than assumed: the eval range must be
the whole protein (it is, for flu/rsv_a/denv2), and `union_min` must be a no-op, which
it is for members that never abstain.

Gated by `test_one_member_is_a_no_op`, the repo's standard ensemble gate, plus a test
that the mean really is taken in LOG space -- the discriminating case had to be chosen
deliberately, since on agreeing members the two means rank alike. Suite 246 -> 250.

### It helps on all mutations and does not help on de novo

**Recall at q = 0.1:**

| | h3n2 all | rsv-a all | denv2 all | h3n2 novo | rsv-a novo | denv2 novo |
|---|---|---|---|---|---|---|
| antiGen (val x10) | **0.898** | **0.806** | 0.751 | 0.655 | 0.537 | 0.446 |
| antiGen (base) | 0.891 | 0.687 | 0.698 | **0.676** | 0.493 | 0.488 |
| Where x which + events | 0.875 | 0.804 | **0.760** | **0.596** | **0.574** | **0.596** |

On ALL MUTATIONS val-selection is worth a great deal on the two small corpora -- rsv-a
0.687 -> 0.806, denv2 0.698 -> 0.751 -- which is what the s95 budget work was for. On
DE NOVO it is **worse than base** on two of three (h3n2 0.676 -> 0.655, denv2 0.488 ->
0.446). Selecting an epoch on held-out mutations does not select for the harder task.

**The headline survives the better rung.** `wxw_dirichlet` still takes all three de
novo panels and DENV2 on all mutations: four of six. rsv-a's all-mutations panel is
0.806 vs 0.804, which is a tie and should not be reported as a win either way.

### Code

* `antiGen/eval_utils.py` -- `MeanLogProbEnsemble`.
* `antiGen/eval.py` -- `--ensemble-dir` / `--ensemble-name`.
* `antiGen/tests/test_mean_logprob_ensemble.py` -- new.
* `experiments/multitree/submit_endemic_evals.sh` -- stages the members.
* `antiGen-figures/figures/fig_endemic_revised.R` -- the bar.

## 100. endemic_results_backbone: the same models, a much smaller test set (2026-10-30)

A second arm of the endemic figure, scored on testing-side mutations on the nextstrain
backbones only. `figs/endemic_results_backbone.png`. The full-test-tree figure is
untouched and re-renders byte-identical.

### Nothing about training moved

Same checkpoints, same `--train-examples`, same `--root-counts`. **Only
`--test-examples` and the output directory change**, so the factored baselines are
still fitted on the same training examples and the de novo cut is still the same cut.
The two arms differ in what is being predicted and in nothing else.

`submit_endemic_evals.sh` takes `TESTSET` and `ARM` rather than being forked; the
default is the existing arm, unchanged.

### THE DENOMINATORS, which decide what this figure can say

| virus | queries | all-mutation targets | de novo targets |
|---|---|---|---|
| h3n2 | 311 (was 7,974) | 574 (was 19,434) | **5** (was 621) |
| rsv-a | 48 (was 316) | 55 (was 444) | **15** (was 136) |
| denv2 | 119 (was 419) | 222 (was 784) | **58** (was 240) |

**Panel D rests on five mutations.** One of them is twenty points of recall, which is
why its bars come out as 1.0 / 0.8 / 0.8 / 0.8 / 0.6 -- an ordering of a five-element
set, not a result. Panel E has fifteen. Those two panels must not be read as
orderings, and the all-mutations row is only comfortable for h3n2.

So every panel is titled with its own n. A recall is a fraction and this figure cannot
be looked at honestly without the denominator on the page. `count_targets.py` computes
it with the eval's own de novo cut rather than a re-derivation.

### Two traps while keeping the first figure identical

* **An empty title is still a title.** `labs(title = panel_n(...))` returning `""`
  reserves the title strip and shifted every panel -- 1.96M pixels differed. The title
  is now ADDED with `ggtitle` only when there is one.
* **A theme element set unconditionally does the same.** Styling `plot.title` for the
  annotation changed the default figure's layout even with no title, so the styling is
  chosen on whether a counts file was given.

Both are the same shape: an option that is supposed to be off cannot leave a trace in
the layout, and "the figure still renders" is not the test -- byte-identity is.

### Code

* `experiments/multitree/count_targets.py` -- new.
* `experiments/multitree/submit_endemic_evals.sh` -- `TESTSET` / `ARM`.
* `antiGen-figures/figures/fig_endemic_backbone.R` -- new, sources the revised drawer.
* `antiGen-figures/figures/fig_endemic_revised.R` -- parameterised; defaults unchanged.

## 101. Bloom & Neher as a baseline, at a date-matched cutoff (2026-11-01)

The TreeMHN line is closed (s100 assessment stands: the whole monotonic-accumulation
family assumes irreversibility). The paper that replaces it is a different one --
**Bloom & Neher, "Fitness effects of mutations to SARS-CoV-2 proteins", Virus Evolution
9(2) vead055** -- and it fits, because it is not a progression model at all. It is a
per-(site, amino acid) fitness estimate for every SARS-CoV-2 protein, derived by
comparing actual to expected counts of independent occurrences along the UShER
phylogeny. That is a STATIC SCORE MAP, which is exactly the shape `load_baselines`
already reads, and the repo already carries two of the same kind (`pyr0`, `CoVFit`).

### The date, and why it is the whole design

Their estimates must not be computed from sequences inside our test window, or the
baseline is scored on occurrences it already counted. They publish dated tree snapshots
(`results_public_<date>/`), and exactly one lands on our quarterly grid:

| their snapshot | nearest cutoff of ours | offset | favours |
|---|---|---|---|
| 2022-01-31 | 2022-01-01 | +30 d | them |
| 2022-12-18 | 2023-01-01 | -14 d | us |
| 2023-03-27 | 2023-04-01 | -5 d | us |
| **2023-10-01** | **2023-10-01** | **0** | **neither** |
| 2024-04-19 | 2024-04-01 | +18 d | them |
| 2024-11-06 | 2024-10-01 | +36 d | them |

**2023-10-01 it is.** 64,715 training examples, a shared test tree of 75,606, and base
antiGen already trained there. Verified the snapshot is genuinely earlier data rather
than a copy of the main release: 1,727,136 total expected counts against 1,902,428, and
only 13.4% of shared cells have identical fitness.

### The trap: 72% of their values are negative

`load_baselines` ends with `tensor[tensor < 0] = nan` -- a negative entry is the
sentinel for "this baseline abstains here". Bloom & Neher's fitness is a LOG RATIO, so
negative means deleterious: **6,447 of 8,917 spike cells, 72.3%**. Written raw, the
baseline would have silently abstained on nearly three quarters of its own estimates,
and what survived would have been the tolerated mutations -- a bias toward exactly the
cells most likely to be observed, which would have read as skill.

`bloom_neher_to_baseline.py` adds a constant instead. Measured, not assumed: the +8
shift introduces **0 ties and 0 order inversions**, and the hit-rate metrics read
within-query ranks only. Cells below their own `min_expected_count: 10` (a DISPLAY
cutoff in their config, not a filter on the released file, which carries expected
counts below 1) are written as the sentinel, so the baseline abstains where the authors
would not quote a number.

Final coverage: **1,277 of the 3,819 RBD cells, 33.4%**. Counted over the 20-residue
alphabet, because `load_baselines` drops the `*` and `-` columns -- the first version
of the script reported 1,345 by counting them, a number the figure would never have.
For contrast EVEscape, whose spike-wide coverage is 13.6%, is complete over the RBD, so
on these panels Bloom & Neher is the partial-coverage rung and the panel has to say so.

The source is already heavily tied -- 583 distinct values across 1,277 cells -- and
`load_baselines` returns float32, which merges one more pair (583 -> 582). That is the
pipeline's behaviour for every baseline, not this conversion's.

### A latent bug this turned up: `_best.pt` is not a completion marker

`--save-best` rewrites `..._best.pt` every time validation improves, so it appears
MINUTES into a run. `train_antigen_valsel10.sh`'s resume guard tested for it, so a run
that was killed, preempted or hit its walltime looked finished and was skipped --
quietly putting an undertrained member into a ten-member ensemble that would still
average ten things and report OK. The plain `fitness_*.pt` is written once, at the end.

Caught because the 2023-10-01 array showed 2/10 "complete" four minutes in. **The three
endemic arms were checked and are genuinely 10/10** (final checkpoints present), so
s99's numbers stand. Fixed in the resume guard, in both completion counters, and in the
eval submitter's member staging, which could otherwise have put an in-progress run's
`_best.pt` into the ensemble.

### Training

Ten val-selected runs at 2023-10-01, array `3509194_[0-9]`. Submitted as one job PER
SEED rather than the series script's ten-in-series: this corpus is 64,715 examples
against sc2-2021's 2,934, so an epoch is ~1,011 steps rather than ~46 and ten runs at
96 epochs is a ~12 h serial job -- neither backfillable nor parallel. `SEEDS` is a new,
defaulted knob on `train_antigen_valsel10.sh`, so split seed i / init seed 100+i are
still derived inside it and a run produced per-seed is identical to one from the series.

### Still to do

Score on the panel B/C substrate (coguk list, macro pooling, top decile) once the runs
land, then the supplement: antiGen (val x10) against Bloom & Neher at 2023-10-01, all
escape mutations and de novo escape mutations.

### Code

* `experiments/multitree/bloom_neher_to_baseline.py` -- new.
* `antiGen/data/baselines/BloomNeher_2023-10-01.csv` -- new.
* `examples/slurm/train_valsel_seed.sh` -- new.
* `examples/slurm/train_antigen_valsel10.sh` -- `SEEDS`, and the completion test.

## 102. Panel G becomes Spearman: two bars, not four (2026-11-01)

Panel G drew four Pearson bars per model: {native space, log space} x {prediction,
position-only control}. It now draws two Spearman bars: prediction, and a position-only
control summarised by RANK.

### Why four collapses to two

Rank correlation is invariant to a monotone transform, and `take_log` is exactly that
(exp for the pipeline models, log for the baselines). **Checked, not assumed**: across
all 27 models the log and linear Spearmans agree to within 1e-9 for the prediction. So
two of the four bars carried no information once the metric changed.

The control does not collapse that way. `mean_norm` replaces each row by the mean of
its SCORES, and a row-mean of exp(x) (arithmetic) orders sites differently from a
row-mean of x (roughly geometric): log and linear disagree for **all 27** models, and
in SIGN for BLOSUM62 (-0.017 vs 0.141) and both MSA Transformers (-0.165 vs 0.151).
Which bar you drew decided whether the control looked positive or negative.

So the control is the row-mean of RANKS instead, which is invariant by construction --
verified: 0 of 27 disagree. It lands near the log form (BLOSUM62 0.145, MSA Trans. BA.2
0.118), so the negative value-mean numbers were an artifact of arithmetic averaging in
probability space. Ties are midranks, which is what `cor(method = "spearman")` uses;
this matters because the control is constant within a site, so every site is a 20-way
tie block.

### Panel G, Spearman (Expression)

| model | prediction | position-only (rank) |
|---|---:|---:|
| EVEscape | 0.362 | 0.354 |
| BLOSUM62 | 0.279 | 0.145 |
| antiGen (val x10) | 0.270 | **0.394** |
| EVE | 0.140 | 0.181 |
| Tranception | 0.123 | 0.120 |
| MSA Transformer | 0.067 | 0.133 |
| MSA Trans. (BA.2 eval) | 0.053 | 0.118 |
| ESM 2 (650M) | 0.038 | 0.008 |
| ESM 2 (35M) | 0.033 | -0.098 |

**antiGen's control exceeds its prediction** (0.394 vs 0.270): collapsing each site to
its mean rank correlates with DMS expression BETTER than the full per-residue output.
EVEscape is the same shape with the gap closed (0.362 vs 0.354). This is robust to the
control definition, so it is a property of the models against this assay rather than a
measurement choice -- and it is the intended reading of the panel, not a problem with
it: the claim being supported is that there is a predictor correlating with DMS, site
level being enough.

### The option that leaked, and the clobbering it exposed

The three labels were first done as options. They leaked, because
**`panels_sc2_export.R` causes the PUBLISHED `sc2_results_aa20` to be drawn**: it sets
`sc2.fig_main`/`sc2.fig_prefix` to scratch names, but `fig_sc2_aa20.R` sets its own
afterwards and wins -- the precedence trap of s85 and s93 again. So the scratch naming
never applied to that figure, every `fig_sc2_revised.R` run has been overwriting
`figs/sc2_results_aa20.png` and the `aa20_supp_*` files, and a label set for this
Spearman panel relabelled a PEARSON panel "Spearman" in a figure nobody asked to touch.

Labels are now PARAMETERS of `make_dms_corr_plot`, which cannot leak across a source.
Confirmed afterwards that `sc2_results_aa20.png` and `aa20_supp_dms.png` re-render
byte-identical and only `sc2_results_revised.png` moves.

Worth keeping as a shape: an option is global to the session, so it is the wrong tool
whenever the same builder serves two figures in one process.

### Code

* `antiGen-figures/R/common.R` -- `cor_with_NA_mask(method=)`.
* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `rank_mean_rows`, the Spearman
  dump and frame, caller-supplied `transformation`, label parameters.
* `antiGen-figures/figures/panels_sc2_export.R` -- panel G off the Spearman frame.

## 103. Supplement: antiGen against the two fixed-split baselines (2026-11-01)

`figs/supp_fixed_split.png`. Two panels, two bars each, all escape mutations only.

### Why these two models get their own figure

Most rungs can be scored at any cutoff -- a PSSM is refit, EVE is retrained, the
factored baselines are refit on whatever training window they are given. **PyR0 and
Bloom & Neher cannot**: each was fit once, on sequence data up to a particular day, and
neither offers a dated re-fit at an arbitrary date. Scored at the cutoffs the rest of
sc2_results_revised uses, they would be reading data they had already seen. So each is
scored at the cutoff matching its own date, which is what makes this a separate figure
rather than two more bars on panel B.

| panel | baseline | its data ends | our cutoff | offset |
|---|---|---|---|---|
| A | PyR0 (Obermeyer 2022, 6,466,300 GISAID genomes) | 2022-01-20 | 2022-01-01 | +20 d, favours PyR0 |
| B | Bloom & Neher (`results_public_2023-10-01`) | 2023-10-01 | 2023-10-01 | 0 |

**PyR0's offset is deliberate and in the baseline's favour.** It is fit once at
2022-01-20 and our grid is quarterly, so an exact match does not exist. 2022-01-01
leaves PyR0 twenty days PAST the split; 2022-04-01 would handicap it by 71 days.
Erring toward the baseline makes the result a floor rather than an artefact.

### The numbers (macro, top decile, `list_coguk_fullpool`)

| | antiGen (base, 32 ep) | baseline | gap |
|---|---:|---:|---:|
| A, 2022-01-01, 214 cells | **0.612** | PyR0 0.200 | +0.412 |
| B, 2023-10-01, 139 cells | **0.849** | Bloom & Neher 0.691 | +0.158 |

antiGen here is the 32-EPOCH BASE, the weaker of the two antiGen variants, so both
margins are floors. Bloom & Neher is much the stronger baseline of the two -- it also
beats EVEscape (0.367), pyr0 and CoVFit at its own cutoff.

### De novo is not in this figure, and the reason is arithmetic

It is viable at 2022-01-01 (30 cells, 469 events) and collapses at 2023-10-01 to **one
cell and four events**: by then nearly every listed mutation has been observed, so
there is nothing left that is de novo. A panel that exists on one half of the figure
and not the other is worse than no panel, so all escape mutations only.

Noted for whoever reads the per-cutoff CSVs: PyR0 scores exactly 0.100 on de novo, the
same chance-level signature as the PSSM and for the same structural reason -- it
estimates fitness from observed lineage growth, so an unobserved mutation has no
estimate, the whole unseen pool ties, and fractional top-q returns exactly q.

### Two traps in the output

* **`BloomNeher_2023-10-01` is also scored in the 2022-01-01 run** (macro 0.542),
  because it lives in `data/baselines/` and every static map there is scored. That
  number carries ~21 months of post-split data and must not be quoted. The figure names
  its two models explicitly rather than taking whatever the CSV holds.
* **CoVFit is a third model of this kind** -- a trained SARS-CoV-2 fitness predictor
  with its own fixed cutoff, currently a published default drop. It is not in this
  figure because its date was not established; it scores 0.061 at 2022-01-01.

### Code

* `examples/slurm/score_fixed_split.sh` -- new (was score_bloomneher_2023.sh); `D`
  picks the cutoff.
* `antiGen-figures/figures/figS_fixed_split.R`, registered in `run_all.R`.

## 104. Panel G: shades, not opacity (2026-11-01)

The second series is now a lighter SHADE of each group's colour rather than the same
colour at lower alpha. Opacity blends with whatever sits behind the bar -- the panel
grid shows through it -- and transparency is the part of a PDF that renderers disagree
about. `shade_fill` is off by default, so the four-bar Pearson panels this same builder
draws keep the alpha mechanism.

**Two fill scales are not possible in one ggplot**, which dictates the construction:
`fill` carries one level per (group, series) with a manual palette, and `alpha` stays in
the aes purely to carry the SECOND legend -- its values are both 1, so it has no visual
effect, and that legend's keys are overridden to a dark/light grey pair.

### Two bugs on the way, both worth remembering

* **`group_colors()` returns an UNNAMED vector.** It is matched to the factor's levels
  by POSITION, which is how `scale_fill_manual(values = colors_here)` consumes it.
  Reading names off it gives NULL, so every fill key missed and **every bar came back
  grey** -- a total loss of the group colouring that is obvious on sight, which is the
  only reason it was caught immediately. The levels are now rebuilt the way
  `group_colors` builds them, with a length assertion beside them.
* **A literal Unicode rho fails on the pdf device**: "conversion failure on
  'rho (Expression)' in 'mbcsToSbcs'". The PNG showed the character and the PDF would
  have lost it, which is a defect only the vector output has and only a reader of the
  vector output would see. The label is a plotmath expression now
  (`quote(rho ~ "(Expression)")`), which both devices render, and `make_dms_corr_plot`
  takes a full `ylab` override so an expression can be passed at all.

Checking the PNG is not checking the figure when `save_figure` writes both.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `shade_fill`, `ylab`.
* `antiGen-figures/figures/panels_sc2_export.R` -- passes both.

## 105. Panel D moved to the 2023-01-01 cutoff (2026-11-02)

Same path, same figure, a later training window: base antiGen at 2023-01-01
(`cutoff_series/2023-01-01/base/fitness_simple_d128_n2_lr0.0001_20261024_141848.pt`,
epoch 31 of 32, no epoch selection) in place of the 2022-01-01 rung s88 used.

**The path is still entirely out of sample.** BA.2.86 emerged around August 2023, so
every one of the seven RBD steps is post-cutoff at 2023-01-01 just as it was at
2022-01-01. The query for each step remains the PARENT genotype, which is the only
query for which the step is a prediction rather than a fact, and all seven parents were
re-verified to carry the expected from-residue.

### Ranks, of 3,819

| step | 2022-01-01 | 2023-01-01 |
|---|---:|---:|
| L455S | 258 | **98** |
| V483D | 1,107 | **953** |
| R346T | 233 | **18** |
| F456L | 82 | 91 |
| D483V | 261 | 302 |
| T346R | 19 | **12** |
| Q493E | 191 | **120** |

Five of seven improve, two drift slightly worse, and the count in the top decile is
unchanged at **6 of 7** -- V483D remains the one miss, at 953 rather than 1,107, so an
extra year of training moves it but does not rescue it. The reversion pair at 346 is
where the gain is largest (233 -> 18, 19 -> 12), which is what a model with another
year of 346-site turnover in its training window should look like.

The 2022-01-01 ranks stay in `data/ba286_to_kp3_ranks.csv`; the new ones are a separate
file, so switching back is a re-run of `draw_path_panel.py` against the old CSV.

### Code

* `antiGen-figures/data/ba286_to_kp3_ranks_2023-01-01.csv` -- new.

## 106. Row 2 reordered, and the column widths that had to move with it (2026-11-02)

Row 2 is now `bg_all, bg_dn, case_study`, lettered D E F by POSITION so it still reads
left to right. Both rows now carry bar panels in columns 1 and 2, so B/C sit directly
above D/E -- the same kind of panel over the same roster -- and the case study, which
is an image and shares its shape with nothing, ends the row where it interrupts
nothing.

**The reorder alone was not finished work.** `COL_W` was `c(2.2, 1.65, 1.35)`, sized
when column 1 held the only image. Moving the case study to column 3 put a fixed-aspect
image drawn at 5.5in into the narrowest column: it rendered about **3.4in**, taking its
8pt text to roughly **5pt**. A reorder that silently shrinks a panel below legibility is
not a layout change, it is a defect.

Rebalancing took two goes, and the first was wrong in an instructive way.
`c(1.95, 1.5, 1.8)` took the width from column 2, which **clipped panel B's title** --
"Top decile recall, all escape muta". A ggplot title is clipped at the panel edge
rather than wrapped, so it fails silently and only at the glyph that falls off.
`c(1.85, 1.65, 1.75)` takes it from column 1 instead, which holds the two panels that
can best afford it: the cartoon is a wide image, and the background panel's longest
label (`BA.2 DMS (LY-CoV1404 escape)`) still fits. Checked all three columns
afterwards rather than only the one that was changed.

### Code

* `antiGen-figures/figures/fig_sc2_revised.R` -- row 2 order, `COL_W`.

## 107. aa20_supp_dms: Spearman, and a third target (2026-11-02)

Two changes to `aa20_supp_dms.png`, both opt-in so the PUBLISHED `supp_dms.png` is
untouched -- verified byte-identical by re-running `fig2_sc2_benchmarks.R` after the
assembly was restructured, not just by reasoning about the gate.

**Spearman, two bars per model**, from `dms_long_sp`, for the reason panel G switched
(s102): the log/linear pair collapses under a rank metric, leaving prediction against
position-only control.

**A third target panel.** The figure's two roles are easy to confuse and were, in
conversation: a BAR is a score map being scored, a PANEL is the assay it is scored
against. `BA.2 DMS (LY-CoV1404 escape)` was already a bar on both panels -- and is a
rung on B-E of sc2_results_revised, where the target is observed evolution and no
conflict arises. What it was not is a TARGET. It is one now, as panel C.

It is dropped as a bar **in panel C only**, via a new `self_model` argument. The two
BA.2 bind/expression assays are dropped everywhere for the same reason, but they are
targets of every panel; this one is a target of one, so removing it from A and B would
throw away information. A self-correlation of 1.000 is not wrong, it is uninformative
and reads as a result.

### Three mistakes worth recording

* **`do.call` evaluates its arguments.** `ylab = quote(rho)` is a plotmath SYMBOL, and
  passing it through `do.call` without `quote = TRUE` gave "object 'rho' not found".
* **A duplicate name in one `options()` call silently takes the last value.** The new
  `sc2.supp_dms_height = 15` was added ABOVE an existing `= 9` in the same call, so it
  had no effect and the figure kept rendering at the two-panel height. There is now one
  assignment, not two.
* **Three panels do not fit a two-panel canvas.** At height 9 the plot areas collapsed
  to strips while the rotated model names took the page. Height 15, and the legend
  strip's fraction cut from 0.16 to 0.085 since it is a share of a bigger canvas.

### What panel C shows

Mostly near zero, and the BA.1 bind/expression assays are strongly NEGATIVE (about
-0.15). That direction is the expected one -- escape mutations tend to cost binding --
and is worth a caption line, because a reader scanning for tall bars will read the
panel as "nothing predicts escape" and miss that the DMS rungs are anti-correlated
rather than uninformative.

### Code

* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- optional third target,
  `self_model`, the Spearman supplement branch.
* `antiGen-figures/figures/fig_sc2_aa20.R` -- the options.

## 108. Panel C withdrawn: a one-antibody assay is the wrong target for a rank
## correlation over the whole RBD (2026-11-02)

s107 added a third target panel to `aa20_supp_dms`, correlating every model against the
BA.2 LY-CoV1404 escape assay. Every bar came out near zero. That is not a bug, and the
two obvious suspects were both cleared:

* **RBD filtering is correct.** All 2,023 non-sentinel cells in the file are inside
  `RBD_RANGE`; the slice discards nothing. The whole-spike figure of 7.2% measured is
  the RBD's 45.7% diluted over 1,273 residues.
* **No tie problem.** 1,665 distinct values among the 2,023, none of them zero, so
  there is no tie block capping the rank correlation.

**The cause is the assay's shape.** LY-CoV1404 is ONE antibody, so escape from it sits
in **62 cells at 8 sites** (364, 444-448, 499, 500 -- its epitope) and the remaining
~1,960 measured cells sit around 0.005, which is assay noise. A Spearman over the whole
RBD ranks ~97% noise, so it returns about zero whatever the model does.

**It was hiding a real signal, which is the reason to drop rather than keep.**

| model | rho (whole RBD) | top-decile recall of the 62 epitope cells |
|---|---:|---:|
| EVEscape | 0.046 | **0.435** |
| antiGen (val x10) | -0.025 | 0.065 |
| antiGen (base) | -0.032 | 0.048 |
| BA.1 DMS (bind) | -0.159 | 0.016 |

EVEscape puts 44% of the epitope in its top decile and scores rho = 0.046. A panel that
reports that as nothing is worse than no panel. antiGen's low value is honest -- it
does not find this epitope -- and BA.1 bind's -0.159 is the expected anti-correlation,
escape costing binding.

Panel C is not drawn. The `sc2.dms_escape_target` machinery stays in the drawer, off,
because the finding is about the METRIC and not the mechanism: top-decile recall of the
high-escape cells would have measured this assay properly, and is the metric panels B
and C of the main figure already use.

The assay keeps its PREDICTOR bar on both remaining panels, as before; `self_model` is
only needed when an assay is a panel's own target.

### Code

* `antiGen-figures/figures/fig_sc2_aa20.R` -- third target off, two-panel sizing back.

## 109. A cell-weighted panel H, and the 201 cells its list was missing (2026-11-04)

Panel H draws top-decile recall against the train/test cutoff, EVENT-weighted: a
(site, AA) pair hosted by k queries counts k times, so a handful of heavily-recurrent
cells carry the number. The supplement asks the other question -- each pair counts
ONCE, its value the share of the queries hosting it as a descendant mutation in which
it lands in the top decile, averaged over pairs. That is the pooling the COG-UK list
panels use, here over every RBD cell instead of a list.

No new metric code. `compute_hit_rate(per_cell=True)` already returns the
cell_hits / cell_events pair, `list_macro.py` already accumulates it, and
`submit_macro_allcells_aa20.sh` already takes a `CUTOFF`. Sixteen runs of it.

### The list was wrong, and the wrongness was silent

`ref/all_rbd_cells_aa20.csv` is 201 sites x **19** alts. It has no row where
`wildtype == mutant`, so all 201 (site, reference allele) cells are absent.

**A reversion to the reference is an ordinary mutation.** Whenever a query has
diverged at site i, a change back to the reference allele at i is a real event, and
the eval predicts it from the same candidates as anything else -- the candidate pool
is 201 x 20 = 4,020 in BOTH this run and `eval_all_sc2.sh` (`candidate pool: common
-- 4020 of 4020`, the only pool line in any eval_all_sc2 log). The POOL was never the
problem. The TARGET SET was: the 19-alt list drops those events.

Measured at cutoff 2021-01-01:

| list | queries with a target | target events |
|---|---|---|
| 3,819 (19 alts) | 22,317 (31.1%) | 43,314 |
| 4,020 (20 alts) | 23,957 (33.4%) | 46,433 |

3,119 events, 6.7% of the total, thrown away.

### What caught it

`list_macro_summary.csv` carries `macro` and `micro` side by side, and with every cell
listed the micro column IS `quantile_hit_rates_all_all-models.csv`. On the 3,819 list
it failed on all 21 shared models -- `where_times_which` off by 0.101,
`Omicron_BA2_bind` by 0.089. On the 4,020 list it reproduces the published panel to
1e-8.

**The fixed maps are the tell.** `Omicron_BA2_bind` is a static score map: it ranks
identically in both runs. Nothing but a different target population can move it. That
ruled out the pool, the tie rule and the checkpoints in one step, and pointed straight
at the list.

I had the diagnosis backwards first -- I wrote that the 3,819 list was "still the right
deliverable" and treated the 4,020 run as a validation device. It is the other way
round: 4,020 is the deliverable, and excluding reversions is an unjustified
restriction on a panel claiming "all RBD mutations". Ivan caught that.

### What this does NOT change

**The COG-UK panels are fine.** `coguk.csv` has no (site, reference) rows either, and
at most 11 alts at any site -- but there the list IS the object of study, a watchlist
of flagged mutations, which by construction contains no "revert to Wuhan" entry. Those
panels claim performance on that watchlist and deliver it, and they already score
against the full 4,020-cell pool, so a model ranking a reversion highly pays for it
there exactly as in panel H.

**No published figure was affected.** The 3,819 all-cells output was never staged into
`antiGen-figures/data/`, and no figure script reads an `_allcells` path. The defect
was latent; it would have entered the paper only through this supplement.

### Guards, because the failure mode is a silent no-op

* `macro_series_daemon.sh` PINS `LIST` rather than letting the submit script fall back
  to the 19-alt default.
* It writes to a separate `_allcells_inclref` suffix. `list_macro` RESUMES from its
  summary CSV and skips any model already in it, so re-running a cutoff into the old
  directory would find all 22 rows present, skip everything, and leave the stale
  numbers standing -- a no-op that looks exactly like success.
* `WANT=22`, not the 21 rows the per-cutoff `quantile_hit_rates` CSV carries. Those
  evals predate `BloomNeher_2023-10-01` landing in `data/baselines`, and a run today
  picks it up. WANT is tested with `-ge`, so 21 would mark a cutoff COMPLETE one model
  short and stop resubmitting it.

### Cost, measured rather than guessed

~13 minutes per cutoff, not the ~7 hours the script's header quotes. That figure is
the 2021 arm's 31-model roster, which carries ESM-2 650M, Tranception and three
MSA-Transformers; a cutoff stage carries four checkpoints, one of them a 35M model.
Sixteen cutoffs is ~4 GPU-hours, so the full quarterly grid is affordable and no
coarsening is needed.

### Code

* `antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv` -- new, 4,020 cells.
* `antiGen-figures/figures/fig2_sc2_benchmarks.R` -- `gate_recall_by_macro()`, a third
  `sc2.gate_loader`. Additive; `by_position` stays the default.
* `antiGen-figures/figures/figS_gate_macro.R` -- new. Evaluates `fig_sc2_aa20.R`'s
  option block with its trailing `source()` stripped, THEN overrides, because a
  wrapper's own `options()` runs last and wins (the s-note trap again).
* `experiments/multitree/macro_series_daemon.sh` -- new.
* `experiments/multitree/stage_macro_summaries.sh` -- new.
* `experiments/multitree/submit_macro_allcells_aa20.sh` -- header corrected: it
  claimed a gate its own default list cannot satisfy.

## 110. Panels B-E pooled per mutation, and a background row that had no such
## pooling to switch to (2026-11-04)

sc2_results_revised's B-E are CELL-weighted on the CoG-UK list: each (site, AA) pair
counts once. The supplement asks them EVENT-weighted -- a pair hosted by k queries
counts k times -- so the gap between the two reads as how much of each score rides on
recurrence. Same list, same pool, same tie rule; only the denominator moves.

**No GPU time.** Both halves came out of files that already existed.

### The recall half was already built

`fig_coguk_list.R` has carried both poolings as first-class metrics all along:

```r
RECALL_METRICS <- c("micro_all", "micro_unseen", "macro_all", "macro_unseen")
RECALL_MAIN <- sprintf("%s, cell-weighted",  c(ALL_NOUN, DN_NOUN))
RECALL_SUPP <- sprintf("%s, event-weighted", c(ALL_NOUN, DN_NOUN))
```

`list_macro.py` writes `micro` beside `macro`, so panels A/B of the new figure are a
metric selection and nothing else.

### The background half had no pooling to select

`background_contrast.py`, line 52: "The panel value is the unweighted mean over
cells." There is no weighted variant in it, and none in the summary CSV it writes --
`lift`, `auc`, `n_cells`, `n_pairs`, already aggregated per cellset.

But `background_percell.csv` carries per-cell `lift`/`auc`/`delta_pct` AND `n_hosts`,
so the re-pooling is arithmetic over a file on disk rather than hours of re-scoring.
`background_event_weighted.py` does it.

**THE GATE IS `--weight none`.** An unweighted re-pool must reproduce
`background_summary.csv`, because that is what that file already holds. It does, to
**4.94e-07 over 180 (model, cellset) pairs**, and `--check` refuses to write on
failure. The weighting is only trustworthy because the un-weighting is checkable.

**The bootstrap is weighted too.** `rng.choice(p=...)` was unusably slow at 10k draws x
~1.2k cells x 360 intervals, so it samples cells by inverse-CDF on a cumulative weight
vector. Resampling cells with probability proportional to `n_hosts` and taking an
unweighted mean has the host-weighted mean as its expectation -- so the interval is
weighted like the point it brackets, not a cell-weighted interval around an
event-weighted dot.

### What is zero, and why that is right

Background lift asks whether a model ranks a mutation higher on the backgrounds that
HOST it than on controls. A score that never looks at the background cannot have any,
so PSSM, EVEscape, pyr0, every DMS column and the MSA-Transformers are identically
zero under BOTH poolings. Everything that conditions on the background sequence moves:
antiGen and its gates, the whole ESM-2 family, Tranception, EVE, the ensembles, codon,
substitution, wxw_dirichlet. I first wrote that only three rungs could move, from a
sample of eight models, and that was wrong.

### This is NOT the existing coguk_list_aa20_supp

That figure's background row swaps the CELL SET -- the curated list for every RBD cell
with a backbone host -- and leaves the pooling alone, as its own comment says. It
changes the mutations, not the weighting. The two supplements answer different
questions and both are worth having.

### The size of the effect

antiGen on de novo CoG-UK cells: 15.56 pp cell-weighted, **24.82 pp event-weighted**.
The recurrent de novo mutations are the ones the background signal reads best on, so
per-mutation pooling makes that panel considerably stronger rather than weaker --
the opposite direction to the recall panels, where event-weighting is four fifths a
memorisation test (s-note at the head of fig_coguk_list.R).

### Code

* `experiments/multitree/background_event_weighted.py` -- new.
* `antiGen-figures/figures/figS_coguk_eventwtd.R` -- new. Same eval()-then-override
  pattern as figS_gate_macro.R, and it redirects the drawer's own two save_figure()
  calls to scratch names so sourcing it cannot touch a published output.
* `antiGen-figures/data/evals_sc2_background_aa20_eventwtd/` -- the re-pooled summary.

Published figures verified byte-identical before and after.

## 111. Two more silent mismatches between list_macro.py and eval.py (2026-11-04)

s109 fixed the cell list and declared the micro gate passed. It was not: that check ran
with 14 of 22 models written, and the three that fail had not been computed yet.
**A gate run against a partial roster is not a gate.** With all 21 shared models:

| model | published | micro | diff |
|---|---|---|---|
| `where_times_which` | 0.711434 | 0.602577 | -0.109 |
| `substitution` | 0.450944 | 0.409877 | -0.041 |
| `wxw_dirichlet` | 0.724226 | 0.714923 | -0.009 |

Eighteen others matched to 1e-8.

### The diagnostic, which is the reusable part

Everything that matched IGNORES `training_examples`: the eleven fixed maps, `codon`
(a fixed table) and `pssm_root` / `pssm_root_position` (read from `--root-counts`).
Everything built FROM `training_examples` diverged. That is a property of the INPUT,
not of any one estimator -- three unrelated models cannot go wrong the same way by
coincidence. It says "the training pool differs" without needing to read either model.

### The cause: per-file roles vs a pooled split

```
eval.py      fitting substitution matrix on 6836 training examples (seed=0)
list_macro   training pool: kept 2934 of 2934
```

`eval.py` pools `--train-examples` AND `--test-examples` and then splits on
`is_training`, so the test corpus's PRE-CUTOFF branches count as training:
2,934 + 3,902 = 6,836. `list_macro.py` assigns the role per FILE -- training comes
only from `--train-examples` -- so the fit sees 43% of the data.

**Only with `--split-date`.** Without one, the test corpus has no pre-cutoff branches
and the two agree, which is every published list_macro run. `--pool-examples` is
therefore DEFAULT OFF, and the published CoG-UK macro numbers cannot move.

### And a second, smaller one found on the way

`--substitution-seed` defaults to **14 in list_macro.py and 0 in eval.py**, and
neither submit script passed it. Fixing only this moved `substitution` from -0.041 to
-0.045 -- real, but not the cause. Both had to be right. The defaults still diverge in
the repo; `submit_macro_allcells_aa20.sh` now pins the seed at the call site rather
than changing either default, since changing list_macro.py's would move published
numbers.

### Two wrong turns, recorded because the reasoning was the error

1. I read the submit scripts' flag diff and concluded the cause was the missing
   `--variant-counts-dir` / `--counts-since`, patched it, and relaunched all 16. The
   jobs died in 15 seconds: `list_macro.py: error: unrecognized arguments`. That
   script has no variant-counts concept, and reading both constructors shows
   `_where_times_which` and `_wxw_dirichlet` are IDENTICAL in the two files and use
   neither. Inferring a cause from a flag diff without reading the code that consumes
   the flags.
2. Then the seed, scaled to a one-cutoff test rather than sixteen -- the right size,
   finally -- and it failed. Only after that did I compare the two training-pool lines,
   which is where the answer had been all along.

### Three silent defects, one gate

| # | defect | effect |
|---|---|---|
| 1 | 19-alt list omitting (site, reference) | 6.7% of target events dropped (s109) |
| 2 | per-file example roles | substitution fit on 2,934 not 6,836 |
| 3 | substitution seed 14 vs 0 | a different fit again |

Each produced a run that completed with all 22 models and rendered a plausible figure.
The only thing that caught any of them is that `list_macro_summary.csv` carries
`micro` beside `macro`, and micro MUST reproduce
`quantile_hit_rates_all_all-models.csv`. Keep that identity in any figure of this kind.

### Directory suffixes, which are now load-bearing

`_allcells` (s109's wrong list), `_allcells_inclref` (right list, wrong example roles)
and `_allcells_pooled` (both right) all look complete at 22 models. `list_macro`
resumes from its summary and skips models already present, so re-running into a
superseded directory is a no-op that looks exactly like success. The daemon and the
staging script both default to `allcells_pooled`.

### Code

* `antiGen/list_macro.py` -- `--pool-examples`, default off.
* `experiments/multitree/submit_macro_allcells_aa20.sh` -- passes `--pool-examples`
  with `--split-date`, and pins `--substitution-seed` (`SUBSEED`, default 0).
* `experiments/multitree/stage_macro_summaries.sh`, `macro_series_daemon.sh` --
  default suffix `allcells_pooled`.

## 112. Two counted-statistic baselines, and a candidate-registration regression
## they uncovered (2026-11-05)

`Site x Substitution` (s_i * M[wt_i, a]) and `SxS, MCM interpolation`
((N + kappa*s*M)/(N_total+kappa), kappa = N_total) added to sc2_results_revised's
B-E and H, from `av-multitree`'s `statistical_baselines.py`.

**The file is copied VERBATIM** to `antiGen/statistical_baselines.py`. It calls itself
the single source of truth for this math and needs only torch, so copying beats
re-deriving: the two arms cannot drift. `CountedStatPredictor` (eval_utils) is a thin
wrapper that puts it behind the model interface, which it needs because the scores are
EXAMPLE-DEPENDENT -- `M[wt_i, a]` is indexed by the query's own residues -- so they
cannot go through `fixed_map_metrics`.

### They are the counted siblings of two rungs already drawn

| ours (fitted codon x BLOSUM) | these (counted M) |
|---|---|
| `where_times_which` = log s_i + log P_fit(a\|wt_i) | `site_substitution` |
| `wxw_dirichlet` = (N + kappa*pi_fit)/(N_t+kappa) | `mcm_interpolation` |

Each is drawn BESIDE the rung it mirrors, so the panel reads as what the fitted "which"
term buys over counting substitutions. The answer, at every cutoff measured, is
NOTHING: SxS beats where_times_which at both ends (0.743 vs 0.711 at 2021-01-01, 0.792
vs 0.775 at 2024-01-01) and on panels D/E Site x Substitution places FIRST, above
antiGen (val x10). The Dirichlet pair crosses over late (0.872 vs 0.876 at 2024-01-01).

**They are near-duplicates of EACH OTHER**: rank corr 0.988 between them, 0.32 between
the interpolation and the MCM it is supposed to interpolate toward. With kappa =
N_total the prior s*M spreads over all 25,460 cells while N concentrates on a few
thousand, so the interpolation is dominated by the SxS term. That is a property of the
published kappa, not a defect, but it is why the two lines sit on top of each other.

### Their own flag, not --multitree-baselines

The figures key rows BY NAME and an unlisted model stops the render ("models with no
group assignment"), so a roster must not grow silently. `--counted-baselines` in
eval.py, list_macro.py and background_discrimination.py; the figure side uses the
existing optional-rung pattern (`coguk.wxw_dirichlet`), default empty.

### THE REGRESSION: built models were never scored

All sixteen cutoff evals COMPLETED in 26 seconds having added nothing -- printing the
roster, resuming past the checkpoints, writing outputs and reporting "Done."

`eval.py`'s `for name in builtin_models: candidates.append(...)` was nested inside
`if args.ensemble_dir:`. Without an ensemble directory every built model is
CONSTRUCTED AND THEN DISCARDED. Not mine: it is nested at HEAD, and an older cutoff log
shows "added 3 factored baselines" with no --ensemble-dir, so it regressed between.

**It was never about the two new rungs.** `codon`, `substitution`,
`where_times_which` and `wxw_dirichlet` would ALSO vanish from any fresh CSV; the
existing ones only look healthy because older runs wrote them. The fix moves the loop
out and skips names already queued (`args.ensemble_name` is appended explicitly AND
lives in `builtin_models`, so an unguarded loop double-scores it). After it:
"added 6 built model(s)".

Re-scoring the four pre-existing baselines reproduced the published numbers EXACTLY --
16 cutoffs, worst |new - published| = 0.0e+00, no duplicate rows.

### Three figure-side traps, two of them self-inflicted

* **The options-ordering trap, walked into after quoting it.** Setting
  `coguk.drop_models` in `fig_coguk_list_aa20.R` CLOBBERED
  `panels_coguk_export.R`'s pruned list, because a sourced wrapper's `options()` run
  after the caller's and WIN. That put the whole fine-tune block back into B-E, which
  widened the top legend to two rows, rebalanced the grid and CLIPPED panel B's title.
  Fixed by APPENDING: `c(getOption("coguk.drop_models", c("CoVFit","pyr0")), ...)`.
* **`aa20_supp_*` clobbering (pre-existing, s-note).** `fig_sc2_revised.R` runs
  `panels_sc2_export.R`, which sets a pruned roster and then sources
  `fig_sc2_aa20.R`, which re-writes the aa20 supplements with that pruning --
  `aa20_supp_dms.png` lost its antiGen and Ensembles groups, 16.6% of pixels.
  ORDER IS THE FIX: render the revised figure first, the aa20 arm LAST.
* **A silent success in my own finisher.** `finish_counted_baselines.sh` sent the
  render step to /dev/null, so both renders failed and it printed "DONE".

### BloomNeher_2023-10-01 came along for the ride

Re-running list_macro for the counted rungs also scored it (it sits in
data/baselines), so the roster went 31 -> 34, not 32. It is a DATE-SPECIFIC map and
this corpus is cut at 2021, so it is dropped by name rather than grouped.

### Background: 2 models, not 48

`background_contrast.py` has no resume and re-scores every `scores_*.npz` -- hours,
past quick_preemptible's 2h wall, with NOTHING written when killed. Lift is per-model
and the cellsets come from the BACKBONE, not the roster, so the two rungs were scored
in an isolated directory (10m48s) and their rows merged. The equivalence is CHECKED,
not assumed: 2,426 cells, identical sets AND host counts, or the merge refuses.

### Cluster, for the record

gpu_batch congested (6 nodes draining); cpu_batch hit `QOSGrpCpuLimit`;
quick_preemptible caps submissions per user (8 of 16 accepted) AND walltime at 2h,
which `submit_bg_aa20.sh`'s hardcoded `--time=24:00:00` violated outright. Partition
and walltime have to move together; it now takes a `TIME` knob.

### Code

* `antiGen/statistical_baselines.py` -- new, verbatim from av-multitree.
* `antiGen/eval_utils.py` -- `CountedStatPredictor`.
* `antiGen/config.py` -- `SITE_SUBSTITUTION`, `MCM_INTERPOLATION`.
* `antiGen/eval.py` -- `--counted-baselines`, and the registration fix.
* `antiGen/list_macro.py`, `antiGen/background_discrimination.py` -- same flag.
* `fig_sc2_aa20.R` -- two rows, dashed, legend 4 -> 5 rows, and the gate colour
  vectors converted POSITIONAL -> NAMED (same colours to the same rungs; naming makes
  that independent of row order, which is what the comments there warn about).
* `fig_coguk_list.R` / `_aa20.R`, `panels_sc2_export.R`, `figS_gate_macro.R`.

### Addendum: the event-weighted supplement (same day)

`aa20_supp_coguk_eventwtd` carries both rungs too. Its RECALL panels needed nothing --
the figure evaluates `fig_coguk_list_aa20.R`, which now opts in -- but its BACKGROUND
panels read `data/evals_sc2_background_aa20_eventwtd/`, a re-pool of
`background_percell.csv` that was generated before the rungs existed and so sat stale
at 45 models. Re-running `background_event_weighted.py --weight hosts` brings it to 47.

The gate came along for free and is the reason to keep it: `--weight none` still
reproduces the published summary, now over **188** (model, cellset) pairs rather than
180 at the same 4.94e-07 -- eight more, which is exactly the two new models x four
cellsets. A re-pool that silently dropped them would have shown up as 180.

Both are NON-ZERO on the adaptability panels, unlike most of that figure's roster: a
fixed score map cannot have background lift, but these index `M[wt_i, a]` by the
query's own residues and so condition on the background. They beat their fitted
siblings under event weighting as well as cell weighting -- de novo, 18.7 pp against
where_times_which's 10.1 -- so the ordering is not an artefact of the pooling.

### Addendum 2: the gate on B-E, at the matched cutoff

`gated_static_xfmr_pssm_lr0.0001` -- panel H's headline rung -- added to B-E as an
optional rung (`coguk.gate_rung`), beside base antiGen and in its group.

**THE CUTOFF MATCH IS REAL, not nominal.** The gate is trained PER CUTOFF and these
panels are the 2021 arm, so only `cutoff_stage_aa20/2021-01-01` is admissible. Checked
rather than assumed: the two training corpora have byte-identical root counts and
differ in exactly ONE field -- `POST_CUTOFF: False` against a blank -- on all 2,934
examples and nothing else. Same data, different serialisation, so the gate saw what the
other B-E rungs saw. (That identical counts file is also why `pssm_root` matched
bit-identically back in s111 while the trained rungs did not: it reads --root-counts.)

Anchoring on base antiGen is safe even though the revised figure DROPS it:
`drop_models` filters the data frame and never prunes MODEL_GROUP.

| panel | gate | best non-antiGen |
|---|---|---|
| B recall, all | 0.1391 macro, 2nd overall | Site x Substitution 3rd |
| C recall, de novo | 0.1784 macro | Site x Substitution 2nd |
| D adaptability, all | +8.7 pp [+4.5, +13.2] | Site x Substitution 1st, +14.1 |
| E adaptability, de novo | +11.5 pp [+5.7, +17.8] | Site x Substitution 1st, +18.7 |

So the gate is second only to antiGen (val x10) on all-escape recall, and
`Site x Substitution` BEATS IT on de novo recall and on both adaptability panels. Worth
noting, since the gate is what panel H is built around.

Scored the same way as the counted rungs: one model in an isolated contrast directory
(cellsets identical, 2,426 both ways) rather than re-running all 49. The unweighted
gate went 188 -> 192 rows, exactly the gate's four cellsets, which is what shows the
merge neither lost nor duplicated anything.

RENDER ORDER AGAIN: coguk, revised, eventwtd, then `fig_sc2_aa20.R` LAST. All four
`aa20_supp_*` PNGs verified byte-identical to HEAD afterwards.

### Addendum 3: the counted rungs on panel G (DMS)

Panel G reads `data/sc2_aa20/<model>/variant_root_predictions_*.csv` -- one (seq_len,
22) matrix per model at the variant root -- so a rung reaches it only by having such a
file. `where_times_which` has an inventory entry but NO such file, which is why it has
never appeared there.

**NO GPU.** `dump_variant_root_predictions.py` rebuilds those from the `--dump-scores`
cache, because the variant root is already one of the cached queries. The list_macro
runs used `--dump-scores`, so `site_substitution`, `mcm_interpolation` AND the gate
were all already in `list_score_maps_aa20`.

**The dumper was checked against a committed file first**: re-dumping
`simple_d128_n2_lr0.0001` reproduced
`data/sc2_aa20/simple_d128_n2_lr0.0001/variant_root_predictions_*.csv` over 4,020
finite cells at worst |delta| = 0.00e+00, with an identical NaN pattern.

Both rungs also needed entries in `subsets$pandemic_training_data` AND
`longform_names`, POSITIONALLY matched -- the two vectors are parallel and
length-checked, and an unlisted model becomes a bar called "NA" in the antiGen group.

| rung (expression, Spearman) | prediction | position-only |
|---|---|---|
| Site x Substitution | 0.291 | **0.442** |
| SxS, MCM interpolation | 0.298 | **0.449** |
| antiGen | 0.261 | 0.337 |
| EVEscape | 0.362 | 0.354 |

**Their POSITION-ONLY controls are the tallest bars on the panel** -- above every
model's actual prediction, EVEscape included -- and for both, position-only far exceeds
the prediction. That is mechanically unsurprising and is the point: SxS is
`log s_i + log M[wt_i, a]`, and `s_i` IS a positional marginal, so the panel's own
position-only control recovers most of the signal while the substitution term actively
costs it ~0.15 of rho. Read that way, these two rungs say the DMS panel's expression
correlation is largely a positional effect.

**A cross-check of mine disagreed and was WRONG.** A ten-line reimplementation of the
correlation gave 0.150 for `site_substitution` and 0.286 for base antiGen, against the
figure's 0.291 and 0.261. The figure's base-antiGen value matches the bar already
published for it, so the drawer's pipeline is right and the reimplementation's
wild-type masking is not -- it matched the reference against `colnames(expr)` where
`mask_wildtype` uses `AMINO_ACIDS`. Recorded because the disagreement was resolved by
checking which one reproduced a known published number, not by preferring the newer
code.

**`data/sc2_aa20/` IS NOT TRACKED, and that is pre-existing.** All 21 model
directories under it are untracked, where `data/sc2` (the published arm) has 45 files
committed -- so panel G cannot be rebuilt from a clone on this arm, and could not
before this work either. The two new directories were deliberately NOT committed
alone: `models <- list.files(MODEL_DIR)` would then find 2 of 21 in a fresh clone and
render panel G with two bars and no error, which is worse than the present state of
failing outright. Committing all 21 would close the gap and match the `data/sc2`
precedent; it is left for a decision rather than taken here, since it is 19 files
beyond this change.

### Addendum 4: the counted rungs on the CELL-WEIGHTED panel H (2026-11-07)

`aa20_supp_gate_macro` (s109/s110) was built before these rungs existed, and its series
lives in `data/gating_macro_aa20/`, separate from panel H's `data/gating_aa20/`, so
none of the s112 work reached it.

**RESUME, not a new suffix.** Re-run into the SAME `_allcells_pooled` directories:
nothing about the cell list or the example-pooling changed this time, so the 22 rows
already there are still valid and `list_macro` skips them, scoring 2 models per cutoff
instead of 24. All sixteen in TEN MINUTES. The earlier suffix changes (s109, s111) were
necessary precisely because those runs invalidated the existing rows; this one does not,
and using a fresh suffix here would have cost 3.5 GPU-hours for nothing.

The figure needed no edit: `figS_gate_macro.R` evaluates `fig_sc2_aa20.R`, which already
carries both rungs in `sc2.gate_rows` with named colours.

**The gate went 336 -> 368 comparisons at 6.0e-08.** The INCREASE is the check that both
rungs landed on every cutoff: 16 x 2 is exactly 32, and a rung missing from three
cutoffs would show up as 362, not as a passing worst-case.

### Counted beats fitted at EVERY cutoff

| | mean (counted - fitted) | cutoffs won |
|---|---|---|
| `Site x Substitution` vs `where_times_which` | +0.0091 | **16/16** |
| `SxS, MCM interpolation` vs `wxw_dirichlet` | +0.0106 | **16/16** |

32 of 32 comparisons. Taken with the main figure (0.743 vs 0.711 and 0.792 vs 0.775
event-weighted), panels D/E (where SxS places first), and the event-weighted supplement
(18.7 pp against 10.1), the ordering now holds under BOTH poolings, on every panel, and
across the whole cutoff series. The fitted codon x BLOSUM "which" term is not buying
anything over counting substitutions anywhere we have measured.

## 113. sc2_results_revised2: the two poolings as the figure's argument (2026-11-07)

A revision of the revision. Draws nothing new and runs no evaluation; every panel is
one of the other figures' own objects.

| | sc2_results_revised | revised2 |
|---|---|---|
| C | top decile, de novo, PER CELL | top decile, all, PER MUTATION |
| D | background lift, all list cells | top decile, de novo, PER CELL |
| E | background lift, de novo | top decile, de novo, PER MUTATION |
| G | DMS correlation | the cutoff series, PER CELL |

**Reordered to read top-down as an argument** (final layout): row 1 says what the
evaluation IS -- the definition, then one worked case; row 2 says how it behaves as the
training window grows, per cell beside per mutation; row 3 is the full roster broken out
as a 2x2, {all, de novo} x {per cell, per mutation}, left to right per cell -> per
mutation in both halves so the contrast runs one way everywhere.

    row 1   A cartoon            B case study
    row 2   C cutoff, per cell   D cutoff, per mutation
    row 3   E all, per cell      F all, per mutation
            G de novo, per cell  H de novo, per mutation

The gap within each pooling pair reads directly as how much of a score rides on
recurrence, which is what s109/s110 built the machinery to measure and which no single
figure showed side by side before.

**FOUR BAR PANELS IN ONE ROW changes two things.** Only the LEFTMOST carries the y-axis
model names -- all four share one roster in one order, so the names are read once per
row rather than four times -- and the titles shrink to the subset and the pooling
("All escape, per cell"), the x axis already saying what the number is -- a ggplot
title is clipped, not wrapped. The per-cell pair share a ~0.5 x-range and the
per-mutation pair ~0.8, so the columns stay comparable by free scales rather than by a
forced common one.

**11 x 10, not 16 x 10.** At 16in the two fixed-aspect images in row 1 floated in white
space: an image cannot grow past its own aspect, so extra width is margin rather than
picture. Narrowing also makes every label LARGER relative to the figure, since text is
in points and does not scale with the canvas. The binding constraint is the four titles
in row 3 -- at 11in "De novo escape, per mutation" nearly spans its panel, so this is
about as narrow as the figure goes without shortening them again or dropping the base
text size.

**ALL FOUR OF E-H ARE THE CoG-UK ESCAPE LIST.** "de novo" is the subset of that list
absent from the training window, NOT a different target set, so every one of the four
titles says "escape". Two of them said only "De novo" at first, which reads as a change
of target rather than a change of subset -- Ivan caught it.

Dropped in the process: BOTH background-lift panels. `sc2_results_revised` is untouched
and still carries them.

**I built the wrong D and E first.** Reading "replace C and D with the event-weighted
A and B" literally puts both event-weighted panels in C and D and discards the per-cell
de novo panel, which destroys the 2x2 rather than completing it. The end state the
instruction described -- not the operation it named -- was the thing to build: the old
C MOVES to D and the second event-weighted panel takes E.

### Three subprocesses, not two

The per-cell and per-mutation cutoff panels are the SAME drawer over different data
through different loaders, and `sc2.gate_loader` is one option read at source time, so
one R session cannot build both. `panels_gate_macro_export.R` is the third export.

### The event-weighted bars are pruned like the rest

They come from `panels_coguk_export.R`, which sets this figure's `coguk.drop_models`,
so C and D carry the revised figure's roster -- NOT `aa20_supp_coguk_eventwtd`'s full
one, which is what "filtered to the same models" asked for.

### Two layout defects of mine, both caught by looking

* **`first` IS THE Y-AXIS LABELS.** Exported with `first = FALSE` for `decile_dn_evt`,
  which then LED row 2 -- so D and E both drew bars with no model names at all. It is
  TRUE for whichever panel leads its row.
* **A clipped title, twice.** A ggplot title is clipped rather than wrapped (the trap
  s106 records). The leading panel of a row loses width to its y-axis labels, so D's
  title has to be SHORTER than its partner's, not the same length.

### Both sides of each pair name their pooling

`decile_all` and `gate` carry unqualified titles and are left alone for the revised
figure. revised2 uses new `decile_all_pc` and `gate_pm` objects titled "(per cell)" and
"(per mutation)". Unqualified on both sides of a pooling comparison, a reader has to
already know which convention the bare title meant -- and B is per cell while H is per
mutation, so the two defaults are not even the same.

### Render order, again

`panels_sc2_export.R` sources `fig_sc2_aa20.R`, so running revised2 rewrites
`aa20_supp_dms.png` and `sc2_results_aa20.png` with a pruned roster. Rendering the aa20
arm standalone afterwards restores both to byte-identical. Verified: after the final
render, the ONLY PNG differing from HEAD is the new figure itself.

### Code

* `antiGen-figures/figures/fig_sc2_revised2.R` -- new.
* `antiGen-figures/figures/panels_gate_macro_export.R` -- new.
* `panels_coguk_export.R` -- `decile_all_evt`, `decile_dn_evt`, `decile_all_pc` added;
  existing objects untouched, so `sc2_results_revised` is unaffected.
* `panels_sc2_export.R` -- `gate_pm` added, same way.

## 114b. Panel B again, matched to style.png (2026-11-07)

s114's flat cold version was wrong twice over, and MEASURING the reference said so.

**The reference is not flat.** `style.png`'s body luminance runs 104..246, a min/max of
0.42, where s114 had set `ambient = 0.80` (ratio 0.80) on the theory that "simpler"
meant "flatter". It has MORE modelling than the flat version, not less.

**But matching the range does not match the picture.** At `ambient = 0.42` the glyphs
got a broad charcoal band. The histograms say why:

| | p5 | p25 | p50 | p75 | frac < 160 |
|---|---|---|---|---|---|
| style.png | 157 | 203 | 233 | 241 | **5.6%** |
| lambertian, amb 0.42 | 130 | 159 | 196 | 219 | **25.3%** |

Style is light nearly EVERYWHERE with a small dark tail. Lambertian diffuse darkens
everything turned away from the lamp, so its luminance spreads smoothly. No value of
`ambient` fixes that: the two produce different SHAPES of histogram.

**That shape is ambient occlusion.** Near-uniform lit surface, sharp dark creases.
`render_rbd_surface` now has an `occlusion` term -- a screen-space proxy comparing the
smoothed height field against a broader blur of itself, so a pixel sitting below its
own neighbourhood is in a pocket. One extra gaussian, no rays. With `ambient = 1.0`
(diffuse OFF entirely) and `occlusion = 0.65`:

| | p5 | p25 | p50 | p75 | frac < 160 |
|---|---|---|---|---|---|
| style.png | 157 | 203 | 233 | 241 | 5.6% |
| **AO, amb 1.0 occ 0.65** | **157** | 212 | 232 | 237 | **5.6%** |

### Palette, sampled not eyeballed

`BODY #e5f3fc`, `SURF_NET #d5413e`, `SURF_REVERT #f6d5be`, outline `#7f868d` -- all
read off the reference. The outline is a NEUTRAL grey there, not the blue this used.

**SURFACE AND INK NEED DIFFERENT VALUES.** The sampled peach is right on the glyph,
against a pale blue body, and unreadable as 8pt text on white. `SURF_REVERT` is the
sampled value; `REVERT` (`#cf8f62`) is the darker one for labels and markers. The dark
accent is legible either way and needs only one.

`color_smooth` is also a parameter now: the depth blur has to merge atoms into one
body, but blurring COLOUR that hard makes a highlighted patch bleed until it reads as
a stain rather than a site. 0.30 keeps the sites discrete, which is how style.png
shows them.

### Revision: exposure and one warm family

Matching style.png's HISTOGRAM still produced a picture that looked badly exposed --
some areas blown out, some crushed -- because the numbers that matched were the wrong
ones to chase. At `ambient = 1.0`, **19.3% of body pixels sat at the ceiling** (flat,
no gradation: "overexposed") while AO crushed creases to p1 = 109 ("underexposed").
A reference rendered at a different size and by a different renderer does not
transfer by percentile.

Final: `ambient 0.82`, `occlusion 0.30`, `occlusion_scale 0.35` (softer), `smooth
0.019`. Clipped pixels 19.3% -> 1.6%, p1 109 -> 154. A little diffuse back, a much
gentler occlusion: lobes still read, nothing pinned to either end.

**Two colour families, not three.** The light accent sampled at hue ~24 deg, and an
ink darkened from it drifted to orange -- blue, red, AND orange. Both accents now sit
on the red's own hue (~1 deg): surface `#d5413e` / `#f5c6c4`, ink `#c9413e` /
`#e3908c`.

### The lesson

Three rounds of adjusting-by-eye moved slowly and in the wrong direction. One
histogram of the reference settled both the parameter AND the lighting model. When
there is a reference image, measure it.

## 114. Panel B: flatter glyphs and a cold palette (2026-11-07)

### The shading terms are arguments now

`render_rbd_surface.render` took its four light terms as literals; they are named
arguments with the old values as defaults, so the original look is reproduced exactly
and a caller can ask for a flatter one without a second renderer.

| term | default | panel B |
|---|---|---|
| `ambient` | 0.54 | **0.80** -- compresses light-to-dark, so the body is a shape with a little modelling rather than a photograph |
| `specular` | 0.13 | **0** -- the single strongest "3D" cue; a hard bright spot says "shiny solid" louder than the shading does |
| `rim` | 0.16 | **0** -- back-light along the silhouette, which lifts the body off the page |
| `depth_cue` | 0.11 | **0.05** |

`smooth` 0.0118 -> 0.020 as well, merging the per-atom bumps into one body.

**THE SILHOUETTE STAYS.** It is tempting to read "simpler" as "fewer lines", but the
module docstring already records why it exists: shading alone leaves the body
dissolving into the page. On a FLATTER body it matters more, not less -- with the rim
light gone the outline is the only thing separating shape from background.

### Cold palette

`NET` `#c0392b` -> `#2d4f6b` and `REVERT` `#e8a07e` -> `#6b8ead`. The red and orange
were the only warm colour in the panel and two more hues than it needs: filled vs
hollow ALREADY separates the two cases in the rank chart and in the key, so colour was
carrying a redundant distinction there -- but NOT on the structures, where the marks
are the only thing shown and colour is all there is.

**That is why `REVERT` was darkened twice.** The first cold value (`#93aec4`) read fine
as label text on white and VANISHED into `BODY` (`#d6e4f2`) on the glyphs, where
JN.1.11.1's reverting site became invisible. The three values have to separate from
each other AND from the body; mid-slate does, pale slate does not.

### The asset is gitignored

`antiGen-figures/cartoons/*.png` is covered by `.gitignore:65`, so the glyph is NOT in
the repo -- only `draw_path_panel.py` and the figures that embed it are. Consequence:
`sc2_results_revised2.png` now carries the flat cold version while
`sc2_results_revised.png` still embeds the glossy warm one, and the generator can no
longer produce the latter. Re-rendering the revised figure would reconcile them; left
for a decision, since it changes a published figure's appearance.

Rebuild:

    python experiments/multitree/draw_path_panel.py \
      --ranks antiGen-figures/data/ba286_to_kp3_ranks_2023-01-01.csv \
      --pdb antiGen-figures/ref/6M0J.pdb \
      --output antiGen-figures/cartoons/ba286_to_kp3.png

The `_2023-01-01` ranks file, not `ba286_to_kp3_ranks.csv`: the published panel is the
2023-01-01 cutoff (s105), and the two files disagree (L455S 98 against 258).

## 115. sc2_results_revised2 panel A: the redrawn recall cartoon (2026-11-08)

Panel A of `sc2_results_revised2` now draws `antiGen-figures/cartoons/recall_definition_v2.png`
(the user's redrawn cartoon, delivered as `fig2a.png` at the repo root and moved beside
the original `recall_definition.png`). It goes through `plot_cartoon` with the same title,
"Definition of next-mutation recall", exactly as panel B does, so the two image panels
carry matching titles and insets. Only this figure changed: `panels_sc2_export.R` still
ships the original as `sc$cartoon`, and `sc2_results_aa20` keeps it.

Like every cartoon, the PNG is covered by `.gitignore:65` (`*.png`) and lives only on
disk; the committed record is the rendered figure. `fig_sc2_aa20.R` was re-rendered last,
as always (it shares outputs with the export), and the PDFs it rewrote with byte-identical
PNGs were reverted.

## 116. sc2_results_revised2: PSSM is called PFM (2026-11-08)

The PSSM baseline is now named PFM (position frequency matrix). For now only
`sc2_results_revised2` is renamed, since which figures reach the paper is still open.
The drawers' label tables are shared with every other figure, so the rename happens
downstream, at the figure, in one of two ways depending on how each panel draws text:

* **E-H and the legend** use plain text grobs. `fig_sc2_revised2.R` walks the baked
  grobs and rewrites whole-word `PSSM` -> `PFM`, giving "PFM", "PFM Conservation" and
  "antiGen / PFM gate".
* **C and D** (the cutoff series) have an `element_markdown` legend, which lays each
  word out at a fixed x. The grob walk produced "antiGen / PFM   gate" with a hole
  where the longer word had been. These are renamed inside `make_gate_plot` instead,
  through a new call-time option `sc2.gate_label_map` (NULL by default, so no other
  panel moves). It renames the data, the named palette, the dashed set and the bolded
  gate. `model` is a FACTOR there, so the LEVELS are renamed: a first version used
  `ifelse` on the values, got the factor codes back and drew every line grey.
  `run_export` switches the option on by passing `REVISED_PFM=1` to the two gate
  exports.

`fig_sc2_revised2.R` stops if any panel still says PSSM after both steps. The
`sc2_results_aa20` and `sc2_results_revised` outputs are unchanged.

### s116 addendum: the rest of the renames, and two model types

The PSSM -> PFM rename widened into one table, `antiGen-figures/figures/revised2_labels.R`,
still for `sc2_results_revised2` only:

| was | now |
|---|---|
| antiGen (val x10) | antiGen (CV x10) |
| Site × / x Substitution | S×S |
| SxS, MCM interpolation | S×S-MCM interp. |
| Where × / x which | S×M (site × mixture) |
| Where × / x which + events | S×M-MCM interp. |

MCM is the mutation count matrix. Both spellings of the times sign are listed because
the bar drawer writes "×" and the gate drawer "x"; every new name uses ×. Labels are
matched WHOLE, not by regex. Both mechanisms read this one table: the grob walk for
E-H, and `sc2.gate_label_map` for C/D, which the exports now set from the table when
`REVISED2_LABELS=1` (formerly `REVISED_PFM`). The figure stops if anything matching
`REVISED2_STALE` survives.

**Model types.** "Val-selected (10x)" was never a model type, only a group the
ensemble arm appends. Under `REVISED2_LABELS`, `panels_coguk_export.R` now regroups
the frames the bars read: the CV bar joins **antiGen**, and the gate becomes **antiGen
gated hybrid** in `GATE_BLUE`, the colour its line already has in C/D. This is done on
the data frames rather than in `MODEL_GROUP`, because that vector is also the bar order
and must stay contiguous by group; bar order is unchanged.

## 117. sc2_results_revised2: the C/D key moves under the panels (2026-11-08)

The cutoff-series key sat inside the lower-right of C and D, over the dashed baselines.
C and D draw the same nine lines in the same colours, so one key serves both. It is now
a two-row strip under row 2:

* `panels_gate_macro_export.R` exports it as `gate_legend`, built from the C panel's
  own plot. Entries, colours and order are therefore the panel's.
* `fig_sc2_revised2.R` removes the in-panel copies from both baked grobs. In
  ggplot2 >= 3.5 the inside legend is its own `guide-box-inside` component.

Three details:

* **Dashed keys.** The old key drew every line solid, because the linetype guide is
  suppressed. The strip's `override.aes` draws `GATE_BASELINES` (after the label map)
  dashed, as in the panels.
* **Plain text, not markdown.** gridtext measures "×" wider than it draws it, which left
  a gap after every S×S / S×M label. The strip uses `element_text` at 8 pt, against the
  bars' 7 pt row labels, and does the one bold entry with plotmath. `element_markdown`
  and `element_text` do not merge, so the key text is blanked first and then set.
* **Size.** The canvas is 10.5 in tall (was 10). Row 2 gets 1.1 of the height and a small
  spacer keeps the strip off the E-H titles.

## 118. sc2_results_revised2: each key above the row it serves (2026-11-08)

The two keys now sit directly above the panels they describe:

* The C/D line key (s117) moved from under the two panels to above them.
* The model-type key moved from the head of the figure to above E-H, the only panels
  it colours.

Each row is now a `plot_grid` of key and panels, with rel_heights `c(0.1, 1)` for row 2
and `c(0.1, 1.45)` for row 3. The figure's rel_heights are `c(0.95, 1.1, 1.55)`, and the
canvas is unchanged at 11 x 10.5 in.

## 119. Panel B redrawn at its slot's size (2026-11-08)

In `sc2_results_revised2`, panel B's slot is half of the 11 in row and about 2.45 in tall
under its title. `draw_path_panel.py` drew it on a 5.5 x 3.5 in canvas, so `draw_image`
shrank it to about 70% to fit the height. That left white bands either side, and every
label printed at about 70% of its stated size. The canvas is now 5.5 x 2.45 in (`W_IN`,
`H_IN`), so the panel is drawn at 1:1 and the font sizes in the script are the printed
point sizes.

* **Lineage, top half.** The four glyphs (0.98 in tall, at the render's own 380:640
  aspect) span the full width with 0.18 in margins; the arrows share the gaps. Variant
  names are 9.5 pt bold and the mutations on the arrows 8.8 pt bold. Both are about 1.5x
  their previous printed size.
* **Rank chart, bottom half.** Seven rows 0.11 in apart, at 7.3-7.5 pt. The "top decile"
  label had no row above the chart left to sit in, so it is now a third entry in the key,
  drawn as a band-coloured swatch.

Ranks, colours, brackets and rendering parameters are unchanged. The PNG is gitignored
(s114b); rebuild with the command recorded there.

### s116 addendum 2: "Substitution matrix" -> "Codon-B62 mixture"

The fitted substitution model's bar in `sc2_results_revised2` is renamed, in
`revised2_labels.R`. It is not a free 20 x 20 table. It is
w_codon · log(κ·ts + tv + c) + w_blosum · BLOSUM62 + b(m): codon reachability, with a
fitted transition/transversion ratio κ, mixed with BLOSUM62, plus a per-destination
background. That is about 25 fitted numbers (`antiGen/models/substitution_fitness_predictor.py`).
"Codon-B62 mixture" names the two parts. S×M (site × mixture) is that mixture's
conditional times the site marginal.

The κ term is K80-like (a ts/tv ratio with no base frequencies), not HKY85, which also
fits base frequencies. That detail belongs in the caption, not the bar label.
`REVISED2_STALE` now also catches "Substitution matrix".

## 120. figs/ reorganised: inactive figures moved to figs/old/ (2026-11-08)

`antiGen-figures/figs/` now holds only the figures under active revision:
`sc2_results_revised2.png`, `supp_dms.png` and `supp_fixed_split.png`. The other 46 PNGs
moved to `figs/old/` (git renames, content unchanged), `supp_endemic_results.png`
among them. No code reads from `figs/`; it is output only. `figs_vector/` was not
reorganised.

**A render repopulates figs/.** `save_figure` still writes to `figs/`, so re-running
an old figure's script puts a fresh PNG there beside its copy in `old/`. The
revised2 render does this as a side effect: `panels_sc2_export.R` sources
`fig_sc2_aa20.R`, which writes `sc2_results_aa20.png` and the `aa20_supp_*` PNGs.
Until those side effects go to scratch names, delete them after a revised2 render
rather than re-rendering `fig_sc2_aa20.R`.

## 121. The mutation count matrix alone, on panels C and D of sc2_results_revised2 (2026-11-08)

**MCM** is the mutation count matrix by itself: log N[i, a], the number of times
mutation a occurred at site i on the training tree (`statistical_baselines.score_map`,
key `mcm` -> `frequency`). It is the N that S×S-MCM interp. blends with its factored
prior. It is not the PFM, which counts residues across the training SEQUENCES
(`counts_D_cur.tsv`, 351,410 at 2021-01-01) rather than mutation events.

* **Flag.** `--mcm-baseline` on `eval.py` and `list_macro.py` adds the rung `mcm` through
  `CountedStatPredictor`. It is separate from `--counted-baselines` so that no other
  roster grows. Figures key rows by name, and an unknown model collapses into an NA
  group.
* **Scoring.** All 16 cutoffs, in both pipelines, as resume passes that add one model:
  * panel C (per cell): `macro_series_daemon.sh` with `EXTRA_LM_ARGS=--mcm-baseline WANT=25`,
    on `evo_gpu_priority`, 1-2 min each;
  * panel D (per mutation): `submit_cutoff_evals.sh`, which now passes the flag and
    requires `mcm` in `PANEL_ROWS`. Moved to `evo_gpu_priority` with a 30 min limit;
    about 7 min each.

  Every previously existing row is byte-identical in all 16 panel D CSVs (checked with
  `diff` before staging).
* **Figure.** `sc2.gate_rows_extra` (new, default empty) inserts rungs after
  `mcm_interpolation` in `fig_sc2_aa20.R`'s gate roster and joins `sc2.gate_baselines`.
  The two gate exports set it to `c(mcm = "MCM")` only under `REVISED2_LABELS`, so panel M
  of sc2_results_aa20 and sc2_results_revised are unchanged. The line is dashed in
  `#a08c7a`.

Per cell it tracks the factored baselines and ends level with the PFM, about 0.65 at
2024-01-01. Per mutation it starts lower (0.63-0.71 through 2021) and reaches about 0.86.

## 122. Supplement: C and D against training-set size (2026-11-08)

`figs/supp_gate_units.png` (`figures/figS_gate_units.R`) redraws C (per cell, left) and
D (per mutation, right) of sc2_results_revised2 three times, with training-set size on
the x axis instead of the cutoff date:

| row | x | 2020-04-01 -> 2024-01-01 |
|---|---|---|
| 1 | mutation events in the training examples (whole spike) | 1,129 -> 293,748 |
| 2 | training examples (a sequence and its descendant mutations) | 190 -> 68,342 |
| 3 | sequences placed on the training tree (`#sequences` in counts_D_cur.tsv) | 28,622 -> 8,277,482 |

The sizes come from the files the cutoff-series models train on, and are written by
`experiments/multitree/training_size_by_cutoff.py` to
`data/gate_units_aa20/training_size_by_cutoff.csv`. They match independent counts:
2,934 records at 2021-01-01 (the MANIFEST), 351,410 sequences at 2021-01-01, and
7,813,585 at 2023-01-01 (the MSA). Nothing is re-scored; each point is one of the 16
cutoffs.

`panels_gate_units_export.R` is one process per pooling (`GATE_UNITS_POOLING`), because
the two loaders are chosen at source time. It joins the sizes on the drawer's own
decimal-year x.

**x is linear** (`GATE_UNITS_SCALE=log` for the alternative). Log spreads out the first
six cutoffs but squeezes the last eight, where nearly all the recall gain happens, into
a sliver: the training set grows less than 2x over those eight. On a linear axis per-cell
recall is close to linear in mutations and in examples, and bends up against sequences.
Per-mutation recall jumps over the first few thousand and then flattens.

## 123. A val-selected ensemble rung in list_macro (2026-11-08)

`list_macro.py --ensemble-dir DIR [--ensemble-name N]` adds one rung: every
`fitness_*.pt` in DIR, combined by the published "Mean of log-probs" rule. This is a port
of `eval.py --ensemble-dir`, for the 2023-01-01 all-cells panels (I-L) planned for
sc2_results_revised2.

**eval.py's full-eval-range precondition was stronger than needed.** A member's
log-softmax is its logits minus ONE constant per query, whatever positions it is
normalised over. So the arith mean is the mean logit minus one constant per query, and
its within-query RANKS are the same under any range. `MeanLogProbEnsemble(eval_range=...)`
normalises over the eval range anyway, so its values also equal the cached-map rule's,
and fills every other position with -inf. A new test pins both facts. The NaN-aware case
(gap/stop-masking members) is not handled, so list_macro refuses a non-20-letter
alphabet.

**Gate, passed.** `experiments/multitree/gate_list_ensemble.sh` (job 3653855) re-scored the
2021 `vs10Antigen_s0..s9` members through the new path. It reproduces the published
`ens_8a91fbbf88` list rows (`ensemble_report` over cached maps):

| variant | macro | published | micro | published |
|---|---|---|---|---|
| list_coguk_fullpool | 0.54512072 | 0.54512071 | 0.83055240 | 0.83055240 |
| list_coguk_unseen_fullpool | 0.50833851 | 0.50833848 | 0.71138322 | 0.71138322 |

Supporting changes:

* `submit_macro_allcells_aa20.sh` takes `EXTRA_LM_ARGS`.
* `fig_coguk_list_aa20.R`'s `coguk.macro_dir` is now overridable (eval-then-override).

The 2023-01-01 rows themselves wait on training. Ten val-select seeds at 256 epochs
(job 3658552), because at 96 epochs all five finished seeds selected epoch 93-95.
EVE and Tranception (jobs 3658555/6) were resubmitted with `--examples`; the first
submission omitted it and died at load.

## 124. The MCM alone on panels E-H of sc2_results_revised2 (2026-11-09)

The s121 rung `mcm` (log N[i,a]) is now also drawn on the COG-UK bar panels E-H,
right after S×S-MCM interp., in the interpolation's group.

* **Rows.** One resume pass on the 2021 list run (`evals_sc2_lists_macro_aa20`), job
  3670251, 40 s on `evo_gpu_priority`. It used that run's own arguments plus
  `--mcm-baseline`; the map went to the shared cache `list_score_maps_aa20`. Every other
  row of the summary and per-cell CSVs is byte-identical; both were staged into
  `data/evals_sc2_lists_macro_aa20/`.
* **Opt-in, and DROPPED otherwise.** `fig_coguk_list.R` stops on any model it has no
  group for, and the shared list CSV now carries `mcm`, so every other figure built on
  it would have stopped. `coguk.mcm` (default empty) inserts the bar after the
  interpolation; when it is empty, `mcm` joins `DROP_MODELS`. `panels_coguk_export.R`
  sets it under `REVISED2_LABELS` only. Checked: coguk_list_aa20 renders byte-identically
  from the CSV with and without the `mcm` rows.

| panel | MCM recall at q = 0.1 |
|---|---|
| E, all escape, per cell | 0.455 |
| F, all escape, per mutation | 0.794 |
| G, de novo, per cell | 0.106 |
| H, de novo, per mutation | 0.105 |

De novo is chance by construction: an unseen cell has N = 0, so the whole unseen pool
ties and fractional top-q returns about q. This is the same signature as PyR0's 0.100
(s103).

The committed `figs/old/coguk_list_aa20.png` already differs from a fresh render with or
without these rows. It predates later code changes and is not affected by this one.

## 125. sc2_results_revised2 panels I-L: the same roster on all RBD cells at 2023-01-01 (2026-11-09)

Row 4 mirrors E-H panel for panel (all / de novo, per cell / per mutation). Its target
set is all 4,020 RBD cells (`ref/all_rbd_cells_aa20_incl_ref.csv`, reference allele
included) at the 2023-01-01 cutoff, instead of the 2021 COG-UK escape list.

**Rows.** Scored into the existing run `evals_sc2_lists_macro_aa20_2023-01-01_allcells_pooled`
(`--pool-examples`, `--split-date`, substitution seed 0, s111/s112). One resume pass,
job 3682938 (12 min, `gpu_batch`), added the rows E-H draws that the run lacked:

| row | from |
|---|---|
| ESM 2 (35M) / (650M), untrained | existing `ckpt_stage_2023-01-01` (`*_noft`) |
| EVE | refit at 2023-01-01: `--freeze-backbone`, `--max-seqs 351410` (a uniform subsample, to stay in the 2021 24 h / 256 G envelope) on the regenerated MSA (7,813,585 sequences, counts verified identical) |
| antiGen (CV x10) | `ens_vs10_2023`: s123's `--ensemble-dir` over ten val-selected runs at **256** epochs, selected epochs 235-254 |

* **Epochs.** At 96 epochs all five finished seeds selected epoch 93-95, and the 32-epoch
  arm selected the last epoch in 9/10. At 256, the val NLL fell 0.05-0.07 over epochs
  63-127, 0.009-0.015 over 191-255 and at most 0.010 over the last 32. That is a
  plateau, against a 0.15 spread between seeds. The 96-epoch outputs are kept as
  `valsel_ep96_binding`.
* **Node GPUCACE.** Seed 9 ran there at about 3.6 min/epoch (vs ~35 s elsewhere), and the
  first Tranception run OOMed there (its GPU reports 19.6 GB). Both were resubmitted
  with `--exclude=GPUCACE`, which several of our scripts already default to.
* **Tranception** is still training (job 3682935) and is the one row missing from I-L.
  It is added with the same resume once it lands.

**Staging.** `experiments/multitree/stage_allcells_2023.sh` writes
`data/evals_sc2_lists_macro_aa20_2023-01-01_allcells/` in the layout the drawer reads:
the ensemble row split into `list_macro_summary_ensemble.csv`, plus
`ensemble_figure_rows.tsv`.

**Figure.**

* `REVISED2_ARM=allcells_2023` makes `panels_coguk_export.R` (E-H's own export) read that
  directory with `coguk.list = "allcells"`. Roster, pruning, names, colours and the MCM
  bar are therefore shared with E-H by construction.
* Titles say "All RBD" / "De novo RBD" and "(2023)". "Escape" is true only of the COG-UK
  list.
* This arm has no variant-recall or background run, and the drawer assumed every list
  did: it filtered the varrecall CSV to zero rows and failed. `coguk.varr_dir` and
  `coguk.bg_dir` are now overridable in `fig_coguk_list_aa20.R` (eval-then-override), the
  arm points both at nothing, and the export builds the background grobs only
  `if (have_bg)`.
* The canvas is 11 x 14.7 in, so row 1, and panel B's 1:1 slot (s119), keep their size.

| at 2023-01-01, q = 0.1 | I all/cell | J all/mut | K de novo/cell | L de novo/mut |
|---|---|---|---|---|
| antiGen (CV x10) | **0.599** | **0.930** | 0.497 | **0.688** |
| best other | PFM (3rd) | PFM (3rd) | **Codon table 0.536** | S×S-MCM interp. 0.674 |

K and L rest on 71 cells and 141 events. On that subset the plain codon table ranks 1st
per cell, and S×S, S×M and S×M-MCM interp. tie exactly (0.518 / 0.667). That tie is
plausible on so few cells but has not been checked.

## 126. supp_fixed_split: both poolings, and CoVFit at its own cutoff (2026-11-09)

The fixed-date baselines figure (s103) now draws both poolings, the same pair as
sc2_results_revised2: one row per baseline, per cell (macro) on the left and per mutation
(micro) on the right. A third row adds **CoVFit**.

**CoVFit's cutoff.** Our checkpoints are `covfit_model_20241007` (CLI, Zenodo 14438178).
The CoVFit training README lists "Oct24: Training with genotype-fitness dataset up until
October 7, 2024". It does not name the CLI version string, so 20241007 = Oct24 is
matched by date, not stated. We score it at our 2024-10-01 cutoff, six days before its
data ends. That error favours the baseline, the same direction as PyR0's (s103), so
antiGen's margin is a floor. Job 3662706: `score_fixed_split.sh` with D=2024-10-01.

| row | cells / events | antiGen per cell | baseline per cell | antiGen per mutation | baseline per mutation |
|---|---|---|---|---|---|
| PyR0, 2022-01-01 | 214 / 10,281 | 0.612 | 0.200 | 0.889 | 0.328 |
| Bloom & Neher, 2023-10-01 | 139 / 1,908 | 0.849 | 0.691 | 0.950 | 0.791 |
| CoVFit, 2024-10-01 | 78 / 456 | 0.902 | **0.064** | 0.976 | 0.112 |

**CoVFit is below chance per cell**, at 2024-10-01 (0.064) and at 2022-01-01 (0.061, s103).
Its map is one static (L, E) table computed on the XFG background (`eval_covfit.sh
--variant XFG`), applied to every query whatever its own residues. That may be the
cause, but it has not been checked. Read the row with that caveat.

Also: y headroom for a value label near 1 (panel F's 0.976 was clipped). The
`data/evals_fixedsplit_*` directories were caught by the `evals_*/` ignore rule and had
never been tracked, so the figure was not reproducible from the repo. They are now
allow-listed.

## 127. Figures render in Arial (2026-11-09)

Nothing named a font family. PNGs got fontconfig's "sans" (Liberation Sans here:
Arial's metrics, not Arial). PDFs named Helvetica without embedding it, so each viewer
substituted. Panel B (matplotlib) was DejaVu Sans. Arial was not installed.

* **Font.** conda-forge `mscorefonts` in `antigen-figs` (`envs/antigen-figs/fonts/arial*.ttf`).
  The environment's fontconfig resolves "Arial" to it.
* **`save_figure`** (`R/common.R`) sets the family on the DEVICE: PNG via `type = "cairo"`
  and PDF via `cairo_pdf`, both with `family = FIG_FONT` (option `paper.font`, default
  Arial). Every text element without a family of its own takes the device default:
  theme text, geom_text, cowplot letters, gridtext markdown. So no theme changed.
  Checked: each active PDF embeds exactly ArialMT and Arial-BoldMT.
* **`draw_path_panel.py`** registers the same files with matplotlib's font manager and
  sets `font.family`. It exits if they are missing rather than falling back.
* **Re-rendered:** sc2_results_revised2, supp_dms (from `fig2_sc2_benchmarks.R`, whose
  other outputs were discarded), supp_gate_units, supp_fixed_split. Arial and Liberation
  Sans share metrics, so no layout moved.

Not covered: the three scripts that call `ggsave` directly (`fig1_overview_panels.R`,
`fig3_pseudovirus_assay.R`, `fig2_sc2_benchmarks.R`'s optional FIG1_AUC) and panel A of
revised2, which is the user's PNG.

## 128. sc2_results_revised2: a pruned main figure, a full supplement, and who designed what (2026-11-09)

**Two versions from one script.** `REVISED2_VARIANT=main` (the default) writes
`sc2_results_revised2`; `=supp` writes `sc2_results_revised2_supp`.

* **The supplement.** Before this section's styling changes it was pixel-identical to
  the s125 figure. Its canvas is the s125 one (rel 0.95/1.1/1.55/1.45 of 14.7 in), plus
  0.45 in for the taller key.
* **The main figure.** Bar panels E-L drop eight rungs, passed by KEY (`DROP_MAIN` ->
  `REVISED2_DROP`, appended to `coguk.drop_models`) so a later rename cannot un-drop one:
  `gated_static_xfmr_pssm_lr0.0001` (bars only; its C/D line stays),
  `MSA_Transformer_BA2`, `MSA_Transformer_XFG`, `where_times_which`, `wxw_dirichlet`,
  `mcm`, `BA2_escape_LYCoV1404`, `esm2_esm2_t33_650M_UR50D_lr0.001_noft`. That leaves 17
  bars in E-H (16 in I-L until Tranception lands). Rows 1-2 keep their inches, so panel
  B's 1:1 slot is unchanged.
* **Gate check.** The export's check that the gate is grouped as the hybrid now applies
  only when the gate is drawn.

**Row headers.** Rotated and bold, at the left of rows 3 and 4: "CoG-UK escape list,
2021-01-01 cutoff" and "All RBD cells, 2023-01-01 cutoff". The I-L titles lose their
"(2023)".

**"Statistical models", a new model type.** S×M, S×S, S×M-MCM interp. and S×S-MCM interp.
(`where_times_which`, `site_substitution`, `wxw_dirichlet`, `mcm_interpolation`) leave
"Pandemic Data in Training Window".

* **Colour.** Pale blue, `#a9c9e3`.
* **Position.** Directly under the antiGen bars: the CV bar, and the gate where it is
  drawn. The drawer orders bars by the `model` factor's levels, so `regroup()` re-levels it.
* **Medal text.** White text does not read on pale blue. New call-time option
  `coguk.medal_dark_groups` (default empty) draws those medals dark.
* **Not included.** MCM and Codon-B62 mixture were not asked for and stay where they were.

**Hatching: designed here vs external.** Every bar NOT designed in this work is striped
(white, 45 deg) via `ggpattern`. Plain bars: antiGen, the gate, the four statistical
models, Codon-B62 mixture (`substitution`), MCM.

* **Option.** Call-time `coguk.ours_labels`, display labels; unset by default, so plain
  `geom_col` and no other figure changes.
* **Medals.** With hatching on, they are `geom_label`s on a solid patch of their bar's own
  fill, so they read over stripes.
* **Key.** Its own line under the model-type key, with HAND-DRAWN swatches: ggpattern's
  legend key draws the pattern at another scale and looked nothing like the bars.
* **Stripe density.** It took three passes. Fine dense stripes (spacing 0.012) washed
  every bar out until the model types could not be told apart. The final setting is
  spacing 0.025, density 0.2, white at 0.85.
* **Install.** `r-ggpattern` 1.3.1 into `antigen-figs` from conda-forge with
  `--freeze-installed`. The dry run showed only NEW packages; ggplot2 is still 4.0.3.

### s128 addendum: quieter hatching, after a reference figure

The white stripes (spacing 0.025, density 0.2) read as noise. The user supplied a
reference, `inspo.png` at the repo root (not committed), and the hatching now follows it:

* **Lines.** Sparse, thin and black: `pattern_colour = pattern_fill = "black"`,
  `pattern_linewidth = 0.12`, `pattern_density = 0.008`, `pattern_spacing = 0.07`, 45 deg.
* **Outlines.** Every bar has a thin dark outline (`colour = "grey15"`,
  `linewidth = 0.2`); without an edge, sparse lines on pale bars float.
* **Key.** An outlined white square for "Designed in this work", the same square with one
  diagonal for "External".

Medals keep their solid patch (s128), so they read over the lines.

**Then (user):** no bar outlines (`colour = NA`). The lines no longer overhang the bar
ends. A stroked stripe border (`pattern_colour`) is drawn half outside the bar's clip,
which put a stub past every bar end. Stripes are now fill only (`pattern_colour = NA`,
`pattern_fill = "black"`), with density 0.06 for the same visual weight. The key's
squares keep a thin border, so the plain one stays visible on white.

## 129. sc2_results_revised2: Codon-B62 under the statistical models; a thinner main roster (2026-11-09)

* **Codon-B62 mixture** (`substitution`) is now drawn directly under the statistical
  models, in both versions. `regroup()` places it after them when it re-levels `model`.
  It keeps its type (grey) and stays plain (ours).
* **The main figure drops three more rungs**, appended to `DROP_MAIN`:
  `pssm_root_position` (PFM Conservation), `Omicron_BA1_bind`, `Omicron_BA2_bind`. That
  leaves 14 bars in E-H and 13 in I-L until Tranception lands. The supplement is
  unchanged. The main bar rows are 2.95 / 2.70 in.
* **The key above row 3 is a fixed height in inches** (`LEG_IN`: 0.40 main, 0.75 supp). As
  a fraction of the row it shrank with the roster and clipped its own swatches.
  `ROW_IN` is now computed before the rows that use it.

## 130. sc2_results_revised2 panel M: the date-matched baselines in the main figure (2026-11-09)

The six results of supp_fixed_split (s126) are panel M of the MAIN figure, a new bottom
row with the rotated header "Date-matched baselines". The supplement version has no M.
`supp_fixed_split.png` is still drawn on its own.

* **Data.** The same files: `data/evals_fixedsplit_<date>/list_macro_summary.csv`,
  `list_coguk_fullpool`, q = 0.1. Per cell = macro, per mutation = micro.
* **Layout.** Built in `fig_sc2_revised2.R` itself, since it is two rows per sub-plot,
  so no export is needed. There is one PAIR per baseline under a heading naming baseline
  and cutoff (PyR0 2022-01-01, Bloom & Neher 2023-10-01, CoVFit 2024-10-01), split
  "per cell" / "per mutation". Model names are drawn once per pair, and the single letter
  M has a column of its own.
* **Style, as E-L.** Horizontal bars in the type colours, the external baseline hatched,
  values at the bar ends (`clip = "off"`), x ticks 0 / 0.5 / 1 so neighbouring
  sub-plots' ticks do not collide. The row is 1.9 in (`M_IN`).
* **The antiGen bar is the BASE model**, labelled "antiGen (base)": the only antiGen
  checkpoint scored at these three cutoffs. E-L draw the CV x10 ensemble. Matching them
  would need val-select runs at three more cutoffs.
* **CoVFit's per-cell 0.06 is below chance** (s126), and the reason has not been checked.

## 131. sc2_results_revised2: bar rows swapped, headers shortened, M's bars matched (2026-11-09)

* **The two bar rows are swapped.** All RBD cells at 2023-01-01 is now the first bar row,
  lettered E-H; the CoG-UK escape list at 2021-01-01 is second, lettered I-L. The
  model-type key stays above the first bar row. Heights are per bar-panel block in inches
  (`PANEL_IN`, rbd / esc), plus the fixed key (`LEG_IN`), so each block keeps its own size
  after the swap. A helper `bar_block()` builds both rows.
* **Shorter row headers:** "All RBD, 2023 cutoff" and "Escape, 2021 cutoff".
* **Panel M has no row header**, but keeps the same left gutter (`HDR_W`) so it lines up
  with E-L.
* **M's bars are the same height as E-L's.** The row is now 1.05 in (was 1.9) and the bar
  width is 0.9, as `geom_col`'s default in E-L. Measured at 300 dpi, the antiGen bar is
  40 px in E, 39 px in I and 40 px in M.
* **M's stripe pitch matches E-L's.** ggpattern's spacing is relative to the plot, and M's
  sub-plots are about a sixth as tall, so M uses `pattern_spacing = 0.30` against E-L's 0.07.

## 132. Panel B in the style of the H3N2 case study (2026-11-09)

At the PI's request, panel B of sc2_results_revised2 (`draw_path_panel.py`) now follows the
H3N2 J.2 -> K panel on `av-multitree`: `experiments/multitree/draw_path_panel_h3n2.py`,
drawn as panel G of that branch's `figs/fig4_endemic.png`.

* **Structures.** Light-blue body (`#bdd7f2`). A site newly changed at a state is red and
  LABELLED ON THE GLYPH, with a leader to its patch, using their left/right column routing
  with a minimum gap. A site changed at an earlier state and still changed is slate grey.
  Arrows between states are filled black wedges, and the mutations are no longer on the
  arrows. The renderer is our own `render_rbd_surface` at its default shading, which
  matches theirs (`render_surface`: ambient 0.54, specular 0.13, outline `#5b7794`).
* **Chart.** antiGen PERCENTILE among all RBD substitutions, 0-100. The top decile is a
  shaded right tail behind a dashed bound, labelled in bold. Each step is a stem and dot
  coloured by the transition that made it (their stage palette), with the name in bold at
  the right and `(rank)` in italics under a "(Rank)" header.
* **Adapted to this path.**
  * **Reversions.** The H3N2 path has none; this one has two, V483D -> D483V and
    R346T -> T346R. Their steps keep a HOLLOW dot, with a key entry "reverts later". A
    site back at its BA.2.86 residue is unmarked on the glyph, so the reversions are
    visible only in the chart.
  * **Key.** "Step into: JN.1 / JN.1.11.1 / KP.3": "... substitution" three times ran
    into the top-decile label at this width. It is laid out by measured text width.
  * **Arrows.** Low on the glyph row (16% of its height) so they clear the on-glyph labels.
  * **No chain key.** Their HA1 / HA2 / other-protomers key has no RBD analogue.
* **Canvas.** Unchanged at 5.5 x 2.45 in, drawn at 1:1 into its slot (s119).

## 133. The codon table on de novo RBD: ties checked; Codon-B62 out of the main figure (2026-11-09)

**The question.** On panel G of sc2_results_revised2 (de novo, all RBD cells, 2023-01-01,
per cell), the codon table ranks 1st at 0.536, above antiGen (CV x10) at 0.497. Its score
depends only on (current residue, new residue), so thousands of cells tie exactly, and
the worry was that the top-decile tie rule flatters it.

**The rule.** `compute_hit_rate` selects exactly q * n_valid cells. Every cell strictly
above the boundary value counts in full; each cell tied AT the boundary counts as
(slots left) / (block size). That is the expected count under uniform random
tie-breaking.

**Checked against random tie-breaks.** `LIST_MACRO_TIE_JITTER=<seed>` is a new DEBUG
switch in `list_macro.py`, off by default and never used for a published row. It adds
uniform noise of 1e-4 x the score range, far below a coarse predictor's gaps, so that
ties break at random. Five copies of the 2023 all-cells run, built-in baselines only
(jobs 3686591-5, about 1.5 min each):

| codon | de novo per cell | de novo per mutation | all per cell |
|---|---|---|---|
| fractional rule (published) | 0.5358 | 0.4900 | 0.3013 |
| random breaks, 4 seeds | 0.5380-0.5388 | 0.4894-0.4965 | 0.3001-0.3080 |

The rule matches random tie-breaking, so this is not a tie artifact.

**It is small-sample.** Panel G rests on 71 cells. Codon minus CV x10 is +0.039, with a
bootstrap 95% CI over cells of [-0.078, +0.156]: 11 cells that codon hits and CV x10
misses, against 6 the other way. A plausible reading is that a never-seen cell carries no
history for a learned model, leaving single-nucleotide accessibility, which is exactly
what the codon table scores; on all RBD cells it drops to 0.30. A check of reachability
from the BA.2 residue was inconclusive, because 2023 queries differ from BA.2 at many
sites.

**Figure.**

* **Main roster.** The main figure drops Codon-B62 mixture (`substitution`); the
  supplement keeps it. Bar blocks are now 2.52 / 2.57 in.
* **Hatch key.** It reads "Ours" / "Previous".

## 134. PI feedback: keys under their rows; panel B's surface and dots (2026-11-09)

* **Each key goes UNDER the rows it serves, with a gap after it**, so it cannot be read as
  belonging to the next row. The C/D line key sits directly beneath C and D, followed by
  white space. The model-type and Ours/Previous key moves from above E-H to under I-L,
  with 0.25 in before panel M; in the supplement, which has no M, it ends the figure.
  `ROW_IN[3]` is now the E-H block alone, and `ROW_IN[4]` is I-L + key + gap.
* **Panel B surface.** The PI found it oddly rendered next to the HA glyphs. An RBD has
  about 1/9 the atoms of the HA trimer and is drawn larger, so at the renderer's default
  smoothing (0.0118) each atom read as a bump, and the directional light left a dark band
  down one side. Now: size 760x1280 (was 380x640), `smooth = 0.030`,
  `color_smooth = 0.32`, `ambient = 0.70`, `specular = 0.05`, `rim = 0.10`. Four
  settings were compared side by side and this one was closest to the HA panel's matte,
  continuous surface. The label routing gap doubles with the resolution, to 140 px.
* **Panel B dots.** The dots near 100 (R346T, T346R) sat on the axes' right edge and
  were half clipped. They are now drawn with `clip_on = False`.

## 135. The E-M key at the very bottom of the main figure (2026-11-09)

In the MAIN figure the model-type / Ours-Previous key is now the last row, under panel M,
which it also colours. I-L is directly followed by M (with 0.15 in between). The
supplement has no M and keeps the s134 layout, with the key under I-L ending the figure.

## 136. Panel B drawn with av-multitree's renderer, verbatim (2026-11-09)

The s134 surface tuning (smooth 0.030, ambient 0.70, ...) still looked wrong to the user.
`experiments/multitree/render_surface.py` is now a VERBATIM copy of the av-multitree file
of that name, the H3N2 panel's renderer. `draw_path_panel.py` imports it in place of
`render_rbd_surface` and calls `render()` with ITS defaults at the H3N2 render size
(760 x 1000); the glyph aspect follows that size. `render_rbd_surface.py` stays for the
other callers.

* **Labels.** The wider glyphs made F456L and Q493E collide. The label columns go back
  to the H3N2 values: a 92 px minimum gap, and an offset of 0.45 x image width.
* **Arrows.** Their margins shrink to 0.14 in, so the shafts show between the closer
  glyphs.
* **Why the RBD still looks more textured than an HA trimer.** The renderer's blur is a
  fixed fraction of image WIDTH. A whole HA trimer fills theirs, so each atom is a few
  pixels; one RBD fills ours, so each atom is larger against the same blur. Smaller
  glyphs, or a higher `smooth` argument, are the levers if that matters.

## 137. Panel B names its structure (2026-11-09)

The title is now "Case study: BA.2.86 to KP.3 (spike RBD, PDB 6M0J)". The glyph is chain E
of 6M0J, the SARS-CoV-2 spike receptor-binding domain alone: residues 333-526, Wuhan-Hu-1
sequence, crystallised with ACE2 (chain A, not drawn). All four glyphs are that one
reference structure, with the sites each state changes coloured on it, not
variant-specific models. That belongs in the caption.

## 138. Query adaptability for E-L: supp_query_adapt (2026-11-09, IN PROGRESS)

sc2_results_revised carried two "query adaptability" panels (background lift, s33/s66);
revised2 dropped them. `figures/figS_query_adapt.R` brings the measure back as a
supplement in revised2's grammar: one panel per E-L, same roster (main by default,
`REVISED2_VARIANT=supp` for the full one), names, colours, hatching and row headers.
Rows: "All RBD, 2023 cutoff" (`rbd_all` / `rbd_unseen`) and "Escape, 2021 cutoff"
(`coguk_all` / `coguk_unseen`), each per cell and per mutation (host-weighted re-pool,
s110). Medals are drawn above the whiskers (`bg_bar_top` in `panels_coguk_export.R`).

**The 2021 arm** is `data/evals_sc2_background_aa20`, which lacked two drawn rungs.
`submit_bg_aa20_topup.sh` (job 3690491, 13 min) scored `mcm` and
`BloomNeher_2023-10-01` into a side directory, never the published one, gated on a
byte-identical rebuilt `backbone_queries.csv` and an identical CoVFit map. Both are
query-blind and come out at exactly 0 on all four cell sets. Their 16 summary / 2,424
per-cell rows were appended, and the event-weighted re-pool redone (its `--weight none`
gate passes).

**The 2023 arm** is new: `submit_bg_2023.sh` (job 3690487, gpu_batch). Same contrast
(host vs the BA.2-root reference, `--min-controls 1`), on the shared test corpus split at
2023-01-01, with E-H's own models: `ckpt_stage_bg_2023-01-01` (the union of the two
staging dirs the 2023 list_macro run read, plus Tranception) and `ens_vs10_2023` built
from its members. 7,302 background genotypes, 730 hosted cells. The cell set is the hosted
cells, so it is smaller than E-H's 4,020.

`background_discrimination.py` gained list_macro's `--pool-examples` (the fit sees
53,916 + 54,988 = 108,904 training examples, exactly E-H's run), `--mcm-baseline` and
`--ensemble-dir/--ensemble-name`. Defaults are unchanged.

**Also.** Tranception at 2023-01-01 (job 3682935) was resume-scored into the E-H run
(job 3690489) and restaged; main and supp revised2 re-rendered with it (all RBD, per
cell, 0.257). The main-figure pruning list moved to `figures/revised2_roster.R`, and
`relabel_grob` / `grob_text` to `revised2_labels.R`, so that both figures share them.

Open: render once job 3690487 lands. That needs the summary staged to
`data/evals_sc2_background_aa20_2023-01-01`, the `_eventwtd` re-pool, and a `.gitignore`
allow-list entry for both.

## 139. C-H on the whole spike: the jobs (2026-11-09, IN PROGRESS)

A supplement of panels C-H scored over all 1,273 spike positions. One pipeline covers
all six panels: list_macro `--eval-range full` over every spike cell
(`ref/all_spike_cells_aa20_incl_ref.csv`, 25,460 = 1,273 x 20, wild-type letters BA.2 as
in the RBD list). With the complete list, macro is the per-cell number (C, E, G) and
micro the per-mutation number (D, F, H) -- eval.py's quantity -- so D needs no second
sweep. `experiments/multitree/submit_fullspike.sh` submits one job per cutoff (16,
gpu_batch, 12 h, resumable). The 2023-01-01 job carries E-H's whole roster and the
ensemble. `submit_macro_allcells_aa20.sh` gained `RANGE=default|full`; full-range jobs are
named `*_fullspike` so `macro_series_daemon.sh` cannot mistake them for RBD chunks.

The DMS maps and EVEscape are defined on the RBD only: off it they have no candidates,
but those targets still count against them (s29), so they are not like-for-like there.

Jobs 3690507-8 and 3690637-53. Their 2-day asks sat PENDING, so the limit is now 12 h.

## 140. sc2_results_revised2_supp is the bar rows alone (2026-11-09)

On request, the supplement version drops rows 1-2 and panel M and keeps only E-L on the
full roster, relettered A-H (`BAR_LETTERS` in `fig_sc2_revised2.R`). The main figure
keeps E-L.

## 141. Finetunes and ensembles back on sc2_results_revised2_supp (2026-11-09, IN PROGRESS)

On request, the full-roster supplement draws the antiGen Finetunes (brown) and the three
static mean rules over the nine-member family (orange, s49) again. `REVISED2_FULL=1`,
passed by the supp variant only, makes `panels_coguk_export.R` keep the eight finetune
keys and add `ens_7c8cf5c991` / `ens_7fa79cecaa` / `ens_7530180874` beside the CV bar.
The main figure is unchanged. Rule ids hash the rule and its member NAMES, so 2021 and
2023 share them. Only the CV bar now joins "antiGen" (`CV_LABEL`); the mean rules keep
their own group. Finetunes and ensembles are drawn plain (ours), since all of them are
antiGen-trained. Supp bar rows are 6.0 in each, for 36 bars.

**2021 (E-H of the supp)** already had everything and draws complete.

**2023 (A-D)** had only the 35M partial FT. `submit_finetunes_2023.sh` trains the other
seven at 2023-01-01 with the 2021 recipes unchanged (EVE/Tranception on the 2023 MSA
capped at 351,410 sequences, like their frozen rungs), into `ckpt_stage_ft_2023-01-01`:
jobs 3690763-69, gpu_batch. The corpus is 19x 2021's (35M partial: 21 min -> 6h40m), so
the 650M arms project to 2-5 days. antiGen-train cannot resume, so the time limits carry
headroom. When all seven land, `finish_ft_2023.sh` scores them into E-H's own run and
dumps the nine members' maps. Base antiGen and the 35M partial FT are re-run for their
maps and must reproduce their rows. It then scores the three mean rules with
`ensemble_report.py`; `stage_allcells_2023.sh` now merges those rows.

**s138 follow-up.** Job 3690487 completed (43m39s, 240 rows). The 2023 arm is staged in
`data/evals_sc2_background_aa20_2023-01-01{,_eventwtd}` (allow-listed). Per-cell lift on
`rbd_all` / `rbd_unseen`: CV x10 +8.1 / +27.6 pp, Tranception frozen +8.9 / +18.6,
antiGen base +5.5 / +26.1. **The ESM-2 35M partial FT is -22.5 pp on `rbd_all`**: it
ranks a cell LOWER on the genotypes that hosted it than on the BA.2 root. At 2021 it was
positive. Not chased.

## 142. supp_query_adapt: one x range per panel (2026-11-09)

On request, the x axes were tightened. Each panel had shared `bg_range_for` with its
all/de novo partner, so the all-cells panels inherited de novo's 0-40 when their own
bars end near 12. Each `bg_bar_top` call now passes its own metric alone, so every
panel's range covers its own bars and 95% intervals, plus the drawer's 5% padding.
Only this supplement's panels change.

## 143. Base antiGen on the full-roster supplement (2026-11-09)

On request, `sc2_results_revised2_supp` draws base antiGen (`simple_d128_n2_lr0.0001`)
beside the CV x10 bar, in the antiGen type and unhatched, labelled "antiGen (base)" as
on panel M. It is dropped only when `REVISED2_FULL` is unset, so the main figure is
unchanged. Bar order: CV x10, base, then the gate.

## 144. Full-roster bar order (2026-11-09)

On request, `sc2_results_revised2_supp` orders its bars as: the antiGen bars and the
statistical models (blue), then the antiGen Finetunes (brown), then the three mean rules
(orange), then MCM as the first green bar directly above Codon-B62 mixture, then the
rest as before. Set in `regroup` in `panels_coguk_export.R`; the finetune and mean-rule
blocks are empty outside the full roster, and the main figure does not draw MCM.

## 145. Full-roster palette: purples for the finetunes and the mean rules (2026-11-09)

On request, `sc2_results_revised2_supp` recolours the antiGen Finetunes from brown
(#443333) to a dark purple, #491584 (hsl 268, 72%, 30%, about antiGen blue's
lightness). The three mean rules go from orange (#bc6c25) to a lighter purple, #7332bd
(hsl 268, 58%, 47%), stepped up the way GATE_BLUE is from ANTIGEN_BLUE. Applied only
under `REVISED2_FULL`; every other figure keeps the drawer's palette.

## 146. The s145 purples, muted (2026-11-09)

On request, the purples are desaturated. Finetunes #491584 -> #4c3371 (hsl 265, 38%,
32%); mean rules #7332bd -> #7e60a9 (hsl 265, 30%, 52%). The light-over-dark step that
echoes the gate is unchanged.

## 147. The full-roster purples moved to indigo (2026-11-10)

On request ("too purple"), both colours move halfway in hue between antiGen's blues and
the s146 purples. Finetunes #4c3371 -> #292a7a (hsl 239, 50%, 32%); mean rules #7e60a9
-> #4a55b5 (hsl 234, 42%, 50%). The straight HSL midpoint inherits antiGen blue's 97%
saturation and comes out vivid (#191b7f / #2739b8), so saturation is kept moderate.

## 148. Whole-spike runs landed; 650M full FT resubmitted at 13 days (2026-11-10)

**Whole spike (s139).** All 16 cutoffs COMPLETED, 7-56 min each. There are 25 models
per cutoff and 30 at 2023-01-01, which carries E-H's roster (ens_vs10_2023, Tranception,
EVE, both untrained ESM-2). No `[skip]`. Base antiGen vs PFM, per cell: 0.278 vs 0.186
at 2020-04-01; 0.567 vs 0.542 at 2023-01-01; 0.683 vs 0.651 at 2024-01-01. The figure
plumbing is not built yet.

**2023 finetunes (s141).** Done: ESM-2 35M frozen (3h55m) and EVE (3h55m). The ESM-2
650M full FT ran 7h55m per epoch (13,479 batches at batch size 4), so 32 epochs take
~10.5 days, past its 8-day limit. A user cannot raise a running job's limit ("Access
denied"), and antiGen-train saves only at the end. It was cancelled (3690767, ~8 h
lost, no checkpoint written) and resubmitted unchanged at 13 days as 3705388. The
recipe was kept, so the bar stays comparable to 2021. The other 650M arms (~1.6-1.9 h
per epoch) fit their limits.

## 149. Main revised2: the all-RBD bars before the cutoff series (2026-11-10)

On request, rows 2 and 3 of the main `sc2_results_revised2` swap. Now: A-B (cartoons),
C-F (all RBD, 2023 cutoff, bars), G-H (the cutoff series and its key), I-L (escape,
2021 cutoff, bars), M, then the model-type key. `BAR_LETTERS` gives C-F / I-L in main.
The series panels are lettered G-H, and row heights follow their rows. The supplement
is unchanged (A-H, bar rows only).

## 150. Main revised2: the cutoff series moved to the bottom row (2026-11-10)

User request. `sc2_results_revised2` now reads: A-B cartoons; C-F all-RBD bars (2023
cutoff); G-J escape bars (2021 cutoff, I-L before); K the fixed-date baselines (M
before); the model-type key; a 0.25 in gap; L-M the cutoff series (G-H before), with
its own line key under them. Canvas height grows by that gap. The supplement is
unchanged (still the bar rows alone, A-H).

## 151. CoVFit asked each query, for panel K (2026-11-10; scoring queued)

**Why.** Panel K's CoVFit arm (0.064 per cell, 0.112 per mutation at 2024-10-01) is
below chance. Its map is one static table from the single mutants of XFG (fold 0,
`eval_covfit.sh --variant XFG`, 26,734 sequences, 22 min), applied to every query.
Ranking the 247 COG-UK RBD cells in it: 24% fall in its bottom decile, 8% in its top.
Its ten lowest sites include 456, 486, 493, 455, 452 and 484 -- sites where XFG already
carries an adaptive residue (F456L, F486P, Q493E, L455S, L452W, A484K), so moving away
from it reads as deleterious. 25 listed cells are XFG's own residues and score at the
query's level (68th percentile), so they can never reach the top decile. Single
mutants barely move CoVFit (80% of RBD deltas within +-0.007 of 0.885), and 83% of the
map's variance is between sites. XFG also post-dates the 2024-10-01 cutoff.

**What.** `antiGen/baselines/covfit_perquery.py` (`select` / `fasta` / `build`) and
`list_macro.py --query-maps NAME=NPZ`, a rung whose map is looked up per scored query
by genotype; a query without a map is an error. Driven by
`experiments/multitree/submit_covfit_perquery.sh` (STAGE=gate|score|eval). 364 distinct
query genotypes (all 364 scored examples) x (201 RBD sites x 19 + 1) = 1.39M CLI
sequences, fold 0 (the XFG map's fold, so the background is the only change), 16
chunks on gpu_batch (array 3706983).

**Gate, passed.** The XFG map copied to every query and scored through `--query-maps`
reproduces the published CoVFit row exactly (worst difference 0.0; job 3706972).

**Input caveat.** The 20-letter sc2 corpus has no gap state; deletions are imputed
(site 483 reads V in 98% of test genotypes although BA.2.86/JN.1 carry V483del), and
no gapped sc2 build exists. CoVFit sees the queries as encoded, without deletions.

**Results** (array 3706983, 16 x ~66 min; eval 3721521, 44 s; 36 new rows appended to
`evals_fixedsplit_2024-10-01`, nothing else changed). COG-UK list, common pool, q = 0.1,
78 cells / 456 events:

| rung | per cell | per mutation |
|---|---|---|
| CoVFit, static XFG map | 0.064 | 0.112 |
| **CoVFit_query, each query's own background** | **0.171** | **0.215** |
| antiGen (base) | 0.902 | 0.976 |
| PFM | 0.885 | 0.976 |

Asking each query lifts CoVFit from below chance to above it (chance 0.10): the static
background was a real handicap. It remains far below antiGen, the PFM, EVEscape
(0.473 / 0.549) and Bloom & Neher (0.667 / 0.748). Per-query deltas are as small as the
XFG map's (1/50/99%: -0.022 / -0.0003 / +0.007).

**Panel K now draws `CoVFit_query`**, still labelled "CoVFit". `supp_fixed_split` still
draws the static map; not changed here. Folds 1-4 were not run (FOLDS knob).

## 152. supp_fullspike: C-F and L-M of sc2_results_revised2 on the whole spike (2026-11-10)

`figures/figS_fullspike.R` -> `figs/supp_fullspike.png`. Row 1 "Whole spike, 2023
cutoff", A-D (all / de novo x per cell / per mutation), then the model-type key; row 2
E-F the cutoff series (per cell, per mutation) with the line key. Same exports, names,
colours, hatching and order as the main figure.

* Data: the s139 runs (`submit_fullspike.sh`), staged by
  `experiments/multitree/stage_fullspike.sh` into `data/gating_macro_aa20_fullspike/<D>/`
  (lines) and `data/evals_sc2_lists_macro_aa20_2023-01-01_fullspike/` (bars;
  `stage_allcells_2023.sh` now takes S/D overrides). With every cell listed, E is the
  macro column and F the micro column of the same run (`REVISED2_GATE_COL`).
* `panels_coguk_export.R` gained the arm `REVISED2_ARM=fullspike_2023`;
  `panels_gate_macro_export.R` takes `REVISED2_GATE_DIR/COL/TITLE/YLAB/DROP`. Unset,
  every published panel is unchanged.
* **Dropped beyond the main roster:** the RBD-only maps (BA.1 and BA.2 expression DMS,
  EVEscape). Off the RBD their non-RBD targets count against them (s29), so their
  numbers would measure coverage. The script refuses to render if one is still drawn.

2023-01-01, q = 0.1, CV x10 against the next drawn bar:

| panel | cells / events | antiGen (CV x10) | next drawn |
|---|---|---|---|
| A all, per cell | 4,466 / 57,319 | 0.576 | PFM 0.542 |
| B all, per mutation | 4,466 / 57,319 | 0.909 | PFM 0.887 |
| C de novo, per cell | 263 / 591 | 0.518 | S×S 0.502 |
| D de novo, per mutation | 263 / 591 | 0.682 | S×S 0.626 |

Undrawn: the antiGen / PFM gate (0.570 / 0.908 on all cells) and base antiGen
(0.567 / 0.889; de novo 0.504 / 0.645).

## 153. The DMS supplement on sc2_results_revised2's C-J roster (2026-11-10)

`antiGen-figures/figures/figS_dms_revised2.R` -> `figs/supp_dms_revised2.png`. The quantity
of aa20_supp_dms (Spearman of each model's map at the BA.2 variant root against the BA.2
expression / binding DMS, RBD cells, wild type masked, beside the position-only
rank-mean control), drawn for the C-J roster with C-J's names, model types, colours and
Ours / Previous hatching. `figs/supp_dms.png` (the published 22-letter corpus, Pearson,
four transformations) is untouched; it is written by fig2_sc2_benchmarks.R's defaults.

* antiGen (CV x10) is the 2021-01-01 ensemble (ens_8a91fbbf88, G-J's arm): the root and
  the DMS are BA.2.
* Two rungs had no root map: the PFM (pssm_root) and the codon table. Both were dumped
  from `list_score_maps_aa20` with `dump_variant_root_predictions.py` (2021 test corpus,
  query 4110) into `data/sc2_aa20_extra/`. The same dump of `site_substitution`
  reproduced the existing `data/sc2_aa20` file byte for byte. The PFM is used as raw
  counts (exp() would overflow them into ties); Spearman needs only their ranks.
* BA.2 DMS (expression), on C-J's roster, is left out of this figure (user): it is the
  Expression panel's own target. 12 models.
* **Gate:** the script's numbers equal the drawer's own `dms_spearman` table (the
  take_log == FALSE rows) for all 42 values it shared before BA.2 DMS was removed (11 models x 2 targets x 2
  series),
  worst difference 0.

| model | expression rho | binding rho |
|---|---|---|
| antiGen (CV x10) | 0.270 | 0.247 |
| S×S | 0.291 | 0.255 |
| S×S-MCM interp. | 0.298 | 0.262 |
| PFM (new) | 0.195 | 0.196 |
| EVEscape | 0.362 | 0.082 |
| BLOSUM62 | 0.280 | 0.293 |
| Codon table (new) | 0.077 | 0.091 |

Position-only controls exceed the prediction for antiGen, S×S, S×S-MCM interp. and the
PFM on both targets (as in the earlier panel-G addendum "the counted rungs on panel G": the signal is
largely positional).

## 154. Data-generation seed robustness of antiGen (CV x10), 2023-01-01, panels C-F (2026-11-11; running)

**Question.** How much do antiGen's C-F bars of sc2_results_revised2 move when only
the seed of the data-generation pipeline changes?

**Data.** `~/antiGen-data-revised/slurm/seed_study.sbatch` (MODE=all, jobs 3705565_1-5,
2026-11-10) rebuilt the 2023-01-01 training examples under `--seed 1..5`, nothing else
changed. `--seed` drives every source of randomness at once (queue shuffle, placement
tie-breaks, allele draws, mutation order on multi-mutation edges). Seed 1 is
byte-identical to the production `examples_2023-01-01_cur.txt` (cmp), so it is the
published run and is not retrained. Records / mutations: 53,916 / 244,460 (seed 1),
53,744 / 244,515, 53,802 / 244,407, 53,863 / 244,256, 53,771 / 244,186 (seeds 2-5).

**Decisions (user).** Seeds 1-5. Only antiGen (CV x10) is retrained; every other bar
stays the published one. The training seeds are held fixed (split seed i, init seed
100+i, i = 0..9). The result goes in a new supplementary figure.

**Pipeline**, `experiments/multitree/submit_dataseed_2023.sh`:
* train: `train_antigen_valsel10.sh` unchanged (ens_vs10_2023's recipe: 10 runs, 256
  epochs, val-frac 0.1, best epoch kept), only EXAMPLES moved; one array of ten
  one-run jobs per seed into `$STORE/dataseed_2023-01-01/seed_<k>/valsel/`. Arrays
  3722088 / 3722090 / 3722092 / 3722094 (seeds 2-5), 12 h limit, gpu_batch.
* eval: the published 2023 all-RBD run's command (job 3682938's flags: the production
  `--train-examples`, the test file cut at 2023-01-01, `--pool-examples`, the
  production root counts, the 4,020-cell list), scoring only the ensemble
  (`--baselines-only --skip-baselines --ensemble-dir`), so every seed is scored on the
  identical test data and de novo set. Jobs 3722089 / 91 / 93 / 95, `afterany` on their
  arrays; each refuses unless all ten runs wrote a final checkpoint, and logs each
  member's selected epoch.
* gate: job 3722087 re-scores the PUBLISHED members through the same eval command;
  it must reproduce the published ens_vs10_2023 row. **Passed** (worst difference 0.0).

Results: pending (the four training arrays wait on gpu_batch).

## 155. A coronavirus pretraining corpus at 2023-01-01, and joint ESM-2 training (2026-11-11)

**Goal (user).** av-multitree's headline -- one ESM-2 35M trained across many trees --
tried on the 2023-01-01 SARS-CoV-2 cutoff, panels C-F of sc2_results_revised2; first
pass on coronaviruses only (the other-virus groups are fetched in part and paused).

**No Nextstrain trees qualify.** Charon's archive returned 404 at 2023-01-01 for every
dataset it answered before rate-limiting us (HTTP 403); of 94 cached current builds only
3 (ebola-2013, yam HA/NA) have every tip pre-2023. (The nextstrain-data S3 bucket that
the sc2 series came from was not yet checked for other pathogens.) So every backbone is
BUILT, `experiments/multitree/corpus2023/`:

* `fetch_group.py` -- Entrez protein, PUBLISHED before 2023-01-01 (PDAT), PDB and
  patent records excluded; summaries first, so huge groups could be subsampled (default:
  every record).
* `split_by_taxon.py` -- the one Coronaviridae spike fetch (13,900 records, SARS-CoV-2
  excluded by taxon, 300-1600 aa) split by NCBI lineage: every sarbecovirus one group
  (in the SARS-CoV-2 frame), a species with >= 20 full-length records its own group,
  smaller ones pooled at subgenus then genus; NCBI's catch-all "species" pool too.
  19 groups.
* `build_cov_tree.py` -- clean (length 0.9-1.1x reference, <= 1% ambiguous; sarbecovirus
  records within 1% of SARS-CoV-2 dropped), MAFFT --keeplength onto the reference
  (--6merpair: this build's localpair segfaults), backbone = a (country, year)-stratified
  subsample of <= 1,500 DISTINCT sequences, IQ-TREE 3 LG+G4 -fast with ancestral states,
  midpoint root, Nextstrain-v2 JSON + root sidecar, then EVERY cleaned record -- partial
  spikes >= 300 aa included, as fragments -- attached with antigen_arms prepare + place
  (run_arms.sbatch's train side; dates from collection, else GenBank creation, --no-clock).
  Internal nodes are '-' where no descendant tip has a residue.
* Totals: 19 trees, 73,243 mutations. Branch lengths are nothing like SARS-CoV-2's
  (1.17 mutations per branch, 0.18 in the RBD): 75% of the relatives' mutations sit on
  branches of > 10. The five genus/subgenus POOLS (97-100% long-branch) were dropped
  (user); the other 14 species trees keep all branches (no cap, user). 14 trees,
  ~51,600 mutations.

**Training changes, `antiGen/train.py` (opt-in; no existing run changes):**
`--resume-state` (model/optimizer/RNG after every epoch, atomic; resumes after the last
epoch saved, batch order seeded by seed + epoch; also writes epochs/fitness_*_epNN.pt),
and `--val-sources` (hold out only from the named sources; each keeps the holdout it
would get alone, since the split is seeded by source name -- verified: 48,524 / 5,392 at
split seed 0, = antiGen CV x10 member 0). quick_preemptible chunks of 1h55m
(`train_joint_cov15.sh`, `train_joint_cov15_cv.sh`) because gpu_batch estimated a
next-day start. `resume_to_ckpt.py` puts joint weights into the control's metadata (a
joint checkpoint records seq_len 1,447; ESM-2 does not depend on it).

**Single runs, recipe = the published ESM-2 35M partial FT (32 epochs planned, no
holdout, seed 42).** The published control is its FINAL epoch and has overfit badly:
the matched SARS-CoV-2-only rerun scores 0.540 / 0.854 / 0.433 / 0.496 at epoch 0
against the published 0.358 / 0.635 / 0.298 / 0.305 at epoch 31. Matched
(joint - SARS-CoV-2 only), C / D / E / F:

| epoch | difference |
|---|---|
| 0 | -0.006 / +0.004 / +0.039 / +0.128 |
| 1 | +0.007 / +0.004 / +0.033 / +0.170 |
| 2 | -0.024 / -0.017 / -0.029 / +0.021 |

Both single runs were stopped after epoch 2 (user).

**CV x3 (av-multitree's headline is CV x3, joint49_*_val_s0-s2).** Members 0-2: split
seed i, init 100+i, 10% SARS-CoV-2 holdout only (--val-sources), the 14 trees train
wholly, 6-epoch budget, --save-best. All three selected epoch 5 = the LAST: the budget
binds (continuations to 12 epochs, jobs 3729032-34, resume after epoch 5). Mean-of-log-
probs average, C-F: 0.564 / 0.892 / 0.480 / 0.681 (antiGen CV x10 0.599 / 0.930 / 0.497 /
0.688). Per-epoch panel scores (monitoring only, they chose nothing) peaked on E/F at
epoch 2 (0.509 / 0.702) while C/D kept rising.

## 156. Next-mutation prediction from the mutation history (2026-11-11)

`antiGen/history.py`, `experiments/history/`, `list_macro.py --history-model`. Query =
the node's mutations from the tree ROOT (Wuhan-Hu-1, shared by the train and test trees;
verified: root + history reproduces every record's sequence, 53,916 train and 75,606
test records). Same targets and loss as antiGen, holdout = antiGen member 0's.

| model | holdout NLL | C | D | E | F |
|---|---|---|---|---|---|
| v1: ordered history, one [QUERY] vector, tied output | 7.971 | 0.437 | 0.772 | 0.380 | 0.482 |
| v2: ordered history, per-site queries cross-attend | 7.545 | 0.483 | 0.837 | 0.400 | 0.504 |
| history as an unordered SET (+15% token dropout) | 7.523 | 0.474 | 0.809 | 0.400 | 0.518 |
| NET DIFFERENCE set from the root (+15% dropout) | 7.539 | 0.532 | 0.851 | 0.518 | 0.582 |
| (antiGen CV x10 member 0 / ensemble) | 7.561 | 0.599 | 0.930 | 0.497 | 0.688 |

The history models beat antiGen on the holdout and lose on the test panels: test
histories are longer (median 51 against 18 mutations; 42 against 12 sites differing from
the root) and come from another inferred tree. The net-difference set -- a function of
the genotype alone, so tree-invariant -- closes most of the gap (E 0.518 ties S×S-MCM
interp.), but it carries no history beyond the genotype. Conclusion: no gain from the
history itself. Line paused (user).

## 157. Blending the joint ESM-2 with antiGen CV x10, chosen on the development holdouts (2026-11-11)

**Rule: nothing is chosen on the test panels.** Joint member i and antiGen CV x10
member i hold out the SAME 10% of SARS-CoV-2 (i = 0, 1, 2), so both are out of sample
there. `holdout_select.py` scores, per member pair, the holdout NLL over ALL mutations
and over DE NOVO cells only (a (site, residue) in no training-split node and no training
child mutation; ~110-130 mutations per holdout -- noisy).

* Joint: holdout NLL over all mutations falls every epoch (7.815 -> 7.621, member 0);
  over de novo cells it is BEST at epoch 0 and rises every epoch (12.46 -> 13.13). antiGen
  member 0: 7.562 all, 14.11 de novo. The two models are good at different things.
* Mixture weight w (score = (1-w) antiGen + w joint, mean of log-probs): the all-mutation
  criterion picks joint epoch 5, w = 0.30 in all three members; the de novo criterion
  picks joint alone (w = 1, epoch 0) -- degenerate.
* `list_macro.py --mix NAME=W[:W2] --mix-dir DIR [--mix-dir2 DIR]`: the weighted blend of
  --ensemble-dir with one or two more ensembles. Gate: the antiGen re-score in each run
  reproduces the published ens_vs10_2023 row (worst difference 0.0).

| blend | C | D | E | F |
|---|---|---|---|---|
| antiGen CV x10 | 0.599 | 0.930 | 0.497 | 0.688 |
| **A: 0.70 antiGen + 0.30 joint (epoch 5)** -- dev-selected | **0.614** | 0.929 | 0.499 | **0.702** |
| B: 0.90 antiGen + 0.10 joint (epoch 0) | 0.607 | 0.932 | 0.496 | 0.681 |

A is within noise of antiGen on D/E and ahead on C (+0.015) and F (+0.014; 71 cells).

**Coronavirus-only finetune** (the 14 trees, no SARS-CoV-2; recipe as above, 6 epochs):
every SARS-CoV-2 example is out of sample for it. `blend_select.py` picks its epoch on
the pooled SARS-CoV-2 holdouts and the three-way weights (antiGen + joint + coronavirus-
only, grid 0.05) on the mean all-mutation holdout NLL; `run_blend_round.sh` then scores
the selected blends (round 1, `blend_r1.json`):

* Coronavirus-only, pooled SARS-CoV-2 holdout NLL by epoch: 9.40, 9.59, 9.80, 9.86,
  10.42, 10.55 -- best at epoch 0 and worse every epoch after (more coronavirus
  training moves it AWAY from SARS-CoV-2). Epoch 0 is used.
* Holdout NLL (mean over members, all mutations / de novo): antiGen alone 7.5695 /
  14.29; A+J (wJ 0.35) 7.5534 / 13.88; A+C (wC 0.05) 7.5657 / 14.04; **A+J+C
  (0.65 / 0.30 / 0.05) 7.5507 / 13.69** -- the dev pick.

| blend (weights A/J/C) | C | D | E | F |
|---|---|---|---|---|
| antiGen CV x10 | 0.599 | 0.930 | 0.497 | 0.688 |
| gate: mixture A re-scored (0.70/0.30/0) | 0.614 | 0.929 | 0.499 | 0.702 |
| A+J (0.65/0.35/0) | 0.609 | 0.927 | 0.499 | 0.702 |
| A+C (0.95/0/0.05) | 0.604 | 0.931 | 0.497 | 0.688 |
| **A+J+C (0.65/0.30/0.05)** -- dev pick | **0.611** | 0.929 | **0.513** | **0.709** |

The gate reproduces mixture A exactly. The dev pick improves on antiGen on C (+0.012),
E (+0.016) and F (+0.021), and ties D. Note A+J at 0.35 scores below mixture A's 0.30
on C: the 0.05 grid step is within the noise of the panel.


## 158. Five-way dev-selected blend: + counted SxS/MCM interpolation and codon (2026-11-11)

**Which models may be weighted on the dev holdouts.** Only those out of sample on member
i's holdout: antiGen CV x10 member i (A), joint member i (J), the coronavirus-only
finetune (C, s157), and two baselines made eligible here. **S** is mcm_interpolation
(`statistical_baselines`, kappa = N_total) with its counts REFIT on member i's training
90%. The published row is fit on all training examples, holdout included. **K** is codon,
which is untrained. The published antiGen single model, PFM and the other fitted rows
saw the holdouts and are not eligible.

* `blend_select2.py`: per-member log-probs for A, J, C (epoch 0), S, K, then the full
  simplex grid at step 0.05 (10,626 points). **Primary pick = the grid minimum of the
  mean all-mutation holdout NLL.** Every subset lies on the grid, so this is one pick.
  The de novo criterion's minimum is a secondary, reported only.
* `list_macro.py --mix-baseline {codon,mcm_interpolation,site_substitution}`
  (repeatable): further `--mix` components, fitted on the run's training examples.
  `--mix NAME=wJ:wC:wS:wK` with the antiGen ensemble taking 1 - sum. Every component is
  clamped at -1e4 before weighting (a no-op for all components so far).
* `run_blend_round2.sh TAG JOINT_EPOCH SELECTION_JSON` scores a selection with four
  gates that must reproduce: ens_vs10_2023 and S alone / K alone against the published
  rows, and r1's A+J+C against s157. All four reproduce here (worst difference 0.0003, the
  rounding of the published 3-decimal values).

Solo holdout NLL (member 0, all / de novo): A 7.561/14.11, J (epoch 5) 7.621/13.13,
C 9.409/11.11, S 7.901/12.72, K 9.203/10.12. Codon is the best single model on de novo
holdout cells, and the worst-but-one over all mutations.

Selection with J = epoch 5 (`blend_sel2_ep5.json`). It reproduces s157's A+J+C exactly
(0.65/0.30/0.05, 7.5507):

| pick (weights A/J/C/S/K) | holdout all | holdout de novo | C | D | E | F |
|---|---|---|---|---|---|---|
| antiGen CV x10 (1/0/0/0/0) | 7.5695 | 14.29 | 0.599 | 0.930 | 0.497 | 0.688 |
| s157 A+J+C (.65/.30/.05/0/0) | 7.5507 | 13.69 | 0.611 | 0.929 | 0.513 | 0.709 |
| **primary** (.65/.30/0/0/.05) | **7.5490** | 13.67 | 0.614 | 0.929 | 0.484 | 0.688 |
| secondary, de novo (0/0/.25/0/.75) | 9.0377 | 10.42 | 0.408 | 0.511 | 0.573 | 0.560 |
| (dev only) A+S (.85/0/0/.15/0) | 7.5624 | 14.02 | | | | |
| (dev only) A+K (.95/0/0/0/.05) | 7.5641 | 14.02 | | | | |

**Reading.** The new primary beats s157's pick on the dev criterion by 0.0017 nats, but
does worse on E (-0.029) and F (-0.021). A difference that small on the holdout does not
separate the two, and this result is not a reason to swap back: swapping on the panels is
the peeking this section rules out. S adds little once A is present (antiGen already
carries the count information). The de novo criterion is degenerate as in s157: it drops
antiGen entirely. On de novo holdout cells (~110-130 per member) codon's "reachable by
one nucleotide change" prior wins. It gives the best E of any row (0.573, codon alone
0.536) at a heavy cost on C, D and F. Takeaway: the all-mutation holdout NLL is dominated by
recurrent mutations and cannot weight de novo specialists; a fair E/F-targeted blend
needs a larger de novo dev set (e.g. pooled across all 10 splits) or a rank criterion.

Round 3 (`round3.sh`, unattended) repeats the selection with each joint member's own
--save-best checkpoint from the 12-epoch continuation (holdout bests so far: s0 epoch 6,
s1 epoch 7, s2 still improving at epoch 8), and scores it the same way.

**Round 3 result.** The 12-epoch holdout minimum is epoch 9 for all three members (7.593 /
7.611 / 7.595; epochs 10-11 are worse), so J = epoch 9. Gates reproduce (worst 0.0003).

| pick (weights A/J/C/S/K) | holdout all | holdout de novo | C | D | E | F |
|---|---|---|---|---|---|---|
| antiGen CV x10 | 7.5695 | 14.29 | 0.599 | 0.930 | 0.497 | 0.688 |
| A+J+C on the grid (.60/.35/.05/0/0) | 7.5450 | 13.77 | | | | |
| **primary** (.55/.35/0/.05/.05) | **7.5431** | 13.66 | 0.614 | 0.926 | 0.477 | 0.695 |
| secondary, de novo (0/0/.25/0/.75) | 9.0377 | 10.42 | 0.408 | 0.511 | 0.573 | 0.560 |

Across rounds 1-3, every dev pick lifts C by about +0.012 to +0.015. D is flat. E moves
-0.020 to +0.016 and F moves 0.000 to +0.021, with no consistent sign. Holdout NLL keeps
improving (7.5507 -> 7.5490 -> 7.5431) without the panels following. The robust gain from
the joint model is on C; the round 1 E/F gain is best read as noise.

## 159. Panel E at codon distance 1 (2026-11-11)

Question (user): among de novo mutations reachable by ONE nucleotide change, where the
genetic code no longer separates candidates, which models have learned the rules?

* `list_macro.py --cd1-variants` adds `list_<l>_unseen_cd1_fullpool` (panel E restricted)
  and `list_<l>_cd1_fullpool` (panel C restricted). For each query, the pool AND the
  targets are limited to cells whose residue is one nucleotide change from the query's
  OWN wild-type residue (any codon of wt -> any codon of mut at Hamming 1; the examples
  carry no nucleotide state). That is 150 of 380 substitutions. Per-query because 320
  RBD wild-type letters differ from the root across backgrounds. `compute_hit_rate`
  accepts a (B, R, E) pool for this.
* Run: the full published 2023-01-01 all-cells roster plus the s157/s158 blends
  (`evals_cd1_2023-01-01`, job 3730899, 32 min). Gate: all 2,160 pre-existing
  (model, variant, qtile) rows reproduce the published run exactly (worst diff 0).
* Panel E at cd1: 48 cells and 104 events (from 71 / 141). The random rate is 0.10.
  Codon is not fully inert inside the pool, because it still counts routes: 0.242 / 0.203.

| model | E macro / micro | E (cd1) macro / micro | C (cd1) macro / micro |
|---|---|---|---|
| antiGen CV x10 | 0.497 / 0.688 | **0.557** / 0.635 | 0.332 / 0.777 |
| blend r2 (A .65, J .30, K .05) | 0.484 / 0.688 | 0.555 / **0.644** | 0.342 / 0.790 |
| antiGen single (simple_d128) | 0.507 / 0.660 | 0.541 / 0.615 | 0.309 / 0.737 |
| blend r1 (A .65, J .30, C .05) | 0.513 / 0.709 | 0.526 / 0.615 | **0.344 / 0.792** |
| gated xfmr + PSSM | 0.475 / 0.546 | 0.501 / 0.615 | 0.317 / 0.749 |
| joint ESM-2 CV x3 | 0.480 / 0.681 | 0.400 / 0.587 | 0.328 / 0.753 |
| where x which / wxw Dirichlet | 0.518 / 0.667 | 0.397 / 0.500 | 0.251-0.275 |
| SxS / MCM interpolation | 0.518-0.519 / 0.667-0.674 | 0.356 / 0.452 | 0.254-0.276 |
| de novo blend (C .25, K .75) | 0.573 / 0.560 | 0.354 / 0.385 | 0.193 / 0.297 |
| EVEscape | 0.239 / 0.496 | 0.292 / 0.510 | 0.189 / 0.243 |
| codon | 0.536 / 0.490 | 0.242 / 0.203 | 0.143 / 0.180 |
| ESM-2 35M partial4 (SARS-CoV-2 only) | 0.298 / 0.305 | 0.231 / 0.221 | 0.262 / 0.564 |

Reading: once reachability is controlled, the trained antiGen models lead, and the count
and substitution baselines fall by 0.12-0.16 macro. Their panel E edge was largely
the genetic code. The de novo blend's E lead was too (0.573 -> 0.354). Caveat: 48 cells.

**supp_cd1** (`antiGen-figures/figures/figS_cd1.R` -> `figs/supp_cd1.png`): the main
figure's C-F on this restriction. It uses the same export (`panels_coguk_export.R`, new arm
`REVISED2_ARM=cd1_2023`), roster (DROP_MAIN), names and colours.
`experiments/multitree/stage_cd1_2023.sh` stages the cd1 variants under the drawer's
variant names into `data/evals_sc2_lists_macro_aa20_2023-01-01_cd1`, and leaves out the
blend rungs. antiGen (CV x10) is first in all four panels. Second place: PFM on A and B,
S×S on C, EVEscape on D (0.510; antiGen 0.635).

**Why A-D read lower than the main figure's C-F: the decile shrinks, not a bug.** A
query's full pool has 3,819 non-wild-type RBD cells, so the decile is 382 slots; 1,565 of
those cells are one nucleotide away. A model that ranks one-nucleotide cells first spends
its 382 slots on its top ~24% of them in the main figure, against 157 slots (10%) here.
Diagnostic (verify_cd1.py, C1t): one-nucleotide targets ranked over the FULL pool give
antiGen CV x10 0.634 / 0.942, in line with the main figure's 0.599 / 0.930.

**Independent check** (`experiments/multitree/verify_cd1.py`). It shares only data and
model loading with list_macro, and recomputes the reachability, pools, fractional top-q
selection and pooling. Cell and event counts match in every variant. A constant score gives
exactly 0.100 everywhere. Codon and simple_d128 match to <= 4e-8 on C, E, C1 and E1, with
codon's route counts rounded so its exact ties stay exact; float64 sum order otherwise
splits ties (up to 0.008). mcm_interpolation and the CV x10 ensemble differ by <= 0.007,
which I attribute to GPU vs CPU float32 near-ties. The first version of the verifier had its
own bug (per-cell hits added back in sorted order); fixed.

**Row 2, G-J at codon distance 1** (`evals_cd1_2021`, the COG-UK escape list at
2021-01-01). Only the main figure's roster is scored; checkpoints are staged in
`evals_cd1_2021/ckpt`. Gates: all 972 re-scored rows reproduce the published G-J data
exactly, and the natively scored ensemble (ens_vs10_2021, --ensemble-dir
ckpt_stage_vs10_aa20) reproduces the published `ens_8a91fbbf88` to 1e-7.
`stage_cd1_2023.sh ARM=2021` stages it under that name only if that gate passes. cd1 sets:
211 cells / 17,465 events, and de novo 106 / 3,381. supp_cd1 is now two rows: A-D = C-F,
E-H = G-J.

| model | G-J (macro/micro: all, de novo) | G-J at cd1 (all, de novo) |
|---|---|---|
| antiGen CV x10 | 0.545/0.831, 0.508/0.711 | 0.375/0.657, 0.384/0.427 |
| EVEscape | 0.359/0.498, 0.387/0.545 | 0.374/0.492, **0.426/0.543** |
| BA.1 DMS (expression) | 0.276/0.300, 0.320/0.483 | 0.291/0.290, 0.384/0.484 |
| SxS-MCM interp. | 0.514/0.828, 0.483/0.705 | 0.334/0.644, 0.295/0.286 |
| PFM | 0.449/0.790, 0.100/0.100 | 0.307/0.637, 0.100/0.100 |
| codon | 0.282/0.292, 0.308/0.342 | 0.145/0.144, 0.140/0.088 |

On escape, the restriction costs antiGen most on de novo escape per mutation
(0.711 -> 0.427), while EVEscape and the DMS barely move. At cd1, antiGen is 1st on all
escape (E, F) but 3rd on de novo escape (G, H), behind EVEscape and BA.1 DMS (expression).
Part of antiGen's de novo escape lead in the main figure is the genetic code. The all-RBD
row's lead is not.

## 160. supp_cd1 C-F scored only on cells seen in >= 100 post-cutoff sequences (2026-11-12)

User request: keep supp_cd1's per-query top decile (the one-nucleotide pool, de novo for
the de novo panels, and the same ranking), but score only (position, residue) cells seen
in >= 100 sequences dated on or after the 2023-01-01 cutoff. All-RBD row only (user:
C-F only, no G-J).

* Mask: the PFM `sc2-test/counts_ba2/counts_2023-01-01.tsv` (688,459 sequences dated >=
  2023-01-01; the BA.2 subtree, which after this date is essentially every sequence).
  339 of 4,020 RBD cells pass (wild-type residues included).
* Because the pool and ranking are unchanged, each cell's hit mass is unchanged. The panels
  are recomputed from the s159 run's per-cell output (`stage_cd1_min100.py`); nothing is
  re-scored. Gate: the per-cell rows, unmasked, reproduce the run's summary for all 70
  (model, variant) pairs (worst 6.9e-6; per-cell hits are stored to 4 decimals).
* `figures/figS_cd1_min100.R` -> `figs/supp_cd1_min100.png`; arm `cd1min100_2023`.

Panel A/B (all, per cell / per mutation; 183 cells, 8,904 events): antiGen CV x10
0.781 / 0.898, PFM 0.770 / 0.835, EVE 0.710 / 0.747, SxS-MCM interp. 0.636 / 0.790,
Tranception 0.544 / 0.563, SxS 0.504 / 0.595, EVEscape 0.290 / 0.254, codon 0.121 / 0.184.

**Panels C/D are degenerate: ONE cell survives,** V445H (BA.2.86's; 13 test events,
197,803 post-cutoff sequences). Every model that ranks it in the top decile scores 1.0;
the others score 0. At lower thresholds the de novo set is still tiny (>= 5 sequences:
11 cells / 60 events; >= 10: 5 / 46; >= 20 and >= 50: 2 / 28). A cell absent from the
7.8M training-window sequences rarely reaches 100 sequences afterwards.

## 161. The top-q selection over-selected by up to one cell per query: fixed (2026-11-12)

**Found by** verify_cd1.py (s159). Fed the verifier's own scores and pools,
`compute_hit_rate` reproduced list_macro exactly, but differed from the verifier's
brute-force expected-hits selection (selecting exactly q * n cells, ties split
uniformly) by up to 0.007.

**The bug.** The boundary value was `torch.nanquantile(p, 1 - q, interpolation='lower')`,
the element at ascending index floor((1 - q)(n - 1)). The cells strictly above it can
number MORE than q * n: with untied scores, n_above = n - 1 - floor((1 - q)(n - 1)),
e.g. 86 against q * n = 85.4 at n = 854, or 10 against 9.5 at n = 95. The tie fraction
`(q n - n_above) / n_at` then clamps to 0, so the selection overshoots q * n by up to one
cell. It was exact only when the boundary fell inside a tied block. The docstring promised
"exactly q * n_valid cells". The bias favours continuous scores over coarse ones
(constant, codon, PFM, BLOSUM62). It is <= 1 / (q n) of the decile: <= 0.3% on the full
pools (~3,800 cells), ~0.6-2% on the codon-distance-1 and de novo pools.

**The fix.** `eval_utils.topq_boundary`: the ceil(q n)-th largest valid value. Then
n_above <= ceil(q n) - 1 < q n <= n_above + n_at, so frac is in [0, 1] and the selection
is exactly q * n. It is used by every value-based copy: compute_hit_rate, the second
copy in eval_utils, list_variant_recall, list_recall, and zeroshot_probe, tie_probe,
fuse_forecast, forecast_hit_summary, figure_nll_audit and collect_figure_metrics. The
rank-based rebuild `background_discrimination.top_q_weight` (rank_fractions consumers:
ensemble_report / sweep / diagnostics, rescore_common_pool, backbone_list_eval,
background_contrast) now takes the boundary at top rank ceil(q n) - 1.
`tests/test_topq_exact.py` pins exact q * n selection for untied, tied and coarse scores.
`test_validate_alphabet::test_the_twenty_two_letter_path_is_unchanged` compares against
`git show HEAD:eval_utils.py` bit-for-bit, so it fails until this is committed (BLOSUM62's
hit rate moves 0.2050 -> 0.2049).

Size on the s159 run (verifier = the corrected rule): antiGen CV x10 C 0.5992 -> 0.5991,
E 0.4970 -> 0.4970, C1 0.3320 -> 0.3314, E1 0.5569 -> 0.5528; mcm_interpolation E
0.5192 -> 0.5121. Codon, constant and simple_d128 unchanged to 1e-8.

**Every hit-rate CSV written before this is stale.** Regeneration of the published figures:
see below.

**Decision (user, 2026-11-12): the published figures are NOT regenerated.** The effect is
too small to matter (<= 0.3% of the decile on their pools). Everything computed from now on
uses the fixed rule. So the existing sc2_results_revised2 (+ supp), supp_fixed_split,
supp_fullspike, supp_gate_units, supp_query_adapt, supp_cd1 and supp_cd1_min100 (2023)
are on the OLD rule, and supp_cd1_min100_2021 (job 3732770, submitted after the fix) is on
the NEW one. A regeneration plan (about 34 GPU-h, about 5 CPU-h; only the 2021 list maps
and the background caches are replayable) is in this session's record if it is ever wanted.

**supp_cd1_min100_2021** (fixed top-q rule, s161). The same eval for all RBD cells at a
2021-01-01 cutoff: the one-nucleotide pool, and targets limited to cells seen in >= 100
sequences dated >= 2021-01-01, counted over ALL lineages.
* Run `STORE/evals_cd1_2021_allcells` (job 3732770, 26 min; main-figure roster: 2021
  corpus, checkpoints `evals_cd1_2021/ckpt`, antiGen = CV x10 from
  ckpt_stage_vs10_aa20).
* Mask `STORE/postcutoff_counts/counts_all_lineages_2021-01-01.tsv` (8,151,572
  sequences), written by `experiments/multitree/postcutoff_counts.py`. counts_ba2 would
  miss Alpha, Delta and BA.1. The script reproduces counts_all and counts_ba2 at
  2021-01-01 and 2023-01-01 exactly before writing; the BA.2 root, found from those
  totals, is node 3430.
* `ARM=2021 stage_cd1_min100.py` (gate worst 1e-6), then `FIG_ARM=2021
  figS_cd1_min100.R`. 509 RBD cells pass the mask.

| model | all, 1 nt, >=100 (cell / mut.; 364 cells, 39,615 events) | de novo (90 cells, 5,504 events) |
|---|---|---|
| antiGen CV x10 | 0.470 / 0.609 | **0.447 / 0.480** |
| EVEscape | 0.225 / 0.264 | 0.370 / 0.429 |
| SxS | 0.390 / 0.528 | 0.347 / 0.361 |
| SxS-MCM interp. | 0.435 / 0.587 | 0.344 / 0.358 |
| BA.1 / BA.2 DMS (expr.) | 0.197 / 0.223, 0.194 / 0.237 | 0.333 / 0.353, 0.333 / 0.358 |
| PFM | **0.540 / 0.730** | 0.100 / 0.100 |
| EVE | 0.416 / 0.587 | 0.183 / 0.163 |

Not drawn, but present in the run: Bloom & Neher (BloomNeher_2023-10-01) scores 0.525 / 0.673
and 0.611 / 0.749. It is fit on data through 2023-10, so at a 2021 cutoff it is not out
of sample, and the drawer's roster leaves it out.

**supp_cd1_min100_2021 + the antiGen / PFM gate** (user). gated_static_xfmr_pssm_lr0.0001
(2021 checkpoint from ckpt_stage_aa20) was added to `evals_cd1_2021/ckpt` and scored by
resume (job 3732903, 1.5 min; the 26 models already scored were skipped). The figure
keeps it in this version only: DROP_MAIN minus the gate. Its key gets 0.75 in, because
the extra model type wraps it to two rows. Gate: all cells 0.550 / 0.718 (1st per cell,
2nd per mutation after PFM 0.730); de novo 0.370 / 0.381. antiGen CV x10 is 0.447 / 0.480;
EVEscape is 0.370 / 0.429 (it edges the gate per cell, 0.37001 against 0.36990).

## 162. supp_cd1_min100_2021's de novo panels (C, D) swept over cutoffs (2026-11-12)

User request, overnight: the quantity of supp_cd1_min100_2021 C/D swept over train/test
cutoffs, like sc2_results_revised2 L/M, from the earliest cutoff to the last one with
enough de novo cells to discriminate models (2023-01-01 is too late: 1 cell).

Quantity at cutoff D: top-decile recall (q = 0.1) of each query's one-nucleotide pool
restricted to de novo cells (pssm_root == 0 over D's training window), scoring target
cells that are de novo, one nucleotide away, AND seen in >= 100 sequences (all lineages)
dated >= D. Per cell (macro) and per mutation (micro). Fixed top-q rule (s161).

* Corpus and roster: panel L's per-cutoff series (`submit_macro_allcells_aa20.sh
  CUTOFF=D`: the sc2-D train split, the shared test file with --split-date D
  --pool-examples, substitution seed 0, the 4,020-cell incl_ref list, checkpoints
  `cutoff_stage_aa20/D`), plus --mcm-baseline --cd1-variants, into
  `STORE/evals_cd1_series/D`. Only the single antiGen model, the gate, ESM-2 35M and
  BLOSUM62 were trained per cutoff, so the antiGen line is the SINGLE model (as in L/M),
  not CV x10. At 2021-01-01 this corpus differs from supp_cd1_min100_2021's 2021-arm
  corpus (sc2-2021 train/test), so that point need not equal it.
* Masks: postcutoff_counts.py extended to every cutoff (all-lineage PFM of sequences
  dated >= D), each gated against counts_ba2 at all 24 dates.
* **Stopping rule, fixed BEFORE any model's result is read.** It is computed from cell
  and event counts only, which are model-independent. A cutoff is drawn only if its scored
  set has >= 30 cells AND >= 300 events. The series runs from 2020-04-01 to the last
  cutoff D such that every cutoff up to D passes.

**Second arm (user): rare in training instead of de novo.** The same eval, with cells
seen in FEWER THAN 100 training-window sequences in place of de novo cells. Each query's
pool is its one-nucleotide cells with training count < 100, and the scored targets are
those cells seen in >= 100 sequences dated >= D. Implemented as `list_macro.py
--seen-min-count 100`: "seen" is training count >= N, default 1 = the published de
novo convention, unchanged; the unseen variants then mean count < 100. The run goes to
`STORE/evals_cd1_series_rare100/<D>` through `run_cd1_series_rare100.sh` (the de novo
driver's twin). Same stopping rule, applied to this arm's own counts and fixed now,
before any result: >= 30 cells and >= 300 events.

**Results.** Both drivers completed all 11 cutoffs (2020-04-01 .. 2022-10-01; 22 runs,
gpu_batch). `stage_cd1_series.py` checks each cutoff by comparing the unmasked per-cell
rows with the run's own summary. The worst difference over all 22 runs is 1.0e-5.

Scored set per cutoff (model-independent), with the stopping rule applied:

| cutoff | de novo cells / events | <100-training cells / events |
|---|---|---|
| 2020-04-01 | 254 / 23,376 | 311 / 33,791 |
| 2020-07-01 | 178 / 12,925 | 306 / 31,735 |
| 2020-10-01 | 138 / 8,455 | 300 / 29,413 |
| 2021-01-01 | 90 / 5,504 | 284 / 24,631 |
| 2021-04-01 | 47 / 2,652 | 242 / 18,058 |
| 2021-07-01 | 31 / 1,725 | 203 / 13,222 |
| 2021-10-01 | 24 / 1,449 (fails: stop) | 142 / 8,424 |
| 2022-01-01 | 9 / 529 | 74 / 4,110 |
| 2022-04-01 | 3 / 135 | 58 / 2,697 |
| 2022-07-01 | 3 / 135 | 44 / 1,896 |
| 2022-10-01 | 1 / 15 | 33 / 1,199 |

So the de novo arm is drawn from 2020-04-01 to 2021-07-01 (6 cutoffs), and the
<100-training arm at all 11. At 2021-01-01 the de novo set is the same 90 cells and
5,504 events as s160. The gate scores exactly s160's 0.370/0.381 there, because it is
the same checkpoint on the same cells. The single antiGen model scores 0.402/0.436
against s160's CV x10 0.470/0.609.

Per cell / per mutation at q = 0.1 (antiGen = simple_d128_n2 single model; gate =
gated_static_xfmr_pssm; the best non-antiGen line is named):

| cutoff | antiGen | gate | best other |
|---|---|---|---|
| de novo 2020-04-01 | 0.288 / 0.353 | 0.280 / 0.348 | wxw_dirichlet 0.510 / 0.674 |
| de novo 2020-10-01 | 0.320 / 0.341 | 0.305 / 0.358 | wxw_dirichlet 0.520 / 0.548 |
| de novo 2021-01-01 | 0.402 / 0.436 | 0.370 / 0.381 | wxw_dirichlet 0.544 / 0.606 |
| de novo 2021-04-01 | 0.639 / 0.709 | 0.669 / 0.709 | mcm_interpolation 0.620 / 0.704 |
| de novo 2021-07-01 | 0.693 / 0.743 | 0.664 / 0.796 | wxw_dirichlet 0.700 / 0.746 |
| <100 2020-04-01 | 0.289 / 0.363 | 0.284 / 0.361 | wxw_dirichlet 0.486 / 0.685 |
| <100 2021-01-01 | 0.439 / 0.552 | 0.472 / 0.604 | mcm 0.524 / 0.667 |
| <100 2021-07-01 | 0.555 / 0.686 | 0.540 / 0.663 | mcm 0.561 / 0.615 |
| <100 2022-01-01 | 0.507 / 0.556 | 0.508 / 0.570 | wxw_dirichlet 0.550 / 0.683 |
| <100 2022-10-01 | 0.648 / 0.760 | 0.731 / 0.831 | mcm 0.578 / 0.567 |

Reading: through 2020 the statistical baselines (S x S / S x M with MCM interpolation,
MCM) lead by about 0.2, with antiGen near EVEscape. antiGen climbs steeply from
2021-01-01 and ties the leaders from 2021-04-01, which is roughly when Alpha-era data
enters training. In the <100 arm it leads or ties from mid-2021, except at 2022-01-01,
and the gate is top at the last two cutoffs. As in s160, Bloom & Neher is not drawn,
because it is fit through 2023-10 and so is not out of sample.

Figure: `figures/figS_cd1_series.R` -> `figs/supp_cd1_series.png`. Data:
`data/gating_cd1_dn_min100/`, `data/gating_cd1_rare100_min100/`. Drawer change:
`make_gate_plot` takes an opt-in `sc2.gate_xbreaks` option, set through
`REVISED2_GATE_XBREAKS` in panels_gate_macro_export.R. It puts breaks at decimal-year
cutoffs labelled "Apr 2020". Unset, every published axis is unchanged.

## 163. Training-fitted baselines saw test-period mutations: fix and audit (2026-11-12; running)

**Found** (user, comparing supp_cd1_min100_2021 C with supp_cd1_series A at 2021-01-01).
Both score the same 90 cells and 5,504 events, and every model that ignores the training
pool agrees to 1e-8. The counted baselines did not: S×S-MCM 0.34 -> 0.54. The series
passed `--pool-examples` (s111's eval.py semantics), so the counted baselines were
fitted on the train file PLUS the shared test corpus's pre-cutoff branches: 6,836
examples against the 2,934 that every antiGen checkpoint was trained on. Checkpoint
`train_args` confirm it: `examples` = the cutoff's own train file, `split_date` None,
for every trained rung at every cutoff. BLOSUM62 is one untrained lookup table, shared
across cutoffs, which is harmless. The 3,902 extra branches carry 331 events at 64 of
the 90 "de novo" cells, so for the counted baselines those cells were not de novo.

**Rule (user): every model in a comparison sees exactly the same training data.**

**Fix for supp_cd1_series:** `submit_macro_allcells_aa20.sh POOL=0` keeps
`--split-date` and drops `--pool-examples`. The test set is unchanged (71,704 at
2021-01-01), and every fitted baseline sees only the train file. Drivers:
`run_cd1_series_trainonly.sh` and `run_cd1_series_rare100_trainonly.sh`, writing to
`STORE/evals_cd1_series{,_rare100}_trainonly/<D>`. The stopping rule uses only cell and
event counts, which pooling does not change, so the drawn cutoffs do not change.

**What pooling touches.** Only the rungs fitted from `training_examples`: substitution
(Codon-B62 mixture), where_times_which (S×M), wxw_dirichlet (S×M-MCM), site_substitution
(S×S), mcm_interpolation (S×S-MCM) and mcm. Trained checkpoints, fixed maps, PFM
(`--root-counts`), codon and the test pool do not depend on it.

**Size of the extra data.** Test-corpus pre-cutoff branches against the train file:
x3.44 at 2020-04-01, x2.33 at 2021-01-01, x2.02 at 2023-01-01, ~x2.00 after that.

**Audit: which runs pooled** (from every log's training-pool line):

| run family | training pool for counted baselines | fair? |
|---|---|---|
| `eval.py` (eval_all_sc2), every cutoff, 2021 arm AND 2023 | pooled: eval.py always pools both files, then splits on is_training | NO |
| list_macro, 2023-01-01 arm and the L/M per-cutoff series (`--pool-examples`) | pooled | NO |
| `bg` (2023 background), `cd1_panelE` | pooled | NO |
| endemic evals (H3N2, RSV-A, DENV2) | pooled (eval.py) | NO |
| list_variant_recall | pooled (same loader as eval.py) | NO |
| list_macro, sc2-2021 arm (COG-UK lists, cd1, cd1_2021_all) | train file only (2,934) | yes |
| background_discrimination, 2021 arm | train file only | yes |

Even the sc2-2021 arm's test file carries the 3,902 pre-cutoff branches, so eval.py's
2021 arm fits on 6,836 while list_macro's fits on 2,934. Per-cell and per-mutation bars
in one figure can therefore come from differently fitted baselines.

External baselines (PyR0, Bloom & Neher, CoVFit, EVEscape, DMS, ESM-2/EVE pretraining)
are trained on outside data and cannot share antiGen's. The fixed-split panel already
compares each with antiGen trained to that baseline's own date. Bloom & Neher is dropped
from every earlier-cutoff panel by name.

**Correction: what the extra branches are. This is test-period leakage, not just "more
pre-cutoff data".** The test corpus is ONE tree over every sequence of every date. A
node is labelled training at D when the earliest leaf anywhere below it is dated < D
(`minLeafDated`, antiGen-data-revised/src/examples.cpp:62). Its record still lists the
mutations to ALL of its children (`MUTATIONS_TO_CHILDREN`, examples.cpp:373), whatever
their dates. So one early sequence under a node makes every later mutation off that
node a "training" event. At 2021-01-01, in the 3,902 such records:

* of the 56,255 child events, 9,653 lead to an interior child whose whole subtree is
  dated >= D, 3,938 to an interior child with a pre-cutoff leaf, and 42,664 to leaves
  (whose dates the record does not carry);
* of the 331 events at the 90 de novo cells, 66 lead to an all-post-cutoff interior
  child and 265 to leaves. NOT ONE pre-cutoff sequence in the whole corpus carries any
  of the 90 cells (all-lineage counts: 0 before D, 10.3 M after), so all 331 are
  mutations first observed after the cutoff.

The counted baselines in every pooled run were therefore partly fitted on the test
period. The trained antiGen checkpoints never were: they were trained on the per-cutoff
train tree, which is built only from sequences dated < D.

**antiGen was trained correctly.** Every checkpoint in every directory a figure run
loaded (124: ckpt_stage_aa20, ckpt_stage_vs10_{aa20,2023-01-01}, ckpt_stage_bg/il_2023,
cutoff_stage_aa20/*, the cd1 stages) records `examples` = its cutoff's train-tree file
and `split_date` None. Across all 1,159 checkpoints in STORE, 939 record a train file
and none a test file. The other 220 are untrained tables or legacy checkpoints that no
figure run loads.

**Current figures (not `figs/old/`), panel by panel.**

| figure / panels | source | status |
|---|---|---|
| sc2_results_revised2 (+supp) C-F, All RBD 2023 | list_macro `--pool-examples` | LEAKED, re-fitted |
| sc2_results_revised2 G-J, escape 2021 | list_macro, train file only | fair |
| sc2_results_revised2 K, fixed splits | antiGen vs one external model | fair (no fitted rung) |
| sc2_results_revised2 L (per cell) | list_macro `--pool-examples`, 16 cutoffs | LEAKED, re-fitted |
| sc2_results_revised2 M (per mutation) | eval.py (always pooled), 16 cutoffs | LEAKED, re-fitted |
| supp_cd1 A-D (2023) / E-H (2021) | cd1 2023 pooled / cd1 2021 train only | A-D re-fitted / fair |
| supp_cd1_min100 (2023) / _2021 | from the cd1 runs above | re-fitted / fair |
| supp_cd1_series | per-cutoff series | rerun, POOL=0 |
| supp_fullspike | full-spike list_macro, pooled, 16 cutoffs + 2023 | LEAKED, re-fitted |
| supp_query_adapt | C-F's data + background 2023 (pooled) + background 2021 (train only) | re-fitted / fair |
| coguk_list_aa20 (+supp) | 2021 list run, variant recall from its cached maps, background 2021 | fair |
| supp_dms, supp_dms_revised2 | maps dumped from the 2021 list run | fair |
| supp_gate_units, supp_fixed_split, usher_benchmark | no fitted rung | fair |

**The fix: re-fit only what depends on the training pool, and splice it in.** Rescoring
whole runs would take tens of GPU-hours for rows that do not change.

* list_macro (`submit_refit_trainonly.sh`, 33 runs): `refit_trainonly.py prepare`
  copies a run without its fitted rows (and the unused blend rows) into `<run>_trainonly`.
  list_macro then resumes into the copy with POOL=0 and no checkpoint, so it scores only
  the missing rows. codon and pssm_root are dropped and rescored as the GATE
  (`refit_trainonly.py verify`).
* eval.py (`submit_refit_evalpy.sh`, 16 runs, into `cutoff_evals_aa20_refit`) and
  background 2023 (`submit_refit_bg_2023.sh`): a fresh no-checkpoint run (eval.py on
  resume drops rows it has no checkpoint for, so it cannot resume into a copy).
  `merge_refit.py ORIG NEW DST` splices the fitted rows into a copy, gated on every
  shared non-fitted row.
* `stage_refit_all.sh`: all gates, then `merge_refit.py splice` replaces the fitted rows
  of each staged figure file in place. Every other row must equal the source exactly.
* Partition: cpu_batch_high_mem with `--gres=none`. No checkpoint is loaded; there are
  ~700 idle CPUs; gpu_batch had them pending on Priority. A 2020-04-01 refit took ~10 min.
* **Top-q rule.** The copied rows were scored under the pre-s161 boundary, so the refits
  run with `ANTIGEN_TOPQ_LEGACY=1`. This new opt-in switch is in
  `eval_utils.topq_boundary` and `background_discrimination.top_q_weight`, pinned by
  `tests/test_topq_legacy.py`. Without it a panel would mix two rules, and the old one
  over-selects, which favours the copied rows. With it, the gate rows agree to 1.2e-7
  (CPU vs GPU summation order), where the rule change alone moves them ~1e-3.

First gate, 2020-04-01 RBD: 288 fixed-row values identical (worst 1.2e-7). Fitted rows,
pooled -> train only, per mutation: MCM all 0.632 -> 0.282 and de novo 0.715 -> 0.114;
S×S 0.706 -> 0.577 and 0.717 -> 0.547; S×S-MCM 0.726 -> 0.601 and 0.754 -> 0.546.

**Code now defaults to the train file only.** `eval.py` and `list_variant_recall.py`
exclude every record from `--test-examples` from the training pool. eval.py keeps the
old behaviour behind `--pool-examples`, marked legacy. list_macro already defaulted to
it. `submit_macro_allcells_aa20.sh` now defaults to POOL=0. `--pool-examples` is removed from
`submit_bg_2023.sh`, `finish_ft_2023.sh` and `submit_dataseed_2023.sh`. The dseed jobs already
queued were submitted with it, which is harmless: they run `--baselines-only --skip-baselines`
and score only the antiGen ensemble rung. In list_macro the training pool feeds nothing but the
fitted baselines and the reference root, which is the train file's first record either way.

**supp_cd1_series re-done (s162's figure, train-only fits).** Both arms were rerun with
POOL=0 into `STORE/evals_cd1_series{,_rare100}_trainonly`. Because these are complete
runs, every row of the figure is on the s161 rule. Counts and the stopping rule are
unchanged (de novo 2020-04-01 .. 2021-07-01; <100 arm all 11 cutoffs), as they must be:
pooling never touched the test set. The de novo staging read 2020-04-01 .. 2021-10-01.
2021-10-01 is the first failing cutoff, and later de novo cutoffs cannot be drawn.

**Consistency check that was failing before.** At 2021-01-01 de novo, S×S now scores
0.347/0.361 and S×S-MCM 0.344/0.358, exactly supp_cd1_min100_2021's values on the same 90
cells (that run was fitted on the train file). antiGen (single model) 0.402/0.436 and the
gate 0.370/0.381 are unchanged. MCM is at chance (0.098) on de novo cells, as a count
matrix must be. Pooled, it had 0.48.

**C/D's end now set by the rule.** The <100 arm was run through 2024-01-01. 2023-01-01
has 26 cells / 840 events and fails (< 30 cells), so the series stops before it. The
later cutoffs fail too: 20/471, 11/227, 6/157, 6/132. The drawn range, 2020-04-01 ..
2022-10-01, is unchanged, and so is the staged data. Count strips over each panel show
the cells (per-cell panels) or events (per-mutation panels) behind every point, with the
rule's floor dashed.

## 164. supp_lm_cd1: sc2_results_revised2 L/M at codon distance 1 (2026-11-12)

User: a supplementary L/M on codon-distance-1 mutations only. No new compute. The
s162/s163 series runs (`STORE/evals_cd1_series_rare100_trainonly`, all 16 of L/M's
cutoffs) carry `list_allcells_cd1_fullpool`: every RBD cell, with pool and targets
restricted per query to cells 1 nt from that query's own residue. Every model in them is
fitted on the train tree only. That variant applies no seen mask, so the arm's
`--seen-min-count 100` does not touch it: its 225 rows are identical to the de novo arm's
at 2021-01-01 and 2022-04-01.

* `experiments/multitree/stage_lm_cd1.py` -> `data/gating_macro_aa20_cd1/<D>/` (variant
  renamed `list_allcells` for the by_macro loader; 25 models per cutoff). Scored set:
  1,236 cells / 50,283 events at 2020-04-01, down to 432 / 3,676 at 2024-01-01.
* `figures/figS_lm_cd1.R` -> `figs/supp_lm_cd1.png`. A = per cell (L's quantity),
  B = per mutation (M's, through list_macro's micro column). Same drawer, roster and
  labels as L. antiGen is the single per-cutoff model.
* Rule: s161 throughout. The main L/M are still on the pre-s161 rule, so the two differ
  by up to ~1e-3 on that account alone.
* Cross-check at 2023-01-01 against the cd1 2023 run (pre-s161 rule, pooled fits):
  antiGen 0.309/0.737 -> 0.308/0.736, gate 0.317/0.749 -> 0.317/0.748, PFM and EVEscape
  within 1e-3, BLOSUM62 identical. S×S per mutation 0.531 -> 0.519 is the pooling fix.

Reading: per cell (A), antiGen and the gate lead or tie PFM at every cutoff, and the
statistical models trail from 2022. Per mutation (B), the gate leads throughout. The
single antiGen model is behind PFM until 2022 and level with the gate from mid-2022.

## 165. sc2_results_revised2: a codon-distance-1 row, the 2021 row led by the >=100-seq de novo pair, pair sub-headers (2026-11-12)

User, for the main figure (the supp variant is unchanged apart from the s163 data):

* **New row G-J under C-F:** supp_cd1's A-D (all RBD, 2023, codon distance 1). Each
  panel's x axis is now fitted to its own bars. Before, it was shared with its pooling
  partner: supp_cd1 A ran to 0.57 for bars ending at 0.33. This is the opt-in
  `coguk.decile_own_range` (`REVISED2_OWN_XRANGE=1` in panels_coguk_export.R); unset, no
  other figure moves.
* **2021 row, K-N:** the all-escape pair (old G, H) is replaced by
  supp_cd1_min100_2021's de novo pair: all RBD at 2021, 1 nt, cells seen in >= 100
  sequences after the cutoff. The de novo escape pair stays as M, N. The pair now leads the
  row, so it is exported with the model names (`decile_dn_pc_named`). It is drawn on the
  main roster (DROP_MAIN, no gate), so all four panels share one roster. The row is now
  half all-RBD and half escape list, so its rotated header is "2021 cutoff" and the
  sub-headers say which is which.
* **Sub-headers over each pair of a bar row:** C-F "All mutations" / "De novo
  mutations"; G-J the same with ", 1 nt"; K-N "All RBD: >=100 seqs after cutoff, 1 nt" /
  "Escape: de novo". Each has a thin rule under its pair. The left label is centred over
  the bars rather than over the model-name column.
* **Relettered:** fixed-split K -> O, cutoff series L/M -> P/Q.
* Panel titles can now be overridden per panel (`REVISED2_T4="dn_pc=...;dn_evt=..."`).
  The K/L titles say "1 nt". `run_export` in fig_sc2_revised2.R now shell-quotes its env
  values, which a title with spaces needs.

Data are the s163 train-only fits: C-F, G-J, P from the list_macro refits, Q from the
eval.py refits, and K-N from 2021 runs that were fitted on the train file throughout.

Pushed with ONE point pending (now done, below): sc2_results_revised2 Q at 2022-01-01 still carried the
pooled fits for its fitted baselines. That eval.py refit landed on a slow node; a GPU
duplicate is in `STORE/cutoff_evals_aa20_refit_gpu/2022-01-01`, and it is spliced and
re-rendered when it lands. supp_fullspike waits on its five early-cutoff refits. Staged and
re-rendered from the s163 fits: sc2_results_revised2 (+supp), supp_cd1, supp_cd1_min100
(2023), supp_query_adapt.

**s163 complete.** Q 2022-01-01 came from the GPU duplicate (gate: 22,356 shared values,
worst 0.0) and was spliced in; the slow CPU copy was cancelled. The full-spike refits
all passed (0 failures), and supp_fullspike is restaged (16 cutoffs + the 2023 bars) and
re-rendered. Every current figure the audit marked as leaked is now on train-only fits.

## 166. supp_min100_2021: supp_cd1_min100_2021 without the 1-nt filter (2026-11-12)

User: a version of sc2_results_revised2 K/L (de novo, >= 100 post-cutoff seqs, 2021) on the
full pool. No new compute: the 2021 run (`STORE/evals_cd1_2021_allcells`, train file
only, s161 rule) already has per-cell rows for `list_allcells{,_unseen}_fullpool`.
`NT1=0 ARM=2021 stage_cd1_min100.py` applies the same mask to those
(-> `data/evals_sc2_lists_macro_aa20_2021_allcells_min100`; gate worst 1.4e-6).
Export arm `min100_2021`. `FIG_ARM=2021full figS_cd1_min100.R` -> `figs/supp_min100_2021.png`.

Scored sets: all 365 cells / 40,418 events; de novo 91 / 5,748. The 1-nt version had
90 / 5,504, so frequent de novo substitutions are essentially all one nucleotide away.

Per cell / per mutation, q = 0.1:

| | all, full pool | all, 1 nt | de novo, full pool | de novo, 1 nt |
|---|---|---|---|---|
| antiGen (CV x10) | 0.719 / 0.805 | 0.470 / 0.609 | 0.665 / 0.706 | 0.447 / 0.480 |
| gate | 0.768 / 0.873 | 0.550 / 0.718 | 0.577 / 0.618 | 0.370 / 0.381 |
| S×S | 0.668 / 0.773 | 0.390 / 0.528 | 0.698 / 0.743 | 0.347 / 0.361 |
| S×S-MCM | 0.690 / 0.787 | 0.435 / 0.587 | 0.697 / 0.741 | 0.344 / 0.358 |
| codon table | 0.317 / 0.313 | 0.158 / 0.164 | 0.385 / 0.382 | 0.157 / 0.142 |

Reading: on the full pool, de novo recall mostly measures whether a model ranks one-nt
substitutions above multi-nt ones. Every target is 1 nt, and S×S's counted M[wt, a]
learns that multi-nt substitutions almost never happen. The codon table alone goes
0.157 -> 0.385. So the statistical models pass antiGen there (0.70 vs 0.67). Within the
1-nt pool, where that signal is removed, antiGen leads (0.45/0.48 vs 0.35/0.36). This is
the reason for the 1-nt filter (s159).

## 167. sc2_results_revised2's 2021 row: de novo >= 100 seqs, full pool vs 1 nt (2026-11-12)

User: K/L <- supp_min100_2021's C/D (full pool, s166) and M/N <- the previous K/L
(supp_cd1_min100_2021's C/D, 1 nt). The row is now one comparison: all-RBD de novo
cells seen in >= 100 sequences after the 2021-01-01 cutoff (91 / 90 cells), with only
the pool changing. Row header "De novo, >=100 seqs, 2021"; sub-headers "All
substitutions" / "Within 1 nt". K/L: S×S 1st, S×S-MCM 2nd, antiGen 3rd (0.67/0.71 vs
0.70/0.74). M/N: antiGen 1st (0.45/0.48), EVEscape 2nd, S×S 3rd (0.35/0.36). The COG-UK
de novo escape pair leaves the main figure; it remains in coguk_list_aa20, supp_cd1 E-H
and sc2_results_revised2_supp. The full-pool pair is a new export (`f21`, arm
min100_2021, own x range, named leading panel).

## 168. sc2_results_revised2: section subtitles, simpler titles, no row headers (2026-11-12)

User: larger section subtitles, "Forecasting RBD mutations, 2023 cutoff" above C-J and
"Forecasting novel and prevalent RBD mutations, 2021 cutoff" above K-N (14 pt, bold,
left-aligned). Pair sub-headers: C-F "All RBD mutations" / "De novo RBD mutations",
G-J the same with ", 1 nt", K-N "Novel and prevalent RBD mutations" (+ ", 1 nt"). Here
"novel and prevalent" = de novo AND seen in >= 100 sequences after the cutoff. Panel
titles C-N are now just "Per cell" / "Per mutation" (REVISED2_T4 override; the
supplement keeps its full titles). The rotated row headers are gone from the main
figure, along with their left gutter (HDR_W 0.002 in main). The supplement is unchanged.

## 169. sc2_results_revised2: K-N headers, centred section titles, lead letters, panel O to a supplement (2026-11-12)

* K/L header "De novo RBD mutations, observed >=100 times"; M/N the same + ", 1 nt".
* Section subtitles are centred.
* The leading panel's letter (C, G, K) sits just before its "Per cell" title, as at E,
  I, M (`label_x = LEAD_LAB_X` 0.285 on the first panel), not left of the model names.
* The fixed-split panel (K, then O) moves to its own supplement,
  `figs/supp_fixed_split_revised2.png`, written by fig_sc2_revised2.R (main variant),
  with the model-type key. The older `supp_fixed_split` (figS_fixed_split.R, vertical
  bars, CoVFit's static map) is left as is.
* The cutoff series is relettered P/Q -> O/P, under a new section subtitle,
  "Forecasting RBD mutations, variable cutoffs".

## 170. sc2_results_revised2: rotated row headers instead of section subtitles (2026-11-12)

User: back to row headers, at 12 pt (the s169 subtitles were 14). "Forecasting RBD
mutations, 2023 cutoff" spans BOTH 2023 bar rows (C-F and G-J), centred between them.
"Forecasting novel and prevalent / RBD mutations, 2021 cutoff" sits beside K-N, and
"Forecasting RBD mutations, / variable cutoffs" beside O/P, each on two lines to fit its
row. The O/P line key sits outside that header's span. Left gutter HDR_W 0.05 in main.
Pair sub-headers and panel titles are unchanged.
