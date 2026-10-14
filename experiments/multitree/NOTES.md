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
