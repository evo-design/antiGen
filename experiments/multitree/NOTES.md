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
