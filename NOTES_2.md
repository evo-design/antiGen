# NOTES_2

Running log of work on the `is-dev` branch, continued from `NOTES.md` (7,100+ lines and
too long to keep appending to). Same convention: newest entry last, one entry per round,
written so that a reader who was not there can tell what was done, what was measured, and
what is still assumed rather than known.

`NOTES.md` remains the record for everything up to and including 2026-09-23. Start here
for anything after.

---

## 2026-09-23 — per-cutoff antiGen training on H3N2, one job per backbone-eval origin

### The ask

> For the H3N2 data (backbone tree + additional sequences), train the model with a
> train-test split at each cutoff date we've been using on our backbone eval, i.e. the
> .25 year mark within the range of years we've been studying. Train for a fixed 32
> epochs, as we have typically been doing. (Yes, we now will be running the tree
> augmentation pipeline that for a while we have not needed.) Submit a separate job for
> each cutoff. Use whatever GPU partition you want except `evo_gpu_priority`, whatever
> will do this fastest. No need to eval the model, just train.

So this is the *generative* antiGen model -- the from-scratch transformer over
`MUTATIONS_TO_CHILDREN` targets, `README.md` parts 1-2 -- not the forecasting work the
rest of `NOTES.md` is about. The two share the H3N2 backbone tree and nothing else. The
augmentation pipeline is back in play: the season/clade rounds ran `--no-augment` because
they wanted the guide tree's own tips, and this round wants the full attached tree.

### The cutoff arithmetic, which is the one place this could quietly go wrong

`antiGen-data` takes its cutoff in **years after the tree's root date**, not as an
absolute decimal year. The backbone eval's origins are horizons in absolute time, so they
have to be shifted.

- `season_of` (in `clade_cluster.py`): season *Y* runs 1 Oct *Y-1* to 1 Apr *Y*.
- `Task.horizon = season + 0.25`, i.e. 1 April of the origin season.
- Backbone eval origins: seasons 2017, 2018, 2020, 2021, 2022, 2023, 2024, 2025 --
  eight of them, predicting targets 2018, 2019, 2021-2026. Season 2020 is dropped as a
  *target* (the COVID season), which is why 2019 is not an origin.
- Root date of `~/h3n2/tree.json`: **2007.758** (`NODE_0000001`), confirmed both from the
  tree JSON and from `cache_h3n2.manifest.json`'s `root_time`.

| origin season | horizon (1 Apr) | `--cutoff` = horizon − 2007.758 |
| ---: | ---: | ---: |
| 2017 | 2017.25 | 9.492 |
| 2018 | 2018.25 | 10.492 |
| 2020 | 2020.25 | 12.492 |
| 2021 | 2021.25 | 13.492 |
| 2022 | 2022.25 | 14.492 |
| 2023 | 2023.25 | 15.492 |
| 2024 | 2024.25 | 16.492 |
| 2025 | 2025.25 | 17.492 |

**Assumption, flagged rather than buried: 2019.25 (cutoff 11.492) is excluded.** The ask
says "each cutoff date we've been using on our backbone eval", and 2019 has never been an
origin there -- its target season 2020 was declared anomalous and dropped from
evaluation, validation and training alike. It is one more job if that reading is wrong.

`std::ostringstream << cutoff` at default precision gives six significant digits, so the
files land as `examples_cutoff_9.492.txt` and so on, exactly -- no float-formatting
surprise to work around.

### The inputs, and what was verified before anything was submitted

Three alignment segments, from `cache_h3n2.manifest.json` rather than guessed:

```
~/h3n2/ref.json, ~/h3n2/tree.json, ~/h3n2/metadata.tsv
~/h3n2/output/nextclade.cds_translation.{SigPep,HA1,HA2}.fasta
```

**The binary was rebuilt** from `antiGen/antiGen-data/`, whose working tree carries
uncommitted changes (+1,797 lines across `main.cpp`, `types.h`, `Makefile`) that the
copy at `~/antiGen-data/antiGen-data` (built 6 July) predates. CPU-only build: the login
node has no `nvcc`. That turns out not to matter -- see the cache below.

**The attached-tree cache is valid and turns generation into a no-op.**
`~/antiGen-data/cache_h3n2` (28 MB, 19 Sep) loads under the freshly built binary:
44,159 nodes, and a full generation at cutoff 9.492 takes **2 seconds**. The cache
manifest carries no cutoff field, and the attachment happens before the split, so one
cache serves all eight cutoffs. That matters for more than speed: the cache is only
*written* on a miss, so eight concurrent jobs all hitting it is race-free, whereas eight
concurrent jobs all rebuilding it would have raced on the same output file. Verified by
running one before submitting any.

Cutoff 9.492 produces 26,877 examples -- 7,422 `TRAINING`, 19,455 `TESTING` -- from
34,655 training and 169,074 testing samples. Later cutoffs will be larger.

### Two operational traps hit on the way

1. **The session scratchpad is node-local `/tmp`.** A first test job wrote its script and
   logs there, "completed" in 5 seconds and produced nothing, because the compute node
   cannot see the login node's `/tmp`. Everything for this round lives on
   `/large_storage/hielab/ispecht/antigen-h3n2-cutoffs/` instead.
2. **`$HOME` is at 97% (31 GB free).** Examples files are ~20 MB each and checkpoints are
   small, so eight of them would fit -- but not with any margin, so outputs go to
   `/large_storage` regardless.

Also worth recording: `srun` from inside this session lands as a *step* of the enclosing
interactive allocation and inherits its (small) limits. Anything needing its own
resources has to go through `sbatch`.

---

## 2026-09-23 — the COVID mask, corrected: it is 2021 that is anomalous, the mask belongs on the eval only, and the headline table changes

### The ask, and the two corrections it took to land it

> Show me what the headline table on the backbone looks like if we include the 2020
> season, which we have been dropping (as anomalous).

Then, on seeing it: call **2021** the anomalous one instead. Then, on seeing that:
**no seasons removed from training — mask the EVAL only.**

Each correction mattered, because `--drop-targets` does not mean what the request
meant. It removes a transition "from evaluation, validation and training alike"
(`season_momentum.rolling`), and for `backbone_compare` it removes it from the rolling
six-transition refit window. So the flag conflates three separable interventions, and the
first two rounds of this entry measured the wrong ones. They are kept below because the
difference between them is the result.

### **THE HEADLINE BACKBONE TABLE IS NOW THIS ONE** — it supersedes section 3 of the 2026-09-22 HEADLINE RESULTS entry in `NOTES.md`

**Eval on season 2021 masked. Nothing held out of any fit.** The selector's validation
folds and every origin's training window see all nine transitions in the record, 2020 and
2021 alike; the 2020→2021 row is excluded from the averages and from nothing else. No
refit is involved — the per-transition numbers are the `--drop-targets ""` run's, with one
row not counted.

8 scored transitions: 2017→2018, 2018→2019, **2019→2020**, 2021→2022, 2022→2023,
2023→2024, 2024→2025, 2025→2026. Persistence 7.970 AA, floor 3.389. Persistence agrees
between our run and theirs to **5.3e-15** and between ours and VaxSeer's to **7.1e-15**,
so the rows are paired and not merely adjacent.

| source | model | mean AA | vs persistence | wins | ours − theirs |
| --- | --- | ---: | ---: | ---: | --- |
| H2020 | `ne_star` | **7.494** | **+5.35%** | 7/8 | +0.168 [−0.25, +0.63] |
| H2020 | `distance_from_consensus` | 7.603 | +4.66% | 4/8 | +0.059 [−0.26, +0.37] |
| H2020 + H2025 | `ne_star-lbi` | 7.610 | +3.59% | 5/8 | +0.052 [−0.35, +0.50] |
| **this work** | **`mlr+mom`** | 7.661 | +3.86% | **8/8** | — |
| this work | `mlr` | 7.717 | +2.96% | 7/8 | −0.056 [−0.26, +0.14] |
| H2020 | `lbi` | 8.144 | −2.56% | 2/8 | −0.483 [−0.77, −0.24] |
| this work | `ewma` | 8.198 | −2.47% | 3/8 | −0.537 [−0.79, −0.25] |
| H2020 | `ep_x-ne_star` | 8.493 | −7.15% | 4/8 | −0.831 [−2.61, +0.70] |
| H2020 | `ep_x` | 8.813 | −10.07% | 4/8 | −1.152 [−2.55, −0.14] |
| BC2021 | consensus, point mass | 8.854 | −11.44% | 3/8 | −1.192 [−2.31, −0.01] |
| Neher 2014 | top-LBI strain, point mass | 10.645 | −35.10% | 1/8 | −2.984 [−4.35, −1.57] |
| **Shi 2025** | **VaxSeer (self-trained)** | **pending** | | | origin 2019 training, job 3002439 |
| — | persistence | 7.970 | | | |
| — | floor | 3.389 | | | |

**What changed against the published table, stated plainly: we are no longer top of the
sequence-only field on the mean.** `mlr+mom` goes 8.220 / +5.22% / 8-of-8 to
7.661 / +3.86% / 8-of-8, and three of their metrics move above us on the mean. **None of
the three is significantly above us** — +0.168, +0.059, +0.052 AA, every paired interval
straddling zero, against our 8 of 8 transitions versus their 7, 4 and 5. So finding 3 of
the headline entry ("what separates our model is stability, not level") survives intact
and is now the *only* thing separating us; finding 2 ("a one-coefficient count of sequence
differences matches a 255-coefficient learned model") is strengthened, since on this panel
`ne_star` and `distance_from_consensus` are both nominally ahead.

Per transition, one fit, the masked column in parentheses:

| model | 2018 | 2019 | 2020 | *(2021)* | 2022 | 2023 | 2024 | 2025 | 2026 | mean of 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 5.688 | 7.325 | 7.574 | *13.367* | 9.969 | 7.487 | 10.114 | 4.591 | 11.014 | **7.970** |
| `mlr+mom` | 5.571 | 6.883 | 7.497 | *13.958* | 9.032 | 7.118 | 9.875 | 4.399 | 10.917 | **7.661** |
| `mlr` | 6.184 | 6.466 | 7.380 | *14.147* | 9.360 | 6.935 | 10.007 | 4.454 | 10.950 | **7.717** |
| `distance_from_consensus` | 5.810 | 7.358 | 7.673 | *12.544* | 8.493 | 6.682 | 10.525 | 3.987 | 10.296 | **7.603** |
| `ne_star` | 5.395 | 6.916 | 7.596 | *13.108* | 9.761 | 7.487 | 8.643 | 4.390 | 9.763 | **7.494** |
| `ne_star-lbi` | 6.082 | 7.455 | 7.237 | *12.985* | 9.504 | 7.499 | 8.703 | 4.310 | 10.085 | **7.610** |
| `lbi` | 6.090 | 7.771 | 7.780 | *13.783* | 10.325 | 7.499 | 10.051 | 4.684 | 10.953 | **8.144** |
| floor | 2.430 | 3.176 | 2.774 | *4.714* | 3.533 | 2.604 | 2.843 | 2.457 | 7.294 | **3.389** |

**The masked row is the only transition in the whole record where we lose**: 2020→2021,
`mlr+mom` 13.958 against persistence 13.367, −4.43%. That is what buys back the 8/8 and
it should be said out loud, because it is the same number read two ways — masking the one
transition we lose raises our win count and lowers our mean skill.

### Why season 2021 and not season 2020, measured rather than asserted

The `--drop-targets 2020` default has been carried since `NOTES.md` line 3122, where it
enters as "2020 dropped as anomalous" with no number behind it. There is none in the log.
The tip counts do not support it:

| target season | 2018 | 2019 | **2020** | **2021** | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequences | 79 | 74 | **84** | **35** | 60 | 96 | 102 | 94 | 68 |

Season 2020 is at 84, squarely inside its neighbours' 74–102. Season 2021 is at 35 — the
season influenza went near-absent worldwide, which the 2026-09-22 COVID entry already
identified and which is the season the counts actually single out. The defensible case for
masking 2020 is that surveillance collapsed part-way through it, so the observed
population is a truncated sample of the real one; that is an argument about the target's
validity, not about thinness, and it was never written down. **The mask moves to 2021.**

### The three ways to mask, and why only one of them is legitimate

Every arm below is the same standing protocol — same grid, same selector, same L1, their
coefficients refitted at every origin — differing only in what the mask removes.

| arm | `mlr+mom` | `ne_star` | `d_cons` | `ne_star-lbi` | persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| published: 2020 out of training, validation and scoring | 8.220 / **+5.22%** / 8/8 | 8.286 / +4.52% / 7/8 | 8.211 / +5.65% / 5/8 | 8.455 / +1.77% / 5/8 | 8.694 |
| nothing masked, 9 rows scored | 8.361 / +2.94% / 8/9 | 8.118 / +4.97% / 8/9 | 8.152 / +4.83% / 5/9 | 8.207 / +3.51% / 6/9 | 8.570 |
| 2021 out of training, validation and scoring | 8.072 / **−0.28%** / 5/8 | 7.672 / +3.53% / 6/8 | 7.632 / +4.23% / 4/8 | 7.805 / +1.35% / 4/8 | 7.970 |
| **eval on 2021 masked only (the new headline)** | **7.661 / +3.86% / 8/8** | 7.494 / +5.35% / 7/8 | 7.603 / +4.66% / 4/8 | 7.610 / +3.59% / 5/8 | 7.970 |

Two things to read off this, both about our model rather than about the data.

**1. Putting 2019→2020 into the training windows costs us 2.04 points, and it is one
origin.** Going from the published arm to the 9-row arm, our skill on the *eight shared
transitions* falls +5.22% → +3.18% before a single new row is averaged in; adding the
ninth row accounts for only the remaining 0.24. Of the +0.249 AA mean degradation on
those eight, +0.228 is origin 2020 alone (12.133 → 13.958), whose `window=2` training set
becomes {2018→2019, 2019→2020} instead of {2017→2018, 2018→2019} — the model learns from
the season COVID truncated and then forecasts into the bottleneck. Their metrics move by
+0.001 to −0.127 over the same change, because one or two coefficients fitted over six
transitions do not care which two sit nearest.

**2. Masking a season out of *training* is catastrophic for us and free for them.** The
third arm punches a two-year hole in the record and our model goes below persistence,
−0.28%, last of the fitted models, with `ne_star` beating us at +0.400 AA **[+0.05, +0.78]
— the first interval excluding zero any of their models has achieved on the backbone**.
The damage localises exactly where the theory says: 2021→2022 +1.791 AA and 2022→2023
+1.288 AA, the two origins whose two-transition window straddles the hole, together 3.08
of the 3.45 AA total. This reproduces treatment C of the 2026-09-22 entry (+5.22% →
−0.02%) by an independent route and confirms it was not an artefact of that treatment's
`--min-season-seqs 50` mechanism.

**So the eval-only mask is the only one of the three that answers the question actually
being asked.** A season declared anomalous as an *observation* is a season we should not
be graded on; it is not a season that stops having happened, and removing it from the
fits is a different and much larger claim. The published arm made that larger claim
without meaning to, and 2.04 of its 5.22 points came from it.

### VaxSeer: present in the figure, absent from the entry, and missing one origin

`Shi et al. 2025` was omitted from the first three tables of this round because the row
list was copied from the 2026-09-22 headline entry, which predates the VaxSeer round by a
day; `eval_figures.py:124` has carried it since. Two things follow.

**It cannot be scored on the new panel yet.** VaxSeer is trained per origin, so its eight
values carry over verbatim under any re-masking — but it was built for the published
panel's origins (2017, 2018, 2020–2025) and has **no origin 2019**, which is precisely the
origin the new panel adds. That run is now going: `process_fasta` at `--end_date 2019-04`
gives 19,536 sequences (between 2018's 15,375 and 2020's 23,448) and `last_time_bin 92` =
(2019−2003)·6 − 4, the no-leak arithmetic verified per origin in the VaxSeer entry;
`support/origin2019.fasta` holds 433 genotypes, matching `n_support` from our own run.
Train job **3002439**, scoring chained as `--dependency=afterok` (**3002440**) rather than
a nohup watcher, which is the operational lesson from that entry.

**On the seven origins where it can be paired, masking 2021 does not rescue it.** Panel:
2017→2018, 2018→2019, 2021→2022, 2022→2023, 2023→2024, 2024→2025, 2025→2026; persistence
8.027, floor 3.477. VaxSeer 9.048 (−10.28%, 5/7), last on the mean, against `ne_star`
7.479 (+6.16%), `d_cons` 7.593 (+5.51%), `ne_star-lbi` 7.663 (+3.47%), **ours 7.685
(+4.27%, 7/7)**, `mlr` 7.765, `lbi` 8.196, `ewma` 8.322.

That is worth more than a row. The COVID entry attributed VaxSeer's collapse to the
bottleneck, but **its two catastrophic transitions are 2020→2021 (16.663) and 2021→2022
(19.909), and only the first is the bottleneck.** With 2020→2021 masked, 2021→2022 alone
carries the entire −10.28%: +10.877 AA against us, on a panel where its other six
transitions mean 7.238 and it wins 5 of 7. **The model breaks on the bottleneck and on
the rebound out of it**, and the mean/median disagreement recorded for the full panel
survives the mask rather than being explained by it. The claim from that entry stands and
gets sharper: on ordinary transitions a 12-layer autoregressive model over 200k sequences
is level with a 255-coefficient linear fit and with a one-coefficient count; across a
regime change in either direction it is far worse than doing nothing.

Dropping 2019→2020 to make the panel paired barely moves the other models (`mlr+mom`
7.661 → 7.685, `ne_star` 7.494 → 7.479, `d_cons` 7.603 → 7.593), so the 7-row and 8-row
readings agree for everything except VaxSeer.

### What is now stale elsewhere

- **`NOTES.md` section 3 of the 2026-09-22 HEADLINE RESULTS entry** (the 8-transition
  backbone table, `mlr+mom` 8.220 / +5.22% / 8-of-8) is superseded by the table above. A
  pointer has been added there. Sections 1 and 2 — the H2020 panel, 1996–2014 and
  2015–2019 — are untouched by this round; nothing about them involves the backbone
  masking, and **the headline claim still rests on section 1**, which remains the only
  comparison out-of-sample for both sides.
- The sentence "All years kept. Excluding the anomalous 2021 season from scoring moves us
  to +4.57% (6/6)" in that section describes treatment B on the *2020-masked* panel and is
  still true of that panel, but it is no longer the sensitivity that matters — the
  eval-only mask on the full-training panel is.
- **`eval_figures.py` regenerates the backbone figure from `dfc/` and `backbone3way/`,
  i.e. the published masking.** It has not been repointed. Doing so needs the VaxSeer 2019
  row first, or the figure loses a series; it is the obvious next task and is not done.
- The 2026-09-22 COVID entry's treatments A/B/C are not withdrawn. A is now understood to
  have been "2020 out of the fits as well", which was not the intent of "all years kept",
  and that entry's own decision text ("dropping seasons for anomaly is a judgement call
  that a reader cannot audit") is the argument for the change made here.

### Caveats that govern the new table

- **The backbone is still our development set.** `mlr+mom` was designed and tuned across
  dozens of rounds on the 2020-masked version of this panel. The re-masking was not
  chosen to flatter it — it costs us 1.36 points and the top slot — but the model's
  hyperparameter *grid* was shaped on the old panel and that does not undo itself.
- **Nothing in the top five rows is separable at this n.** Eight origins on 6-month
  pivots with a 12-month horizon is about four independent years; the four models between
  7.494 and 7.717 AA are a tie, and the ordering among them must not be reported as a
  ranking.
- **Three maskings were computed and one was chosen after seeing all three.** The choice
  is defended on the principle (do not remove data from a fit in order to excuse a
  forecast) and on the tip counts, both of which are stated above and auditable — not on
  which table came out best, and in fact it is not the one that came out best for us.

### Artifacts

- `~/antiGen-count-loss-runs/covid2020/` — `ours.slurm` (job 3002149, 9 origins,
  `--drop-targets ""`), `theirs.slurm` (3002150), `out/c20_*.json`,
  `theirs_with2020.json`, `summarize_c20.py`, **`summarize_evalmask.py`** (prints the new
  headline table, the two comparison arms and the masked row),
  **`summarize_evalmask_vax.py`** (the same with VaxSeer, on both the 7-origin paired
  panel and the full 8).
- `~/antiGen-count-loss-runs/covid2021/` — `ours.slurm` (3002382), `theirs.slurm`
  (3002383), `out/c21_*.json`, `theirs_drop2021.json`, `summarize_c21.py`. The
  mask-from-training arm; kept because the −0.28% is the measurement, not a failed run.
- `~/antiGen-count-loss-runs/vaxseer/` — `processed/2003-10_to_2019-04_2M/`,
  `support/origin2019.fasta`, `score2019.slurm`; jobs 3002439 and 3002440 outstanding.
- No code changed. Every arm is the existing `season_momentum.py` and
  `backbone_compare.py` under a different `--drop-targets`, and the eval-only mask is a
  summariser that drops a row it does not refit.

---

## 2026-09-23 — convolving `mlr+mom` with antiGen's forward pass: the selector picks `mu = 0` everywhere, and so does the oracle

### The ask

> Start with `mlr+mom`, and suppose it predicts fraction `x_v` for sequence `v` in next
> year's season. Do a forward pass of trained antiGen on `v`. Then reweight `x_v` to
> `(1 - mu) x_v`, and for each single-AA neighbour of `v`, add mass
> `mu * x_v * [forward pass probability assigned to the substitution]`. Select `mu` using
> validation. Also evaluate whether there's some oracle value of `mu` for which we can
> improve performance, even if selecting that value is impossible. [...] Ignore any
> previous mutation_kernel work and start fresh.

This is the convolution of two distributions over different things: the forecasting side
predicts over genotypes already sequenced, the generative side predicts, for one
genotype, which substitution its children carry. `antiGen-infer`'s output is exactly
`pi(q, b | v)` — one softmax over the whole (position x residue) map with each position's
wild-type cell removed, renormalised — so no new quantity had to be defined.

**The answer is a clean and complete negative.** On all nine origins of the standing
backbone panel, at every one of twelve mixing weights, the convolution makes the forecast
worse; the validation selector returns `mu = 0` at all eight scored origins, the
per-transition oracle returns `mu = 0` at all nine, and a single panel-wide oracle `mu`
also returns 0. This is not a selection failure — there is nothing to select.

### What was built (fresh, as asked)

- **`antiGen/antigen_kernel.py`** — the proposal and the cost block. `forward_probs` is
  `antiGen.infer.score_batch` written out where it is used and asserted equal to it;
  `checkpoint_for` resolves an origin to the checkpoint whose train/test split is that
  origin's own horizon and **raises** when it is absent, because a neighbour's checkpoint
  would be trained past the horizon it is forecasting from.
- **`antiGen/antigen_convolve.py`** — the driver. The parent is held at exactly what the
  standing protocol selects for `mlr+mom` (same grid, same `top_k = 5` averaging, same
  window), so the only thing chosen here that was not chosen before is `mu`, over twelve
  candidates rather than over a grid multiplied by a new field. `mu` is selected on the
  selector's own validation folds — the `val_folds = 3` transitions immediately before
  the one being predicted, each forecast by the same combos and convolved by **its own**
  origin's checkpoint.
- **`antiGen/tests/test_antigen_kernel.py`** — fifteen tests, all passing, and the 136
  existing `mutation_kernel` / `season_momentum` / `backbone_compare` tests still pass.
- Two lines of wiring in `season_momentum.py`: `Task.candidates` dispatches to the new
  module for the `kagen` arm, and `--antigen-checkpoints` / `--antigen-sites-only` /
  `--antigen-drop-gap-stop` reach the panel. So `--families mlr+mom+K --kernels kagen`
  also works and is the *other* reading of the ask, in which `mu` joins the grid and is
  chosen jointly with the parent's hyperparameters. **That arm has not been run** — see
  "what is not done".

Nothing in the previous kernel round's arms, gate or break-even bar is used. The only
things imported from `mutation_kernel` are `Forecast`, `convolve` and `golden_mu`, which
are generic transport plumbing rather than a kernel, and `Task.score` requires the first
of them.

### The three checkpoints the retrain array did not cover

The 2026-09-23 training array covered the eight origins of the *published* panel
(2017, 2018, 2020–2025). The rolling protocol also computes a forecast at every
**validation** origin, and the eval-only-2021-mask panel scores origin **2019** as well,
so three more cutoffs were needed: 2015 (7.492), 2016 (8.492), 2019 (11.492). All three
are trained, same recipe, 32 epochs. `order` on this panel is 2015..2025 and `min_train`
is 2, so origins 2016–2025 all get a forward pass; 2015 is only ever a training
transition and its checkpoint is never read.

**One operational scar worth recording.** Two identical fill arrays were submitted within
thirteen seconds of each other, from this session and from elsewhere, writing the same
`examples_cutoff_*.txt` paths. One of them truncated the file while the other's
`antiGen-train` was reading it: `h3n2fill_3002484_1` validated 26,338 examples and then
loaded 4,543 (3,837 TRAINING + 706 TESTING against a true 3,837 + 22,501). Only the
TESTING side was lost, so that model's weights are fine, but the near miss is the lesson:
the generator's output path carries no run id, so two concurrent jobs at the same cutoff
race. Season 2015's directory now holds two checkpoints; `checkpoint_for` picks the last
by name, says so, and 2015 is never read anyway. The other eight fills read their files
whole (26,911 for 2019, 27,122 for 2016) and are clean.

### The measurement

Standing panel: `--drop-targets ""`, nothing held out of any fit, target 2021 masked from
the averages only — the headline panel of the entry above. Parent grid, `top-k 5`,
`val-folds 3`, field for field `covid2020/ours.slurm`.

**`mu = 0` reproduces the headline exactly: 7.6615 against the published 7.661.** That is
the nesting check, and it is the reason the rest of the table can be read as a difference
rather than as a different run.

Panel means over the eight scored transitions, all three arms:

| arm | `mu = 0` | `mu-hat` (validation) | `mu*` (oracle, per transition) | `mu` common (oracle, one for the panel) |
| --- | ---: | ---: | ---: | ---: |
| full proposal | **7.6615** | 7.6615 at `mu = 0`, 8/8 | 7.6615 at `mu = 0`, 0/8 improved | 7.6615 at `mu = 0` |
| varying sites only | 7.6615 | 7.6615 at `mu = 0`, 8/8 | 7.6615 at `mu = 0`, 0/8 improved | 7.6615 at `mu = 0` |
| gap/stop dropped | 7.6615 | 7.6615 at `mu = 0`, 8/8 | 7.6615 at `mu = 0`, 0/8 improved | 7.6615 at `mu = 0` |

The whole curve, averaged over the eight scored transitions (full proposal):

| `mu` | 0 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMD | 7.6615 | 7.6687 | 7.6759 | 7.6975 | 7.7335 | 7.8055 | 7.9501 | 8.0964 | 8.2436 | 8.3927 |
| vs `mu = 0` | — | −0.09% | −0.19% | −0.47% | −0.94% | −1.88% | −3.77% | −5.68% | −7.60% | −9.54% |

**The curve is monotone increasing and, to within 1–5%, linear in `mu` at every single
origin.** Per transition, the ratio of the slope at zero to the total rise over `[0, 1]`
is 0.95–1.00 (0.990, 1.000, 0.995, 0.994, 0.982, 0.953, 0.976, 0.990, 0.989 for origins
2017–2025). A convex function that is linear over its whole domain is one where the
transport plan never reorganises: the proposal mass is simply being carried one step
further from the target, and no part of it is ever close enough to change which target it
serves.

The panel slope at `mu = 0` is **+0.720 EMD per unit `mu`**, per origin 0.61 to 0.78,
against a worst case of exactly +1 by the triangle-inequality bound. So roughly
three quarters of every unit of mass the kernel moves goes strictly uphill and almost
none of it goes downhill.

### What that number means, and why it is worse than "it did not help"

`W_1` is convex in `mu` and its derivative at zero is `1 - p_flat - 2 p_down`, `p_down`
being the `x`-weighted share of proposal mass on substitutions that move a genotype
*closer* to the next season in the dual's geometry. A measured derivative of +0.72 puts
`2 p_down + p_flat` at 0.28, so `p_down` is at most 0.14 and likely well below it — against
the 0.446 a kernel needs to break even at all. **antiGen's per-genotype substitution
distribution is not merely uninformative about where next season goes; it is close to
the worst case a one-step kernel can achieve.**

This is not because the target is unreachable. The oracle floor over the support widened
by one substitution is **2.451** against the unextended **3.389** — widening the support
by one substitution would buy 0.938 EMD of headroom, a quarter of the standing 7.66 — and
the convolution takes none of it. The reachable set moves; the proposal does not point
into it.

**Both sensitivities are null.** About 9.8% of antiGen's proposal mass (8.7%–11.1% by
origin) sits on positions the panel never sees varying, where a substitution is one step
further from every target by construction. Renormalising that mass away — the generous
reading — moves the `mu = 1` mean from 8.3927 to 8.3652, i.e. it recovers 3.8% of the
damage and changes no selected or oracle `mu`. Dropping the `*` and `-` columns (0.2% of
the mass) moves the fourth decimal. Neither choice is load-bearing, which is worth knowing
because both were judgement calls made before any number came back.

### Caveats, stated rather than buried

- **One parent, one dataset, one metric.** This is `mlr+mom` on the H3N2 backbone under
  Hamming EMD. A kernel that is useless for relocating transport mass is not thereby
  useless for ranking substitutions, which is what antiGen's own evaluations measure and
  what Part 4 of the README reports.
- **The oracle is the strong form of the result.** `mu-hat = 0` alone would be consistent
  with a good kernel and a bad selector; `mu* = 0` at every origin, including the masked
  2020->2021, rules that out. There is no value of `mu` at which any transition improves.
- **The backbone is still the development set**, and the parent forecast was tuned on it
  over dozens of rounds. That cuts the usual way — it makes the parent hard to beat — and
  it is the reason `mu = 0` reproducing 7.661 is the check that matters here.
- **The checkpoints are 32-epoch runs of one recipe** (`d_model 128`, 2 layers, lr 1e-4).
  Nothing here says a differently trained antiGen would behave the same way, only that
  this one does, at nine horizons.

### What is not done

- **The joint-selection arm.** `season_momentum --families mlr+mom+K,mlr+mom+K@mu*
  --kernels kagen` puts `mu` into the selector's grid alongside the parent's fields. The
  plumbing is in and tested; it was not run because it costs about five hours per origin
  (5,400 collapsed combinations x 3 folds, most of them a 25k–50k-row transport solve)
  and, with the oracle at `mu = 0` everywhere, it can only return `mu = 0` more
  expensively. It should be run if anything about the proposal changes.
- **The descent diagnostic.** `mutation_kernel.descent` computes `p_down` and `p_flat`
  directly from the dual rather than inferring a bound on them from the slope, and it
  duck-types onto the new `Candidates`. One transport solve per origin. That would turn
  "`p_down <= 0.14`" into a number, and it is the obvious next thing.
- **No figure, and `eval_figures.py` is untouched.** Nothing here belongs in the headline
  table; it is a negative result about an extension, not a change to a reported model.

### Artifacts

- `antiGen/antigen_kernel.py`, `antiGen/antigen_convolve.py`,
  `antiGen/tests/test_antigen_kernel.py`; two small edits in `antiGen/season_momentum.py`.
- `~/antiGen-count-loss-runs/agenconv/` — `run.slurm` (job 3002573, three arms),
  `out/agenconv_{full,sites,nogapstop}.json` (per-transition curves at all twelve `mu`,
  the validation curve per fold, the selected and oracle `mu`, both floors), `logs/`.
- `/large_storage/hielab/ispecht/antigen-h3n2-cutoffs/checkpoints/season_{2015,2016,2019}/`
  — the three fill checkpoints, `train_fill.slurm` (job 3002484) and
  `train_gap_cutoffs.slurm` (job 3002488, the duplicate, cancelled part way).
