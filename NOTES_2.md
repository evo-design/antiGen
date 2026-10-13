# NOTES_2

Running log of work on the `is-dev` branch, continued from `NOTES.md` (7,100+ lines and
too long to keep appending to). Same convention: newest entry last, one entry per round,
written so that a reader who was not there can tell what was done, what was measured, and
what is still assumed rather than known.

`NOTES.md` remains the record for everything up to and including 2026-09-23. Start here
for anything after.

---

## STANDING CONVENTIONS

Read before adding a number to any table. Each was adopted on the date given and applies
to everything reported after it; entries above that date describe the arm they were run
under and are not retroactively wrong, only superseded.

### 1. Every hyperparameter is selected causally (2026-09-28)

**No number this project reports may depend on a parameter chosen with information from
after its own forecast date.** Not the coefficients only -- the *hyperparameters*, the
grid, the kernel's operating point, every constant a run had a choice about. If a value
was picked by looking at how a transition turned out, that transition may not then be
scored with it.

The rule is about what a critic can say, not about what the number comes out as. A
forecast whose temperature was tuned on the season it is forecasting is not a forecast,
however small the effect turns out to be -- and in this project the effect has twice been
small, which is an argument for making the change, not against it:

- the kernel's `(tau_e, tau_k, m)` came from a descent gate pooled over five of the eight
  scored backbone transitions, and `p_down` is an oracle quantity. Resolved per origin
  instead, from gate rows whose targets had closed: **+0.0124 AA [−0.0735, +0.0865]**.
- the backbone feature set was chosen from allele frequencies over the whole record.
  Censored at each origin's horizon: **−0.0002 AA [−0.0020, +0.0019]**.

Practically, this means:

- `eval_figures` defaults to `--kernel-arm causal` and `--their-arm selected`. The
  superseded arms are reachable by flag, are labelled in the blurb wherever they appear,
  and are never the default.
- `season_momentum --kernel-gate-table` resolves the kernel operating point per origin
  and **refuses a gate table not stamped `causal_table`**.
- `backbone_compare --select` is on by default, and every output records which arm wrote
  it.
- A panel with no causal arm **raises** rather than serving the oracle-selected one
  unlabelled. Two currently do: the `published` masking panel has neither a
  `backbone_compare --select` run nor a `causalgate` run.
- Anything still exposed goes in the "still exposed" list of the entry that found it, and
  stays there until it is fixed or measured. As of 2026-09-28 that list is: the `--mus`
  ceiling (binding at 0.4 at every origin); `top_k`, `val_folds` and the grid ranges
  (shaped on the backbone over dozens of rounds); the eval mask (chosen after computing
  three); the H2020 kernel arm (running at a backbone-derived operating point, finding A2);
  and the backbone tree itself, which is one 2026 build and so makes their `lbi` and
  `ne_star` non-causal in a way only per-horizon tree rebuilds would fix.

### 1a. A momentum term selects both its timescales (2026-09-28)

Corollary of the above, recorded separately because it was missed for four days. The
genotype-level momentum column selects `tau_f` and `tau_s` independently; the site-allele
momentum the consensus kernel proposes with is the same construction at allele resolution
and had its slow window pinned at `CONS_SLOW_MULT = 3` times its fast one. A pinned ratio
is a selected parameter with a value nobody chose on evidence -- the 3 was borrowed from
the panel's `tau_s` grid and never revisited. **Both windows are now swept in the gate and
resolved per origin.** An arm that reads no slow table (`kcons`, `kemp`, `kunif`) gets a
single `None` entry rather than a sweep of a field it cannot see, which `reads_slow`
enforces and a test pins; otherwise the argmin among identical configurations would be
settled by iteration order.

### 2. Both sides get the same protocol (2026-09-24)

Their hyperparameters are re-chosen per origin by the same selector, folds and averaging
ours gets. Comparing a per-origin selector against constants frozen in 2019 compares
protocols as much as models. See the 2026-09-24 entry; it costs us two places in the
ranking, which is why it needs no further argument.

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

> **Pointer added 2026-09-24.** The *their-side* columns below are the published-config
> arm, with their hyperparameters frozen while ours were selected per origin. That
> asymmetry was found and measured in the 2026-09-24 causality entry at the end of this
> file, and **the reporting default has changed**: their models now get the same
> per-origin selector. On that default `lbi` is 7.677 / +4.20% / 5-of-8 rather than
> 8.144 / −2.56% / 2-of-8, and `mlr+mom` is third of six rather than fourth. The rows
> below remain correct for the arm they describe and are reachable as
> `eval_figures --their-arm published`.

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

### 7. VaxSeer, the third side of the table, which nothing had mentioned

The headline table is three-sided and a panel swap moves all three. There was no
additive-panel VaxSeer run and nothing in the previous two entries said so; it would have
been found at the end, when the table was assembled and one column had no rows.

**The checkpoints are not retrained, and the argument is stronger here than it was for
the tree swap.** `prep_data.py` builds VaxSeer's training corpus from
`nextclade.cds_translation.HA.fasta` + `metadata.tsv` — the *corpus*. The additive panel
is a subsample **of** that corpus and changes neither file. The 2026-09-28 swap had to
argue that the tree was not a training input; this one only has to observe that a
subsample of a corpus is not the corpus. The preflight asserts
`sha256(metadata.tsv) == manifest.corpus_sha` (`addf94424319`, corpus 2026-04-08) rather
than asserting it in prose.

What is panel-dependent is the support set and the score, and both are redone.
`make_support_fasta.py` and `score_vaxseer.py` are `competitors_t0924`'s with the panel
reader swapped for `add_panel_source_args`/`tips_from_args` — the single reader every
other CLI uses — and `score_vaxseer.py` now writes the manifest hash beside its numbers.
Origin 2022's support is **1,825 genotypes against the backbone's 379–707**.

**`vaxseer.slurm` is job `3106147`, 7 GPU tasks**, the same seven origins `score.slurm`
runs. The 2017 and 2019 checkpoints exist and are not used, for the same two reasons the
scored array skips those origins.

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

---

## 2026-09-23 — status: `kemp` scored (14/17), consensus kernels gated, antiGen moved to a column; two arrays failed on a path bug

Brief status entry, written mid-round. Three strands, one conclusion: **clearing a
descent gate certifies a slope, and a slope is not a forecast.** No headline number
changes.

### 1. `kemp`, scored at last

Never scored in 2026-09-21 — array 2991023 died on the `golden_mu` bug and the round
moved to Huddleston. Gate rerun on the fixed tree reproduces the published four
decimals: `kunif` `p_down` 0.0800 / `dW/dmu` +0.7380, `kemp` 0.4529 / −0.1975 at
`tau_k` 0.1, `m` 0.

Six of nine transitions scored:

| predict | persistence | `mlr+mom` | `mlr+mom+K` | delta | `mu` | oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 5.688 | 5.5710 | 5.5710 | +0.0000 | 0 | 5.5356 |
| 2019 | 7.325 | 6.8829 | 6.8689 | −0.0140 | 0.05 | 6.8829 |
| 2020 | 7.574 | 7.4974 | 7.5373 | +0.0399 | 0.05 | 7.4974 |
| 2021 | 13.367 | 12.1330 | 12.1151 | −0.0179 | 0.05 | 12.1330 |
| 2022 | 9.969 | 9.4649 | 9.6607 | +0.1958 | 0.05 | 9.2036 |
| 2023 | 7.487 | 6.6962 | 6.7207 | +0.0245 | 0.05 | 6.6962 |

`mlr+mom+K` is **+0.0381 AA worse** than its parent (2 better, 3 worse, 1 tie); skill
+5.66% → +5.24%. `season+K` at the declared `mu = 0.1` — nothing fitted, nothing
selected — is +0.0084 AA, i.e. nothing, and is the cleanest row because nothing was
chosen.

**The oracle generalises the result.** `mu* > 0` at 2 of 6; elsewhere golden section
returns the parent exactly, so no `mu` improves anything. Total oracle gain −0.0494 AA
against a standing error of 8.04.

**Why, arithmetically.** Realisable gain is `slope x mu`, and `mu` is floored by the
grid. At −0.1975 and `mu = 0.05`, first order offers **0.010 AA** — the whole prize. The
realised number is worse than first order, so the curve turns up before 0.05. A future
kernel arm needs finer `mu` than `{0, 0.05, 0.1, 0.2, 0.4}` before its gate number means
anything.

**A mis-specified constant, flagged not patched.** On the `--drop-targets ""` panel
`kemp` gives 0.4324, under the hardcoded 0.4462. That constant is *uniform's*
break-even; the proposal derives the bar as `(1 - p_flat)/2`, a **per-arm** quantity, and
`kemp`'s `p_flat` of 0.27 puts its own bar at 0.366 — cleared by 0.07, `dW/dmu` negative
either way. `BREAK_EVEN` is right for `kunif`, wrong for any arm whose proposal is not
flat. Left alone: it is the pre-registered number.

### 2. Consensus kernels: `kcmom` clears three times faster, after a bug that hid it

Two leak-free arms in `mutation_kernel`, proposing toward where each site's alleles are
going rather than toward what a genotype can mutate into:

    pi(q, b | v)  ∝  f_hat_q(b) * (1 - f_hat_q(v_q)),   b != v_q

`kcons` uses the fast site-allele table (delta = 0); `kcmom` extrapolates the
log-frequency trend (delta = 1). The `1 - f_hat_q(v_q)` factor is the conditioning on
`v`: spend the proposal where `v`'s own residue is the one being left behind.

**The first numbers were a null because of a bug in the new code.** The flat
`(site, residue)` index was built as `np.arange(J, dtype=C.dtype) * nA` with
`panel.codes` uint8; under NEP 50 the multiply stays uint8, wraps every 13 sites, and
collapses all 313 sites onto ~13 bins. The table came back near-uniform, summed to one
per site, and raised nothing anywhere. Found by reading a real table: position 0 of HA
is M in 1681 of 1683 sequences and was being reported at 0.037.

Scope audited — the only other uint8 array, `consensus_distance`'s `cons`, is only ever
compared, never used in index arithmetic. **No published number is affected.** Pre-fix
outputs quarantined as `BUGGY_uint8wrap_*` with a README. The regression test checks
against a per-site loop on a fixture big enough to trip the wrap *and* asserts that it
does trip it; the five-site fixture cannot, since `J * nA` must exceed 255.

Gate on the corrected table (published panel, training transitions, bar 0.4462):

| arm | raw (`tau_k` 1) | best | `dW/dmu` |
| --- | ---: | ---: | ---: |
| `kunif` | 0.0800 | — | +0.7380 |
| `kemp` | 0.1172 | 0.4529 | −0.1975 |
| `kcons` | 0.5028 | 0.7391 | −0.5487 |
| `kcmom` | 0.5258 | 0.7631 | −0.5958 |

`kcmom` reaches 60% of the greedy oracle, clears **unsharpened** where `kemp` never did,
and the sign agrees on all five training transitions (−0.481, −0.565, −0.666, −0.746,
−0.520) where `kemp` pointed uphill on two of five. Both optima are **interior**:
`tau_e` 0.15 −0.569 | 0.25 −0.596 | 0.5 −0.581 | 1.0 −0.550 | 2.0 −0.482, and `tau_k` a
plateau over 0.005–0.03. Operating point `tau_e` 0.25, `tau_k` 0.02, `m` 0. `kcons` runs
at identical settings, not its own argmax, so the only difference between them is delta.

**The motivating prediction was wrong.** The arm was built on the argument that the
level term would be regressive and momentum would carry it. `kcons` clears at 0.5028
raw: the level does most of the work, momentum adds a secondary −0.047.

**Written down before any forecast:** at −0.5958 and `mu = 0.05`, first order offers
~0.030 AA. Three times `kemp`'s prize is three times very little. Expect a clean gate
and a negligible forecast unless the `mu` grid is refined — which the queued array does
not do.

### 3. antiGen as a column, not a kernel

Section 1 explains why the 2026-09-23 convolution negative is hard for *any* kernel to
escape. So this strand stops moving mass: unextended support, floor unchanged, antiGen
entering where `distance_from_consensus` enters — one standardised column, one
coefficient (`antiGen/antigen_feature.py`). Frozen trunk, nothing refitted.

- **`agen`**, arrival rate: `sum over u in V_T, d(u,v)=1 of x_u * pi(q,b|u)` — how much
  mass the population is poised to mutate *into* `v`. A property of `v`'s
  **neighbourhood**, not of its residues. Feasibility checked first: 65–69% of the
  support has an in-support one-step neighbour, carrying 57–82% of the mass, mean
  in-degree 1.2.
- **`agenf`**, flow overlap: `sum over (q,b) of pi(q,b|v) * f_hat_q(b)`. Dense, but a
  function of `v`, so the one that might be redundant.

Families `mlr+agen`, `mlr+mom+agen`, `mlr+agenf`, `mlr+mom+agenf`, wired as the `dfc`
arms: hyperparameter-free from the selector's side, each its parent plus a scalar.

**The redundancy test needed correcting before it meant anything.** `dfc` was discarded
for being spanned by the one-hot allele design (in-sample `R^2` 0.968–0.997). Applied
naively that test has almost no power here: 239 allele columns at rank 124–167 against a
support of 271–442, so **pure Gaussian noise scores 0.67–0.88 in sample**. The gate
therefore uses cross-validated `R^2`, ridge calibrated by sweep (1e-4 maximises the gap
between noise and a column built as `S @ beta`), intercept unpenalised (penalising it
breaks shift-invariance — a test catches this). And **no absolute bar**: even
cross-validated, an exactly-spanned column only reaches ~0.80, because holding out a
fifth of a rank-deficient fit loses real information. Each column is read against two
anchors computed on its own horizon's design — a noise column and a spanned column —
and called redundant above their midpoint.

This does **not** overturn the `dfc` finding: 0.968–0.997 sits well above the 0.67–0.88
null, and `dfc` comes back redundant at 7 of 7 horizons under the corrected statistic
too. It does mean those published numbers were flattered by the design's rank, and that
a new column must not be judged on the same scale.

Gate, published panel, seven training transitions (anchors: noise −0.558, spanned
+0.742, midpoint +0.092):

| feature | CV `R^2` | in-sample | redundant at | Spearman vs growth | rho>0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `agen` | **−0.298** | 0.751 | **0 of 7** | **+0.156** | 5/7 |
| `agenf` | +0.895 | 0.991 | 7 of 7 | −0.167 | 1/7 |
| `dfc` (reference) | +0.873 | 0.980 | 7 of 7 | −0.137 | 2/7 |

**`agen` clears, `agenf` does not.** The arrival rate sits at the noise anchor — the
allele design cannot express it out of fold, which is what a neighbourhood quantity
should do and the structural reason the arm was built that way. The flow column sits on
top of `dfc`, the outcome its being a function of `v` made likely.

Spearman is the second, weaker diagnostic: correlation with realised next-season log
growth against `assigned`, the nearest-neighbour target the multinomial fit uses. It
reads the transition's own future, so it is an oracle diagnostic restricted to training
transitions. `agen` at +0.156, positive at 5 of 7, is a signal but a weak one. The two
diagnostics answer different questions — non-redundant says the design did not already
have it; positive Spearman says anyone has it at all.

**On refitting the head, which was the suggestion: not done, deliberately.**
`output_proj` is 128 x 22 ~ 2,816 parameters, and the MLR objective supplies 2–3
training transitions per origin; a mean-pooled scalar head would still be 129 against
~3 effective observations, on a panel where one extra grid field has been priced at
about a point and a one-coefficient count matches a 255-coefficient model. Establish
whether the frozen column carries anything first. If `agen` scores at or above its
parent, a refit head has something to sharpen; if not, 129 parameters on three
transitions will not rescue it.

### State of play, and two arrays that need resubmitting

**`kemp` scoring: 14 of 17 tasks COMPLETED** (array 3003636), all clean. The three
missing are published-panel origins 2023–2025: they were preempted at 1h44m off
`cpu_preemptible` — which `scontrol show partition` gives as **`PriorityTier=0`, the
bottom of the cluster**, against 1000 for `cpu` and `cpu_batch` — so the eviction was
structural rather than bad luck. They were cancelled and resubmitted to `cpu_batch` as
job 3004050.

**Both newer arrays then failed.** 3004050 (3 tasks) and 3004049 (34 tasks, the
consensus scoring) each failed within seconds, exit code 1, every task with

    /home/ivan.specht/miniconda/bin/python3: No module named antiGen.season_momentum

The `echo` line before the `cd` is in the logs, so the scripts started. The identical
command with the identical script structure ran fine in array 3003636 earlier the same
day, and `python3 -c "import antiGen.season_momentum"` from the repo root succeeds now.
**The cause is not established.** Candidates not yet separated: the working tree was
being edited during that window (the consensus arms, `--tau-e`, `antigen_feature.py`),
and `$HOME` is at 98% with 27 GB free. Both arrays need diagnosing before resubmission —
resubmitting blind would just reproduce it.

**Not done:**

- The four outstanding `kemp` transitions (predict-2024, 2025, 2026 on the published
  panel). The six reported above are the smaller supports; predict-2026, where
  persistence is 11.0 and the floor 7.3, is among the missing. **The sign could move.**
- No consensus forecast exists. The gate is the only measurement `kcons`/`kcmom` have.
- `agen` is gated, not scored.
- **The oracle `mu` is still recorded nowhere.** `combo_*@mu*["mu"]` holds the grid's
  value, never the golden-section result, so `mu*` can only be inferred from
  "oracle != parent". Worth fixing before the next scoring array.
- Nothing has been run on the H2020 panel; everything here is the backbone.

**Tests:** 188 passing across `test_antigen_feature`, `test_mutation_kernel`,
`test_antigen_kernel`, `test_season_momentum`, `test_season_emd`.

**Code added this round:** `antiGen/antigen_feature.py`, `antiGen/feature_gate.py`,
`kcons`/`kcmom` and `site_allele_freq`/`consensus_freq` in `antiGen/mutation_kernel.py`,
the `agen` families and `--tau-e` in `antiGen/season_momentum.py` (the latter closing a
gap where the gate could sweep a timescale no scoring run could reach), and the kernel
settings now recorded in every scored JSON so two runs differing only in them can be
told apart.

**Artifacts:** `~/antiGen-count-loss-runs/kempscore/` (`score.slurm`, `refill.slurm`,
`summarize_kemp.py`, `out/`, `gate/` including the quarantined `BUGGY_uint8wrap_*`);
`~/antiGen-count-loss-runs/consscore/` (`score.slurm`, empty `out/`);
`~/antiGen-count-loss-runs/agenfeat/gate_pub.json`.

---

## 2026-09-23 — one branch switch took out three jobs; VaxSeer completed; the backbone figure repointed to the eval-mask panel

### The cause of the "No module named antiGen.season_momentum" failures, established

The previous entry left this open: arrays 3004050 (kempfill, 3 tasks) and 3004049
(consscore, 34 tasks) each lost every task in under two seconds to

    /home/ivan.specht/miniconda/bin/python3: No module named antiGen.season_momentum

and the entry recorded "the cause is not established", with the working tree being
edited and `$HOME` at 98% as unseparated candidates. **Both were wrong.** `git reflog`
against the `sacct` start times gives it exactly:

| time | event |
| --- | --- |
| 20:55:30 | commit `78d9625` on `is-dev` — the consensus arms, `--tau-e`, `antigen_feature.py` |
| **20:58:16** | **checkout `is-dev` → `is-dev-2` (`de963f6`)** |
| 21:17–21:57 | every task of both arrays starts, and dies |
| 21:58:08 | the VaxSeer origin-2019 **scoring** job 3002440 dies the same way |
| **22:27:53** | **checkout back to `is-dev`** |

`de963f6` is an older branch whose `antiGen/` holds only `config.py`, `train.py`,
`eval.py`, `load.py` and friends — **no `season_momentum.py`, no `mutation_kernel.py`,
no `antigen_feature.py`**. The jobs `cd /home/ivan.specht/antiGen` and import from the
live working tree, so for those 89 minutes the module genuinely was not on disk. 37 + 1
tasks, one cause. Array 3003636 was untouched because its tasks had already imported
before the switch, which is why the failure looked selective and was not.

Nothing about the cluster, the environment or disk pressure was involved. The lesson is
narrow and worth keeping: **a job that reads code from a mutable working tree is a job
whose correctness depends on what someone does to that tree while it queues.**

### The fix: jobs read from an immutable snapshot

`git archive HEAD` of `78d9625` into
`/large_storage/hielab/ispecht/antigen-code/78d9625`, made read-only and stamped with
`.SNAPSHOT_COMMIT`. The working tree was clean apart from `NOTES_2.md`, so the snapshot
is byte-identical to what the scripts meant to run — verified with `diff -rq` against
`~/antiGen/antiGen`. Both `kempscore/refill.slurm` and `consscore/score.slurm` now `cd`
there and run a preflight that asserts the commit stamp and imports
`antiGen.season_momentum` + `antiGen.mutation_kernel`, exiting 42 with a named error
otherwise. Originals kept as `*.slurm.bak`.

The preflight fired and passed on every task of the resubmission:

    preflight OK: code /large_storage/hielab/ispecht/antigen-code/78d9625 @ 78d9625269...

### A second operational trap: `cpu_batch`'s cap is shared, not per-user

The resubmitted jobs then sat `PENDING (QOSGrpCpuLimit)` and would have sat there for
days. `cpu_batch`'s QOS carries `GrpTRES=cpu=288` **shared across all users**, and it
was pinned at 288/288 with 286 jobs queued, one user holding 232 CPUs across eight jobs
with **7-day and 14-day time limits already 20h–1d19h in**. The nodes themselves had
~558 idle CPUs; the cap was the whole constraint. The previous entry chose `cpu_batch`
to escape `cpu_preemptible`'s `PriorityTier=0` evictions, which was right about the
eviction and traded it for an indefinite block.

Runtimes from array 3003636 are **10m52s to 5h19m**, and the outstanding tasks are the
largest supports — so no preemptible partition is viable (a 5h task evicted at 1h44m
never lands) and `quick_preemptible`'s 2h cap excludes most of them.

**Moved to `cpu_batch,cpu_high_mem`.** `cpu_high_mem` runs QOS `nogpu`, whose
`MaxTRESPU=cpu=513` is **per-user**, with no job-count or wall limit, `PriorityTier=1`,
non-preemptible, ~508 CPUs idle and our usage at zero. Listing both partitions keeps
`cpu_batch` (tier 1000) in play if it frees. Done with `scontrol update` rather than
cancel-and-resubmit, so the job IDs and the `afterok` dependency survived. Two
side-effects: the pending reason cleared immediately and tasks started within seconds,
and CPUs-per-task fell 6 → 3, because `cpu_high_mem`'s `MaxMemPerCPU=8192` against
`cpu_batch`'s 4096 stops the 24G request inflating the allocation.

Live: **3004889** kempfill (3 tasks, the missing `k_pub_{2023,2024,2025}`), **3004890**
consscore smoke (task 1, `kcmom` pub 2017), **3004891** consscore 2–34 at `%12`, gated
`afterok` on the smoke. The smoke gate is there because the `kcmom`/`kcons` *scoring*
path has never run end-to-end — only the gate has.

### VaxSeer origin 2019, and what it does to the comparison

Training (3002439) completed in 5h57m; only the chained scoring died, and only at the
last CPU-only step, after the GPU inference had already written `test_results.csv`. So
the rerun was one Python call, not a 6-hour retrain.

    2019->2020  floor  2.774 | persistence  7.574 | vaxseer 10.031 | n_eff 6.0 / 433

`n_support` 433 matches our own run, as the previous entry predicted it would. The row
is bad for VaxSeer — **+2.534 AA against `mlr+mom`, −32% against persistence** — and it
is the third-worst of its nine. The panel is now complete:

| model | mean AA (8) | vs persistence | wins |
| --- | ---: | ---: | ---: |
| `ne_star` | 7.494 | +5.35% | 7/8 |
| `distance_from_consensus` | 7.603 | +4.66% | 4/8 |
| `ne_star-lbi` | 7.610 | +3.59% | 5/8 |
| **`mlr+mom`** | **7.661** | **+3.86%** | **8/8** |
| `mlr` | 7.717 | +2.96% | 7/8 |
| `lbi` | 8.144 | −2.56% | 2/8 |
| `ewma` | 8.198 | −2.47% | 3/8 |
| `ep_x-ne_star` | 8.493 | −7.15% | 4/8 |
| `ep_x` | 8.813 | −10.07% | 4/8 |
| consensus, point mass | 8.854 | −11.44% | 3/8 |
| **VaxSeer (self-trained)** | **9.171** | **−13.05%** | **5/8** |
| top-LBI strain, point mass | 10.645 | −35.10% | 1/8 |
| persistence | 7.970 | | |
| floor | 3.389 | | |

**VaxSeer's 5-of-8 against a mean of 9.171 is the whole story in two numbers**, and
adding 2019 sharpens rather than softens the reading in the previous entry. Its per
transition spread: −0.449, −0.415, −0.014, −1.159 on four of them (it beats us), against
+1.142, +2.534 and **+10.877** on 2021→2022. One transition carries the entire deficit.
The claim stands and is now measured on the full panel: on ordinary transitions a
12-layer autoregressive model over 200k sequences is level with a 255-coefficient linear
fit; across a regime change it is far worse than doing nothing.

### The backbone figure, repointed

The previous entry flagged this as "the obvious next task and is not done", blocked on
the 2019 row. Done now. `eval_figures.py` gains `BACKBONE_PANELS` and a `--backbone`
flag; `load_backbone(which)` takes the sources and the mask from it.

- **`evalmask` (new default)** — ours from `covid2020/out/c20_*.json`, theirs from
  `covid2020/theirs_with2020.json`, origin 2020 dropped *after loading and before any
  average*, which is what makes it an eval-only mask: the fits behind the numbers saw
  the row. Scores 2018, 2019, 2020, 2022–2026.
- **`published`** — the old `dfc/` + `backbone3way/theirs.json`, kept reproducible
  because the 2026-09-22 entry and everything citing it report it.

The cross-run assertions were kept and hold on the new panel: persistence agrees
ours/theirs to **5.3e-15** and ours/VaxSeer to **3.6e-15**. The figure's hardcoded "8
transitions" became `panel['n']`, and the caption now names the masking, because that is
precisely what changed and a reader cannot tell 8 transitions from 8 transitions.

**Regression check: `--backbone published` reproduces `table_backbone.csv`
byte-identically** against the copy archived before the change, and prints `mlr+mom`
8.220 / +5.22% / 8-of-8 with persistence 8.694 — the published row of the comparison
table in the entry above. So the repoint is a repoint and not a silent recomputation.

### Artifacts

- `~/antiGen-count-loss-runs/figures/` — regenerated. `fig_backbone_{light,dark}.{png,pdf}`,
  `table_backbone.csv`, `table_long.csv` (608 rows) now on the eval-mask panel and
  carrying VaxSeer for all 8; the four `fig_h2020_*` files and their two CSVs are
  unchanged, that panel having no backbone masking in it. `README.md` rewritten with a
  "which backbone panel this is" section. `ARCHIVE_published_panel/` holds the previous
  backbone figures and CSVs.
- `~/antiGen-count-loss-runs/vaxseer/out/vaxseer_2019.json` — the missing row.
- `/large_storage/hielab/ispecht/antigen-code/78d9625/` — the read-only snapshot the
  scoring arrays now run from.
- `covid2020/summarize_evalmask_vax.py` — de-staled; it printed "VaxSeer pending" and a
  7-origin subpanel that no longer exists. The subpanel logic is kept, guarded on an
  actual gap, so it still speaks up when a model is missing an origin.

### Not done

- The three `kemp` transitions and the 34 consensus scorings are **running, not
  finished**. Every number in the entry above this one that depends on them still does.
- `BREAK_EVEN` is still `kunif`'s bar applied to every arm, and the oracle `mu` is still
  recorded nowhere. Both were deliberately left alone rather than changed mid-array:
  editing `season_momentum.py` now would desynchronise these runs from the 14 `kemp`
  outputs already on disk.
- `python -m antiGen.season_momentum --help` raises `TypeError: %o format` from a
  literal `%` in an argparse help string. Cosmetic — real runs are unaffected — but it
  means `--help` cannot be used to check a flag exists.

---

## 2026-09-24 — three backbone findings die out of sample, `kemp` is settled, the consensus kernel is the largest effect yet, and it is now queued on H2020

### 1. The assumptions sweep, and what it cost to believe it

44 one-at-a-time perturbations of the constants `mlr+mom` never selects (job 3005272,
all COMPLETED). Baseline arm reproduces `covid2020/out/c20_2022.json` bit-identically,
so every movement is the perturbation.

**The lookback is inert.** From L = 1.5 to L = 10 years the forecast is unchanged to
four decimal places, while the support grows 72 -> 637 genotypes at origin 2022. EWMA
base mass beyond three years is 1e-4, and the tilt never amplifies it. Only L = 1.0
moves anything (+0.040, CI straddling zero), and below ~2 years the *floor* starts to
rise (+0.050 at L = 1), so the support has begun losing transport destinations and the
problem itself has changed. **L can go from 5 to ~2 at no cost**; L = 1 is a different
panel. `maxiter` is likewise inert from 40 to 400 -- exactly 0.0000 -- so the fit is
fully converged well before the 80 we use.

**An audit that had to be added after the fact.** Five of 43 arms moved the panel
itself. `maxsup_200` looked like the best row in the table at 7.520 AA; its persistence
baseline had moved by 0.956 and its floor by 1.298, and on its own scale it is −1.63%
skill at 3/8. It is not comparable and neither is `maxsup_400` or `minseas_50`. The
check -- does this arm change `emd_season` or `floor`? -- is now the first thing the
summariser prints, and it should have been there before the first table was read.

**Load-bearing:** dropping `window=0` from the grid costs +0.154 AA [+0.036, +0.310];
one or two validation folds cost +0.14 and +0.12; and both narrowing and widening the
lambda grid hurt, which says it is well centred.

### 2. Three significant wins, none of which survived — the point of the exercise

| arm | backbone | H2020 validation (23) | H2020 test (8) | verdict |
| --- | ---: | --- | --- | --- |
| `topk1` | −0.070 (sig) | +0.018 [−0.017, +0.054] | +0.015 [−0.032, +0.060] | no effect |
| `maf05` | −0.122 (sig) | **+0.136 [+0.010, +0.291]** | **+0.120 [+0.013, +0.262]** | **worse, significantly** |
| `vf5` | −0.061 (sig) | −0.009 [−0.095, +0.071] | +0.060 [−0.015, +0.155] | no effect |

Jobs 3005335/6/7, 93 tasks, all COMPLETED; panel checks clean on all three (d naive and
d floor both 0.0e+00), so all three are comparable. Each arm's `args.txt` is the
baseline's with exactly one flag appended, verified by diff.

**`maf05` is the lesson.** The largest apparent win on the backbone is significantly
*harmful* on both H2020 arms, sign reversed, two intervals excluding zero. Development-set
overfitting, caught by the only test that could catch it.

**`topk1` is a different case and should not be filed with the other two.** Top-5
averaging was never the intended protocol -- the argmin was -- so this was a check on a
drift, not a search. The answer is that the drift is **costless out of sample** (+0.018,
+0.015, both null). So reverting to `--top-k 1` is free and restores the intended
design; it must not be reported as an improvement, because it is not one.

### 3. `kemp`, complete: the sign did not move

The refill (job 3004889) landed `k_pub_{2023,2024,2025}`; the pre-registered reference
rows match exactly (`season` 8.6944, `mlr+mom` 8.2199). On the complete eight
transitions `mlr+mom+K` is **8.2493 against 8.2199, +0.029 AA worse**, −0.53% skill
[−1.43, +0.20], better in 4/8, P(no better) 0.903. The three new rows came in at −0.070,
+0.132, −0.016, so predict-2026 -- the one the last entry flagged as able to move the
sign -- did not. **`kemp` does not help. That question is closed.**

### 4. The consensus kernel works, and it is not a small effect

All 17 `kcmom` tasks and 16 of 17 `kcons` (job 3004891; `c_kcons_nodrop_2025` still
running). Parent-vs-kemp agreement is 0.0e+00 on every panel, so these are paired with
the kemp run exactly as the script declared.

| arm / panel | parent | `+K` | delta | 95% CI | wins |
| --- | ---: | ---: | ---: | --- | ---: |
| `kcmom` published | 8.2199 | **8.0383** | **−0.182** | [−0.215, −0.148] | **8/8** |
| `kcons` published | 8.2199 | 8.0458 | −0.174 | [−0.201, −0.144] | 8/8 |
| `kcmom` eval-mask | 7.6615 | **7.4758** | **−0.186** | [−0.220, −0.150] | **8/8** |

On the headline panel this takes the top of the table: `mlr+mom+K` 7.476 / +6.42% / 8-of-8
against `ne_star` 7.494, with the tightest interval in the table by an order of magnitude.

Three things worth recording. The gate's first-order estimate was ~0.030 AA and the
realised gain is **six times** it -- the opposite of `kemp`, where realised was worse
than first order, and a warning against trusting that extrapolation in either direction.
`season+K`, with nothing fitted and nothing selected, improves −0.060 at 8/8, which is
the cleanest row because nothing was chosen. And **`kcons` ~ `kcmom`** (−0.174 vs
−0.182): momentum adds 0.008 AA, the level term does all the work, which **falsifies the
arm's motivating prediction** exactly as the gate had already hinted.

**Two caveats that must travel with the number.** `mlr+mom+K` places mass on one-step
neighbours *outside* the observed support, so it is a different model class from the
rows beneath it -- same targets and same cost matrix, so the EMD comparison is fair, but
its entitled floor is 2.780 against 3.632. And the operating point (`tau_e` 0.25,
`tau_k` 0.02, m 0) was chosen on the descent gate over training transitions: leak-free,
but still chosen on this panel.

### 5. So it is now running on H2020, which required building the support

`huddleston_panel.py` had no kernel: no `--kernels`/`--mus`/`--tau-e`, no `+K` families.
But `PanelTask` subclasses `Task` and `run_family_panel` delegates unknown families to
`run_family`, which already handles `+K`, so this was plumbing -- five flags into
`make_grid`, plus stamping `panel.kernel_tau_e`, because `Task.candidates` resolves
`tau_e` off the panel rather than through `run_family`'s signature. **Verified: with no
kernel flags the grid is bit-identical at 2880 combinations**, so every earlier run on
that panel reproduces.

**The smoke test earned its keep on the first attempt.** `huddleston_panel.py:681` read
`w.x` off a `Forecast`, whose attribute is `w.w` -- dead code until now, because before
the kernel was wired through no family on that panel ever returned a `Forecast`. Every
`+K` origin would have crashed. Fixed, and the `d_top` caveat recorded rather than
approximated: `per_sup` is indexed over the observed support, so for a `+K` family whose
top mass sits on a proposed candidate, `d_top` is the best *observed* haplotype and not
the forecast's own argmax. A `d_top_support_only_<fam>` flag now records when that
applies. EMD is unaffected -- it sees the extended support.

Submitted gated rather than blind, since the round is long and unattended: **job 3009789
is origin 2015-04-01 alone** (task 24 of the real array, so nothing is wasted) and **job
3009790 is the other 30 origins on `afterok`**. Because `afterok` only sees an exit code,
the gate task carries its own numerical assertion: the four families it shares with
`huddleston-lbi` must reproduce that run *exactly* -- which also tests the assumption
that trimming 14 families to 6 is inert, since selection is per-family -- and the
`emd_mlr+mom+K` column must exist. It exits 43 otherwise and the 30 never start.

Expect this to be slow. The comparable backbone runs were 29 min to 6h38m per origin
(median 3h54m), the H2020 `mlr+mom+K` grid is 7200 against the backbone's 5400, and the
array runs single-threaded. **8-20 hours wall is the honest range.**

### Code and snapshots

`antiGen/backbone_compare.py` (their per-origin hyperparameter selector, `--select`),
`antiGen/eval_figures.py` (`--backbone` panel selector), `antiGen/huddleston_panel.py`
(kernel support + the `Forecast` fix). All three are uncommitted working-tree changes.
Runs no longer read `~/antiGen`: the scoring arrays take an immutable snapshot on
`/large_storage`, stamped and preflight-checked. `78d9625` is the commit-pinned one;
`wt-2871dff130a4` is content-hashed off the working tree and is what the H2020 kernel
array runs from.

### Not done

- `c_kcons_nodrop_2025` still running; it affects only the `kcons` eval-mask row.
- The H2020 kernel array is **queued, not finished**. Until it reports, the +0.186 AA is
  a development-set number of exactly the kind that just failed to replicate three times
  in section 2. It should not enter any table before then.
- `--top-k 1` has not been made the default anywhere. The evidence says it is free and
  it is what was intended; that is a one-line change nobody has made.
- `BREAK_EVEN` is still `kunif`'s bar applied to every arm, and the oracle `mu` is still
  recorded nowhere.
- The lookback has not been reduced from 5 to 2 anywhere either, though the sweep says
  it is free.

---

## 2026-09-24 — the causality audit: two exposures found in our model and two on theirs, one of them worth 0.467 AA to `lbi`; **their hyperparameters are now selected by default**

### The ask

> Read notes and notes_2 to get oriented. I want you to audit the mlr+mom model for
> anything that isn't selected causally, i.e. any parameters that we need future
> information to select.

and then, after the first pass:

> For A1, can we simply switch to doing causal selection without losing performance? If
> so we should do this. For A3, yes, make the proposed change and submit array.

and then:

> Can you also audit whether there's anything non-causal about our fits of the other
> models, e.g. Huddleston, LBI, etc. on our "backbone" dataset? Or any hyperparameter
> selection that would unfairly advantage or disadvantage them?

Three rounds, recorded as one entry because the third changes how the first two get
reported.

### What is clean, stated first so the rest reads in proportion

The per-origin machinery does what the docstrings say. `select` takes folds strictly
earlier than the test origin, and because seasons are annual with `horizon = season +
0.25` and a target closing at `season + 1.25`, `order[j] <= order[i] - 1` gives
`t_horizon(j) <= horizon(i)` exactly -- the 6-month-pivot leak that
`huddleston_panel.usable_before` exists for cannot recur on this grid. `landscape`,
`site_allele_freq`, `empirical_counts`, `timepoints`, `growth_cv`, `emb_residual_pcs`,
`dfc` and `checkpoint_for` all read `dates <= horizon` or the support alone.
`descent_gate` refuses structurally on an unscored transition. `mlr+mom+K@mu*` is labelled
oracle everywhere it appears. `assigned` is only ever consumed from training tasks. The
leaks are not in the forecast loop; they are in constants that sit outside it.

### A1 — the kernel's operating point was selected on five of the eight scored transitions

`--tau-e 0.25 --kernel-tau-k 0.02 --kernel-m 0`, the constants behind `mlr+mom+K`
7.476 / +6.42% / 8-of-8, come from the descent gate. **`p_down` is an oracle quantity**:
`phi` is the c-transform of the optimal *target* potential, so a gate row knows where its
season actually went. `kernel_gate.py`'s `--gate-through 2022` calls its transitions
"training", and they are -- for the *fit*. They are not held out of the *evaluation*:
`rolling` scores every origin from `min_train` on. The gate JSONs say it plainly:

    gate_cons_ext_te0.25.json:  gated_on [2017, 2018, 2020, 2021, 2022]
                                held_out [2023, 2024, 2025]

Five of the eight scored rows are in the first list. The same constant reaches
`antigen_feature.FLOW_TAU_E` and `mutation_kernel.BREAK_EVEN` ("evaluated at the panel's
measured `p_flat`").

**The measurement survives, and that is checkable rather than hoped for.** Splitting the
realised `kcmom` gain by whether the gate had read the transition:

| panel | gated (5) | gate-blind |
| --- | ---: | ---: |
| published | −0.161 | **−0.216** (3) |
| eval-mask | −0.173 | **−0.193** (4) |

The gain is *larger* where the gate was blind. If the oracle exposure were manufacturing
the result the sign would be the other way. But surviving is not being causal.

**So it was made causal.** `mutation_kernel.causal_operating_point` picks
`(tau_e, tau_k, m)` per origin from the gate rows whose targets had closed by that
origin's horizon -- row `j` reads season `j + 1`, closing at `(j+1)+0.25 <= i+0.25`, i.e.
`j < i`, the same rule `train_set` and `select` obey, stated on the gate's index.
`kernel_gate --full-table` produces the rows for every transition and stamps the file;
`season_momentum --kernel-gate-table` consumes them and **refuses a file that is not
stamped**, because a table that stops at `--gate-through` would leave the early origins
silently on the declared defaults. Producing and consuming are separated so the causality
is one tested filter rather than where the driver happened to stop. `Task.kernel_point`
is the single override point, applied before the cache key. Six tests in
`test_mutation_kernel.py`, including a fixture where the future-fed configuration wins
outright unless the filter removes it.

**Feasibility was checked before any of it was built**, because the ask was conditional on
not losing performance. The causal picks track the pooled point to within ±0.08 in slope
on held-out transitions, and the surface is flat (median configuration −0.53 against the
best −0.596).

**The gate table, which is worth having on its own.** 2,376 rows, 216 configurations x 11
transitions, tau_e over {0.15, 0.25, 0.5, 1, 2, 4} and tau_k over {0.001 ... 1}, both
wider than the sweep that chose 0.25/0.02 -- deliberately, because choosing causally from
a menu drawn after seeing the oracle answer is still an exposure. A first pass floored at
tau_k 0.005 selected 0.005 at seven of nine origins, which is the grid-edge defect this
round is flagging in the `mus` ceiling, so the menu was extended a decade and rerun. Mean
`dW/dmu` over all eleven transitions, `kcmom`, m = 0:

|  tau_e \ tau_k | 0.001 | 0.005 | 0.02 | 0.1 | 1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | −0.5504 | −0.5507 | **−0.5534** | −0.5207 | −0.1592 |
| **0.5** | −0.5602 | −0.5604 | −0.5539 | −0.5122 | −0.1543 |
| 1 | −0.5416 | −0.5377 | −0.5256 | −0.4702 | −0.1062 |

`tau_e` has a genuine interior optimum at **0.5** -- the published 0.25 is second -- and
`tau_k` is a plateau from 0.001 to 0.02 spread over 0.005, so the per-origin picks
wandering inside it are noise, not disagreement. `m` is inert at these temperatures
(identical to four decimals for tau_k <= 0.1), because the proposal has already collapsed
below ten candidates a parent.

**And the sharper picks are not a sharper kernel.** The causal arm resolves tau_k to
0.001-0.002 rather than 0.02, which reads alarming. It is not: at tau_e 0.25 the
candidate count falls 51,224 -> 3,758 while `p_down` moves 0.7425 -> 0.7415 and `p_flat`
0.0684 -> 0.0675. The 47,000 extra candidates carry no mass; it is the same distribution
with a zero-mass tail pruned.

**Status: array 3011610 running, nine origins, not finished.** Until it reports, the
+6.42% remains a number whose operating point was chosen partly on five of its own rows.

### A3 — the backbone feature set was chosen over the whole record. Shipped, measured, null.

`TipPanel` has always taken a `feature_mask` and `huddleston_panel` has always passed one.
`season_momentum` had no flag, so every backbone number in this log chose its
(position, residue) columns from allele frequencies counted over all 1,683 tips running to
2026. The coefficients are still fitted on earlier transitions, so the exposure is in the
column *set* -- but it has the flattering sign, since an emerging allele gets its own
coefficient precisely because it later became common.

`--censor-features` added (needs exactly one `--transitions` origin; the geometry --
varying sites, alphabet, `codes`, every transport cost -- is untouched either way, since
dropping a site that only varies later would stop counting real substitutions). Four tests.
Array 3011506, nine origins, `covid2020/ours.slurm` field for field plus the flag.

| origin | tips counted | features | uncensored | dropped |
| ---: | ---: | ---: | ---: | ---: |
| 2017 | 440 | 164 | 239 | 75 |
| 2020 | 864 | 184 | 239 | 55 |
| 2022 | 1025 | 210 | 239 | 29 |
| 2025 | 1536 | 233 | 239 | 6 |

Panel check exact: `d emd_season` and `d floor` both 0.0e+00, `n_support` and `n_target`
unchanged. And the result is a clean null --

| family | uncensored | censored | delta | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `mlr+mom` | 7.6615 | 7.6613 | **−0.0002** | [−0.0020, +0.0019] |
| `mlr` | 7.7170 | 7.7181 | +0.0011 | [+0.0001, +0.0022] |

`mlr+mom` is 7.661 / 8-of-8 either way. `mlr`'s +0.0011 has an interval excluding zero and
is still a thousandth of an amino acid. **The exposure was real in structure and carries
nothing**, which is what the H2020 measurement (0.00 test, +0.22 validation) predicted and
what had never been checked here. Censoring is free and should now be on.

### The other side of the table, audited

**Their tree metrics are not causal on the backbone.** `backbone_compare` never reads the
`lbi`/`ep`/`ne_star` values Nextstrain shipped, and recomputes LBI per horizon with tips
censored -- all documented. But the *tree* is one 2026 build: topology, internal-node
dates and the ancestral mutation placements `mutational_load` walks were inferred with the
whole record. `lbi`, `ne_star` and both composites are functions of that inference.

This is specific to the backbone, and this codebase already contains the evidence that it
matters: `huddleston_panel.py:123` records that on the H2020 panel, where their pipeline
builds a tree per timepoint, the value differs between timepoints in **100%** of cases for
`lbi` and `ep_x` and 40% for `ne_star`. So per-horizon rebuilding materially changes these
metrics and the backbone hands their models a tree they would not have had. Direction is
plausibly in their favour, which is the safe direction for our claim, but it is
**unmeasured** and removing it means per-horizon augur/treetime builds. Not started.
`mlr+mom` has no equivalent exposure: it reads tip dates and tip sequences, and a tip's
sequence is the composition of mutations along its root path whatever the reconstruction
does with them.

**26 of 1,683 tip dates are TreeTime-inferred** (`num_date.inferred: true`), confidence
widths up to 1.0 year; 17 have intervals spanning two seasons and 5 straddle an origin
horizon, so they can move between support and non-support or between target seasons. This
is in `TipPanel` and hits both sides identically -- a panel-level wrinkle, not a fairness
one, and small against seasons of 35-102.

**Everything else on their side is causal**: `ep_x`'s past window, `ne_star`'s ancestor
walk, the per-origin standardisation, `their_train_idx`, and `select_their`'s folds all
obey the same rule ours do.

**Their optimiser is not a handicap, measured.** `fit_their_model` is one Nelder-Mead start
at `seed=0`. Refitting from 24 random starts at seven origins: every 1-D model (`lbi`,
`ne_star`, `ep_x`, `distance_from_consensus`) is already at the global optimum. 2-D
`ne_star-lbi` does stick once -- training objective 8.191 against 8.026 at origin 2022 --
and the better-optimised fit scores **worse** on test, +0.121 AA averaged. Fixing their
optimiser would hurt them.

### **THE STANDING DEFAULT CHANGES: their hyperparameters are now selected per origin**

Every published "theirs" number -- `backbone3way/theirs.json`,
`covid2020/theirs_with2020.json` -- was produced **without `--select`**, at the config
their paper ships, while ours were re-chosen at every origin off a 1,080-combination grid
by 3-fold validation with the top 5 averaged. `backbone_compare` has carried `--select`
since it was written, its own comment says "that asymmetry is in our favour and this
closes it", and **the flag had never been run**. The capability had been mistaken for the
measurement.

Job 3011787, eval-mask panel, same folds, same top-5, same objective, same optimiser:

| model | published config | selected per origin | selection worth |
| --- | ---: | ---: | ---: |
| `lbi` | 8.144 (−2.56%, 2/8) | **7.677 (+4.20%, 5/8)** | **+0.467** |
| `distance_from_consensus` | 7.603 (4/8) | 7.566 (6/8) | +0.037 |
| `ne_star-lbi` | 7.610 (5/8) | 7.624 (6/8) | −0.014 |
| `ep_x` | 8.813 | 8.899 | −0.086 |
| `ne_star` | 7.494 (7/8) | 7.772 (6/8) | −0.278 |
| `ep_x-ne_star` | 8.493 | 8.870 | −0.377 |

Two readings, and the second is the one that matters. **The module's premise is wrong on
average** -- across six models the selector costs them 0.042 AA, because three folds over
16-144 candidates on eight transitions mostly adds variance to a one- or two-coefficient
model, which is a fact about our protocol as much as about theirs. **But it is right about
`lbi`, and dramatically**: giving `lbi_tau` what our `tau_f` gets takes their weakest row
from −2.56% / 2-of-8 to +4.20% / 5-of-8, and the selector picks `lbi_tau = 1.0` at nearly
every origin against their published 0.3. The published table's most damning row about
their models was substantially an artefact of a frozen constant.

**Decision, taken with the user: the selected arm is the default for reporting, for their
models and ours.** It is the apples-to-apples comparison -- both sides get the same
selector on the same folds with the same averaging -- and it is what a reader should see
first. The published arm stays reproducible, because it is what their papers specify and
what every entry above this one reports.

- `backbone_compare --select` is **on by default**; `--no-select` restores the published
  config. **A pre-2026-09-24 script that passed no flag now produces the other arm.** The
  arm is therefore recorded in the output JSON, along with `val_folds`, `top_k`, `window`
  and the panel settings -- until today that file held `per_transition` and nothing else,
  which is exactly how the asymmetry went unnoticed for two days.
- `eval_figures --their-arm {selected,published}` defaults to `selected`;
  `BACKBONE_PANELS` gains `theirs_published`. The `published` masking panel has **no**
  selected arm (`backbone_compare --select --drop-targets 2020` has never been run) and
  asking for one raises instead of serving frozen numbers under the selected label.
- `theirselect/theirs_select.json` was stamped after the fact, since it predates the
  metadata.

**The headline table on the new default** (8 transitions, season 2021 masked from the
evaluation only, nothing held out of any fit):

| model | source | mean AA | vs persistence | wins |
| --- | --- | ---: | ---: | ---: |
| `mlr+mom+K` | this work | **7.476** | **+6.42%** | **8/8** |
| `distance_from_consensus` | H2020 | 7.566 | +5.21% | 6/8 |
| `ne_star-lbi` | H2020/H2025 | 7.624 | +4.80% | 6/8 |
| **`mlr+mom`** | **this work** | **7.661** | **+3.86%** | **8/8** |
| `lbi` | H2020 | 7.677 | +4.20% | 5/8 |
| `mlr` | this work | 7.717 | +2.96% | 7/8 |
| `ne_star` | H2020 | 7.772 | +2.35% | 6/8 |
| `ewma` | this work | 8.198 | −2.47% | 3/8 |
| consensus, point mass | BC2021 | 8.854 | −11.44% | 3/8 |
| `ep_x-ne_star` | H2020 | 8.870 | −11.47% | 3/8 |
| `ep_x` | H2020 | 8.899 | −11.23% | 3/8 |
| VaxSeer (self-trained) | Shi2025 | 9.171 | −13.05% | 5/8 |
| top-LBI strain, point mass | Neher2014/BC2021 | 10.645 | −35.10% | 1/8 |
| persistence | — | 7.970 | | |
| floor | — | 3.389 | | |

`mlr+mom` goes fourth to third of six fitted models. **Nothing in the top five is
separable**: every paired interval against us straddles zero (`d_cons` +0.096, `ne_star-lbi`
+0.037, `lbi` −0.016, `ne_star` −0.111). **The 8-of-8 win count remains uniquely ours**, so
finding 3 of the headline entry -- what separates our model is stability, not level --
survives, and is now more clearly the only thing. The `mlr+mom+K` row at the top is still
the oracle-selected operating point; it must be repointed to `causalgate/` when 3011610
lands.

### Two asymmetries left standing, both against us, recorded so the ledger is honest

- **Our base measure is the weaker one.** We tilt a selectable EWMA level; they tilt fixed
  whole-season persistence, which the baseline table has said all along is stronger
  (8.68 against 9.05).
- **VaxSeer gets no hyperparameter selection at all**, one published recipe, against our
  1,080-combination selector. No arm of `--their-arm` changes that.

Not an asymmetry: our grid is 1,080 and theirs 16-144, but that is because their models
have fewer knobs. That is the comparison.

### Still exposed, and not fixed

- **The `--mus` ceiling binds.** Every origin selects `mu = 0.4`, the top of
  `0,0.05,0.1,0.2,0.4`, and the per-transition oracle beats it at all nine (5.403 ->
  5.240, 4.141 -> 3.880, ...). The reported gain is a lower bound and the ceiling is a
  constant nobody selected. Left alone deliberately this round: changing two things at
  once would make neither measurable.
- **`top_k = 5`, `val_folds = 3`, and the grid ranges** (`lambdas`, `windows` including 0,
  `tau_f`/`tau_s`, `gamma_pens`) were shaped over dozens of rounds on this panel, and the
  assumptions sweep priced several of them as load-bearing. The 144-point fixed lattice on
  H2020 is the standing rebuttal and should be cited wherever the grid is.
- **The masking** was chosen after computing three of them, defended on the tip counts and
  on the principle, which is auditable and is not the same as causal.
- **`BREAK_EVEN` is still `kunif`'s bar** applied to every arm, and the oracle `mu` is
  still recorded nowhere. Unchanged from the entry above.
- **Per-horizon tree builds for their metrics.** The one open causality item on their
  side. Not scoped.

### Artifacts

- `antiGen/season_momentum.py` (`--censor-features`, `--kernel-gate-table`,
  `Task.kernel_point`), `antiGen/mutation_kernel.py` (`causal_operating_point`),
  `antiGen/kernel_gate.py` (`--full-table`, `--tau-es`, per-row `tau_e`, and a verdict
  line that no longer reports one timescale's `p_down` under another's label),
  `antiGen/backbone_compare.py` (`--select` on by default, run metadata in the output),
  `antiGen/eval_figures.py` (`--their-arm`). 390 tests pass; the 4 failures in
  `test_tranception_integration` predate this round and are unrelated.
- `~/antiGen-count-loss-runs/censor/` -- `ours.slurm` (3011506), `out/cen_*.json`,
  `summarize_censor.py`.
- `~/antiGen-count-loss-runs/causalgate/` -- `gate.slurm` (3011558), `gate/g_*.json`,
  `merge_gate.py`, `gate_table.json`, `score.slurm` (3011610, **running**),
  `summarize_causalgate.py`.
- `~/antiGen-count-loss-runs/theirselect/` -- `select.slurm` (3011787),
  `theirs_select.json`, `summarize_theirselect.py`.
- `~/antiGen-count-loss-runs/figures/` regenerated on the new default;
  `ARCHIVE_publishedtheirs_0924/` holds the previous set.
- Code snapshot `/large_storage/hielab/ispecht/antigen-code/wt-010a07373114`.

---

## 2026-09-28 — the causal gate lands, the consensus kernel replicates out of panel, and causal selection becomes the convention

### The ask

> Report on the completed jobs: causal gate array and mlr+mom+K on H2020 data

then

> please regenerate the headline table, figures, and CSV with causal selection. Also
> update notes to record this as the new convention: I always want models to be selected
> causally, as otherwise we could be criticized for peeking at the future.

The convention is now **standing convention 1** at the top of this file. The rest of this
entry is the two measurements that made it cheap to adopt.

### 1. The causal gate: it costs nothing, and the headline claim is intact

Array 3011610, nine origins, all COMPLETED, 1h01m to 10h07m. `(tau_e, tau_k, m)` resolved
per origin from the gate rows whose targets had closed by that origin's horizon, against
the previous arm's one point pooled over five of the eight scored transitions.

Nesting check exact: `season`, `ewma`, `mlr` and `mlr+mom` all reproduce `consscore` to
**0.0e+00**, so the only thing that moved is the kernel.

| arm | mean AA | vs persistence | beats persistence | beats parent |
| --- | ---: | ---: | ---: | ---: |
| `mlr+mom` (parent) | 7.6615 | +3.86% | 8/8 | — |
| `mlr+mom+K` oracle-pooled | 7.4758 | +6.42% | 8/8 | 8/8 |
| **`mlr+mom+K` causal** | **7.4882** | **+6.11%** | **8/8** | 7/8 |

**causal − oracle: +0.0124 AA [−0.0735, +0.0865], worse in 5 of 8.** The interval is an
order of magnitude wider than the point estimate; the two arms are indistinguishable.

**A distinction that has to be stated because it nearly went the wrong way in the
reporting.** On the causal arm `mlr+mom+K` beats its own parent at 7 of 8 rather than 8,
but beats **persistence at 8 of 8**. The headline convention is versus persistence, so
finding 3 -- what separates our model is stability, not level -- survives untouched. The
summariser now prints both counts, because printing only the first would have retired a
headline claim on a comparison it was never about.

The prediction recorded four days ago was −0.178 against the pooled −0.186 by scaling the
realised gain with the descent slope; realised is −0.173. The reasoning held too: the
causal picks land at `tau_e` 0.5–1.0 and `tau_k` 0.001–0.002 rather than 0.25/0.02, and
that does not matter because they are different points on the same plateau. The
gate-read/gate-blind split is now moot but unchanged in shape: −0.168 on the four the old
gate had read, −0.179 on the four it had not.

### 2. `mlr+mom+K` on H2020: the validation arm replicates decisively, the test arm timed out

Array 3009790 lost **tasks 25–31 to `--time=24:00:00`**, and those seven are 2015-10-01
through 2018-10-01 -- **seven of the eight test origins**. What survived is the 23
validation origins and the single test origin 2015-04-01, which was the gate task. The arm
the round needed is the one that died. The cause was the self-imposed limit and nothing
else: both `cpu_high_mem` and `cpu_batch` carry `MaxTime=14-00:00:00`, and completed
runtimes climb steeply with support (20h41m at 2012-10-01).

Validation arm, 23 origins, 2003-10-01 .. 2014-10-01. The four families shared with
`huddleston-lbi` reproduce that run to **0.0e+00**, so the pairing is exact:

| | mean AA | vs naive | beats naive |
| --- | ---: | ---: | ---: |
| naive | 6.3703 | | |
| `mlr+mom` | 5.8956 | +7.04% | 18/23 |
| **`mlr+mom+K`** | **5.7502** | **+9.46%** | **19/23** |
| `season+K` | 6.3182 | +0.84% | |

**Kernel delta −0.1455 AA [−0.2062, −0.0791], better at 21 of 23.**

That is a genuine out-of-panel replication of the backbone's −0.186, and the context is
what makes it worth something: three significant backbone findings (`maf05`, `topk1`,
`vf5`) had just failed to replicate on this very panel, one with the sign reversed. The
consensus kernel does not. The single test origin 2015-04-01 gives +0.0446, i.e. the
kernel is worse there; n = 1 and nothing should be read into it.

**Refill submitted: job 3060596**, tasks 25–31, five days and 32G, on the **same code
snapshot** `wt-2871dff130a4` and the same `args.txt` as the 24 that landed. A panel whose
origins came from two working trees is not one run, so the causal-gate and censoring work
stays out of it. Expect 1–3 days.

Two caveats travel with the H2020 number. It is the **validation** arm, not their held-out
test arm. And its kernel operating point is still the backbone-gate one transferred
across -- finding A2 -- which for this panel is future information, since that gate read
2018–2023 targets and every H2020 origin predates them. Deliberately not fixed in a run
that was already 24/31 done; it is now on the standing "still exposed" list.

### 3. What was regenerated

`eval_figures` gains `--kernel-arm {causal,oracle}`, default **causal**, the same shape as
`--their-arm`. `BACKBONE_PANELS` gains `kernel_oracle`. The `published` masking panel has
`kernel=None` as well as `theirs=None`, so asking it for either causal arm raises and
names the run that would produce it, rather than serving a superseded number unlabelled.
The blurb says "mlr+mom+K at the SUPERSEDED oracle-pooled operating point" whenever the
oracle arm is drawn.

**The headline table, on the standing convention -- both sides selected, everything
causal** (8 transitions, season 2021 masked from the evaluation only, nothing held out of
any fit):

| model | source | mean AA | vs persistence | 95% CI | wins |
| --- | --- | ---: | ---: | --- | ---: |
| **`mlr+mom+K`** | this work | **7.488** | **+6.11%** | [+3.82, +8.46] | **8/8** |
| `distance_from_consensus` | H2020 | 7.566 | +5.21% | [+0.71, +9.86] | 6/8 |
| `ne_star-lbi` | H2020/H2025 | 7.624 | +4.80% | [+1.05, +9.09] | 6/8 |
| **`mlr+mom`** | this work | **7.661** | **+3.86%** | [+2.13, +5.88] | **8/8** |
| `lbi` | H2020 | 7.677 | +4.20% | [+1.07, +7.55] | 5/8 |
| `mlr` | this work | 7.717 | +2.96% | [−1.10, +6.66] | 7/8 |
| `ne_star` | H2020 | 7.772 | +2.35% | [−2.09, +6.24] | 6/8 |
| `ewma` | this work | 8.198 | −2.47% | [−4.90, +0.01] | 3/8 |
| consensus, point mass | BC2021 | 8.854 | −11.44% | [−26.54, +4.16] | 3/8 |
| `ep_x-ne_star` | H2020 | 8.870 | −11.47% | [−29.63, +2.33] | 3/8 |
| `ep_x` | H2020 | 8.899 | −11.23% | [−27.78, +1.33] | 3/8 |
| VaxSeer (self-trained) | Shi2025 | 9.171 | −13.05% | [−41.20, +7.39] | 5/8 |
| top-LBI strain, point mass | Neher2014/BC2021 | 10.645 | −35.10% | [−55.29, −15.69] | 1/8 |
| persistence | — | 7.970 | | | |
| floor | — | 3.389 | | | |

`mlr+mom+K` keeps the top of the table with nothing chosen from the future, and it is the
only row that is not a reweighting of the observed support -- its entitled floor is 1.696
against 3.389, which the CSV carries as a per-row note and which `captured` enforces.
`mlr+mom` is third of six fitted models and **nothing in the top five is separable**.
The 8-of-8 is shared only with its own parent.

**Regression check**: `--backbone published --their-arm published --kernel-arm oracle`
still prints `mlr+mom+K` 8.038 / +7.58% / 8-of-8, `d_cons` 8.211, `mlr+mom` 8.220 /
+5.22% / 8-of-8 -- the published panel unchanged, so the repoint is a repoint.

### Artifacts

- `antiGen/eval_figures.py` (`--kernel-arm`, `KERNEL_ARMS`, `kernel_oracle`, the raising
  guard); `causalgate/summarize_causalgate.py` (wins against persistence as well as
  against the parent, and the gate-read/gate-blind labels corrected -- on the masked
  panel the split is 4 and 4, not 5 and 3).
- `~/antiGen-count-loss-runs/figures/` regenerated on the causal default;
  `ARCHIVE_oracle_kernel_0928/` holds the previous set and
  `ARCHIVE_publishedtheirs_0924/` the one before that.
- `~/antiGen-count-loss-runs/panel_kernel_refill.slurm` (job 3060596).
- No model code changed this round; `eval_figures` is a reporting change and the two
  measurements were already on disk. 390 tests pass.

### 4. The LaTeX derivation, brought onto the convention

`writeup/mlr_mom_K.tex` was written on 2026-09-24 and its caveat 2 read "the kernel's own
constants were not chosen by the causal selector". That is now false, so the note was
rewritten rather than patched:

- **New `\subsection{The kernel's own constants}`** derives the Danskin right-derivative
  `dW/dmu|_0 = 1 - p_flat - 2 p_down`, says plainly that it is an *oracle* quantity, and
  states the readability rule (`row j` is readable at origin `i` iff `j < i`) as a
  displayed quote. The claim that every quantity entering the forecast is a function of
  strictly earlier data now covers the kernel's constants and is made in one place.
- **The result table is on the standing convention**: causal kernel arm, their models
  selected. `mlr+mom+K` 7.488 / +6.11% / 8-of-8; `d_cons` 7.566, `ne_star-lbi` 7.624,
  `mlr+mom` 7.661, `lbi` 7.677, `ne_star` 7.772; floors 3.389 and **2.541** (the
  eval-mask panel's extended floor, where the note previously carried the published
  panel's 2.780 under a footnote). The H2020 validation replication is a new paragraph.
- **Caveat 1 changed its meaning and had to be rewritten, not just renumbered.** Under the
  superseded arm `mu = 0.4` was selected at *every* origin, so the optimum sat on the grid
  boundary. Under the causal arm the selector uses 0.05, 0.1, 0.2 and 0.4, and `mu = 0` at
  the masked origin. The ceiling still binds at four of eight, so the defect is partial
  rather than total -- and a differently tempered kernel wanting a different mixing weight
  is the expected behaviour, not a surprise.
- Caveats added for the two exposures this round did not close: the H2020 run's
  transferred operating point (A2), and the symmetric-protocol choice with its
  direction stated (it costs their six models 0.042 AA on average, is worth +0.467 to
  `lbi`, and moves us from fourth to third).
- A note in §5.3 records why the small selected `tau_k` is not a sharper kernel: 14x fewer
  candidates, `p_down` 0.7425 -> 0.7415.

**Build fix, unrelated to content.** `pdflatex` on this machine fails fatally on
`microtype`'s font expansion against the default bitmap Computer Modern ("auto expansion
is only possible with scalable fonts"). `\usepackage{lmodern}` added before `microtype`:
same typeface as scalable Type 1 outlines, so nothing about the appearance changes. Three
passes, **zero errors, zero undefined references, zero overfull boxes**, 11 pages. Every
figure in the note was cross-checked against the run JSONs programmatically rather than
copied by hand.


---

## 2026-09-28 (later) — the allele momentum was selecting one timescale where its own analogue selects two

### The ask

> This is pedantic, but I don't love the asymmetry of how for the sequence momentum both
> the fast and slow timescales are selected, but for the allele momentum, we fix their
> ratio at 3. Can we select both the fast and slow timescales for allele momentum using
> validation? Then resubmit the array

Not pedantic. `CONS_SLOW_MULT = 3.0` had been in `mutation_kernel.py` since the consensus
arms were written, its comment says the pair was "borrowed rather than invented" from the
panel's `--tau-s-years` grid, and nothing since had revisited it. Under standing
convention 1 a pinned ratio is not a free pass: it is a selected parameter whose value was
chosen by analogy rather than by evidence.

### Where it is selected, and why not on the outer folds

The ask says "using validation", and the literal reading -- put both windows in the outer
grid beside `tau_f` and `tau_s` -- is not affordable. `mlr+mom+K`'s collapsed grid is
already 5,400 combinations times 3 folds and runs 1h to 10h per origin; six fast values
times nine slow ones is a further 20x, i.e. of order 200 hours per origin. **They go in
the descent gate instead**, which is where `tau_e`, `tau_k` and `m` already live: it is
causal per origin under convention 1, it scores no forecast so it costs the multiple-looks
budget nothing, and the whole gate table now takes 3.5 minutes per transition. The
substantive property the ask is about -- both timescales chosen from data rather than one
pinned to the other -- is delivered either way.

### The grid, and the guard that had to come with it

Fast over `{0.15, 0.25, 0.5, 1, 2, 4}` as before; slow over
`{0.3, 0.5, 0.75, 1, 1.5, 2, 3, 6, 12}`, spanning ratios from 1.2x to 80x and bracketing
the old 3x on both sides. Pairs with slow <= fast are skipped, and `consensus_freq` now
**raises** on them rather than producing a momentum term that points backwards -- the ratio
is the entire content of the term, so that is a different model, not a bad setting.

`reads_slow(arm)` is new and load-bearing. `kcons` is the `delta = 0` control and never
builds a slow table, so sweeping one for it would emit configurations identical in every
number the gate reports, and `causal_operating_point`'s argmin among them would be settled
by tie-break. One-timescale arms get exactly one slow entry, `None`, and their rows say so.

### What the gate says, before any forecast is scored

Table: 8,118 rows, 738 configurations x 11 transitions (job 3061557, 24s to 3m40s a task).

**The regression check is the one that matters and it passes exactly.** `kcons` reads no
slow table, so freeing the window must leave it untouched: its causal pick and its `deriv`
are **identical at all nine origins** to the pinned run's. The sweep reaches only the arm
that can see it.

For `kcmom`, mean `dW/dmu` at each origin's causal pick, pinned against free:

| origin | pinned 3x | free slow | gain | ratio selected |
| ---: | ---: | ---: | ---: | --- |
| 2017 | −0.4738 | −0.6043 | −0.1305 | 0.25 / 6 = **24x** |
| 2018 | −0.4822 | −0.5418 | −0.0596 | 0.15 / 0.75 = 5x |
| 2019 | −0.5101 | −0.5400 | −0.0299 | 0.25 / 2 = 8x |
| 2020 | −0.4918 | −0.5167 | −0.0249 | 0.25 / 2 = 8x |
| 2021 | −0.5068 | −0.5576 | −0.0508 | 0.25 / 2 = 8x |
| 2022 | −0.5364 | −0.5711 | −0.0346 | 0.15 / 0.75 = 5x |
| 2023 | −0.5340 | −0.5649 | −0.0309 | 0.15 / 0.75 = 5x |
| 2024 | −0.5487 | −0.5787 | −0.0300 | 0.15 / 0.75 = 5x |
| 2025 | −0.5636 | −0.5892 | −0.0256 | 0.15 / 0.75 = 5x |

Better at all nine. **Not one origin chose a ratio near 3**: the selector wants 5x or 8x,
and the fast window it pairs them with is shorter than the old 0.25 at six of nine. A
direct smoke at 2017->2018 puts the same point starkly -- at fast 0.25 the derivative runs
−0.4629 (slow 0.5), −0.4323 (slow 1.0), −0.3119 (slow 3.0), so the pinned ratio was the
worst of the three offered.

**This does not license a forecast claim.** A first-order slope at `mu = 0` has been six
times too pessimistic once (`kcmom`) and too optimistic once (`kemp`) in this line of
work. The sign of the expected change is known; the magnitude is not, and the scored array
is the measurement.

### Status

- Gate: job **3061557**, complete, merged to `kslow/gate_table.json`.
- Scoring: job **3061603**, nine origins, `--time=2-00:00:00` and 32G. The 16h `causalgate`
  used would probably have held, but the H2020 array lost seven tasks to a self-imposed
  24h cap the same week and the partitions allow fourteen days; there is no reason to cut
  it fine. **Running, not finished.**
- Until it reports, the headline `mlr+mom+K` row stays at 7.488 / +6.11% / 8-of-8 from
  `causalgate/`, and `eval_figures` stays pointed there.
- `writeup/mlr_mom_K.tex` will need a third pass: §5.2 still says the slow window is
  "declared, not selected, and is borrowed from the panel's own `tau_s` grid", which this
  round makes false. Deliberately not edited yet -- the numbers it would carry do not
  exist.

### Artifacts

- `antiGen/mutation_kernel.py` (`consensus_freq(..., tau_slow)`, `reads_slow`,
  `causal_operating_point` over four fields), `antiGen/kernel_gate.py` (`--tau-e-slows`,
  per-row `tau_e_slow`, the degenerate-sweep collapse), `antiGen/season_momentum.py`
  (`--tau-e-slow`, four-tuple `kernel_point`), tests updated plus two new ones. 392 pass.
- `~/antiGen-count-loss-runs/kslow/` — `gate.slurm`, `merge_gate.py`, `gate_table.json`,
  `score.slurm`, `summarize_kslow.py`.
- Code snapshot `/large_storage/hielab/ispecht/antigen-code/wt-3db611efcadb`.

---

## 2026-09-28 (later still) — the backbone tree was truncated mid-season; the sweep is cancelled, the tree is swapped, and the panel is re-run with its control

### The ask

> Currently we are running a sweep with the momentum `tau`'s for allele frequencies
> selected by validation. However, I just realized that in the backbone dataset, the 2026
> season is truncated. Cancel that sweep, swap out the backbone tree with the new one I
> just downloaded at `tree_pulled_2026-09-28.json`, and resubmit the sweep.

### What was wrong, stated precisely, because "truncated" understates it

`~/h3n2/tree.json` was the Nextstrain `seasonal-flu` build stamped `meta.updated =
2026-04-02`. Its last tip is **2026.193**. Season 2026 runs 2025.75 to **2026.25** and is
the *target* of this panel's last scored transition, 2025 -> 2026. So that transition was
being scored against a season that had not finished arriving — and not only the last three
weeks of it: sequences from a season keep landing for months, which is why the season-2026
tip count goes up by a quarter in a build pulled five months later.

But the tail is not the whole difference. **A rolling Nextstrain build re-subsamples**, so
every season moved:

| tree | updated | tips | last tip | s2015 | s2016 | s2023 | s2024 | s2025 | s2026 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tree_2026-04-02` | 2026-04-02 | 1683 | 2026.193 | 82 | 73 | 96 | 102 | 94 | **68** |
| `tree_2026-09-24` | 2026-09-24 | 1664 | 2026.675 | 78 | 64 | 91 | 92 | 86 | **84** |

and the root moved **2007.758 -> 2008.241**. So this is not a patch to one transition:
**every transition on the backbone panel has to be recomputed, and nothing from the old
tree may be mixed with or compared against anything from the new one.**

What did *not* change, checked rather than assumed: both builds carry the same
`root_sequence`, identical to `~/h3n2/ref.json` in all three segments, and both come from
`nextstrain/seasonal-flu`. So the genotype construction in `load_tips` is unchanged and
`ref.json` did not have to be reissued. The annotation set moved a little
(`emerging_haplotype` -> `emerging_haplotype_ha`, `qc.overallStatus` ->
`qc.overallStatus_ha`, `kikawa_2025a_cTiterSub` gone), which is harmless here: the only
`node_attrs` key any of this code reads is `num_date`, and `backbone_compare` recomputes
`lbi` and `ne_star` from the topology rather than reading the tree's own columns.

**Install.** The old build is kept as `~/h3n2/tree_2026-04-02.json`, the new one as
`~/h3n2/tree_2026-09-24.json` — named by `meta.updated`, which is the auditable
provenance — and `~/h3n2/tree.json` is now a symlink, so `ls -l` says which build is live
rather than leaving it to a file date. Every consumer picks it up without a flag.

### Cancelled

Job **3061603** (`kslow/score.slurm`), 20m23s in, all nine tasks. `kslow/out/` was empty,
so there is no partial output to be mistaken later for a finished arm. `kslow/` is left
intact as the record of the old-tree round and is superseded in full.

**Not cancelled: job 3060596**, the H2020 kernel refill. It runs `huddleston_panel
--repo`, a different dataset that never touches the backbone tree, and it is 1h40m into a
five-day allocation.

### The re-run is two arms, not one, and that was the one judgement call

Resubmitting the free-slow sweep alone would have produced a number with nothing to
compare it against: the arm it is measured against is `causalgate/out/cg_kcmom_*.json`,
the pinned-3x run, which stood on the truncated tree. **Asked and confirmed: both arms
go.** The second is not only a control — `causalgate/` is where the headline `mlr+mom+K`
7.488 / +6.11% / 8-of-8 came from, so that row inherits the truncation and has to be
recomputed whatever the slow window turns out to be worth.

- `causalgate_t0924/` — slow pinned at 3x fast. `causalgate/`'s protocol exactly.
- `kslow_t0924/` — slow swept. `kslow`'s protocol, plus the grid fix below.

Both on the **same code snapshot** `wt-3db611efcadb`, so the two arms differ in one thing.
That is a change from `causalgate/`, which ran `wt-010a07373114`; under the newer snapshot
omitting `--tau-e-slows` gives `tau_e_slows = [None]` and `consensus_freq` falls back to
`CONS_SLOW_MULT * fast`, i.e. the same pinned 3x. The equivalence is measured, not
assumed — job 3061557's `kcons` arm reproduced 3011558's causal pick and `deriv`
identically at all nine origins under this snapshot — and it is re-checked below.

No model code changed this round. The only edit to the repo is the notes.

### A grid defect the re-run made free to fix, found in the first merge

The pinned arm's candidate set is the six pairs `(f, 3f)` over the fast grid. The free
arm's slow grid was `{0.3, 0.5, 0.75, 1, 1.5, 2, 3, 6, 12}` — which contains `3f` for five
of the six fast values but **not `3 x 0.15 = 0.45`**. And the pinned arm selects `fast =
0.15` at eight of nine origins. So the free arm could not express the pinned arm's own
operating point, and on the first merge that cost it the comparison at 2017->2018
(−0.5306 against the pinned −0.5483) and a hair at 2020: **a lattice artifact wearing the
clothes of a measurement.**

The defect was in 3061603 too — the tree swap did not introduce it — but comparability
with 3061603 was already gone, so the fix was free to make. **Asked and confirmed: `0.45`
added to `--tau-e-slows`.** It pairs only with `fast = 0.15`; every other fast value skips
it under the `slow > fast` rule, so it is 18 extra `kcmom` configurations (774 against
738) and about four minutes of gate. The free candidate set now **strictly contains** the
pinned one, so the free arm's gate criterion cannot lose to the pinned arm by
construction, and any difference the scored arrays report is the measurement rather than
the grid.

### The gate tables, before either forecast is scored

Free: 8,514 rows, 774 configurations x 11 transitions (job 3061903; 3061846 was the
pre-fix merge and is superseded). Pinned: 2,376 rows, 216 x 11 (job 3061864).

| origin | free fast/slow | ratio | free dW/dmu | pin fast | pin dW/dmu | gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2017 | 0.25 / 3 | 12x | −0.5999 | 0.15 | −0.5909 | −0.0090 |
| 2018 | 0.15 / 0.45 | 3x | −0.5483 | 0.15 | −0.5483 | tie |
| 2019 | 0.25 / 3 | 12x | −0.6011 | 0.15 | −0.5864 | −0.0148 |
| 2020 | 0.15 / 0.45 | 3x | −0.5373 | 0.15 | −0.5373 | tie |
| 2021 | 0.15 / 1 | 7x | −0.5584 | 0.15 | −0.5427 | −0.0157 |
| 2022 | 0.15 / 1 | 7x | −0.5876 | 0.15 | −0.5654 | −0.0222 |
| 2023 | 0.15 / 0.75 | 5x | −0.5787 | 0.15 | −0.5605 | −0.0182 |
| 2024 | 0.15 / 0.75 | 5x | −0.5876 | 0.15 | −0.5719 | −0.0157 |
| 2025 | 0.15 / 0.75 | 5x | −0.5973 | 0.15 | −0.5837 | −0.0136 |

**Free better at 7 of 9, tied at 2, worse at none; mean gain −0.0121.** The two ties are
origins where the free selector chose the pinned point itself, which is what a superset
grid is for. (They print as `+0.0000` rather than `0`: `0.45` parsed from the grid against
`CONS_SLOW_MULT * 0.15 = 0.44999999999999996` differ in the sixteenth digit, so
`compare_gate_arms.py` calls anything under 1e-9 a tie rather than reporting the width of
a double as a result.)

**Two things changed against the old tree and both are worth recording.** The gains are
about a third of what the truncated tree reported (−0.012 mean against −0.060), and the
selected ratios moved: 2017 and 2019 now want **12x**, where before no origin went above
8x. The direction survives the tree change — **not one origin selects a ratio near 3
except the two that tie there** — but the magnitude did not, which is the second time in
this line of work that a gate slope has not transferred across a change of panel.

**The `kcons` regression check passes at 9 of 9.** `kcons` has `delta = 0` and reads no
slow table, so freeing the window must leave its pick and its `deriv` untouched; identical
in both tables at every origin. The sweep reaches only the arm that can see it, which is
the check that makes the pinned/free pair interpretable at all.

**And none of this licenses a forecast claim.** `dW/dmu` is a first-order slope at
`mu = 0`; it has been six times too pessimistic once (`kcmom`) and too optimistic once
(`kemp`). The scored arrays are the measurement.

### Status

- **Pinned control**: job **3061914**, nine origins, running.
- **Free slow window**: job **3061932**, nine origins, running.
- Both `--time=2-00:00:00`, 32G, 9-wide. `causalgate`'s 3011610 ran 1h01m to 10h07m on the
  same nine origins, so **expect both to land together in about 10–12 hours**; the two-day
  limit is slack, not an estimate.
- Every task carries a **tree preflight** that refuses to run unless `$H/tree.json` is the
  2026-09-24 build with a tip past 2026.4, and a **gate-table preflight** — the free arm
  refuses a one-timescale table, the pinned arm refuses a two-timescale one. Two
  directories holding the same filenames and differing only in which gate table they read
  is exactly the mistake worth spending a preflight on.

### What is now known to be standing on the truncated tree, and is not yet fixed

Everything downstream of `causalgate/`, which is to say the current headline table and
everything built from it. Listed so it is not rediscovered:

- `causalgate/` — the `mlr+mom+K` row, 7.488 / +6.11% / 8-of-8. **3061914 replaces it.**
- `censor/` (3011506), `theirselect/` (3011787), and the `backbone_compare` runs behind
  every their-side column — `d_cons`, `lbi`, `ne_star`, `ne_star-lbi`, `ep_x`, VaxSeer.
  **Not resubmitted this round**; the headline table cannot be regenerated end to end
  until they are.
- `figures/` and the CSV, and `writeup/mlr_mom_K.tex`, which carries the old numbers
  throughout and already needed a third pass for §5.2.
- The `published` masking panel, unchanged and unaffected — it is a different panel.

Not affected: the H2020 panel (`huddleston-*`, job 3060596), which never reads this tree,
and the generative antiGen cutoff runs of 2026-09-23 — though note that the
`~/antiGen-data/cache_h3n2` manifest pins `tree_json` by path, size and mtime and records
`root_time: 2007.758`, so the next generative run against the swapped tree will miss the
cache and rebuild it, and **its cutoff arithmetic changes**: cutoffs are years after the
root date, and the root moved by 0.483 years.

### Artifacts

- `~/h3n2/tree_2026-09-24.json` (from `antiGen/tree_pulled_2026-09-28.json`),
  `~/h3n2/tree_2026-04-02.json` (the superseded build), `~/h3n2/tree.json` -> symlink.
- `~/antiGen-count-loss-runs/causalgate_t0924/` — `gate.slurm` (3061864), `merge_gate.py`,
  `gate_table.json`, `score.slurm` (3061914, **running**).
- `~/antiGen-count-loss-runs/kslow_t0924/` — `gate.slurm` (3061846 pre-fix, 3061903 final),
  `merge_gate.py`, `gate_table.json`, `compare_gate_arms.py`, `score.slurm` (3061932,
  **running**), `summarize_kslow.py`.
- Code snapshot unchanged: `/large_storage/hielab/ispecht/antigen-code/wt-3db611efcadb`.

---

## 2026-09-29 — both arms land on the corrected tree: the slow window is worse when freed, and the 8-of-8 does not survive the retree

### The ask

> Report on the completed array with the new tree

Jobs **3061914** (pinned 3x) and **3061932** (slow selected), nine origins each, all
eighteen tasks COMPLETED, 42m to 11h49m. The two arms ran within four minutes of each
other at every origin, which is what a one-variable change should cost.

### The two checks that make the rest readable, both pass

**Free against pinned, families that read no kernel: `0.0e+00` at all four.** `season`,
`ewma`, `mlr` and `mlr+mom` are bit-identical between the arms, so the only thing that
moved is the slow window.

**New tree against old, same four families: they move, by up to 1.332 AA.** This is the
check the tree preflight cannot do — a preflight reads which file is on disk, not which
numbers came out of it. The panel really is a different panel.

| transition | support, old -> new | target, old -> new |
| --- | ---: | ---: |
| 2017->2018 | 271 -> 249 | 54 -> 52 |
| 2018->2019 | 348 -> 324 | 54 -> 52 |
| 2019->2020 | 433 -> 408 | 64 -> 63 |
| 2020->2021 | 435 -> 447 | 29 -> 30 |
| 2021->2022 | 404 -> 409 | 37 -> 36 |
| 2022->2023 | 375 -> 381 | 63 -> 70 |
| 2023->2024 | 392 -> 403 | 73 -> 66 |
| 2024->2025 | 418 -> 418 | 69 -> 66 |
| **2025->2026** | 442 -> 433 | **43 -> 60** |

The transition the swap was for gains **40% more target sequences**. Everything else moves
by a handful either way, which is the resubsampling.

### 1. The freed slow window is WORSE, and the gate said the opposite

| arm | mean AA | vs persistence | beats persistence | beats parent |
| --- | ---: | ---: | ---: | ---: |
| `mlr+mom` (parent) | 7.5725 | +5.96% [+1.16, +11.45] | 6/8 | — |
| **`mlr+mom+K`, slow pinned at 3x** | **7.4915** | **+7.17% [+2.04, +12.76]** | **7/8** | 7/8 |
| `mlr+mom+K`, slow selected | 7.4976 | +7.09% [+1.93, +12.73] | 6/8 | 7/8 |
| `season+K`, pinned | 7.9709 | +0.77% | 8/8 | 3/8 |
| `season+K`, selected | 7.9750 | +0.72% | 8/8 | 2/8 |
| persistence | 8.0274 | | | |
| floor | 3.6355 | | | |

**freed − pinned: +0.0061 AA [+0.0017, +0.0129], worse in 7 of 8, tied in 1, better in
none.** The interval excludes zero. It is a small effect — a tenth of the kernel's own
gain — but unlike every previous null in this project it is *consistent*: the freed arm
does not win a single transition, and the one tie is 2018->2019, where the free selector
chose the pinned point itself.

**This is the third time a gate slope has failed to predict a scored outcome, and the
first time it has failed in sign.** The gate, on this same tree, had the freed window
better at 7 of 9 origins by a mean `dW/dmu` of −0.0121 and worse at none. The forecast
says the reverse at 7 of 8. The round's own script said the magnitude was not known; what
it did not say, and should have, is that the sign was not known either. `dW/dmu` is a
first-order slope at `mu = 0` in a quantity computed from the realised target, and the
selector reads it as if it ranked forecasts. It does not.

So **the pinned 3x survives its own audit.** The asymmetry that motivated the round — the
genotype momentum selects two timescales, the allele momentum selected one — is real and
was worth testing, and the answer is that the borrowed 3 was not costing anything. It is
now a measured choice rather than an inherited one, which was the point.

| transition | pers | parent | pinned | free | free − pin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2017->2018 | 6.148 | 6.105 | 6.108 | 6.110 | +0.0018 |
| 2018->2019 | 7.025 | 5.551 | 5.540 | 5.540 | tie |
| 2019->2020 | 7.724 | 7.906 | 7.883 | 7.888 | +0.0044 |
| 2020->2021 | 12.895 | 13.529 | 13.583 | 13.583 | tie (masked) |
| 2021->2022 | 9.972 | 8.726 | 8.690 | 8.692 | +0.0013 |
| 2022->2023 | 7.983 | 8.035 | 7.963 | 7.991 | +0.0281 |
| 2023->2024 | 9.960 | 9.396 | 9.319 | 9.319 | +0.0005 |
| 2024->2025 | 4.401 | 3.969 | 3.725 | 3.732 | +0.0071 |
| 2025->2026 | 11.007 | 10.893 | 10.705 | 10.710 | +0.0054 |

The selected ratios were 12x at 2017 and 2019, 7x at 2021-2022, 5x at 2023-2025, and the
pinned point itself at 2018 and 2020. Both arms select the same `mu` at every origin, at
the 0.4 ceiling in 4 of 8 — unchanged from the old tree, so the ceiling exposure is
neither better nor worse than it was.

### 2. **THE 8-OF-8 DOES NOT SURVIVE THE CORRECTED TREE.** This retires finding 3 as it was stated

| | old tree (truncated) | new tree, pinned |
| --- | ---: | ---: |
| `mlr+mom+K` mean AA | 7.4882 | 7.4915 |
| vs persistence | +6.11% | **+7.17%** |
| **beats persistence** | **8/8** | **7/8** |
| `mlr+mom` beats persistence | 8/8 | **6/8** |

It is one transition, **2019->2020**, and it is a genuine reversal rather than a margin
wobble: on the old tree `mlr+mom+K` beat persistence 7.511 against 7.574, on the new one
it loses 7.883 against 7.724. The parent loses there and at 2022->2023 as well.

**Finding 3 of the 2026-09-22 headline entry — "what separates our model is stability, not
level", resting on a win count no other model matched — is no longer supported in that
form.** What is left is stronger on level and weaker on stability: the percentage gain
went *up* (+6.11% -> +7.17%), because persistence got worse on the corrected panel
(7.9704 -> 8.0274) while our arm barely moved. Whether 7-of-8 is still uniquely ours
cannot be said yet, because their columns have not been re-run (below).

**The tree delta itself is not separable: +0.0033 AA [−0.4759, +0.5095], higher in 4 of
8.** The mean sits still while the individual transitions move by up to 1.3 AA, so the
retree is not a small perturbation that happened to cancel — it is a different panel whose
average lands in the same place. The transition that motivated the swap, 2025->2026,
barely moved in level (10.608 -> 10.705) despite gaining 40% of its target sequences; the
movement is at 2018->2019 (6.824 -> 5.540) and 2022->2023 (6.692 -> 7.963), neither of
which was truncated. **So the headline changed because of the resubsampling, not because
of the truncation the swap was for.** That is worth stating plainly: the defect was real
and had to be fixed, and fixing it is not what moved the numbers.

The floors moved with the panel: 3.389 -> **3.6355** on support, 2.5407 -> **2.7455**
extended. `mlr+mom+K` is still the only row entitled to the second.

### What can and cannot be reported from this

**Can**: the pinned-vs-free comparison, which is a controlled one-variable measurement on
one tree, and the `mlr+mom+K` / `mlr+mom` / `mlr` / `ewma` / persistence rows above.

**Cannot: the headline table.** Every their-side column — `d_cons`, `lbi`, `ne_star`,
`ne_star-lbi`, `ep_x`, VaxSeer — is still on the truncated tree, as are `censor/` (3011506)
and `theirselect/` (3011787). Mixing them with the rows above would be comparing two
panels, which is the error this whole round exists to avoid. `eval_figures` still points
at `causalgate/`, and has deliberately not been repointed: under standing convention 1 a
panel with no causal arm raises rather than serving a superseded number unlabelled, and
the same rule should apply to a panel with no current-tree arm. **Until their side is
re-run, the reportable headline table is the old-tree one, and it is superseded in full.**

`writeup/mlr_mom_K.tex` is now wrong in three places rather than one: the result table,
caveat 1's ceiling count (unchanged at 4 of 8, so that one survives), and §5.2's claim
that the slow window is "declared, not selected" — which this round makes false, and makes
false in the direction of the declared value being right.

### Still exposed, unchanged

The `--mus` ceiling (4 of 8), `top_k`/`val_folds`/grid ranges, the eval mask, the H2020
kernel arm's transferred operating point, and per-horizon tree rebuilds for their metrics —
which this round makes concrete rather than theoretical: one build swap moved individual
transitions by over an AA, so their `lbi` and `ne_star`, read off a single present-day
tree, are exposed to the same thing at every origin.

### Artifacts

- `~/antiGen-count-loss-runs/causalgate_t0924/out/cg_kcmom_*.json` (3061914, 9/9),
  `~/antiGen-count-loss-runs/kslow_t0924/out/ks_kcmom_*.json` (3061932, 9/9).
- `kslow_t0924/summarize_t0924.py` — the nesting check, the tree check, both operating
  points, the paired bootstrap and the tree delta in one place.
- `kslow_t0924/compare_gate_arms.py` — the gate-side picks, retained because the gap
  between what it predicted and what section 1 measured is the finding.
- No model code changed; the only repo edit is this file.

---

## 2026-10-01 — the competitor columns are re-run on the corrected tree, the headline table exists again, and `distance_from_consensus` is now first

### The ask

> We need to re-run all of the competitor models on the new tree. Please implement this
> and submit an array.

The 2026-09-29 entry ended with "Cannot: the headline table": `causalgate_t0924` (3061914)
had recomputed our side on `tree_2026-09-24` and every their-side column — `d_cons`,
`lbi`, `ne_star`, `ne_star-lbi`, `ep_x`, `ep_x-ne_star`, the two point masses and VaxSeer
— still stood on the truncated 2026-04-02 build. This closes that.

### What was submitted, and the one thing in it that is not a competitor

All in `~/antiGen-count-loss-runs/competitors_t0924/`, all on snapshot `wt-3db611efcadb`,
which is the snapshot `causalgate_t0924` and `kslow_t0924` ran under.

| job | script | what | elapsed |
| ---: | --- | --- | --- |
| **3092889** | `theirs.slurm`, array 1-2 | `backbone_compare` both arms: `--select` (the reporting default) and `--no-select` (their published config) | 13m14s / 14s |
| **3092890** | `ours.slurm`, array 1-9 | `season,ewma,mlr,mlr+mom`, the control | 11-44s |
| **3092891** | `vaxseer.slurm`, array 1-9 | new support FASTAs, GPU inference on the existing checkpoints, re-scored | 30-36s |

20 of 20 tasks COMPLETED.

**Both their-arms, not just the default.** `eval_figures` exposes `--their-arm published`
and every entry before 2026-09-24 cites it. Re-running only the selected arm would have
left the flag quietly serving a different *panel*, not a different protocol.

**`ours.slurm` is the control, and it is there for one specific reason.**
`eval_figures.load_backbone` asserts

```
kern[s]["emd_mlr+mom"] == ours[s]["emd_mlr+mom"]      EXACTLY, 0.0
```

because `mu = 0` nests the parent. Serving both sides of that out of `causalgate_t0924`
would make the check compare a file with itself — it would pass forever, and stop meaning
anything at exactly the moment the panel is being rebuilt. It costs 11-44s a task. It
passes at 0.0e+00 across all nine origins, between two separately submitted jobs.

**VaxSeer is NOT retrained, and that is a claim with an argument.** `train.slurm` reads
`processed/2003-10_to_<origin>-04_2M/...`, which `prep_data.py` builds from
`~/h3n2/output/nextclade.cds_translation.HA.fasta` (202,046 sequences) and
`~/h3n2/metadata.tsv`. Neither is the tree and neither moved in the swap — only the
1,664-tip Nextstrain subsample did. So the dominance predictor's training set is
bit-identical under both builds and the nine checkpoints are still the right models. What
*is* tree-dependent is the support set (`Task(panel, origin).sup`) and the score, and that
is what the array redoes. The job asserts the mtimes rather than asserting it in prose,
and **refuses to reuse `vaxseer/scoring/`**: those `test_results.csv` files are inference
against the *old* support FASTAs, the ids are `sup00000...` in both, and `score_all.sh`
skips an origin whose CSV exists — so a stale CSV would have scored silently against the
wrong genotypes. That is the failure this round was one flag away from.

### The checks, which come before any average

**Four independent constructions of the same panel agree.** Each of the four sources
builds `TipPanel` and `Task` itself, so persistence and the floor are recomputed four
times over: max |delta| **7.11e-15** on persistence and **8.88e-16** on the floor, against
a 1e-9 tolerance. Plus the exact nesting identity above.

**Their columns move against the old tree, by a lot.** The check a tree preflight cannot
do — it reads which file is on disk, not which numbers came out of it. Max |delta| per
model: `lbi` **5.226**, `ne_star-lbi` 2.450, `d_cons` 2.156, `ep_x` 1.569, `ne_star`
1.406, `ep_x-ne_star` 0.914.

### **THE HEADLINE BACKBONE TABLE, ON THE CORRECTED TREE.** It supersedes the 2026-09-23 table in full

8 transitions, season 2021 masked from the averages only, their hyperparameters selected
per origin by our protocol, `mlr+mom+K` at the causal operating point with the slow window
pinned at 3x (which won its own audit on 2026-09-29).

| model | source | mean AA | vs persistence | 95% CI | beats pers |
| --- | --- | ---: | ---: | --- | ---: |
| **distance_from_consensus** | **H2020** | **7.2469** | **+10.23%** | [+3.33, +17.16] | **7/8** |
| mlr+mom+K | this work | 7.4915 | +7.17% | [+2.03, +12.74] | 7/8 |
| mlr+mom | this work | 7.5725 | +5.96% | [+1.23, +11.39] | 6/8 |
| mlr | this work | 7.5790 | +5.97% | [+2.00, +10.49] | 7/8 |
| ne_star-lbi | H2020/H2025 | 7.6562 | +5.08% | [−4.62, +14.76] | 6/8 |
| ne_star | H2020 | 7.9114 | +1.42% | [−0.53, +3.41] | 6/8 |
| season+K | this work | 7.9709 | +0.77% | [+0.57, +1.03] | 8/8 |
| lbi | H2020 | 8.0639 | +0.68% | [−16.15, +15.60] | 5/8 |
| ewma | this work | 8.2062 | −1.68% | [−6.28, +2.36] | 3/8 |
| consensus (point mass) | BC2021 | 8.3137 | −2.80% | [−15.40, +9.65] | 4/8 |
| ep_x | H2020 | 8.7078 | −7.58% | [−25.88, +6.35] | 5/8 |
| VaxSeer (self-trained) | Shi2025 | 8.7124 | −6.25% | [−33.05, +10.92] | 6/8 |
| ep_x-ne_star | H2020 | 8.9712 | −10.46% | [−29.71, +4.24] | 5/8 |
| top-LBI strain (point mass) | Neher2014/BC2021 | 10.9051 | −35.98% | [−74.72, +0.67] | 3/8 |
| persistence | | 8.0274 | | | |
| floor (support) | | 3.6355 | | | |
| floor (extended, `mlr+mom+K`) | | 2.7455 | | | |

### **WE ARE NO LONGER FIRST.** `distance_from_consensus` takes the top row and this has to be said straight

On the truncated tree `mlr+mom+K` was 7.4882 against `d_cons` 7.5656. On the corrected
tree it is 7.4915 against **7.2469**. Our arm barely moved (+0.0033); theirs improved by
**−0.3187**, and the ordering flipped.

Paired, over the 8 scored transitions:

| | ours − theirs | 95% CI | ours better in |
| --- | ---: | --- | ---: |
| old tree | −0.0774 | [−0.3753, +0.1952] | 3/8 |
| **new tree** | **+0.2446** | **[−0.0378, +0.5552]** | **2/8** |

**The interval straddles zero, so the gap is not separable at 95% — but the win count is
2 of 8 and the direction is consistent.** Per transition, ours − `d_cons`: +0.030, +0.394,
−0.266, +0.475, +1.068, −0.356, +0.126, +0.485. We lose at 2019->2020 and 2023->2024 and
lose everywhere else, worst at 2022->2023 by over an AA. The honest statement is that
`mlr+mom+K` and `distance_from_consensus` are not separated by this panel and that
`d_cons` is ahead on both the mean and the count.

This is the third thing the retree has cost: the 8-of-8 (2026-09-29), finding 3's
stability claim (same), and now the top row. In all three cases the *defect* was real, the
fix was mandatory, and **the resubsampling rather than the truncation is what moved the
numbers** — 2025->2026, the transition the swap was for, gained 40% of its targets and
barely moved.

### The symmetric protocol, re-measured on this tree

Standing convention 2 says both sides get the same per-origin selector. On the corrected
tree it is worth **−0.0311 AA** to their six on average — i.e. it helps them, as it did
before, but barely, and it is not uniform: `ne_star-lbi` −0.284 and `lbi` −0.257 gain,
`ep_x-ne_star` +0.317 and `ne_star` +0.220 lose. `d_cons` is indifferent (7.2469 selected,
7.2253 published), so **the row that now leads the table does not lead it because of the
protocol choice.** That is worth knowing: the one decision that could be read as
flattering their side is not what put them on top.

### The tree delta, their side

| model | old | new | delta | old wins | new wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| distance_from_consensus | 7.5656 | 7.2469 | **−0.3187** | 6/8 | 7/8 |
| ne_star | 7.7724 | 7.9114 | +0.1390 | 6/8 | 6/8 |
| ne_star-lbi | 7.6240 | 7.6562 | +0.0322 | 6/8 | 6/8 |
| ep_x-ne_star | 8.8700 | 8.9712 | +0.1012 | 3/8 | 5/8 |
| lbi | 7.6772 | 8.0639 | **+0.3866** | 5/8 | 5/8 |
| ep_x | 8.8989 | 8.7078 | −0.1911 | 3/8 | 5/8 |
| consensus (point mass) | 8.8537 | 8.3137 | −0.5399 | 3/8 | 4/8 |
| top-LBI (point mass) | 10.6452 | 10.9051 | +0.2600 | 1/8 | 3/8 |
| VaxSeer | 9.1709 | 8.7124 | −0.4585 | — | — |
| persistence | 7.9704 | 8.0274 | +0.0571 | | |

`lbi` moves most and in the direction that hurts it, which is the concrete form of the
standing exposure: `lbi` and `ne_star` are read off a *single present-day tree* at every
origin, and one build swap is worth 0.39 AA to `lbi`. Per-horizon tree rebuilds remain the
only fix and remain not done.

### Code

`antiGen/eval_figures.py` only; no model code changed.

- The `evalmask` panel points at `competitors_t0924/` and `causalgate_t0924/`.
- **Every panel now carries a `tree` field and it prints in the blurb**, so no figure or
  CSV can be read without knowing which build it stands on. The field is the label; the
  four-way persistence assertion is the check, and it cannot pass across a build change.
- `vaxseer` became panel-scoped. `published` keeps the old-tree VaxSeer because the runs
  behind it (`dfc/`, `backbone3way/`) were not re-run and are wholly 2026-04-02.
- `evalmask`'s `kernel_oracle` is `None`: the oracle arm only ever ran on the truncated
  build, so asking for it raises rather than reaching across trees for it.

### Still standing on the truncated tree, and not fixed by this round

- The **`published` masking panel** in its entirety — `dfc/` and `backbone3way/`. It is
  internally consistent and now labelled 2026-04-02 wherever it appears.
- `censor/` (3011506), finding A3's feature-censoring measurement.
- `figures/` and the CSV, which have not been regenerated, and
  `writeup/mlr_mom_K.tex`, which is now wrong in four places rather than three: the
  result table, §5.2's "declared, not selected" claim, and now the ordering — the note
  says `mlr+mom+K` leads and it does not.

### Still exposed, unchanged

The `--mus` ceiling (4 of 8), `top_k`/`val_folds`/grid ranges, the eval mask, the H2020
kernel arm's transferred operating point, and per-horizon tree rebuilds for their metrics
— which this round makes sharper still, since `lbi` moved 0.39 AA for one build swap.

### Artifacts

- `~/antiGen-count-loss-runs/competitors_t0924/` — `theirs.slurm` (3092889),
  `ours.slurm` (3092890), `vaxseer.slurm` (3092891), snapshot-pathed copies of
  `make_support_fasta.py` and `score_vaxseer.py`, `summarize_competitors.py`, and
  `out/` holding `theirs_select.json`, `theirs_published.json`, `ours_*.json`,
  `vaxseer_*.json`.
- Code snapshot unchanged: `/large_storage/hielab/ispecht/antigen-code/wt-3db611efcadb`.

---

## 2026-10-02 — the additive panel is built: `panel_manifest.py`, two real manifests, and additivity that passes on the actual origins

### The ask

> Read NOTES_2 to get oriented. We are in the process of implementing
> PROPOSAL_additive_panel.md. Please continue with implementation, and as you do, record
> to NOTES_2 what progress you make

`PROPOSAL_additive_panel.md` is the plan; this is the first entry that reports anything
built against it. **No forecast has been scored on the new panel.** Everything below is
either code, a structural test, or a zero-fit measurement on data already on disk. The
headline table of 2026-10-01 still stands on `tree_2026-09-24.json` and is untouched by
this entry.

### 1. What now exists

`antiGen/panel_manifest.py` (914 lines) and `antiGen/tests/test_panel_manifest.py`
(39 tests). The three layers of the proposal's §3 are three things in the module:

- **the corpus**, read by `read_corpus` from `metadata.tsv` + `nextclade.tsv` (+ an
  optional `--extra-metadata` for the columns a GISAID re-pull would supply);
- **the subsample**, `admit`, which is twelve lines: order each month-cell by
  `u(s) = sha256(accession ‖ salt)`, take the first `K`, and union with whatever a prior
  manifest already admitted, which is what makes a refresh insert-never-evict;
- **the horizon**, `Manifest.accessions_at(T, level)`, the date-prefix.

Plus `verify` (never writes), `preflight` (the hash check, exit 42 on a mismatch), and
`show` (header, cell sizes, and the additivity report over a list of origins).

`season_momentum.load_alignment_dated` gained `--manifest`, and with it two behaviours
that are easy to miss:

1. the read is restricted to the admitted accessions, but **the horizon prefix is not
   applied at load time** — `TipPanel.support(T)` already takes it, and taking it twice
   would give a panel that could serve only one origin;
2. **dates come from the manifest, not from the metadata.** The pin is the fix for the 23
   tips the two backbone builds re-dated; a corpus that disagrees prints how many and is
   overruled.

`add_panel_source_args` / `tips_from_args` are now the single panel reader for
`season_momentum`, `kernel_gate` and `feature_gate` (and `--manifest` is wired through
`embed_genotypes` and `plm_marginals`), so a gate table and the score it gates cannot be
read off different sets of sequences.

### 2. The two manifests, built

```
panel_K30_v1.tsv  4,577 rows over 156 cells, 151 full at K=30
                  sha256 d4881a783e29fc641793f0b6c5aa43d7520601fa799a53ac85f0e43a3f4a4560
panel_K90_v1.tsv 13,277 rows over 156 cells, 139 full at K=90, 0 over K
                  sha256 ecf7a3d8130121027d1a1e8bd7b52b0991fe3f66cf33d44ae5480a52840ba38e
```

Both in `~/h3n2/`, both from salt `h3n2-ha-v1`, `cells=month`, `admission=topk`,
corpus version `2026-04-08` (`corpus_sha addf9442…`), nextclade dataset
`2026-01-14--19-24-43Z`. Of the 202,493 accessions, **every one carries a full
unambiguous Y-M-D**, so Huddleston's ambiguous-date exclusion costs nothing, as §1 said.
The filters drop `qc 5,661` and `coverage 452`.

**The coverage rule had a hole and it is closed.** `passes` read
`min_coverage is not None and rec.coverage is not None and rec.coverage < min_coverage`,
so a record with *no* coverage figure passed a threshold it had never been measured
against. There are 444 such accessions, and they are exactly the 444 that nextclade could
not align and that are therefore absent from `nextclade.cds_translation.HA.fasta` — the
13,689-of-13,792 shortfall §6 measured. Admitting them declares a panel size with no
sequence behind it. The rule now reads "a declared threshold the record has no figure for
is a failure, not a pass", stays pointwise, and is pinned by
`test_a_record_with_no_coverage_figure_fails_a_declared_threshold`.

`min_coverage` is **declared, not selected**: Huddleston's ">= 900 nucleotides" against
this nextclade dataset's 1,718 nt HA reference, i.e. 0.5239. Nothing about the record was
looked at to choose it.

### 3. **ADDITIVITY PASSES ON THE REAL ORIGINS.** `K = 90`, level 1

| T | 2017.25 | 2018.25 | 2019.25 | 2020.25 | 2021.25 | 2022.25 | 2023.25 | 2024.25 | 2025.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \|S(T)\| | 4,528 | 5,607 | 6,693 | 7,765 | 8,359 | 9,231 | 10,311 | 11,390 | 12,476 |
| added | — | +1,079 | +1,086 | +1,072 | +594 | +872 | +1,080 | +1,079 | +1,086 |
| **lost** | — | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

Against the backbone's 459 of 1,682 lost over one five-month build gap. This is §13's
first success criterion, and it is a property of the rule rather than of this draw —
`test_additivity_every_horizon_prefix_contains_the_last` and the four tests beside it
assert it structurally, including under a refresh that backfills the past, under the
force-include designation rule, and at level 2.

### 4. The gate panel is a **subset** of the scored panel, not merely similar to it

§6 recommends gating at `K = 30` and scoring at `K = 90`, carrying the operating point
across, and records that carry as an exposure of the same shape as the H2020 kernel arm's
transferred point. Top-`K` in hash order makes it a much cheaper exposure to defend than
that one: the first 30 of a cell are the first 30 of its first 90, so

```
S(K=30) ⊂ S(K=90)        4,577 of 13,277, measured, and at every horizon
```

Every sequence the gate saw, the scored run also sees; nothing the gate chose on has been
redrawn. Pinned by `test_a_smaller_K_is_a_subset_of_a_larger_one_under_the_same_rule`.
It does not make the carry free — the gate still reads a thinner support than the one the
score runs on — but it removes the part of the exposure that was about resampling.

### 5. Season counts, measured on the filtered panels

| | s2015 | s2016 | s2017 | s2018 | s2019 | s2020 | s2021 | s2022 | s2023 | s2024 | s2025 | s2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tree (1,664 tips) | — | — | 94 | 77 | 75 | 87 | **36** | 54 | 91 | 92 | 86 | 84 |
| `K = 30` | 182 | 180 | 181 | 182 | 181 | 179 | 175 | 180 | 179 | 180 | 184 | 89\* |
| `K = 90` | 546 | 539 | 543 | 539 | 544 | 539 | **408** | 542 | 542 | 537 | 549 | 268\* |

\* s2026 is half-missing from this corpus (§7.2). The swing of the tree's per-season
count, 36–94, becomes 408–549 at `K = 90`, and **season 2021 holds 408 sequences instead
of 36** — which is what makes §13's third item a measurement rather than an assumption.

Transport sizes, at the origins §6 priced:

| `K` | origin | \|sup\| | \|tgt\| | product |
| ---: | --- | ---: | ---: | ---: |
| — | current backbone | 379–707 | 30–70 | ~3.1e4 |
| 30 | 2018.25 / 2022.25 / 2025.25 | 907 / 825 / 859 | 108 / 102 / 56 | ~7.7e4 |
| 90 | 2018.25 / 2022.25 / 2025.25 | 2,166 / 1,825 / 1,925 | 252 / 257 / 155 | ~4.4e5 |

So **~2.5x the current transport problem at `K = 30` and ~14x at `K = 90`** — cheaper
than §6's 4x/25x estimate, because that measurement was taken before the QC filter.

### 6. The panel's hash is in every run's output

`tips_from_args` now sets `a.panel_provenance` — source, manifest path, `manifest_sha256`,
`K`, cells, salt, corpus version, corpus sha, level, tree — and `season_momentum`,
`kernel_gate` and `feature_gate` all write it into their JSON under `panel`. With
`--manifest-sha` the run **refuses to start** on a mismatch, the direct analogue of the
tree preflight. A `--manifest` pointed at a tree run is now refused before the file is
opened, so the error says what is wrong rather than failing on whichever file is missing.

### 7. The short-lookback origins are now stated rather than discovered

§7.3: the corpus starts 2013-01-01, so at `lookback = 5` every origin before 2018.25 wants
support the panel does not hold. The backbone never showed this, because the Nextstrain
build's window was anchored to the build date and so always covered its own origins. The
roll now prints

```
SHORT LOOKBACK -- the panel starts at 2013.001 and these origins want history before it:
2013->2014 (-4.75y), 2014->2015 (-3.75y), 2015->2016 (-2.75y), 2016->2017 (-1.75y),
2017->2018 (-0.75y)
```

and records `lookback_shortfall` on every row. The proposal says such an origin is
"either dropped or carried with the shortfall stated"; it is carried, and the shortfall is
in the output. **The scored set on this panel is origins 2018, 2020, 2021, 2022, 2023,
2024, 2025 — seven transitions**, against the backbone eval's eight: 2017 is lost to the
short lookback and 2019 is absent as an origin for the same reason it always was, its
target being the COVID season 2020. That is the real loss of power §7.3 priced, and it is
7 rather than 8 exactly as predicted.

### 8. End to end, on the real panel

```
python -u -m antiGen.season_momentum \
  --alignment ~/h3n2/output/nextclade.cds_translation.HA.fasta \
  --metadata ~/h3n2/metadata.tsv --manifest ~/h3n2/panel_K30_v1.tsv \
  --manifest-sha d4881a783e29... --transitions 2022 --families season,mlr ...
```

runs in well under a minute on a 5-combination grid and reports
`2022->2023 floor 1.844 | season 6.627 mlr 6.438`. These are **not** comparable to any
number in this document — different panel, different floor — and are quoted only as
evidence that the path works. 4,571 of 4,577 sequences survive the 2% ambiguity floor;
2,337 distinct genotypes; 229 features over 98 varying sites.

### 9. The round is scaffolded and **not submitted**

`~/antiGen-count-loss-runs/panel_v1/` — `README.md`, `gate.slurm`, `score.slurm`,
`merge_gate.py`, `make_snapshot.sh`, and empty `gate/`, `out/`, `logs/`. Nothing is
queued. §11 prices this round as a third full re-run of both sides and §12 says every
number in the ledger moves again when it lands; that is a decision to take deliberately,
not a side effect of a session that was implementing the protocol. Two things are left
blank on purpose: `CODE=` (no snapshot has been made — the code was still moving today)
and the submission itself.

- **`gate.slurm`**, K = 30, 13 tasks. Field for field `causalgate_t0924/gate.slurm` —
  same grid, same arms, `--full-table --min-train 0 --drop-targets ""` — with the tree
  swapped for the manifest. Thirteen rather than eleven because the corpus starts 2013;
  the five short-lookback origins are gated but not scored, because
  `causal_operating_point` at origin 2018 may read only transitions closed by 2018.25,
  which is exactly 2013->2014 .. 2016->2017. **They are the only past this corpus has**,
  and that is an asymmetry against the earliest scored origin, recorded rather than
  discovered.
- **`score.slurm`**, K = 90, 7 tasks: 2018, 2020, 2021, 2022, 2023, 2024, 2025. Memory
  64G rather than 32G and a 4-day limit rather than 2, for ~14x the transport problem.
  Task 7 is the half-season — it is run so the shape of the shortfall is measured, not so
  the number is used.
- **Three preflights**, two of them new. The snapshot stamp as before; then
  `panel_manifest preflight` against the hash written into the script, which is the direct
  analogue of the tree preflight; then, in `score.slurm` only, a check that the merged
  gate table is stamped `causal_table` **and was gated at K = 30 on salt `h3n2-ha-v1`**.
  A gate table from the backbone tree or from the scored panel itself is refused by exit
  code. The manifest hash is checked twice by two code paths, since every run also gets
  `--manifest-sha`.
- **`merge_gate.py`** is `causalgate_t0924`'s with one assertion added, and it is the
  assertion the round exists to make: every shard must carry the same
  `panel.manifest_sha256` and the same level. Two shards differing only in which draw of
  the corpus they read is the failure of §1 placed inside a single table.
- **`make_snapshot.sh` declares the snapshot naming convention, which was never written
  down.** `78d9625` is a commit; the five `wt-*` directories were "content-hashed off the
  working tree" by a scheme that is not recorded and cannot be recovered — none of the
  five is a git object. From here `wt-<12 hex>` means sha256 over `<path>NUL<sha256>NUL` for
  every `*.py`/`*.toml` under `antiGen/` plus `pyproject.toml`, path-sorted. Deterministic
  across machines, independent of mtimes and of the git index. It does **not** reproduce
  the historical names and does not try to. The working tree as of this entry hashes to
  `651e4796fa86`.

### 10. Both paths smoke-tested on the real panel

`kernel_gate` on `panel_K30_v1`, origin 2022, one `tau_e` and two `tau_k`:

```
2022->2023: season EMD 6.6267  support 825  targets 102
  kcmom tau_e 0.5  tau_k 0.01  p_down 0.7741  dW/dmu -0.6952   61,113 candidates
  kcmom tau_e 0.5  tau_k 0.1   p_down 0.7726  dW/dmu -0.6429  476,025 candidates
  the bar is p_down > 0.4462 -> CLEARS
```

The shard carries the `panel` block, so `merge_gate.py`'s assertions have something to
read. Again: **not comparable to anything in this document** — different panel, different
floor. It is evidence the path runs, and a first sight of what the gate costs at this
size (577 candidates per parent at `tau_k = 0.1`, against the backbone's support of
379–707 genotypes rather than 825).

### What is not done

- **No scored run.** The gate table does not exist and neither array has been submitted.
  §11's "third full re-run of both sides" has not started.
- **`backbone_compare` cannot read a manifest**, and should not until §8's per-origin
  trees exist: `lbi` and `ne_star` are read off the topology. `~/h3n2/usher_work/final.pb`
  is a MAT holding 203,732 samples — the 1,683 backbone tips plus all 202,049
  nextclade-aligned sequences — so pruning it to `S(T)` is one `matUtils extract` away,
  but a prune of a present-day MAT keeps present-day topology and so does **not** give
  what §8 asks for. Re-estimating per origin needs `augur`/IQ-TREE, neither of which is
  installed here. This is the next decision, not a next step.
- **`huddleston_panel` takes no manifest, and needs none**: it runs on *their* panel,
  which §2 measured as already additive (0 of 2,441 in-overlap strain-timepoints dropped,
  0 of 13,347 re-dated).
- **Region stratification, passage exclusion and level 2 all wait on a GISAID re-pull**
  (§7.1, §5). `cells=region-month` raises rather than guessing; the passage rule is
  implemented and silently inert on bare `EPI_ISL_*` names, which is stated in
  `is_passaged`'s docstring; `--level 2` refuses a manifest with no submission column.
- **The corpus still ends 2025-12-31**, so s2026 is 268 sequences against 537–549 for a
  full season, and the 2025->2026 transition reproduces the exact truncation the
  2026-09-28 tree swap was for. §7.2 unchanged.
- No `outliers.txt` and no force-include list are pinned: both header fields are `.`.
  The force-include mechanism exists and is causal (a strain enters only at origins at or
  after its designation date) but has no list to read.

### Still exposed, unchanged

The `--mus` ceiling, `top_k`/`val_folds`/grid ranges, the eval mask, the H2020 kernel
arm's transferred operating point, and per-horizon tree rebuilds for their metrics. The
panel adds one of its own: **the `K = 30` gate / `K = 90` score carry**, reduced but not
removed by §4 above.

### Artifacts

- `antiGen/panel_manifest.py`, `antiGen/tests/test_panel_manifest.py` (39 tests).
- `~/h3n2/panel_K30_v1.tsv` @ `d4881a783e29`, `~/h3n2/panel_K90_v1.tsv` @ `ecf7a3d81301`.
  `panel_manifest verify` against the corpus they were built from: `would add: 0`,
  `verify: OK`. A rebuild from the same corpus and salt is **byte-identical**.
- `~/antiGen-count-loss-runs/panel_v1/` — `README.md`, `gate.slurm`, `score.slurm`,
  `merge_gate.py`, `make_snapshot.sh`. Unsubmitted; `CODE=` unfilled.
- Modified: `antiGen/season_momentum.py` (`--manifest`, `--manifest-sha`, `--level`,
  `observable`, `panel_provenance`, the short-lookback report),
  `antiGen/kernel_gate.py`, `antiGen/feature_gate.py`, `antiGen/embed_genotypes.py`,
  `antiGen/plm_marginals.py`.
- 435 tests pass.

---

## 2026-10-02 (later) — the gate is submitted on the additive panel; the candidate cap bites at K = 30, and `backbone_compare` needs more than one tree per origin

### The ask

> Read NOTES_2 to get oriented. We are in the process of implementing
> PROPOSAL_additive_panel.md. Please continue with implementation, and as you do, record
> to NOTES_2 what progress you make

The session opened by asking which of three open items to take, and was corrected:

> I am a bit confused--our goal is simply to replace the sequence set that was previously
> extracted from the backbone tree tips with a new sequence set that it additive per
> origin, like Huddleston et al. (2020). We should be working towards that goal of
> ultimately re-running with the new sequence set, and recomputing the evals.

**That is the directive, and the two side quests are dropped.** The corpus-version
sensitivity measurement (§13.2) and everything waiting on a GISAID re-pull are not being
worked on. The only question is what it takes to get both sides of the headline table
onto `panel_K*_v1` and the evals recomputed. The question the session asked was
over-framed and should not have been asked in that shape.

### 1. The snapshot exists, and `make_snapshot.sh` had two bugs

`wt-651e4796fa86` — **the hash the previous entry predicted**, which is the first evidence
that the naming convention declared earlier today actually reproduces on a second run.
`CODE=` is now filled in both `gate.slurm` and `score.slurm`.

It did not build on the first attempt. Two fixes, both in `make_snapshot.sh`:

- `tar -cf - antiGen pyproject.toml --exclude=...` puts the excludes *after* the paths,
  where GNU tar ignores them, warns, and exits 2. Moved in front.
- that failed run left `wt-651e4796fa86/` behind holding a half-extracted tree, and the
  script's `if [ -e "$DEST" ]` then reported **"snapshot already exists ... nothing to
  do"** and exited 0. A half-copied snapshot that claims to be complete is worse than no
  snapshot at all: the stamp assertion in both `.slurm` files would have passed on a
  `CODE` holding an arbitrary subset of the code. The check now requires a non-empty
  `.SNAPSHOT_COMMIT` and refuses the directory otherwise.

### 2. Smoke-tested at the worst origin, then submitted

Origin 2013 is the thinnest shard the gate runs: the panel starts 2013.001, so a 5-year
lookback has about three months behind it. Run from the snapshot at a 1x1x1 grid:

```
2013->2014: season EMD 4.4737  support 49  targets 94
  kcmom tau_e 0.5  tau_k 0.01  m 0  p_down 0.6182  dW/dmu -0.2662  --> CLEARS
```

Support **49 genotypes** — thin, but it produces a row rather than falling under
`--min-season-seqs`, which is what `merge_gate.py`'s raggedness assertion needs, and the
shard carries its `panel` block. Submitted on that basis: **`gate.slurm` is job
`3105705`, 13 tasks, K = 30.**

### 3. **Two shards failed, and the cause is the panel size meeting a guard**

Tasks 8 and 9 — origins 2020 and 2021 — exited non-zero on

```
ValueError: 530528 candidates for arm 'kcmom' at tau_k=0.1, m=0; the cap is 500000.
Truncate with a smaller m rather than scoring a kernel nobody asked for
```

520,654 for origin 2021. This is **not a bug and the guard did its job** —
`MAX_CANDIDATES = 500_000` in `mutation_kernel.py` exists to stop a kernel nobody asked
for from being scored. It is §6's arithmetic arriving: origin 2020 carries **944 support
genotypes** against the backbone's 379–707, and the candidate set is the support times
the admitted single-mutation neighbourhood, so it grows with the panel.

The previous session's smoke test is the warning that was there to be read and was not:
it reported **476,025 candidates at `tau_k = 0.1`** for origin 2022 and recorded the
number without noticing it was 95% of a cap.

**This is a grid problem, not a panel problem.** `--tau-ks 0.001,...,0.1,0.25,1` was
sized against the backbone's support. The two entries above `0.1` were always the
expensive end and on this panel the top of that range is unreachable at `m = 0`. The
tasks still running were predicted here to die the same way, and **that prediction is
wrong** — left in rather than quietly fixed. Origins 2017, 2018, 2022 and 2025 all
completed. Only **2020 and 2021** exceed the cap: the two COVID-era origins, the pair
whose support is anomalous for the same reason the eval mask exists. 2022 passes at
476,025, within 5% of the cap, so the margin elsewhere is thin rather than comfortable.

**Nothing has been changed in response yet.** The fix is a grid decision and it touches
standing convention 1: the grid ranges are already on the "still exposed" list as shaped
on the backbone, and re-cutting the `tau_k` ceiling because of a compute cap is a
declared, non-oracle reason — but it is still a choice, it must be made the same way at
every origin, and it must be recorded here before it is run rather than discovered in a
merged table afterwards.

### 4. Their side: what `backbone_compare` actually needs, measured rather than assumed

The directive is to recompute **the evals**, which is a two-sided table, so their columns
have to stand on the same sequence set. Read off the code rather than off the earlier
entry's one-line claim that "a manifest is not a tree":

| their metric | needs the tree? | why |
| --- | --- | --- |
| `distance_from_consensus` | no | `consensus_distance(p.codes, task.sup, w)`, sequence only — and it is the current top row |
| `ep_x` | no | `epitope_cross_immunity` over panel codes |
| `lbi` | **yes** | `local_branching_index` walks nodes and branch lengths |
| `ne_star` | **yes** | `mutational_load` walks a tip up to its ancestor in the prior season |

So two of four, and the pair that needs a tree is exactly their designated best
sequence-only model, `ne_star-lbi`. A tree-free run would produce a table missing their
headline arm, which is not a comparison worth reporting.

Three things found while sizing the per-origin tree build, all of which change the
estimate in §8 and in the previous entry:

1. **The tooling is here after all.** The earlier entry said `augur`/IQ-TREE are not
   installed and stopped there. `~/miniconda/envs/usher/bin` holds `usher`,
   `usher-sampled`, `matUtils`, `matOptimize`, `faToVcf` and `mafft`; `~/miniconda/envs/bte`
   holds `treetime`; there is a `chronumental` env. §8's exact recipe (`augur align` +
   IQ-TREE + `augur refine`) is not available, but a parsimony-placement analogue with
   TreeTime or chronumental dating is. **Any tree built this way is parsimony, not ML,
   and that deviation belongs in the entry that reports the first number off it.**
2. **`branch_length(n)` is `date(n) - date(parent)`**, not divergence. LBI therefore needs
   node *dates* and nothing else from the branch lengths, which means the dating step is
   the load-bearing one and the substitution model behind the topology matters less than
   it looks.
3. **A tree holding only `S(T)` is not sufficient**, and this is the finding that matters.
   `rolling` builds **one** `BackbonePanel` and reads `bp.panel` for every season in the
   roll — the support at each origin, the training transitions behind it, and `task.tgt_w`,
   **the target season, which is after the horizon by construction**. A per-origin tree
   containing only the date-prefix has no target in it to score against. So §8's "one tree
   per origin" is underspecified: either the tree carries `S(T)` plus the target season
   as tips whose metrics are never read, or `BackbonePanel` is split so that the panel
   comes from the manifest and only `lbi`/`ne_star` come from the per-origin tree.
   **The second is the right shape** — it is the same separation the manifest made on our
   side — and it is a real change to `backbone_compare`, not a flag.

### Status at the time of writing

- `gate.slurm` job `3105705`: **11 of 13 shards written** — 2013, 2014, 2015, 2016, 2017,
  2018, 2019, 2022, 2025 at the time of writing, with 2023 and 2024 still running and no
  reason to expect them to fail. **Tasks 8 and 9 (origins 2020, 2021) are dead on the
  candidate cap and the table cannot be merged without them**: `merge_gate.py` refuses a
  ragged table, and rightly — a gate table missing two transitions would hand
  `causal_operating_point` a hole at exactly the origins the eval mask is about.
- `score.slurm` is **not submitted** and cannot be until the gate merges.
- `backbone_compare` is unchanged. Nothing has been written toward the per-origin trees.
- No scored number exists on the additive panel. The 2026-10-01 headline table still
  stands on `tree_2026-09-24.json` and is untouched.

### What the next session should do first

1. **Settle the `tau_k` grid** against the cap, at every origin identically, and record
   the reason before re-running. Re-submit the gate.
2. **Split `BackbonePanel`** so the panel is the manifest and the tree supplies only
   `lbi` and `ne_star`. Until that exists there is no per-origin tree worth building,
   because there is nothing that could read one.

---

## 2026-10-02 (third) — the cap is made relative and the gate lands; the K = 90 score array and the per-origin trees are both running

### The ask

> Read NOTES_2 to get oriented. We are in the process of implementing and running
> PROPOSAL_additive_panel.md. Please continue with the task, and as you do, record to
> NOTES_2 what progress you make

Picking up the previous entry's two-item list: settle the `tau_k` grid against the
candidate cap and re-submit the gate, then split `BackbonePanel` so the panel comes from
the manifest and only `lbi`/`ne_star` come from a tree. Both are done, and the two arrays
that follow from them are queued.

### 1. The candidate cap was the wrong *kind* of number, and that is the fix

The previous entry framed this as a grid decision. It is not: **the cap was an absolute
row count sized against a panel that no longer exists**, and re-cutting the grid to fit it
would have confounded the one thing this round is for.

`MAX_CANDIDATES = 500_000` was documented as firing "only ever on an untruncated PLM
arm, where the block would be 2.3M rows". It fired on `kcmom` and `kcons`, both
`residues="observed"`, at origins 2020 and 2021, whose dense blocks are **530,528**
(944 parents x 562 columns) and **520,654**. An `observed` arm at `m = 0` is asking for
exactly its own one-step neighbourhood over residues the support already carries — that
is not a kernel nobody asked for, it is the kernel, and it was refused for being 6% over
a constant that had nothing to do with it.

So the guard is now **relative to the leak-free dense block**: `|sup|` times the number of
leak-free `(site, residue)` columns, floored at `MIN_CANDIDATE_CAP = 500_000` so nothing
smaller than the backbone changes verdict. An `observed` arm can never exceed its own
block and is never refused at any panel size; a `residues="all"` arm still is, by ~13x at
origin 2020 (7.1M against 530k), which is the case the guard was written for. It never
needs re-cutting when the corpus grows.

Two measurements were taken before choosing, and both are reasons the alternative was
rejected:

- **Cutting the `tau_k` ceiling changes nothing anyway.** Restricting the backbone gate
  table to `tau_k <= 0.1` and re-resolving `causal_operating_point` gives the identical
  `(tau_e, tau_k, m)` at **all 8 backbone origins for both arms** — the selected `tau_k`
  is 0.001–0.01 everywhere and the top of the range is uniformly dominated.
- **The cap was never protecting a resource.** Origin 2020 at the densest configuration
  (`tau_k = 1, m = 0`, 530,528 candidates) runs in **17 s at 900 MB peak RSS**.

Even so the grid was left alone, because this round exists to change the panel and only
the panel, and `gate.slurm` is otherwise field for field `causalgate_t0924/gate.slurm`.
Pinned by three tests: a leak-free arm is never refused its own dense block, an
all-residue arm at `m = 0` still trips, and the floor reproduces the old absolute
behaviour below the backbone's size.

### 2. The gate is merged, over all thirteen transitions

All 13 shards of job **`3105784`** (snapshot `wt-1c179d0cca7a`) wrote. The whole array was
re-run rather than only the two that died: the old shards are numerically unaffected by
the cap change — it raises or does not raise and computes nothing — but a merged table
whose shards came from two snapshots is not worth the saving. They are in
`ARCHIVE_gate_3105705_capfail/`.

```
gate_table.json  2,808 rows, 216 configurations x 13 transitions (2013->2014 .. 2025->2026)
                 panel d4881a783e29, K=30, cells=month, level 1, corpus 2026-04-08
```

`merge_gate.py` passed its raggedness check and its one new assertion, that every shard
carries the same `panel.manifest_sha256` and level. The operating point each scored origin
resolves to, printed before the scoring run rather than discovered in its logs:

| origin | kcmom | kcons |
| --- | --- | --- |
| 2018 | `0.15 / 0.001 / m 0` −0.5006 | `0.15 / 0.001 / m 0` −0.4492 |
| 2020 | `0.5 / 0.001 / m 0` −0.4384 | `0.15 / 0.001 / m 0` −0.4381 |
| 2021 | `0.5 / 0.001 / m 0` −0.4508 | `2.0 / 0.001 / m 10` −0.4465 |
| 2022 | `0.5 / 0.001 / m 0` −0.4648 | `0.15 / 0.001 / m 0` −0.4547 |
| 2023 | `0.5 / 0.001 / m 0` −0.4856 | `0.15 / 0.001 / m 0` −0.4832 |
| 2024 | `0.5 / 0.001 / m 0` −0.5090 | `0.15 / 0.001 / m 0` −0.5074 |
| 2025 | `0.5 / 0.001 / m 0` −0.5218 | `0.15 / 0.001 / m 0` −0.5195 |

**Every origin selects the smallest `tau_k` in the grid**, and `dW/dmu` is monotone in
`tau_k` across the whole table (−0.4989 at 0.001 rising to −0.1247 at 1). The argmin is on
the grid's lower boundary at every origin, which is a *different* exposure from the one
section 1 was about and is the same one the backbone had — "the grid ranges, shaped on the
backbone over dozens of rounds" is already on the still-exposed list. It is now visible
rather than inferred: an interior optimum would have been evidence the range was wide
enough, and this is not that.

### 3. The scored array is running

**`score.slurm` is job `3106005`, 7 tasks, K = 90**, origins 2018, 2020, 2021, 2022, 2023,
2024, 2025. All seven preflights passed — snapshot stamp, manifest hash, and the gate
table's `causal_table` + `K=30` + salt check — and every task printed its causally
resolved operating point before starting. This is the first scored run of
`PROPOSAL_additive_panel.md`; §11's "third full re-run of both sides" has started.

Sized from a single-combination probe at origin 2022, K = 90: **901 s wall, 8.2 GB peak
RSS** — on the login node at load 21 with 4 cores, so an unknown multiple of what a
dedicated node costs. 64G is comfortable. The 4-day limit is the thing to watch and is
the reason it is stated here: if the biggest origins hit it, the grid is the lever, not
the panel.

### 4. `BackbonePanel` is split, and the split is byte-identical on the old path

`backbone_compare` fused two things that only looked like one, because both came off the
same `tree.json`: the panel every model is scored on, and the topology two of their four
metrics need. The panel is now an argument and the tree is a `TreeSource`.

- `BackbonePanel.from_tree(...)` is the pre-2026-10-02 path, panel and tree both off one
  Nextstrain build. `OneTree` serves it at every origin.
- `BackbonePanel(panel, trees, maps, ref)` takes a `TipPanel` from anywhere — a manifest,
  say — and `PerOriginTrees` reads `tree_<season>.json` from a directory, refusing an
  origin it has no build for rather than falling back to another origin's.
- **The join from a tree tip to a panel genotype is now by sequence**, not by row order.
  The old code paired `tip_nodes[i]` with `panel.tip_gid[i]`, sound only because both
  lists were the same tips sorted the same way; once the two are separate objects that
  coincidence is gone. Tips the panel cannot name are counted in `join_misses` and written
  into the output, not dropped quietly.

**The finding that makes this the right shape, confirmed:** a per-origin tree needs to
hold only `S(T)`. The previous entry worried that `rolling` reads `task.tgt_w`, the target
season, which is after the horizon — but the target is read off the *panel*, and the tree
is asked only for `lbi` and `ne_star` over the support. No target tips. That is what makes
§8's rebuilds tractable.

Regression, on `tree_2026-09-24.json` with the same flags, old snapshot against new:

```
--no-select arm:  per_transition identical     (8 transitions, 6 models + 2 point masses)
--select arm:     per_transition identical
tree_join_misses: 0 at every origin
```

Also recorded in the output now: `panel` (the provenance block with the manifest hash),
`trees` (which source wrote `lbi` and `ne_star`), and `tree_join_misses`.

### 5. `antiGen/panel_tree.py` — the per-origin builds, and §8's recipe is available

The previous entry said `augur`/IQ-TREE are not installed and the entry after it found
usher and treetime. Both were incomplete: **`~/miniconda/envs/tpcmp/bin` holds IQ-TREE
3.1.3 and TreeTime 0.12.1**, so §8's exact recipe is available in substance —
`augur refine` and `augur ancestral` are wrappers on TreeTime, and the alignment step is
free because the panel is already a 566-residue nextclade translation.

Per origin: IQ-TREE `--fast` for the topology, then one TreeTime run with `--aa` for the
dates and the ancestral sequences, then a conversion to the subset of the auspice schema
`load_tree` reads.

**The deviation to declare, and it belongs in the entry that reports the first number off
one of these trees: this is an amino-acid tree.** Nextstrain builds H3N2 phylogenies from
nucleotides, 1,737 sites; we hold only the translation, 566 residues. Fewer informative
characters means a less resolved topology. It is a smaller deviation than the one it
removes — an amino-acid tree estimated at origin `T` is causal, a nucleotide tree
estimated in 2026 is not.

**Two bugs, both silent, both caught by `panel_tree verify` rather than by reading code.**
The first tree built lost **all 91 of its tips** to the sequence join:

1. **TreeTime's own `branch_attrs.mutations` is a strict subset of the branch's
   substitutions.** It listed five on a tip that differed from its parent at eighteen
   positions. `load_tree` rebuilds a node's sequence by applying the list it is handed, so
   an incomplete list produces a tip sequence that is not the tip's. Branch mutations are
   now diffed from the reconstructed sequences, which makes the round trip exact by
   construction.
2. **The inferred root is not the reference.** `load_tree` starts its reconstruction from
   the SigPep+HA1+HA2 concatenation, so a tree rooted anywhere else returns tip sequences
   that are wrong everywhere the two differ. The reference-to-root difference now sits on
   the root's own branch, which costs `ne_star` nothing: `mutational_load` walks
   `while n.parent is not None`, so a root's mutation list is never counted.

Tips take their **observed** sequence rather than TreeTime's reconstruction of it (they
agreed at every tip of the first build, and the count is recorded per tree). After both
fixes:

```
origin 2013: 91 tips, 0 not in the panel, 0 after the horizon -- OK
```

`verify` is a preflight in both `trees.slurm` and `theirs.slurm`, and exits non-zero.

### 6. Both remaining arrays

**`trees.slurm` is job `3106092`, 13 tasks, K = 90**, `%7` so it does not crowd out the
scoring array. Thirteen and not seven because `rolling` refits their coefficients over a
six-transition training window and a training row at origin `j` reads origin `j`'s
metrics. **Origin `j`'s metrics always read tree `j`**, including when `j` is a training
row for a later `i`: that is Huddleston's own convention — their per-timepoint
tip-attribute tables are computed at their own timepoint — and it is causal, since tree
`j` is estimated from `S(T_j)` and nothing after. The alternative, re-reading every
training row off tree `i`, is also defensible and is not what their pipeline does.

`|S(T)|` runs 272 (2013) to 12,476 (2025). Origin 2013 landed in about a minute including
`verify`.

**Job `3106092` was cancelled, because `verify` failed two of the first three trees — and
the failure was silent in every other respect.** This is the third thing the preflight has
caught, and the argument for having written it:

```
origin 2013:  272 tips, 0 not in the panel,  0 after the horizon -- OK
origin 2014: 1287 tips, 0 not in the panel,  1 after the horizon -- FAILED
origin 2015: 2373 tips, 0 not in the panel, 12 after the horizon -- FAILED
```

**The real failure was the dates, and only the dates.** TreeTime re-estimates a tip date
that does not fit its clock, and 13 tips across two origins came back dated *after their
own origin's horizon*. Those fall outside `local_branching_index`'s censoring and outside
`_live`'s window — a tip the forecaster did hold, removed from their metrics by a dating
run. **Tip dates are now the manifest's pins**, which is what §3's date pin exists for: a
sequence's position in time is not revisable, and a dating run is not an exception to
that. Internal node dates stay TreeTime's, which is what the dating run is for.

**The 2,373-against-2,374 was me misreading my own panel, and the correction is worth
more than the original claim.** It was first written up here as TreeTime pruning a clock
outlier at its default 4 IQRs. It is not: 2,374 is the *manifest* count at origin 2015 and
2,373 is the *panel* count, because `load_alignment_dated` drops 21 of the 13,277 admitted
accessions at its 2% ambiguity floor and one of them falls before 2015.25. Measured:
manifest minus panel is 0 at origins 2013 and 2014, 1 at 2015, 2 at 2016. Nothing was
pruned. Two things follow:

- `--clock-filter 0` stays, but it is a **precaution and not a fix for anything observed**.
  An H3N2 panel subsampled 90 per month will hold clock outliers and they are part of the
  panel both sides are scored on, so TreeTime must not be allowed to decide otherwise.
- The guard that would actually catch a prune is the new one in `build_origin`: the tree's
  tip set must equal `S(T)` *as the panel holds it*, exactly, or it is not written. That
  is the assertion the misreading should have been, and it is now there.

Re-submitted as **job `3106127`** under snapshot `wt-6c411403256d`, with
`~/h3n2/trees_K90_v1/` cleared first so no tree from the cancelled array can survive into
the set `theirs.slurm` reads.

**`theirs.slurm` is written and NOT submitted**, because it cannot run until all thirteen
trees exist. Its preflight checks, for every origin, that the tree exists, is stamped for
*that* origin, was built on *this* manifest, and joins to the panel with zero misses.
`--select` stays on, which is standing convention 2.

### What is not done

- **No number on the additive panel is reportable yet.** The scored array is running; the
  2026-10-01 headline table still stands on `tree_2026-09-24.json` and is untouched. The
  smoke figures quoted in this and the two previous entries are not comparable to anything
  in this document — different panel, different floor.
- **Their side has not run at all** on the additive panel. It waits on `trees.slurm`.
- **`eval_figures` has no `additive` entry** in `BACKBONE_PANELS`, and adding one is not
  purely mechanical: on the backbone, `ours` and `kernel` came from two separate runs
  (`competitors_t0924` and `causalgate_t0924`) and the loader cross-checks four sources
  against each other on the persistence baseline and the floor. On panel_v1 a single run
  writes both families, so there are three sources rather than four and one of the
  identities the loader asserts becomes trivially true. That is a weaker check and it
  should be said in the entry that first draws the panel, not discovered later.
- The corpus still ends 2025-12-31, so task 7 of the scored array (2025->2026) is the
  half-season §7.2 describes and is run to measure the shortfall, not to be used.
- Region stratification, passage exclusion and level 2 still wait on a GISAID re-pull.
- No `outliers.txt` and no force-include list are pinned; both header fields are `.`.

### Still exposed

Unchanged, plus one made visible: the `--mus` ceiling; `top_k`, `val_folds` and the grid
ranges — and **the `tau_k` grid's lower boundary is now binding at every origin on this
panel**, which is the same exposure stated sharply; the eval mask; the H2020 kernel arm's
transferred operating point; the `K = 30` gate / `K = 90` score carry. Removed once
`theirs.slurm` lands: per-horizon tree rebuilds for their metrics, the oldest item on the
list. A new one takes its place and is smaller: **their trees are amino-acid, parsimony-
seeded ML builds rather than the nucleotide builds Nextstrain makes.**

### Artifacts

- `antiGen/panel_tree.py`, `antiGen/tests/test_panel_tree.py` (8 tests).
- `antiGen/mutation_kernel.py`: `MIN_CANDIDATE_CAP` + `candidate_cap()` replacing
  `MAX_CANDIDATES`; 3 tests in `test_mutation_kernel.py`.
- `antiGen/backbone_compare.py`: `TreeSource` / `OneTree` / `PerOriginTrees`, the
  `BackbonePanel` split, the sequence join, `--trees-dir` and the panel-source flags;
  3 tests in `test_backbone_compare.py`.
- `antiGen/season_momentum.py`: `load_alignment_dated(..., with_names=True)`.
- `~/antiGen-count-loss-runs/panel_v1/`: `gate_table.json`, `trees.slurm`, `theirs.slurm`,
  `ARCHIVE_gate_3105705_capfail/`.
- `~/h3n2/trees_K90_v1/` — per-origin trees, being written.
- Snapshots `wt-1c179d0cca7a` (gate + score) and `wt-6c411403256d` (trees + theirs).
- Jobs: gate `3105784` done, score `3106005` running, trees `3106127` running
  (`3106092` cancelled, see above).
- 449 tests pass, up from 435: +3 on the candidate cap, +3 on the `BackbonePanel` split,
  +8 on `panel_tree`.

### What the next session should do first

1. **Check `3106005` and `3106127`.** If a score task hit the 4-day wall, cut the grid,
   not the panel, and say so here first.
2. **Submit `theirs.slurm`** once all thirteen trees pass `verify`.
3. Only then is there a two-sided table to compare, and §12's "every number moves again"
   can be measured rather than predicted.

---

## 2026-10-04 — the K = 90 score array is 84% oracle diagnostic; `score.slurm` is split on `--families` and the split is proved inert

### The ask

> We are re-running a sweep on the re-generated, additive dataset. However, jobs have been
> running over 2 days. Can you check that these jobs are progressing appropriately? Is
> there a way to make them faster? Should we try to extend the time caps on the jobs?

Progressing, yes. Going to finish, no — and the reason is not the panel, it is one family.

### 1. The array is healthy and still will not land, and nothing in its logs said so

Job **`3106005`**, seven tasks, submitted 2026-10-02 15:21, `--time=4-00:00:00`, so
`EndTime` is 2026-10-06 15:21. At 2026-10-04 16:40 all seven were `RUNNING` at
**2-01:19** with **not one finished, origin 2018 included, which is the cheapest**.

They are not stuck. `ssh` to the nodes is key-denied and `sstat` returns an empty table
for these jobs, so the processes were read directly with
`srun --jobid=3106005 --overlap --cpu-bind=none`:

```
origin 2025   elapsed 177641s   cpu 177637s   99.9%   RSS 7.3 GB
origin 2024   elapsed 177641s   cpu 177630s   99.9%   RSS 6.1 GB
```

CPU time equal to wall time to within four seconds over two days: no swapping, no I/O
wait, no thrash. 64G was the right call and is not close to binding.

**The logs are empty by construction and that is the finding worth keeping.**
`season_momentum` prints one line per transition and each task runs one, so a task that
has not finished has printed nothing since its preflight. Forty-nine hours of silence is
indistinguishable from forty-nine hours of deadlock, and the only reason it was not read
as one is that this session went to the nodes. The previous session had already written
`bench.slurm` (**`3122475`**) for exactly this gap, which was the right instinct.

There is also no checkpointing: `main` writes its JSON after `rolling` returns, so a task
that hits the wall loses all of it. And the seven are holding slots while 102 `og_jun`
tasks sit behind `QOSMaxJobsPerUserLimit`.

### 2. Where the time goes: `mlr+mom+K@mu*` is 84% of it

Counted with `collapse()` on `score.slurm`'s own grid rather than estimated:

| family | collapsed combos | extended transport solves, x3 folds |
| --- | ---: | ---: |
| `season` | — | fixed baseline, no search |
| `ewma` | 8 | 0 |
| `mlr` | 120 | 0 |
| `mlr+mom` | 1,080 | 0 |
| `season+K` | 1 | 3 |
| `mlr+mom+K` | 5,400 | 12,960 |
| **`mlr+mom+K@mu*`** | **1,080** | **68,040** |

81,003 extended-support solves per task, of which the oracle is **84.0%**, plus 6,864
unextended ones. `golden_mu` has `tol = 1e-3` on `[0, 1]`, so about fifteen bisections
plus the bracket plus the three endpoints — **~21 solves per collapsed combination**
against `mlr+mom+K`'s four, because `mu = 0` returns a `Forecast` with no block and
scores unextended.

**The fits were never the problem.** `_momentum_prediction` memoises the parent on
`RELEVANT["mlr+mom"]` plus the training seasons, and `mu` is the fastest-varying axis in
`make_grid`, so the five `mu`s of one combination share one fit and `mlr+mom+K`'s 5,400
cost 1,080 fits. What does not memoise is `task.score`, which is an exact network-simplex
solve per call.

**And it is the one family whose number can never be reported.** `mlr+mom+K@mu*` is
`ORACLE_FAMILIES` (`season_momentum.py:1617`); `report` prints that it "chooses mu on the
transition being predicted: an oracle diagnostic, not a forecast". Five sixths of a
four-day wall was going to the one row that cannot go in a table.

### 3. The per-solve cost, two independent measurements, and one number I could not reconcile

- **`bench.slurm` task 1** — 5-combination grid at origin 2018, so ~78 extended solves —
  had run **1h23m without finishing**: **≳63 s/solve**.
- **`verify_family_split.py`** (section 4) at origin 2016 on a 2-combination grid: the
  oracle leg took **904.6 s** for ~42 extended solves, **~21 s/solve**, on the login node
  at a support of 3,444 tips against origin 2018's 5,605.

Either way 81,003 solves is of order **20 to 60 days per task**, and 49 h of wall had
bought something between **3% and 10%** of one. The backbone comparison in `score.slurm`'s
header — "~4.4e5 cells, i.e. ~14x" — is the wrong model: `ot.emd` is a network simplex and
its cost is superlinear in the problem, so 14x the cells is very much more than 14x the
time. The backbone round's own 42m–11h49m spread over nine origins says the same thing in
the other direction.

**The 901 s single-combination probe quoted in the 2026-10-02 entry §3 does not reconcile
with either measurement** and I could not make it. It is also the number the 4-day cap was
sized from. Until someone reproduces it, treat the two measurements above as the live ones
and that one as unexplained.

### 4. `score.slurm` is split on `--families`, and the split is *measured* inert

```
score_headline.slurm   season,ewma,mlr,mlr+mom,season+K,mlr+mom+K   -> shard_headline/
score_oracle.slurm     mlr+mom+K@mu*                                -> shard_oracle/
merge_families.py      re-joins them into out/pv1_kcmom_<S>.json
verify_family_split.py one origin, run whole and run sharded, diffed
```

Both are `score.slurm` character for character except `--families`, the output path and
`--time=14-00:00:00`, the partition maximum on `cpu_batch` and `cpu_high_mem`.

**The grid stays 5,400 with all five `--mus` in both shards.** `floor_ext`,
`kernel_point` and every `n_grid_` are computed from the grid and not from the family
list, so shortening `--mus` would be a different run while dropping a family is not. Both
scripts assert `len(grid) == 5400` as a preflight, alongside the collapsed sizes of the
two kernel families.

**Why it is inert.** `select()` reads `tasks`, `order`, `i`, `grid`, `val_folds`,
`maxiter`, `gamma_pen` and `top_k` and nothing else, so no family's chosen combination or
score depends on which other families were asked for. The only shared mutable state is
`task._cache`, a memoisation dict, so the split recomputes `mlr+mom`'s 1,080 fits a second
time and changes no number. There is no RNG anywhere in the fit or score path.

**The one place it could have gone wrong.** `report` draws a single `default_rng(0)` and
consumes it **once per family in list order** (`:2327`), and picks its reference column as
`season` when present and `families[0]` otherwise. A shard's own `summary` is therefore not
comparable with a full run's. `merge_families.py` discards them and re-runs `report` over
the merged row with the families in `score.slurm`'s order.

**It is checked, not asserted.** The merge refuses the pair unless every field `rolling`
computes *before* the family loop — `floor`, `floor_alt`, `floor_ext`, `kernel_point`,
`n_support`, `n_target`, `support_dropped`, `n_next_seqs`, `lookback_shortfall`, `metric`
— agrees through `repr`, with no tolerance; agreement to 1e-12 would be a finding, not a
pass. It also refuses overlapping family lists, differing panel/grid/gate-table
provenance, and any row field it does not know how to place.

**Measured, origin 2016 -> 2017, 2-combination grid, same panel and gate table and snapshot:**

```
mono      season,ewma,mlr,mlr+mom,season+K,mlr+mom+K,mlr+mom+K@mu*    959.5s
headline  season,ewma,mlr,mlr+mom,season+K,mlr+mom+K                   94.9s
oracle    mlr+mom+K@mu*                                               904.6s

all 44 row fields identical    summary identical over 9 entries
```

Every `emd_`, every `combo_`, and the pre-family-loop fields. The 94.9 against 904.6 is
also the section 2 claim end to end, although at a 2-combination grid the oracle's share
is structurally higher than the 84% the real grid predicts: `collapse` gives it one
combination against `mlr+mom+K`'s two, where at 5,400 it gets 1,080 against 5,400.

### What this does not fix

- **The headline shard is still slow.** 12,963 extended solves is of order a week at
  origin 2018 on the section 3 numbers and worse at 2025. 14 days will probably carry the
  early origins and probably not 2024 and 2025.
- **The oracle array will very likely not finish at the late origins**, which is said in
  its header rather than left to be discovered. `merge_families.py --partial` writes a
  table from the origins whose oracle shard is missing and names them in
  `families_missing`, so a reader that indexes `emd_mlr+mom+K@mu*` raises rather than
  finding a zero.
- **The real fix was not attempted.** `select`'s combination loop is embarrassingly
  parallel and deterministic, so it can be made bit-identical, and Slurm is already
  handing these tasks `cpu=8` while `OMP_NUM_THREADS=1` and the code is single-threaded.
  One caveat for whoever does it: `mu` is the fastest-varying axis in `make_grid` and
  `_momentum_prediction`'s cache is per-task, so a worker split must keep combinations
  sharing an `mlr+mom` key in one worker — contiguous chunks of five — or it will do five
  times the fits it needs.
- Two micro-costs, both behaviour-preserving, neither taken: `Forecast.stacked` re-`vstack`s
  the whole cost matrix on every score call although the candidate block is fixed by the
  gate's pinned `(tau_k, m)`; and `emd()` recomputes `W = sum(G * M)` over the dense plan
  although POT returns the cost in its log.
- **Nothing was submitted and no time limit was changed.** `sbatch` and `scontrol update`
  were both refused to this session by its own permission rule. Raising `TimeLimit` on a
  running job is operator-only in stock Slurm, so `3106005` itself may need the admin
  rather than the user; the partitions do allow 14 days.
- Still **no reportable number on the additive panel**, which is where the 2026-10-02
  entry left it.

### Still exposed

Unchanged. Nothing here touched the grid, the gate, the panel or the protocol — the whole
point of the round was to change which process computes which family and nothing else.
One item is retired in substance: the 2026-10-02 entry's "if a score task hit the 4-day
wall, cut the grid, not the panel" is now answered — **neither**, the wall was being spent
on a diagnostic.

### Artifacts

- `~/antiGen-count-loss-runs/panel_v1/`: `score_headline.slurm`, `score_oracle.slurm`,
  `merge_families.py`, `verify_family_split.py`, and a new README section carrying the
  section 2 cost table.
- Code snapshot unchanged: `wt-1c179d0cca7a`. **No file in `antiGen/` was modified**,
  which is what makes the comparison in section 4 a comparison.
- Jobs found running and left alone: score `3106005` (7 tasks, wall 2026-10-06 15:21),
  bench `3122475` (2 tasks), theirs `3122479` — whose log had passed origin 2019 with all
  thirteen trees stamped and joining at zero misses, so `theirs.slurm` did land.

### What the next session should do first

1. **Read `bench.slurm`'s `BENCH` line** if `3122475` landed; it is the only clean
   measurement of the marginal cost of a combination and section 3 is two bounds without it.
2. **Submit `score_headline.slurm`.** It is the table. Then `score_oracle.slurm` behind it.
3. **Decide about `3106005` deliberately.** It cannot finish, it blocks 102 queued tasks,
   and its three cheapest origins have about 46 h left in which they have a small chance.
   Extending it needs the admin; cancelling it costs three low-probability rows.
4. Only then `merge_families.py`, and `eval_figures`' missing `additive` entry — which the
   2026-10-02 entry already warned is not mechanical, because a single run writes both
   families and one of the loader's four cross-checks becomes trivially true.

---

## 2026-10-04, later — the selector runs in a pool, the split is bit-identical, and both arrays are launched

### The ask

> Continue with the task of getting the array run launched on the new (additive) dataset.

Launched: headline `3124528`, oracle `3124535`, both on snapshot `wt-9bc6726d3598`, all
seven headline tasks running. But the previous entry's plan — "submit
`score_headline.slurm`" — would have returned one or two rows of seven, and the number
that says so arrived between the two entries.

### 1. `bench.slurm` landed and it resolves the two bounds to the pessimistic one

```
BENCH combos=1 wall=5155.98 s maxrss=6991576 KB
```

One collapsed combination at origin 2018, all seven families, ~78 extended transport
solves. **A solve costs of order 60 s**, which is the `≳63 s` end of the previous entry's
spread and not the `~21 s` end — and the 901 s single-combination probe of 2026-10-02 §3
stays unreconciled and unused. Three consequences, and only the third was a surprise:

| | extended solves | at ~60 s/solve |
| --- | ---: | ---: |
| `score.slurm` (`3106005`, all seven families) | 81,003 | **~58 days** against a 4-day wall |
| `score_headline` at origin 2018, the cheapest | 12,963 | **9–12 days** against 14 |
| `score_headline` at 2024–2025, ~2x the support, superlinear simplex | 12,963 | **past 14 days** |

So `3106005` had **no** chance on any task rather than a small one on its three cheapest,
which is what the previous entry left open; it was cancelled, and with it `3122475_2`,
the 12-combination bench leg, which needed ~16 h of an 8 h wall and would have printed
nothing. And `score_headline.slurm` as it stood was not a table. **The previous entry's
"the real fix was not attempted" is the thing this entry does.**

### 2. `--select-workers`, and why it cannot move a number

`select`'s combination loop now runs in a pool of forked processes.

- It reads `tasks`, `order`, `i`, `grid`, `val_folds`, `maxiter`, `gamma_pen` and
  `top_k`; no iteration writes anything another iteration reads; there is no RNG under
  `run_family` or `score`. **This is the same independence the family split rests on**,
  used a second time.
- `combo_error` is one iteration's body, lifted out so the serial path and a worker run
  one piece of code rather than two copies of it.
- The one piece of shared mutable state is `task._cache`, a memoisation of pure
  functions: a cold cache costs time and returns the same arrays.
- **`select_chunks` cuts the grid only at cache-group boundaries**, so a run of `mu`s
  sharing one `mlr+mom` fit stays in one worker and the pool does 1,080 fits, not 5,400.
  `mu` is the fastest-varying axis in `make_grid`, which is what makes those runs
  contiguous; `cache_groups` derives the boundaries from `RELEVANT["mlr+mom"]` rather
  than hardcoding the five, so a grid with a different `--mus` still chunks correctly.
- **The argmin and the `top_k` list are taken by the same ascending scan over `errs` the
  serial path uses**, lifted out of the loop so both paths run it. `e < best_err` means
  the *first* strict minimum in grid order wins and means a NaN never wins at all; the
  pool returns its pairs out of order and has to re-impose both. `min` over the items
  would have differed on each.
- `fork`, not `spawn`: the children inherit the panel, the cost matrices and the
  candidate blocks as copy-on-write pages. Measured on a running task — 33 processes,
  largest single RSS 6.7 GB, the same as the serial bench, and 188 GB summed across them,
  which double-counts the shared pages. `OMP_NUM_THREADS=1` stops being a nicety and
  becomes a requirement.

### 3. It is measured, not argued

`verify_workers.slurm` / `verify_select_workers.py`, the same shape of proof
`verify_family_split.py` makes for the family split: one origin, run serial and pooled,
differing in `--select-workers` and nothing else, every field through `repr` with no
tolerance. **All seven families, including `mlr+mom+K@mu*`** — `score_oracle.slurm` uses
the pool too, and `golden_mu` is the only caller that scores *inside* `run_family` rather
than after it, so it is the only one whose solves happen on a worker's own stack.

```
origin 2015->2016, 7 families, grid 20
  select-workers   0     1106.6s
  select-workers  32      331.3s
  all 44 fields identical    summary identical over 9 entries
```

**The 3.3x is not the speedup and saying so is the point.** This grid gives `mlr+mom+K`
20 combinations in **4 chunks**, so at most four workers were ever busy; the real grid
gives it 5,400 in 120 chunks and the oracle 1,080 in 135. Solving `S + L = 1106.6` and
`S + L/4 = 331.3` puts the fixed setup at ~73 s and the loop at ~1,034 s serial against
~258 s on four workers — **linear**, which is the claim that matters. The verify script
now prints how many workers its own grid could have used, because the end-to-end ratio is
otherwise read as a ceiling.

**A smaller finding, caught by the first submission dying:** `verify_family_split.py`'s
usage line suggested `--origin 2014` as the cheap case. `rolling` starts its loop at
`min_train = 2`, so 2015 is the first origin it scores and a 2014 run writes an empty
`per_transition`. Both scripts say so now.

`select_workers` is recorded in the output JSON and is deliberately **not** in
`merge_families.py`'s `SHARED_META`: a merge that refused two shards for differing in it
would assert the opposite of what this verifies.

### 4. What is launched, and the prediction to check it against

- **`3124528`, headline, 7 tasks, 14 d, 34 cpus / 400G, 32 workers.** All seven RUNNING,
  grid asserted at 5,400, gate table asserted causal and K=30.
- **`3124535`, oracle, 7 tasks, `%3`.** Two running, five held on
  `QOSMaxCpuPerUserLimit`; they roll in as headline tasks land. The `nogpu` QoS allows
  513 CPUs per user and each task is billed 50.
- `3122479`, `theirs`, still running, past origin 2023 of thirteen.

**The prediction, stated here so the next session measures it rather than re-derives it.**
12,963 solves at 60 s over 32 workers is ~6.8 h of loop at origin 2018, plus setup; the
late origins carry roughly twice the support against a superlinear simplex, so of order a
day. **All seven headline tasks should land inside two days**, against the 14-day wall.
The oracle's 68,040 solves are ~1.5 days at 2018 and perhaps 4–5 at 2025 — so the
previous entry's "the oracle array will very likely not finish at the late origins" is
now a genuinely open question rather than a safe bet, and `merge_families.py --partial`
is still the supported answer if it goes the other way.

### What is not done

- **Still no reportable number on the additive panel**, which is where the last three
  entries left it. Nothing in this round touched the grid, the gate, the panel or the
  protocol — only which process evaluates a combination.
- Their side has not been merged with ours; `theirs` has not finished.
- `eval_figures` still has no `additive` entry, and the 2026-10-02 warning stands: a
  single run writes both families on panel_v1, so one of the loader's four cross-checks
  becomes trivially true and that must be said in the entry that first draws the panel.
- The corpus still ends 2025-12-31, so headline task 7 is the half-season of §7.2.
- Region stratification, passage exclusion and level 2 still wait on a GISAID re-pull.
- The two micro-costs are still untaken: `Forecast.stacked` re-`vstack`s the whole cost
  matrix per score call although the candidate block is fixed by the gate's pinned
  `(tau_k, m)`, and `emd()` recomputes `W = sum(G * M)` over the dense plan although POT
  returns the cost in its log. Both are now worth less than they were, since the thing
  they would speed up is divided by 32.

### Still exposed

Unchanged — the `--mus` ceiling; `top_k`, `val_folds` and the grid ranges, with the
`tau_k` grid's lower boundary binding at every origin on this panel; the eval mask; the
H2020 kernel arm's transferred operating point; the `K = 30` gate / `K = 90` score carry;
their amino-acid trees. One item retires: **"what would make the oracle affordable … not
done in this round"** is done and verified.

### Artifacts

- `antiGen/season_momentum.py`: `combo_error`, `cache_groups`, `select_chunks`,
  `_select_parallel`, `_select_chunk`, `select(..., workers=)`,
  `rolling(..., select_workers=)`, `--select-workers`, and `select_workers` in the output
  provenance.
- `antiGen/tests/test_season_momentum.py`: 5 tests — serial/pool identity through `repr`
  on three families, the tie-break contract, and that chunks cover the grid once and
  never cut a cache group. **454 tests pass, up from 449.**
- `~/antiGen-count-loss-runs/panel_v1/`: `verify_select_workers.py`,
  `verify_workers.slurm`, `score_headline.slurm` and `score_oracle.slurm` updated,
  `*.slurm.bak` of the serial versions, a README section carrying §1's cost table.
- Snapshot `wt-9bc6726d3598`.
- Jobs: headline `3124528`, oracle `3124535`, theirs `3122479`; cancelled `3106005` and
  `3122475`; verification `3124381` (origin 2015, passed) and `3124373_2` (origin 2016).

### What the next session should do first

1. **Check `3124528` against §4's prediction.** If a task is far off it, the first thing
   to read is whether the pool is saturated — `select_chunks` on that family's collapsed
   grid — not the panel.
2. Read `3124373_2`, the origin-2016 verification, which is the stronger of the two.
3. `merge_families.py` once both shards have an origin in common, then the first
   two-sided table on the additive panel.
4. `eval_figures`' missing `additive` entry, with the cross-check caveat stated.

---

## 2026-10-12 — the first two-sided table on the additive panel

### The ask

> Check on the jobs. … Yes run the merge and the comparison to theirs.

### 1. The arrays landed, and §4 of the 2026-10-04 entry was half right

| job | outcome |
| --- | --- |
| `3124528` headline, 7 tasks | **all COMPLETED**, 17:50 to 1-03:45 each |
| `3124535` oracle, 7 tasks | **5 COMPLETED** (3-10:15 to 6-05:17); 2024 and 2025 still running, walls to Oct 25 |
| `3122479` theirs | COMPLETED 5:03:06 |
| `3124373_2` verify @ 2016 | passed — 44 fields identical, 4161.5 s serial against 1130.2 s pooled |

**The headline prediction was right and the oracle prediction was wrong by ~2.3x.** I
said all seven headline tasks inside two days: 17.8 h to 27.8 h. I said the oracle would
be ~1.5 d at 2018 and 4–5 d at 2025: origin 2018 took 3.4 d and origin 2020 took 6.2 d.
The pool is saturated there — 1,080 collapsed combinations in 135 chunks against 32
workers — so **the miss is in the per-solve cost and not in the parallelism**.
`golden_mu`'s interior `mu` values put mass across more of the extended support than the
grid's five do, and the simplex is dearer for it. The ~60 s/solve from `bench.slurm` is a
mixed-family average and should not be carried to the oracle.

Nothing hit a wall and nothing was requeued. `merge_families.py` passed its shared-field
check at all seven origins, so the family split was inert in the event and not only in
the argument.

### 2. `compare_theirs.py`, and what it checks before it compares

The two sides are two different programs reading one manifest. The table is only legible
if they are scoring the same problem, so that is asserted first, per origin:
`n_support`, `n_target` and `floor` **bit for bit**, and `emd_season` to a few ULP.

**The ULP tolerance on `emd_season` is not a weakening of `merge_families.py`'s rule and
the distinction is the point.** There, two shards of *one* program repeat an identical
call on identical inputs and anything but equality means something leaked across
`--families`. Here two *different* programs solve the same transport problem; `ot.emd` is
a network simplex and a degenerate optimum has several optimal vertices, so the two can
return different plans with the same objective and `sum(G * M)` then differs in the last
bits. Measured: identical at six of seven origins, **1 ULP at 2024**. Anything larger is
a different LP and the script refuses.

Their trees join the panel at **zero misses at all thirteen origins**.

### 3. The table — six transitions, one panel, one set of targets

`2025->2026` is dropped: the corpus ends 2025-12-31 and that target holds 155 sequences
against 252–295 for a full season (§7.2). `mlr+mom+K@mu*` is out of the table because
origin 2024's oracle shard is still running; the merged row names it in
`families_missing` and a reader that indexes it raises.

```
       family  mean EMD  vs season             95% CI   P<=0   wins
       season   10.0096     +0.00%  [  +0.00,  +0.00]  1.000   0/6
         ewma    9.6526     +3.96%  [  +0.87,  +7.85]  0.003   5/6
          mlr    9.3198     +7.51%  [  -0.95, +16.71]  0.041   4/6
      mlr+mom    9.3786     +6.84%  [  -2.53, +17.44]  0.085   3/6
     season+K    9.9601     +0.60%  [  +0.34,  +0.93]  0.000   6/6
    mlr+mom+K    9.2080     +8.99%  [  -0.90, +20.16]  0.039   3/6
          lbi   10.0815     -1.60%  [  -6.37,  +3.66]  0.727   2/6
      ne_star   11.5091    -11.76%  [ -34.90,  -0.04]  0.987   2/6
  ne_star-lbi   12.8109    -23.60%  [ -62.57,  +0.83]  0.967   1/6
         ep_x   10.0204     -0.09%  [  -0.33,  +0.06]  0.739   2/6
  dist_from_consensus 9.9585 +1.24% [  -3.79,  +6.10]  0.318   4/6
 ep_x-ne_star   10.2119     -1.30%  [  -3.80,  +0.01]  0.968   3/6
point_consensus 10.0752     -0.68%  [ -15.59, +13.89]  0.538   3/6
 point_toplbi   13.9894    -37.86%  [ -58.09, -19.61]  1.000   0/6
        floor    2.6730
      floor+K    1.8091
```

**What this says, and the four things it does not.**

- Our best fitted family is ahead of their best by about **7.7 points** of reduction
  against the same baseline: `mlr+mom+K` +8.99% against
  `distance_from_consensus` +1.24%. Every other Huddleston family is negative on the mean.
- **It is not separation.** `mlr+mom+K` wins 3 of 6 and its CI crosses zero;
  `distance_from_consensus` wins 4 of 6 and its CI crosses zero. Six transitions do not
  tell these two apart and the table should not be read as though they did.
- **Our mean is carried by two transitions.** `mlr+mom+K` is +30.42% at 2023->2024 and
  +21.81% at 2024->2025, and negative at 2018, 2020 and 2022. `ewma` is the only family
  on either side that is both positive on the mean and tight (+3.96%, CI clear of zero,
  5/6) — it is small and it is the one that reproduces.
- **`season+K` is the cleanest result in the table and the smallest**: +0.60%, 6/6,
  P = 0.000, CI [+0.34, +0.93]. The kernel on a fixed baseline helps every time and
  helps very little.

**Two of their numbers want explaining rather than reporting.** At 2021->2022 `ne_star`
is **-69.5%** and `ne_star-lbi` **-117.3%** against persistence — 21.57 and 27.66 EMD
against season's 12.73. One transition is most of their deficit on the mean. 2021 is the
post-COVID rebound origin and their coefficients are refit on a six-transition window
that spans the gap.

**The sharpest evidence at that origin is not the blow-up but the collapse beside it, and
it only shows in the exact values.** `ep_x` and `lbi` both score
`12.726851851851848` against season's `12.726851851851851` — the same number to **3
ULP**, which is not "close to persistence" but *is* persistence, down to the last bits
the simplex resolves. `ep_x-ne_star` is `12.726759594915329`, a genuine 1e-4 away and so
a different forecast. So at 2021->2022 two of their six families returned the base
measure exactly and two more went sharply the wrong way; only
`distance_from_consensus` (11.918, +6.35%) did anything useful.

The earlier draft of this paragraph said three families "sit exactly on persistence
(+0.00%)", which was read off the rounded print. Two do, one does not, and the ULP
distance is the fact worth keeping — a percentage rounded to two places cannot tell a
collapsed fit from a nearly-collapsed one. **This is not yet diagnosed** and until it is,
their mean is a number about one origin as much as about their method.

`point_toplbi` at -37.86%, 0/6, is the expected shape: a single-strain prediction scored
by EMD against a whole season's distribution is not a reweighting and loses to one.

### What is not done

- **Both oracle tasks are still running**, so the oracle column is absent from two
  origins and from the table. Rerun `compare_theirs.py` when they land; nothing else
  changes.
- **Their 2021->2022 blow-up is undiagnosed** and is the first thing worth looking at:
  whether it is the training window spanning the COVID gap, the tree at that origin, or
  the metric.
- `eval_figures` still has no `additive` entry, and the 2026-10-02 caveat stands: a
  single run writes both families on panel_v1, so one of the loader's four cross-checks
  becomes trivially true and that must be said in the entry that first draws the panel.
- Region stratification, passage exclusion and level 2 still wait on a GISAID re-pull.
- `captured` / share-of-achievable is not quoted anywhere above, deliberately: the kernel
  families sit above `floor+K` 1.8091 and everyone else above `floor` 2.6730, and
  comparing shares across two floors is the thing `captured` refuses to do.

### Still exposed

Unchanged. One addition, from §1: **the `~60 s/solve` figure is a mixed-family average
and does not transfer to `mlr+mom+K@mu*`**, which ran ~2.3x dearer per solve. Any future
sizing that quotes it should say which families it is sizing.

### Artifacts

- `~/antiGen-count-loss-runs/panel_v1/compare_theirs.py` — the two-sided table, with the
  pre-comparison checks and the ULP argument in its docstring.
- `out/pv1_kcmom_{2018,2020,2021,2022,2023,2024,2025}.json` — merged, five with seven
  families and two partial.
- Jobs: `3124528` done, `3124535` tasks 6–7 running, `3122479` done.

### What the next session should do first

1. **Rerun `compare_theirs.py` when `3124535_6` and `_7` land**, which adds the oracle
   column at every origin and changes nothing else.
2. **Diagnose 2021->2022 on their side.** It is one origin and it is most of the gap.
3. `eval_figures`' `additive` entry, with the cross-check caveat stated in the entry that
   first draws it.

### Addendum, same day — status at the time of writing

`3124535_6` (origin 2024) at 1-21:46 and `3124535_7` (origin 2025) at 1-06:01, both
RUNNING, walls 2026-10-25. The five that finished took 3-10:15 to 6-05:17, so these are
roughly half way and nothing suggests they will not land. Until they do, the oracle
column is absent from two origins and `compare_theirs.py` leaves it out of the table
entirely rather than reporting it over four — which is the same refusal
`merge_families.py --partial` makes one level up.

Nothing else is queued or running on this round. `eval_figures` has not been touched.
