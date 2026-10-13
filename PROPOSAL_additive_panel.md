# Proposal — a strictly additive sequence panel, to replace the rolling Nextstrain backbone

Status: **plan, not a result.** Nothing below has been fitted and no forecast has been
scored. Sections 1 and 6 are zero-fit measurements on data already on disk; everything
else is a decision to be made or a test to be written. The panel this replaces is
`~/h3n2/tree_2026-09-24.json`, 1,664 tips, on which the 2026-10-01 headline table stands.

The ask:

> there's been an inherent issue all along in using these "backbone" trees, which is that
> the future can update the past, i.e. a tree from a later date might not contain all of
> the same sequences as a tree from an earlier date. Can you suggest a protocol for which
> sequences are strictly additive, while maintaining a similar inclusion/exclusion
> criteria as these backbone trees (from nextstrain)? If relevant, feel free to look to
> the Huddleston (2020) paper for inspiration.

**Read section 1 first.** It is the measurement that sets the size of the problem, and it
is larger than the framing of the 2026-09-28 tree-swap entry assumed: not a truncated
tail, a **27.3% churn across the whole record**. Section 2 is the measurement that settles
the design — Huddleston's protocol is empirically additive, checked rather than taken on
their word — and after that the proposal is mostly bookkeeping.

---

## 1. The measurement that sizes the problem

`tree_2026-04-02.json` and `tree_2026-09-24.json` are two builds of
`nextstrain/seasonal-flu` five months apart. Diffed by tip name, restricted to the window
both builds span (2009.264–2026.193, so the September build's extra tail is excluded and
cannot flatter the count):

| | |
| --- | ---: |
| April tips inside the overlap window | 1,682 |
| **absent from the September build** | **459 (27.3%)** |
| new in the September build, same window | 441 |
| shared tips whose `num_date` changed | 23 |

Lost tips by season:

| s2015 | s2016 | s2017 | s2018 | s2019 | s2020 | s2022 | s2023 | s2024 | s2025 | s2026 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 22 | 36 | 18 | 18 | 22 | 19 | 31 | 41 | 24 | 38 |

**This is not the tail.** A quarter of the panel is redrawn every build, uniformly across
eighteen years, and the losses at 2015 and 2017 are the same order as the loss at 2026.
Dates move too — `A/Norway/466/2014` 2014.041 -> 2014.092,
`A/SouthAustralia/55/2014` 2014.432 -> 2014.460 — so a sequence's *position in time* is
also revisable.

This is the mechanism behind the 2026-09-29 finding that the headline "changed because of
the resampling, not because of the truncation the swap was for." A 27% churn rate predicts
exactly that, and it predicts that the next build will do it again. **The backbone panel
is a random variable indexed by download date**, and every number the project reports is
conditioned on a draw it never recorded.

Three things already in the ledger are consequences rather than coincidences: the 8-of-8
that did not survive the retree, finding 3's stability claim, and the top row going to
`distance_from_consensus`. A fourth is the standing exposure that `lbi` and `ne_star` are
read off a single present-day tree — one build swap was worth **0.3866 AA** to `lbi`.

## 2. Huddleston's protocol, and the check that it actually holds

`elife-60067-v2.pdf` p. 24, *Strain selection for natural populations* and *Phylogenetic
inference*:

> Prior to our analyses, we downloaded all HA sequences and metadata from GISAID. For
> model training and validation, we selected 15,583 HA sequences >= 900 nucleotides that
> were sampled between October 1, 1990 and October 1, 2015. To account for known
> variation in sequence availability by region, we subsampled the selected sequences to a
> representative set of **90 viruses per month** with even sampling across 10 global
> regions [...] We excluded all egg-passaged strains and all strains with ambiguous year,
> month, and day annotations. We prioritized strains with more available HI titer
> measurements [...]
>
> **For each timepoint in model training, validation, and testing, we selected the
> subsampled HA sequences with collection dates up to that timepoint.**

The second paragraph is the whole protocol: *one* download, *one* subsample, and every
timepoint is a date-prefix of it. Nothing is ever redrawn.

**Checked, not assumed.** `~/flu-forecasting`,
`natural_sample_1_with_90_vpm_sliding/tip_attributes_with_weighted_distances.tsv`, 39
timepoints 1996-10-01 to 2015-10-01, 13,347 distinct strains.

Their per-timepoint tables are *populations*, not cumulative sets — a ~6-month trailing
window of strains with non-zero KDE frequency, which is the analogue of our
`TipPanel.support(horizon)` with its `lookback`, and is deliberately non-nested. So the
naive nesting test is the wrong one and fails loudly (38 of 38 consecutive pairs "lose"
strains). The right test holds the window fixed: for consecutive timepoints `a < b`, take
the strains in `a`'s population whose date is at or after the earliest date `b`'s
population actually spans, and ask how many `b` dropped.

| | |
| --- | ---: |
| in-overlap strain-timepoints checked | 2,441 |
| **dropped by the later timepoint** | **0** |
| strains carrying two different `numdate`s | **0 of 13,347** |

**Zero.** Their past is never resampled and never re-dated. The protocol works, and it
works for a structural reason worth naming: *a sequence's membership is decided by a rule
that reads only its own month.*

## 3. The design: three layers Nextstrain fuses into one rolling job

A rolling build re-filters, re-subsamples and re-windows the entire record on every run,
and the analysis then takes a prefix of the result. Separate the three and the problem
disappears.

### Layer 1 — the corpus, append-only

One dated GISAID download. We have one: `~/h3n2/metadata.tsv` + `~/h3n2/combined.fasta`,
**202,493 H3N2 HA sequences**, 2013-01-01 to 2025-12-31, keyed by `EPI_ISL_*`. Measured
and not assumed: **every one of the 202,493 carries a full unambiguous Y-M-D**, so
Huddleston's ambiguous-date exclusion is free here and costs zero sequences.

A refresh may only *add* accessions. The two ways a corpus can shrink are handled
explicitly rather than silently:

- **Retractions** go in `revoked.txt`, applied to every horizon at once, forcing a panel
  version bump and a notes entry. A retraction must never produce a per-horizon
  difference, because that is the exact failure mode this proposal exists to remove.
- **Curation of dates.** The 23 re-dated tips in section 1 are this. The manifest (layer
  2) pins each accession's date at admission; a corpus refresh that disagrees is a
  version bump, not a silent correction.

### Layer 2 — the subsample, drawn once, by a cell-local rule

This is the crux, and it is the one place the Nextstrain config is structurally unable to
give what we need.

`augur filter --group-by region year month --subsample-max-sequences N` sets a **global**
budget and spreads it over the groups. The quota for 2015-03 therefore depends on how many
groups exist — on how many months have happened, and on how many sequences arrived in all
of them. **It is a function of the future.** Huddleston's 90-per-month is not: a cell is
filled from its own month's sequences and from nothing else.

So: fix a quota `K` per cell, where a cell is `(month)` or `(region, month)`. Within a
cell, order by

```
u(s) = sha256(accession ‖ salt)
```

and admit the first `K`. Reproducible from the accession alone, and — unlike Huddleston's
"prioritized strains with more available HI titer measurements", which reads titers that
accrue for years after collection and is one of the few places their protocol leaks — a
pure function of the sequence's own identity. The salt is declared once in the manifest
header and never changed.

> **Additivity.** Membership of `s` depends only on `u(s)` and on the other sequences in
> `s`'s own month-cell, every one of which shares `s`'s month. Hence for the horizon
> prefix `S(T) = { s in S : date(s) <= T }` and any `T1 < T2`, `S(T1) ⊆ S(T2)`. ∎

**The one leak, and how to close it.** Backfill into an already-filled cell can displace an
admitted sequence: a 2015-03 sequence appearing in a 2027 refresh with a small `u` would
push out one already in. Close it with **insert-never-evict** — admit the newcomer and let
the cell run over `K`. Cells become slightly uneven; the set only ever grows, under any
future refresh, forever. The manifest records `admitted_at_corpus_version` so the overage
is visible rather than inferred.

The alternative — a fixed admission threshold `u(s) < p_c`, membership a function of `s`
and nothing else, additive with no caveat at all — is cleaner in principle but gives up
the flat cell sizes that the quota exists to produce. **Recommend top-`K` with
insert-never-evict**, and record the threshold variant as the fallback if cell overage ever
becomes material.

### Layer 3 — the horizon, anchored to the forecast origin

At origin `T` the panel is the date-prefix of the frozen `S`. Lookback windows are fine:
`TipPanel.lookback = 5.0` is a deterministic function of `T`, so `S(T)` is not nested
across origins but it *is* reproducible and causal, which is the property that matters.

What is not fine is Nextstrain's 12-year window anchored to the **build date** — which is
why the root moved 2007.758 -> 2008.241 between the two builds, a change nothing in the
analysis asked for. **Anchor every window to the forecast origin, never to "today."**

## 4. Their inclusion/exclusion criteria, and an additive analogue for each

Read off the tree itself (`tree_2026-09-24.json`, 1,664 tips) rather than off a config we
do not have:

- QC: `good` 1,659, `mediocre` 3, `bad` 2 — so `qc.overallStatus_ha != bad` with a handful
  of force-included exceptions.
- 43 tips carry `-egg` or `-cell` in the name (`A/Switzerland/9715293/2013-egg`), which
  Huddleston excluded outright. These are the vaccine and reference force-includes.
- Regions are balanced across **exactly Huddleston's 10**: Europe 303, Oceania 208, North
  America 173, Southeast Asia 154, China 148, Africa 145, Japan Korea 142, West Asia 135,
  South America 130, South Asia 126.

| Nextstrain step | additive? | additive analogue |
| --- | --- | --- |
| min length / coverage | yes, pointwise | keep; `coverage` is in `output/nextclade.tsv` for all 202k |
| nextclade QC `!= bad` | pointwise, but depends on the dataset version | keep; **pin the nextclade dataset version in the manifest** |
| unambiguous Y-M-D | yes | keep — free, the corpus is already 100% full dates |
| exclude egg/cell passage | yes, pointwise | keep; needs a passage column (section 7) |
| `outliers.txt` exclusion | **no** — the list grows | pin the commit; a bump is a panel version bump |
| force-include vaccine/reference strains | **no** — the list grows forward | force-include only strains designated **on or before `T`**, which is additive *and* causal |
| `--group-by region year month --subsample-max-sequences N` | **no** — global budget | fixed per-cell quota, hash order, insert-never-evict (§3) |
| 12y window from the build date | **no** — the root moves | window anchored to `T`, or a fixed start |

Two of the eight are structurally non-additive and both are fixed by the same move:
**replace anything whose value depends on the whole record with something that depends on
one cell or on `T`.**

## 5. Additive is not causal, and the gap is three months

Standing convention 1 is about causality; this proposal is about additivity. They are
different properties and the protocol above delivers only the second.

A collection-date prefix assumes instantaneous reporting. It does not hold:
`~/flu-forecasting-delays` (Huddleston & Bedford 2025, `elife-104282-v2.pdf`) measures the
lag between collection and GISAID availability at **~3 months on average** and shows it
costs real accuracy — "reducing submission lags decreased uncertainty in current clade
frequencies by 50%."

- **Level 1**, what this proposal does: collection-date prefix, with the assumption stated.
  Note which way it cuts — the support at `T` contains late-reported sequences a real
  forecaster would not have had, so it flatters every model's *inputs* equally and the
  target not at all.
- **Level 2**, if a GISAID re-pull supplies `date_submitted`:
  `S(T) = { s : collection(s) <= T and submission(s) <= T }`. Still monotone in `T`, since
  both predicates are. Costs one metadata column.

**Censor the support, never the target.** The target for season `Y+1` is the question, not
an input; submission-censoring it would be asking what we could have known about an
outcome rather than what the outcome was. The asymmetry is deliberate and should be
asserted in code, not left to a docstring.

## 6. What this buys beyond hygiene: the panel stops being sample-starved

The Nextstrain 12-year build is ~8 sequences per month. Huddleston is 90. **An 11-fold
difference**, and it is the better argument of the two.

Measured: hash-order the corpus into `YYYY-MM` cells, take the first `K`, and count what
comes out. All 156 month-cells are non-empty; at `K = 90`, 148 of 156 fill completely.

| | s2017 | s2018 | s2019 | s2020 | s2021 | s2022 | s2023 | s2024 | s2025 | s2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tree (1,664 tips) | 94 | 77 | 75 | 87 | **36** | 54 | 91 | 92 | 86 | 84 |
| `K = 30` (4,680) | 179 | 184 | 181 | 178 | 181 | 180 | 180 | 176 | 180 | 90* |
| `K = 90` (13,792) | 540 | 548 | 545 | 536 | **516** | 547 | 542 | 536 | 542 | 268* |
| `K = 180` (26,446) | 1,081 | 1,093 | 1,098 | 1,073 | **679** | 1,094 | 1,087 | 1,078 | 1,091 | 537* |

\* season 2026 is half-missing from this corpus — see section 7.

Two things fall out.

**The season counts go flat.** The tree's per-season tip count swings 36–94; at `K = 90`
it is 516–548. The EMD target for 2025->2026 is currently **60 sequences**. At `K = 90` it
would be ~540. A target that thin is why a resample moves a transition by 1.3 AA, and
flattening it attacks the variance that section 1 is about from the other side.

**Season 2021 may stop needing a mask.** The eval mask is a standing exposure — chosen
after computing three. The reason 2021 is anomalous is that the tree holds **36** tips in
it. The corpus holds **679**, of which `K = 90` admits 516. The underlying collapse is
real and no quota can manufacture data — 679 against ~20,000 in a normal season is a
30-fold shortfall in the world, not in the sampling — but at 516 target sequences instead
of 36, *whether the anomaly survives becomes a measurement rather than an assumption*.
That is worth the round on its own.

### Tractability, which is the thing that could kill it

Support and target in **distinct genotypes**, which is what the transport problem is sized
by. Measured on `output/nextclade.cds_translation.HA.fasta`; 13,689 of the 13,792 selected
at `K = 90` are present and translated, all **566 residues** — exactly the
SigPep+HA1+HA2 concatenation `load_tips` builds, so no realignment is needed. Only 24 of
13,689 exceed the 2% ambiguity floor.

| `K` | origin | support seqs | support genotypes | target genotypes | `|sup| x |tgt|` |
| ---: | --- | ---: | ---: | ---: | ---: |
| — | current backbone | 379–707 | **249–447** | **30–70** | ~3.1e4 |
| 30 | 2018.25 / 2022.25 / 2025.25 | ~1,780 | 1,036 / 1,045 / 1,029 | 121 / 94 / 51 | ~1.3e5 |
| **90** | 2018.25 / 2022.25 / 2025.25 | ~5,100 | **2,575 / 2,412 / 2,439** | **290 / 283 / 139** | ~7.5e5 |
| 180 | 2018.25 / 2022.25 / 2025.25 | ~9,900 | 3,944–4,279 | 268–517 | ~2.2e6 |

`K = 90` is **~25x the current transport problem**. POT in float64 handles 2,575 x 293
without complaint, but the scored arrays currently run 42m to 11h49m and the gate sweeps
774 configurations x 11 transitions, so a 25x cost is the difference between an overnight
array and a week. Two levers, both already present:

- `TipPanel.max_support` exists for exactly this, caps by abundance within the window, is
  deterministic, uses no selector-chosen hyperparameter, and gives every family the
  identical support and floor. It can only restrict the model, never help it.
- `K = 30` is still **2x the tree** per season and ~1,030 support genotypes, a 4x
  transport cost rather than 25x.

**Recommend running the gate at `K = 30` and the scored arrays at `K = 90`**, with the
operating point carried across — and recording that carry as an exposure, because it is
the same shape as the H2020 kernel arm's transferred operating point already on the list.

## 7. What blocks this today

Stated first because three of the four are real and none is hidden.

1. **No geography in the corpus.** `metadata.tsv` is `strain` + `date` and nothing else;
   `combined.fasta` headers are bare `EPI_ISL_*`. The tree's tips carry `region`, but join
   on `accession_ha` (`EPI269899`, a *segment* accession) and ours are *isolate*
   accessions (`EPI_ISL_196611`) — different namespaces, and the corpus has no strain
   names to fall back on. **Region stratification needs a GISAID metadata re-pull.**
   Without it the quota is month-only, which skews toward whoever sequences most: 2025
   alone is 45,190 of 202,493. Tier 1 is implementable today and is still a strict
   improvement; tier 2 needs one download.
2. **The corpus ends 2025-12-31.** Season 2026 runs Oct 2025 – Mar 2026; we hold the
   Oct–Dec half (22,381 sequences) and not Jan–Mar. A corpus-based panel reproduces the
   exact truncation the 2026-09-28 tree swap was for. The `K = 90` target for s2026 is 139
   genotypes against 283–290 for a full season, which is the same signature. **A refresh
   is required before the last transition can be scored**, and the protocol is built to
   absorb refreshes.
3. **The corpus starts 2013-01-01.** With a 5-year lookback, origin 2017.25 wants support
   back to 2012.25. Origins 2018.25 onward are clean (support 5,145–5,403 sequences);
   2017->2018 is short and should either be dropped or carried with the shortfall stated.
   That takes the panel from 9 origins to 8, or 8 scored transitions to 7 under the 2021
   mask — a real loss of power, to be weighed against section 6's variance reduction.
4. **`lbi` and `ne_star` need a tree.** Our side does not — `season_momentum
   --alignment/--metadata` already bypasses the tree entirely and `load_alignment_dated`
   reads exactly the FASTA + TSV pair this protocol produces. `backbone_compare`
   recomputes both metrics from the topology and so needs one tree per origin.

## 8. The tree, which the protocol makes tractable rather than harder

Per-horizon tree rebuilds have been on the "still exposed" list since 2026-09-24 and were
made concrete on 2026-10-01: `lbi` moved **0.3866 AA** for one build swap. The additive
panel is what makes the fix cheap, because Huddleston's own pipeline is the recipe — at
each timepoint, take `S(T)` and run `augur align` + IQ-TREE + `augur refine` (TreeTime).

Tip sets are then nested by construction, which is the property section 3 proves.
Topologies are not, and should not be: a tree re-estimated from more data genuinely
changes shape, and a forecaster at origin `T` faces exactly that. What is removed is the
illusion that a 2026 build knew what a 2018 forecaster knew.

This closes the additivity hole and the `lbi`/`ne_star` causality hole in one pass. Nine
origins at a few thousand tips each is an afternoon on `cpu_batch`.

## 9. What to build

A **manifest** is the artifact, and the panel's identity is its hash. The discipline is
the one `tree_<meta.updated>.json` + symlink already established on 2026-09-28: the
provenance is in the filename and in a preflight, not in a file date.

```
panel_v1.tsv      header: salt, corpus_sha, nextclade_dataset, outliers_commit, K, cells
                  rows:   accession, date, region, cell, u_hash, admitted_at_corpus_version
```

- `antiGen/panel_manifest.py` — build it from `metadata.tsv` + `nextclade.tsv`, and
  `--verify` an existing one against a refreshed corpus, reporting additions and refusing
  to write a manifest that drops a row.
- `load_alignment_dated` gains a `--manifest` argument: restrict to the admitted
  accessions, and assert the manifest hash matches the one the run records.
- A **manifest preflight** in every `score.slurm`, refusing to run unless the hash matches
  — the direct analogue of the tree preflight, and for the same reason: two directories
  holding the same filenames and differing only in which panel they read is exactly the
  mistake worth spending a preflight on.

## 10. Tests to write before anything is run

The point of the protocol is a structural guarantee, so the guarantee goes in `pytest`,
not in this file.

1. **Additivity.** For every consecutive origin pair, `set(S(T1)) <= set(S(T2))`. One line
   over the manifest, and it is the whole proposal.
2. **Cell locality.** Admission of `s` is invariant to deleting any sequence outside
   `cell(s)`. Fuzz it: drop a random 10% of some other cell, rebuild, assert the manifest
   is unchanged.
3. **Insert-never-evict.** Inject a synthetic accession with an artificially small `u`
   into a filled historical cell; assert the cell grows and that no previously admitted
   accession is dropped.
4. **Date stability.** No accession carries two dates across manifest versions — the test
   that the 23 re-dated tips of section 1 would have caught.
5. **Target is not submission-censored.** Under Level 2, assert the target set for season
   `Y+1` is identical with and without the submission predicate. Pins section 5's
   asymmetry as code.
6. **Hash determinism.** The manifest is byte-identical across two builds from the same
   corpus and salt, on two machines.

## 11. Cost, and what it is being spent on

- One GISAID metadata re-pull (region, passage, `date_submitted`) and one corpus refresh
  through 2026-03. Everything else in §7 follows from those two.
- `panel_manifest.py` plus the `--manifest` path through `load_alignment_dated`: small,
  and the alignment path already exists and is already exercised.
- Nine per-origin tree builds for `backbone_compare`.
- **A third full re-run of both sides.** `competitors_t0924` was 20 tasks and under 15
  minutes; `causalgate_t0924` was 42m–11h49m, and §6 says to expect 4x–25x of that.

## 12. What this costs the record, stated plainly

**This is a third panel.** Every number moves again. The 2026-04-02 -> 2026-09-24 swap
moved individual transitions by up to 1.332 AA, moved `lbi` by 0.3866, and flipped the top
row to `distance_from_consensus`; a move from a 1,664-tip Nextstrain subsample to a
13,792-sequence quota panel is a larger change than that, and the ordering could go either
way. `writeup/mlr_mom_K.tex` is already wrong in four places and this would make it five.

The argument for paying it is that **it is the last time**. After this the panel is a
declared, hashed, append-only object, and the next GISAID refresh adds rows instead of
redrawing a quarter of the record. Every result the project has had to retire in the last
week — the 8-of-8, finding 3, the top row — was retired by a draw nobody recorded. That
stops.

## 13. What would make this round a success

Not a better headline number. Three things, in order:

1. **The additivity test passes**, and the manifest hash is in every run's output. The
   claim "the past is fixed" becomes checkable by someone who was not here.
2. **The retree sensitivity is measured and bounded.** Re-run the panel under two corpus
   versions and report the delta, the way 2026-09-29 reported +0.0033 AA
   [−0.4759, +0.5095] for the tree swap. Under an additive panel that interval should
   shrink toward zero as the corpus grows, and if it does not, that is a finding.
3. **Season 2021's mask is settled by evidence.** 516 target sequences instead of 36 is
   enough to ask whether the anomaly is the world or the sampling. Either answer retires
   a standing exposure.

A fourth, if the tree rebuilds land: `lbi` and `ne_star` stop being read off a present-day
tree, which is the oldest unfixed item on the list and the one that most directly flatters
the comparison against us.
