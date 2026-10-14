# Causal forecasting evaluation: train on the past, score on what happened next

The random held-out split answers "can the model fit these trees". It is not a forecast:
a held-out node's parent usually sits in the training half, and — more fundamentally —
the tree's topology, ancestral reconstruction and tip subsampling were all inferred from
data that includes the held-out examples. This design removes both problems by training
on a tree *as it was published on a past date* and scoring against substitutions that
only appear in today's tree.

## The mechanism

nextstrain.org serves dated dataset versions through the Charon API:

```
https://nextstrain.org/charon/getDataset?prefix=/ncov/open/global/all-time@2025-01-01
```

returns the build exactly as published then — verified: that snapshot reports
`updated=2024-12-27` with a latest tip of 2024.92, against 2026.68 in today's tree.
Nothing after the forecast date can leak into training, including the phylogenetic
inference itself. `fetch_snapshots.py` wraps this and records what it retrieved.

## Decisions (locked 2026-10-13, before anything was scored)

**D = 2025-01-01.** Chosen on post-D mutation *counts* only -- sample-size reasoning, not
outcome reasoning; no model has been run against any post-D data. Per the standing causal-
selection rule, D is fixed from here: epochs, learning rate and the arm are selected on a
random split of the snapshot examples alone, and the post-D set is opened once at the end.

**Queries are anchored on the snapshot, not on today's tree.** See *Test* below.

## Feasibility (probed 2026-10-13)

Retention, by `@date` against the Charon dataset path (Y = served, `.` = 404):

```
                             2022 23-01 23-07 24-01 24-07 25-01 25-07 26-01
  ncov                         .    Y     Y     Y     Y     Y     Y     Y
  measles, evd68, mumps, zika  Y    Y     Y     Y     Y     Y     Y     Y
  mpox                         .    Y     Y     Y     Y     Y     Y     Y
  dengue denv1-4               .    .     .     .     Y     Y     Y     Y
  avian flu h5n1, h5nx         .    .     .     .     Y     Y     Y     Y
  rsv_a, rsv_b                 .    .     .     .     .     Y     Y     Y
  WNV                          .    .     .     .     .     .     Y     Y
```

Post-D amino-acid substitutions (branches with parent date <= D < child date), measured
against the **live** trees:

| tree | gene | D = 2024-07-01 | D = 2025-01-01 |
|---|---|---|---|
| h5nx | HA | 490 | **318** |
| h5n1 | HA | 388 | 307 |
| ncov | S | 120 | **101** |
| rsv_b | F | 80 (no snapshot) | **77** |
| rsv_a | F | 69 (no snapshot) | **52** |
| measles | genome | 36 | **31** |
| denv2 | E | 51 | **18** |
| denv1 | E | 40 | **15** |
| mumps, zika | | 5 | **5** |
| evd68 | genome | 12 | 0 |
| WNV | | 17 (no snapshot) | 7 (no snapshot) |

**~620 post-D mutations at D = 2025-01-01** using one avian tree (denv3/denv4 not yet
measured live, so the real figure is nearer 650). That is comparable to or larger than
the 517 the random-split comparison used, so power is no longer the binding constraint.

Four findings constrain the design:

1. **Three dataset paths in the old table were wrong.** `/measles/genome` and
   `/dengue/denv2` are not dataset paths; the real ones are `/measles/genome/global` and
   `/dengue/denvN/E`. The earlier probe resolved them to a redirect page, which is why
   dengue looked frozen and the panel looked small. Fixed in `fetch_snapshots.py`.
2. **Seasonal flu is delisted, not merely un-snapshotted.** Of the 318 datasets Charon
   serves, *zero* match `flu/seasonal`. Those trees cannot be forecast-evaluated by any
   route. They remain valid training-only context for D > 2024.3.
3. **But avian flu HA is live and snapshotted, and it dominates the test set.** This
   retires the old "the trees with the strongest joint-training signal are unavailable"
   limitation: h5nx/h5n1 are HA, the same protein as h3n2 and h1n1pdm, and contribute
   more post-D mutations than the entire rest of the panel combined.
   **h5n1 and h5nx share 1588 tip names (37% of the smaller tree)** -- they are
   overlapping clade cuts of one dataset. Use exactly one; this design uses **h5nx**
   (more post-D mutations, broader clade). Dengue serotypes share zero tips.
4. **The local tree copies are stale and must not be used here.** dengue stops at 2023.87,
   avian flu at 2024.24, seasonal flu at 2024.27 -- all fetched from
   `data.nextstrain.org`, which is a different artifact from the Charon dataset path.
   Re-fetch everything from Charon.

## Protocol

**Train.** Generate examples from each snapshot with the existing generator (no cutoff,
everything TRAINING). The snapshot's own **dated root-sequence sidecar** must be used:
the dataset JSON has no `root_sequence` key, so without it the snapshot tree would be
built against today's reference. `fetch_snapshots.py` now retrieves
`&type=root-sequence` at the same `@date`. Verified for ncov that the sidecar is
identical across dates; that is not guaranteed for other builds.

Optionally add the frozen trees (seasonal flu, and any dataset whose last tip precedes D)
as extra training-only context -- this is the "does joint training help" question, asked
where the answer matters.

**Select.** Epochs, learning rate and the arm are chosen on a random held-out split *of
the snapshot examples only*. Enforced by construction: the post-D set does not exist as a
file until selection is finished.

**Test -- anchored on today's tree.** NOT the design below: snapshot anchoring was
built, measured and abandoned on 2026-10-13 (see NOTES.md s11, *Query anchoring*).
Branch-level tip-set matching recovered only 3% of h5nx's crossing branches -- most
crossing parents have zero pre-D descendant tips -- and four trees share zero tip
names across builds because Nextstrain renamed them. The accepted cost is that the
query genotype is mildly hindsight-informed; it never touches training or epoch
selection. The original intent is kept below for the record.

**Test -- snapshot-anchored (NOT IMPLEMENTED; superseded above).** The earlier draft
took the query genotype from *today's* tree. That leaks: today's ancestral reconstruction used post-D tips, including the very
descendants whose substitutions are the target, so parsimony can push a descendant's
substitution up into the parent state. Instead:

- **Query** = an internal node of the **snapshot** tree. Its genotype is inferred from
  pre-D data only, so it carries no post-D information of any kind.
- **Match** into today's tree by **descendant tip-name set**. Internal node names are
  regenerated every build and are not stable across versions; tip (strain) names are. The
  match for snapshot node `N` is the MRCA in today's tree of `tips(N)` intersected with
  today's tips.
- **Validate** each match: the matched node's pre-D descendants must equal
  `tips(N) & today's tips`. Nodes failing this are dropped and the match rate is reported,
  not assumed.
- **Target** = substitutions on branches below the matched node with
  parent date <= D < child date.

Two subsets are scored separately: *all* post-D substitutions, and *novel* ones --
absent from the entire snapshot tree. The novel subset is the forecast that matters and
the one every count-based baseline is structurally unable to make.

**Sensitivity.** Also build the today's-tree-anchored version and report how often the two
reconstructions of the same pre-D ancestor actually disagree. That measures the size of
the concession the simpler construction would have made, rather than asserting it is small.

**Score.** NLL per post-D mutation and top-q hit rate (q = 1, 5, 10, 25%) over the full
sequence, with the paired bootstrap from `compare_runs.py`.

**Baselines**, all fitted on pre-D data only: uniform, codon accessibility (data-free),
the learned substitution matrix, the per-position `where_only` map, the factorized
`where_times_which` predictor, and the empirical `frequency` joint. On the novel subset
`frequency` should collapse to its smoothing floor while the substitution-level rungs hold
up -- that contrast is the point of the experiment.

## Arms

Panel at D = 2025-01-01: **ncov, rsv_a, rsv_b, measles, h5nx, denv1-4, evd68, mumps,
zika, mpox**.

| arm | trained on | asks |
|---|---|---|
| `single_<tree>@D` | one snapshot | per-tree forecasting ability |
| `joint_panel@D` | the whole forecastable panel | does joint training improve forecasts |
| `joint_all@D` | panel + every frozen tree (seasonal flu, ...) | does unrelated extra data help a forecast |
| baselines | pre-D only | the bar |

`joint_all` vs `joint_panel` is the causal version of the joint5-vs-joint3 question, and
the one worth reporting: extra trees are free at inference time, so if they improve a
genuine forecast that is a result regardless of effect size.

## Honest limitations

- **One forecast date.** D is a single draw. 2024-07-01 is the natural robustness
  replicate (more horizon, ~930 mutations) but has no RSV or WNV snapshots.
- **The panel is dominated by one tree.** h5nx alone is roughly half the post-D
  mutations, so the pooled number is largely an avian-flu HA result. Report per-tree
  alongside pooled, and expect every tree except h5nx, ncov and rsv_b to be underpowered.
- **No seasonal flu.** The specific trees from the random-split result cannot be tested;
  avian HA is a related but distinct panel.
- **Targets still come from today's inference.** Unavoidable -- that is the observation.
  Only the *input* side is made leak-free.

## Status

Done: `fetch_snapshots.py` (corrected paths, dated root-sequence sidecars, retention
probed and recorded above); D and the query anchoring locked.

Still to build: `--testing-after DATE` in `make_tree_examples.py`, which needs real date
handling -- `Node.__slots__` has no date field, `parse_tree` never reads `num_date`, and
every example is written `TRAINING` unconditionally. The subtle part is
`collapse_zero_mutation_branches`: collapsing merges branches, so a collapsed branch can
straddle D and silently change which branches count as crossing. Then the tip-set matcher,
the arm scripts, and the scoring pass.
