# antiGen-data-revised

Phylogeny inputs, kept as files rather than re-fetched on demand.

## `data/ncov_global@2020-12-29.json`

The Nextstrain `/ncov/global` build as it stood just before 2021-01-01, for the
D = 2021-01-01 arm (antiGen NOTES s27 / s29).

**It does not come from Charon.** The ncov archive floor is **2022-05-01** — a
`getDataset` for 2022-04-29 returns 404, bisected. No 2021 snapshot exists there, and
none of the snapshots under `antiGen/nextstrain/` reaches back past **2024-07-01**.
It comes from the Wayback Machine:

| file | capture | URL |
|---|---|---|
| `ncov_global@2020-12-29.json` | `20201229050041` | `https://web.archive.org/web/20201229050041id_/https://nextstrain.org/charon/getDataset?prefix=/ncov/global` |
| `ncov_global@2020-12-29_root-sequence.json` | `20201121013321` | `https://web.archive.org/web/20201121013321id_/https://nextstrain.org/charon/getDataset?prefix=/ncov/global&type=root-sequence` |

The `id_` in the path is load-bearing: it returns the original bytes rather than the
archive's rewritten page.

`20201229050041` is the **last capture before the cutoff**. The CDX index for
2020-12-15 .. 2021-01-15 holds 2020-12-15, -20, -21, -23, **-29**, then 2021-01-14,
which is post-cutoff. The sidecar has only two captures in range, 2020-11-21 and
2021-01-14, so the November one is the only pre-cutoff option.

### A REQUESTED CAPTURE DATE IS NOT A GUARANTEE

Wayback silently falls back to the nearest capture. Asking for `/ncov/north-america`
and `/ncov/africa` at 2020-12-20 returned builds **updated 2021-04-29 carrying ~2,200
post-cutoff tips each**; `/ncov/south-america` returned 2021-01-15. Nothing in the
request or the response status says so. Only the payload does. Re-fetch by exact
timestamp as above, and re-run the guards:

```
python3 -I guard.py data/ncov_global@2020-12-29.json
```

### Guards, as verified on download (2026-10-21)

| check | value |
|---|---|
| `meta.updated` | **2020-12-28** |
| `meta.title` | Genomic epidemiology of novel coronavirus - Global subsampling |
| tips | **3,848** |
| tip `num_date` range | 2019.9849 .. **2020.9768** |
| tips at or after 2021.0 | **0** |

All four match what NOTES s27 recorded for this tree, independently of this fetch.

### The sidecar is a different date, on purpose

The root-sequence capture is 2020-11-21, five weeks before the tree. For ncov this is
a fixed reference genome (Wuhan-Hu-1) rather than a re-inferred ancestral state, so it
carries no post-cutoff information: `nuc` is 29,903 nt and `S` is 1,274 aa
(1,273 + stop), beginning `MFVFLVLLPLVSSQCVNLTTRTQLPPAYTN` and ending `*`.

Against today's reference it differs at **exactly one S residue, 943 (F here, S
today)** — which is the single "from-base mismatch on the train side" NOTES s27
reports for this arm, and so is expected rather than a fetch error. `nuc` differs at
114 of 29,903 sites, concentrated at the masked ends. The 2020 build also carries
`ORF10` and `ORF14`, which today's does not.

## `data/ncov_open_global_all-time@2026-03-15.json`

The **new starting tree**: Nextstrain's `ncov/open/global/all-time` build — open
(GenBank) data, global, subsampled evenly across the whole pandemic — at the latest
snapshot **before 2026-03-22**, which is the last collection date in
`~/sc2-open-data/metadata.tsv`.

**This one does come from Nextstrain's own archive**, unlike the 2020 tree: the archive
floor is 2022-05-01, so a 2026 date is in range and the Wayback detour is not needed.

| file | request |
|---|---|
| `ncov_open_global_all-time@2026-03-15.json` | `https://nextstrain.org/charon/getDataset?prefix=/ncov/open/global/all-time@2026-03-21` |
| `ncov_open_global_all-time@2026-03-15_root-sequence.json` | the same, plus `&type=root-sequence` |

The file is named for the build's own `meta.updated`, 2026-03-15, not for the date
requested.

### `@date` resolves to the latest build at or before it, which Wayback does not

Worth bisecting rather than assuming, because the 2020 tree's whole provenance problem
was an archive that silently returns a *later* capture. Nextstrain does not:

| requested | returned `meta.updated` |
|---|---|
| `@2026-03-08` | 2026-03-08 |
| `@2026-03-14` | **2026-03-08** — the earlier one, not the nearer 03-15 |
| `@2026-03-16` | 2026-03-15 |
| `@2026-03-21` | 2026-03-15 |
| `@2026-03-22` | 2026-03-22 |
| none (current) | **2026-10-18**, 3,018 tips — the build is still running, so the date is doing real work |

Builds are weekly, on Sundays. **2026-03-15 is the latest one strictly before 2026-03-22**;
a 2026-03-22 build does exist, and taking it instead would be the "on or before" reading.

### Guards, as verified on download (2026-10-22)

```
python3 -I guard.py "data/ncov_open_global_all-time@2026-03-15.json" 2026.2192
```

| check | value |
|---|---|
| `meta.updated` | **2026-03-15** |
| `meta.title` | Genomic epidemiology of SARS-CoV-2 with subsampling focused globally since pandemic start |
| `meta.data_provenance` | GenBank — the open build, no GISAID |
| tips | **2,939** (5,688 nodes, largest multifurcation 15) |
| tip `num_date` range | 2019.9770 .. **2026.1660** (≈ 2026-03-01) |
| tips at or after 2026.2192 (2026-03-22) | **0** |

`guard.py` now takes the cutoff as a second argument and still defaults to 2021.0, so the
2020 tree's guard is the same command it always was.

### The sidecar is the current reference, and that is checkable

`S` is 1,274 aa (1,273 + stop), `nuc` is 29,903 nt, no ambiguity codes, and it differs
from the 2020 sidecar at **exactly one S residue, 943 (F there, S here)** — which is the
single difference the 2020 section above already predicted against "today's reference".
Two independently fetched references agreeing to one documented residue is the strongest
check available that this is the right file.

### It parses, and the machinery runs on it unchanged

```
tree         : 5688 nodes, 2939 tips, 5687 edges, largest multifurcation 15
               1450 edges carry S mutations (2570 mutations total)
               65796 residues recorded against the root, 8335 of them missing
               every branch's from-allele matches the path above it
allele sets  : 505 of 1273 sites carry more than one allele
               parsimony score 2133 against 2570 Nextstrain mutations
check        : 0 observations outside their own allele set
               0 (node, site) pairs where Nextstrain's reconstruction is outside the set
examples     : 1055 records, 2133 mutations; check_examples.py clean
```

Six years of diversity rather than one, and it shows: 505 polymorphic sites against 386,
ambiguity at 1,633 (node, site) pairs against 29 with sets reaching size 4, and a longest
resolved edge of 27 mutations against 5 — that one is Omicron's branch.

## Not included

The derived training examples this tree produced are already in the main repo and are
what the 2021 arm actually consumed:

```
antiGen/experiments/multitree/data/forecast2021/examples_ncov@2021-01-01_{train,test,novel}.txt
antiGen/experiments/multitree/data/forecast2021/examples_ncov@2021-01-01_test_novelty.json
```

## `data/trees/` — the quarterly cutoff series

The same tree the D = 2021-01-01 arm used, taken again at every quarter from
**2020-04-01 to 2026-01-01** — 24 cutoffs. For each one: the latest global,
all-time-subsampled ncov build **strictly before** the cutoff, its root-sequence
sidecar, and the guards. `data/trees/manifest.tsv` is the index, one row per
(chain, cutoff); `fetch_trees.py` rebuilds the whole thing and is resumable.

```
python3 -I fetch_trees.py data/trees both
```

### Neither Charon nor the Wayback Machine reaches back far enough

Both of the routes the two existing trees came by run out, and in opposite directions:

| route | coverage | why it is not enough |
|---|---|---|
| `getDataset?prefix=…@date` | **floor 2022-04-30** — 2022-04-29 is a 404, bisected | misses nine cutoffs; and `/ncov/global` is a 404 at *every* date, the name is retired |
| Wayback CDX | `/ncov` 2020-01-21..2020-04-05 (47 captures), `/ncov/global` 2020-04-09..2021-06-20 (105), `/ncov/gisaid/global` 2021-07-13..2021-09-04 (7) | a hole from **2021-09-05 to 2022-04-29**, which is three cutoffs, and only 6 sidecar captures in all |

The public **`nextstrain-data` S3 bucket** has no hole. Every build was written
under its own dated key and all of them are still there, anonymously listable and
readable. That is where this series comes from.

### One build, six names, no gap

| served as | S3 key prefix | dated builds | range |
|---|---|---|---|
| `/ncov` | `ncov_` | 66 | 2020-01-22 .. 2020-04-08 |
| `/ncov/global` | `ncov_global_` | 328 | 2020-04-08 .. 2021-07-06 |
| `/ncov/gisaid/global` | `ncov_gisaid_global_` | 188 | 2021-07-07 .. 2022-04-29 |
| `/ncov/gisaid/global/all-time` | `ncov_gisaid_global_all-time_` | 603 | 2022-04-30 .. **2025-10-10** |
| `/ncov/open/global` | `ncov_open_global_` | 180 | 2021-07-07 .. 2022-04-29 |
| `/ncov/open/global/all-time` | `ncov_open_global_all-time_` | 928 | 2022-04-30 .. 2026-10-18 |

The first four are one continuous chain — **the GISAID chain**, which is what
`/ncov/global` always was and so what the 2021-01-01 arm already ran on. The open
(GenBank) chain begins only at the 2021-07 split, so it cannot reach the 2020
cutoffs at all. Both are downloaded, under `data/trees/gisaid/` and
`data/trees/open/`; **the GISAID chain is the series**, and the open one is there
because the placement queue is open data and the 2026 tree above is an open build.

### Every cutoff is available, and the lag is a day or two

24 of 24 on the GISAID chain, 18 of 24 on the open chain. The build is 1–4 days
before its cutoff everywhere **except 2026-01-01 on the GISAID chain, where it is
83 days**: `ncov_gisaid_global_all-time` stopped on 2025-10-10 and never resumed.
That is the only cutoff where the two chains differ materially — open has
2025-12-28, four days out — and the only place the "latest available preceding
build" rule does visible work.

### The 2021-01-01 tree comes back byte-identical

`ncov_global_2020-12-28.json` from the bucket and `ncov_global@2020-12-29.json`
from the Wayback Machine are the same file, `md5 4edfa514a7b7df6bf3a519c77f670be2`,
and the sidecars agree too — same `S`, same `nuc`, all 29,903 of it. Two
independent routes to one build is the check that says the other 23 are the right
files as well.

### Sidecars, where no dated one exists

Dated `_root-sequence.json` keys exist only for the all-time families, and only
from 2022-07 — `ncov_open_global_all-time_2022-06-30_root-sequence.json` is a 404.
Everywhere else the undated key carries **S3 object-version history**, and the
version at or before the build's own day is the sidecar as it stood:

| key | versions | range |
|---|---|---|
| `ncov_global_root-sequence.json` | 67 | 2020-11-05 .. 2021-07-06 |
| `ncov_gisaid_global_root-sequence.json` | 1 | 2021-07-19 |
| `ncov_gisaid_global_all-time_root-sequence.json` | 593 | 2022-06-08 .. 2025-10-24 |
| `ncov_open_global_root-sequence.json` | 1 | 2021-07-19 |
| `ncov_open_global_all-time_root-sequence.json` | 895 | 2022-06-08 .. 2026-10-18 |

**Three builds predate every sidecar there is.** 2020-03-31, 2020-06-30 and
2020-09-30 are all before 2020-11-05, when the first one was written; they get that
first one, and `manifest.tsv` says so in `root_version_date`. The cost is bounded
and known: the sidecar is the Wuhan-Hu-1 reference rather than a re-inferred
ancestral state, and the only S residue that has ever moved across the whole series
is **943**, which the 2020 section above already reports. It reads `F` through
2021-02-12, `X` by 2021-06-07, and `S` today.

### Guards, as verified on download (2026-10-23)

All 42 trees pass: `post_cutoff_tips` **0** on every one, S mutations present on
every one, `S` 1,274 aa on every sidecar. The per-cutoff numbers are in
`manifest.tsv`. The shape of the series, on the GISAID chain:

| cutoff | build | tips | S mutations |
|---|---|---|---|
| 2020-04-01 | 2020-03-31 | 2,489 | 192 |
| 2020-10-01 | 2020-09-30 | 4,939 | 893 |
| 2021-01-01 | 2020-12-28 | 3,848 | 1,322 |
| 2022-01-01 | 2021-12-31 | 3,492 | 2,538 |
| 2023-01-01 | 2022-12-30 | 3,122 | 2,337 |
| 2024-01-01 | 2023-12-30 | 3,728 | 3,630 |
| 2025-01-01 | 2024-12-28 | 3,377 | 3,549 |
| 2026-01-01 | 2025-10-10 | 3,401 | 4,030 |

Tips stay near 3–4,000 throughout — the subsampling target — while the S mutation
count climbs from 192 to 4,030. The earliest arm is thin by construction: at
2020-03-31 the whole tree carries 192 S substitutions on 178 edges, against 1,322
nine months later.
