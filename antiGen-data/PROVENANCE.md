# Input provenance: trees, sidecars and snapshots

Phylogeny inputs, kept as files rather than re-fetched on demand.

## `data/ncov_global@2020-12-29.json`

The Nextstrain `/ncov/global` build as it stood just before 2021-01-01, for the
D = 2021-01-01 arm.

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

All four match what was recorded for this tree when the 2021 arm was built,
independently of this fetch.

### The sidecar is a different date, on purpose

The root-sequence capture is 2020-11-21, five weeks before the tree. For ncov this is
a fixed reference genome (Wuhan-Hu-1) rather than a re-inferred ancestral state, so it
carries no post-cutoff information: `nuc` is 29,903 nt and `S` is 1,274 aa
(1,273 + stop), beginning `MFVFLVLLPLVSSQCVNLTTRTQLPPAYTN` and ending `*`.

Against today's reference it differs at **exactly one S residue, 943 (F here, S
today)** — which is the single "from-base mismatch on the train side" this arm
shows, and so is expected rather than a fetch error. `nuc` differs at
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

The corpora built from this tree are not committed: they are written to
`output/sc2-2021/{train,test}/` by `slurm/run_2021.sbatch` and `slurm/place_full.sbatch`.

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

## The h3n2, rsv-a and denv2 snapshots

The same pipeline, three more pathogens. `~/h3n2`, `~/rsv-a` and `~/denv2` each
already hold a current `tree.json` + `ref.json`; these are the **historical
snapshots** alongside them, one per arm, at a requested date rounded to the
nearest quarter and then resolved the way the ncov series resolves a cutoff —
**the latest build strictly before it**.

```
python3 -I fetch_snapshots.py all
```

| arm | requested | nearest quarter | build taken | lag | files |
|---|---|---|---|---|---|
| rsv-a | 2023-11-11 | **2023-10-01** | 2023-09-26 | 5 d | `~/rsv-a/{tree,ref}_2023-09-26.json` |
| h3n2 | 2022-11-09 | **2022-10-01** | 2022-09-23 | 8 d | `~/h3n2/{tree,ref}_2022-09-23.json` |
| denv2 | 2019-09-15 | **2019-10-01** | 2019-08-16 | 46 d | `~/denv2/{tree,ref}_2019-08-16.json` |

### Object versions, not dated keys

ncov is the only build Nextstrain ever wrote under *dated* S3 keys — the thing
`fetch_trees.py` relies on. Flu, RSV and dengue overwrite one undated key on
every rebuild. The history is still there, because `nextstrain-data` is
versioned, so here a snapshot is an **S3 object version** and "the build as of
D" is the latest version strictly before D. Keys get renamed, so each arm is a
*chain* and the pick scans all of it:

| arm | chain | versions | range |
|---|---|---|---|
| h3n2 | `flu_seasonal_h3n2_ha_12y` → `seasonal-flu_h3n2_ha_12y` | 248 + 159 | 2019-12-11 .. today |
| rsv-a | `rsv_a_genome` → `rsv_a_genome_all-time` | 105 + 135 | 2022-09-12 .. today |
| denv2 | `dengue_denv2_tree`/`_meta` → `dengue_denv2` → `dengue_denv2_genome` | 14 + 29 + 321 | 2018-01-28 .. today |

Which key is which build was settled against the trees already in the folders,
not assumed: the current `seasonal-flu_h3n2_ha_12y` reproduces `~/h3n2/tree.json`'s
tip-date floor of **2009.264** exactly, `rsv_a_genome_all-time` reproduces
`~/rsv-a/tree.json`'s **1977.999**, and `dengue_denv2_genome` is the only dengue
key whose `nuc` is the full **10,723 nt** genome that `~/denv2/ref.json` carries.

### denv2 is auspice v1, and is converted

The dengue build was rebuilt rarely before 2023 — fourteen versions in six
years — and the only one before 2019-10-01 is **2019-08-16**, which predates
the v2 export. It arrives as auspice **v1**: a tree file and a meta file, node
state under `attr`, branch mutations under `muts`/`aa_muts`, CDS starts counted
from 0. `from_v1()` in `fetch_snapshots.py` converts it; the two halves are kept
as fetched next to it, as `tree_2019-08-16_v1-source_{tree,meta}.json`.

The converter is checked against the one build Nextstrain exported **both**
ways. `dengue_denv2_{tree,meta}.json` at 2019-11-09 and `dengue_denv2.json` at
2019-11-10 are a day apart (`meta.updated` 2019-11-08 and 2019-11-09), so the
comparison is by reconstructed tip sequence rather than by node name — internal
names are assigned per export. Of the 1,022 tips the two builds share, **1,021
reconstruct to an identical E protein**; the one that differs does so at a
single residue, E-155, which is the day's rebuild and not the conversion.

### Every snapshot needs its own sidecar

For ncov the sidecar is a fixed reference (Wuhan-Hu-1) and the series could
share one across years. Here it is the **re-inferred ancestral root**, and it
moves. Against each folder's current `ref.json`:

| arm | gene | residues differing |
|---|---|---|
| h3n2 | HA1 (329 aa) | 1 — position 8, D in 2022, N now |
| rsv-a | F (575 aa) | 7 |
| denv2 | E (495 aa) | 11 |

`nuc` moves further still — 956 of 10,723 sites for denv2 — and its *length*
moves too, 1,701 → 1,737 for h3n2 and 15,160 → 15,225 for rsv-a, so a snapshot
tree read against today's `ref.json` would be wrong in the coordinate frame as
well as in the bases. Each tree is paired with the sidecar version from its own
build day.

Even the word "root" is not quite the sidecar's. These sidecars are the
*reference* the branch mutations are stated against, and a build whose inferred
ancestor sits off that reference says so as **branch mutations on the root node
itself**. The two historical snapshots and the denv2 pair carry none; the
current `~/h3n2/tree.json` carries 13 on HA1+HA2 and the current
`~/rsv-a/tree.json` carries 7 on F. ncov's root never does, which is why this
surfaced only here (see `pinnedRoot` in `src/`).

**From mid-2024 the sidecar is inside the tree.** Nextstrain stopped publishing
`<key>_root-sequence.json` during 2024 — dengue's last version is 2024-01-16 and
RSV's 2024-05-21, while the tree keys themselves are current — and put a
top-level `"root_sequence"` in the dataset JSON instead. Taking the newest
surviving sidecar for a later build is not a near miss but a wrong answer, since
the root is re-inferred every rebuild: the 2024-12-31 dengue tree read against
its own 2024-01-16 sidecar shows 66 from-allele mismatches, 28 of them real
conflicts, and none at all against the inline one. `fetch_snapshots.py` prefers
the inline root sequence and falls back to the sidecar key only when a build has
none, which is every build before mid-2024.

The **gene set** moves as well: the 2023 rsv-a build annotates one `M2` CDS of
195 aa where today's splits it into `M2-1` (195) and `M2-2` (91). `F` is
untouched, and `F` is the gene this arm is about, but anything keyed by gene
name has to read each build's own `genome_annotations` rather than assume
today's.

### Guards, as verified on download (2026-10-28)

All three PASS: `post_cutoff_tips` **0**, mutations present on the headline
gene, sidecar carries both `nuc` and that gene. `data/snapshots_manifest.tsv`
is the index.

| arm | build | tips | nodes | tip dates | gene | edges | mutations |
|---|---|---|---|---|---|---|---|
| h3n2 | 2022-09-23 | 2,344 | 4,034 | 2005.86 .. 2022.686 | HA1 | 1,091 | 1,465 |
| rsv-a | 2023-09-26 | 1,918 | 3,504 | 1956.02 .. 2023.295 | F | 557 | 768 |
| denv2 | 2019-08-16 | 1,198 | 2,203 | 1944.15 .. 2019.575 | E | 388 | 661 |

And the check the ncov README calls "every branch's from-allele matches the path
above it", run here against the sidecar for **every** CDS, not just the headline
one:

| arm | mutations checked | from-state mismatches |
|---|---|---|
| h3n2 | 6,005 nuc + 2,061 aa | **0** everywhere |
| rsv-a | 39,888 nuc + 7,547 aa | **0** everywhere |
| denv2 | 31,779 nuc + 4,801 aa | 0 on every CDS; **18 on `nuc`** |

### The denv2 sidecar is a later build's, and the cost is two sites

2019-08-16 predates every version of `dengue_denv2_root-sequence.json` — the
first is 2019-11-10 — so it gets that first one, the same fallback
`fetch_trees.py` applies to the three 2020 ncov builds, and
`snapshots_manifest.tsv` says so in `root_version_date`.

The 18 `nuc` mismatches above are all of the cost, and they are **two sites**:
**6732** (NS4A, reads T in the August root and C in the November one) and
**8289** (NS5, C then T). Both are third codon positions, and both are
synonymous — which is what the 0 mismatches across all 4,801 amino-acid
mutations already say independently. The amino-acid frame, which is all this
pipeline reads, is exact.
