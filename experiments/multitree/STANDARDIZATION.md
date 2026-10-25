# Standardized train/test splits — conventions and recipe

How splits are built on the `av-multitree` branch, so Ivan's data and our own data go
through one consistent scheme and stay comparable. Two generators, one on-disk format.

## The two arms

| arm | generator | method | used for |
|---|---|---|---|
| **attach** | `antiGen-data-revised/` (vendored C++; `make` → `antigen`) | parsimony attachment of a sequence corpus onto the guide tree | SC2 (Ivan's quarterly series + test) |
| **no-attach** | `experiments/multitree/make_tree_examples.py` (pure Python) | guide tree only — internal nodes + branch intermediates, no corpus | our own viruses (measles, hmpv, chikv, rsv-a, ebola-bdbv) |

**Never mix the two arms for one virus in a single comparison table.** They answer
different questions and are byte-format-identical on disk, so a mislabelled directory
cannot be caught automatically. Label every split's arm explicitly in its path/filename.

## The on-disk format (identical across both generators)

One record per inferred genotype, `EXAMPLE_ID:`-labelled:

```
EXAMPLE_ID: <n>
PARENT_EXAMPLE_ID: <n|None>
MUTATION_FROM_PARENT:
<from,pos,to | None>
<split label>                 # see below
IS_DESCENDED_FROM_VARIANT:
<label,True|False lines, or empty>
IS_VARIANT_ROOT:
<label,True|False lines, or empty>
<full aligned AA sequence>
MUTATIONS_TO_CHILDREN:
<pos,aa lines>
<blank line>
```

Fixed everywhere:
- **Alphabet** `"ACDEFGHIKLMNPQRSTVWY*-"`, **E = 22** (stop + gap included). The revised
  C++ generator emits only the 20 standard residues; the gap/stop columns simply stay
  unpopulated — no model change, and existing masks still apply.
- **Positions are 0-indexed** in both generators (verified empirically on the sc2 2021
  arm: 2933/2933 `MUTATION_FROM_PARENT` records fit base 0, max pos 1272 < 1273). The
  loader does **not** re-base positions; if a future file is 1-indexed it will fail the
  range check loudly rather than shift silently.

### Split-label dialects (the only on-disk difference)

- `antiGen-data` / `make_tree_examples.py`: the bare word `TRAINING` or `TESTING`.
- `antiGen-data-revised`: `POST_CUTOFF: True` / `POST_CUTOFF: False` (post-cutoff ==
  testing). **Single-line form only.** The multi-cutoff block form
  (`--post-cutoff d1,d2,...` → a list of `<date>,True` lines) is *not* consumed — build
  one file per cutoff.

`antiGen/load.py:_parse_split_label` accepts both; both pass `antiGen-validate-examples`.

## Naming and layout

- Attach (SC2): `antiGen-data-revised/output/sc2-<D>/train/examples_<D>_cur.txt` and the
  shared test set `.../sc2-test/examples_all_cur.txt` (or per-arm `sc2-<D>/test/`).
- No-attach (own viruses): `experiments/multitree/data/examples_<virus>@<D>_{train,test,
  novel}.txt` + `_test_novelty.json`.
- **Tracked vs local:** the `.txt` example/MSA files are large and gitignored (regenerable);
  the small JSON novelty sidecars and `data/trees.json` are tracked, so a figure stays
  reproducible from the commit that adds it.

## Recipe — build a no-attach split for one of our viruses

Our `data/*/` trees are auspice v2 with an embedded `root_sequence`, so no `--ref` is
needed. Pick the gene(s) from the tree's mutation namespace (measles `F`, chikv `E1`,
hmpv `F`/`G`/…).

```bash
# 1. Training snapshot: every node is TRAINING, no cutoff.
python experiments/multitree/make_tree_examples.py \
    --tree data/measles/measles_f_tree.json --genes F \
    --out experiments/multitree/data/examples_measles@<D>_train.txt
# verified: measles F -> 266 records, seq_len 550, 507 child mutations, 0 from-base mismatches

# 2. Post-D test set from today's tree + the dated snapshot (writes _test, _novel, _novelty.json).
python experiments/multitree/make_forecast_testset.py \
    --current  <today's auspice json> --snapshot <snapshot auspice json> \
    --genes F --date <D> \
    --out experiments/multitree/data/examples_measles@<D>_test.txt
```

For a *dated* snapshot of a public build, `fetch_snapshots.py` pulls it from the Charon
`@YYYY-MM-DD` API; for our own trees, use the tree as it stood at `D`.

## Recipe — rebuild the SC2 (attach) series from scratch

```bash
cd antiGen-data-revised && make                 # -> ./antigen (g++, OpenMP, no CUDA)
python3 -I fetch_trees.py data/trees both       # dated ncov builds from the S3 bucket
# then prepare -> place per cutoff; see slurm/run_2021.sbatch for the exact 3-command chain.
```

Ivan's already-generated 2021 arm is copied under `antiGen-data-revised/output/sc2-2021/`
(gitignored): `train/examples_2021_cur.txt` (2,934 records, all pre-cutoff) and
`test/examples_all_cur.txt` (75,606 records: 3,902 pre-2021 training + 71,704 post-2021
testing).

## Models

Unchanged — this branch adds no models. Train/eval with the existing roster and the
joint-training code already on `main`; point them at a split with `--examples <path>`.
