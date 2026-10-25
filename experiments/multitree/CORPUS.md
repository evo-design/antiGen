# The multi-virus no-attach training corpus

One unified training corpus of guide-tree-only (no-attach) examples across every virus/protein
we have collected and trained on. All splits are **TRAINING** (no temporal cutoff); hold out a
validation slice at train time with `--val-fraction`.

- **Manifest:** `experiments/multitree/data/trees.json` — one entry per dataset with
  `dataset`, `genes`, `path`, `seq_len`, `examples`, `source` (`catalogue`/`local`),
  `attach_mode: none`. Tracked in git.
- **Example files:** `experiments/multitree/data/examples_<short>_nocutoff.txt` — large,
  **gitignored** (regenerable byte-for-byte), local only.
- **Total:** 28 datasets, 35,032 examples, 21 distinct sequence lengths.

## Sources

- **catalogue (23):** regenerated from Ivan's downloaded Nextstrain trees at
  `/home/ivan.specht/antiGen/nextstrain/` via `generate_all.py` — flu (h3n2/h1n1pdm/vic/yam
  HA + NA, h5n1, h5nx), dengue denv1–4, ncov S, measles (H), mumps, evd68 (VP1), rabies,
  mpox, WNV, zika, rsv_a F, rsv_b F. Yellow fever is auto-skipped (22 examples, < 50 floor).
- **local (5):** proteins the catalogue lacks, from our own `data/*/` trees — `chikv_e1` (E1),
  `chikv_e2` (E2), `hmpv_f` (F), `measles_f` (F), `norovirus_vp1` (VP1).

### Not included, and why
- **4 local trees overlap the catalogue by protein** and were left out to avoid double-counting
  the same (virus, protein): local ev_d68 VP1 (≈ catalogue `evd68`), measles H (≈ `measles`),
  wnv env (≈ `wnv`), rsv_b F (≈ `rsv_b`). To use the local build instead, regenerate that one
  with `make_tree_examples.py` and swap the manifest path.
- **rsv-a** has no local auspice tree (only FASTAs); the catalogue `rsv_a` (Nextstrain F) covers it.
- **sc2/denv/flu as *attach* builds** live on the other arm (`antiGen-data-revised/`, SC2) and in
  the registry — not mixed into this no-attach corpus (see STANDARDIZATION.md).
- **ebola-bdbv** (`data/new_heldout_virus/`) is the held-out *test* virus — deliberately excluded
  from training.

## Regenerate

```bash
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python   # deps live in the ag_env conda env

# catalogue (23) from Ivan's trees
$PY experiments/multitree/generate_all.py \
    --trees /home/ivan.specht/antiGen/nextstrain \
    --out experiments/multitree/data --manifest experiments/multitree/data/trees.json

# the 5 local-unique proteins
$PY experiments/multitree/make_tree_examples.py --tree data/chikv/chikv_e1_tree.json  --genes E1  --out experiments/multitree/data/examples_chikv_e1_nocutoff.txt
$PY experiments/multitree/make_tree_examples.py --tree data/chikv/chikv_tree.json     --genes E2  --out experiments/multitree/data/examples_chikv_e2_nocutoff.txt
$PY experiments/multitree/make_tree_examples.py --tree data/hmpv/hmpv_tree.json       --genes F   --out experiments/multitree/data/examples_hmpv_f_nocutoff.txt
$PY experiments/multitree/make_tree_examples.py --tree data/measles/measles_f_tree.json --genes F --out experiments/multitree/data/examples_measles_f_nocutoff.txt
$PY experiments/multitree/make_tree_examples.py --tree data/norovirus/norovirus_gii_tree.json --genes VP1 --out experiments/multitree/data/examples_norovirus_vp1_nocutoff.txt
```

(The local entries must then be re-added to `trees.json`; `generate_all.py` only writes the
catalogue ones.)

## Train on the whole corpus (your existing joint trainer)

Mixed sequence lengths mean only length-agnostic models qualify (`esm2`, `blosum62`); every batch
stays single-dataset. Feed one `--dataset-spec NAME:0:PATH` per dataset (cutoff `0` is unused when
an explicit `.txt` path is given):

```bash
SPECS=$($PY - <<'PY'
import json
m=json.load(open("experiments/multitree/data/trees.json"))
print(" ".join(f"--dataset-spec {k}:0:{m[k]['path']}" for k in sorted(m)))
PY
)
antiGen-train $SPECS --model-type esm2 --freeze-backbone --finetune-layers 4 \
    --val-fraction 0.1 --epochs <N> --run-tag multivirus_noattach \
    --checkpoint-dir <dir> --output-dir <dir>
```
