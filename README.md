# antiGen

Welcome to antiGen, a model that predicts future mutations to viral proteins based on inferred phylogenetic trees. This repository is intended for (1) generating training data from phylogenetic trees, (2) training antiGen on said data, (3) querying substitution probabilities predicted by antiGen, (4) running the evaluations reported in the antiGen manuscript, and (5) reproducing the manuscript figures.

**Just want predictions?** [TUTORIAL.md](https://github.com/evo-design/antiGen/blob/main/TUTORIAL.md) scores every substitution of a
SARS-CoV-2 spike (BA.2.86 in the example) with the pretrained 2023-01-01 antiGen that
ships with the package, and lists the top-decile RBD mutations. CPU only, no download.

## Install

Install the modeling package and its command-line tools from PyPI (Python ≥ 3.10):

```bash
pip install antigen-model            # core: from-scratch transformer + BLOSUM62 / PSSM / gated (no model downloads)
pip install "antigen-model[esm]"     # + ESM-2 pretrained language model
pip install "antigen-model[dca]"     # + EVcouplings (needs external `plmc`)
```

Note that the distribution is named `antigen-model`, but the import name and CLIs keep the
`antiGen` spelling (`import antiGen`; `antiGen-train`/`-infer`/`-eval`). This is
all you need to train antiGen and score substitutions (Parts 2–3). The
training-data generator (Part 1, C++/CUDA) and the figure-reproduction code
(Part 5, R) are not part of the distribution; they can be obtained by cloning this repo.

```bash
git clone https://github.com/evo-design/antiGen.git
cd antiGen
pip install ".[esm]"                 # optional source install of the same package
```

## System requirements

The `antiGen` Python package is OS-independent (pure-Python wheel); the `antiGen-data`
generator is C++ and the figure code is R. No non-standard hardware is required. A GPU
accelerates training and data generation but is optional. The software has been tested on:

- **Operating system:** tested on Ubuntu 22.04.5 LTS (Linux kernel 5.15, x86_64) and macOS 26.3.1
  (Apple Silicon, arm64). The Python package and CLIs are platform-independent (pure-Python wheel)
  and require only Python ≥ 3.10.
- **Python:** 3.12 (requires ≥ 3.10), with PyTorch 2.7–2.13, NumPy 2.5, pandas 3.0,
  Biopython 1.87, and PyYAML 6.0.
- **Data generator (`antiGen-data`, Part 1):** g++ 11.4 (C++17) for the CPU build; the
  GPU build was compiled and run with CUDA 12.6 on an NVIDIA H100.
- **Figure reproduction (Part 5):** R 4.5.3 with the packages listed in that section.

## Quickstart (generate → train → infer, runs on CPU)

Runs the whole pipeline on a toy example in under a minute: **generate** training and
test examples by placing newer sequences on a tiny Nextstrain tree with the included
[`antiGen-data`](antiGen-data/) tool, then **train** a model and **score** substitutions.
No GPU is required.

```bash
# 1. build the data-generation tool (C++17 + OpenMP)
cd antiGen-data && make && cd ..

# 2. place the toy sequences on the toy tree; records first sampled after 2020-08-15
#    are the test set
antiGen-data/antigen prepare --tree antiGen-data/example/tree.json \
    --root-seq antiGen-data/example/root-sequence.json \
    --fasta antiGen-data/example/seqs.fasta --metadata antiGen-data/example/metadata.tsv \
    --date-cutoff none --no-clock --out toy_prepared.tsv
antiGen-data/antigen place --tree antiGen-data/example/tree.json \
    --root-seq antiGen-data/example/root-sequence.json --prepared toy_prepared.tsv \
    --post-cutoff 2020-08-15 --out toy_tree.tsv --out-examples toy_examples.txt

# 3. (optional) sanity-check the examples file
antiGen-validate-examples toy_examples.txt --split-date 2020-08-15 --alphabet ACDEFGHIKLMNPQRSTVWY

# 4. train a small from-scratch transformer on the pre-2020-08-15 records
antiGen-train --examples toy_examples.txt --split-date 2020-08-15 --model-type transformer \
    --d-model 32 --num-layers 1 --epochs 3 --batch-size 4 --output-dir ./checkpoints

# 5. score substitutions for the toy query sequences
antiGen-infer ./checkpoints/fitness_*.pt antiGen-data/example/seqs.fasta -o scores.csv
```

**Just want predictions from the paper's model?** [TUTORIAL.md](TUTORIAL.md) scores a
SARS-CoV-2 spike with the pretrained antiGen that ships with the package.

---

# 1. Generating training data

Training and test corpora come from **[`antiGen-data/`](antiGen-data/)**, a C++17/OpenMP
tool that reads a Nextstrain mutation-annotated tree, computes parsimony-consistent allele
sets, places newer aligned sequences on it by parsimony, and writes the examples antiGen
trains and is evaluated on, plus residue count matrices. Build it with `make` in that
directory; `./antigen` prints every command and option.

```bash
antigen prepare --tree <dataset.json> --root-seq <root-sequence.json> \
    --fasta <aligned.fasta> --metadata <strain_date.tsv> --date-cutoff <D> --out prepared.tsv
antigen place   --tree <dataset.json> --root-seq <root-sequence.json> --prepared prepared.tsv \
    --out tree.tsv --out-examples examples.txt [--post-cutoff D1,D2,...] [--variant BA.2]
```

**Inputs.** A Nextstrain v2 dataset JSON (nodes carry `node_attrs.num_date.value` and
`branch_attrs.mutations`, e.g. `"S": ["D614G"]`), its root-sequence sidecar
(`{"S": "MFVF…*"}`), an aligned FASTA of the sequences to place (protein, in the sidecar's
frame), and a `strain`⇥`date` metadata TSV.

**Train and test.** A *train corpus* for cutoff D is built from the latest tree before D
plus every sequence collected before D (`--date-cutoff D`). The *test tree* is built once,
from every sequence, with `--post-cutoff` listing all cutoffs. Each record then lists the
cutoffs it falls after, and the test set at D is the records carrying D. Every model,
statistical baselines included, is trained on the train corpus alone; the test tree's
pre-D records are never training data.

**The examples file.** One record per internal node: its sequence, the mutation from its
parent, its `POST_CUTOFF` cutoffs, variant flags, and the mutations to its children (the
training target; a mutation listed twice is a count of two). Positions are 0-based.
`antiGen-validate-examples <file> --split-date D` checks a file.

**The paper's corpora.** `antiGen-data/slurm/` holds the drivers. `run_series.sbatch`
builds a train corpus per quarterly cutoff (2020-04-01 … 2026-01-01), `run_all.sbatch`
builds the shared test tree (plus `counts_ba2/`, the BA.2-clade residue counts after each
cutoff), and `run_2021.sbatch` and `place_full.sbatch` build the 2021-01-01 arm. The input
trees and sidecars are fetched by `fetch_trees.py` and `fetch_snapshots.py`, and
[`antiGen-data/PROVENANCE.md`](antiGen-data/PROVENANCE.md) documents every file's source
and integrity checks. See [`antiGen-data/README.md`](antiGen-data/README.md).

---

# 2. Training antiGen

**Pointing antiGen at data.** Give the examples file with `--examples <path>`, plus
`--split-date D` when its records carry per-cutoff `POST_CUTOFF` labels. This is how every
model in the paper was trained, on the train corpus of its cutoff from Part 1, e.g. the
2023-01-01 base model:

```bash
antiGen-train --examples output/sc2-2023-01-01/train/examples_2023-01-01_cur.txt \
    --model-type transformer --d-model 128 --num-layers 2 \
    --epochs 32 --batch-size 64 --lr 1e-4 --variant BA.2 --ignore-dead-ends \
    --output-dir ./checkpoints
```

[`REPRODUCE.md`](REPRODUCE.md) lists the exact command for every checkpoint the figures
use: the cross-validated ten (`--val-mode random --save-best --split-seed i --seed 100+i`,
256 epochs), the gated model, the ESM-2 / EVE / Tranception fine-tunes, and the per-cutoff
series. Each checkpoint also stores its `train_args`, so a model can always be retrained
from the record it carries.

**Dataset shortcuts.** `--flu`, `--rsv-a`, `--denv2` and the SARS-CoV-2 default select a
dataset's variant, alphabet and evaluation region. With `--cutoff` instead of `--examples`,
they resolve `examples_cutoff_<cutoff>.txt` under `$FITNESS_DATA_ROOT` (or `--data-dir`),
a local path only; nothing is downloaded.

| Flag | Dataset | Variant | Eval region |
|------|---------|---------|-------------|
| *(default)* | SARS-CoV-2 spike | BA.2 (or `--variant`) | RBD (330–530, 0-indexed) |
| `--flu` | Influenza HA | J or K | full sequence |
| `--rsv-a` | RSV-A F | A.D.1.6 | full sequence |
| `--denv2` | Dengue 2 envelope | 2II_F.1.1.2_annotated | full sequence |

**Alphabet.** antiGen trains over the **20 standard amino acids** by default, with no gap
and no stop column. `--alphabet ACDEFGHIKLMNPQRSTVWY*-` trains in the older 22-token
frame, for data that still carries gaps and stops. A checkpoint stores its alphabet, so
`antiGen-infer` and `antiGen-eval` reload it without being told.

## Model architectures (`--model-type`)

| Model | `--model-type` | Extra install | Description |
|-------|----------------|---------------|-------------|
| Transformer | `transformer` | — (core) | Small transformer trained from scratch. **Default; no downloads.** |
| ESM-2 | `esm2` | `.[esm]` | ESM-2 protein LM (default `esm2_t12_35M_UR50D`), frozen/partial/full finetune |
| BLOSUM62 | `blosum62` | — (core) | Static BLOSUM62 substitution matrix (no training, eval only) |
| Codon | `codon` | — (core) | Single-nucleotide accessibility under the standard genetic code (no training, eval only) |
| PSSM | `pssm` | — (core) | Per-position `log(count+1)` from the root PSSM (no training, eval only) |
| Gated | `gated` | — (core) | Learnable per-position blend of two predictors (`--gate-partner-a/-b`, `--gate-variant`) |
| EVcouplings | `evcouplings` | `.[dca]` + `plmc` | DCA Potts model fit from a pandemic MSA via `plmc` (pseudo-likelihood; eval only) |
| EVE | `eve` | — (core) | EVE VAE (frozen or finetuned) trained on a pandemic MSA |
| Tranception | `tranception` | — (core) | Tranception autoregressive scorer (frozen or finetuned) |

For the pretrained ESM-2 backbone, we provide three finetuning strategies: **frozen**
(`--freeze-backbone`, head only), **partial** (`--freeze-backbone --finetune-layers N`), and
**full** (no freeze flags). MSA models (`evcouplings`/`eve`/`tranception`) additionally need
`--msa` / `--metadata`; their date cutoff is derived from the dataset MRCA + `--cutoff` (or
`--date-cutoff`).

```bash
# from-scratch transformer (the default)
antiGen-train --model-type transformer --d-model 128 --num-layers 2 \
    --epochs 64 --batch-size 64 --lr 1e-4 --flu --output-dir ./checkpoints

# ESM-2, partial finetune of the last 4 layers
antiGen-train --model-type esm2 --freeze-backbone --finetune-layers 4 --lr 1e-4 --flu

# codon-accessibility baseline (no training; --codon-kappa up-weights transitions)
antiGen-train --model-type codon --codon-kappa 2.5 --flu

# gated blend of a from-scratch transformer and the root PSSM
antiGen-train --model-type gated --gate-partner-a fitness_predictor --gate-partner-b pssm \
    --gate-variant static --flu
```

Checkpoints are written to `--output-dir` as `fitness_<run_prefix>.pt`, alongside a
`manifest.json` recording the full recipe. They are self-describing (they store the complete
`ModelConfig`), so `antiGen-infer` / `antiGen-eval` reload them standalone.

---

# 3. Querying substitution probabilities

`antiGen-infer <checkpoint> <queries.fasta>` runs a forward pass on one or more query sequences
and writes the model's probability distribution over substitutions relative to each query.

```bash
antiGen-infer ./checkpoints/fitness_*.pt my_queries.fasta -o scores.csv
```

Default **long** output reports one row per candidate substitution:

```
query_id,position,ref_aa,alt_aa,prob
toy_root,0,M,A,0.00148030
toy_root,0,M,C,0.00164760
...
```

`prob` is the inferred probability of each possible amino acid substitution relative to the
query. Use `--format wide` for a position × amino-acid matrix. `--drop-gap-stop` excludes the
`-`/`*` columns, and is a no-op on a model trained over the 20 standard residues.

**Length / alignment.** antiGen operates on a **fixed, aligned coordinate frame**: provide
query sequences aligned to the training reference (same length and alignment).

---

# 4. Running the manuscript evaluations

`antiGen-eval` scores every checkpoint in a directory, plus the statistical baselines and
the fixed external maps in `data/baselines/`, on the post-cutoff records of a test tree:

```bash
antiGen-eval --dataset sc2 --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples $CORPORA/sc2-2023-01-01/train/examples_2023-01-01_cur.txt \
    --test-examples  $CORPORA/sc2-test/examples_all_cur.txt --split-date 2023-01-01 \
    --root-counts    $CORPORA/sc2-2023-01-01/train/counts_2023-01-01_cur.tsv \
    --checkpoint-dir ./checkpoints --output-dir ./evals
```

It reports **next-mutation recall** (how often the mutations that occur next fall in a
model's top-q predictions) per mutation, and `antiGen/list_macro.py` reports it per cell
(each (site, residue) counted once) over a mutation list or every cell of a region. The
statistical baselines are fitted on the train corpus only, the data the checkpoints saw.
Ties at a top-q boundary are counted fractionally (the expected value under random
tie-breaking).

The paper's evaluations, one named step per figure family, are in
[`reproduce/eval.sh`](reproduce/eval.sh) (Slurm; paths and partitions in
`reproduce/config.sh`):

```bash
bash reproduce/stage_checkpoints.sh          # check the checkpoint directories the evals read
bash reproduce/eval.sh                       # list the steps
bash reproduce/eval.sh rbd_2023              # submit one; outputs under $OUT/rbd_2023-01-01/
bash reproduce/stage.sh                      # outputs -> antiGen-figures/data/
```

[`REPRODUCE.md`](REPRODUCE.md) maps every figure to its staged data, the step that
produces it, the checkpoints it reads and the published output it reproduces.
`python reproduce/train_commands.py` prints the `antiGen-train` command behind every
checkpoint. The training recipes are in [`examples/slurm/`](examples/slurm/).

---

# 5. Reproducing the paper figures

[`antiGen-figures/`](antiGen-figures/) is R code that draws the staged evaluation outputs
into the manuscript figures; it trains and evaluates nothing, and everything it reads is
committed. From that directory:

```bash
bash render_all.sh                    # every current figure -> figs/*.png, figs_vector/*.pdf
bash render_all.sh supp_cd1_series    # one figure
```

It needs R (4.5.3 was used) with cowplot, dplyr, forcats, ggpattern, ggplot2, ggtext,
magick, reshape2, scales, tibble and tidyr.

---

## Citing

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
