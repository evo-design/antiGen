# antiGen

Welcome to antiGen, a model that predicts future mutations to viral proteins based on inferred phylogenetic trees. This repository is intended for (1) generating training data from phylogenetic trees, (2) training antiGen on said data, (3) querying substitution probabilities predicted by antiGen, (4) running the evaluations reported in the antiGen manuscript, and (5) reproducing the manuscript figures.

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

Runs the whole pipeline on a toy example: **generate** training data from a tiny
Nextstrain-format mutation-annotated tree and MSA using the included `antiGen-data` tool, then **train** a model and **score** substitutions.
No GPU is required for this example.

```bash
# 1. build the data-generation tool (make auto-selects nvcc if present, else g++)
cd antiGen-data && make && cd ..

# 2. generate training data from the toy Nextstrain JSON + MSA (~ms; GPU if present, else CPU)
antiGen-data/antiGen-data antiGen-data/example/out 0.6 toy \
    antiGen-data/example/ref.json antiGen-data/example/tree.json \
    antiGen-data/example/metadata.tsv antiGen-data/example/seqs.fasta

# 3. (optional) sanity-check the generated examples file
antiGen-validate-examples antiGen-data/example/out/examples_cutoff_0.6.txt

# 4. train a small from-scratch transformer on it
antiGen-train --model-type transformer \
    --examples antiGen-data/example/out/examples_cutoff_0.6.txt \
    --d-model 32 --num-layers 1 --epochs 3 --batch-size 4 --output-dir ./checkpoints

# 5. score substitutions for the toy query sequences
antiGen-infer ./checkpoints/fitness_*.pt antiGen-data/example/seqs.fasta -o scores.csv
```

---

# 1. Generating training data

antiGen training data is produced by the included **[`antiGen-data/`](antiGen-data/)** tool, a
C++/CUDA application that grows a mutation-annotated phylogeny by parsimonious attachment and returns training/testing examples, as well as position-specific scoring matrices
(PSSMs) for the training window and at various cutoff times in the testing window. To build it, cd into the antiGen-data sub-directory and run `make` (needs a C++17 compiler; `nvcc` is optional, for GPU support).

The raw inputs (reference sequences, mutation-annotated trees, metadata, and MSAs) for the four datasets analyzed in the paper are archived on Zenodo at [doi:10.5281/zenodo.21842015](https://doi.org/10.5281/zenodo.21842015).

Run it as:

```bash
antiGen-data <output_dir> <cutoff> <dataset_type> <ref_json> <tree_json> <metadata_tsv> <alignment_files...> [max_seqs] [--cpu]
```

- `output_dir` — where outputs are written (created if absent).
- `cutoff` — train/test split, in **years after the tree's root date**: examples dated before `root + cutoff` are labeled `TRAINING`, later ones `TESTING`.
- `dataset_type` — one of `sc2`, `h3n2`, `rsv_a`, `rsv_b`, `denv2`, or `toy`; selects the alphabet, segment layout, and evaluation region.
- `ref_json`, `tree_json`, `metadata_tsv` — the reference sequence, guide phylogeny, and per-strain dates (formats under **Input files** below).
- `alignment_files` — one or more MSA FASTAs of the sequences to attach, one per segment (1 for most datasets; 3 for H3N2 HA).
- `max_seqs` *(optional)* — cap on how many sequences to attach.
- `--cpu` *(optional)* — force the CPU path; a GPU is used automatically when present.

**Input files.** The three inputs follow the Nextstrain convention; see
[`antiGen-data/example/`](antiGen-data/example/) for a complete, runnable set.

- `ref_json` — root sequence per segment, e.g. `{"S": "MFVF…"}`.
- `tree_json` — a Nextstrain tree whose nodes carry `node_attrs.num_date.value` (decimal year) and `branch_attrs.mutations`, one list per segment of `<ref><pos><alt>` strings, 1-indexed (e.g. `"S": ["A222V", "D614G"]`).
- `metadata_tsv` — `strain`⇥`date` (ISO `YYYY-MM-DD`) for each sequence in the alignment(s).

Each run writes to `<output_dir>/`, formatted as follows:

```
output_<dataset>/
  examples_cutoff_<value>.txt     # training/testing examples (one entry per inferred ancestral genotype; see below)
  pssms/root.csv                  # PSSM of all sequences in the training window
  pssms/<variant>/time_*.csv      # PSSM of all sequences in the testing window up through the cutoff time specified in the file name
```

Each example within `examples_cutoff_*.txt` holds a sequence, its `TRAINING`/`TESTING` designation, whether it is the root example of or is descended from a supplied variant of interest, and the substitutions observed in that sequence's children (the training target).
Each line of the `MUTATIONS_TO_CHILDREN:` block is `<pos>,<alt>,<descendants>`, where `descendants` counts the observed sequences descended from that substitution — everything at or below the child it leads to. The count stays on the example's own side of the split: sequences from before the cutoff for a `TRAINING` example, from after it for a `TESTING` one. Summing the counts on the global root therefore gives the number of training sequences, less those whose genotype is the root's own.
`antiGen-validate-examples <file> [--variant V]` checks a file is well-formed (single consistent
sequence length, alphabet `ACDEFGHIKLMNPQRSTVWY*-`, in-range positions, ≥1 training example). The
Quickstart above generates a miniature example of this format from the toy inputs in
[`antiGen-data/example/`](antiGen-data/example/).

---

# 2. Training antiGen

**Pointing antiGen at data.** Give an explicit file with `--examples <path>` (works for any
dataset with no code edits), or use a built-in dataset shortcut which resolves
`examples_cutoff_<cutoff>.txt` under `$FITNESS_DATA_ROOT` (or an explicit `--data-dir`). These
shortcuts only build a local path — the data is never downloaded, so you must have generated it
first (Part 1) and placed it under `$FITNESS_DATA_ROOT`:

| Flag | Dataset | Variant | `--cutoff` | Eval region |
|------|---------|---------|------------|-------------|
| *(default)* | SARS-CoV-2 spike | BA.2 (or `--variant`) | 1, 4, or 5 | RBD (330–530, 0-indexed) |
| `--flu` | Influenza HA | J or K | 12 | full sequence |
| `--rsv-a` | RSV-A F | A.D.1.6 | 74 | full sequence |
| `--denv2` | Dengue 2 envelope | 2II_F.1.1.2_annotated | 191 | full sequence |

```bash
# built-in dataset: resolve the file from $FITNESS_DATA_ROOT via a shortcut
export FITNESS_DATA_ROOT=/path/to/antiGen-data      # holds output1/, output_flu_full_cutoff12/, ...
antiGen-train --flu --cutoff 12 --model-type transformer --output-dir ./checkpoints

# your own data: point --examples straight at a file you generated in Part 1
# (no $FITNESS_DATA_ROOT, no shortcut — works for any protein/pathogen). seq_len is read from
# the file (alphabet is the standard 22 AAs); add --variant <label> if it has a variant of interest.
antiGen-train --examples ./mydata/examples_cutoff_5.txt \
    --model-type transformer --d-model 128 --num-layers 2 \
    --epochs 64 --batch-size 64 --lr 1e-4 --output-dir ./checkpoints
```

## Model architectures (`--model-type`)

| Model | `--model-type` | Extra install | Description |
|-------|----------------|---------------|-------------|
| Transformer | `transformer` | — (core) | Small transformer trained from scratch. **Default; no downloads.** |
| ESM-2 | `esm2` | `.[esm]` | ESM-2 protein LM (default `esm2_t12_35M_UR50D`), frozen/partial/full finetune |
| BLOSUM62 | `blosum62` | — (core) | Static BLOSUM62 substitution matrix (no training, eval only) |
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
query (including premature stop codons `*` and gaps `-`, i.e. point indels). Use `--format wide`
for a position × amino-acid matrix, and `--drop-gap-stop` to exclude the `-`/`*` columns.

**Length / alignment.** antiGen operates on a **fixed, aligned coordinate frame**: provide
query sequences aligned to the training reference (same length and alignment).

---

# 4. Running the manuscript evaluations

`antiGen-eval` scans a directory of checkpoints, groups them by model, and writes unified
per-variant comparison CSVs against a set of fixed baselines.

```bash
antiGen-eval --dataset sc2 --data-dir $FITNESS_DATA_ROOT/output1 \
    --checkpoint-dir ./checkpoints --output-dir ./evals_sc2
```

It produces two metric families per variant: **quantile hit-rate** (how often observed mutations
fall in the model's top-q% predictions) and **variant-emergence** (across top-q quantiles
q = 0.1 … 1.0). These are termed next-mutation recall and PSSM precision in the paper, respectively. Baselines come from `data/baselines*/` plus two built from the root PSSM
(`pssm_root` and the position-only `pssm_root_position`). Ties at a top-q boundary are broken
**fractionally** (the expected value under random tie-breaking).

The SLURM scripts used for the manuscript runs are in [`examples/slurm/`](examples/slurm/); they
embed lab-specific paths/partitions and are meant to be copied and edited for your environment.

---

# 5. Reproducing the paper figures

[`antiGen-figures/`](antiGen-figures/) is R code that plots the evaluation outputs into the
manuscript figures (it trains and evaluates nothing). From that directory, `Rscript run_all.R`
regenerates everything into `figs/` (PNG) and `figs_vector/` (PDF), or `Rscript figures/<script>.R`
runs a single figure; the inputs it reads live in `antiGen-figures/data/` and sibling directories.
Requires R with ggplot2, cowplot, reshape2, dplyr, tidyr, tibble, forcats, magick, ggtext,
jsonlite, and scales.

---

## Citing

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
