# antiGen

Welcome to antiGen, a model that predicts future mutations to viral proteins based on inferred phylogenetic trees. This repository is intended for (1) generating training data from phylogenetic trees, (2) training antiGen on said data, (3) querying substitution probabilities predicted by antiGen, (4) running the evaluations reported in the antiGen manuscript, and (5) reproducing the manuscript figures.

**The paper.** Specht, Park, Vermani, Chithrananda, Driscoll, Brixi, Palacios and Hie,
*Forecasting viral evolution from phylogenetic trees*, bioRxiv
[10.64898/2026.08.27.741102](https://www.biorxiv.org/content/10.64898/2026.08.27.741102v2).

**Where to start.** To score mutations relative to BA.2.86 (or your own spike) with the
pretrained model, follow the [tutorial](https://github.com/evo-design/antiGen/blob/main/TUTORIAL.md).
To generate data and train and evaluate your own models, continue with this README: the
Quickstart, then Parts 1–5. To regenerate the paper's evaluations and figures, see
[REPRODUCE.md](https://github.com/evo-design/antiGen/blob/main/REPRODUCE.md).

## Install

Install the modeling package and its command-line tools from PyPI (Python ≥ 3.10):

```bash
pip install antigen-model            # core: from-scratch transformer + BLOSUM62 / PFM / gated (no model downloads)
pip install "antigen-model[esm]"     # + ESM-2 pretrained language model
pip install "antigen-model[dca]"     # + EVcouplings (needs external `plmc`)
```

Note that the distribution is named `antigen-model`, but the import name and CLIs keep the
`antiGen` spelling (`import antiGen`; `antiGen-train`/`-infer`/`-eval`/`-top-decile`/
`-validate-examples`). This is all you need to train antiGen and score substitutions
(Parts 2–3). The training-data generator (Part 1, C++/CUDA) and the figure-reproduction code
(Part 5, R) are not part of the distribution; they can be obtained by cloning this repo.

```bash
git clone https://github.com/evo-design/antiGen.git
cd antiGen
pip install ".[esm]"                 # optional source install of the same package
```

## Scoring a spike with the pretrained model

The package ships a pretrained SARS-CoV-2 spike model trained on the phylogeny of
sequences dated **before 2023-01-01**. `antiGen-top-decile` ranks every candidate
substitution of a query in a coordinate range — here the receptor-binding domain,
201 × 19 = 3,819 cells — and writes the top decile, the pool and cutoff of the paper's
all-RBD panels. It runs on a CPU in a few seconds and downloads nothing beyond the
package.

The query below is `tutorial/BA.2.86_spike.fasta` in a clone of this repository; without
one, download
[BA.2.86_spike.fasta](https://github.com/evo-design/antiGen/releases/latest/download/BA.2.86_spike.fasta).

```bash
pip install antigen-model
antiGen-top-decile tutorial/BA.2.86_spike.fasta --start 331 --end 531 -o top_rbd.csv
```

```text
model: antiGen (CV x10), SARS-CoV-2 spike, 2023-01-01 cutoff
BA.2.86: 3819 candidates in 331-531; top 0.1: 382 listed
```

```
query_id,rank,mutation,position,ref_aa,alt_aa,score,prob,percentile
BA.2.86,1,S446G,446,S,G,-4.289715,7.121325e-02,100.00
BA.2.86,2,K440N,440,K,N,-4.365720,6.600123e-02,99.97
BA.2.86,3,N417K,417,N,K,-4.577544,5.340207e-02,99.95
...
```

BA.2.86 was first sampled months after that cutoff. R346T, gained on the way to
JN.1.11.1, ranks 9th here; L455S, the substitution that took BA.2.86 to JN.1, ranks
114th — both inside the top decile, from a model that saw no post-2022 sequence.
`antiGen-infer --pretrained sc2-2023-01-01 <query.fasta>` writes the full distribution
instead of a ranking. The [tutorial](https://github.com/evo-design/antiGen/blob/main/TUTORIAL.md)
does the same from Python, and runs in
[Colab](https://colab.research.google.com/github/evo-design/antiGen/blob/main/tutorial/tutorial.ipynb).

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

Runs the whole pipeline on a toy example in under a minute: **generate** a train corpus
and a test tree with the included [`antiGen-data`](antiGen-data/) tool, then **train** a
model and **score** substitutions. No GPU is required.

The train and test data come from **two separate `antiGen-data` runs**, exactly as in the
paper:

* the **train corpus**: an *earlier* tree (a build from before the cutoff) plus only the
  sequences collected before the cutoff;
* the **test tree**: a *later* tree plus the sequences collected up to a later date, each
  record labelled with the cutoffs it falls after.

Never train on any record of the test tree, including its pre-cutoff records: the later
tree and the sequences placed on it carry information from after the cutoff.
`antiGen-train` refuses a file that holds any post-cutoff record.

```bash
# 1. build the data-generation tool (C++17 + OpenMP)
cd antiGen-data && make && cd ..
E=antiGen-data/example

# 2. TRAIN corpus: the earlier tree (a 2020-06-01 build) plus the sequences dated
#    before the 2020-08-15 cutoff
antiGen-data/antigen prepare --tree $E/tree.json --root-seq $E/root-sequence.json \
    --fasta $E/seqs.fasta --metadata $E/metadata.tsv \
    --date-cutoff 2020-08-15 --no-clock --out train_prepared.tsv
antiGen-data/antigen place --tree $E/tree.json --root-seq $E/root-sequence.json \
    --prepared train_prepared.tsv --out train_tree.tsv --out-examples train_examples.txt

# 3. TEST tree: the later tree (a 2020-11-15 build) plus every sequence dated before
#    2020-11-15; records first sampled on or after 2020-08-15 are the test set
antiGen-data/antigen prepare --tree $E/tree_late.json --root-seq $E/root-sequence.json \
    --fasta $E/seqs.fasta --metadata $E/metadata.tsv \
    --date-cutoff 2020-11-15 --no-clock --out test_prepared.tsv
antiGen-data/antigen place --tree $E/tree_late.json --root-seq $E/root-sequence.json \
    --prepared test_prepared.tsv --post-cutoff 2020-08-15 \
    --out test_tree.tsv --out-examples test_examples.txt

# 4. (optional) sanity-check both files
antiGen-validate-examples train_examples.txt --alphabet ACDEFGHIKLMNPQRSTVWY
antiGen-validate-examples test_examples.txt --split-date 2020-08-15 --alphabet ACDEFGHIKLMNPQRSTVWY

# 5. train a small from-scratch transformer on the train corpus (and only on it)
antiGen-train --examples train_examples.txt --model-type transformer \
    --d-model 32 --num-layers 1 --epochs 3 --batch-size 4 --output-dir ./checkpoints

# 6. score substitutions for the toy query sequences
antiGen-infer ./checkpoints/fitness_*.pt $E/seqs.fasta -o scores.csv
```

The train corpus holds 5 records, every one of them training data. The test tree holds 7,
of which the 2 on or after 2020-08-15 are the test set; its other 5 are never used. The
evaluations in Part 4 score a model on the test records of the test tree.

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

**Alphabet.** `--alphabet` picks how stops and gaps are read. Give the same value to every
command of a run (`place` refuses a queue `prepare`d under another):

* `--alphabet 20` (**the default**, and what the paper's corpora use): the 20 standard
  amino acids. A stop (`*`) or a gap (`-`) is treated like `X`, as missing or ambiguous
  data at that site, never as a residue of its own.
* `--alphabet 22`: stop and gap are states like the other 20, so the alphabet has 22
  tokens and only `X` is missing. A run of `-` longer than `prepare --max-gap-run`
  (default 10), or one touching either end of the gene, is still read as missing
  coverage rather than a deletion.

Train on the same alphabet the corpus was built with (Part 2, *Alphabet*).

**Train and test come from two separate runs.** For a cutoff D:

* the **train corpus** is an *earlier* tree, the last build before D, plus only the
  sequences collected before D (`prepare --date-cutoff D`; `place` without
  `--post-cutoff`). Every one of its records is training data.
* the **test tree** is a *later* tree plus the sequences collected up to a later date,
  placed with `--post-cutoff` listing the cutoffs. Each record lists the cutoffs it falls
  after, and the test set at D is the records that list D.

Every model, the statistical baselines included, is trained or fitted on the train
corpus alone. No record of the test tree is training data, including its pre-cutoff
records: the later tree and the sequences placed on it carry information from after D.
`antiGen-train` and the evaluations refuse any training file that holds a post-cutoff
record. The Quickstart above is this procedure in miniature.

**The examples file.** One record per internal node: its sequence, the mutation from its
parent, its `POST_CUTOFF` cutoffs, variant flags, and the mutations to its children (the
training target; a mutation listed twice is a count of two). Positions are 0-based.
`antiGen-validate-examples <file>` checks a file (add `--split-date D` for a test tree).

**The paper's corpora.** `antiGen-data/slurm/` holds the drivers. `run_series.sbatch`
builds a train corpus per quarterly cutoff (2020-04-01 … 2026-01-01), `run_all.sbatch`
builds the shared test tree (plus `counts_ba2/`, the BA.2-clade residue counts after each
cutoff), and `run_2021.sbatch` and `place_full.sbatch` build the 2021-01-01 arm. The input
trees and sidecars are fetched by `fetch_trees.py` and `fetch_snapshots.py`, and
[`antiGen-data/PROVENANCE.md`](antiGen-data/PROVENANCE.md) documents every file's source
and integrity checks. See [`antiGen-data/README.md`](antiGen-data/README.md).

---

# 2. Training antiGen

**Pointing antiGen at data.** `--examples` names the train corpus of the cutoff (Part 1),
and is required. Every model in the paper was trained this way; for example the
2023-01-01 base model:

```bash
antiGen-train --examples $CORPORA/sc2-2023-01-01/train/examples_2023-01-01_cur.txt \
    --model-type transformer --d-model 128 --num-layers 2 --epochs 32 --batch-size 64 \
    --variant BA.2 --ignore-dead-ends --output-dir ./checkpoints
```

`python reproduce/train_commands.py` prints the exact command behind every checkpoint the
figures use, read back from the `train_args` each checkpoint stores
([`REPRODUCE.md`](REPRODUCE.md)).

**Validation.** By default every record of the train corpus trains, for a fixed number of
epochs. `--val-mode random --val-frac 0.1 --save-best` instead holds out a random tenth
of the train corpus and keeps the epoch that scores best on it. The paper's antiGen
(CV ×10) is ten such models (`--split-seed 0…9`, `--seed 100…109`, 256 epochs), combined
as the mean of their log-probabilities. Validation never reads the test tree.

**Alphabet.** antiGen trains over the **20 standard amino acids** by default, with no gap
and no stop column, matching antiGen-data's default `--alphabet 20` (Part 1).
`--alphabet ACDEFGHIKLMNPQRSTVWY*-` trains in the 22-token frame, for a corpus built with
`--alphabet 22`. A checkpoint stores its alphabet, so
`antiGen-infer` and `antiGen-eval` reload it without being told.

## Model architectures (`--model-type`)

| Model | `--model-type` | Extra install | Description |
|-------|----------------|---------------|-------------|
| Transformer | `transformer` | — (core) | Small transformer trained from scratch. **Default; no downloads.** |
| ESM-2 | `esm2` | `.[esm]` | ESM-2 protein LM (default `esm2_t12_35M_UR50D`), frozen/partial/full finetune |
| BLOSUM62 | `blosum62` | — (core) | Static BLOSUM62 substitution matrix (no training, eval only) |
| Codon | `codon` | — (core) | Single-nucleotide accessibility under the standard genetic code (no training, eval only) |
| Codon-aware BLOSUM62 | `codon_blosum` | — (core) | BLOSUM62 target frequencies restricted to codon-adjacent substitutions (`--codon-blosum-kappa`; fixed constants, no training) |
| Codon × BLOSUM62 mixture | `substitution` | — (core) | The same two terms as a log-linear model with 25 fitted scalars (κ, the two weights, a per-destination background); position-blind, so one fit covers any number of phylogenies |
| PFM | `pfm` | — (core) | Position frequency matrix: per-position `log(count+1)` of the train corpus's root counts (no training, eval only). `pssm`, the model's former name, is still accepted, and is the spelling kept inside checkpoints and evaluation output. |
| MCM | — (`antiGen-eval --mcm-baseline`) | — (core) | Mutation count matrix: per-cell `log N[i,a]`, the number of `i`→`a` mutations in the train corpus (fitted at eval time) |
| S×S | — (`antiGen-eval --counted-baselines`) | — (core) | Site × substitution: `s_i · M[wt_i, a]`, the site's share of train-corpus mutations `s_i` times the counted substitution matrix `M` (fitted at eval time) |
| S×S–MCM Interp. | — (`antiGen-eval --counted-baselines`) | — (core) | `(N[i,a] + κ · s_i · M[wt_i, a]) / (N_total + κ)` with κ = N_total: the MCM counts blended with S×S (fitted at eval time) |
| S×M | — (`antiGen-eval --multitree-baselines`) | — (core) | Site × mixture: `s_i · B[wt_i, a]`, the same `s_i` times the fitted codon × BLOSUM62 mixture `B` in place of `M` (fitted at eval time) |
| S×M–MCM Interp. | — (`antiGen-eval --multitree-baselines`) | — (core) | `(N[i,a] + κ · s_i · B[wt_i, a]) / (N_total + κ)` with κ = N_total: the MCM counts blended with S×M (fitted at eval time) |
| Gated | `gated` | — (core) | Learnable per-position blend of two predictors (`--gate-partner-a/-b`, `--gate-variant`) |
| EVcouplings | `evcouplings` | `.[dca]` + `plmc` | DCA Potts model fit from an MSA via `plmc` (pseudo-likelihood; eval only) |
| EVE | `eve` | — (core) | EVE VAE (frozen or finetuned) trained on an MSA |
| Tranception | `tranception` | — (core) | Tranception autoregressive scorer (frozen or finetuned) |

For the pretrained ESM-2 backbone there are three finetuning strategies: **frozen**
(`--freeze-backbone`, head only), **partial** (`--freeze-backbone --finetune-layers N`), and
**full** (no freeze flags). The MSA models (`evcouplings`/`eve`/`tranception`) also take
`--msa`: the alignment of the train corpus's own sequences, all dated before the cutoff,
which antiGen-data writes alongside it (`antiGen-data/slurm/msa_2023-01-01.sbatch`). The
paper's commands, all on the 2023-01-01 train corpus (`TRAIN=$CORPORA/sc2-2023-01-01/train`):

```bash
# ESM-2 35M, partial finetune of the last 4 layers
antiGen-train --examples $TRAIN/examples_2023-01-01_cur.txt --model-type esm2 \
    --freeze-backbone --finetune-layers 4 --epochs 32 --variant BA.2 --ignore-dead-ends

# EVE, frozen, on the train corpus's alignment
antiGen-train --examples $TRAIN/examples_2023-01-01_cur.txt --model-type eve \
    --freeze-backbone --epochs 32 --msa $TRAIN/msa_2023-01-01_cur.fasta --max-seqs 351410 \
    --variant BA.2 --ignore-dead-ends

# gated blend of a from-scratch transformer and the root PFM of the train corpus
# (pssm_root.csv from reproduce/tools/counts_to_pssm_csv.py on counts_2023-01-01_cur.tsv)
antiGen-train --examples $TRAIN/examples_2023-01-01_cur.txt --model-type gated \
    --gate-partner-b pfm --pfm-path pssm_root.csv --d-model 128 --num-layers 2 \
    --epochs 32 --batch-size 64 --variant BA.2 --ignore-dead-ends
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
query_id,position,ref_aa,alt_aa,prob,position_1based,mutation
new1,0,M,A,1.907314e-03,1,M1A
new1,0,M,C,3.051861e-03,1,M1C
...
```

`prob` is the inferred probability of each possible amino acid substitution relative to the
query. `position` is 0-based; `position_1based` and `mutation` name the same substitution in
the usual 1-based notation (e.g. `L455S`). Use `--format wide` for a position × amino-acid
matrix. `--drop-gap-stop` excludes the `-`/`*` columns, and is a no-op on a model trained
over the 20 standard residues.

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
checkpoint. The training recipes are in [`reproduce/slurm/`](reproduce/slurm/).

---

# 5. Reproducing the paper figures

[`antiGen-figures/`](antiGen-figures/) is R code that draws the staged evaluation outputs
into the manuscript figures; it trains and evaluates nothing, and everything it reads is
committed. From that directory:

```bash
bash render_all.sh                    # every current figure -> figs/*.png, figs_vector/*.pdf
bash render_all.sh supp_fullspike     # one figure (names: see the list in the script)
```

It needs R (4.5.3 was used) with cowplot, dplyr, forcats, ggpattern, ggplot2, ggtext,
gtable, magick, png, reshape2, scales, tibble and tidyr.

---

## Citing

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
