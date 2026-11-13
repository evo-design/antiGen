# antiGen-data: training and test corpora from a phylogeny

`antigen` turns a Nextstrain phylogeny plus a queue of newer aligned sequences into the
examples antiGen trains and is evaluated on. It:

1. reads a **Nextstrain v2 dataset JSON** (the mutation-annotated tree) and its
   **root-sequence sidecar**;
2. computes parsimony-consistent **allele sets** at every (node, site) (Hartigan);
3. **places** newer sequences on the tree by parsimony (date-ordered, deterministic for a
   given `--seed`, multithreaded);
4. writes the **examples file**: one record per internal node, holding its sequence and the
   mutations to its children (the training target). Each record also lists **which
   train/test cutoffs it falls after**, so one test tree serves every cutoff.

It is C++17 with OpenMP and needs no GPU.

```bash
cd antiGen-data && make          # builds ./antigen
./antigen                        # usage: commands and every option
```

## Commands

| command | does |
|---|---|
| `tree` | read the tree and report it |
| `sets` | + allele sets; `--out-examples` writes examples from the tree alone |
| `prepare` | select the sequences eligible for placement (date cutoff, coverage, molecular-clock filter) from an aligned FASTA + metadata |
| `place` | append `prepare`'s sequences to the tree; write the tree, examples, count matrices |

The options that matter most:

* `--tree`, `--root-seq`: the dataset JSON and its sidecar (`{"S": "MFVF…*"}`; each gene's
  terminal stop is dropped). `--gene` picks the CDS (default `S`; `HA1,HA2` for flu).
* `prepare --fasta <aligned.fasta> --metadata <strain⇥date tsv> --date-cutoff YYYY-MM-DD`:
  keeps sequences collected before the cutoff (`none` keeps all). `--no-clock` and
  `--min-covered` relax the filters.
* `place --prepared <prepare's output> --out-examples <file>`;
  `--post-cutoff d1,d2,…` labels each record with the cutoffs every sequence below it is
  at or after (the test tree); `--variant BA.2` flags a clade root; `--out-counts` and
  `--out-variant-counts` write (site × 20) residue count matrices.
* `--alphabet 20|22|nuc`, `--threads`, `--seed`.

## Output: the examples file

```
EXAMPLE_ID: 6
PARENT_EXAMPLE_ID: 5
MUTATION_FROM_PARENT:
I,20,V                      # from-residue, 0-based position, to-residue
POST_CUTOFF:
2020-08-15,True             # one line per cutoff this record is after (none in a train corpus)
IS_DESCENDED_FROM_VARIANT:
IS_VARIANT_ROOT:
MTEYKLVVVGAGGVGKSTLTVRLIQ   # the record's sequence
MUTATIONS_TO_CHILDREN:
4,N                         # 0-based position, to-residue; one line per child edge
```

(This is a record of the toy example below.) antiGen reads it with `antiGen.load.load_examples`. A file whose `POST_CUTOFF` lists
cutoffs needs the split named: `--split-date` on `antiGen-train`, `antiGen-eval`,
`antiGen-validate-examples` and the list evals. A record is training data at cutoff D
exactly when D is not among its lines. Train every model on a **train corpus**, built
from sequences before D only, and evaluate on the post-D records of the **test tree**.
The test tree's pre-D records are not training data (docs/NOTES.md s163).

## Toy example (seconds, CPU)

`example/` holds a 25-residue toy tree, its sidecar, seven newer sequences and their
dates. Four of them form a clade first sampled after 2020-08-15.

```bash
./antigen prepare --tree example/tree.json --root-seq example/root-sequence.json \
    --fasta example/seqs.fasta --metadata example/metadata.tsv \
    --date-cutoff none --no-clock --out toy_prepared.tsv
./antigen place --tree example/tree.json --root-seq example/root-sequence.json \
    --prepared toy_prepared.tsv --post-cutoff 2020-08-15 --out toy_tree.tsv \
    --out-examples toy_examples.txt
#   2020-08-15: 2 records at or after it, 6 before it
antiGen-validate-examples toy_examples.txt --split-date 2020-08-15 --alphabet ACDEFGHIKLMNPQRSTVWY
```

## The paper's SARS-CoV-2 corpora

Inputs: the Nextstrain trees and sidecars under `data/` (fetch with `fetch_trees.py` and
`fetch_snapshots.py`; every file's source, date and integrity checks are in
[`PROVENANCE.md`](PROVENANCE.md)), and the open (GenBank) spike alignment and metadata
(`$HOME/sc2-open-data/spike.fasta`, `metadata.tsv`; override with `ANTIGEN_FASTA` /
`ANTIGEN_META`).

| corpus (under `output/`) | built by | what |
|---|---|---|
| `sc2-<D>/train/` | `sbatch --array=… slurm/run_series.sbatch` | the train corpus of each quarterly cutoff D in `slurm/cutoffs.tsv` (2020-04-01 … 2026-01-01): the build just before D, plus every sequence collected before D |
| `sc2-test/` | `sbatch slurm/run_all.sbatch` | **the** test tree: every sequence on the 2026-03-15 all-time build, each record labelled against every cutoff. Also `counts_ba2/counts_<D>.tsv`, the BA.2-clade residue counts dated ≥ D |
| `sc2-2021/train/`, `sc2-2021/test/` | `slurm/run_2021.sbatch`, `slurm/place_full.sbatch` | the 2021-01-01 arm on the 2020-12-29 `/ncov/global` build |
| `sc2-2023-01-01/train/msa_*.fasta` | `slurm/msa_2023-01-01.sbatch` | the 2023-01-01 alignment the EVE / Tranception models train on |

`slurm/run_arms*.sbatch` (with `arms*.tsv`) build the H3N2, RSV-A and DENV2 corpora the
same way. `check_examples.py` and `check_variant_counts.py` audit a corpus;
`seed_study.py` rebuilds one under other seeds; `bench/` is the placement benchmark
behind `antiGen-figures/figs/usher_benchmark.png`.

`NOTES.md` is this pipeline's own running record of design decisions.
