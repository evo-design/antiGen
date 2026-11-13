# Tutorial: scoring mutations with pretrained antiGen

This tutorial uses the pretrained antiGen model that ships with the package to score
every amino-acid substitution of a SARS-CoV-2 spike, relative to a query you choose. Here
the query is **BA.2.86**, the starting point of the paper's BA.2.86 → KP.3 case study.
The model was trained on the phylogeny of sequences dated before **2023-01-01**, months
before BA.2.86 was first sampled. Everything runs on a CPU, in a few seconds.

You will:

1. get the substitution probabilities of every mutation relative to the query, and
2. get a ranked list of the top decile of mutations in a coordinate range (the RBD),
   computed the way the paper's all-RBD evaluation ranks them.

## 1. Install

The package is on PyPI as `antigen-model` (Python ≥ 3.10; version 0.2.0 or later ships
the pretrained models). To stay CPU-only, install the CPU build of PyTorch first. Plain
`pip install torch` would pull the much larger CUDA build on Linux.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install antigen-model
```

The distribution is named `antigen-model`; the import name and the command-line tools
keep the `antiGen` spelling (`import antiGen`, `antiGen-infer`, `antiGen-top-decile`).
To work from source instead, clone the repository and run `pip install .` in it.

**The pretrained models come with the install.** No download step and no GPU are
needed. Two models are included, about 25 MB in all:

| name | what it is |
|---|---|
| `sc2-2023-01-01` (default) | **antiGen (CV ×10)**: ten models, each trained on a different 90/10 train/validation split and kept at its best validation epoch, combined as the mean of their log-probabilities. The "antiGen (CV x10)" of the paper's bar panels. |
| `sc2-2023-01-01-base` | the single base model trained at the same cutoff, the one behind the paper's case study (panel B) and its cutoff-series panels |

```python
from antiGen import pretrained
pretrained.available()
# {'sc2-2023-01-01': 'antiGen (CV x10), SARS-CoV-2 spike, 2023-01-01 cutoff',
#  'sc2-2023-01-01-base': 'antiGen single base model, SARS-CoV-2 spike, 2023-01-01 cutoff'}
```

## 2. The query

The models read a spike protein in the 20-letter amino-acid alphabet, **aligned to the
1,273-residue Wuhan-Hu-1 frame**: position *i* of the query is spike position *i*. The
BA.2.86 spike ships with the package in that frame, exactly as the paper scored it.
Write it to the working directory:

```bash
python -c "from antiGen import pretrained; pretrained.write_example('BA.2.86_spike.fasta')"
```

To score your own sequence, align it to the Wuhan-Hu-1 spike (GenBank NC_045512.2) and
write it at length 1,273 using only `ACDEFGHIKLMNPQRSTVWY`. The tools refuse any other
length or character, rather than silently scoring a misaligned sequence.

## 3. Substitution probabilities for every mutation

```bash
antiGen-infer --pretrained sc2-2023-01-01 BA.2.86_spike.fasta -o ba286_scores.csv
```

`ba286_scores.csv` has one row for each of the 1,273 × 19 = 24,187 possible
substitutions:

```
query_id,position,ref_aa,alt_aa,prob,position_1based,mutation
BA.2.86,345,R,T,6.750526e-03,346,R346T
BA.2.86,454,L,S,2.021199e-04,455,L455S
BA.2.86,455,F,L,3.186422e-04,456,F456L
...
```

**What `prob` means.** It is antiGen's predicted probability that the next mutation on
this background is that substitution. It is **one distribution over all substitutions
of the whole sequence** (it sums to 1 over the 24,187 rows), not a separate categorical
per position. That is the quantity the model is trained on. `position` is 0-based;
`position_1based` and `mutation` use the usual notation. `--format wide` writes one row
per position, with a column per amino acid.

The same thing in Python:

```python
import torch
from antiGen import pretrained

model, info = pretrained.load("sc2-2023-01-01", device="cpu")
seq = "".join(l.strip() for l in open("BA.2.86_spike.fasta")
              if not l.startswith(">"))
x = pretrained.encode(seq, info["alphabet"])          # (1, 1273) residue indices
with torch.no_grad():
    scores = model(x)                                 # (1, 1273, 20) log-scores
probs = pretrained.substitution_probs(scores, x)      # sums to 1; wild type = 0

aa = info["alphabet"]
print(probs[0, 455 - 1, aa.index("S")])               # P(L455S) ≈ 2.0e-4
```

## 4. A ranked top-decile list for the RBD

The paper evaluates antiGen by ranking every candidate substitution in a region and
asking whether the mutations that actually happen next fall in the top decile. To get
that list for the receptor-binding domain (positions 331–531):

```bash
antiGen-top-decile BA.2.86_spike.fasta --start 331 --end 531 -o ba286_top_rbd.csv
```

```
model: antiGen (CV x10), SARS-CoV-2 spike, 2023-01-01 cutoff
BA.2.86: 3819 candidates in 331-531; top 0.1: 382 listed
```

The RBD has 201 positions × 19 non-wild-type residues = **3,819 candidates**, and the
top decile is the best ⌈0.1 × 3,819⌉ = **382**. This is the candidate pool and cutoff the
paper's all-RBD panels use. The list, best first:

```
query_id,rank,mutation,position,ref_aa,alt_aa,score,prob,percentile
BA.2.86,1,S446G,446,S,G,-4.289715,7.121325e-02,100.00
BA.2.86,2,K440N,440,K,N,-4.365720,6.600123e-02,99.97
BA.2.86,3,N417K,417,N,K,-4.577544,5.340207e-02,99.95
BA.2.86,4,K460N,460,K,N,-4.587424,5.287705e-02,99.92
BA.2.86,5,K484E,484,K,E,-4.730417,4.583174e-02,99.90
...
BA.2.86,9,R346T,346,R,T,-5.196780,2.874930e-02,99.79
...
BA.2.86,114,L455S,455,L,S,-8.705295,8.607933e-04,97.04
...
```

`rank` 1 is the most likely; tied cells share their best rank. `score` is the model's
log-score, which is what the ranking uses. `prob` is the substitution probability
renormalised over the 3,819 candidates in the range, and `percentile` is 100 at the top.

**Reading it against what happened.** L455S, the substitution that took BA.2.86 to JN.1
(the dominant lineage of early 2024), is rank **114 of 3,819**: inside the top decile and
the top 3%, from a model that never saw a post-2022 sequence. R346T, gained on the way
to JN.1.11.1 (and later reverted on the path to KP.3), is rank 9.

Options: `--qtile 0.05` lists the top 5% instead; `--all` writes every candidate in
the range; omit `--start`/`--end` to rank the whole spike; `--pretrained
sc2-2023-01-01-base` uses the single base model; `--checkpoint my_model.pt` uses your
own trained model. In Python:

```python
from antiGen.top_decile import rank_substitutions, score_query, top_decile

ranked = rank_substitutions(score_query(model, seq, info["alphabet"]), seq,
                            info["alphabet"], start=331, end=531)
top = top_decile(ranked, qtile=0.1)                   # 382 dicts, best first
```

## 5. Reproducing the paper's case study

Panel B of the paper's main SARS-CoV-2 figure ranks each step of BA.2.86 → JN.1 →
JN.1.11.1 → KP.3 with the single base model. Its first step, L455S from BA.2.86, is
reproduced exactly:

```bash
antiGen-top-decile BA.2.86_spike.fasta --start 331 --end 531 \
    --pretrained sc2-2023-01-01-base --all -o base_rbd.csv
grep ",L455S," base_rbd.csv        # rank 98 of 3,819, as in the figure
```

`antiGen/tests/test_pretrained.py` checks this number, so the shipped models stay pinned
to the paper's.

## Notes

* **Speed.** One query takes about 0.3 s with the ten-model ensemble on a single CPU;
  each command, including Python start-up, about 2 s. FASTA files with
  several records are scored one after another.
* **What the scores are not.** antiGen predicts which substitution happens next on the
  query's background, learned from the phylogeny. It is not a measured fitness or
  escape value.
* **Other cutoffs, other viruses.** Train your own model with `antiGen-train` (see the
  README) and pass it with `--checkpoint`; every command above works the same way.
