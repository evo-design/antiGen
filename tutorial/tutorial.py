# %% [markdown]
# # Scoring mutations with pretrained antiGen
#
# Score every amino-acid substitution of a SARS-CoV-2 spike with the pretrained antiGen
# that ships with the `antigen-model` package, then rank the receptor-binding domain
# (RBD). The query is **BA.2.86**, the start of the paper's BA.2.86 → KP.3 case study.
# The model was trained on the phylogeny of sequences dated before **2023-01-01**, months
# before BA.2.86 was first sampled. Everything runs on a CPU in a few seconds.
#
# The same cells are in [`tutorial/tutorial.ipynb`](https://github.com/evo-design/antiGen/blob/main/tutorial/tutorial.ipynb)
# (Jupyter, or [open it in Colab](https://colab.research.google.com/github/evo-design/antiGen/blob/main/tutorial/tutorial.ipynb))
# and [`tutorial/tutorial.py`](https://github.com/evo-design/antiGen/blob/main/tutorial/tutorial.py)
# (`python tutorial.py`, or cell by cell in an editor that understands `# %%`).

# %% [markdown]
# ## 1. Install
#
# Python ≥ 3.10. To stay CPU-only on Linux, install the CPU build of PyTorch first
# (`pip install torch --index-url https://download.pytorch.org/whl/cpu`); plain
# `pip install torch` pulls the much larger CUDA build. In a notebook this is the
# `%pip install antigen-model` cell. The distribution is named `antigen-model`; the
# import name keeps the `antiGen` spelling.

# %%
# %pip install antigen-model

# %% [markdown]
# ## 2. The query
#
# The model reads a spike in the 20-letter amino-acid alphabet, **aligned to the
# 1,273-residue Wuhan-Hu-1 frame**: position *i* of the query is spike position *i*.
# The BA.2.86 spike, in that frame exactly as the paper scored it:
# **[download BA.2.86_spike.fasta](https://github.com/evo-design/antiGen/releases/latest/download/BA.2.86_spike.fasta)**.
# Put it in the working directory; if it is not there, this cell fetches it.

# %%
import os
import urllib.request

QUERY = "BA.2.86_spike.fasta"
URL = "https://github.com/evo-design/antiGen/releases/latest/download/BA.2.86_spike.fasta"
if not os.path.exists(QUERY):
    urllib.request.urlretrieve(URL, QUERY)

def read_fasta(path):
    """The first record of a FASTA file, as (name, sequence)."""
    lines = [l.strip() for l in open(path) if l.strip()]
    return lines[0][1:].split()[0], "".join(l for l in lines[1:] if not l.startswith(">"))

name, seq = read_fasta(QUERY)
print(name, len(seq), "residues")

# %% [markdown]
# ## 3. Load the pretrained model
#
# Two models ship with the package (about 25 MB in all):
#
# * `sc2-2023-01-01` (default): **antiGen (CV ×10)**, ten models each trained on a
#   different 90/10 split of the 2023-01-01 train corpus and kept at its best validation
#   epoch, combined as the mean of their log-probabilities;
# * `sc2-2023-01-01-base`: the single base model behind the case study and the cutoff
#   series.

# %%
from antiGen import pretrained

print(pretrained.available())
model, info = pretrained.load("sc2-2023-01-01", device="cpu")
alphabet = info["alphabet"]
print(info["description"], "| alphabet", alphabet, "| length", info["seq_len"])

# %% [markdown]
# ## 4. Substitution probabilities for every mutation
#
# One forward pass gives a score for every (position, residue) cell.
# `substitution_probs` turns it into antiGen's predicted probability that the **next
# mutation** on this background is each substitution. It is **one distribution over all
# substitutions of the whole sequence** (it sums to 1 over the 1,273 × 19 = 24,187
# substitutions), not a separate categorical per position.

# %%
import pandas as pd
import torch

x = pretrained.encode(seq, alphabet)                 # (1, 1273) residue indices
with torch.no_grad():
    scores = model(x)                                # (1, 1273, 20) log-scores
probs = pretrained.substitution_probs(scores, x)[0]  # (1273, 20); wild type = 0

rows = [(pos + 1, seq[pos], alt, float(probs[pos, j]))
        for pos in range(len(seq)) for j, alt in enumerate(alphabet) if alt != seq[pos]]
table = pd.DataFrame(rows, columns=["position", "ref_aa", "alt_aa", "prob"])
table["mutation"] = table.ref_aa + table.position.astype(str) + table.alt_aa
print(len(table), "substitutions; probabilities sum to", round(table.prob.sum(), 6))
table.to_csv("ba286_scores.csv", index=False)
print(table.set_index("mutation").loc[["R346T", "L455S", "F456L"]])

# %% [markdown]
# ## 5. The top decile of the RBD
#
# The paper evaluates antiGen by ranking every candidate substitution in a region and
# asking whether the mutations that actually happen next fall in the top decile. For the
# RBD (positions 331–531) there are 201 × 19 = **3,819** candidates, and the top decile is
# the best ⌈0.1 × 3,819⌉ = **382**: the pool and cutoff of the paper's all-RBD panels.
# `rank` 1 is the most likely (tied cells share their best rank), `prob` is renormalised
# over the candidates in the range, and `percentile` is 100 at the top.

# %%
from antiGen.top_decile import rank_substitutions, top_decile

ranked = pd.DataFrame(rank_substitutions(scores[0], seq, alphabet, start=331, end=531))
top = pd.DataFrame(top_decile(ranked.to_dict("records"), qtile=0.1))
print(len(ranked), "candidates in the RBD;", len(top), "in the top decile")
top.to_csv("ba286_top_rbd.csv", index=False)
print(top.head(10).to_string(index=False))

# %% [markdown]
# **Reading it against what happened.** L455S took BA.2.86 to JN.1, the dominant lineage
# of early 2024. R346T was gained on the way to JN.1.11.1 (and later reverted on the path
# to KP.3). Both are inside the top decile, from a model that never saw a post-2022
# sequence:

# %%
print(ranked.set_index("mutation").loc[["L455S", "R346T"], ["rank", "prob", "percentile"]])

# %% [markdown]
# ## 6. The paper's case study
#
# Panel B of the paper's main SARS-CoV-2 figure ranks each step of BA.2.86 → JN.1 →
# JN.1.11.1 → KP.3 with the single base model. Its first step, L455S from BA.2.86, is
# rank **98 of 3,819** there; `antiGen/tests/test_pretrained.py` pins this number.

# %%
from antiGen.top_decile import score_query

base, _ = pretrained.load("sc2-2023-01-01-base", device="cpu")
ranked_base = pd.DataFrame(rank_substitutions(score_query(base, seq, alphabet), seq,
                                              alphabet, start=331, end=531))
print(ranked_base.set_index("mutation").loc[["L455S"], ["rank", "percentile"]])

# %% [markdown]
# ## Notes
#
# * **Your own sequence.** Align it to the Wuhan-Hu-1 spike (GenBank NC_045512.2) and
#   write it at length 1,273 using only `ACDEFGHIKLMNPQRSTVWY`. `pretrained.encode`
#   refuses any other character, and the model any other length, rather than silently
#   scoring a misaligned sequence.
# * **What the scores are not.** antiGen predicts which substitution happens next on the
#   query's background, learned from the phylogeny. It is not a measured fitness or
#   escape value.
# * **From the command line,** the same two results:
#   `antiGen-infer --pretrained sc2-2023-01-01 BA.2.86_spike.fasta -o ba286_scores.csv` and
#   `antiGen-top-decile BA.2.86_spike.fasta --start 331 --end 531 -o ba286_top_rbd.csv`.
# * **Other cutoffs, other viruses.** Train your own model (README, Parts 1–2) and load it
#   with `antiGen.models.factory.load_model`; every cell above works the same way.
