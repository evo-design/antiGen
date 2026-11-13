# %% [markdown]
# # Scoring mutations with pretrained antiGen
#
# Score every amino-acid substitution of a SARS-CoV-2 spike with the pretrained antiGen
# that ships with the `antigen-model` package, then rank the receptor-binding domain
# (RBD). The query is **BA.2.86**, the start of the paper's BA.2.86 → KP.3 case study.
# The model was trained on the phylogeny of sequences dated before **2023-01-01**, months
# before BA.2.86 was first sampled. Everything runs on a CPU in a few seconds.
#
# The same cells are in [`tutorial.ipynb`](https://github.com/evo-design/antiGen/blob/main/tutorial.ipynb)
# (Jupyter, or [open it in Colab](https://colab.research.google.com/github/evo-design/antiGen/blob/main/tutorial.ipynb))
# and [`tutorial.py`](https://github.com/evo-design/antiGen/blob/main/tutorial.py)
# (`python tutorial.py`, or cell by cell in an editor that understands `# %%`).
#
# **The query.** Download
# **[BA.2.86_spike.fasta](https://github.com/evo-design/antiGen/releases/latest/download/BA.2.86_spike.fasta)**
# and save it in a folder named `queries` inside the directory you run the tutorial
# from, as `queries/BA.2.86_spike.fasta`. In a clone of the repository it is already
# there. It is the BA.2.86 spike aligned to the 1,273-residue Wuhan-Hu-1 frame (position
# *i* is spike position *i*), in the 20-letter amino-acid alphabet, exactly as the paper
# scored it.

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
# ## 2. Load the model and the query
#
# Two models ship with the package (about 25 MB in all):
#
# * `sc2-2023-01-01` (default): **antiGen (CV ×10)**, ten models each trained on a
#   different 90/10 split of the 2023-01-01 train corpus and kept at its best validation
#   epoch, combined as the mean of their log-probabilities;
# * `sc2-2023-01-01-base`: the single base model behind the paper's case study and
#   cutoff series (pass its name to `pretrained.load` to use it instead).

# %%
from antiGen import pretrained

model, info = pretrained.load("sc2-2023-01-01", device="cpu")
alphabet = info["alphabet"]
print(info["description"])

seq = "".join(l.strip() for l in open("queries/BA.2.86_spike.fasta") if not l.startswith(">"))
print(len(seq), "residues")

# %% [markdown]
# ## 3. Substitution probabilities for every mutation
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
# ## 4. The top decile of the RBD
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
# ## 5. A FASTA of the query and its top k variants
#
# The query followed by its `k` best single-amino-acid variants from the RBD ranking
# above, each the query with that one substitution, ready to pass to anything that reads
# FASTA. For the whole spike instead, rank with
# `rank_substitutions(scores[0], seq, alphabet)` (no `start` / `end`).

# %%
K = 10
NAME = "BA.2.86"

records = [(NAME, seq)]
for r in ranked.head(K).itertuples():
    i = r.position - 1
    assert seq[i] == r.ref_aa
    records.append((f"{NAME}_{r.mutation} rank={r.rank} prob={r.prob:.3g}",
                    seq[:i] + r.alt_aa + seq[i + 1:]))

out = f"ba286_top{K}_variants.fasta"
with open(out, "w") as f:
    for header, s in records:
        f.write(f">{header}\n" + "\n".join(s[j:j + 60] for j in range(0, len(s), 60)) + "\n")
print(f"wrote {out}: the query + {len(records) - 1} variants")
print("\n".join(h for h, _ in records))

# %% [markdown]
# ## Notes
#
# * **Your own sequence.** Align it to the Wuhan-Hu-1 spike (GenBank NC_045512.2), write
#   it at length 1,273 using only `ACDEFGHIKLMNPQRSTVWY`, save it in `queries/`, and put
#   its file name in section 2. `pretrained.encode` refuses any other character, and the
#   model any other length, rather than silently scoring a misaligned sequence.
# * **Gaps and missing data in the query.** The model has no gap or unknown symbol, so
#   every position needs a real amino acid. Where your sequence has a gap, an `X` or an
#   uncalled site, fill in the residue of a known recent ancestor at that position (for
#   example the lineage's parent, or the nearest ancestral node on the tree), so the
#   query is a genotype that actually exists rather than an imputed one.
# * **What the scores are not.** antiGen predicts which substitution happens next on the
#   query's background, learned from the phylogeny. It is not a measured fitness or
#   escape value.
# * **From the command line,** the same two results:
#   `antiGen-infer --pretrained sc2-2023-01-01 queries/BA.2.86_spike.fasta -o ba286_scores.csv`
#   and `antiGen-top-decile queries/BA.2.86_spike.fasta --start 331 --end 531 -o ba286_top_rbd.csv`.
# * **Other cutoffs, other viruses.** Train your own model (README, Parts 1–2) and load it
#   with `antiGen.models.factory.load_model`; every cell above works the same way.
