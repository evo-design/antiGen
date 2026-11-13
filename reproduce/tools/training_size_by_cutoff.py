#!/usr/bin/env python
"""Training-set size at each cutoff, in the three units of figS_gate_units.

For each cutoff D, from the files the cutoff-series models train on
(antiGen-data-revised/output/sc2-D/train/):

  n_mutations   total mutation EVENTS over the training examples: the sum of every
                example's descendant-mutation matrix (what statistical_baselines.fit
                calls N_total, here over examples_D_cur.txt alone)
  n_examples    training examples: a sequence together with its descendant mutations
  n_sequences   sequences placed on the training tree, read from the '#sequences'
                header of counts_D_cur.tsv -- the MSA's row count (checked against the
                2023-01-01 MSA: 7,813,585)
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from antiGen.load import load_examples

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/ivan.specht/antiGen-data-revised/output")
ap.add_argument("--cutoffs", required=True, help="comma-separated YYYY-MM-DD")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = []
for d in a.cutoffs.split(","):
    tr = os.path.join(a.data, f"sc2-{d}", "train")
    with open(os.path.join(tr, f"counts_{d}_cur.tsv")) as f:
        head = f.readline().rstrip("\n").split("\t")
    assert head[0] == "#sequences", head
    n_seq = int(head[1])
    ex = load_examples(os.path.join(tr, f"examples_{d}_cur.txt"),
                       "ACDEFGHIKLMNPQRSTVWY", "BA.2")
    ex = ex[0] if isinstance(ex, tuple) else ex
    n_mut = float(sum(float(e.mat.sum()) for e in ex))
    rows.append((d, len(ex), int(round(n_mut)), n_seq))
    print(f"{d}  examples={len(ex):>7,}  mutations={int(n_mut):>9,}  sequences={n_seq:>10,}",
          flush=True)

with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cutoff", "n_examples", "n_mutations", "n_sequences"])
    w.writerows(rows)
print("wrote", a.out)
