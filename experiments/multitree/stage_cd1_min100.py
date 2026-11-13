#!/usr/bin/env python
"""supp_cd1_min100 (NOTES s160): supp_cd1's all-RBD row (main figure C-F at codon
distance 1) with the SCORED cells further limited to those observed in >= MIN sequences
dated on or after the 2023-01-01 cutoff.

The top decile is unchanged: same per-query pool (the one-nucleotide cells, de novo
for the de novo panels) and the same ranking, so every cell's hit mass is unchanged. The
mask only chooses which (position, residue) cells are averaged. So the panels are
recomputed from list_macro's per-cell output; no model is re-scored.

Mask: sc2-test/counts_ba2/counts_2023-01-01.tsv, the PFM of sequences dated >= 2023-01-01
(688,459 sequences: the BA.2 subtree, which after this date is essentially every
sequence). A cell is kept when its count is >= MIN.

Gate: the per-cell rows, unmasked, reproduce list_macro's summary (per-cell hits are
stored to 4 decimals, hence the 1e-4 tolerance).
"""
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from antiGen.load import load_counts_matrix

AA = "ACDEFGHIKLMNPQRSTVWY"
MIN = int(os.environ.get("MIN_COUNT", "100"))
ARM = os.environ.get("ARM", "2023")
STORE = os.environ.get("STORE", "/large_storage/hielab/ispecht/fitness-inference")
FIG = os.environ.get("FIGDATA", "/home/ivan.specht/antiGen/antiGen-figures/data")
if ARM == "2023":
    RUN = f"{STORE}/evals_cd1_2023-01-01/out"
    PFM = os.environ.get("CORPORA", "/home/ivan.specht/antiGen-data-revised/output") + "/sc2-test/counts_ba2/counts_2023-01-01.tsv"
    OUT = f"{FIG}/evals_sc2_lists_macro_aa20_2023-01-01_cd1_min{MIN}"
    ENS = "ens_vs10_2023"
else:
    # 2021-01-01: all RBD cells (no COG-UK list). The mask is the ALL-LINEAGE PFM of
    # sequences dated >= 2021-01-01 (postcutoff_counts.py; counts_ba2 would miss
    # Alpha, Delta and BA.1).
    RUN = f"{STORE}/evals_cd1_2021_allcells/out"
    PFM = f"{STORE}/postcutoff_counts/counts_all_lineages_2021-01-01.tsv"
    OUT = f"{FIG}/evals_sc2_lists_macro_aa20_2021_allcells_cd1_min{MIN}"
    ENS = "ens_vs10_2021"
# NOTES s163: RUN_DIR / OUT_DIR override, to stage a re-fitted run somewhere the
# published file can then be spliced from.
RUN = os.environ.get("RUN_DIR", RUN)
PFM = os.environ.get("PFM", PFM)
OUT = os.environ.get("OUT_DIR", OUT)
EXTRA = {"joint_cv3", "covonly", "blend_r1_AJC", "blend_r2_AJK", "blend_dn_CK"}
REN = {"list_allcells_cd1_fullpool": "list_allcells_fullpool",
       "list_allcells_unseen_cd1_fullpool": "list_allcells_unseen_fullpool"}
# NT1=0 (NOTES s166): the same mask on the FULL pool (every substitution, no 1-nt
# filter) -- the run's own *_fullpool variants, staged under their own names.
if os.environ.get("NT1", "1") == "0":
    REN = {"list_allcells_fullpool": "list_allcells_fullpool",
           "list_allcells_unseen_fullpool": "list_allcells_unseen_fullpool"}
    OUT = os.environ.get("OUT_DIR", OUT.replace("_cd1_min", "_min"))

pfm = np.asarray(load_counts_matrix(PFM, AA))
keep = pfm >= MIN                                     # (1273, 20), site index = pos1 - 1
print(f"mask: {int(keep[330:531].sum())} of {201 * 20} RBD cells seen in >= {MIN} "
      f"post-cutoff sequences")

cells = collections.defaultdict(list)                 # (model, variant) -> [(pos1, alt, k, hits)]
for r in csv.DictReader(open(f"{RUN}/list_macro_percell.csv")):
    if r["variant"] in REN and r["qtile"] == "0.10":
        cells[(r["model"], r["variant"])].append(
            (int(r["pos1"]), r["alt"], float(r["k_events"]), float(r["hits"])))
summ = {(r["model"], r["variant"]): r for r in csv.DictReader(open(f"{RUN}/list_macro_summary.csv"))
        if r["qtile"] == "0.10" and r["variant"] in REN}


def pool(rows):
    rate = [h / k for _, _, k, h in rows]
    return (float(np.mean(rate)) if rate else float("nan"),
            sum(h for *_, h in rows) / max(sum(k for _, _, k, _ in rows), 1),
            len(rows), sum(k for _, _, k, _ in rows))


worst = 0.0
for key, rows in cells.items():
    mac, mic, n, ev = pool(rows)
    s = summ[key]
    assert n == int(s["n_cells"]) and abs(ev - float(s["n_events"])) < 0.5, key
    worst = max(worst, abs(mac - float(s["macro"])), abs(mic - float(s["micro"])))
assert worst < 1e-4, worst
print(f"GATE OK: {len(cells)} (model, variant) pairs, unmasked per-cell rows reproduce the "
      f"summary, worst {worst:.1e}")

os.makedirs(OUT, exist_ok=True)
hdr = ["model", "variant", "qtile", "macro", "micro", "n_cells", "n_events"]
out = {"o": [], "e": []}
for (m, v), rows in sorted(cells.items()):
    if m in EXTRA:
        continue
    kept = [x for x in rows if keep[x[0] - 1, AA.index(x[1])]]
    mac, mic, n, ev = pool(kept)
    out["e" if m == ENS else "o"].append(dict(model=m, variant=REN[v], qtile="0.10",
                                              macro=f"{mac:.8f}", micro=f"{mic:.8f}",
                                              n_cells=n, n_events=f"{ev:.0f}"))
for name, part in (("list_macro_summary_aa20.csv", out["o"]),
                   ("list_macro_summary_ensemble.csv", out["e"])):
    w = csv.DictWriter(open(f"{OUT}/{name}", "w", newline=""), fieldnames=hdr)
    w.writeheader(); w.writerows(part)
open(f"{OUT}/ensemble_figure_rows.tsv", "w").write(
    f"{ENS}\tantiGen, 10x val-selected epoch (mean of log-probs)\t"
    "Val-selected, averaged over 10 holdouts\n")
for r in out["e"]:
    print(f"{ENS} {r['variant']}: macro {float(r['macro']):.3f} micro {float(r['micro']):.3f} "
          f"cells {r['n_cells']} events {r['n_events']}")
print(f"staged {len({r['model'] for r in out['o']})} models + {ENS} -> {OUT}")
