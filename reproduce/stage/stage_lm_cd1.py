#!/usr/bin/env python
"""Stage sc2_results_revised2 L/M at codon distance 1.

Source: the per-cutoff series runs of s162/s163 (STORE/evals_cd1_series_rare100_trainonly),
which carry `list_allcells_cd1_fullpool` -- every RBD cell, the pool AND the targets
restricted per query to cells one nucleotide from that query's own residue -- for all 16
of L/M's cutoffs, with every model fitted on the train tree only. That variant applies no
seen mask, so --seen-min-count does not touch it (checked: identical rows in the de novo
arm's runs at every cutoff both have).

Writes antiGen-figures/data/gating_macro_aa20_cd1/<D>/list_macro_summary.csv with the
variant renamed to `list_allcells`, the name the by_macro loader reads. macro = per cell
(L), micro = per mutation (M). The runs use the s161 top-q rule throughout.
"""
import csv
import glob
import os
import shutil

STORE = os.environ.get("STORE", "/large_storage/hielab/ispecht/fitness-inference")
SRC = os.environ.get("RUNS", f"{STORE}/evals_cd1_series_rare100_trainonly")
OUT = os.environ.get("FIGDATA", "/home/ivan.specht/antiGen/antiGen-figures/data") + "/gating_macro_aa20_cd1"
VAR = "list_allcells_cd1_fullpool"
CUTS = sorted(os.path.basename(d) for d in glob.glob(f"{SRC}/20*"))
assert len(CUTS) == 16, CUTS
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
for d in CUTS:
    rows = [r for r in csv.DictReader(open(f"{SRC}/{d}/list_macro_summary.csv")) if r["variant"] == VAR]
    models = {r["model"] for r in rows}
    assert len(models) == 25, (d, len(models))
    os.makedirs(f"{OUT}/{d}")
    with open(f"{OUT}/{d}/list_macro_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "variant": "list_allcells"})
    q = [r for r in rows if r["qtile"] == "0.10"][0]
    print(f"{d}: {len(models)} models, {q['n_cells']} cells, {q['n_events']} events")
print(f"staged {len(CUTS)} cutoffs -> {OUT}")
