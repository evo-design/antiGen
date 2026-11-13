#!/usr/bin/env python
"""Stage the de novo, codon-distance-1, >= 100 post-cutoff-sequence series.

Per cutoff D (STORE/evals_cd1_series/<D>):
  * per-cell rows of variant list_allcells_unseen_cd1_fullpool: the one-nucleotide pool
    restricted to D's de novo cells, at q = 0.10;
  * kept cells: those seen in >= MIN sequences (all lineages) dated >= D
    (STORE/postcutoff_counts/counts_all_lineages_<D>.tsv);
  * per cell (macro) and per mutation (micro), every model.

GATE per cutoff: the unmasked per-cell rows reproduce the run's own summary.
STOPPING RULE (pre-registered in s162, model-independent): a cutoff is drawn only if
its kept set has >= MIN_CELLS cells and >= MIN_EVENTS events, and the series stops
at the first cutoff that fails.

Writes antiGen-figures/data/gating_cd1_dn_min100/<D>/list_macro_summary.csv in the
layout the drawer's by_macro loader reads (variant "list_allcells"). Only q = 0.10
exists per cell, so q = 0.20 and 0.50 are written as NA; L/M draw q = 0.10.
"""
import csv
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from antiGen.load import load_counts_matrix

AA = "ACDEFGHIKLMNPQRSTVWY"
MIN, MIN_CELLS, MIN_EVENTS = 100, 30, 300
STORE = os.environ.get("STORE", "/large_storage/hielab/ispecht/fitness-inference")
# ARM=denovo (default): pool and targets are de novo cells (training count 0).
# ARM=rare100: cells seen in < 100 training sequences (list_macro --seen-min-count 100).
ARM = os.environ.get("ARM", "denovo")
RUNS = f"{STORE}/evals_cd1_series" + ("" if ARM == "denovo" else "_rare100")
# NOTES s163: every model fitted on the train tree only (POOL=0). The pooled runs above
# gave the fitted baselines the test corpus's pre-cutoff records.
RUNS += os.environ.get("RUN_SUFFIX", "_trainonly")
RUNS = os.environ.get("RUNS", RUNS)                  # reproduce/stage.sh
POSTCUT = os.environ.get("POSTCUTOFF", f"{STORE}/postcutoff_counts")
OUT = (os.environ.get("FIGDATA", "/home/ivan.specht/antiGen/antiGen-figures/data") + "/gating_cd1_"
       + ("dn" if ARM == "denovo" else "rare100") + "_min100")
VAR = "list_allcells_unseen_cd1_fullpool"
CUTS = sys.argv[1:] or ["2020-04-01", "2020-07-01", "2020-10-01", "2021-01-01", "2021-04-01",
                        "2021-07-01", "2021-10-01", "2022-01-01", "2022-04-01", "2022-07-01",
                        "2022-10-01", "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01",
                        "2024-01-01"]

table, passing, stopped = [], [], None
for d in CUTS:
    run = f"{RUNS}/{d}"
    pfm = np.asarray(load_counts_matrix(f"{POSTCUT}/counts_all_lineages_{d}.tsv", AA))
    keep = pfm >= MIN
    cells = {}
    for r in csv.DictReader(open(f"{run}/list_macro_percell.csv")):
        if r["variant"] == VAR and r["qtile"] == "0.10":
            cells.setdefault(r["model"], []).append(
                (int(r["pos1"]), r["alt"], float(r["k_events"]), float(r["hits"])))
    summ = {r["model"]: r for r in csv.DictReader(open(f"{run}/list_macro_summary.csv"))
            if r["variant"] == VAR and r["qtile"] == "0.10"}
    worst = 0.0
    for m, rows in cells.items():
        mac = np.mean([h / k for *_, k, h in rows]); mic = sum(h for *_, h in rows) / sum(k for _, _, k, _ in rows)
        assert len(rows) == int(summ[m]["n_cells"]), (d, m)
        worst = max(worst, abs(mac - float(summ[m]["macro"])), abs(mic - float(summ[m]["micro"])))
    assert worst < 1e-4, (d, worst)
    out = {}
    for m, rows in cells.items():
        k_ = [x for x in rows if keep[x[0] - 1, AA.index(x[1])]]
        ev = sum(x[2] for x in k_)
        out[m] = (float(np.mean([h / k for *_, k, h in k_])) if k_ else float("nan"),
                  sum(x[3] for x in k_) / ev if ev else float("nan"), len(k_), ev)
    n_cells, n_ev = next(iter(out.values()))[2:]
    assert all(v[2] == n_cells and v[3] == n_ev for v in out.values()), d   # model-independent
    ok = n_cells >= MIN_CELLS and n_ev >= MIN_EVENTS
    table.append((d, n_cells, n_ev, ok, worst))
    if stopped is None and ok:
        passing.append((d, out))
    elif stopped is None:
        stopped = d

print("cutoff      kept cells  events  pass   gate (unmasked vs summary)")
for d, c, e, ok, w in table:
    print(f"{d}  {c:10d}  {e:6.0f}  {'yes' if ok else 'NO ':4s}  {w:.1e}")
print(f"series: {passing[0][0]} .. {passing[-1][0]} ({len(passing)} cutoffs)"
      + (f"; stops before {stopped}" if stopped else "; every cutoff passes"))

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
hdr = ["model", "variant", "qtile", "macro", "micro", "n_cells", "n_events"]
for d, out in passing:
    os.makedirs(f"{OUT}/{d}")
    with open(f"{OUT}/{d}/list_macro_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for m, (mac, mic, n, ev) in sorted(out.items()):
            w.writerow(dict(model=m, variant="list_allcells", qtile="0.10", macro=f"{mac:.8f}",
                            micro=f"{mic:.8f}", n_cells=n, n_events=f"{ev:.0f}"))
            for q in ("0.20", "0.50"):
                w.writerow(dict(model=m, variant="list_allcells", qtile=q, macro="NA",
                                micro="NA", n_cells=n, n_events=f"{ev:.0f}"))
print(f"staged {len(passing)} cutoffs -> {OUT}")
