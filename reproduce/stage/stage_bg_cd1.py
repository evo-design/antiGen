#!/usr/bin/env python
"""Stage the one-nucleotide query-adaptability row of supp_query_adapt.

Source: a background_contrast.py --cd1 run on the 2023-01-01 background maps (cell sets
rbd_cd1_all / rbd_cd1_unseen) and its host-weighted re-pool (background_event_weighted.py).
Writes antiGen-figures/data/evals_sc2_background_aa20_2023-01-01_cd1{,_eventwtd}/ with the
cell sets renamed from <set>_cd1_* to <set>_* (rbd_all, coguk_unseen, ...), the names the drawer reads, so the 1-nt row is
drawn by exactly the code that draws the all-RBD row.

    python reproduce/stage/stage_bg_cd1.py RUN_DIR          # RUN_DIR and RUN_DIR_eventwtd
"""
import csv
import os
import sys

RUN = sys.argv[1].rstrip("/")
FIG = os.environ.get("FIGDATA", os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "antiGen-figures", "data"))

for src, dst, files in [
        (RUN, "evals_sc2_background_aa20_2023-01-01_cd1",
         ("background_summary.csv", "background_percell.csv", "background_cellsets.csv")),
        (RUN + "_eventwtd", "evals_sc2_background_aa20_2023-01-01_cd1_eventwtd",
         ("background_summary.csv", "background_cellsets.csv"))]:
    out = os.path.join(FIG, dst)
    os.makedirs(out, exist_ok=True)
    for f in files:
        rows = list(csv.DictReader(open(os.path.join(src, f))))
        if "cellset" in rows[0]:
            # Each set may also carry a `_perquery` twin (the per-query pool convention).
            def ren(name):
                assert "_cd1_" in name, f"{src}/{f}: not a --cd1 cell set: {name}"
                return name.replace("_cd1_", "_", 1)    # rbd_cd1_all -> rbd_all, coguk_cd1_unseen -> coguk_unseen
            for r in rows:
                r["cellset"] = ren(r["cellset"])
        with open(os.path.join(out, f), "w", newline="") as o:
            w = csv.DictWriter(o, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    print(f"staged {src} -> {out}")
