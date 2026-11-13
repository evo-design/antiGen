#!/usr/bin/env python
"""Stage the whole-spike codon-distance-1 run for supp_fullspike's second row (NOTES s178).

Source: STORE/evals_sc2_lists_macro_aa20_2023-01-01_allcells_fullspike_cd1, a fresh
run of the published whole-spike command (train-only fits, ANTIGEN_TOPQ_LEGACY=1) with
--cd1-variants added. Writes data/evals_sc2_lists_macro_aa20_2023-01-01_fullspike_cd1/
with the 1-nt variants renamed to the names the drawer reads (as s159 did), the CV x10
row split off into list_macro_summary_ensemble.csv, and ensemble_figure_rows.tsv.

GATE: the run's FULL-POOL rows (list_allcells{,_unseen}_fullpool) must equal the
published supp_fullspike rows to 1e-6 for every model both have: the same command, so
any difference means the run is not the published one.
"""
import csv
import os

STORE = os.environ.get("STORE", "/large_storage/hielab/ispecht/fitness-inference")
RUN = os.environ.get("RUN_DIR", f"{STORE}/evals_sc2_lists_macro_aa20_2023-01-01_allcells_fullspike_cd1") + "/list_macro_summary.csv"
FIG = os.environ.get("FIGDATA", "/home/ivan.specht/antiGen/antiGen-figures/data")
PUB = [f"{FIG}/evals_sc2_lists_macro_aa20_2023-01-01_fullspike/list_macro_summary_aa20.csv",
       f"{FIG}/evals_sc2_lists_macro_aa20_2023-01-01_fullspike/list_macro_summary_ensemble.csv"]
OUT = f"{FIG}/evals_sc2_lists_macro_aa20_2023-01-01_fullspike_cd1"
ENS = "ens_vs10_2023"
REN = {"list_allcells_cd1_fullpool": "list_allcells_fullpool",
       "list_allcells_unseen_cd1_fullpool": "list_allcells_unseen_fullpool"}
key = lambda r: (r["model"], r["variant"], r["qtile"])

run = list(csv.DictReader(open(RUN)))
hdr = list(run[0])
runk = {key(r): r for r in run}
pub = {key(r): r for f in PUB for r in csv.DictReader(open(f))}
shared = [k for k in pub if k in runk and k[1] in REN.values()]
worst = max(abs(float(pub[k][c]) - float(runk[k][c])) for k in shared for c in ("macro", "micro"))
if not shared or worst > 1e-6:
    raise SystemExit(f"GATE FAIL: {len(shared)} shared full-pool rows, worst {worst:.2e}")
print(f"GATE OK: {len(shared)} full-pool rows reproduce supp_fullspike (worst {worst:.1e}), "
      f"{len({k[0] for k in shared})} models")
cd1 = [{**r, "variant": REN[r["variant"]]} for r in run if r["variant"] in REN]
os.makedirs(OUT, exist_ok=True)
for name, part in (("list_macro_summary_aa20.csv", [r for r in cd1 if r["model"] != ENS]),
                   ("list_macro_summary_ensemble.csv", [r for r in cd1 if r["model"] == ENS])):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr, lineterminator="\n"); w.writeheader(); w.writerows(part)
assert any(r["model"] == ENS for r in cd1), "no CV x10 row"
open(f"{OUT}/ensemble_figure_rows.tsv", "w").write(
    f"{ENS}\tantiGen, 10x val-selected epoch (mean of log-probs)\tVal-selected, averaged over 10 holdouts\n")
q = [r for r in cd1 if r["qtile"] == "0.10" and r["model"] == ENS]
for r in q:
    print(f"  {ENS} {r['variant']}: {float(r['macro']):.3f}/{float(r['micro']):.3f} "
          f"({r['n_cells']} cells, {r['n_events']} events)")
print(f"staged {len({r['model'] for r in cd1})} models -> {OUT}")
