#!/bin/bash
# Stage the codon-distance-1 run (NOTES s159) for supp_cd1: sc2_results_revised2's
# panels C-F with pool AND targets restricted, per query, to substitutions one
# nucleotide change from the query's wild-type residue. The drawer reads
# list_allcells_fullpool / list_allcells_unseen_fullpool, so the cd1 variants are staged
# UNDER THOSE NAMES (and the unrestricted ones dropped) into their own data dir, laid out
# as stage_allcells_2023.sh lays out the main arm. The s157/s158 blend rungs scored in
# the same run are left out: this figure is the main figure's roster.
set -euo pipefail
S=${S:-/large_storage/hielab/ispecht/fitness-inference/evals_cd1_2023-01-01/out}
D=${D:-/home/ivan.specht/antiGen/antiGen-figures/data/evals_sc2_lists_macro_aa20_2023-01-01_cd1}
ENS=${ENS:-ens_vs10_2023}
mkdir -p "$D"
python3 - "$S/list_macro_summary.csv" "$D" "$ENS" <<'PY'
import csv, sys
src, out, ens = sys.argv[1:]
EXTRA = {"joint_cv3", "covonly", "blend_r1_AJC", "blend_r2_AJK", "blend_dn_CK"}
REN = {"list_allcells_cd1_fullpool": "list_allcells_fullpool",
       "list_allcells_unseen_cd1_fullpool": "list_allcells_unseen_fullpool"}
rows = list(csv.DictReader(open(src))); hdr = list(rows[0].keys())
keep = []
for r in rows:
    if r["model"] in EXTRA or r["variant"] not in REN:
        continue
    keep.append(dict(r, variant=REN[r["variant"]]))
e = [r for r in keep if r["model"] == ens]
o = [r for r in keep if r["model"] != ens]
assert e, f"no {ens} row"
for name, part in (("list_macro_summary_aa20.csv", o), ("list_macro_summary_ensemble.csv", e)):
    w = csv.DictWriter(open(f"{out}/{name}", "w", newline=""), fieldnames=hdr)
    w.writeheader(); w.writerows(part)
open(f"{out}/ensemble_figure_rows.tsv", "w").write(
    f"{ens}\tantiGen, 10x val-selected epoch (mean of log-probs)\t"
    "Val-selected, averaged over 10 holdouts\n")
print(f"staged {len({r['model'] for r in o})} models + {ens} -> {out}")
PY
