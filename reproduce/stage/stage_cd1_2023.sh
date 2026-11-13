#!/bin/bash
# Stage a codon-distance-1 run (NOTES s159) for supp_cd1: sc2_results_revised2's bar
# panels with pool AND targets restricted, per query, to substitutions one nucleotide
# change from the query's wild-type residue. The drawer reads list_<L>_fullpool /
# list_<L>_unseen_fullpool, so the cd1 variants are staged UNDER THOSE NAMES (the
# unrestricted ones dropped) into their own data dir, laid out as stage_allcells_2023.sh
# lays out the main arm. Rungs not on the main figure's roster are left out.
#
#   ARM=2023 (default)  all RBD cells at 2023-01-01          -> C-F
#   ARM=2021            COG-UK escape list at 2021-01-01     -> G-J
# For 2021 the antiGen rung is scored natively (--ensemble-dir, ens_vs10_2021) and staged
# as the published ens_8a91fbbf88 ONLY IF its unrestricted rows reproduce the published
# ones (gate below).
set -euo pipefail
ARM=${ARM:-2023}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
FIG=${FIGDATA:-/home/ivan.specht/antiGen/antiGen-figures/data}
if [ "$ARM" = 2023 ]; then
  S=$STORE/evals_cd1_2023-01-01/out; D=$FIG/evals_sc2_lists_macro_aa20_2023-01-01_cd1
  LIST=allcells; SRC_ENS=ens_vs10_2023; ENS=ens_vs10_2023; PUB=""
else
  S=$STORE/evals_cd1_2021/out; D=$FIG/evals_sc2_lists_macro_aa20_cd1
  LIST=coguk; SRC_ENS=ens_vs10_2021; ENS=ens_8a91fbbf88
  PUB=$FIG/evals_sc2_lists_macro_aa20/list_macro_summary_ensemble.csv
fi
S=${RUN_DIR:-$S}      # reproduce/stage.sh; the published 2023 run is out_trainonly (s163)
mkdir -p "$D"
python3 - "$S/list_macro_summary.csv" "$D" "$LIST" "$SRC_ENS" "$ENS" "$PUB" <<'PY'
import csv, sys
src, out, lst, src_ens, ens, pub = sys.argv[1:]
EXTRA = {"joint_cv3", "covonly", "blend_r1_AJC", "blend_r2_AJK", "blend_dn_CK"}
REN = {f"list_{lst}_cd1_fullpool": f"list_{lst}_fullpool",
       f"list_{lst}_unseen_cd1_fullpool": f"list_{lst}_unseen_fullpool"}
rows = list(csv.DictReader(open(src))); hdr = list(rows[0].keys())
if pub:   # gate: the native ensemble must reproduce the published rule on the unrestricted rows
    old = {(r["variant"], r["qtile"]): r for r in csv.DictReader(open(pub)) if r["model"] == ens}
    new = {(r["variant"], r["qtile"]): r for r in rows if r["model"] == src_ens and r["variant"] in REN.values()}
    assert new and set(new) <= set(old), (len(new), len(old))
    w = max(abs(float(new[k][c]) - float(old[k][c])) for k in new for c in ("macro", "micro"))
    assert w < 1e-6, f"native {src_ens} does not reproduce published {ens}: worst {w}"
    print(f"GATE OK: {src_ens} reproduces published {ens} on {len(new)} rows, worst {w:.1e}")
keep = []
for r in rows:
    if r["model"] in EXTRA or r["variant"] not in REN:
        continue
    keep.append(dict(r, variant=REN[r["variant"]],
                     model=ens if r["model"] == src_ens else r["model"]))
e = [r for r in keep if r["model"] == ens]
o = [r for r in keep if r["model"] != ens]
assert e, f"no {src_ens} row"
for name, part in (("list_macro_summary_aa20.csv", o), ("list_macro_summary_ensemble.csv", e)):
    w = csv.DictWriter(open(f"{out}/{name}", "w", newline=""), fieldnames=hdr)
    w.writeheader(); w.writerows(part)
open(f"{out}/ensemble_figure_rows.tsv", "w").write(
    f"{ens}\tantiGen, 10x val-selected epoch (mean of log-probs)\t"
    "Val-selected, averaged over 10 holdouts\n")
print(f"staged {len({r['model'] for r in o})} models + {ens} -> {out}")
PY
