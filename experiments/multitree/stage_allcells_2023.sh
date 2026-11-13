#!/bin/bash
# Stage the 2023-01-01 all-cells run for sc2_results_revised2 panels I-L (NOTES s125).
#
# The drawer reads every list_macro_summary_*.csv in its macro dir, plus an ensemble
# CSV and ensemble_figure_rows.tsv for `ens_*` rows -- the layout
# data/evals_sc2_lists_macro_aa20/ has. So the val-selected ensemble row is split out
# of the run's one summary into its own file, and the figure-rows TSV names it.
set -euo pipefail
S=${S:-/large_storage/hielab/ispecht/fitness-inference/evals_sc2_lists_macro_aa20_2023-01-01_allcells_pooled}
D=${D:-/home/ivan.specht/antiGen/antiGen-figures/data/evals_sc2_lists_macro_aa20_2023-01-01_allcells}
ENS=${ENS:-ens_vs10_2023}
# The three mean rules over the nine-member family (NOTES s141), once finish_ft_2023.sh
# has written them; absent until then, and the figure draws those bars empty.
REPORT=${REPORT:-/large_storage/hielab/ispecht/fitness-inference/ensembles8_aa20_2023-01-01/report/list_macro_summary_ensemble.csv}
mkdir -p "$D"
python3 - "$S/list_macro_summary.csv" "$D" "$ENS" "$REPORT" <<'PY'
import csv, sys
import os
src, out, ens, report = sys.argv[1:]
rows = list(csv.DictReader(open(src))); hdr = list(rows[0].keys())
e = [r for r in rows if r["model"] == ens]
o = [r for r in rows if r["model"] != ens]
if not e:
    sys.exit(f"no {ens} row in {src}")
means = []
if os.path.exists(report):
    means = [{k: r.get(k, "") for k in hdr} for r in csv.DictReader(open(report))
             if r["model"].startswith("ens_") and r["model"] != ens]
    print(f"mean rules from {report}: {sorted({r['model'] for r in means})}")
e += means
for name, part in (("list_macro_summary_aa20.csv", o), ("list_macro_summary_ensemble.csv", e)):
    w = csv.DictWriter(open(f"{out}/{name}", "w", newline=""), fieldnames=hdr)
    w.writeheader(); w.writerows(part)
open(f"{out}/ensemble_figure_rows.tsv", "w").write(
    f"{ens}\tantiGen, 10x val-selected epoch (mean of log-probs)\t"
    "Val-selected, averaged over 10 holdouts\n")
print(f"staged {len({r['model'] for r in o})} models + {ens} -> {out}")
PY
