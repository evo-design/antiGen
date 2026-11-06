#!/bin/bash
# Finish the counted-baseline work after bg_contrast2 lands. Idempotent; refuses to
# merge anything it cannot first prove equivalent.
#
# WHY A MERGE AT ALL. background_contrast.py has no resume and re-scores every
# scores_*.npz it finds, which is ~48 models and hours -- past quick_preemptible's 2h
# wall, with nothing written when it is killed. Lift is per-model and independent, and
# the cellsets come from the BACKBONE (hosts/controls), not from the roster, so the two
# new rungs can be scored alone and their rows merged in. That equivalence is the whole
# basis of the shortcut, so it is checked rather than assumed.
set -euo pipefail
R=/large_storage/hielab/ispecht/fitness-inference/evals_sc2_background_aa20
W=/large_storage/hielab/ispecht/fitness-inference/bg_contrast_newrungs
F=/home/ivan.specht/antiGen/antiGen-figures

[ -f "$W/background_summary.csv" ] || { echo "bg_contrast2 has not written a summary yet" >&2; exit 1; }

echo "== 1. cellsets must be identical to the published run =="
python3 - "$W" "$F" <<'PY'
import csv, sys
W, F = sys.argv[1], sys.argv[2]
def load(p):
    return {(r['cellset'], r['pos1'], r['alt']): r['n_hosts']
            for r in csv.DictReader(open(p))}
a = load(f"{W}/background_cellsets.csv")
b = load(f"{F}/data/evals_sc2_background_aa20/background_cellsets.csv")
if set(a) != set(b):
    sys.exit(f"CELLSETS DIFFER: isolated {len(a)} vs published {len(b)}; shortcut invalid")
bad = [k for k in a if a[k] != b[k]]
if bad:
    sys.exit(f"HOST COUNTS DIFFER on {len(bad)} cell(s); shortcut invalid")
print(f"  OK: {len(a)} cells, identical sets and host counts")
PY

echo "== 2. merge the two rows into summary and percell (backup first) =="
for f in background_summary.csv background_percell.csv; do
  cp "$R/$f" "$R/$f.pre_counted"
  python3 - "$R/$f" "$W/$f" <<'PY'
import csv, sys
dst, src = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(dst))); hdr = rows[0].keys()
keep = [r for r in rows if r['model'] not in ('site_substitution', 'mcm_interpolation')]
new  = [r for r in csv.DictReader(open(src))
        if r['model'] in ('site_substitution', 'mcm_interpolation')]
w = csv.DictWriter(open(dst, 'w', newline=''), fieldnames=list(hdr))
w.writeheader(); w.writerows(keep + new)
print(f"  {dst.split('/')[-1]}: {len(keep)} kept + {len(new)} new = {len(keep)+len(new)}")
PY
done

echo "== 3. stage into the figures tree =="
cp "$R/background_summary.csv" "$R/background_percell.csv" "$R/background_cellsets.csv" \
   "$F/data/evals_sc2_background_aa20/"
echo "  staged"

echo "== 4. render =="
cd "$F"
RS=/home/ivan.specht/miniconda/envs/antigen-figs/bin/Rscript
"$RS" figures/fig_coguk_list_aa20.R  >/dev/null 2>&1 && echo "  coguk_list_aa20 ok"
"$RS" figures/fig_sc2_revised.R      >/dev/null 2>&1 && echo "  sc2_results_revised ok"

echo "== 5. did the rungs land in every panel? =="
for f in data/evals_sc2_background_aa20/background_summary.csv \
         data/evals_sc2_lists_macro_aa20/list_macro_summary_aa20.csv \
         data/gating_aa20/2021-01-01/quantile_hit_rates_all_all-models.csv; do
  n=$(cut -d, -f1 "$f" | sort -u | grep -cE '^(site_substitution|mcm_interpolation)$')
  printf "  %-62s %s/2\n" "$(basename $(dirname $f))" "$n"
done
echo "DONE -- review figs/sc2_results_revised.png"
