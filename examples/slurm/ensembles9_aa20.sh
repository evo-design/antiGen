#!/bin/bash
#SBATCH --job-name=ens9_aa20
#SBATCH --partition=gpu,gpu_batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --output=logs/ens9_aa20_%j.out
#SBATCH --error=logs/ens9_aa20_%j.err

# The nine-member roster ensembles (NOTES s49/s50) on the revised 20-letter corpus, so
# `coguk_list_aa20` carries the three "Mean of ..." bars the published figure carries.
#
# WHY ONLY NOW. The nine members are antiGen, EVE full FT, Tranception full FT and the
# six ESM-2 arms. Until s62 scored EVE and Tranception on this corpus, two of the nine
# had no cached map here and the family could not be formed at all. No forward pass is
# needed now: everything is read from `list_score_maps_aa20`.
#
# THE RULES ARE CARRIED, AND THAT IS A CLAIM THIS JOB CHECKS RATHER THAN ASSUMES. The
# family is nine parameter-free rules -- three normalisations x three poolings -- and the
# published figure draws the three ARITHMETIC ones, a choice s49/s50 made on the
# published corpus's SELECT cells (the ~1,235 observed RBD cells NOT on the CoG-UK list).
# Carrying that choice to a different corpus is importing a selection made elsewhere,
# which is exactly what s54's Open warned about. So stage 2 re-scores all nine on THIS
# corpus's own select cells before stage 3 draws anything, and the two columns are
# printed side by side. If the means do not lead on select here, that is a finding and it
# goes in the notes beside the bars -- it does not quietly change which bars are drawn,
# because choosing the rule on the cells the figure reports is the one thing this whole
# design exists to prevent.
#
# TWO REPORTS, MERGED, RATHER THAN ONE OVER ELEVEN RULES. `ensemble_report` writes the
# figure's ensemble files WHOLESALE, so a run over the nine alone would delete s56's
# antiGen bar and s57's hybrid bar rather than join them -- but putting all eleven rules
# in ONE run is wrong for a subtler reason that a first attempt at this job walked into.
#
# The auto-added "panel incumbent" comparator is the MAX OVER `sub.members`, i.e. over
# every member whose map the run loaded. The published nine-member report loaded exactly
# nine, so its incumbent is "the best single rung of the family" -- which is the
# comparison that makes an ensemble bar mean anything. A combined run also loads the ten
# vs10Antigen runs and the ten hybrid blends, so the incumbent for the FAMILY's bars
# became `vs10Antigen_s4`: the best of ten unlabelled runs of a different arm, picked by
# maximum. Not wrong, and thoroughly misleading.
#
# So each arm is reported with its own member set -- the family against its own nine,
# the two carried rules against the eleven and sixteen rungs s56/s57 used -- and the
# four output files are concatenated. The rule_ids are disjoint, both runs score the same
# cells, and stage 5 gates both of those before writing anything.
#
# WHAT IS DRAWN IS A SUBSET OF WHAT IS REPORTED, which is the published convention and
# not a trim for flattery: the CSV and the contrasts carry all nine rules including the
# six losers, and the row list -- which is the only thing the figure reads -- carries the
# three means plus the two val-selected bars. The published directory has exactly this
# shape: nine rule_ids in `list_macro_summary_ensemble.csv`, three rows in
# `ensemble_figure_rows.tsv`.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"
set -o pipefail

REPO=${REPO:-/home/ivan.specht/antiGen}
REV=${REV:-/home/ivan.specht/antiGen-data-revised/output/sc2-2021/train}
MAPS=${MAPS:-/large_storage/hielab/ispecht/fitness-inference/list_score_maps_aa20}
WORK=${WORK:-/large_storage/hielab/ispecht/fitness-inference/ensembles9_aa20}
ALPHABET=${ALPHABET:-ACDEFGHIKLMNPQRSTVWY}
N=${N:-10}
cd "$REPO" || exit 1
mkdir -p "$WORK/report" "$WORK/sweep"

echo ""
echo "=== 1. the rule files, built against the maps that actually exist ==="
python -I - "$MAPS" "$WORK" "$N" <<'PY' || exit 1
import json, os, sys
maps, work, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

# The nine members of NOTES s49: base antiGen and its eight fine-tunes, which are the two
# leftmost legend groups of the figure. The two *_noft rungs are excluded because they
# are not fine-tunes, and everything outside the family because it is not in it.
MEMBERS = ["simple_d128_n2_lr0.0001", "eve_ft_lr0.0001", "tranception_ft_both_lr0.0001",
           "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4",
           "esm2_esm2_t12_35M_UR50D_lr0.001_frozen",
           "esm2_esm2_t12_35M_UR50D_lr1e-05_full",
           "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4",
           "esm2_esm2_t33_650M_UR50D_lr0.001_frozen",
           "esm2_esm2_t33_650M_UR50D_lr1e-05_full"]
NORMS = [("pctile", "ranks"), ("logsoftmax", "log-probs"), ("zscore", "z-scores")]
POOLS = [("arith", "Mean"), ("max", "Max"), ("min", "Min")]
ENS_GROUP = "Ensembles"
VS_GROUP = "Val-selected, averaged over 10 holdouts"

def check(members, what):
    missing = [m for m in members
               if not os.path.exists(os.path.join(maps, f"scores_{m}.npy"))]
    if missing:
        print(f"ABORT: {what} names {len(missing)} member(s) with no cached map:")
        for m in missing:
            print(f"  scores_{m}.npy")
        raise SystemExit(1)

check(MEMBERS, "the nine-member family")
sweep, report = [], []
for norm, nlab in NORMS:
    for pool, plab in POOLS:
        sweep.append({"norm": norm, "pool": pool, "nan": "union_min",
                      "members": MEMBERS})
        report.append({"norm": norm, "pool": pool, "nan": "union_min",
                       "members": MEMBERS,
                       "figure_label": f"{plab} of {nlab}",
                       "figure_group": ENS_GROUP})

# s56 and s57, carried so the report does not delete their bars. Their members are ten
# runs apiece and are NOT drawn on this figure, so they get their own legend group: the
# "an ensemble bar sits beside its own parts" caveat is true of the nine and false of
# these two, and one noun cannot be right for both.
vs = [("vs10Antigen", f"antiGen, {n}x val-selected epoch (mean of log-probs)"),
      ("hybridWxwPssm", f"Where x which + PSSM hybrid, {n}x val-selected blend")]
for prefix, label in vs:
    members = [f"{prefix}_s{i}" for i in range(n)]
    check(members, label)
    report.append({"norm": "logsoftmax", "pool": "arith", "nan": "union_min",
                   "members": members, "figure_label": label,
                   "figure_group": VS_GROUP})

json.dump(sweep, open(os.path.join(work, "rules9_sweep.json"), "w"), indent=1)
json.dump(report[:9], open(os.path.join(work, "rules9_report.json"), "w"), indent=1)
json.dump(report[9:], open(os.path.join(work, "rules2_report.json"), "w"), indent=1)
# The bars. The three arithmetic rules of the family, plus the two carried arms.
drawn = [f"{p} of {n_}" for p, n_ in [("Mean", "ranks"), ("Mean", "log-probs"),
                                      ("Mean", "z-scores")]] + [l for _, l in vs]
open(os.path.join(work, "drawn_labels.txt"), "w").write("\n".join(drawn) + "\n")
print(f"wrote rules9_sweep.json ({len(sweep)} rules), rules9_report.json (9), "
      f"rules2_report.json ({len(report) - 9}), drawn_labels.txt ({len(drawn)})")
PY

echo ""
echo "=== 2. the transfer check: all nine rules on THIS corpus's select cells ==="
# `--rules-json` scores exactly these rules and runs no search. The substrate scores the
# report cells (the curated list, which panels C-F draw) and the select cells (observed
# RBD cells NOT on the list) in one pass, so the carried choice can be read against the
# cells it was never allowed to see.
python -u "$REPO/antiGen/ensemble_sweep.py" \
    --substrate list --maps-dir "$MAPS" \
    --alphabet "$ALPHABET" \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --rules-json "$WORK/rules9_sweep.json" \
    --ungrounded-queries drop \
    --device cuda \
    --output-dir "$WORK/sweep" || { echo "SWEEP FAILED"; exit 1; }

NINE=simple_d128_n2_lr0.0001,eve_ft_lr0.0001,tranception_ft_both_lr0.0001,esm2_esm2_t12_35M_UR50D_lr0.0001_partial4,esm2_esm2_t12_35M_UR50D_lr0.001_frozen,esm2_esm2_t12_35M_UR50D_lr1e-05_full,esm2_esm2_t33_650M_UR50D_lr0.0001_partial4,esm2_esm2_t33_650M_UR50D_lr0.001_frozen,esm2_esm2_t33_650M_UR50D_lr1e-05_full

echo ""
echo "=== 3a. the family, against its own nine members ==="
# --members is EXACTLY the nine, so the auto-added panel incumbent is the best single
# rung of the family -- "max over 9 rungs", as the published report reads.
python -u "$REPO/antiGen/ensemble_report.py" \
    --substrate list --maps-dir "$MAPS" \
    --alphabet "$ALPHABET" \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --members "$NINE" \
    --compare "$NINE" \
    --rule-json "$WORK/rules9_report.json" \
    --ungrounded-queries drop \
    --output-dir "$WORK/report9" || { echo "REPORT (family) FAILED"; exit 1; }

echo ""
echo "=== 3b. the two carried arms, on the member sets s56/s57 used ==="
python -u "$REPO/antiGen/ensemble_report.py" \
    --substrate list --maps-dir "$MAPS" \
    --alphabet "$ALPHABET" \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --members "vs10Antigen,hybridWxwPssm,simple_d128_n2_lr0.0001,where_times_which,pssm_root" \
    --compare "simple_d128_n2_lr0.0001,where_times_which,pssm_root" \
    --rule-json "$WORK/rules2_report.json" \
    --ungrounded-queries drop \
    --output-dir "$WORK/report2" || { echo "REPORT (carried) FAILED"; exit 1; }

echo ""
echo "=== 4. merge the two reports, gated ==="
python -I - "$WORK" <<'PY' || exit 1
import csv, os, sys
work = sys.argv[1]
a, b, out = (os.path.join(work, d) for d in ("report9", "report2", "report"))
os.makedirs(out, exist_ok=True)
for name in ("list_macro_summary_ensemble.csv", "ensemble_contrasts.csv",
             "ensemble_percell.csv"):
    ra = list(csv.DictReader(open(os.path.join(a, name))))
    rb = list(csv.DictReader(open(os.path.join(b, name))))
    fa, fb = ra[0].keys(), rb[0].keys()
    if list(fa) != list(fb):
        raise SystemExit(f"ABORT: {name} columns differ between the two reports")
    key = "model" if "model" in fa else "rule_id"
    ida, idb = {r[key] for r in ra}, {r[key] for r in rb}
    if ida & idb:
        # Two runs reporting the same rule would put one bar's evidence in the file
        # twice, and the figure would draw whichever row it read last.
        raise SystemExit(f"ABORT: {name} rule overlap between reports: {sorted(ida & idb)}")
    if name == "list_macro_summary_ensemble.csv":
        # Both runs must have scored the SAME cells, or the two halves of this file are
        # two different denominators printed in one column.
        ca = {(r["variant"], r["qtile"]): r["n_cells"] for r in ra}
        cb = {(r["variant"], r["qtile"]): r["n_cells"] for r in rb}
        bad = {k: (ca[k], cb[k]) for k in ca.keys() & cb.keys() if ca[k] != cb[k]}
        if bad:
            raise SystemExit(f"ABORT: the two reports disagree on n_cells: {bad}")
        print(f"  n_cells agree on all {len(ca.keys() & cb.keys())} (variant, qtile) keys")
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fa))
        w.writeheader()
        w.writerows(ra + rb)
    print(f"  {name}: {len(ra)} + {len(rb)} = {len(ra) + len(rb)} rows, "
          f"{len(ida)} + {len(idb)} rules")
rows = []
for d in (a, b):
    rows += [l.rstrip("\n") for l in open(os.path.join(d, "ensemble_figure_rows.tsv"))
             if l.strip()]
open(os.path.join(out, "ensemble_figure_rows.tsv"), "w").write("\n".join(rows) + "\n")
print(f"  ensemble_figure_rows.tsv: {len(rows)} rows")
PY

echo ""
echo "=== 5. the drawn rows ==="
# The report writes one row per rule it scored; the figure draws every row in this file.
# Keeping the losers in the CSV and out of the bar panels is the published shape.
python -I - "$WORK" <<'PY' || exit 1
import os, sys
work = sys.argv[1]
p = os.path.join(work, "report", "ensemble_figure_rows.tsv")
drawn = [l.strip() for l in open(os.path.join(work, "drawn_labels.txt")) if l.strip()]
rows = [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
keep = [r for r in rows if r[1] in drawn]
if len(keep) != len(drawn):
    got = {r[1] for r in rows}
    raise SystemExit(f"ABORT: {len(keep)} of {len(drawn)} drawn labels found; "
                     f"missing {sorted(set(drawn) - got)}")
with open(p, "w") as f:
    for r in keep:
        f.write("\t".join(r) + "\n")
print(f"{len(rows)} rules reported, {len(keep)} drawn:")
for r in keep:
    print(f"  {r[0]}  {r[1]}   [{r[2] if len(r) > 2 else '<default group>'}]")
PY

echo ""
echo "=== outputs ==="
for f in "$WORK/report/list_macro_summary_ensemble.csv" \
         "$WORK/report/ensemble_figure_rows.tsv" \
         "$WORK/report/ensemble_contrasts.csv" \
         "$WORK/report/ensemble_percell.csv"; do
    if [ -s "$f" ]; then echo "  OK   $f"; else echo "  MISSING $f"; exit 1; fi
done

echo ""
echo "=== to draw it:"
echo "  cp $WORK/report/{list_macro_summary_ensemble.csv,ensemble_figure_rows.tsv,ensemble_contrasts.csv,ensemble_percell.csv} \\"
echo "     $REPO/antiGen-figures/data/evals_sc2_lists_macro_aa20/"
echo "  (cd $REPO/antiGen-figures && Rscript figures/fig_coguk_list_aa20.R)"

echo "Job finished at: $(date)"
echo "ENS9_AA20 OK"
