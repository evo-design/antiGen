#!/bin/bash
# Panels C-H of sc2_results_revised2 on the WHOLE SPIKE, for the supplement
# supp_fullspike (NOTES s139).
#
# ONE PIPELINE FOR ALL SIX PANELS. list_macro with --eval-range full and every spike cell
# listed (ref/all_spike_cells_aa20_incl_ref.csv: 1,273 sites x 20, reference allele
# included), at each of the sixteen cutoffs. With the COMPLETE list, the macro column is
# the per-cell number (C, E, G) and the micro column the per-mutation number (D, F, H):
# the same quantity eval.py's quantile_hit_rates reports, through another accumulator
# (the gate in submit_macro_allcells_aa20.sh's header). So D does not need a second
# sixteen-eval sweep.
#
# THE ROSTER, as on the RBD:
#   * every cutoff: its cutoff_stage_aa20/<D> checkpoints (base antiGen, the gate,
#     ESM-2 35M partial FT, BLOSUM62), the refitted statistical rungs and the fixed
#     maps, plus mcm (C/D carry it, NOTES s121);
#   * 2023-01-01 also carries E-H's whole roster -- ckpt_stage_bg_2023-01-01 (the
#     series checkpoints plus ESM-2 untrained, EVE, Tranception) and the val-selected
#     ensemble -- so one run serves C/D's 2023 point and E-H.
#
# Baselines defined only on the RBD (the DMS maps, EVEscape) keep their RBD cells as
# candidates while non-RBD targets count against them, so they degrade rather than
# vanish; their full-spike numbers are not like-for-like (NOTES s29).
#
# gpu_batch (CLAUDE.md), GPUCACE excluded (NOTES s125). One job per cutoff; each
# resumes from its own summary CSV, so a re-run picks up only what is missing.
set -euo pipefail

STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
LIST=$REPO/antiGen-figures/ref/all_spike_cells_aa20_incl_ref.csv
CUTOFFS=${CUTOFFS:-"2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01
         2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01
         2023-04-01 2023-07-01 2023-10-01 2024-01-01"}
TIME=${TIME:-12:00:00}
cd "$REPO"
[ "$(($(wc -l < "$LIST") - 1))" = 25460 ] || { echo "$LIST is not 25,460 cells" >&2; exit 1; }

for D in $CUTOFFS; do
    out=$STORE/evals_sc2_lists_macro_aa20_${D}_allcells_fullspike
    if [ "$D" = 2023-01-01 ]; then
        ckpts=$STORE/ckpt_stage_bg_2023-01-01
        extra="--mcm-baseline --ensemble-dir $STORE/ckpt_stage_vs10_2023-01-01 --ensemble-name ens_vs10_2023"
    else
        ckpts=$STORE/cutoff_stage_aa20/$D
        extra="--mcm-baseline"
    fi
    echo "== $D"
    CUTOFF=$D RANGE=full LIST=$LIST CKPTS=$ckpts OUTDIR=$out \
    PARTITION=gpu_batch TIME=$TIME EXTRA_LM_ARGS="$extra" \
        ./experiments/multitree/submit_macro_allcells_aa20.sh | sed 's/^/   /'
done
