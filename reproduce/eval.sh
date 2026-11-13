#!/bin/bash
# Every evaluation behind the figures, from the checkpoints, one named step at a time.
#
#   bash reproduce/eval.sh                      # list the steps
#   bash reproduce/eval.sh <step> [cutoff ...]  # submit one (cutoff steps: all cutoffs by default)
#   DRY=1 bash reproduce/eval.sh <step>         # print the job, submit nothing
#   PARTITION=... TIME=01:55:00 bash ...        # another partition / time limit (list_macro resumes)
#
# Each step submits Slurm job(s) writing to $OUT/<step>[/<cutoff>] (config.sh), and
# reproduce/stage.sh turns those outputs into antiGen-figures/data/. REPRODUCE.md maps
# each figure to its staged data, its step and its checkpoints. Run
# `bash reproduce/stage_checkpoints.sh` first: the steps read the checkpoint dirs it
# lays out.
#
# THE TOP-q RULE. Most of the published outputs were produced before a later change to
# how ties at the top-q boundary are counted, and are reproduced with
# ANTIGEN_TOPQ_LEGACY=1; the steps marked [current rule] were run after it and are
# reproduced without. Every step sets this for you.
#
# FITTED BASELINES (substitution, where x which, PSSM, codon, mcm, ...) are fitted on the
# train corpus only, the same data the checkpoints were trained on.
#
# list_macro steps write list_macro_summary.csv (+ list_macro_percell.csv) and RESUME
# from it: resubmitting a step that was cut short scores only the missing models.
set -euo pipefail
. "$(dirname "$0")/config.sh"
cd "$REPO"; mkdir -p "$LOGS" "$OUT"

# ---------------------------------------------------------------------------------
# helpers

# submit NAME TIME gpu|cpu LEGACY(0|1) COMMAND...
submit() {
    local name=$1 time=${TIME:-$2} kind=$3 legacy=$4; shift 4   # TIME= overrides every limit
    local cmd="set -euo pipefail; cd $REPO; export DATA_DIR=''"
    [ "$legacy" = 1 ] && cmd="$cmd ANTIGEN_TOPQ_LEGACY=1"
    cmd="$cmd; $*"
    local res=(--partition="$PARTITION" --gres=gpu:1 --cpus-per-task=8 --mem=192G)
    [ "$kind" = cpu ] && res=(--partition="$CPU_PARTITION" --cpus-per-task=8 --mem=64G)
    [ -n "$EXCLUDE" ] && res+=(--exclude="$EXCLUDE")
    if [ "${DRY:-0}" = 1 ]; then
        printf '\n# job rp_%s (%s, %s, legacy=%s)\n%s\n' "$name" "$kind" "$time" "$legacy" "$cmd"
        return
    fi
    # A bash job script per step (sbatch --wrap runs /bin/sh), kept beside the logs as
    # the record of exactly what ran.
    local script=$LOGS/rp_$name.sh
    printf '#!/bin/bash\n%s\n' "$cmd" > "$script"
    sbatch --parsable --job-name="rp_$name" --time="$time" "${res[@]}" \
        --output="$LOGS/rp_${name}_%j.out" --error="$LOGS/rp_${name}_%j.err" "$script"
}

# list_macro.py exactly as reproduce/slurm/eval_sc2_list_macro.sh ran it:
#   lm RANGE CKPT_DIR OUT_DIR [list_macro args...]
lm() {
    local range=$1 ck=$2 out=$3; shift 3
    echo "$PYTHON -u antiGen/list_macro.py --dataset sc2 --eval-range $range" \
         "--checkpoint-dir $ck --output-dir $out" \
         "--qtiles 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --percell-qtile 0.1" \
         "--mutation-list coguk=$COGUK --alphabet $ALPHABET $*"
}

# the corpus of cutoff D: its train corpus against the shared test tree, cut at D
split_args() {
    echo "--train-examples $(train_ex "$1") --test-examples $TEST_ALL" \
         "--root-counts $(counts "$1") --split-date $1"
}
ARM2021="--train-examples $TRAIN_2021 --test-examples $TEST_2021 --root-counts $COUNTS_2021"

# ---------------------------------------------------------------------------------
# steps

# 2021-01-01 arm, COG-UK list, the whole 2021 roster; dumps every model's score maps
# for the DMS comparison (dms_dumps); a second pass adds the antiGen (CV x10) members'
# maps to the same cache. 2021 list_macro runs used its default
# substitution seed (14); the 2023 and per-cutoff runs pass 0, matching eval.py.
list2021() {
    submit list2021 12:00:00 gpu 1 "$(lm default "$CKPT/ckpt_stage_aa20" "$OUT/list2021" \
        $ARM2021 --multitree-baselines --counted-baselines --dump-scores "$OUT/list2021_maps")
$(lm default "$CKPT/ckpt_stage_vs10_aa20" "$OUT/list2021_vs10" $ARM2021 --skip-baselines --dump-scores "$OUT/list2021_maps")"
}

# The DMS panels' per-model predictions at the BA.2 clade root, from list2021's maps.
dms_dumps() {
    local d="$PYTHON reproduce/tools/dump_variant_root_predictions.py --maps-dir $OUT/list2021_maps --examples $TEST_2021"
    submit dms_dumps 02:00:00 cpu 1 "$d --output-dir $OUT/dms_dumps/models
$d --output-dir $OUT/dms_dumps/ensembles --rule-json $REPO/reproduce/rules/rules_vs10_2021.json"
}

# 2021 arm, COG-UK list, the 1-nt (codon distance 1) variants.
cd1_2021() {
    submit cd1_2021 06:00:00 gpu 1 "$(lm default "$CKPT/evals_cd1_2021/ckpt" "$OUT/cd1_2021" \
        $ARM2021 --multitree-baselines --counted-baselines --cd1-variants \
        --ensemble-dir "$CKPT/ckpt_stage_vs10_aa20" --ensemble-name ens_vs10_2021)"
}

# 2021 arm, every RBD cell, 1-nt variants. [current rule]
cd1_2021_allcells() {
    submit cd1_2021_allcells 06:00:00 gpu 0 "$(lm default "$CKPT/evals_cd1_2021/ckpt" "$OUT/cd1_2021_allcells" \
        $ARM2021 --mutation-list allcells=$RBD_CELLS --multitree-baselines --counted-baselines --cd1-variants \
        --ensemble-dir "$CKPT/ckpt_stage_vs10_aa20" --ensemble-name ens_vs10_2021)"
}

# 2023-01-01, every RBD cell: the full roster (fine-tunes included) and antiGen (CV x10),
# with the 1-nt variants. Feeds sc2_results_revised2 C-F, its supplement and the cd1 supps.
rbd_2023() {
    submit rbd_2023 12:00:00 gpu 1 "$(lm default "$CKPT/ckpt_stage_rbd_2023-01-01" "$OUT/rbd_2023-01-01" \
        $(split_args 2023-01-01) --substitution-seed 0 --mutation-list allcells=$RBD_CELLS \
        --multitree-baselines --counted-baselines --mcm-baseline --cd1-variants \
        --ensemble-dir "$CKPT/ckpt_stage_vs10_2023-01-01" --ensemble-name ens_vs10_2023)"
}

# 2023-01-01 mean-rule ensembles of eight members: dump the members' maps, then combine.
ens8_2023() {
    local D=2023-01-01 o=$OUT/ens8_2023-01-01
    local eight=simple_d128_n2_lr0.0001,eve_ft_lr0.0001,tranception_ft_both_lr0.0001,esm2_esm2_t12_35M_UR50D_lr0.0001_partial4,esm2_esm2_t12_35M_UR50D_lr0.001_frozen,esm2_esm2_t12_35M_UR50D_lr1e-05_full,esm2_esm2_t33_650M_UR50D_lr0.0001_partial4,esm2_esm2_t33_650M_UR50D_lr0.001_frozen
    submit ens8_2023 12:00:00 gpu 1 "$(lm default "$CKPT/ckpt_stage_ens8_$D" "$o/members" \
        $(split_args $D) --substitution-seed 0 --mutation-list allcells=$RBD_CELLS \
        --skip-baselines --dump-scores "$o/maps")
$PYTHON -u antiGen/ensemble_report.py --substrate list --maps-dir $o/maps --alphabet $ALPHABET \
    --root-counts $(counts $D) --mutation-list allcells=$RBD_CELLS --list allcells \
    --members $eight --compare $eight --rule-json $REPO/reproduce/rules/rules_mean3_8.json \
    --ungrounded-queries drop --output-dir $o/report"
}

# 2023-01-01, the whole spike (every one of 1,273 x 20 cells), with the 1-nt variants.
fullspike_2023() {
    submit fullspike_2023 12:00:00 gpu 1 "$(lm full "$CKPT/ckpt_stage_bg_2023-01-01" "$OUT/fullspike_2023-01-01" \
        $(split_args 2023-01-01) --substitution-seed 0 --mutation-list allcells=$SPIKE_CELLS \
        --multitree-baselines --counted-baselines --mcm-baseline --cd1-variants \
        --ensemble-dir "$CKPT/ckpt_stage_vs10_2023-01-01" --ensemble-name ens_vs10_2023)"
}

# Cutoff series, per mutation (antiGen-eval): sc2_results_revised2's per-mutation panel.
cutoff_eval() {
    for D in "$@"; do
        submit cutoff_eval_$D 24:00:00 gpu 1 "antiGen-eval --dataset sc2 \
            --checkpoint-dir $CKPT/cutoff_stage_aa20/$D --output-dir $OUT/cutoff_eval/$D \
            --priority-models blosum62,esm2 --alphabet $ALPHABET $(split_args "$D") \
            --variant-counts-dir $BA2_COUNTS --counts-since $D \
            --multitree-baselines --counted-baselines --mcm-baseline"
    done
}

# Cutoff series, per cell over every RBD cell: the per-cell cutoff panel.
cutoff_rbd() {
    for D in "$@"; do
        submit cutoff_rbd_$D 12:00:00 gpu 1 "$(lm default "$CKPT/cutoff_stage_aa20/$D" "$OUT/cutoff_rbd/$D" \
            $(split_args "$D") --substitution-seed 0 --mutation-list allcells=$RBD_CELLS \
            --multitree-baselines --counted-baselines --mcm-baseline)"
    done
}

# Cutoff series on the whole spike (supp_fullspike). At 2023-01-01 the roster is the
# whole 2023 one without fine-tunes, plus antiGen (CV x10).
cutoff_spike() {
    for D in "$@"; do
        local ck=$CKPT/cutoff_stage_aa20/$D ens=""
        [ "$D" = 2023-01-01 ] && { ck=$CKPT/ckpt_stage_bg_$D
            ens="--ensemble-dir $CKPT/ckpt_stage_vs10_$D --ensemble-name ens_vs10_2023"; }
        submit cutoff_spike_$D 12:00:00 gpu 1 "$(lm full "$ck" "$OUT/cutoff_spike/$D" \
            $(split_args "$D") --substitution-seed 0 --mutation-list allcells=$SPIKE_CELLS \
            --multitree-baselines --counted-baselines --mcm-baseline $ens)"
    done
}

# 1-nt series, de novo cells (supp_cd1_series, de novo arm; 2020-04-01 .. 2021-10-01). [current rule]
cd1_series_denovo() {
    for D in "$@"; do
        submit cd1dn_$D 12:00:00 gpu 0 "$(lm default "$CKPT/cutoff_stage_aa20/$D" "$OUT/cd1_series_denovo/$D" \
            $(split_args "$D") --substitution-seed 0 --mutation-list allcells=$RBD_CELLS \
            --multitree-baselines --counted-baselines --mcm-baseline --cd1-variants)"
    done
}

# 1-nt series, cells seen in < 100 training sequences (supp_cd1_series rare arm, and
# supp_lm_cd1 from the same runs' unmasked 1-nt variant). [current rule]
cd1_series_rare100() {
    for D in "$@"; do
        submit cd1r_$D 12:00:00 gpu 0 "$(lm default "$CKPT/cutoff_stage_aa20/$D" "$OUT/cd1_series_rare100/$D" \
            $(split_args "$D") --substitution-seed 0 --mutation-list allcells=$RBD_CELLS \
            --multitree-baselines --counted-baselines --mcm-baseline --cd1-variants --seen-min-count 100)"
    done
}

# Post-cutoff residue counts over all lineages, the >= 100-sequence masks of the cd1 series.
postcutoff_counts() {
    submit postcutoff_counts 04:00:00 cpu 0 "$PYTHON reproduce/tools/postcutoff_counts.py $OUT/postcutoff_counts"
}

# Background discrimination + contrast (supp_query_adapt), 2021 arm.
background_2021() {
    local o=$OUT/background_2021
    submit background_2021 24:00:00 gpu 1 "$PYTHON antiGen/background_discrimination.py --dataset sc2 \
    --tree-json $REPO/data/mutation-annotated-trees/sc2_guide.json \
    --ref-json $REPO/data/mutation-annotated-trees/sc2_guide_ref.json \
    --checkpoint-dir $CKPT/ckpt_stage_aa20 --output-dir $o --alphabet $ALPHABET \
    --train-examples $TRAIN_2021 --root-counts $COUNTS_2021 --backbone-examples $TEST_2021 \
    --multitree-baselines --counted-baselines --mcm-baseline \
    --ensemble-dir $CKPT/ckpt_stage_vs10_aa20 --ensemble-name ens_8a91fbbf88
$PYTHON -u antiGen/background_contrast.py --background-dir $o --root-counts $COUNTS_2021 \
    --mutation-list coguk=$COGUK --reference-query 0 --min-hosts 1 --min-controls 1
$PYTHON reproduce/tools/background_event_weighted.py --background-dir $o --output-dir $o/check_unweighted --weight none --check
$PYTHON reproduce/tools/background_event_weighted.py --background-dir $o --output-dir ${o}_eventwtd --weight hosts
cp $o/background_cellsets.csv ${o}_eventwtd/"
}

# Background discrimination + contrast (supp_query_adapt), 2023-01-01.
background_2023() {
    local D=2023-01-01 o=$OUT/background_2023-01-01
    submit background_2023 24:00:00 gpu 1 "$PYTHON antiGen/background_discrimination.py --dataset sc2 \
    --checkpoint-dir $CKPT/ckpt_stage_bg_$D --output-dir $o --alphabet $ALPHABET \
    --train-examples $(train_ex $D) --backbone-examples $TEST_ALL --root-counts $(counts $D) \
    --split-date $D --substitution-seed 0 --multitree-baselines --counted-baselines --mcm-baseline \
    --ensemble-dir $CKPT/ckpt_stage_vs10_$D --ensemble-name ens_vs10_2023
REF=\$($PYTHON -I -c \"import json; q=json.load(open('$o/backbone_manifest.json'))['variant_root_query']; assert q >= 0, q; print(q)\")
$PYTHON -u antiGen/background_contrast.py --background-dir $o --root-counts $(counts $D) \
    --mutation-list coguk=$COGUK --reference-query \$REF --min-hosts 1 --min-controls 1
$PYTHON reproduce/tools/background_event_weighted.py --background-dir $o --output-dir $o/check_unweighted --weight none --check
$PYTHON reproduce/tools/background_event_weighted.py --background-dir $o --output-dir ${o}_eventwtd --weight hosts
cp $o/background_cellsets.csv ${o}_eventwtd/"
}

# antiGen-data seed study (supp_dataseed): antiGen (CV x10) retrained on the 2023-01-01
# train corpus rebuilt under antigen --seed k, scored as C-J are (+ 1-nt variants). Seed 1
# is the production corpus, i.e. the published ensemble; the corpora are built by
# antiGen-data's seed_study.sbatch and the members by reproduce/slurm/submit_dataseed_2023.sh.
dataseed() {
    for k in "$@"; do
        local ens=$CKPT/dataseed_2023-01-01/seed_$k/stage name=ens_vs10_2023_dseed$k
        [ "$k" = 1 ] && { ens=$CKPT/ckpt_stage_vs10_2023-01-01; name=ens_vs10_2023; }
        submit dataseed_$k 02:00:00 gpu 1 "mkdir -p $OUT/dataseed/empty; $(lm default "$OUT/dataseed/empty" "$OUT/dataseed/seed_$k/eval_cd1" \
            $(split_args 2023-01-01) --substitution-seed 0 --mutation-list allcells=$RBD_CELLS \
            --cd1-variants --baselines-only --skip-baselines --ensemble-dir $ens --ensemble-name $name)"
    done
}

# Fixed train/test splits (supp_fixed_split_revised2): the base antiGen of cutoff D
# against the fixed external maps, COG-UK list. At 2024-10-01 the per-query CoVFit maps
# join if built (reproduce/slurm/submit_covfit_perquery.sh STAGE=gate|score|eval).
fixedsplit() {
    for D in "$@"; do
        local q=""; [ -f "$OUT/covfit_perquery_$D/covfit_query.npz" ] && q="--query-maps CoVFit_query=$OUT/covfit_perquery_$D/covfit_query.npz"
        submit fixedsplit_$D 06:00:00 gpu 1 "$PYTHON -u antiGen/list_macro.py --dataset sc2 --alphabet $ALPHABET \
            $(split_args "$D") --checkpoint-dir $CKPT/cutoff_series/$D/base --output-dir $OUT/fixedsplit/$D \
            --mutation-list coguk=$COGUK $q"
    done
}

# Training-set size per cutoff in three units (supp_gate_units' x axes).
gate_units() {
    submit gate_units 02:00:00 cpu 0 "mkdir -p $OUT/gate_units; $PYTHON reproduce/tools/training_size_by_cutoff.py \
        --data $CORPORA --cutoffs $(echo $CUTOFFS | tr ' ' ,) --out $OUT/gate_units/training_size_by_cutoff.csv"
}

# Panel B: rank each step of BA.2.86 -> KP.3 with the 2023-01-01 base model, then draw it.
panel_b() {
    local ck; ck=$(ls "$CKPT"/cutoff_stage_aa20/2023-01-01/fitness_simple_d128_n2_lr0.0001_*.pt)
    submit panel_b 01:00:00 cpu 0 "mkdir -p $OUT/panel_b
$PYTHON reproduce/tools/rank_path_cells.py --path-csv antiGen-figures/data/ba286_to_kp3_path.csv \
    --examples $TEST_2021 --checkpoint $ck --output $OUT/panel_b/ba286_to_kp3_ranks_2023-01-01.csv
$PYTHON reproduce/tools/draw_path_panel.py --ranks $OUT/panel_b/ba286_to_kp3_ranks_2023-01-01.csv \
    --pdb antiGen-figures/ref/6M0J.pdb --output $OUT/panel_b/ba286_to_kp3.png"
}

# ---------------------------------------------------------------------------------
STEPS="list2021 dms_dumps cd1_2021 cd1_2021_allcells rbd_2023 ens8_2023 fullspike_2023
       cutoff_eval cutoff_rbd cutoff_spike cd1_series_denovo cd1_series_rare100
       postcutoff_counts background_2021 background_2023 dataseed fixedsplit gate_units panel_b"
if [ $# = 0 ]; then
    echo "steps (bash reproduce/eval.sh <step> [cutoff ...]):"
    for s in $STEPS; do echo "  $s"; done
    exit 0
fi
step=$1; shift
case " $(echo $STEPS) " in *" $step "*) ;; *) echo "unknown step: $step" >&2; exit 1 ;; esac
case $step in
    cutoff_eval|cutoff_rbd|cutoff_spike|cd1_series_rare100) [ $# = 0 ] && set -- $CUTOFFS ;;
    cd1_series_denovo) [ $# = 0 ] && set -- $DENOVO_CUTOFFS ;;
    fixedsplit) [ $# = 0 ] && set -- $FIXED_SPLITS ;;
    dataseed) [ $# = 0 ] && set -- 1 2 3 ;;
esac
"$step" "$@"
