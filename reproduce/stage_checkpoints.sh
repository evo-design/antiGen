#!/bin/bash
# Lay out the checkpoint directories the evals read, as symlinks into the training
# outputs, from reproduce/checkpoints.tsv.
#
#   bash reproduce/stage_checkpoints.sh            # report what is present / missing
#   bash reproduce/stage_checkpoints.sh --link     # create the missing links
#
# Each row is <link dir> <file name> <target>, paths relative to $CKPT. A target of '-'
# is a real file the training wrote there directly (it is checked, not linked). The
# targets are the training output paths of the paper's runs; retrained models get new
# timestamps in their file names, so after retraining, edit the targets (or rebuild the
# table from your own layout) before linking.
#
# The directories, and what reads them:
#   ckpt_stage_aa20               2021 arm roster (eval.py, list_macro, background)
#   ckpt_stage_vs10_aa20          2021 antiGen (CV x10) members
#   evals_cd1_2021/ckpt           2021 1-nt / de novo roster
#   ckpt_stage_bg_2023-01-01      2023 roster without the fine-tunes (background, whole spike)
#   ckpt_stage_ft_2023-01-01      2023 fine-tunes (real files)
#   ckpt_stage_ft6_cd1_2023-01-01 the six 2023 fine-tunes drawn
#   ckpt_stage_rbd_2023-01-01     the two above together: the 2023 all-RBD roster
#   ckpt_stage_vs10_2023-01-01    2023 antiGen (CV x10) members
#   ckpt_stage_ens8_2023-01-01    the eight members of the 2023 mean-rule ensembles
#   evals_cd1_2023-01-01/ckpt     (historical: the 2023 roster before the fine-tunes)
#   cutoff_stage_aa20/<D>         per cutoff: base antiGen, the gate, ESM-2 35M partial, BLOSUM62
set -euo pipefail
. "$(dirname "$0")/config.sh"
LINK=0; [ "${1:-}" = --link ] && LINK=1

ok=0; made=0; missing=0
while IFS=$'\t' read -r dir name target; do
    case $dir in \#*|"") continue ;; esac
    path=$CKPT/$dir/$name
    if [ "$target" = - ]; then
        if [ -f "$path" ]; then ok=$((ok + 1)); else echo "MISSING (trained file) $dir/$name"; missing=$((missing + 1)); fi
        continue
    fi
    if [ ! -e "$CKPT/$target" ]; then
        echo "MISSING target $target (for $dir/$name)"; missing=$((missing + 1)); continue
    fi
    if [ -e "$path" ]; then
        [ "$(readlink -f "$path")" = "$(readlink -f "$CKPT/$target")" ] || {
            echo "CONFLICT $dir/$name exists and does not point at $target"; missing=$((missing + 1)); }
        ok=$((ok + 1)); continue
    fi
    if [ $LINK = 1 ]; then
        mkdir -p "$CKPT/$dir"
        ln -s "$CKPT/$target" "$path"; made=$((made + 1))
    else
        echo "absent  $dir/$name -> $target"; missing=$((missing + 1))
    fi
done < "$(dirname "$0")/checkpoints.tsv"
echo "present $ok, linked $made, missing/absent $missing"
[ $missing = 0 ]
