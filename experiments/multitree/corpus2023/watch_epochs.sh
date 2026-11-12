#!/bin/bash
# Score every new epoch of the joint run and the matched SARS-CoV-2-only control as it
# lands (NOTES s155). Epochs with a per-epoch checkpoint (epochs/) are used directly;
# joint epochs from chunk 1, which predates those, are converted from the resume state.
R=/home/ivan.specht/antiGen; B=/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023
PY=/home/ivan.specht/miniconda/bin/python
declare -A done_
while :; do
  for run in esm2_35m_partial_s42:ep control_sc2only_s42:ctl; do
    d=$B/${run%%:*}; pre=${run##*:}
    # per-epoch checkpoints
    for f in $d/epochs/fitness_*_ep*.pt; do
      [ -f "$f" ] || continue
      e=$(echo "$f" | sed -E 's/.*_ep([0-9]+)\.pt/\1/'); tag=$pre$e
      [ -n "${done_[$tag]:-}" ] || [ -d $B/eval/$tag ] && { done_[$tag]=1; continue; }
      # A joint checkpoint records the longest training protein (1,447) as seq_len and
      # list_macro refuses it for SARS-CoV-2 (1,273); ESM-2 is length-agnostic, so the
      # weights go into the control's metadata (resume_to_ckpt.py, strict shapes).
      mkdir -p $d/epochs_converted
      $PY $R/experiments/multitree/corpus2023/resume_to_ckpt.py "$f" $d/epochs_converted/fitness_ep$e.pt >/dev/null && \
      PARTITION=quick_preemptible $R/experiments/multitree/corpus2023/eval_joint_epoch.sh $d/epochs_converted/fitness_ep$e.pt $tag >/dev/null && done_[$tag]=1
    done
    # resume state (chunk-1 epochs of the joint run)
    if [ -f $d/resume_state.pt ]; then
      m=$(stat -c %Y $d/resume_state.pt)
      if [ "${done_[$pre-mtime]:-}" != "$m" ]; then
        e=$($PY -c "import torch;print('%02d'%torch.load('$d/resume_state.pt',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
        tag=$pre$e
        if [ -n "$e" ] && [ ! -d $B/eval/$tag ] && [ ! -f $d/epochs/fitness_*_ep$e.pt ]; then
          mkdir -p $d/epochs_converted
          $PY $R/experiments/multitree/corpus2023/resume_to_ckpt.py $d/resume_state.pt $d/epochs_converted/fitness_ep$e.pt >/dev/null && \
          PARTITION=quick_preemptible $R/experiments/multitree/corpus2023/eval_joint_epoch.sh $d/epochs_converted/fitness_ep$e.pt $tag >/dev/null
        fi
        done_[$pre-mtime]=$m
      fi
    fi
  done
  sleep 60
done
