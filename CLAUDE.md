# antiGen — working rules

## Cluster: partition policy (read this before any `sbatch`)

> **`evo_gpu_priority` is for VERY SHORT jobs only — order of minutes.**
> **Everything else goes to `gpu_batch`.**

It is a shared fast lane. A multi-hour job parked on it blocks other people's short
work, which is the entire point of the partition.

**Do not trust a script's default.** Many files under `examples/slurm/` carry
`#SBATCH --partition=evo_gpu_priority` from before this rule — among them
`eval_all_sc2.sh`, `list_recall_sc2.sh`, `eval_emergence_sc2.sh`, `eval_all_flu.sh`,
`eval_all_denv2.sh`, `eval_all_rsv_a.sh` and the `train_*` / `score_*` valsel scripts.
Submitting through one of those without an override puts a long job on the fast lane.

Pass the partition explicitly, or give the wrapper a knob:

```bash
PARTITION=${PARTITION:-gpu_batch}
sbatch --partition="$PARTITION" ...
```

Already queued or running in the wrong place:

```bash
scontrol update jobid=<id> partition=gpu_batch   # PENDING only
scancel <id> && <resubmit>                       # RUNNING: no in-place move
```

## Cluster: CPU partitions are not interchangeable

> **Batch work on CPU goes to `cpu_batch`, never `cpu`.**

`cpu` is the DEFAULT partition and carries `QoS=cpu_interact`, which caps a user at
**2 concurrent jobs** (`sacctmgr show qos` -> `cpu_interact MaxJobsPU=2`,
`MaxTRESPU=cpu=64`). A sweep submitted there runs two at a time however many cores are
idle and whatever its priority, and `squeue` explains it only as
`(QOSMaxJobsPerUserLimit)` -- which reads like a cluster-wide quota rather than a
property of the partition that was picked by default.

`cpu_batch` has `MaxJobsPU=20`, `MaxSubmitPU=200`. Same nodes.

```bash
scontrol update jobid=<id> partition=cpu_batch   # PENDING only
```

**Measured runtimes, as a guide to what "short" means:**

| job | time | partition |
|---|---|---|
| `list_macro` resume, one model | ~45 s | `evo_gpu_priority` ok |
| `list_variant_recall`, full rebuild | ~90 s | `evo_gpu_priority` ok |
| per-cutoff eval (`cut_<date>`) | 4 min – 1h14m | `gpu_batch` |
| `background_discrimination`, full | ~2h21m | `gpu_batch` |
| `background_contrast`, revised corpus | hours | `gpu_batch` |
| main `eval_all_sc2` | ~11h | `gpu_batch` |

## Working log

`experiments/multitree/NOTES.md` is the running record — numbered sections, newest at
the end. Work that lands in the repo gets a section; code comments reference them as
`NOTES s<N>`, so a referenced section has to exist.
