# Examples

## `slurm/`

The SLURM batch scripts behind the manuscript's checkpoints: one recipe per model family
(`train_transformer.sh`, `train_esm2_{noft,frozen,partial,full}.sh` and their 650M
versions, `train_eve.sh`, `train_tranception.sh`), the per-cutoff series
(`train_cutoff_series.sh`, `train_cutoff_gate.sh`), the 10-fold validation-selected
ensembles (`train_antigen_valsel10.sh`, `train_valsel_seed.sh`, `train_esm2_valsel10.sh`,
`best_epoch.py`), and the external baselines that need their own run (`eval_covfit.sh`,
`download_covfit.sh`, `msa_transformer_spike*.sh`). `eval_sc2_list_macro.sh` wraps
`antiGen/list_macro.py`.

**They are lab-specific**: they hard-code the maintainers' data paths and cluster
partitions. Treat them as templates. The per-model recipes pass any extra arguments
through to `antiGen-train`, and must be given the train corpus that way
(`--examples <train corpus>`); `antiGen-train` refuses to run without one, and refuses
any file that holds a post-cutoff record. `python reproduce/train_commands.py` prints the
exact `antiGen-train` command behind every checkpoint the figures use, and
`reproduce/eval.sh` runs every evaluation; see [`REPRODUCE.md`](../REPRODUCE.md).

For a starting point, see the Quickstart in the top-level `README.md`, which generates
training data from the toy inputs in `antiGen-data/example/`, then trains and runs
inference on it.
