# Producers and stagers behind `reproduce/`

The scripts here are called by [`reproduce/eval.sh`](../../reproduce/eval.sh) and
[`reproduce/stage.sh`](../../reproduce/stage.sh), or build an input the evaluations read.
[`REPRODUCE.md`](../../REPRODUCE.md) says which figure each one serves. `docs/NOTES.md` is
the record of how each came about; the section numbers in their headers (`s<N>`) point
there.

| script | does |
|---|---|
| `stage_allcells_2023.sh`, `stage_cd1_2023.sh`, `stage_cd1_min100.py`, `stage_cd1_series.py`, `stage_lm_cd1.py`, `stage_fullspike_cd1_2023.py` | turn a list_macro run into the staged figure data (`reproduce/stage.sh` calls them with the paths set) |
| `postcutoff_counts.py` | all-lineage residue counts dated on or after each cutoff: the >= 100-sequence masks |
| `background_event_weighted.py` | the host-weighted background summary (supp_query_adapt) |
| `dump_variant_root_predictions.py` | each model's prediction at the BA.2 clade root, from a list_macro score cache (supp_dms_revised2) |
| `training_size_by_cutoff.py` | training-set size per cutoff (supp_gate_units) |
| `rank_path_cells.py`, `draw_path_panel.py`, `render_surface.py` | panel B of sc2_results_revised2: ranks along BA.2.86 -> KP.3, and the drawing |
| `ensemble_test_rows.py` | cached-map ensemble rows for the 2021 per-mutation eval |
| `counts_to_pssm_csv.py` | the root PSSM the gated model trains with (`examples/slurm/train_cutoff_gate.sh`) |
| `submit_cutoff_gates.sh`, `submit_cutoff_esm2_retry.sh`, `submit_finetunes_2023.sh`, `submit_dataseed_2023.sh` | training drivers: the gate and ESM-2 35M at every cutoff, the 2023 fine-tunes, the 2023 CV x10 data-seed replicates |
| `finish_ft_2023.sh` | adds a finished 2023 fine-tune to the all-RBD run |
| `submit_covfit_perquery.sh` | per-query CoVFit maps for the 2024-10-01 fixed split |
| `bloom_neher_to_baseline.py`, `make_escape_baseline.py` | write two of the external maps in `data/baselines/` |
