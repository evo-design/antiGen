# Reproducing the paper's figures

Every figure in `antiGen-figures/figs/` is drawn from **staged data** committed under
`antiGen-figures/data/`. This file traces each one back through the evaluation that
produced it to the checkpoints and corpora behind that evaluation. There are five stages,
and you can start from the published output of any of them:

```
corpora ──► checkpoints ──► evaluations ──► staged data ──► figures
antiGen-data   antiGen-train    reproduce/eval.sh  reproduce/stage.sh  antiGen-figures/render_all.sh
(README there) reproduce/train_commands.py
```

| stage | command | input | output |
|---|---|---|---|
| corpora | `antiGen-data/` (its README, *The paper's SARS-CoV-2 corpora*) | Nextstrain trees, the open spike alignment | `$CORPORA/sc2-*/` |
| checkpoints | `python reproduce/train_commands.py` prints every `antiGen-train` command; then `bash reproduce/stage_checkpoints.sh --link` | corpora | `$CKPT/...` |
| evaluations | `bash reproduce/eval.sh <step>` (Slurm) | checkpoints + corpora | `$OUT/<step>/` |
| staged data | `bash reproduce/stage.sh [target ...]` | `$OUT/<step>/` | `antiGen-figures/data/` |
| figures | `cd antiGen-figures && bash render_all.sh [figure ...]` | `data/`, `ref/`, `baselines/`, `dms/`, `cartoons/` | `figs/*.png`, `figs_vector/*.pdf` |

Paths and partitions are set in [`reproduce/config.sh`](reproduce/config.sh): `CORPORA`,
`CKPT`, `OUT` (default `$CKPT/reproduce`), `PARTITION` (default `gpu_batch`) and
`CPU_PARTITION` (default `cpu_batch`). Override any of them in the environment.

**Rendering alone** needs only R (4.5.3 was used) with cowplot, dplyr, forcats, ggpattern,
ggplot2, ggtext, magick, reshape2, scales, tibble and tidyr. Everything it reads is
committed.

### Where things are

| path | what |
|---|---|
| `reproduce/config.sh` | paths and partitions for everything below |
| `reproduce/train_commands.py` | the `antiGen-train` command behind every checkpoint |
| `reproduce/checkpoints.tsv`, `stage_checkpoints.sh` | the checkpoint directories the evals read |
| `reproduce/eval.sh` | every evaluation, one named step per figure family |
| `reproduce/stage.sh`, `compare_staged.py` | eval outputs → `antiGen-figures/data/`, and a value-level check against the committed data |
| `reproduce/rules/` | the ensemble rule definitions |
| `reproduce/slurm/` | the training recipes and their submitters (one per model family, the cutoff series, the CV ×10 ensembles, the 2023 fine-tunes, the data-seed replicates), and the external-baseline runs |
| `reproduce/stage/` | the stagers `stage.sh` calls |
| `reproduce/tools/` | producers the steps call (post-cutoff masks, host-weighted background, DMS dumps, training sizes, panel B) and the builders of two external maps in `data/baselines/` |

## 1. Figures, their data and the step behind each

| figure (`figs/<name>.png`) | staged data (`antiGen-figures/data/`) | `eval.sh` step(s) |
|---|---|---|
| `sc2_results_revised2` B (case study) | `ba286_to_kp3_ranks_2023-01-01.csv`, `cartoons/ba286_to_kp3.png` | `panel_b` |
| ″ C–F (all RBD, 2023-01-01) | `evals_sc2_lists_macro_aa20_2023-01-01_allcells/` | `rbd_2023`, `ens8_2023` |
| ″ G–J (1 nt, 2023-01-01) | `evals_sc2_lists_macro_aa20_2023-01-01_cd1/` | `rbd_2023` |
| ″ K–N (de novo, ≥ 100 sequences, 2021-01-01) | `evals_sc2_lists_macro_aa20_2021_allcells_min100/`, `…_2021_allcells_cd1_min100/` | `cd1_2021_allcells`, `postcutoff_counts` |
| ″ O/P (cutoff series) | `gating_aa20/<D>/` (per mutation), `gating_macro_aa20/<D>/` (per cell) | `cutoff_eval`, `cutoff_rbd` |
| ″ model-type key | `evals_sc2_lists_macro_aa20/` | `list2021` |
| `sc2_results_revised2` Q (fixed splits) | `evals_fixedsplit_{2022-01-01,2023-10-01,2024-10-01}/` | `fixedsplit` (+ CoVFit per query, §4) |
| `sc2_results_revised2_supp` | as C–J and O/P | as above |
| `supp_min100_2021` | `…_2021_allcells_min100/` | `cd1_2021_allcells`, `postcutoff_counts` |
| `supp_fullspike` | `gating_macro_aa20_fullspike/<D>/`, `…_2023-01-01_fullspike/`, `…_2023-01-01_fullspike_cd1/` | `cutoff_spike`, `fullspike_2023` |
| `supp_query_adapt` | `evals_sc2_background_aa20{,_eventwtd}/`, `…_2023-01-01{,_eventwtd}/` | `background_2021`, `background_2023` |
| `supp_dms_revised2` | `sc2_aa20/`, `sc2_aa20_extra/` (+ `baselines/`, `dms/`) | `list2021`, `dms_dumps` |
| `supp_gate_units` | `gate_units_aa20/`, `gating_aa20/`, `gating_macro_aa20/` | `gate_units`, `cutoff_eval`, `cutoff_rbd` |
| `supp_dataseed` | `dataseed_2023-01-01/seed_summary.csv` | `dataseed` |
| `supp_usher_benchmark` | `runtime/usher_benchmark_results.csv` | antiGen-data `bench/` (§4) |

Retired to `figs/old/` (no longer rendered): `supp_cd1`, `supp_cd1_min100`, `supp_cd1_min100_2021`,
`supp_cd1_series`, `supp_lm_cd1`.

`<D>` runs over the sixteen quarterly cutoffs 2020-04-01 … 2024-01-01; the de novo 1-nt
arm uses 2020-04-01 … 2021-10-01, and its stopping rule draws the first six.

## 2. The evaluation steps

`bash reproduce/eval.sh` lists them; `DRY=1 bash reproduce/eval.sh <step>` prints the job
without submitting it. Each submitted job's script is kept as `logs/rp_<step>.sh`.

| step | what | checkpoints (`$CKPT/…`) | top-q rule | published source (`$CKPT/…`) | time |
|---|---|---|---|---|---|
| `list2021` | list_macro, 2021-01-01 arm, COG-UK list, whole roster; dumps score maps | `ckpt_stage_aa20`, `ckpt_stage_vs10_aa20` | legacy | `evals_sc2_lists_macro_aa20`, `list_score_maps_aa20` | ~1 h |
| `dms_dumps` | each model's prediction at the BA.2 clade root, from those maps (CPU) | — | — | written straight into `data/sc2_aa20{,_extra}` | min |
| `cd1_2021` | 2021 arm, COG-UK list, + 1-nt variants | `evals_cd1_2021/ckpt` | legacy | `evals_cd1_2021/out` | ~30 min |
| `cd1_2021_allcells` | 2021 arm, every RBD cell, + 1-nt variants | `evals_cd1_2021/ckpt` | current | `evals_cd1_2021_allcells/out` | ~30 min |
| `rbd_2023` | 2023-01-01, every RBD cell, full roster with fine-tunes + CV ×10, + 1-nt | `ckpt_stage_rbd_2023-01-01`, `ckpt_stage_vs10_2023-01-01` | legacy | `evals_cd1_2023-01-01/out_trainonly`; the codon-aware B62 rows from `evals_cb62_2023-01-01` (same flags, run later) | ~1 h |
| `ens8_2023` | the three mean-rule ensembles of eight members | `ckpt_stage_ens8_2023-01-01` | legacy | `ensembles8_aa20_2023-01-01/report` | ~1 h |
| `fullspike_2023` | 2023-01-01, whole spike, + 1-nt | `ckpt_stage_bg_2023-01-01`, `ckpt_stage_vs10_2023-01-01` | legacy | `evals_sc2_lists_macro_aa20_2023-01-01_allcells_fullspike_cd1` | ~50 min |
| `cutoff_eval <D>` | antiGen-eval per cutoff (per mutation) | `cutoff_stage_aa20/<D>` | legacy | `cutoff_evals_aa20_trainonly/<D>` | 4 min – 1.3 h |
| `cutoff_rbd <D>` | list_macro per cutoff, every RBD cell | `cutoff_stage_aa20/<D>` | legacy | `evals_sc2_lists_macro_aa20_<D>_allcells_pooled_trainonly` | 30 min – 1 h |
| `cutoff_spike <D>` | the same on the whole spike | `cutoff_stage_aa20/<D>` (2023-01-01: `ckpt_stage_bg_…` + CV ×10) | legacy | `evals_sc2_lists_macro_aa20_<D>_allcells_fullspike_trainonly` | ~1 h |
| `cd1_series_denovo <D>` | 1-nt, de novo cells | `cutoff_stage_aa20/<D>` | current | `evals_cd1_series_trainonly/<D>` | ~30 min |
| `cd1_series_rare100 <D>` | 1-nt, cells seen < 100 times in training | `cutoff_stage_aa20/<D>` | current | `evals_cd1_series_rare100_trainonly/<D>` | ~30 min |
| `postcutoff_counts` | all-lineage residue counts dated ≥ each cutoff (the ≥ 100 masks; CPU) | — | — | `postcutoff_counts` | min |
| `background_2021` | background discrimination + contrast, then the host-weighted summary | `ckpt_stage_aa20`, `ckpt_stage_vs10_aa20` | legacy | `evals_sc2_background_aa20` (+ `_topup`, see §5) | ~2.5 h |
| `background_2023` | the same at 2023-01-01 | `ckpt_stage_bg_2023-01-01`, `ckpt_stage_vs10_2023-01-01` | legacy | `evals_sc2_background_aa20_2023-01-01_trainonly` | hours |
| `dataseed <k>` | antiGen (CV ×10) retrained on the 2023-01-01 corpus rebuilt with `antigen --seed k` (k = 1, 2, 3; 1 is the production corpus), scored as C–J | `ckpt_stage_vs10_2023-01-01` (k = 1), `dataseed_2023-01-01/seed_<k>/stage` | legacy | `dataseed_2023-01-01/seed_<k>/eval_cd1` | ~2 min |
| `fixedsplit <D>` | base antiGen of cutoff D against the fixed maps, COG-UK list | `cutoff_series/<D>/base` | legacy | `evals_fixedsplit_<D>` | ~30 min |
| `gate_units` | training-set size per cutoff in three units (CPU) | — | — | — | ~2 min |
| `panel_b` | ranks of each BA.2.86 → KP.3 step, and the panel image (CPU) | `cutoff_stage_aa20/2023-01-01` base | — | — | ~20 s |

**The top-q rule.** A later change fixed how ties at the top-q boundary are counted.
Most published outputs predate it and are reproduced with `ANTIGEN_TOPQ_LEGACY=1` (the
"legacy" rows); `eval.sh` sets the flag per step. Regenerating everything under the
current rule is a follow-up.

**Fitted baselines** (substitution, where × which, PSSM, codon, site × substitution, the
MCM rungs) are fitted on the train corpus only, the data every checkpoint was trained
on. Each step's test set is the test tree's records after the cutoff.

**The model roster** of a step is whatever checkpoints its directory holds, plus the
statistical rungs and the fixed external maps in `data/baselines/` (globbed by every
eval). A step can hold more models than its figure draws; the drawers pick their
rows by name (`antiGen-figures/figures/revised2_roster.R`).

## 3. Checkpoints

`python reproduce/train_commands.py` reads the `train_args` stored in every checkpoint
of every family and prints the `antiGen-train` command that produced it (`--family` to
pick one, `--sh` for bare commands). The families:

| family | directory | contents |
|---|---|---|
| `2021-roster` | `ckpt_stage_aa20` | 2021 arm: base antiGen, the gate, ESM-2 35M/650M (untrained, frozen, partial, full), EVE, Tranception, BLOSUM62 |
| `2021-cv10` | `ckpt_stage_vs10_aa20` | 2021 antiGen (CV ×10): ten 90/10 splits, each at its best validation epoch |
| `2023-roster` | `ckpt_stage_bg_2023-01-01` | 2023 base, gate, ESM-2, EVE / Tranception frozen, BLOSUM62 |
| `2023-cv10` | `ckpt_stage_vs10_2023-01-01` | 2023 antiGen (CV ×10) |
| `2023-finetunes` | `ckpt_stage_ft_2023-01-01` | 2023 ESM-2 / EVE / Tranception fine-tunes |
| `2023-codon-blosum` | `ckpt_codon_blosum_2023-01-01` | codon-aware BLOSUM62 (`--model-type codon_blosum`, κ = 4): training-free, so its "checkpoint" is the fixed lookup table |
| `dataseed-<k>` | `dataseed_2023-01-01/seed_<k>/stage` | the CV ×10 members trained on the seed-k corpus |
| `cutoff-<D>` | `cutoff_stage_aa20/<D>` | per cutoff: base antiGen, the gate, ESM-2 35M partial, BLOSUM62 |

Inputs beyond a plain `antiGen-train` call:

* **The gate** (`gated_static_xfmr_pssm`) takes the cutoff's root PSSM as a CSV:
  `reproduce/slurm/train_cutoff_gate.sh <D>` runs `reproduce/tools/counts_to_pssm_csv.py`
  on `counts_<D>_cur.tsv` and then trains (`submit_cutoff_gates.sh` does all sixteen).
* **EVE and Tranception** train on the cutoff's alignment, built by antiGen-data
  (`slurm/run_2021.sbatch --out-msa`, `slurm/msa_2023-01-01.sbatch`). The 2023 models use
  `--max-seqs 351410`.
* **CV ×10**: `reproduce/slurm/train_antigen_valsel10.sh` (2021, 96 epochs) and
  `train_valsel_seed.sh 2023-01-01` (EPOCHS=256); `best_epoch.py` picks each split's epoch.
* **Fine-tunes at 2023-01-01**: `reproduce/slurm/submit_finetunes_2023.sh`.

The evals read checkpoint **directories of symlinks** into the training outputs.
[`reproduce/checkpoints.tsv`](reproduce/checkpoints.tsv) lists every link (directory, file
name, target) and `bash reproduce/stage_checkpoints.sh` checks them (`--link` creates the
missing ones). Retrained models get new time stamps in their file names; edit the
targets before linking.

## 4. Inputs with their own producers

| input | producer |
|---|---|
| `data/baselines/CoVFit.csv` | `reproduce/slurm/eval_covfit.sh` (CoVFit CLI, XFG, fold 0) |
| `data/baselines/MSA_Transformer{,_BA2,_XFG}.csv` | `reproduce/slurm/msa_transformer_spike{,_ba2,_xfg}.sh` |
| `data/baselines/BloomNeher_2023-10-01.csv` | `reproduce/tools/bloom_neher_to_baseline.py` |
| `data/baselines/BA2_escape_LYCoV1404.csv` | `reproduce/tools/make_escape_baseline.py` |
| CoVFit per-query maps (fixed split 2024-10-01) | `reproduce/slurm/submit_covfit_perquery.sh` with `STAGE=gate`, `score`, `eval` |
| `reproduce/rules/rules_drawn5.json`, `rules_mean3_8.json` | the ensemble rule definitions (the members and how they combine); `rules_vs10_2021.json` is the CV ×10 rule alone, for `dms_dumps` |
| `runtime/usher_benchmark_results.csv` | antiGen-data `bench/` (`run_n.sbatch` per size, then `bench/plot.py`) |
| `ref/all_{rbd,spike}_cells_aa20_incl_ref.csv` | every (site, residue) cell, reference allele included |

Fixed inputs with no producer in the repository: EVEscape, the Omicron DMS maps and pyr0
in `data/baselines/`; `data/ba286_to_kp3_path.csv` (the case-study path);
`ref/6M0J.pdb`; the pseudovirus assay tables under `antiGen-figures/experiments/`.

## 5. What has been checked, and what is known to differ

* **Staging.** `reproduce/stage.sh` run on the published evaluation outputs (each step
  name pointed at the source in §2) reproduces every committed CSV it writes, compared
  row by row with `reproduce/compare_staged.py`: worst difference 6e-8, on the codon
  and PSSM rows. The figures draw the same rows.
* **From the checkpoints.** `panel_b` reproduces the case-study ranks and the panel image
  byte for byte, and `gate_units` reproduces its table exactly. GPU spot checks through
  `eval.sh`: `cd1_2021_allcells` reproduces every row exactly, `cutoff_eval 2020-04-01`
  every row to 1e-8, `cutoff_rbd 2020-04-01` 1,796 of 1,800 rows exactly and the other
  four (mcm_interpolation on the COG-UK variants, not drawn) within 5e-5.
* **Rendering.** `render_all.sh` on the committed data reproduces every PNG byte for byte.
* **The 2021 background** was assembled from three runs (the roster, a top-up adding
  mcm and Bloom-Neher, and a separate gate contrast run) whose rows were merged by hand.
  `background_2021` is the single run that covers all of them. Its antiGen (CV ×10)
  rows come from the native `--ensemble-dir` path, which has been checked against the
  published rule for list_macro only.
* **The DMS dumps** (`dms_dumps`) were not recorded when they were first made; the step is
  a reconstruction, checked byte for byte on `site_substitution`.
* **Not drawn, not reproduced here:** `evals_sc2_aa20/` (the 2021 per-mutation eval) and
  `evals_sc2_lists_varrecall_aa20/` are read by figure code that no current figure
  draws. Their fitted rows predate the train-only fits (they were fitted on the test tree's
  pre-cutoff records too), so a re-run will not match them.
