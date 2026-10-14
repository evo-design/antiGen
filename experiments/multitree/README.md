# Multi-tree ESM-2 finetuning (exploratory)

Can **one** ESM-2 finetune learn anything that transfers across different viral proteins
and different phylogenies? This arm trains a single model jointly on SARS-CoV-2 spike,
H3N2 HA and H1N1pdm HA examples, exploiting the fact that ESM-2 is the only trainable
model in the repo that is length-agnostic.

Two deliberate simplifications make it quick to run, and **both make the numbers here
non-comparable with the manuscript's**:

1. **No attachment phase.** Examples come straight from the guide tree's internal nodes
   and branches — no MSA, no metadata, no C++ build.
2. **No temporal cutoff, random validation split.** Every example is `TRAINING`, and
   `--val-mode random` holds out a random 10% per tree. This is *not* a forecast: a
   held-out node's parent can sit in the training half, so validation scores are
   optimistic. The arm is tagged `experiment: multitree` in the manifest and in
   `train_args`, and `antiGen-eval` skips such checkpoints unless run with
   `--include-experimental`.

## 1. Generate the examples

`make_tree_examples.py` walks a Nextstrain/auspice v2 JSON and writes the standard
examples format (`antiGen/load.py`, `antiGen-validate-examples` both read it unchanged).
It mirrors `antiGen-data/src/main.cpp` — collapsing zero-mutation branches, reconstructing
each node's ancestral genotype, shuffling each branch's mutations, emitting `k-1`
intermediate examples per `k`-mutation branch plus one node example per node with
children — minus attachment and the cutoff.

```bash
cd /home/ivan.specht/antiGen
P=experiments/multitree/make_tree_examples.py
D=experiments/multitree/data

python $P --tree nextstrain/ncov_open_global_all-time.json \
          --ref antiGen-data/inputs/sc2/ref.json --genes S --strip-trailing-stop \
          --out $D/examples_ncov_S_nocutoff.txt

python $P --tree nextstrain/flu_seasonal_h3n2_ha_12y.json \
          --ref antiGen-data/inputs/h3n2/ref.json --genes SigPep HA1 HA2 \
          --out $D/examples_h3n2_HA_nocutoff.txt

python $P --tree nextstrain/flu_seasonal_h1n1pdm_ha_12y.json \
          --ref experiments/multitree/refs/h1n1pdm_ha_12y_root-sequence.json \
          --genes SigPep HA1 HA2 --out $D/examples_h1n1pdm_HA_nocutoff.txt
```

Reference sequences: the repo's `sc2` and `h3n2` `ref.json` are verified consistent with
these trees (zero from-base mismatches over 2672 / 1605 aa mutations). H1N1pdm has none in
the repo, so `refs/h1n1pdm_ha_12y_root-sequence.json` is the Nextstrain sidecar
(`https://data.nextstrain.org/flu_seasonal_h1n1pdm_ha_12y_root-sequence.json`). Segment
lengths are read from the reference, not hardcoded — H1N1pdm's HA is 17+327+222 where
H3N2's is 16+329+221, both summing to 566.

Expected output (regenerating with the same `--seed` is byte-identical):

| tree | seq_len | node ex. | intermediate ex. | total | child mutations |
|---|---|---|---|---|---|
| ncov S | 1273 | 327 | 1188 | 1515 | 2672 |
| h3n2 HA | 566 | 287 | 530 | 817 | 1605 |
| h1n1pdm HA | 566 | 207 | 455 | 662 | 1378 |

The filenames deliberately avoid the `examples_cutoff_*` pattern so `antiGen-eval`'s glob
never picks them up. These files have no `TESTING` examples and no variant annotations, so
the variant-emergence eval and the `pssm`/`gated` model types do not apply to them.

## 2. Train

```bash
sbatch experiments/multitree/train_multitree_esm2.sh   # partition gpu_batch, 1 GPU
```

Finetuning strategy is the usual three flags: `--freeze-backbone --finetune-layers 4`
(the script's default, the conservative start on ~2.7k training examples),
`--freeze-backbone` alone for head-only, or neither for a full finetune.

Per epoch the trainer prints, and appends to `<run_prefix>_val_metrics_by_epoch.json`,
the validation NLL per observed mutation **per tree**, each next to two reference points:

- **uniform** — `log(L·21)`, what a predictor that knows nothing scores. Raw NLL is not
  comparable across trees (the softmax in `compute_probs` runs over the whole `L×E` map),
  so the *gap to uniform* is the comparable quantity.
- **codon** — the genetic code alone: the log number of single-nucleotide routes from the
  wild-type amino acid to each alternative (`--model-type codon`, `antiGen/models/
  codon_fitness_predictor.py`). Data-free and position-blind, it still beats uniform
  comfortably, because most observed substitutions are one nucleotide away.
- **frequency** — a Laplace-smoothed per-position mutability map fit to that tree's
  training half only. This is the bar: it enjoys the same within-tree leakage as the
  random split, and it is strong (hit@10% ≈ 0.74–0.84 on these trees).

A zero-shot row (`epoch: -1`) is recorded before the first optimizer step, which separates
"ESM-2 already knew this" from "finetuning learned something". Measured there, the three
reference points order as expected on all three trees — e.g. on H3N2 HA, NLL/mutation
9.38 (uniform) > 8.58 (codon) > 7.75 (frequency), with hit@10% 0.29 (codon) vs 0.82
(frequency) and 0.16 for zero-shot ESM-2.

## 3. Reading the result

The experiment succeeds in its narrow sense if per-tree validation NLL improves over the
zero-shot row and beats uniform on **all three** trees. Beating the per-tree frequency
baseline would be the strong result. A model that improves only on the tree contributing
the most examples is the negative result worth knowing — with ~2.7k training examples
total, overfitting is the expected failure mode.

## First run (2026-10-12)

`esm2_t12_35M`, last 4 layers + LM head (11.3M trainable), lr 1e-4, batch 8
(`--max-batch-tokens 8192`), 20 epochs, ~10 min on one H100. Artifacts in
`runs/multitree/`; `python experiments/multitree/summarize_run.py runs/multitree` prints
the full curve.

Validation NLL per held-out mutation, at each tree's best epoch, against that tree's
reference points:

| tree | zero-shot | **best** (epoch) | codon | frequency | uniform |
|---|---|---|---|---|---|
| ncov S | 10.89 | **7.19** (5) | 9.98 | 8.35 | 10.19 |
| h3n2 HA | 9.23 | **6.81** (5) | 8.58 | 7.75 | 9.38 |
| h1n1pdm HA | 9.18 | **7.17** (3) | 8.46 | 8.04 | 9.38 |

One jointly trained model beats uniform, codon and the per-tree frequency baseline on
**all three** trees, and beats frequency on hit@10% on two of three (h3n2 0.875 vs 0.823,
h1n1pdm 0.881 vs 0.739; ncov 0.791 vs 0.836 stays just below).

Two things to carry forward:

1. **It overfits after ~epoch 5.** Training loss falls monotonically (8.48 -> 5.20) while
   validation NLL bottoms out at epoch 3-5 on every tree and then climbs — by epoch 19
   h3n2 is back to 7.90, worse than the frequency baseline. `train.py` saves only the
   final checkpoint, so the saved `.pt` here is the *overfit* one; re-run with
   `--epochs 6` for a usable model. Interestingly hit@10% degrades far less than NLL
   (0.82-0.85 at epoch 19), i.e. the ranking survives while the calibration does not.
2. **This does not yet demonstrate transfer.** It shows a model trained on three trees
   beats the baselines on all three — not that the joint training helped. The missing
   control is three single-tree runs with the same recipe; only if the joint model beats
   those per-tree models is anything crossing between trees.

## Single-tree controls

```bash
sbatch experiments/multitree/train_singletree_esm2.sh      # array 0-2, one tree per job
python experiments/multitree/compare_runs.py \
    runs/multitree /large_storage/hielab/ispecht/multitree/single/*
```

Same recipe as the joint run, one tree per job. The source NAME and `--split-seed 0` are
kept identical, so `split_examples` draws each tree's held-out set from the same
per-source RNG stream (`[split_seed, crc32(name)]`) and the validation sets match example
for example — verified: identical train and val sets for all three trees. `compare_runs.py`
re-checks the held-out mutation counts and warns if they ever diverge.

`delta = joint - single` on validation NLL per held-out mutation: negative means training
across trees helped that tree.

## Control result (2026-10-13): no measurable transfer

Job array `3273115`, same recipe, one tree per job, identical held-out sets. Validation
NLL per held-out mutation at each run's best epoch:

| tree | held-out muts | single-tree | joint | delta | hit@10% single -> joint |
|---|---|---|---|---|---|
| ncov S | 215 | 7.239 (ep 4) | 7.194 (ep 5) | **-0.045** | 0.781 -> 0.791 |
| h3n2 HA | 176 | 6.824 (ep 5) | 6.813 (ep 5) | **-0.011** | 0.881 -> 0.875 |
| h1n1pdm HA | 126 | 7.217 (ep 6) | 7.174 (ep 3) | **-0.043** | 0.849 -> 0.825 |

The joint model is nominally ahead on all three trees, but by 0.01-0.05 nats — far below
what 126-215 held-out mutations can resolve — and hit@10% moves the *other* way on two of
the three. **Read this as a tie: training across trees neither helped nor hurt.** What the
earlier joint run showed was that ESM-2 can fit several phylogenies at once, not that
anything transfers between them.

The one suggestive detail is early-epoch behavior: at epoch 0 the joint model is ahead on
every tree (ncov 8.04 vs 8.10, h3n2 7.17 vs 7.34, h1n1pdm 7.44 vs 7.67), i.e. the extra
trees act as a mild warm start, but the single-tree runs catch up within a few epochs.

Resolving a difference this small needs a paired test, which this setup cannot currently
support: `train.py` saves only the final (overfit) checkpoint, so the best-epoch models are
gone, and only aggregate NLL is logged, not per-mutation values. To make it answerable:
save a best-by-validation checkpoint, log per-example losses, and repeat over several
`--split-seed` values.

## Note on this environment

`--model-type esm2` currently fails here with
`ModuleNotFoundError: Could not import module 'EsmForMaskedLM'`. The cause is unrelated to
this experiment: `torchvision 0.21.0` is built for torch 2.6 and raises
`operator torchvision::nms does not exist` under the installed torch 2.7.1, and
transformers 4.57 imports `torchvision.transforms` on the way to ESM-2. Fix with
`pip install 'torchvision==0.22.1'` (matching torch 2.7.1) or `pip uninstall torchvision`.
