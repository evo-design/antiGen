#!/usr/bin/env python
"""
Simple fitness training: predict which mutations were observed from each ancestral sequence.
"""

import os
import gc
import json
import random
import math
import zlib

import argparse
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from antiGen.config import (ModelConfig, ModelType, TransformerConfig, AlphabetConfig, ESM2Config,
                            EVcouplingsConfig, EVEConfig, TranceptionConfig, PSSMConfig, CodonConfig,
                            SubstitutionConfig, GatedFitnessConfig, STANDARD_AAS, LEGACY_AAS)
from antiGen.models.factory import create_model
from antiGen.load import load_examples
from antiGen.batching import LengthBucketBatchSampler
from antiGen.eval_utils import (compute_probs, compute_loss_batch, compute_hit_rate,
                                validate_per_source, frequency_baseline_scores,
                                fixed_map_metrics, fit_substitution_baseline,
                                where_only_baseline_scores, positional_marginal_counts,
                                WhereTimesWhichPredictor, CellDumper)
from antiGen import datasets


# ── Dataset ───────────────────────────────────────────────────────────────────

class MutationDataset(Dataset):
    def __init__(self, training_examples):
        self.aa_indices_list = [e.aa_idx for e in training_examples]
        self.mutation_matrices = [e.mat for e in training_examples]

    def __len__(self):
        return len(self.aa_indices_list)

    def __getitem__(self, idx):
        return self.aa_indices_list[idx], self.mutation_matrices[idx]


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train fitness model on observed mutations')
    parser.add_argument('--cutoff', type=float, default=1.0, help='Train/test cutoff (years since MRCA)')
    parser.add_argument('--ignore-dead-ends', action='store_true', default=False, help='Do not make examples for sequences with no downstream mutants')

    parser.add_argument('--model-type', type=str, default='transformer', choices=['transformer', 'esm2', 'blosum62', 'codon', 'substitution', 'pssm', 'evcouplings', 'eve', 'tranception', 'gated'],
                        help='Model architecture: from-scratch transformer, ESM-2 finetune, BLOSUM62 (eval only), PSSM (eval only), EVcouplings (eval only), EVE VAE, Tranception LM, or gated blend of two predictors')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension (transformer only)')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers (transformer only)')
    parser.add_argument('--esm2-model-name', type=str, default='facebook/esm2_t12_35M_UR50D',
                        help='HuggingFace ESM-2 model name (esm2 only)')
    parser.add_argument('--freeze-backbone', action='store_true', default=False,
                        help='Freeze pretrained backbone, only train output head (esm2 only)')
    parser.add_argument('--finetune-layers', type=int, default=0,
                        help='Number of last encoder layers to unfreeze (requires --freeze-backbone, esm2 only)')
    parser.add_argument('--codon-kappa', type=float, default=1.0,
                        help='Transition/transversion ratio for the codon baseline '
                             '(1.0 = every nucleotide change counts equally)')
    parser.add_argument('--codon-pseudocount', type=float, default=0.01,
                        help='Floor for codon-inaccessible substitutions')
    parser.add_argument('--no-blosum', action='store_true', default=False,
                        help='Drop the BLOSUM62 term from --model-type substitution, '
                             'leaving a codon-only model with learned kappa')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-batches', type=int, default=None, help='Stop after N batches total')
    parser.add_argument('--warmup-steps', type=int, default=0, help='LR warmup steps')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--examples', type=str, nargs='+', default=None,
                        help='One or more examples_cutoff_*.txt paths, optionally NAME=PATH. '
                             'Overrides dataset routing. More than one file trains a single '
                             'model jointly on all of them; each file is one "source" '
                             'phylogeny and may have its own sequence length (esm2 only).')
    parser.add_argument('--split-date', default=None, metavar='YYYY-MM-DD',
                    help="Which train/test split to read, for a corpus whose "
                         "POST_CUTOFF field is a SET of cutoff dates rather than one "
                         "boolean (antiGen-data-revised's single test tree, which "
                         "serves 24 of them). A record is a TEST example exactly when "
                         "its set carries this date. Required the moment any record "
                         "carries a non-empty set -- a default would silently pick a "
                         "split -- and ignored on the train arms, whose sets are all "
                         "empty.")
    parser.add_argument('--alphabet', type=str, default=STANDARD_AAS,
                        help=f'Amino-acid alphabet, one character per output column '
                             f'(default: the 20 standard residues, {STANDARD_AAS}). '
                             f'The published checkpoints were trained in the 22-token '
                             f'frame {LEGACY_AAS}, where index 20 is stop and 21 is gap; '
                             f'pass that to train on data that still carries them.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes (0 for small CPU runs)')
    parser.add_argument('--max-batch-tokens', type=int, default=0,
                        help='If >0, cap tokens per batch: a length-L bucket uses '
                             'min(--batch-size, N // L). Keeps a long-sequence source from '
                             'OOMing at a batch size tuned for a short one.')
    parser.add_argument('--val-mode', choices=['file', 'random'], default='file',
                        help="'file' (default): the repo's temporal split — TRAINING examples "
                             "train, TESTING examples validate. 'random': EXPERIMENTAL arm; "
                             "drops every TESTING example and randomly splits the TRAINING "
                             "pool. Nothing after the forecast date is read, but the resulting "
                             "validation score is not a forecast.")
    parser.add_argument('--val-frac', type=float, default=0.1,
                        help='Validation fraction per source (--val-mode random only)')
    parser.add_argument('--test-frac', type=float, default=0.0,
                        help='Carve a third, selection-free split of this size per '
                             'source (--val-mode random only). Nothing in training or '
                             'epoch selection reads it; it exists so reported numbers '
                             'come from data that had no part in choosing the model. '
                             'Without it the epoch is chosen on val and reported on '
                             'val, which is not a held-out estimate.')
    parser.add_argument('--split-seed', type=int, default=0,
                        help='Seed for the random split, independent of --seed so the split is '
                             'stable across training seeds (--val-mode random only)')
    parser.add_argument('--val-qtiles', type=float, nargs='+', default=[0.1, 0.25],
                        help='Top-q quantiles for the per-epoch validation hit rate')
    parser.add_argument('--dump-cells', action='store_true',
                        help='Write the full per-cell validation probability maps at the '
                             'best-scoring epoch to <prefix>_val_cells.npz, for '
                             'position/substitution decomposition. Adds no change to any '
                             'metric; it only observes the tensors validation already '
                             'computes.')
    parser.add_argument('--save-best', action='store_true',
                        help='Also save the checkpoint at the epoch with the best pooled '
                             'validation NLL, as fitness_<prefix>_best.pt. Without this '
                             'only the final epoch is kept, which is typically well past '
                             'the best epoch.')
    parser.add_argument('--save-best-per-source', action='store_true',
                        help='Also save, for every source, the checkpoint at THAT '
                             'source\'s best validation epoch, as '
                             'fitness_<prefix>_best_<source>.pt. A joint run\'s trees '
                             'peak at different epochs, so the pooled --save-best '
                             'checkpoint is not the best one for any single tree; this '
                             'is what the per-tree epoch rule needs. Selection uses '
                             'validation data only, exactly as --save-best does.')
    parser.add_argument('--mask-gap-stop', action='store_true',
                        help='Mask gap/stop columns in validation metrics (drop them as '
                             'candidate cells and sentinel target events). Off by default; '
                             'the forecast arms pass it so val recall@10 matches the masked '
                             'test metric.')
    parser.add_argument('--save-every-epoch', action='store_true',
                        help='Also save a checkpoint at EVERY epoch as '
                             'fitness_<prefix>_epoch<N>.pt, for per-epoch test scoring / '
                             'val-vs-test correspondence and seed ensembling.')
    parser.add_argument('--select-metric', choices=['nll_pooled', 'recall10_macro'],
                        default='nll_pooled',
                        help="Checkpoint-selection objective for --save-best / "
                             "--save-best-per-source. 'nll_pooled' (default) keeps the "
                             "original mutation-count-weighted pooled validation NLL. "
                             "'recall10_macro' selects the epoch with the highest "
                             "recall@10%% MACRO-averaged across sources (every tree "
                             "weighted equally), and per source its own best recall@10%%. "
                             "Requires 0.1 in --val-qtiles.")
    parser.add_argument('--experiment', type=str, default=None,
                        help='Free-text tag marking an experimental arm; recorded in the '
                             'manifest and train_args, and skipped by antiGen-eval unless it '
                             'is run with --include-experimental.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing examples_cutoff_{cutoff}.txt for a named '
                             'dataset (else resolved from $FITNESS_DATA_ROOT).')
    parser.add_argument('--date-cutoff', type=str, default=None,
                        help='YYYY-MM-DD MSA date cutoff for evcouplings/eve/tranception '
                             '(else derived from the dataset MRCA + --cutoff years).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hit-rate-buckets', type=int, default=10,
                        help='Number of quantile buckets for hit-rate evaluation')
    parser.add_argument('--variant', type=str, default=None,
                        help='Variant name (default: BA.2 for SARS-CoV-2, J for flu)')
    parser.add_argument('--flu', action='store_true', default=False,
                        help='Flu mode: uses flu data, full-sequence eval, variant J or K')
    parser.add_argument('--rsv-a', action='store_true', default=False,
                        help='RSV-A mode: uses RSV-A data, full-sequence eval, variant A.D.1.6')
    parser.add_argument('--denv2', action='store_true', default=False,
                        help='DENV2 mode: uses DENV2 data, full-sequence eval, variant 2II_F.1.1.2_annotated')
    parser.add_argument('--msa', type=str, default=None,
                        help='Path to pandemic MSA FASTA (required for evcouplings/eve)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata TSV with strain/date columns, used to '
                             'restrict the MSA to sequences collected before '
                             '--date-cutoff. OMIT IT when the MSA is already restricted '
                             'to the training window -- antiGen-data-revised writes one '
                             'per run beside the examples it generated from the same '
                             'sequences -- and every sequence in the file is then used. '
                             'Omitting it on an MSA that is NOT so restricted leaks the '
                             'test window into pretraining, which nothing downstream '
                             'can detect, so it is never the default.')
    parser.add_argument('--max-seqs', type=int, default=0,
                        help='Max MSA sequences for evcouplings/eve/tranception (0=unlimited)')
    parser.add_argument('--evc-pseudo-count', type=float, default=0.5,
                        help='Pseudo-count for EVcouplings mean-field DCA regularization (default: 0.5)')
    parser.add_argument('--scoring-direction', type=str, default='both',
                        choices=['forward', 'backward', 'both'],
                        help='Autoregressive scoring direction for Tranception (default: both)')
    _partner_choices = ['fitness_predictor', 'esm2_fitness_predictor',
                        'blosum62', 'codon', 'substitution', 'pssm', 'evcouplings', 'eve', 'tranception']
    parser.add_argument('--gate-partner-a', type=str, default='fitness_predictor',
                        choices=_partner_choices,
                        help='ModelType.value of the first partner for --model-type gated '
                             '(reuses that model type\'s CLI flags).')
    parser.add_argument('--gate-partner-b', type=str, default='blosum62',
                        choices=_partner_choices,
                        help='ModelType.value of the second partner for --model-type gated.')
    parser.add_argument('--gate-variant', type=str, default='static',
                        choices=['static', 'uncertainty'],
                        help='Gate module: "static" (per-position learnable, data-blind) or '
                             '"uncertainty" (MLP on partner softmax entropies).')
    return parser.parse_args()


# ── Run-identity helpers ──────────────────────────────────────────────────────

_PARTNER_SHORT = {
    'esm2_fitness_predictor': 'esm2',
    'fitness_predictor': 'xfmr',
    'blosum62': 'blosum62',
    'codon': 'codon',
    'substitution': 'subst',
    'pssm': 'pssm',
    'evcouplings': 'evc',
    'eve': 'eve',
    'tranception': 'tran',
}


def parse_source_spec(token):
    """``NAME=PATH`` -> ``(name, path)``; a bare path -> a name derived from the path.

    Bare-path naming prefers the parent directory (``.../output_flu_full_cutoff12/
    examples_cutoff_12.txt`` -> ``output_flu_full_cutoff12``), falling back to the stem.
    """
    if '=' in token and not os.path.exists(token):
        name, _, path = token.partition('=')
        return name, path
    parent = os.path.basename(os.path.dirname(os.path.abspath(token)))
    stem = os.path.splitext(os.path.basename(token))[0]
    return (stem if stem.startswith('examples_') and stem != 'examples' else parent or stem), token


def build_checkpoint(model, model_config, args, run_prefix, seq_len_by_source,
                     source_summary, epoch, global_step, loss, optimizer_state):
    """The checkpoint dict, shared by the final save and the best-epoch save.

    ``optimizer_state`` is None for the best-epoch save: resuming from a mid-training
    snapshot is not a use case here, and the optimizer state doubles the file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss,
        'config': model_config.to_dict(),
        'train_args': vars(args),
        'run_prefix': run_prefix,
        'seq_lens': seq_len_by_source,
        'sources': source_summary,
    }

    # EVE/Tranception: persist the MSA-derived dims so eval.py can rebuild the
    # architecture and load weights without re-running MSA pretraining. The
    # alphabet-mapping buffers live in state_dict already; only L_focus and the
    # focus-column indices are needed to reconstruct the module shapes.
    if args.model_type in ('eve', 'tranception'):
        checkpoint['eve_state'] = {
            'L_focus': model._L_focus,
            'focus_cols': model._focus_to_aligned.tolist(),
        }
    return checkpoint


def held_out_split(examples_by_source, val_frac, test_frac, split_seed):
    """Three-way ``(train, val, test)`` split of the TRAINING pool, per source.

    The two-way :func:`split_examples` gives no set that is untouched by model
    selection: the epoch is chosen on ``val`` and then reported on ``val``. Any
    number read off that set is the minimum of a noisy curve over ~20 epochs, not an
    estimate of held-out performance. This carves a third split that nothing in
    training or tuning is allowed to see, so reported numbers come from data that
    had no part in choosing the model.

    The test block is taken from the FRONT of the permutation, so for a fixed
    ``split_seed`` a run's test set does not move when ``val_frac`` changes.

    Returns three dicts keyed by source name.
    """
    train_by, val_by, test_by = {}, {}, {}
    for name, examples in examples_by_source.items():
        pool = [e for e in examples if e.is_training]
        dropped = len(examples) - len(pool)
        if dropped:
            print(f"  [{name}] dropped {dropped} TESTING example(s); post-cutoff data "
                  f"is not used in this arm")
        rng = np.random.default_rng([int(split_seed), zlib.crc32(name.encode())])
        perm = rng.permutation(len(pool))
        n_test = int(round(test_frac * len(pool)))
        n_val = int(round(val_frac * len(pool)))
        # Never starve training, however small the tree.
        n_test = min(n_test, max(len(pool) - 2, 0))
        n_val = min(n_val, max(len(pool) - n_test - 1, 0))
        test_idx = set(perm[:n_test].tolist())
        val_idx = set(perm[n_test:n_test + n_val].tolist())
        train_by[name] = [e for i, e in enumerate(pool)
                          if i not in test_idx and i not in val_idx]
        val_by[name] = [e for i, e in enumerate(pool) if i in val_idx]
        test_by[name] = [e for i, e in enumerate(pool) if i in test_idx]
        print(f"  [{name}] train={len(train_by[name])} val={len(val_by[name])} "
              f"test={len(test_by[name])} (test is never read during training)")
    return train_by, val_by, test_by


def split_examples(examples_by_source, mode, val_frac, split_seed):
    """Return ``(train_by_source, val_by_source)``.

    ``file``   — the repo default: the TRAINING/TESTING label written by antiGen-data,
                 i.e. the temporal cutoff.
    ``random`` — experimental: every TESTING example is dropped outright, so no
                 post-forecast-date information reaches training, validation or any
                 hyperparameter choice. The TRAINING pool is then split per source, each
                 source drawing from its own stream so adding or removing one tree does
                 not perturb another tree's split.

    NOTE: this yields no selection-free set — the epoch is chosen on ``val`` and
    reported on ``val``. Use :func:`held_out_split` with ``--test-frac`` for anything
    whose numbers are reported.
    """
    train_by, val_by = {}, {}
    for name, examples in examples_by_source.items():
        if mode == 'file':
            train_by[name] = [e for e in examples if e.is_training]
            val_by[name] = [e for e in examples if not e.is_training]
            continue
        pool = [e for e in examples if e.is_training]
        dropped = len(examples) - len(pool)
        if dropped:
            print(f"  [{name}] --val-mode random: dropped {dropped} TESTING example(s); "
                  f"post-cutoff data is not used in this arm")
        # crc32, not hash(): str hashing is salted by PYTHONHASHSEED across processes.
        rng = np.random.default_rng([int(split_seed), zlib.crc32(name.encode())])
        perm = rng.permutation(len(pool))
        n_val = int(round(val_frac * len(pool)))
        n_val = min(max(n_val, 1 if len(pool) > 1 else 0), max(len(pool) - 1, 0))
        val_idx = set(perm[:n_val].tolist())
        train_by[name] = [e for i, e in enumerate(pool) if i not in val_idx]
        val_by[name] = [e for i, e in enumerate(pool) if i in val_idx]
    for name in examples_by_source:
        print(f"  [{name}] train={len(train_by[name])} val={len(val_by[name])} "
              f"train_mutations={sum(float(e.mat.sum()) for e in train_by[name]):.0f} "
              f"val_mutations={sum(float(e.mat.sum()) for e in val_by[name]):.0f}")
    return train_by, val_by


def _print_validation(val, baseline_metrics, sources):
    """One line overall, then one per source tree, each against its reference points."""
    print(f"  val nll/mut (all sources)={val['overall']['nll_per_mutation']:.6f} "
          f"over {val['overall']['n_mutations']:.0f} mutations")
    for name in sources:
        stats = val['per_source'].get(name)
        if stats is None:
            continue
        refs = baseline_metrics.get(name, {})
        freq = refs.get('frequency', {})
        codon = refs.get('codon', {})
        subst = refs.get('substitution', {})
        where = refs.get('where_only', {})
        wxw = refs.get('where_times_which', {})
        q = sorted(stats['hit_rate'])[0]
        print(f"    [{name}] L={stats['seq_len']} muts={stats['n_mutations']:.0f} "
              f"nll/mut={stats['nll_per_mutation']:.4f} "
              f"(uniform {stats['uniform_nll_per_mutation']:.4f}, "
              f"codon {codon.get('nll_per_mutation', float('nan')):.4f}, "
              f"subst {subst.get('nll_per_mutation', float('nan')):.4f}, "
              f"where {where.get('nll_per_mutation', float('nan')):.4f}, "
              f"wxw {wxw.get('nll_per_mutation', float('nan')):.4f}, "
              f"frequency {freq.get('nll_per_mutation', float('nan')):.4f})  "
              f"hit@{q}={stats['hit_rate'][q]:.4f} "
              f"(codon {codon.get('hit_rate', {}).get(q, float('nan')):.4f}, "
              f"subst {subst.get('hit_rate', {}).get(q, float('nan')):.4f}, "
              f"where {where.get('hit_rate', {}).get(q, float('nan')):.4f}, "
              f"wxw {wxw.get('hit_rate', {}).get(q, float('nan')):.4f}, "
              f"frequency {freq.get('hit_rate', {}).get(q, float('nan')):.4f})")


def compute_run_family(args) -> str:
    """Stable run-family identifier (no timestamps) for grouping outputs.

    Non-gated: the model_type. Gated: gated_<gate_variant>_<partner_a>_<partner_b>.
    """
    if args.model_type == 'gated':
        a = _PARTNER_SHORT.get(args.gate_partner_a, args.gate_partner_a)
        b = _PARTNER_SHORT.get(args.gate_partner_b, args.gate_partner_b)
        return f"gated_{args.gate_variant}_{a}_{b}"
    if args.examples and len(args.examples) > 1:
        return f"{args.model_type}_multitree"
    return args.model_type


def write_manifest(args, model_config, output_dir, run_prefix,
                   source_summary=None) -> None:
    """Write manifest.json next to the run's outputs capturing the full recipe.

    Aggregators consume this rather than guessing from filenames, so new gate
    variants / model combos don't break discovery.
    """
    import json
    import subprocess
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    manifest = {
        "schema_version": 2,
        "run_family": compute_run_family(args),
        "run_prefix": run_prefix,
        "model_type": args.model_type,
        "model_config": model_config.to_dict(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_batches": args.max_batches,
            "warmup_steps": args.warmup_steps,
            "freeze_backbone": args.freeze_backbone,
            "finetune_layers": args.finetune_layers,
        },
        "data": {
            "cutoff_years": args.cutoff,
            "data_dir": args.data_dir,
            "examples": args.examples,
            "ignore_dead_ends": args.ignore_dead_ends,
            "variant": args.variant,
            "dataset_mode": (
                "flu" if args.flu else
                "rsv_a" if args.rsv_a else
                "denv2" if args.denv2 else
                "sc2"
            ),
            # schema_version 2: multi-source training and experimental-arm labeling.
            "sources": source_summary,
            "seq_lens": (sorted({s["seq_len"] for s in source_summary})
                         if source_summary else None),
            "split_mode": args.val_mode,
            "val_frac": args.val_frac if args.val_mode == 'random' else None,
            "split_seed": args.split_seed if args.val_mode == 'random' else None,
            "experiment": args.experiment,
            "forecast_valid": args.val_mode == 'file',
        },
        "git_commit": git_commit,
    }
    if args.model_type == 'gated':
        manifest["gating"] = {
            "gate_variant": args.gate_variant,
            "partner_a": args.gate_partner_a,
            "partner_b": args.gate_partner_b,
        }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # An arm is experimental if it trains across trees or validates on a random split;
    # the tag travels in train_args and the manifest so antiGen-eval can skip it.
    if args.experiment is None and ((args.examples and len(args.examples) > 1)
                                    or args.val_mode == 'random'):
        args.experiment = 'multitree'

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    allowed_aas = args.alphabet
    E = len(allowed_aas)
    if len(set(allowed_aas)) != E:
        raise SystemExit(f"--alphabet has repeated characters: {allowed_aas!r}")
    print(f"Alphabet: {allowed_aas} (size {E})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if sum([args.flu, args.rsv_a, args.denv2]) > 1:
        raise ValueError("Cannot specify more than one of --flu, --rsv-a, --denv2")

    # Dataset name (for defaults + MSA date-cutoff). Custom data via --examples still uses
    # 'sc2' defaults for the (train-irrelevant) variant unless overridden.
    dataset_name = 'flu' if args.flu else 'rsv_a' if args.rsv_a else 'denv2' if args.denv2 else 'sc2'
    if args.variant is None:
        args.variant = datasets.default_variant(dataset_name)
    cutoff_int = int(args.cutoff)

    if args.examples:
        # Fully custom: train on any examples file(s) (new protein/pathogen, zero code edits).
        spec = [parse_source_spec(tok) for tok in args.examples]
        data_dir = args.data_dir  # may be None; pssm/gated partners need it (pssms/root.csv)
    else:
        datasets.validate(dataset_name, cutoff_int, args.variant)
        data_dir = args.data_dir or datasets.default_data_dir(dataset_name, cutoff_int)
        spec = [(dataset_name, datasets.examples_path(data_dir, cutoff_int))]

    sources, examples_by_source = [], {}
    for name, path in spec:
        if name in examples_by_source:
            raise SystemExit(f"Duplicate source name '{name}' — use NAME=PATH to disambiguate")
        print(f"Loading examples from {path} (source '{name}')...")
        examples = load_examples(path, allowed_aas, args.variant, source=name,
                                 split_date=args.split_date)
        if args.ignore_dead_ends:
            # Filter before splitting so train and val are drawn from one population.
            examples = [e for e in examples if e.mat.sum() >= 0.5]
        if not examples:
            raise SystemExit(f"Source '{name}': no examples left after filtering ({path})")
        lens = {len(e.aa_idx) for e in examples}
        if len(lens) != 1:
            raise SystemExit(f"Source '{name}' has mixed sequence lengths {sorted(lens)} — "
                             f"run antiGen-validate-examples on {path}")
        sources.append(name)
        examples_by_source[name] = examples

    seq_len_by_source = {n: len(examples_by_source[n][0].aa_idx) for n in sources}
    seq_lens = sorted(set(seq_len_by_source.values()))
    seq_len_aa = max(seq_lens)
    print("Sources: " + ", ".join(
        f"{n}(L={seq_len_by_source[n]}, n={len(examples_by_source[n])})" for n in sources))

    # Only length-agnostic models may see more than one length: the from-scratch
    # transformer's positional encoding is a (1, seq_len, d_model) parameter.
    if len(seq_lens) > 1:
        from antiGen.infer import VARIABLE_LENGTH_OK
        variable_ok = {'esm2': ModelType.ESM2_FITNESS_PREDICTOR,
                       'blosum62': ModelType.BLOSUM62,
                       'codon': ModelType.CODON,
                       'substitution': ModelType.SUBSTITUTION}
        if variable_ok.get(args.model_type) not in VARIABLE_LENGTH_OK:
            raise SystemExit(
                f"--model-type {args.model_type} predicts on a fixed coordinate frame and "
                f"cannot train on mixed sequence lengths {seq_lens}. Use --model-type esm2.")

    if args.test_frac > 0:
        if args.val_mode != 'random':
            raise SystemExit('--test-frac requires --val-mode random')
        train_by_source, val_by_source, _test_by_source = held_out_split(
            examples_by_source, args.val_frac, args.test_frac, args.split_seed)
        # Deliberately dropped on the floor: the test split is reconstructed from
        # (split_seed, val_frac, test_frac) at scoring time and must not be touched
        # here, not even to report its size beyond the line held_out_split prints.
        del _test_by_source
    else:
        train_by_source, val_by_source = split_examples(
            examples_by_source, args.val_mode, args.val_frac, args.split_seed)

    training_examples = [e for n in sources for e in train_by_source[n]]
    total_mutations_train = sum(float(e.mat.sum()) for e in training_examples)
    print(f"Training examples: {len(training_examples)}, "
          f"Validation examples: {sum(len(val_by_source[n]) for n in sources)}")
    print(f"Total mutations in training set: {total_mutations_train:.0f}")

    source_summary = [
        {'name': n, 'path': dict(spec)[n], 'seq_len': seq_len_by_source[n],
         'n_train': len(train_by_source[n]), 'n_val': len(val_by_source[n]),
         'n_mutations_train': sum(float(e.mat.sum()) for e in train_by_source[n]),
         'n_mutations_val': sum(float(e.mat.sum()) for e in val_by_source[n])}
        for n in sources
    ]

    train_dataset = MutationDataset(training_examples)
    train_sampler = LengthBucketBatchSampler(
        [len(e.aa_idx) for e in training_examples], batch_size=args.batch_size,
        shuffle=True, seed=args.seed, max_batch_tokens=args.max_batch_tokens)
    # batch_sampler is mutually exclusive with batch_size / shuffle / drop_last.
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    # Validation: one loader per source. Each source is one protein, hence one length,
    # so the default collate is fine and the per-tree breakdown falls out for free.
    val_loaders = {
        n: DataLoader(MutationDataset(val_by_source[n]), batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers, pin_memory=True)
        for n in sources if val_by_source.get(n)
    }
    if not val_loaders:
        print("No validation examples found — skipping per-epoch validation.")

    print(f"Train batches per epoch: {len(train_sampler)} "
          f"buckets={train_sampler.bucket_sizes()}")

    allowed_aas_str = allowed_aas
    alphabet_config = AlphabetConfig(amino_acids=allowed_aas_str)

    if args.model_type == 'esm2':
        model_config = ModelConfig(
            model_type=ModelType.ESM2_FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            esm2=ESM2Config(
                model_name=args.esm2_model_name,
                freeze_backbone=args.freeze_backbone,
                finetune_layers=args.finetune_layers,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: ESM-2 ({args.esm2_model_name}), freeze_backbone={args.freeze_backbone}, finetune_layers={args.finetune_layers}")
    elif args.model_type == 'blosum62':
        model_config = ModelConfig(
            model_type=ModelType.BLOSUM62,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
        )
        model = create_model(model_config, device)
        print("Model: BLOSUM62 (no training, eval only)")
    elif args.model_type == 'codon':
        model_config = ModelConfig(
            model_type=ModelType.CODON,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            codon=CodonConfig(kappa=args.codon_kappa,
                              pseudocount=args.codon_pseudocount),
        )
        model = create_model(model_config, device)
        print(f"Model: codon accessibility (kappa={args.codon_kappa}, no training, eval only)")
    elif args.model_type == 'substitution':
        model_config = ModelConfig(
            model_type=ModelType.SUBSTITUTION,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            substitution=SubstitutionConfig(init_kappa=args.codon_kappa,
                                            pseudocount=args.codon_pseudocount,
                                            use_blosum=not args.no_blosum),
        )
        model = create_model(model_config, device)
        print(f"Model: learned substitution matrix (codon x BLOSUM62, "
              f"{model.get_num_parameters():,} free scalars)")
    elif args.model_type == 'pssm':
        model_config = ModelConfig(
            model_type=ModelType.PSSM,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            pssm=PSSMConfig(data_dir=data_dir),
        )
        model = create_model(model_config, device)
        print(f"Model: PSSM log(count+1) (no training, eval only) — source: {data_dir}/pssms/root.csv")
    elif args.model_type == 'evcouplings':
        if args.msa is None:
            raise ValueError("--msa is required for evcouplings model type")
        if not args.metadata:
            date_cutoff = ""
            print(f"EVcouplings: MSA used as given -- no --metadata, so no date filter. "
                  f"The file must already be restricted to the training window.")
        else:
            date_cutoff = args.date_cutoff or datasets.date_cutoff_str(dataset_name, args.cutoff)
            print(f"EVcouplings: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.EVCOUPLINGS,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            evcouplings=EVcouplingsConfig(
                msa_path=args.msa,
                metadata_path=args.metadata or "",
                date_cutoff=date_cutoff,
                max_seqs=args.max_seqs,
                pseudo_count=args.evc_pseudo_count,
            ),
        )
        model = create_model(model_config, device)
        print("Model: EVcouplings mean-field DCA (no training, eval only)")
    elif args.model_type == 'eve':
        if args.msa is None:
            raise ValueError("--msa is required for eve model type")
        if not args.metadata:
            date_cutoff = ""
            print(f"EVE: MSA used as given -- no --metadata, so no date filter. "
                  f"The file must already be restricted to the training window.")
        else:
            date_cutoff = args.date_cutoff or datasets.date_cutoff_str(dataset_name, args.cutoff)
            print(f"EVE: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.EVE,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            eve=EVEConfig(
                msa_path=args.msa,
                metadata_path=args.metadata or "",
                date_cutoff=date_cutoff,
                freeze_backbone=args.freeze_backbone,
                max_seqs=args.max_seqs,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: EVE VAE, freeze_backbone={args.freeze_backbone}")
    elif args.model_type == 'tranception':
        if args.msa is None:
            raise ValueError("--msa is required for tranception model type")
        if not args.metadata:
            date_cutoff = ""
            print(f"Tranception: MSA used as given -- no --metadata, so no date filter. "
                  f"The file must already be restricted to the training window.")
        else:
            date_cutoff = args.date_cutoff or datasets.date_cutoff_str(dataset_name, args.cutoff)
            print(f"Tranception: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.TRANCEPTION,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            tranception=TranceptionConfig(
                msa_path=args.msa,
                metadata_path=args.metadata or "",
                date_cutoff=date_cutoff,
                freeze_backbone=args.freeze_backbone,
                max_seqs=args.max_seqs,
                scoring_direction=args.scoring_direction,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: Tranception LM, scoring={args.scoring_direction}, freeze_backbone={args.freeze_backbone}")
    elif args.model_type == 'gated':
        # Populate every per-partner sub-config so any partner_type works; the selected
        # partners pick up their CLI flags through their existing sub-config dataclass.
        model_config = ModelConfig(
            model_type=ModelType.GATED_FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            transformer=TransformerConfig(
                d_model=args.d_model, nhead=4,
                dim_feedforward=4 * args.d_model, num_layers=args.num_layers,
            ),
            esm2=ESM2Config(
                model_name=args.esm2_model_name,
                freeze_backbone=args.freeze_backbone, finetune_layers=args.finetune_layers,
            ),
            pssm=PSSMConfig(data_dir=data_dir),
            gated=GatedFitnessConfig(
                partner_a_type=args.gate_partner_a,
                partner_b_type=args.gate_partner_b,
                gate_variant=args.gate_variant,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: Gated[{args.gate_partner_a} + {args.gate_partner_b}, gate={args.gate_variant}], "
              f"freeze_backbone={args.freeze_backbone}, finetune_layers={args.finetune_layers}")
    else:
        model_config = ModelConfig(
            model_type=ModelType.FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            transformer=TransformerConfig(
                d_model=args.d_model,
                nhead=4,
                dim_feedforward=4 * args.d_model,
                num_layers=args.num_layers,
            ),
        )
        model = create_model(model_config, device)
    print(f"Model: {model.get_num_parameters(trainable_only=True):,} trainable params")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build run-specific prefix for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    arm_tag = ""
    if args.examples and len(args.examples) > 1:
        arm_tag += f"_multitree{len(args.examples)}"
    if args.val_mode == 'random':
        arm_tag += f"_randval{args.val_frac:g}s{args.split_seed}"
    if args.model_type == 'esm2':
        model_name_short = args.esm2_model_name.split('/')[-1]
        run_prefix = f"esm2_{model_name_short}{arm_tag}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'blosum62':
        run_prefix = f"blosum62_{timestamp}"
    elif args.model_type == 'codon':
        run_prefix = f"codon_k{args.codon_kappa:g}_{timestamp}"
    elif args.model_type == 'substitution':
        tag = "codononly" if args.no_blosum else "blosum"
        run_prefix = f"subst_{tag}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'evcouplings':
        run_prefix = f"evcouplings_{timestamp}"
    elif args.model_type == 'eve':
        if args.freeze_backbone:
            run_prefix = f"eve_frozen_{timestamp}"
        else:
            run_prefix = f"eve_ft_lr{args.lr}_{timestamp}"
    elif args.model_type == 'tranception':
        if args.freeze_backbone:
            run_prefix = f"tranception_frozen_{args.scoring_direction}_{timestamp}"
        else:
            run_prefix = f"tranception_ft_{args.scoring_direction}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'gated':
        a = _PARTNER_SHORT.get(args.gate_partner_a, args.gate_partner_a)
        b = _PARTNER_SHORT.get(args.gate_partner_b, args.gate_partner_b)
        run_prefix = f"gated_{args.gate_variant}_{a}_{b}_lr{args.lr}_{timestamp}"
    else:
        run_prefix = f"simple_d{args.d_model}_n{args.num_layers}_lr{args.lr}_{timestamp}"

    write_manifest(args, model_config, args.output_dir, run_prefix,
                   source_summary=source_summary)
    val_metrics_path = os.path.join(args.output_dir,
                                    f"{run_prefix}_val_metrics_by_epoch.json")
    val_records = []

    global_step = 0
    epoch = 0
    avg_train_loss = float('nan')
    optimizer_state = None
    best_val_nll, best_epoch = float('inf'), None
    best_val_nll_by_source, best_epoch_by_source = {}, {}

    skip_training = args.model_type in ('blosum62', 'codon', 'pssm', 'evcouplings') or \
        (args.model_type == 'eve' and args.freeze_backbone) or \
        (args.model_type == 'tranception' and args.freeze_backbone)
    if skip_training:
        print(f"{args.model_type} mode: skipping training (no learnable weights); saving checkpoint only.")
        model.eval()
        del train_loader, train_dataset, training_examples
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.warmup_steps > 0:
            def warmup_fn(step):
                if step < args.warmup_steps:
                    return float(step) / float(max(1, args.warmup_steps))
                return 1.0
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
        else:
            scheduler = None

        # Per-tree reference points, computed once. A uniform predictor scores
        # log(L * (E - 1)) per mutation. The frequency baseline is the empirical *joint*
        # over (position, alt_aa) from that tree's TRAIN half only -- it knows both where
        # a mutation lands and which residue it becomes, so it is the memorization
        # ceiling, not a positional map. where_only is that same map with its amino-acid
        # axis flattened to uniform, so the gap between them is exactly the value of
        # knowing which residue appears.
        baseline_metrics = {}
        for name, loader in val_loaders.items():
            counts = frequency_baseline_scores(
                train_by_source[name], seq_len_by_source[name], E)
            where = where_only_baseline_scores(
                train_by_source[name], seq_len_by_source[name], E)
            baseline_metrics[name] = {
                'frequency': fixed_map_metrics(counts, loader, device, mask_gap_stop=args.mask_gap_stop,
                                               qtiles=tuple(args.val_qtiles)),
                'where_only': fixed_map_metrics(where, loader, device, mask_gap_stop=args.mask_gap_stop,
                                                qtiles=tuple(args.val_qtiles)),
            }
        # Learned substitution matrix (codon x BLOSUM62, 25 free scalars) fitted on each
        # source's training half. Position-blind, so it is the bar for "has the model
        # learned more than substitution chemistry".
        for name, loader in (val_loaders.items()
                             if args.model_type != 'substitution' else ()):
            fitted = fit_substitution_baseline(train_by_source[name], device, allowed_aas)
            scored = validate_per_source(fitted, {name: loader}, device, mask_gap_stop=args.mask_gap_stop,
                                         qtiles=tuple(args.val_qtiles))
            baseline_metrics[name]['substitution'] = scored['per_source'][name]
            baseline_metrics[name]['substitution']['params'] = {
                'kappa': float(torch.exp(fitted.log_kappa)),
                'w_codon': float(fitted.w_codon),
                'w_blosum': float(fitted.w_blosum),
            }

            # P(i, a) = P(i) * P(a | wt_i): the strongest predictor that assumes where
            # and which are independent. The model only earns the word "learned" for
            # position-specific substitution preference if it beats this.
            factored = WhereTimesWhichPredictor(
                fitted.config,
                positional_marginal_counts(train_by_source[name],
                                           seq_len_by_source[name], E),
                fitted).to(device)
            factored_val = validate_per_source(factored, {name: loader}, device, mask_gap_stop=args.mask_gap_stop,
                                               qtiles=tuple(args.val_qtiles))
            baseline_metrics[name]['where_times_which'] = \
                factored_val['per_source'][name]
            del factored, fitted

        # Codon accessibility: a data-free reference point that knows only the genetic
        # code, scored through the same machinery as the model.
        if val_loaders and args.model_type != 'codon':
            codon_model = create_model(
                ModelConfig(model_type=ModelType.CODON, seq_len=seq_len_aa,
                            alphabet=alphabet_config,
                            codon=CodonConfig(kappa=args.codon_kappa,
                                              pseudocount=args.codon_pseudocount)),
                device)
            codon_val = validate_per_source(codon_model, val_loaders, device, mask_gap_stop=args.mask_gap_stop,
                                            qtiles=tuple(args.val_qtiles))
            for name, stats in codon_val['per_source'].items():
                baseline_metrics[name]['codon'] = stats
            del codon_model
        if val_loaders:
            zero_shot = validate_per_source(model, val_loaders, device, mask_gap_stop=args.mask_gap_stop,
                                            qtiles=tuple(args.val_qtiles))
            print("Zero-shot (before any optimizer step):")
            _print_validation(zero_shot, baseline_metrics, sources)
            val_records.append({'epoch': -1, 'global_step': 0,
                                'avg_train_loss': None, 'val': zero_shot,
                                'baselines': baseline_metrics})
            with open(val_metrics_path, 'w') as f:
                json.dump(val_records, f, indent=2)

        print("Starting training...")

        best_path = os.path.join(args.output_dir, f"fitness_{run_prefix}_best.pt")
        cells_path = os.path.join(args.output_dir, f"{run_prefix}_val_cells.npz")

        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)   # without this, every epoch replays one order
            total_train_loss = 0.0
            n_batches = 0

            for batch_num, (aa_batch, target_batch) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                aa_batch = aa_batch.to(device)
                target_batch = target_batch.to(device)

                logits = model(aa_batch).float()
                probs = compute_probs(logits, target_batch, aa_batch)
                loss = compute_loss_batch(probs, target_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_loss_val = loss.item()
                total_train_loss += train_loss_val
                n_batches += 1
                global_step += 1

                if batch_num % 10 == 0:
                    print(f"  Epoch {epoch}, batch {batch_num}: train_loss={train_loss_val:.8f}")

                if args.max_batches is not None and global_step >= args.max_batches:
                    print(f"Reached max batches ({args.max_batches}), stopping.")
                    break

            avg_train_loss = total_train_loss / max(n_batches, 1)
            print(f"Epoch {epoch}: avg_train_loss={avg_train_loss:.8f}")

            if val_loaders:
                dumper = CellDumper() if args.dump_cells else None
                val = validate_per_source(model, val_loaders, device, mask_gap_stop=args.mask_gap_stop,
                                          qtiles=tuple(args.val_qtiles),
                                          cell_sink=dumper)
                _print_validation(val, baseline_metrics, sources)
                val_records.append({'epoch': epoch, 'global_step': global_step,
                                    'avg_train_loss': avg_train_loss, 'val': val,
                                    'baselines': baseline_metrics})
                # Rewritten every epoch so a wall-clock kill still leaves the curve.
                with open(val_metrics_path, 'w') as f:
                    json.dump(val_records, f, indent=2)

                # The final epoch is typically well past the best one, so without this
                # the saved weights and cells are the overfit model, not the scored one.
                # Per-source best epochs, for the per-tree epoch rule. A joint run's
                # sources peak at different epochs, so the pooled checkpoint above is
                # not the best one for any particular tree. Selection still uses only
                # validation data, so this is as causally clean as --save-best.
                # Selection objective (lower is better for both). For recall10_macro the
                # per-source key is 1 - recall@10 and the pooled key is its MACRO mean
                # across sources (every tree equal); the recorded hit_rate curve is
                # untouched, so recall stays readable from the logs.
                if args.select_metric == 'recall10_macro':
                    _probe = next(iter(val['per_source'].values()))['hit_rate']
                    if '0.1' not in _probe:
                        raise SystemExit("--select-metric recall10_macro needs 0.1 in "
                                         "--val-qtiles")
                    per_src_sel = {s: 1.0 - st['hit_rate']['0.1']
                                   for s, st in val['per_source'].items()}
                    pooled_sel = sum(per_src_sel.values()) / max(len(per_src_sel), 1)
                    sel_desc = 'macro(1-recall@10)'
                else:
                    per_src_sel = {s: st['nll_per_mutation']
                                   for s, st in val['per_source'].items()}
                    pooled_sel = val['overall']['nll_per_mutation']
                    sel_desc = 'pooled NLL/mut'

                if args.save_best_per_source:
                    for src, stats in val['per_source'].items():
                        s_nll = per_src_sel[src]
                        if s_nll < best_val_nll_by_source.get(src, float('inf')):
                            best_val_nll_by_source[src] = s_nll
                            best_epoch_by_source[src] = epoch
                            torch.save(
                                build_checkpoint(model, model_config, args, run_prefix,
                                                 seq_len_by_source, source_summary,
                                                 epoch, global_step, avg_train_loss,
                                                 None),
                                os.path.join(args.output_dir,
                                             f"fitness_{run_prefix}_best_{src}.pt"))

                pooled = pooled_sel
                improved = pooled < best_val_nll
                if improved:
                    best_val_nll, best_epoch = pooled, epoch
                    if args.save_best:
                        torch.save(
                            build_checkpoint(model, model_config, args, run_prefix,
                                             seq_len_by_source, source_summary, epoch,
                                             global_step, avg_train_loss, None),
                            best_path)
                        print(f"  New best {sel_desc} {pooled:.8f}; "
                              f"saved {best_path}")
                    if dumper is not None:
                        n_bytes = dumper.save(cells_path, epoch=epoch,
                                              nll_per_mutation=pooled)
                        print(f"  Dumped per-cell maps for epoch {epoch} to "
                              f"{cells_path} ({n_bytes / 1e6:.1f} MB uncompressed)")
                if getattr(args, 'save_every_epoch', False):
                    torch.save(
                        build_checkpoint(model, model_config, args, run_prefix,
                                         seq_len_by_source, source_summary, epoch,
                                         global_step, avg_train_loss, None),
                        os.path.join(args.output_dir,
                                     f"fitness_{run_prefix}_epoch{epoch}.pt"))
                del dumper

            if args.max_batches is not None and global_step >= args.max_batches:
                break

        # Free training-only objects to reclaim memory before saving
        optimizer_state = optimizer.state_dict()
        del optimizer, scheduler, train_loader, train_dataset, training_examples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Freed training objects (optimizer, scheduler, train_loader, etc.) before saving")


    output_path = os.path.join(args.output_dir, f"fitness_{run_prefix}.pt")

    print(f"Saving model to {output_path}...")
    checkpoint = build_checkpoint(model, model_config, args, run_prefix,
                                  seq_len_by_source, source_summary,
                                  epoch, global_step, avg_train_loss, optimizer_state)
    torch.save(checkpoint, output_path)

    if best_epoch_by_source:
        print("Per-source best epochs: " + ', '.join(
            f"{k}={v}" for k, v in sorted(best_epoch_by_source.items())))
    if best_epoch is not None and args.save_best:
        print(f"Best pooled validation NLL {best_val_nll:.8f} at epoch {best_epoch} "
              f"(saved separately as fitness_{run_prefix}_best.pt)")

    print("Done.")


if __name__ == "__main__":
    main()