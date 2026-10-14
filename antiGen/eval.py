#!/usr/bin/env python
"""Unified multi-model evaluation over trained checkpoints + baselines.

Scans a checkpoint directory, groups checkpoints by run "prefix" (everything
between ``fitness_`` and the trailing ``_YYYYMMDD_HHMMSS.pt``), picks the newest
per group, keeps only those whose stored ``train_args`` match ``--dataset``,
loads the evaluation data once, then runs every surviving model plus all
baselines and writes one unified comparison CSV per eval variant.

Re-runs inference from checkpoints — no re-training. EVE/Tranception checkpoints
carry an ``eve_state`` (L_focus, focus_cols) so they rebuild without re-running
MSA pretraining (see eval_utils.load_checkpoint + config.pretrained_state).
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from antiGen.config import ModelType
from antiGen import datasets
from antiGen.models.factory import create_model
from antiGen.load import load_examples, load_pssms
from antiGen.eval_utils import (
    RBD_RANGE,
    evaluate_full,
    variant_hit_rates,
    load_baselines,
    build_special_mask,
    build_model_gap_stop_mask,
    load_checkpoint,
)


# ── Dataset routing ────────────────────────────────────────────────────────────
# Named-dataset constants (paths, variants, cutoffs) live in antiGen.datasets so
# train.py and eval.py share one portable source of truth (no hardcoded paths).

# Top-q quantiles for variant-emergence: q = 0.1 (top decile) … 1.0 (all candidates).
EMERGENCE_QUANTILES = [round(0.1 * i, 4) for i in range(1, 11)]


def dataset_of_train_args(train_args):
    """Which --dataset a checkpoint was trained on, or None if train_args absent."""
    if train_args is None:
        return None
    if train_args.get('flu'):
        return 'flu'
    if train_args.get('rsv_a'):
        return 'rsv_a'
    if train_args.get('denv2'):
        return 'denv2'
    return 'sc2'


# ── Dataset ─────────────────────────────────────────────────────────────────────

class MutationDataset(Dataset):
    def __init__(self, training_examples):
        self.aa_indices_list = [e.aa_idx for e in training_examples]
        self.mutation_matrices = [e.mat for e in training_examples]

    def __len__(self):
        return len(self.aa_indices_list)

    def __getitem__(self, idx):
        return self.aa_indices_list[idx], self.mutation_matrices[idx]


# ── Checkpoint discovery ─────────────────────────────────────────────────────────

_CKPT_RE = re.compile(r"fitness_(.*)_(\d{8}_\d{6})\.pt$")


def discover_checkpoints(checkpoint_dir, model_filters=None, exclude_filters=None):
    """Group ``fitness_*.pt`` files by prefix.

    Returns dict prefix -> list of (timestamp_str, path) sorted newest-first.
    ``model_filters`` (list of substrings) keeps only prefixes matched by at
    least one filter. ``exclude_filters`` (list of substrings) drops matching
    prefixes — but only when the user has NOT explicitly requested them via
    ``model_filters`` (so an explicit --models overrides the default exclusions).
    Selection of the newest *dataset-matching* checkpoint per prefix happens later
    (once the target seq_len is known).
    """
    groups = {}  # prefix -> list of (timestamp_str, path)
    excluded = set()
    for path in glob.glob(os.path.join(checkpoint_dir, "fitness_*.pt")):
        m = _CKPT_RE.search(os.path.basename(path))
        if not m:
            continue
        prefix, ts = m.group(1), m.group(2)
        if model_filters:
            if not any(f in prefix for f in model_filters):
                continue
            # Explicit include via --models overrides the default exclusions.
        elif exclude_filters and any(x in prefix for x in exclude_filters):
            excluded.add(prefix)
            continue
        groups.setdefault(prefix, []).append((ts, path))
    # Timestamps are YYYYMMDD_HHMMSS -> lexicographic == chronological
    for prefix in groups:
        groups[prefix].sort(key=lambda tp: tp[0], reverse=True)
    if excluded:
        print(f"Excluding {len(excluded)} prefix(es) by default (--exclude): {sorted(excluded)}")
    return groups


def peek_checkpoint_meta(path):
    """Cheaply read (train_args, config_dict, global_step) without loading weights.

    Uses mmap so large checkpoints (e.g. ESM-2 650M) aren't read into RAM just to
    route by dataset / derive a config label.
    """
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 -- fall back to a full load if mmap unsupported
        ck = torch.load(path, map_location='cpu', weights_only=False)
    train_args = ck.get('train_args')
    config_dict = ck.get('config', {}) or {}
    global_step = ck.get('global_step')
    del ck
    return train_args, config_dict, global_step


def config_label(prefix, config_dict, global_step):
    """Distinguishing label for a checkpoint whose filename prefix is ambiguous.

    ESM-2 encodes only the learning rate in its run prefix, so the four
    finetuning configs collide (notably frozen vs noft, which share lr=0.001).
    Recover the config from freeze_backbone / finetune_layers, and use
    global_step to tell a trained frozen head (>0) from an untrained noft (0).
    Other model types keep their prefix unchanged.
    """
    mt = config_dict.get("model_type")
    if mt != "esm2_fitness_predictor":
        return prefix
    sub = config_dict.get("esm2", {}) or {}
    if not sub.get("freeze_backbone", True):
        tag = "full"
    elif (sub.get("finetune_layers") or 0) > 0:
        tag = f"partial{sub.get('finetune_layers')}"
    elif (global_step or 0) > 0:
        tag = "frozen"
    else:
        tag = "noft"
    return f"{prefix}_{tag}"


# ── Unified output writers ───────────────────────────────────────────────────────

def write_unified_hit_rate(output_dir, file_suffix, qtile_labels,
                           model_rows, baseline_rows):
    """Write one CSV: rows = model prefixes then baselines, cols = qtiles."""
    csv_path = os.path.join(output_dir, f"quantile_hit_rates_{file_suffix}_all-models.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + qtile_labels)
        for name in sorted(model_rows):
            writer.writerow([name] + [f"{v:.8f}" for v in model_rows[name]])
        for name, vals in baseline_rows.items():
            writer.writerow([name] + [f"{v:.8f}" for v in vals])


def write_unified_variant_emergence(output_dir, file_suffix, n_steps, quantiles,
                                    model_rows, baseline_rows):
    """Write one emergence CSV: rows = (model|baseline, top_quantile), cols = time steps.

    model_rows / baseline_rows map name -> {q: [hit rate per step]}. One row per
    (name, q) so every top-q quantile (q = 0.1 … 0.9) is reported.
    """
    step_labels = [f"t_{(i + 1) * 0.1:.1f}" for i in range(n_steps)]
    csv_path = os.path.join(output_dir, f"variant_emergence_{file_suffix}_all-models.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "top_quantile"] + step_labels)

        def emit(name, qmap):
            for q in quantiles:
                if q in qmap:
                    writer.writerow([name, f"{q:.1f}"] + [f"{v:.8f}" for v in qmap[q]])

        for name in sorted(model_rows):
            emit(name, model_rows[name])
        for name, qmap in baseline_rows.items():
            emit(name, qmap)


def read_emergence_csv(path, baseline_names, prefix_set):
    """Read an emergence CSV into name -> {q: [vals]} dicts (model_rows, baseline_rows).

    Handles the (model, top_quantile, steps...) row layout for resume. Returns
    ({}, {}) if the file is absent or in the old (no top_quantile column) layout.
    """
    model_rows, baseline_rows = {}, {}
    if not os.path.exists(path):
        return model_rows, baseline_rows
    baseline_names = set(baseline_names)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2 or header[1] != "top_quantile":
            return model_rows, baseline_rows  # old layout -> treat as absent (recompute)
        for row in reader:
            if not row or len(row) < 2:
                continue
            name = row[0]
            try:
                q = round(float(row[1]), 4)
                vals = [float(x) for x in row[2:]]
            except ValueError:
                continue
            if name in baseline_names:
                baseline_rows.setdefault(name, {})[q] = vals
            elif name in prefix_set:
                model_rows.setdefault(name, {})[q] = vals
    return model_rows, baseline_rows


def read_unified_csv(path, baseline_names, prefix_set):
    """Read an existing unified CSV into (model_rows, baseline_rows) float-value dicts.

    Rows whose label is a known baseline go to baseline_rows; rows whose label is a
    known model prefix go to model_rows. Used to resume a run: models already on disk
    are skipped and their rows reused. Returns ({}, {}) if the file is absent.
    """
    model_rows, baseline_rows = {}, {}
    if not os.path.exists(path):
        return model_rows, baseline_rows
    baseline_names = set(baseline_names)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            name = row[0]
            try:
                vals = [float(x) for x in row[1:]]
            except ValueError:
                continue
            if name in baseline_names:
                baseline_rows[name] = vals
            elif name in prefix_set:
                model_rows[name] = vals
    return model_rows, baseline_rows


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Unified multi-model fitness evaluation')
    parser.add_argument('--dataset', type=str, required=True, choices=['sc2', 'flu', 'rsv_a', 'denv2'])
    parser.add_argument('--include-experimental', action='store_true', default=False,
                        help='Also evaluate checkpoints tagged with an experimental arm '
                             '(antiGen-train --experiment), which are not forecast-valid.')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory to scan for fitness_*.pt checkpoints')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for unified comparison CSVs')
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Override the data directory (else derived from checkpoints' train_args)")
    parser.add_argument('--variant', type=str, default=None,
                        help='Override the variant name (else from train_args or dataset default)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated prefix substrings to filter which model groups run '
                             '(explicit include; overrides --exclude)')
    parser.add_argument('--exclude', type=str, default='transformer_,esmc,vespag',
                        help='Comma-separated prefix substrings to skip. Defaults to the legacy '
                             '"transformer_" hyperparameter-sweep checkpoints (the from-scratch '
                             'transformer to report is the "simple_" run) and the retired '
                             '"esmc"/"vespag" models (removed from the package). Ignored for '
                             'prefixes explicitly named in --models.')
    parser.add_argument('--priority-models', type=str, default=None,
                        help='Comma-separated label substrings to evaluate FIRST (before the '
                             'other models). Baselines are always computed first regardless.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for baseline tie-break noise')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hit-rate-buckets', type=int, default=10)
    parser.add_argument('--emergence-only', action='store_true', default=False,
                        help='Recompute only the variant-emergence CSVs (one forward pass per '
                             'model on the testing root; no test-set hit-rate passes). Reuses/'
                             'leaves the quantile_hit_rates CSVs untouched. Fast.')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Bit-reproducible mode: seed all RNGs, force deterministic torch '
                             'algorithms (needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the env), and '
                             're-seed before each model so stochastic rebuilds (EVE/Tranception '
                             'sampling) are reproducible. NOTE: the EVcouplings plmc fit is an '
                             'external binary and is not covered by this flag.')
    return parser.parse_args()


def _enable_determinism(seed: int) -> None:
    """Make the eval bit-reproducible run-to-run *without changing the numerics*.

    Seeds every RNG (the only genuine randomness is EVE/Tranception's sampling and
    the EVcouplings MSA subsample), enables deterministic algorithms in warn-only
    mode, and disables cuDNN autotuning. Caller must also export
    CUBLAS_WORKSPACE_CONFIG=:4096:8 *before* CUDA initializes.

    Deliberately does NOT override the scaled-dot-product-attention backend: the
    default (flash / mem-efficient) forward is already bit-reproducible within a
    GPU architecture, and forcing the math backend would shift the from-scratch
    transformer's logits (~4e-6) enough to flip near-boundary cells in the ranking
    metrics — i.e. it would diverge from the non-deterministic reference results.
    Keeping the default backend reproduces those results exactly (on the same GPU
    class). The EVcouplings plmc fit is external and not covered by this flag.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.deterministic:
        _enable_determinism(args.seed)
        print(f"Mode: --deterministic (seed={args.seed}, deterministic torch algorithms; "
              f"EVcouplings plmc fit is external and not covered)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)
    gap_idx = allowed_aas.index('-')
    stop_idx = allowed_aas.index('*')
    n_buckets = args.hit_rate_buckets
    emergence_only = args.emergence_only
    if emergence_only:
        print("Mode: --emergence-only (recompute variant-emergence CSVs; hit-rate CSVs untouched)")

    model_filters = [s.strip() for s in args.models.split(',') if s.strip()] if args.models else None
    exclude_filters = [s.strip() for s in args.exclude.split(',') if s.strip()] if args.exclude else None

    # ── 1. Discover checkpoint groups (prefix -> newest-first list) ──────────
    groups = discover_checkpoints(args.checkpoint_dir, model_filters, exclude_filters)
    if not groups:
        print(f"No checkpoints matched in {args.checkpoint_dir} (filters={model_filters}). Nothing to do.")
        return
    print(f"Found {len(groups)} checkpoint group(s): {sorted(groups)}")

    # ── 2. Resolve data paths ────────────────────────────────────────────────
    if args.data_dir:
        data_dir = args.data_dir
    else:
        allowed_cutoffs = datasets.DATASETS[args.dataset]["allowed_cutoffs"]
        if allowed_cutoffs is not None and len(allowed_cutoffs) == 1:
            # flu/rsv/denv: single fixed cutoff -> deterministic dir (needs $FITNESS_DATA_ROOT)
            data_dir = datasets.default_data_dir(args.dataset, allowed_cutoffs[0])
        else:
            # sc2 (multiple cutoffs): derive the cutoff from any matching checkpoint's train_args
            data_dir = None
            for prefix in groups:
                for _ts, p in groups[prefix]:
                    ta, _cfg, _gs = peek_checkpoint_meta(p)
                    if ta is not None and dataset_of_train_args(ta) == args.dataset:
                        data_dir = datasets.default_data_dir(args.dataset, ta['cutoff'])
                        break
                if data_dir:
                    break
            if data_dir is None:
                raise SystemExit(
                    f"Cannot determine data dir for '{args.dataset}' without train_args; pass --data-dir")

    variant = args.variant or datasets.default_variant(args.dataset)

    examples_files = sorted(glob.glob(os.path.join(data_dir, "examples_cutoff_*.txt")))
    if not examples_files:
        raise SystemExit(f"No examples_cutoff_*.txt found in {data_dir}")
    examples_path = examples_files[0]
    print(f"Dataset={args.dataset} variant={variant}")
    print(f"Loading examples from {examples_path}...")
    all_examples = load_examples(examples_path, allowed_aas, variant)

    seq_len = len(all_examples[0].aa_idx)
    print(f"Sequence length: {seq_len}")
    eval_range = list(range(seq_len)) if args.dataset in ('flu', 'rsv_a', 'denv2') else RBD_RANGE

    # ── 3. Select the newest dataset-matching checkpoint per (prefix, config) ─
    # Match by train_args when present, else (legacy checkpoints) by seq_len.
    # A single filename prefix can hold multiple distinct configs (ESM-2
    # encode only lr in the prefix, so full/partial/frozen/noft collide). Group by
    # a config-derived label and keep the newest dataset-matching checkpoint per
    # label, so all configs surface as separate rows.
    candidates = []  # (label, path, train_args)
    for prefix in sorted(groups):
        chosen_by_label = {}  # label -> (path, train_args)  (first hit = newest)
        for _ts, path in groups[prefix]:  # newest-first
            ta, cfg, gstep = peek_checkpoint_meta(path)
            if ta is not None:
                # Experimental arms (multi-tree, random validation split) are not
                # forecast-valid and must not land in a comparison CSV by accident.
                if ta.get('experiment') and not args.include_experimental:
                    print(f"[skip] {prefix}: experimental arm "
                          f"'{ta['experiment']}' (pass --include-experimental)")
                    continue
                if dataset_of_train_args(ta) != args.dataset:
                    continue
                matched = True
            else:
                matched = (cfg.get("seq_len") == seq_len)
            if not matched:
                continue
            label = config_label(prefix, cfg, gstep)
            if label not in chosen_by_label:  # newest-first => first seen is newest
                chosen_by_label[label] = (path, ta)
        if chosen_by_label:
            for label in sorted(chosen_by_label):
                path, ta = chosen_by_label[label]
                tag = "train_args" if ta is not None else f"legacy seq_len={seq_len}"
                print(f"[keep] {label}: {os.path.basename(path)} ({tag})")
                candidates.append((label, path, ta))
        else:
            print(f"[skip] {prefix}: no checkpoint matches dataset '{args.dataset}'")
    if not candidates:
        print(f"No checkpoints match --dataset {args.dataset}. Nothing to do.")
        return

    # Optionally evaluate some models first (e.g. the from-scratch transformer), so
    # their rows land early even if the heavy models (650M) push the job to the wall.
    if args.priority_models:
        prio = [s.strip() for s in args.priority_models.split(',') if s.strip()]
        is_prio = lambda label: any(p in label for p in prio)
        first = [c for c in candidates if is_prio(c[0])]
        rest = [c for c in candidates if not is_prio(c[0])]
        candidates = first + rest
        print(f"Priority-first models: {[c[0] for c in first]}")

    testing_examples = [e for e in all_examples if not e.is_training]
    variant_testing_examples = [e for e in testing_examples if e.is_descended_from_variant]
    training_examples = [e for e in all_examples if e.is_training]
    global_wt_aa_idx = training_examples[0].aa_idx.clone()
    print(f"Testing examples: {len(testing_examples)}, "
          f"variant-descended: {len(variant_testing_examples)}")

    # PSSMs + baselines
    pssm_dir = os.path.join(data_dir, "pssms")
    pssm_root, pssm_variant = load_pssms(pssm_dir, variant, E)

    if args.dataset == 'flu':
        baseline_subdir = "baselines_flu"
    elif args.dataset == 'rsv_a':
        baseline_subdir = "baselines_rsv_a"
    elif args.dataset == 'denv2':
        baseline_subdir = "baselines_denv2"
    else:
        baseline_subdir = "baselines"
    baseline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", baseline_subdir)
    baselines = load_baselines(baseline_dir, allowed_aas, args.seed, pssm_root, global_wt_aa_idx)

    # Loaders
    test_loader = DataLoader(
        MutationDataset(testing_examples), batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    variant_test_loader = None
    if variant_testing_examples:
        variant_test_loader = DataLoader(
            MutationDataset(variant_testing_examples), batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False,
        )

    # ── 4. Eval-variant masks ────────────────────────────────────────────────
    special_mask = build_special_mask(seq_len, E, gap_idx, stop_idx)
    hit_rate_seen_mask = pssm_root > 0  # (seq_len, E) True = seen in training root

    test_variants = [
        ("All", "all", None),
        ("Unseen-mutations", "unseen", hit_rate_seen_mask),
        ("All (no gap/stop)", "all_nogapstop", special_mask),
        ("Unseen (no gap/stop)", "unseen_nogapstop", hit_rate_seen_mask | special_mask),
    ]
    variant_variants = [
        ("Variant-descended", "variant", None),
        ("Variant-descended (no gap/stop)", "variant_nogapstop", special_mask),
    ]

    qtile_labels = [f"top_{(i + 1) * 100 / n_buckets:.1f}%" for i in range(n_buckets - 1)]

    # ── 5. Variant-emergence (testing-root) setup ────────────────────────────
    testing_root_examples = [e for e in all_examples if e.is_testing_root]
    do_emergence = bool(pssm_variant) and len(testing_root_examples) >= 1
    if do_emergence:
        testing_root_example = testing_root_examples[0]
        wt_idx = testing_root_example.aa_idx[eval_range]
        wt_mask_2d = torch.ones(len(eval_range), E, dtype=torch.bool)
        wt_mask_2d.scatter_(1, wt_idx.unsqueeze(-1), False)

        pssm_variant_sub = [pv[eval_range, :] for pv in pssm_variant]
        pssm_root_sub = pssm_root[eval_range, :]
        emergence_seen_mask = pssm_root_sub > 0
        pssm_variant_sub_unseen = []
        for pv in pssm_variant_sub:
            pvm = pv.clone()
            pvm[emergence_seen_mask] = float('nan')
            pssm_variant_sub_unseen.append(pvm)

        # wt masks for the no-gap/stop and unseen+no-gap/stop specs
        wt_mask_nogapstop = wt_mask_2d.clone()
        wt_mask_nogapstop[:, gap_idx] = False
        wt_mask_nogapstop[:, stop_idx] = False
        combined_mask = emergence_seen_mask.clone()
        combined_mask[:, gap_idx] = True
        combined_mask[:, stop_idx] = True
        wt_mask_unseen_nogapstop = wt_mask_2d.clone()
        wt_mask_unseen_nogapstop[combined_mask] = False

        variant_thresholds = [100, 1] if variant == 'BA.2' else [10, 1]
        T0 = variant_thresholds[0]

        # Model-independent baseline score maps per spec (subset to eval range + masks)
        def _sub(bp):
            return bp[eval_range, :].clone()

        bl_base = [(n, _sub(bp)) for n, bp in baselines]
        bl_unseen = []
        for n, bp in baselines:
            s = _sub(bp)
            s[emergence_seen_mask] = float('nan')
            bl_unseen.append((n, s))
        bl_nogapstop = []
        for n, bp in baselines:
            s = _sub(bp)
            s[:, gap_idx] = float('nan')
            s[:, stop_idx] = float('nan')
            bl_nogapstop.append((n, s))
        bl_unseen_nogapstop = []
        for n, bp in baselines:
            s = _sub(bp)
            s[combined_mask] = float('nan')
            bl_unseen_nogapstop.append((n, s))

        # spec: (file_suffix, pssm_list, model_prob_key, wt_mask, baseline_set, threshold)
        emergence_specs = [
            ("all", pssm_variant_sub, "base", wt_mask_2d, bl_base, T0),
            ("unseen", pssm_variant_sub_unseen, "unseen", wt_mask_2d, bl_unseen, T0),
            ("all_thresh1", pssm_variant_sub, "base", wt_mask_2d, bl_base, 1),
            ("unseen_thresh1", pssm_variant_sub_unseen, "unseen", wt_mask_2d, bl_unseen, 1),
            ("all_nogapstop", pssm_variant_sub, "nogapstop", wt_mask_nogapstop, bl_nogapstop, T0),
            ("all_nogapstop_thresh1", pssm_variant_sub, "nogapstop", wt_mask_nogapstop, bl_nogapstop, 1),
            ("unseen_nogapstop", pssm_variant_sub_unseen, "unseen_nogapstop", wt_mask_unseen_nogapstop, bl_unseen_nogapstop, T0),
            ("unseen_nogapstop_thresh1", pssm_variant_sub_unseen, "unseen_nogapstop", wt_mask_unseen_nogapstop, bl_unseen_nogapstop, 1),
        ]

        n_steps = len(pssm_variant_sub)
        # Accumulators: suffix -> {name: {q: [hit rate per step]}}
        emergence_model_rows = {suffix: {} for suffix, *_ in emergence_specs}
        emergence_baseline_rows = {}
        for suffix, pssm_list, _mp, wt_m, bl_set, thr in emergence_specs:
            emergence_baseline_rows[suffix] = {
                name: variant_hit_rates(bp, wt_m, pssm_list, threshold_count=thr,
                                        quantiles=EMERGENCE_QUANTILES)
                for name, bp in bl_set
            }

    # ── 6. Resume: preload existing unified CSVs so completed models are skipped ──
    prefix_set = {p for p, _, _ in candidates}
    baseline_names = [n for n, _ in baselines]

    hit_rate_model_rows = {suffix: {} for _, suffix, _ in test_variants}
    variant_hr_model_rows = {suffix: {} for _, suffix, _ in variant_variants}
    test_baseline_rows = {}
    variant_baseline_rows = {}

    for _, suffix, _ in test_variants:
        mr, br = read_unified_csv(
            os.path.join(args.output_dir, f"quantile_hit_rates_{suffix}_all-models.csv"),
            baseline_names, prefix_set)
        hit_rate_model_rows[suffix] = mr
        if br:
            test_baseline_rows[suffix] = br
    if variant_test_loader is not None:
        for _, suffix, _ in variant_variants:
            mr, br = read_unified_csv(
                os.path.join(args.output_dir, f"quantile_hit_rates_{suffix}_all-models.csv"),
                baseline_names, prefix_set)
            variant_hr_model_rows[suffix] = mr
            if br:
                variant_baseline_rows[suffix] = br
    if do_emergence:
        for suffix, *_ in emergence_specs:
            mr, _br = read_emergence_csv(
                os.path.join(args.output_dir, f"variant_emergence_{suffix}_all-models.csv"),
                baseline_names, prefix_set)
            emergence_model_rows[suffix].update(mr)

    def flush_outputs():
        """(Re)write unified CSVs from current accumulators. Called after the baseline
        sweep and after each model so a timeout/crash never loses finished work. In
        --emergence-only mode the hit-rate CSVs are left untouched."""
        if not emergence_only:
            for _, suffix, _ in test_variants:
                if suffix in test_baseline_rows:
                    write_unified_hit_rate(args.output_dir, suffix, qtile_labels,
                                           hit_rate_model_rows[suffix], test_baseline_rows[suffix])
            if variant_test_loader is not None:
                for _, suffix, _ in variant_variants:
                    if suffix in variant_baseline_rows:
                        write_unified_hit_rate(args.output_dir, suffix, qtile_labels,
                                               variant_hr_model_rows[suffix], variant_baseline_rows[suffix])
        if do_emergence:
            for suffix, *_ in emergence_specs:
                write_unified_variant_emergence(args.output_dir, suffix, n_steps, EMERGENCE_QUANTILES,
                                                emergence_model_rows[suffix], emergence_baseline_rows[suffix])

    # ── 7. Baseline sweep (once, no model) — before models so baselines land on disk
    #        immediately and are reused on resume. Skipped entirely in emergence-only
    #        mode (variant-emergence baselines were already computed in step 5). ─────
    if not emergence_only:
        if any(suffix not in test_baseline_rows for _, suffix, _ in test_variants):
            print("\n=== Baseline sweep (test set) ===")
            base_results = evaluate_full(
                None, test_loader, device,
                variants=[(label, sm) for label, _, sm in test_variants],
                baselines=baselines, n_buckets=n_buckets, eval_range=eval_range,
            )
            for label, suffix, _ in test_variants:
                _, _, baseline_hits = base_results[label]
                test_baseline_rows[suffix] = baseline_hits
        else:
            print("\nReusing test-set baseline rows from existing CSVs.")

        if variant_test_loader is not None and any(
                suffix not in variant_baseline_rows for _, suffix, _ in variant_variants):
            print("=== Baseline sweep (variant-descended) ===")
            vb_results = evaluate_full(
                None, variant_test_loader, device,
                variants=[(label, sm) for label, _, sm in variant_variants],
                baselines=baselines, n_buckets=n_buckets, eval_range=eval_range,
            )
            for label, suffix, _ in variant_variants:
                _, _, baseline_hits = vb_results[label]
                variant_baseline_rows[suffix] = baseline_hits

    # Persist baselines / emergence baselines now, before the first (possibly slow) model.
    flush_outputs()

    # ── 8. Per-model evaluation (incremental writes, resume-safe) ─────────────
    # A model is "done" when its rows exist in the outputs this run produces:
    # emergence CSVs in --emergence-only mode, else the hit-rate 'all' CSV.
    if emergence_only:
        done_ref = emergence_model_rows[emergence_specs[0][0]] if do_emergence else {}
        already_done = set(done_ref)
    else:
        already_done = set(hit_rate_model_rows[test_variants[0][1]])
    if already_done:
        print(f"Resuming: skipping {len(already_done)} model(s) already in output CSVs: {sorted(already_done)}")

    for prefix, path, train_args in candidates:
        if prefix in already_done:
            print(f"[resume-skip] {prefix}: already present in output CSVs")
            continue
        # Re-seed per model so stochastic rebuilds (EVE/Tranception sampling) are
        # reproducible regardless of evaluation order / prior RNG consumption.
        if args.deterministic:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        print(f"\n=== Evaluating model group: {prefix} ===")
        try:
            ckpt, model_config = load_checkpoint(path, map_location=device)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {prefix}: load_checkpoint failed ({exc})")
            continue

        if model_config.seq_len != seq_len:
            print(f"[skip] {prefix}: checkpoint seq_len={model_config.seq_len} != data seq_len={seq_len}")
            continue

        try:
            model = create_model(model_config, device)
            missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if missing:
                print(f"  [warn] {prefix}: {len(missing)} missing state_dict keys (e.g. {missing[:3]})")
            if unexpected:
                print(f"  [warn] {prefix}: {len(unexpected)} unexpected state_dict keys (e.g. {unexpected[:3]})")
            model.eval()
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {prefix}: model build/load failed ({exc})")
            del ckpt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        model_mask = build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx)

        # Quantile hit rates over the test set (the expensive part; skipped in
        # --emergence-only mode, which touches only the single testing-root pass below).
        if not emergence_only:
            results = evaluate_full(
                model, test_loader, device,
                variants=[(label, sm) for label, _, sm in test_variants],
                baselines=[], n_buckets=n_buckets, eval_range=eval_range,
                model_gap_stop_mask=model_mask,
            )
            for label, suffix, _ in test_variants:
                _, hit_rate_by_qtile, _ = results[label]
                hit_rate_model_rows[suffix][prefix] = hit_rate_by_qtile

            # Variant-descended hit rates
            if variant_test_loader is not None:
                v_results = evaluate_full(
                    model, variant_test_loader, device,
                    variants=[(label, sm) for label, _, sm in variant_variants],
                    baselines=[], n_buckets=n_buckets, eval_range=eval_range,
                    model_gap_stop_mask=model_mask,
                )
                for label, suffix, _ in variant_variants:
                    _, hit_rate_by_qtile, _ = v_results[label]
                    variant_hr_model_rows[suffix][prefix] = hit_rate_by_qtile

        # Variant emergence (single testing-root forward pass; per top-q quantile)
        if do_emergence:
            with torch.no_grad():
                aa_input = testing_root_example.aa_idx.unsqueeze(0).to(device)
                tr_scores = model(aa_input).float().squeeze(0).cpu()[eval_range, :]
            mp_base = tr_scores
            mp_unseen = tr_scores.clone()
            mp_unseen[emergence_seen_mask] = float('nan')
            mp_nogapstop = tr_scores.clone()
            mp_nogapstop[:, gap_idx] = float('nan')
            mp_nogapstop[:, stop_idx] = float('nan')
            mp_unseen_nogapstop = tr_scores.clone()
            mp_unseen_nogapstop[combined_mask] = float('nan')
            mp_by_key = {
                "base": mp_base, "unseen": mp_unseen,
                "nogapstop": mp_nogapstop, "unseen_nogapstop": mp_unseen_nogapstop,
            }
            for suffix, pssm_list, mp_key, wt_m, _bl, thr in emergence_specs:
                emergence_model_rows[suffix][prefix] = variant_hit_rates(
                    mp_by_key[mp_key], wt_m, pssm_list, threshold_count=thr,
                    quantiles=EMERGENCE_QUANTILES,
                )

        # Persist after each model so a timeout/crash keeps all finished models.
        flush_outputs()
        print(f"[write] flushed outputs after model: {prefix}")

        # Free model between groups
        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final flush (outputs were already written incrementally after each model).
    flush_outputs()
    print(f"\n=== Wrote unified outputs to {args.output_dir} ===")
    print("Done.")


if __name__ == "__main__":
    main()
