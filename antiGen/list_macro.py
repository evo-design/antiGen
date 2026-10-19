#!/usr/bin/env python
"""Cell-weighted (macro) recall on a curated mutation list.

**The metric this adds.** Figure 2's hit rate, and every curated-list variant built on
it, is an average over mutation *events*: every (example, position, residue) target in
the test set counts once, so a cell that arises on 972 branches counts 972 times and a
cell that arises once counts once. On a curated list that weighting dominates the
result -- CoG-UK's 247 in-RBD cells carry 26,491 test events, median 46 per cell, and
81% of those events sit on the 104 cells the model already saw in training (NOTES s31).

This script scores each listed mutation **once**:

    for a cell c observed relative to k base sequences,
        score(c) = (1 / k) * #{those base sequences whose top-q the model puts c in}
    metric  = mean over listed cells of score(c)

The inner question is unchanged -- "is this cell in this query's top 10%?" is decided
by exactly the same ``compute_hit_rate`` call, with the same per-example pool and the
same fractional tie-breaking -- so the two numbers differ only in how they are averaged.
Both are reported side by side, and the event-weighted column reproduces the published
``quantile_hit_rates_list_*`` CSVs cell for cell, which is how this script is checked.

**Only 10.7% of the test set is scored, and that is exact, not an approximation.** An
example with no listed cell among its targets contributes nothing to either the
numerator or the denominator of any list variant, so dropping it changes no number.
17,978 of 167,600 examples carry a CoG-UK RBD target; restricting the forward passes to
those is what makes this a ~1h job rather than a ~10h one.

Variants, matching ``eval.py``'s naming:

    <list>_fullpool           targets = listed cells, pool = every eval-range cell
                              (Figure 2's question asked of a subset -- the headline)
    <list>                    targets = pool = listed cells (the de novo convention;
                              a near-negative-free pool, see NOTES s25)
    <list>_unseen_fullpool    as _fullpool, restricted to cells absent from the
                              training window (pssm_root == 0)
    <list>_unseen             targets = pool = those cells

Usage:
    python antiGen/list_macro.py \
        --data-dir /home/ivan.specht/antiGen-data/output1 \
        --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
        --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2_lists_macro \
        --mutation-list coguk=coguk.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import os

import torch
from torch.utils.data import DataLoader

from antiGen import datasets
from antiGen.config import (AlphabetConfig, CodonConfig, ModelConfig, ModelType)
from antiGen.eval import (MutationDataset, config_label, dataset_of_train_args,
                          discover_checkpoints, peek_checkpoint_meta)
from antiGen.eval_utils import (RBD_RANGE, build_model_gap_stop_mask,
                                compute_hit_rate, fit_substitution_baseline,
                                load_baselines, load_checkpoint,
                                positional_marginal_counts,
                                WhereTimesWhichPredictor)
from antiGen.load import load_examples, load_pssms
from antiGen.models.factory import create_model
from antiGen.mutation_lists import describe as describe_list, load_mutation_list

SUMMARY = "list_macro_summary.csv"
PERCELL = "list_macro_percell.csv"


def build_variants(lists, seen_mask):
    """(suffix, score_mask, target_mask) per list variant. Masks are True = exclude."""
    variants = []
    for label, in_full in lists:
        variants.append((f"list_{label}_fullpool", None, ~in_full))
        variants.append((f"list_{label}", ~in_full, ~in_full))
        variants.append((f"list_{label}_unseen_fullpool", seen_mask, (~in_full) | seen_mask))
        variants.append((f"list_{label}_unseen", (~in_full) | seen_mask, (~in_full) | seen_mask))
    return variants


def cell_pass(score_fn, loader, variants, device, eval_range, qtiles):
    """Accumulate per-cell hit mass and event counts over the loader.

    ``score_fn(aa_batch) -> (B, seq_len, E)`` is the only thing that differs between a
    model (forward pass) and a baseline (a fixed map broadcast to the batch), so both
    go through this one function and are masked and ranked identically.

    Returns ``acc[(suffix, qtile)] = (cell_hits, cell_events)``, each (R, E).
    """
    acc = {}
    n_batches = len(loader)
    for batch_idx, (aa_batch, target_batch) in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"    batch {batch_idx}/{n_batches}", flush=True)
        aa_batch = aa_batch.to(device)
        target_batch = target_batch.to(device)
        scores = score_fn(aa_batch)

        for suffix, score_mask, target_mask in variants:
            if score_mask is None:
                v_scores = scores
            else:
                v_scores = scores.clone()
                v_scores[score_mask.to(device).unsqueeze(0).expand_as(v_scores)] = float('nan')
            if target_mask is None:
                v_target = target_batch
            else:
                v_target = target_batch.clone()
                v_target[target_mask.to(device).unsqueeze(0).expand_as(v_target)] = 0

            for qtile in qtiles:
                _h, _n, cell_hits, cell_events = compute_hit_rate(
                    v_scores, v_target, aa_batch, qtile, eval_range=eval_range,
                    per_cell=True)
                key = (suffix, qtile)
                if key not in acc:
                    acc[key] = [torch.zeros_like(cell_hits), torch.zeros_like(cell_events)]
                acc[key][0] += cell_hits
                acc[key][1] += cell_events

        if batch_idx % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {k: (v[0].cpu(), v[1].cpu()) for k, v in acc.items()}


def _masked_scores(model, aa_batch, model_gap_stop_mask):
    """One forward pass, with the cells a model does not legitimately emit NaN-ed."""
    with torch.no_grad():
        scores = model(aa_batch).float()
    scores[:, model_gap_stop_mask] = float('nan')
    return scores


def summarize(acc):
    """acc -> rows of (suffix, qtile, macro, micro, n_cells, n_events)."""
    rows = []
    for (suffix, qtile), (cell_hits, cell_events) in sorted(acc.items()):
        observed = cell_events > 0
        n_events = float(cell_events.sum())
        n_cells = int(observed.sum())
        macro = float((cell_hits[observed] / cell_events[observed]).mean()) if n_cells else float('nan')
        micro = float(cell_hits.sum() / n_events) if n_events else float('nan')
        rows.append((suffix, qtile, macro, micro, n_cells, n_events))
    return rows


def write_summary(output_dir, by_model):
    path = os.path.join(output_dir, SUMMARY)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "qtile", "macro", "micro", "n_cells", "n_events"])
        for model in by_model:
            for suffix, qtile, macro, micro, n_cells, n_events in by_model[model]:
                w.writerow([model, suffix, f"{qtile:.2f}", f"{macro:.8f}",
                            f"{micro:.8f}", n_cells, f"{n_events:.0f}"])
    os.replace(tmp, path)


def append_percell(output_dir, model, acc, qtile_wanted, allowed_aas, eval_range):
    """Append this model's per-cell rates at the headline quantile, for inspection.

    Appended rather than rewritten so a resumed run keeps the models already scored.
    """
    path = os.path.join(output_dir, PERCELL)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["model", "variant", "qtile", "pos1", "alt", "k_events", "hits", "rate"])
        for (suffix, qtile), (cell_hits, cell_events) in sorted(acc.items()):
            if abs(qtile - qtile_wanted) > 1e-9:
                continue
            for r, e in (cell_events > 0).nonzero().tolist():
                w.writerow([model, suffix, f"{qtile:.2f}", eval_range[r] + 1,
                            allowed_aas[e], f"{cell_events[r, e]:.0f}",
                            f"{cell_hits[r, e]:.4f}",
                            f"{float(cell_hits[r, e] / cell_events[r, e]):.6f}"])


def read_done(output_dir):
    path = os.path.join(output_dir, SUMMARY)
    by_model = {}
    if not os.path.exists(path):
        return by_model
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            model, suffix, qtile, macro, micro, n_cells, n_events = row
            by_model.setdefault(model, []).append(
                (suffix, float(qtile), float(macro), float(micro), int(n_cells), float(n_events)))
    return by_model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='sc2', choices=sorted(datasets.DATASETS))
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mutation-list', action='append', required=True,
                    help='LABEL=PATH, repeatable')
    ap.add_argument('--eval-range', default='default', choices=['default', 'full'])
    ap.add_argument('--variant', default=None)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--models', default=None, help='comma-separated substrings to keep')
    ap.add_argument('--exclude', default='transformer_,esmc,vespag',
                    help='Comma-separated prefix substrings to skip. The default is '
                         'eval.py\'s, so this script scores the same roster the '
                         'published CSVs do: the legacy "transformer_" sweep (the '
                         'from-scratch transformer to report is "simple_") and the '
                         'retired esmc/vespag models, which no longer build.')
    ap.add_argument('--include-experimental', action='store_true')
    ap.add_argument('--qtiles', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    ap.add_argument('--percell-qtile', type=float, default=0.1)
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--multitree-baselines', action='store_true',
                    help='Also score the cross-tree factored baselines (codon, '
                         'substitution, where x which). They read the residue at each '
                         'position of the query, so they cannot be expressed as the '
                         'fixed (L, E) maps data/baselines/ holds and go through the '
                         'model path instead -- which also means they inherit the '
                         'gap/stop masking. Fitted on the TRAINING examples only.')
    ap.add_argument('--substitution-seed', type=int, default=14)
    ap.add_argument('--skip-baselines', action='store_true',
                    help='Score only the checkpoints. The CSV baselines are fixed maps '
                         'and cost ~2 min each; skipping them lets a second job get at '
                         'the checkpoints without re-deriving rows another run already '
                         'has. The rows it does write are identical either way.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)
    gap_idx, stop_idx = allowed_aas.index('-'), allowed_aas.index('*')
    qtiles = [float(q) for q in args.qtiles.split(',') if q.strip()]
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    variant = args.variant or datasets.default_variant(args.dataset)
    examples_files = sorted(glob.glob(os.path.join(args.data_dir, "examples_cutoff_*.txt")))
    if not examples_files:
        raise SystemExit(f"No examples_cutoff_*.txt found in {args.data_dir}")
    all_examples = load_examples(examples_files[0], allowed_aas, variant)
    seq_len = len(all_examples[0].aa_idx)
    eval_range = list(range(seq_len)) if args.eval_range == 'full' else (
        list(range(seq_len)) if args.dataset in ('flu', 'rsv_a', 'denv2') else RBD_RANGE)
    print(f"Sequence length {seq_len}; eval range {len(eval_range)} positions "
          f"({eval_range[0] + 1}-{eval_range[-1] + 1}, 1-based)")

    testing_examples = [e for e in all_examples if not e.is_training]
    training_examples = [e for e in all_examples if e.is_training]
    global_wt_aa_idx = training_examples[0].aa_idx.clone()

    # ── Lists ───────────────────────────────────────────────────────────────
    root_seq = "".join(allowed_aas[i] for i in global_wt_aa_idx.tolist())
    lists = []
    in_range = torch.zeros(seq_len, E, dtype=torch.bool)
    in_range[eval_range, :] = True
    for item in (args.mutation_list or []):
        if '=' not in item:
            raise SystemExit(f"--mutation-list wants LABEL=PATH, got {item!r}")
        label, path = item.split('=', 1)
        cells, report = load_mutation_list(path, allowed_aas, seq_len,
                                           root_seq=root_seq, label=label)
        in_full = torch.zeros(seq_len, E, dtype=torch.bool)
        for pos, aa in cells:
            in_full[pos, aa] = True
        in_full &= in_range
        print(describe_list(report, in_range=int(in_full.sum())))
        lists.append((label, in_full))

    any_listed = torch.zeros(seq_len, E, dtype=torch.bool)
    for _label, in_full in lists:
        any_listed |= in_full

    # ── Only the examples that carry a listed target ────────────────────────
    # Exact, not a sample: an example with no listed target contributes 0 to both the
    # numerator and the denominator of every variant here.
    scored = [e for e in testing_examples if bool((e.mat[any_listed] > 0).any())]
    n_events = sum(float(e.mat[any_listed].sum()) for e in scored)
    print(f"Testing examples: {len(testing_examples)}; carrying a listed target: "
          f"{len(scored)} ({100 * len(scored) / len(testing_examples):.1f}%), "
          f"{n_events:.0f} listed target events")
    if not scored:
        raise SystemExit("No test example carries a listed cell in the eval range.")

    loader = DataLoader(MutationDataset(scored), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    pssm_root, _pssm_variant = load_pssms(os.path.join(args.data_dir, "pssms"), variant, E)
    seen_mask = pssm_root > 0
    variants = build_variants(lists, seen_mask)
    for label, in_full in lists:
        n_listed = int(in_full.sum())
        n_unseen = int((in_full & ~seen_mask).sum())
        print(f"  {label}: {n_listed} cells in eval range, {n_unseen} de novo "
              f"(absent from the training window)")

    baseline_subdir = {'flu': 'baselines_flu', 'rsv_a': 'baselines_rsv_a',
                       'denv2': 'baselines_denv2'}.get(args.dataset, 'baselines')
    baseline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", baseline_subdir)
    baselines = load_baselines(baseline_dir, allowed_aas, args.seed, pssm_root, global_wt_aa_idx)

    # ── Checkpoints ─────────────────────────────────────────────────────────
    model_filters = [s.strip() for s in args.models.split(',') if s.strip()] if args.models else None
    exclude_filters = [s.strip() for s in args.exclude.split(',') if s.strip()] if args.exclude else None
    candidates = []
    if not args.baselines_only:
        groups = discover_checkpoints(args.checkpoint_dir, model_filters, exclude_filters)
        for prefix in sorted(groups):
            chosen = {}
            for _ts, path in groups[prefix]:          # newest-first
                ta, cfg, gstep = peek_checkpoint_meta(path)
                if ta is not None:
                    if ta.get('experiment') and not args.include_experimental:
                        continue
                    if dataset_of_train_args(ta) != args.dataset:
                        continue
                elif cfg.get("seq_len") != seq_len:
                    continue
                label = config_label(prefix, cfg, gstep)
                chosen.setdefault(label, path)
            for label in sorted(chosen):
                print(f"[keep] {label}: {os.path.basename(chosen[label])}")
                candidates.append((label, chosen[label]))
        if not candidates:
            # Discovering groups and filtering every one away is a misconfiguration,
            # not a no-op (NOTES s31). --multitree-baselines is the one legitimate
            # no-checkpoint case: it supplies its own models.
            msg = (f"No checkpoints match --dataset {args.dataset} "
                   f"({len(groups)} group(s) discovered, all filtered out).")
            if args.multitree_baselines:
                print(msg + " Continuing: --multitree-baselines supplies its own models.")
            else:
                raise SystemExit(msg)

    # ── The cross-tree factored baselines, built the same way eval.py builds them
    builtin_models = {}
    if args.multitree_baselines:
        print("\n=== Building cross-tree factored baselines ===", flush=True)
        _fit_cache = {}

        def _fit():
            if "fit" not in _fit_cache:
                print(f"  fitting substitution matrix on {len(training_examples)} "
                      f"training examples (seed={args.substitution_seed})...", flush=True)
                _fit_cache["fit"] = fit_substitution_baseline(
                    training_examples, device, allowed_aas, seed=args.substitution_seed)
            return _fit_cache["fit"]

        builtin_models = {
            "codon": lambda: create_model(ModelConfig(
                model_type=ModelType.CODON, seq_len=seq_len,
                alphabet=AlphabetConfig(amino_acids=allowed_aas),
                codon=CodonConfig()), device),
            "substitution": _fit,
            "where_times_which": lambda: WhereTimesWhichPredictor(
                _fit().config,
                positional_marginal_counts(training_examples, seq_len, E),
                _fit()).to(device),
        }
        for name in builtin_models:
            candidates.append((name, None))
        print(f"  added {len(builtin_models)}: {', '.join(builtin_models)}")

    by_model = read_done(args.output_dir)
    skipped = []
    if by_model:
        print(f"Resuming: {len(by_model)} model(s) already in {SUMMARY}: {sorted(by_model)}")

    def record(name, acc):
        by_model[name] = summarize(acc)
        write_summary(args.output_dir, by_model)
        append_percell(args.output_dir, name, acc, args.percell_qtile,
                       allowed_aas, eval_range)
        for suffix, qtile, macro, micro, n_cells, n_ev in by_model[name]:
            if abs(qtile - args.percell_qtile) < 1e-9:
                print(f"  {name:45s} {suffix:32s} q={qtile:.2f}  "
                      f"macro={macro:.4f}  micro={micro:.4f}  "
                      f"cells={n_cells} events={n_ev:.0f}", flush=True)

    # ── Baselines (fixed maps, no forward pass) ─────────────────────────────
    for name, probs_2d in (() if args.skip_baselines else baselines):
        if name in by_model:
            print(f"[resume-skip] {name}")
            continue
        print(f"\n=== Baseline: {name} ===", flush=True)
        bp = probs_2d.to(device)
        record(name, cell_pass(lambda aa, bp=bp: bp.unsqueeze(0).expand(aa.shape[0], -1, -1),
                               loader, variants, device, eval_range, qtiles))
        del bp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Models ──────────────────────────────────────────────────────────────
    for prefix, path in candidates:
        if prefix in by_model:
            print(f"[resume-skip] {prefix}")
            continue
        print(f"\n=== Model: {prefix} ===", flush=True)
        if prefix in builtin_models:
            model = builtin_models[prefix]()
            model.eval()
            model_mask = build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx).to(device)
            record(prefix, cell_pass(
                (lambda aa, m=model, mm=model_mask: _masked_scores(m, aa, mm)),
                loader, variants, device, eval_range, qtiles))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        # A model that cannot be built is skipped so one bad checkpoint does not cost
        # the rows already scored -- but the skips are collected and the run exits
        # NON-ZERO at the end (NOTES s30: a silent [skip] plus exit 0 is how eight
        # model groups went missing from a complete-looking set of CSVs).
        try:
            ckpt, model_config = load_checkpoint(path, map_location=device)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {prefix}: load_checkpoint failed ({exc})")
            skipped.append((prefix, f"load_checkpoint: {exc}"))
            continue
        if model_config.seq_len != seq_len:
            print(f"[skip] {prefix}: checkpoint seq_len={model_config.seq_len} != {seq_len}")
            skipped.append((prefix, f"seq_len {model_config.seq_len} != {seq_len}"))
            continue
        try:
            model = create_model(model_config, device)
            missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {prefix}: model build/load failed ({exc})")
            skipped.append((prefix, f"build/load: {exc}"))
            del ckpt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if missing:
            print(f"  [warn] {len(missing)} missing state_dict keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected state_dict keys (e.g. {unexpected[:3]})")
        model.eval()
        model_mask = build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx).to(device)

        record(prefix, cell_pass(
            (lambda aa, m=model, mm=model_mask: _masked_scores(m, aa, mm)),
            loader, variants, device, eval_range, qtiles))
        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n=== Wrote {os.path.join(args.output_dir, SUMMARY)} ===")
    print(f"Scored {len(by_model)} model(s)")
    if skipped:
        raise SystemExit(
            f"{len(skipped)} model group(s) were skipped and are MISSING from "
            f"{SUMMARY}: " + "; ".join(f"{n} ({why})" for n, why in skipped))


if __name__ == "__main__":
    main()
