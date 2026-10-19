#!/usr/bin/env python
"""CoG-UK yield of the BA.2 top decile, from one forward pass on the variant root.

**The question.** Put one query to the model -- the BA.2 root, the sequence a forecaster
would actually have had in hand -- and take its top decile over the RBD, ~442 of 4,422
cells. That set is fixed. Then sweep time and ask what share of it turns out to be
CoG-UK mutations that emerged downstream of BA.2 by t:

    value(t) = |top-q(RBD pool) & list & emerged_t| / (q * N_RBD)

The denominator is the size of the model's own top decile and does not move with t, so
the curve is monotone increasing and its max is simply the end-of-series value.

**How this relates to the two neighbouring metrics**, because both are easy to confuse
it with:

* ``eval.py``'s ``variant_emergence_list_<L>`` (Figure 2's PSSM precision, restricted to
  a list) NaN-masks everything off-list BEFORE ranking, so its top decile is ~25 of the
  247 listed cells. Different pool, therefore a different top-q set and a different
  numerator: **not this metric, and not proportional to it.**
* Dividing the same numerator by ``|list & emerged_t|`` instead gives the recall
  reading. That one is **proportional to this at each fixed t**, up to the per-model
  constant N_RBD, so it ranks models almost identically -- but its denominator grows
  with t, which makes the curve fall rather than rise and puts its max at the first time
  step with a single emerged cell, where several models tie at 1.000. That degeneracy is
  why the fixed denominator is used here.

**N_RBD is per model, not global.** It is the count of cells that model actually ranks:
the RBD's 4,422 less the query's own residue, less the gap/stop columns a model does not
legitimately emit. That is ~4,221 for a model that emits gaps and ~3,819 for one that
does not, a ~10% spread, so an abstaining model is scored against a slightly smaller
top decile. This is the convention ``variant_hit_rates`` already uses (its denominator
is likewise ``q * n_valid``), kept for consistency with Figure 2.

**The attainable ceiling is well below 1** and should be read with the numbers: only 234
of the 247 listed RBD cells are mutations BA.2 could make at all (it already carries 13),
of which 200 ever emerge at 1+ cases and 79 at 100+. Against a top decile of ~442 that
caps the metric near 0.45 and 0.18 respectively.

**Emerged** means what it means everywhere else in this pipeline: the cell has at least
``threshold`` observations in the variant PSSM at that time step, and the variant PSSMs
are built from BA.2-descended sequences, so "emerged by t" is already "arose downstream
of the variant root by t". Thresholds follow eval.py: 100 and 1 for BA.2.

Conventions held identical to the recall panels so the numbers can be read against them:
the query's own residue is NaN-masked, so is any cell a model does not legitimately
emit (``build_model_gap_stop_mask``), the de novo cut is the same ``pssm_root > 0``
training-window mask, and the top-q boundary is split fractionally, as
``compute_hit_rate`` and ``variant_hit_rates`` both do, so a coarse predictor cannot
sweep a tied block into its top decile.

A cell count, not an event count: each listed mutation counts once, which is the whole
point of the figure this feeds.

Usage:
    python antiGen/list_variant_recall.py \
        --data-dir /home/ivan.specht/antiGen-data/output1 \
        --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
        --output-dir .../evals_sc2_lists_varrecall \
        --mutation-list coguk=coguk.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import os

import torch

from antiGen import datasets
from antiGen.config import AlphabetConfig, CodonConfig, ModelConfig, ModelType
from antiGen.eval import (config_label, dataset_of_train_args, discover_checkpoints,
                          peek_checkpoint_meta)
from antiGen.eval_utils import (RBD_RANGE, build_model_gap_stop_mask,
                                fit_substitution_baseline, load_baselines,
                                load_checkpoint, positional_marginal_counts,
                                WhereTimesWhichPredictor)
from antiGen.load import load_examples, load_pssms
from antiGen.models.factory import create_model
from antiGen.mutation_lists import describe as describe_list, load_mutation_list

OUT = "list_variant_recall.csv"


def top_q_weights(scores_2d, qtile):
    """Fractional membership of each cell in the top-q of ``scores_2d``.

    1.0 strictly inside, ``frac`` on the boundary block, 0 elsewhere -- the expected
    membership under random tie-breaking, which is what makes the metric seed-free and
    fair to coarse-valued predictors. NaN cells (wild type, gap/stop, -inf sentinels,
    anything masked out) are never selected and never counted in the pool.
    """
    p = scores_2d.float()
    valid = ~torch.isnan(p)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return torch.zeros_like(p), 0
    vb = torch.nanquantile(p, 1.0 - qtile, interpolation='lower')
    above = p > vb
    at = p == vb
    n_above = float(above.sum())
    n_at = max(float(at.sum()), 1.0)
    frac = min(max((qtile * n_valid - n_above) / n_at, 0.0), 1.0)
    return above.float() + frac * at.float(), n_valid


def curve(scores_2d, target_masks, qtile):
    """[(n_targets, n_selected, value)] per time step, for one score map.

    ``n_selected`` is the denominator -- the size of this model's top decile -- and is
    the same at every time step. A step where nothing has emerged yet gives 0, not NaN:
    the top decile exists and contains none of the list, which is a real zero.
    """
    w, n_valid = top_q_weights(scores_2d, qtile)
    n_selected = max(qtile * n_valid, 1e-9)
    return [(int(tm.sum()), n_selected, float((w * tm.float()).sum()) / n_selected)
            for tm in target_masks]


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
    ap.add_argument('--qtile', type=float, default=0.1)
    ap.add_argument('--models', default=None)
    ap.add_argument('--exclude', default='transformer_,esmc,vespag')
    ap.add_argument('--include-experimental', action='store_true')
    ap.add_argument('--multitree-baselines', action='store_true')
    ap.add_argument('--substitution-seed', type=int, default=14)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)
    gap_idx, stop_idx = allowed_aas.index('-'), allowed_aas.index('*')
    print(f"Device: {device}")

    variant = args.variant or datasets.default_variant(args.dataset)
    files = sorted(glob.glob(os.path.join(args.data_dir, "examples_cutoff_*.txt")))
    if not files:
        raise SystemExit(f"No examples_cutoff_*.txt in {args.data_dir}")
    all_examples = load_examples(files[0], allowed_aas, variant)
    seq_len = len(all_examples[0].aa_idx)
    eval_range = (list(range(seq_len)) if args.eval_range == 'full'
                  else (list(range(seq_len)) if args.dataset in ('flu', 'rsv_a', 'denv2')
                        else RBD_RANGE))

    roots = [e for e in all_examples if e.is_testing_root]
    if not roots:
        raise SystemExit(f"No testing root for variant '{variant}' -- nothing to query.")
    root = roots[0]
    training_examples = [e for e in all_examples if e.is_training]
    global_wt_aa_idx = training_examples[0].aa_idx.clone()
    print(f"Variant root: '{variant}', {len(roots)} candidate(s), querying the first")
    print(f"Eval range: {len(eval_range)} positions "
          f"({eval_range[0] + 1}-{eval_range[-1] + 1}, 1-based)")

    pssm_root, pssm_variant = load_pssms(os.path.join(args.data_dir, "pssms"), variant, E)
    if not pssm_variant:
        raise SystemExit("No variant PSSMs: there is no time grid to sweep.")
    pssm_variant_sub = [pv[eval_range, :] for pv in pssm_variant]
    seen_sub = pssm_root[eval_range, :] > 0
    n_steps = len(pssm_variant_sub)
    print(f"Variant PSSM time steps: {n_steps}")

    # The query's own residue is not a mutation.
    wt_idx = root.aa_idx[eval_range]
    wt_keep = torch.ones(len(eval_range), E, dtype=torch.bool)
    wt_keep.scatter_(1, wt_idx.unsqueeze(-1), False)

    lists = []
    for item in (args.mutation_list or []):
        label, path = item.split('=', 1)
        cells, report = load_mutation_list(
            path, allowed_aas, seq_len, label=label,
            root_seq="".join(allowed_aas[i] for i in global_wt_aa_idx.tolist()))
        in_full = torch.zeros(seq_len, E, dtype=torch.bool)
        for pos, aa in cells:
            in_full[pos, aa] = True
        in_sub = in_full[eval_range, :]
        print(describe_list(report, in_range=int(in_sub.sum())))
        lists.append((label, in_sub))

    thresholds = [100, 1] if variant == 'BA.2' else [10, 1]

    # Target sets: listed cells that have emerged downstream of the variant root by t,
    # at each threshold, for the whole list and for its de novo subset. The wild-type
    # mask applies here too -- a cell the query already carries is not a mutation it
    # could make.
    target_sets = {}      # (list, subset, threshold) -> [mask per time step]
    for label, in_sub in lists:
        for subset, base in (("all", in_sub & wt_keep),
                             ("unseen", in_sub & wt_keep & ~seen_sub)):
            for thr in thresholds:
                target_sets[(label, subset, thr)] = [
                    base & (pv >= thr) for pv in pssm_variant_sub]
        for subset in ("all", "unseen"):
            n_end = int(target_sets[(label, subset, thresholds[-1])][-1].sum())
            n_pool = int((in_sub & wt_keep).sum()) if subset == "all" else \
                int((in_sub & wt_keep & ~seen_sub).sum())
            print(f"  {label}/{subset}: {n_pool} listed cells in range, "
                  f"{n_end} emerged downstream by the last time step "
                  f"(threshold {thresholds[-1]})")

    rows = []

    def record(name, scores_2d):
        """scores_2d: (len(eval_range), E), already gap/stop masked."""
        s = scores_2d.clone().float()
        s[~wt_keep] = float('nan')
        for (label, subset, thr), masks in target_sets.items():
            for i, (n_targets, n_selected, value) in enumerate(curve(s, masks, args.qtile)):
                rows.append(dict(model=name, list=label, subset=subset, threshold=thr,
                                 step=i, n_targets=n_targets,
                                 n_selected=round(n_selected, 2), value=value))
        done = [r for r in rows if r['model'] == name and r['subset'] == 'all'
                and r['threshold'] == thresholds[0]]
        if done:
            print(f"  {name:45s} max (all, {thresholds[0]}+) = "
                  f"{max(r['value'] for r in done):.4f}  "
                  f"(top decile = {done[0]['n_selected']:.0f} cells)", flush=True)
        write()

    def write():
        path = os.path.join(args.output_dir, OUT)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "list", "subset", "threshold",
                                              "step", "n_targets", "n_selected",
                                              "value"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        os.replace(tmp, path)

    # ── Fixed-map baselines ─────────────────────────────────────────────────
    baseline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                                "baselines")
    for name, probs_2d in load_baselines(baseline_dir, allowed_aas, args.seed,
                                         pssm_root, global_wt_aa_idx):
        record(name, probs_2d[eval_range, :])

    # ── Checkpoints ─────────────────────────────────────────────────────────
    model_filters = [s.strip() for s in args.models.split(',') if s.strip()] if args.models else None
    exclude_filters = [s.strip() for s in args.exclude.split(',') if s.strip()] if args.exclude else None
    groups = discover_checkpoints(args.checkpoint_dir, model_filters, exclude_filters)
    candidates = []
    for prefix in sorted(groups):
        chosen = {}
        for _ts, path in groups[prefix]:
            ta, cfg, gstep = peek_checkpoint_meta(path)
            if ta is not None:
                if ta.get('experiment') and not args.include_experimental:
                    continue
                if dataset_of_train_args(ta) != args.dataset:
                    continue
            elif cfg.get("seq_len") != seq_len:
                continue
            chosen.setdefault(config_label(prefix, cfg, gstep), path)
        for label in sorted(chosen):
            print(f"[keep] {label}: {os.path.basename(chosen[label])}")
            candidates.append((label, chosen[label]))

    builtin = {}
    if args.multitree_baselines:
        _cache = {}

        def _fit():
            if "f" not in _cache:
                _cache["f"] = fit_substitution_baseline(
                    training_examples, device, allowed_aas, seed=args.substitution_seed)
            return _cache["f"]

        builtin = {
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
        candidates += [(n, None) for n in builtin]

    if not candidates:
        raise SystemExit("No checkpoints matched and no builtin models requested.")

    aa_input = root.aa_idx.unsqueeze(0).to(device)
    skipped = []
    for prefix, path in candidates:
        print(f"\n=== {prefix} ===", flush=True)
        try:
            if prefix in builtin:
                model = builtin[prefix]()
            else:
                ckpt, cfg = load_checkpoint(path, map_location=device)
                if cfg.seq_len != seq_len:
                    print(f"[skip] {prefix}: seq_len {cfg.seq_len} != {seq_len}")
                    skipped.append((prefix, "seq_len"))
                    continue
                model = create_model(cfg, device)
                missing, unexpected = model.load_state_dict(ckpt['model_state_dict'],
                                                            strict=False)
                if missing:
                    print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})")
                del ckpt
            model.eval()
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {prefix}: build/load failed ({exc})")
            skipped.append((prefix, str(exc)))
            continue

        mask = build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx)
        with torch.no_grad():
            scores = model(aa_input).float().squeeze(0).cpu()
        scores[mask] = float('nan')
        record(prefix, scores[eval_range, :])

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n=== Wrote {os.path.join(args.output_dir, OUT)} ===")
    if skipped:
        raise SystemExit(f"{len(skipped)} model(s) skipped and MISSING from {OUT}: "
                         + "; ".join(f"{n} ({w})" for n, w in skipped))


if __name__ == "__main__":
    main()
