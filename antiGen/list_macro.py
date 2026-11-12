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

import numpy as np
import torch
from torch.utils.data import DataLoader

from antiGen import datasets
from antiGen import score_cache as sc
from antiGen.config import (AlphabetConfig, CodonConfig, LEGACY_AAS, ModelConfig,
                            ModelType)
from antiGen.eval import (MutationDataset, config_label, dataset_of_train_args,
                          discover_checkpoints, peek_checkpoint_meta)
from antiGen import eval_utils as eu
from antiGen.eval_utils import (CountedStatPredictor, MeanLogProbEnsemble, RBD_RANGE,
                                build_model_gap_stop_mask,
                                compute_hit_rate, fit_substitution_baseline,
                                load_baselines, load_checkpoint,
                                positional_marginal_counts,
                                where_times_which_dirichlet,
                                WhereTimesWhichPredictor)
from antiGen.load import load_counts_matrix, load_examples, load_pssms
from antiGen.models.factory import create_model
from antiGen.mutation_lists import describe as describe_list, load_mutation_list

SUMMARY = "list_macro_summary.csv"
PERCELL = "list_macro_percell.csv"


def build_variants(lists, seen_mask, reach1=None):
    """(suffix, score_mask, target_mask, reach1) per list variant. Masks are True =
    exclude. ``reach1`` (E, E) bool, when given, adds the codon-distance-1 variants
    (NOTES s159): pool AND targets further restricted, per query, to the cells one
    nucleotide change away from that query's own wild-type residue."""
    variants = []
    for label, in_full in lists:
        variants.append((f"list_{label}_fullpool", None, ~in_full, None))
        variants.append((f"list_{label}", ~in_full, ~in_full, None))
        variants.append((f"list_{label}_unseen_fullpool", seen_mask, (~in_full) | seen_mask, None))
        variants.append((f"list_{label}_unseen", (~in_full) | seen_mask, (~in_full) | seen_mask, None))
        if reach1 is not None:
            variants.append((f"list_{label}_unseen_cd1_fullpool", seen_mask,
                             (~in_full) | seen_mask, reach1))
            variants.append((f"list_{label}_cd1_fullpool", None, ~in_full, reach1))
    return variants


## Set by LIST_MACRO_TIE_JITTER=<seed>: a debugging aid, never used for a published row.
_TIE_JITTER = (int(os.environ["LIST_MACRO_TIE_JITTER"])
               if os.environ.get("LIST_MACRO_TIE_JITTER") else None)


def cell_pass(score_fn, loader, variants, device, eval_range, qtiles, sink=None,
              common_pool=None):
    """Accumulate per-cell hit mass and event counts over the loader.

    ``score_fn(aa_batch) -> (B, seq_len, E)`` is the only thing that differs between a
    model (forward pass) and a baseline (a fixed map broadcast to the batch), so both
    go through this one function and are masked and ranked identically.

    ``sink(start, scores_eval)``, if given, receives each batch's eval-range slice
    *before* any variant masking, which is what makes one cached map serve all four
    variants: the gap/stop mask already applied is a property of the MODEL, while the
    wild-type mask and the de novo mask are properties of the query and the variant and
    are applied downstream. ``start`` is the running query offset, counted rather than
    derived from the batch index, so a short final batch cannot silently shift it.

    Returns ``acc[(suffix, qtile)] = (cell_hits, cell_events)``, each (R, E).
    """
    acc = {}
    n_batches = len(loader)
    start = 0
    for batch_idx, (aa_batch, target_batch) in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"    batch {batch_idx}/{n_batches}", flush=True)
        aa_batch = aa_batch.to(device)
        target_batch = target_batch.to(device)
        scores = score_fn(aa_batch)
        if _TIE_JITTER is not None:
            # DEBUG ONLY (NOTES s133): break exact ties at random, to check that the
            # fractional top-q rule equals the mean over random tie-breaks. The noise is
            # far below the smallest gap between distinct scores of a coarse predictor.
            g = torch.Generator(device="cpu").manual_seed(_TIE_JITTER * 100003 + start)
            fin = torch.isfinite(scores)
            span = (scores[fin].max() - scores[fin].min()).item() if fin.any() else 1.0
            noise = torch.rand(scores.shape, generator=g).to(scores.device)
            scores = torch.where(fin, scores + noise * 1e-4 * max(span, 1e-12), scores)
        if sink is not None:
            sub = scores[:, eval_range, :] if scores.shape[1] != len(eval_range) else scores
            sink(start, sub.detach().float().cpu().numpy())
        start += int(aa_batch.shape[0])

        for suffix, score_mask, target_mask, reach1 in variants:
            # With a common pool the candidate set is handed to compute_hit_rate as a
            # mask instead of being burned into the scores, so that a cell inside the
            # pool which this model cannot score RANKS LAST rather than leaving the
            # pool -- which is what makes "the top decile" the same size for every
            # model (NOTES s51). Without one, the published per-model behaviour.
            v_pool = None
            if common_pool is None:
                if score_mask is None:
                    v_scores = scores
                else:
                    v_scores = scores.clone()
                    v_scores[score_mask.to(device).unsqueeze(0).expand_as(v_scores)] = float('nan')
            else:
                v_scores = scores
                v_pool = common_pool.to(device)
                if score_mask is not None:
                    v_pool = v_pool & ~score_mask.to(device)[eval_range, :]
            if target_mask is None:
                v_target = target_batch
            else:
                v_target = target_batch.clone()
                v_target[target_mask.to(device).unsqueeze(0).expand_as(v_target)] = 0
            if reach1 is not None:
                if v_pool is None:
                    raise SystemExit("--cd1-variants needs the common pool (not --legacy-pool)")
                keep = reach1.to(device)[aa_batch]                    # (B, L, E)
                v_pool = v_pool.unsqueeze(0) & keep[:, eval_range, :]
                v_target = v_target * keep.to(v_target.dtype)

            for qtile in qtiles:
                _h, _n, cell_hits, cell_events = compute_hit_rate(
                    v_scores, v_target, aa_batch, qtile, eval_range=eval_range,
                    per_cell=True, pool_mask=v_pool)
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
    ap.add_argument('--data-dir', default=None,
                    help='Directory holding examples_cutoff_*.txt and pssms/. Not '
                         'needed when --train-examples / --test-examples and '
                         '--root-counts name every input explicitly.')
    ap.add_argument('--alphabet', default=LEGACY_AAS,
                    help='Residue alphabet, which fixes E and the column order of '
                         'every map. The default is the 22-token frame every '
                         'published checkpoint and cached score map was written in. '
                         'Pass STANDARD_AAS (ACDEFGHIKLMNPQRSTVWY) to score the '
                         '20-letter arm (NOTES s53): a checkpoint of the other width '
                         'then fails on shape rather than scoring wrongly.')
    ap.add_argument('--train-examples', action='append', default=None, metavar='PATH',
                    help='Examples file(s) to take the TRAINING pool from -- only the '
                         'pre-cutoff records in them are kept. Overrides the '
                         '--data-dir glob. antiGen-data writes one file carrying both '
                         'halves; antiGen-data-revised writes a train file and a test '
                         'file, and the test file still carries the pre-cutoff spine '
                         'of the tree, which is not the training pool.')
    ap.add_argument('--test-examples', action='append', default=None, metavar='PATH',
                    help='Examples file(s) to take the TESTING pool from -- only the '
                         'post-cutoff records in them are kept.')
    ap.add_argument('--split-date', default=None, metavar='YYYY-MM-DD',
                    help="Which train/test split to read, for a corpus whose "
                         "POST_CUTOFF field is a SET of cutoff dates rather than one "
                         "boolean (antiGen-data-revised's single test tree, which "
                         "serves 24 of them). A record is a TEST example exactly when "
                         "its set carries this date. Required the moment any record "
                         "carries a non-empty set -- a default would silently pick a "
                         "split -- and ignored on the train arms, whose sets are all "
                         "empty.")
    ap.add_argument('--root-counts', default=None, metavar='TSV',
                    help="antiGen-data-revised's positional count matrix over the "
                         "training window, used in place of pssms/root.csv. It is the "
                         "same quantity -- how often each residue was observed at each "
                         "site before the cutoff -- and it defines the de novo cut, so "
                         "it has to come from the same corpus the models were trained "
                         "on.")
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
    ap.add_argument('--legacy-pool', action='store_true',
                    help="Score in each MODEL's own emission set, which is what every "
                         "published CSV before NOTES s51 used. The default is the common "
                         "candidate pool: the 20 real amino-acid columns, identical for "
                         "every model, so 'the top decile' is the same size for all of "
                         "them. Under the published convention it was not -- n_valid ran "
                         "3,801 to 4,221 across the roster and up to 45.7%% of a model's "
                         "top decile went to gap/stop cells that can never be a listed "
                         "target. Only pass this to reproduce a pre-s51 number.")
    ap.add_argument('--percell-qtile', type=float, default=0.1)
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--multitree-baselines', action='store_true',
                    help='Also score the cross-tree factored baselines (codon, '
                         'substitution, where x which). They read the residue at each '
                         'position of the query, so they cannot be expressed as the '
                         'fixed (L, E) maps data/baselines/ holds and go through the '
                         'model path instead -- which also means they inherit the '
                         'gap/stop masking. Fitted on the TRAINING examples only.')
    ap.add_argument('--pool-examples', action='store_true',
                    help='pool --train-examples and --test-examples and split on '
                         'is_training, as eval.py does, instead of taking the role '
                         'from which flag named the file. Matters only with '
                         '--split-date, where the test corpus carries pre-cutoff '
                         'branches that belong to the training window.')
    ap.add_argument('--counted-baselines', action='store_true',
                    help='add the counted-statistic rungs: site_substitution '
                         '(s_i * M[wt_i,a]) and mcm_interpolation '
                         '((N + kappa*s*M)/(N_total+kappa), kappa=N_total). Separate '
                         'from --multitree-baselines so an existing roster does not '
                         'change: the figures key rows by name and an unknown model '
                         'collapses into an NA group (NOTES s-note).')
    ap.add_argument('--ensemble-dir', default=None, metavar='DIR',
                    help='add one rung: every fitness_*.pt in DIR combined by the '
                         'published "Mean of log-probs" rule (ensemble_rules '
                         'logsoftmax/arith), each member normalised over the EVAL '
                         'RANGE as the cached-map rule does (NOTES s123). Port of '
                         'eval.py --ensemble-dir, which instead needs the full range.')
    ap.add_argument('--ensemble-name', default='antigen_valsel',
                    help='row name for --ensemble-dir')
    ap.add_argument('--history-model', action='append', default=[], metavar='NAME=PT',
                    help='a HistoryTransformer checkpoint (antiGen.history, NOTES s156): '
                         'each scored query is its mutation history from the root of the '
                         'tree it came from, rebuilt from its own examples file.')
    ap.add_argument('--query-maps', action='append', default=[], metavar='NAME=NPZ',
                    help='a precomputed PER-QUERY map rung (NOTES s151): an NPZ with '
                         '`aa` (Nq, L) int8 genotypes, `eval_range` and `scores` '
                         '(Nq, R, E). Each scored query is looked up by genotype; a '
                         'query the NPZ lacks is an error, not a skip.')
    ap.add_argument('--mix-dir', default=None, metavar='DIR',
                    help='a SECOND ensemble directory (fitness_*.pt), mixed with --ensemble-dir '
                         'by --mix (NOTES s157)')
    ap.add_argument('--mix-dir2', default=None, metavar='DIR',
                    help='an optional THIRD ensemble directory for --mix NAME=W:W2')
    ap.add_argument('--mix', action='append', default=[], metavar='NAME=W[:W2[:W3...]]',
                    help='a rung scoring (1 - sum W) * mean-log-prob(--ensemble-dir) + '
                         'W * mean-log-prob(--mix-dir) + W2 * (--mix-dir2) + W3.. * each '
                         '--mix-baseline in order, every component normalised per query '
                         'over the eval range as ens_vs10_2023 is. Repeatable.')
    ap.add_argument('--seen-min-count', type=int, default=1,
                    help='training-window sequence count at which a cell counts as seen '
                         '(default 1: unseen = de novo). NOTES s162 uses 100.')
    ap.add_argument('--cd1-variants', action='store_true',
                    help='add list_<l>_cd1_fullpool and list_<l>_unseen_cd1_fullpool: pool '
                         'and targets restricted, per query, to substitutions one nucleotide '
                         'change from the query\'s own wild-type residue (NOTES s159)')
    ap.add_argument('--mix-baseline', action='append', default=[], metavar='NAME',
                    choices=['codon', 'mcm_interpolation', 'site_substitution'],
                    help='an untrained / counted model as a further --mix component, fitted '
                         'on this run\'s training examples (NOTES s158)')
    ap.add_argument('--mcm-baseline', action='store_true',
                    help='add the mutation count matrix alone, "mcm": log N[i,a] '
                         '(statistical_baselines "frequency"). NOTES s121.')
    ap.add_argument('--substitution-seed', type=int, default=14)
    ap.add_argument('--skip-baselines', action='store_true',
                    help='Score only the checkpoints. The CSV baselines are fixed maps '
                         'and cost ~2 min each; skipping them lets a second job get at '
                         'the checkpoints without re-deriving rows another run already '
                         'has. The rows it does write are identical either way.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dump-scores', default=None, metavar='DIR',
                    help='Also cache each model\'s (queries, eval positions, residues) '
                         'score map under DIR, so every later question about combining '
                         'models is answerable without a forward pass. Mirrors '
                         'background_discrimination.py\'s convention on the backbone '
                         'substrate. MUST be a new directory: the maps are keyed to this '
                         'run\'s query set by a hash, and two query sets produce maps of '
                         'identical shape that cannot be told apart afterwards.')
    ap.add_argument('--dump-max-bytes', type=float, default=64e9,
                    help='Refuse a dump projected above this without --yes-big. '
                         '--eval-range full multiplies the payload 6.3x (1273 positions '
                         'against 201), which is 60 GB over this roster.')
    ap.add_argument('--yes-big', action='store_true',
                    help='Acknowledge a dump above --dump-max-bytes.')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allowed_aas = args.alphabet
    E = len(allowed_aas)
    gap_idx = allowed_aas.index('-') if '-' in allowed_aas else None
    stop_idx = allowed_aas.index('*') if '*' in allowed_aas else None
    qtiles = [float(q) for q in args.qtiles.split(',') if q.strip()]
    print(f"Device: {device}")
    print(f"Alphabet: '{allowed_aas}' (E={E})")

    # ── Data ────────────────────────────────────────────────────────────────
    variant = args.variant or datasets.default_variant(args.dataset)
    if not args.data_dir and not (args.train_examples and args.test_examples):
        raise SystemExit("--data-dir, or --train-examples with --test-examples.")
    if args.train_examples or args.test_examples:
        if not (args.train_examples and args.test_examples):
            raise SystemExit("--train-examples and --test-examples go together: "
                             "one of the two pools would otherwise be empty.")
        training_examples, testing_examples = [], []
        if args.pool_examples:
            # eval.py's semantics: pool every file, THEN split on is_training. With
            # --split-date the test corpus carries pre-cutoff branches that belong to
            # the training window, and eval.py counts them as training. This script
            # assigns the role per FILE, so it drops them -- 2,934 against eval.py's
            # 6,836 at the 2021-01-01 cutoff, i.e. the fit sees 43% of the data.
            #
            # Everything built on fit_substitution_baseline then differs from the
            # published panel: substitution, where_times_which and wxw_dirichlet all
            # missed it, while codon (fixed table) and pssm_root (read from
            # --root-counts) matched to 1e-8. Nothing that ignores training_examples
            # moved, which is what identified this.
            #
            # OFF BY DEFAULT. Without --split-date the two agree, which is every
            # published list_macro run, and those numbers must not move.
            pooled = []
            for path in list(args.train_examples) + list(args.test_examples):
                recs = load_examples(path, allowed_aas, variant,
                                     split_date=args.split_date)
                if args.history_model:
                    from antiGen.history import attach_histories
                    attach_histories(recs)       # per file: parents precede children
                print(f"  pooled {len(recs)} from {path}")
                pooled += recs
            training_examples = [e for e in pooled if e.is_training]
            testing_examples = [e for e in pooled if not e.is_training]
            print(f"  pooled split: {len(training_examples)} training, "
                  f"{len(testing_examples)} testing (eval.py semantics)")
            all_examples = training_examples + testing_examples
            return_early = True
        else:
            return_early = False
        for path in ([] if return_early else args.train_examples):
            recs = load_examples(path, allowed_aas, variant,
                                 split_date=args.split_date)
            kept = [e for e in recs if e.is_training]
            print(f"  training pool: kept {len(kept)} of {len(recs)} from {path}")
            training_examples += kept
        for path in ([] if return_early else args.test_examples):
            recs = load_examples(path, allowed_aas, variant,
                                 split_date=args.split_date)
            kept = [e for e in recs if not e.is_training]
            print(f"  testing pool: kept {len(kept)} of {len(recs)} from {path}")
            testing_examples += kept
        if not return_early:
            all_examples = training_examples + testing_examples
    else:
        examples_files = sorted(glob.glob(os.path.join(args.data_dir, "examples_cutoff_*.txt")))
        if not examples_files:
            raise SystemExit(f"No examples_cutoff_*.txt found in {args.data_dir}")
        all_examples = load_examples(examples_files[0], allowed_aas, variant,
                                     split_date=args.split_date)
        testing_examples = [e for e in all_examples if not e.is_training]
        training_examples = [e for e in all_examples if e.is_training]
    if not training_examples or not testing_examples:
        raise SystemExit(f"{len(training_examples)} training and "
                         f"{len(testing_examples)} testing examples; both are needed.")
    # load_examples checks this within a file; across files it is this script's to make,
    # and two corpora of different length would otherwise index the same (L, E) grid.
    seq_lens = {len(e.aa_idx) for e in all_examples}
    if len(seq_lens) != 1:
        raise SystemExit(f"examples span several sequence lengths {sorted(seq_lens)}")
    seq_len = len(all_examples[0].aa_idx)
    eval_range = list(range(seq_len)) if args.eval_range == 'full' else (
        list(range(seq_len)) if args.dataset in ('flu', 'rsv_a', 'denv2') else RBD_RANGE)
    print(f"Sequence length {seq_len}; eval range {len(eval_range)} positions "
          f"({eval_range[0] + 1}-{eval_range[-1] + 1}, 1-based)")

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

    # ── The score-map cache, if asked for ───────────────────────────────────
    # Everything below is set up BEFORE any model is built, so a misconfigured dump
    # fails in seconds rather than after the first forward pass.
    dump_dir = args.dump_scores
    if dump_dir:
        R = len(eval_range)
        per_model = sc.expected_bytes(len(scored), R, E)
        projected = per_model * max(1, len(glob.glob(os.path.join(args.checkpoint_dir,
                                                                  'fitness_*.pt'))))
        print(f"\n=== Score-map cache: {dump_dir} ===")
        print(f"  {len(scored)} queries x {R} positions x {E} residues x float32 "
              f"= {per_model / 1e6:.1f} MB per model")
        if projected > args.dump_max_bytes and not args.yes_big:
            raise SystemExit(
                f"projected dump is ~{projected / 1e9:.1f} GB, above --dump-max-bytes "
                f"({args.dump_max_bytes / 1e9:.1f} GB). Pass --yes-big to proceed. "
                f"(--eval-range {args.eval_range} gives {R} positions per query.)")
        aa_all = torch.stack([e.aa_idx for e in scored])              # (N, seq_len)
        dump_sha1 = sc.query_sha1(aa_all[:, eval_range])
        sc.write_query_manifest(
            dump_dir, dataset=args.dataset, variant=variant, seq_len=seq_len,
            eval_range=eval_range,
            list_specs=[[lab, int(m.sum())] for lab, m in lists],
            sha1=dump_sha1, n_queries=len(scored), n_events=n_events, args=args)
        if not os.path.exists(os.path.join(dump_dir, sc.QUERIES)):
            # Every eval-range target, not only the listed ones: the cells NOT on the
            # list are what a later selection pass is scored on, and they come off this
            # same forward pass for free (NOTES: the val set for the ensembling arm).
            qi, po, al, ct = [], [], [], []
            for i, e in enumerate(scored):
                nz = (e.mat[eval_range, :] > 0).nonzero()
                for r, a in nz.tolist():
                    qi.append(i); po.append(eval_range[r]); al.append(a)
                    ct.append(float(e.mat[eval_range[r], a]))
            sc.write_queries(dump_dir, aa_all.to(torch.int8).numpy(),
                             (np.array(qi), np.array(po), np.array(al),
                              np.array(ct, dtype=np.float32)))
            print(f"  wrote {sc.QUERIES}: {len(scored)} genotypes, {len(qi)} target cells "
                  f"({sum(ct):.0f} events) over the whole eval range")
        print(f"  query_sha1 {dump_sha1}", flush=True)

    if args.root_counts:
        pssm_root = load_counts_matrix(args.root_counts, allowed_aas)
        if pssm_root.shape[0] != seq_len:
            raise SystemExit(f"{args.root_counts}: {pssm_root.shape[0]} sites against "
                             f"a {seq_len}-residue sequence")
    elif args.data_dir:
        pssm_root, _pssm_variant = load_pssms(os.path.join(args.data_dir, "pssms"), variant, E)
    else:
        raise SystemExit("--root-counts, or a --data-dir carrying pssms/root.csv.")
    # A cell is "seen" from --seen-min-count training-window sequences up (default 1:
    # the published de novo convention, pssm_root == 0). NOTES s162 sets 100, so the
    # `unseen` variants become "rare in training" (< 100 sequences).
    seen_mask = pssm_root >= args.seen_min_count
    if args.seen_min_count != 1:
        print(f"seen = training-window count >= {args.seen_min_count}: the `unseen` "
              f"variants are cells seen in fewer than {args.seen_min_count} sequences")
    reach1 = None
    if args.cd1_variants:
        from antiGen.models.codon_fitness_predictor import build_codon_lookup
        reach1 = build_codon_lookup(allowed_aas, pseudocount=0.01) > float(np.log(0.01)) + 1e-4
        reach1.fill_diagonal_(False)
        print(f"codon-distance-1 variants: {int(reach1.sum())} of {E * (E - 1)} "
              f"amino-acid substitutions reachable by one nucleotide change")
    variants = build_variants(lists, seen_mask, reach1)
    common_pool = (None if args.legacy_pool
                   else eu.real_substitution_mask(len(eval_range), allowed_aas))
    if common_pool is None:
        print("candidate pool: per-model emission set (LEGACY, pre-s51)")
    else:
        print(f"candidate pool: common -- {int(common_pool.sum())} of "
              f"{len(eval_range) * E} cells, gap/stop excluded for every model")
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
            if args.multitree_baselines or args.ensemble_dir or args.counted_baselines:
                print(msg + " Continuing: another flag supplies its own models.")
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

        builtin_models.update({
            "codon": lambda: create_model(ModelConfig(
                model_type=ModelType.CODON, seq_len=seq_len,
                alphabet=AlphabetConfig(amino_acids=allowed_aas),
                codon=CodonConfig()), device),
            "substitution": _fit,
            "where_times_which": lambda: WhereTimesWhichPredictor(
                _fit().config,
                positional_marginal_counts(training_examples, seq_len, E),
                _fit()).to(device),
            "wxw_dirichlet": lambda: where_times_which_dirichlet(
                _fit(), training_examples, seq_len, E).to(device),
        })
        for name in builtin_models:
            candidates.append((name, None))
        print(f"  added {len(builtin_models)}: {', '.join(builtin_models)}")

    if args.counted_baselines or args.mcm_baseline:
        print("\n=== Building counted-statistic baselines ===", flush=True)
        from antiGen.statistical_baselines import fit as _cs_fit
        _cs_cache = {}

        def _cs_stats():
            if "st" not in _cs_cache:
                print(f"  fitting counted statistics on {len(training_examples)} "
                      f"training examples...", flush=True)
                _cs_cache["st"] = _cs_fit(training_examples, seq_len, E)
                print(f"  N_total = {_cs_cache['st']['N_total']:.0f}", flush=True)
            return _cs_cache["st"]

        _cs_cfg = ModelConfig(model_type=ModelType.SITE_SUBSTITUTION, seq_len=seq_len,
                              alphabet=AlphabetConfig(amino_acids=allowed_aas))
        counted_models = {}
        if args.counted_baselines:
            counted_models.update({
                "site_substitution": lambda: CountedStatPredictor(
                    _cs_cfg, _cs_stats(), "site_substitution").to(device),
                "mcm_interpolation": lambda: CountedStatPredictor(
                    _cs_cfg, _cs_stats(), "mcm_interpolation").to(device),
            })
        if args.mcm_baseline:
            counted_models["mcm"] = (lambda: CountedStatPredictor(
                _cs_cfg, _cs_stats(), "mcm").to(device))
        builtin_models.update(counted_models)
        for _n in counted_models:
            candidates.append((_n, None))
        print(f"  added {len(counted_models)}: {', '.join(counted_models)}")

    if args.ensemble_dir:
        ens_paths = sorted(glob.glob(os.path.join(args.ensemble_dir, "fitness_*.pt")))
        if not ens_paths:
            raise SystemExit(f"--ensemble-dir {args.ensemble_dir} has no fitness_*.pt")
        if gap_idx is not None or stop_idx is not None:
            # A member that NaN-masks gap/stop would need a NaN-aware logsumexp to
            # match the cached rule's union_min; not needed for aa20, so refused.
            raise SystemExit("--ensemble-dir is implemented for the 20-letter alphabet "
                             "only (no gap/stop columns); see NOTES s123")

        def _ensemble():
            mems = []
            for ep in ens_paths:
                eck, ecfg = load_checkpoint(ep, map_location=device)
                if ecfg.seq_len != seq_len:
                    raise SystemExit(f"{ep}: seq_len {ecfg.seq_len} != {seq_len}")
                em = create_model(ecfg, device)
                em.load_state_dict(eck['model_state_dict'], strict=False)
                em.eval()
                mems.append(em)
                print(f"    member {os.path.basename(ep)}", flush=True)
            return MeanLogProbEnsemble(mems, eval_range=eval_range).to(device)

        builtin_models[args.ensemble_name] = _ensemble
        candidates.append((args.ensemble_name, None))

        if args.mix:
            if not args.mix_dir:
                raise SystemExit("--mix needs --mix-dir")
            mix_paths = sorted(glob.glob(os.path.join(args.mix_dir, "fitness_*.pt")))
            if not mix_paths:
                raise SystemExit(f"--mix-dir {args.mix_dir} has no fitness_*.pt")
            _mix_cache = {}

            def _members(paths):
                mems = []
                for ep in paths:
                    eck, ecfg = load_checkpoint(ep, map_location=device)
                    if ecfg.seq_len != seq_len:
                        raise SystemExit(f"{ep}: seq_len {ecfg.seq_len} != {seq_len}")
                    em = create_model(ecfg, device)
                    em.load_state_dict(eck['model_state_dict'], strict=False)
                    em.eval()
                    mems.append(em)
                return MeanLogProbEnsemble(mems, eval_range=eval_range).to(device)

            mix2_paths = (sorted(glob.glob(os.path.join(args.mix_dir2, "fitness_*.pt")))
                          if args.mix_dir2 else [])

            def _baseline(name):
                """A --mix-baseline component (NOTES s158): codon is untrained; the
                counted ones are fitted on this run's training examples."""
                if name == 'codon':
                    bm = create_model(ModelConfig(
                        model_type=ModelType.CODON, seq_len=seq_len,
                        alphabet=AlphabetConfig(amino_acids=allowed_aas),
                        codon=CodonConfig()), device)
                else:
                    from antiGen.statistical_baselines import fit as _bfit
                    bm = CountedStatPredictor(
                        ModelConfig(model_type=ModelType.SITE_SUBSTITUTION, seq_len=seq_len,
                                    alphabet=AlphabetConfig(amino_acids=allowed_aas)),
                        _bfit(training_examples, seq_len, E), name).to(device)
                bm.eval()
                return MeanLogProbEnsemble([bm], eval_range=eval_range).to(device)

            class _Mix(torch.nn.Module):
                """(1 - sum w) * A + sum_k w_k * comps[k] over the eval range; -inf outside.
                Each component is clamped at -1e4 first, as blend_select.py does (a no-op
                for every component used so far: none is -inf inside the eval range)."""
                def __init__(self, A, comps, ws):
                    super().__init__()
                    self.A, self.comps = A, list(comps)
                    self.ws = [float(w) for w in ws]

                def forward(self, aa):
                    a_ = self.A(aa)
                    clamp = lambda t: torch.nan_to_num(t[:, eval_range, :], neginf=-1e4)
                    s_ = (1 - sum(self.ws)) * clamp(a_)
                    for w_, c_ in zip(self.ws, self.comps):
                        if w_:
                            s_ = s_ + w_ * clamp(c_(aa))
                    out = torch.full_like(a_, float('-inf'))
                    out[:, eval_range, :] = s_
                    return out

            for item in args.mix:
                mname, ws = item.split('=', 1)
                ws = [float(x) for x in ws.split(':')]
                if len(ws) > 2 + len(args.mix_baseline):
                    raise SystemExit(f"--mix {item}: more weights than components")
                ws += [0.0] * (2 + len(args.mix_baseline) - len(ws))
                if ws[1] and not mix2_paths:
                    raise SystemExit(f"--mix {item} has a third weight but no --mix-dir2")

                def _mk(ws=tuple(ws)):
                    if 'A' not in _mix_cache:
                        _mix_cache['A'] = _ensemble()
                        _mix_cache['B'] = _members(mix_paths)
                        _mix_cache['C'] = _members(mix2_paths) if mix2_paths else None
                        _mix_cache['X'] = [_baseline(b) for b in args.mix_baseline]
                    return _Mix(_mix_cache['A'],
                                [_mix_cache['B'], _mix_cache['C']] + _mix_cache['X'], ws)
                builtin_models[mname] = _mk
                candidates.append((mname, None))
            print(f"  mix rungs: {', '.join(m.split('=')[0] for m in args.mix)} "
                  f"({len(ens_paths)} + {len(mix_paths)} members)")
        print(f"  ensemble rung '{args.ensemble_name}': {len(ens_paths)} member(s) "
              f"from {args.ensemble_dir}, normalised over the eval range")

    # ── Per-query precomputed maps (NOTES s151) ──────────────────────────────
    # A baseline that must be asked each query, scored outside this process (CoVFit's
    # CLI). Every scored query must have its own map -- checked here, before any pass.
    query_maps = {}
    for item in args.query_maps:
        qname, qpath = item.split('=', 1)
        z = np.load(qpath)
        if z['eval_range'].tolist() != list(eval_range):
            raise SystemExit(f"{qpath}: eval range differs from this run's")
        key = {bytes(g): i for i, g in enumerate(z['aa'].astype(np.int8))}
        rows = [key.get(bytes(e.aa_idx.to(torch.int8).numpy())) for e in scored]
        if any(r is None for r in rows):
            raise SystemExit(f"{qpath}: {sum(r is None for r in rows)} of {len(scored)} "
                             f"scored queries have no map")
        full = torch.full((len(scored), seq_len, E), float('nan'))
        full[:, eval_range, :] = torch.from_numpy(z['scores'][np.array(rows)]).float()
        query_maps[qname] = full
        print(f"  per-query rung '{qname}': {len(set(rows))} maps for {len(scored)} "
              f"scored queries ({qpath})")

    by_model = read_done(args.output_dir)
    skipped = []
    if by_model:
        print(f"Resuming: {len(by_model)} model(s) already in {SUMMARY}: {sorted(by_model)}")

    def _have_map(name):
        return bool(dump_dir) and os.path.exists(sc.map_path(dump_dir, name))

    def _resume_skip(name):
        """True only if every artifact this run is asked for already exists.

        Resume is per-ARTIFACT, not per-row. Keying it on the summary CSV alone is how a
        --dump-scores run pointed at a directory that already holds rows would print
        [resume-skip] for all 30 models, write ZERO maps, report 'Scored 30 model(s)' and
        exit 0 -- the same shape as the three silent successes NOTES s30/s31 record.
        """
        if name not in by_model:
            return False
        if dump_dir and not _have_map(name):
            print(f"[re-run] {name}: has a {SUMMARY} row but no cached map")
            return False
        return True

    def _verify_against_row(name, acc):
        """A model re-run only for its map must reproduce the row already on disk.

        Free, and it is the one place a cached map and a published number are produced by
        the same process, so a disagreement here means the scoring path moved.
        """
        old = {(suf, round(q, 4)): (ma, mi) for suf, q, ma, mi, _c, _e in by_model[name]}
        worst = 0.0
        for suf, q, ma, mi, _c, _e in summarize(acc):
            ref = old.get((suf, round(q, 4)))
            if ref is None:
                raise SystemExit(f"{name}: re-run produced variant {suf} q={q} that the "
                                 f"existing {SUMMARY} row does not have")
            worst = max(worst, abs(ref[0] - ma), abs(ref[1] - mi))
        if worst > 1e-6:
            raise SystemExit(f"{name}: re-run disagrees with the existing {SUMMARY} row "
                             f"by {worst:.2e}; refusing to publish a map against a row "
                             f"it does not reproduce")
        print(f"  [verified] {name} reproduces its existing row to {worst:.1e}")

    def _sink_for(name):
        """(sink, finish) for one model, or (None, no-op) when not dumping."""
        if not dump_dir:
            return None, (lambda ok=True: None)
        w = sc.MapWriter(dump_dir, name, len(scored), len(eval_range), E)

        def finish(ok=True):
            if not ok:
                w.abort()
                return
            side = w.close(manifest_sha1=dump_sha1,
                           extra={"n_queries": len(scored),
                                  "eval_range_len": len(eval_range)})
            print(f"  [map] {name}: {side['n_nan']} NaN, {side['n_neginf']} -inf cells",
                  flush=True)
        return w.write_batch, finish

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
        if _resume_skip(name):
            print(f"[resume-skip] {name}")
            continue
        print(f"\n=== Baseline: {name} ===", flush=True)
        bp = probs_2d.to(device)
        sink, finish = _sink_for(name)
        try:
            acc = cell_pass(
                lambda aa, bp=bp: bp.unsqueeze(0).expand(aa.shape[0], -1, -1),
                loader, variants, device, eval_range, qtiles, sink=sink,
                common_pool=common_pool)
        except BaseException:
            finish(ok=False)
            raise
        if name in by_model:
            _verify_against_row(name, acc)
            finish()
        else:
            finish()
            record(name, acc)
        del bp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── History models (NOTES s156): same unshuffled-loader offset as the query maps.
    if args.history_model and not args.pool_examples:
        raise SystemExit("--history-model is implemented for --pool-examples runs only")
    for item in args.history_model:
        hname, hpath = item.split('=', 1)
        if _resume_skip(hname):
            print(f"[resume-skip] {hname}")
            continue
        from antiGen.history import load_history_model, make_tokens, pad_histories
        hm = load_history_model(hpath, device)
        print(f"\n=== History model: {hname} ({hpath}) ===", flush=True)
        sink, finish = _sink_for(hname)
        _hoff = [0]

        def _hfn(aa, m=hm, off=_hoff):
            rows = scored[off[0]:off[0] + aa.shape[0]]
            off[0] += aa.shape[0]
            for r, e in zip(range(aa.shape[0]), rows):
                if not torch.equal(aa[r].cpu(), e.aa_idx):
                    raise SystemExit(f"{hname}: loader row and scored query disagree")
            hist, pad = pad_histories([make_tokens(e, m.tokens) for e in rows], aa.device)
            with torch.no_grad():
                return m(aa, hist, pad).float()
        try:
            acc = cell_pass(_hfn, loader, variants, device, eval_range, qtiles,
                            sink=sink, common_pool=common_pool)
        except BaseException:
            finish(ok=False)
            raise
        if _hoff[0] != len(scored):
            raise SystemExit(f"{hname}: consumed {_hoff[0]} of {len(scored)} queries")
        if hname in by_model:
            _verify_against_row(hname, acc)
        finish()
        if hname not in by_model:
            record(hname, acc)
        del hm

    # ── Per-query maps: the loader is unshuffled, so batch rows are `scored` in order.
    for name, full in query_maps.items():
        if _resume_skip(name):
            print(f"[resume-skip] {name}")
            continue
        print(f"\n=== Per-query baseline: {name} ===", flush=True)
        sink, finish = _sink_for(name)
        _off = [0]

        def _qfn(aa, full=full, off=_off):
            out = full[off[0]:off[0] + aa.shape[0]].to(aa.device)
            off[0] += aa.shape[0]
            return out
        try:
            acc = cell_pass(_qfn, loader, variants, device, eval_range, qtiles,
                            sink=sink, common_pool=common_pool)
        except BaseException:
            finish(ok=False)
            raise
        if _off[0] != len(scored):
            raise SystemExit(f"{name}: consumed {_off[0]} of {len(scored)} query maps")
        if name in by_model:
            _verify_against_row(name, acc)
        finish()
        if name not in by_model:
            record(name, acc)

    # ── Models ──────────────────────────────────────────────────────────────
    for prefix, path in candidates:
        if _resume_skip(prefix):
            print(f"[resume-skip] {prefix}")
            continue
        print(f"\n=== Model: {prefix} ===", flush=True)
        if prefix in builtin_models:
            model = builtin_models[prefix]()
            model.eval()
            model_mask = build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx).to(device)
            sink, finish = _sink_for(prefix)
            try:
                acc = cell_pass(
                    (lambda aa, m=model, mm=model_mask: _masked_scores(m, aa, mm)),
                    loader, variants, device, eval_range, qtiles, sink=sink,
                common_pool=common_pool)
            except BaseException:
                finish(ok=False)
                raise
            if prefix in by_model:
                _verify_against_row(prefix, acc)
                finish()
            else:
                finish()
                record(prefix, acc)
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

        sink, finish = _sink_for(prefix)
        try:
            acc = cell_pass(
                (lambda aa, m=model, mm=model_mask: _masked_scores(m, aa, mm)),
                loader, variants, device, eval_range, qtiles, sink=sink,
                common_pool=common_pool)
        except BaseException:
            finish(ok=False)
            raise
        if prefix in by_model:
            _verify_against_row(prefix, acc)
            finish()
        else:
            finish()
            record(prefix, acc)
        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n=== Wrote {os.path.join(args.output_dir, SUMMARY)} ===")
    print(f"Scored {len(by_model)} model(s)")

    if dump_dir:
        # "Wrote no maps and exited 0" is the failure this block exists to make
        # impossible. Check the artifacts on disk against the roster this run was asked
        # for, and name the gap rather than reporting success.
        want = sorted({n for n, _ in (() if args.skip_baselines else baselines)}
                      | {p for p, _ in candidates})
        have = set(sc.list_models(dump_dir))
        gap = [m for m in want if m not in have]
        tmp = sc.stray_tmp(dump_dir)
        exp = sc.expected_bytes(len(scored), len(eval_range), E)
        wrong = [m for m in want if m in have
                 and os.path.exists(sc.map_path(dump_dir, m))
                 and os.path.getsize(sc.map_path(dump_dir, m)) - 128 != exp]
        print(f"Score-map cache: {len(have)} map(s) in {dump_dir}")
        if gap or tmp or wrong:
            raise SystemExit(
                f"score-map cache is incomplete: "
                + (f"{len(gap)} missing ({', '.join(gap[:6])})" if gap else "")
                + (f"; stray partial writes: {', '.join(tmp)}" if tmp else "")
                + (f"; wrong payload size: {', '.join(wrong[:6])}" if wrong else ""))

    if skipped:
        raise SystemExit(
            f"{len(skipped)} model group(s) were skipped and are MISSING from "
            f"{SUMMARY}: " + "; ".join(f"{n} ({why})" for n, why in skipped))


if __name__ == "__main__":
    main()
