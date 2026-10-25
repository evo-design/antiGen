#!/usr/bin/env python
"""Score every backbone query genotype, and store where each cell ranks in each.

**The question.** Figure 2 and the CoG-UK list panels ask, for a fixed query, where a
mutation ranks among that query's candidates -- "within a query, rank the mutations".
This script is the input to the transpose: for a fixed *mutation*, rank the *queries*.
Fixing the cell and varying the background cancels every per-mutation bias a model has,
including the one NOTES s32 identified as the reason base antiGen trails in the BA.2
yield panel (its decile fills with cells it saw in training). What is left is how much a
model's prediction moves with the sequence it is given.

**What is written.** Not a metric, and not a rank: the model's raw score for every
eval-range cell of every backbone query genotype, (Q, R, E) float32, ~7 MB per model.
Everything downstream -- which pool a rank is taken in, which cells are targets, which
backgrounds are compared -- is then decided without a second forward pass.

Keeping scores rather than ranks is not an optimisation, it is the thing that makes the
contrast valid. ``compute_hit_rate`` masks each query's own residue at every eval-range
position, so the candidate pool differs from query to query; a derived genotype has had
more of its high-scoring cells masked away, which lifts the percentile of everything
left. Compared ACROSS queries that is worth ~0.1 of AUC to predictors with no access to
the query at all -- measured on the pilot run: CoVFit 0.609, pyr0 0.614, Omicron BA.1
bind 0.600, pssm_root 0.418, all of which are fixed (L, E) maps and must sit at 0.5.
``antiGen/background_contrast.py`` therefore ranks each cell in a pool held fixed across
queries, and reports the published per-query convention beside it as the diagnostic it
turned out to be.

``rank_fractions`` / ``top_q_weight`` here are the bridge back to the published metric:
they rebuild ``compute_hit_rate``'s fractional top-q weight from a pool exactly, which
``antiGen/tests/test_background_ranks.py`` pins, and the contrast script uses them for
its per-query column.

The pool is the eval range, not the mutation list -- the ``_fullpool`` convention of the
list evals, for the reason NOTES s31 gives: handing a model only the listed cells
changes its task from "find these in the protein" to "sort a list you were given".

**Queries are the backbone genotypes**, i.e. the guide tree before ``antiGen-data``'s
attachment phase, restricted to post-cutoff nodes and deduplicated by sequence -- 402 of
them for sc2 spike. See ``antiGen/background_tree.py`` for why the attached tree is the
wrong substrate for this question. 402 forward passes is a minute of GPU per model
rather than the hours ``eval.py`` spends, so the whole roster fits in one short job.

Roster, filtering, resume and the exit-non-zero-on-skip behaviour follow
``antiGen/list_macro.py``; the models scored here are the models the published CSVs
report.

Usage:
    python antiGen/background_discrimination.py \\
        --data-dir /home/ivan.specht/antiGen-data/output1 \\
        --tree-json antiGen-data/tree.json --ref-json antiGen-data/ref.json \\
        --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \\
        --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2_background
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import hashlib
import json
import os

import numpy as np
import torch

from antiGen import datasets
from antiGen.background_tree import (build_backbone, build_backbone_from_examples,
                                     describe as describe_backbone)
from antiGen.config import AlphabetConfig, CodonConfig, ModelConfig, ModelType
from antiGen.eval import (config_label, dataset_of_train_args, discover_checkpoints,
                          peek_checkpoint_meta)
from antiGen.eval_utils import (RBD_RANGE, build_model_gap_stop_mask,
                                fit_substitution_baseline, load_baselines,
                                load_checkpoint, positional_marginal_counts,
                                where_times_which_dirichlet,
                                WhereTimesWhichPredictor)
from antiGen.config import LEGACY_AAS
from antiGen.load import load_counts_matrix, load_examples, load_pssms
from antiGen.models.factory import create_model

def peek_root_example(path):
    """The root example's sequence, without loading the whole examples file.

    ``load_examples`` materialises a dense (seq_len, E) target matrix per example --
    ~19 GB for output1's 171,248 -- and all this script needs from the file is the
    sequence length and the root genotype, so the first block is parsed directly. The
    root is example 0 and its sequence is the line before ``MUTATIONS_TO_CHILDREN:``.
    """
    head = []
    with open(path) as fh:
        for line in fh:
            head.append(line.rstrip("\n"))
            if line.startswith("MUTATIONS_TO_CHILDREN:"):
                break
            if len(head) > 500:
                raise SystemExit(f"{path}: no MUTATIONS_TO_CHILDREN: in the first block")
    if not head or not head[0].startswith("EXAMPLE_ID: 0"):
        raise SystemExit(f"{path}: does not start with EXAMPLE_ID: 0")
    return head[-2]


QUERIES = "backbone_queries.csv"
NODES = "backbone_nodes.csv"
HOSTS = "backbone_hosts.csv"
MANIFEST = "backbone_manifest.json"


def rank_fractions(scores, aa_batch, eval_range, pool_mask=None, return_filled=False,
                   wt_pool_mask=None):
    """(above, tie, n_valid): rank fractions over the eval-range pool.

    ``scores`` is (B, seq_len, E) or (B, R, E); ``aa_batch`` is (B, seq_len). ``above``
    and ``tie`` are (B, R, E) and NaN wherever the cell is not a candidate; ``n_valid``
    is (B,), the size of each query's pool. The pool is masked exactly as
    ``compute_hit_rate`` masks it -- the query's own residue at every eval-range
    position, and ``-inf`` sentinels -- and :func:`top_q_weight` turns these three back
    into that function's fractional top-q weight, cell for cell.

    ``pool_mask`` is the (R, E) COMMON CANDIDATE POOL and is passed through to keep that
    identity holding: it must mean here exactly what it means in ``compute_hit_rate``,
    or the ensemble path and the published path stop being the same metric. A cell in
    the pool that the model cannot score ranks last; a cell outside it leaves the pool.

    ``wt_pool_mask`` is the (B, R) query-dependent half -- ``eval_utils`` builds it with
    :func:`~antiGen.eval_utils.real_wildtype_mask` -- and drops every candidate at a
    position whose query residue is a gap or a stop, so ``-123A`` cannot be selected.
    The same argument means the same thing in ``compute_hit_rate``, for the same
    reason.

    ``return_filled`` adds a fourth return value: the (B, R, E) mask of in-pool cells the
    model could not score, which were filled to rank last. **A caller using a pool mask
    must zero its top-q weights there.** Ranking last is not the same as never being
    selected: the filled cells form one tie block at the bottom, so once q is large
    enough that the quota exceeds the SCORABLE count, the boundary lands inside that
    block and every cell in it takes fractional weight. Measured on BLOSUM62, which
    carries query-dependent ``-inf`` (460 in-pool cells at one test query, 80 at
    another): nothing at q <= 0.5, but **75.8 of the 3,437.1 selected at q = 0.9**,
    worth -0.0039 of its recall AUC. That is a baseline being credited for cells it
    declined to score, so it is removed rather than reported.
    """
    aa_sub = aa_batch if aa_batch.shape[1] == len(eval_range) else aa_batch[:, eval_range]
    sub = scores[:, eval_range, :] if scores.shape[1] != len(eval_range) else scores
    sub = sub.clone().float()
    pm = None
    if pool_mask is not None:
        pm = pool_mask.to(sub.device).unsqueeze(0).expand_as(sub)
    if wt_pool_mask is not None:
        wt = wt_pool_mask.to(sub.device)
        if wt.shape != sub.shape[:2]:
            raise ValueError(f"wt_pool_mask {tuple(wt.shape)} does not match the "
                             f"(batch, eval-range) grid {tuple(sub.shape[:2])}")
        wt = wt.unsqueeze(-1).expand_as(sub)
        pm = wt if pm is None else (pm & wt)
    if pm is not None:
        unscorable = (torch.isnan(sub) | torch.isneginf(sub)) & pm
        filled_mask = unscorable.clone()
        if unscorable.any():
            finite = torch.where(torch.isfinite(sub), sub,
                                 torch.full_like(sub, float('inf')))
            floor = finite.reshape(sub.shape[0], -1).min(dim=1).values
            floor = torch.where(torch.isfinite(floor), floor,
                                torch.zeros_like(floor)).view(-1, 1, 1) - 1.0
            sub = torch.where(unscorable, floor.expand_as(sub), sub)
        sub = torch.where(pm, sub, torch.full_like(sub, float('nan')))
    else:
        filled_mask = torch.zeros_like(sub, dtype=torch.bool)
    sub.scatter_(2, aa_sub.unsqueeze(-1), float('nan'))
    filled_mask = filled_mask & ~torch.isnan(sub)
    sub = torch.where(torch.isneginf(sub), torch.full_like(sub, float('nan')), sub)

    B = sub.shape[0]
    flat = sub.reshape(B, -1)
    valid = ~torch.isnan(flat)
    n_valid = valid.sum(dim=1).clamp(min=1).float()                  # (B,)
    # Invalid cells sort to the end and never count as "above" a finite score.
    filled = torch.where(valid, flat, torch.full_like(flat, float('inf')))
    sorted_vals, _ = torch.sort(filled, dim=1)
    probe = torch.where(valid, flat, torch.zeros_like(flat))
    n_le = torch.searchsorted(sorted_vals.contiguous(), probe.contiguous(), right=True)
    n_lt = torch.searchsorted(sorted_vals.contiguous(), probe.contiguous(), right=False)
    n_above = n_valid.unsqueeze(1) - n_le.float()
    n_tie = (n_le - n_lt).float()

    above = (n_above / n_valid.unsqueeze(1)).reshape(sub.shape)
    tie = (n_tie / n_valid.unsqueeze(1)).reshape(sub.shape)
    bad = torch.isnan(sub)
    above[bad] = float('nan')
    tie[bad] = float('nan')
    if return_filled:
        return above, tie, n_valid, filled_mask
    return above, tie, n_valid


def top_q_weight(above, tie, n_valid, q):
    """``compute_hit_rate``'s fractional top-q weight, rebuilt from the stored ranks.

    The boundary is the value ``torch.nanquantile(..., 1 - q, interpolation='lower')``
    picks: the element at index ``floor((n_valid - 1) * (1 - q))`` of the ascending
    pool, i.e. the one ``r = n_valid - 1 - that`` places from the top. A cell with ``a``
    cells strictly above it and a tied block of ``t`` occupies top ranks ``[a, a + t)``,
    so the boundary block is the one whose interval contains ``r``, and ``A`` is that
    block's ``a``. Cells above the boundary take the whole weight; the boundary block
    splits what is left of ``q * n_valid``; everything below takes nothing.

    Deriving ``A`` this way rather than from ``r`` directly is what makes the identity
    hold under ties -- with a coarse predictor (BLOSUM62's handful of integer levels) a
    single block can straddle the boundary, and then the count strictly above it is far
    less than ``r``. Reproducing the boundary at all, rather than assuming the selection
    is exactly ``q * n_valid`` cells, matters for the same reason in the other
    direction: the two differ by up to one cell, and the published recall numbers are
    defined by the former.
    """
    flat_shape = (above.shape[0], -1)
    nv = n_valid.reshape(-1, 1).float()
    # The stored fractions are counts over the pool, so round back to the integers they
    # came from: at n_valid ~ 4,200 the float32 round trip is off by ~1e-4, which an
    # exact comparison against an integer rank would get wrong.
    a = torch.round(above.reshape(flat_shape) * nv)
    t = torch.round(tie.reshape(flat_shape) * nv)
    r = nv - 1.0 - torch.floor((nv - 1.0).double() * (1.0 - q) + 1e-9).float()

    TOL = 1e-3
    is_boundary = (a <= r + TOL) & (r < a + t - TOL) & ~torch.isnan(a)
    A = torch.where(is_boundary, a, torch.full_like(a, float('inf'))).min(dim=1, keepdim=True).values
    T = torch.where(is_boundary, t, torch.full_like(t, float('nan'))).nanmean(dim=1, keepdim=True)
    frac = torch.clamp((q * nv - A) / T.clamp(min=1e-12), 0.0, 1.0)

    w = torch.where(a + t <= A + TOL, torch.ones_like(a),
                    torch.where((a - A).abs() < TOL, frac.expand_as(a),
                                torch.zeros_like(a)))
    w = torch.where(torch.isnan(a), torch.zeros_like(w), w)
    return w.reshape(above.shape)


def _masked_scores(model, aa_batch, model_gap_stop_mask):
    with torch.no_grad():
        scores = model(aa_batch).float()
    scores[:, model_gap_stop_mask] = float('nan')
    return scores


def score_pass(score_fn, aa_queries, eval_range, device, batch_size):
    """(Q, R, E) eval-range scores for every query genotype.

    Raw scores, not ranks. The contrast this feeds compares one cell ACROSS queries, and
    which pool a rank is taken in then stops being an implementation detail: the
    published convention masks each query's own residues, so the pool differs per query,
    and a derived genotype's pool has had more of its high-scoring cells removed. That
    is worth ~0.1 of AUC to a predictor with no access to the query at all (measured:
    CoVFit 0.609, pyr0 0.614, pssm_root 0.418, where all three should sit at 0.5), which
    would swamp the effect being measured. Storing scores lets the pool be chosen
    downstream, and lets both conventions be reported from one forward pass.

    Only the model's illegitimate gap/stop columns are masked here, because that mask is
    a property of the model and not of the query. ``-inf`` sentinels are kept as they
    are: for the codon baseline they mean "not reachable from THIS query", which is
    query adaptation and not an artefact.
    """
    out = []
    for start in range(0, aa_queries.shape[0], batch_size):
        aa_batch = aa_queries[start:start + batch_size].to(device)
        scores = score_fn(aa_batch)
        sub = scores[:, eval_range, :] if scores.shape[1] != len(eval_range) else scores
        out.append(sub.float().cpu().numpy().astype(np.float32))
        del scores, sub
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out)


def scores_path(output_dir, model):
    safe = model.replace('/', '_')
    return os.path.join(output_dir, f"scores_{safe}.npz")


def write_backbone(output_dir, bb, allowed_aas, eval_range, args):
    with open(os.path.join(output_dir, QUERIES), "w", newline="") as f:
        w = csv.writer(f)
        # eval_seq is the genotype restricted to the eval range: enough to tell whether
        # a cell is a mutation from this query at all, which is what the contrast needs,
        # without copying 402 full spike sequences into the directory.
        w.writerow(["qidx", "n_nodes", "n_branches", "t_min", "t_max", "t_mean",
                    "clades", "eval_seq", "seq_sha1"])
        for q, seq in enumerate(bb.query_seq):
            nodes = bb.query_nodes[q]
            ts = bb.node_time[nodes]
            clades = sorted({bb.node_clade[i] for i in nodes})
            w.writerow([q, len(nodes), bb.query_branches[q], f"{ts.min():.4f}",
                        f"{ts.max():.4f}", f"{ts.mean():.4f}", "|".join(clades),
                        "".join(seq[i] for i in eval_range),
                        hashlib.sha1(seq.encode()).hexdigest()])

    with open(os.path.join(output_dir, NODES), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "parent", "dfs_in", "dfs_out", "time", "clade", "qidx"])
        for i in range(len(bb.node_time)):
            w.writerow([i, int(bb.node_parent[i]), int(bb.node_in[i]),
                        int(bb.node_out[i]), f"{bb.node_time[i]:.4f}",
                        bb.node_clade[i], int(bb.node_qidx[i])])

    with open(os.path.join(output_dir, HOSTS), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos1", "alt", "node", "qidx", "n_events"])
        for (pos, aa) in sorted(bb.hosts):
            for node, k in sorted(bb.hosts[(pos, aa)].items()):
                w.writerow([pos + 1, allowed_aas[aa], node, int(bb.node_qidx[node]), k])

    manifest = dict(bb.report)
    manifest.update(eval_range_start1=eval_range[0] + 1, eval_range_end1=eval_range[-1] + 1,
                    n_eval_positions=len(eval_range), allowed_aas=allowed_aas,
                    args={k: v for k, v in vars(args).items()})
    with open(os.path.join(output_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='sc2', choices=sorted(datasets.DATASETS))
    ap.add_argument('--alphabet', default=LEGACY_AAS,
                    help='Residue alphabet. The default is the 22-token frame every '
                         'published invocation uses; the revised corpus passes 20.')
    ap.add_argument('--train-examples', nargs='+', default=None,
                    help='The training pool, in place of the data-dir glob. The revised '
                         'generator writes the corpus halves as separate files.')
    ap.add_argument('--root-counts', default=None, metavar='TSV',
                    help="The revised count matrix, where pssms/root.csv used to be.")
    ap.add_argument('--split-date', default=None, metavar='YYYY-MM-DD',
                    help='Which split to read, for a POST_CUTOFF set corpus.')
    ap.add_argument('--data-dir', default=None,
                    help='antiGen-data output dir (examples + pssms), for the alphabet, '
                         'sequence length, root PSSM and the training examples the '
                         'factored baselines are fitted on')
    ap.add_argument('--tree-json', default=None)
    ap.add_argument('--ref-json', default=None)
    ap.add_argument('--backbone-examples', default=None, metavar='FILE',
                    help="Take the backgrounds from antiGen-data-revised's example "
                         "corpus instead of a Nextstrain tree: every record is already "
                         "a genotype with its parent and its child mutations, so no "
                         "ancestral reconstruction runs and there is nothing for it to "
                         "disagree with. This is the ATTACHED tree, and it carries no "
                         "dates, so the run it feeds must be contrasted with "
                         "background_contrast.py --reference-query rather than the "
                         "published time-matched rule.")
    ap.add_argument('--segment', default='S')
    ap.add_argument('--cutoff', type=float, default=1.0,
                    help="years after the root date, as antiGen-data's CLI takes it")
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--eval-range', default='default', choices=['default', 'full'])
    ap.add_argument('--variant', default=None)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--models', default=None, help='comma-separated substrings to keep')
    ap.add_argument('--exclude', default='transformer_,esmc,vespag',
                    help="eval.py's default, so this scores the published roster")
    ap.add_argument('--include-experimental', action='store_true')
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--skip-baselines', action='store_true')
    ap.add_argument('--multitree-baselines', action='store_true',
                    help='also score codon / substitution / where x which')
    ap.add_argument('--substitution-seed', type=int, default=14)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allowed_aas = args.alphabet
    E = len(allowed_aas)
    # None when the alphabet has no such column; build_model_gap_stop_mask masks
    # nothing then, which is the right no-op at 20 letters (NOTES s54).
    gap_idx = allowed_aas.index('-') if '-' in allowed_aas else None
    stop_idx = allowed_aas.index('*') if '*' in allowed_aas else None
    print(f"Device: {device}")

    variant = args.variant or datasets.default_variant(args.dataset)
    if args.train_examples:
        examples_files = list(args.train_examples)
    elif args.data_dir:
        examples_files = sorted(glob.glob(
            os.path.join(args.data_dir, "examples_cutoff_*.txt")))
    else:
        raise SystemExit("--data-dir, or --train-examples.")
    if not examples_files:
        raise SystemExit(f"No examples_cutoff_*.txt found in {args.data_dir}")
    examples_root = peek_root_example(examples_files[0])
    seq_len = len(examples_root)
    eval_range = list(range(seq_len)) if args.eval_range == 'full' else (
        list(range(seq_len)) if args.dataset in ('flu', 'rsv_a', 'denv2') else RBD_RANGE)
    global_wt_aa_idx = torch.tensor([allowed_aas.index(c) for c in examples_root],
                                    dtype=torch.long)
    # The full examples file is only needed to FIT the factored baselines; loading it
    # costs ~19 GB, so it is loaded only when they are asked for.
    training_examples = None
    if args.multitree_baselines:
        all_examples = load_examples(examples_files[0], allowed_aas, variant,
                                     split_date=args.split_date)
        training_examples = [e for e in all_examples if e.is_training]
        if "".join(allowed_aas[i] for i in training_examples[0].aa_idx.tolist()) != examples_root:
            raise SystemExit("the root example parsed two ways disagree")
    print(f"Sequence length {seq_len}; eval range {len(eval_range)} positions "
          f"({eval_range[0] + 1}-{eval_range[-1] + 1}, 1-based)")

    # ── The backbone, before attachment ─────────────────────────────────────
    if args.backbone_examples:
        bb = build_backbone_from_examples(args.backbone_examples, seq_len, allowed_aas,
                                          variant, eval_range=eval_range,
                                          split_date=args.split_date)
    else:
        if not (args.tree_json and args.ref_json):
            raise SystemExit("--tree-json with --ref-json, or --backbone-examples.")
        bb = build_backbone(args.tree_json, args.ref_json, args.segment, seq_len,
                            allowed_aas, args.cutoff, eval_range=eval_range)
    print(describe_backbone(bb.report))
    if bb.report['n_wt_mismatch']:
        raise SystemExit(f"{bb.report['n_wt_mismatch']} branch mutations disagree with "
                         f"the reconstructed parent residue; the reconstruction is wrong")
    # The reference the backbone is grown from must be the sequence antiGen-data put at
    # the root of the examples file; if it is not, every genotype here is off.
    if args.backbone_examples:
        print("  backbone and corpus are the same file; no reconstruction to check")
    elif bb.root_seq != examples_root:
        diff = sum(1 for a, b in zip(bb.root_seq, examples_root) if a != b)
        raise SystemExit(f"backbone root genotype differs from the examples root at "
                         f"{diff} position(s); the backbone and the corpus are not the "
                         f"same tree")
    print("  root genotype matches the examples file's root")

    write_backbone(args.output_dir, bb, allowed_aas, eval_range, args)

    aa_queries = torch.tensor(
        [[allowed_aas.index(c) for c in seq] for seq in bb.query_seq], dtype=torch.long)
    print(f"Query tensor: {tuple(aa_queries.shape)}")

    if args.root_counts:
        pssm_root = load_counts_matrix(args.root_counts, allowed_aas)
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir, or --root-counts.")
        pssm_root, _ = load_pssms(os.path.join(args.data_dir, "pssms"), variant, E)
    baseline_subdir = {'flu': 'baselines_flu', 'rsv_a': 'baselines_rsv_a',
                       'denv2': 'baselines_denv2'}.get(args.dataset, 'baselines')
    baseline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                                baseline_subdir)
    baselines = load_baselines(baseline_dir, allowed_aas, args.seed, pssm_root,
                               global_wt_aa_idx)

    # ── Checkpoints ─────────────────────────────────────────────────────────
    model_filters = [s.strip() for s in args.models.split(',') if s.strip()] if args.models else None
    exclude_filters = [s.strip() for s in args.exclude.split(',') if s.strip()] if args.exclude else None
    candidates, groups = [], {}
    if not args.baselines_only:
        groups = discover_checkpoints(args.checkpoint_dir, model_filters, exclude_filters)
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
        if not candidates:
            msg = (f"No checkpoints match --dataset {args.dataset} "
                   f"({len(groups)} group(s) discovered, all filtered out).")
            if args.multitree_baselines:
                print(msg + " Continuing: --multitree-baselines supplies its own models.")
            else:
                raise SystemExit(msg)

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
                _fit().config, positional_marginal_counts(training_examples, seq_len, E),
                _fit()).to(device),
            "wxw_dirichlet": lambda: where_times_which_dirichlet(
                _fit(), training_examples, seq_len, E).to(device),
        }
        for name in builtin_models:
            candidates.append((name, None))
        print(f"  added {len(builtin_models)}: {', '.join(builtin_models)}")

    skipped, scored = [], []

    def record(name, scores):
        np.savez_compressed(scores_path(args.output_dir, name), scores=scores,
                            eval_range=np.asarray(eval_range, dtype=np.int32))
        scored.append(name)
        finite = np.isfinite(scores)
        print(f"  {name:45s} queries={scores.shape[0]} cells={scores.shape[1]}x"
              f"{scores.shape[2]} finite/query={finite.sum() / scores.shape[0]:.0f}",
              flush=True)

    for name, probs_2d in (() if args.skip_baselines else baselines):
        if os.path.exists(scores_path(args.output_dir, name)):
            print(f"[resume-skip] {name}")
            continue
        print(f"\n=== Baseline: {name} ===", flush=True)
        bp = probs_2d.to(device)
        record(name, score_pass(lambda aa, bp=bp: bp.unsqueeze(0).expand(aa.shape[0], -1, -1),
                                aa_queries, eval_range, device, args.batch_size))
        del bp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for prefix, path in candidates:
        if os.path.exists(scores_path(args.output_dir, prefix)):
            print(f"[resume-skip] {prefix}")
            continue
        print(f"\n=== Model: {prefix} ===", flush=True)
        if prefix in builtin_models:
            model = builtin_models[prefix]()
            model.eval()
            mm = build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx).to(device)
            record(prefix, score_pass(lambda aa, m=model, mm=mm: _masked_scores(m, aa, mm),
                                      aa_queries, eval_range, device, args.batch_size))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        # A model that cannot be built is skipped so one bad checkpoint does not cost
        # the rows already scored -- but the run exits NON-ZERO naming it (NOTES s30).
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
            print(f"  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        model.eval()
        mm = build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx).to(device)
        record(prefix, score_pass(lambda aa, m=model, mm=mm: _masked_scores(m, aa, mm),
                                  aa_queries, eval_range, device, args.batch_size))
        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n=== Wrote {len(scored)} score matrices to {args.output_dir} ===")
    if skipped:
        raise SystemExit(
            f"{len(skipped)} model group(s) were skipped and are MISSING: "
            + "; ".join(f"{n} ({why})" for n, why in skipped))


if __name__ == "__main__":
    main()
