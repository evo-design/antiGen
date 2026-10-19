#!/usr/bin/env python
"""The CoG-UK list panels, scored on the guide tree alone -- no attachment phase.

**What this controls for.** Every number in `coguk_list_results` is computed over
`output1`'s test set, which is the guide phylogeny *after* `antiGen-data` has attached
~170k aligned sequences to it by parsimony. Two things about that are worth being able
to rule out:

* **the attachment itself.** Attached tips are placed by a parsimony criterion, and the
  test set's ancestral genotypes and target mutations are partly a product of that
  placement rather than of the Nextstrain build.
* **recurrence.** NOTES s31/s32: CoG-UK's 247 in-RBD cells carry 26,491 test events at a
  median of 46 per cell (max 972), 81% of them on cells already in the training window,
  which is why the event-weighted and cell-weighted readings disagree so sharply that
  the figure has to show both. Almost all of that multiplicity comes from attachment --
  the same mutation recorded against hundreds of near-identical backgrounds.

On the backbone alone the same list carries ~4 events per cell instead of ~46, so the
two weightings nearly coincide and the question the main figure splits into two columns
largely stops being a question.

**Nothing about the scoring changes.** The candidate pool, the wild-type and gap/stop
masking, the fractional tie rule and the de novo mask are the published ones: the top-q
weights come from ``top_q_weight``, which ``antiGen/tests/test_background_ranks.py``
pins against ``compute_hit_rate`` cell for cell. Only the test set changes.

**The test set.** Every post-cutoff guide-tree node with at least one post-cutoff child
branch is an example; its targets are the eval-range mutations on those branches. Unlike
the examples file, a multi-mutation branch is NOT decomposed into intermediate genotypes
in a random order -- all of its mutations are targets of the parent genotype, which is
the simpler statement and does not depend on `antiGen-data`'s RNG. Nodes sharing a spike
genotype are one query (a model sees only the sequence), so the per-cell accumulation is
done per genotype with the target counts summed over its nodes.

**The BA.2 yield row** keeps its query -- the backbone's BA.2 root genotype is
byte-identical to the IS_VARIANT_ROOT example the published run uses, checked here --
and replaces only "emerged by t", which in the published version counts attached
sequences through the variant PSSMs. Here it counts guide-tree tips descended from the
BA.2 root whose genotype carries the residue by t.

This re-uses the score matrices ``antiGen/background_discrimination.py`` already wrote,
so it needs no GPU and no forward passes.

Usage:
    python antiGen/backbone_list_eval.py \\
        --background-dir .../evals_sc2_background \\
        --data-dir /home/ivan.specht/antiGen-data/output1 \\
        --tree-json antiGen-data/tree.json --ref-json antiGen-data/ref.json \\
        --mutation-list coguk=coguk.csv \\
        --output-dir .../evals_sc2_lists_macro_backbone
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from antiGen.background_discrimination import rank_fractions, top_q_weight
from antiGen.background_tree import build_backbone, describe as describe_backbone
from antiGen.eval_utils import RBD_RANGE
from antiGen.load import load_pssms
from antiGen.mutation_lists import load_mutation_list

SUMMARY = "list_macro_summary_backbone.csv"
PERCELL = "list_macro_percell.csv"
VARR = "list_variant_recall.csv"


def target_matrix(bb, eval_range, E):
    """(Q, R, E) counts: how often each cell is a target of each query genotype."""
    pos_of = {p: i for i, p in enumerate(eval_range)}
    T = np.zeros((len(bb.query_seq), len(eval_range), E), dtype=np.float32)
    for (pos, aa), per_node in bb.hosts.items():
        r = pos_of[pos]
        for node, k in per_node.items():
            T[int(bb.node_qidx[node]), r, aa] += k
    return T


def ranks_under(scores, aa_idx, score_mask):
    """above/tie/n_valid with ``score_mask`` (True = exclude) taken out of the pool."""
    s = scores.clone()
    if score_mask is not None:
        s[:, score_mask] = float('nan')
    return rank_fractions(s, aa_idx, list(range(s.shape[1])))


def summarize(cell_hits, cell_events):
    observed = cell_events > 0
    n_events = float(cell_events.sum())
    n_cells = int(observed.sum())
    macro = float((cell_hits[observed] / cell_events[observed]).mean()) if n_cells else float('nan')
    micro = float(cell_hits.sum() / n_events) if n_events else float('nan')
    return macro, micro, n_cells, n_events


def ba2_emergence(bb, eval_range, E, root_node, n_steps, timestep):
    """(n_steps, R, E) counts of BA.2-descended guide-tree tips carrying each cell by t.

    The analogue of the variant PSSM the published metric reads, built from the guide
    tree's own tips instead of the attached alignment. "Descended from BA.2" is tree
    position, not the clade label, so 22C/22D/23I/25C -- all BA.2 descendants carrying
    their own labels -- are included.
    """
    allowed = bb.allowed_aas
    in_lo, in_hi = bb.node_in[root_node], bb.node_out[root_node]
    children = {}
    for i in range(len(bb.node_time)):
        p = int(bb.node_parent[i])
        if p >= 0:
            children.setdefault(p, []).append(i)
    counts = np.zeros((n_steps, len(eval_range), E), dtype=np.float32)
    n_tips = 0
    # Only genotypes inside the BA.2 subtree are needed, so the walk starts there and
    # each node's genotype is its parent's with that branch's mutations applied.
    seqs = {root_node: _genotype_of(bb, root_node)}
    order = [root_node]
    while order:
        i = order.pop()
        for c in children.get(i, []):
            seqs[c] = _apply(seqs[i], bb, c)
            order.append(c)
    for i, seq in seqs.items():
        if children.get(i):
            continue                              # tips only
        if not (in_lo <= bb.node_in[i] < in_hi):
            continue
        step = int(np.floor((bb.node_time[i] - bb.root_time) / timestep))
        step = max(0, min(n_steps - 1, step))
        n_tips += 1
        for r, p in enumerate(eval_range):
            counts[step, r, allowed.index(seq[p])] += 1
    return np.cumsum(counts, axis=0), n_tips


def _genotype_of(bb, node):
    """Rebuild one node's genotype from the root by replaying its path."""
    path = []
    i = node
    while i >= 0:
        path.append(i)
        i = int(bb.node_parent[i])
    seq = list(bb.root_seq)
    for i in reversed(path):
        for (pos, aa) in bb.branch_muts[i]:
            seq[pos] = bb.allowed_aas[aa]
    return "".join(seq)


def _apply(parent_seq, bb, node):
    seq = list(parent_seq)
    for (pos, aa) in bb.branch_muts[node]:
        seq[pos] = bb.allowed_aas[aa]
    return "".join(seq)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--background-dir', required=True,
                    help='holds the scores_*.npz written by background_discrimination.py')
    ap.add_argument('--data-dir', required=True, help='for pssms/root.csv, the de novo mask')
    ap.add_argument('--tree-json', required=True)
    ap.add_argument('--ref-json', required=True)
    ap.add_argument('--segment', default='S')
    ap.add_argument('--cutoff', type=float, default=1.0)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mutation-list', action='append', required=True, help='LABEL=PATH')
    ap.add_argument('--variant', default='21L (BA.2)')
    ap.add_argument('--qtiles', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    ap.add_argument('--yield-qtile', type=float, default=0.1)
    ap.add_argument('--n-steps', type=int, default=60)
    ap.add_argument('--timestep', type=float, default=0.1)
    ap.add_argument('--thresholds', default='1,100')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)
    qtiles = [float(q) for q in args.qtiles.split(',') if q.strip()]
    thresholds = [int(t) for t in args.thresholds.split(',') if t.strip()]

    with open(os.path.join(args.background_dir, "backbone_manifest.json")) as fh:
        manifest = json.load(fh)
    seq_len = manifest["seq_len"]
    eval_range = list(range(manifest["eval_range_start1"] - 1, manifest["eval_range_end1"]))
    R = len(eval_range)

    bb = build_backbone(args.tree_json, args.ref_json, args.segment, seq_len,
                        allowed_aas, args.cutoff, eval_range=eval_range)
    print(describe_backbone(bb.report))
    if bb.report['n_wt_mismatch']:
        raise SystemExit("branch mutations disagree with the reconstruction")

    T = target_matrix(bb, eval_range, E)
    print(f"test set: {len(bb.query_seq)} query genotypes, "
          f"{int((T > 0).any(axis=(1, 2)).sum())} of them carrying an eval-range target, "
          f"{T.sum():.0f} target events on {int((T.sum(axis=0) > 0).sum())} cells")

    pssm_root, _ = load_pssms(os.path.join(args.data_dir, "pssms"), args.variant, E)
    seen = (pssm_root > 0)[eval_range, :]

    lists = []
    for item in args.mutation_list:
        label, path = item.split('=', 1)
        cells, report = load_mutation_list(path, allowed_aas, seq_len,
                                           root_seq=bb.root_seq, label=label)
        in_full = torch.zeros(R, E, dtype=torch.bool)
        pos_of = {p: i for i, p in enumerate(eval_range)}
        for pos, aa in cells:
            if pos in pos_of:
                in_full[pos_of[pos], aa] = True
        lists.append((label, in_full))
        ev = float(T[:, in_full.numpy()].sum())
        n_obs = int(((T.sum(axis=0) > 0) & in_full.numpy()).sum())
        print(f"  {label}: {int(in_full.sum())} cells in range, {n_obs} observed on the "
              f"backbone, {ev:.0f} events ({ev / max(n_obs, 1):.1f} per observed cell)")

    # (suffix, score_mask, target_mask); True = exclude. Same four variants as eval.py.
    variants = []
    for label, in_full in lists:
        variants.append((f"list_{label}_fullpool", None, ~in_full))
        variants.append((f"list_{label}", ~in_full, ~in_full))
        variants.append((f"list_{label}_unseen_fullpool", seen, (~in_full) | seen))
        variants.append((f"list_{label}_unseen", (~in_full) | seen, (~in_full) | seen))

    aa_idx = torch.tensor([[allowed_aas.index(c) for c in seq[eval_range[0]:eval_range[-1] + 1]]
                           for seq in bb.query_seq], dtype=torch.long)
    T_t = torch.from_numpy(T)

    # The BA.2 root, and the check that it is the query the published run uses.
    ba2_nodes = [i for i, c in enumerate(bb.node_clade) if c == args.variant]
    if not ba2_nodes:
        raise SystemExit(f"no node labelled {args.variant!r} in the guide tree")
    ba2_root = min(ba2_nodes, key=lambda i: bb.node_in[i])
    ba2_q = int(bb.node_qidx[ba2_root])
    if ba2_q < 0:
        raise SystemExit(f"{args.variant} root is not a post-cutoff query genotype")
    print(f"{args.variant} root: node {ba2_root}, t = {bb.node_time[ba2_root]:.3f}, "
          f"query genotype {ba2_q}")
    emerged, n_tips = ba2_emergence(bb, eval_range, E, ba2_root, args.n_steps, args.timestep)
    print(f"  {n_tips} guide-tree tips descend from it")
    for label, in_full in lists:
        fin = in_full.numpy()
        for thr in thresholds:
            ever = int(((emerged[-1] >= thr) & fin).sum())
            print(f"  {label}: {ever} listed cells reach {thr}+ tips by the end of the grid")

    summary_rows, percell_rows, varr_rows = [], [], []
    paths = sorted(glob.glob(os.path.join(args.background_dir, "scores_*.npz")))
    if not paths:
        raise SystemExit(f"no scores_*.npz in {args.background_dir}")
    for path in paths:
        model = os.path.basename(path)[len("scores_"):-len(".npz")]
        with np.load(path) as z:
            scores = torch.from_numpy(z["scores"].astype(np.float32))
        for suffix, score_mask, target_mask in variants:
            above, tie, n_valid = ranks_under(scores, aa_idx, score_mask)
            tgt = T_t.clone()
            if target_mask is not None:
                tgt[target_mask.unsqueeze(0).expand_as(tgt)] = 0
            for q in qtiles:
                W = top_q_weight(above, tie, n_valid, q)
                cell_hits = (W * tgt).sum(dim=0)
                cell_events = tgt.sum(dim=0)
                macro, micro, n_cells, n_events = summarize(cell_hits, cell_events)
                summary_rows.append([model, suffix, f"{q:.2f}", f"{macro:.8f}",
                                     f"{micro:.8f}", n_cells, f"{n_events:.0f}"])
                if abs(q - 0.1) < 1e-9:
                    for r, e in (cell_events > 0).nonzero().tolist():
                        percell_rows.append([model, suffix, f"{q:.2f}", eval_range[r] + 1,
                                             allowed_aas[e], f"{cell_events[r, e]:.0f}",
                                             f"{cell_hits[r, e]:.4f}",
                                             f"{float(cell_hits[r, e] / cell_events[r, e]):.6f}"])
        # ── The BA.2 yield series, on the backbone's own emergence ──────────
        above, tie, n_valid = ranks_under(scores, aa_idx, None)
        W = top_q_weight(above, tie, n_valid, args.yield_qtile)[ba2_q]
        n_selected = float(args.yield_qtile * n_valid[ba2_q])
        for label, in_full in lists:
            fin = in_full.numpy()
            for thr in thresholds:
                for step in range(args.n_steps):
                    for subset, extra in (("all", None), ("unseen", ~seen.numpy())):
                        tmask = fin & (emerged[step] >= thr)
                        if extra is not None:
                            tmask = tmask & extra
                        num = float((W.numpy() * tmask).sum())
                        varr_rows.append([model, label, subset, thr, step,
                                          f"{num / n_selected:.8f}", int(tmask.sum()),
                                          f"{n_selected:.2f}"])
        print(f"  {model:45s} done", flush=True)

    with open(os.path.join(args.output_dir, SUMMARY), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "qtile", "macro", "micro", "n_cells", "n_events"])
        w.writerows(summary_rows)
    with open(os.path.join(args.output_dir, PERCELL), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "qtile", "pos1", "alt", "k_events", "hits", "rate"])
        w.writerows(percell_rows)
    with open(os.path.join(args.output_dir, VARR), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "list", "subset", "threshold", "step", "value",
                    "n_targets", "n_selected"])
        w.writerows(varr_rows)
    print(f"\nWrote {SUMMARY} ({len(summary_rows)} rows) and {VARR} ({len(varr_rows)} rows) "
          f"to {args.output_dir}")


if __name__ == "__main__":
    main()
