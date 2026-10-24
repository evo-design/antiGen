#!/usr/bin/env python
"""A weighted blend of where x which and the root PSSM, with the weight chosen on a holdout.

**What this is.** The two strongest composition baselines on the CoG-UK panels say
different things. ``where_times_which`` is the factorized ``P(i) * P(a | wt_i)``: it knows
where mutations happen and what substitutions are chemically and codon-wise likely, but
nothing position-specific about *which* residue. ``pssm_root`` is the training-window
count of each residue at each site: entirely position-specific, and blind to the query's
own residue. Neither contains the other, so a blend can beat both -- and on the 20-letter
arm they sit at 0.4581 and 0.4491 on panel C, within 0.01 of each other, which is exactly
the regime where that is worth asking.

**The blend is geometric, in log-probability space**, which is the convention the rest of
this pipeline combines predictors in (``ensemble_rules`` ``logsoftmax``/``arith``, the
published "Mean of log-probs"):

    blend_w = w * log_softmax(wxw) + (1 - w) * log_softmax(log1p(pssm))

``log1p`` matters and is not cosmetic. The published ``pssm_root`` baseline passes RAW
counts to the metric, which only ranks, so any monotone transform of them is the same
baseline -- but a blend adds the two components, so their scales have to be commensurate
first. ``log1p`` is the repo's own reading of a PSSM (``--model-type pssm`` prints
"PSSM log(count+1)"), and because it is monotone, **w = 0 reproduces the published
pssm_root ranking cell for cell**. ``w = 1`` likewise reproduces ``where_times_which``.
Both are checked by ``--gate``.

**The weight is selected on a random holdout, N times, and the N blends are averaged** --
the protocol of NOTES s56, so the two arms are read the same way. Per split:

* the 90% train half fits ``where_times_which`` (its positional marginal AND its
  substitution matrix, both from examples, so both are genuinely re-fitted);
* the 10% val half chooses ``w`` over a grid, by NLL per mutation;
* the blended map is written to the score cache as ``<prefix>_s<i>``.

``ensemble_report.py`` then takes the mean of the N maps' log-probs, exactly as it does
for the s56 antiGen arm.

**ONE COMPONENT IS NOT RE-FITTED, AND THE BIAS HAS A DIRECTION.** ``pssm_root`` is a count
over the training window's 351,410 *sequences*, not over the 2,934 *examples*, so there is
no 90% of it to take -- the examples and the MSA rows are not in correspondence. The val
examples are branches of the same pre-cutoff tree whose sequences those counts include, so
``pssm_root`` has in that sense already seen them, and a ``w`` chosen on val is biased
**towards the PSSM**. The selected weights are printed for exactly that reason: if they
sit hard against w = 0 the selection is suspect, and if they are interior it is less so.
The test set is post-cutoff and is untouched by this either way.

Usage:
    python antiGen/hybrid_baseline.py \\
        --maps-dir .../list_score_maps_aa20 \\
        --train-examples .../examples_2021_cur.txt \\
        --root-counts .../counts_2021_cur.tsv \\
        --alphabet ACDEFGHIKLMNPQRSTVWY --n-splits 10 --output-dir .../hybrid_arm
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from antiGen import score_cache as sc
from antiGen.config import (AlphabetConfig, LEGACY_AAS, ModelConfig, ModelType)
from antiGen.eval_utils import (compute_probs, fit_substitution_baseline,
                                positional_marginal_counts,
                                WhereTimesWhichPredictor)
from antiGen.load import load_counts_matrix, load_examples
from antiGen.train import split_examples

REPORT = "hybrid_weights.csv"


def log_norm(x):
    """log_softmax over the flattened (L, E) grid, per query. Shape-preserving."""
    flat = torch.log_softmax(x.reshape(x.shape[0], -1), dim=1)
    return flat.reshape(x.shape)


def fit_wxw(train_examples, seq_len, E, allowed_aas, device, subst_seed):
    """``where_times_which`` fitted on one split's train half, in eval mode."""
    subst = fit_substitution_baseline(train_examples, device, allowed_aas,
                                      seed=subst_seed)
    counts = positional_marginal_counts(train_examples, seq_len, E)
    model = WhereTimesWhichPredictor(subst.config, counts, subst).to(device)
    model.eval()
    return model


@torch.no_grad()
def blend_batch(wxw, pssm_log, aa_batch):
    """(B, L, E) normalised log-prob maps for each component, same query batch.

    Returned separately rather than already mixed, so one forward pass serves every
    weight on the grid -- the search is then free and the components are provably the
    same ones the chosen weight is applied to.
    """
    a = log_norm(wxw(aa_batch).float())
    b = log_norm(pssm_log.unsqueeze(0).expand(aa_batch.shape[0], -1, -1).contiguous())
    return a, b


@torch.no_grad()
def val_nll(a, b, w, target_batch, aa_batch):
    """NLL per mutation of ``w*a + (1-w)*b`` on one batch -> (sum, n).

    Goes through ``compute_probs`` so the wild-type masking and the renormalisation are
    the pipeline's, not a second implementation of them.
    """
    probs = compute_probs(w * a + (1.0 - w) * b, target_batch, aa_batch)
    norm = probs / probs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    loss = -(target_batch * torch.log(norm))
    return float(loss.sum()), float(target_batch.sum())


def select_weight(wxw, pssm_log, val_examples, grid, device, batch_size):
    """-> (best_w, [(w, nll)]) over the grid, by pooled val NLL per mutation."""
    totals = {w: [0.0, 0.0] for w in grid}
    for i in range(0, len(val_examples), batch_size):
        chunk = val_examples[i:i + batch_size]
        aa = torch.stack([e.aa_idx for e in chunk]).to(device)
        tg = torch.stack([e.mat for e in chunk]).to(device)
        a, b = blend_batch(wxw, pssm_log, aa)
        for w in grid:
            s, n = val_nll(a, b, w, tg, aa)
            totals[w][0] += s
            totals[w][1] += n
    curve = [(w, totals[w][0] / max(totals[w][1], 1.0)) for w in grid]
    best = min(curve, key=lambda t: t[1])[0]
    return best, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--maps-dir', required=True,
                    help='The score cache to write into. Must already describe this '
                         'query set: the manifest is checked, never overwritten.')
    ap.add_argument('--train-examples', action='append', required=True)
    ap.add_argument('--root-counts', required=True)
    ap.add_argument('--alphabet', default=LEGACY_AAS)
    ap.add_argument('--variant', default='BA.2')
    ap.add_argument('--n-splits', type=int, default=10)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--split-seed-base', type=int, default=0)
    ap.add_argument('--substitution-seed', type=int, default=14)
    ap.add_argument('--weight-grid', default=None,
                    help='Comma-separated weights on the where x which component. '
                         'Default: 0.00 to 1.00 in steps of 0.05.')
    ap.add_argument('--name-prefix', default='hybridWxwPssm')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--gate', action='store_true',
                    help='Also write the w=0 and w=1 endpoint maps, under '
                         '<prefix>Gate0 / <prefix>Gate1, so that the blend can be shown '
                         'to reproduce pssm_root and where_times_which exactly at its '
                         'endpoints rather than approximately.')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allowed_aas = args.alphabet
    E = len(allowed_aas)
    grid = ([float(x) for x in args.weight_grid.split(',')] if args.weight_grid
            else [round(0.05 * k, 2) for k in range(21)])
    print(f"Device: {device}")
    print(f"Alphabet: '{allowed_aas}' (E={E})")
    print(f"Weight grid: {grid[0]} .. {grid[-1]} in {len(grid)} steps")

    # ── The query set comes from the cache, not from a re-derivation ────────────
    # Reading the genotypes the maps were built over makes the new maps aligned to the
    # existing ones by construction. Re-loading the test corpus and filtering it again
    # would produce the same queries only if every step matched, and a silent reordering
    # has no array-level signature.
    man = sc.read_query_manifest(args.maps_dir)
    recorded = (man.get("args") or {}).get("alphabet")
    if recorded and recorded != allowed_aas:
        raise SystemExit(f"{args.maps_dir} was written over '{recorded}' but --alphabet "
                         f"is '{allowed_aas}'")
    eval_range = man["eval_range"]
    seq_len = man["seq_len"]
    aa_all, _tgt = sc.read_queries(args.maps_dir)
    aa_all = torch.from_numpy(aa_all.astype(np.int64))
    N, R = aa_all.shape[0], len(eval_range)
    print(f"Queries from the cache: {N} x {R} eval positions "
          f"(sha1 {man['query_sha1']})")

    # ── The two components ──────────────────────────────────────────────────────
    train_examples = []
    for path in args.train_examples:
        recs = load_examples(path, allowed_aas, args.variant)
        kept = [e for e in recs if e.is_training]
        print(f"  training pool: kept {len(kept)} of {len(recs)} from {path}")
        train_examples += kept
    if not train_examples:
        raise SystemExit("no training examples")
    if len(train_examples[0].aa_idx) != seq_len:
        raise SystemExit(f"examples are L={len(train_examples[0].aa_idx)}, the cache "
                         f"says {seq_len}")

    pssm = load_counts_matrix(args.root_counts, allowed_aas)
    pssm_log = torch.log1p(pssm.clamp(min=0)).to(device)

    rows = []
    jobs = [(f"{args.name_prefix}_s{i}", args.split_seed_base + i)
            for i in range(args.n_splits)]
    if args.gate:
        jobs += [(f"{args.name_prefix}Gate0", None), (f"{args.name_prefix}Gate1", None)]

    for name, split_seed in jobs:
        print(f"\n=== {name} ===", flush=True)
        if split_seed is None:                      # an endpoint gate
            w = 0.0 if name.endswith("Gate0") else 1.0
            wxw = fit_wxw(train_examples, seq_len, E, allowed_aas, device,
                          args.substitution_seed)
            n_tr, n_va, curve = len(train_examples), 0, []
            print(f"  endpoint gate at w={w}: must reproduce "
                  f"{'pssm_root' if w == 0 else 'where_times_which'} cell for cell")
        else:
            tr_by, va_by = split_examples({"src": train_examples}, "random",
                                          args.val_frac, split_seed)
            tr, va = tr_by["src"], va_by["src"]
            n_tr, n_va = len(tr), len(va)
            n_val_mut = sum(float(e.mat.sum()) for e in va)
            print(f"  split-seed {split_seed}: train {n_tr}, val {n_va} "
                  f"({n_val_mut:.0f} val mutations)")
            wxw = fit_wxw(tr, seq_len, E, allowed_aas, device, args.substitution_seed)
            w, curve = select_weight(wxw, pssm_log, va, grid, device, args.batch_size)
            best_nll = dict(curve)[w]
            print(f"  selected w = {w:.2f}  (val NLL {best_nll:.4f}; "
                  f"w=0 {dict(curve)[grid[0]]:.4f}, w=1 {dict(curve)[grid[-1]]:.4f})")
            if w in (grid[0], grid[-1]):
                print(f"  [warn] the selected weight is at the edge of the grid")

        writer = sc.MapWriter(args.maps_dir, name, N, R, E)
        try:
            for start in range(0, N, args.batch_size):
                stop = min(start + args.batch_size, N)
                aa = aa_all[start:stop].to(device)
                a, b = blend_batch(wxw, pssm_log, aa)
                blended = (w * a + (1.0 - w) * b)[:, eval_range, :]
                writer.write_batch(start, blended.detach().float().cpu().numpy())
                if (start // args.batch_size) % 50 == 0:
                    print(f"    query {start}/{N}", flush=True)
            side = writer.close(manifest_sha1=man["query_sha1"],
                                extra={"n_queries": N, "eval_range_len": R,
                                       "hybrid_weight": w,
                                       "n_train": n_tr, "n_val": n_va})
        except BaseException:
            writer.abort()
            raise
        print(f"  [map] {name}: w={w:.2f}, {side['n_nan']} NaN, "
              f"{side['n_neginf']} -inf cells", flush=True)
        rows.append(dict(model=name, split_seed=split_seed, weight=w,
                         n_train=n_tr, n_val=n_va,
                         curve=json.dumps([[a_, round(b_, 6)] for a_, b_ in curve])))
        del wxw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    path = os.path.join(args.output_dir, REPORT)
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    sel = [r["weight"] for r in rows if r["split_seed"] is not None]
    print(f"\n=== Wrote {path} ===")
    if sel:
        print(f"Selected weights on where x which: min {min(sel):.2f} "
              f"median {sorted(sel)[len(sel) // 2]:.2f} max {max(sel):.2f} "
              f"mean {sum(sel) / len(sel):.3f}")
        print(f"  (w = 1 is pure where x which, w = 0 is pure PSSM)")


if __name__ == "__main__":
    main()
