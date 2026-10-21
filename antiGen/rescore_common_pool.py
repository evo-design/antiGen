#!/usr/bin/env python
"""Re-score the CoG-UK recall panels and the yield row in a COMMON candidate pool.

**The defect.** "The top decile" was a different size for every model. ``n_valid`` is each
model's own emission set, and across the 38 cached rungs it takes the values 3,801 /
3,804 / 3,819 / 4,020 / 4,221 at the BA.2 root -- a **10% spread**, almost all of it the
gap and stop columns some models emit and others do not. Two consequences, in opposite
directions:

* on the recall panels the top-q SET is ``q * n_valid`` cells, so a model with a larger
  pool gets more chances to catch a target;
* on the yield row ``q * n_valid`` is the DENOMINATOR, so a model with a smaller pool
  scores higher for the same catch. Measured: the nine-member mean of log-probs caught
  **98.0** listed cells to EVE full FT's **94.0** and reported 0.2322 against 0.2461.

Either way the comparison turns partly on a model's alphabet rather than on its
predictions, which is not what any of these panels claim to measure.

**The fix.** One pool, defined without reference to any model: the variant's own cell set
intersected with the 20 real amino-acid columns (:func:`eval_utils.real_substitution_mask`).
Every model then ranks the same cells and selects the same number of them. A cell inside
the pool that a model cannot score is **ranked last** rather than dropped, so one model's
missing measurements do not shrink the pool for everyone -- the convention
``background_contrast.common_percentiles`` already uses for the K-L row, which is why that
row needed no change here.

**It costs no target.** 0 of the 247 CoG-UK listed RBD cells are a gap or a stop, so no
model loses a cell it could have been credited for. The change is to the denominator and
to the size of the selected set, not to what counts as a hit.

**``--legacy-pool`` is the regression, and it runs first.** Under it this script must
reproduce the published CSVs cell for cell from the cached maps. Only once that holds does
the common-pool number mean anything: otherwise a difference could be this script rather
than the convention.

Output: ``list_macro_summary_commonpool.csv`` in ``list_macro.py``'s exact schema and
``list_variant_recall_commonpool.csv`` in ``list_variant_recall.py``'s.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from antiGen import datasets
from antiGen import ensemble_rules as er
from antiGen import ensemble_sweep as es
from antiGen import score_cache as sc
from antiGen.background_discrimination import rank_fractions, top_q_weight
from antiGen.eval_utils import RBD_RANGE, real_substitution_mask
from antiGen.list_macro import load_examples, load_mutation_list, load_pssms, summarize
from antiGen.list_variant_recall import curve

ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALLOWED_AAS)
R = len(RBD_RANGE)
MACRO_OUT = "list_macro_summary_commonpool.csv"
VARR_OUT = "list_variant_recall_commonpool.csv"


def build_variants(in_sub, seen_sub, label, common):
    """[(suffix, pool_mask, target_mask)] -- pool and target are True = KEEP.

    Mirrors ``list_macro.build_variants``, which states them as exclusion masks, and
    intersects every pool with ``common``. When ``common`` is all-True this is exactly
    the published definition, which is what ``--legacy-pool`` relies on.
    """
    allc = torch.ones(R, E, dtype=torch.bool)
    return [
        (f"list_{label}_fullpool",        allc & common,                 in_sub),
        (f"list_{label}",                 in_sub & common,               in_sub),
        (f"list_{label}_unseen_fullpool", (~seen_sub) & common,          in_sub & ~seen_sub),
        (f"list_{label}_unseen",          in_sub & ~seen_sub & common,   in_sub & ~seen_sub),
    ]


def load_rules(path):
    if not path:
        return []
    with open(path) as f:
        specs = json.load(f)
    if isinstance(specs, dict):
        specs = [specs]
    out = []
    for spec in specs:
        spec = dict(spec)
        spec.pop("family", None)
        spec.pop("figure_label", None)
        out.append(er.Rule(family="frozen",
                           **{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in spec.items()}))
    return out


def dense_targets(tgt_coo, n_queries):
    """COO -> a (N, R, E) sparse-backed lookup, as per-query index lists."""
    qidx, pos, aa, cnt = tgt_coo
    keep = np.isin(pos, np.asarray(RBD_RANGE))
    qidx, pos, aa, cnt = qidx[keep], pos[keep], aa[keep], cnt[keep]
    rowof = {p: i for i, p in enumerate(RBD_RANGE)}
    r = np.asarray([rowof[int(p)] for p in pos], dtype=np.int64)
    order = np.argsort(qidx, kind="stable")
    return qidx[order], r[order], aa[order].astype(np.int64), cnt[order].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--mutation-list", action="append", required=True)
    ap.add_argument("--rule-json", default=None)
    ap.add_argument("--models", default=None, help="comma-separated; default all cached")
    ap.add_argument("--qtiles", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--qtile-varr", type=float, default=0.1)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--variants", default=None,
                    help="Comma-separated variant suffixes to score, e.g. "
                         "'fullpool,unseen_fullpool'. Default: all four. Only the two "
                         "fullpool variants are DRAWN -- fig_coguk_list.R's `recall_long` "
                         "points all four of its metrics at `*_fullpool` -- so the "
                         "list-restricted pair costs half the runtime and reaches no "
                         "panel. Restricting is a runtime choice, not a metric one.")
    ap.add_argument("--legacy-pool", action="store_true",
                    help="each model's own emission set, i.e. the published convention")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    qtiles = [float(q) for q in args.qtiles.split(",") if q.strip()]

    variant = args.variant or datasets.default_variant("sc2")
    ex = load_examples(sorted(glob.glob(os.path.join(args.data_dir,
                                                     "examples_cutoff_*.txt")))[0],
                       ALLOWED_AAS, variant)
    seq_len = len(ex[0].aa_idx)
    train0 = [e for e in ex if e.is_training][0]
    root = [e for e in ex if e.is_testing_root][0]

    pssm_root, pssm_var = load_pssms(os.path.join(args.data_dir, "pssms"), variant, E)
    seen_sub = (pssm_root[RBD_RANGE, :] > 0)

    label, path = args.mutation_list[0].split("=", 1)
    cells, _ = load_mutation_list(path, ALLOWED_AAS, seq_len, label=label,
                                  root_seq="".join(ALLOWED_AAS[i]
                                                   for i in train0.aa_idx.tolist()))
    in_full = torch.zeros(seq_len, E, dtype=torch.bool)
    for p, a in cells:
        in_full[p, a] = True
    in_sub = in_full[RBD_RANGE, :]

    common = (torch.ones(R, E, dtype=torch.bool) if args.legacy_pool
              else real_substitution_mask(R, ALLOWED_AAS))
    variants = build_variants(in_sub, seen_sub, label, common)
    if args.variants:
        # Match on the short suffix ("fullpool", "unseen_fullpool", "unseen") or on the
        # full variant name. The bare list-restricted variant is "" / the list label.
        def short(name):
            return name[len(f"list_{label}"):].lstrip("_")
        want = {v.strip() for v in args.variants.split(",") if v.strip()}
        chosen = [v for v in variants if short(v[0]) in want or v[0] in want]
        if not chosen:
            raise SystemExit(
                f"--variants {sorted(want)} matched nothing; available short names: "
                f"{[short(v[0]) or '(bare)' for v in variants]}")
        variants = chosen
    mode = "LEGACY per-model pool" if args.legacy_pool else "COMMON pool"
    print(f"{mode}: {int(common.sum())} of {R * E} eval-range cells are candidates "
          f"before the per-variant restriction")
    for suffix, pool, tgt in variants:
        print(f"  {suffix:34s} pool {int(pool.sum()):5d} cells   targets {int(tgt.sum()):4d}")

    models = (args.models.split(",") if args.models else sc.list_models(args.maps_dir))
    rules = load_rules(args.rule_json)
    rule_members = sorted({m for r in rules for m in r.members})
    print(f"\n{len(models)} model(s), {len(rules)} rule(s)")

    aa_all, tgt_coo = sc.read_queries(args.maps_dir)
    N = aa_all.shape[0]
    tq, tr, ta, tc = dense_targets(tgt_coo, N)
    starts = np.searchsorted(tq, np.arange(N + 1))

    acc = {}            # name -> {(suffix, q): [cell_hits, cell_events]}

    def accumulate(name, s, aa_b, tgt_b):
        """One SINGLE model. A single map has no normalisation, so the candidate pool
        only has to reach ``rank_fractions``; there is no support for it to perturb."""
        a = acc.setdefault(name, {})
        for suffix, pool, tmask in variants:
            if args.legacy_pool:
                # The published convention restricts the pool by NaN-ing the excluded
                # cells before ranking -- exactly what cell_pass does with score_mask --
                # and then lets n_valid fall out of whatever the MODEL emits inside it.
                sv = s.clone()
                sv[:, ~pool] = float("nan")
                above, tie, nv = rank_fractions(sv, aa_b, list(range(R)))
                filled = None
            else:
                above, tie, nv, filled = rank_fractions(
                    s, aa_b, list(range(R)), pool_mask=pool, return_filled=True)
            t = tgt_b * tmask.unsqueeze(0).float()
            for q in qtiles:
                W = top_q_weight(above, tie, nv, q)
                if filled is not None and filled.any():
                    W = torch.where(filled, torch.zeros_like(W), W)
                h, e = (W * t).sum(dim=0), t.sum(dim=0)
                k = (suffix, q)
                if k not in a:
                    a[k] = [torch.zeros_like(h), torch.zeros_like(e)]
                a[k][0] += h
                a[k][1] += e

    for start in range(0, N, args.block):
        stop = min(start + args.block, N)
        aa_b = torch.from_numpy(aa_all[start:stop].astype(np.int64))[:, RBD_RANGE]
        tgt_b = torch.zeros(stop - start, R, E)
        lo, hi = starts[start], starts[stop]
        if hi > lo:
            tgt_b[torch.from_numpy((tq[lo:hi] - start).astype(np.int64)),
                  torch.from_numpy(tr[lo:hi]), torch.from_numpy(ta[lo:hi])] = \
                torch.from_numpy(tc[lo:hi])
        member_blocks = {}
        for m in sorted(set(models) | set(rule_members)):
            blk, _ = sc.load_block(args.maps_dir, m, start, stop, neginf_to_nan=True)
            t = torch.from_numpy(blk)
            if m in rule_members:
                member_blocks[m] = t
            if m in models:
                accumulate(m, t, aa_b, tgt_b)
        # RULES. The pool a member is normalised over must be the pool the metric ranks
        # it in -- `ensemble_sweep._wt_masked`'s docstring, and the reason the loop cannot
        # be hoisted out of the variant. Applying one rule per block and restricting the
        # pool only at ranking time normalises each member over its whole map instead,
        # which perturbs every rule whose normalisation depends on the support's
        # composition. Measured, when this code did that: up to 0.0498 against
        # `ensemble_report`, with `(logsoftmax, arith)` agreeing to 0.0000 because a
        # log-softmax over a larger support differs by a per-query scalar that cancels in
        # an arithmetic mean. The cross-job agreement check in `fig_coguk_list.R` is what
        # caught it.
        for suffix, pool, tmask in variants:
            excl = ~pool
            masked = {m: es._wt_masked(v, aa_b, excl) for m, v in member_blocks.items()}
            t = tgt_b * tmask.unsqueeze(0).float()
            for r in rules:
                name = f"ens_{r.rule_id}"
                a = acc.setdefault(name, {})
                # _wt_masked already restricted the pool, so no pool_mask here: that is
                # exactly ensemble_report.percell_for_rules' call.
                above, tie, nv = rank_fractions(er.apply(masked, r), aa_b,
                                                list(range(R)))
                for q in qtiles:
                    W = top_q_weight(above, tie, nv, q)
                    h, e = (W * t).sum(dim=0), t.sum(dim=0)
                    k = (suffix, q)
                    if k not in a:
                        a[k] = [torch.zeros_like(h), torch.zeros_like(e)]
                    a[k][0] += h
                    a[k][1] += e
            masked.clear()
        member_blocks.clear()
        print(f"  queries {stop}/{N}", flush=True)

    rows = []
    for name, a in acc.items():
        for suffix, q, macro, micro, n_cells, n_events in summarize(a):
            rows.append([name, suffix, f"{q:.2f}", f"{macro:.8f}", f"{micro:.8f}",
                         n_cells, f"{n_events:.0f}"])
    out = os.path.join(args.output_dir, MACRO_OUT)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "qtile", "macro", "micro", "n_cells", "n_events"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")

    # ── G-J, from the single variant-root query ──────────────────────────────
    want = root.aa_idx.numpy().astype(aa_all.dtype)
    hits = np.flatnonzero((aa_all == want[None, :]).all(axis=1))
    if len(hits) != 1:
        raise SystemExit(f"variant root matches {len(hits)} queries")
    qi = int(hits[0])
    wt_keep = torch.ones(R, E, dtype=torch.bool)
    wt_keep.scatter_(1, root.aa_idx[RBD_RANGE].unsqueeze(-1), False)
    pssm_var_sub = [pv[RBD_RANGE, :] for pv in pssm_var]
    thresholds = [100, 1] if variant == "BA.2" else [10, 1]
    target_sets = {}
    for subset, base in (("all", in_sub & wt_keep),
                         ("unseen", in_sub & wt_keep & ~seen_sub)):
        for thr in thresholds:
            target_sets[(subset, thr)] = [base & (pv >= thr) for pv in pssm_var_sub]

    # The query's own residue is NOT a candidate, so it leaves the pool rather than
    # being filled and ranked last inside it. Getting this wrong put n_valid at 4,020
    # instead of 3,819 -- a denominator 5% too large for every model alike, so the
    # comparison stayed fair while the level was wrong. `compute_hit_rate` applies its
    # wild-type mask AFTER the pool fill and so never had the problem; `top_q_weights`
    # receives the scores already wild-type-masked, and the fill resurrected them.
    varr_pool = None if args.legacy_pool else (common & wt_keep)
    vrows = []
    blocks = {}
    for m in sorted(set(models) | set(rule_members)):
        blk, _ = sc.load_block(args.maps_dir, m, qi, qi + 1, neginf_to_nan=True)
        blocks[m] = torch.from_numpy(blk)
    named = [(m, blocks[m][0]) for m in models]
    if rules:
        aa_root = torch.from_numpy(aa_all[qi:qi + 1].astype("int64"))[:, RBD_RANGE]
        excl = ~(common & wt_keep) if not args.legacy_pool else None
        masked_root = {m: es._wt_masked(v, aa_root, excl) for m, v in blocks.items()}
        named += [(f"ens_{r.rule_id}", er.apply(masked_root, r)[0]) for r in rules]
    for name, s2d in named:
        s = s2d.clone().float()
        s[~wt_keep] = float("nan")
        for (subset, thr), masks in target_sets.items():
            for i, (n_t, n_sel, val) in enumerate(curve(s, masks, args.qtile_varr,
                                                        pool_mask=varr_pool)):
                vrows.append([name, label, subset, thr, i, n_t, round(n_sel, 2), val])
    outv = os.path.join(args.output_dir, VARR_OUT)
    with open(outv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "list", "subset", "threshold", "step", "n_targets",
                    "n_selected", "value"])
        w.writerows(vrows)
    sel = sorted({r[6] for r in vrows})
    print(f"wrote {len(vrows)} rows -> {outv}")
    print(f"top-decile sizes across all models: {sel}"
          f"{'  <- ONE size, which was the point' if len(sel) == 1 else '  <- STILL VARIES'}")


if __name__ == "__main__":
    main()
