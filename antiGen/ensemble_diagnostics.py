#!/usr/bin/env python
"""How different are the members, and how much headroom is there? Read before sweeping.

NOTES s14's precedent is that this step decides which rules are worth searching at all:
there, the agreement table showed the model and `where x which` were genuinely different
predictors with a large per-target complementarity, and the oracle showed +0.096 of
headroom -- and then nine families of mixing rules failed to reach any of it. Measuring
the ceiling first is what made that a finding rather than a surprise.

What it reports, per candidate pool and target cell set:

* **pairwise Spearman** of the score maps, over cells both members may use, averaged over
  queries. The within-group blocks of the figure's five model groups are printed
  separately, because "the finetunes agree with each other more than with the baselines"
  is the structure a hierarchical rule would exploit.
* **top-q overlap**, as a share of the selection budget: two members can correlate well
  overall and still disagree about the decile, which is the only part the metric reads.
* **per-cell complementarity**, the 2x2 of NOTES s14: for each pair, the share of the
  metric caught by both, by one, by neither.
* **the per-cell oracle** -- picking the best member per cell, with the answer -- and the
  **greedy oracle-union curve**, which says how many members the headroom needs.
* **a self-agreement scale**, without which the Spearman numbers have no units. There is
  no clean one: all 178 published checkpoints are `--seed 42`. The honest substitutes, in
  order, are the three antiGen `valsel`/`fix32` seed pairs (a genuine spread, on a 90%
  training subsample), the six mt2021 rungs, and the 81 same-config antiGen duplicates --
  which share a seed and differ only by CUDA jitter, and therefore measure NOTHING. Which
  one is in use is printed with the table rather than left to be assumed.

The oracle numbers are CEILINGS computed with the answer and are not proposals. Quote
them as the headroom, never as a result.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os

import numpy as np
import torch

from antiGen import ensemble_sweep as es
from antiGen.background_discrimination import rank_fractions, top_q_weight


def spearman_and_overlap(a, b, q):
    """(mean Spearman, mean top-q overlap share) between two members' maps, per query.

    Restricted to cells BOTH members may use, so a member that abstains on gap/stop is
    not credited with disagreeing there. Mid-ranks within tied blocks, via the same
    searchsorted identity the rest of this arm uses.
    """
    from antiGen import ensemble_rules as er
    both = (~torch.isnan(a)) & (~torch.isnan(b))
    ra = er.pctile_ranks(torch.where(both, a, torch.full_like(a, float('nan'))), both)
    rb = er.pctile_ranks(torch.where(both, b, torch.full_like(b, float('nan'))), both)
    B = a.shape[0]
    rho, ov = [], []
    for i in range(B):
        m = both[i]
        if int(m.sum()) < 10:
            continue
        x, y = ra[i][m].double(), rb[i][m].double()
        x = x - x.mean(); y = y - y.mean()
        d = float((x.norm() * y.norm()))
        rho.append(float((x * y).sum() / d) if d > 0 else float('nan'))
        n = max(1, int(round(q * int(m.sum()))))
        ta = set(torch.topk(torch.where(m, a[i], torch.full_like(a[i], -float('inf')))
                            .reshape(-1), n).indices.tolist())
        tb = set(torch.topk(torch.where(m, b[i], torch.full_like(b[i], -float('inf')))
                            .reshape(-1), n).indices.tolist())
        ov.append(len(ta & tb) / n)
    return (float(np.nanmean(rho)) if rho else float('nan'),
            float(np.mean(ov)) if ov else float('nan'))


def per_cell_rates(sub, member, pool_name, q, device="cpu"):
    """(cell_hits, cell_events) for one member: the per-cell rate the oracle works on."""
    pmask = sub.pool_mask(pool_name)
    ch = ce = None
    for start, stop in sub.blocks():
        aa_b = sub.aa_eval[start:stop].to(device)
        tgt_b = sub.targets[start:stop].to(device)
        s = es._wt_masked(sub.raw(member, start, stop).to(device), aa_b, pmask)
        above, tie, n_valid = rank_fractions(s, aa_b, list(range(s.shape[1])))
        W = top_q_weight(above, tie, n_valid, q)
        h, e = (W * tgt_b).sum(dim=0), tgt_b.sum(dim=0)
        ch = h if ch is None else ch + h
        ce = e if ce is None else ce + e
    return ch.cpu(), ce.cpu()


def greedy_union(rates, cells, k_max=8):
    """Greedy oracle-union: add the member that raises the per-cell max most.

    A ceiling. NOTES s14 found a per-genotype oracle of this shape showing +0.096 that no
    pre-registered rule could reach, so the gap between this curve and the sweep's best
    is itself the result.
    """
    chosen, cur, out = [], np.zeros(len(cells)), []
    pool = sorted(rates)
    for _ in range(min(k_max, len(pool))):
        best = None
        for m in pool:
            if m in chosen:
                continue
            g = float(np.maximum(cur, rates[m]).mean())
            if best is None or g > best[0]:
                best = (g, m)
        g, m = best
        chosen.append(m)
        cur = np.maximum(cur, rates[m])
        out.append((m, g))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--substrate', choices=['backbone', 'list'], required=True)
    ap.add_argument('--background-dir'); ap.add_argument('--maps-dir')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--tree-json'); ap.add_argument('--ref-json')
    ap.add_argument('--segment', default='S')
    ap.add_argument('--cutoff', type=float, default=1.0)
    ap.add_argument('--variant', default=None)
    ap.add_argument('--mutation-list', action='append', required=True)
    ap.add_argument('--list', default=None)
    ap.add_argument('--members', default=None); ap.add_argument('--exclude', default=None)
    ap.add_argument('--qtile', type=float, default=0.1)
    ap.add_argument('--k-max', type=int, default=8)
    ap.add_argument('--select-position-disjoint', action='store_true')
    ap.add_argument('--query-block', type=int, default=2048)
    ap.add_argument('--spearman-queries', type=int, default=64,
                    help='Subsample of queries for the 435-pair agreement table, which '
                         'is O(K^2) and is a descriptive statistic rather than a reported '
                         'number. 0 disables it.')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    sub, label = es.substrate_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    rows_pair, rows_oracle = [], []

    for pool_name in es.POOLS:
        sets = sub.cellsets(pool_name, position_disjoint=args.select_position_disjoint)
        print(f"\n=== {pool_name} (q = {args.qtile}) ===", flush=True)
        rates, ch_ce = {}, {}
        for m in sub.members:
            ch, ce = per_cell_rates(sub, m, pool_name, args.qtile, device=args.device)
            ch_ce[m] = (ch, ce)
        for cs_name, cs_mask in sets.items():
            obs = None
            for m in sub.members:
                ch, ce = ch_ce[m]
                e = (ce * cs_mask).numpy()
                if obs is None:
                    obs = e > 0
                rates[m] = np.divide((ch * cs_mask).numpy()[obs], e[obs],
                                     out=np.zeros(int(obs.sum())), where=e[obs] > 0)
            cells = np.arange(int(obs.sum()))
            if len(cells) == 0:
                print(f"  {cs_name}: no observed cells"); continue
            singles = sorted(((float(v.mean()), k) for k, v in rates.items()), reverse=True)
            oracle = float(np.max(np.stack([rates[m] for m in sorted(rates)]), axis=0).mean())
            print(f"  {cs_name}: {len(cells)} observed cells | best single "
                  f"{singles[0][0]:.4f} ({singles[0][1]}) | per-cell ORACLE {oracle:.4f} "
                  f"(+{oracle - singles[0][0]:.4f})")
            gu = greedy_union(rates, cells, k_max=args.k_max)
            for i, (m, g) in enumerate(gu, 1):
                print(f"    k={i}: +{m[:48]:48s} union={g:.4f}")
                rows_oracle.append(dict(pool=pool_name, cellset=cs_name, k=i, member=m,
                                        union=g, n_cells=len(cells),
                                        best_single=singles[0][0], oracle=oracle))
            for s, m in singles:
                rows_oracle.append(dict(pool=pool_name, cellset=cs_name, k=0, member=m,
                                        union=s, n_cells=len(cells),
                                        best_single=singles[0][0], oracle=oracle))

        if args.spearman_queries:
            nq = min(args.spearman_queries, sub.n_queries)
            idx = np.linspace(0, sub.n_queries - 1, nq).astype(int)
            pm = sub.pool_mask(pool_name)
            maps = {}
            for m in sub.members:
                blk = torch.stack([sub.raw(m, int(i), int(i) + 1)[0] for i in idx])
                maps[m] = es._wt_masked(blk, sub.aa_eval[idx], pm)
            print(f"  pairwise agreement over {nq} queries "
                  f"({len(sub.members) * (len(sub.members) - 1) // 2} pairs)...", flush=True)
            for a, b in itertools.combinations(sorted(maps), 2):
                rho, ov = spearman_and_overlap(maps[a], maps[b], args.qtile)
                rows_pair.append(dict(pool=pool_name, a=a, b=b, spearman=rho, overlap=ov))
            del maps
            rr = [r['spearman'] for r in rows_pair if r['pool'] == pool_name]
            oo = [r['overlap'] for r in rows_pair if r['pool'] == pool_name]
            print(f"    Spearman: median {np.nanmedian(rr):.3f}, "
                  f"range [{np.nanmin(rr):.3f}, {np.nanmax(rr):.3f}]")
            print(f"    top-q overlap: median {np.median(oo):.3f}, "
                  f"range [{np.min(oo):.3f}, {np.max(oo):.3f}]")

    for name, rows, fields in (
            ("ensemble_pairs.csv", rows_pair, ["pool", "a", "b", "spearman", "overlap"]),
            ("ensemble_oracle.csv", rows_oracle,
             ["pool", "cellset", "k", "member", "union", "n_cells", "best_single", "oracle"])):
        if not rows:
            continue
        with open(os.path.join(args.output_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {os.path.join(args.output_dir, name)} ({len(rows)} rows)")
    with open(os.path.join(args.output_dir, "diagnostics_manifest.json"), "w") as f:
        json.dump(dict(substrate=args.substrate, members=sub.members, list=label,
                       qtile=args.qtile,
                       args=dict(sorted(vars(args).items()))), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
