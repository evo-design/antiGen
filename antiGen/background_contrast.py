#!/usr/bin/env python
"""Does a model's score for a mutation move with the background it is asked about?

**The transpose of the recall panels.** Figure 2 and the CoG-UK list panels fix a query
and rank the mutations in it. This script fixes a *mutation* and ranks the *queries*:
among the backbone genotypes from which that mutation could have arisen, does the model
rank it higher in the ones it actually arose from? Because the cell is held fixed, every
per-mutation bias cancels -- a model that simply scores N501Y high everywhere gains
nothing -- and what is left is how much the prediction is adapted to the sequence it was
given. That is the quantity NOTES s32 left unmeasured when it found antiGen's deficit
"far sharper on the single BA.2 query than averaged over 17,978 queries".

**Hosts and controls.** For a cell ``c = (position, residue)``:

* a **host** is a backbone genotype with a post-cutoff child branch carrying ``c``;
* a **control** is a genotype that (i) does not already carry ``c``'s residue, so ``c``
  is a mutation it could make, (ii) has a node within ``--window`` years of a host node,
  and (iii) is neither an ancestor nor a descendant of any host node of that cell.

(ii) is there because the guide tree is recency-heavy -- a quarter of its post-cutoff
nodes are dated after 2025.7 -- so an unmatched contrast would partly measure "does this
model score mutations higher on later genotypes". (iii) is there because a node two
branches above where 346T arose is not a negative in any useful sense; excluding the
host's own lineage makes the comparison between parts of the tree that did and did not
host the mutation, rather than punishing a model for scoring it high across the clade
where it kept arising.

**The ruler, and why it is not the published one.** A cell's standing in a query is its
percentile: the share of a reference pool the model scores above it (lower = ranked
higher). The published recall metric takes that pool to be the eval range minus *this
query's own residues* -- which is right when the query is fixed and wrong here, because
the pool then differs from query to query and a derived genotype has had more of its
high-scoring cells masked away, lifting everything that is left. Measured on the pilot,
that is worth ~0.1 of AUC to predictors that cannot see the query at all: CoVFit 0.609,
pyr0 0.614, Omicron BA.1 bind 0.600, pssm_root 0.418, every one of them a fixed (L, E)
map that must sit at 0.5.

So the headline ranks each cell in a pool held **fixed across queries**: the eval-range
cells that no backbone genotype carries and that the model scores finitely in every
query, plus the target cell itself. Identical for every query by construction, so a
fixed map scores every query identically and lands at exactly 0.5 -- the null is then a
fact about the method rather than a hope. The published per-query convention is reported
beside it as ``_perquery`` rows, where it serves as the diagnostic it turned out to be.

**The statistic, and why it is not the AUC.** Three readings of the same contrast are
written, per cell:

    lift(c)   P(c in the query's top decile | host) - P(... | matched control)
    AUC(c)    mean over hosts h of mean over controls k of 1[pct_h < pct_k] + 0.5 ties
    delta(c)  mean control percentile - mean host percentile, in percentile points

The panel value is the unweighted mean over cells; chance is 0 for lift and delta, 0.5
for the AUC, and in all three the query-blind rungs sit on chance exactly.

**lift is the headline.** The AUC is scale-free, so it scores the sign of a
0.7-percentile-point nudge exactly as it scores a 28-point swing -- and on the pilot
those are the two regimes actually present: base antiGen's percentile for a cell moves
with the background about eight times as much as the EVE rungs' (median sd 0.054 against
0.0066). A model that shifts a cell's rank by less than a percentile point produces the
same top decile for every background in the tree; its direction can be right while its
forecast is, operationally, background-blind. lift combines sign and size, is in the
units the recall panels are already read in, and goes to zero for a model whose decile
does not move. Measured on the pilot, listed cells: antiGen +4.2 pp, antiGen as EVE full
FT +0.7 pp, EVE -0.3 pp, every fixed map exactly 0.0 -- where the AUC has them at 0.528,
0.602, 0.574 and 0.500. Both are reported; the disagreement is the finding, not a
nuisance.

**Cells a model cannot score.** A non-finite score means the cell is not a candidate for
that query: a gap/stop column the model does not emit, or a ``-inf`` sentinel (the codon
baseline's "not reachable by one nucleotide substitution from this query"). Those are
ranked last (percentile 1.0) rather than dropped, because "impossible from here" is a
statement about the background and dropping it would discard exactly the signal being
measured. ``--drop-nan`` reports the contrast with those pairs dropped instead.

**The null is measured, not assumed.** Every ``data/baselines/`` rung is a fixed
(L, E) map with no access to the query, so it should sit at 0.5 -- but not exactly,
because the candidate pool is masked per query (the query's own residues), which moves a
cell's percentile a little. Those rows are the empirical null band for the panel.

Usage:
    python antiGen/background_contrast.py \\
        --background-dir .../evals_sc2_background \\
        --data-dir /home/ivan.specht/antiGen-data/output1 \\
        --mutation-list coguk=coguk.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from antiGen.background_discrimination import rank_fractions
from antiGen.load import load_pssms
from antiGen.mutation_lists import load_mutation_list

SUMMARY = "background_summary.csv"
PERCELL = "background_percell.csv"
CELLSETS = "background_cellsets.csv"


def read_backbone(background_dir):
    with open(os.path.join(background_dir, "backbone_manifest.json")) as fh:
        manifest = json.load(fh)
    with open(os.path.join(background_dir, "backbone_queries.csv")) as fh:
        queries = list(csv.DictReader(fh))
    nodes = np.genfromtxt(os.path.join(background_dir, "backbone_nodes.csv"),
                          delimiter=",", names=True, dtype=None, encoding="utf-8")
    with open(os.path.join(background_dir, "backbone_hosts.csv")) as fh:
        hosts = list(csv.DictReader(fh))
    return manifest, queries, nodes, hosts


def build_contrast_sets(manifest, queries, nodes, hosts, window,
                        min_control_branches=0):
    """cell -> (host qidx list, control qidx list), under the matching rule."""
    eval_range = list(range(manifest["eval_range_start1"] - 1, manifest["eval_range_end1"]))
    pos_of = {p: i for i, p in enumerate(eval_range)}
    eval_seq = [q["eval_seq"] for q in queries]
    n_branches = [int(q["n_branches"]) for q in queries]
    q_nodes = {}
    for i in range(len(nodes["node"])):
        q = int(nodes["qidx"][i])
        if q >= 0:
            q_nodes.setdefault(q, []).append(i)
    node_time = nodes["time"]
    node_in, node_out = nodes["dfs_in"], nodes["dfs_out"]

    by_cell = {}
    for row in hosts:
        cell = (int(row["pos1"]), row["alt"])
        by_cell.setdefault(cell, []).append((int(row["node"]), int(row["qidx"])))

    out = {}
    for cell, entries in by_cell.items():
        host_nodes = [n for n, _q in entries]
        host_qs = sorted({q for _n, q in entries})
        hin, hout = node_in[host_nodes], node_out[host_nodes]
        htimes = node_time[host_nodes]
        col = pos_of[cell[0] - 1]
        controls = []
        for q in range(len(queries)):
            if q in host_qs:
                continue
            if eval_seq[q][col] == cell[1]:          # already carries it
                continue
            # Hosts are "bigger" nodes than controls by construction -- a genotype with
            # more post-cutoff child branches has more chances to host any given
            # mutation (median 28 branches against 4). A uniform "this background is
            # productive" response cancels in a WITHIN-query rank, which is why the
            # statistic is a percentile and not a raw score; this flag is the direct
            # check on that argument.
            if n_branches[q] < min_control_branches:
                continue
            # Related to a host ANYWHERE in the tree disqualifies the genotype, not
            # just at the node that happens to be time-matched: the model is shown the
            # sequence, so a genotype that also occurs inside a host's clade is not a
            # background that failed to host the mutation.
            qn = q_nodes[q]
            qin = node_in[qn]
            if np.any(qin[:, None] >= hin[None, :]) and np.any(
                    (qin[:, None] >= hin[None, :]) & (qin[:, None] < hout[None, :])):
                continue
            if np.any((qin[:, None] <= hin[None, :]) &
                      (hin[None, :] < node_out[qn][:, None])):
                continue
            if np.any(np.abs(node_time[qn][:, None] - htimes[None, :]) <= window):
                controls.append(q)
        out[cell] = (host_qs, controls, host_nodes)
    return out, eval_range, pos_of


def common_percentiles(scores, pool_mask, cells):
    """percentile of each listed cell in every query, against a query-independent pool.

    ``pool_mask`` is (R, E) and the same for every query, so the only thing that moves a
    cell's percentile between queries is the model's scores. The target cell is taken
    out of its own reference set and a non-finite score ranks last.
    """
    Q = scores.shape[0]
    pool_idx = np.flatnonzero(pool_mask.reshape(-1))
    flat = scores.reshape(Q, -1)
    pool_vals = flat[:, pool_idx]                              # (Q, n_pool)
    out = {}
    for (r, e) in cells:
        c_flat = r * scores.shape[2] + e
        s_c = flat[:, c_flat]
        keep = pool_idx != c_flat
        vals = pool_vals if keep.all() else pool_vals[:, keep]
        n = vals.shape[1]
        above = (vals > s_c[:, None]).sum(axis=1)
        ties = (vals == s_c[:, None]).sum(axis=1)
        pct = (above + 0.5 * ties) / float(n)
        out[(r, e)] = np.where(np.isfinite(s_c), pct, 1.0)
    return out


def perquery_percentiles(scores, eval_seq_idx, cells):
    """The published convention: the pool is the eval range minus THIS query's residues.

    Computed with ``rank_fractions``, the same function whose top-q weights reproduce
    ``compute_hit_rate`` cell for cell, so these rows are on the published ruler.
    """
    import torch
    above, tie, _nv = rank_fractions(torch.from_numpy(scores),
                                     torch.from_numpy(eval_seq_idx),
                                     list(range(scores.shape[1])))
    pct = (above + tie / 2.0).numpy()
    return {(r, e): np.where(np.isfinite(pct[:, r, e]), pct[:, r, e], 1.0)
            for (r, e) in cells}


def cell_stats(pct_cell, host_qs, control_qs, qtile, drop_nan=False):
    """(lift, auc, delta, n_pairs) for one cell, or None if it cannot be scored."""
    h = pct_cell[host_qs].astype(np.float64)
    k = pct_cell[control_qs].astype(np.float64)
    if drop_nan:
        h, k = h[h < 1.0], k[k < 1.0]
    if h.size == 0 or k.size == 0:
        return None
    cmp = (h[:, None] < k[None, :]).astype(np.float64)
    cmp += 0.5 * (h[:, None] == k[None, :])
    auc = float(cmp.mean(axis=1).mean())
    lift = float((h < qtile).mean() - (k < qtile).mean())
    return lift, auc, float(k.mean() - h.mean()), int(h.size * k.size)


def bootstrap_ci(values, n_boot, seed, alpha=0.05):
    if len(values) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = np.asarray(values)[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--background-dir', required=True)
    ap.add_argument('--data-dir', required=True, help='for pssms/root.csv (the de novo mask)')
    ap.add_argument('--output-dir', default=None, help='defaults to --background-dir')
    ap.add_argument('--mutation-list', action='append', default=None, help='LABEL=PATH')
    ap.add_argument('--variant', default='21L (BA.2)')
    ap.add_argument('--window', type=float, default=0.5,
                    help='years; a control must have a node this close to a host node')
    ap.add_argument('--min-controls', type=int, default=10)
    ap.add_argument('--min-control-branches', type=int, default=0,
                    help='require this many post-cutoff child branches of a control, to '
                         'check the result is not a host/control fecundity difference')
    ap.add_argument('--qtile', type=float, default=0.1,
                    help='the operating point the lift is read at (the recall panels\' decile)')
    ap.add_argument('--min-hosts', type=int, default=1)
    ap.add_argument('--drop-nan', action='store_true',
                    help='drop non-candidate pairs instead of ranking them last')
    ap.add_argument('--legacy-pool', action='store_true',
                    help="the published per-model pool: keep gap/stop and cut each "
                         "model's own non-finite cells. The regression for the common "
                         "pool -- under it this must reproduce the published CSV.")
    ap.add_argument('--skip-perquery', action='store_true',
                    help="do not also report the published per-query pool convention")
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out_dir = args.output_dir or args.background_dir
    os.makedirs(out_dir, exist_ok=True)

    manifest, queries, nodes, hosts = read_backbone(args.background_dir)
    allowed_aas = manifest["allowed_aas"]
    E = len(allowed_aas)
    print(f"{len(queries)} query genotypes, {len(hosts)} host rows, "
          f"eval range {manifest['eval_range_start1']}-{manifest['eval_range_end1']}")

    sets, eval_range, pos_of = build_contrast_sets(manifest, queries, nodes, hosts,
                                                   args.window,
                                                   args.min_control_branches)
    seq_len = manifest["seq_len"]
    pssm_root, _ = load_pssms(os.path.join(args.data_dir, "pssms"), args.variant, E)
    seen = (pssm_root > 0).numpy()

    # ── Cell sets ───────────────────────────────────────────────────────────
    lists = {}
    for item in (args.mutation_list or []):
        label, path = item.split('=', 1)
        cells, report = load_mutation_list(path, allowed_aas, seq_len, label=label)
        lists[label] = {(p + 1, allowed_aas[a]) for p, a in cells}
        print(f"  list {label}: {report['n_cells']} cells")

    usable = {}
    for cell, (hq, cq, _hn) in sets.items():
        if len(hq) >= args.min_hosts and len(cq) >= args.min_controls:
            usable[cell] = (hq, cq)
    print(f"{len(sets)} cells hosted; {len(usable)} with >={args.min_hosts} host(s) and "
          f">={args.min_controls} controls")

    def is_denovo(cell):
        return not seen[cell[0] - 1, allowed_aas.index(cell[1])]

    def no_indel(cell):
        return cell[1] not in '-*'

    cellsets = {}
    for label, members in lists.items():
        cellsets[f"{label}_all"] = [c for c in usable if c in members]
        cellsets[f"{label}_unseen"] = [c for c in usable if c in members and is_denovo(c)]
    cellsets["rbd_all"] = [c for c in usable if no_indel(c)]
    cellsets["rbd_unseen"] = [c for c in usable if no_indel(c) and is_denovo(c)]
    for name, cells in cellsets.items():
        print(f"  cellset {name}: {len(cells)} cells")

    with open(os.path.join(out_dir, CELLSETS), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cellset", "pos1", "alt", "n_hosts", "n_controls", "de_novo"])
        for name, cells in cellsets.items():
            for c in sorted(cells):
                w.writerow([name, c[0], c[1], len(usable[c][0]), len(usable[c][1]),
                            int(is_denovo(c))])

    # ── The query-independent AND model-independent pool ────────────────────
    # A cell is in it when no backbone genotype carries that residue, so no query is
    # ranking one of its own residues, and the pool is literally the same set for every
    # query.
    #
    # It is now also the same set for every MODEL. It used to carry a per-model
    # finiteness cut, which left the pool at 4,134 cells for antiGen, the PSSMs, pyr0 and
    # CoVFit, 3,933 for ESM-2, 3,734 for EVE and Tranception and 3,696 for BLOSUM62 -- an
    # 11% spread, and 400 of antiGen's were gap or stop columns. That did not bias `lift`
    # the way it biased the recall panels, since lift is a difference of two probabilities
    # for the SAME model and chance is 0 whatever the pool size. It did something subtler:
    # a model that emits gap and stop ranks them HIGH (measured: 39.8% of antiGen's top
    # decile at the BA.2 root), so a tenth of its reference pool was cells it scored above
    # most real ones, pushing every real cell down in percentile and compressing its lift
    # toward zero. Excluding them for everyone removes that, and makes the whole figure
    # one convention (NOTES s51).
    #
    # A pool cell a model cannot score is ranked LAST rather than dropped -- it is filled
    # with -inf below, so it never counts as above a finite target score while keeping the
    # denominator identical across models. That is the same treatment `compute_hit_rate`
    # and `rank_fractions` now give an unscorable in-pool cell.
    R, Ecols = len(eval_range), E
    carried = np.zeros((R, Ecols), dtype=bool)
    for q in queries:
        for r, ch in enumerate(q["eval_seq"]):
            carried[r, allowed_aas.index(ch)] = True
    eval_seq_idx = np.asarray(
        [[allowed_aas.index(ch) for ch in q["eval_seq"]] for q in queries], dtype=np.int64)
    real_col = np.asarray([a not in "-*" for a in allowed_aas], dtype=bool)
    common_pool = (~carried) if args.legacy_pool else ((~carried) & real_col[None, :])
    print(f"pool: {int((~carried).sum())} of {R * Ecols} eval-range cells are carried by "
          f"no backbone genotype")
    print(f"      {'LEGACY per-model pool' if args.legacy_pool else 'COMMON pool'}: "
          f"{int(common_pool.sum())} cells"
          f"{'' if args.legacy_pool else ' (gap and stop columns excluded for every model)'}")

    target_cells = sorted({(pos_of[c[0] - 1], allowed_aas.index(c[1])) for c in usable})

    # ── Models ──────────────────────────────────────────────────────────────
    paths = sorted(glob.glob(os.path.join(args.background_dir, "scores_*.npz")))
    if not paths:
        raise SystemExit(f"no scores_*.npz in {args.background_dir}")
    summary_rows, percell_rows = [], []
    for path in paths:
        model = os.path.basename(path)[len("scores_"):-len(".npz")]
        with np.load(path) as z:
            scores = z["scores"].astype(np.float32)
        if args.legacy_pool:
            pool_mask = common_pool & np.isfinite(scores).all(axis=0)
        else:
            # Model-independent by construction. An unscorable cell inside it is filled
            # with -inf so it ranks last without leaving the denominator.
            pool_mask = common_pool
            bad = pool_mask[None, :, :] & ~np.isfinite(scores)
            if bad.any():
                scores = scores.copy()
                scores[bad] = -np.inf
        if pool_mask.sum() < 100:
            print(f"  [skip] {model}: only {int(pool_mask.sum())} pool cells")
            continue
        conventions = {"": common_percentiles(scores, pool_mask, target_cells)}
        if not args.skip_perquery:
            conventions["_perquery"] = perquery_percentiles(scores, eval_seq_idx,
                                                            target_cells)
        shown = {}
        for suffix, pcts in conventions.items():
            per_cell = {}
            for cell, (hq, cq) in usable.items():
                key = (pos_of[cell[0] - 1], allowed_aas.index(cell[1]))
                got = cell_stats(pcts[key], hq, cq, args.qtile,
                                 drop_nan=args.drop_nan)
                if got is not None:
                    per_cell[cell] = got
            for name, cells in cellsets.items():
                vals = [per_cell[c] for c in cells if c in per_cell]
                if not vals:
                    continue
                lifts = [v[0] for v in vals]
                aucs = [v[1] for v in vals]
                deltas = [v[2] for v in vals]
                llo, lhi = bootstrap_ci(lifts, args.n_boot, args.seed)
                alo, ahi = bootstrap_ci(aucs, args.n_boot, args.seed)
                summary_rows.append([model, name + suffix, f"{np.mean(lifts):.6f}",
                                     f"{llo:.6f}", f"{lhi:.6f}", f"{np.mean(aucs):.6f}",
                                     f"{alo:.6f}", f"{ahi:.6f}",
                                     f"{np.mean(deltas):.6f}", len(lifts),
                                     sum(v[3] for v in vals)])
                if suffix == "":
                    shown[name] = f"{100 * np.mean(lifts):+.2f}"
            if suffix == "":
                for cell, (lift, auc, delta, n_pairs) in sorted(per_cell.items()):
                    percell_rows.append([model, cell[0], cell[1], len(usable[cell][0]),
                                         len(usable[cell][1]), f"{lift:.6f}",
                                         f"{auc:.6f}", f"{delta:.6f}", n_pairs,
                                         int(is_denovo(cell))])
        print(f"  {model:45s} lift(pp) " +
              "  ".join(f"{k}={v}" for k, v in sorted(shown.items())), flush=True)
        del scores

    with open(os.path.join(out_dir, SUMMARY), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "cellset", "lift", "lift_lo", "lift_hi", "auc",
                    "auc_lo", "auc_hi", "delta_pct", "n_cells", "n_pairs"])
        w.writerows(summary_rows)
    with open(os.path.join(out_dir, PERCELL), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pos1", "alt", "n_hosts", "n_controls", "lift", "auc",
                    "delta_pct", "n_pairs", "de_novo"])
        w.writerows(percell_rows)
    print(f"\nWrote {os.path.join(out_dir, SUMMARY)} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
