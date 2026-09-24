#!/usr/bin/env python
"""Huddleston's fitness metrics, recomputed honestly on our backbone tree.

The comparison against Huddleston et al. 2020 ran on *their* panel, which stops in 2019.
Neither that paper nor its 2025 successor evaluates anything after October 2019, so the
COVID bottleneck and the rebound after it are untouched by this literature. Our backbone
tree runs to 2026, and this module puts their models on it.

Everything their sequence-only models need is derivable from the tree we already have --
1,683 dated tips, 1,624 internal nodes with inferred dates, amino-acid mutations on every
branch -- with one exception and one hazard.

**The exception is `cTiter`.** HI antigenic novelty needs titer tables. The claim that
these are simply "not public" was wrong and is corrected here: Shi et al. 2025 (VaxSeer)
release their whole extraction of the Crick Worldwide Influenza Centre reports --
`vaxseer/data/antigenicity/hi_processed/a_h3n2_pairs.csv`, 63,299 virus-vaccine pairs over
6,981 distinct viruses. The real obstacle is coverage of *this panel*: matching our 1,683
tips by strain name finds 116, which is 6-9 titered strains in each season from 2015 to
2022 and **zero** from 2023 on (their set holds 8 distinct 2023 strains and nothing later,
so that gap is real, not a name-matching artefact). Three of the eight scored origins
therefore have no titer data at all, and the other five have far too few titered strains to
fit the Neher et al. 2016 antigenic-tree model `cTiter` is read off. Their only model that
survived 2015-2019 out of sample is therefore not comparable here, and the comparison is
sequence-only on both sides. That is a limitation of the comparison, not a result.

**The hazard is that the tree already carries their annotations.** `lbi`, `ep`, `ne_star`
and `cTiter` sit on every node of `tree.json` -- computed by Nextstrain on the build dated
2026-04-02, i.e. with the whole record in hand. Reading them would leak the future into
every forecast, which is the failure their own `oracle_x` demonstrates. So LBI is
recomputed here from scratch on the tree pruned to tips dated at or before each horizon,
and the shipped values are never touched.

Metrics implemented, following their Methods:

  lbi                      Neher et al. 2014's local branching index, up/down polarizer
                           messages with tau = 0.3 over a time tree, tips within the last
                           `window` = 0.5 years counted as alive. Recomputed per horizon
                           on the censored tree.
  ne_star                  mutational load: amino-acid substitutions at Luksza-Lassig
                           non-epitope sites accumulated since the tip's ancestor in the
                           previous season, read off the branch mutations.
  ep_x                     epitope antigenic novelty: the Luksza-Lassig cross-immunity
                           sum over previously circulating strains, exp(-D_ep / D0) with
                           D0 = 14 and D_ep the epitope-site distance.
  distance_from_consensus  substitutions from the current population's consensus. Their
                           most *stable* sequence-only metric on the 2020 panel.

Fitted as they fit them -- `xhat_v` proportional to `x_v exp(f_v)`, `f_v` linear in the
standardised metrics, coefficients minimising the earth-mover's distance over a rolling
training window under an L1 penalty of 0.1, by Nelder-Mead -- and refitted at every
origin, which the 2026-09-22 entry established as the fairer protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy.optimize import minimize

from antiGen.clade_cluster import season_of
from antiGen.season_emd import emd
from antiGen.season_momentum import (H3N2_SEGMENTS, Task, TipPanel,
                                     consensus_distance, cost_matrix)

L1_LAMBDA = 0.1          # their config/config.json, fitness_model.l1_lambda
LBI_TAU, LBI_WINDOW = 0.3, 0.5          # their config, lbi.tau and lbi.window
D0 = 14.0                # Luksza and Lassig's cross-immunity decay constant
MAX_YEARS_BACK = 5.0     # their max_years_for_distances


# ── the tree, with dates and mutations ──────────────────────────────────────

class Node:
    __slots__ = ("name", "date", "parent", "kids", "muts", "seq", "is_tip", "alive",
                 "up", "down", "lbi")

    def __init__(self, name, date, parent):
        self.name, self.date, self.parent = name, date, parent
        self.kids, self.muts, self.seq = [], [], None
        self.is_tip = False
        self.alive = False
        self.up = self.down = self.lbi = 0.0


def load_tree(tree_json, ref_json, segments=H3N2_SEGMENTS):
    """Every node with its inferred date, its branch's amino-acid mutations, and the
    reconstructed sequence implied by the path from the root.

    Positions in `branch_attrs.mutations` are 1-indexed within their segment, so they are
    shifted by the segment's offset in the concatenation -- the same convention
    `season_momentum.load_tips` uses, and the reconstructed tip sequences agree with it.
    """
    ref = json.load(open(ref_json))
    offset, base = {}, []
    for s in segments:
        offset[s] = sum(len(ref[x]) for x in segments[:segments.index(s)])
        base.append(ref[s])
    base = list("".join(base))
    raw = json.load(open(tree_json))
    root_raw = raw["tree"] if "tree" in raw else raw
    nodes = []

    def walk(rn, parent, seq):
        nd = (rn.get("node_attrs") or {}).get("num_date") or {}
        node = Node(rn.get("name"), nd.get("value"), parent)
        muts = (rn.get("branch_attrs") or {}).get("mutations") or {}
        undo = []
        for seg, lst in muts.items():
            if seg not in offset:
                continue
            for m in lst:
                pos = int(m[1:-1]) - 1 + offset[seg]
                node.muts.append((pos, m[0], m[-1]))
                undo.append((pos, seq[pos]))
                seq[pos] = m[-1]
        node.seq = "".join(seq)
        nodes.append(node)
        kids = rn.get("children")
        if kids:
            for k in kids:
                node.kids.append(walk(k, node, seq))
        else:
            node.is_tip = True
        for pos, prev in reversed(undo):
            seq[pos] = prev
        return node

    root = walk(root_raw, None, base)
    # an internal node with no inferred date cannot contribute a branch length; inherit
    # the parent's date so the branch is zero-length rather than undefined
    for n in nodes:
        if n.date is None:
            n.date = n.parent.date if n.parent is not None else 0.0
    return root, nodes


def branch_length(n):
    return 0.0 if n.parent is None else max(0.0, float(n.date) - float(n.parent.date))


# ── local branching index, recomputed per horizon on the censored tree ──────

def local_branching_index(nodes, horizon, tau=LBI_TAU, window=LBI_WINDOW):
    """Neher et al. 2014's LBI, over the tree restricted to tips dated <= `horizon`.

    Censoring is what makes this honest: the values Nextstrain shipped on `tree.json`
    were computed on the 2026 build and know the whole record. Here a node is *alive*
    only if it is a tip inside `(horizon - window, horizon]`, and only branches whose
    child is dated at or before the horizon carry messages, so nothing after the horizon
    can raise anyone's index.
    """
    keep = [n for n in nodes if n.date is not None and float(n.date) <= horizon + 1e-9]
    kept = set(id(n) for n in keep)
    for n in keep:
        n.alive = bool(n.is_tip and float(n.date) > horizon - window)
        n.up = n.down = n.lbi = 0.0
    post = sorted(keep, key=lambda n: -float(n.date))
    for n in post:                                   # up messages, children before parents
        s = sum(c.up for c in n.kids if id(c) in kept)
        bl = branch_length(n) / tau
        n.up = s * np.exp(-bl) + (tau * (1 - np.exp(-bl)) if n.alive else 0.0)
    pre = sorted(keep, key=lambda n: float(n.date))
    for n in pre:                                    # down messages, parents before children
        ks = [c for c in n.kids if id(c) in kept]
        tot = sum(c.up for c in ks)
        for c in ks:
            bl = branch_length(c) / tau
            c.down = (n.down + tot - c.up) * np.exp(-bl) + (
                tau * (1 - np.exp(-bl)) if c.alive else 0.0)
    out = {}
    for n in keep:
        n.lbi = n.down + sum(c.up for c in n.kids if id(c) in kept)
        if n.is_tip:
            out[id(n)] = n.lbi
    return out


# ── the sequence-based metrics ──────────────────────────────────────────────

def site_mask(path, segments=H3N2_SEGMENTS, ref_json=None):
    """A boolean mask over the concatenated protein from one of their distance maps."""
    ref = json.load(open(ref_json))
    offset = {s: sum(len(ref[x]) for x in segments[:segments.index(s)]) for s in segments}
    total = sum(len(ref[s]) for s in segments)
    m = np.zeros(total, dtype=bool)
    spec = json.load(open(path))["map"]
    for seg, sites in spec.items():
        if seg not in offset:
            continue
        for p in sites:
            i = int(p) - 1 + offset[seg]
            if 0 <= i < total:
                m[i] = True
    return m


def mutational_load(tip, season, nonepi):
    """`ne_star`: non-epitope substitutions since the tip's ancestor in the prior season.

    Their definition is "the number of putatively deleterious mutations accumulated since
    their ancestor in the previous season", which needs the tree: walk from the tip up to
    the first node dated at or before the end of the previous season and count the
    non-epitope amino-acid changes on the way.

    A tip that itself predates the cutoff scores zero -- nothing has accumulated since.
    That is the same behaviour as their `augur distance --compare-to ancestor`, where a
    tip older than the reference date is its own ancestor, and it makes the metric a
    function of a strain's *recent ancestry* rather than of its genotype alone.
    """
    cutoff = season - 1 + 0.25                       # 1 April of the previous season
    n, count = tip, 0
    while n.parent is not None and float(n.date) > cutoff:
        count += sum(1 for pos, a, b in n.muts if nonepi[pos] and a != b)
        n = n.parent
    return float(count)


def epitope_cross_immunity(seq_codes, sup, past_codes, past_w, epi_idx):
    """`ep_x`: sum over previously circulating strains of `w_j exp(-D_ep(i,j) / D0)`.

    `D_ep` counts differences at the epitope sites only. Higher means the strain looks
    *less* like what the population has already seen, so the metric is negated to match
    their sign convention (their `ep_x` is the negative of the cross-immunity burden).
    """
    A = seq_codes[sup][:, epi_idx]
    out = np.zeros(len(sup))
    step = max(1, 2_000_000 // max(len(past_codes), 1))
    P = past_codes[:, epi_idx]
    for a in range(0, len(sup), step):
        b = min(a + step, len(sup))
        D = (A[a:b, None, :] != P[None, :, :]).sum(2)
        out[a:b] = (np.exp(-D / D0) * past_w).sum(axis=1)
    return -out


# ── their model, fitted the way they fit it ─────────────────────────────────

def standardize(P, w):
    """Centre on the frequency-weighted mean, scale by the frequency-weighted sd --
    exactly what `DistanceExponentialGrowthModel.predict` does within a timepoint."""
    P = np.asarray(P, dtype=float).copy()
    mu = np.average(P, weights=w, axis=0)
    sd = np.sqrt(np.average((P - mu) ** 2, weights=w, axis=0))
    nz = np.where(sd > 0)[0]
    P[:, nz] = (P[:, nz] - mu[nz]) / sd[nz]
    return P


def project(beta, P, x0):
    f = np.asarray(P, float) @ np.asarray(beta, float)
    p = x0 * np.exp(f - f.max())
    s = p.sum()
    return p / s if s > 0 else x0.copy()


def fit_their_model(train, l1=L1_LAMBDA, seed=0):
    """Coefficients minimising mean EMD over the training transitions, plus an L1 term.

    Their optimiser, not ours: Nelder-Mead from a random start in [-0.5, 0.5], which is
    what `fit_model.py` does and is the sensible choice for a piecewise-linear objective
    in one to three dimensions. Our own L-BFGS-with-Danskin route is for 255 coefficients
    and would be the wrong tool here.
    """
    if not train:
        return None
    d = train[0][0].shape[1]
    rng = np.random.default_rng(seed)

    def obj(beta):
        tot = 0.0
        for P, x0, tgt_w, M in train:
            tot += emd(project(beta, P, x0), tgt_w, M)[0]
        return tot / len(train) + l1 * float(np.abs(beta).sum())

    res = minimize(obj, rng.random(d) - 0.5, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-4})
    return res.x


# ── the panel, with their metrics attached ──────────────────────────────────

class BackbonePanel:
    """Our backbone panel, plus Huddleston's metrics computed per season and genotype."""

    def __init__(self, tree_json, ref_json, maps_dir, min_allele_freq=0.005,
                 lookback=5.0, verbose=True):
        self.root, self.nodes = load_tree(tree_json, ref_json)
        tips = [(float(n.date), n.seq) for n in self.nodes
                if n.is_tip and n.date is not None]
        self.panel = TipPanel(sorted(tips), min_allele_freq=min_allele_freq,
                              lookback=lookback)
        self.tip_nodes = sorted([n for n in self.nodes if n.is_tip and n.date is not None],
                                key=lambda n: (float(n.date), n.seq))
        self.nonepi = site_mask(os.path.join(maps_dir, "luksza_nonepitope.json"),
                                ref_json=ref_json)
        epi = site_mask(os.path.join(maps_dir, "luksza.json"), ref_json=ref_json)
        # epitope sites, expressed in the panel's varying-site coordinates
        self.epi_idx = np.array([j for j, p in enumerate(self.panel.sites) if epi[p]],
                                dtype=int)
        self._cache = {}
        if verbose:
            print(f"backbone: {len(tips)} tips, {len(self.panel.genotypes)} genotypes, "
                  f"{self.nonepi.sum()} non-epitope and {epi.sum()} epitope sites, "
                  f"{len(self.epi_idx)} epitope sites varying in the panel")

    def metrics(self, task):
        """Their four sequence-only metrics, per support genotype, at one horizon."""
        key = float(task.horizon)
        if key in self._cache:
            return self._cache[key]
        p = self.panel
        sup = task.sup
        w = task.baseline("season")
        w = w if w.sum() > 0 else task.level(1.0 / 12)

        lbi_by_node = local_branching_index(self.nodes, task.horizon)
        # collapse per-tip metrics onto genotypes, weighted by how many tips carry each
        lbi = np.zeros(len(sup)); nes = np.zeros(len(sup)); cnt = np.zeros(len(sup))
        pos = {int(g): k for k, g in enumerate(sup)}
        for i, n in enumerate(self.tip_nodes):
            if float(n.date) > task.horizon or float(n.date) <= task.horizon - p.lookback:
                continue
            g = int(p.tip_gid[i])
            if g not in pos:
                continue
            k = pos[g]
            lbi[k] += lbi_by_node.get(id(n), 0.0)
            nes[k] += mutational_load(n, task.season, self.nonepi)
            cnt[k] += 1
        nz = cnt > 0
        lbi[nz] /= cnt[nz]; nes[nz] /= cnt[nz]

        # the previously circulating population the cross-immunity term sums over
        m = ((p.dates <= task.horizon) & (p.dates > task.horizon - MAX_YEARS_BACK))
        pg, pc = np.unique(p.tip_gid[m], return_counts=True)
        out = dict(
            lbi=lbi, ne_star=nes,
            ep_x=epitope_cross_immunity(p.codes, sup, p.codes[pg], pc / pc.sum(),
                                        self.epi_idx),
            distance_from_consensus=consensus_distance(p.codes, sup, w))
        self._cache[key] = out
        return out


# ── the roll ────────────────────────────────────────────────────────────────

THEIR_MODELS = {
    "ne_star-lbi": ("ne_star", "lbi"),          # their designated best sequence-only
    "lbi": ("lbi",),
    "ne_star": ("ne_star",),
    "ep_x": ("ep_x",),
    "distance_from_consensus": ("distance_from_consensus",),
    "ep_x-ne_star": ("ep_x", "ne_star"),
}


def rolling(bp, seasons=None, drop_targets=(2020,), min_seqs=30, window=6,
            min_train=2, verbose=True):
    """Score their models season by season, refitting coefficients at every origin.

    The training window is their six years, which on annual seasons is six transitions,
    and only transitions whose target season had closed by the origin are usable -- the
    same rule the panel round settled on. Their coefficients are refitted at each origin
    rather than frozen, which the 2026-09-22 entry established as the fairer protocol.
    """
    p = bp.panel
    counts = {s: int((p.season == s).sum()) for s in p.seasons}
    usable = [s for s in p.seasons if counts.get(s, 0) >= min_seqs]
    order = [s for s in usable if (s + 1) in usable and (s + 1) not in drop_targets]
    if seasons:
        order = [s for s in order if s in set(seasons) or True]
    tasks = {s: Task(p, s) for s in order}
    rows = []
    for i, s in enumerate(order):
        if i < min_train:
            continue
        task = tasks[s]
        train_idx = order[max(0, i - window):i]
        row = dict(season=int(s), next=int(s + 1), n_support=len(task.sup),
                   n_target=len(task.tgt_g), floor=task.floor,
                   emd_season=task.score(task.baseline("season")))
        met = {j: bp.metrics(tasks[j]) for j in train_idx + [s]}
        for name, preds in THEIR_MODELS.items():
            train = []
            for j in train_idx:
                t = tasks[j]
                x0 = t.baseline("season")
                if x0.sum() <= 0:
                    continue
                P = standardize(np.column_stack([met[j][q] for q in preds]), x0)
                train.append((P, x0, t.tgt_w, t.M))
            beta = fit_their_model(train)
            x0 = task.baseline("season")
            P = standardize(np.column_stack([met[s][q] for q in preds]), x0)
            w = x0 if beta is None else project(beta, P, x0)
            row["emd_" + name] = task.score(w)
            row["beta_" + name] = None if beta is None else [float(b) for b in beta]
        # the Barrat-Charlaix single-strain baselines, as point masses
        x0 = task.baseline("season")
        cons = consensus_distance(p.codes, task.sup, x0)
        row["emd_point_consensus"] = task.score(
            np.eye(len(task.sup))[int(np.argmin(cons))])
        lb = met[s]["lbi"]
        row["emd_point_toplbi"] = task.score(np.eye(len(task.sup))[int(np.argmax(lb))])
        per = task.tgt_w @ task.M.T
        row["d_best"] = float(per.min()); row["d_mean"] = float(x0 @ per)
        row["d_consensus"] = float(per[int(np.argmin(cons))])
        row["d_toplbi"] = float(per[int(np.argmax(lb))])
        rows.append(row)
        if verbose:
            print(f"  {s}->{s+1} floor {task.floor:6.3f} | season {row['emd_season']:6.3f} "
                  + "  ".join(f"{n} {row['emd_' + n]:6.3f}" for n in THEIR_MODELS),
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.path.expanduser("~/h3n2/tree.json"))
    ap.add_argument("--ref", default=os.path.expanduser("~/h3n2/ref.json"))
    ap.add_argument("--maps", default=os.path.expanduser(
        "~/flu-forecasting/config/distance_maps/h3n2/ha"))
    ap.add_argument("--min-allele-freq", type=float, default=0.005)
    ap.add_argument("--lookback", type=float, default=5.0)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--min-season-seqs", type=int, default=30)
    ap.add_argument("--drop-targets", default="2020")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    bp = BackbonePanel(a.tree, a.ref, a.maps, a.min_allele_freq, a.lookback)
    drop = tuple(int(x) for x in a.drop_targets.split(",") if x)
    rows = rolling(bp, drop_targets=drop, min_seqs=a.min_season_seqs, window=a.window)
    nai = np.array([r["emd_season"] for r in rows])
    print(f"\n{'model':26}{'mean EMD':>10}{'vs persistence':>16}{'wins':>8}")
    for name in THEIR_MODELS:
        v = np.array([r["emd_" + name] for r in rows])
        print(f"{'theirs: ' + name:26}{v.mean():10.3f}"
              f"{100 * np.mean((nai - v) / nai):+15.2f}%{int((v < nai).sum()):5d}/{len(v)}")
    for k, lab in (("emd_point_consensus", "consensus (point mass)"),
                   ("emd_point_toplbi", "top-LBI strain (point mass)")):
        v = np.array([r[k] for r in rows])
        print(f"{lab:26}{v.mean():10.3f}"
              f"{100 * np.mean((nai - v) / nai):+15.2f}%{int((v < nai).sum()):5d}/{len(v)}")
    print(f"{'persistence (baseline)':26}{nai.mean():10.3f}")
    print(f"{'floor':26}{np.mean([r['floor'] for r in rows]):10.3f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(per_transition=rows), open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
