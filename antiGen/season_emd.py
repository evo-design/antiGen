#!/usr/bin/env python
"""Fit a genotype-fitness model season to season by earth-mover's distance.

No clades, and no phylogeny. A season is just its bag of sequences: if x_{v,i} is the
fraction of season i's sequences carrying genotype v, the model is

    xhat_{v,i+1}  proportional to  x_{v,i} exp(f_v),     f_v = sum_{p,a} beta_pa 1[v_p = a]

and beta is fitted by minimising the earth-mover's distance between the predicted and the
observed next-season distributions, with Hamming distance between genotypes as the ground
metric. The baseline is f = 0, which makes xhat_{i+1} = x_i -- persistence.

Why this objective rather than the clade one it replaces. The clades were a choice, and
above any frequency cutoff they nest, so a season's population was counted several times
over and the effective number of observations was never the number of rows (see the
2026-09-20 entries). Here the prediction units are the genotypes themselves: a partition
of the season by construction, no cutoff, no subjective grouping. And the supports of
consecutive seasons barely overlap -- new mutations appear, old genotypes vanish -- which
is precisely the case where a metric-aware divergence is the right thing and a pointwise
one is not defined. Wasserstein charges a prediction the Hamming distance it has to move
mass, so putting weight one substitution away from where the population went is penalised
a little and putting it twenty away a lot.

Gradients. W_1 is the value of a linear program, so by LP duality
W_1(a, b) = max { u.a + v.b : u_i + v_j <= M_ij }, and by Danskin's theorem
dW/da_v = u_v at the optimal dual. Chaining through the softmax-like xhat(beta),

    dW/dbeta_k = sum_v u_v xhat_v (S_vk - sum_w xhat_w S_wk) = Cov_xhat(u, S_k)

which is a covariance under the predicted distribution, and is therefore invariant to the
constant shift that the dual potentials are only defined up to -- the one piece of
sloppiness that would otherwise matter. Verified against central differences in the tests.

The cost matrix does not depend on beta, so it is built once per season pair and reused
across every iteration and every lambda.

Two sources of sequences, which is the comparison the module exists to make:

    --clade-dir     the genotypes of the guide tree's own tips, per season, as the
                    generator's strain tables record them: Nextstrain's subsample, about
                    150 sequences a year, evenly spread by time and place.
    --alignment     every sequence in the alignment, straight from the FASTAs, with no
                    tree involved at all.

Usage:
    python -m antiGen.season_emd --clade-dir path/to/output_flu_seasons_noaug_refit
    python -m antiGen.season_emd --alignment aln.{SigPep,HA1,HA2}.fasta --metadata md.tsv
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

LAMBDAS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


# ── the season panel ────────────────────────────────────────────────────────

class SeasonPanel:
    """Per-season genotype distributions, plus the feature space over varying sites."""

    def __init__(self, seasons, min_allele_freq=0.0, max_genotypes=0, min_season_seqs=0):
        # seasons: {season -> Counter(sequence -> count)}
        self.dropped_mass = {}
        keep = {}
        for t, c in sorted(seasons.items()):
            if sum(c.values()) < min_season_seqs:
                continue
            items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
            total = float(sum(c.values()))
            if max_genotypes and len(items) > max_genotypes:
                cut = items[:max_genotypes]
                self.dropped_mass[t] = 1.0 - sum(m for _, m in cut) / total
                items = cut
            else:
                self.dropped_mass[t] = 0.0
            keep[t] = items
        self.seasons = sorted(keep)
        self.seq_len = len(next(iter(keep[self.seasons[0]]))[0]) if self.seasons else 0
        self.n_seqs = {t: float(sum(m for _, m in keep[t])) for t in self.seasons}

        # Positions that never vary add the same constant to every Hamming distance, so
        # dropping them leaves every distance and every transport plan unchanged.
        seen = [set() for _ in range(self.seq_len)]
        for t in self.seasons:
            for g, _ in keep[t]:
                for p, ch in enumerate(g):
                    seen[p].add(ch)
        self.sites = [p for p in range(self.seq_len) if len(seen[p]) > 1]
        alpha = sorted({ch for p in self.sites for ch in seen[p]})
        code = {ch: i for i, ch in enumerate(alpha)}

        self.genotypes, self.w, self.codes = {}, {}, {}
        for t in self.seasons:
            self.genotypes[t] = [g for g, _ in keep[t]]
            m = np.asarray([c for _, c in keep[t]], dtype=np.float64)
            self.w[t] = m / m.sum()
            self.codes[t] = np.asarray(
                [[code[g[p]] for p in self.sites] for g in self.genotypes[t]],
                dtype=np.uint8)

        # Features, on the same convention as the clade model: one per (position,
        # residue) that clears the allele-frequency floor at a site that varies. A
        # per-position constant is unidentifiable -- every sequence has one residue per
        # position -- and cancels out of the prediction, since xhat is a ratio.
        tot = 0.0
        cnt = [Counter() for _ in self.sites]
        for t in self.seasons:
            for g, mm in zip(self.genotypes[t], self.w[t] * self.n_seqs[t]):
                tot += mm
                for j, p in enumerate(self.sites):
                    cnt[j][g[p]] += mm
        self.features = []
        for j, p in enumerate(self.sites):
            al = {a: v / tot for a, v in cnt[j].items() if v / tot >= min_allele_freq}
            if len(al) > 1:
                self.features.extend((p, a) for a in sorted(al))
        self.feature_index = {f: i for i, f in enumerate(self.features)}
        self.n_sites = len({p for p, _ in self.features})

        self.onehot = {t: self._onehot(t) for t in self.seasons}

    def _onehot(self, t):
        D = len(self.features)
        indptr, indices = [0], []
        for g in self.genotypes[t]:
            for p, a in self.features:
                if g[p] == a:
                    indices.append(self.feature_index[(p, a)])
            indptr.append(len(indices))
        return sparse.csr_matrix((np.ones(len(indices)), indices, indptr),
                                 shape=(len(self.genotypes[t]), D))


def panel_from_clade_tables(clade_dir, tag="seasons", **kw):
    """The guide tree's own tips: one row per (season, node) with a multiplicity.

    Node sequences are how the generator records a tip's genotype, so this is the tree's
    sample set without using the tree for anything else. Strains sitting outside T_t are
    not in these tables -- 6 of 947 on the H3N2 backbone -- which is the one difference
    from counting the tips directly."""
    import pandas as pd
    strains = pd.read_csv(os.path.join(clade_dir, f"clade_strains_{tag}.csv"))
    nodes = pd.read_csv(os.path.join(clade_dir, f"clade_nodes_{tag}.csv"))
    seq = {int(r.node_id): r.seq for r in nodes.itertuples()}
    seasons = defaultdict(Counter)
    for r in strains.itertuples():
        seasons[int(r.season)][seq[int(r.node_id)]] += int(r.m_i)
    return SeasonPanel(seasons, **kw)


def panel_from_alignment(alignments, metadata, keep=None, **kw):
    """Every sequence in the alignment, with no tree involved."""
    from antiGen.clade_cluster import SeasonSequences
    d = SeasonSequences(alignments, metadata, keep=keep)
    return SeasonPanel({t: Counter(c) for t, c in d.seasons.items()}, **kw)


# ── the objective ───────────────────────────────────────────────────────────

class Pair:
    """One (season i -> season i+1) transition, with its cost matrix built once."""

    __slots__ = ("cur", "nxt", "S", "w_cur", "w_next", "M", "floor")

    def __init__(self, panel, cur, nxt):
        self.cur, self.nxt = cur, nxt
        self.S = panel.onehot[cur]
        self.w_cur = panel.w[cur]
        self.w_next = panel.w[nxt]
        A, B = panel.codes[cur], panel.codes[nxt]
        # Hamming in blocks, so a 5000 x 5000 pair does not allocate a bool cube.
        M = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
        step = max(1, 2_000_000 // max(B.shape[0], 1))
        for s in range(0, A.shape[0], step):
            e = min(s + step, A.shape[0])
            M[s:e] = (A[s:e, None, :] != B[None, :, :]).sum(2)
        self.M = M
        # The best any reweighting of season i could do: send each next-season genotype's
        # mass to the nearest genotype that exists in season i. No model of any kind that
        # only reweights the season can beat this, so it is the ceiling the reduction
        # should be read against.
        self.floor = float(self.w_next @ M.min(axis=0))


def emd(a, b, M, duals=False):
    """Exact earth-mover's distance and the dual potential for `a`.

    With `duals` the target potential comes back as well, so the caller gets the pair
    (u, z) with u_i + z_j <= M_ij and u.a + z.b = W_1. The source potential alone is
    what the Danskin gradient needs -- dW/da_i = u_i -- and it is all this returned
    until the mutation kernel, which needs the *other* one: the c-transform
    phi_w = min_j (d(w, j) - z_j) extends the optimal source potential to genotypes
    outside the support at no extra transport solve, which is how the kernel's
    mu = 0 derivative is evaluated over a hundred thousand candidates for the price of
    the solve that scoring already pays."""
    import ot
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()
    G, log = ot.emd(a, b, M, log=True, numItermax=10_000_000)
    # A failed LP is not a small numerical event here: POT warns *once* per process,
    # returns an all-zero plan, and the cost comes back as 0.0 -- an EMD smaller than any
    # attainable one, which a search over a hyperparameter would then select. Raising
    # turns that into a stack trace at the call that caused it. (The failure it caught:
    # a source mass of -1e-17 out of the mutation kernel's convolution at mu = 1.)
    if log.get("warning"):
        raise RuntimeError(
            f"transport solve failed: {log['warning']} -- {len(a)} x {len(b)}, "
            f"sum(a) - 1 = {a.sum() - 1:.3e}, min(a) = {a.min():.3e}, "
            f"sum(b) - 1 = {b.sum() - 1:.3e}, min(b) = {b.min():.3e}")
    W = float(np.sum(G * M))
    return (W, log["u"], log["v"]) if duals else (W, log["u"])


def predict(beta, pair):
    """xhat_{i+1} = x_i e^f / sum, with the shift that keeps the exponential in range."""
    f = pair.S @ beta
    e = np.exp(f - f.max())
    w = pair.w_cur * e
    s = w.sum()
    return w / s if s > 0 else pair.w_cur.copy()


def emd_loss_grad(beta, pairs, lam):
    """Mean earth-mover's distance over the pairs, with its gradient.

    The gradient of each term is Cov_xhat(u, S), u being the optimal dual potential of the
    predicted distribution. Seasons are averaged rather than summed so that lambda means
    the same thing whatever the number of training pairs."""
    D = beta.shape[0]
    loss, grad = 0.0, np.zeros(D)
    for pr in pairs:
        xhat = predict(beta, pr)
        w, u = emd(xhat, pr.w_next, pr.M)
        loss += w
        Eu = float(xhat @ u)
        ES = pr.S.T @ xhat
        grad += (pr.S.T @ (xhat * u)) - Eu * ES
    n = max(len(pairs), 1)
    return loss / n + lam * float(beta @ beta), grad / n + 2.0 * lam * beta


def fit(pairs, lam, D, beta0=None, maxiter=120):
    from scipy.optimize import minimize
    x0 = np.zeros(D) if beta0 is None else beta0.copy()
    if not pairs:
        return x0
    res = minimize(emd_loss_grad, x0, args=(pairs, lam), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 20})
    return res.x


def pick_lambda(pairs, grid, D, maxiter=120, rule="loso"):
    """Choose lambda inside the training transitions, scored by held-out EMD.

    `loso` leaves each transition out in turn -- the thorough option, and the one used
    wherever it is affordable. `last` validates on the most recent training transition
    only, fitting on everything before it. That is one fit per lambda instead of one per
    (lambda, fold), which is the difference between hours and days once a season carries
    five thousand genotypes, and the transition it validates on is the one closest in time
    to the test. Neither rule ever touches the test transition.

    Walked from the strongest penalty down with a warm start inside each fold only: the
    objective is not convex, so warm starting across folds would let a held-out season
    shape the initialisation of the fit that is then scored on it."""
    if len(pairs) < 2:
        return grid[len(grid) // 2], {}
    folds = range(len(pairs)) if rule == "loso" else [len(pairs) - 1]
    err = {lam: [] for lam in grid}
    for h in folds:
        if rule == "last":
            tr = pairs[:h]                       # strictly earlier, never a later one
        else:
            tr = [p for k, p in enumerate(pairs) if k != h]
        te = pairs[h]
        if not tr:
            continue
        warm = None
        for lam in sorted(grid, reverse=True):
            b = fit(tr, lam, D, beta0=warm, maxiter=maxiter)
            warm = b
            w, _ = emd(predict(b, te), te.w_next, te.M)
            err[lam].append(w)
    means = {lam: float(np.mean(v)) for lam, v in err.items() if v}
    if not means:
        return grid[len(grid) // 2], {}
    return min(means, key=means.get), means


# ── rolling evaluation ──────────────────────────────────────────────────────

def evaluate(panel, lam_grid=LAMBDAS, min_train=2, fixed_lambda=None, maxiter=120,
             verbose=True, only=None, rule="loso", train_window=0,
             window_side="recent"):
    ss = panel.seasons
    trans = [(ss[k], ss[k + 1]) for k in range(len(ss) - 1)
             if ss[k + 1] == ss[k] + 1]
    if verbose:
        print(f"{len(trans)} consecutive-season transitions: "
              + ", ".join(f"{a}->{b}" for a, b in trans))
    cache = {}

    def pair(a, b):
        if (a, b) not in cache:
            cache[(a, b)] = Pair(panel, a, b)
        return cache[(a, b)]

    D = len(panel.features)
    rows = []
    for k in range(min_train, len(trans)):
        cur, nxt = trans[k]
        # `only` lets one process handle one transition, which is how the uncapped
        # all-sequences arm is parallelised: each transition is independent given the
        # panel, since the training set is always the transitions strictly before it.
        if only is not None and cur not in only:
            continue
        avail = trans[:k]
        # `train_window` keeps only the most recent transitions. The point is to break the
        # confound in the rolling protocol: the transitions run in calendar order, so the
        # size of the training set and the year being predicted move together and a trend
        # in one cannot be told from a trend in the other. Holding the test transition
        # fixed and varying the window measures the learning curve on its own.
        if train_window:
            avail = (avail[:train_window] if window_side == "oldest"
                     else avail[-train_window:])
        train = [pair(a, b) for a, b in avail]
        test = pair(cur, nxt)
        lam = fixed_lambda
        if lam is None:
            lam, _ = pick_lambda(train, list(lam_grid), D, maxiter=maxiter, rule=rule)
        beta = fit(train, lam, D, maxiter=maxiter)
        model, _ = emd(predict(beta, test), test.w_next, test.M)
        base, _ = emd(test.w_cur, test.w_next, test.M)
        rows.append(dict(
            season=int(cur), next=int(nxt), n_train=len(train),
            n_available=k, train_window=int(train_window),
            window_side=window_side, lam=float(lam),
            n_cur=len(panel.genotypes[cur]), n_next=len(panel.genotypes[nxt]),
            seqs_cur=panel.n_seqs[cur], seqs_next=panel.n_seqs[nxt],
            emd_model=model, emd_base=base, emd_floor=test.floor,
            reduction=(base - model) / base if base > 0 else float("nan"),
            captured=((base - model) / (base - test.floor)
                      if base - test.floor > 1e-12 else float("nan")),
            beta_norm=float(np.linalg.norm(beta))))
        if verbose:
            r = rows[-1]
            print(f"  {cur}->{nxt}: base {base:7.4f}  model {model:7.4f}  "
                  f"floor {test.floor:7.4f}  reduction {100 * r['reduction']:+6.2f}%  "
                  f"of achievable {100 * r['captured']:+6.1f}%  lambda {lam:g}")
    return rows


def summarise(rows, n_boot=4000, seed=0):
    """Mean reduction over held-out transitions, with a bootstrap over transitions.

    A transition is the independent unit here, as a season was for the clade work: there
    is one of them per year and they are what a claim about forecasting has to hold over."""
    if not rows:
        return {}
    red = np.asarray([r["reduction"] for r in rows])
    cap = np.asarray([r["captured"] for r in rows])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(red), size=(n_boot, len(red)))
    bs = red[draws].mean(axis=1)
    return dict(
        n=len(rows), mean_reduction=float(red.mean()),
        ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        p_no_better=float(np.mean(bs <= 0.0)),
        median_reduction=float(np.median(red)),
        n_better=int((red > 0).sum()),
        mean_captured=float(np.nanmean(cap)),
        mean_emd_base=float(np.mean([r["emd_base"] for r in rows])),
        mean_emd_model=float(np.mean([r["emd_model"] for r in rows])),
        mean_emd_floor=float(np.mean([r["emd_floor"] for r in rows])))


# ── driver ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clade-dir", help="generator output; uses its strain tables, i.e. "
                                         "the guide tree's tips only")
    src.add_argument("--alignment", nargs="+", help="per-segment aligned FASTAs")
    ap.add_argument("--metadata", help="TSV with strain and date columns (--alignment)")
    ap.add_argument("--tag", default="seasons")
    ap.add_argument("--keep-strains", default=None,
                    help="file of strain names, one per line, to restrict --alignment to")
    ap.add_argument("--min-allele-freq", type=float, default=0.005)
    ap.add_argument("--min-season-seqs", type=int, default=30,
                    help="skip seasons with fewer sequences than this")
    ap.add_argument("--max-genotypes", type=int, default=0,
                    help="keep only the this many most abundant genotypes per season and "
                         "renormalise; 0 keeps all. The dropped mass is reported")
    ap.add_argument("--lambdas", default="0,0.001,0.01,0.1,1,10,100")
    ap.add_argument("--fixed-lambda", type=float, default=None,
                    help="skip selection. A sweep over this is a DIAGNOSTIC: reading the "
                         "best value off held-out numbers would be selecting on the test "
                         "set")
    ap.add_argument("--min-train", type=int, default=2)
    ap.add_argument("--train-window", type=int, default=0,
                    help="fit on only this many of the most recent training transitions; "
                         "0 uses all of them. Sweeping it at a fixed test transition is "
                         "how the learning curve is separated from the calendar")
    ap.add_argument("--window-side", default="recent", choices=("recent", "oldest"),
                    help="which end of the available history --train-window keeps. "
                         "Comparing the two at a fixed test transition and a fixed window "
                         "size is a test of stationarity: if the seasons are exchangeable "
                         "it cannot matter, and if the landscape drifts the recent end "
                         "wins")
    ap.add_argument("--transitions", default="",
                    help="evaluate only these transitions, named by the season predicted "
                         "*from*, comma separated. One process per transition is how the "
                         "uncapped all-sequences arm is parallelised")
    ap.add_argument("--lambda-rule", default="loso", choices=("loso", "last"),
                    help="'loso' leaves out each training transition in turn; 'last' "
                         "validates on the most recent training transition only, which "
                         "is what makes the uncapped arm affordable")
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    if a.clade_dir:
        panel = panel_from_clade_tables(
            a.clade_dir, a.tag, min_allele_freq=a.min_allele_freq,
            max_genotypes=a.max_genotypes, min_season_seqs=a.min_season_seqs)
        source = f"clade tables (guide-tree tips) {a.clade_dir}"
    else:
        if not a.metadata:
            ap.error("--alignment needs --metadata")
        keep = None
        if a.keep_strains:
            keep = {l.strip() for l in open(a.keep_strains) if l.strip()}
        panel = panel_from_alignment(
            a.alignment, a.metadata, keep=keep, min_allele_freq=a.min_allele_freq,
            max_genotypes=a.max_genotypes, min_season_seqs=a.min_season_seqs)
        source = f"alignment ({len(a.alignment)} segments)"

    print(f"source: {source}")
    print(f"{len(panel.features)} features over {panel.n_sites} varying of "
          f"{panel.seq_len} sites; {len(panel.seasons)} seasons "
          f"{panel.seasons[0]}-{panel.seasons[-1]}")
    for t in panel.seasons:
        d = panel.dropped_mass.get(t, 0.0)
        print(f"   season {t}: {int(panel.n_seqs[t]):6d} sequences, "
              f"{len(panel.genotypes[t]):5d} genotypes"
              + (f", {100 * d:.2f}% of mass dropped by the cap" if d > 0 else ""))
    print()

    grid = tuple(float(x) for x in a.lambdas.split(","))
    only = ({int(x) for x in a.transitions.split(",")} if a.transitions else None)
    rows = evaluate(panel, grid, a.min_train, a.fixed_lambda, a.maxiter,
                    only=only, rule=a.lambda_rule, train_window=a.train_window,
                    window_side=a.window_side)
    s = summarise(rows, a.n_boot)
    if s:
        print(f"\nHeld-out, one season ahead, over {s['n']} transitions:")
        print(f"  mean EMD   baseline (f=0) {s['mean_emd_base']:.4f}   "
              f"model {s['mean_emd_model']:.4f}   floor {s['mean_emd_floor']:.4f}  "
              f"(Hamming units)")
        print(f"  mean reduction vs baseline {100 * s['mean_reduction']:+.2f}% "
              f"[{100 * s['ci'][0]:+.2f}, {100 * s['ci'][1]:+.2f}]  "
              f"P(<=0) {s['p_no_better']:.3f}")
        print(f"  better than baseline in {s['n_better']} of {s['n']} transitions; "
              f"mean share of the achievable gain {100 * s['mean_captured']:+.1f}%")
    if a.output:
        with open(a.output, "w") as f:
            json.dump(dict(source=source, n_features=len(panel.features),
                           max_genotypes=a.max_genotypes,
                           dropped_mass=panel.dropped_mass,
                           lambda_rule=a.lambda_rule,
                           fixed_lambda=a.fixed_lambda,
                           per_transition=rows, summary=s), f, indent=2)
        print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
