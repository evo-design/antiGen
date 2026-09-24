#!/usr/bin/env python
"""A one-step substitution kernel, and the transport machinery that scores it.

Every model in `season_momentum` predicts `xhat_v ∝ base_v exp(f_v)`: a *reweighting* of
the genotypes the support already holds. That has a hard floor -- the mass-weighted
distance from each next-season genotype to the nearest one the model may touch -- and on
the eight backbone transitions the floor is 3.632 against a standing error of 8.220, so
44% of what is left is structurally unreachable. This module moves the floor instead of
chasing what sits above it, by convolving a finished forecast with a substitution kernel:

    x'  =  (1 - mu) x  +  mu K x,      (K x)_w = sum_v x_v pi(q, b | v),  w = v ⊕ (q,b)

with `pi(· | v)` a distribution over one-step substitutions. `mu = 0` is the parent
forecast exactly, and nothing fitted ever sees a candidate: the kernel is applied *after*
the forecast, which is what keeps `beta`, the design matrices and the floor of the fitted
model byte-identical to the parent's. Arm 2 of `season_emd` and the full-alignment
momentum run both expanded the support a model was *fitted* on and inverted the result,
from +4.3% to -9.2%; this is the same operation done on the other side of the fit.

What the algebra says before any kernel is built
-----------------------------------------------
**The downside is bounded by `mu k`.** The coupling that holds `(1-mu)x` in place and
ships `mu x` along `K` costs at most `mu k`, so `|W_1(x', nu) - W_1(x, nu)| <= mu k`. At
`k = 1` the convolution risks `mu` against an oracle upside of 0.940 EMD units.

**`W_1` is convex in `mu`**, since `x'(mu)` is affine and `W_1(·, nu)` convex in its first
argument. One optimum, and `golden_mu` finds it in a handful of transport solves.

**The `mu = 0` derivative costs one transport solve.** By Danskin it is the supremum of
`u . (Kx - x)` over the dual argmax set; the candidates carry no mass at `mu = 0`, so the
c-transform `phi_w = min_j (d(w,j) - z*_j)` of the optimal *target* potential attains it:

    dW_1/dmu |_0  =  sum_v x_v sum_{(q,b)} pi(q,b|v) [ phi_{v ⊕ (q,b)} - phi_v ]

`phi` is 1-Lipschitz in Hamming by dual feasibility and the cost is integer-valued, so
every bracket is exactly -1, 0 or +1. Writing `p_down`, `p_flat` for the `x`-weighted
shares of proposal mass on each,

    dW_1/dmu |_0  =  1 - p_flat - 2 p_down,   so   p_down* = (1 - p_flat) / 2

**A kernel must place more than 44.6% of its proposal mass on downhill directions to be
worth anything at all.** A uniform kernel achieves 7.7%, so its optimal `mu` is provably
zero -- a convex function with right-derivative +0.739 at zero attains its minimum there.
That measurement is what `descent` computes and what `gate` turns into a pre-scoring
filter: an arm that does not clear `BREAK_EVEN` on the *training* transitions is never
scored on a held-out one, so it costs the exposure budget nothing.

The cost matrix needs no sequences
----------------------------------
Site `q` contributes `1[v_q != u_q]` to `d(v, u)` before the substitution and
`1[b != u_q]` after, so

    d(v ⊕ (q,b), u)  =  d(v, u)  +  1[v_q = u_q]  -  1[b = u_q]

and the candidate block is a rank-structured update of `M`, built in
`O(|support| x |targets|)` per `(q, b)` pair rather than by materialising a single
candidate sequence. The signs are the way round they are because the *old* residue's
agreement is what is lost and the *new* residue's is what is gained; the transposed form
disagrees with brute-force Hamming on 191 of 200 random draws, which is why
`test_the_recursion_matches_brute_force_hamming` exists.

Rows with identical costs are interchangeable in the LP, so the block is deduplicated by
cost row -- 144,628 triples give 143,058 distinct sequences but only 27,570 distinct rows,
a fivefold reduction. The merge is exact and it is a scoring-time reformulation only: the
forecast itself is never defined on the merged index.

Leakage
-------
`kemp` reads only sequences dated at or before the horizon and is the primary arm for
exactly that reason. `kobs` and `kplm` read ESM-2, and UniRef contains H3N2 HA from every
test season, so no protocol on our side makes them leak-free; they are labelled leaky
wherever they appear. `kobs` exists to bound `kplm`: it is the same model at the same
temperature and truncation with the novel-residue channel switched off, and the oracle
floors put that channel's whole prize at 0.088 EMD units.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from antiGen.season_emd import emd

# The twenty proteinogenic residues, in the order the ESM-2 marginals are written.
AA = "ACDEFGHIKLMNPQRSTVWY"

# The share of proposal mass a kernel must place on strictly downhill directions before
# the convolution can help at all: p_down* = (1 - p_flat) / 2, evaluated at the panel's
# measured p_flat of 0.1076. Declared here rather than recomputed per arm so the gate is
# a fixed bar; `descent` also returns the arm's own break-even for the report.
BREAK_EVEN = 0.4462

# The kernel arms. `leaky` marks the ones that read a model pretrained on data from the
# test seasons; `conditional` marks the ones whose proposal depends on the parent
# genotype beyond excluding its own residue, which section 2 makes the whole game.
ARMS = {
    "kunif": dict(residues="observed", leaky=False, conditional=False,
                  needs_plm=False),
    "kemp": dict(residues="observed", leaky=False, conditional=False, needs_plm=False),
    "kcons": dict(residues="observed", leaky=False, conditional=True, needs_plm=False),
    "kcmom": dict(residues="observed", leaky=False, conditional=True, needs_plm=False),
    "kobs": dict(residues="observed", leaky=True, conditional=True, needs_plm=True),
    "kplm": dict(residues="all", leaky=True, conditional=True, needs_plm=True),
    "kemp*kplm": dict(residues="all", leaky=True, conditional=True, needs_plm=True),
}

# The consensus arms' timescales, declared here rather than selected -- the
# `lambda_alpha` precedent, and the same reason `kemp`'s tau_e is declared: the gate
# scores no forecast, so settling them there costs the multiple-looks budget nothing,
# whereas putting them in front of the outer selector would cost about a point each.
#
# `CONS_FAST` is the fast timescale in years when no `tau_e` is given, and the slow one
# is `CONS_SLOW_MULT` times it. 1 and 3 years are the panel's own `--tau-s-years` grid,
# so the pair is borrowed rather than invented.
CONS_FAST = 1.0
CONS_SLOW_MULT = 3.0
# The momentum exponent per arm. `kcons` is the level-only control -- it proposes toward
# where the population *is* -- and `kcmom` extrapolates the log-frequency trend one step,
# proposing toward where it is *heading*. Running both is what separates "the consensus
# is the right direction" from "the change in the consensus is".
CONS_DELTA = {"kcons": 0.0, "kcmom": 1.0}

# Refuse to materialise a candidate block bigger than this. At `m = 0` over observed
# residues a transition carries 63k-168k triples, so the guard only ever fires on an
# untruncated PLM arm, where the block would be 2.3M rows and the LP no longer a
# transport problem. Raising rather than silently truncating: a truncation the caller did
# not ask for would change which kernel was scored.
MAX_CANDIDATES = 500_000


class Candidates:
    """One kernel's proposal, and the deduplicated cost block the LP sees.

    `parent`, `site` and `res` name each `(v, q, b)` triple -- `site` indexes the panel's
    varying sites and `res` the panel's residue code, extended past the end of the
    alphabet for residues the panel never observed. `pi` is the proposal probability,
    summing to one over each parent that has any candidate at all. `row` maps a candidate
    to its row of `cost`, which holds the *distinct* cost rows only."""

    __slots__ = ("parent", "site", "res", "pi", "row", "cost", "n_parents", "arm",
                 "tau_k", "m")

    def __init__(self, parent, site, res, pi, row, cost, n_parents, arm, tau_k, m):
        self.parent, self.site, self.res, self.pi = parent, site, res, pi
        self.row, self.cost = row, cost
        self.n_parents, self.arm, self.tau_k, self.m = n_parents, arm, tau_k, m

    def __len__(self):
        return len(self.parent)

    @property
    def n_rows(self):
        return self.cost.shape[0]

    def mass(self, x):
        """`K x` on the deduplicated rows: the candidate mass a forecast `x` sends out.

        Sums to `x` over the parents that have candidates, and to less than one when some
        parent has none -- a single-genotype support proposes nothing, and that mass
        simply stays where it is rather than being invented somewhere."""
        w = np.asarray(x, dtype=np.float64)[self.parent] * self.pi
        return np.bincount(self.row, weights=w, minlength=self.cost.shape[0])

    def moved(self, x):
        """The share of `x` the kernel actually moves; one whenever every parent has a
        candidate, which is every real transition."""
        return float(np.asarray(x, dtype=np.float64)[self.parent] @ self.pi)

    @property
    def p_parent(self):
        """Total proposal weight per parent: one where the arm proposes anything at all
        and zero where it does not.

        This is the row sum of `K`, and it is what keeps the convolution
        mass-conserving: a parent with an empty row has `mu` multiplying nothing, so its
        mass stays where it is rather than being scaled away into a forecast that no
        longer sums to one.

        **Clipped at one, and the clip is not cosmetic.** `pi` is renormalised per
        parent, so the row sum is one up to float error -- but a bincount of a few
        hundred renormalised entries lands at `1 + 1e-15` for a good fraction of them,
        and `convolve` then leaves `x_v (1 - mu p_v)` *negative* at `mu = 1`. POT reports
        a negative source mass as an infeasible LP, warns once, and returns a zero plan
        whose cost is `0.0` -- an EMD better than any real one, which the oracle-`mu`
        search would then walk straight to. One is what the row sum of a row-stochastic
        matrix is, so taking the minimum is exact rather than a tolerance."""
        return np.minimum(
            np.bincount(self.parent, weights=self.pi, minlength=self.n_parents), 1.0)



# ── the convolved forecast ──────────────────────────────────────────────────

class Forecast:
    """A prediction that may live on the extended support `V_T ∪ N_1(V_T)`.

    `w` is the mass the forecast leaves on the task's own support and `blocks` holds the
    mass the kernel moved off it, one `(mass, cost)` pair per convolution that
    contributed. A *tuple* of blocks rather than a single one because `top_k` averages
    the predictions of the best few candidates, and those candidates may carry different
    arms or different truncations, whose dedup indexes are not comparable. Averaging in
    the extended space is then a concatenation of blocks with masses divided by K, which
    is exact: a distribution over a disjoint union is what concatenation means.

    A plain `np.ndarray` is still what every unconvolved family returns, and `mu = 0`
    returns a `Forecast` with **no** block rather than an empty one, so the kernel
    families nest their parents bit-identically."""

    __slots__ = ("w", "blocks")

    def __init__(self, w, blocks=()):
        self.w = np.asarray(w, dtype=np.float64)
        self.blocks = tuple((np.asarray(b, dtype=np.float64), c) for b, c in blocks)

    def sum(self):
        return float(self.w.sum()) + float(sum(b.sum() for b, _ in self.blocks))

    @property
    def off_support(self):
        """The share of the forecast sitting on genotypes the support does not hold."""
        t = self.sum()
        return (t - float(self.w.sum())) / t if t > 0 else 0.0

    def stacked(self, M):
        """The forecast and its cost matrix as one distribution over the extended
        support: masses concatenated, cost rows stacked underneath `M`.

        The order is support first, then the blocks in the order they were added, which
        is the order `average` concatenated them in -- fixed, so the transport problem
        handed to the solver is a deterministic function of the forecast."""
        if not self.blocks:
            return self.w, M
        a = np.concatenate([self.w] + [b for b, _ in self.blocks])
        C = np.vstack([np.asarray(M, dtype=np.float64)]
                      + [np.asarray(c, dtype=np.float64) for _, c in self.blocks])
        return a, C


def convolve(w, cand, mu):
    """`x' = (1 - mu) x + mu K x`, as a forecast on the extended support.

    A parent the arm proposes nothing for keeps its mass rather than losing it: its row
    of `K` is empty, so `mu` multiplies nothing there and the total is conserved either
    way. That is what `Candidates.p_parent` is for, and it is why a single-genotype
    support -- which has no substitution to propose at all -- returns the parent rather
    than a forecast that no longer sums to one.

    `mu = 0` returns the parent bit-identically, with no block at all, which is the
    nesting the families rely on."""
    w = np.asarray(w, dtype=np.float64)
    mu = float(mu)
    if mu <= 0.0 or len(cand) == 0:
        return Forecast(w)
    kept = w * (1.0 - mu * cand.p_parent)
    return Forecast(kept, ((mu * cand.mass(w), cand.cost),))


def average(preds):
    """The mean of several forecasts, which may live on different extended supports.

    Plain arrays average as they always did. Once any prediction carries a block the
    mean is taken in the extended space: the support parts average pointwise, every
    block comes along with its mass divided by the number of predictions, and the result
    is renormalised. Nothing is merged across blocks, because two arms' dedup indexes
    name different rows."""
    if not preds:
        raise ValueError("nothing to average")
    if not any(isinstance(p, Forecast) for p in preds):
        w = np.mean(np.vstack([np.asarray(p, dtype=np.float64) for p in preds]), axis=0)
        s = w.sum()
        return w / s if s > 0 else w
    fcs = [p if isinstance(p, Forecast) else Forecast(p) for p in preds]
    k = len(fcs)
    w = np.mean(np.vstack([f.w for f in fcs]), axis=0)
    out = Forecast(w, tuple((b / k, c) for f in fcs for b, c in f.blocks))
    t = out.sum()
    if t > 0:
        out.w = out.w / t
        out.blocks = tuple((b / t, c) for b, c in out.blocks)
    return out


# ── the empirical substitution table ────────────────────────────────────────

def empirical_counts(panel, horizon, tau_e):
    """Discounted counts of the one-step substitutions the record shows up to `horizon`.

    Every pair of genotypes at Hamming distance one whose *earlier* member is dated at or
    before the horizon contributes the substitution that separates them, directed by
    first observation: the older residue is the old one. An occurrence is discounted by
    `exp(-(T - t)/tau_e)` in the derived genotype's first-observation date, so a
    substitution the record showed a decade ago counts for less than one it showed last
    year.

    **Nothing dated after the horizon is read**, which is what makes this the leak-free
    arm and is asserted directly as a test. A tie in first observation contributes half a
    count in each direction rather than an arbitrary one.

    `tau_e` is the panel's own lookback by default -- the window the support already uses
    -- and is deliberately *not* a field the selector varies: the 2026-09-21 grid-size
    entry priced one extra field in front of the selector at about a point, and this one
    can be settled by declaration.

    Returns a dense `(n_sites, |alphabet|, |alphabet|)` table of counts.

    Memoised on the panel, keyed by `(horizon, tau_e)`: the table is a function of those
    two and of the record, the same for every temperature and truncation of every arm
    that reads it, and building it is an all-pairs Hamming scan over every genotype dated
    at or before the horizon. Without the memo a gate sweep over `(tau_k, m)` would
    rebuild the identical table once per cell."""
    tau_e = float(panel.lookback if tau_e is None else tau_e)
    memo = getattr(panel, "_emp_counts", None)
    if memo is None:
        memo = panel._emp_counts = {}
    if (horizon, tau_e) in memo:
        return memo[(horizon, tau_e)]
    first = np.full(len(panel.genotypes), np.inf)
    np.minimum.at(first, panel.tip_gid, panel.dates)
    keep = np.flatnonzero(first <= horizon)
    nA = len(panel.alphabet)
    cnt = np.zeros((len(panel.sites), nA, nA))
    memo[(horizon, tau_e)] = cnt
    if len(keep) < 2:
        return cnt

    C = panel.codes[keep]
    t0 = first[keep]
    wt = np.exp(-(horizon - t0) / max(tau_e, 1e-9))
    step = max(1, 2_000_000 // max(len(keep), 1))
    for a in range(0, len(keep), step):
        b = min(a + step, len(keep))
        D = (C[a:b, None, :] != C[None, :, :]).sum(2)
        ii, jj = np.nonzero(D == 1)
        ii = ii + a
        # Each unordered pair is seen twice by the blocked scan; keep one ordering and
        # let the dates, not the row order, decide the direction.
        m = ii < jj
        ii, jj = ii[m], jj[m]
        if not len(ii):
            continue
        q = np.argmax(C[ii] != C[jj], axis=1)
        ci, cj = C[ii, q], C[jj, q]
        older_i = t0[ii] < t0[jj]
        tie = t0[ii] == t0[jj]
        # i older: ci -> cj, weighted by the derived genotype's own age.
        f = older_i & ~tie
        np.add.at(cnt, (q[f], ci[f], cj[f]), wt[jj[f]])
        f = (~older_i) & ~tie
        np.add.at(cnt, (q[f], cj[f], ci[f]), wt[ii[f]])
        np.add.at(cnt, (q[tie], ci[tie], cj[tie]), 0.5 * wt[jj[tie]])
        np.add.at(cnt, (q[tie], cj[tie], ci[tie]), 0.5 * wt[ii[tie]])
    return cnt


# ── the proposal ────────────────────────────────────────────────────────────

def _columns(panel, A, residues):
    """The `(site, residue)` pairs an arm may propose, as two parallel code arrays.

    `observed` keeps residues the support already carries at that site, and so drops
    sites carrying only one: substituting to the single residue already there is a no-op,
    which is the degenerate case the tests pin and part of why the leak-free ceiling is
    2.780 rather than 2.692.

    `all` opens **every** site the panel sees varying to all twenty residues, including
    sites that happen to be monomorphic within this horizon's support -- those are exactly
    where a novel residue has something to offer, and dropping them would make the arm's
    ceiling something other than the `k = 1` Hamming ball the floor table reports.
    Residues the panel never observed are coded past the end of its alphabet, so they
    equal no target's code and are a mismatch everywhere, which is precisely what the
    Hamming recursion needs a novel residue to mean.

    Sites the panel never sees varying are outside `panel.codes` altogether, and leaving
    them out costs nothing: every genotype agrees there, so a substitution at such a site
    can only open a new mismatch and can never lower a distance."""
    nA = len(panel.alphabet)
    extra = {c: nA + i for i, c in enumerate(a for a in AA if a not in panel.code)}
    site, res = [], []
    for j in range(A.shape[1]):
        obs = np.unique(A[:, j])
        if residues == "observed":
            if len(obs) < 2:
                continue                   # nothing to substitute to; see test 13
            cand = obs
        else:
            cand = np.asarray(sorted({panel.code.get(c, extra.get(c, -1)) for c in AA}
                                     | set(int(o) for o in obs)))
            cand = cand[cand >= 0]
        site.extend([j] * len(cand))
        res.extend(int(c) for c in cand)
    return np.asarray(site, dtype=np.int32), np.asarray(res, dtype=np.int32)


def site_allele_freq(panel, horizon, tau):
    """`f_q(b)`: the discounted share of residue `b` at varying site `q` at `horizon`.

    Every sequence dated at or before the horizon contributes `exp(-(T - t)/tau)` to its
    own residue at each site, and each site is normalised to sum to one. **Nothing dated
    after the horizon is read**, which is what makes the consensus arms leak-free and is
    asserted directly as a test.

    This is the same quantity `TipPanel.landscape` computes for the escape term, but over
    the panel's own alphabet and varying-site index rather than the feature set, which
    drops residues below the allele-frequency floor and would leave the kernel unable to
    name a column it is entitled to propose. Recomputing it here keeps the arm
    self-contained and the index alignment testable, at the cost of one extra table per
    `(horizon, tau)`.

    Ages are measured from the most recent sequence, not from the horizon: the shares are
    normalised per site so a common factor cancels exactly, but without the shift a short
    `tau` underflows every weight to zero and the table falls back to uniform, which is
    the opposite of what a short memory means. Same stabilisation as `landscape` and as
    `predict`'s `f - f.max()`.

    Memoised on the panel, keyed by `(horizon, tau)`: the two consensus arms ask for the
    same table at every temperature and truncation of the gate's sweep.

    Returns a dense `(n_varying_sites, |alphabet|)` array of shares."""
    key = (float(horizon), float(tau))
    memo = getattr(panel, "_saf", None)
    if memo is None:
        memo = panel._saf = {}
    if key in memo:
        return memo[key]
    nA, J = len(panel.alphabet), panel.codes.shape[1]
    f = np.full((J, nA), 1.0 / nA)
    memo[key] = f
    idx = np.where(panel.dates <= horizon)[0]
    if not len(idx) or not J:
        return f
    age = horizon - panel.dates[idx]
    wt = (np.ones_like(age) if not np.isfinite(tau)
          else np.exp(-(age - age.min()) / max(tau, 1e-12)))
    gw = np.zeros(len(panel.genotypes))
    np.add.at(gw, panel.tip_gid[idx], wt)
    nz = np.flatnonzero(gw > 0)
    if not len(nz):
        return f
    # One bincount over a flattened (site, residue) index rather than a loop over sites.
    # int64 explicitly: `panel.codes` is uint8, and building the site offsets in the
    # codes' own dtype wraps at 256 -- under NEP 50 the `* nA` stays uint8 too, so the
    # offsets cycle every 13 sites and every site collapses onto a shared bin. The table
    # comes back near-uniform, which is a plausible-looking distribution and not an
    # error, so nothing downstream would have raised. `test_site_allele_freq_is_not
    # _corrupted_by_the_flat_index` is the regression.
    C = panel.codes[nz].astype(np.int64)                    # (k, J)
    flat = (C + np.arange(J, dtype=np.int64)[None, :] * nA).ravel()
    acc = np.bincount(flat, weights=np.repeat(gw[nz], J),
                      minlength=J * nA).reshape(J, nA)
    tot = acc.sum(axis=1, keepdims=True)
    np.divide(acc, tot, out=f, where=tot > 0)
    return f


def consensus_freq(panel, horizon, arm, tau_e):
    """`f_hat_q(b)`: the site-allele shares the consensus arms propose toward.

    `kcons` returns the fast table as it stands -- where the population *is*. `kcmom`
    extrapolates the log-frequency trend one step,

        log f_hat  =  log f_fast  +  delta * (log f_fast - log f_slow)

    renormalised per site, which is the site-allele reading of the momentum column
    `Task.design` already builds at genotype level. The point of the arm is that the
    transport-downhill direction is a property of *where the population is going*, not of
    what a given genotype can plausibly mutate into -- the one thing that separates the
    only arm that has ever cleared the gate (`kemp`, a population signal) from the two
    that did not (`kobs`/`kplm`, sequence-intrinsic ones).

    The momentum ratio is floored at the panel's own `min_allele_freq` before the logs,
    and only inside the ratio: the level term keeps its exact share. Without the floor an
    allele absent from the slow window but present in the fast one gives an infinite
    ratio, and the arm would put all of a site's mass on whichever allele happened to
    appear most recently. The floor is the panel's declared one rather than a new
    constant, so it is auditable against a number already in the protocol."""
    fast = float(CONS_FAST if tau_e is None else tau_e)
    f_fast = site_allele_freq(panel, horizon, fast)
    delta = CONS_DELTA[arm]
    if delta == 0.0:
        return f_fast
    f_slow = site_allele_freq(panel, horizon, fast * CONS_SLOW_MULT)
    eps = max(float(getattr(panel, "min_allele_freq", 0.0)) or 1e-3, 1e-12)
    with np.errstate(divide="ignore"):
        lg = (np.log(np.maximum(f_fast, 1e-300))
              + delta * (np.log(np.maximum(f_fast, eps))
                         - np.log(np.maximum(f_slow, eps))))
    lg = lg - lg.max(axis=1, keepdims=True)
    w = np.where(np.isfinite(lg), np.exp(lg), 0.0)
    tot = w.sum(axis=1, keepdims=True)
    out = np.full_like(w, 1.0 / w.shape[1])
    np.divide(w, tot, out=out, where=tot > 0)
    return out


def _log_weights(panel, sup, A, arm, col_site, col_res, horizon, tau_e):
    """`log(s_q(v) psi_q(v_q -> b))` for every parent and every candidate column.

    The factorisation `pi(q,b|v) ∝ s_q(v) psi_q(v_q -> b)` is where the arms differ, and
    only in where the two factors come from. `kunif` is flat, so its site propensity is
    the count of alternatives a site carries. `kemp` is the discounted substitution
    count, which gives every parent the same propensities. The two PLM arms read
    `p(b | v)` directly, which carries *both* factors at once: a site where the model is
    confident of the wild type puts little mass on any alternative, so proposal mass
    concentrates on the sites the model is unsure about, and section 2 makes exactly that
    kind of conditioning the whole game."""
    n, C = len(sup), len(col_site)
    A_at = A[:, col_site]                                   # (n, C) the parent's residue
    with np.errstate(divide="ignore"):
        if arm == "kunif":
            L = np.zeros((n, C))
        elif arm in ("kemp", "kemp*kplm"):
            cnt = empirical_counts(panel, horizon, tau_e)
            nA = cnt.shape[1]
            safe = np.where(col_res < nA, col_res, 0)[None, :]
            L = np.log(np.where(col_res[None, :] < nA,
                                cnt[col_site[None, :], A_at, safe], 0.0))
        elif arm in CONS_DELTA:
            # psi_q(v_q -> b) = f_hat_q(b): propose the residue the population is
            # heading toward. s_q(v) = 1 - f_hat_q(v_q): spend that proposal at the
            # sites where v's *own* residue is the one being left behind, which is the
            # whole of the arm's conditioning on v and the reason a site where v already
            # carries the rising allele is not proposed at.
            fh = consensus_freq(panel, horizon, arm, tau_e)
            nA = fh.shape[1]
            safe = np.where(col_res < nA, col_res, 0)[None, :]
            psi = np.where(col_res[None, :] < nA, fh[col_site[None, :], safe], 0.0)
            keep = np.clip(1.0 - fh[col_site[None, :], A_at], 1e-12, None)
            L = np.log(psi) + np.log(keep)
        else:
            L = np.zeros((n, C))
        if arm in ("kobs", "kplm", "kemp*kplm"):
            if panel.marg is None:
                raise ValueError(f"arm {arm!r} needs PLM marginals; pass --plm-marginals")
            # The panel's residue codes are alphabetical over what it observed; the
            # marginals are over `AA`. Map one onto the other, and give a panel residue
            # with no PLM column (a gap, say) probability zero rather than a wrong one.
            aa_of = np.full(len(panel.alphabet) + len(AA), -1, dtype=np.int64)
            for i, c in enumerate(panel.alphabet):
                if c in panel.marg_alphabet:
                    aa_of[i] = panel.marg_alphabet.index(c)
            extra = [c for c in AA if c not in panel.code]
            for i, c in enumerate(extra):
                aa_of[len(panel.alphabet) + i] = panel.marg_alphabet.index(c)
            k = aa_of[col_res]
            P = panel.marg[sup]                             # (n, n_sites, |AA|)
            safe = np.where(k < 0, 0, k)[None, :]
            lp = P[np.arange(n)[:, None], col_site[None, :], safe]
            lp = np.where(k[None, :] < 0, -np.inf, lp)
            L = L + lp
    # A parent never proposes its own residue, and `kunif`'s site propensity is flat over
    # the alternatives a site carries rather than over the site itself.
    L = np.where(A_at == col_res[None, :], -np.inf, L)
    return L


def _softmax_truncate(L, tau_k, m):
    """`pi ∝ exp(L / tau_k)` per parent, kept to its top `m` columns and renormalised.

    Temperature and truncation are the two levers the round actually has: section 2 makes
    the question "can the top of the proposal ranking be made sharp enough", and these are
    how enrichment is converted into a forecast. `m = 0` keeps everything; `tau_k -> 0`
    approaches point mass on the single best direction per parent, whose derivative the
    panel measures at -0.995.

    Ties at the truncation boundary are broken by column order, which is fixed by
    `_columns`, so the kernel is a deterministic function of its arguments."""
    n, C = L.shape
    Z = L / max(float(tau_k), 1e-12)
    if m and m < C:
        # `argpartition` alone would not be stable, and a tie decided by the partition's
        # internal order would make the kernel depend on numpy's pivot choice.
        order = np.argsort(-Z, axis=1, kind="stable")[:, :m]
        keep = np.zeros_like(Z, dtype=bool)
        np.put_along_axis(keep, order, True, axis=1)
        Z = np.where(keep, Z, -np.inf)
    Z = Z - np.max(np.where(np.isfinite(Z), Z, -np.inf), axis=1, keepdims=True)
    P = np.where(np.isfinite(Z), np.exp(Z), 0.0)
    s = P.sum(axis=1, keepdims=True)
    return np.divide(P, s, out=np.zeros_like(P), where=s > 0)


def _cost_block(M, A, B, parent, site, res):
    """The candidate block, by the recursion, and deduplicated by cost row.

    `d(v ⊕ (q,b), u) = d(v, u) + 1[v_q = u_q] - 1[b = u_q]`. Built one `(q, b)` group at
    a time so the whole `|candidates| x |targets|` array is the only thing materialised,
    and never a candidate sequence."""
    K, T = len(parent), B.shape[0]
    R = np.empty((K, T), dtype=np.int16)
    Mi = np.rint(M).astype(np.int16)
    order = np.lexsort((res, site))
    starts = np.flatnonzero(np.r_[True, (site[order][1:] != site[order][:-1])
                                  | (res[order][1:] != res[order][:-1])])
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        idx = order[a:b]
        q, rr = int(site[idx[0]]), int(res[idx[0]])
        vs = parent[idx]
        same = (A[vs, q][:, None] == B[None, :, q])
        hit = (rr == B[:, q])[None, :]
        R[idx] = Mi[vs] + same.astype(np.int16) - hit.astype(np.int16)
    # Merging identical cost rows is exact: the LP only ever sees the row, so rows with
    # identical costs are interchangeable and summing their masses leaves the optimal
    # value unchanged. It is a scoring-time reformulation -- the forecast is never
    # defined on the merged index -- and it buys a factor of five to eight.
    view = np.ascontiguousarray(R).view([("", R.dtype)] * T).ravel()
    _, first, inv = np.unique(view, return_index=True, return_inverse=True)
    return R[first], inv.astype(np.int32)


def candidates(panel, sup, M, tgt_g, arm, tau_k=1.0, m=0, horizon=None, tau_e=None):
    """Every one-step candidate an arm proposes, with its proposal mass and cost rows."""
    if arm not in ARMS:
        raise ValueError(f"unknown kernel arm {arm!r}; have {sorted(ARMS)}")
    A, B = panel.codes[sup], panel.codes[tgt_g]
    col_site, col_res = _columns(panel, A, ARMS[arm]["residues"])
    if not len(col_site):
        empty = np.zeros(0, dtype=np.int32)
        return Candidates(empty, empty, empty, np.zeros(0),
                          empty, np.zeros((0, len(tgt_g)), dtype=np.int16),
                          len(sup), arm, tau_k, m)
    L = _log_weights(panel, sup, A, arm, col_site, col_res, horizon, tau_e)
    P = _softmax_truncate(L, tau_k, m)
    # A parent the arm has nothing to say about -- every count zero, say -- falls back to
    # the control's row rather than to no row at all, so `K` stays row-stochastic and the
    # arm is never credited with a proposal it did not make.
    dead = np.flatnonzero(P.sum(axis=1) <= 0)
    if len(dead):
        obs_site, obs_res = _columns(panel, A, "observed")
        if len(obs_site):
            fb = np.zeros_like(P)
            cols = {(int(q), int(r)): c for c, (q, r) in enumerate(zip(col_site, col_res))}
            for v in dead:
                ok = [cols[(int(q), int(r))] for q, r in zip(obs_site, obs_res)
                      if A[v, q] != r and (int(q), int(r)) in cols]
                if ok:
                    fb[v, ok] = 1.0 / len(ok)
            P[dead] = fb[dead]

    vv, cc = np.nonzero(P)
    if len(vv) > MAX_CANDIDATES:
        raise ValueError(
            f"{len(vv)} candidates for arm {arm!r} at tau_k={tau_k:g}, m={m}; "
            f"the cap is {MAX_CANDIDATES}. Truncate with a smaller m rather than "
            f"scoring a kernel nobody asked for")
    parent = vv.astype(np.int32)
    site = col_site[cc]
    res = col_res[cc]
    pi = P[vv, cc]
    cost, row = _cost_block(M, A, B, parent, site, res)
    return Candidates(parent, site, res, pi, row, cost, len(sup), arm, tau_k, m)



def one_step_floor(panel, sup, M, tgt_g, tgt_w, residues="observed"):
    """The oracle floor once the support is widened by one substitution.

    The best any forecast on `V_T ∪ N_1(V_T)` could do: send each next-season genotype's
    mass to the nearest genotype anything is *allowed* to sit on. Computed by the same
    recursion the candidate block uses, streamed one `(q, b)` pair at a time so the
    2.3M-row untruncated block of an all-residues arm is never materialised -- only a
    running minimum over targets is kept.

    **`residues` is the whole point of the pair of numbers.** `observed` is the leak-free
    ceiling, reachable by a kernel that proposes only residues the support already
    carries; `all` opens every site to all twenty and is the ceiling a PLM's novel-residue
    channel is bounded by. On the backbone panel they are 2.780 and 2.692 against an
    unextended 3.632, which puts that channel's entire prize at 0.088 EMD units.

    This is a *ceiling*, not a score: it ignores the proposal entirely, so it is the same
    number for every temperature and every truncation of an arm with the same residue
    class. It moves `Task.floor`, and a reduction computed against one floor may never be
    compared with one computed against the other."""
    A, B = panel.codes[sup], panel.codes[tgt_g]
    col_site, col_res = _columns(panel, A, residues)
    best = np.rint(M).astype(np.float64).min(axis=0)
    Mi = np.rint(M).astype(np.float64)
    for q, b in zip(col_site.tolist(), col_res.tolist()):
        # A parent already carrying `b` at `q` has no substitution to make there, so it
        # contributes no candidate and must not lower the floor.
        ok = A[:, q] != b
        if not ok.any():
            continue
        R = (Mi[ok] + (A[ok, q][:, None] == B[None, :, q])
             - (b == B[:, q])[None, :])
        np.minimum(best, R.min(axis=0), out=best)
    return float(np.asarray(tgt_w, dtype=np.float64) @ best)


# ── the gate ────────────────────────────────────────────────────────────────

def descent(cand, x, M, tgt_w):
    """The `mu = 0` derivative of the transport cost, and the shares behind it.

    One transport solve, no fit and no sweep. `phi` is the c-transform of the optimal
    target potential, so `phi_w - phi_v` lies in `{-1, 0, +1}` for a one-step neighbour
    under a Hamming cost and the derivative is `p_up - p_down = 1 - p_flat - 2 p_down`.

    **This reads the transition's own targets**, so it is an oracle diagnostic: computing
    it on a transition before that transition has been scored would be selection on the
    test set, which is what `season_momentum` gates by refusing it outright."""
    x = np.asarray(x, dtype=np.float64)
    x = x / x.sum()
    W, _, z = emd(x, tgt_w, M, duals=True)
    phi_v = (M - z[None, :]).min(axis=1)
    if len(cand) == 0:
        return dict(emd=W, p_down=0.0, p_flat=0.0, deriv=float("nan"),
                    break_even=float("nan"), greedy=float("nan"),
                    x_with_a_downhill=0.0, n_cand=0, moved=0.0)
    phi_w = (cand.cost.astype(np.float64) - z[None, :]).min(axis=1)
    d = phi_w[cand.row] - phi_v[cand.parent]
    w = x[cand.parent] * cand.pi
    tot = float(w.sum())
    p_down = float(w[d < -0.5].sum()) / tot if tot > 0 else 0.0
    p_flat = float(w[np.abs(d) <= 0.5].sum()) / tot if tot > 0 else 0.0
    # The greedy arm: point mass on the single best direction per parent. A parent with
    # no candidate at all keeps its mass and so contributes nothing, which is zero rather
    # than the `inf` the accumulator starts at.
    best = np.full(cand.n_parents, np.inf)
    np.minimum.at(best, cand.parent, d)
    best[~np.isfinite(best)] = 0.0
    return dict(
        emd=W, p_down=p_down, p_flat=p_flat,
        deriv=float(w @ d) / tot if tot > 0 else float("nan"),
        break_even=0.5 * (1.0 - p_flat),
        greedy=float(x @ best),
        x_with_a_downhill=float(x[best < -0.5].sum()),
        n_cand=len(cand), moved=tot)


def golden_mu(score, lo=0.0, hi=1.0, tol=1e-3, maxit=40):
    """The `mu` minimising a convex score on `[lo, hi]`, by golden section.

    `W_1((1-mu)x + mu Kx, nu)` is convex in `mu` -- the forecast is affine in it and the
    transport cost convex in the forecast -- so a golden-section search has no local
    minimum to fall into and a handful of transport solves locates the optimum exactly.
    Returns `(mu, score(mu))`, and the endpoints are evaluated so a monotone score comes
    back at the endpoint rather than at an interior point the bracket never left."""
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = float(lo), float(hi)
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = score(c), score(d)
    for _ in range(maxit):
        if b - a < tol:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = score(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = score(d)
    cands = [(score(lo), float(lo)), (score(hi), float(hi)),
             (fc, c), (fd, d), (score(0.5 * (a + b)), 0.5 * (a + b))]
    f, mu = min(cands)
    return mu, f


def gate(stats, break_even=BREAK_EVEN):
    """Which `(arm, tau_k, m)` operating point clears the bar, from the gate statistics.

    `stats` maps `(arm, tau_k, m)` to the mean `p_down` over the training transitions.
    An arm's operating point is its best `p_down`; an arm whose best does not clear
    `break_even` is reported with its number and **never scored on a held-out
    transition**, so it costs the multiple-looks budget nothing. This is the first
    pre-scoring filter in this line of work, and it is the reason the round can close on
    a measurement rather than on a sweep."""
    out = {}
    by_arm = defaultdict(list)
    for (arm, tau_k, m), p in stats.items():
        by_arm[arm].append((p, tau_k, m))
    for arm, rows in by_arm.items():
        p, tau_k, m = max(rows)
        out[arm] = dict(arm=arm, tau_k=float(tau_k), m=int(m), p_down=float(p),
                        passes=bool(p > break_even), break_even=float(break_even),
                        leaky=bool(ARMS[arm]["leaky"]))
    return out


def apply_substitution(seq, site_positions, q, res, panel):
    """`v ⊕ (q,b)` as an actual string, for the tests that brute-force the recursion.

    Never called in the forecasting path: the whole point of the recursion is that no
    candidate sequence is ever materialised."""
    chars = panel.alphabet + [c for c in AA if c not in panel.code]
    out = list(seq)
    out[site_positions[q]] = chars[int(res)]
    return "".join(out)
