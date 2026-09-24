"""antiGen as a *column*, not as a kernel: two scalars per genotype, one coefficient each.

Why this rather than another convolution
----------------------------------------
The 2026-09-23 convolution round put antiGen's forward pass in front of the transport
problem and got a clean negative: the selector chose `mu = 0` at every origin, the
per-transition oracle chose `mu = 0` at every origin, and the panel slope at zero was
+0.720 against a worst case of +1. The `kemp` round then showed the more general
problem. A kernel's realisable gain is `slope x mu`, and `mu` is bounded below by the
grid's resolution: `kemp` cleared its gate at `dW/dmu = -0.1975` and, at the selected
`mu = 0.05`, had about 0.010 AA to offer. It returned +0.041 AA *worse*. Clearing a
descent gate certifies a slope; it does not buy a forecast.

So this module stops moving mass. The forecast keeps the unextended support `V_T`, the
floor does not move, and antiGen enters where `distance_from_consensus` enters -- as one
standardised design column carrying one coefficient. `mlr+mom+agen` is `mlr+mom` plus a
scalar at every combination in the grid.

The trap this design has to clear, and the test that decides it
---------------------------------------------------------------
`distance_from_consensus` was added as a column on 2026-09-23 and **discarded**: it is a
linear function of the one-hot allele design (`R^2` 0.968-0.997 at the eight horizons),
so `mlr` could already express it and the arm added a shrinkage target rather than a
feature. Any antiGen scalar that is a function of `v`'s own residues risks exactly that,
because the transformer's output for `v` is a (smooth, but still) function of `v`'s
sequence and the design already carries an indicator for every observed allele.

Both columns here are therefore **interactions between the model and the population**,
not functions of `v` alone:

`agen` -- the arrival rate. How much mass the *current population* is poised to mutate
    into `v`, under antiGen's own substitution distribution:

        arr_v  =  sum over u in V_T with d(u, v) = 1  of  x_u * pi(q, b | u)

    where `(q, b)` is the single site at which `u` and `v` differ and `b = v_q`. This is
    a property of `v`'s *neighbourhood* in the support -- who is one step away and how
    much weight they carry -- so it cannot be written as a function of `v`'s residues,
    and that is the whole reason it is the primary arm. On the backbone it is non-zero
    for 65-69% of the support carrying 57-82% of the mass, with mean in-degree ~1.2.

`agenf` -- the flow overlap. How much of antiGen's proposal off `v` points where the
    population is already heading:

        flow_v  =  sum over (q, b) of  pi(q, b | v) * f_hat_q(b)

    with `f_hat` the momentum-extrapolated site-allele table `mutation_kernel`'s
    consensus arms use, at the operating point the descent gate settled. Dense where
    `arr` is sparse. It *is* a function of `v` through `pi`, so it is the one of the two
    that could plausibly fail the R^2 test, and it is run beside `arr` rather than
    instead of it for that reason.

**Pre-registered, before any forecast is scored:** each column is regressed on the
family's own one-hot allele design at each *training* horizon. A column whose `R^2`
clears the bar `distance_from_consensus` failed at is not new information, and is
reported with its number and not scored. `feature_gate` is the driver; it fits nothing
and reads no target, so its cost to the multiple-looks budget is zero.

Leakage
-------
Leak-free by the same construction the rest of the round uses. `checkpoint_for` resolves
an origin to the checkpoint whose train/test split is that origin's own horizon and
**raises** when it is absent rather than borrowing a neighbour's; `x_u` is the support's
own pre-horizon weights; `f_hat` reads only sequences dated at or before the horizon.
Nothing here reads the season being forecast.
"""
from __future__ import annotations

import numpy as np

from antiGen.antigen_kernel import checkpoint_for, forward_probs
from antiGen.mutation_kernel import consensus_freq

# The declared settings of the `f_hat` table `agenf` reads. Not new constants: they are
# the operating point the descent gate settled for `kcmom` on the training transitions
# (tau_e 0.25 with the slow window a declared multiple of it, an interior optimum of the
# sweep in both directions). Carried rather than re-chosen, so this module adds no field
# to the selector's grid -- the `lambda_alpha` and `dfc` precedent.
FLOW_TAU_E = 0.25
FLOW_ARM = "kcmom"

# The ridge on the redundancy regression, calibrated rather than guessed. Swept over
# 1e-8..1 on five backbone horizons against two references computed on the same design:
# a Gaussian noise column, which the allele indicators must NOT be able to predict, and
# a column built as `S @ beta`, which they must. 1e-4 is where the two are furthest
# apart -- noise -0.47, spanned +0.80, `distance_from_consensus` +0.69.
REDUNDANCY_RIDGE = 1e-4

# There is deliberately NO absolute bar. With the allele design at rank 124-167 against
# a support of 271-442, an in-sample R^2 is mostly overfitting -- pure noise scores 0.67
# to 0.88 -- and even cross-validated, a column the design exactly spans only reaches
# ~0.80 because holding out a fifth of a rank-deficient fit genuinely loses information.
# So a column is judged against the two anchors measured on its own horizon: it is
# declared redundant when its CV R^2 sits above the midpoint of `null` and `spanned`.
# `REDUNDANCY_ANCHORS` is how many draws of each are averaged.
REDUNDANCY_ANCHORS = 5

FEATURES = ("agen", "agenf")


def _probs(panel, horizon, season):
    """antiGen's `(n_genotypes, L, E)` map for every genotype the panel holds.

    Computed over the whole genotype list rather than the horizon's support, and
    memoised per origin: the two columns and every combination in the selector's grid
    ask for the same forward pass, and a support is a subset of the list. One pass per
    origin, ~1,063 sequences on the backbone."""
    memo = getattr(panel, "_agen_probs", None)
    if memo is None:
        memo = panel._agen_probs = {}
    key = int(season)
    if key in memo:
        return memo[key]
    root = getattr(panel, "agen_root", None)
    if not root:
        raise ValueError(
            "the antiGen feature columns need --antigen-checkpoints, the directory of "
            "per-origin checkpoints laid out as season_<S>/fitness_*.pt")
    ckpt = checkpoint_for(root, season)
    probs, alphabet = forward_probs(panel.genotypes, ckpt,
                                    device=getattr(panel, "agen_device", None))
    memo[key] = (probs, alphabet)
    return memo[key]


def _model_code(panel, alphabet):
    """Panel residue code -> column of antiGen's alphabet, or -1 where it has none."""
    out = np.full(len(panel.alphabet), -1, dtype=np.int64)
    at = {c: i for i, c in enumerate(alphabet)}
    for i, c in enumerate(panel.alphabet):
        if c in at:
            out[i] = at[c]
    return out


def arrival(panel, sup, x, horizon, season):
    """`arr_v`: the mass the current population is poised to mutate into `v`.

    One all-pairs Hamming scan over the support to find the distance-one pairs, then one
    lookup into antiGen's map per ordered pair. A genotype with no in-support neighbour
    scores exactly zero, which is a statement about it rather than missing data: nothing
    the population currently carries is one substitution away.

    `x` is the source measure over `sup` -- whole-season persistence, the same weights
    `Task.dfc` standardises under, so the column reads no hyperparameter and the family
    collapses over its parent's grid."""
    probs, alphabet = _probs(panel, horizon, season)
    code_of = _model_code(panel, alphabet)
    sites = np.asarray(panel.sites, dtype=np.int64)
    C = panel.codes[sup]
    n = len(sup)
    out = np.zeros(n)
    if n < 2:
        return out
    # Distance-one ordered pairs (u -> v). The scan is the same shape as
    # `empirical_counts`'s and is bounded by |sup|^2 x |sites|, which is 442^2 x 313 at
    # the largest backbone horizon.
    D = (C[:, None, :] != C[None, :, :]).sum(2)
    uu, vv = np.nonzero(D == 1)
    if not len(uu):
        return out
    q = np.argmax(C[uu] != C[vv], axis=1)              # the one differing varying site
    b = code_of[C[vv, q]]                              # v's residue there, in model codes
    ok = b >= 0
    p = np.zeros(len(uu))
    p[ok] = probs[np.asarray(sup)[uu[ok]], sites[q[ok]], b[ok]]
    np.add.at(out, vv, x[uu] * p)
    return out


def flow(panel, sup, horizon, season, tau_e=FLOW_TAU_E, arm=FLOW_ARM):
    """`flow_v`: the overlap between antiGen's proposal off `v` and where the population
    is heading.

    `sum_(q,b) pi(q,b|v) f_hat_q(b)`, restricted to the panel's varying sites and to
    residues it has a code for. Mass antiGen spends on invariant positions or on
    characters the panel never observed contributes nothing rather than being
    renormalised away: the model spent it, and pretending otherwise would flatter the
    column. That is the same choice `antigen_kernel.candidates` makes by default."""
    probs, alphabet = _probs(panel, horizon, season)
    code_of = _model_code(panel, alphabet)
    fh = consensus_freq(panel, horizon, arm, tau_e)     # (n_varying_sites, |alphabet|)
    sites = np.asarray(panel.sites, dtype=np.int64)
    keep = np.flatnonzero(code_of >= 0)
    # (n_sites, |alphabet_model|) table of f_hat laid out on antiGen's residue columns.
    F = np.zeros((len(sites), probs.shape[2]))
    F[:, code_of[keep]] = fh[:, keep]
    P = probs[np.asarray(sup)][:, sites, :]             # (n, n_sites, E)
    return np.einsum("nqe,qe->n", P, F)


def column(panel, sup, x, horizon, season, which):
    """One standardised design column, ready to stack beside the one-hot alleles.

    Standardised under the same weights, for the reason `Task.dfc` gives: the column
    joins a design whose other entries are one-hots and a centred log-ratio, so a raw
    scale would make the weak `gamma_pen` mean something different on it than on
    momentum. `arr` is put through `log1p` first -- it is a sum of probability-weighted
    masses spanning orders of magnitude with a third of the support at exactly zero, and
    the untransformed column would be a near-indicator of its own largest entry."""
    if which == "agen":
        v = np.log1p(arrival(panel, sup, x, horizon, season) / _scale(x))
    elif which == "agenf":
        v = flow(panel, sup, horizon, season)
    else:
        raise ValueError(f"unknown antiGen feature {which!r}; have {FEATURES}")
    m = float(np.average(v, weights=x)) if x.sum() > 0 else float(v.mean())
    sd = float(np.sqrt(np.average((v - m) ** 2, weights=x))) if x.sum() > 0 else \
        float(v.std())
    return (v - m) / sd if sd > 0 else v - m


def _scale(x):
    """The unit `arr` is measured in before `log1p`: the mean source weight.

    `log1p` is not scale-free, so applying it to a quantity whose typical size is set by
    `1/|sup|` would make the transform's curvature depend on how big the support is, and
    the column would not mean the same thing at two horizons. Dividing by the mean
    weight first makes `arr = 1` mean "one average genotype's worth of arrival"
    everywhere."""
    n = len(x)
    return float(x.sum() / n) if n and x.sum() > 0 else 1.0


def redundancy(S, col, w=None, folds=5, seed=0, lam=REDUNDANCY_RIDGE):
    """How much of `col` the one-hot allele design can already express.

    Returns `(cv_r2, in_r2, n_eff)`. **`cv_r2` is the number the gate reads.**

    The test `distance_from_consensus` failed, but computed the way it has to be to mean
    anything. The allele design carries 239 columns against a support of 271-442, with
    rank 124-167, so an *in-sample* R^2 is mostly overfitting: measured on the backbone,
    a column of pure Gaussian noise scores **0.67 to 0.88** against this design. Read
    against 1.0, that makes every column look redundant; read against its own null, it
    says almost nothing. So the gate uses K-fold cross-validated R^2 instead, which a
    column outside the span cannot fake -- noise scores about zero, and a column the
    indicators genuinely span still scores ~1 because the relationship is real and
    holds out of fold.

    This does not overturn the `dfc` finding: 0.968-0.997 in sample sits well above that
    0.67-0.88 null, and the structural argument (dfc is exactly `n_sites` minus the sum
    of the consensus indicators) says it should be ~1 out of fold too. It does mean the
    published numbers were flattered by the design's rank, and that a *new* column must
    not be judged against the same scale.

    Least squares through a weighted normal equation with a small ridge: `S` is rank
    deficient by construction -- the indicators at each site sum to one -- so the
    unregularised system is singular, and the fitted values, which are all this needs,
    are unaffected by breaking the tie."""
    A = np.asarray(S.todense() if hasattr(S, "todense") else S, dtype=np.float64)
    y = np.asarray(col, dtype=np.float64)
    n = len(y)
    w = np.ones(n) / n if w is None else np.asarray(w, dtype=np.float64) / np.sum(w)
    A = np.hstack([A, np.ones((n, 1))])

    def _fit(rows):
        sw = np.sqrt(w[rows])
        Aw, yw = A[rows] * sw[:, None], y[rows] * sw
        G = Aw.T @ Aw
        # The intercept is NOT penalised. Shrinking it toward zero would make the fit
        # depend on where the column's origin happens to sit, so `redundancy(col)` and
        # `redundancy(col + c)` would differ -- which they must not, since R^2 is a
        # statement about shape. `test_redundancy_is_invariant_to_affine_rescaling`
        # fails on the penalised version.
        G[np.diag_indices_from(G)] += lam * scale * pen
        return np.linalg.solve(G, Aw.T @ yw)

    scale = max(float(np.trace(A.T @ A)) / A.shape[1], 1e-12)
    pen = np.ones(A.shape[1])
    pen[-1] = 0.0
    ybar = float(w @ y)
    ss_tot = float(w @ (y - ybar) ** 2)
    if ss_tot <= 0:
        return 1.0, 1.0, n

    beta = _fit(np.arange(n))
    in_r2 = 1.0 - float(w @ (y - A @ beta) ** 2) / ss_tot

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    pred = np.empty(n)
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test, assume_unique=False)
        pred[test] = A[test] @ _fit(train)
    cv_r2 = 1.0 - float(w @ (y - pred) ** 2) / ss_tot
    return cv_r2, in_r2, n


def anchors(S, w=None, n_draws=REDUNDANCY_ANCHORS, seed=0, **kw):
    """The two references a column's CV `R^2` is read against, on its own design.

    `null` -- Gaussian noise. The allele indicators must not be able to predict it, so
        this is where a column carrying information the design does not have should sit.
    `spanned` -- a column built as `S @ beta`, which the indicators span exactly. This
        is where `distance_from_consensus` belongs and where a redundant column sits.

    Both are needed because neither 0 nor 1 is the right reference: the design is rank
    deficient, so out-of-fold recovery of even an exactly-spanned column is attenuated,
    and the attenuation depends on the horizon's support size. Measuring both on the
    same design is what makes the comparison mean something."""
    A = np.asarray(S.todense() if hasattr(S, "todense") else S, dtype=np.float64)
    n, k = A.shape
    rng = np.random.default_rng(seed)
    null = [redundancy(S, rng.standard_normal(n), w=w, **kw)[0] for _ in range(n_draws)]
    span = [redundancy(S, A @ rng.standard_normal(k) + 2.0, w=w, **kw)[0]
            for _ in range(n_draws)]
    return float(np.mean(null)), float(np.mean(span))
