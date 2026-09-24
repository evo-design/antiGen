"""antiGen as a design column: the arrival graph, the redundancy test, and the leak guard.

The tests that exist because something could go wrong quietly:

* `arr` is claimed to be a *neighbourhood* quantity, which is the whole reason to expect
  it to clear the redundancy bar `distance_from_consensus` failed. If it were secretly a
  function of `v`'s own residues the arm would be the `dfc` round again with a slower
  forward pass, so that property is asserted directly rather than argued.
* `redundancy` is the gate. It has to return 1 for a column that *is* in the design's
  span, on a design that is rank deficient by construction, which is the case a naive
  `lstsq` gets wrong or refuses.
* The columns must read nothing dated after the horizon, the same guarantee the kernel
  arms carry and the same test.
"""
import numpy as np
import pytest

from antiGen.antigen_feature import (FEATURES, REDUNDANCY_RIDGE, anchors,
                                     arrival, column, redundancy)
from antiGen.season_momentum import (AGEN_FAMILIES, RELEVANT, Task, TipPanel,
                                     design_for, make_grid)

TIPS = [
    (2000.80, "AAAAA"), (2000.85, "AAAAA"), (2000.90, "CAAAA"),
    (2001.05, "GAAAA"),
    (2001.10, "ACAAA"), (2001.20, "ACAAA"), (2001.22, "ACCAA"),
    (2001.80, "ACCAA"), (2001.90, "ACCGA"), (2002.10, "ACCGA"), (2002.20, "GCCGA"),
    (2002.80, "GCCGA"), (2002.90, "GCCGC"), (2003.10, "GCCGC"), (2003.20, "GDCGC"),
    (2003.80, "GDCGC"), (2003.90, "GDCGW"), (2004.10, "GDCGW"), (2004.20, "GDCGW"),
]


@pytest.fixture
def panel():
    return TipPanel(TIPS, min_allele_freq=0.0, lookback=5.0)


def _fake_probs(panel, rng=None, flat=False):
    """A stand-in for antiGen's forward pass, shaped exactly as `forward_probs` returns.

    Random rather than real: what these tests pin is the indexing, the graph and the
    standardisation, and a real checkpoint would make the suite need a GPU and a 28 MB
    model to assert identities that hold for any table. The one real forward pass this
    module depends on is asserted equal to `antiGen.infer` in `test_antigen_kernel`."""
    rng = np.random.default_rng(0) if rng is None else rng
    alphabet = list(panel.alphabet) + ["*", "-"]
    L, E = panel.seq_len, len(alphabet)
    n = len(panel.genotypes)
    if flat:
        p = np.ones((n, L, E))
    else:
        p = rng.random((n, L, E)) + 0.01
    # zero the wild-type cell and renormalise over the whole map, as forward_probs does
    at = {c: i for i, c in enumerate(alphabet)}
    for i, g in enumerate(panel.genotypes):
        for q, c in enumerate(g):
            p[i, q, at[c]] = 0.0
    p /= p.sum(axis=(1, 2), keepdims=True)
    return p, alphabet


def _install(panel, **kw):
    panel._agen_probs = {s: _fake_probs(panel, **kw) for s in panel.seasons}
    panel.agen_root = "unused -- probs are pre-installed"
    return panel


# ── the arrival graph ───────────────────────────────────────────────────────

def test_arrival_is_the_in_support_one_step_inflow(panel):
    """Recomputed the slow way: every ordered support pair at Hamming distance one,
    materialised, with the differing site and residue looked up by hand."""
    _install(panel)
    t = Task(panel, 2001)
    x = t.baseline("season")
    x = x if x.sum() > 0 else t.level(1 / 12)
    got = arrival(panel, t.sup, x, t.horizon, t.season)

    probs, alphabet = panel._agen_probs[t.season]
    at = {c: i for i, c in enumerate(alphabet)}
    seqs = [panel.genotypes[g] for g in t.sup]
    want = np.zeros(len(t.sup))
    for iu, u in enumerate(seqs):
        for iv, v in enumerate(seqs):
            diff = [k for k in range(len(u)) if u[k] != v[k]]
            if len(diff) != 1:
                continue
            q = diff[0]
            want[iv] += x[iu] * probs[t.sup[iu], q, at[v[q]]]
    assert np.allclose(got, want)
    assert got.sum() > 0, "fixture has no distance-one support pairs to test"


def test_arrival_is_zero_for_a_genotype_with_no_in_support_neighbour(panel):
    """An isolated genotype scores exactly zero, and that is a statement about it --
    nothing the population carries is one substitution away -- not missing data."""
    _install(panel)
    t = Task(panel, 2001)
    x = t.baseline("season")
    x = x if x.sum() > 0 else t.level(1 / 12)
    got = arrival(panel, t.sup, x, t.horizon, t.season)
    C = panel.codes[t.sup]
    D = (C[:, None, :] != C[None, :, :]).sum(2)
    isolated = np.flatnonzero((D == 1).sum(0) == 0)
    assert np.all(got[isolated] == 0.0)


def test_arrival_depends_on_the_neighbourhood_and_not_on_the_genotype_alone(panel):
    """The property the whole arm rests on.

    Two panels whose supports contain the same genotype `v` with the same antiGen map,
    differing only in how much mass sits on `v`'s one-step neighbours. `arr_v` must
    move. A column that is a function of `v`'s residues -- which is what
    `distance_from_consensus` turned out to be -- cannot do this, and that is why it
    failed the redundancy test and why this one might not."""
    _install(panel)
    t = Task(panel, 2001)
    x = t.baseline("season")
    x = x if x.sum() > 0 else t.level(1 / 12)
    C = panel.codes[t.sup]
    D = (C[:, None, :] != C[None, :, :]).sum(2)
    v = int(np.argmax((D == 1).sum(0)))
    nbr = np.flatnonzero(D[:, v] == 1)
    assert len(nbr) > 0

    base = arrival(panel, t.sup, x, t.horizon, t.season)[v]
    y = x.copy()
    y[nbr[0]] *= 3.0                      # the neighbour's mass, not v's own
    bumped = arrival(panel, t.sup, y, t.horizon, t.season)[v]
    assert not np.isclose(base, bumped)

    z = x.copy()
    z[v] *= 3.0                           # v's own mass: arr_v must NOT move
    same = arrival(panel, t.sup, z, t.horizon, t.season)[v]
    assert np.isclose(base, same)


def test_a_flat_antigen_map_makes_arrival_a_pure_neighbour_mass_count(panel):
    """With a uniform proposal the model contributes nothing and `arr` collapses to the
    weighted in-degree, which is the sanity check that the probability lookup is not
    silently constant in the real case."""
    _install(panel, flat=True)
    t = Task(panel, 2001)
    x = t.baseline("season")
    x = x if x.sum() > 0 else t.level(1 / 12)
    got = arrival(panel, t.sup, x, t.horizon, t.season)
    C = panel.codes[t.sup]
    D = (C[:, None, :] != C[None, :, :]).sum(2)
    probs, _ = panel._agen_probs[t.season]
    # every non-wild-type cell carries the same probability under a flat map
    p = probs[t.sup[0]][probs[t.sup[0]] > 0][0]
    assert np.allclose(got, (D == 1).T @ x * p)


# ── the redundancy gate ─────────────────────────────────────────────────────

def test_redundancy_scores_a_spanned_column_above_a_noise_column(panel):
    """The gate's only real requirement: it must separate the two known cases.

    Absolute values are not asserted. The allele design is rank deficient, so an
    exactly-spanned column cannot be recovered perfectly out of fold and its CV R^2 is
    attenuated by an amount that depends on the support size -- which is exactly why
    the gate compares against anchors measured on the same design rather than a fixed
    bar."""
    rng = np.random.default_rng(3)
    t = Task(panel, 2001)
    S = np.asarray(t.S.todense())
    spanned = S @ rng.standard_normal(S.shape[1]) + 2.0
    noise = rng.standard_normal(len(t.sup))
    assert redundancy(t.S, spanned)[0] > redundancy(t.S, noise)[0]


def test_the_in_sample_r2_is_the_one_that_overfits(panel):
    """Recorded because it is the trap the first version of this gate fell into: with
    239 allele columns against 271-442 genotypes, in-sample R^2 of a *noise* column
    comes back at 0.67-0.88 on the real panel. The CV number must be far lower."""
    rng = np.random.default_rng(5)
    t = Task(panel, 2001)
    cv, insample, _ = redundancy(t.S, rng.standard_normal(len(t.sup)))
    assert insample > cv


def test_anchors_bracket_and_are_ordered():
    """On a design with real headroom. The five-site fixture saturates -- its design
    has as many columns as the support has rows -- so both anchors sit at 1 there and
    the test would pass or fail on numerical noise. The gate's own output carries the
    anchors per horizon precisely because this depends on the support size."""
    rng = np.random.default_rng(11)
    res = "ACGT"
    tips = [(2000.0 + 0.004 * i,
             "".join(res[int(c)] for c in rng.integers(0, 4, 12)))
            for i in range(400)]
    p = TipPanel(tips, min_allele_freq=0.0, lookback=50.0)
    t = Task(p, 2001)
    assert len(t.sup) > np.linalg.matrix_rank(np.asarray(t.S.todense())) + 5
    null, spanned = anchors(t.S, n_draws=3)
    assert null < spanned


def test_redundancy_is_invariant_to_affine_rescaling_of_the_column(panel):
    rng = np.random.default_rng(4)
    t = Task(panel, 2001)
    col = rng.standard_normal(len(t.sup))
    a = redundancy(t.S, col)[0]
    b = redundancy(t.S, 3.5 * col - 2.0)[0]
    assert a == pytest.approx(b, abs=1e-6)


def test_the_ridge_is_the_calibrated_one():
    """Swept on five backbone horizons against both anchors; 1e-4 maximised their
    separation (noise -0.47, spanned +0.80). Pinned so a later change is deliberate."""
    assert REDUNDANCY_RIDGE == 1e-4


# ── the columns, standardisation, and the wiring ────────────────────────────

@pytest.mark.parametrize("which", FEATURES)
def test_a_column_is_standardised_under_the_population_weights(panel, which):
    _install(panel)
    t = Task(panel, 2001)
    x = t.baseline("season")
    x = x if x.sum() > 0 else t.level(1 / 12)
    col = column(panel, t.sup, x, t.horizon, t.season, which)
    assert len(col) == len(t.sup)
    assert np.isfinite(col).all()
    assert float(np.average(col, weights=x)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.sqrt(np.average(col ** 2, weights=x))) == pytest.approx(1.0, abs=1e-9)


def test_the_column_reads_nothing_dated_after_the_horizon(panel):
    """Same guarantee and same test as `kemp` and the consensus arms.

    The antiGen map is held fixed across the variants -- it is keyed by origin and comes
    from a checkpoint trained at that cutoff, which is a separate leak guard tested in
    `test_antigen_kernel` -- so what this pins is the *other* input, the population
    weights and the support the column is built over."""
    horizon_season = 2001
    ref = TipPanel(TIPS, min_allele_freq=0.0, lookback=5.0)
    _install(ref)
    t = Task(ref, horizon_season)
    x = t.baseline("season")
    x = x if x.sum() > 0 else t.level(1 / 12)
    base = arrival(ref, t.sup, x, t.horizon, t.season)

    late = [tp for tp in TIPS if tp[0] > t.horizon]
    assert len(late) > 1
    for variant in ([tp for tp in TIPS if tp != late[0]],
                    [(tp[0] + 0.01, tp[1]) if tp == late[0] else tp for tp in TIPS],
                    TIPS + [(2003.85, "GDCGC")]):
        p = TipPanel(variant, min_allele_freq=0.0, lookback=5.0)
        assert p.alphabet == ref.alphabet and p.sites == ref.sites
        _install(p)
        tv = Task(p, horizon_season)
        xv = tv.baseline("season")
        xv = xv if xv.sum() > 0 else tv.level(1 / 12)
        assert np.array_equal(tv.sup, t.sup)
        assert np.allclose(arrival(p, tv.sup, xv, tv.horizon, tv.season), base)


def test_the_families_add_one_column_and_no_grid_field(panel):
    """The `dfc` contract: at every combination the family is its parent plus a scalar,
    and it collapses over the identical candidate list -- so the paired delta against
    the parent is not confounded with how much the selector had to choose among."""
    _install(panel)
    t = Task(panel, 2001)
    grid = make_grid((1.0,), (3.0,), (1.0,), (2,))
    combo = grid[0]
    for fam, parent in (("mlr+mom+agen", "mlr+mom"), ("mlr+mom+agenf", "mlr+mom"),
                        ("mlr+agen", "mlr"), ("mlr+agenf", "mlr")):
        Z, _, n_mom, _ = design_for(t, combo, fam)
        Zp, _, n_mom_p, _ = design_for(t, combo, parent)
        assert Z.shape[1] == Zp.shape[1] + 1, fam
        assert n_mom == n_mom_p + 1, fam
        # The momentum-keeping arms see exactly their parent's fields. The
        # momentum-dropping ones gain `gamma_pen`, because the new column is the one
        # thing carrying that weak penalty once momentum is gone -- the same asymmetry
        # `mlr+dfc` has against `mlr`, and it is copied from there deliberately.
        extra = () if "mom" in fam else ("gamma_pen",)
        assert RELEVANT[fam] == RELEVANT[parent] + extra, fam


def test_every_antigen_family_is_registered():
    assert AGEN_FAMILIES == ("mlr+agen", "mlr+mom+agen", "mlr+agenf", "mlr+mom+agenf")
    assert FEATURES == ("agen", "agenf")
