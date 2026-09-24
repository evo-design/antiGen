"""The season-to-season earth-mover's objective: cost matrix, duals, and the gradient."""
from collections import Counter

import numpy as np
import pytest

from antiGen.season_emd import (Pair, SeasonPanel, emd, emd_loss_grad, fit, predict,
                                summarise)

# Four positions, of which 1 and 3 vary. Season 2001 holds three genotypes, 2002 holds
# three, and the two supports overlap in only one of them -- the case the objective exists
# for.
SEASONS = {
    2001: Counter({"AAAA": 6, "ACAA": 3, "AAAG": 1}),
    2002: Counter({"ACAA": 5, "ACAG": 4, "AGAA": 1}),
    2003: Counter({"ACAG": 7, "AGAG": 3}),
}


@pytest.fixture
def panel():
    return SeasonPanel(SEASONS, min_allele_freq=0.0)


def test_only_varying_sites_are_features(panel):
    assert panel.sites == [1, 3]
    assert set(panel.features) == {(1, "A"), (1, "C"), (1, "G"), (3, "A"), (3, "G")}
    assert panel.n_sites == 2


def test_weights_are_within_season_frequencies(panel):
    w = dict(zip(panel.genotypes[2001], panel.w[2001]))
    assert w["AAAA"] == pytest.approx(0.6)
    assert w["ACAA"] == pytest.approx(0.3)
    assert w["AAAG"] == pytest.approx(0.1)
    assert sum(panel.w[2001]) == pytest.approx(1.0)


def test_cost_matrix_is_hamming_over_the_full_sequence(panel):
    pr = Pair(panel, 2001, 2002)
    for i, g in enumerate(panel.genotypes[2001]):
        for j, h in enumerate(panel.genotypes[2002]):
            assert pr.M[i, j] == sum(x != y for x, y in zip(g, h))


def test_invariant_sites_do_not_change_distances():
    """Padding every genotype with a constant column must leave the cost matrix alone."""
    padded = {t: Counter({("Z" * 5) + g: m for g, m in c.items()})
              for t, c in SEASONS.items()}
    a = Pair(SeasonPanel(SEASONS, 0.0), 2001, 2002)
    b = Pair(SeasonPanel(padded, 0.0), 2001, 2002)
    assert np.array_equal(a.M, b.M)


def test_zero_fitness_is_the_persistence_baseline(panel):
    pr = Pair(panel, 2001, 2002)
    assert np.allclose(predict(np.zeros(len(panel.features)), pr), pr.w_cur)


def test_a_per_position_constant_does_not_change_the_prediction(panel):
    """Every genotype carries one residue per position, so such a shift cancels in the
    ratio; the prediction must be invariant to it even though beta is not."""
    pr = Pair(panel, 2001, 2002)
    rng = np.random.default_rng(0)
    beta = rng.standard_normal(len(panel.features))
    shifted = beta.copy()
    for p in {p for p, _ in panel.features}:
        c = rng.standard_normal()
        for k, (q, _) in enumerate(panel.features):
            if q == p:
                shifted[k] += c
    assert np.allclose(predict(beta, pr), predict(shifted, pr))


def test_emd_against_an_exact_small_case(panel):
    """A case whose optimal plan can be written down: identical distributions cost zero,
    and moving all mass one substitution costs exactly one."""
    same = Pair(panel, 2002, 2002)
    w, _ = emd(same.w_cur, same.w_next, same.M)
    assert w == pytest.approx(0.0, abs=1e-12)
    # one genotype to one genotype, one substitution apart
    M = np.array([[1.0]])
    w2, _ = emd(np.array([1.0]), np.array([1.0]), M)
    assert w2 == pytest.approx(1.0)


def test_floor_is_the_best_any_reweighting_could_do(panel):
    """The floor sends each next-season genotype to its nearest season-i neighbour; no
    distribution supported on season i can beat it, and a search confirms it here."""
    pr = Pair(panel, 2001, 2002)
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = rng.random(len(pr.w_cur))
        a /= a.sum()
        w, _ = emd(a, pr.w_next, pr.M)
        assert w >= pr.floor - 1e-9


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0])
def test_gradient_matches_central_differences(panel, scale):
    """The Danskin gradient through the optimal dual, which is the one thing here that
    can be silently wrong: the duals are only defined up to a constant, and the gradient
    is only well posed because it is a covariance under xhat."""
    pairs = [Pair(panel, 2001, 2002), Pair(panel, 2002, 2003)]
    D = len(panel.features)
    rng = np.random.default_rng(2)
    beta = scale * rng.standard_normal(D)
    _, g = emd_loss_grad(beta, pairs, lam=0.3)
    eps = 1e-6
    for k in range(D):
        d = np.zeros(D)
        d[k] = eps
        hi, _ = emd_loss_grad(beta + d, pairs, lam=0.3)
        lo, _ = emd_loss_grad(beta - d, pairs, lam=0.3)
        assert g[k] == pytest.approx((hi - lo) / (2 * eps), rel=1e-4, abs=1e-7)


def test_dual_shift_leaves_the_gradient_alone(panel):
    """Adding a constant to the dual potential is a different valid optimum; the gradient
    must not move, or the fit would depend on the solver's arbitrary normalisation."""
    pr = Pair(panel, 2001, 2002)
    beta = np.zeros(len(panel.features))
    xhat = predict(beta, pr)
    _, u = emd(xhat, pr.w_next, pr.M)
    S = pr.S
    def grad(uu):
        return (S.T @ (xhat * uu)) - float(xhat @ uu) * (S.T @ xhat)
    assert np.allclose(grad(u), grad(u + 7.5))


def test_fitting_reduces_the_objective_in_sample(panel):
    pairs = [Pair(panel, 2001, 2002), Pair(panel, 2002, 2003)]
    D = len(panel.features)
    start, _ = emd_loss_grad(np.zeros(D), pairs, lam=0.0)
    beta = fit(pairs, 0.0, D, maxiter=200)
    end, _ = emd_loss_grad(beta, pairs, lam=0.0)
    assert end <= start + 1e-12


def test_genotype_cap_reports_the_mass_it_drops():
    p = SeasonPanel(SEASONS, 0.0, max_genotypes=2)
    assert len(p.genotypes[2001]) == 2
    assert p.dropped_mass[2001] == pytest.approx(0.1)      # the singleton "AAAG"
    assert sum(p.w[2001]) == pytest.approx(1.0)


def test_summarise_counts_transitions_not_genotypes():
    rows = [dict(reduction=0.2, captured=0.5, emd_base=1.0, emd_model=0.8, emd_floor=0.6),
            dict(reduction=-0.1, captured=-0.2, emd_base=1.0, emd_model=1.1, emd_floor=0.5)]
    s = summarise(rows, n_boot=500)
    assert s["n"] == 2
    assert s["mean_reduction"] == pytest.approx(0.05)
    assert s["n_better"] == 1
