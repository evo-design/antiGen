"""The time-weighted predictor: tip reconstruction, decay weights, momentum, gradient."""
import json

import numpy as np
import pytest
from scipy import sparse

from antiGen.season_momentum import (MONTH, Task, TipPanel, base_grid, fit, load_tips,
                                     loss_grad, make_grid, penalty, predict, train_set)

# Two seasons of dated tips. Season 2001 is Oct 2000 - Mar 2001, season 2002 the year
# after. Positions 1 and 3 vary. "ACAA" arrives late in season 2001 and is the only thing
# seen in the last month, which is what makes the decay timescales distinguishable.
TIPS = [
    (2000.80, "AAAA"), (2000.85, "AAAA"), (2000.90, "AAAA"),
    (2001.10, "AAAG"), (2001.20, "ACAA"), (2001.24, "ACAA"),
    (2001.60, "ACAA"),                                  # out of season (summer)
    (2001.80, "ACAA"), (2001.90, "ACAG"), (2002.10, "ACAG"), (2002.20, "ACAG"),
]


@pytest.fixture
def panel():
    return TipPanel(TIPS, min_allele_freq=0.0, lookback=5.0)


def test_tips_are_reconstructed_from_the_tree(tmp_path):
    """A tip's genotype is the reference with the mutations on its path applied, and a
    sibling must not see its brother's mutations -- the undo on the way back up."""
    ref = {"SigPep": "MM", "HA1": "AAA", "HA2": "CC", "nuc": "x" * 21}
    tree = {"tree": {
        "name": "root", "branch_attrs": {"mutations": {"HA1": ["A1G"]}},
        "children": [
            {"name": "t1", "branch_attrs": {"mutations": {"HA1": ["A2T"]}},
             "node_attrs": {"num_date": {"value": 2001.1}}},
            {"name": "t2", "branch_attrs": {"mutations": {"HA2": ["C1W"]}},
             "node_attrs": {"num_date": {"value": 2002.1}}},
        ]}}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "tree.json").write_text(json.dumps(tree))
    tips, skipped = load_tips(str(tmp_path / "tree.json"), str(tmp_path / "ref.json"))
    assert skipped == 0
    got = dict((d, g) for d, g in tips)
    # SigPep MM, then HA1 with A1G on both paths, then HA2.
    assert got[2001.1] == "MM" + "GTA" + "CC"          # t1 also carries A2T
    assert got[2002.1] == "MM" + "GAA" + "WC"          # t2 does NOT carry A2T
    assert "nuc" not in "".join(got.values())


def test_undated_tips_are_counted_not_guessed(tmp_path):
    ref = {"SigPep": "M", "HA1": "A", "HA2": "C"}
    tree = {"tree": {"name": "root", "children": [
        {"name": "a", "node_attrs": {"num_date": {"value": 2001.0}}},
        {"name": "b", "node_attrs": {}}]}}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "tree.json").write_text(json.dumps(tree))
    tips, skipped = load_tips(str(tmp_path / "tree.json"), str(tmp_path / "ref.json"))
    assert len(tips) == 1 and skipped == 1


def test_out_of_season_tips_are_kept_for_weights_but_are_not_a_season(panel):
    assert (panel.season < 0).sum() == 1                # the 2001.60 summer sample
    g, w = panel.target(2001)
    assert w.sum() == pytest.approx(1.0)
    # the summer sequence is not part of any season's observed distribution
    total_in_season = int((panel.season > 0).sum())
    assert total_in_season == len(TIPS) - 1


def test_decay_weights_favour_recent_sequences(panel):
    task = Task(panel, 2001)
    fast = panel.decayed(task.horizon, 0.5 * MONTH, task.sup, task.mask)
    slow = panel.decayed(task.horizon, np.inf, task.sup, task.mask)
    pos = {int(g): k for k, g in enumerate(task.sup)}
    i_acaa = pos[panel.gid["ACAA"]]
    i_aaaa = pos[panel.gid["AAAA"]]
    # ACAA is the most recent thing before 1 April 2001; AAAA is the oldest.
    assert fast[i_acaa] > slow[i_acaa]
    assert fast[i_aaaa] < slow[i_aaaa]
    assert fast.sum() == pytest.approx(1.0)


def test_infinite_tau_is_uniform_over_the_lookback(panel):
    task = Task(panel, 2001)
    w = panel.decayed(task.horizon, np.inf, task.sup, task.mask)
    pos = {int(g): k for k, g in enumerate(task.sup)}
    # six sequences at or before 2001.25: three AAAA, one AAAG, two ACAA
    assert w[pos[panel.gid["AAAA"]]] == pytest.approx(3 / 6)
    assert w[pos[panel.gid["ACAA"]]] == pytest.approx(2 / 6)
    assert w[pos[panel.gid["AAAG"]]] == pytest.approx(1 / 6)


def test_every_genotype_in_support_keeps_positive_weight(panel):
    """The point of a decay rather than a hard window: recency without losing support."""
    task = Task(panel, 2001)
    for tau in (0.1 * MONTH, MONTH, 1.0, np.inf):
        w = panel.decayed(task.horizon, tau, task.sup, task.mask)
        assert (w > 0).all()


def test_hard_windows_do_lose_support(panel):
    task = Task(panel, 2001)
    w = panel.window(task.horizon, MONTH, task.sup, task.mask)
    assert (w == 0).any()                    # that is the trade-off being measured
    assert w.sum() == pytest.approx(1.0)


def test_momentum_is_signed_by_whether_a_share_is_rising(panel):
    task = Task(panel, 2001)
    Z, base = task.design(0.5 * MONTH, np.inf)
    mom = np.asarray(Z[:, -1].todense()).ravel()
    pos = {int(g): k for k, g in enumerate(task.sup)}
    assert mom[pos[panel.gid["ACAA"]]] > 0   # arriving
    assert mom[pos[panel.gid["AAAA"]]] < 0   # fading
    # centred under the base measure, so it is a tilt and not an offset
    assert float(base @ mom) == pytest.approx(0.0, abs=1e-9)


def test_zero_theta_reproduces_the_base_measure(panel):
    task = Task(panel, 2001)
    Z, base = task.design(MONTH, 1.0)
    assert np.allclose(predict(np.zeros(Z.shape[1]), Z, base), base)


def test_penalty_puts_the_weak_penalty_on_the_momentum_columns():
    p = penalty(5, 2, 10.0, 0.01)
    assert list(p) == [10.0, 10.0, 10.0, 0.01, 0.01]
    assert list(penalty(3, 0, 2.0, 0.5)) == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("scale", [0.0, 0.4])
def test_gradient_matches_central_differences(panel, scale):
    tasks = []
    for s in (2001,):
        t = Task(panel, s)
        Z, base = t.design(MONTH, 1.0)
        tasks.append((Z, base, t.tgt_w, t.M))
    D = tasks[0][0].shape[1]
    rng = np.random.default_rng(0)
    theta = scale * rng.standard_normal(D)
    pen = penalty(D, 1, 0.5, 0.01)
    _, g = loss_grad(theta, tasks, pen)
    eps = 1e-6
    for k in range(D):
        d = np.zeros(D)
        d[k] = eps
        hi, _ = loss_grad(theta + d, tasks, pen)
        lo, _ = loss_grad(theta - d, tasks, pen)
        assert g[k] == pytest.approx((hi - lo) / (2 * eps), rel=1e-4, abs=1e-7)


def test_fit_recovers_the_base_measure_as_the_penalty_grows(panel):
    """With everything shrunk to nothing the prediction must be the base measure, which is
    the check that the preconditioning did not stall the whole vector at zero for the
    wrong reason."""
    t = Task(panel, 2001)
    Z, base = t.design(MONTH, 1.0)
    bundles = [(Z, base, t.tgt_w, t.M)]
    theta = fit(bundles, 1e10, 1e10, Z.shape[1], n_mom=1, maxiter=50)
    assert np.allclose(predict(theta, Z, base), base, atol=1e-6)


def test_fit_leaves_momentum_free_when_only_beta_is_penalised(panel):
    t = Task(panel, 2001)
    Z, base = t.design(0.5 * MONTH, np.inf)
    bundles = [(Z, base, t.tgt_w, t.M)]
    theta = fit(bundles, 1e8, 1e-4, Z.shape[1], n_mom=1, maxiter=100)
    assert abs(theta[-1]) > 1e-3              # gamma moved
    assert np.abs(theta[:-1]).max() < 1e-3    # beta did not


def test_train_set_takes_the_most_recent_and_never_the_future():
    tasks = {y: y for y in range(2010, 2020)}
    order = list(range(2010, 2020))
    assert train_set(tasks, order, 5, 0) == order[:5]
    assert train_set(tasks, order, 5, 2) == order[3:5]
    assert train_set(tasks, order, 0, 3) == []


def test_grid_never_lets_the_slow_timescale_undercut_the_fast():
    g = make_grid((1, 6), (1.0, np.inf), (1.0,), (0,))
    for c in g:
        if np.isfinite(c["tau_s"]):
            assert c["tau_s"] > c["tau_f"]


def test_base_grid_covers_every_no_fit_candidate():
    g = base_grid((1, 3))
    kinds = {c["base"] for c in g}
    assert {"season", "last3m", "last1m", "all"} <= kinds
    assert "ewma" in kinds


# ── the multinomial-logit estimator ─────────────────────────────────────────

def test_assignment_is_nearest_neighbour_and_conserves_mass(panel):
    """The MLR target: every next-season sequence's weight lands on its nearest already
    seen genotype, ties split, and nothing is created or lost."""
    from antiGen.season_momentum import Task as T
    t = T(panel, 2001)
    assert t.assigned.sum() == pytest.approx(1.0)
    nearest = t.M.min(axis=0)
    for j in range(t.M.shape[1]):
        hit = np.flatnonzero(t.M[:, j] <= nearest[j] + 1e-12)
        assert len(hit) >= 1
        # no mass may sit on a genotype that is not among the nearest for some target
    reachable = np.zeros(len(t.sup), dtype=bool)
    for j in range(t.M.shape[1]):
        reachable[np.flatnonzero(t.M[:, j] <= nearest[j] + 1e-12)] = True
    assert not (t.assigned[~reachable] > 0).any()


def test_assignment_attains_the_floor(panel):
    """Transporting by the assignment must cost exactly the floor, which is what makes it
    the right surrogate and not an arbitrary matching."""
    from antiGen.season_momentum import Task as T
    t = T(panel, 2001)
    nearest = t.M.min(axis=0)
    assert float(t.tgt_w @ nearest) == pytest.approx(t.floor)


def test_mlogit_gradient_matches_central_differences(panel):
    from antiGen.season_momentum import Task as T, mlogit_loss_grad, penalty as pen_of
    t = T(panel, 2001)
    Z, base = t.design(MONTH, 1.0)
    tasks = [(Z, base, t.assigned)]
    D = Z.shape[1]
    rng = np.random.default_rng(3)
    theta = 0.3 * rng.standard_normal(D)
    pen = pen_of(D, 1, 0.5, 0.01)
    _, g = mlogit_loss_grad(theta, tasks, pen)
    eps = 1e-6
    for k in range(D):
        d = np.zeros(D)
        d[k] = eps
        hi, _ = mlogit_loss_grad(theta + d, tasks, pen)
        lo, _ = mlogit_loss_grad(theta - d, tasks, pen)
        assert g[k] == pytest.approx((hi - lo) / (2 * eps), rel=1e-5, abs=1e-9)


def test_mlogit_is_convex_so_the_start_does_not_matter(panel):
    """Two very different starting points must reach the same optimum, which is the
    property the earth-mover's fit does not have."""
    from antiGen.season_momentum import Task as T, mlogit_loss_grad, penalty as pen_of
    from scipy.optimize import minimize
    t = T(panel, 2001)
    Z, base = t.design(MONTH, 1.0)
    tasks = [(Z, base, t.assigned)]
    D = Z.shape[1]
    pen = pen_of(D, 1, 1.0, 1.0)
    outs = []
    for start in (np.zeros(D), np.full(D, 2.0), np.full(D, -3.0)):
        r = minimize(mlogit_loss_grad, start, args=(tasks, pen), jac=True,
                     method="L-BFGS-B", options={"maxiter": 500})
        outs.append(r.x)
    for o in outs[1:]:
        assert np.allclose(o, outs[0], atol=1e-4)


def test_smoothing_bounds_the_smallest_base_probability(panel):
    from antiGen.season_momentum import Task as T
    t = T(panel, 2001)
    raw = t.level(0.5 * MONTH)
    sm = t.smooth(raw, 0.2)
    assert sm.sum() == pytest.approx(1.0)
    assert sm.min() >= 0.2 / len(raw) - 1e-12
    assert sm.min() > raw.min()
    assert np.allclose(t.smooth(raw, 0.0), raw)


def test_mlogit_se_shrinks_like_one_over_root_n(panel):
    """The loss is a mean cross-entropy over normalised mass, so its own Hessian has no
    idea how many sequences that mass stands for; the scaling to log-likelihood units is
    what makes the interval mean anything. If it is missing, the SE does not move with N
    at all."""
    from antiGen.season_momentum import Task as T, fit_mlogit, mlogit_se
    t = T(panel, 2001)
    Z, base = t.design(0.5 * MONTH, np.inf)
    tasks = [(Z, base, t.assigned)]
    theta, pen = fit_mlogit(tasks, 1.0, 0.01, Z.shape[1], n_mom=1)
    se10 = mlogit_se(theta, tasks, pen, 10)
    se1000 = mlogit_se(theta, tasks, pen, 1000)
    assert np.isfinite(se10) and se10 > 0
    assert se10 / se1000 == pytest.approx(10.0, rel=0.05)      # sqrt(1000/10)


def test_rare_genotypes_carry_little_information(panel):
    """A genotype the model gives almost no probability contributes almost nothing to the
    observed information, which is the multinomial weighting doing the work."""
    from antiGen.season_momentum import Task as T
    t = T(panel, 2001)
    Z, base = t.design(0.5 * MONTH, np.inf)
    Zd = Z.toarray()
    for p_small in (1e-6, 1e-3):
        p = np.full(len(base), p_small)
        p[0] = 1.0 - p_small * (len(base) - 1)
        m = Zd.T @ p
        info = Zd.T @ (Zd * p[:, None]) - np.outer(m, m)
        # nearly all mass on one genotype leaves the information nearly singular
        assert np.linalg.norm(info) < 1.0


def test_support_cap_keeps_the_most_abundant_and_reports_what_it_drops():
    """The cap must be by abundance, deterministic, and must restrict the mask too, so the
    weights are computed over exactly the genotypes the support holds."""
    tips = ([(2000.9, "AAAA")] * 5 + [(2001.0, "ACAA")] * 3 + [(2001.1, "AAAG")] * 1
            + [(2002.1, "ACAG")] * 4)
    p = TipPanel(tips, min_allele_freq=0.0, lookback=5.0, max_support=2)
    sup, mask = p.support(2001.25)
    assert len(sup) == 2
    kept = {p.genotypes[i] for i in sup}
    assert kept == {"AAAA", "ACAA"}                 # 5 and 3 beat the singleton
    assert p.support_dropped[2001.25] == pytest.approx(1 / 9)
    # the mask may not still select the dropped genotype's sequences
    assert set(p.tip_gid[mask]) == set(sup)


def test_support_cap_off_by_default_keeps_everything():
    tips = [(2000.9, "AAAA"), (2001.0, "ACAA"), (2001.1, "AAAG")]
    p = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    sup, _ = p.support(2001.25)
    assert len(sup) == 3
    assert p.support_dropped[2001.25] == 0.0


def test_alignment_loader_reads_dates_and_resolves_ambiguity(tmp_path):
    (tmp_path / "md.tsv").write_text("strain\tdate\na\t2001-01-15\nb\t2001-02-15\n"
                                     "c\t2001-03-15\n")
    (tmp_path / "s1.fasta").write_text(">a\nMK\n>b\nMK\n>c\nMK\n")
    (tmp_path / "s2.fasta").write_text(">a\nAC\n>b\nAX\n>c\nAC\n")
    from antiGen.season_momentum import load_alignment_dated
    tips, dropped = load_alignment_dated(
        [str(tmp_path / "s1.fasta"), str(tmp_path / "s2.fasta")],
        str(tmp_path / "md.tsv"), max_ambig=0.5)
    assert dropped == 0
    got = {round(d, 3): g for d, g in tips}
    # b's X at the last position resolves to the consensus there, which is C
    assert all(g == "MKAC" for g in got.values())


def test_grid_collapses_to_what_a_family_can_distinguish():
    """A family that ignores a field must not be offered the same candidate twice."""
    from antiGen.season_momentum import collapse
    g = make_grid((1, 3), (1.0, np.inf), (0.3, 3.0), (2, 3), (0.0, 0.05))
    # ewma sees only tau_f and alpha: 2 x 2 distinct candidates
    assert len(collapse(g, "ewma")) == 4
    # mlr ignores the slow timescale: tau_f x alpha x lam x window = 2*2*2*2
    assert len(collapse(g, "mlr")) == 16
    # mlr+mom sees everything the grid varies
    assert len(collapse(g, "mlr+mom")) == len(g)
    # an unknown family is left alone
    assert len(collapse(g, "season")) == len(g)


def test_collapse_keeps_the_first_of_each_duplicate_group():
    from antiGen.season_momentum import collapse
    g = make_grid((1,), (1.0, np.inf), (0.3,), (2,), (0.0,))
    out = collapse(g, "ewma")
    assert len(out) == 1 and out[0] is g[0]


def test_restricted_family_list_runs_end_to_end(panel, capsys):
    """Regression: both the progress line and the report indexed hardcoded family keys,
    so any --families that excluded one crashed *after* the expensive work was done. Every
    subset must survive the whole path."""
    from antiGen.season_momentum import make_grid, report, rolling
    # Four consecutive seasons, so at least one transition has history to train on.
    tips = []
    for y, g in ((2001, "AAAA"), (2002, "ACAA"), (2003, "ACAG"), (2004, "AGAG")):
        # y + 0.2 falls inside the last month before the horizon at y + 0.25, so the
        # one-month window baseline is defined rather than nan for every season.
        tips += [(y - 0.1, g), (y - 0.05, g), (y + 0.1, "AAAG"), (y + 0.2, g)]
    wide = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.05,))
    for fams in (("season", "mlr"), ("mlr+mom",), ("last3m", "ewma"),
                 ("season", "last3m", "last1m", "all", "ewma", "base_sel", "mlr")):
        rows = rolling(wide, drop_targets=(), min_seqs=1, min_train=1, val_folds=1,
                       grid=grid, maxiter=10, families=fams, verbose=True)
        assert rows, "the fixture must yield at least one scored transition"
        s = report(rows, fams)
        assert set(s) >= set(fams)
        for f in fams:
            assert np.isfinite(s[f]["mean_emd"])


# ── the extended design ─────────────────────────────────────────────────────

def test_extended_design_columns_are_what_they_claim(panel):
    """momentum, log base, density within 1, density within 2 -- and all centred under
    the base measure, so none of them is smuggling in an offset."""
    from antiGen.season_momentum import Task as T, extended_design
    t = T(panel, 2001)
    combo = dict(tau_f=MONTH, tau_s=np.inf, lam=1.0, window=0, alpha=0.0,
                 base_kind="ewma")
    Z, base, n_extra = extended_design(t, combo)
    assert n_extra == 4
    assert Z.shape[1] == len(panel.features) + 4
    extra = np.asarray(Z[:, -4:].todense())
    for k in range(4):
        assert float(base @ extra[:, k]) == pytest.approx(0.0, abs=1e-9)
    # the log-base column must track the base, up to the centring
    logb = np.log(base)
    assert np.corrcoef(extra[:, 1], logb)[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_density_counts_recent_relatives_within_the_radius(panel):
    from antiGen.season_momentum import Task as T
    t = T(panel, 2001)
    n1 = t.neighbours(np.inf, 1)
    n0 = t.neighbours(np.inf, 0)
    pos = {int(g): k for k, g in enumerate(t.sup)}
    # radius 0 is just the genotype's own weight
    assert np.allclose(n0, t.level(np.inf))
    # radius 1 must be at least that, and strictly more where a neighbour exists
    assert (n1 >= n0 - 1e-12).all()
    # AAAA and AAAG differ at one position, so each is in the other's radius-1 ball
    assert n1[pos[panel.gid["AAAA"]]] > n0[pos[panel.gid["AAAA"]]]


def test_season_base_is_uniform_over_that_season(panel):
    from antiGen.season_momentum import Task as T
    t = T(panel, 2001)
    b = t.base_measure("season", MONTH)
    pos = {int(g): k for k, g in enumerate(t.sup)}
    # season 2001 holds three AAAA, one AAAG, two ACAA
    assert b[pos[panel.gid["AAAA"]]] == pytest.approx(3 / 6)
    assert b[pos[panel.gid["ACAA"]]] == pytest.approx(2 / 6)
    assert b.sum() == pytest.approx(1.0)
    assert not np.allclose(b, t.base_measure("ewma", MONTH))


def test_base_kind_is_invisible_to_families_that_ignore_it():
    """Only the extended family reads base_kind, so adding it to the grid must not
    multiply anyone else's candidate list."""
    from antiGen.season_momentum import collapse
    g = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma", "season"))
    assert len(g) == 2
    assert len(collapse(g, "mlr")) == 1
    assert len(collapse(g, "mlr+mom")) == 1
    assert len(collapse(g, "mlr++")) == 2


def test_top_k_averaging_returns_several_candidates(panel):
    from antiGen.season_momentum import Task as T, select
    tips = []
    for y, gg in ((2001, "AAAA"), (2002, "ACAA"), (2003, "ACAG"), (2004, "AGAG")):
        tips += [(y - 0.1, gg), (y - 0.05, gg), (y + 0.1, "AAAG"), (y + 0.2, gg)]
    wide = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    order = [2001, 2002, 2003]
    tasks = {s: T(wide, s) for s in order}
    grid = make_grid((1, 3), (np.inf,), (0.3, 3.0), (0,), (0.0,))
    one, _ = select(tasks, order, 2, grid, "mlr", 2, 10, 0.01, top_k=1)
    three, _ = select(tasks, order, 2, grid, "mlr", 2, 10, 0.01, top_k=3)
    assert len(one) == 1
    assert 1 < len(three) <= 3
    assert three[0] == one[0]          # the best is still first


# ── the embedding metric ────────────────────────────────────────────────────

def _emb_panel(dim=3, seed=0):
    rng = np.random.default_rng(seed)
    tips = []
    for y, g in ((2001, "AAAA"), (2002, "ACAA"), (2003, "ACAG"), (2004, "AGAG")):
        tips += [(y - 0.1, g), (y - 0.05, g), (y + 0.1, "AAAG"), (y + 0.2, g)]
    seqs = sorted({g for _, g in tips})
    emb = {g: rng.standard_normal(dim) for g in seqs}
    return tips, emb


def test_euclidean_cost_matches_a_direct_distance_computation():
    from antiGen.season_momentum import Task as T, cost_matrix
    tips, emb = _emb_panel()
    p = TipPanel(tips, 0.0, 5.0, 0, emb, "euclidean")
    t = T(p, 2002)
    M = cost_matrix(p, t.sup, t.tgt_g, "euclidean")
    for i, gi in enumerate(t.sup):
        for j, gj in enumerate(t.tgt_g):
            want = np.linalg.norm(p.emb[gi] - p.emb[gj])
            assert M[i, j] == pytest.approx(want, abs=1e-9)
    assert (M >= 0).all()


def test_the_two_metrics_give_different_geometries():
    from antiGen.season_momentum import Task as T
    tips, emb = _emb_panel()
    ph = TipPanel(tips, 0.0, 5.0, 0, emb, "hamming")
    pe = TipPanel(tips, 0.0, 5.0, 0, emb, "euclidean")
    th, te = T(ph, 2002), T(pe, 2002)
    assert th.M.shape == te.M.shape
    assert not np.allclose(th.M / max(th.M.max(), 1e-9), te.M / max(te.M.max(), 1e-9))
    # each task carries the other metric too, and they are swapped
    assert np.allclose(th.M_alt, te.M)
    assert np.allclose(te.M_alt, th.M)


def test_alt_scoring_uses_the_other_metric():
    from antiGen.season_momentum import Task as T
    tips, emb = _emb_panel()
    p = TipPanel(tips, 0.0, 5.0, 0, emb, "euclidean")
    t = T(p, 2002)
    w = t.baseline("season")
    assert t.score(w) != pytest.approx(t.score(w, alt=True))
    # the Hamming reading of the same forecast must match a Hamming-built task
    ph = TipPanel(tips, 0.0, 5.0, 0, emb, "hamming")
    th = T(ph, 2002)
    assert t.score(w, alt=True) == pytest.approx(th.score(w), abs=1e-9)


def test_assignment_and_floor_follow_the_metric():
    from antiGen.season_momentum import Task as T
    tips, emb = _emb_panel()
    th = T(TipPanel(tips, 0.0, 5.0, 0, emb, "hamming"), 2002)
    te = T(TipPanel(tips, 0.0, 5.0, 0, emb, "euclidean"), 2002)
    assert th.floor == pytest.approx(float(th.tgt_w @ th.M.min(axis=0)))
    assert te.floor == pytest.approx(float(te.tgt_w @ te.M.min(axis=0)))
    # the multinomial target is a different vector under a different geometry
    assert not np.allclose(th.assigned, te.assigned)


def test_a_mismatched_embedding_is_refused():
    tips, emb = _emb_panel()
    emb.pop("ACAA")
    with pytest.raises(ValueError, match="different panel"):
        TipPanel(tips, 0.0, 5.0, 0, emb, "euclidean")


def test_embedding_residual_is_orthogonal_to_the_allele_design():
    """The point of using the residual: the added columns must carry only what the allele
    features cannot already express, or they merely duplicate beta."""
    from antiGen.season_momentum import Task as T
    tips, emb = _emb_panel(dim=6, seed=2)
    p = TipPanel(tips, 0.0, 5.0, 0, emb, "hamming")
    t = T(p, 2003)
    pcs = t.emb_residual_pcs(3)
    S = np.asarray(t.S.todense())
    if pcs.shape[1]:
        X = np.hstack([S, np.ones((S.shape[0], 1))])
        assert np.abs(X.T @ pcs).max() < 1e-6


def test_embedding_pcs_are_capped_by_the_available_rank():
    from antiGen.season_momentum import Task as T
    tips, emb = _emb_panel(dim=3, seed=3)
    p = TipPanel(tips, 0.0, 5.0, 0, emb, "hamming")
    t = T(p, 2003)
    # never more columns than the residual actually has directions
    assert t.emb_residual_pcs(50).shape[1] <= 3
    assert t.emb_residual_pcs(1).shape[1] <= 1


def test_emb_family_needs_an_embedding_and_others_do_not():
    from antiGen.season_momentum import Task as T, collapse
    g = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma",), (4, 8))
    # emb_pcs is invisible to families that do not read it
    assert len(collapse(g, "mlr")) == 1
    assert len(collapse(g, "mlr+emb")) == 2


def test_gamma_pen_is_a_selected_field_not_a_fixed_argument():
    """It was worth 1-3 points to every fitted family in the ablation, so it has to be
    chosen on validation folds like any other hyperparameter, and it must be invisible to
    families with no weakly-penalised column."""
    from antiGen.season_momentum import collapse
    g = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma",), (8,), (0.01, 1.0))
    assert len(g) == 2
    assert len(collapse(g, "mlr")) == 1          # no momentum column to penalise
    assert len(collapse(g, "mlr+mom")) == 2
    assert len(collapse(g, "mlr+emb")) == 2


def test_combo_gamma_pen_overrides_the_call_argument(panel):
    """run_family must use the grid's value, or the selector's choice is silently
    discarded and every candidate is fitted identically."""
    from antiGen.season_momentum import Task as T, run_family
    t = T(panel, 2001)
    base = dict(tau_f=MONTH, tau_s=np.inf, lam=1e8, window=0, alpha=0.0,
                base_kind="ewma", emb_pcs=4)
    loose = run_family(t, {**base, "gamma_pen": 1e-4}, "mlr+mom", [t], 60, 9.9)
    tight = run_family(t, {**base, "gamma_pen": 1e8}, "mlr+mom", [t], 60, 9.9)
    assert not np.allclose(loose, tight)
    # with the momentum crushed too, the prediction collapses to the base measure
    assert np.allclose(tight, t.smooth(t.level(MONTH), 0.0), atol=1e-5)


# ── the immune landscape and the escape table ───────────────────────────────

# Four seasons, a tri-allelic site at position 0, a second varying site appearing late at
# position 2, and a biallelic one at position 3. Enough that the `full` escape table is
# strictly wider than the per-(site, new residue) restriction, which is what the
# identifiability claim is about.
ESC_TIPS = [
    (2000.80, "AAAA"), (2000.85, "AAAA"), (2000.90, "CAAA"), (2001.10, "AAAG"),
    (2001.20, "CAAA"), (2001.50, "CAAG"),
    (2001.80, "CAAG"), (2001.90, "GAAG"), (2002.10, "GAAG"), (2002.20, "GAAA"),
    (2002.80, "GAAA"), (2002.90, "GAAG"), (2003.10, "CAAG"), (2003.20, "GAAG"),
    (2003.80, "GACG"), (2003.90, "GAAA"), (2004.10, "GACG"), (2004.20, "AAAG"),
    (2004.80, "GACG"), (2004.90, "AACA"), (2005.10, "GAAA"), (2005.20, "AACG"),
]
ESC_COMBO = dict(tau_f=MONTH, tau_s=np.inf, lam=1.0, window=0, alpha=0.0,
                 base_kind="ewma", emb_pcs=4, gamma_pen=0.01,
                 tau_c=2.0, lam_c=1.0, rho_e=1.0)


@pytest.fixture
def esc_panel():
    return TipPanel(ESC_TIPS, min_allele_freq=0.0, lookback=10.0)


def test_the_landscape_is_a_distribution_over_each_site(esc_panel):
    """sum_a h_{q,a} = 1 is not cosmetic: it is what makes the gauge freedom exactly
    absorbable by beta, since adding a constant to c_{q,.,b} shifts the escape term by
    that constant times the site's total."""
    for tau in (0.25, 2.0, np.inf):
        h = esc_panel.landscape(2002.25, tau)
        assert (h >= 0).all()
        for _, lo, hi in esc_panel.site_span:
            assert h[lo:hi].sum() == pytest.approx(1.0)


def test_the_landscape_reads_only_sequences_up_to_the_horizon(esc_panel):
    p = esc_panel
    # At 1 April 2001 five sequences exist: AAAA, AAAA, CAAA, AAAG, CAAA. Position 0
    # carries A three times and C twice, and G not at all; position 3 carries A four
    # times and G once.
    h = p.landscape(2001.25, np.inf)
    assert h[p.feature_index[(0, "A")]] == pytest.approx(0.6)
    assert h[p.feature_index[(0, "C")]] == pytest.approx(0.4)
    assert h[p.feature_index[(0, "G")]] == pytest.approx(0.0)
    assert h[p.feature_index[(3, "A")]] == pytest.approx(0.8)
    assert h[p.feature_index[(3, "G")]] == pytest.approx(0.2)
    # G at position 0 first appears in Nov 2001, so a later horizon must see it and the
    # earlier one must not.
    assert p.landscape(2002.25, np.inf)[p.feature_index[(0, "G")]] > 0


def test_the_discount_interpolates_between_the_last_sequence_and_the_whole_record(esc_panel):
    p = esc_panel
    # tau -> 0 puts everything on the most recent sequence, GAAA at 2002.20
    h0 = p.landscape(2002.25, 1e-6)
    assert h0[p.feature_index[(0, "G")]] == pytest.approx(1.0)
    assert h0[p.feature_index[(3, "A")]] == pytest.approx(1.0)
    # tau -> inf is the plain count over everything up to the horizon
    h1 = p.landscape(2002.25, np.inf)
    seen = [g for d, g in ESC_TIPS if d <= 2002.25]
    for q, lo, hi in p.site_span:
        for k in range(lo, hi):
            want = sum(1 for g in seen if g[q] == p.features[k][1]) / len(seen)
            assert h1[k] == pytest.approx(want)


def test_the_escape_block_widths_and_the_zero_diagonal(esc_panel):
    p = esc_panel
    # three sites with 3, 2 and 2 alleles: 7 features, sum n_q(n_q - 1) = 6 + 2 + 2 = 10
    assert len(p.features) == 7
    assert len(p.triples) == 10
    assert all(ai != bi for _, ai, bi in p.triples)        # c_{q,b,b} = 0 by gauge
    t = Task(p, 2002)
    _, spec = t.escape(2.0, "escf")
    assert spec == ((1, "sigma"), (3, "site"), (7, "new"),
                    (len(p.res_pairs), "pair"), (10, "full"))


def test_escape_columns_are_h_times_the_one_hot(esc_panel):
    """Column (q, a, b) of the free block is the scalar h_{q,a} times the one-hot column
    1[v_q = b] that beta already uses. That identity is the whole implementation."""
    p = esc_panel
    t = Task(p, 2002)
    C, spec = t.escape(2.0, "escf")
    h = p.landscape(t.horizon, 2.0)
    S = np.asarray(t.S.todense())
    at = {}
    off = 0
    for n, b in spec:
        at[b] = (off, off + n)
        off += n
    dense = np.asarray(C.todense())
    lo, hi = at["full"]
    for k, (_, ai, bi) in enumerate(p.triples):
        assert np.allclose(dense[:, lo + k], h[ai] * S[:, bi])
    # and the three pooled blocks are sums of those columns, which under the gauge is
    # what collapses them to one weight per feature: sum_{a != b} h_{q,a} = 1 - h_{q,b}
    lo, hi = at["new"]
    for k in range(len(p.features)):
        assert np.allclose(dense[:, lo + k], (1.0 - h[k]) * S[:, k])


def test_the_pooled_column_is_the_distance_to_the_immune_landscape(esc_panel):
    """A constant table reduces the escape term to sigma times the discounted expected
    Hamming distance from v to the past population -- the Luksza-Lassig cross-immunity
    term, checked against both the closed form and the definition."""
    from antiGen.season_momentum import escape_term
    p = esc_panel
    t = Task(p, 2002)
    h = p.landscape(t.horizon, 2.0)
    col = np.asarray(t.escape(2.0, "esc1")[0].todense()).ravel()
    want = np.asarray([sum(1.0 - h[p.feature_index[(q, p.genotypes[int(g)][q])]]
                           for q, _, _ in p.site_span) for g in t.sup])
    assert np.allclose(col, want)
    c = {k: 1.0 for k in p.triples}
    assert np.allclose(col, escape_term(p, h, t.sup, c))


def test_the_gauge_freedom_is_exactly_absorbed_by_beta(esc_panel):
    """Adding kappa_{q,b} to c_{q,.,b} for every a -- the diagonal included -- and taking
    the same kappa out of beta_{q,b} leaves the fitness unchanged. Those are the flat
    directions the c_{q,b,b} = 0 gauge removes, and the reason the landscape has to be
    normalised per site."""
    from antiGen.season_momentum import escape_term
    p = esc_panel
    t = Task(p, 2002)
    h = p.landscape(t.horizon, 2.0)
    rng = np.random.default_rng(0)
    c = {(j, ai, bi): float(rng.normal())
         for j, (_, lo, hi) in enumerate(p.site_span)
         for ai in range(lo, hi) for bi in range(lo, hi)}       # diagonal included
    beta = rng.normal(size=len(p.features))
    S = np.asarray(t.S.todense())
    kappa = rng.normal(size=len(p.features))
    f0 = S @ beta + escape_term(p, h, t.sup, c)
    f1 = S @ (beta + kappa) + escape_term(
        p, h, t.sup, {k: v - kappa[k[2]] for k, v in c.items()})
    assert np.allclose(f0, f1)


def test_penalty_gives_each_escape_block_its_own_weight():
    pen = penalty(10, 1, 7.0, 0.5, ((2, 0.01), (3, 4.0)))
    assert list(pen) == [7.0] * 4 + [0.5] + [0.01] * 2 + [4.0] * 3
    # with no blocks it is the two-block behaviour, unchanged
    assert list(penalty(4, 1, 7.0, 0.5)) == [7.0, 7.0, 7.0, 0.5]


def test_crushing_the_escape_penalty_recovers_mlr_plus_momentum(esc_panel):
    """The nesting, asserted rather than assumed. This is the exact shape of the trap the
    quadratic clade model hit: a block penalised thousands of times more stiffly than the
    rest sets the curvature, the line search crawls, and every coefficient stalls near
    zero -- so the fit silently returns the null model, worst of all at the large ratios
    that are supposed to reproduce the simpler one."""
    from antiGen.season_momentum import run_family
    t1, t2 = Task(esc_panel, 2002), Task(esc_panel, 2003)
    ref = run_family(t2, ESC_COMBO, "mlr+mom", [t1], 80, 0.01)
    assert not np.allclose(ref, t2.level(MONTH))            # beta really is being fitted
    for fam in ("mlr+mom+esc1", "mlr+mom+escs", "mlr+mom+escf"):
        got = run_family(t2, {**ESC_COMBO, "lam_c": 1e12}, fam, [t1], 80, 0.01)
        assert np.allclose(got, ref, atol=1e-6), fam
        loose = run_family(t2, {**ESC_COMBO, "lam_c": 1e-3}, fam, [t1], 80, 0.01)
        assert not np.allclose(loose, ref, atol=1e-6), fam   # and it is not inert


def test_the_gradient_is_right_on_the_escape_design(esc_panel):
    from antiGen.season_momentum import design_for, mlogit_loss_grad
    t = Task(esc_panel, 2003)
    Z, base, n_mom, blocks = design_for(t, ESC_COMBO, "mlr+mom+escf")
    bundles = [(Z, t.smooth(base, 0.0), t.assigned)]
    pen = penalty(Z.shape[1], n_mom, 1.0, 0.01, blocks)
    rng = np.random.default_rng(1)
    th = rng.normal(scale=0.2, size=Z.shape[1])
    _, g = mlogit_loss_grad(th, bundles, pen)
    for k in rng.choice(Z.shape[1], 12, replace=False):
        e = np.zeros_like(th)
        e[k] = 1e-6
        num = (mlogit_loss_grad(th + e, bundles, pen)[0]
               - mlogit_loss_grad(th - e, bundles, pen)[0]) / 2e-6
        assert g[k] == pytest.approx(num, abs=1e-6, rel=1e-5)


def test_the_old_residue_index_is_unidentified_at_two_training_transitions(esc_panel):
    """The rank claim, as a test rather than a table. Per (q, b) group the stacked columns
    {h_{q,a}(T_t) 1[v_q = b]}_a span at most T - 1 dimensions beyond beta's own column, so
    at two training transitions the free table adds exactly what the per-(q, b)
    restriction adds and the ancestral-residue index carries no information at all. Once
    enough transitions accumulate it adds strictly more.

    This is a property of the design, not of the fit: it holds at every penalty."""
    p = esc_panel
    tasks = [Task(p, s) for s in (2001, 2002, 2003, 2004)]

    def rank(trs, arm):
        A = []
        for t in trs:
            Z, base = t.design(MONTH, np.inf)
            M = np.asarray(Z.todense())
            if arm:
                M = np.hstack([M, np.asarray(t.escape(2.0, arm)[0].todense())])
            A.append(M - base @ M)                          # centred within each task
        return np.linalg.matrix_rank(np.vstack(A), tol=1e-9)

    two, four = tasks[:2], tasks
    assert rank(two, "escfree") - rank(two, None) == rank(two, "escn") - rank(two, None)
    assert rank(four, "escfree") - rank(four, None) > rank(four, "escn") - rank(four, None)


def test_uncentred_escape_columns_are_prediction_and_gradient_neutral(esc_panel):
    """Why the escape columns are left uncentred while the other added ones are not: a
    per-task constant in the exponent cancels in the prediction, and because both p and
    the assigned target sum to one it drops out of the gradient exactly too. Centring is
    numerical hygiene there, not a requirement, and skipping it here keeps each column
    literally h_{q,a} times its one-hot."""
    from antiGen.season_momentum import design_for, mlogit_loss_grad
    t = Task(esc_panel, 2003)
    Z, base, n_mom, blocks = design_for(t, ESC_COMBO, "mlr+mom+escf")
    base = t.smooth(base, 0.0)
    pen = penalty(Z.shape[1], n_mom, 1.0, 0.01, blocks)
    rng = np.random.default_rng(2)
    th = rng.normal(scale=0.2, size=Z.shape[1])
    Zs = np.asarray(Z.todense())
    shifted = Zs.copy()
    shifted[:, -1] += 3.7
    l0, g0 = mlogit_loss_grad(th, [(Z, base, t.assigned)], pen)
    l1, g1 = mlogit_loss_grad(th, [(sparse.csr_matrix(shifted), base, t.assigned)], pen)
    assert l0 == pytest.approx(l1)
    assert np.allclose(g0, g1)
    assert np.allclose(predict(th, Z, base), predict(th, sparse.csr_matrix(shifted), base))


def test_sigma_rises_when_the_distant_genotype_wins(esc_panel):
    """The sign convention, which is the falsifiable part: sigma > 0 has to mean 'far from
    the immune landscape is fitter'. With the target mass on the most distant genotype the
    gradient at sigma = 0 must be negative, so the fit raises it."""
    from antiGen.season_momentum import mlogit_loss_grad
    t = Task(esc_panel, 2003)
    C, _ = t.escape(2.0, "esc1")
    col = np.asarray(C.todense()).ravel()
    q = np.zeros(len(t.sup))
    q[int(np.argmax(col))] = 1.0
    _, g = mlogit_loss_grad(np.zeros(1), [(C, t.level(MONTH), q)], np.zeros(1))
    assert g[0] < 0


def test_sigma_report_returns_the_coefficient_the_fit_produced(esc_panel):
    from antiGen.season_momentum import design_for, fit_family, sigma_report
    t1, t2 = Task(esc_panel, 2002), Task(esc_panel, 2003)
    for fam in ("mlr+esc1", "mlr+mom+esc1"):
        Z, _, n_mom, blocks = design_for(t2, ESC_COMBO, fam)
        theta = fit_family(t2, ESC_COMBO, fam, [t1], 0.01, Z.shape[1], n_mom, blocks)
        sig, se = sigma_report(t2, ESC_COMBO, fam, [t1], 0.01)
        assert sig == pytest.approx(float(theta[-1]))        # sigma is the last column
        assert np.isfinite(se) and se > 0
    # and with no training transitions there is nothing to report rather than a crash
    assert not np.isfinite(sigma_report(t2, ESC_COMBO, "mlr+mom+esc1", [], 0.01)[0])


def test_the_escape_hyperparameters_are_selected_and_scoped():
    """tau_c, lam_c and rho_e are grid fields, not fixed arguments -- sweeping a penalty
    and reading the best cell off the test transitions reversed the sign of a conclusion
    once already. And rho_e must be invisible to the arms with no full table for it to act
    on, or the argmin is decided among identical candidates by iteration order."""
    from antiGen.season_momentum import collapse
    g = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma",), (8,), (0.01,),
                  (1.0, 4.0), (0.1, 10.0), (1.0, 100.0))
    assert len(g) == 8
    assert len(collapse(g, "mlr+mom")) == 1
    assert len(collapse(g, "mlr+mom+esc1")) == 4
    assert len(collapse(g, "mlr+mom+escs")) == 4
    assert len(collapse(g, "mlr+mom+escf")) == 8
    assert len(collapse(g, "mlr+esc1")) == 4


def test_the_escape_families_run_end_to_end(esc_panel):
    from antiGen.season_momentum import report, rolling
    fams = ("season", "mlr+mom", "mlr+esc1", "mlr+mom+esc1", "mlr+mom+escs",
            "mlr+mom+escf")
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma",), (4,), (0.01,),
                     (2.0,), (0.3, 3.0), (1.0, 100.0))
    rows = rolling(esc_panel, drop_targets=(), min_seqs=3, min_train=2, val_folds=2,
                   grid=grid, families=fams, verbose=False)
    assert rows
    s = report(rows, fams)
    for f in fams:
        assert np.isfinite(s[f]["mean_emd"])
    for f in ("mlr+esc1", "mlr+mom+esc1"):
        assert len(s[f]["sigma"]) == len(rows)
        assert all(np.isfinite(x) for x in s[f]["sigma"])


# ── the base measure the tilt multiplies, and the blend against persistence ──


def test_the_new_fields_are_invisible_to_the_families_that_ignore_them():
    """`base_kind` and `eta` each add a family. Neither may multiply anyone else's
    candidate list -- least of all `mlr+mom`'s, which is the reference row both are
    compared against, and which must be fitted over exactly the grid it always was."""
    from antiGen.season_momentum import collapse
    g = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma", "season"),
                  (8,), (0.01,), (2.0,), (1.0,), (1.0,), (0.5, 1.0))
    assert len(g) == 4
    assert len(collapse(g, "mlr+mom")) == 1
    assert len(collapse(g, "mlr+mom+sb")) == 2       # base_kind only
    assert len(collapse(g, "mlr+mom+mix")) == 2      # eta only
    assert len(collapse(g, "mlr++")) == 2            # base_kind, as before


def test_the_selected_base_family_nests_its_parent_and_departs_from_it(panel):
    """`mlr+mom+sb` is `mlr+mom` with one field freed, so at base_kind 'ewma' it must be
    the parent to the last digit, and at 'season' it must not be -- otherwise the family
    is either broken or pointless."""
    from antiGen.season_momentum import Task as T, run_family
    t = T(panel, 2001)
    combo = dict(tau_f=MONTH, tau_s=np.inf, lam=1.0, window=0, alpha=0.0,
                 emb_pcs=4, gamma_pen=0.01, eta=1.0)
    parent = run_family(t, {**combo, "base_kind": "ewma"}, "mlr+mom", [t], 60, 0.01)
    same = run_family(t, {**combo, "base_kind": "ewma"}, "mlr+mom+sb", [t], 60, 0.01)
    other = run_family(t, {**combo, "base_kind": "season"}, "mlr+mom+sb", [t], 60, 0.01)
    assert np.allclose(parent, same)
    assert not np.allclose(parent, other)
    assert other.sum() == pytest.approx(1.0)


def test_the_selected_base_family_tilts_the_measure_it_was_given(panel):
    """With the tilt crushed the forecast is the base measure itself, which is the only
    check that `base_kind` reaches the prediction rather than just the grid."""
    from antiGen.season_momentum import Task as T, run_family
    t = T(panel, 2001)
    dead = dict(tau_f=MONTH, tau_s=np.inf, lam=1e8, window=0, alpha=0.0, emb_pcs=4,
                gamma_pen=1e8, eta=1.0, base_kind="season")
    w = run_family(t, dead, "mlr+mom+sb", [t], 60, 0.01)
    assert np.allclose(w, t.base_measure("season", MONTH), atol=1e-5)


def test_the_blend_reaches_both_of_the_models_it_mixes(panel):
    """eta = 1 must be `mlr+mom` exactly and eta = 0 whole-season persistence exactly.
    A family that cannot recover either endpoint is not nesting them."""
    from antiGen.season_momentum import Task as T, run_family
    t = T(panel, 2001)
    combo = dict(tau_f=MONTH, tau_s=np.inf, lam=1.0, window=0, alpha=0.0,
                 base_kind="ewma", emb_pcs=4, gamma_pen=0.01)
    mom = run_family(t, {**combo, "eta": 1.0}, "mlr+mom", [t], 60, 0.01)
    assert np.allclose(run_family(t, {**combo, "eta": 1.0}, "mlr+mom+mix",
                                  [t], 60, 0.01), mom)
    assert np.allclose(run_family(t, {**combo, "eta": 0.0}, "mlr+mom+mix",
                                  [t], 60, 0.01), t.baseline("season"))
    half = run_family(t, {**combo, "eta": 0.5}, "mlr+mom+mix", [t], 60, 0.01)
    assert np.allclose(half, 0.5 * mom + 0.5 * t.baseline("season"))


def test_the_blend_is_never_worse_than_interpolating_the_two_scores(panel):
    """The reason to expect anything from the blend is that the earth-mover's distance
    is convex in the forecast. That is a property of the score, so it can be checked
    rather than argued: the blend's EMD must sit at or below the straight line between
    the two endpoints' EMDs, at every weight."""
    from antiGen.season_momentum import Task as T, run_family
    t = T(panel, 2001)
    combo = dict(tau_f=MONTH, tau_s=np.inf, lam=1.0, window=0, alpha=0.0,
                 base_kind="ewma", emb_pcs=4, gamma_pen=0.01)
    ends = [t.score(run_family(t, {**combo, "eta": e}, "mlr+mom+mix", [t], 60, 0.01))
            for e in (0.0, 1.0)]
    for eta in (0.25, 0.5, 0.75):
        got = t.score(run_family(t, {**combo, "eta": eta}, "mlr+mom+mix",
                                 [t], 60, 0.01))
        assert got <= eta * ends[1] + (1 - eta) * ends[0] + 1e-9


def test_the_blend_memo_does_not_leak_across_combos_or_training_sets(panel):
    """The parent's forecast is cached so that five values of eta do not refit it five
    times. A key that missed a field would silently hand one combination's fit to
    another, which is the kind of bug that makes a family look better than it is."""
    from antiGen.season_momentum import Task as T, run_family, _momentum_prediction
    t0, t1 = T(panel, 2001), T(panel, 2002)
    combo = dict(tau_f=MONTH, tau_s=np.inf, lam=1.0, window=0, alpha=0.0,
                 base_kind="ewma", emb_pcs=4, gamma_pen=0.01, eta=1.0)
    cached = _momentum_prediction(t1, combo, [t0], 60, 0.01)
    assert np.allclose(cached, run_family(t1, combo, "mlr+mom", [t0], 60, 0.01))
    # a different penalty, and a different training set, must both miss the cache
    assert not np.allclose(cached, _momentum_prediction(t1, {**combo, "lam": 1e8},
                                                        [t0], 60, 0.01))
    assert not np.allclose(cached, _momentum_prediction(t1, combo, [t0, t1], 60, 0.01))


def test_the_two_new_families_run_end_to_end(panel, capsys):
    from antiGen.season_momentum import rolling
    tips = []
    for y, gg in ((2001, "AAAA"), (2002, "ACAA"), (2003, "ACAG"), (2004, "AGAG"),
                  (2005, "AGGG")):
        tips += [(y - 0.1, gg), (y - 0.05, gg), (y + 0.1, "AAAG"), (y + 0.2, gg)]
    wide = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma", "season"),
                     (8,), (0.01,), (2.0,), (1.0,), (1.0,), (0.5, 1.0))
    rows = rolling(wide, drop_targets=(), min_seqs=1, min_train=2, val_folds=2,
                   grid=grid, maxiter=20,
                   families=("season", "mlr+mom", "mlr+mom+sb", "mlr+mom+mix"),
                   verbose=False)
    assert rows
    for r in rows:
        for f in ("mlr+mom", "mlr+mom+sb", "mlr+mom+mix"):
            assert np.isfinite(r["emd_" + f])
        assert r["combo_mlr+mom+mix"]["eta"] in (0.5, 1.0)
        assert r["combo_mlr+mom+sb"]["base_kind"] in ("ewma", "season")


# ── the within-season fit: a growth rate read off season i's own time course ──

# Monthly sequences over five and a bit years. 'C' at site 1 sweeps in logistically,
# steeply enough that it moves *within* a one-year window rather than only across the
# panel -- a window whose composition is constant has no slope to find, which is a way to
# write a passing test that measures nothing. 'G' at site 3 holds exactly a third of every
# timepoint, so that site is flat by construction. "ACAT" is seen once at the very start:
# inside the lookback, so it is in the support at every horizon, and never inside any
# window used below. Its site-1 allele is the one that is sweeping and its site-3 allele
# is its own, which is what makes it the test case for the two bases -- `feat` must give
# it a rate borrowed from 'C', and `free` must give it exactly zero.
def _win_tips(months=64, per_month=6, start=2000.0, mid=2003.75, scale=0.25):
    tips = []
    for k in range(months):
        t = start + k / 12.0
        nC = int(round(per_month / (1.0 + np.exp(-(t - mid) / scale))))
        for j in range(per_month):
            s1 = "C" if j < nC else "A"
            s3 = "G" if j % 3 == 0 else "A"
            tips.append((t, f"A{s1}A{s3}"))
    tips.append((start + 1 / 24.0, "ACAT"))
    return tips


WIN_COMBO = dict(tau_f=MONTH, tau_s=np.inf, lam=1.0, window=0, alpha=0.0,
                 base_kind="ewma", emb_pcs=4, gamma_pen=0.01, tau_c=2.0, lam_c=1.0,
                 rho_e=1.0, eta=1.0, W=1.0, wbasis="feat", Delta=1.0,
                 lam_as=(0.01, 0.1, 1.0, 10.0, 100.0), wbin=0)


@pytest.fixture
def win_panel():
    return TipPanel(_win_tips(), min_allele_freq=0.0, lookback=10.0)


def test_the_window_is_the_half_open_interval_it_claims(win_panel):
    """(T - W, T], grouped by distinct date, shares over the window total, and `tbar` the
    count-weighted mean date. Every later claim rests on that last one in particular."""
    t = Task(win_panel, 2004)
    times, Q, tbar = t.timepoints(1.0)
    assert len(times) == 12
    assert times.min() > t.horizon - 1.0 and times.max() <= t.horizon
    assert Q.sum() == pytest.approx(1.0)
    assert tbar == pytest.approx(float(times @ Q.sum(axis=1)))
    # every sequence in the window, and nothing else, is accounted for
    inw = ((win_panel.dates > t.horizon - 1.0) & (win_panel.dates <= t.horizon)
           & t.mask).sum()
    assert Q.sum(axis=1).min() > 0 and inw == 12 * 6


def test_the_window_cannot_see_past_the_horizon(win_panel):
    """The half of the leakage test that matters: nothing dated after `T` may touch
    `alpha_hat`, however much of it there is. A window-boundary off-by-one is exactly
    what this would catch, and it is silent otherwise.

    The added and removed sequences carry genotypes the panel already holds, so the
    support and the feature space are untouched and the only thing under test is the
    window."""
    base = _win_tips()
    t0 = Task(win_panel, 2004)
    a0 = t0.growth(1.0, "feat", 1.0)
    for edit in (base + [(2004.30, "ACAA"), (2004.9, "ACAG"), (2005.2, "AAAA")],
                 [x for x in base if x[0] <= 2004.25],
                 [(d if d <= 2004.25 else d + 0.01, g) for d, g in base]):
        p = TipPanel(edit, min_allele_freq=0.0, lookback=10.0)
        assert p.features == win_panel.features
        t = Task(p, 2004)
        assert np.array_equal(t.sup, t0.sup)
        assert np.allclose(t.growth(1.0, "feat", 1.0), a0, atol=1e-12)


def test_moving_one_date_across_the_horizon_does_change_it(win_panel):
    """The other half. A window that never changed would pass the test above trivially."""
    t0 = Task(win_panel, 2004)
    a0 = t0.growth(1.0, "feat", 1.0)
    moved = [((2004.24 if d > 2004.25 and g == "ACAA" and d < 2004.4 else d), g)
             for d, g in _win_tips()]
    assert moved != _win_tips()
    t = Task(TipPanel(moved, min_allele_freq=0.0, lookback=10.0), 2004)
    assert not np.allclose(t.growth(1.0, "feat", 1.0), a0)


def test_the_inner_fit_recovers_a_known_growth_rate():
    """The correctness test; everything else here checks plumbing.

    Sequences are simulated from the inner model itself with a known per-feature rate --
    'C' at site 1 rising at +2 a year, 'G' at site 3 falling at -1.5 -- and the fit has
    to give back the sign of every one of them and a magnitude in the right neighbourhood.
    Only the within-site *contrast* is identified, since adding a constant to a site's
    rates cancels in the softmax, so the contrasts are what is compared."""
    true = {(1, "A"): 0.0, (1, "C"): 2.0, (3, "A"): 0.0, (3, "G"): -1.5}
    gts = [f"A{s1}A{s3}" for s1 in "AC" for s3 in "AG"]
    rate = {g: true[(1, g[1])] + true[(3, g[3])] for g in gts}
    tips, N = [], 60
    times = [2003.25 + (k + 1) / 12.0 for k in range(12)]
    tbar = float(np.mean(times))
    for t in times:
        f = np.asarray([rate[g] * (t - tbar) for g in gts])
        p = np.exp(f - f.max())
        p /= p.sum()
        n = np.floor(N * p).astype(int)
        n[np.argmax(p)] += N - n.sum()                  # largest-remainder, kept exact
        for g, c in zip(gts, n):
            tips += [(t, g)] * int(c)
    tips += [(2004.8, g) for g in gts]                  # a next season to predict
    panel = TipPanel(tips, min_allele_freq=0.0, lookback=10.0)
    a = Task(panel, 2004).growth(1.0, "feat", 1e-4)
    got = {g: v for g, v in zip([panel.genotypes[i]
                                 for i in Task(panel, 2004).sup], a)}
    site1 = got["ACAA"] - got["AAAA"]                   # C over A at site 1
    site3 = got["AAAG"] - got["AAAA"]                   # G over A at site 3
    assert site1 > 0 and site3 < 0                      # the signs, first
    assert site1 == pytest.approx(2.0, rel=0.25)
    assert site3 == pytest.approx(-1.5, rel=0.25)


def test_the_inner_fit_is_convex_so_the_start_does_not_matter(win_panel):
    """The same property the outer multinomial fit is held to, on the inner bundles --
    which differ from the outer ones in carrying a target that sums to the timepoint's
    share of the window rather than to one."""
    from antiGen.season_momentum import mlogit_loss_grad, penalty as pen_of
    from scipy.optimize import minimize
    t = Task(win_panel, 2004)
    times, Q, tbar = t.timepoints(1.0)
    B = t.growth_basis("feat")
    bundles = [((times[j] - tbar) * B, Q.sum(axis=0), Q[j]) for j in range(len(times))]
    D = B.shape[1]
    pen = pen_of(D, 0, 0.05, 0.05)
    outs = []
    for start in (np.zeros(D), np.full(D, 3.0), np.full(D, -2.0)):
        r = minimize(mlogit_loss_grad, start, args=(bundles, pen), jac=True,
                     method="L-BFGS-B", options={"maxiter": 800})
        outs.append(r.x)
    for o in outs[1:]:
        assert np.allclose(o, outs[0], atol=1e-4)


def test_the_inner_gradient_matches_central_differences(win_panel):
    """The inner bundles are the one place the multinomial target does *not* sum to one:
    each timepoint carries its share of the window, so a month with two sequences weighs
    a tenth of one with twenty. The gradient is then Z'(q_total p - q), and dropping that
    factor -- which is invisible everywhere else, since every outer target is a
    distribution -- makes the loss and its gradient different functions."""
    from antiGen.season_momentum import mlogit_loss_grad, penalty as pen_of
    t = Task(win_panel, 2004)
    times, Q, tbar = t.timepoints(1.0)
    B = t.growth_basis("feat")
    bundles = [((times[j] - tbar) * B, Q.sum(axis=0), Q[j]) for j in range(len(times))]
    D = B.shape[1]
    pen = pen_of(D, 0, 0.1, 0.1)
    rng = np.random.default_rng(0)
    th = rng.normal(scale=0.5, size=D)
    _, g = mlogit_loss_grad(th, bundles, pen)
    eps = 1e-6
    for k in range(D):
        e = np.zeros(D)
        e[k] = eps
        hi, _ = mlogit_loss_grad(th + e, bundles, pen)
        lo, _ = mlogit_loss_grad(th - e, bundles, pen)
        assert g[k] == pytest.approx((hi - lo) / (2 * eps), abs=1e-6)


def test_the_growth_rate_does_not_depend_on_the_time_origin(win_panel):
    """A slope is not a function of where time is measured from. Shifting every date by a
    year -- and the horizon with it -- must leave `alpha_hat` alone; this is the test that
    catches a `tbar` mistake, which is the likeliest silent bug in the inner fit."""
    t0 = Task(win_panel, 2004)
    shifted = TipPanel([(d + 1.0, g) for d, g in _win_tips()],
                       min_allele_freq=0.0, lookback=10.0)
    t1 = Task(shifted, 2005)
    assert t1.horizon == pytest.approx(t0.horizon + 1.0)
    assert np.allclose(t1.growth(1.0, "feat", 1.0), t0.growth(1.0, "feat", 1.0),
                       atol=1e-8)


def test_scaling_time_scales_the_growth_rate_inversely(win_panel):
    """The units half of the same check, on the estimator directly: with dates in units
    `c` times as long, the same fit in the same units of the penalty must give
    `alpha_hat / c`. The penalty is on `theta`, so it scales by `c**2` -- which is the
    statement that `lambda_alpha` has units, and is worth pinning down."""
    from antiGen.season_momentum import fit_mlogit
    t = Task(win_panel, 2004)
    times, Q, tbar = t.timepoints(1.0)
    B = t.growth_basis("feat")
    p0 = Q.sum(axis=0)

    def theta_at(c, lam):
        bundles = [(c * (times[j] - tbar) * B, p0, Q[j]) for j in range(len(times))]
        return fit_mlogit(bundles, lam, lam, B.shape[1], n_mom=0, maxiter=500)[0]

    lam, c = 0.05, 4.0
    assert np.allclose(theta_at(c, lam * c * c), theta_at(1.0, lam) / c, atol=1e-6)


def test_binning_nests_the_exact_per_date_likelihood():
    """Twelve monthly bins is the literal reading of 'month by month' and a coarsening of
    the exact per-sequence likelihood, which is what runs by default. When every bin holds
    exactly one date, and that date is the bin's own centre, the two must coincide
    exactly -- otherwise the binning is doing something besides grouping."""
    horizon, W, nbin = 2004.25, 1.0, 12
    centres = [horizon - W + W * (b + 0.5) / nbin for b in range(nbin)]
    tips = []
    for b, t in enumerate(centres):
        tips += [(t, "ACAA")] * (1 + b % 3) + [(t, "AAAA")] * (3 - b % 3)
    tips += [(2003.0, "AAAG"), (2004.8, "ACAA"), (2004.9, "AAAA")]
    panel = TipPanel(tips, min_allele_freq=0.0, lookback=10.0)
    t = Task(panel, 2004)
    x0, y0, b0 = t.timepoints(W, 0)
    x1, y1, b1 = t.timepoints(W, nbin)
    assert np.allclose(x0, x1) and np.allclose(y0, y1) and b0 == pytest.approx(b1)
    assert np.allclose(t.growth(W, "feat", 1.0, 0), t.growth(W, "feat", 1.0, nbin))


def test_a_degenerate_window_falls_back_to_the_parent(win_panel):
    """`W = 0.5` at 2022 holds 35 sequences, so a window too thin to carry a slope is not
    hypothetical. Fewer than two timepoints must give a zero column and the parent
    model's forecast, not an exception and not junk."""
    from antiGen.season_momentum import run_family
    t = Task(win_panel, 2004)
    assert len(t.timepoints(0.02)[0]) < 2
    assert np.allclose(t.growth(0.02, "feat", 1.0), 0.0)
    thin = {**WIN_COMBO, "W": 0.02}
    assert np.allclose(run_family(t, thin, "mlr+wmom", [t], 60, 0.01),
                       run_family(t, thin, "mlr", [t], 60, 0.01))
    assert np.allclose(run_family(t, thin, "mlr+mom+wmom", [t], 60, 0.01),
                       run_family(t, thin, "mlr+mom", [t], 60, 0.01))
    assert np.allclose(run_family(t, thin, "wmlr", [t], 60, 0.01),
                       t.level(WIN_COMBO["tau_f"]))


def test_crushing_the_inner_penalty_recovers_the_parent_model(win_panel):
    """`lambda_alpha -> inf` sends `theta_a` to zero under every basis, which sends the
    column to zero, which must give back the parent coefficient for coefficient. A nested
    family whose limit is not its parent is not a fair comparison against it -- the
    failure the quadratic clade model hit and the escape round nearly repeated."""
    from antiGen.season_momentum import run_family
    t = Task(win_panel, 2004)
    for wb in ("feat", "free", "both"):
        dead = {**WIN_COMBO, "wbasis": wb, "lam_as": (1e12,)}
        assert np.allclose(t.growth(1.0, wb, 1e12), 0.0, atol=1e-8)
        assert np.allclose(run_family(t, dead, "mlr+wmom", [t], 80, 0.01),
                           run_family(t, dead, "mlr", [t], 80, 0.01), atol=1e-8)
        assert np.allclose(run_family(t, dead, "mlr+mom+wmom", [t], 80, 0.01),
                           run_family(t, dead, "mlr+mom", [t], 80, 0.01), atol=1e-8)
    # and the column is doing something at a penalty that does not crush it
    assert not np.allclose(run_family(t, WIN_COMBO, "mlr+wmom", [t], 80, 0.01),
                           run_family(t, WIN_COMBO, "mlr", [t], 80, 0.01))


def test_zero_lead_time_makes_wmlr_the_base_measure(win_panel):
    """`wmlr` is a reweighting of the base measure rather than a covariate, so at
    `Delta = 0` it must be that measure exactly, and at a year it must not be."""
    from antiGen.season_momentum import run_family
    t = Task(win_panel, 2004)
    w0 = run_family(t, {**WIN_COMBO, "Delta": 0.0}, "wmlr", [t], 60, 0.01)
    assert np.allclose(w0, t.level(WIN_COMBO["tau_f"]))
    w1 = run_family(t, WIN_COMBO, "wmlr", [t], 60, 0.01)
    assert w1.sum() == pytest.approx(1.0) and not np.allclose(w1, w0)


def test_the_growth_column_adds_exactly_one_direction(win_panel):
    """The identification structure, as a test rather than a table. Within one task
    `alpha = S theta_a` lies in the column space of `S`, so the column is *exactly*
    redundant there; stacked over transitions it is `sum_t e_t (x) S_t alpha_hat_t`, which
    leaves beta's span only because `alpha_hat_t` differs between horizons. One new
    direction, not none and not two. This is the escape term's structure with one
    parameter instead of 354, and like it, a property of the design at every penalty."""
    from antiGen.season_momentum import growth_column
    tasks = [Task(win_panel, s) for s in (2002, 2003, 2004)]

    def rank(trs, with_alpha):
        A = []
        for t in trs:
            Z, base = t.design(MONTH, np.inf)
            M = np.asarray(Z[:, :-1].todense())
            if with_alpha:
                M = np.hstack([M, growth_column(t, WIN_COMBO)[:, None]])
            A.append(M - base @ M)

        return np.linalg.matrix_rank(np.vstack(A), tol=1e-9)

    assert rank(tasks[:1], True) - rank(tasks[:1], False) == 0
    assert rank(tasks, True) - rank(tasks, False) == 1


def test_every_support_genotype_gets_a_rate_and_the_denominator_does_not_move(win_panel):
    """The column is defined on the whole support, including genotypes the window never
    held, and the inner softmax normalises over the same index set at every timepoint --
    never over a moving set of whoever happens to have been sequenced that month."""
    from antiGen.season_momentum import growth_column, predict
    t = Task(win_panel, 2004)
    times, Q, tbar = t.timepoints(1.0)
    unheld = np.flatnonzero(Q.sum(axis=0) == 0)
    assert len(unheld) and len(growth_column(t, WIN_COMBO)) == len(t.sup)
    B = t.growth_basis("feat")
    th = np.full(B.shape[1], 0.3)
    for j in range(len(times)):
        p = predict(th, (times[j] - tbar) * B, Q.sum(axis=0))
        assert len(p) == len(t.sup) and p.sum() == pytest.approx(1.0)


def test_the_two_bases_differ_exactly_where_they_should(win_panel):
    """The pooling difference in one line. A genotype the window never held has `q = 0` at
    every timepoint and, because `tbar` is count-weighted, a gradient of exactly zero, so
    the free basis leaves its rate at zero to machine precision. Under `feat` it still
    gets one, borrowed from its alleles. That is what `free` against `feat` isolates, and
    table E of the proposal says it applies to 60-89% of the support."""
    t = Task(win_panel, 2004)
    _, Q, _ = t.timepoints(1.0)
    unheld = np.flatnonzero(Q.sum(axis=0) == 0)
    assert len(unheld)
    free = t.growth(1.0, "free", 0.1)
    feat = t.growth(1.0, "feat", 0.1)
    assert np.allclose(free[unheld], 0.0, atol=1e-12)
    assert np.abs(feat[unheld]).max() > 1e-6
    held = np.flatnonzero(Q.sum(axis=0) > 0)
    assert np.abs(free[held]).max() > 1e-6


def test_the_inner_penalty_is_chosen_in_the_window_and_reports_its_curve(win_panel):
    """`lambda_alpha` is settled by leave-one-timepoint-out CV inside the window rather
    than by the outer selector, which the 2026-09-21 round priced at about a point per
    added field. The curve comes back with it so the fit's sensitivity can be reported
    instead of assumed."""
    t = Task(win_panel, 2004)
    las = (0.01, 1.0, 100.0)
    best, curve = t.growth_cv(1.0, "feat", las)
    assert set(curve) == {float(x) for x in las}
    assert best == min(curve, key=curve.get)
    assert np.allclose(t.growth_selected(1.0, "feat", las),
                       t.growth(1.0, "feat", best)
                       - float(t.timepoints(1.0)[1].sum(axis=0)
                               @ t.growth(1.0, "feat", best)))


def test_the_within_season_fields_are_scoped_to_the_families_that_read_them():
    """Three new fields, and none of them may multiply the candidate list of a family
    that cannot see it -- least of all `mlr+mom`'s, the reference row everything here is
    compared against. `mlr+wmom` drops the momentum column, so it must not see `tau_s`
    either, or its argmin is decided by iteration order among identical candidates."""
    from antiGen.season_momentum import collapse
    g = make_grid((1,), (1.0, np.inf), (1.0,), (0,), (0.0,), ("ewma",), (8,), (0.01,),
                  (2.0,), (1.0,), (1.0,), (1.0,), Ws=(0.5, 1.0), wbases=("feat", "free"),
                  deltas=(1.0, 2.0))
    assert len(g) == 2 * 2 * 2 * 2
    assert len(collapse(g, "mlr+mom")) == 2         # tau_s alone
    assert len(collapse(g, "mlr")) == 1
    assert len(collapse(g, "mlr+wmom")) == 4        # W x wbasis, and no tau_s
    assert len(collapse(g, "mlr+mom+wmom")) == 8    # tau_s x W x wbasis
    assert len(collapse(g, "wmlr")) == 8            # W x wbasis x Delta, no tau_s
    assert len(collapse(g, "mlr+mom+esc1")) == 2


def test_the_within_season_families_run_end_to_end(win_panel, capsys):
    """End to end through the rolling selector, with phi reported for the fitted arms."""
    from antiGen.season_momentum import rolling, report
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma",), (8,), (0.01,),
                     (2.0,), (1.0,), (1.0,), (1.0,), Ws=(1.0,), wbases=("feat",),
                     deltas=(1.0,), lam_as=(1.0,))
    fams = ("season", "mlr", "mlr+mom", "mlr+wmom", "mlr+mom+wmom", "wmlr")
    rows = rolling(win_panel, drop_targets=(), min_seqs=1, min_train=2, val_folds=2,
                   grid=grid, maxiter=20, families=fams, verbose=False)
    assert rows
    for r in rows:
        r_sup = Task(win_panel, r["season"]).sup
        for f in fams:
            assert np.isfinite(r["emd_" + f])
        assert r["combo_mlr+wmom"]["W"] == 1.0
        assert r["combo_mlr+wmom"]["wbasis"] == "feat"
        assert r["combo_wmlr"]["Delta"] == 1.0
        for f in ("mlr+wmom", "mlr+mom+wmom"):
            assert np.isfinite(r["sigma_" + f]) and np.isfinite(r["sigma_se_" + f])
        # the inner fit's own diagnostic: what it chose, and how flat the curve it chose
        # from was, so section 3's claim that lambda_alpha matters little is measured
        w = r["wfit_mlr+wmom"]
        assert w["lam_alpha"] == 1.0 and w["n_support"] == len(r_sup)
        assert w["n_in_window"] <= w["n_support"] and w["n_timepoints"] >= 2
        assert set(w["cv_curve"]) == {"1.0"} and w["cv_spread"] == 0.0
    out = report(rows, fams)
    assert out["mlr+wmom"]["coef_name"] == "phi"
    assert "phi" in capsys.readouterr().out


# ── distance from consensus as a design column ──────────────────────────────


def test_the_consensus_is_the_weighted_plurality_and_the_distance_counts_sites():
    """`consensus_distance` by hand on four genotypes, and the weights decide it.

    Three varying sites. Under uniform weights 'A' is the plurality at site 1 and the
    consensus is "A.AA"; put enough weight on the single 'C' carrier and the consensus
    moves to it, which flips who is closest. The metric is a count of *sites*, never of
    tips, so a genotype two substitutions out scores 2 whatever its abundance."""
    from antiGen.season_momentum import consensus_distance
    tips = [(2000.8, "AAAA"), (2000.9, "AAAG"), (2001.0, "ACAA"), (2001.1, "AAAA")]
    p = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    sup = np.arange(len(p.genotypes))
    seq = {g: k for k, g in enumerate(p.genotypes)}
    n = len(p.genotypes)
    uni = np.full(n, 1.0 / n)
    d = consensus_distance(p.codes, sup, uni)
    # consensus "AAAA": itself 0, "AAAG" 1 site away, "ACAA" 1 site away
    assert d[seq["AAAA"]] == 0 and d[seq["AAAG"]] == 1 and d[seq["ACAA"]] == 1
    w = np.zeros(n); w[seq["ACAA"]] = 0.9; w[seq["AAAA"]] = 0.1
    d2 = consensus_distance(p.codes, sup, w)
    assert d2[seq["ACAA"]] == 0 and d2[seq["AAAA"]] == 1 and d2[seq["AAAG"]] == 2


def test_the_backbone_comparison_and_the_design_column_are_the_same_function():
    """The metric `backbone_compare` scores as a model of its own and the column
    `mlr+mom+dfc` fits are one function, so the two cannot drift apart. The row in the
    headline table and the arm that tries to beat it have to mean the same thing."""
    from antiGen import backbone_compare
    from antiGen.season_momentum import consensus_distance
    assert backbone_compare.consensus_distance is consensus_distance


def test_the_column_is_the_metric_standardised_under_season_persistence(win_panel):
    """`Task.dfc` is `consensus_distance` under whole-season weights -- the same weights
    `BackbonePanel.metrics` uses -- centred and scaled under those same weights. Centring
    is hygiene: an additive constant in f cancels in the prediction. The scaling is not,
    because it is what makes `gamma_pen` mean the same thing here as on momentum."""
    from antiGen.season_momentum import consensus_distance
    t = Task(win_panel, 2004)
    w = t.baseline("season")
    raw = consensus_distance(win_panel.codes, t.sup, w)
    z = t.dfc()
    assert np.average(z, weights=w) == pytest.approx(0.0, abs=1e-12)
    assert np.average(z ** 2, weights=w) == pytest.approx(1.0)
    sd = np.sqrt(np.average((raw - np.average(raw, weights=w)) ** 2, weights=w))
    assert z == pytest.approx((raw - np.average(raw, weights=w)) / sd)
    # and it reads no hyperparameter: the same task hands back the same column whatever
    # the combination asks for
    assert t.dfc() is z


def test_the_consensus_column_is_redundant_within_a_task_and_new_only_when_stacked():
    """What the arm actually buys, as a test rather than a claim.

    `dfc_v` counts the sites at which `v` differs from the consensus, so with every
    observed allele carrying a column it is exactly `n_sites - sum_j S[v, (j, cons_j)]`
    -- a linear function of the one-hot design. Within one task it therefore adds *no*
    direction at all: `mlr` can already express it. Stacked over transitions it adds
    exactly one, because the consensus and the scaling move between horizons and the fit
    pools a single theta over them. So the column is a shrinkage target, not a feature:
    it exempts one direction of beta from `lam` and hands it `gamma_pen` instead."""
    p = TipPanel(_win_tips(), min_allele_freq=0.0, lookback=10.0)
    tasks = [Task(p, s) for s in (2002, 2003, 2004)]

    def rank(trs, with_dfc):
        A = []
        for t in trs:
            Z, base = t.design(MONTH, np.inf)
            M = np.asarray(Z[:, :-1].todense())
            if with_dfc:
                M = np.hstack([M, t.dfc()[:, None]])
            A.append(M - base @ M)
        return np.linalg.matrix_rank(np.vstack(A), tol=1e-9)

    assert rank(tasks[:1], True) - rank(tasks[:1], False) == 0
    assert rank(tasks, True) - rank(tasks, False) == 1


def test_the_consensus_column_carries_the_weak_penalty_and_sits_last(win_panel):
    """Columns are [beta | momentum | dfc], so `penalty` must put `gamma_pen` on the
    last two of `mlr+mom+dfc` and on the last one of `mlr+dfc`, and `sigma_report` --
    which reads `theta[-1]` -- must be reading the consensus coefficient, not momentum."""
    from antiGen.season_momentum import design_for, penalty
    t = Task(win_panel, 2004)
    combo = dict(WIN_COMBO)
    Zp, _, _, _ = design_for(t, combo, "mlr+mom")
    for fam, n_mom in (("mlr+mom+dfc", 2), ("mlr+dfc", 1)):
        Z, base, nm, blocks = design_for(t, combo, fam)
        assert nm == n_mom and blocks == ()
        assert Z.shape[1] == Zp.shape[1] + (1 if fam == "mlr+mom+dfc" else 0)
        assert np.asarray(Z[:, -1].todense()).ravel() == pytest.approx(t.dfc())
        pen = penalty(Z.shape[1], nm, combo["lam"], combo["gamma_pen"])
        assert (pen[-n_mom:] == combo["gamma_pen"]).all()
        assert (pen[:-n_mom] == combo["lam"]).all()


def test_the_consensus_arms_add_no_field_to_the_selectors_grid():
    """The column reads no hyperparameter, so `mlr+mom+dfc` must collapse over exactly
    the candidate list `mlr+mom` does. That is the point of taking the consensus under
    season persistence rather than under the combination's own base measure: the paired
    delta between parent and child is then not confounded with how much the selector had
    to choose among -- the confound the 2026-09-21 round priced at about a point."""
    from antiGen.season_momentum import collapse
    g = make_grid((1, 2), (1.0, np.inf), (1.0, 3.0), (0, 2), (0.0,), ("ewma",), (8,),
                  (0.01, 0.3))
    assert len(collapse(g, "mlr+mom+dfc")) == len(collapse(g, "mlr+mom"))
    # `mlr+dfc` drops momentum, so like `mlr+wmom` it must not see `tau_s`; it does see
    # `gamma_pen`, which `mlr` does not, so it is `mlr`'s list times those values
    assert len(collapse(g, "mlr+dfc")) == 2 * len(collapse(g, "mlr"))


def test_the_consensus_families_run_end_to_end(win_panel, capsys):
    """Through the rolling selector, with delta reported and its sign readable.

    The sign is the falsifiable part, as it is for sigma and phi: a strain further from
    the population consensus should be *less* fit, so delta < 0 is the prediction and a
    column that helps with delta > 0 is a covariate that happens to correlate with
    something rather than the metric Huddleston's code implements."""
    from antiGen.season_momentum import rolling, report
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,), ("ewma",), (8,), (0.01,))
    fams = ("season", "mlr", "mlr+mom", "mlr+dfc", "mlr+mom+dfc")
    rows = rolling(win_panel, drop_targets=(), min_seqs=1, min_train=2, val_folds=2,
                   grid=grid, maxiter=20, families=fams, verbose=False)
    assert rows
    for r in rows:
        for f in fams:
            assert np.isfinite(r["emd_" + f])
        for f in ("mlr+dfc", "mlr+mom+dfc"):
            assert np.isfinite(r["sigma_" + f]) and np.isfinite(r["sigma_se_" + f])
    out = report(rows, fams)
    assert out["mlr+mom+dfc"]["coef_name"] == "delta"
    assert "delta" in capsys.readouterr().out


# ── the feature set as a function of the past only (`--censor-features`) ────
#
# `TipPanel` has always accepted a `feature_mask`, and `huddleston_panel` has always
# passed one; the backbone driver never could, so every backbone number in the log was
# produced with the (position, residue) columns chosen over the whole record -- 1996-2026
# of allele frequencies deciding which alleles the 2017 forecast is allowed a coefficient
# for. The fitted values are still fitted on earlier transitions, so this is an exposure
# in the column *set* and not in the coefficients, but it is one, and at origin 2017 of
# the real backbone 42 of the 239 columns exist only because of post-horizon data and
# vary over the support. These tests pin the flag that removes it.

CENSOR_REF = {"SigPep": "M", "HA1": "AAAA", "HA2": "C", "nuc": "x" * 18}
# Season Y runs 1 Oct Y-1 to 1 Apr Y, so .80 lands in season Y+1 and .10 in season Y.
CENSOR_SEASONS = (2001, 2002, 2003, 2004, 2005)


def _censor_tree(tmp_path, future_allele):
    """A flat tree: four tips a season, position 2 of HA1 carrying the varying allele.

    Every season holds three "A" tips and one that differs, so the panel has something to
    fit. `future_allele` is the residue the *post-horizon* seasons carry at HA1 position 3
    -- the one whose global frequency decides whether that column exists. Seasons up to
    2003 never carry it, so censoring at origin 2003's horizon must make the column
    disappear and rewriting it must then change nothing.
    """
    kids = []
    for k, s in enumerate(CENSOR_SEASONS):
        for j in range(4):
            muts = []
            if j == 3:
                muts.append("A2C")                      # the ordinary varying site
            if s > 2003 and j >= 2:
                muts.append(f"A3{future_allele}")       # the post-horizon allele
            date = (s - 1) + 0.80 + 0.02 * j
            kids.append({"name": f"t{k}_{j}",
                         "branch_attrs": {"mutations": {"HA1": muts}} if muts else {},
                         "node_attrs": {"num_date": {"value": date}}})
    tree = {"tree": {"name": "root", "branch_attrs": {}, "children": kids}}
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "ref.json").write_text(json.dumps(CENSOR_REF))
    (tmp_path / "tree.json").write_text(json.dumps(tree))
    return tmp_path


def _censor_run(tmp_path, future_allele, censor, origin=2003):
    """`season_momentum.main` end to end, returning the output JSON."""
    import sys
    from antiGen import season_momentum as sm

    d = _censor_tree(tmp_path, future_allele)
    out = d / "out.json"
    argv = ["season_momentum", "--tree", str(d / "tree.json"), "--ref", str(d / "ref.json"),
            "--min-allele-freq", "0.1", "--min-season-seqs", "2", "--min-train", "2",
            "--val-folds", "2", "--maxiter", "20", "--drop-targets", "",
            "--tau-f-months", "1", "--tau-s-years", "inf", "--lambdas", "1",
            "--windows", "0", "--alphas", "0", "--gamma-pens", "0.01",
            "--families", "season,mlr,mlr+mom", "--transitions", str(origin),
            "-o", str(out)]
    if censor:
        argv.append("--censor-features")
    old = sys.argv
    try:
        sys.argv = argv
        sm.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text())


def test_the_censor_fixture_can_actually_show_a_feature_set_leak(tmp_path):
    """Without this the invariance test below would pass vacuously.

    The allele at HA1 position 3 appears only in seasons 2004 and 2005, i.e. strictly
    after origin 2003's horizon. Counted over the whole record it clears the 0.1 floor
    and the site contributes two columns; counted only up to the horizon the site is
    monomorphic and contributes none."""
    a = _censor_run(tmp_path / "all", "G", censor=False)
    c = _censor_run(tmp_path / "cut", "G", censor=True)
    assert a["n_features"] > c["n_features"], (
        "censoring removed no column; the fixture proves nothing")
    assert a["censor_at"] is None and c["censor_at"] == 2003.25
    assert c["n_feature_tips"] == 12          # three seasons of four tips, 2001-2003


def test_rewriting_the_future_cannot_move_a_censored_feature_set(tmp_path):
    """The audit in one assertion. Two records identical at or before 2003.25 and
    different after it must give the same columns once the counts are censored -- and
    must not, uncensored, or the fixture is not exercising the exposure."""
    cut_g = _censor_run(tmp_path / "cg", "G", censor=True)
    cut_w = _censor_run(tmp_path / "cw", "W", censor=True)
    assert cut_g["n_features"] == cut_w["n_features"]
    assert cut_g["n_feature_tips"] == cut_w["n_feature_tips"]
    all_g = _censor_run(tmp_path / "ag", "G", censor=False)
    all_w = _censor_run(tmp_path / "aw", "W", censor=False)
    # Same *count* either way here, but the columns are different residues; the point is
    # that the censored panel cannot tell the two records apart at all and the
    # uncensored one is reading a residue no pre-horizon sequence carries.
    assert all_g["n_features"] > cut_g["n_features"]
    assert all_w["n_features"] > cut_w["n_features"]


def test_the_censored_run_still_scores_the_origin_it_was_censored_at(tmp_path):
    """A censored panel must be a panel: the roll still happens and the target is
    untouched. The geometry -- varying sites, codes, transport costs -- is global by
    construction, so the floor may not move when only the feature counts are censored."""
    a = _censor_run(tmp_path / "all", "G", censor=False)
    c = _censor_run(tmp_path / "cut", "G", censor=True)
    (ra,), (rc,) = a["per_transition"], c["per_transition"]
    assert ra["season"] == rc["season"] == 2003
    assert rc["floor"] == pytest.approx(ra["floor"], abs=1e-12)
    assert rc["emd_season"] == pytest.approx(ra["emd_season"], abs=1e-12)
    assert np.isfinite(rc["emd_mlr+mom"])


def test_censoring_refuses_a_panel_it_cannot_censor(tmp_path):
    """One panel, one censoring horizon. Two origins in a single run would need two
    feature sets, and silently taking the first would censor the other at the wrong
    date."""
    for bad in ("", "2003,2004"):
        d = _censor_tree(tmp_path / f"b{len(bad)}", "G")
        import sys
        from antiGen import season_momentum as sm
        argv = ["season_momentum", "--tree", str(d / "tree.json"),
                "--ref", str(d / "ref.json"), "--censor-features"]
        if bad:
            argv += ["--transitions", bad]
        old = sys.argv
        try:
            sys.argv = argv
            with pytest.raises(SystemExit):
                sm.main()
        finally:
            sys.argv = old


# ── the parallel selector ───────────────────────────────────────────────────
#
# `--select-workers` is a performance flag and the tests below are what entitles it to be
# described that way. The selector's combination loop is 84% of a scored task's wall on
# the additive panel at K = 90 -- 13,000 network-simplex solves at about a minute each --
# and the only reason it may be split across processes is that no iteration reads what
# another writes. These check that claim on the two things that could break it: the
# numbers themselves, and the tie-break that decides which of two equal candidates wins.

def _wide_tasks():
    """Four consecutive seasons as `rolling` would set them up: `(tasks, order)`.

    Built fresh on each call so that the serial and the parallel run below cannot share a
    warm `task._cache` -- which is the one piece of state the two paths differ in, and so
    the one the comparison has to hold apart rather than let coincide."""
    tips = []
    for y, g in ((2001, "AAAA"), (2002, "ACAA"), (2003, "ACAG"), (2004, "AGAG")):
        tips += [(y - 0.1, g), (y - 0.05, g), (y + 0.1, "AAAG"), (y + 0.2, g)]
    panel = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    order = [s for s in panel.seasons if (s + 1) in panel.seasons]
    return {s: Task(panel, s) for s in order}, order


@pytest.mark.parametrize("family", ["mlr", "mlr+mom", "ewma"])
def test_worker_pool_reproduces_the_serial_selector_exactly(family):
    """Same chosen combos and the same validation error for every candidate, bit for bit.

    `repr` and not `pytest.approx`: a worker pool that agreed to 1e-12 would mean the two
    paths were doing arithmetic in different orders, which is the finding this test exists
    to rule out, not a pass."""
    from antiGen.season_momentum import select

    tasks, order = _wide_tasks()
    grid = make_grid((1, 3), (1.0, np.inf), (0.3, 3.0), (0, 2), (0.0, 0.05))
    i = len(order) - 1
    ser_c, ser_e = select(tasks, order, i, grid, family, 3, 10, 0.01, top_k=2)
    tasks2, order2 = _wide_tasks()
    par_c, par_e = select(tasks2, order2, i, grid, family, 3, 10, 0.01, top_k=2,
                          workers=3)
    assert repr(ser_e) == repr(par_e)
    assert [sorted(c.items(), key=repr) for c in ser_c] == \
           [sorted(c.items(), key=repr) for c in par_c]


def test_the_tie_break_is_grid_order_on_both_paths():
    """Among candidates with equal validation error the earliest in the grid wins, and a
    NaN never wins at all. The parallel path returns its pairs out of order, so it has to
    re-impose this rather than inherit it from the loop."""
    from antiGen.season_momentum import select_chunks

    # The ordering contract, stated directly on the scan `select` runs over `errs`.
    errs = {0: 2.0, 1: 1.0, 2: 1.0, 3: float("nan"), 4: 3.0}
    best, best_err = None, np.inf
    for c in sorted(errs):
        if errs[c] < best_err:
            best, best_err = c, errs[c]
    assert best == 1, "the first of two equal minima, not the last"
    assert sorted(errs, key=lambda c: errs[c])[:2] == [1, 2], "stable within a tie"
    # And the chunking never reorders the grid it is cutting.
    assert [a for a, _ in select_chunks([{}] * 10, 3)] == \
           sorted(a for a, _ in select_chunks([{}] * 10, 3))


def test_chunks_cover_the_grid_once_and_never_split_a_cache_group():
    """`mu` is the fastest-varying axis and `_momentum_prediction` memoises the parent a
    run of `mu`s shares, so a chunk boundary inside such a run would refit it once per
    piece -- five times the fits for the same numbers."""
    from antiGen.season_momentum import cache_groups, collapse, select_chunks

    grid = make_grid((1, 3), (1.0, np.inf), (0.3, 3.0), (0, 2), (0.0,),
                     mus=(0, 0.05, 0.1, 0.2, 0.4), kernels=("kcmom",))
    g = collapse(grid, "mlr+mom+K")
    groups = cache_groups(g)
    assert [b - a for a, b in groups] == [5] * (len(g) // 5), "one group per mu-run"
    for n in (1, 2, 3, 7, 64):
        chunks = select_chunks(g, n)
        assert chunks[0][0] == 0 and chunks[-1][1] == len(g)
        assert all(x[1] == y[0] for x, y in zip(chunks, chunks[1:])), "contiguous cover"
        starts = {a for a, _ in groups}
        assert all(a in starts for a, _ in chunks), "no cut inside a cache group"
