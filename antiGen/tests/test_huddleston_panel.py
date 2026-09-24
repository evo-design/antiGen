"""Our model on the Huddleston et al. panel: pivots, populations, and the naive identity."""
import os

import numpy as np
import pandas as pd
import pytest

from antiGen.huddleston_panel import (BUILD_DIR, TIP_ATTRS, PanelTask, build_panel,
                                      censor_horizon, decimal, make_tasks, read_build,
                                      read_forecasts, read_published, shift_months)
from antiGen.huddleston_panel import (PANEL_FAMILIES, collapse_panel, run_family_panel,
                                      select_panel, train_before, usable_before)
from antiGen.season_momentum import collapse, make_grid, predict, run_family
from scipy import sparse

REPO = os.path.expanduser("~/flu-forecasting")
needs_repo = pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO, "results", "builds", "natural",
                                    BUILD_DIR["validation"], TIP_ATTRS)),
    reason="needs a clone of github.com/blab/flu-forecasting at ~/flu-forecasting")

# Four sites, two builds, pivots six months apart. "ACAA" is the newcomer: it appears
# late in 2001 and takes over in 2002, which is what a momentum column is for.
SEQS = {"a": "AAAA", "b": "AAAG", "c": "ACAA", "d": "ACAG"}


def _rows(build, spec):
    """spec: {timepoint: [(strain, key, numdate, frequency), ...]}."""
    out = []
    for t, tips in spec.items():
        for s, k, d, f in tips:
            out.append(dict(strain=s, timepoint=t, numdate=d, aa_sequence=SEQS[k],
                            frequency=f))
    return pd.DataFrame(out)


def _write(tmp_path, build, df):
    d = tmp_path / "results" / "builds" / "natural" / BUILD_DIR[build]
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / TIP_ATTRS, sep="\t", index=False)
    return d


VAL = {
    "2000-10-01": [("s1", "a", 2000.70, 0.6), ("s2", "b", 2000.72, 0.4)],
    "2001-04-01": [("s3", "a", 2001.20, 0.5), ("s4", "c", 2001.22, 0.5)],
    "2001-10-01": [("s5", "c", 2001.70, 0.7), ("s6", "b", 2001.72, 0.3)],
    "2002-04-01": [("s7", "c", 2002.20, 0.8), ("s8", "d", 2002.22, 0.2)],
    "2002-10-01": [("s9", "d", 2002.70, 1.0)],
}


@pytest.fixture
def panel_fixture(tmp_path):
    _write(tmp_path, "validation", _rows("validation", VAL))
    return build_panel(str(tmp_path), ["validation"], min_allele_freq=0.0, lookback=5.0,
                       verbose=False)


# ── the date conventions ────────────────────────────────────────────────────

def test_pivots_are_on_the_twelfth_grid_not_the_day_count():
    """1 April is year.25 and 1 October year.75, which is where augur puts its pivots.

    Their 2002-04-01 population holds a strain dated 2002.249144. That sits above the day
    count reading of the same label, 2002 + 90/365.25, and below its pivot -- so only the
    pivot convention leaves the population censored."""
    assert decimal("2002-04-01") == pytest.approx(2002.25)
    assert decimal("2015-10-01") == pytest.approx(2015.75)
    assert 2002 + 90 / 365.25 < 2002.249144 < decimal("2002-04-01")


def test_twelve_months_on_lands_on_the_same_month():
    assert shift_months("2015-10-01", 12) == "2016-10-01"
    assert shift_months("2015-04-01", 12) == "2016-04-01"
    assert shift_months("2015-10-01", 6) == "2016-04-01"


# ── the panel ───────────────────────────────────────────────────────────────

def test_every_population_sequence_is_a_genotype_of_the_panel(panel_fixture):
    panel, _, pops, _ = panel_fixture
    for (_, t), (sup, w, n, hz, pv) in pops.items():
        assert len(sup) == len(w) and w.sum() == pytest.approx(1.0)
        assert set(sup) <= set(range(len(panel.genotypes)))


def test_a_strain_reconstructed_two_ways_keeps_both_haplotypes(tmp_path):
    """175 validation strains carry different HA translations at their two timepoints,
    because each timepoint reconstructs ambiguous sites on its own tree. Deduplicating by
    strain alone would drop one of the two, and a population would then hold a genotype
    the panel has never seen."""
    spec = {
        "2001-04-01": [("twin", "a", 2001.20, 1.0)],
        "2001-10-01": [("twin", "b", 2001.20, 1.0)],       # same strain, other residue
    }
    _write(tmp_path, "validation", _rows("validation", spec))
    panel, _, pops, seq_at = build_panel(str(tmp_path), ["validation"],
                                         min_allele_freq=0.0, verbose=False)
    assert SEQS["a"] in panel.gid and SEQS["b"] in panel.gid
    assert seq_at[("validation", "2001-04-01", "twin")] == SEQS["a"]
    assert seq_at[("validation", "2001-10-01", "twin")] == SEQS["b"]


def test_a_population_from_the_future_is_refused(tmp_path):
    spec = {"2001-04-01": [("s1", "a", 2001.20, 0.5), ("late", "c", 2001.60, 0.5)]}
    _write(tmp_path, "validation", _rows("validation", spec))
    with pytest.raises(ValueError, match="not censored"):
        build_panel(str(tmp_path), ["validation"], min_allele_freq=0.0, verbose=False)


def test_tasks_need_a_target_twelve_months_on(panel_fixture):
    panel, in_build, pops, _ = panel_fixture
    tasks, order = make_tasks(panel, in_build, pops, lookback=5.0)
    # Five timepoints, so three have a target a year later; they come out in order.
    assert [t for _, t in order] == ["2000-10-01", "2001-04-01", "2001-10-01"]
    assert tasks[("validation", "2001-04-01")].tlabel == "2002-04-01"


# ── the task ────────────────────────────────────────────────────────────────

def test_the_support_is_theirs_and_so_is_the_floor(panel_fixture):
    panel, in_build, pops, _ = panel_fixture
    tasks, _ = make_tasks(panel, in_build, pops, lookback=5.0)
    task = tasks[("validation", "2001-04-01")]
    sup, kde, _, _, _ = pops[("validation", "2001-04-01")]
    assert np.array_equal(task.sup, sup)
    assert task.floor == pytest.approx(float(task.tgt_w @ task.M.min(axis=0)))
    assert task.assigned.sum() == pytest.approx(1.0)


def test_the_season_baseline_is_their_naive_forecast(panel_fixture):
    """On this panel `season` is not a uniform measure over a season's sequences: it is
    their KDE frequency, which is exactly what their naive model predicts."""
    panel, in_build, pops, _ = panel_fixture
    tasks, _ = make_tasks(panel, in_build, pops, lookback=5.0)
    task = tasks[("validation", "2001-04-01")]
    kde = pops[("validation", "2001-04-01")][1]
    assert np.allclose(task.baseline("season"), kde)
    assert np.allclose(task.base_measure("season", 1.0), kde)
    # and it is a different vector from the EWMA level, or the row would be vacuous
    assert not np.allclose(task.level(1.0 / 12), kde)


def test_the_mask_holds_only_the_past_and_honours_the_corpus_mode(tmp_path):
    """Under `corpus="build"` a task averages over its own build only; under `union` it
    pools them. Never, in either mode, over a sequence dated past its horizon."""
    _write(tmp_path, "validation", _rows("validation", VAL))
    # Timepoints that give the other build no origin of its own (no target a year on),
    # so it contributes only tips -- which is the thing being tested.
    other = {
        "2001-04-01": [("x1", "a", 2001.21, 0.5), ("x2", "c", 2001.23, 0.5)],
        "2001-10-01": [("x3", "c", 2001.71, 1.0)],
    }
    _write(tmp_path, "test", _rows("test", other))
    panel, in_build, pops, _ = build_panel(str(tmp_path), ["validation", "test"],
                                            min_allele_freq=0.0, verbose=False)
    tasks, _ = make_tasks(panel, in_build, pops, lookback=5.0, corpus="build")
    task = tasks[("validation", "2001-04-01")]
    assert in_build["validation"][task.mask].all()
    assert panel.dates[task.mask].max() <= task.horizon + 1e-12
    # every support genotype has some tip behind it, so no base weight is structurally 0
    assert (task.level(1.0 / 12) > 0).all()
    # pooling can only add tips, and never any dated after the horizon
    pooled = make_tasks(panel, in_build, pops, lookback=5.0,
                        corpus="union")[0][("validation", "2001-04-01")]
    assert pooled.mask.sum() >= task.mask.sum()
    assert panel.dates[pooled.mask].max() <= pooled.horizon + 1e-12


def test_a_forecast_table_becomes_weights_over_the_support(panel_fixture):
    """Their `forecasts_*.tsv` are per strain; ours is per haplotype. Strains collapse
    onto haplotypes the same way the target does, and the naive table comes back as the
    KDE vector it started from."""
    panel, in_build, pops, seq_at = panel_fixture
    tasks, _ = make_tasks(panel, in_build, pops, lookback=5.0)
    task = tasks[("validation", "2001-04-01")]
    rows = [(s, f) for s, _, _, f in VAL["2001-04-01"]]
    assert np.allclose(task.weights_of(rows, seq_at), task.baseline("season"))


def test_a_strain_outside_the_support_is_refused(panel_fixture):
    panel, in_build, pops, seq_at = panel_fixture
    tasks, _ = make_tasks(panel, in_build, pops, lookback=5.0)
    task = tasks[("validation", "2001-04-01")]
    seq_at[("validation", "2001-04-01", "ghost")] = SEQS["d"]
    with pytest.raises(KeyError):
        task.weights_of([("ghost", 1.0)], seq_at)


def test_zero_coefficients_reproduce_the_base_measure(panel_fixture):
    """The nesting that makes a percentage against `season` meaningful: with no tilt the
    forecast is the base measure, so a family can only differ by what it fits."""
    panel, in_build, pops, _ = panel_fixture
    tasks, _ = make_tasks(panel, in_build, pops, lookback=5.0)
    task = tasks[("validation", "2001-10-01")]
    Z, base = task.design(1.0 / 12, 1.0)
    assert np.allclose(predict(np.zeros(Z.shape[1]), Z, base), base)


def test_families_run_end_to_end_on_the_fixture(panel_fixture):
    panel, in_build, pops, _ = panel_fixture
    tasks, order = make_tasks(panel, in_build, pops, lookback=5.0)
    grid = make_grid((1.0,), (1.0,), (1.0,), (2,), (0.0,), ("ewma", "season"))
    i = len(order) - 1
    for fam in ("season", "last3m", "ewma", "mlr", "mlr+mom", "mlr+mom+sb"):
        tr = train_before(tasks, order, i, grid[0]["window"])
        w = run_family(tasks[order[i]], grid[0], fam, tr, 20, 0.01)
        assert w.shape == tasks[order[i]].sup.shape
        assert w.sum() == pytest.approx(1.0)
        assert np.isfinite(tasks[order[i]].score(w))


# ── against their own files ─────────────────────────────────────────────────

@needs_repo
def test_our_naive_is_their_naive_on_the_real_panel():
    """The one identity that makes every percentage in this round comparable with theirs:
    our `season` baseline and their shipped `naive` forecast are the same distribution,
    so they must score identically under our transport solve."""
    panel, in_build, pops, seq_at = build_panel(REPO, ["test"], verbose=False)
    tasks, _ = make_tasks(panel, in_build, pops, lookback=5.0)
    forecasts = read_forecasts(REPO, "test")["naive"]
    for label in ("2015-10-01", "2018-10-01"):
        task = tasks[("test", label)]
        g = forecasts[label]
        theirs = task.weights_of(zip(g.strain.values, g.projected_frequency.values),
                                 seq_at)
        assert task.score(theirs) == pytest.approx(task.score(task.baseline("season")),
                                                   abs=1e-9)


@needs_repo
def test_the_scored_origins_are_the_ones_they_publish():
    panel, in_build, pops, _ = build_panel(REPO, ["validation", "test"], verbose=False)
    tasks, _ = make_tasks(panel, in_build, pops, lookback=5.0)
    evalset, published = read_published(REPO)
    evalset = {k: v for k, v in evalset.items() if k in tasks}
    assert sum(1 for v in evalset.values() if v == "validation") == 23
    assert sum(1 for v in evalset.values() if v == "test") == 8
    assert all(k in tasks for k in evalset)


# ── leakage: mechanical, not argued ─────────────────────────────────────────
#
# The invariance these tests turn on: a forecast for origin `i` must be a function of
# what was observable at `i`. Rewriting everything dated after `i` -- the sequences, and
# so the target, the assignment, the cost matrix and the allele statistics -- must leave
# the forecast for `i` bit-identical. Anything that moves came from the future.
# Comparison is keyed by sequence rather than by index, because rewriting the future
# renumbers genotype ids and can permute the support's order.
#
# The fixture is built to make the *feature-set* exposure visible, which takes some care:
# residue C at site 1 is carried by 2 of the 12 pre-horizon tips (0.167, under the 0.2
# allele-frequency floor) and by a support genotype at the origin and at an earlier
# origin, so the column has a gradient if it exists at all. Whether it exists depends on
# how common C becomes *after* the origin -- which is exactly the leak.

MAF = 0.2
LEAK = {
    "2000-10-01": [("a1", "a", 2000.70, 0.4), ("a2", "b", 2000.72, 0.3),
                   ("a3", "a", 2000.74, 0.3)],
    "2001-04-01": [("b1", "a", 2001.20, 0.4), ("b2", "c", 2001.22, 0.3),
                   ("b3", "b", 2001.24, 0.3)],
    "2001-10-01": [("c1", "a", 2001.70, 0.4), ("c2", "b", 2001.72, 0.3),
                   ("c3", "a", 2001.74, 0.3)],
    "2002-04-01": [("d1", "a", 2002.20, 0.4), ("d2", "c", 2002.22, 0.3),
                   ("d3", "b", 2002.24, 0.3)],
    # everything below is after the origin's horizon
    "2002-10-01": [("e1", "c", 2002.70, 0.5), ("e2", "c", 2002.72, 0.5)],
    "2003-04-01": [("f1", "c", 2003.20, 0.5), ("f2", "c", 2003.22, 0.5)],
}
ORIGIN, HZ = "2002-04-01", 2002.25
# The future with C taken out of it: same dates, same names, same frequencies.
REWRITTEN = {"c": "b", "a": "a", "b": "b", "d": "d"}


def _rewrite_future(spec, horizon):
    return {t: [(s, REWRITTEN[k] if d > horizon else k, d, f) for s, k, d, f in tips]
            for t, tips in spec.items()}


def _forecast(tmp_path, spec, origin, family, censor, maf=MAF):
    _write(tmp_path, "validation", _rows("validation", spec))
    at = censor_horizon(str(tmp_path), ["validation"], origin) if censor else None
    panel, in_build, pops, _ = build_panel(str(tmp_path), ["validation"],
                                            min_allele_freq=maf, verbose=False,
                                            censor_at=at)
    tasks, order = make_tasks(panel, in_build, pops, lookback=5.0)
    i = order.index(("validation", origin))
    grid = make_grid((1.0, 3.0), (1.0,), (0.3, 3.0), (2,), (0.0,), ("ewma",))
    combos, _ = select_panel(tasks, order, i, grid, family, 2, 40, 0.01)
    tr = train_before(tasks, order, i, combos[0]["window"])
    w = run_family(tasks[order[i]], combos[0], family, tr, 40, 0.01)
    return ({panel.genotypes[g]: float(x) for g, x in zip(tasks[order[i]].sup, w)},
            set(panel.features))


def test_the_fixture_can_actually_show_a_feature_set_leak(tmp_path):
    """Without this the invariance tests below would pass vacuously.

    Counting allele frequencies over the whole record puts (site 1, C) in the feature
    set; counting them only up to the origin leaves it out, because C is at 0.167 there
    against a floor of 0.2. So the uncensored feature set is a function of the future,
    and this fixture is one where that function actually varies."""
    _, f_all = _forecast(tmp_path / "a", LEAK, ORIGIN, "mlr", censor=False)
    _, f_cut = _forecast(tmp_path / "b", LEAK, ORIGIN, "mlr", censor=True)
    _, f_alt = _forecast(tmp_path / "c", _rewrite_future(LEAK, HZ), ORIGIN, "mlr",
                         censor=False)
    assert f_all != f_cut, "censoring changed nothing; the fixture proves nothing"
    assert f_all != f_alt, "rewriting the future changed no feature; likewise"
    assert f_cut == _forecast(tmp_path / "d", _rewrite_future(LEAK, HZ), ORIGIN, "mlr",
                              censor=True)[1]


@pytest.mark.parametrize("family", ["ewma", "mlr", "mlr+mom"])
def test_rewriting_the_future_cannot_move_a_censored_forecast(tmp_path, family):
    """The whole audit in one assertion, per family."""
    a, _ = _forecast(tmp_path / "a", LEAK, ORIGIN, family, censor=True)
    b, _ = _forecast(tmp_path / "b", _rewrite_future(LEAK, HZ), ORIGIN, family,
                     censor=True)
    assert set(a) == set(b)
    for g in a:
        assert a[g] == pytest.approx(b[g], abs=1e-12), f"{family} moved at {g}"


def test_uncensored_features_leave_the_forecast_exposed(tmp_path):
    """The exposure the flag exists to remove, as a measured fact rather than a worry.

    Recorded as a test so that the claim in NOTES -- that the feature-set exposure is
    second-order but not zero -- has something behind it, and so that the day it becomes
    zero somebody has to come and say so."""
    a, _ = _forecast(tmp_path / "a", LEAK, ORIGIN, "mlr", censor=False)
    b, _ = _forecast(tmp_path / "b", _rewrite_future(LEAK, HZ), ORIGIN, "mlr",
                     censor=False)
    keys = set(a) | set(b)
    moved = max(abs(a.get(g, 0.0) - b.get(g, 0.0)) for g in keys)
    assert moved > 1e-12, ("uncensored features no longer depend on the future on this "
                           "fixture; if that is now true in general, say so and drop "
                           "the flag")


def test_only_origins_whose_target_has_closed_may_be_used(tmp_path):
    """The invariant the rolling protocol rests on, on the indices rather than in a
    docstring: nothing a fit or a fold reads may still have been unfolding at the
    origin's horizon.

    On a 6-month pivot grid with a 12-month horizon that costs exactly one transition --
    `order[:i]` would include an origin forecasting to six months past `i`."""
    _write(tmp_path, "validation", _rows("validation", LEAK))
    panel, in_build, pops, _ = build_panel(str(tmp_path), ["validation"],
                                            min_allele_freq=MAF, verbose=False)
    tasks, order = make_tasks(panel, in_build, pops, lookback=5.0)
    for i in range(len(order)):
        T = tasks[order[i]].horizon
        use = usable_before(tasks, order, i)
        assert use == list(range(max(0, i - 1)))          # j <= i - 2
        for j in use:
            assert tasks[order[j]].t_horizon <= T + 1e-9
        for t_ in train_before(tasks, order, i, 0):
            assert t_.t_horizon <= T + 1e-9


# ── their metrics as columns of our design ──────────────────────────────────

PRED = {
    "2000-10-01": [("a1", "a", 2000.70, 0.4), ("a2", "b", 2000.72, 0.6)],
    "2001-04-01": [("b1", "a", 2001.20, 0.5), ("b2", "c", 2001.22, 0.5)],
    "2001-10-01": [("c1", "a", 2001.70, 0.4), ("c2", "b", 2001.72, 0.6)],
    "2002-04-01": [("d1", "a", 2002.20, 0.3), ("d2", "c", 2002.22, 0.7)],
    "2002-10-01": [("e1", "c", 2002.70, 0.5), ("e2", "b", 2002.72, 0.5)],
    "2003-04-01": [("f1", "c", 2003.20, 0.6), ("f2", "a", 2003.22, 0.4)],
}


def _write_with_predictors(tmp_path, spec, lbi_of):
    df = _rows("validation", spec)
    df["lbi"] = [lbi_of[s] for s in df.strain]
    df["ne_star"] = [len(s) for s in df.strain]
    df["delta_frequency"] = 0.0
    df["cTiter_x"] = -7.0 + df["lbi"]
    d = tmp_path / "results" / "builds" / "natural" / BUILD_DIR["validation"]
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / TIP_ATTRS, sep="\t", index=False)


@pytest.fixture
def pred_panel(tmp_path):
    lbi = {s: 0.1 * (i + 1) for i, s in enumerate(
        [t[0] for tips in PRED.values() for t in tips])}
    _write_with_predictors(tmp_path, PRED, lbi)
    panel, in_build, pops, seq_at = build_panel(
        str(tmp_path), ["validation"], min_allele_freq=0.0, verbose=False,
        predictors=("lbi", "ne_star", "delta_frequency", "cTiter_x"))
    return make_tasks(panel, in_build, pops, lookback=5.0), pops


def test_their_metrics_are_standardised_within_the_timepoint(pred_panel):
    """Centred on the KDE-weighted mean and scaled by the KDE-weighted std, which is what
    their own `predict` does -- so a coefficient means the same thing in both models."""
    (tasks, _), pops = pred_panel
    task = tasks[("validation", "2001-04-01")]
    x = task._preds["lbi"]
    w = task.baseline("season")
    assert float(w @ x) == pytest.approx(0.0, abs=1e-12)
    assert float(np.sqrt(w @ (x - float(w @ x)) ** 2)) == pytest.approx(1.0, abs=1e-12)


def test_a_haplotype_gets_the_frequency_weighted_mean_of_its_strains(tmp_path):
    """Two strains, one haplotype: the column is their KDE-weighted mean, not either
    value and not the unweighted mean."""
    spec = {"2001-04-01": [("x", "a", 2001.20, 0.25), ("y", "a", 2001.22, 0.75)],
            "2001-10-01": [("z", "b", 2001.70, 1.0)]}
    _write_with_predictors(tmp_path, spec, {"x": 0.0, "y": 1.0, "z": 0.5})
    _, _, pops, _ = build_panel(str(tmp_path), ["validation"], min_allele_freq=0.0,
                                verbose=False, predictors=("lbi",))
    assert pops[("validation", "2001-04-01")][4]["lbi"][0] == pytest.approx(0.75)


def test_a_zero_coefficient_on_their_metric_reproduces_the_parent(pred_panel):
    """`mlr+mom+lbi` must nest `mlr+mom`, and `hudd` must nest their naive model. Without
    that the families are a horse race rather than a decomposition."""
    (tasks, order), _ = pred_panel
    i = len(order) - 1
    task = tasks[order[i]]
    combo = make_grid((1.0,), (1.0,), (1.0,), (2,), (0.0,), ("ewma",))[0]
    Z = sparse.hstack([task.design(combo["tau_f"], combo["tau_s"])[0],
                       task.their_design(("lbi",))]).tocsr()
    base = task.base_measure("ewma", combo["tau_f"])
    assert np.allclose(predict(np.zeros(Z.shape[1]), Z, base), base)
    Zh = task.their_design(("ne_star", "lbi"))
    assert np.allclose(predict(np.zeros(Zh.shape[1]), Zh, task.baseline("season")),
                       task.baseline("season"))


@pytest.mark.parametrize("family", sorted(PANEL_FAMILIES))
def test_every_panel_family_runs_and_returns_a_distribution(pred_panel, family):
    (tasks, order), _ = pred_panel
    i = len(order) - 1
    grid = make_grid((1.0,), (1.0,), (1.0,), (2,), (0.0,), ("ewma",))
    tr = train_before(tasks, order, i, 2)
    w = run_family_panel(tasks[order[i]], grid[0], family, tr, 40, 0.3)
    assert w.shape == tasks[order[i]].sup.shape
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_a_family_not_in_the_table_is_delegated_unchanged(pred_panel):
    """Every arm run before these families existed must be reproducible to the bit."""
    (tasks, order), _ = pred_panel
    i = len(order) - 1
    grid = make_grid((1.0,), (1.0,), (1.0,), (2,), (0.0,), ("ewma",))
    tr = train_before(tasks, order, i, 2)
    for fam in ("season", "ewma", "mlr", "mlr+mom"):
        a = run_family_panel(tasks[order[i]], grid[0], fam, tr, 40, 0.3)
        b = run_family(tasks[order[i]], grid[0], fam, tr, 40, 0.3)
        assert np.array_equal(a, b), fam


def test_the_hudd_families_cannot_see_fields_they_have_no_column_for(pred_panel):
    """`hudd` has no allele features and no momentum, so `lam` and `tau_s` change nothing
    about it; letting the selector range over them would decide the argmin by iteration
    order among identical candidates."""
    grid = make_grid((0.5, 1.0), (1.0, np.inf), (0.3, 3.0), (2,), (0.0,), ("ewma",))
    assert len(collapse_panel(grid, "hudd")) == 1
    assert len(collapse_panel(grid, "mlr+mom+lbi")) == len(collapse(grid, "mlr+mom"))


def test_two_builds_at_one_pivot_need_an_explicit_owner(tmp_path):
    """At a 6-month horizon the validation build gains an origin that collides with the
    test build's. Silently keeping either would let one span of evolution vote twice in
    the history, so an unresolved collision raises and `prefer` settles it."""
    val = {"2001-04-01": [("v1", "a", 2001.20, 1.0)],
           "2001-10-01": [("v2", "b", 2001.70, 1.0)]}
    tst = {"2001-04-01": [("t1", "c", 2001.21, 1.0)],
           "2001-10-01": [("t2", "d", 2001.71, 1.0)]}
    _write(tmp_path, "validation", _rows("validation", val))
    _write(tmp_path, "test", _rows("test", tst))
    panel, in_build, pops, _ = build_panel(str(tmp_path), ["validation", "test"],
                                           min_allele_freq=0.0, verbose=False)
    with pytest.raises(ValueError, match="no preference"):
        make_tasks(panel, in_build, pops, lookback=5.0, delta_months=6)
    tasks, order = make_tasks(panel, in_build, pops, lookback=5.0, delta_months=6,
                              prefer={"2001-04-01": "test"})
    assert order == [("test", "2001-04-01")]
    assert ("validation", "2001-04-01") not in tasks


def test_the_twelve_month_horizon_needs_no_preference(tmp_path):
    """The collision is specific to shorter horizons; at 12 months the two builds'
    scoreable origins are disjoint and nothing is dropped."""
    val = {"2001-04-01": [("v1", "a", 2001.20, 1.0)],
           "2002-04-01": [("v2", "b", 2002.20, 1.0)]}
    tst = {"2003-04-01": [("t1", "c", 2003.20, 1.0)],
           "2004-04-01": [("t2", "d", 2004.20, 1.0)]}
    _write(tmp_path, "validation", _rows("validation", val))
    _write(tmp_path, "test", _rows("test", tst))
    panel, in_build, pops, _ = build_panel(str(tmp_path), ["validation", "test"],
                                           min_allele_freq=0.0, verbose=False)
    tasks, order = make_tasks(panel, in_build, pops, lookback=5.0, delta_months=12)
    assert order == [("validation", "2001-04-01"), ("test", "2003-04-01")]
