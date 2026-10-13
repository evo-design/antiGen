"""The mutation kernel: the recursion, the dedup, the duals, the gate, and the nesting.

The fourteen tests `PROPOSAL_mutation_kernel.md` section 9 asks for, in its order. Three
of them exist because something was already wrong: the recursion's signs were transposed
in the probe and in the log (191 of 200 brute-force draws disagreed), the dedup by cost
row is claimed to be exact rather than approximate, and `kemp`'s window is the one place
a one-year off-by-one would turn the primary arm into a leaky one without changing a
single number visibly.
"""
import numpy as np
import pytest

from antiGen.mutation_kernel import (AA, ARMS, BREAK_EVEN, CONS_DELTA, CONS_FAST,
                                     CONS_SLOW_MULT, Candidates, Forecast,
                                     apply_substitution, average, candidates,
                                     causal_operating_point, convolve, reads_slow,
                                     consensus_freq, descent, empirical_counts, gate,
                                     golden_mu, one_step_floor, site_allele_freq,
                                     candidate_cap, MIN_CANDIDATE_CAP, _columns)
from antiGen.season_emd import emd
from antiGen.season_momentum import (K_FAMILIES, MU_FIXED, Task, TipPanel, design_for,
                                     fit_family, make_grid, run_family, train_set)

# Five varying sites over four seasons, and the 2001 support is chosen so that all three
# floors are distinct. Its targets are ACCAA (in the support), ACCGA (one substitution
# out) and GCCGA (two), giving an unextended floor of 1.0. Site 0 carries a G in the
# support, so an *observed*-residue kernel can reach GCCGA's neighbourhood and the
# observed ceiling is 0.75. Site 3 is monomorphic within the support -- every genotype
# there carries A -- while varying panel-wide, so only a novel residue reaches ACCGA and
# the all-residue ceiling is 0.25. That is the degenerate case of test 13 and the
# fixture's version of the 0.088 gap the PLM's whole prize sits in.
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


@pytest.fixture
def task(panel):
    return Task(panel, 2001)


def _marg_panel(seed=0, tips=TIPS):
    """A panel carrying synthetic PLM marginals, for the two conditional arms.

    Random rather than real: what the arms are being tested on is the plumbing -- the
    alphabet mapping, the novel-residue channel, the global renormalisation -- and a real
    ESM-2 pass would make the test need a GPU and a download to assert an equality that
    holds for any table."""
    base = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    rng = np.random.default_rng(seed)
    lp = {}
    for g in base.genotypes:
        z = rng.standard_normal((len(base.sites), len(AA)))
        lp[g] = (z - np.log(np.exp(z).sum(axis=1, keepdims=True))).astype(np.float32)
    return TipPanel(tips, min_allele_freq=0.0, lookback=5.0,
                    marginals=dict(alphabet=AA, sites=base.sites, logp=lp))


def _brute_cost(panel, sup, tgt_g, cand):
    """The candidate block computed by materialising every sequence, the slow way."""
    chars = panel.alphabet + [c for c in AA if c not in panel.code]
    out = np.empty((len(cand), len(tgt_g)), dtype=np.int64)
    for i, (v, q, b) in enumerate(zip(cand.parent, cand.site, cand.res)):
        w = apply_substitution(panel.genotypes[sup[v]], panel.sites, q, b, panel)
        for j, u in enumerate(tgt_g):
            t = panel.genotypes[u]
            out[i, j] = sum(1 for a, c in zip(w, t) if a != c)
    assert chars                                  # the encoding the helper relies on
    return out


# ── 1. nesting ──────────────────────────────────────────────────────────────

def test_mu_zero_reproduces_the_parent_bit_identically(task):
    """`mu = 0` must be the parent *exactly*, not to within a tolerance: it is what makes
    `mlr+mom+K` nest `mlr+mom` and what makes a selector that returns zero in every
    transition a clean result rather than a failed run."""
    cand = task.candidates("kunif")
    x = task.baseline("season")
    fc = convolve(x, cand, 0.0)
    assert fc.blocks == ()                        # no block at all, not an empty one
    assert np.array_equal(fc.w, x)
    assert task.score(fc) == task.score(x)


def test_zero_candidate_mass_scores_as_the_unextended_forecast(task):
    """A forecast carrying a block of zeros must score identically to one carrying no
    block. Otherwise the extended transport problem is charging for rows nothing sits on,
    and every kernel row would be shifted by a constant nobody could see."""
    cand = task.candidates("kunif")
    x = task.baseline("season")
    padded = Forecast(x, ((np.zeros(cand.n_rows), cand.cost),))
    assert padded.sum() == pytest.approx(1.0)
    assert task.score(padded) == pytest.approx(task.score(x), abs=1e-9)


# ── 2. K is a kernel ────────────────────────────────────────────────────────

@pytest.mark.parametrize("arm", ["kunif", "kemp", "kcons", "kcmom"])
def test_the_kernel_is_row_stochastic_with_no_self_loops(task, arm):
    cand = task.candidates(arm, horizon_free := 1.0)
    assert horizon_free == 1.0                    # tau_k, positionally
    A = task.panel.codes[task.sup]
    # No self-loop: a parent never proposes the residue it already carries.
    assert not np.any(A[cand.parent, cand.site] == cand.res)
    # Row-stochastic: every parent's proposal sums to one. `kemp`'s counts are sparse, so
    # the fallback to the control's row is what keeps this true rather than a claim that
    # every count is positive.
    p = cand.p_parent
    assert np.allclose(p, 1.0)
    assert cand.moved(task.baseline("season")) == pytest.approx(1.0)


def test_the_convolution_conserves_mass_even_when_a_parent_proposes_nothing(panel):
    """A support with one genotype has no substitution to propose, so `K` has an empty
    row and that parent's mass must stay where it is. Scaling it by `1 - mu` anyway would
    hand the solver a forecast that no longer sums to one."""
    one = TipPanel([(2000.9, "AAAAA"), (2001.8, "ACAAA"), (2001.9, "ACAAA")],
                   min_allele_freq=0.0, lookback=5.0)
    t = Task(one, 2001)
    assert len(t.sup) == 1
    cand = t.candidates("kunif")
    assert len(cand) == 0
    fc = convolve(t.baseline("season"), cand, 0.4)
    assert fc.sum() == pytest.approx(1.0)
    assert fc.blocks == ()


# ── 3. the recursion ────────────────────────────────────────────────────────

@pytest.mark.parametrize("arm", ["kunif", "kemp", "kcons", "kcmom"])
def test_the_recursion_matches_brute_force_hamming(task, arm):
    """`d(v ⊕ (q,b), u) = d(v,u) + 1[v_q = u_q] - 1[b = u_q]`.

    The transposed form -- the one the 2026-09-21 kernel entry and `probe2.py` both
    recorded -- disagreed with brute force on 191 of 200 random draws, which is why this
    is first among the substantive tests."""
    cand = task.candidates(arm)
    got = cand.cost[cand.row].astype(np.int64)
    assert np.array_equal(got, _brute_cost(task.panel, task.sup, task.tgt_g, cand))


def test_a_novel_residue_is_a_mismatch_everywhere(task):
    """A residue the panel never observed is coded past the end of the alphabet, so it
    equals no target's code. That is the only way the recursion can price a substitution
    to something it has no column for."""
    A = task.panel.codes[task.sup]
    col_site, col_res = _columns(task.panel, A, "all")
    novel = col_res >= len(task.panel.alphabet)
    assert novel.any(), "the fixture must leave some residue of AA unobserved"
    B = task.panel.codes[task.tgt_g]
    for q, b in zip(col_site[novel].tolist(), col_res[novel].tolist()):
        assert not np.any(B[:, q] == b)


# ── 4. the dedup is exact ───────────────────────────────────────────────────

def test_cost_row_dedup_leaves_the_transport_value_unchanged(task):
    """Rows with identical costs are interchangeable in the LP, so summing their masses
    cannot change the optimum. The claim is exactness, not a tolerance, so it is checked
    against the unmerged problem on a real transition."""
    cand = task.candidates("kunif")
    assert cand.n_rows < len(cand), "the fixture must actually have duplicate rows"
    x = task.baseline("season")
    mu = 0.3
    merged = convolve(x, cand, mu)
    a_m, C_m = merged.stacked(task.M)
    # The same problem with one row per candidate, masses unmerged.
    full = np.concatenate([x * (1.0 - mu * cand.p_parent), mu * x[cand.parent] * cand.pi])
    C_f = np.vstack([task.M, cand.cost[cand.row].astype(np.float64)])
    assert emd(a_m, task.tgt_w, C_m)[0] == pytest.approx(
        emd(full, task.tgt_w, C_f)[0], abs=1e-9)


def test_dedup_preserves_the_total_candidate_mass(task):
    cand = task.candidates("kemp")
    x = task.baseline("season")
    assert cand.mass(x).sum() == pytest.approx(float(x @ cand.p_parent))


# ── 5. Lipschitz and integrality ────────────────────────────────────────────

def test_phi_moves_by_exactly_minus_one_zero_or_plus_one(task):
    """`phi` is 1-Lipschitz in Hamming by dual feasibility and the cost is
    integer-valued, so every one-step neighbour's bracket is -1, 0 or +1. That is what
    turns the derivative into `1 - p_flat - 2 p_down` and the break-even into an exact
    number rather than an estimate."""
    x = task.baseline("season")
    cand = task.candidates("kunif")
    _, _, z = emd(x, task.tgt_w, task.M, duals=True)
    phi_v = (task.M - z[None, :]).min(axis=1)
    phi_w = (cand.cost.astype(np.float64) - z[None, :]).min(axis=1)
    d = phi_w[cand.row] - phi_v[cand.parent]
    assert np.allclose(d, np.rint(d), atol=1e-9)
    assert set(np.rint(d).astype(int).tolist()) <= {-1, 0, 1}


# ── 6. Danskin ──────────────────────────────────────────────────────────────

def test_the_finite_difference_converges_to_the_analytic_derivative(task):
    """`(W(eps) - W(0)) / eps -> dW/dmu|_0`. The whole gate rests on the analytic form
    being the derivative of the thing that actually gets scored."""
    x = task.baseline("season")
    cand = task.candidates("kunif")
    task.score(x)                                 # the leak guard's key
    d = task.descent_gate(x, "kunif")
    w0 = task.score(convolve(x, cand, 0.0))
    # The score is piecewise linear in `mu` -- it is the value of an LP whose right-hand
    # side is affine -- so on the first linear piece the difference quotient is the
    # derivative to solver precision at every step size, not only in the limit.
    for eps in (1e-2, 1e-3, 1e-4):
        slope = (task.score(convolve(x, cand, eps)) - w0) / eps
        assert slope == pytest.approx(d["deriv"], abs=1e-6)
    # And the identity the break-even is derived from.
    assert d["deriv"] == pytest.approx(1.0 - d["p_flat"] - 2.0 * d["p_down"], abs=1e-9)
    assert d["break_even"] == pytest.approx(0.5 * (1.0 - d["p_flat"]), abs=1e-12)


# ── 7. convexity ────────────────────────────────────────────────────────────

def test_the_score_is_convex_in_mu_and_golden_section_finds_the_optimum(task):
    """`x'(mu)` is affine and `W_1(·, nu)` convex, so the score is convex on [0,1]: one
    optimum, and golden section cannot fall into a local one. Both halves are asserted,
    because it is convexity that turns a positive derivative at zero into a proof that
    the optimal `mu` *is* zero."""
    x = task.baseline("season")
    cand = task.candidates("kunif")
    mus = np.linspace(0.0, 1.0, 21)
    W = np.asarray([task.score(convolve(x, cand, m)) for m in mus])
    second = W[2:] - 2 * W[1:-1] + W[:-2]
    assert (second >= -1e-9).all()
    mu, f = golden_mu(lambda m: task.score(convolve(x, cand, m)), tol=1e-6)
    assert f <= W.min() + 1e-6            # the bracket is 1e-6 wide, so this is tight
    assert abs(mu - mus[W.argmin()]) <= 0.05 + 1e-6      # within one grid step


def _wide_panel():
    """A support with many polymorphic sites and a target sitting close to it.

    This is the geometry of the real panel, and the reason a *diffuse* kernel is near
    worst-case there: the mass-weighted distance to the targets is 3.6 substitutions over
    hundreds of varying sites, so a site drawn at random is overwhelmingly one that
    already *agrees* with the target and changing it opens a new mismatch. Here the base
    genotype sits one substitution from the target across fourteen sites, twelve of them
    polymorphic in the support, which reproduces the panel's numbers closely enough to
    assert the conclusion on: `p_down` comes out at 1/15 against the panel's 0.0766 and
    the derivative at +0.867 against +0.739."""
    base = "A" * 14
    tips = [(2000.80, base), (2000.85, base), (2000.90, base)]
    for k in range(12):
        tips.append((2001.0 + 0.001 * k, base[:k] + "C" + base[k + 1:]))
    tgt = base[:12] + "C" + base[13:]
    tips += [(2001.80, tgt), (2001.90, tgt)]
    return TipPanel(tips, min_allele_freq=0.0, lookback=5.0)


def test_a_uniform_kernel_that_points_uphill_has_its_optimum_at_zero():
    """The round's central fact. A convex function on [0,1] with a positive
    right-derivative at zero attains its minimum at zero, so a diffuse kernel does not
    merely lose -- its optimal `mu` is provably zero, at every transition, with no fit, no
    sweep and no selector. There is nothing to run.

    The bar follows from the same identity: `p_down* = (1 - p_flat) / 2`. Uniform sits far
    below it while the *greedy* arm -- point mass on the single best direction per parent
    -- descends at nearly -1, so the problem is entirely one of selection and not of
    availability."""
    t = Task(_wide_panel(), 2001)
    x = t.baseline("season")
    t.score(x)
    d = t.descent_gate(x, "kunif")
    assert d["deriv"] > 0
    assert d["p_down"] < d["break_even"]
    # Availability is not the problem: four fifths of the mass has a downhill direction
    # and the greedy arm takes it. The greedy figure is not -0.8 but -0.6, because the
    # base genotype has no downhill direction at all and a kernel forced to move every
    # parent is charged +1 for it -- which is why the panel's -0.995 sits just short of
    # -1 rather than at it.
    assert d["x_with_a_downhill"] > 0.75
    assert d["greedy"] < -0.5
    assert d["greedy"] == pytest.approx(-d["x_with_a_downhill"]
                                        + (1.0 - d["x_with_a_downhill"]), abs=1e-9)
    cand = t.candidates("kunif")
    mu, f = golden_mu(lambda m: t.score(convolve(x, cand, m)), tol=1e-6)
    assert mu == pytest.approx(0.0, abs=1e-5)
    assert f == pytest.approx(t.score(x), abs=1e-9)


# ── 8. the mu*k bound ───────────────────────────────────────────────────────

@pytest.mark.parametrize("arm", ["kunif", "kemp", "kcons", "kcmom"])
def test_the_downside_is_bounded_by_mu_times_k(task, arm):
    """The coupling that holds `(1-mu)x` in place and ships `mu x` one substitution costs
    at most `mu`, so the score moves by at most that. This is the bound no previous round
    could state, and it is what licenses a declared `mu` instead of a fitted one."""
    x = task.baseline("season")
    cand = task.candidates(arm)
    w0 = task.score(convolve(x, cand, 0.0))
    for mu in (0.05, 0.1, 0.2, 0.4, 1.0):
        assert abs(task.score(convolve(x, cand, mu)) - w0) <= mu + 1e-9


# ── 8b. the float error at mu = 1, which the panel had and the fixture did not ──

def test_a_row_sum_never_exceeds_one_and_mu_one_stays_a_distribution():
    """The regression this suite missed. `pi` is renormalised per parent, so `p_parent`
    is one -- but a bincount of a few hundred renormalised entries overshoots by 1e-15
    for a good fraction of parents, and `convolve` then leaves `x_v (1 - mu p_v)`
    *negative* at `mu = 1`. POT calls a negative source mass an infeasible LP, warns once
    per process, and returns a zero plan whose cost is `0.0` -- an EMD below any
    attainable one, which `golden_mu` evaluates at its `hi` endpoint and takes as the
    minimum. The oracle family would then report `mu = 1` at an EMD of zero on every
    transition.

    Nine candidates on one parent is the smallest instance of the overshoot: nine copies
    of `1/9` sum to strictly more than one in float64. The real panel had 225 parents of
    271 over one at predict-2018."""
    n = 9
    pi = np.full(n, 1.0 / n)
    assert np.bincount(np.zeros(n, int), weights=pi, minlength=1)[0] > 1.0   # the cause
    cand = Candidates(parent=np.zeros(n, int), site=np.zeros(n, int),
                      res=np.arange(n), pi=pi, row=np.arange(n),
                      cost=np.ones((n, 1)), n_parents=1, arm="kunif", tau_k=1.0, m=0)
    assert cand.p_parent.max() <= 1.0
    fc = convolve(np.ones(1), cand, 1.0)
    assert (fc.w >= 0.0).all()
    assert fc.sum() == pytest.approx(1.0)


def test_a_failed_transport_solve_raises_rather_than_scoring_zero(task):
    """`emd` must not hand back the zero plan's cost of `0.0` when the LP fails. That
    value is smaller than any real EMD, so anything minimising over a hyperparameter
    walks straight to whatever configuration broke the solver -- which is exactly how the
    negative mass above would have been selected rather than noticed."""
    a = task.baseline("season").copy()
    a[int(np.argmax(a))] += 1e-9
    a[int(np.argmin(a[a > 0]))] = -1e-9           # a negative source mass, as before
    with pytest.raises(RuntimeError, match="transport solve failed"):
        emd(a, task.tgt_w, task.M)


def test_the_oracle_mu_search_obeys_the_bound_at_its_endpoints(task):
    """`golden_mu` evaluates `lo` and `hi` explicitly, so `mu = 1` is scored on every
    transition the oracle family touches whether or not the optimum is near it. The
    `mu k` bound therefore has to hold *there*, which is the assertion the zero-cost plan
    violated by 5.7 EMD."""
    x = task.baseline("season")
    cand = task.candidates("kemp")
    w0 = task.score(x)
    mu, f = golden_mu(lambda u: task.score(convolve(x, cand, u)), tol=1e-6)
    assert 0.0 <= mu <= 1.0
    assert f >= w0 - 1.0 - 1e-9
    assert f <= w0 + 1e-9                         # the search can always return mu = 0


# ── 9. kemp is leak-free ────────────────────────────────────────────────────

def test_kemp_reads_nothing_dated_after_the_horizon():
    """Bit-identical when a sequence dated after the horizon is removed, perturbed or
    added -- and it *changes* when one sequence's date crosses the horizon. The first half
    is what a one-year window off-by-one would break while leaving every printed number
    looking reasonable.

    The perturbations deliberately keep the alphabet and the set of varying sites fixed:
    they are properties of the whole panel, so changing them would re-index the count
    table for reasons that have nothing to do with the horizon."""
    horizon = 2001.25
    base = TipPanel(TIPS, min_allele_freq=0.0, lookback=5.0)
    ref = empirical_counts(base, horizon, tau_e=5.0)

    late = [t for t in TIPS if t[0] > horizon]
    assert len(late) > 1
    for variant in (
            [t for t in TIPS if t != late[0]],                       # removed
            [(t[0] + 0.01, t[1]) if t == late[0] else t for t in TIPS],   # perturbed
            TIPS + [(2003.85, "GDCGC")],                             # added
    ):
        p = TipPanel(variant, min_allele_freq=0.0, lookback=5.0)
        assert p.alphabet == base.alphabet and p.sites == base.sites
        assert np.array_equal(empirical_counts(p, horizon, tau_e=5.0), ref)

    # Now move one sequence from after the horizon to before it. The table must move.
    crossed = [(2001.05, t[1]) if t == late[0] else t for t in TIPS]
    p = TipPanel(crossed, min_allele_freq=0.0, lookback=5.0)
    assert not np.array_equal(empirical_counts(p, horizon, tau_e=5.0), ref)


def test_kemp_counts_are_directed_by_first_observation(panel):
    """The older residue is the old one. ACAAA (2001.10) precedes ACCAA (2001.22) at site
    2, so the count must sit on A -> C and not on C -> A."""
    cnt = empirical_counts(panel, 2001.25, tau_e=5.0)
    q = panel.sites.index(2)
    a, c = panel.code["A"], panel.code["C"]
    assert cnt[q, a, c] > 0
    assert cnt[q, c, a] == 0


# ── 10. no fitted parameter sees a candidate ────────────────────────────────

def test_the_parents_fit_is_byte_identical_with_the_kernel_on_and_off(panel):
    """The two rounds that widened the support a model was *fitted* on went from +4.3% to
    -9.2%. The kernel is applied after the forecast precisely so that cannot happen here,
    and this is the assertion that it did not: same design, same theta, same floor."""
    order = [2001, 2002, 2003]
    tasks = {s: Task(panel, s) for s in order}
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,),
                     mus=(0.0, 0.4), kernels=(None, "kunif"))
    off = next(c for c in grid if c["kernel"] is None and c["mu"] == 0.0)
    on = next(c for c in grid if c["kernel"] == "kunif" and c["mu"] == 0.4)
    t, tr = tasks[2003], train_set(tasks, order, 2, 0)
    Z0, b0, n0, bl0 = design_for(t, off, "mlr+mom")
    Z1, b1, n1, bl1 = design_for(t, on, "mlr+mom")
    assert np.array_equal(Z0.toarray(), Z1.toarray())
    assert np.array_equal(b0, b1) and (n0, bl0) == (n1, bl1)
    th0 = fit_family(t, off, "mlr+mom", tr, 0.01, Z0.shape[1], n0, bl0)
    th1 = fit_family(t, on, "mlr+mom", tr, 0.01, Z1.shape[1], n1, bl1)
    assert np.array_equal(th0, th1)
    # And the parent's own prediction over the support is untouched: the kernel family's
    # support part is the parent scaled by (1 - mu), nothing re-fitted.
    parent = run_family(t, off, "mlr+mom", tr, 10, 0.01)
    fc = run_family(t, on, "mlr+mom+K", tr, 10, 0.01)
    cand = t.candidates("kunif")
    assert np.allclose(fc.w, parent * (1.0 - 0.4 * cand.p_parent))


# ── 11. the observed-residue restriction ────────────────────────────────────

def test_kobs_proposes_no_residue_the_support_does_not_carry(panel):
    p = _marg_panel()
    t = Task(p, 2001)
    A = p.codes[t.sup]
    for arm, novel_allowed in (("kobs", False), ("kplm", True)):
        cand = t.candidates(arm, 1.0, 0)
        seen = [set(np.unique(A[:, q]).tolist()) for q in range(A.shape[1])]
        novel = [int(b) not in seen[int(q)] for q, b in zip(cand.site, cand.res)]
        assert any(novel) == novel_allowed, arm


def test_the_observed_floor_is_the_leak_free_ceiling(task):
    """Both floors, and the order between them. `observed` is what a kernel proposing only
    residues the support carries could reach; `all` opens the novel-residue channel, and
    the gap between them is the entire prize that channel can deliver -- 0.088 EMD units
    on the backbone panel, against a 3.632 unextended floor."""
    obs = task.floor_ext("observed")
    al = task.floor_ext("all")
    assert al <= obs <= task.floor + 1e-12
    assert obs < task.floor, "the fixture must have somewhere to move mass to"
    # The `observed` ceiling is attained by the untruncated observed candidate set: no
    # forecast on it can score below, and the floor is exactly that minimum.
    cand = task.candidates("kunif")
    stacked = np.vstack([task.M, cand.cost.astype(np.float64)])
    assert obs == pytest.approx(float(task.tgt_w @ stacked.min(axis=0)), abs=1e-9)
    assert one_step_floor(task.panel, task.sup, task.M, task.tgt_g, task.tgt_w,
                          "observed") == pytest.approx(obs, abs=1e-12)


# ── 12. both floors are present, and never reduced across ───────────────────

def test_a_reduction_is_never_computed_across_two_floors(panel):
    from antiGen.season_momentum import captured, report, rolling
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,),
                     mus=(0.0, 0.2), kernels=("kunif",))
    fams = ("season", "mlr+mom", "season+K")
    rows = rolling(panel, drop_targets=(), min_seqs=1, min_train=1, val_folds=1,
                   grid=grid, maxiter=10, families=fams, verbose=False)
    assert rows
    for r in rows:
        assert "floor" in r and r.get("floor_ext")
        assert all(v <= r["floor"] + 1e-12 for v in r["floor_ext"].values())
        # An unextended family is measured above the unextended floor, a kernel family
        # above the extended one, and asking for the wrong one is not possible: the
        # family decides.
        assert captured(r, "mlr+mom") != pytest.approx(captured(r, "season+K")) or \
            r["emd_mlr+mom"] != r["emd_season+K"]
        bare = dict(r)
        bare.pop("floor_ext")
        with pytest.raises(KeyError):
            captured(bare, "season+K")
    s = report(rows, fams)
    assert "floor_ext" in s and s["floor_ext"]["observed"] <= s["floor"] + 1e-12


# ── 13. degenerate cases ────────────────────────────────────────────────────

def test_a_monomorphic_site_contributes_no_observed_candidate(task):
    """Site 3 carries only A across the 2001 support, so an observed-residue arm has no
    substitution to propose there. An all-residue arm does -- that is exactly where a
    novel residue is worth something -- and the 2002 target needs a G there."""
    A = task.panel.codes[task.sup]
    j = task.panel.sites.index(3)
    assert len(np.unique(A[:, j])) == 1
    obs_site, _ = _columns(task.panel, A, "observed")
    assert j not in obs_site.tolist()
    all_site, all_res = _columns(task.panel, A, "all")
    assert j in all_site.tolist()
    assert task.panel.code["G"] in all_res[all_site == j].tolist()


def test_a_transition_with_no_downhill_direction_returns_mu_zero(panel):
    """`p_down = 0` must come back as `mu = 0`, not as a raise and not as a nan that the
    selector then sorts to the front."""
    one = TipPanel([(2000.9, "AAAAA"), (2001.8, "ACAAA"), (2001.9, "ACAAA")],
                   min_allele_freq=0.0, lookback=5.0)
    t = Task(one, 2001)
    x = t.baseline("season")
    t.score(x)
    d = t.descent_gate(x, "kunif")
    assert d["p_down"] == 0.0 and d["n_cand"] == 0
    mu, _ = golden_mu(lambda m: t.score(convolve(x, t.candidates("kunif"), m)))
    assert mu == pytest.approx(0.0, abs=1e-3)


def test_the_gate_reads_an_operating_point_and_a_verdict():
    stats = {("kunif", 1.0, 0): 0.077, ("kemp", 1.0, 0): 0.30,
             ("kemp", 0.25, 5): 0.51, ("kplm", 0.5, 3): 0.60}
    v = gate(stats)
    assert v["kunif"]["passes"] is False and v["kunif"]["p_down"] == 0.077
    assert v["kemp"]["passes"] is True
    assert (v["kemp"]["tau_k"], v["kemp"]["m"]) == (0.25, 5)     # the arm's best
    assert v["kplm"]["leaky"] is True and v["kunif"]["leaky"] is False
    assert v["kemp"]["break_even"] == pytest.approx(BREAK_EVEN)


# ── 14. the gate is leak-free by construction ───────────────────────────────

def test_the_descent_gate_refuses_an_unscored_transition(panel):
    """`phi` reads the optimal target potential, so `p_down` knows where the season
    actually went. Measuring it before anything has scored the transition would choose an
    arm on the test set, and the refusal is what makes that impossible rather than
    discouraged."""
    t = Task(panel, 2001)
    x = t.baseline("season")
    assert t.scored is False
    with pytest.raises(RuntimeError, match="has not been scored"):
        t.descent_gate(x, "kunif")
    t.score(x)
    assert t.scored is True
    assert 0.0 <= t.descent_gate(x, "kunif")["p_down"] <= 1.0


# ── the surrounding machinery ───────────────────────────────────────────────

def test_averaging_concatenates_blocks_rather_than_merging_them(task):
    """`top_k` averages the best few candidates' predictions, and two combos' dedup
    indexes name different rows. Concatenating with masses divided by K is exact --
    a distribution over a disjoint union is what that means -- and merging would be
    arithmetic on incomparable indexes."""
    x = task.baseline("season")
    a = convolve(x, task.candidates("kunif"), 0.2)
    b = convolve(x, task.candidates("kemp", 0.5, 2), 0.4)
    m = average([a, b])
    assert len(m.blocks) == 2
    assert m.sum() == pytest.approx(1.0)
    assert np.allclose(m.w, 0.5 * (a.w + b.w))
    # And the score of the average is at most the average of the scores: W_1 is convex in
    # the forecast, which is the same fact `mlr+mom+mix` is built on.
    assert task.score(m) <= 0.5 * (task.score(a) + task.score(b)) + 1e-9
    # A plain array still averages as an array.
    assert isinstance(average([x, x]), np.ndarray)


def test_the_kernel_is_invisible_to_families_that_do_not_convolve(panel):
    from antiGen.season_momentum import collapse
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,),
                     mus=(0.0, 0.1, 0.4), kernels=(None, "kunif"))
    assert len(collapse(grid, "mlr+mom")) == 1
    assert len(collapse(grid, "season+K")) == 2          # the arm, and nothing else
    assert len(collapse(grid, "mlr+mom+K")) == 6         # mu x kernel
    assert len(collapse(grid, "mlr+mom+K@mu*")) == 2     # mu is chosen by golden section


def test_season_plus_k_ignores_the_grids_mu_and_uses_the_declared_one(panel):
    """`season+K`'s whole claim is that it owes nothing to the selector. Its `mu` is
    declared in the module, so a grid that says otherwise must not reach it."""
    t = Task(panel, 2003)
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,),
                     mus=(0.9,), kernels=("kunif",))
    fc = run_family(t, grid[0], "season+K", [], 10, 0.01)
    x = t.baseline("season")
    cand = t.candidates("kunif")
    assert np.allclose(fc.w, x * (1.0 - MU_FIXED * cand.p_parent))


def test_the_oracle_arm_never_scores_worse_than_its_parent(panel):
    """Golden section over a convex score with `mu = 0` in the bracket: the oracle arm is
    the parent or better by construction. That is what makes it the diagnostic separating
    'the kernel points the wrong way' from 'mu was chosen badly' -- if the oracle column
    is also flat, the direction is the problem."""
    order = [2001, 2002, 2003]
    tasks = {s: Task(panel, s) for s in order}
    grid = make_grid((1,), (np.inf,), (1.0,), (0,), (0.0,),
                     mus=(0.0,), kernels=("kunif",))
    t, tr = tasks[2003], train_set(tasks, order, 2, 0)
    parent = run_family(t, grid[0], "mlr+mom", tr, 10, 0.01)
    oracle = run_family(t, grid[0], "mlr+mom+K@mu*", tr, 10, 0.01)
    assert t.score(oracle) <= t.score(parent) + 1e-6


def test_the_kernel_refuses_a_non_hamming_metric():
    """The recursion counts agreeing residues, which is a statement about Hamming. Under
    the embedding metric a candidate's distance would need its own ESM vector, and no
    candidate sequence is ever materialised."""
    rng = np.random.default_rng(0)
    seqs = sorted({g for _, g in TIPS})
    emb = {g: rng.standard_normal(3) for g in seqs}
    p = TipPanel(TIPS, min_allele_freq=0.0, lookback=5.0, embedding=emb,
                 metric="euclidean")
    t = Task(p, 2001)
    with pytest.raises(ValueError, match="Hamming-only"):
        t.candidates("kunif")


def test_every_arm_is_declared_leaky_or_not_and_the_families_are_named():
    assert set(ARMS) == {"kunif", "kemp", "kcons", "kcmom",
                         "kobs", "kplm", "kemp*kplm"}
    assert [a for a in ARMS if not ARMS[a]["leaky"]] == ["kunif", "kemp",
                                                         "kcons", "kcmom"]
    assert [a for a in ARMS if ARMS[a]["needs_plm"]] == ["kobs", "kplm", "kemp*kplm"]
    # The consensus arms condition on the parent beyond excluding its own residue --
    # their site propensity is 1 - f_hat(v_q) -- which is what section 2 makes the game.
    assert [a for a in ARMS if ARMS[a]["conditional"]] == ["kcons", "kcmom",
                                                           "kobs", "kplm", "kemp*kplm"]
    assert K_FAMILIES == ("season+K", "mlr+mom+K", "mlr+mom+K@mu*")


# ── the consensus arms ──────────────────────────────────────────────────────

def test_site_allele_freq_reads_nothing_dated_after_the_horizon():
    """The same guarantee `kemp` gets, by the same construction and the same test.

    `kcons`/`kcmom` are the leak-free population arms; a window off-by-one here would
    make the table read the season it is forecasting into, which is the one failure that
    would look like a result."""
    horizon = 2001.25
    base = TipPanel(TIPS, min_allele_freq=0.0, lookback=5.0)
    ref = site_allele_freq(base, horizon, 1.0)

    late = [t for t in TIPS if t[0] > horizon]
    assert len(late) > 1
    for variant in (
            [t for t in TIPS if t != late[0]],                            # removed
            [(t[0] + 0.01, t[1]) if t == late[0] else t for t in TIPS],   # perturbed
            TIPS + [(2003.85, "GDCGC")],                                  # added
    ):
        p = TipPanel(variant, min_allele_freq=0.0, lookback=5.0)
        assert p.alphabet == base.alphabet and p.sites == base.sites
        assert np.allclose(site_allele_freq(p, horizon, 1.0), ref)

    crossed = [(2001.05, t[1]) if t == late[0] else t for t in TIPS]
    p = TipPanel(crossed, min_allele_freq=0.0, lookback=5.0)
    assert not np.allclose(site_allele_freq(p, horizon, 1.0), ref)


def test_site_allele_freq_is_a_distribution_per_site(panel):
    for tau in (0.5, 1.0, 5.0, np.inf):
        f = site_allele_freq(panel, 2001.25, tau)
        assert f.shape == (len(panel.sites), len(panel.alphabet))
        assert np.all(f >= 0.0)
        assert np.allclose(f.sum(axis=1), 1.0)


def test_site_allele_freq_matches_a_hand_count_at_infinite_tau(panel):
    """tau = inf is a plain unweighted share over every sequence up to the horizon, so
    it can be checked by counting. At 2001.25 the panel holds AAAAA x2, CAAAA, GAAAA,
    ACAAA x2, ACCAA -- seven sequences, four carrying A at site 0."""
    f = site_allele_freq(panel, 2001.25, np.inf)
    seqs = [g for d, g in TIPS if d <= 2001.25]
    assert len(seqs) == 7
    for j, p in enumerate(panel.sites):
        for c in set(s[p] for s in seqs):
            want = sum(1 for s in seqs if s[p] == c) / len(seqs)
            assert f[j, panel.code[c]] == pytest.approx(want)


def test_site_allele_freq_is_not_corrupted_by_the_flat_index():
    """The vectorised bincount must agree with a plain per-site loop on a panel with
    ENOUGH SITES TO OVERFLOW a uint8 index.

    `panel.codes` is uint8. The first version of `site_allele_freq` built its site
    offsets as `np.arange(J, dtype=C.dtype) * nA`, which under NEP 50 stays uint8 and
    wraps every 256 // nA sites, collapsing most sites onto a shared bin. The table came
    back near-uniform -- a perfectly plausible distribution, summing to one per site,
    raising nothing anywhere downstream. It was only caught by reading a real H3N2 table
    and noticing that position 0 of HA, which is M in 1681 of 1683 sequences, was being
    reported at 0.037.

    The five-site fixture cannot express this: J * nA has to exceed 255 before the wrap
    happens at all, which is why this test builds its own panel and why the existing
    hand-count test passed throughout."""
    rng = np.random.default_rng(0)
    # The offsets are j * nA, so the uint8 wrap needs j * nA > 255, i.e. more than
    # 255 // nA sites. With a four-letter alphabet that is 64, so 80 sites clears it
    # with margin and the whole table is corrupted rather than only its tail.
    L, res = 80, "ACGT"
    seqs = []
    for i in range(60):
        seqs.append("".join(res[int(c)] for c in rng.integers(0, 4, L)))
    tips = [(2000.0 + 0.01 * i, sq) for i, sq in enumerate(seqs)]
    p = TipPanel(tips, min_allele_freq=0.0, lookback=50.0)
    assert len(p.sites) * len(p.alphabet) > 255, "fixture too small to trip the wrap"
    assert p.codes.dtype == np.uint8, "the bug needs the narrow dtype to exist"

    horizon, tau = 2000.40, 0.25
    got = site_allele_freq(p, horizon, tau)

    # The reference: one loop, no flattening, no shared index.
    idx = np.where(p.dates <= horizon)[0]
    age = horizon - p.dates[idx]
    wt = np.exp(-(age - age.min()) / tau)
    want = np.zeros((len(p.sites), len(p.alphabet)))
    for k, w in zip(idx, wt):
        code = p.codes[p.tip_gid[k]]
        for j in range(len(p.sites)):
            want[j, code[j]] += w
    want /= want.sum(axis=1, keepdims=True)
    assert np.allclose(got, want)

    # And prove the fixture actually trips the wrap, rather than passing because the
    # bug is unreachable here: recompute the flat index the buggy way and require it to
    # disagree. Without this the test would still pass if someone shrank the fixture.
    nz = np.flatnonzero(np.bincount(p.tip_gid[idx], weights=wt,
                                    minlength=len(p.genotypes)) > 0)
    C = p.codes[nz]
    J, nA = len(p.sites), len(p.alphabet)
    bad = (C + np.arange(J, dtype=C.dtype)[None, :] * nA).ravel()
    good = (C.astype(np.int64)
            + np.arange(J, dtype=np.int64)[None, :] * nA).ravel()
    assert not np.array_equal(bad, good), "fixture does not exercise the uint8 wrap"


def test_kcons_proposes_toward_the_majority_residue(panel):
    """The arm's whole claim, at its simplest: at a site where the population is mostly
    C, a parent carrying A proposes C more than it proposes any rarer residue."""
    horizon, sup = 2001.25, None
    task = Task(panel, 2001)
    sup = task.sup
    cand = task.candidates("kcons", tau_k=1.0, m=0)
    f = site_allele_freq(panel, task.horizon, CONS_FAST)
    # For each parent and site, the proposal must rank residues exactly as f does.
    seen = 0
    for v in range(len(sup)):
        rows = np.flatnonzero(cand.parent == v)
        for q in np.unique(cand.site[rows]):
            at = rows[cand.site[rows] == q]
            if len(at) < 2:
                continue
            order_pi = np.argsort(-cand.pi[at], kind="stable")
            order_f = np.argsort(-f[q, cand.res[at]], kind="stable")
            assert np.array_equal(cand.res[at][order_pi], cand.res[at][order_f])
            seen += 1
    assert seen > 0


def test_kcons_spends_less_at_a_site_where_the_parent_already_carries_the_top_allele():
    """The conditioning on `v`, isolated. For two parents differing at site 0 **and
    nowhere else**, the one already carrying the site's dominant residue must place
    strictly less of its proposal at that site than the one carrying a minority residue.

    The pairing is the whole of the test's validity: `pi` is normalised per parent over
    every column, so the share a parent spends at one site depends on what it has to
    propose at the others, and comparing two parents that differ at more than one site
    compares those denominators rather than the conditioning. An earlier version of this
    test did exactly that and failed in the direction that looks like a broken arm.

    This is the factor a non-conditional population arm like `kemp` does not have, and
    it is why the arm is declared `conditional`."""
    tips = ([(2000.0 + 0.01 * i, "AAA") for i in range(20)]
            + [(2000.5 + 0.01 * i, "CAA") for i in range(3)]
            + [(2001.0, "AAC"), (2001.05, "CAC")]
            + [(2001.80, "AAA"), (2001.90, "CAA"), (2002.00, "AAC")])
    p = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    t = Task(p, 2001)
    assert len(t.tgt_g) > 0 and len(t.sup) > 1
    cand = t.candidates("kcons", tau_k=1.0, m=0)
    f = site_allele_freq(p, t.horizon, CONS_FAST)
    j0 = p.sites.index(0)
    top = int(np.argmax(f[j0]))
    seqs = [p.genotypes[g] for g in t.sup]

    def share_at(v, q):
        m = (cand.parent == v) & (cand.site == q)
        return float(cand.pi[m].sum())

    pairs = 0
    for a in range(len(seqs)):
        for b in range(len(seqs)):
            differ = [k for k in range(len(seqs[a])) if seqs[a][k] != seqs[b][k]]
            if differ != [0]:
                continue
            if p.code[seqs[a][0]] != top or p.code[seqs[b][0]] == top:
                continue
            assert share_at(a, j0) < share_at(b, j0)
            pairs += 1
    assert pairs > 0


def test_kcmom_follows_the_trend_where_kcons_follows_the_level():
    """A residue that is still a minority overall but rising must gain mass under
    `kcmom` relative to `kcons`. Built so the two arms disagree in a known direction:
    C at site 0 is rare across the whole record and common in the last year.

    If this passes with delta = 0 the arm is not doing what its name says, so the
    control is asserted to be unchanged in the same breath."""
    old = [(2000.0 + 0.02 * i, "AAA") for i in range(40)]
    new = [(2002.6 + 0.01 * i, "CAA") for i in range(12)] + \
          [(2002.6 + 0.01 * i, "AAA") for i in range(12)]
    p = TipPanel(old + new, min_allele_freq=0.0, lookback=5.0)
    horizon = 2002.75
    q = p.sites.index(0)
    c = p.code["C"]
    lvl = consensus_freq(p, horizon, "kcons", None)
    mom = consensus_freq(p, horizon, "kcmom", None)
    assert np.allclose(lvl, site_allele_freq(p, horizon, CONS_FAST))
    assert mom[q, c] > lvl[q, c]
    assert np.allclose(mom.sum(axis=1), 1.0)


def test_the_momentum_ratio_is_floored_so_a_new_allele_cannot_take_a_whole_site():
    """An allele absent from the slow window and present in the fast one has an
    infinite raw ratio. The floor is what keeps `kcmom` a distribution rather than a
    point mass on whatever appeared most recently, and the table must stay finite."""
    tips = ([(2000.0 + 0.02 * i, "AAA") for i in range(40)]
            + [(2002.70, "CAA"), (2002.72, "AAA")])
    p = TipPanel(tips, min_allele_freq=0.005, lookback=5.0)
    mom = consensus_freq(p, 2002.75, "kcmom", None)
    assert np.all(np.isfinite(mom))
    assert np.allclose(mom.sum(axis=1), 1.0)
    q, c = p.sites.index(0), p.code["C"]
    assert mom[q, c] < 1.0


def test_the_consensus_arms_declare_their_timescales_rather_than_selecting_them(panel):
    """`tau_e` is the fast timescale and the slow one is a declared multiple of it, so
    the pair is settleable by the gate and never reaches the outer selector's grid."""
    assert CONS_DELTA == {"kcons": 0.0, "kcmom": 1.0}
    assert CONS_FAST == 1.0 and CONS_SLOW_MULT == 3.0
    a = consensus_freq(panel, 2001.25, "kcons", None)
    b = consensus_freq(panel, 2001.25, "kcons", CONS_FAST)
    assert np.allclose(a, b)
    assert not np.allclose(consensus_freq(panel, 2001.25, "kcons", 0.25), a)


def test_the_panels_tau_e_reaches_a_scoring_run(panel):
    """`--tau-e` must change the kernel a *scored* forecast is built from.

    `run_family` calls `task.candidates(arm, tau_k, m_k)` and threads no `tau_e`, so
    before this was wired the gate could sweep a timescale that no scoring run could
    ever reach: the gate would settle an operating point, and the scorer would silently
    build a different kernel at each arm's built-in default. Nothing raised, and the
    scored file did not record the setting either, so the two would have been
    indistinguishable after the fact.

    `None` stays meaningful -- it means the arm's own declared default -- which is why
    the not-given marker is a sentinel and not `None`."""
    ref = panel.__class__(TIPS, min_allele_freq=0.0, lookback=5.0)
    t_def = Task(ref, 2001)
    base = t_def.candidates("kcmom", tau_k=1.0, m=0)

    short = panel.__class__(TIPS, min_allele_freq=0.0, lookback=5.0)
    short.kernel_tau_e = 0.05
    t_short = Task(short, 2001)
    got = t_short.candidates("kcmom", tau_k=1.0, m=0)

    assert len(base) == len(got)
    assert not np.allclose(base.pi, got.pi), "--tau-e did not reach the kernel"

    # An explicit argument still wins over the panel's, and an explicit None still
    # means the arm's default rather than "unset".
    same = t_short.candidates("kcmom", tau_k=1.0, m=0, tau_e=None)
    assert np.allclose(same.pi, base.pi)


# ── choosing the operating point causally (`causal_operating_point`) ────────
#
# `p_down` is an oracle quantity: `phi` is the c-transform of the optimal TARGET
# potential, so a gate row knows where its season actually went. The round that settled
# `tau_e 0.25, tau_k 0.02, m 0` pooled the gate over transitions 2017->2018 .. 2022->2023
# and applied the one point at every origin -- and five of those six transitions are also
# scored, so the operating point behind the row reported at origin 2018 was chosen partly
# from origin 2018's own target. These tests pin the rule that removes it.

def _rows(arm="kcmom", **per_cfg):
    """Gate rows in the shape `kernel_gate` writes them.

    `per_cfg` maps a config name to `{season: deriv}`; config names are decoded as
    `te<fast>_ts<slow>_tk<tau_k>_m<m>` so a test can say what it means in the call. The
    slow window may be `n` for None, which is what an arm that reads no slow table
    records."""
    out = []
    for name, by_season in per_cfg.items():
        te, ts, tk, m = name.split("_")
        slow = None if ts[2:] == "n" else float(ts[2:])
        for s, dv in by_season.items():
            out.append(dict(season=int(s), next=int(s) + 1, arm=arm,
                            tau_e=float(te[2:]), tau_e_slow=slow,
                            tau_k=float(tk[2:]), m=int(m[1:]),
                            deriv=float(dv), p_down=0.5, p_flat=0.1))
    return out


def test_a_gate_row_from_the_future_is_never_readable():
    """The whole correction in one assertion.

    `te1.0_tk0.1_m0` is far the best config -- but only because of transition
    2022->2023, which closes at 2023.25, after origin 2021's horizon of 2021.25. At
    origin 2021 the choice must be made from 2017..2020 alone, where the other config
    wins."""
    rows = _rows(**{"te1.0_ts3.0_tk0.1_m0": {2017: -0.1, 2018: -0.1, 2019: -0.1, 2020: -0.1,
                                              2021: -0.1, 2022: -9.9},
                    "te0.25_ts0.75_tk0.02_m0": {2017: -0.5, 2018: -0.5, 2019: -0.5,
                                                2020: -0.5, 2021: -0.5, 2022: -0.5}})
    te, ts, tk, m, info = causal_operating_point(rows, "kcmom", 2021)
    assert (te, ts, tk, m) == (0.25, 0.75, 0.02, 0)
    assert info["gated_on"] == [2017, 2018, 2019, 2020]
    # And with everything readable, the future-fed config does win -- so the test is
    # about the filter and not about the arithmetic.
    assert causal_operating_point(rows, "kcmom", 2026)[:4] == (1.0, 3.0, 0.1, 0)


def test_the_previous_transition_is_readable_and_this_one_is_not():
    """The boundary, which is where an off-by-one would live. Transition `j` reads the
    targets of season `j + 1`, closing at `(j + 1) + 0.25`; origin `i` forecasts from
    `i + 0.25`. So `j = i - 1` closes exactly at the horizon and is in; `j = i` is out."""
    rows = _rows(**{"te0.25_ts0.75_tk0.02_m0": {2019: -0.5, 2020: -0.5, 2021: -0.5}})
    assert causal_operating_point(rows, "kcmom", 2021)[4]["gated_on"] == [2019, 2020]
    assert causal_operating_point(rows, "kcmom", 2020)[4]["gated_on"] == [2019]


def test_a_config_measured_on_fewer_transitions_is_dropped_not_averaged():
    """A mean over four transitions and a mean over five are not comparable, and taking
    the argmin across both lets a config win by having been measured on the easy ones."""
    rows = _rows(**{"te0.5_ts1.5_tk0.1_m0": {2019: -9.9},            # one easy row only
                    "te0.25_ts0.75_tk0.02_m0": {2018: -0.5, 2019: -0.5}})
    te, ts, tk, m, info = causal_operating_point(rows, "kcmom", 2020)
    assert (te, ts, tk, m) == (0.25, 0.75, 0.02, 0)
    assert info["n_configs"] == 1
    # Asked not to require completeness, the ragged config wins -- so the guard is what
    # is being tested, not the fixture.
    assert causal_operating_point(rows, "kcmom", 2020,
                                  require_complete=False)[:4] == (0.5, 1.5, 0.1, 0)


def test_an_origin_with_nothing_readable_raises_rather_than_defaulting():
    """Falling back to the declared flag would leave half a panel causal and half not,
    with nothing in the output saying which half."""
    rows = _rows(**{"te0.25_ts0.75_tk0.02_m0": {2020: -0.5, 2021: -0.5}})
    with pytest.raises(ValueError, match="no gate row"):
        causal_operating_point(rows, "kcmom", 2020)
    with pytest.raises(ValueError, match="no gate row"):
        causal_operating_point(rows, "kcons", 2025)          # arm absent entirely


def test_each_arm_gets_its_own_operating_point():
    rows = (_rows("kcmom", **{"te0.25_ts0.75_tk0.02_m0": {2019: -0.9},
                              "te1.0_ts3.0_tk0.1_m0": {2019: -0.1}})
            + _rows("kcons", **{"te0.25_tsn_tk0.02_m0": {2019: -0.1},
                                "te1.0_tsn_tk0.1_m0": {2019: -0.9}}))
    assert causal_operating_point(rows, "kcmom", 2020)[:4] == (0.25, 0.75, 0.02, 0)
    # kcons reads no slow table, so its rows carry None and it is never asked to choose
    # between two settings of a field it cannot see.
    assert causal_operating_point(rows, "kcons", 2020)[:4] == (1.0, None, 0.1, 0)


def test_the_resolved_point_overrides_the_declared_flags(panel):
    """`Task.kernel_point` is the one place the override happens, and it happens before
    the cache key -- so the key still names the kernel that was actually built, and a
    task asked for two different declared points returns the resolved one both times."""
    task = Task(panel, 2001)
    declared = task.candidates("kcmom", tau_k=1.0, m=0, tau_e=1.0)
    task._cache.clear()
    task.kernel_point = {"kcmom": (0.25, 0.75, 0.02, 0)}
    a = task.candidates("kcmom", tau_k=1.0, m=0, tau_e=1.0)
    b = task.candidates("kcmom", tau_k=0.5, m=3, tau_e=2.0)     # different flags
    assert a is b, "the override must collapse onto one cache key"
    resolved = [k for k in task._cache if k[0] == "cand"]
    assert resolved == [("cand", "kcmom", 0.02, 0, 0.25, 0.75)]
    assert not np.allclose(np.sort(a.pi), np.sort(declared.pi)), (
        "the fixture's two operating points give the same proposal; it proves nothing")
    # An arm the table did not resolve is untouched.
    task.candidates("kcons", tau_k=0.5, m=3, tau_e=2.0)
    assert ("cand", "kcons", 0.5, 3, 2.0, None) in task._cache


def test_the_slow_window_is_free_and_must_exceed_the_fast_one(panel):
    """The asymmetry this replaces: the genotype momentum column selects `tau_f` and
    `tau_s` independently, so the site-allele momentum -- the same construction at allele
    resolution -- may not have its slow window pinned to a multiple of its fast one.

    Three things are pinned here: the slow window changes the table, the 3x fallback is
    still what `None` means, and a slow window at or below the fast one raises rather
    than producing a momentum term that points backwards."""
    a = consensus_freq(panel, 2001.25, "kcmom", 0.5, 1.5)
    b = consensus_freq(panel, 2001.25, "kcmom", 0.5, 4.0)
    assert not np.allclose(a, b), "the slow window does not reach the table"
    # `None` is still 3x the fast window, so every pre-2026-09-28 run reproduces.
    assert np.allclose(consensus_freq(panel, 2001.25, "kcmom", 0.5, None),
                       consensus_freq(panel, 2001.25, "kcmom", 0.5, 0.5 * CONS_SLOW_MULT))
    for bad in (0.5, 0.25):
        with pytest.raises(ValueError, match="longer than the fast one"):
            consensus_freq(panel, 2001.25, "kcmom", 0.5, bad)


def test_only_the_momentum_arm_reads_a_slow_window(panel):
    """`kcons` is the delta = 0 control, so the slow table is never built for it. If the
    gate swept a slow window for it every configuration would be identical and the argmin
    would be decided by iteration order -- which is what `reads_slow` exists to stop."""
    assert reads_slow("kcmom") and not reads_slow("kcons")
    assert not reads_slow("kemp") and not reads_slow("kunif")
    for slow in (1.5, 4.0, None):
        assert np.allclose(consensus_freq(panel, 2001.25, "kcons", 0.5, slow),
                           consensus_freq(panel, 2001.25, "kcons", 0.5, 99.0))


# ── the candidate cap is relative to the leak-free block (2026-10-02) ───────

def test_a_leak_free_arm_is_never_refused_for_its_own_dense_block(task, monkeypatch):
    """The absolute 500k cap refused `kcmom` and `kcons` at origins 2020 and 2021 of
    `panel_K30_v1` -- 530,528 and 520,654 rows -- for being 6% over a constant sized
    against the backbone's 379-707-genotype support. An `observed` arm at `m = 0` asks
    for exactly its own dense block and must never be refused it, at any panel size.

    The floor is monkeypatched to 1 so the fixture's tiny support exercises the relative
    branch rather than the floor."""
    monkeypatch.setattr("antiGen.mutation_kernel.MIN_CANDIDATE_CAP", 1)
    A = task.panel.codes[task.sup]
    col_site, _ = _columns(task.panel, A, "observed")
    dense = len(task.sup) * len(col_site)
    for arm in ("kunif", "kemp", "kcons", "kcmom"):
        assert candidate_cap(task.panel, A, arm, col_site) == dense
        cand = task.candidates(arm, 1.0, 0)          # tau_k = 1: as dense as it gets
        assert 0 < len(cand.parent) <= dense


def test_an_all_residue_arm_at_m_zero_still_trips_the_cap(monkeypatch):
    """What the guard exists for. `kplm` opens every varying site to all twenty residues,
    so its untruncated block is a large multiple of the leak-free one -- ~13x at origin
    2020 of `panel_K30_v1`, 7.1M rows against 530k -- and the LP is no longer a transport
    problem. Truncating with `m` is the caller's job, and the arm says so."""
    monkeypatch.setattr("antiGen.mutation_kernel.MIN_CANDIDATE_CAP", 1)
    p = _marg_panel()
    t = Task(p, 2001)
    A = p.codes[t.sup]
    obs = len(_columns(p, A, "observed")[0])
    assert len(_columns(p, A, "all")[0]) > obs, "the fixture must have a novel channel"
    assert candidate_cap(p, A, "kplm") == len(t.sup) * obs
    with pytest.raises(ValueError, match="the cap is"):
        t.candidates("kplm", 1.0, 0)
    # ...and the same arm truncated is fine, which is what the message tells the caller.
    assert len(t.candidates("kplm", 1.0, 2).parent) <= 2 * len(t.sup)


def test_the_cap_floor_reproduces_the_old_absolute_behaviour(task):
    """Below the backbone's size nothing changes: the floor is the constant the absolute
    cap was, so no panel smaller than the one that forced this change sees a different
    verdict from the one it would have seen before 2026-10-02."""
    A = task.panel.codes[task.sup]
    assert MIN_CANDIDATE_CAP == 500_000
    for arm in ARMS:
        assert candidate_cap(task.panel, A, arm) == MIN_CANDIDATE_CAP
