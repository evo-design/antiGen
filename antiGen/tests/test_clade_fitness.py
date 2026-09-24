"""The clade fitness model: membership, the exact eq.-23 objective, and its gradient."""
import numpy as np
import pytest

from antiGen.clade_fitness import (CladeData, build_design, exp_loss_grad, fit_exp,
                                   predict_exp, ridge_dual)

# A four-node tree.  0 is the root; 1 and 3 hang off it; 2 hangs off 1.
#   0 AAAAAA
#   |- 1 ACAAAA
#   |   '- 2 ACAAAG
#   '- 3 AAAAAT
NODES = [(0, "", "AAAAAA"), (1, "0", "ACAAAA"), (2, "1", "ACAAAG"), (3, "0", "AAAAAT")]
# season -> {node: multiplicity}.  m(2001) = 60, m(2002) = 100.
STRAINS = {2001: {1: 10, 2: 20, 3: 30}, 2002: {1: 40, 2: 50, 3: 10}}
M_T = {2001: 60, 2002: 100, 2003: 80}


def _write(tmp_path):
    (tmp_path / "clade_nodes_seasons.csv").write_text(
        "node_id,parent_node_id,seq\n"
        + "".join(f"{n},{p if p else 'NA'},{s}\n" for n, p, s in NODES))
    (tmp_path / "clade_strains_seasons.csv").write_text(
        "season,node_id,m_i\n"
        + "".join(f"{t},{n},{m}\n" for t, d in STRAINS.items() for n, m in d.items()))
    (tmp_path / "clades_seasons_summary.csv").write_text(
        "season,seqs_t,lca_node\n"
        + "".join(f"{t},{M_T[t]},0\n" for t in sorted(STRAINS)))

    # Clades, one per node subtree, with X_t read off STRAINS and an arbitrary but
    # non-degenerate next-season frequency.
    rows = ["season,clade_id,root_nodes,n_mut,X_t,X_next,seqs_t,seqs_next"]
    subtree = {0: [1, 2, 3], 1: [1, 2], 2: [2], 3: [3]}
    x_next = {0: 1.0, 1: 0.9, 2: 0.5, 3: 0.1}
    for t in sorted(STRAINS):
        for cid, members in subtree.items():
            seqs_t = sum(STRAINS[t][n] for n in members)
            nxt = x_next[cid]
            rows.append(f"{t},{cid},{cid},{cid},{seqs_t / M_T[t]},{nxt},"
                        f"{seqs_t},{int(nxt * M_T[t + 1])}")
    (tmp_path / "clades_seasons.csv").write_text("\n".join(rows) + "\n")
    return str(tmp_path)


@pytest.fixture
def data(tmp_path):
    return CladeData(_write(tmp_path), "seasons")


def test_features_are_the_varying_sites(data):
    # Only positions 1 (A/C) and 5 (A/G/T) vary between the strains.
    assert sorted({p for p, _ in data.features}) == [1, 5]
    assert set(data.features) == {(1, "A"), (1, "C"), (5, "A"), (5, "G"), (5, "T")}


def test_membership_and_profiles(data):
    ids, Q, q_pop, cover, S, W, u, m_i = data.season_profiles(2001)
    assert ids == [0, 1, 2, 3]
    assert cover == pytest.approx(1.0)
    assert list(m_i) == [10, 20, 30]          # rows follow the strain table: nodes 1, 2, 3
    # Clade 1 is nodes 1 and 2, i.e. 10 and 20 of the season's 60 sequences, so within
    # the clade they weigh 1/3 and 2/3.  Both carry C at position 1; only node 2 has G
    # at position 5.
    j_c, j_g = data.feature_index[(1, "C")], data.feature_index[(5, "G")]
    assert Q[1, j_c] == pytest.approx(1.0)
    assert Q[1, j_g] == pytest.approx(2.0 / 3.0)
    # The season as a whole is half C at position 1 (30 of 60) and a third G.
    assert q_pop[j_c] == pytest.approx(0.5)
    assert q_pop[j_g] == pytest.approx(1.0 / 3.0)
    # Every profile sums to one per position.
    for k in range(Q.shape[0]):
        for pos in (1, 5):
            cols = [i for i, (p, _) in enumerate(data.features) if p == pos]
            assert Q[k, cols].sum() == pytest.approx(1.0)


def test_design_columns_sum_to_zero_within_a_position(data):
    Z, y, rows, blocks = build_design(data, sorted(data.strains), cutoff=0.0)
    assert len(rows) == len(Z) == 8          # four clades in each of two seasons
    assert [b.season for b in blocks] == [2001, 2002]
    for pos in (1, 5):
        cols = [i for i, (p, _) in enumerate(data.features) if p == pos]
        assert np.abs(Z[:, cols].sum(axis=1)).max() == pytest.approx(0.0, abs=1e-12)


def test_zero_fitness_predicts_no_change(data):
    _, _, _, blocks = build_design(data, sorted(data.strains), cutoff=0.0)
    beta = np.zeros(len(data.features))
    for b in blocks:
        # All strains equally fit, so every clade's <e^f> equals the season's.
        assert np.abs(predict_exp(beta, b)).max() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0])
def test_exact_gradient_matches_central_differences(data, scale):
    _, _, _, blocks = build_design(data, sorted(data.strains), cutoff=0.0)
    D = len(data.features)
    rng = np.random.default_rng(7)
    beta = rng.normal(scale=scale, size=D) if scale else np.zeros(D)
    loss, grad = exp_loss_grad(beta, blocks, lam=0.3)
    h = 1e-6
    for j in range(D):
        bp, bm = beta.copy(), beta.copy()
        bp[j] += h
        bm[j] -= h
        num = (exp_loss_grad(bp, blocks, 0.3)[0] - exp_loss_grad(bm, blocks, 0.3)[0]) / (2 * h)
        assert num == pytest.approx(grad[j], rel=1e-5, abs=1e-6)


def test_linear_model_is_the_first_order_expansion(data):
    """The exact objective's gradient at beta = 0 is the linear design's, and the two
    predictions agree to first order in beta."""
    Z, y, rows, blocks = build_design(data, sorted(data.strains), cutoff=0.0)
    D = len(data.features)
    _, g0 = exp_loss_grad(np.zeros(D), blocks, lam=0.0)
    assert g0 == pytest.approx(2 * Z.T @ (-y), rel=1e-9, abs=1e-9)

    rng = np.random.default_rng(3)
    direction = rng.normal(size=D)
    errs = []
    for eps in (1e-2, 1e-3):
        beta = eps * direction
        exact = np.concatenate([predict_exp(beta, b) for b in blocks])
        errs.append(np.abs(exact - Z @ beta).max() / eps**2)
    # The gap is second order: dividing by eps^2 leaves it bounded.
    assert errs[0] == pytest.approx(errs[1], rel=0.2)


def test_exact_fit_beats_its_own_linear_approximation_in_sample(data):
    """Not a claim about prediction -- just that the extra flexibility is real."""
    Z, y, rows, blocks = build_design(data, sorted(data.strains), cutoff=0.0)
    D = len(data.features)
    lam = 1e-3
    beta_exp = fit_exp(blocks, lam, D)
    r_exp = np.concatenate([predict_exp(beta_exp, b) for b in blocks]) - y
    r_lin = Z @ ridge_dual(Z, y, lam) - y
    assert float(r_exp @ r_exp) <= float(r_lin @ r_lin) + 1e-9


def test_non_overlapping_shells_partition_the_season(data):
    """The paper's eqs. 29-31. Nested clades count a season's strains many times over;
    the shells count each strain exactly once."""
    from antiGen.clade_fitness import build_design, nonoverlapping_shells

    Zo, yo, nested, _ = build_design(data, sorted(data.strains), cutoff=0.0)
    Zn, yn, shells, blocks = build_design(data, sorted(data.strains), cutoff=0.0,
                                          non_overlapping=True)
    for season in (2001, 2002):
        nest = [r for r in nested if r["season"] == season]
        shell = [r for r in shells if r["season"] == season]
        # Clade 0 is the whole season and clades 1..3 sit inside it, so the nested
        # frequencies sum to well over one while the shells sum to exactly one.
        assert sum(r["X_t"] for r in nest) > 1.5
        assert sum(r["X_t"] for r in shell) == pytest.approx(1.0)
        assert sum(r["seqs_t"] for r in shell) == 60 if season == 2001 else 100
        assert all(r["shell"] for r in shell)
    # Clade 1 is nodes 1 and 2; clade 2 is node 2 alone; so clade 1's shell is node 1.
    s1 = [r for r in shells if r["season"] == 2001 and r["clade_id"] == 1][0]
    assert s1["seqs_t"] == 10
    # Every shell's strain weights still form a distribution.
    for b in blocks:
        assert np.allclose(np.asarray(b.Wc.sum(axis=1)).ravel(), 1.0)


def test_identical_membership_clades_collapse(data):
    """Two clades holding the same strains are one unit: eq. 31 subtracts proper subsets,
    so identical sets never remove each other and would otherwise double-count."""
    from antiGen.clade_fitness import nonoverlapping_shells
    members = {1: {0, 1}, 2: {0, 1}, 3: {0}}
    shells, inner = nonoverlapping_shells({1, 2, 3}, members)
    assert sorted(shells) == [2, 3]            # the later of the identical pair survives
    assert shells[2] == {1} and shells[3] == {0}
    assert set().union(*shells.values()) == {0, 1}


# ── the quadratic model ─────────────────────────────────────────────────────

@pytest.fixture
def quad(tmp_path):
    return CladeData(_write(tmp_path), "seasons", pairs=True, min_pair_count=1,
                     gauge=True)


@pytest.fixture
def gauged(tmp_path):
    return CladeData(_write(tmp_path), "seasons", gauge=True)


def test_gauge_drops_the_reference_allele(gauged):
    # Position 1 is A in node 3 (30 sequences in 2001) and C in nodes 1,2; position 5 is
    # A, G, T. The reference is the most abundant residue, and it carries no feature.
    assert (1, gauged.wt[1]) not in set(gauged.features)
    assert (5, gauged.wt[5]) not in set(gauged.features)
    assert len(gauged.features) == 5 - 2


def test_interactions_are_only_between_non_reference_alleles(quad):
    pairs = [f for f in quad.features if len(f) == 4]
    assert pairs, "expected at least one interaction"
    for i, j, k, l in pairs:
        assert i < k                                  # each unordered pair once
        assert j != quad.wt[i] and l != quad.wt[k]    # the gauge
    assert quad.n_linear + quad.n_pairs == len(quad.features)
    assert quad.is_pair.sum() == quad.n_pairs


def test_interactions_are_only_those_some_strain_carries(quad):
    """A pair no sequence carries is a column of zeros; it must not be a feature."""
    _, _, _, _, S, _, _, _ = quad.season_profiles(2001)
    cols = np.asarray(S.sum(axis=0)).ravel()
    assert (cols[quad.is_pair] >= 0).all()
    seen = set()
    for g in quad.strains.values():
        for nid in g.node_id:
            seq = quad.seq[int(nid)]
            act = [(p, seq[p]) for p in quad.varying_sites
                   if (p, seq[p]) in set(quad.features)]
            for a in range(len(act)):
                for b in range(a + 1, len(act)):
                    seen.add((act[a][0], act[a][1], act[b][0], act[b][1]))
    assert {f for f in quad.features if len(f) == 4} <= seen


def test_pair_column_is_the_product_of_its_two_linear_columns(quad):
    """The design has to actually encode 1[a_i=j]*1[a_k=l], or gamma means nothing."""
    _, _, _, _, S, _, _, _ = quad.season_profiles(2001)
    X = np.asarray(S.todense())
    for idx, f in enumerate(quad.features):
        if len(f) != 4:
            continue
        i, j, k, l = f
        a = X[:, quad.feature_index[(i, j)]]
        b = X[:, quad.feature_index[(k, l)]]
        assert np.allclose(X[:, idx], a * b)


def test_quadratic_contains_the_gauged_linear_model(tmp_path, quad, gauged):
    """Sending lambda_gamma to infinity has to recover the gauged linear fit, or the
    comparison between the two is not a comparison of nested models."""
    from antiGen.clade_fitness import build_design, fit_exp, penalty_vector, predict_exp
    seasons = sorted(quad.strains)
    _, _, _, bq = build_design(quad, seasons, cutoff=0.0)
    _, _, _, bl = build_design(gauged, seasons, cutoff=0.0)
    bq_fit = fit_exp(bq, penalty_vector((1.0, 1e12), quad.is_pair), len(quad.features))
    bl_fit = fit_exp(bl, 1.0, len(gauged.features))
    pq = np.concatenate([predict_exp(bq_fit, b) for b in bq])
    pl = np.concatenate([predict_exp(bl_fit, b) for b in bl])
    assert np.allclose(pq, pl, atol=1e-4)
    assert np.abs(bq_fit[quad.is_pair]).max() < 1e-6


@pytest.mark.parametrize("scale", [0.05, 0.3])
def test_quadratic_gradient_matches_central_differences(quad, scale):
    from antiGen.clade_fitness import build_design, exp_loss_grad, penalty_vector
    _, _, _, blocks = build_design(quad, sorted(quad.strains), cutoff=0.0)
    D = len(quad.features)
    rng = np.random.default_rng(0)
    beta = scale * rng.standard_normal(D)
    lam = penalty_vector((0.7, 3.0), quad.is_pair)
    _, g = exp_loss_grad(beta, blocks, lam)
    eps = 1e-6
    for idx in rng.choice(D, size=min(D, 12), replace=False):
        d = np.zeros(D); d[idx] = eps
        hi, _ = exp_loss_grad(beta + d, blocks, lam)
        lo, _ = exp_loss_grad(beta - d, blocks, lam)
        assert g[idx] == pytest.approx((hi - lo) / (2 * eps), rel=2e-4, abs=1e-6)


def test_vector_penalty_ridge_matches_the_explicit_solve():
    from antiGen.clade_fitness import ridge_dual
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((6, 11))
    y = rng.standard_normal(6)
    d = np.abs(rng.standard_normal(11)) + 0.5
    want = np.linalg.solve(Z.T @ Z + np.diag(d), Z.T @ y)
    assert np.allclose(ridge_dual(Z, y, d), want, atol=1e-8)
    lam = 2.3
    want2 = np.linalg.solve(Z.T @ Z + lam * np.eye(11), Z.T @ y)
    assert np.allclose(ridge_dual(Z, y, lam), want2, atol=1e-8)
