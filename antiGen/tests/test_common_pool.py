"""The common candidate pool: one top-decile size for every model.

Without a pool mask, ``n_valid`` is whatever a model emits -- 3,819 to 4,221 across the
real roster, a 10% spread that is a property of a model's alphabet. These tests pin the
four properties the fix rests on: the pool is model-independent, an unscorable cell
inside it ranks last rather than leaving it, the two implementations of the metric still
agree cell for cell, and passing no mask reproduces the published behaviour exactly.
"""
import pytest
import torch

from antiGen.background_discrimination import rank_fractions, top_q_weight
from antiGen.eval_utils import compute_hit_rate, real_substitution_mask

AAS = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(AAS)
R = 12
GAP, STOP = AAS.index('-'), AAS.index('*')


def _scores(seed, drop_cols=()):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(4, R, E, generator=g)
    for c in drop_cols:
        s[:, :, c] = float('nan')
    return s


def _aa(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 20, (4, R), generator=g)


def test_real_substitution_mask_drops_exactly_gap_and_stop():
    m = real_substitution_mask(R, AAS)
    assert m.shape == (R, E)
    assert not m[:, GAP].any() and not m[:, STOP].any()
    keep = [i for i in range(E) if i not in (GAP, STOP)]
    assert m[:, keep].all()
    assert int(m.sum()) == R * 20


@pytest.mark.parametrize("drop", [(), (STOP,), (GAP, STOP)])
def test_pool_makes_n_valid_model_independent(drop):
    """The whole point: a model that declines gap/stop gets the same pool as one that
    does not. Without the mask these differ by exactly the dropped columns."""
    aa = _aa()
    pool = real_substitution_mask(R, AAS)
    full = _scores(1)
    partial = _scores(1, drop_cols=drop)

    _, _, nv_full = rank_fractions(full, aa, list(range(R)), pool_mask=pool)
    _, _, nv_part = rank_fractions(partial, aa, list(range(R)), pool_mask=pool)
    assert torch.equal(nv_full, nv_part)

    # and without the mask they differ by the dropped columns (when any were dropped)
    _, _, nv_full_legacy = rank_fractions(full, aa, list(range(R)))
    _, _, nv_part_legacy = rank_fractions(partial, aa, list(range(R)))
    expected_gap = float(len(drop) * R)
    assert torch.allclose(nv_full_legacy - nv_part_legacy,
                          torch.full_like(nv_full_legacy, expected_gap))


def test_unscorable_cell_in_pool_ranks_last_and_still_counts():
    """A NaN inside the pool stays in n_valid and is never selected."""
    aa = _aa()
    pool = real_substitution_mask(R, AAS)
    s = _scores(2)
    s[:, 0, 0] = float('nan')          # a real amino-acid column, so inside the pool

    _, _, nv = rank_fractions(s, aa, list(range(R)), pool_mask=pool)
    _, _, nv_ref = rank_fractions(_scores(2), aa, list(range(R)), pool_mask=pool)
    assert torch.equal(nv, nv_ref), "an unscorable in-pool cell must not shrink the pool"

    above, tie, nvv = rank_fractions(s, aa, list(range(R)), pool_mask=pool)
    for q in (0.1, 0.5):
        W = top_q_weight(above, tie, nvv, q)
        assert float(W[:, 0, 0].max()) == 0.0, "a ranked-last cell must never be selected"


@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.9])
@pytest.mark.parametrize("drop", [(), (STOP,), (GAP, STOP)])
def test_hit_rate_and_rank_fractions_agree_under_the_pool(q, drop):
    """The identity the ensemble path leans on must survive the pool mask."""
    aa = _aa(3)
    pool = real_substitution_mask(R, AAS)
    s = _scores(4, drop_cols=drop)
    g = torch.Generator().manual_seed(5)
    tgt = (torch.rand(4, R, E, generator=g) < 0.05).float()

    h, _ = compute_hit_rate(s.clone(), tgt, aa, qtile=q, eval_range=list(range(R)),
                            pool_mask=pool)
    above, tie, nv = rank_fractions(s.clone(), aa, list(range(R)), pool_mask=pool)
    h2 = float((top_q_weight(above, tie, nv, q) * tgt).sum())
    assert h == pytest.approx(h2, abs=1e-4)


@pytest.mark.parametrize("q", [0.1, 0.5])
def test_no_mask_is_the_published_behaviour(q):
    """pool_mask=None must change nothing, so the legacy regression is meaningful."""
    aa = _aa(6)
    s = _scores(7, drop_cols=(GAP,))
    g = torch.Generator().manual_seed(8)
    tgt = (torch.rand(4, R, E, generator=g) < 0.05).float()

    h_none, n_none = compute_hit_rate(s.clone(), tgt, aa, qtile=q,
                                      eval_range=list(range(R)), pool_mask=None)
    h_omit, n_omit = compute_hit_rate(s.clone(), tgt, aa, qtile=q,
                                      eval_range=list(range(R)))
    assert h_none == h_omit and n_none == n_omit

    a1, t1, v1 = rank_fractions(s.clone(), aa, list(range(R)), pool_mask=None)
    a2, t2, v2 = rank_fractions(s.clone(), aa, list(range(R)))
    assert torch.equal(v1, v2)
    assert torch.equal(torch.nan_to_num(a1, nan=-1), torch.nan_to_num(a2, nan=-1))


def test_pool_shrinks_the_selected_set_for_a_gap_emitting_model():
    """Sanity on the direction: dropping gap/stop from the pool shrinks q*n_valid."""
    aa = _aa(9)
    s = _scores(10)
    pool = real_substitution_mask(R, AAS)
    _, _, nv_legacy = rank_fractions(s, aa, list(range(R)))
    _, _, nv_pool = rank_fractions(s, aa, list(range(R)), pool_mask=pool)
    assert float(nv_pool[0]) < float(nv_legacy[0])
    assert float(nv_legacy[0] - nv_pool[0]) == pytest.approx(2 * R, abs=1e-6)


# ── An unscorable in-pool cell must never be SELECTED, at any q ──────────────
# Ranking last is not the same as never being selected: the filled cells form one tie
# block at the bottom, so once q*n_valid exceeds the SCORABLE count the boundary lands
# inside that block and every cell in it takes fractional weight. Measured on BLOSUM62,
# which carries query-dependent -inf: 75.8 of the 3,437.1 cells selected at q = 0.9,
# worth -0.0039 of its recall AUC.

@pytest.mark.parametrize("q", [0.1, 0.5, 0.8, 0.9, 0.95])
def test_unscorable_cells_are_never_selected_at_any_q(q):
    aa = _aa(11)
    pool = real_substitution_mask(R, AAS)
    s = _scores(12)
    # abstain on MOST of the pool, so the quota must exceed the scorable count
    s[:, :, 2:18] = float('nan')
    uns = (torch.isnan(s) | torch.isneginf(s)) & pool.unsqueeze(0)
    above, tie, nv, filled = rank_fractions(s.clone(), aa, list(range(R)),
                                            pool_mask=pool, return_filled=True)
    W = top_q_weight(above, tie, nv, q)
    W = torch.where(filled, torch.zeros_like(W), W)
    leaked = float(torch.nan_to_num(W[uns & ~torch.isnan(W)], nan=0.0).sum())
    assert leaked == pytest.approx(0.0, abs=1e-9), f"{leaked} weight on unscorable cells"


@pytest.mark.parametrize("q", [0.5, 0.9])
def test_hit_rate_gives_no_credit_for_unscorable_cells(q):
    """compute_hit_rate must not count a target the model declined to score."""
    aa = _aa(13)
    pool = real_substitution_mask(R, AAS)
    s = _scores(14)
    s[:, :, 2:18] = float('nan')
    tgt = torch.zeros(4, R, E)
    tgt[:, :, 2:18] = 1.0          # every target sits on an unscorable cell
    tgt = tgt * pool.unsqueeze(0)
    h, n = compute_hit_rate(s.clone(), tgt, aa, qtile=q, eval_range=list(range(R)),
                            pool_mask=pool)
    assert n > 0
    assert h == pytest.approx(0.0, abs=1e-6), f"credited {h} hits on unscorable cells"


def test_filled_mask_is_empty_without_a_pool():
    """pool_mask=None must stay bit-exact: nothing is filled, nothing is zeroed."""
    aa = _aa(15)
    s = _scores(16, drop_cols=(GAP,))
    out = rank_fractions(s.clone(), aa, list(range(R)), return_filled=True)
    assert len(out) == 4 and not out[3].any()
