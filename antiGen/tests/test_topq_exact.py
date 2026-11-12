"""The top-q selection holds EXACTLY q * n cells (NOTES s161)."""
import torch

from antiGen.eval_utils import compute_hit_rate, topq_boundary

R0, R1, E = 330, 531, 20


def _selected_mass(scores, pool):
    tt = torch.zeros(1, 1273, E); tt[0, R0:R1] = pool.float()
    aa = torch.zeros(1, 1273, dtype=torch.long)
    _h, _n, ch, _ce = compute_hit_rate(scores, tt, aa, 0.1, eval_range=list(range(R0, R1)),
                                       per_cell=True, pool_mask=pool)
    return float(ch.sum()), int(pool.sum())


def test_untied_scores_select_exactly_q_n():
    g = torch.Generator().manual_seed(0)
    for n_keep in (95, 474, 854, 954, 1490, 1896, 3819):
        sc = torch.randn(1, R1 - R0, E, generator=g)
        pool = torch.zeros(1, R1 - R0, E, dtype=torch.bool)
        pool.view(-1)[torch.randperm((R1 - R0) * E, generator=g)[:n_keep]] = True
        pool[0, :, 0] = False                       # the wild type (A) is never a candidate
        got, n = _selected_mass(sc, pool)
        assert abs(got - 0.1 * n) < 1e-4, (n, got)


def test_tied_and_coarse_scores_select_exactly_q_n():
    g = torch.Generator().manual_seed(1)
    for levels in (1, 2, 8, 50):
        sc = torch.randint(0, levels, (1, R1 - R0, E), generator=g).float()
        pool = torch.ones(1, R1 - R0, E, dtype=torch.bool); pool[0, :, 0] = False
        got, n = _selected_mass(sc, pool)
        assert abs(got - 0.1 * n) < 1e-3, (levels, got, 0.1 * n)


def test_boundary_is_the_ceil_q_n_th_largest():
    p = torch.tensor([[5., 4., 3., 2., 1., float('nan')], [1., 1., 1., 1., float('nan'), float('nan')]])
    vb = topq_boundary(p, 0.3, dim=1)               # n = 5 -> k = 2; n = 4 -> k = 2
    assert vb.flatten().tolist() == [4.0, 1.0]
