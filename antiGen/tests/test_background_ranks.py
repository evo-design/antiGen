"""Tests for ``rank_fractions``, the stored form of the background-discrimination scores.

``antiGen/background_discrimination.py`` does not store a metric; it stores, per (query,
cell), the share of the query's candidate pool scoring strictly above that cell and the
share tied with it. Everything downstream -- the host-vs-control contrast, any top-q
reading -- is computed from those two numbers without a second forward pass. That is only
legitimate if they carry exactly the information ``compute_hit_rate`` extracts, so these
tests pin the identity

    top_q_weight(above, tie, n_valid, Q)  ==  compute_hit_rate's fractional weight

against ``compute_hit_rate(per_cell=True)`` itself, over random scores, tied scores
(BLOSUM62-style integer levels, where the fractional tie rule actually bites), NaN and
-inf sentinels, and every quantile the evals report.

Core dependencies only (torch); no model downloads.
"""
import torch

from antiGen.background_discrimination import rank_fractions, top_q_weight
from antiGen.eval_utils import compute_hit_rate


ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALPHABET)
L = 16
EVAL_RANGE = list(range(4, 12))
QTILES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _batch(seed, B=6, levels=None, sentinels=False):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(B, L, E, generator=g)
    if levels is not None:                       # coarse, heavily tied scores
        scores = torch.round(scores * levels) / levels
    aa = torch.randint(0, E, (B, L), generator=g)
    if sentinels:
        scores[:, :, ALPHABET.index('-')] = float('-inf')
        scores[0, EVAL_RANGE[0], ALPHABET.index('*')] = float('nan')
    target = (torch.rand(B, L, E, generator=g) < 0.08).float()
    target.scatter_(2, aa.unsqueeze(-1), 0.0)
    return scores, aa, target


def test_reproduces_compute_hit_rate_per_cell():
    for seed, levels, sentinels in ((0, None, False), (1, 4, False), (2, 4, True),
                                    (3, None, True)):
        scores, aa, target = _batch(seed, levels=levels, sentinels=sentinels)
        above, tie, n_valid = rank_fractions(scores, aa, EVAL_RANGE)
        for q in QTILES:
            _h, _n, cell_hits, cell_events = compute_hit_rate(
                scores, target, aa, q, eval_range=EVAL_RANGE, per_cell=True)
            w = top_q_weight(above, tie, n_valid, q)
            mine = (w * target[:, EVAL_RANGE, :]).sum(dim=0)
            assert torch.allclose(mine, cell_hits, atol=1e-5), (seed, q)
            assert abs(float(mine.sum()) - _h) < 1e-4, (seed, q)
            assert float(cell_events.sum()) == float(target[:, EVAL_RANGE, :].sum())


def test_non_candidates_are_nan_and_candidates_sum_to_one():
    scores, aa, _ = _batch(5, sentinels=True)
    above, tie, _n_valid = rank_fractions(scores, aa, EVAL_RANGE)
    aa_sub = aa[:, EVAL_RANGE]
    # The query's own residue is never a candidate.
    own = torch.gather(above, 2, aa_sub.unsqueeze(-1)).squeeze(-1)
    assert torch.isnan(own).all()
    # Nor is the -inf column.
    assert torch.isnan(above[:, :, ALPHABET.index('-')]).all()
    # Rank fractions are a partition of the pool: the best cell has nothing above it,
    # and above + tie never exceeds 1.
    valid = ~torch.isnan(above)
    assert float(above[valid].min()) == 0.0
    assert float((above + tie)[valid].max()) <= 1.0 + 1e-6
    for b in range(above.shape[0]):
        v = valid[b]
        n = int(v.sum())
        # Each distinct score level contributes tie == (count / n_valid).
        assert abs(float(tie[b][v].min()) - 1.0 / n) < 1e-6


def test_ties_share_the_same_fractions():
    scores, aa, _ = _batch(7, levels=2)          # many exact ties
    above, tie, _n_valid = rank_fractions(scores, aa, EVAL_RANGE)
    sub = scores[:, EVAL_RANGE, :]
    b = 0
    flat_s, flat_a, flat_t = sub[b].reshape(-1), above[b].reshape(-1), tie[b].reshape(-1)
    valid = ~torch.isnan(flat_a)
    for level in torch.unique(flat_s[valid]):
        sel = valid & (flat_s == level)
        assert float(flat_a[sel].std(unbiased=False)) < 1e-6
        assert float(flat_t[sel].std(unbiased=False)) < 1e-6
