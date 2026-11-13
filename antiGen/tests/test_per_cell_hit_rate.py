"""Tests for ``compute_hit_rate(per_cell=True)``, the cell-weighted list metric.

The macro (per-mutation) list recall in ``antiGen/list_macro.py`` exists because the
published list metric averages over mutation EVENTS, and a curated list's cells recur
at wildly different rates (CoG-UK RBD: 1 to 972 events per cell). The only thing that
makes the two readings comparable is that the *inner* question -- "is this cell inside
this query's top q?" -- is decided identically. These tests pin that down:

  * the per-cell tensors sum back to the scalars the existing path returns, so no
    hit mass is created or lost by the regrouping;
  * hit mass lands on the cell it belongs to, not merely in the right total;
  * events are counted per cell, duplicates included, so a cell's denominator is the
    number of base sequences it was observed relative to.

Core dependencies only (torch); no model downloads.
"""
import torch

from antiGen.eval_utils import compute_hit_rate


ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALPHABET)
L = 16
EVAL_RANGE = list(range(4, 12))


def _batch(seed, B=6):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(B, L, E, generator=g)
    aa = torch.randint(0, E, (B, L), generator=g)
    target = (torch.rand(B, L, E, generator=g) < 0.05).float()
    # A target never lands on the query's own residue -- that is not a mutation, and
    # compute_hit_rate NaN-masks those cells.
    target.scatter_(2, aa.unsqueeze(-1), 0.0)
    return scores, aa, target


def test_per_cell_sums_to_the_scalar_metric():
    for seed in (0, 1, 2):
        scores, aa, target = _batch(seed)
        for qtile in (0.1, 0.25, 0.5):
            n_hits, n_muts = compute_hit_rate(scores, target, aa, qtile,
                                              eval_range=EVAL_RANGE)
            n_hits2, n_muts2, cell_hits, cell_events = compute_hit_rate(
                scores, target, aa, qtile, eval_range=EVAL_RANGE, per_cell=True)
            assert (n_hits, n_muts) == (n_hits2, n_muts2)
            assert cell_hits.shape == (len(EVAL_RANGE), E)
            assert torch.isclose(cell_hits.sum(), torch.tensor(n_hits), atol=1e-4)
            assert torch.isclose(cell_events.sum(), torch.tensor(n_muts), atol=1e-4)


def test_hit_mass_lands_on_the_right_cell():
    """One example, one target, a score that is unambiguously inside the top decile."""
    scores = torch.zeros(1, L, E)
    aa = torch.zeros(1, L, dtype=torch.long)
    target = torch.zeros(1, L, E)
    pos, alt = 7, 5                      # inside EVAL_RANGE, not the wild-type index
    scores[0, pos, alt] = 10.0
    target[0, pos, alt] = 1.0

    n_hits, n_muts, cell_hits, cell_events = compute_hit_rate(
        scores, target, aa, 0.1, eval_range=EVAL_RANGE, per_cell=True)

    row = EVAL_RANGE.index(pos)
    assert n_muts == 1.0
    assert cell_events[row, alt] == 1.0
    assert cell_events.sum() == 1.0
    assert cell_hits[row, alt] > 0.99
    assert torch.isclose(cell_hits.sum(), cell_hits[row, alt])


def test_events_count_per_cell_not_per_example():
    """Two base sequences carrying the same mutation give that cell a denominator of 2.

    One of them ranks it top, the other bottom, so the cell's rate is 1/2 -- the
    quantity ``list_macro.py`` averages over cells.
    """
    scores = torch.zeros(2, L, E)
    aa = torch.zeros(2, L, dtype=torch.long)
    target = torch.zeros(2, L, E)
    pos, alt = 9, 3
    scores[0, pos, alt] = 10.0          # example 0 ranks it at the top
    scores[1, pos, alt] = -10.0         # example 1 ranks it at the bottom
    scores[1, 5, 6] = 10.0
    target[:, pos, alt] = 1.0

    _h, n_muts, cell_hits, cell_events = compute_hit_rate(
        scores, target, aa, 0.1, eval_range=EVAL_RANGE, per_cell=True)

    row = EVAL_RANGE.index(pos)
    assert n_muts == 2.0
    assert cell_events[row, alt] == 2.0
    assert abs(float(cell_hits[row, alt] / cell_events[row, alt]) - 0.5) < 1e-4


def test_a_cell_observed_twice_in_one_example_counts_twice():
    """``mat`` holds counts, not flags: two children with the same mutation make 2."""
    scores = torch.zeros(1, L, E)
    aa = torch.zeros(1, L, dtype=torch.long)
    target = torch.zeros(1, L, E)
    pos, alt = 6, 4
    scores[0, pos, alt] = 10.0
    target[0, pos, alt] = 2.0

    _h, n_muts, cell_hits, cell_events = compute_hit_rate(
        scores, target, aa, 0.1, eval_range=EVAL_RANGE, per_cell=True)

    row = EVAL_RANGE.index(pos)
    assert n_muts == 2.0
    assert cell_events[row, alt] == 2.0
    assert cell_hits[row, alt] > 1.98        # both counted, both hits


def test_ties_are_split_fractionally_per_cell():
    """A flat score map ties every cell, so each target gets exactly q of a hit."""
    scores = torch.zeros(1, L, E)
    aa = torch.zeros(1, L, dtype=torch.long)
    target = torch.zeros(1, L, E)
    target[0, 6, 4] = 1.0
    target[0, 9, 3] = 1.0

    for qtile in (0.1, 0.3):
        _h, _n, cell_hits, cell_events = compute_hit_rate(
            scores, target, aa, qtile, eval_range=EVAL_RANGE, per_cell=True)
        rates = cell_hits[cell_events > 0] / cell_events[cell_events > 0]
        assert torch.allclose(rates, torch.full_like(rates, qtile), atol=1e-4)
