"""Invariants of the ensemble rule algebra.

The load-bearing one is :func:`test_one_member_is_a_no_op`. Every later verification in
this arm -- the backbone replay at 1e-9, the attached-tree replay at 1e-7 -- is a
comparison between a *one-member ensemble* and a published single-model row, so those
checks only mean something if a one-member ensemble is guaranteed to be the identity on
ranks. ``collect_figure_metrics.logp_model`` states the same invariant for the multi-tree
arm's averaging ("a one-element ensemble is a no-op"); this generalises it over every
(normalisation, pooling, abstention) combination in ``LEGAL``.
"""
from __future__ import annotations

import pytest
import torch

from antiGen import ensemble_rules as er
from antiGen.eval_utils import compute_hit_rate

R, E, B = 9, 22, 6
SEQ_LEN = 40
EVAL_RANGE = list(range(12, 12 + R))


def _fixture(seed=0, gap_stop_nan=True, coarse=False):
    """A map shaped like a real member's: NaN gap/stop columns, some ties."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, R, E, generator=g)
    if coarse:                      # a BLOSUM62-shaped member: ~8 distinct levels
        x = (x * 2).round() / 2
    if gap_stop_nan:
        x[:, :, 20] = float('nan')  # '*'
        x[:, :, 21] = float('nan')  # '-'
    aa = torch.randint(0, 20, (B, SEQ_LEN), generator=g)
    tgt = torch.zeros(B, SEQ_LEN, E)
    for b in range(B):
        for _ in range(3):
            p = int(torch.randint(0, R, (1,), generator=g))
            e = int(torch.randint(0, 20, (1,), generator=g))
            tgt[b, EVAL_RANGE[p], e] += 1.0
    return x, aa, tgt


def _all_legal_rules(members):
    out = []
    for norm, pl in sorted(er.LEGAL):
        for nan in er.NAN_CONVENTIONS:
            kw = {}
            if pl == "power":
                kw = {"beta": -1.0}
            elif pl == "trimmed":
                kw = {"trim": 0.25}
            elif pl == "rrf":
                kw = {"k": 60.0}
            elif pl == "vote":
                kw = {"qprime": 0.1}
            out.append(er.Rule(norm=norm, pool=pl, nan=nan, members=members, **kw))
    return out


@pytest.mark.parametrize("coarse", [False, True])
def test_one_member_is_a_no_op(coarse):
    """A one-member ensemble scores identically to the raw map, under every rule.

    Checked through ``compute_hit_rate`` itself rather than on the ranks, because that
    is the function whose output the published CSVs hold: a rule that reordered a tie
    block would change ``cell_hits`` even while preserving the sort order.
    """
    x, aa, tgt = _fixture(coarse=coarse)
    for q in (0.1, 0.5):
        ref = compute_hit_rate(x.clone(), tgt, aa, q, eval_range=EVAL_RANGE,
                               per_cell=True)
        for rule in _all_legal_rules(("m",)):
            got = compute_hit_rate(er.apply({"m": x.clone()}, rule), tgt, aa, q,
                                   eval_range=EVAL_RANGE, per_cell=True)
            assert abs(got[0] - ref[0]) < 1e-9, (rule.label, q)
            assert got[1] == ref[1], (rule.label, q)
            assert torch.allclose(got[2], ref[2], atol=1e-9), (rule.label, q)
            assert torch.allclose(got[3], ref[3], atol=1e-9), (rule.label, q)


def test_repeated_member_is_also_a_no_op():
    """Greedy selection with replacement duplicates members; that must change nothing."""
    x, aa, tgt = _fixture(seed=3)
    ref = compute_hit_rate(x.clone(), tgt, aa, 0.1, eval_range=EVAL_RANGE, per_cell=True)
    for rule in _all_legal_rules(("m", "m", "m")):
        got = compute_hit_rate(er.apply({"m": x.clone()}, rule), tgt, aa, 0.1,
                               eval_range=EVAL_RANGE, per_cell=True)
        assert torch.allclose(got[2], ref[2], atol=1e-9), rule.label


def test_power_beta_one_is_arithmetic_and_beta_zero_is_geometric():
    xs = [_fixture(seed=i)[0] for i in range(3)]
    maps = {f"m{i}": x for i, x in enumerate(xs)}
    names = tuple(maps)
    base = dict(norm="pctile", nan="union_min", members=names)
    a = er.apply(maps, er.Rule(pool="arith", **base))
    p1 = er.apply(maps, er.Rule(pool="power", beta=1.0, **base))
    assert torch.allclose(a[~torch.isnan(a)], p1[~torch.isnan(p1)], atol=1e-6)
    g = er.apply(maps, er.Rule(pool="geom", **base))
    p0 = er.apply(maps, er.Rule(pool="power", beta=0.0, **base))
    assert torch.allclose(g[~torch.isnan(g)], p0[~torch.isnan(p0)], atol=1e-6)


def test_pooling_is_permutation_invariant():
    maps = {f"m{i}": _fixture(seed=i)[0] for i in range(4)}
    fwd = tuple(maps)
    rev = tuple(reversed(fwd))
    for pl, kw in (("arith", {}), ("geom", {}), ("max", {}), ("min", {}), ("median", {}),
                   ("trimmed", {"trim": 0.25}), ("rrf", {"k": 60.0})):
        a = er.apply(maps, er.Rule(norm="pctile", pool=pl, nan="union_min",
                                   members=fwd, **kw))
        b = er.apply(maps, er.Rule(norm="pctile", pool=pl, nan="union_min",
                                   members=rev, **kw))
        m = ~torch.isnan(a)
        assert torch.allclose(a[m], b[m], atol=1e-6), pl


def test_union_minus_intersect_is_exactly_the_symmetric_difference():
    """The pool-size asymmetry NOTES s32 measured, pinned as an identity."""
    x0, _, _ = _fixture(seed=1, gap_stop_nan=True)    # abstains on '*' and '-'
    x1, _, _ = _fixture(seed=2, gap_stop_nan=False)   # emits every column
    stack = torch.stack([x0, x1])
    normed = torch.stack([er.normalize(stack[i], "pctile", ~torch.isnan(stack[i]))
                          for i in range(2)])
    _, v_i = er.resolve_nan(normed, "intersect")
    _, v_u = er.resolve_nan(normed, "union_min")
    assert bool((v_u | v_i).eq(v_u).all())
    finite = ~torch.isnan(stack)
    only_one = finite.any(dim=0) & ~finite.all(dim=0)
    assert bool((v_u & ~v_i).eq(only_one).all())
    assert int(v_u.sum()) - int(v_i.sum()) == int(only_one.sum()) == B * R * 2


def test_pctile_is_monotone_and_handles_ties():
    x, _, _ = _fixture(seed=5)
    valid = ~torch.isnan(x)
    p = er.pctile_ranks(x, valid)
    for b in range(B):
        v = x[b][valid[b]]
        pr = p[b][valid[b]]
        o = torch.argsort(v)
        assert bool((pr[o].diff() >= -1e-9).all())
        assert abs(float(pr.min())) < 1e-9 and abs(float(pr.max()) - 1.0) < 1e-9
    # A flat map asserts no ordering: every cell takes the mid-rank.
    flat = torch.zeros(1, R, E)
    pf = er.pctile_ranks(flat, torch.ones_like(flat, dtype=torch.bool))
    assert torch.allclose(pf, torch.full_like(pf, 0.5))


def test_globalz_preserves_cross_query_level_but_pctile_destroys_it():
    """Why `globalz` exists: the background panels read the level, not the rank."""
    x, _, _ = _fixture(seed=7)
    x[0] += 5.0                                   # one uniformly high-scoring query
    valid = ~torch.isnan(x)
    gz = er.normalize(x, "globalz", valid)
    pc = er.normalize(x, "pctile", valid)
    means_gz = torch.stack([gz[b][valid[b]].mean() for b in range(B)])
    means_pc = torch.stack([pc[b][valid[b]].mean() for b in range(B)])
    assert float(means_gz[0] - means_gz[1:].mean()) > 1.0
    assert float((means_pc - means_pc.mean()).abs().max()) < 1e-6


def test_rule_id_collapses_duplicates_and_ignores_provenance():
    a = er.Rule(norm="pctile", pool="arith", nan="union_min", members=("x", "y"),
                weights=(1.0, 1.0), family="greedy")
    b = er.Rule(norm="pctile", pool="arith", nan="union_min", members=("y", "x"),
                weights=(1.0, 1.0), family="simplex")
    assert a.rule_id == b.rule_id
    c = er.Rule(norm="pctile", pool="arith", nan="union_min", members=("x", "y"),
                weights=(2.0, 1.0))
    assert c.rule_id != a.rule_id


def test_illegal_combinations_are_refused():
    with pytest.raises(ValueError):
        er.Rule(norm="zscore", pool="geom", nan="union_min", members=("x",))
    with pytest.raises(ValueError):
        er.Rule(norm="pctile", pool="arith", nan="union_min", members=())
    with pytest.raises(ValueError):
        er.Rule(norm="pctile", pool="arith", nan="union_min", members=("x",),
                weights=(1.0, 2.0))


def test_grid_has_no_duplicate_rule_ids():
    for stage in ("families", "gate", "nan"):
        rules = er.enumerate_grid(("a", "b", "c"), stage)
        ids = [r.rule_id for r in rules]
        assert len(ids) == len(set(ids)), stage
        assert rules, stage


def test_missing_member_exits_rather_than_guessing():
    x, _, _ = _fixture()
    with pytest.raises(SystemExit):
        er.apply({"m": x}, er.Rule(norm="pctile", pool="arith", nan="union_min",
                                   members=("m", "absent")))
