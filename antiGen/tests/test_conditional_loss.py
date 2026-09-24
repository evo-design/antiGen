"""The conditional loss and the descendant counts it reads."""
import math

import numpy as np
import pytest
import torch

from antiGen.eval_utils import (compute_conditional_loss_batch, conditional_log_probs,
                                conditional_nll)
from antiGen.load import load_examples, split_by_window
from antiGen.eval_counts import Accumulator, score_reference, score_persistence

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"


def _example(cells, seq_len=4, E=3):
    """(mat, n_mat) for one example from {(pos, alt): n}."""
    mat = torch.zeros(1, seq_len, E)
    n_mat = torch.zeros(1, seq_len, E)
    for (pos, alt), n in cells.items():
        mat[0, pos, alt] = 1.0
        n_mat[0, pos, alt] = float(n)
    return mat, n_mat


def test_normalizes_over_the_occupied_cells_only():
    mat, n_mat = _example({(0, 1): 3, (1, 2): 1})
    logits = torch.zeros(1, 4, 3)
    logits[0, 0, 1] = 1.0
    logits[0, 2, 0] = 50.0  # an unoccupied cell, however confident, must not enter
    log_q, occupied = conditional_log_probs(logits, mat)
    assert float(log_q[0][occupied[0]].exp().sum()) == pytest.approx(1.0)
    q = torch.softmax(torch.tensor([1.0, 0.0]), 0)
    nll, n_obs, n_ex = conditional_nll(logits, mat, n_mat)
    assert float(nll) == pytest.approx(float(-(3 * q[0].log() + 1 * q[1].log())))
    assert float(n_obs) == 4.0 and int(n_ex) == 1


def test_a_single_child_costs_nothing_whatever_the_model_says():
    mat, n_mat = _example({(2, 0): 9})
    for logits in (torch.zeros(1, 4, 3), torch.randn(1, 4, 3) * 100):
        assert float(conditional_nll(logits, mat, n_mat)[0]) == pytest.approx(0.0)


def test_a_dead_end_is_skipped_rather_than_scored():
    mat, n_mat = _example({})
    nll, n_obs, n_ex = conditional_nll(torch.randn(1, 4, 3), mat, n_mat)
    assert float(nll) == 0.0 and float(n_obs) == 0.0 and int(n_ex) == 0


def test_uniform_logits_give_log_k_per_descendant():
    mat, n_mat = _example({(0, 0): 5, (1, 1): 3, (2, 2): 2})
    loss = compute_conditional_loss_batch(torch.zeros(1, 4, 3), mat, n_mat)
    assert float(loss) == pytest.approx(math.log(3))


def test_the_empirical_split_is_the_floor():
    """No score map beats q_i = n_i / sum(n): that is the MLE for the categorical."""
    mat, n_mat = _example({(0, 0): 7, (1, 1): 3})
    n = torch.tensor([7.0, 3.0])
    oracle = torch.zeros(1, 4, 3)
    oracle[0, 0, 0], oracle[0, 1, 1] = math.log(0.7), math.log(0.3)
    best = float(compute_conditional_loss_batch(oracle, mat, n_mat))
    assert best == pytest.approx(float(-(n * (n / n.sum()).log()).sum() / n.sum()))
    for _ in range(20):
        other = torch.randn(1, 4, 3)
        assert float(compute_conditional_loss_batch(other, mat, n_mat)) >= best - 1e-6


def test_non_finite_scores_are_excluded_not_propagated():
    mat, n_mat = _example({(0, 1): 2, (1, 2): 2})
    logits = torch.zeros(1, 4, 3)
    logits[0, 2, 2] = float("nan")   # a baseline's wild-type mask, outside the occupied set
    assert float(compute_conditional_loss_batch(logits, mat, n_mat)) == pytest.approx(math.log(2))


def test_loader_reads_the_third_field(tmp_path):
    seq = "MTEY"
    body = (
        "EXAMPLE_ID: 0\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nTRAINING\n"
        "IS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\n" + seq + "\n"
        "MUTATIONS_TO_CHILDREN:\n0,L,3\n2,K,2\n0,L,4\n\n"
    )
    path = tmp_path / "examples_cutoff_1.txt"
    path.write_text(body)
    ex, = load_examples(str(path), ALPHABET, None)
    L, K = ALPHABET.index("L"), ALPHABET.index("K")
    # repeated (pos, alt) rows accumulate in both matrices
    assert ex.mat[0, L] == 2.0 and ex.n_mat[0, L] == 7.0
    assert ex.mat[2, K] == 1.0 and ex.n_mat[2, K] == 2.0
    assert ex.n_mat.sum() == 9.0


def test_loader_reports_absent_counts_rather_than_inventing_them(tmp_path):
    body = (
        "EXAMPLE_ID: 0\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nTRAINING\n"
        "IS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEY\n"
        "MUTATIONS_TO_CHILDREN:\n0,L\n2,K\n\n"
    )
    path = tmp_path / "examples_cutoff_1.txt"
    path.write_text(body)
    ex, = load_examples(str(path), ALPHABET, None)
    assert ex.n_mat is None
    assert ex.mat.sum() == 2.0


def test_loader_rejects_a_half_counted_file(tmp_path):
    body = (
        "EXAMPLE_ID: 0\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nTRAINING\n"
        "IS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEY\n"
        "MUTATIONS_TO_CHILDREN:\n0,L,3\n2,K\n\n"
    )
    path = tmp_path / "examples_cutoff_1.txt"
    path.write_text(body)
    with pytest.raises(ValueError, match="descendant count"):
        load_examples(str(path), ALPHABET, None)


# ── The unconditional variant ─────────────────────────────────────────────────

from antiGen.eval_utils import (compute_unconditional_loss_batch, non_wt_log_probs,  # noqa: E402
                                unconditional_nll)


def test_uniform_logits_give_log_of_the_non_wildtype_grid():
    seq_len, E = 4, 3
    aa = torch.zeros(1, seq_len, dtype=torch.long)     # wild-type is token 0 everywhere
    _, n_mat = _example({(0, 1): 3, (1, 2): 1}, seq_len, E)
    loss = compute_unconditional_loss_batch(torch.zeros(1, seq_len, E), aa, n_mat)
    assert float(loss) == pytest.approx(math.log(seq_len * (E - 1)))


def test_wild_type_cells_carry_no_mass():
    aa = torch.tensor([[0, 1, 2, 0]])
    log_p = non_wt_log_probs(torch.randn(1, 4, 3), aa)
    assert float(log_p.exp().sum()) == pytest.approx(1.0)
    for pos, wt in enumerate(aa[0].tolist()):
        assert float(log_p[0, pos, wt].exp()) == pytest.approx(0.0)


def test_unconditional_is_conditional_plus_coverage():
    """uncond = cond - log(mass on the occupied cells), for any score map."""
    seq_len, E = 5, 4
    aa = torch.randint(0, E, (1, seq_len))
    mat, n_mat = _example({(0, (int(aa[0, 0]) + 1) % E): 5,
                           (2, (int(aa[0, 2]) + 2) % E): 2}, seq_len, E)
    for _ in range(10):
        scores = torch.randn(1, seq_len, E)
        log_p = non_wt_log_probs(scores, aa)
        mass = float(log_p[0][mat[0] > 0].exp().sum())
        cond = float(compute_conditional_loss_batch(scores, mat, n_mat))
        uncond = float(compute_unconditional_loss_batch(scores, aa, n_mat))
        assert uncond == pytest.approx(cond - math.log(mass), abs=1e-5)


def test_a_single_child_is_not_free_here():
    """Unlike the conditional loss, k == 1 still costs: the mass has to land on that cell."""
    seq_len, E = 4, 3
    aa = torch.zeros(1, seq_len, dtype=torch.long)
    _, n_mat = _example({(2, 1): 9}, seq_len, E)
    flat = float(compute_unconditional_loss_batch(torch.zeros(1, seq_len, E), aa, n_mat))
    assert flat == pytest.approx(math.log(seq_len * (E - 1)))
    confident = torch.full((1, seq_len, E), -10.0)
    confident[0, 2, 1] = 10.0
    assert float(compute_unconditional_loss_batch(confident, aa, n_mat)) < 0.01


def test_counts_not_child_counts_are_what_is_weighted():
    """Two children with wildly different descendant counts pull the loss differently."""
    seq_len, E = 4, 3
    aa = torch.zeros(1, seq_len, dtype=torch.long)
    _, lopsided = _example({(0, 1): 100, (1, 2): 1}, seq_len, E)
    scores = torch.zeros(1, seq_len, E)
    scores[0, 0, 1] = 5.0     # all confidence on the cell with the 100 descendants
    good = float(compute_unconditional_loss_batch(scores, aa, lopsided))
    scores2 = torch.zeros(1, seq_len, E)
    scores2[0, 1, 2] = 5.0    # ...on the cell with 1
    assert good < float(compute_unconditional_loss_batch(scores2, aa, lopsided))


def test_unconditional_skips_examples_with_no_children():
    seq_len, E = 4, 3
    aa = torch.zeros(1, seq_len, dtype=torch.long)
    _, n_mat = _example({}, seq_len, E)
    nll, n_obs = unconditional_nll(torch.randn(1, seq_len, E), aa, n_mat)
    assert float(nll) == 0.0 and float(n_obs) == 0.0


# ── KL against the normalized counts ──────────────────────────────────────────

from antiGen.eval_utils import compute_kl_loss_batch, kl_to_empirical  # noqa: E402


def test_kl_is_zero_exactly_at_the_empirical_split():
    mat, n_mat = _example({(0, 1): 7, (1, 2): 3})
    perfect = torch.full((1, 4, 3), -50.0)
    perfect[0, 0, 1], perfect[0, 1, 2] = math.log(0.7), math.log(0.3)
    assert float(compute_kl_loss_batch(perfect, mat, n_mat)) == pytest.approx(0.0, abs=1e-6)


def test_kl_is_never_negative():
    mat, n_mat = _example({(0, 1): 7, (1, 2): 3, (2, 0): 1})
    for _ in range(50):
        assert float(compute_kl_loss_batch(torch.randn(1, 4, 3), mat, n_mat)) >= -1e-6


def test_kl_against_uniform_matches_the_closed_form():
    mat, n_mat = _example({(0, 1): 7, (1, 2): 3})
    expected = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
    assert float(compute_kl_loss_batch(torch.zeros(1, 4, 3), mat, n_mat)) == pytest.approx(expected)


def test_kl_equals_conditional_cross_entropy_less_the_entropy():
    """The two objectives differ by a per-example constant, hence share their gradients."""
    mat, n_mat = _example({(0, 1): 5, (1, 2): 2, (3, 0): 1})
    n = torch.tensor([5.0, 2.0, 1.0])
    p_hat = n / n.sum()
    entropy = float(-(p_hat * p_hat.log()).sum())
    for _ in range(10):
        scores = torch.randn(1, 4, 3)
        cross = float(compute_conditional_loss_batch(scores, mat, n_mat))  # per descendant
        assert float(compute_kl_loss_batch(scores, mat, n_mat)) == pytest.approx(cross - entropy,
                                                                                 abs=1e-5)


def test_kl_leaves_out_single_child_examples_rather_than_averaging_in_zeros():
    """Two examples, one of them k == 1: the mean must be over the one that has content."""
    mat = torch.zeros(2, 4, 3)
    n_mat = torch.zeros(2, 4, 3)
    mat[0, 0, 1] = mat[0, 1, 2] = 1.0
    n_mat[0, 0, 1], n_mat[0, 1, 2] = 7.0, 3.0
    mat[1, 2, 0] = 1.0
    n_mat[1, 2, 0] = 99.0
    scores = torch.zeros(2, 4, 3)
    expected = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
    assert float(compute_kl_loss_batch(scores, mat, n_mat)) == pytest.approx(expected)
    _, n_ex = kl_to_empirical(scores, mat, n_mat)
    assert int(n_ex) == 1


def test_kl_weights_a_huge_example_no_more_than_a_tiny_one():
    """The point of normalizing: descendant counts set the split, not the example's weight."""
    mat = torch.zeros(2, 4, 3)
    n_mat = torch.zeros(2, 4, 3)
    mat[0, 0, 1] = mat[0, 1, 2] = 1.0
    n_mat[0, 0, 1], n_mat[0, 1, 2] = 70000.0, 30000.0   # same 0.7/0.3 split, vast
    mat[1, 0, 1] = mat[1, 1, 2] = 1.0
    n_mat[1, 0, 1], n_mat[1, 1, 2] = 7.0, 3.0           # ...and tiny
    scores = torch.zeros(2, 4, 3)
    both = float(compute_kl_loss_batch(scores, mat, n_mat))
    one = float(compute_kl_loss_batch(scores[:1], mat[:1], n_mat[:1]))
    assert both == pytest.approx(one)


# ── The count transform ───────────────────────────────────────────────────────

from antiGen.train import COUNT_TRANSFORMS, MutationDataset  # noqa: E402
from antiGen.load import Example  # noqa: E402


def _tiny_example(counts, seq_len=4, E=3, is_training=True):
    mat = torch.zeros(seq_len, E)
    n_mat = torch.zeros(seq_len, E)
    for (pos, alt), n in counts.items():
        mat[pos, alt] = 1.0
        n_mat[pos, alt] = float(n)
    return Example(aa_idx=torch.zeros(seq_len, dtype=torch.long), mat=mat,
                   is_training=is_training, n_mat=n_mat)


def test_the_transform_compresses_the_weights_it_does_not_reorder_them():
    ex = _tiny_example({(0, 1): 100, (1, 2): 1})
    for name in ("log1p", "sqrt"):
        ds = MutationDataset([ex], name)
        _, _, n, _ = ds[0]
        big, small = float(n[0, 1]), float(n[1, 2])
        assert big > small                       # order preserved
        assert big / small < 100.0               # ...but the ratio is compressed


def test_no_transform_leaves_the_counts_alone():
    ex = _tiny_example({(0, 1): 100, (1, 2): 1})
    _, _, n, _ = MutationDataset([ex], "none")[0]
    assert float(n[0, 1]) == 100.0 and float(n[1, 2]) == 1.0


def test_log1p_of_a_single_descendant_stays_positive():
    """log(1 + 1) > 0, so a mutation with one descendant still carries weight."""
    _, _, n, _ = MutationDataset([_tiny_example({(0, 1): 1})], "log1p")[0]
    assert float(n[0, 1]) == pytest.approx(math.log(2))


def test_a_pre_count_example_still_loads_under_a_transform():
    ex = Example(aa_idx=torch.zeros(4, dtype=torch.long), mat=torch.zeros(4, 3),
                 is_training=True, n_mat=None)
    _, _, n, _ = MutationDataset([ex], "log1p")[0]
    assert float(n.sum()) == 0.0


def test_transformed_counts_change_which_split_the_loss_prefers():
    """The whole point: under log1p a 100:1 split is no longer worth 100 times as much."""
    ex = _tiny_example({(0, 1): 100, (1, 2): 1})
    mat = ex.mat.unsqueeze(0)
    raw = ex.n_mat.unsqueeze(0)
    logged = COUNT_TRANSFORMS["log1p"](ex.n_mat).unsqueeze(0)
    # a prediction that matches the raw split beats one matching the compressed split under
    # raw weighting, and the ordering flips once the counts are compressed
    fit_raw, fit_log = torch.zeros(1, 4, 3), torch.zeros(1, 4, 3)
    fit_raw[0, 0, 1] = math.log(100.0)
    fit_log[0, 0, 1] = math.log(math.log(101.0) / math.log(2.0))
    assert (float(compute_conditional_loss_batch(fit_raw, mat, raw))
            < float(compute_conditional_loss_batch(fit_log, mat, raw)))
    assert (float(compute_conditional_loss_batch(fit_log, mat, logged))
            < float(compute_conditional_loss_batch(fit_raw, mat, logged)))


# ── The three readings of top-1 ───────────────────────────────────────────────

from antiGen.eval_counts import Accumulator, _ci, _paired_ci  # noqa: E402


def _feed(acc, scores, counts):
    """One example: `scores` are log-probs at the occupied cells, `counts` the descendants."""
    acc.add(torch.tensor(scores, dtype=torch.float32),
            torch.tensor(counts, dtype=torch.float32), len(counts))


def test_a_flat_scorer_is_scored_as_the_coin_flip_it_is():
    """Uniform must not be credited with whichever cell happens to come first."""
    acc = Accumulator()
    _feed(acc, [0.0, 0.0, 0.0], [90.0, 5.0, 5.0])
    s = acc.summary()["k>=2"]
    assert s["top1_pick"] == pytest.approx(1 / 3)          # one of three is the true max
    assert s["top1_share"] == pytest.approx(100 / 3 / 100)  # expected descendants / total


def test_the_three_readings_come_apart_on_one_big_and_one_small_example():
    """A model right on a huge example and wrong on many small ones."""
    acc = Accumulator()
    _feed(acc, [1.0, 0.0], [10000.0, 1.0])     # right, and enormous
    for _ in range(9):
        _feed(acc, [0.0, 1.0], [10.0, 1.0])    # wrong, and small
    s = acc.summary()["k>=2"]
    assert s["top1_pick"] == pytest.approx(0.1)            # 1 of 10 examples
    assert s["top1_acc"] > 0.99                            # but ~all descendants
    assert s["top1_share"] < 0.2                           # per-example share is low too


def test_descendant_weighting_is_what_the_headline_number_uses():
    acc = Accumulator()
    _feed(acc, [1.0, 0.0], [70.0, 30.0])
    _feed(acc, [0.0, 1.0], [5.0, 1.0])
    s = acc.summary()["k>=2"]
    assert s["top1_acc"] == pytest.approx((70.0 + 1.0) / 106.0)
    assert s["top1_share"] == pytest.approx((70 / 100 + 1 / 6) / 2)
    assert s["top1_pick"] == pytest.approx(0.5)


def test_single_child_examples_are_excluded_from_the_k2_subset():
    acc = Accumulator()
    _feed(acc, [0.0], [42.0])
    _feed(acc, [1.0, 0.0], [7.0, 3.0])
    assert acc.summary()["k>=2"]["n_examples"] == 1
    assert acc.summary()["all"]["n_examples"] == 2


def test_confidence_intervals_bracket_the_point_estimate_and_are_reproducible():
    values = [float(v) for v in range(20)]
    point, lo, hi = _ci(values)
    assert lo < point < hi
    assert (point, lo, hi) == _ci(values)      # seeded, so repeat calls agree


def test_a_narrow_interval_needs_many_examples_not_many_descendants():
    """One vast example gives a useless CI; many small ones give a usable one."""
    one = Accumulator()
    _feed(one, [1.0, 0.0], [100000.0, 1.0])
    lo, hi = one.summary()["k>=2"]["top1_pick_ci"]
    assert hi - lo == pytest.approx(0.0)       # a single example: the bootstrap cannot move

    many = Accumulator()
    for i in range(200):
        _feed(many, [1.0, 0.0] if i % 2 else [0.0, 1.0], [7.0, 3.0])
    lo, hi = many.summary()["k>=2"]["top1_pick_ci"]
    assert 0.0 < hi - lo < 0.2


def test_stratifying_by_k_partitions_the_examples():
    acc = Accumulator()
    _feed(acc, [0.0, 1.0], [3.0, 1.0])
    _feed(acc, [0.0, 1.0, 0.0], [3.0, 1.0, 1.0])
    _feed(acc, [0.0] * 6, [1.0] * 6)
    _feed(acc, [0.0] * 12, [1.0] * 12)
    by_k = acc.summary()["by_k"]
    assert {lab: b["n_examples"] for lab, b in by_k.items()} == {
        "k=2": 1, "k=3-4": 1, "k=5-9": 1, "k>=10": 1}


def test_pairing_finds_a_consistent_small_edge_that_unpaired_intervals_miss():
    """Genotype difficulty varies hugely; the model wins on every one by a little."""
    rng = torch.Generator().manual_seed(0)
    base, model = [], []
    for _ in range(300):
        difficulty = float(torch.rand(1, generator=rng))   # shared, and dominant
        base.append(difficulty)
        model.append(difficulty + 0.02)                    # a small consistent edge
    lo_u, hi_u = _ci(model)[1:], _ci(base)[1:]
    assert lo_u[0] < hi_u[1]                               # unpaired intervals overlap
    d, lo, hi, p = _paired_ci(model, base)
    assert d == pytest.approx(0.02)
    assert lo > 0 and p == 0.0                             # paired, the edge is unambiguous


def test_pairing_reports_no_edge_when_there_is_none():
    rng = torch.Generator().manual_seed(1)
    base = [float(v) for v in torch.rand(300, generator=rng)]
    d, lo, hi, p = _paired_ci(base, base)
    assert d == 0.0 and lo == 0.0 and hi == 0.0 and p == 1.0


from antiGen.eval_counts import _paired_ratio_ci, effective_n  # noqa: E402


def test_descendant_weighted_pairing_tracks_the_big_genotypes():
    """One huge genotype the model wins, many small ones it loses: the weighted delta is positive."""
    big_a, big_b, w = [9000.0], [0.0], [10000.0]
    small_a, small_b, small_w = [0.0] * 50, [2.0] * 50, [4.0] * 50
    d, lo, hi, p = _paired_ratio_ci(big_a + small_a, big_b + small_b, w + small_w)
    assert d == pytest.approx((9000.0 - 100.0) / (10000.0 + 200.0))
    assert d > 0                     # the one big genotype carries it, as intended
    assert hi > lo


def test_effective_n_falls_as_the_weights_concentrate():
    assert effective_n([1.0] * 100) == pytest.approx(100.0)
    assert effective_n([1000.0] + [1.0] * 99) < 2.0
    assert 1.0 < effective_n([10.0] + [1.0] * 99) < 100.0


def test_effective_n_of_the_real_shape_is_tiny():
    """A handful of vast genotypes among thousands of small ones."""
    weights = [100000.0, 50000.0, 20000.0] + [3.0] * 3000
    assert effective_n(weights) < 10.0


# ── Interval-mode example files ───────────────────────────────────────────────

_IV_HEAD = ("EXAMPLE_ID: {id}\nTIME_INTERVAL: {w}\nTIME_RANGE: {lo},{hi}\n"
            "PARENT_EXAMPLE_ID: {parent}\nMUTATION_FROM_PARENT:\n{mut}\nTRAINING\n"
            "IS_DESCENDED_FROM_VARIANT:\nJ,False\nIS_VARIANT_ROOT:\nJ,False\nMTEY\n"
            "MUTATIONS_TO_CHILDREN:\n{rows}\n\n")


def _interval_file(tmp_path, body):
    path = tmp_path / "examples_intervals.txt"
    path.write_text(body)
    return str(path)


def test_loader_reads_the_window_and_both_counts(tmp_path):
    body = _IV_HEAD.format(id=0, w=2019, lo=2019.5, hi=2020.5, parent="None", mut="None",
                           rows="0,L,3,17\n2,K,0,4")
    ex, = load_examples(_interval_file(tmp_path, body), ALPHABET, "J")
    L, K = ALPHABET.index("L"), ALPHABET.index("K")
    assert ex.time_interval == 2019 and ex.time_range == (2019.5, 2020.5)
    # n_mat is always the count the file is about -- here, the next window's samples
    assert ex.n_mat[0, L] == 3.0 and ex.n_mat[2, K] == 0.0
    assert ex.total_mat[0, L] == 17.0 and ex.total_mat[2, K] == 4.0
    # a substitution with no descendants in the next window is still a listed candidate
    assert ex.mat[2, K] == 1.0


def test_the_same_genotype_appears_once_per_window(tmp_path):
    body = (_IV_HEAD.format(id=0, w=2019, lo=2019.5, hi=2020.5, parent="None", mut="None",
                            rows="0,L,3,17")
            + _IV_HEAD.format(id=1, w=2020, lo=2020.5, hi=2021.5, parent="None", mut="None",
                              rows="0,L,8,17"))
    a, b = load_examples(_interval_file(tmp_path, body), ALPHABET, "J")
    L = ALPHABET.index("L")
    assert (a.time_interval, b.time_interval) == (2019, 2020)
    assert a.aa_idx.tolist() == b.aa_idx.tolist()      # same genotype
    assert a.n_mat[0, L] == 3.0 and b.n_mat[0, L] == 8.0   # different targets
    assert a.total_mat[0, L] == b.total_mat[0, L]      # same all-time total


def test_cutoff_files_still_parse_and_carry_no_window(tmp_path):
    body = ("EXAMPLE_ID: 0\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nTRAINING\n"
            "IS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEY\n"
            "MUTATIONS_TO_CHILDREN:\n0,L,3\n\n")
    path = tmp_path / "examples_cutoff_1.txt"
    path.write_text(body)
    ex, = load_examples(str(path), ALPHABET, None)
    assert ex.time_interval is None and ex.time_range is None
    assert ex.total_mat is None                         # no fourth field to read
    assert ex.n_mat[0, ALPHABET.index("L")] == 3.0


# ── Splitting an interval file on its windows ─────────────────────────────────

def _three_windows(tmp_path):
    body = "".join(
        _IV_HEAD.format(id=i, w=w, lo=w + 0.5, hi=w + 1.5, parent="None", mut="None",
                        rows="0,L,3,17")
        for i, w in enumerate([2021, 2022, 2022, 2023, 2024])
    )
    return load_examples(_interval_file(tmp_path, body), ALPHABET, "J")


def test_the_test_window_is_held_out_and_later_windows_are_dropped(tmp_path):
    train, test, dropped = split_by_window(_three_windows(tmp_path), 2023)
    assert [e.time_interval for e in train] == [2021, 2022, 2022]
    assert [e.time_interval for e in test] == [2023]
    # The most recent window is counted against a partly observed year, so it is neither
    # trained on nor tested on.
    assert [e.time_interval for e in dropped] == [2024]


def test_splitting_the_last_window_drops_nothing(tmp_path):
    train, test, dropped = split_by_window(_three_windows(tmp_path), 2024)
    assert len(train) == 4 and len(test) == 1 and dropped == []


def test_an_empty_test_window_is_an_error_not_an_empty_split(tmp_path):
    with pytest.raises(ValueError, match="No examples in window 2025"):
        split_by_window(_three_windows(tmp_path), 2025)


def test_a_cutoff_file_has_no_windows_to_split_on(tmp_path):
    body = ("EXAMPLE_ID: 0\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nTRAINING\n"
            "IS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEY\n"
            "MUTATIONS_TO_CHILDREN:\n0,L,3\n\n")
    path = tmp_path / "examples_cutoff_1.txt"
    path.write_text(body)
    with pytest.raises(ValueError, match="TIME_INTERVAL"):
        split_by_window(load_examples(str(path), ALPHABET, None), 2023)


# ── Zero-descendant substitutions, which only interval files produce ──────────

def test_the_oracle_survives_a_substitution_with_no_descendants():
    """0 * log 0 = 0. The oracle puts log 0 on an unused cell, and 0 * -inf is NaN."""
    acc = Accumulator()
    n = torch.tensor([7.0, 0.0, 3.0])
    acc.add(torch.log(n / n.sum()), n, 3)
    row, = acc.rows
    assert math.isfinite(row["nll_c"]) and math.isfinite(row["nll_u"])
    # the oracle's loss per descendant is exactly the entropy of the observed split
    assert row["nll_c"] / row["n"] == pytest.approx(row["entropy"], abs=1e-6)


def test_an_example_with_no_descendants_at_all_is_scored_by_nobody():
    acc = Accumulator()
    acc.add(torch.tensor([-1.0, -2.0]), torch.tensor([0.0, 0.0]), 2)
    assert acc.rows == []


# ── The persistence baseline ──────────────────────────────────────────────────

_IV5 = ("EXAMPLE_ID: 0\nTIME_INTERVAL: 2019\nTIME_RANGE: 2019.5,2020.5\n"
        "PARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nTRAINING\n"
        "IS_DESCENDED_FROM_VARIANT:\nJ,False\nIS_VARIANT_ROOT:\nJ,False\nMTEY\n"
        "MUTATIONS_TO_CHILDREN:\n{rows}\n\n")


def _one(tmp_path, rows):
    path = tmp_path / "examples_intervals.txt"
    path.write_text(_IV5.format(rows=rows))
    ex, = load_examples(str(path), ALPHABET, "J")
    return ex


def test_loader_reads_the_own_window_count(tmp_path):
    ex = _one(tmp_path, "0,L,3,17,5\n2,K,0,4,2")
    L, K = ALPHABET.index("L"), ALPHABET.index("K")
    assert ex.n_mat[0, L] == 3.0 and ex.cur_mat[0, L] == 5.0   # target vs this year
    assert ex.n_mat[2, K] == 0.0 and ex.cur_mat[2, K] == 2.0
    assert ex.total_mat[0, L] == 17.0


def test_a_four_field_file_has_no_own_window_counts(tmp_path):
    ex = _one(tmp_path, "0,L,3,17")
    assert ex.cur_mat is None and ex.total_mat is not None


def test_persistence_predicts_this_years_split_smoothed(tmp_path):
    ex = _one(tmp_path, "0,L,3,17,9\n2,K,1,4,1")
    acc = score_persistence([ex], alpha=1.0)
    row, = acc.rows
    # q = (9+1, 1+1)/12 = (10/12, 2/12), scored against n = (3, 1)
    want = -(3 * math.log(10 / 12) + 1 * math.log(2 / 12))
    assert row["nll_c"] == pytest.approx(want, rel=1e-6)


def test_the_pseudocount_is_what_keeps_a_new_substitution_finite(tmp_path):
    """c = 0 next year's winner. Without smoothing this is an infinite loss."""
    ex = _one(tmp_path, "0,L,7,17,0\n2,K,1,4,50")
    finite = score_persistence([ex], alpha=0.5).rows[0]["nll_c"]
    assert math.isfinite(finite)
    assert not math.isfinite(score_persistence([ex], alpha=0.0).rows[0]["nll_c"])


def test_a_huge_pseudocount_collapses_persistence_to_uniform(tmp_path):
    ex = _one(tmp_path, "0,L,3,17,9\n2,K,1,4,1")
    big = score_persistence([ex], alpha=1e9).rows[0]["nll_c"]
    uni = score_reference([ex], "uniform", n_cells=84).rows[0]["nll_c"]
    assert big == pytest.approx(uni, rel=1e-5)


def test_persistence_beats_uniform_when_the_split_persists(tmp_path):
    """The whole point: if this year's split predicts next year's, this baseline is hard."""
    ex = _one(tmp_path, "0,L,90,100,900\n2,K,10,20,100")
    per = score_persistence([ex], alpha=1.0).rows[0]
    uni = score_reference([ex], "uniform", n_cells=84).rows[0]
    assert per["nll_c"] / per["n"] < uni["nll_c"] / uni["n"]


# ── Late fusion: persistence counts + learned sequence residual ───────────────

from antiGen.config import ModelConfig, ModelType, ResidualConfig, TransformerConfig  # noqa: E402
from antiGen.models.factory import create_model  # noqa: E402
from antiGen.eval_utils import conditional_log_probs  # noqa: E402


def _residual(seq_len=6, alpha=1.0, beta_init=1.0, lambda_init=0.0):
    cfg = ModelConfig(
        model_type=ModelType.RESIDUAL_FITNESS_PREDICTOR, seq_len=seq_len,
        transformer=TransformerConfig(d_model=16, nhead=4, dim_feedforward=32, num_layers=1),
        residual=ResidualConfig(alpha=alpha, beta_init=beta_init, lambda_init=lambda_init),
    )
    # eval(), so dropout does not make two forward passes disagree -- these tests compare
    # against closed forms.
    return create_model(cfg, "cpu").eval()


def _cur(seq_len=6, cells=((0, 1, 9.0), (2, 3, 1.0))):
    c = torch.zeros(1, seq_len, len(ALPHABET))
    for pos, alt, v in cells:
        c[0, pos, alt] = v
    return c


def test_at_lambda_zero_the_model_is_exactly_the_persistence_baseline():
    """The property that makes lambda_init = 0 meaningful: training starts AT the baseline."""
    model = _residual(lambda_init=0.0)
    aa, cur = torch.zeros(1, 6, dtype=torch.long), _cur()
    mat = (cur > 0).float()
    got, _ = conditional_log_probs(model(aa, cur), mat)
    want, _ = conditional_log_probs(torch.log(cur + 1.0), mat)
    assert torch.allclose(got[mat.reshape(1, -1) > 0], want[mat.reshape(1, -1) > 0], atol=1e-6)


def test_a_nonzero_lambda_moves_the_prediction_off_the_baseline():
    model = _residual(lambda_init=0.5)
    aa, cur = torch.zeros(1, 6, dtype=torch.long), _cur()
    mat = (cur > 0).float()
    sel = mat.reshape(1, -1) > 0
    off, _ = conditional_log_probs(model(aa, cur), mat)
    base, _ = conditional_log_probs(torch.log(cur + 1.0), mat)
    assert not torch.allclose(off[sel], base[sel], atol=1e-6)


def test_beta_scales_how_hard_the_prevalence_term_pushes():
    """beta < 1 flattens this window's split; beta > 1 sharpens it."""
    aa, cur = torch.zeros(1, 6, dtype=torch.long), _cur()
    mat = (cur > 0).float()
    sel = mat.reshape(1, -1) > 0

    def gap(beta):
        lp, _ = conditional_log_probs(_residual(beta_init=beta)(aa, cur), mat)
        v = lp[sel]
        return float(v.max() - v.min())

    assert gap(0.5) < gap(1.0) < gap(2.0)


def test_with_no_counts_it_degrades_to_the_sequence_only_predictor():
    """Scoring a de novo sequence still works: the prevalence term cancels in the softmax."""
    model = _residual(lambda_init=0.7)
    aa = torch.zeros(1, 6, dtype=torch.long)
    assert torch.allclose(model(aa), 0.7 * model.inner(aa), atol=1e-6)


def test_a_nonpositive_pseudocount_is_rejected_at_construction():
    with pytest.raises(ValueError, match="alpha must be positive"):
        _residual(alpha=0.0)


def test_the_config_survives_a_checkpoint_roundtrip():
    cfg = ModelConfig(model_type=ModelType.RESIDUAL_FITNESS_PREDICTOR, seq_len=6,
                      residual=ResidualConfig(alpha=0.3, beta_init=0.8, lambda_init=0.2,
                                              learn_beta=False, learn_lambda=True))
    assert ModelConfig.from_dict(cfg.to_dict()).residual == cfg.residual


def test_freezing_leaves_only_the_trainable_scalar():
    cfg = ModelConfig(
        model_type=ModelType.RESIDUAL_FITNESS_PREDICTOR, seq_len=6,
        transformer=TransformerConfig(d_model=16, nhead=4, dim_feedforward=32, num_layers=1),
        residual=ResidualConfig(learn_beta=True, learn_lambda=False),
    )
    model = create_model(cfg, "cpu")
    assert model.beta.requires_grad and not model.lam.requires_grad


def test_the_dataset_yields_the_own_window_counts_untransformed(tmp_path):
    """count_transform reweights the TARGET; cur_mat is a model input and must not move."""
    ex = _one(tmp_path, "0,L,3,17,100\n2,K,1,4,1")
    _, _, n, cur = MutationDataset([ex], "log1p")[0]
    L = ALPHABET.index("L")
    assert float(cur[0, L]) == 100.0                      # input, untouched
    assert float(n[0, L]) == pytest.approx(math.log1p(3))  # target, transformed


# ── Beyond-the-baseline diagnostics ──────────────────────────────────────────

from antiGen.eval_beyond import pair_stats, fit_beta_lambda, _nll  # noqa: E402


def _row(f, c, n):
    return (np.array(f, dtype=float), np.array(c, dtype=float), np.array(n, dtype=float))


def test_pair_classes_split_on_what_the_baseline_can_see():
    # j=0 vs j=1: c 5>1 and n 9>1  -> concordant
    # j=0 vs j=2: c 5>1 and n 9>20 -> discordant (2 overtook 0)
    # j=1 vs j=2: c 1==1, n 1!=20  -> c-tied
    stats, = pair_stats([_row([0, 0, 0], [5, 1, 1], [9, 1, 20])])
    assert stats["concordant"][1] == 1
    assert stats["discordant"][1] == 1
    assert stats["c-tied"][1] == 1


def test_a_pair_with_no_outcome_ordering_is_dropped():
    """n_j == n_l gives no ground truth, so the pair cannot score anyone."""
    stats, = pair_stats([_row([1, 0], [5, 1], [7, 7])])
    assert sum(stats[c][1] for c in stats) == 0


def test_a_flat_model_scores_exactly_chance_not_tensor_order():
    stats, = pair_stats([_row([0, 0, 0], [5, 1, 1], [9, 1, 20])])
    for cls in ("concordant", "discordant", "c-tied"):
        assert stats[cls][0] == pytest.approx(0.5 * stats[cls][1])


def test_a_model_that_knows_the_overtake_scores_one_on_discordant():
    # ranks the eventual winner (index 2) top, against this year's leader (index 0)
    stats, = pair_stats([_row([0.0, 0.0, 5.0], [5, 1, 1], [9, 1, 20])])
    assert stats["discordant"][0] == pytest.approx(1.0)   # called the overtake
    assert stats["c-tied"][0] == pytest.approx(1.0)       # ...and where the baseline is blind
    # the concordant pair is a tie in the model's scores, so it scores the coin flip
    assert stats["concordant"][0] == pytest.approx(0.5)


def test_the_descendant_weighted_tally_tracks_the_bigger_pairs():
    stats, = pair_stats([_row([0.0, 5.0], [5, 1], [1, 1000])])
    assert stats["discordant"][3] == pytest.approx(1001.0)


def test_fitting_lambda_recovers_a_signal_the_counts_cannot_express():
    """Rows where the model score, not the count, determines who wins."""
    rows = [_row([0.0, 3.0], [10, 10], [1, 99]) for _ in range(40)]
    _, lam = fit_beta_lambda(rows, alpha=0.3, learn_lambda=True)
    assert lam > 0.3


def test_fitting_lambda_leaves_it_near_zero_when_the_model_is_noise():
    rng = np.random.default_rng(0)
    rows = [_row(rng.normal(size=2), [9, 1], [9, 1]) for _ in range(60)]
    _, lam = fit_beta_lambda(rows, alpha=0.3, learn_lambda=True)
    assert abs(lam) < 0.3


def test_the_two_parameter_loss_matches_the_conditional_loss_it_stands_in_for():
    rows = [_row([0.5, -0.2, 1.0], [4, 2, 0], [3, 1, 5])]
    total, obs = _nll(rows, beta=1.0, lam=0.0, alpha=1.0)
    f, c, n = rows[0]
    s = np.log(c + 1.0)
    want = -(n * (s - np.log(np.exp(s).sum()))).sum()
    assert total == pytest.approx(want, rel=1e-9) and obs == 9.0


# ── Test C: the DPO reduction ────────────────────────────────────────────────

from antiGen.eval_beyond import (build_pairs, _flatten, fit_pair_logistic,  # noqa: E402
                                 pair_logloss, pair_accuracy)


def test_pairs_are_oriented_winner_first_and_classified_by_the_baseline():
    # counts: n = [9, 1, 20], c = [5, 1, 1]
    pairs = build_pairs([_row([0, 0, 0], [5, 1, 1], [9, 1, 20])], alpha=1.0,
                        min_count=1, min_ratio=1.0)
    _, _, cls, _, _ = pairs[0]
    assert sorted(cls) == ["c-tied", "concordant", "discordant"]


def test_the_min_count_filter_removes_the_sampling_noise_pairs():
    """1-vs-0 is not a preference; it is one sequence."""
    rows = [_row([0, 1], [3, 1], [1, 0])]
    assert build_pairs(rows, alpha=1.0, min_count=5, min_ratio=2.0) == []
    assert len(build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)) == 1


def test_the_min_ratio_filter_removes_pairs_that_are_too_close():
    rows = [_row([0, 1], [3, 1], [11, 10])]
    assert build_pairs(rows, alpha=1.0, min_count=5, min_ratio=2.0) == []
    assert len(build_pairs(rows, alpha=1.0, min_count=5, min_ratio=1.0)) == 1


def test_each_example_carries_total_weight_one_whatever_its_k():
    small = build_pairs([_row([0, 1], [1, 9], [20, 5])], 1.0, 1, 1.0)
    big = build_pairs([_row(list(range(6)), [1] * 6, [60, 50, 40, 30, 20, 10])], 1.0, 1, 1.0)
    for pairs in (small, big):
        _, _, _, w, _ = _flatten(pairs)
        assert w.sum() == pytest.approx(1.0)


def test_selecting_informative_pairs_drops_the_concordant_ones():
    pairs = build_pairs([_row([0, 0, 0], [5, 1, 1], [9, 1, 20])], 1.0, 1, 1.0)
    _, _, cls, _, _ = _flatten(pairs, ("discordant", "c-tied"))
    assert "concordant" not in set(cls)


def test_the_fit_finds_a_positive_slope_when_the_score_predicts_the_winner():
    rows = [_row([0.0, 2.0], [4, 4], [3, 30]) for _ in range(30)]
    pairs = build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)
    dc, df, _, w, _ = _flatten(pairs)
    _, lam = fit_pair_logistic(dc, df, w, use_df=True)
    assert lam > 0.5


def test_the_fit_leaves_the_slope_near_zero_on_a_noise_feature():
    rng = np.random.default_rng(1)
    rows = [_row(rng.normal(size=2), [9, 1], [30, 3]) for _ in range(80)]
    pairs = build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)
    dc, df, _, w, _ = _flatten(pairs)
    _, lam = fit_pair_logistic(dc, df, w, use_df=True)
    assert abs(lam) < 0.4


def test_a_flat_scorer_gets_exactly_chance_on_the_pairs_it_cannot_order():
    pairs = build_pairs([_row([0, 0], [4, 4], [30, 3])], 1.0, 1, 1.0)
    dc, df, cls, w, _ = _flatten(pairs)
    # gamma = 0 too, so z == 0 everywhere: the tie must score 0.5, not 0 or 1
    acc = pair_accuracy(dc, df, cls, w, gamma=0.0, lam=0.0)
    assert acc["c-tied"][0] == pytest.approx(0.5)
    assert pair_logloss(dc, df, w, 0.0, 0.0) == pytest.approx(math.log(2), rel=1e-9)


def test_ratio_weighting_favours_the_lopsided_pair_within_an_example():
    rows = [_row([0, 1, 2], [1, 1, 1], [100, 1, 50])]
    pairs = build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)
    _, _, _, w_eq, _ = _flatten(pairs, None, "equal")
    _, _, _, w_lr, _ = _flatten(pairs, None, "logratio")
    # the 100-vs-1 pair should gain weight relative to equal, and every scheme still
    # totals 1 for the example
    assert w_lr.max() > w_eq.max()
    assert w_eq.sum() == pytest.approx(1.0) and w_lr.sum() == pytest.approx(1.0)


def test_linear_ratio_concentrates_harder_than_log_ratio():
    rows = [_row([0, 1, 2], [1, 1, 1], [1000, 1, 3])]
    pairs = build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)
    _, _, _, w_lr, _ = _flatten(pairs, None, "logratio")
    _, _, _, w_ra, _ = _flatten(pairs, None, "ratio")
    assert w_ra.max() > w_lr.max()


def test_weighting_never_shifts_weight_between_examples():
    """Each genotype keeps total weight 1 under every scheme -- that is the whole point."""
    rows = [_row([0, 1], [1, 1], [100, 1]),
            _row(list(range(5)), [1] * 5, [50, 40, 30, 20, 10])]
    pairs = build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)
    for sch in ("equal", "logratio", "ratio"):
        _, _, _, w, idx = _flatten(pairs, None, sch)
        for e in (0, 1):
            assert w[idx == e].sum() == pytest.approx(1.0)


from antiGen.eval_beyond import _subset  # noqa: E402


def test_restricting_to_a_class_renormalizes_each_example_back_to_one():
    rows = [_row([0, 1, 2], [5, 1, 1], [9, 1, 20]),
            _row([0, 1], [1, 9], [30, 3])]
    pairs = build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)
    dc, df, cls, w, idx = _flatten(pairs)
    m = cls == "discordant"
    _, _, sw, sidx = _subset(dc, df, w, idx, m)
    for e in np.unique(sidx):
        assert sw[sidx == e].sum() == pytest.approx(1.0)


def test_fitting_on_a_class_would_read_the_outcome_which_is_why_we_do_not():
    """A discordant pair has Δc < 0 by construction, so a fit on that class alone drives
    gamma negative and 'predicts' perfectly. This pins the artifact the code avoids."""
    rows = [_row([0.0, 0.0], [10, 1], [1, 50]) for _ in range(30)]   # all discordant
    pairs = build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)
    dc, df, cls, w, idx = _flatten(pairs)
    assert set(cls) == {"discordant"}
    gamma, _ = fit_pair_logistic(dc, df, w, use_df=False)
    assert gamma < -1.0            # the degenerate fit, reproduced deliberately


def test_a_fit_on_all_pairs_keeps_gamma_positive():
    """The population the code actually fits on: prevalence is predictive overall."""
    rows = ([_row([0.0, 0.0], [10, 1], [50, 1]) for _ in range(30)]      # concordant
            + [_row([0.0, 0.0], [10, 1], [1, 50]) for _ in range(5)])    # a few discordant
    pairs = build_pairs(rows, alpha=1.0, min_count=1, min_ratio=1.0)
    dc, df, cls, w, _ = _flatten(pairs)
    gamma, _ = fit_pair_logistic(dc, df, w, use_df=False)
    assert gamma > 0.0
