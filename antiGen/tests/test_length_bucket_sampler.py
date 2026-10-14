"""Tests for the length-bucketed batch sampler used by multi-source training.

Core dependencies only (torch); no model downloads.
"""
import pytest

from antiGen.batching import LengthBucketBatchSampler


LENGTHS = [566] * 7 + [1273] * 5 + [566] * 3   # 15 items, two lengths, interleaved


def test_batches_are_length_homogeneous():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=4, seed=0)
    for batch in sampler:
        assert len({LENGTHS[i] for i in batch}) == 1, batch


def test_every_example_appears_exactly_once():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=4, seed=0)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(len(LENGTHS)))


def test_len_matches_realized_batch_count():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=4, seed=0)
    assert len(sampler) == len(list(sampler))
    # 10 of length 566 -> 3 batches, 5 of length 1273 -> 2 batches
    assert len(sampler) == 5


def test_set_epoch_changes_the_order():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=4, seed=0)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    sampler.set_epoch(0)
    assert list(sampler) == first        # and it is reproducible


def test_no_shuffle_is_deterministic_and_ordered():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=4, shuffle=False, seed=0)
    batches = list(sampler)
    assert batches == list(sampler)
    assert batches[0] == [0, 1, 2, 3]    # first four length-566 items, in order


def test_max_batch_tokens_shrinks_the_long_bucket():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=8, seed=0,
                                       max_batch_tokens=4096)
    for batch in sampler:
        length = LENGTHS[batch[0]]
        assert len(batch) <= max(1, 4096 // length)
    # 4096 // 566 = 7 (capped by batch_size 8), 4096 // 1273 = 3
    sizes_by_length = {}
    for batch in sampler:
        sizes_by_length.setdefault(LENGTHS[batch[0]], []).append(len(batch))
    assert max(sizes_by_length[1273]) == 3


def test_drop_last_drops_partial_batches():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=4, drop_last=True, seed=0)
    for batch in sampler:
        assert len(batch) == 4


def test_bucket_sizes():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=4, seed=0)
    assert sampler.bucket_sizes() == {566: 10, 1273: 5}


@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_various_batch_sizes_cover_the_dataset(batch_size):
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=batch_size, seed=0)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(len(LENGTHS)))
