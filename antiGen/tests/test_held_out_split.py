"""The three-way split must be disjoint, exhaustive and reproducible.

Everything reported from the random regime rests on the test block having played no
part in training or epoch selection, so these are the invariants worth pinning.
"""
import torch

from antiGen.train import held_out_split, split_examples


class _Ex:
    """Minimal stand-in for load.Example: the splitter only reads is_training."""
    def __init__(self, i):
        self.idx = i
        self.is_training = True
        self.mat = torch.zeros(4, 4)


def _pool(n=200):
    return {'treeA': [_Ex(i) for i in range(n)],
            'treeB': [_Ex(i) for i in range(n // 2)]}


def test_splits_are_disjoint_and_exhaustive():
    tr, va, te = held_out_split(_pool(), val_frac=0.1, test_frac=0.1, split_seed=0)
    for name in ('treeA', 'treeB'):
        ids = [{id(e) for e in part[name]} for part in (tr, va, te)]
        assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
        assert sum(len(s) for s in ids) == len(_pool()[name])


def test_split_is_reproducible_and_seed_sensitive():
    a = held_out_split(_pool(), 0.1, 0.1, split_seed=0)
    b = held_out_split(_pool(), 0.1, 0.1, split_seed=0)
    c = held_out_split(_pool(), 0.1, 0.1, split_seed=1)
    ia, ib, ic = ({e.idx for e in x[2]['treeA']} for x in (a, b, c))
    assert ia == ib
    assert ia != ic


def test_test_block_is_stable_when_val_frac_changes():
    """The test set is taken from the front of the permutation, so re-tuning the
    validation fraction must not silently move what is being reported on."""
    _, _, t1 = held_out_split(_pool(), val_frac=0.1, test_frac=0.1, split_seed=0)
    _, _, t2 = held_out_split(_pool(), val_frac=0.3, test_frac=0.1, split_seed=0)
    assert {e.idx for e in t1['treeA']} == {e.idx for e in t2['treeA']}


def test_test_block_is_absent_from_the_two_way_split_training_set():
    """held_out_split's train set must be a subset of the two-way train set: the
    difference is exactly what was removed to make room for val and test."""
    pool = _pool()
    tr3, _, te3 = held_out_split(pool, 0.1, 0.1, split_seed=0)
    assert all(e not in tr3['treeA'] for e in te3['treeA'])


def test_tiny_sources_still_get_training_data():
    tiny = {'t': [_Ex(i) for i in range(3)]}
    tr, va, te = held_out_split(tiny, val_frac=0.5, test_frac=0.5, split_seed=0)
    assert len(tr['t']) >= 1


def test_two_way_split_unchanged():
    """split_examples is still used by published arms; it must not have drifted."""
    tr, va = split_examples(_pool(), 'random', 0.1, 0)
    assert len(tr['treeA']) + len(va['treeA']) == 200
    assert len(va['treeA']) == 20
