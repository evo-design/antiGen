"""A cached score map replays through the metric to the same numbers as the live pass.

This is the Stage-2 exit check of the ensembling arm in miniature. The real one compares
a replay of each cached map against the published ``list_macro_summary_*.csv`` row for
that model; this pins the same identity on synthetic data, so a regression in the dump
path is caught by ``pytest`` rather than by a 10-hour GPU job.

Two properties, and the second is the one that is easy to lose:

* the map written by ``cell_pass``'s ``sink`` reproduces the accumulator that the same
  call returned, cell for cell, through ``compute_hit_rate``;
* it does so with ``compute_hit_rate`` called UNMODIFIED. That is why the cache stores
  ``aa_idx`` and the target coordinates at full sequence length while storing the scores
  sliced to the eval range: that function slices the eval range itself and only
  re-slices ``scores`` when the shape tells it to, so the replay runs the identical code
  path the published numbers came from rather than a parallel one.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from antiGen import score_cache as sc
from antiGen.eval_utils import compute_hit_rate
from antiGen.list_macro import build_variants, cell_pass, summarize

ALLOWED = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALLOWED)
SEQ_LEN, N = 60, 23
EVAL_RANGE = list(range(20, 41))          # 21 positions
R = len(EVAL_RANGE)


class _DS(Dataset):
    def __init__(self, aa, mat):
        self.aa, self.mat = aa, mat

    def __len__(self):
        return self.aa.shape[0]

    def __getitem__(self, i):
        return self.aa[i], self.mat[i]


def _world(seed=0):
    g = torch.Generator().manual_seed(seed)
    aa = torch.randint(0, 20, (N, SEQ_LEN), generator=g)
    mat = torch.zeros(N, SEQ_LEN, E)
    for i in range(N):
        for _ in range(4):
            p = EVAL_RANGE[int(torch.randint(0, R, (1,), generator=g))]
            a = int(torch.randint(0, 20, (1,), generator=g))
            mat[i, p, a] += 1.0
    # A model-shaped score map: deterministic in the query, gap/stop not emitted,
    # one -inf sentinel, and a coarse block so there are real ties.
    scores = torch.randn(N, SEQ_LEN, E, generator=g)
    scores[:, :, ALLOWED.index('-')] = float('nan')
    scores[:, :, ALLOWED.index('*')] = float('nan')
    scores[3, EVAL_RANGE[2], 5] = float('-inf')
    scores[:, EVAL_RANGE[0]:EVAL_RANGE[0] + 3, :20] = \
        (scores[:, EVAL_RANGE[0]:EVAL_RANGE[0] + 3, :20] * 2).round() / 2
    return aa, mat, scores


def test_dumped_map_replays_to_the_same_metric(tmp_path):
    aa, mat, scores = _world()
    loader = DataLoader(_DS(aa, mat), batch_size=8, shuffle=False, drop_last=False)
    in_full = torch.zeros(SEQ_LEN, E, dtype=torch.bool)
    in_full[EVAL_RANGE[1], 3] = True
    in_full[EVAL_RANGE[4], 7] = True
    in_full[EVAL_RANGE[9], 11] = True
    seen = torch.zeros(SEQ_LEN, E, dtype=torch.bool)
    seen[EVAL_RANGE[4], 7] = True
    variants = build_variants([("t", in_full)], seen)
    qtiles = [0.1, 0.5]
    dev = torch.device("cpu")

    w = sc.MapWriter(str(tmp_path), "m", N, R, E)
    live = cell_pass(lambda ab: scores[:ab.shape[0]].clone() if False else
                     scores[_offset(ab, aa)].clone(),
                     loader, variants, dev, EVAL_RANGE, qtiles,
                     sink=w.write_batch)
    w.close(manifest_sha1="x")

    arr, n_neg = sc.load_map(str(tmp_path), "m")
    assert arr.shape == (N, R, E)
    assert n_neg == 1, "the -inf sentinel must survive to disk and convert on load"

    cached = torch.from_numpy(arr)
    replay = {}
    for suffix, score_mask, target_mask in variants:
        v = cached.clone()
        if score_mask is not None:
            v[:, score_mask[EVAL_RANGE, :]] = float('nan')
        t = mat.clone()
        if target_mask is not None:
            t[target_mask.unsqueeze(0).expand_as(t)] = 0
        for q in qtiles:
            _h, _n, ch, ce = compute_hit_rate(v, t, aa, q, eval_range=EVAL_RANGE,
                                              per_cell=True)
            replay[(suffix, q)] = (ch, ce)

    assert set(replay) == set(live)
    for key in sorted(live):
        lh, le = live[key]
        rh, re_ = replay[key]
        assert torch.allclose(lh, rh, atol=1e-7), key
        assert torch.allclose(le, re_, atol=1e-7), key

    for (suf, q, ma, mi, nc, ne), (rsuf, rq, rma, rmi, rnc, rne) in zip(
            summarize(live), summarize({k: replay[k] for k in replay})):
        assert (suf, q, nc) == (rsuf, rq, rnc)
        # NaN-aware: a variant whose target set has no observed cell scores NaN on both
        # sides by construction (summarize returns NaN at n_cells == 0), and that is the
        # agreement being checked, not a hole.
        def _same(x, y):
            return (np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-7
        assert _same(ma, rma) and _same(mi, rmi), (suf, q, ma, rma, mi, rmi)


def _offset(aa_batch, aa_all):
    """Row indices of this batch inside the full query set (the loader is unshuffled)."""
    idx = []
    for row in aa_batch:
        hit = (aa_all == row).all(dim=1).nonzero()
        idx.append(int(hit[0]))
    return torch.tensor(idx)


def test_sink_offsets_survive_a_short_final_batch(tmp_path):
    """The running offset, not batch_idx * batch_size, is what lines the map up.

    With 23 queries at batch 8 the last batch holds 7. Deriving the offset from the batch
    index would still be right here, but a sink that did so would be wrong the moment a
    loader drops or pads, so the invariant is pinned: the map's row i is query i.
    """
    aa, mat, scores = _world(seed=5)
    loader = DataLoader(_DS(aa, mat), batch_size=8, shuffle=False, drop_last=False)
    variants = build_variants([("t", torch.ones(SEQ_LEN, E, dtype=torch.bool))],
                              torch.zeros(SEQ_LEN, E, dtype=torch.bool))
    w = sc.MapWriter(str(tmp_path), "m", N, R, E)
    cell_pass(lambda ab: scores[_offset(ab, aa)].clone(), loader, variants,
              torch.device("cpu"), EVAL_RANGE, [0.1], sink=w.write_batch)
    w.close(manifest_sha1="x")
    # Raw, so the stored bytes are compared: the -inf sentinel is stored as -inf and
    # only load_map's default folds it into NaN.
    arr, n_neg = sc.load_map(str(tmp_path), "m", neginf_to_nan=False)
    ref = scores[:, EVAL_RANGE, :].numpy()
    both = np.isfinite(arr) & np.isfinite(ref)
    assert np.allclose(arr[both], ref[both])
    assert (np.isnan(arr) == np.isnan(ref)).all()
    assert (np.isneginf(arr) == np.isneginf(ref)).all()
