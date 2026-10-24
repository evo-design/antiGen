"""The query-dependent half of the pool: no model may predict `-123A` either.

``real_substitution_mask`` (NOTES s51) drops the gap and stop COLUMNS, so no model can
pick ``A123-``. It says nothing about the source residue, so until s52 a position whose
query residue was a gap carried a full 20 candidates -- one MORE than a normal
position's 19, because the wild-type ``scatter_`` removes nothing when the query's own
residue is not a pool column. ``real_wildtype_mask`` drops those rows.

These tests pin the properties the change rests on: the mask is the row-wise mirror of
the column one, it is model-independent at every query, an ungrounded position
contributes no candidates at all, the three implementations still agree under it, the
target-handling modes do what they say, and passing no mask is bit-exact with s51.
"""
import os
import tempfile

import pytest
import torch

import antiGen.ensemble_sweep as es
from antiGen.config import LEGACY_AAS, STANDARD_AAS
from antiGen.background_discrimination import rank_fractions, top_q_weight
from antiGen.eval_utils import (compute_hit_rate, real_substitution_mask,
                                real_wildtype_mask)

AAS = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(AAS)
R = 12
B = 4
GAP, STOP = AAS.index('-'), AAS.index('*')
POOL = real_substitution_mask(R, AAS)


def _scores(seed, drop_cols=()):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(B, R, E, generator=g)
    for c in drop_cols:
        s[:, :, c] = float('nan')
    return s


def _aa(seed=0, ungrounded=()):
    """Query residues, with ``ungrounded`` a list of (row, position, GAP-or-STOP)."""
    g = torch.Generator().manual_seed(seed)
    aa = torch.randint(0, 20, (B, R), generator=g)
    for b, r, c in ungrounded:
        aa[b, r] = c
    return aa


def _targets(cells):
    t = torch.zeros(B, R, E)
    for b, r, e, n in cells:
        t[b, r, e] = n
    return t


# ── the mask itself ───────────────────────────────────────────────────────────

def test_mask_is_the_row_mirror_of_the_column_mask():
    aa = _aa(ungrounded=[(0, 3, GAP), (1, 7, STOP)])
    m = real_wildtype_mask(aa, AAS)
    assert m.shape == (B, R)
    assert not m[0, 3] and not m[1, 7]
    assert int((~m).sum()) == 2
    # every other (query, position) is grounded
    keep = m.clone()
    keep[0, 3] = keep[1, 7] = True
    assert keep.all()


def test_mask_depends_only_on_the_query_not_the_model():
    """Model-independence is the property the whole arm exists to establish."""
    aa = _aa(ungrounded=[(2, 5, GAP)])
    assert torch.equal(real_wildtype_mask(aa, AAS), real_wildtype_mask(aa, AAS))
    for drop in ((), (STOP,), (GAP, STOP)):
        nvs = []
        for seed in (1, 2, 3):
            above, tie, nv = rank_fractions(_scores(seed, drop), aa, list(range(R)),
                                            pool_mask=POOL,
                                            wt_pool_mask=real_wildtype_mask(aa, AAS))
            nvs.append(nv)
        assert all(torch.equal(nvs[0], x) for x in nvs[1:])


# ── what it does to the pool ──────────────────────────────────────────────────

def test_an_ungrounded_position_contributes_no_candidates():
    """And note the arithmetic it corrects: before s52 it contributed 20, not 19."""
    aa_ok = _aa()
    aa_bad = _aa(ungrounded=[(0, 3, GAP), (0, 4, STOP)])
    s = _scores(1)
    _, _, nv_ok = rank_fractions(s, aa_ok, list(range(R)), pool_mask=POOL,
                                 wt_pool_mask=real_wildtype_mask(aa_ok, AAS))
    _, _, nv_bad = rank_fractions(s, aa_bad, list(range(R)), pool_mask=POOL,
                                  wt_pool_mask=real_wildtype_mask(aa_bad, AAS))
    # a grounded position gives 19 candidates (20 columns minus the query's own
    # residue); an ungrounded one now gives 0, where s51 gave it the full 20.
    assert int(nv_ok[0]) == R * 19
    assert int(nv_bad[0]) == (R - 2) * 19
    assert torch.equal(nv_ok[1:], nv_bad[1:])


def test_no_cell_at_an_ungrounded_position_is_ever_selected():
    aa = _aa(ungrounded=[(0, 3, GAP), (1, 9, STOP)])
    wtm = real_wildtype_mask(aa, AAS)
    s = _scores(5)
    # make the ungrounded rows the highest-scoring cells in the map, so nothing but the
    # mask could keep them out of the top decile.
    s[0, 3, :] = 99.0
    s[1, 9, :] = 99.0
    above, tie, nv = rank_fractions(s, aa, list(range(R)), pool_mask=POOL,
                                    wt_pool_mask=wtm)
    for q in (0.1, 0.5, 0.9):
        W = top_q_weight(above, tie, nv, q)
        assert float(W[0, 3].sum()) == 0.0
        assert float(W[1, 9].sum()) == 0.0


def test_gap_to_letter_and_letter_to_gap_are_both_blocked():
    """The two halves together, which is the point of s52."""
    aa = _aa(ungrounded=[(0, 2, GAP)])
    s = _scores(7)
    s[0, 2, 5] = 99.0        # -123A  : blocked by the row mask
    s[0, 6, GAP] = 99.0      # A123-  : blocked by the column mask
    above, tie, nv = rank_fractions(s, aa, list(range(R)), pool_mask=POOL,
                                    wt_pool_mask=real_wildtype_mask(aa, AAS))
    W = top_q_weight(above, tie, nv, 0.1)
    assert float(W[0, 2, 5]) == 0.0
    assert float(W[0, 6, GAP]) == 0.0


# ── targets ───────────────────────────────────────────────────────────────────

def test_an_ungrounded_target_can_never_be_hit():
    """Whatever the mode, no model is ever CREDITED for one."""
    aa = _aa(ungrounded=[(0, 3, GAP)])
    s = _scores(11)
    s[0, 3, :] = 99.0
    t = _targets([(0, 3, 5, 1.0)])
    n_hits, n_mut = compute_hit_rate(s, t, aa, qtile=0.5, eval_range=list(range(R)),
                                     pool_mask=POOL,
                                     wt_pool_mask=real_wildtype_mask(aa, AAS))
    assert n_hits == 0.0
    assert n_mut == 1.0          # 'keep': still in the denominator


def test_drop_removes_the_event_from_both_sides():
    """The 'drop' mode, as ``Substrate.targets_block`` implements it."""
    aa = _aa(ungrounded=[(0, 3, GAP)])
    wtm = real_wildtype_mask(aa, AAS)
    t = _targets([(0, 3, 5, 7.0), (1, 4, 6, 3.0)])
    dropped = t * wtm.unsqueeze(-1).float()
    assert float(dropped[0, 3, 5]) == 0.0      # ungrounded: gone
    assert float(dropped[1, 4, 6]) == 3.0      # grounded: untouched
    assert float(dropped.sum()) == 3.0


def test_substrate_modes(tmp_path):
    aa = _aa(ungrounded=[(0, 3, GAP)])
    t = _targets([(0, 3, 5, 7.0), (1, 4, 6, 3.0)])
    kw = dict(maps_dir=str(tmp_path), aa_eval=aa, targets=t,
              seen=torch.zeros(R, E, dtype=torch.bool),
              in_full=torch.zeros(R, E, dtype=torch.bool),
              observed=t.sum(0) > 0, eval_range=list(range(R)), members=["m"])
    assert es.Substrate(**kw).wt_pool(0, B) is None
    assert float(es.Substrate(**kw).targets_block(0, B).sum()) == 10.0
    keep = es.Substrate(ungrounded="keep", **kw)
    assert keep.wt_pool(0, B) is not None
    assert float(keep.targets_block(0, B).sum()) == 10.0
    drop = es.Substrate(ungrounded="drop", **kw)
    assert float(drop.targets_block(0, B).sum()) == 3.0
    with pytest.raises(ValueError):
        es.Substrate(ungrounded="nonsense", **kw)


# ── the implementations still agree, and nothing moves without the mask ───────

@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("drop", [(), (GAP, STOP)])
def test_hit_rate_and_rank_fractions_agree_under_the_query_pool(q, drop):
    aa = _aa(ungrounded=[(0, 3, GAP), (1, 7, STOP), (2, 0, GAP)])
    wtm = real_wildtype_mask(aa, AAS)
    s = _scores(13, drop)
    t = _targets([(0, 1, 4, 2.0), (1, 5, 9, 1.0), (3, 8, 2, 5.0), (0, 3, 6, 4.0)])
    n_hits, _ = compute_hit_rate(s, t, aa, qtile=q, eval_range=list(range(R)),
                                 pool_mask=POOL, wt_pool_mask=wtm)
    above, tie, nv, filled = rank_fractions(s, aa, list(range(R)), pool_mask=POOL,
                                            return_filled=True, wt_pool_mask=wtm)
    W = top_q_weight(above, tie, nv, q)
    if filled.any():
        W = torch.where(filled, torch.zeros_like(W), W)
    assert float((W * t).sum()) == pytest.approx(n_hits, abs=1e-4)


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_no_wt_mask_is_bit_exact_with_s51(q):
    """The regression the whole comparison rests on: s52 is opt-in."""
    aa = _aa(ungrounded=[(0, 3, GAP), (1, 7, STOP)])
    s = _scores(17)
    t = _targets([(0, 1, 4, 2.0), (0, 3, 6, 4.0), (2, 5, 9, 1.0)])
    a = compute_hit_rate(s, t, aa, qtile=q, eval_range=list(range(R)), pool_mask=POOL)
    b = compute_hit_rate(s, t, aa, qtile=q, eval_range=list(range(R)), pool_mask=POOL,
                         wt_pool_mask=None)
    assert a == b
    x = rank_fractions(s, aa, list(range(R)), pool_mask=POOL)
    y = rank_fractions(s, aa, list(range(R)), pool_mask=POOL, wt_pool_mask=None)
    assert all(torch.equal(torch.nan_to_num(i, nan=-1.0),
                           torch.nan_to_num(j, nan=-1.0)) for i, j in zip(x, y))


def test_wt_masked_applies_the_query_pool_before_normalisation():
    """It has to bite in ``_wt_masked``, not only at ranking time -- NOTES s51."""
    aa = _aa(ungrounded=[(0, 3, GAP)])
    wtm = real_wildtype_mask(aa, AAS)
    s = _scores(19)
    out = es._wt_masked(s, aa, ~POOL, wtm)
    assert torch.isnan(out[0, 3]).all()
    assert not torch.isnan(out[1, 3, :20]).all()
    assert torch.isnan(out[:, :, GAP]).all() and torch.isnan(out[:, :, STOP]).all()


def test_wrong_shaped_mask_is_rejected():
    aa = _aa()
    s = _scores(23)
    with pytest.raises(ValueError):
        rank_fractions(s, aa, list(range(R)), pool_mask=POOL,
                       wt_pool_mask=torch.ones(B, R + 1, dtype=torch.bool))
    with pytest.raises(ValueError):
        compute_hit_rate(s, _targets([]), aa, eval_range=list(range(R)),
                         pool_mask=POOL,
                         wt_pool_mask=torch.ones(B + 1, R, dtype=torch.bool))


# ── The alphabet of a cached map directory (NOTES s56) ──────────────────────────
# `ensemble_sweep` built every mask from a module constant, so a 20-letter map
# directory would have been read in the 22-token frame: the pool mask would drop two
# columns that are not there, and `alt` would name every residue one along. Neither
# fails on shape once the masks agree with each other, so the alphabet is read from the
# maps and then checked against them.

def _fake_maps(tmp, n_res, alphabet=None):
    """A map directory with one model, written the way score_cache writes one."""
    import json
    import numpy as np
    from antiGen import score_cache as sc
    os.makedirs(tmp, exist_ok=True)
    w = sc.MapWriter(tmp, "m", 2, 3, n_res)
    w.write_batch(0, np.zeros((2, 3, n_res), dtype=np.float32))
    w.close(manifest_sha1="x", extra={"n_queries": 2, "eval_range_len": 3})
    man = {"seq_len": 10, "eval_range": [0, 1, 2]}
    if alphabet is not None:
        man["args"] = {"alphabet": alphabet}
    with open(os.path.join(tmp, "query_manifest.json"), "w") as f:
        json.dump(man, f)
    return man


def test_map_alphabet_comes_from_the_manifest():
    from antiGen.ensemble_sweep import alphabet_of_maps
    with tempfile.TemporaryDirectory() as tmp:
        man = _fake_maps(tmp, 20, alphabet=STANDARD_AAS)
        assert alphabet_of_maps(tmp, man) == STANDARD_AAS


def test_a_manifest_without_an_alphabet_falls_back_to_the_legacy_frame():
    """Every map directory dumped before the flag existed is the 22-token frame."""
    from antiGen.ensemble_sweep import alphabet_of_maps
    with tempfile.TemporaryDirectory() as tmp:
        man = _fake_maps(tmp, 22)
        assert alphabet_of_maps(tmp, man) == LEGACY_AAS


def test_an_alphabet_that_disagrees_with_the_maps_is_refused():
    """The check that makes the fallback safe: width is verified against a map file."""
    from antiGen.ensemble_sweep import alphabet_of_maps
    with tempfile.TemporaryDirectory() as tmp:
        man = _fake_maps(tmp, 20, alphabet=STANDARD_AAS)
        with pytest.raises(SystemExit, match="20 residue columns"):
            alphabet_of_maps(tmp, man, LEGACY_AAS)
    with tempfile.TemporaryDirectory() as tmp:
        man = _fake_maps(tmp, 22)          # legacy maps, mislabelled as 20
        with pytest.raises(SystemExit, match="22 residue columns"):
            alphabet_of_maps(tmp, man, STANDARD_AAS)


def test_an_override_wins_over_the_manifest_when_they_agree_in_width():
    from antiGen.ensemble_sweep import alphabet_of_maps
    with tempfile.TemporaryDirectory() as tmp:
        man = _fake_maps(tmp, 20, alphabet=STANDARD_AAS)
        other = "ACDEFGHIKLMNPQRSTVWX"
        assert alphabet_of_maps(tmp, man, other) == other
