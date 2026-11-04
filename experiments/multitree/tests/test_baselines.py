"""Pin the renamed baseline registry and the statistical_baselines surface.

Covers what test_count_baselines.py does not: the new nomenclature
(baseline_names), the key aliases, the shim re-export, and that site_mixture_map
reproduces the inline ``where_times_which`` formula the scoring scripts used.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import baseline_names as names
import statistical_baselines as sb
import count_baselines as cb


class _Ex:
    def __init__(self, aa_idx, mat): self.aa_idx = aa_idx; self.mat = mat


def _toy(L=6, E=22, seed=0):
    g = torch.Generator().manual_seed(seed)
    exs = []
    for _ in range(40):
        aa = torch.randint(0, 20, (L,), generator=g)
        mat = torch.zeros(L, E)
        for i in range(L):
            if torch.rand(1, generator=g).item() < 0.5:
                a = torch.randint(0, 20, (1,), generator=g).item()
                if a != aa[i].item(): mat[i, a] += torch.randint(1, 4, (1,), generator=g).item()
        exs.append(_Ex(aa, mat))
    return exs, L, E


def test_registry_renames():
    assert names.display("mcm") == "Mutation Count Matrix (MCM)"
    assert names.display("frequency") == "Mutation Count Matrix (MCM)"      # via alias
    assert names.display("where_times_which") == "Site × Mixture (S×M)"
    assert names.display("cpssm_dir") == "S×S, MCM Interpolation"
    assert names.display("site_prop") == "Site × Substitution (S×S)"


def test_categories_partition():
    stat = {b.key for b in names.STATISTICAL}
    zero = {b.key for b in names.ZERO_SHOT}
    assert stat == {"occurrence_pssm", "mcm", "site_mixture",
                    "site_substitution", "mcm_interpolation"}
    assert zero == {"esm2_zeroshot", "codon_blosum"}
    assert stat.isdisjoint(zero)
    for b in names.STATISTICAL:
        assert b.category == names.CATEGORY_STATISTICAL
    for b in names.ZERO_SHOT:
        assert b.category == names.CATEGORY_ZEROSHOT


def test_canonical_key_aliases_match_math_keys():
    exs, L, E = _toy()
    st = sb.fit(exs, L, E)
    ab = torch.stack([exs[k].aa_idx for k in range(3)])
    for canon, math_key in [("occurrence_pssm", "pssm"), ("mcm", "frequency"),
                            ("site_substitution", "site_prop"),
                            ("mcm_interpolation", "cpssm_dir")]:
        assert torch.allclose(sb.score_map(canon, st, ab),
                              sb.score_map(math_key, st, ab)), canon


def test_site_mixture_matches_inline_formula():
    exs, L, E = _toy()
    st = sb.fit(exs, L, E)
    ab = torch.stack([exs[k].aa_idx for k in range(3)])
    log_sub = torch.log_softmax(torch.randn(E, E, generator=torch.Generator().manual_seed(1)), dim=-1)

    got = sb.site_mixture_map(st, ab, log_sub)

    # inline reference from score_perevent*.py: sub_mat[aa] + log(pos_counts).view(L,1),
    # pos_counts == eval_utils.positional_marginal_counts == (N + alpha).sum(dim=1)
    pos_counts = (st["N"] + sb.ALPHA).sum(dim=1)
    for b in range(ab.shape[0]):
        ref = log_sub[ab[b]] + torch.log(pos_counts.clamp(min=1e-30)).view(L, 1)
        assert torch.allclose(got[b], ref)


def test_shim_reexports():
    assert cb.fit is sb.fit and cb.score_map is sb.score_map
    assert cb.DISPLAY_NAMES["frequency"] == "Mutation Count Matrix (MCM)"
