"""Pin count_baselines to the canonical counted_pssm identities (kappa limits).

counted_pssm is DEFINED by: kappa->0 == the event-count PSSM, kappa->inf == site_propensity
(antiGen/models/counted_pssm_fitness_predictor.py docstring). count_baselines.score_map must
honour both, so the forecast rungs stay in agreement with the canonical model class.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import count_baselines as cb


class _Ex:
    def __init__(self, aa_idx, mat): self.aa_idx = aa_idx; self.mat = mat


def _toy(L=6, E=22, seed=0):
    g = torch.Generator().manual_seed(seed)
    exs = []
    for _ in range(40):
        aa = torch.randint(0, 20, (L,), generator=g)          # wild types (real residues)
        mat = torch.zeros(L, E)
        for i in range(L):
            if torch.rand(1, generator=g).item() < 0.5:
                a = torch.randint(0, 20, (1,), generator=g).item()
                if a != aa[i].item(): mat[i, a] += torch.randint(1, 4, (1,), generator=g).item()
        exs.append(_Ex(aa, mat))
    return exs, L, E


def test_kappa_limits():
    exs, L, E = _toy()
    st = cb.fit(exs, L, E)
    ab = torch.stack([exs[k].aa_idx for k in range(3)])        # (3, L)

    # kappa -> inf limit IS site_propensity (same code path, must match exactly)
    assert torch.allclose(cb.score_map('cpssm_kinf', st, ab),
                          cb.score_map('site_prop', st, ab)), "kappa->inf != site_propensity"

    # kappa -> 0 limit IS the raw event-count PSSM
    assert torch.allclose(cb.score_map('cpssm_k0', st, ab),
                          torch.log(st['N'].clamp(min=1e-30)).unsqueeze(0).expand(3, -1, -1)), \
        "kappa->0 != event-count PSSM"

    # cpssm_dir (kappa = N_total) is strictly between: not equal to either limit
    dir_ = cb.score_map('cpssm_dir', st, ab)
    assert not torch.allclose(dir_, cb.score_map('site_prop', st, ab))
    assert not torch.allclose(dir_, cb.score_map('cpssm_k0', st, ab))


def test_substitution_matrix_drops_self():
    exs, L, E = _toy()
    st = cb.fit(exs, L, E)
    # M[x, x] forced to 0 (a mutation to the residue already there is not an event)
    assert float(st['M'].diagonal().abs().max()) == 0.0
