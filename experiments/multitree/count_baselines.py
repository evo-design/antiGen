"""Single source of truth for the forecast-side count baselines.

Consolidates the math that was duplicated in score_usermodels.py and score_perevent.py.
It is the SAME model family as antiGen/models/{site_propensity,counted_pssm}; the limits
kappa->0 == event-count PSSM and kappa->inf == site_propensity are the canonical classes'
defining identities (verified in tests/test_count_baselines.py), so this module and those
classes stay in agreement by construction.

    fit(train_examples, L, E)         -> dict of sufficient stats (N, occ, s, M)
    score_map(name, stats, aa_batch)  -> (B, L, E) log-score tensor for one rung

Rungs:
  pssm        occurrence PSSM: log1p of residue-composition counts (wild-type-agnostic)
  frequency   event-count PSSM: log N[i,a]            (== counted_pssm kappa->0; the
              Laplace-smoothed version is eval_utils.frequency_baseline_scores)
  site_prop   s_i * M[wt_i,a]                          (== counted_pssm kappa->inf)
  cpssm_dir   (N[i,a] + kappa*s_i*M[wt_i,a]) / (N_total + kappa), kappa = N_total
              -- your PSSM + Dirichlet-backoff model (antiGen/models/counted_pssm)
"""
from __future__ import annotations
import torch

ALPHA = 0.5          # add-alpha smoothing on s and M, matching the canonical classes
_FLOOR = 1e-30


def fit(train_examples, L: int, E: int):
    """Sufficient statistics off one source's TRAINING split (event + occurrence counts)."""
    N = torch.zeros(L, E)      # mutation-EVENT counts
    occ = torch.zeros(L, E)    # residue-OCCURRENCE counts
    M = torch.zeros(E, E)      # substitution events accumulated by each example's own wild type
    for ex in train_examples:
        N += ex.mat
        occ[torch.arange(L), ex.aa_idx] += 1.0
        M.index_add_(0, ex.aa_idx.to(torch.long), ex.mat)
    s = N.sum(dim=1) + ALPHA
    s = s / s.sum()
    Msm = M.clone()
    Msm.fill_diagonal_(0.0)                # drop self-substitutions before smoothing
    Msm = Msm + ALPHA
    Msm.fill_diagonal_(0.0)                # keep the diagonal exactly 0 after smoothing
    Msm = Msm / Msm.sum(dim=1, keepdim=True).clamp(min=_FLOOR)
    return {"N": N, "occ": occ, "s": s, "M": Msm, "N_total": float(N.sum())}


def score_map(name: str, st: dict, aa_batch: torch.Tensor) -> torch.Tensor:
    """(B, L, E) log-scores for `name`, given a batch of wild-type indices aa_batch (B, L)."""
    N, occ, s, M, Nt = st["N"], st["occ"], st["s"], st["M"], st["N_total"]
    L, E = occ.shape
    B = aa_batch.shape[0]
    if name == "pssm":
        return torch.log1p(occ).unsqueeze(0).expand(B, -1, -1).clone()
    if name in ("frequency", "cpssm_k0"):
        return torch.log(N.clamp(min=_FLOOR)).unsqueeze(0).expand(B, -1, -1).clone()
    Mrows = M[aa_batch]                                   # (B, L, E) = M[wt_i, a]
    logs = torch.log(s.clamp(min=_FLOOR)).view(1, L, 1)
    if name in ("site_prop", "cpssm_kinf"):
        return logs + torch.log(Mrows.clamp(min=_FLOOR))
    if name == "cpssm_dir":                               # kappa = N_total (equal weight)
        q = (N.unsqueeze(0) + Nt * (s.view(1, L, 1) * Mrows)) / (Nt + Nt)
        return torch.log(q.clamp(min=_FLOOR))
    raise ValueError(f"unknown count baseline '{name}'")
