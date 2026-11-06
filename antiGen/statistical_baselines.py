"""Statistical-model baselines for the forecast test set -- self-contained.

The counted/empirical family, fit off ONE source's TRAINING split. This is the
single source of truth for the math (``count_baselines`` is now a thin shim over
it) and the only dependency is torch, so the file drops into other contexts as-is.

Display names and categories live in ``baseline_names`` (pure Python); this module
only needs them for ``DISPLAY_NAMES`` and degrades gracefully if that file is absent.

Rungs (math key -> display name; see baseline_names for abbreviations/category):
  pssm         "Occurrence PSSM"             log1p(occ[i,a]), wild-type-agnostic
  frequency /  "Mutation Count Matrix (MCM)" log N[i,a]            (== counted_pssm kappa->0)
   cpssm_k0
  site_prop /  "Site x Substitution (SxS)"   s_i * M[wt_i,a]       (== counted_pssm kappa->inf)
   cpssm_kinf
  cpssm_dir    "SxS, MCM Interpolation"      (N + kappa*s*M)/(N_total+kappa), kappa=N_total
  <site_mixture_map>  "Site x Mixture (SxM)"  log s_i + log P_mix(a|wt_i); the mixture
               substitution matrix is EXTERNAL (fitted), so it is a separate function
               that takes the matrix as an argument rather than a score_map rung.

Identities (pinned in tests/test_count_baselines.py): kappa->0 == MCM,
kappa->inf == SxS. Same family as antiGen/models/{site_propensity,counted_pssm}.
"""
from __future__ import annotations
import torch

try:                                             # names are nice-to-have, not required
    from baseline_names import DISPLAY_NAMES     # math-key -> canonical display string
except Exception:                                # pragma: no cover - standalone fallback
    DISPLAY_NAMES = {}

ALPHA = 0.5          # add-alpha smoothing on s and M, matching the canonical classes
_FLOOR = 1e-30

# Friendly keys -> the locked math keys score_map already understands. Lets callers
# use the registry's canonical keys without touching the ~40 scripts that pass the
# originals.
_KEY_ALIASES = {
    "occurrence_pssm": "pssm",
    "mcm": "frequency",
    "site_substitution": "site_prop",
    "mcm_interpolation": "cpssm_dir",
}


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
    """(B, L, E) log-scores for `name`, given a batch of wild-type indices aa_batch (B, L).

    Accepts the locked math keys (pssm, frequency, cpssm_k0, site_prop, cpssm_kinf,
    cpssm_dir) and the registry's canonical aliases (occurrence_pssm, mcm,
    site_substitution, mcm_interpolation). Site x Mixture is `site_mixture_map`.
    """
    name = _KEY_ALIASES.get(name, name)
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


def site_mixture_map(st: dict, aa_batch: torch.Tensor, log_sub_mat: torch.Tensor,
                     alpha: float = ALPHA) -> torch.Tensor:
    """Site x Mixture (SxM): positional marginal x an EXTERNAL substitution mixture.

    SxM = log s_i + log P_mix(a | wt_i), where the mixture ``log_sub_mat`` is an
    (E, E) tensor of log P(a | x) supplied by the caller -- typically
    ``log_softmax`` of the fitted codon x BLOSUM substitution matrix
    (``fit_substitution_baseline(...).score_matrix()``). Kept out of ``score_map``
    precisely because that matrix is fitted, not a pure count statistic.

    The "where" term uses the same add-alpha positional counts as
    ``eval_utils.positional_marginal_counts`` (row sums of N + alpha), so this
    reproduces the inline ``where_times_which`` rung the scoring scripts used.

    Returns (B, L, E) log-scores.
    """
    N = st["N"]
    L, E = N.shape
    B = aa_batch.shape[0]
    pos_counts = (N + alpha).sum(dim=1)                   # (L,) smoothed positional marginal
    where = torch.log(pos_counts.clamp(min=_FLOOR)).view(1, L, 1)
    which = log_sub_mat[aa_batch]                         # (B, L, E) = log P(a | wt_i)
    return where + which
