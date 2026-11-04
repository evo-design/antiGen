"""Back-compat shim -- the count baselines now live in ``statistical_baselines``.

Kept so the ~40 scoring/figure scripts that do ``import count_baselines`` and call
``count_baselines.fit`` / ``count_baselines.score_map`` with the original math keys
(pssm, frequency, cpssm_k0, site_prop, cpssm_kinf, cpssm_dir) keep working unchanged.
New code should import ``statistical_baselines`` directly.

The math and its canonical-identity tests (tests/test_count_baselines.py) are
unchanged; only the human-facing names moved to ``baseline_names`` (the renames:
Event Counts -> Mutation Count Matrix (MCM), Where x which -> Site x Mixture (SxM),
Smoothed Counts -> "SxS, MCM Interpolation"; Site x Substitution keeps its name,
abbreviated SxS).
"""
from __future__ import annotations

from statistical_baselines import (  # noqa: F401  (re-exported surface)
    ALPHA,
    _FLOOR,
    fit,
    score_map,
    site_mixture_map,
)

# math key -> canonical display name. Previously defined here; now owned by
# baseline_names (single source of truth) and re-exported for any old reader.
from baseline_names import DISPLAY_NAMES  # noqa: F401
