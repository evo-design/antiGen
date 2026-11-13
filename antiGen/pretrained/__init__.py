"""Pretrained antiGen models shipped with the package.

Two SARS-CoV-2 spike models, both trained on the tree built from sequences dated before
**2023-01-01** (the paper's 2023 cutoff), on the 20-residue alphabet and the 1273-residue
Wuhan-Hu-1 spike frame:

``sc2-2023-01-01``  (default)
    antiGen (CV x10): ten models trained on different 90/10 train/validation splits,
    each kept at its best validation epoch, combined as the mean of their per-query
    log-probabilities. This is the "antiGen (CV x10)" of the paper's bar panels.
``sc2-2023-01-01-base``
    The single base model trained at that cutoff, the one behind the paper's BA.2.86 to
    KP.3 case study (panel B) and its cutoff-series panels.

Both are small (~2.3 MB a model, 25 MB in all) and run on a CPU.

    >>> from antiGen import pretrained
    >>> model, info = pretrained.load("sc2-2023-01-01")
    >>> scores = model(aa_idx)        # (batch, 1273, 20) log-scores; higher = more likely

A model's output ranks every (position, residue) cell of a query; turn it into a
distribution over substitutions with :func:`substitution_probs`.
"""
from __future__ import annotations

import os

import torch

from antiGen.eval_utils import MeanLogProbEnsemble   # the paper's "Mean of log-probs" rule
from antiGen.models.factory import load_model

_HERE = os.path.dirname(os.path.abspath(__file__))

#: name -> (relative member paths, description)
REGISTRY = {
    "sc2-2023-01-01": (
        [f"sc2_2023-01-01/cv10/member_s{i}.pt" for i in range(10)],
        "antiGen (CV x10), SARS-CoV-2 spike, 2023-01-01 cutoff",
    ),
    "sc2-2023-01-01-base": (
        ["sc2_2023-01-01/base.pt"],
        "antiGen single base model, SARS-CoV-2 spike, 2023-01-01 cutoff",
    ),
}
DEFAULT = "sc2-2023-01-01"


def available():
    """Names of the shipped models, with a one-line description each."""
    return {k: v[1] for k, v in REGISTRY.items()}


def paths(name=DEFAULT):
    """Absolute paths of a shipped model's checkpoint file(s)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown pretrained model {name!r}; available: {sorted(REGISTRY)}")
    out = [os.path.join(_HERE, p) for p in REGISTRY[name][0]]
    missing = [p for p in out if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{name}: missing {missing} (is the package installed with its data?)")
    return out


def load(name=DEFAULT, device="cpu"):
    """Load a shipped model. Returns ``(model, info)``.

    ``info`` holds the model's ``alphabet``, ``seq_len``, ``name``, ``description`` and
    the members' ``provenance`` records. A one-member model is returned as is.
    """
    members = [load_model(p, device=device)[0].eval() for p in paths(name)]
    ## Each member log-softmaxed over the whole (position x residue) map, then averaged:
    ## the evaluation's ensemble, whose within-query ranking the paper reports.
    if len(members) == 1:
        model = members[0]
    else:
        model = MeanLogProbEnsemble(members).to(device).eval()
        model.config = members[0].config   # alphabet / seq_len, as on a single model
    prov = [torch.load(p, map_location="cpu", weights_only=False).get("provenance", {})
            for p in paths(name)]
    info = {"name": name, "description": REGISTRY[name][1],
            "alphabet": members[0].config.alphabet.amino_acids,
            "seq_len": members[0].config.seq_len, "provenance": prov}
    return model, info


def encode(seq, alphabet):
    """A sequence as a (1, L) index tensor in the model's alphabet."""
    bad = sorted({c for c in seq if c not in alphabet})
    if bad:
        raise ValueError(f"characters not in the model alphabet {alphabet!r}: {bad}")
    return torch.tensor([[alphabet.index(c) for c in seq]], dtype=torch.long)


def substitution_probs(scores, aa_idx):
    """(B, L, E) scores -> the distribution over substitutions of each query.

    A softmax over the whole (position x residue) map with each position's wild-type
    residue removed, renormalised to sum to 1: the quantity antiGen is trained on (one
    distribution over mutation events per sequence, not a per-position categorical).
    """
    s = scores.float().clone()
    s.scatter_(2, aa_idx.unsqueeze(-1).to(s.device), float("-inf"))
    return torch.softmax(s.flatten(1), dim=1).reshape(s.shape)
