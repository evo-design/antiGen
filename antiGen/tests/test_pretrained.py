"""The shipped 2023-01-01 models load on CPU and reproduce the paper (TUTORIAL.md)."""
import os

import pytest
import torch

from antiGen import pretrained
from antiGen.top_decile import rank_substitutions, score_query, top_decile

FASTA = pretrained.example_fasta("BA.2.86")


def _seq():
    return "".join(l.strip() for l in open(FASTA) if not l.startswith(">"))


@pytest.mark.parametrize("name", sorted(pretrained.available()))
def test_loads_and_normalises(name):
    model, info = pretrained.load(name, device="cpu")
    assert info["seq_len"] == 1273 and info["alphabet"] == "ACDEFGHIKLMNPQRSTVWY"
    seq = _seq()
    x = pretrained.encode(seq, info["alphabet"])
    with torch.no_grad():
        p = pretrained.substitution_probs(model(x), x)
    assert abs(float(p.sum()) - 1.0) < 1e-5
    assert float(p[0, 454, info["alphabet"].index("L")]) == 0.0    # wild type L455


def test_base_model_reproduces_panel_b():
    """sc2_results_revised2 panel B: L455S from BA.2.86 is rank 98 of 3,819 RBD cells."""
    model, info = pretrained.load("sc2-2023-01-01-base")
    seq = _seq()
    ranked = rank_substitutions(score_query(model, seq, info["alphabet"]), seq,
                                info["alphabet"], 331, 531)
    assert len(ranked) == 3819 and len(top_decile(ranked)) == 382
    assert {r["mutation"]: r["rank"] for r in ranked}["L455S"] == 98


def test_cv10_rank_is_stable():
    model, info = pretrained.load("sc2-2023-01-01")
    seq = _seq()
    ranked = rank_substitutions(score_query(model, seq, info["alphabet"]), seq,
                                info["alphabet"], 331, 531)
    assert {r["mutation"]: r["rank"] for r in ranked}["L455S"] == 114
