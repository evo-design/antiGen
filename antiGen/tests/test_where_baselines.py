"""Tests for the where-only and factorized where x which reference baselines.

These two exist to answer "has the model learned position-specific substitution
preference, or only where mutations land and generic substitution chemistry". The
properties asserted here are what make that reading valid: where_only must carry the
frequency baseline's positional signal and none of its amino-acid signal, and the
factorized predictor must be an exact product of its two terms.

Core dependencies only (torch); no model downloads.
"""
import math

import pytest
import torch

from antiGen.config import AlphabetConfig, ModelConfig, ModelType, SubstitutionConfig
from antiGen.eval_utils import (compute_probs, frequency_baseline_scores,
                                positional_marginal_counts,
                                where_only_baseline_scores, WhereTimesWhichPredictor)
from antiGen.models.factory import create_model


ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALPHABET)
L = 12


class FakeExample:
    """Minimal stand-in for load.Example: the baselines only read ``.mat``."""

    def __init__(self, mat):
        self.mat = mat


def _examples(cells):
    """One example whose child-mutation counts are the given {(pos, aa): count}."""
    mat = torch.zeros(L, E)
    for (pos, aa), count in cells.items():
        mat[pos, aa] = float(count)
    return [FakeExample(mat)]


def test_where_only_preserves_the_positional_marginal():
    """It must keep frequency's per-position mass, only flattening the AA axis."""
    examples = _examples({(3, 5): 7, (3, 9): 2, (8, 1): 4})
    freq = frequency_baseline_scores(examples, L, E)
    where = where_only_baseline_scores(examples, L, E)
    assert torch.allclose(freq.sum(dim=1), where.sum(dim=1))


def test_where_only_is_uniform_across_amino_acids():
    """Every row must be flat, i.e. it carries no 'which residue' information."""
    examples = _examples({(3, 5): 7, (3, 9): 2, (8, 1): 4})
    where = where_only_baseline_scores(examples, L, E)
    for row in where:
        assert torch.allclose(row, row[0].expand_as(row))


def test_flat_counts_score_exactly_uniform():
    """With no positional signal, where_only must equal the uniform reference.

    log(L * (E - 1)) is what ``validate_per_source`` reports as
    ``uniform_nll_per_mutation``, so this pins the new baseline to the existing scale.
    """
    examples = [FakeExample(torch.zeros(L, E))]   # only Laplace smoothing survives
    where = where_only_baseline_scores(examples, L, E)

    aa_batch = torch.zeros(1, L, dtype=torch.long)       # wild type is 'A' everywhere
    target = torch.zeros(1, L, E)
    target[0, 4, 7] = 1.0                                 # one observed mutation
    logits = torch.log(where).unsqueeze(0)
    probs = compute_probs(logits, target, aa_batch)
    normalized = probs / probs.sum(dim=(1, 2), keepdim=True)
    nll = -(target * torch.log(normalized)).sum().item()

    assert nll == pytest.approx(math.log(L * (E - 1)), abs=1e-5)


def test_positional_marginal_matches_frequency_row_sums():
    examples = _examples({(0, 2): 3, (5, 11): 1})
    marginal = positional_marginal_counts(examples, L, E)
    assert torch.allclose(marginal, frequency_baseline_scores(examples, L, E).sum(dim=1))


def _substitution_model():
    config = ModelConfig(
        model_type=ModelType.SUBSTITUTION, seq_len=L,
        alphabet=AlphabetConfig(amino_acids=ALPHABET),
        substitution=SubstitutionConfig())
    return create_model(config, torch.device('cpu'))


def test_factored_predictor_is_the_product_of_its_terms():
    """Output must be log P(i) + log P(a | wt_i), with the which term a real conditional."""
    examples = _examples({(2, 4): 6, (7, 13): 3})
    subst = _substitution_model()
    marginal = positional_marginal_counts(examples, L, E)
    model = WhereTimesWhichPredictor(subst.config, marginal, subst)

    x = torch.randint(0, 20, (2, L))
    out = model(x)
    assert out.shape == (2, L, E)

    which = torch.log_softmax(subst.score_matrix()[x], dim=-1)
    expected = which + torch.log(marginal).view(1, -1, 1)
    assert torch.allclose(out, expected, atol=1e-6)

    # The which term is a proper conditional: it normalizes over the destination axis,
    # so all the positional mass comes from the where term and the two cannot trade off.
    assert torch.allclose(torch.logsumexp(which, dim=-1),
                          torch.zeros(2, L), atol=1e-5)


def test_factored_predictor_tracks_the_wild_type_residue():
    """Two positions sharing a wild type must get identical which-terms.

    This is what makes it 'position-independent chemistry': the conditional depends on
    the residue at i, never on i.
    """
    examples = _examples({(1, 3): 2})
    subst = _substitution_model()
    marginal = torch.ones(L)                      # flat where term, isolating which
    model = WhereTimesWhichPredictor(subst.config, marginal, subst)

    x = torch.zeros(1, L, dtype=torch.long)
    x[0, 4] = 9
    x[0, 6] = 9                                   # same residue, different position
    out = model(x)
    assert torch.allclose(out[0, 4], out[0, 6], atol=1e-6)
    assert not torch.allclose(out[0, 4], out[0, 0], atol=1e-6)
