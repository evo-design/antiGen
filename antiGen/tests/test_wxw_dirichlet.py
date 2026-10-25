"""The Dirichlet blend of where_times_which with the observed mutation events.

The model is a posterior mean: the factored joint P(i) * P(a | wt_i) is the base
measure carrying concentration kappa, and each observed mutation event is a conjugate
update. What makes that claim checkable is its two limits -- kappa -> inf is the
factored map and kappa -> 0 is the event-count map, both exactly, both up to a single
additive constant over the whole map that within-query ranking cannot see.

The first limit is a rung drawn on the same panels, so this is the same shape of gate
as ensemble_test_rows' --verify-member: a construction that must reproduce something
already on the figure.

Core dependencies only (torch); no model downloads.
"""
import pytest
import torch

from antiGen.config import AlphabetConfig, ModelConfig, ModelType, SubstitutionConfig
from antiGen.eval_utils import (frequency_baseline_scores, positional_marginal_counts,
                                where_times_which_dirichlet,
                                WhereTimesWhichDirichletPredictor,
                                WhereTimesWhichPredictor)
from antiGen.models.factory import create_model


ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALPHABET)
L = 12


class FakeExample:
    """Minimal stand-in for load.Example: the baselines only read ``.mat``."""

    def __init__(self, mat):
        self.mat = mat


def _examples(cells):
    mat = torch.zeros(L, E)
    for (pos, aa), count in cells.items():
        mat[pos, aa] = float(count)
    return [FakeExample(mat)]


def _substitution_model():
    config = ModelConfig(
        model_type=ModelType.SUBSTITUTION, seq_len=L,
        alphabet=AlphabetConfig(amino_acids=ALPHABET),
        substitution=SubstitutionConfig())
    return create_model(config, torch.device('cpu'))


def _spread(a, b):
    """Max minus min of (a - b): zero iff they differ by one additive constant."""
    d = a - b
    return float(d.max() - d.min())


CELLS = {(2, 4): 6.0, (7, 13): 3.0, (2, 9): 1.0}


def test_is_a_normalized_posterior_mean():
    """Every map must be a distribution over (i, a): positive, summing to one.

    The counts go in raw, so this is also the assertion that the prior alone keeps
    every cell off zero -- an unsmoothed count map would put -inf in most of them.
    """
    subst = _substitution_model()
    model = where_times_which_dirichlet(subst, _examples(CELLS), L, E)
    x = torch.randint(0, 20, (3, L))
    q = torch.exp(model(x))
    assert torch.isfinite(q).all() and (q > 0).all()
    assert torch.allclose(q.sum(dim=(1, 2)), torch.ones(3), atol=1e-5)


def test_large_kappa_is_the_factored_map():
    """kappa -> inf must reproduce where_times_which, which is a drawn rung."""
    examples = _examples(CELLS)
    subst = _substitution_model()
    pm = positional_marginal_counts(examples, L, E)
    wxw = WhereTimesWhichPredictor(subst.config, pm, subst)
    blend = where_times_which_dirichlet(subst, examples, L, E, kappa=1e12)

    x = torch.randint(0, 20, (2, L))
    assert _spread(blend(x), wxw(x)) < 1e-4


def test_small_kappa_is_the_event_count_map():
    """kappa -> 0 must reproduce the raw event counts, on the cells that carry them.

    Only the nonzero cells: with no prior mass the empty ones are log(0), which is
    the thing the blend exists to avoid rather than a value to compare.
    """
    examples = _examples(CELLS)
    subst = _substitution_model()
    counts = frequency_baseline_scores(examples, L, E, alpha=0.0)
    blend = where_times_which_dirichlet(subst, examples, L, E, kappa=1e-9)

    x = torch.randint(0, 20, (2, L))
    seen = counts > 0
    got = blend(x)[:, seen]
    want = torch.log(counts[seen]).unsqueeze(0).expand(2, -1)
    assert _spread(got, want) < 1e-4


def test_default_kappa_weights_prior_and_data_equally():
    """The setting the idea arrived with: kappa = N_total."""
    examples = _examples(CELLS)
    model = where_times_which_dirichlet(_substitution_model(), examples, L, E)
    assert model.kappa == pytest.approx(sum(CELLS.values()))
    assert model.n_total == pytest.approx(sum(CELLS.values()))


def test_counts_enter_raw_not_smoothed():
    """Double smoothing would break both limits; this pins that N carries no alpha."""
    examples = _examples(CELLS)
    model = where_times_which_dirichlet(_substitution_model(), examples, L, E)
    assert torch.allclose(model.event_counts,
                          frequency_baseline_scores(examples, L, E, alpha=0.0))
    assert float(model.event_counts.min()) == 0.0


def test_an_observed_cell_is_lifted_above_its_prior():
    """The update has to do something, and in the direction the model claims.

    Two cells with the same wild type and the same position: the one the training
    window observed must outrank the one it did not, which the factored map alone
    cannot express.
    """
    examples = _examples({(5, 4): 50.0})
    subst = _substitution_model()
    pm = positional_marginal_counts(examples, L, E)
    wxw = WhereTimesWhichPredictor(subst.config, pm, subst)
    blend = where_times_which_dirichlet(subst, examples, L, E)

    x = torch.zeros(1, L, dtype=torch.long)
    w, b = wxw(x)[0], blend(x)[0]
    # The factored map ranks these two cells by chemistry alone; the blend must move
    # the observed one up relative to the unobserved one at the same position.
    assert (b[5, 4] - b[5, 6]) > (w[5, 4] - w[5, 6]) + 1.0


def test_wild_type_counts_do_not_leak_through_the_prior():
    """The which term is a conditional on the example's own residue, as the parent's is.

    Two positions sharing a wild type and sharing counts must get identical rows, so
    the blend inherits the property that makes the factored term position-independent.
    """
    examples = _examples({})
    subst = _substitution_model()
    # kappa explicitly: with no events the default kappa would be N_total = 0.
    blend = WhereTimesWhichDirichletPredictor(
        subst.config, torch.zeros(L, E), torch.ones(L), subst, kappa=1.0)
    x = torch.zeros(1, L, dtype=torch.long)
    x[0, 4] = 9
    x[0, 6] = 9
    out = blend(x)
    assert torch.allclose(out[0, 4], out[0, 6], atol=1e-6)
    assert not torch.allclose(out[0, 4], out[0, 0], atol=1e-6)


def test_a_fit_with_no_events_refuses_rather_than_returning_nan():
    """N_total = 0 makes the default kappa 0 and the posterior 0/0.

    An all-NaN map does not fail -- it ranks as a tie in every cell and would be read
    as a rung that simply performs at chance. Refusing is the only honest option.
    """
    with pytest.raises(ValueError, match="no mutation events"):
        where_times_which_dirichlet(_substitution_model(), _examples({}), L, E)


def test_no_events_but_a_prior_is_exactly_the_prior():
    """The other half of the same edge: kappa > 0 and no data must give the base measure."""
    examples = _examples({})
    subst = _substitution_model()
    pm = positional_marginal_counts(examples, L, E)
    wxw = WhereTimesWhichPredictor(subst.config, pm, subst)
    blend = where_times_which_dirichlet(subst, examples, L, E, kappa=1.0)
    x = torch.randint(0, 20, (2, L))
    assert _spread(blend(x), wxw(x)) < 1e-5
