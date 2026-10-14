"""Tests for the learned codon x BLOSUM62 substitution baseline."""
import torch

from antiGen.config import AlphabetConfig, ModelConfig, ModelType, SubstitutionConfig
from antiGen.models.codon_fitness_predictor import build_codon_lookup
from antiGen.models.factory import create_model
from antiGen.models.substitution_fitness_predictor import (
    SubstitutionFitnessPredictor, build_transition_transversion_counts)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"


def _model(**kwargs):
    config = ModelConfig(model_type=ModelType.SUBSTITUTION, seq_len=8,
                         alphabet=AlphabetConfig(amino_acids=ALPHABET),
                         substitution=SubstitutionConfig(**kwargs))
    return SubstitutionFitnessPredictor(config)


def test_transition_transversion_split_reproduces_the_codon_matrix():
    # ts + tv at kappa = 1 must equal the fixed codon baseline's route counts.
    ts, tv = build_transition_transversion_counts(ALPHABET)
    combined = ts + tv
    reference = torch.exp(build_codon_lookup(ALPHABET, kappa=1.0, pseudocount=0.01)) - 0.01
    assert torch.allclose(combined, reference, atol=1e-4)


def test_split_matches_known_codons():
    ts, tv = build_transition_transversion_counts(ALPHABET)
    k, e, n = ALPHABET.index("K"), ALPHABET.index("E"), ALPHABET.index("N")
    # K(AAA/AAG) -> E(GAA/GAG): position 1 A->G, a transition.
    assert ts[k, e] == 1.0 and tv[k, e] == 0.0
    # K -> N(AAT/AAC): position 3 A->T/C, both transversions.
    assert ts[k, n] == 0.0 and tv[k, n] == 2.0


def test_parameter_count_is_small_and_trainable():
    model = _model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    # log_kappa + w_codon + w_blosum + 22-dim background
    assert sum(p.numel() for p in trainable) == 25
    assert model.get_num_parameters() == 25


def test_gradients_flow_to_every_parameter():
    model = _model()
    x = torch.randint(0, 20, (3, 8))
    model(x).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_kappa_upweights_transitions_in_the_score():
    model = _model()
    k, e = ALPHABET.index("K"), ALPHABET.index("E")
    with torch.no_grad():
        base = model.score_matrix()[k, e].clone()
        model.log_kappa.fill_(float(torch.log(torch.tensor(4.0))))
        raised = model.score_matrix()[k, e]
    assert raised > base


def test_scores_are_finite_including_gap_and_stop():
    model = _model()
    matrix = model.score_matrix()
    assert torch.isfinite(matrix).all()
    # BLOSUM does not score '*'/'-'; those cells must still be usable predictions.
    assert torch.isfinite(matrix[:, ALPHABET.index("*")]).all()
    assert torch.isfinite(matrix[:, ALPHABET.index("-")]).all()


def test_background_shifts_one_destination_only():
    model = _model()
    gap = ALPHABET.index("-")
    with torch.no_grad():
        before = model.score_matrix().clone()
        model.background[gap] += 2.0
        after = model.score_matrix()
    assert torch.allclose(after[:, gap], before[:, gap] + 2.0)
    other = ALPHABET.index("A")
    assert torch.allclose(after[:, other], before[:, other])


def test_no_blosum_ablation_ignores_the_blosum_term():
    model = _model(use_blosum=False)
    with torch.no_grad():
        before = model.score_matrix().clone()
        model.w_blosum.fill_(5.0)
        after = model.score_matrix()
    assert torch.allclose(before, after)


def test_length_agnostic_forward():
    model = _model()
    for seq_len in (8, 31):
        x = torch.randint(0, 20, (2, seq_len))
        assert model(x).shape == (2, seq_len, len(ALPHABET))


def test_factory_and_variable_length_registration():
    from antiGen.infer import VARIABLE_LENGTH_OK
    config = ModelConfig(model_type=ModelType.SUBSTITUTION, seq_len=8,
                         alphabet=AlphabetConfig(amino_acids=ALPHABET))
    model = create_model(config, device="cpu")
    assert isinstance(model, SubstitutionFitnessPredictor)
    assert ModelType.SUBSTITUTION in VARIABLE_LENGTH_OK
