"""The codon-aware BLOSUM62 baseline: a fixed, training-free (wild type x alt) lookup."""
import math

import torch

from antiGen.config import AlphabetConfig, CodonBlosumConfig, ModelConfig, ModelType
from antiGen.models.codon_blosum_fitness_predictor import build_score_matrix
from antiGen.models.factory import create_model

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def test_rows_are_distributions_without_wild_type():
    m = build_score_matrix(AA20)
    for i in range(20):
        assert m[i, i] == float("-inf")
        assert math.isclose(float(torch.exp(m[i]).sum()), 1.0, rel_tol=1e-5)


def test_one_nucleotide_changes_outrank_distant_ones():
    m = build_score_matrix(AA20)
    a = AA20.index
    # Leu -> Phe is one nucleotide away (CTT -> TTT); Leu -> Asp is not.
    assert m[a("L"), a("F")] > m[a("L"), a("D")]
    # Kappa up-weights transitions: Ile -> Val (ATT -> GTT, A<->G) vs Ile -> Phe (A<->T).
    assert build_score_matrix(AA20, kappa=4.0)[a("I"), a("V")] > build_score_matrix(AA20, kappa=1.0)[a("I"), a("V")]


def test_stop_and_gap_are_never_scored():
    m = build_score_matrix(AA20 + "*-")
    assert torch.isinf(m[:, 20:]).all() and torch.isinf(m[20:, :]).all()


def test_model_is_the_lookup_and_round_trips(tmp_path):
    cfg = ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=5,
                      alphabet=AlphabetConfig(amino_acids=AA20),
                      codon_blosum=CodonBlosumConfig(kappa=4.0))
    model = create_model(cfg, "cpu").eval()
    x = torch.tensor([[AA20.index(c) for c in "MLKVA"]])
    out = model(x)
    assert out.shape == (1, 5, 20)
    assert torch.equal(out[0, 1], build_score_matrix(AA20)[AA20.index("L")])
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
