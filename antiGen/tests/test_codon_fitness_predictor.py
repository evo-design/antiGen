"""Tests for the codon-accessibility baseline (core deps only, no downloads)."""
import torch

from antiGen.config import AlphabetConfig, CodonConfig, ModelConfig, ModelType
from antiGen.models.codon_fitness_predictor import (CODON_TABLE, CodonFitnessPredictor,
                                                    build_codon_lookup)
from antiGen.models.factory import create_model

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"


def _counts(kappa=1.0, pseudocount=0.01):
    """The lookup as raw route counts (undo the log and the pseudocount)."""
    return torch.exp(build_codon_lookup(ALPHABET, kappa, pseudocount)) - pseudocount


def test_genetic_code_is_complete_and_standard():
    assert len(CODON_TABLE) == 64
    assert sum(1 for aa in CODON_TABLE.values() if aa == '*') == 3
    assert CODON_TABLE["ATG"] == "M"          # start
    assert CODON_TABLE["TGG"] == "W"
    assert {CODON_TABLE[c] for c in ("TAA", "TAG", "TGA")} == {"*"}
    assert len(set(CODON_TABLE.values())) == 21   # 20 amino acids + stop


def test_methionine_row_matches_the_single_atg_codon():
    # M has exactly one codon, ATG, so counts are exact single-nucleotide neighbours:
    # ATA/ATC/ATT -> I (3); CTG/TTG -> L (2); AAG -> K, AGG -> R, ACG -> T, GTG -> V (1).
    counts = _counts()
    row = counts[ALPHABET.index("M")]
    assert row[ALPHABET.index("I")] == 3.0
    assert row[ALPHABET.index("L")] == 2.0
    for aa in "KRTV":
        assert row[ALPHABET.index(aa)] == 1.0
    assert row[ALPHABET.index("D")] == 0.0     # not reachable in one step


def test_tryptophan_can_reach_a_stop_codon():
    # TGG -> TGA / TAG are both stops, so '*' is a genuine two-route prediction.
    row = _counts()[ALPHABET.index("W")]
    assert row[ALPHABET.index("*")] == 2.0
    assert row[ALPHABET.index("C")] == 2.0     # TGT, TGC


def test_gap_has_no_codon_and_stays_at_the_floor():
    counts = _counts()
    assert torch.all(counts[ALPHABET.index("-")] == 0.0)   # whole row is the floor
    assert torch.all(counts[:, ALPHABET.index("-")] == 0.0)  # and the whole column


def test_inaccessible_pairs_are_finite_but_lowest():
    lookup = build_codon_lookup(ALPHABET)
    k, w = ALPHABET.index("K"), ALPHABET.index("W")
    assert torch.isfinite(lookup[k, w])                     # floor, not -inf
    assert lookup[k, w] < lookup[k, ALPHABET.index("N")]    # ranks below a real route


def test_kappa_upweights_transitions_only():
    plain, weighted = _counts(kappa=1.0), _counts(kappa=4.0)
    # K (AAA/AAG) -> E (GAA/GAG) is A<->G at codon position 1: a transition.
    k, e = ALPHABET.index("K"), ALPHABET.index("E")
    assert torch.isclose(weighted[k, e], 4.0 * plain[k, e], atol=1e-4)
    # K -> N (AAT/AAC) changes position 3 A->T/C: transversions, so unchanged.
    n = ALPHABET.index("N")
    assert torch.isclose(weighted[k, n], plain[k, n], atol=1e-4)


def test_forward_is_a_lookup_and_length_agnostic():
    config = ModelConfig(model_type=ModelType.CODON, seq_len=5,
                         alphabet=AlphabetConfig(amino_acids=ALPHABET))
    model = CodonFitnessPredictor(config)
    assert model.get_num_parameters() == 0
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    for seq_len in (5, 17):                      # trained seq_len is irrelevant
        x = torch.randint(0, 20, (3, seq_len))
        out = model(x)
        assert out.shape == (3, seq_len, len(ALPHABET))
        assert torch.isfinite(out).all()
        assert torch.equal(out[0, 0], model.lookup[x[0, 0]])


def test_factory_builds_it_with_config_kappa():
    config = ModelConfig(model_type=ModelType.CODON, seq_len=4,
                         alphabet=AlphabetConfig(amino_acids=ALPHABET),
                         codon=CodonConfig(kappa=3.0))
    model = create_model(config, device="cpu")
    assert isinstance(model, CodonFitnessPredictor)
    k, e = ALPHABET.index("K"), ALPHABET.index("E")
    assert torch.isclose(torch.exp(model.lookup[k, e]) - 0.01,
                         torch.tensor(3.0), atol=1e-4)


def test_gap_stop_flags_reflect_the_genetic_code():
    # '*' is a real state of the code; '-' has no codon.
    assert CodonFitnessPredictor.OUTPUTS_LEGITIMATE_STOP is True
    assert CodonFitnessPredictor.OUTPUTS_LEGITIMATE_GAP is False
