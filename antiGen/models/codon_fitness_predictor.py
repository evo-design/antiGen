"""Codon accessibility (the standard genetic code) as a fitness predictor.

Amino-acid substitutions are not equally easy to reach: most arise from a *single*
nucleotide change in the underlying codon, and how many such single-nucleotide paths
exist between two amino acids is fixed by the genetic code. This predictor turns that
into a baseline — the protein-level analogue of a transition/transversion prior — with
no learnable parameters and no training data.

For wild-type amino acid ``w`` and mutant ``m`` the score is

    score(w -> m) = log( mean_{c in codons(w)} #{c' in codons(m) : hamming(c, c') == 1}
                         + pseudocount )

i.e. the expected number of single-nucleotide routes out of a random codon of ``w`` into
*some* codon of ``m``, averaged over ``w``'s synonymous codons because the examples format
carries no nucleotide state. Pairs with no one-step route (and ``w == m``, which the
scoring machinery masks anyway) fall back to the pseudocount, so they rank below every
accessible substitution without becoming ``-inf``.

The stop codon ``*`` is a genuine amino-acid state of the genetic code and gets a real
score; the gap ``-`` has no codon and gets the pseudocount floor.

Set ``--codon-weighting observed`` to weight transitions over transversions
(``--codon-kappa``, default 1.0 = unweighted), which makes the baseline strictly stronger
on real phylogenies without fitting anything to the data.
"""
from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import torch

from antiGen.models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from antiGen.config import ModelConfig

# Standard genetic code (NCBI translation table 1), stop codons as '*'.
BASES = "TCAG"
_AA_BY_CODON_ORDER = (
    "FFLLSSSSYY**CC*W"
    "LLLLPPPPHHQQRRRR"
    "IIIMTTTTNNKKSSRR"
    "VVVVAAAADDEEGGGG"
)
CODON_TABLE = {
    "".join(codon): _AA_BY_CODON_ORDER[i]
    for i, codon in enumerate(product(BASES, repeat=3))
}

# Transitions (purine<->purine, pyrimidine<->pyrimidine) happen more often than
# transversions; kappa scales their weight when --codon-weighting observed is used.
_PURINES, _PYRIMIDINES = {"A", "G"}, {"C", "T"}


def _is_transition(a: str, b: str) -> bool:
    return ({a, b} <= _PURINES) or ({a, b} <= _PYRIMIDINES)


def build_codon_lookup(internal_alphabet: str, kappa: float = 1.0,
                       pseudocount: float = 0.01) -> torch.Tensor:
    """Build the ``(vocab_size, vocab_size)`` log one-step-accessibility matrix.

    Args:
        internal_alphabet: the model alphabet, e.g. ``"ACDEFGHIKLMNPQRSTVWY*-"``.
        kappa: transition/transversion ratio. 1.0 counts every nucleotide change
            equally; >1 up-weights transitions.
        pseudocount: floor added to every entry, so inaccessible substitutions rank
            last without producing ``-inf``.

    Returns:
        ``lookup[w, m]`` = log(mean one-step routes from a codon of ``w`` to a codon of
        ``m``, + pseudocount).
    """
    codons_by_aa: dict[str, list[str]] = {}
    for codon, aa in CODON_TABLE.items():
        codons_by_aa.setdefault(aa, []).append(codon)

    vocab_size = len(internal_alphabet)
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)

    for wt_idx, wt_aa in enumerate(internal_alphabet):
        wt_codons = codons_by_aa.get(wt_aa)
        if not wt_codons:            # '-' has no codon
            continue
        for mut_idx, mut_aa in enumerate(internal_alphabet):
            if not codons_by_aa.get(mut_aa):
                continue
            total = 0.0
            for wt_codon in wt_codons:
                for position in range(3):
                    for base in BASES:
                        if base == wt_codon[position]:
                            continue
                        neighbour = (wt_codon[:position] + base
                                     + wt_codon[position + 1:])
                        if CODON_TABLE[neighbour] != mut_aa:
                            continue
                        total += kappa if _is_transition(wt_codon[position], base) else 1.0
            counts[wt_idx, mut_idx] = total / len(wt_codons)

    return torch.log(counts + pseudocount).float()


class CodonFitnessPredictor(BaseFitnessPredictor):
    """Genetic-code accessibility as a non-trainable fitness predictor.

    For each position in the input sequence, outputs the log number of single-nucleotide
    routes from the wild-type amino acid to every amino acid in the internal alphabet.

    Input: ``(batch, seq_len)`` integer AA indices.
    Output: ``(batch, seq_len, vocab_size)`` log-accessibility scores.
    """

    # '-' has no codon, so the gap column is a floor, not a prediction. '*' is a real
    # state of the genetic code, so the stop column is a genuine prediction.
    OUTPUTS_LEGITIMATE_GAP = False
    OUTPUTS_LEGITIMATE_STOP = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        codon_cfg = getattr(config, "codon", None)
        kappa = getattr(codon_cfg, "kappa", 1.0) if codon_cfg else 1.0
        pseudocount = getattr(codon_cfg, "pseudocount", 0.01) if codon_cfg else 0.01
        lookup = build_codon_lookup(config.alphabet.amino_acids, kappa=kappa,
                                    pseudocount=pseudocount)
        self.register_buffer("lookup", lookup)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.lookup[x]

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        return 0
