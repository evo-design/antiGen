"""A learned log-linear substitution model: codon accessibility x BLOSUM62.

This is the strong form of the codon baseline. Rather than fixing the genetic-code
accessibility counts and the transition/transversion ratio by hand, it scores

    s(w -> m) = w_codon * log(ts(w,m) * kappa + tv(w,m) + pseudocount)
              + w_blosum * blosum62(w, m)
              + background(m)

and fits ``kappa``, ``w_codon``, ``w_blosum`` and a 22-dim per-destination background by
the same objective used to train every other model here. That is 25 scalars in total, so
it cannot memorise anything: it is a substitution matrix whose shape is dictated by the
genetic code and by chemical similarity, with only the mixing and the amino-acid
background left free. The background term matters because gaps (point deletions) and stop
codons occur far more often than codon accessibility and BLOSUM alone would imply, and
BLOSUM does not score them at all.

Being position-blind, it is also length-agnostic, so a single instance can be fitted
across any number of phylogenies at once and compared directly with ESM-2 on each.

What it deliberately does *not* know is which positions are mutable. The per-position
frequency baseline in ``eval_utils.frequency_baseline_scores`` covers that axis; this one
isolates the question "given that a site mutates, which amino acid does it go to", which
is where the genetic code and chemical similarity actually carry information.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from antiGen.models.base import BaseFitnessPredictor
from antiGen.models.codon_fitness_predictor import BASES, CODON_TABLE, _is_transition

if TYPE_CHECKING:
    from antiGen.config import ModelConfig


def build_transition_transversion_counts(internal_alphabet: str):
    """Split one-step codon accessibility into transition and transversion counts.

    Returns ``(ts, tv)``, each ``(vocab, vocab)``, where ``ts[w, m]`` is the mean number of
    single-nucleotide *transitions* out of a codon of ``w`` that produce ``m`` (averaged
    over ``w``'s synonymous codons), and ``tv`` the same for transversions. Weighting them
    by a free ``kappa`` and summing recovers the fixed matrix used by the codon baseline.
    """
    codons_by_aa: dict[str, list[str]] = {}
    for codon, aa in CODON_TABLE.items():
        codons_by_aa.setdefault(aa, []).append(codon)

    vocab = len(internal_alphabet)
    ts = torch.zeros(vocab, vocab, dtype=torch.float64)
    tv = torch.zeros(vocab, vocab, dtype=torch.float64)

    for wt_idx, wt_aa in enumerate(internal_alphabet):
        wt_codons = codons_by_aa.get(wt_aa)
        if not wt_codons:
            continue
        for mut_idx, mut_aa in enumerate(internal_alphabet):
            if not codons_by_aa.get(mut_aa):
                continue
            n_ts = n_tv = 0.0
            for wt_codon in wt_codons:
                for position in range(3):
                    for base in BASES:
                        if base == wt_codon[position]:
                            continue
                        neighbour = (wt_codon[:position] + base + wt_codon[position + 1:])
                        if CODON_TABLE[neighbour] != mut_aa:
                            continue
                        if _is_transition(wt_codon[position], base):
                            n_ts += 1.0
                        else:
                            n_tv += 1.0
            ts[wt_idx, mut_idx] = n_ts / len(wt_codons)
            tv[wt_idx, mut_idx] = n_tv / len(wt_codons)

    return ts.float(), tv.float()


def build_blosum_matrix(internal_alphabet: str):
    """BLOSUM62 as a finite ``(vocab, vocab)`` matrix plus a 'not scored' mask.

    ``blosum62_fitness_predictor`` uses ``-inf`` for pairs BLOSUM does not cover (``*``,
    ``-``). A ``-inf`` feature would make the log-linear score ``-inf`` regardless of the
    other terms, so here those cells are filled with the matrix minimum and flagged, and
    the flagged columns get their own learnable bias.
    """
    from Bio.Align.substitution_matrices import load
    blosum = load("BLOSUM62")
    covered = set(blosum.alphabet)

    vocab = len(internal_alphabet)
    matrix = torch.zeros(vocab, vocab)
    scored = torch.zeros(vocab, vocab, dtype=torch.bool)
    for wt_idx, wt_char in enumerate(internal_alphabet):
        for mut_idx, mut_char in enumerate(internal_alphabet):
            if wt_char in covered and mut_char in covered:
                matrix[wt_idx, mut_idx] = float(blosum[wt_char, mut_char])
                scored[wt_idx, mut_idx] = True
    matrix[~scored] = float(matrix[scored].min())
    return matrix, scored


class SubstitutionFitnessPredictor(BaseFitnessPredictor):
    """Log-linear mixture of codon accessibility and BLOSUM62, with five free scalars.

    Input: ``(batch, seq_len)`` integer AA indices.
    Output: ``(batch, seq_len, vocab_size)`` scores.
    """

    # Both special columns carry their own learnable bias, so both are real predictions.
    OUTPUTS_LEGITIMATE_GAP = True
    OUTPUTS_LEGITIMATE_STOP = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        alphabet = config.alphabet.amino_acids
        sub_config = getattr(config, "substitution", None)
        self.pseudocount = getattr(sub_config, "pseudocount", 0.01) if sub_config else 0.01
        use_blosum = getattr(sub_config, "use_blosum", True) if sub_config else True

        ts, tv = build_transition_transversion_counts(alphabet)
        self.register_buffer("ts_counts", ts)
        self.register_buffer("tv_counts", tv)

        if use_blosum:
            blosum, scored = build_blosum_matrix(alphabet)
        else:
            blosum = torch.zeros(len(alphabet), len(alphabet))
            scored = torch.ones(len(alphabet), len(alphabet), dtype=torch.bool)
        self.register_buffer("blosum", blosum)
        self.register_buffer("blosum_scored", scored)

        init_kappa = getattr(sub_config, "init_kappa", 1.0) if sub_config else 1.0
        self.log_kappa = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_kappa)))))
        self.w_codon = nn.Parameter(torch.tensor(1.0))
        self.w_blosum = nn.Parameter(torch.tensor(0.0))
        # Per-destination background, so '-' and '*' get their own rates rather than
        # sharing one "BLOSUM cannot score this" bias.
        self.background = nn.Parameter(torch.zeros(len(alphabet)))

    def score_matrix(self) -> torch.Tensor:
        """The current ``(vocab, vocab)`` substitution score matrix."""
        kappa = torch.exp(self.log_kappa)
        accessibility = self.ts_counts * kappa + self.tv_counts + self.pseudocount
        score = self.w_codon * torch.log(accessibility) + self.w_blosum * self.blosum
        return score + self.background.unsqueeze(0)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.score_matrix()[x]

    def extra_repr(self) -> str:
        return (f"kappa={float(torch.exp(self.log_kappa)):.3f}, "
                f"w_codon={float(self.w_codon):.3f}, w_blosum={float(self.w_blosum):.3f}")
