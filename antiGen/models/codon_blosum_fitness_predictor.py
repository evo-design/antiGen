"""BLOSUM62 corrected for codon distance and known amino-acid mutability.

WHY THIS EXISTS
---------------
BLOSUM62 is the repo's substitution-matrix baseline, and it is a weak one in a specific,
correctable way: it was estimated from *distant homologs*, where any amino acid can reach any other
through many intermediate substitutions. Viral evolution over a few years cannot. On this repo's
own data, 95.5-100% of observed substitutions are reachable by a single nucleotide change, and a
null built from codon adjacency alone sits closer to the dengue substitution matrix than any other
virus does (``examples/analysis/virus_similarity.ipynb``).

So a model can beat BLOSUM62 simply by having learned the genetic code. That is a real capability,
but it is not the capability the manuscript claims, and BLOSUM62 cannot distinguish the two. This
predictor closes that gap: it is still training-free and still knows nothing about any particular
virus, but it knows which substitutions are one nucleotide away and which residues are mutable at
all. Any margin a trained model retains over *this* baseline is a margin over the genetic code.

WHAT IT IS MADE OF -- all three ingredients are external
-------------------------------------------------------
1. **Exchangeability**: BLOSUM62 target frequencies ``q(j|i)``, reconstructed from the published
   half-bit log-odds (``q_ij = p_i p_j 2^(S_ij/2)``) so the matrix is a probability model rather
   than a score, then row-conditioned.
2. **Codon accessibility** ``C(j|i)``: averaging over the source residue's codons, which amino
   acids a single nucleotide change reaches, transitions weighted by ``kappa``. Mixed with a small
   uniform floor ``epsilon`` so that two-nucleotide substitutions -- 4.5% of dengue's, and real --
   are merely unlikely rather than impossible.
3. **Relative mutability** ``m_i`` -- available but **off by default**. Derived from PAM30 as
   ``1 - T_ii``
   after converting the log-odds back to a transition matrix. This reproduces Dayhoff's classic
   ordering (most mutable S/Q/N, least mutable C/L/W), which the module asserts at import rather
   than trusting. It matters because eval softmaxes over the *flattened* (position x AA) map, so a
   per-residue term shifts probability mass between positions, not just within one.

   It is off by default because it is not cleanly sourced. BLOSUM62 *already* encodes mutability,
   in its diagonal (``m_i = 1 - q_ii/p_i``), which step 1 discards by row-conditioning; re-adding
   it from PAM30 mixes two external matrices that rank residues differently (Spearman 0.54, with
   sharp disagreement on Gly and His), so choosing one would need defending. Leaving it off keeps
   the baseline single-sourced: BLOSUM62 plus one mechanistic correction.

Nothing here is fitted to any antiGen dataset, so this remains a drop-in baseline for every virus.

    score(pos, j) = log m_i + log [ q(j|i)^w_blosum C(j|i)^w_codon / Z_i ]

where Z_i row-normalises the product, so the only per-position weighting is the explicit (and by
default absent) m_i. Note the two factors are NOT the same kind of object: q is an evolutionary
OUTCOME distribution -- mutation x selection over long divergence, estimated from ~62%-identity
alignment blocks -- while C is a one-step MUTATIONAL distribution with no selection in it. Their
log-ranks correlate at Spearman 0.36, i.e. BLOSUM62 already encodes some codon reachability, so
the product double-counts mutation. Multiplying them is a product-of-experts heuristic, not a
derivation; the principled form would divide q by its own implied mutational component to recover
a fixation term first. This is a baseline, and that approximation is the price of it.

with the wild-type cell and the gap/stop columns set to -inf, exactly as BLOSUM62FitnessPredictor
does. Scores are used as logits; only their ranking within the flattened map matters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from antiGen.models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from antiGen.config import ModelConfig

_SENTINEL = float("-inf")

# The 20 canonical amino acids, alphabetical -- the order every matrix below is built in.
_AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Published Henikoff background frequencies for BLOSUM62. Recovering these from the matrix by
# fixed-point iteration does not work: it has a degenerate corner attractor and silently converges
# to p in {0, 1}, producing a garbage reference matrix.
_HENIKOFF_ORDER = "ARNDCQEGHILKMFPSTWYV"
_HENIKOFF_P = np.array([0.074, 0.052, 0.045, 0.054, 0.025, 0.034, 0.054, 0.074, 0.026, 0.068,
                        0.099, 0.058, 0.025, 0.047, 0.039, 0.057, 0.051, 0.013, 0.032, 0.073])

_TRANSITIONS = {frozenset("AG"), frozenset("CT")}


def _blosum62_conditional() -> np.ndarray:
    """(20,20) row-conditional BLOSUM62 substitution probabilities, diagonal removed."""
    from Bio.Align.substitution_matrices import load
    b = load("BLOSUM62")
    s = np.array([[float(b[x, y]) for y in _AA20] for x in _AA20], dtype=np.float64)
    p = np.array([_HENIKOFF_P[_HENIKOFF_ORDER.index(c)] for c in _AA20])
    q = np.outer(p, p) * np.power(2.0, s / 2.0)      # half-bit scale
    # Zero the diagonal BEFORE normalising: normalising first would leave rows summing to
    # different totals, which is not a conditional distribution.
    np.fill_diagonal(q, 0.0)
    return q / q.sum(axis=1, keepdims=True)


def _codon_conditional(kappa: float, epsilon: float) -> np.ndarray:
    """(20,20) probability that one nucleotide change takes residue i to residue j."""
    from Bio.Data import CodonTable
    table = CodonTable.unambiguous_dna_by_id[1]
    codons_of: dict[str, list[str]] = {}
    for codon, aa in table.forward_table.items():
        if aa in _AA20:
            codons_of.setdefault(aa, []).append(codon)

    m = np.zeros((20, 20), dtype=np.float64)
    for aa, codons in codons_of.items():
        i = _AA20.index(aa)
        for codon in codons:
            for pos in range(3):
                for base in "ACGT":
                    if base == codon[pos]:
                        continue
                    nxt = codon[:pos] + base + codon[pos + 1:]
                    if nxt in table.stop_codons:
                        continue
                    aa2 = table.forward_table.get(nxt)
                    if aa2 is None or aa2 == aa or aa2 not in _AA20:
                        continue
                    w = kappa if frozenset((codon[pos], base)) in _TRANSITIONS else 1.0
                    m[i, _AA20.index(aa2)] += w / len(codons)
    np.fill_diagonal(m, 0.0)
    m = m / m.sum(axis=1, keepdims=True)
    # A uniform floor over the 19 non-self residues. Without it, every two-nucleotide substitution
    # scores -inf and is ranked below every one-nucleotide one, which is too strong: 4.5% of
    # dengue's observed substitutions are not codon-adjacent.
    off = 1.0 - np.eye(20)
    return (1.0 - epsilon) * m + epsilon * off / 19.0


def _relative_mutability() -> np.ndarray:
    """(20,) Dayhoff-style relative mutability, derived from BioPython's PAM30.

    PAM log-odds are ``S_ij = 10 log10(M_ij / p_j)``, so ``M_ij = p_j 10^(S_ij/10)``; after
    row-normalising, ``1 - M_ii`` is the probability that residue i changes at all. PAM30 is used
    rather than PAM250 because the latter is saturated, and rather than BioPython's ``DAYHOFF``
    matrix, which uses a different scaling and inverts the ordering.
    """
    from Bio.Align.substitution_matrices import load
    pam = load("PAM30")
    s = np.array([[float(pam[x, y]) for y in _AA20] for x in _AA20], dtype=np.float64)
    p = np.array([_HENIKOFF_P[_HENIKOFF_ORDER.index(c)] for c in _AA20])
    t = p[None, :] * np.power(10.0, s / 10.0)
    t = t / t.sum(axis=1, keepdims=True)
    mut = 1.0 - np.diag(t)
    # Checked here rather than trusted: the derived values must reproduce Dayhoff's classic
    # ordering, with tryptophan and cysteine among the least mutable. If a BioPython release
    # changes PAM30's scaling this fails loudly instead of silently reversing the term.
    least = {_AA20[i] for i in np.argsort(mut)[:4]}
    if not {"W", "C"} <= least:
        raise RuntimeError(
            f"derived relative mutability does not rank Trp/Cys among the least mutable residues "
            f"(got {sorted(least)}); PAM30's scaling may have changed")
    return mut / mut.mean()


def build_score_matrix(internal_alphabet: str, kappa: float = 4.0, epsilon: float = 0.05,
                       w_blosum: float = 1.0, w_codon: float = 1.0,
                       use_mutability: bool = False) -> torch.Tensor:
    """(vocab, vocab) log-score lookup: ``lookup[wt, mut]``.

    Cells involving '*' or '-', and the wild-type cell itself, are -inf -- matching
    BLOSUM62FitnessPredictor, whose OUTPUTS_LEGITIMATE_{GAP,STOP} are both False.
    """
    q = _blosum62_conditional()
    c = _codon_conditional(kappa, epsilon)
    combined = np.power(q, w_blosum) * np.power(c, w_codon)
    # Row-normalise BEFORE any per-position term. The product of two row-stochastic matrices is
    # NOT row-stochastic -- sum_j q(j|i)c(j|i) varies 3x across wild-type residues -- and because
    # eval softmaxes over the FLATTENED (position x AA) map, leaving it unnormalised would smuggle
    # in an unintended per-residue mutability weight. It would also make the w_codon=0 and
    # w_blosum=0 arms differ from the combined ones in that respect as well as in their terms,
    # confounding the ablation.
    combined = combined / combined.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        s20 = np.log(combined)
        if use_mutability:
            # The only per-position weighting the model has, and it is explicit and opt-in.
            s20 = s20 + np.log(_relative_mutability())[:, None]
    np.fill_diagonal(s20, -np.inf)

    vocab = len(internal_alphabet)
    lookup = torch.full((vocab, vocab), _SENTINEL, dtype=torch.float32)
    for wi, wc in enumerate(internal_alphabet):
        if wc not in _AA20:
            continue
        for mi, mc in enumerate(internal_alphabet):
            if mc not in _AA20:
                continue
            lookup[wi, mi] = float(s20[_AA20.index(wc), _AA20.index(mc)])
    return lookup


class CodonBLOSUMFitnessPredictor(BaseFitnessPredictor):
    """Training-free baseline: BLOSUM62 x codon accessibility x relative mutability.

    Input:  (batch, seq_len) integer AA indices
    Output: (batch, seq_len, vocab_size) log scores for every substitution
    """

    OUTPUTS_LEGITIMATE_GAP = False
    OUTPUTS_LEGITIMATE_STOP = False

    def __init__(self, config: "ModelConfig"):
        super().__init__(config)
        cb = config.codon_blosum
        lookup = build_score_matrix(
            config.alphabet.amino_acids, kappa=cb.kappa, epsilon=cb.epsilon,
            w_blosum=cb.w_blosum, w_codon=cb.w_codon, use_mutability=cb.use_mutability)
        # A buffer, not a parameter: there is nothing to train, and registering it keeps the
        # matrix in the checkpoint so eval reconstructs exactly what scored.
        self.register_buffer("lookup", lookup)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.lookup[x]
