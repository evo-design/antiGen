"""Zero-shot baselines -- external, training-free, virus-agnostic.

These are deliberately kept SEPARATE from the statistical models
(``statistical_baselines``): nothing here is fit to any antiGen split. Two members:

  * Codon-aware BLOSUM62 (kappa=4) -- BLOSUM62 exchangeability corrected for
    single-nucleotide codon reachability. Training-free; see
    ``antiGen/models/codon_blosum_fitness_predictor.py`` for the full construction.
  * ESM-2 zero-shot -- pretrained ESM-2 650M masked-marginal logits, weights
    loaded but never fine-tuned on antiGen data.

Names/category are in ``baseline_names`` (keys ``codon_blosum``, ``esm2_zeroshot``).

On kappa=4
----------
kappa is the transition:transversion weighting in the codon term. It is a single
virus-agnostic external constant, NOT tuned on the eval metric: it sits where two
on-data analyses point -- the fitted ``substitution`` baseline recovers kappa in
4.3-9.2 across viruses, and a grid fit of codon(kappa) against the trained models'
implied substitution matrices has its rank optimum at kappa ~ 2.5-4 (the R^2 surface
is near-flat, so only the rank is informative). 4 is also the textbook RNA-virus
ts/tv ratio. See experiments/multitree/NOTES.md (substitution-matrix analysis).
"""
from __future__ import annotations

KAPPA_DEFAULT = 4.0
EPSILON_DEFAULT = 0.05


def codon_blosum_log_scores(alphabet: str, kappa: float = KAPPA_DEFAULT,
                            epsilon: float = EPSILON_DEFAULT):
    """(E, E) codon-aware BLOSUM62 log-score matrix for ``alphabet``.

    Thin wrapper over the canonical builder so callers get the zero-shot matrix
    without reaching into antiGen.models directly. Imported lazily to keep this
    module importable (and droppable) without pulling antiGen at module load.

    Returns a torch tensor indexed (wild_type, alt) of log scores, with the
    wild-type and gap/stop columns set to -inf exactly as the predictor does.
    """
    from antiGen.models.codon_blosum_fitness_predictor import build_score_matrix
    return build_score_matrix(alphabet, kappa=kappa, epsilon=epsilon)


# ESM-2 zero-shot is loaded via antiGen.create_model with a pretrained config and
# NO state_dict load (see e.g. score_fig4_denovo_strike.py / score_zeroshot_fig13.py);
# it is not reimplemented here because it is just "the pretrained model, untrained".
__all__ = ["KAPPA_DEFAULT", "EPSILON_DEFAULT", "codon_blosum_log_scores"]
