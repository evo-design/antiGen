"""A per-virus substitution matrix times a per-position propensity scalar.

THE MODEL. One site-independent substitution matrix ``M[x, a]`` -- the probability that a mutation
away from wild-type residue ``x`` lands on ``a`` -- multiplied by one scalar ``s_i`` per alignment
column saying how mutable that column is::

    score(i, a) = s_i * M[wt_i, a]          emitted as log s_i + log M[wt_i, a]

Both are counted off the TRAINING split of one virus. Nothing is gradient-fitted, so this goes
through ``train.py`` with ``skip_training`` like the other count-based baselines.

WHY IT SITS WHERE IT DOES. The two training-free references already in this repo bracket it:

  * ``pssm`` gives every (position, residue) cell its own count -- fully position-specific, and
    with a few thousand mutations over ~500 columns x 22 residues most cells are 0 or 1, so it is
    mostly memorising which exact substitutions were seen.
  * ``blosum62`` / ``codon_blosum`` give every column the same matrix -- fully position-agnostic,
    and cannot know that some columns never move and others move constantly.

This is the rank-1 factorisation in between: the PSSM's ``(L, E)`` count table approximated as an
outer product of a length-``L`` mutability vector and a shared ``(E, E)`` exchangeability matrix.
It has ``L + E^2`` effective parameters against the PSSM's ``L * E``, so it answers "how much of
the PSSM's advantage is just knowing which positions move, rather than which substitutions
happen there".

THE WILD TYPE IS PER EXAMPLE, NOT PER COLUMN. Every example carries its own ``aa_idx``, and
ancestral genotypes at the same column disagree. So the matrix is accumulated by each example's
own wild type (``M[aa_idx[i], a] += counts[i, a]``) and applied the same way at scoring time, by
gathering rows with the query's own ``aa_idx``. Taking a single consensus wild type per column
would silently merge substitutions out of different residues into one row.

SELF-SUBSTITUTIONS ARE DROPPED. ``M[x, x]`` is forced to zero before and after smoothing: a
mutation to the residue already there is not an event, ``eval_utils.compute_probs`` zeroes those
cells anyway, and leaving the diagonal in would put the largest row mass on a cell that can never
be scored.

SMOOTHING IS ADD-ALPHA ON BOTH FACTORS and is not cosmetic -- it decides what the model says about
things it never saw. A column with zero observed mutations gets propensity ``alpha_position``
rather than 0, so it is merely unlikely rather than impossible; the same for a substitution type
absent from the matrix. With zeros the log is -inf and the flattened softmax in eval would hand
those cells exactly zero probability, which is a far stronger claim than this little model has
evidence for.

GAP AND STOP ARE LEGITIMATE OUTPUTS here, unlike BLOSUM62 and the codon model, which emit
sentinels there. Every number this model produces at a ``-`` or ``*`` column is a smoothed count
of substitutions actually observed into that symbol on this virus's own training split, so the
class attributes stay at the base-class default of True and eval does not NaN-mask those columns.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch

from antiGen.models.base import BaseFitnessPredictor

# Log-space floor for cells with exactly zero mass -- the forced-zero diagonal, and any factor the
# caller un-smooths by passing alpha=0. FINITE on purpose: log(0) = -inf would make the emitted
# logits non-finite, and while eval_utils.compute_probs zeroes wild-type cells anyway, a -inf that
# reaches a softmax whose whole row is -inf yields NaN rather than a uniform, and z-scoring in the
# gated predictor would propagate it. -69 is log(1e-30) -- utterly negligible next to real values
# here (around log(1/(L*E)) ~ -9) and still a normal float. Same reasoning as CodonBlosumConfig's
# epsilon floor, which exists so non-codon-adjacent substitutions are unlikely, not impossible.
_LOG_FLOOR = -69.0

if TYPE_CHECKING:
    from antiGen.config import ModelConfig


def fit_from_examples(examples: Iterable, seq_len: int, vocab_size: int,
                      alpha_matrix: float = 1.0, alpha_position: float = 1.0,
                      verbose: bool = True):
    """Count ``M`` and ``s`` off training examples. -> (log_sub_matrix, log_propensity, stats).

    Accumulated in float64: the counts are small integers but a long virus contributes hundreds of
    thousands of them, and the row sums are divided afterwards.
    """
    M = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)
    s = torch.zeros(seq_len, dtype=torch.float64)
    n_examples = 0
    for ex in examples:
        mat = ex.mat.to(torch.float64)
        if mat.shape != (seq_len, vocab_size):
            raise ValueError(f"example has mat {tuple(mat.shape)}, expected {(seq_len, vocab_size)}")
        s += mat.sum(dim=1)
        # index_add_ over the example's OWN wild-type row per column -- see the module docstring.
        M.index_add_(0, ex.aa_idx.to(torch.long), mat)
        n_examples += 1
    if n_examples == 0:
        raise ValueError("site_propensity needs TRAINING examples to fit; got none")

    total_mut = float(s.sum())
    dead_positions = int((s == 0).sum())
    empty_rows = int((M.sum(dim=1) == 0).sum())

    M.fill_diagonal_(0.0)                       # self-substitutions are not events
    M += alpha_matrix
    M.fill_diagonal_(0.0)                       # ... and get no pseudocount either
    M /= M.sum(dim=1, keepdim=True).clamp_min(1e-12)

    s += alpha_position
    s /= s.sum()

    stats = {
        "n_train_examples": n_examples,
        "total_mutations": total_mut,
        "positions_with_no_mutation": dead_positions,
        "wt_residues_never_mutated": empty_rows,
        "frac_positions_dead": dead_positions / max(1, seq_len),
    }
    if verbose:
        print(f"  [site_propensity] fitted on {n_examples} training examples, "
              f"{total_mut:.0f} mutation events")
        print(f"  [site_propensity] {dead_positions}/{seq_len} positions "
              f"({100 * stats['frac_positions_dead']:.1f}%) had no observed mutation -> "
              f"propensity floor alpha={alpha_position}")
        print(f"  [site_propensity] {empty_rows}/{vocab_size} wild-type residues never mutated "
              f"-> their matrix row is uniform over non-self")
    log_M = M.clamp_min(1e-30).log().clamp_min(_LOG_FLOOR)
    log_s = s.clamp_min(1e-30).log().clamp_min(_LOG_FLOOR)
    return log_M.to(torch.float32), log_s.to(torch.float32), stats


class SitePropensityFitnessPredictor(BaseFitnessPredictor):
    """Substitution matrix x per-position propensity. No gradient-trained parameters.

    The two fitted tensors are registered buffers, so they ride along in ``model_state_dict`` and
    a checkpoint reloads standalone through the ordinary ``create_model`` + ``load_state_dict``
    path -- eval never needs the training examples back.

    Input:  (batch, seq_len) integer AA indices -- the query's OWN wild type, used to pick rows.
    Output: (batch, seq_len, vocab_size) log scores.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        E = len(config.alphabet.amino_acids)
        L = config.seq_len
        # Uniform until fitted: a freshly constructed model is a valid (if useless) predictor, which
        # is what lets eval build the shape first and fill it from the checkpoint afterwards.
        self.register_buffer("log_sub_matrix", torch.full((E, E), -torch.log(torch.tensor(float(E)))))
        self.register_buffer("log_propensity", torch.full((L,), -torch.log(torch.tensor(float(L)))))
        self._fitted = False

    def fit(self, examples, alpha_matrix: float = 1.0, alpha_position: float = 1.0,
            verbose: bool = True) -> dict:
        """Count both factors off `examples` and install them. Returns the fit diagnostics."""
        logM, logs, stats = fit_from_examples(
            examples, seq_len=self.log_propensity.shape[0],
            vocab_size=self.log_sub_matrix.shape[0],
            alpha_matrix=alpha_matrix, alpha_position=alpha_position, verbose=verbose)
        self.log_sub_matrix.copy_(logM.to(self.log_sub_matrix.device))
        self.log_propensity.copy_(logs.to(self.log_propensity.device))
        self._fitted = True
        return stats

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) integer AA indices. Gather each column's row by THAT query's wild type, then
        # add the column's log propensity -- a product in probability space.
        rows = self.log_sub_matrix[x.to(torch.long)]        # (B, L, E)
        return rows + self.log_propensity.view(1, -1, 1)

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return 0
        return self.log_sub_matrix.numel() + self.log_propensity.numel()
