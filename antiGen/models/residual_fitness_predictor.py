"""Late fusion of the persistence counts with a learned sequence residual.

The persistence baseline -- this time window's split between a genotype's child
substitutions, smoothed by a pseudocount, used to predict the next window's -- is hard to
beat because it carries information the sequence does not: which lineage is currently
large. This model puts that term in the architecture instead of competing with it, so the
network is only ever asked for the part the counts cannot supply.

    s_j = beta * log(c_j + alpha) + lam * f(sequence)[pos_j, alt_j]

Under the conditional objective only differences of s within an example matter, so
exponentiating says

    n_j  proportional to  (c_j + alpha)^beta * exp(lam * f_j)

i.e. next window's share is this window's share times a sequence-derived multiplier. That
is discrete-time replicator dynamics, and it re-purposes the network's output from "which
substitutions occur" to "which of them is growing" -- a growth advantage, which is the
quantity the counts leave undetermined.

Two consequences worth keeping in view:

  * at beta = 1, lam = 0 this is *exactly* the persistence baseline, so training from
    lambda_init = 0 starts at that baseline's loss;
  * with no counts supplied the prevalence term is constant across cells and cancels in
    the softmax, so the model degrades to lam * f(sequence) -- the sequence-only
    predictor. Scoring a de novo sequence therefore still works, but it is answering a
    different question from the one this model was trained on.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from antiGen.config import ModelConfig, ModelType
from antiGen.models.base import BaseFitnessPredictor


class ResidualFitnessPredictor(BaseFitnessPredictor):
    """Persistence counts plus a gated sequence residual (see module docstring)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        rc = config.residual
        if rc.alpha <= 0:
            raise ValueError(
                f"residual.alpha must be positive (got {rc.alpha}); it is the pseudocount "
                f"that keeps log(c + alpha) finite for the c = 0 cells, which are common."
            )
        self.alpha = float(rc.alpha)

        # The inner predictor is the ordinary from-scratch transformer, unchanged, built
        # from the same ModelConfig so --d-model / --num-layers route through as usual.
        from antiGen.models.fitness_predictor import FitnessPredictor
        inner_config = ModelConfig(
            model_type=ModelType.FITNESS_PREDICTOR,
            seq_len=config.seq_len,
            alphabet=config.alphabet,
            transformer=config.transformer,
        )
        self.inner = FitnessPredictor(inner_config)

        beta = torch.tensor(float(rc.beta_init))
        lam = torch.tensor(float(rc.lambda_init))
        self.beta = nn.Parameter(beta, requires_grad=rc.learn_beta)
        self.lam = nn.Parameter(lam, requires_grad=rc.learn_lambda)

    def forward(
        self, x: torch.Tensor, cur_mat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scores of shape (batch, seq_len, vocab_size).

        Args:
            x: (batch, seq_len) integer AA indices, or one-hot as the base class allows.
            cur_mat: (batch, seq_len, vocab_size) counts of samples from each example's
                own time window descended from each candidate substitution. None scores
                without the prevalence term (see module docstring).
        """
        if x.dim() == 3:
            x = torch.argmax(x, dim=-1)
        residual = self.lam * self.inner(x)
        if cur_mat is None:
            return residual
        prevalence = self.beta * torch.log(cur_mat.to(residual.dtype) + self.alpha)
        return prevalence + residual

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """The sequence-only half, for callers holding the one-argument contract."""
        return self.lam * self.inner(x)
