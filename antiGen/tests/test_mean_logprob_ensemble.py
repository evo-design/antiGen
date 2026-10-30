"""The val-selected bar: N checkpoints as the mean of their per-query log-probs.

The rule is `ensemble_rules`' ``logsoftmax`` / ``arith``, expressed as a MODEL so the
eval scores it through the same masks, candidate pool and quantile machinery as every
other rung. What has to hold for that to be the same rule:

1. a ONE-member ensemble cannot change any rank (the repo's standard ensemble gate), and
2. the mean really is taken in LOG space, not probability space -- the two differ, and
   only one of them is the published rule.
"""
import torch

from antiGen.eval_utils import MeanLogProbEnsemble, compute_probs


class Toy(torch.nn.Module):
    """A stand-in model: fixed logits per query, independent of the input values."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, aa_batch):
        return self.logits[: aa_batch.shape[0]]


def _ranks(x):
    """Within-query ranking of the flattened cells."""
    return torch.argsort(torch.argsort(x.flatten(1), dim=1), dim=1)


def test_one_member_is_a_no_op():
    torch.manual_seed(0)
    lg = torch.randn(4, 7, 5)
    ens = MeanLogProbEnsemble([Toy(lg)])
    aa = torch.zeros(4, 7, dtype=torch.long)
    assert torch.equal(_ranks(ens(aa)), _ranks(lg)), (
        "a one-member ensemble reordered its own member's cells")


def test_composition_with_compute_probs_preserves_the_ranking():
    """`compute_probs` softmaxes over the same flattened axis, so ranks survive it.

    This is the step that lets the ensemble be a model at all: the eval re-normalises
    whatever a model returns, and that re-normalisation must not undo the rule.
    """
    torch.manual_seed(1)
    lg = torch.randn(3, 6, 4)
    ens = MeanLogProbEnsemble([Toy(lg)])
    aa = torch.full((3, 6), 3, dtype=torch.long)       # a wild-type outside 0..2
    tgt = torch.zeros(3, 6, 4)
    out = ens(aa)
    probs = compute_probs(out, tgt, aa)
    # Compare only the non-wild-type cells, which are the ones compute_probs keeps.
    keep = torch.ones_like(probs, dtype=torch.bool)
    keep.scatter_(2, aa.unsqueeze(-1), False)
    a = torch.where(keep, out, torch.full_like(out, -1e9))
    b = torch.where(keep, probs, torch.full_like(probs, -1.0))
    assert torch.equal(_ranks(a), _ranks(b)), (
        "compute_probs' per-query softmax changed the ensemble's ranking")


def test_the_mean_is_taken_in_log_space():
    """Mean-of-log-probs, not mean-of-probs: on two disagreeing members they differ."""
    # Cell 0 is member A's top call and is all but vetoed by member B. In log space
    # the veto drags it below cell 1; averaging probabilities, A's near-certainty
    # survives the average and cell 0 stays on top. That disagreement is the point.
    a = torch.tensor([[[10.0, 1.0, 0.0, 0.0]]])
    b = torch.tensor([[[-10.0, 1.0, 0.0, 0.0]]])
    ens = MeanLogProbEnsemble([Toy(a), Toy(b)])
    aa = torch.full((1, 1), 3, dtype=torch.long)
    got = ens(aa)

    la = torch.log_softmax(a.flatten(1), dim=1)
    lb = torch.log_softmax(b.flatten(1), dim=1)
    assert torch.allclose(got.flatten(1), (la + lb) / 2, atol=1e-6)

    mean_of_probs = (la.exp() + lb.exp()) / 2
    assert not torch.equal(_ranks(got), _ranks(mean_of_probs.reshape(got.shape))), (
        "this pair was chosen because the two means rank differently; if they now "
        "agree the test has stopped discriminating and needs new numbers")


def test_members_are_registered_so_device_moves_apply():
    """A plain list would leave members on the CPU when the ensemble is .to(device)."""
    ens = MeanLogProbEnsemble([Toy(torch.zeros(1, 2, 3)), Toy(torch.zeros(1, 2, 3))])
    assert len(list(ens.children())) == 1 and len(ens.members) == 2
    assert any(b.numel() for b in ens.buffers()), "member buffers are not registered"
