"""Hero-path smoke test: the from-scratch transformer train + inference contract.

Runs with CORE dependencies only (torch/numpy) — no ESM/plmc/MSA — so it always executes
in CI. Verifies the training step is finite and that inference produces a valid
whole-sequence substitution distribution (sums to 1, wild-type cells are zero).
"""
import torch

from antiGen.config import ModelConfig, ModelType, AlphabetConfig, TransformerConfig
from antiGen.models.factory import create_model, save_checkpoint, load_model
from antiGen.eval_utils import compute_probs, compute_loss_batch
from antiGen.infer import score_batch

SEQ_LEN = 12
ALPHA = AlphabetConfig().amino_acids  # the 20 standard residues


def _tiny_model():
    mc = ModelConfig(
        model_type=ModelType.FITNESS_PREDICTOR, seq_len=SEQ_LEN, alphabet=AlphabetConfig(),
        transformer=TransformerConfig(d_model=16, nhead=2, dim_feedforward=32, num_layers=1),
    )
    return create_model(mc, "cpu"), mc


def test_training_step_is_finite():
    model, mc = _tiny_model()
    x = torch.randint(0, len(ALPHA), (2, SEQ_LEN))
    target = torch.zeros(2, SEQ_LEN, len(ALPHA))
    target[0, 3, 5] = 1.0  # one observed substitution
    logits = model(x).float()
    loss = compute_loss_batch(compute_probs(logits, target, x), target)
    assert torch.isfinite(loss)


def test_checkpoint_roundtrips(tmp_path):
    model, mc = _tiny_model()
    p = str(tmp_path / "ckpt.pt")
    save_checkpoint(p, model, mc, torch.optim.Adam(model.parameters()))
    m2, _ = load_model(p, device="cpu")
    assert m2.config.seq_len == SEQ_LEN
    assert m2.config.model_type == ModelType.FITNESS_PREDICTOR


def test_inference_is_a_substitution_distribution():
    model, mc = _tiny_model()
    seq = "ACDEFGHIKLMN"  # length SEQ_LEN
    probs = score_batch(model, [seq], ALPHA, torch.device("cpu"), drop_gap_stop=False)
    assert probs.shape == (1, SEQ_LEN, len(ALPHA))
    # Whole-sequence distribution over substitutions sums to ~1.
    assert abs(probs.sum().item() - 1.0) < 1e-4
    # Wild-type cells carry no probability mass.
    wt_idx = torch.tensor([ALPHA.index(c) for c in seq])
    wt_vals = probs[0, torch.arange(SEQ_LEN), wt_idx]
    assert torch.allclose(wt_vals, torch.zeros(SEQ_LEN), atol=1e-8)
