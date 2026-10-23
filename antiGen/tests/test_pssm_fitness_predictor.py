"""Tests for PSSMFitnessPredictor and its use as a gate partner."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch


from antiGen.config import (
    AlphabetConfig,
    LEGACY_AAS,
    GatedFitnessConfig,
    ModelConfig,
    ModelType,
    PSSMConfig,
    TransformerConfig,
)
from antiGen.models.factory import create_model


def _write_synthetic_pssm(tmp_dir: str, seq_len: int, vocab_size: int = 22) -> str:
    """Write a deterministic <tmp_dir>/pssms/root.csv: cell (pos, aa) = pos*100 + aa."""
    pssms_dir = os.path.join(tmp_dir, "pssms")
    os.makedirs(pssms_dir, exist_ok=True)
    path = os.path.join(pssms_dir, "root.csv")
    aas = list("ACDEFGHIKLMNPQRSTVWY*-")[:vocab_size]
    with open(path, "w") as f:
        f.write("pos," + ",".join(aas) + "\n")
        for pos in range(seq_len):
            row = [str(pos)] + [str(pos * 100 + aa_idx) for aa_idx in range(vocab_size)]
            f.write(",".join(row) + "\n")
    return path


def _make_config(seq_len: int, data_dir: str | None = None, path: str | None = None) -> ModelConfig:
    return ModelConfig(
        model_type=ModelType.PSSM,
        seq_len=seq_len,
        alphabet=AlphabetConfig(amino_acids=LEGACY_AAS),
        pssm=PSSMConfig(data_dir=data_dir, path=path),
    )


def test_forward_shape():
    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 12
        _write_synthetic_pssm(tmp, seq_len)
        model = create_model(_make_config(seq_len, data_dir=tmp), device="cpu")
        x = torch.randint(0, 22, (3, seq_len))
        out = model(x)
        assert out.shape == (3, seq_len, 22), out.shape
        assert torch.isfinite(out).all()


def test_zero_trainable_params():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_pssm(tmp, 12)
        model = create_model(_make_config(12, data_dir=tmp), device="cpu")
        assert model.get_num_parameters(trainable_only=True) == 0
        # buffers are persistent state but not trainable
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0


def test_lookup_correctness():
    """Synthetic PSSM with cell(pos, aa) = pos*100 + aa; verify log1p applied."""
    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 5
        _write_synthetic_pssm(tmp, seq_len)
        model = create_model(_make_config(seq_len, data_dir=tmp), device="cpu")
        out = model(torch.zeros(1, seq_len, dtype=torch.long))
        for pos in range(seq_len):
            for aa in range(22):
                expected = float(np.log1p(pos * 100 + aa))
                assert abs(out[0, pos, aa].item() - expected) < 1e-5, (pos, aa)


def test_determinism_across_calls():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_pssm(tmp, 10)
        model = create_model(_make_config(10, data_dir=tmp), device="cpu")
        x = torch.randint(0, 22, (4, 10))
        a = model(x)
        b = model(x)
        assert torch.equal(a, b)


def test_explicit_path_overrides_data_dir():
    with tempfile.TemporaryDirectory() as tmp:
        # Write the PSSM in a non-standard location and pass via path=
        pssms_dir = os.path.join(tmp, "elsewhere")
        os.makedirs(pssms_dir)
        path = os.path.join(pssms_dir, "explicit.csv")
        with open(path, "w") as f:
            aas = list("ACDEFGHIKLMNPQRSTVWY*-")
            f.write("pos," + ",".join(aas) + "\n")
            f.write("0," + ",".join(["7"] * 22) + "\n")
        config = _make_config(seq_len=1, path=path)
        model = create_model(config, device="cpu")
        out = model(torch.zeros(1, 1, dtype=torch.long))
        assert abs(out[0, 0, 0].item() - float(np.log1p(7))) < 1e-5


def test_seq_len_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_pssm(tmp, 12)
        config = _make_config(seq_len=99, data_dir=tmp)  # 99 ≠ 12
        try:
            create_model(config, device="cpu")
        except ValueError as e:
            assert "seq_len" in str(e)
            return
        assert False, "expected ValueError on seq_len mismatch"


def test_missing_path_and_data_dir_raises():
    config = _make_config(seq_len=12)  # no path, no data_dir
    try:
        create_model(config, device="cpu")
    except ValueError as e:
        assert "path" in str(e) or "data_dir" in str(e)
        return
    assert False, "expected ValueError when both path and data_dir are unset"


def test_gated_xfmr_pssm_static_dispatch():
    """End-to-end: GatedFitnessPredictor accepts pssm_fitness_predictor as a partner."""
    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 8
        _write_synthetic_pssm(tmp, seq_len)
        config = ModelConfig(
            model_type=ModelType.GATED_FITNESS_PREDICTOR,
            seq_len=seq_len,
            alphabet=AlphabetConfig(amino_acids=LEGACY_AAS),
            transformer=TransformerConfig(d_model=32, nhead=4, num_layers=1),
            pssm=PSSMConfig(data_dir=tmp),
            gated=GatedFitnessConfig(
                partner_a_type="fitness_predictor",
                partner_b_type="pssm",
                gate_variant="static",
            ),
        )
        model = create_model(config, device="cpu")
        x = torch.randint(0, 22, (2, seq_len))
        out = model(x)
        assert out.shape == (2, seq_len, 22)
        assert torch.isfinite(out).all()


def test_gated_xfmr_pssm_uncertainty_dispatch():
    """Same partners as above with the uncertainty gate variant."""
    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 8
        _write_synthetic_pssm(tmp, seq_len)
        config = ModelConfig(
            model_type=ModelType.GATED_FITNESS_PREDICTOR,
            seq_len=seq_len,
            alphabet=AlphabetConfig(amino_acids=LEGACY_AAS),
            transformer=TransformerConfig(d_model=32, nhead=4, num_layers=1),
            pssm=PSSMConfig(data_dir=tmp),
            gated=GatedFitnessConfig(
                partner_a_type="fitness_predictor",
                partner_b_type="pssm",
                gate_variant="uncertainty",
            ),
        )
        model = create_model(config, device="cpu")
        x = torch.randint(0, 22, (2, seq_len))
        out = model(x)
        assert out.shape == (2, seq_len, 22)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all tests passed")
