"""ANTIGEN_TOPQ_LEGACY=1 must reproduce the pre-s161 boundary exactly."""
import torch

from antiGen.eval_utils import topq_boundary


def test_legacy_matches_nanquantile(monkeypatch):
    torch.manual_seed(0)
    p = torch.randn(7, 854)
    p[0, :10] = float("nan")
    monkeypatch.setenv("ANTIGEN_TOPQ_LEGACY", "1")
    old = torch.nanquantile(p, 0.9, dim=1, keepdim=True, interpolation="lower")
    assert torch.equal(topq_boundary(p, 0.1, dim=1), old)


def test_default_is_exact(monkeypatch):
    monkeypatch.delenv("ANTIGEN_TOPQ_LEGACY", raising=False)
    p = torch.arange(854, dtype=torch.float32).flip(0).unsqueeze(0)
    vb = topq_boundary(p, 0.1, dim=1)
    assert int((p > vb).sum()) < 85.4 <= int((p >= vb).sum())
