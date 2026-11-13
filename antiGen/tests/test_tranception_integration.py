"""Integration test for Tranception with actual MSA pre-training.

Creates a small synthetic MSA, pre-trains the autoregressive LM, and verifies
the full pipeline produces sensible, non-degenerate scores in all three
scoring directions.
"""
import sys
import tempfile
from pathlib import Path

import torch


from antiGen.config import AlphabetConfig, TranceptionConfig, ModelConfig, ModelType, LEGACY_AAS
from antiGen.models.tranception_fitness_predictor import (
    TranceptionFitnessPredictor, _SENTINEL, TRANCEPTION_Q,
)

REFERENCE_SEQ = "ACDEFGHIKLMNPQR"
SEQ_LEN = len(REFERENCE_SEQ)

VARIANT_SEQS = [
    "ACDEFGHIKLMNPQR",
    "WCDEFGHIKLMNPQR",
    "ADDEFGHIKLMNPQR",
    "ACEEFGHIKLMNPQR",
    "ACDDFGHIKLMNPQR",
    "ACDEAGHIKLMNPQR",
    "ACDEFWHIKLMNPQR",
    "ACDEFGAIKLMNPQR",
    "ACDEFGHLKLMNPQR",
    "ACDEFGHIAKLMNPQR"[:SEQ_LEN],
    "ACDEFGHIKAMNPQR",
    "ACDEFGHIKLAPQR-"[:SEQ_LEN],
    "ACDEFGHIKLMAQR-"[:SEQ_LEN],
    "ACDEFGHIKLMNAR-"[:SEQ_LEN],
    "ACDEFGHIKLMNPAR",
    "WCDEAGHIKLMNPQR",
    "ADDEFGAIKLMNPQR",
    "ACEEFGHIAKLMNPQR"[:SEQ_LEN],
    "ACDDFGHIKAMNPQR",
]


def _write_test_files(tmp_dir):
    fasta_path = Path(tmp_dir) / "test.fasta"
    metadata_path = Path(tmp_dir) / "metadata.tsv"

    with open(fasta_path, "w") as f:
        f.write(f">reference\n{REFERENCE_SEQ}\n")
        for i, seq in enumerate(VARIANT_SEQS):
            f.write(f">strain_{i}\n{seq}\n")

    with open(metadata_path, "w") as f:
        f.write("strain\tdate\n")
        f.write("reference\t2020-01-01\n")
        for i in range(len(VARIANT_SEQS)):
            f.write(f"strain_{i}\t2020-06-{(i % 28) + 1:02d}\n")

    return str(fasta_path), str(metadata_path)


def _make_predictor(fasta_path, metadata_path, direction="both"):
    config = ModelConfig(
        model_type=ModelType.TRANCEPTION,
        seq_len=SEQ_LEN,
        alphabet=AlphabetConfig(amino_acids=LEGACY_AAS),
        tranception=TranceptionConfig(
            msa_path=fasta_path,
            metadata_path=metadata_path,
            date_cutoff="2021-01-01",
            theta=1.0,
            n_embd=64,
            n_head=4,
            n_layer=2,
            dropout=0.0,
            pretrain_steps=500,
            pretrain_lr=3e-4,
            pretrain_batch_size=8,
            freeze_backbone=True,
            max_seqs=0,
            scoring_direction=direction,
        ),
    )
    return TranceptionFitnessPredictor(config)


def test_non_degenerate(direction="both"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path, metadata_path = _write_test_files(tmp_dir)
        predictor = _make_predictor(fasta_path, metadata_path, direction)

        internal_alpha_map = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY*-")}
        x = torch.tensor([[internal_alpha_map[c] for c in REFERENCE_SEQ]])

        output = predictor(x)
        assert output.shape == (1, SEQ_LEN, len(internal_alpha_map))

        for pos in range(SEQ_LEN):
            aa_scores = []
            for c in "ACDEFGHIKLMNPQRSTVWY":
                idx = internal_alpha_map[c]
                s = output[0, pos, idx].item()
                if s != _SENTINEL:
                    aa_scores.append(s)
            unique = len(set(round(s, 6) for s in aa_scores))
            assert unique > 1, (
                f"Position {pos}: all mapped AA scores identical ({aa_scores[0]})"
            )


def test_different_sequences_give_different_scores():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path, metadata_path = _write_test_files(tmp_dir)
        predictor = _make_predictor(fasta_path, metadata_path, "both")

        internal_alpha_map = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY*-")}
        seq1 = REFERENCE_SEQ
        seq2 = "WCDEFGHIKLMNPQR"
        x = torch.tensor([
            [internal_alpha_map[c] for c in seq1],
            [internal_alpha_map[c] for c in seq2],
        ])

        output = predictor(x)

        diffs = 0
        for pos in range(SEQ_LEN):
            for c in "ACDEFGHIKLMNPQRSTVWY":
                idx = internal_alpha_map[c]
                s1 = output[0, pos, idx].item()
                s2 = output[1, pos, idx].item()
                if s1 != _SENTINEL and s2 != _SENTINEL and abs(s1 - s2) > 1e-6:
                    diffs += 1
        assert diffs > 0, "Scores identical despite different input sequences"


def test_forward_backward_differ():
    """Forward-only and backward-only should give different scores (different conditioning)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path, metadata_path = _write_test_files(tmp_dir)
        predictor_fwd = _make_predictor(fasta_path, metadata_path, "forward")

        # Build a backward predictor reusing the same weights
        predictor_bwd = _make_predictor(fasta_path, metadata_path, "backward")

        internal_alpha_map = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY*-")}
        x = torch.tensor([[internal_alpha_map[c] for c in REFERENCE_SEQ]])

        out_fwd = predictor_fwd(x)
        out_bwd = predictor_bwd(x)

        diffs = 0
        for pos in range(SEQ_LEN):
            for c in "ACDEFGHIKLMNPQRSTVWY":
                idx = internal_alpha_map[c]
                sf = out_fwd[0, pos, idx].item()
                sb = out_bwd[0, pos, idx].item()
                if sf != _SENTINEL and sb != _SENTINEL and abs(sf - sb) > 1e-6:
                    diffs += 1
        assert diffs > 0, "Forward and backward scores are identical"


def test_output_are_log_probs():
    """Scores should be log-probabilities (negative, sum to ~1 after exp)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path, metadata_path = _write_test_files(tmp_dir)
        predictor = _make_predictor(fasta_path, metadata_path, "forward")

        internal_alpha_map = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY*-")}
        x = torch.tensor([[internal_alpha_map[c] for c in REFERENCE_SEQ]])

        output = predictor(x)

        for pos in range(SEQ_LEN):
            log_probs = []
            for c in "ACDEFGHIKLMNPQRSTVWY":
                idx = internal_alpha_map[c]
                s = output[0, pos, idx].item()
                if s != _SENTINEL:
                    log_probs.append(s)
                    assert s <= 0, f"Log-prob at pos {pos}, AA {c} is positive: {s}"
            prob_sum = sum(torch.tensor(log_probs).exp().tolist())
            assert abs(prob_sum - 1.0) < 0.01, (
                f"Position {pos}: probabilities sum to {prob_sum}, expected ~1.0"
            )


if __name__ == "__main__":
    test_non_degenerate("both")
    print("PASS: test_non_degenerate (both)")
    test_non_degenerate("forward")
    print("PASS: test_non_degenerate (forward)")
    test_non_degenerate("backward")
    print("PASS: test_non_degenerate (backward)")
    test_different_sequences_give_different_scores()
    print("PASS: test_different_sequences_give_different_scores")
    test_forward_backward_differ()
    print("PASS: test_forward_backward_differ")
    test_output_are_log_probs()
    print("PASS: test_output_are_log_probs")
    print("\nAll Tranception integration tests passed.")
