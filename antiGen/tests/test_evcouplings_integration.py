"""Integration test for EVcouplings with actual MSA fitting.

Creates a small synthetic MSA (20 sequences, 15 positions) with realistic
variation, fits a mean-field DCA model, and verifies the full pipeline
produces sensible, non-degenerate scores.
"""
import sys
import tempfile
from pathlib import Path

import torch


import shutil
import pytest

pytest.importorskip("evcouplings", reason="EVcouplings tests require the evcouplings package")
pytestmark = pytest.mark.skipif(shutil.which("plmc") is None,
                                reason="EVcouplings integration requires the external plmc binary")

from antiGen.config import AlphabetConfig, EVcouplingsConfig, ModelConfig, ModelType, LEGACY_AAS
from antiGen.models.evcouplings_fitness_predictor import EVcouplingsFitnessPredictor, _SENTINEL

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
    "ACDEFGHIALNPQR-",
    "ACDEFGHIKAMNPQR",
    "ACDEFGHIKLAPQR-",
    "ACDEFGHIKLMAQR-",
    "ACDEFGHIKLMNAR-",
    "ACDEFGHIKLMNPAR",
    "WCDEAGHIKLMNPQR",
    "ADDEFGAIKLMNPQR",
    "ACEEFGHIALNPQR-",
    "ACDDFGHIKAMNPQR",
]


def _write_test_files(tmp_dir):
    """Write a small FASTA and metadata TSV to temp directory."""
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


def _make_predictor(fasta_path, metadata_path):
    config = ModelConfig(
        model_type=ModelType.EVCOUPLINGS,
        seq_len=SEQ_LEN,
        alphabet=AlphabetConfig(amino_acids=LEGACY_AAS),
        evcouplings=EVcouplingsConfig(
            msa_path=fasta_path,
            metadata_path=metadata_path,
            date_cutoff="2021-01-01",
            theta=1.0,
            pseudo_count=0.5,
            max_seqs=0,
        ),
    )
    return EVcouplingsFitnessPredictor(config)


def test_full_pipeline_non_degenerate():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path, metadata_path = _write_test_files(tmp_dir)
        predictor = _make_predictor(fasta_path, metadata_path)

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
            unique = len(set(round(s, 8) for s in aa_scores))
            assert unique > 1, (
                f"Position {pos}: all mapped AA scores are identical ({aa_scores[0]})"
            )


def test_full_pipeline_wildtype_zero():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path, metadata_path = _write_test_files(tmp_dir)
        predictor = _make_predictor(fasta_path, metadata_path)

        internal_alpha_map = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY*-")}
        x = torch.tensor([[internal_alpha_map[c] for c in REFERENCE_SEQ]])

        output = predictor(x)

        for pos, c in enumerate(REFERENCE_SEQ):
            if c == "-":
                continue
            idx = internal_alpha_map[c]
            wt_score = output[0, pos, idx].item()
            assert abs(wt_score) < 1e-4, (
                f"Position {pos} ({c}): WT score = {wt_score}, expected ~0"
            )


def test_full_pipeline_variant_differs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path, metadata_path = _write_test_files(tmp_dir)
        predictor = _make_predictor(fasta_path, metadata_path)

        internal_alpha_map = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY*-")}
        seq1 = REFERENCE_SEQ
        seq2 = "WCDEFGHIKLMNPQR"
        x = torch.tensor([
            [internal_alpha_map[c] for c in seq1],
            [internal_alpha_map[c] for c in seq2],
        ])

        output = predictor(x)

        diffs = 0
        for pos in range(1, SEQ_LEN):
            for c in "ACDEFGHIKLMNPQRSTVWY":
                idx = internal_alpha_map[c]
                s1 = output[0, pos, idx].item()
                s2 = output[1, pos, idx].item()
                if s1 != _SENTINEL and s2 != _SENTINEL and abs(s1 - s2) > 1e-6:
                    diffs += 1
        assert diffs > 0, "Coupling terms had no effect across sequences"


def test_full_pipeline_score_range():
    """Scores should span a meaningful range, not be near-zero or near-constant."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path, metadata_path = _write_test_files(tmp_dir)
        predictor = _make_predictor(fasta_path, metadata_path)

        internal_alpha_map = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY*-")}
        x = torch.tensor([[internal_alpha_map[c] for c in REFERENCE_SEQ]])

        output = predictor(x)

        finite_scores = output[output != _SENTINEL]
        score_range = finite_scores.max().item() - finite_scores.min().item()
        assert score_range > 0.1, f"Score range too narrow: {score_range}"
        num_unique = len(torch.unique(torch.round(finite_scores * 1e4)))
        assert num_unique > 10, f"Only {num_unique} unique score values across all positions"


if __name__ == "__main__":
    test_full_pipeline_non_degenerate()
    print("PASS: test_full_pipeline_non_degenerate")
    test_full_pipeline_wildtype_zero()
    print("PASS: test_full_pipeline_wildtype_zero")
    test_full_pipeline_variant_differs()
    print("PASS: test_full_pipeline_variant_differs")
    test_full_pipeline_score_range()
    print("PASS: test_full_pipeline_score_range")
    print("\nAll integration tests passed.")
