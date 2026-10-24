"""``validate_per_source`` over a 20-letter alphabet.

NOTES s56. This function read ``alphabet.amino_acids.index('*')`` unconditionally to
find the stop column it NaN-masks before the hit-rate pass. On the 20 standard residues
there is no such column and it raised ``ValueError: substring not found``.

It survived the s53 training arm untouched for a reason worth keeping: every record in
that corpus is pre-cutoff, so ``--val-mode file`` left every validation loader empty and
this function was never called. The first run that asked for a random holdout -- ten of
them, for the s56 arm -- failed in all ten, after building the model and writing the
manifest. A unit test is what that should have been.

What is pinned: it runs at all under 20 letters, both alphabets agree on the quantities
that do not involve the stop column, and the masking still happens when there IS a stop
column and the model does not emit one.
"""
from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from antiGen.config import (AlphabetConfig, LEGACY_AAS, ModelConfig, ModelType,
                            STANDARD_AAS, TransformerConfig)
from antiGen.eval import MutationDataset
from antiGen.eval_utils import validate_per_source
from antiGen.load import Example
from antiGen.models.factory import create_model

L = 24
DEV = torch.device("cpu")


def _model(alphabet):
    cfg = ModelConfig(
        model_type=ModelType.FITNESS_PREDICTOR, seq_len=L,
        alphabet=AlphabetConfig(amino_acids=alphabet),
        transformer=TransformerConfig(d_model=32, nhead=2, dim_feedforward=64,
                                      num_layers=1, dropout=0.0),
    )
    torch.manual_seed(0)
    m = create_model(cfg, DEV)
    m.eval()
    return m


def _loader(alphabet, n=6):
    """A handful of examples over the first 20 residues, so both alphabets can host them."""
    E = len(alphabet)
    g = torch.Generator().manual_seed(1)
    examples = []
    for i in range(n):
        aa = torch.randint(0, 20, (L,), generator=g)
        mat = torch.zeros(L, E)
        for j in range(3):
            pos = int(torch.randint(0, L, (1,), generator=g))
            alt = int(torch.randint(0, 20, (1,), generator=g))
            if alt != int(aa[pos]):
                mat[pos, alt] += 1.0
        examples.append(Example(aa_idx=aa, mat=mat, is_training=False))
    return DataLoader(MutationDataset(examples), batch_size=3, shuffle=False)


def test_it_runs_over_twenty_letters():
    """The regression: this raised ValueError before s56."""
    m = _model(STANDARD_AAS)
    out = validate_per_source(m, {"src": _loader(STANDARD_AAS)}, DEV)
    assert out["per_source"]["src"]["n_mutations"] > 0
    assert out["overall"]["n_mutations"] > 0


def test_it_runs_over_twenty_letters_for_a_model_that_abstains_on_stop():
    """BLOSUM62 sets ``OUTPUTS_LEGITIMATE_STOP = False``, which is what used to make
    ``mask_stop`` true and send the old code to ``stop_idx``. Under 20 letters there is
    no column to mask, so the flag must simply not be consulted."""
    cfg = ModelConfig(model_type=ModelType.BLOSUM62, seq_len=L,
                      alphabet=AlphabetConfig(amino_acids=STANDARD_AAS))
    m = create_model(cfg, DEV)
    m.eval()
    assert not m.OUTPUTS_LEGITIMATE_STOP
    out = validate_per_source(m, {"src": _loader(STANDARD_AAS)}, DEV)
    assert out["per_source"]["src"]["n_mutations"] > 0


def test_the_twenty_two_letter_path_is_unchanged():
    """The gate: bit-exact against ``git show HEAD:`` for both emission behaviours.

    The guard is only safe if it is a no-op wherever a stop column exists, so that is
    checked against the previous implementation rather than reasoned about -- on a model
    that masks the column (BLOSUM62) and one that does not (the transformer).
    """
    import importlib.util
    import subprocess
    import tempfile as _tf
    src = subprocess.run(["git", "show", "HEAD:antiGen/eval_utils.py"],
                         cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
                         capture_output=True, text=True, check=True).stdout
    with _tf.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eval_utils_head.py")
        with open(path, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("eu_head", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)

    cases = [
        ("transformer", lambda: _model(LEGACY_AAS)),
        ("blosum62", lambda: create_model(
            ModelConfig(model_type=ModelType.BLOSUM62, seq_len=L,
                        alphabet=AlphabetConfig(amino_acids=LEGACY_AAS)), DEV)),
    ]
    for name, build in cases:
        a = old.validate_per_source(build().eval(), {"src": _loader(LEGACY_AAS)}, DEV)
        b = validate_per_source(build().eval(), {"src": _loader(LEGACY_AAS)}, DEV)
        assert a["overall"] == b["overall"], name
        pa, pb = a["per_source"]["src"], b["per_source"]["src"]
        assert pa["n_mutations"] == pb["n_mutations"], name
        assert pa["nll_per_mutation"] == pb["nll_per_mutation"], name
        assert pa["hit_rate"] == pb["hit_rate"], name


def test_the_two_alphabets_agree_on_the_shared_columns():
    """Both alphabets must report the same mutation count over the same examples.

    The 20- and 22-letter models differ in output width, so their scores differ; what
    must not differ is the denominator, since every target cell is one of the first 20
    residues.
    """
    outs = {}
    for ab in (STANDARD_AAS, LEGACY_AAS):
        o = validate_per_source(_model(ab), {"src": _loader(ab)}, DEV)
        outs[ab] = o["per_source"]["src"]
    assert outs[STANDARD_AAS]["n_mutations"] == outs[LEGACY_AAS]["n_mutations"]
    assert outs[STANDARD_AAS]["seq_len"] == outs[LEGACY_AAS]["seq_len"] == L
