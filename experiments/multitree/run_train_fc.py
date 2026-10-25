#!/usr/bin/env python
"""Forecast-arm launcher: run the FEATURE-COMPLETE antiGen.train from Ivan's checkout.

Why this exists. The `av-multitree` forecast pipeline (`--val-mode random`, `--save-best`,
`--save-best-per-source`, `--dump-cells`, `--experiment`, multi-source `--examples NAME=PATH`)
lives in `/home/ivan.specht/antiGen`. The copy editable-installed into the `ag_env` conda env
is this repo's checkout, which is behind it and lacks those flags -- in particular `--save-best`,
without which `train.py` persists only the final (overfit) checkpoint, which the forecast
experiment must not use.

PYTHONPATH does not override the setuptools editable finder, but an explicit
`sys.path.insert(0, ...)` does (verified). So we prepend Ivan's repo, then apply the same
torchvision shim his `run_train.py` uses (transformers 4.57 imports torchvision on the way to
`EsmForMaskedLM`, and the env's torchvision is ABI-mismatched to torch), then hand off to
`antiGen.train.main`. Every argument is passed straight through.

Delete once this repo's checkout catches up and `antiGen.train` here has `--save-best` et al.
"""
import sys

IVAN_REPO = "/home/ivan.specht/antiGen"
sys.path.insert(0, IVAN_REPO)

import transformers.utils.import_utils as _iu  # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

import antiGen  # noqa: E402
import os as _os
_pkg = _os.path.dirname(antiGen.__file__)
if not _pkg.startswith(IVAN_REPO):
    sys.exit(f"ABORT: resolved antiGen at {_pkg}, not under {IVAN_REPO}; "
             f"refusing to run the forecast arm against the stale checkout.")

from antiGen.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
