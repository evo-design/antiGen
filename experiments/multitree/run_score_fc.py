#!/usr/bin/env python
"""Run Ivan's score_forecast.py against the feature-complete antiGen (see run_train_fc.py).

score_forecast.py imports antiGen.eval_utils helpers (fit_substitution_baseline,
WhereTimesWhichPredictor, validate_per_source, ...) that live only in Ivan's checkout; the
env's editable install is this repo and lacks them. Prepend Ivan's repo, apply the
torchvision shim, then hand off. All CLI args pass straight through to score_forecast.
"""
import os
import runpy
import sys

IVAN_REPO = "/home/ivan.specht/antiGen"
SCORER = f"{IVAN_REPO}/experiments/multitree/score_forecast.py"
sys.path.insert(0, IVAN_REPO)

import transformers.utils.import_utils as _iu  # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

import antiGen  # noqa: E402
if not os.path.dirname(antiGen.__file__).startswith(IVAN_REPO):
    sys.exit(f"ABORT: antiGen resolved at {antiGen.__file__}, not under {IVAN_REPO}")

# score_forecast reads sys.argv[1:]; run_name=__main__ triggers its main().
runpy.run_path(SCORER, run_name="__main__")
