#!/usr/bin/env python
"""Print "<best epoch> <best NLL> <last epoch> <holdout mutations>" for one run dir.

The budget check the val-selection drivers run after training. A run whose best epoch IS
its last epoch has not selected an epoch at all: its held-out curve was still descending
when the budget ran out, so the arm collapses to "train to the budget", which is what the
base rung already is. The driver compares columns 1 and 3 and fails the job when they
agree -- see NOTES s66, where nine of ten runs at the 2023-01-01 cutoff did exactly that
under a 32-epoch budget while the job exited OK.

Reads `<prefix>_val_metrics_by_epoch.json`, which `train.py` writes per run. Epoch -1 is
the pre-training baseline row and is not a selectable epoch. Prints a bare "?" when the
file is absent or carries no scored epoch, which the caller treats as "cannot tell"
rather than as a pass.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main(run_dir):
    hits = sorted(glob.glob(os.path.join(run_dir, "*_val_metrics_by_epoch.json")))
    rows = [r for r in (json.load(open(hits[-1])) if hits else [])
            if r.get("epoch", -1) >= 0
            and (r.get("val") or {}).get("overall", {}).get("nll_per_mutation") is not None]
    if not rows:
        print("?")
        return 0
    best = min(rows, key=lambda r: r["val"]["overall"]["nll_per_mutation"])
    print(best["epoch"],
          round(best["val"]["overall"]["nll_per_mutation"], 4),
          max(r["epoch"] for r in rows),
          round(best["val"]["overall"]["n_mutations"]))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: best_epoch.py <run-directory>")
    sys.exit(main(sys.argv[1]))
