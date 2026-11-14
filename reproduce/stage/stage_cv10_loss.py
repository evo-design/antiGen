#!/usr/bin/env python
"""Stage the training curves of antiGen (CV x10) at the 2023-01-01 cutoff, for supp_cv10_loss.

Each of the ten members (split seed i, init seed 100 + i, 256 epochs, a random 10% of the
train corpus held out) wrote <prefix>_val_metrics_by_epoch.json beside its checkpoints:
one record per epoch with `avg_train_loss` (the epoch's mean over batches of the per-batch
NLL per mutation) and the held-out `val.overall.nll_per_mutation` (pooled). Record
`epoch: -1` is the untrained model, with a validation value only. The files hold every
validation example's loss too (~430 MB each), so they are scanned line by line rather than
loaded.

GATE: each member's minimum-validation epoch must be the epoch its _best.pt checkpoint
stores (what --save-best kept).

    python reproduce/stage/stage_cv10_loss.py [VALSEL_DIR]

VALSEL_DIR defaults to $STORE/cutoff_series/2023-01-01/valsel (antigen_vs_s<i>/).
Writes antiGen-figures/data/cv10_loss_2023-01-01/loss_by_epoch.csv.
"""
import csv
import glob
import os
import re
import sys

import torch

STORE = os.environ.get("STORE", "/large_storage/hielab/ispecht/fitness-inference")
ROOT = sys.argv[1] if len(sys.argv) > 1 else f"{STORE}/cutoff_series/2023-01-01/valsel"
FIG = os.environ.get("FIGDATA", os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "antiGen-figures", "data"))
OUT = f"{FIG}/cv10_loss_2023-01-01"

NUM = r"(-?[0-9.eE+-]+|null|NaN)"
EPOCH = re.compile(r'^\s*"epoch":\s*(-?\d+),')
TRAIN = re.compile(rf'^\s*"avg_train_loss":\s*{NUM},')
VAL = re.compile(rf'^\s*"nll_per_mutation":\s*{NUM},')


def scan(path):
    """[(epoch, train_loss or None, val_nll)], from the first nll_per_mutation per record."""
    rows, cur = [], None
    with open(path) as f:
        for line in f:
            m = EPOCH.match(line)
            if m:
                cur = {"epoch": int(m.group(1)), "train": None, "val": None}
                rows.append(cur)
                continue
            if cur is None:
                continue
            m = TRAIN.match(line)
            if m:
                cur["train"] = None if m.group(1) in ("null", "NaN") else float(m.group(1))
                continue
            if cur["val"] is None:
                m = VAL.match(line)
                if m:
                    cur["val"] = float(m.group(1))   # the first one is val.overall
    return rows


out = []
for d in sorted(glob.glob(f"{ROOT}/antigen_vs_s[0-9]"), key=lambda p: int(p.rsplit("s", 1)[1])):
    seed = int(d.rsplit("s", 1)[1])
    js = glob.glob(f"{d}/*_val_metrics_by_epoch.json")
    best_pt = glob.glob(f"{d}/fitness_*_best.pt")
    assert len(js) == 1 and len(best_pt) == 1, (d, js, best_pt)
    rows = scan(js[0])
    assert all(r["val"] is not None for r in rows), d
    trained = [r for r in rows if r["epoch"] >= 0]
    assert len(trained) == 256 and all(r["train"] is not None for r in trained), (d, len(trained))
    best = min(trained, key=lambda r: r["val"])["epoch"]
    stored = torch.load(best_pt[0], map_location="cpu", weights_only=False)["epoch"]
    if best != stored:
        raise SystemExit(f"GATE FAIL: seed {seed}: min-val epoch {best}, _best.pt stores {stored}")
    for r in rows:
        out.append(dict(seed=seed, epoch=r["epoch"] + 1,      # 0 = untrained, 1..256 = after epoch
                        train_nll=("" if r["train"] is None else f"{r['train']:.6f}"),
                        val_nll=f"{r['val']:.6f}", best=int(r["epoch"] == best)))
    print(f"seed {seed}: best epoch {best + 1} of 256, val {min(r['val'] for r in trained):.4f}, "
          f"final train {trained[-1]['train']:.4f} / val {trained[-1]['val']:.4f}")
assert len({r["seed"] for r in out}) == 10, sorted({r["seed"] for r in out})
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/loss_by_epoch.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
    w.writeheader(); w.writerows(out)
print(f"GATE OK: every member's min-validation epoch is its _best.pt epoch\nwrote {OUT}/loss_by_epoch.csv")
