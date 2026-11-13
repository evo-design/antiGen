#!/usr/bin/env python
"""Print the exact `antiGen-train` command behind each checkpoint the figures use.

Every checkpoint stores the arguments it was trained with (`train_args`). This reads them
back and writes each one as an `antiGen-train` command line: only the flags that differ
from the defaults, with corpus and checkpoint paths written relative to $CORPORA (the
antiGen-data output root) and $CKPT (the checkpoint root of reproduce/config.sh).

    python reproduce/train_commands.py                    # every family, as a table
    python reproduce/train_commands.py --family 2023-cv10 --sh > train_2023_cv10.sh

The families and the checkpoint directory each one is read from are in FAMILIES below
(the layout the evals in reproduce/eval.sh read). A command reproduces its model up to
GPU nondeterminism; the evals were run on the checkpoints these commands produced.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

STORE = os.environ.get("STORE", "/large_storage/hielab/ispecht/fitness-inference")
CORPORA_ABS = os.environ.get("CORPORA_ABS", "/home/ivan.specht/antiGen-data-revised/output")

#: family -> (checkpoint directory, what it is)
FAMILIES = {
    "2021-roster": (f"{STORE}/ckpt_stage_aa20", "2021 arm: base, ESM-2 (frozen / partial / full / untrained), EVE, Tranception, BLOSUM62"),
    "2021-gate": (f"{STORE}/evals_cd1_2021/ckpt", "2021 arm, with the antiGen / PFM gate"),
    "2021-cv10": (f"{STORE}/ckpt_stage_vs10_aa20", "2021 arm: antiGen (CV x10) members"),
    "2023-roster": (f"{STORE}/ckpt_stage_bg_2023-01-01", "2023 arm: base, gate, ESM-2, EVE / Tranception frozen, BLOSUM62"),
    "2023-cv10": (f"{STORE}/ckpt_stage_vs10_2023-01-01", "2023 arm: antiGen (CV x10) members"),
    "2023-finetunes": (f"{STORE}/ckpt_stage_ft_2023-01-01", "2023 arm: ESM-2 / EVE / Tranception fine-tunes"),
    "2023-codon-blosum": (f"{STORE}/ckpt_codon_blosum_2023-01-01", "2023 arm: codon-aware BLOSUM62 (training-free)"),
    **{f"dataseed-{k}": (f"{STORE}/dataseed_2023-01-01/seed_{k}/stage", f"antiGen (CV x10) on the corpus built with antigen --seed {k}")
       for k in (2, 3)},
    **{f"cutoff-{d}": (f"{STORE}/cutoff_stage_aa20/{d}", f"cutoff series at {d}: base, gate, ESM-2 35M partial, BLOSUM62")
       for d in ["2020-04-01", "2020-07-01", "2020-10-01", "2021-01-01", "2021-04-01", "2021-07-01",
                 "2021-10-01", "2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01", "2023-01-01",
                 "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"]},
}


def _defaults():
    """The antiGen-train parser's (dest -> (default, option strings, action))."""
    import argparse as ap
    import antiGen.train as T
    out = {}
    orig = ap.ArgumentParser.parse_args

    def grab(self, *a, **k):
        for act in self._actions:
            if act.dest != "help":
                out[act.dest] = (act.default, act.option_strings, act)
        raise SystemExit
    ap.ArgumentParser.parse_args = grab
    saved, sys.argv = sys.argv, ["antiGen-train"]
    try:
        T.parse_args()
    except SystemExit:
        pass
    finally:
        ap.ArgumentParser.parse_args, sys.argv = orig, saved
    return out


def _rel(v):
    s = str(v)
    return (s.replace(CORPORA_ABS, "$CORPORA").replace(f"{STORE}/", "$CKPT_TRAIN/"))


def command(train_args, defaults):
    flags = []
    for k, v in train_args.items():
        if k not in defaults:
            continue                       # a field older code wrote that the CLI no longer takes
        dv, opts, act = defaults[k]
        if v is None or v == dv or not opts:
            continue
        o = opts[0]
        if isinstance(act, argparse._StoreTrueAction):
            if v:
                flags.append(o)
        elif isinstance(v, (list, tuple)):
            flags.append(f"{o} " + " ".join(_rel(x) for x in v))
        else:
            flags.append(f"{o} {_rel(v)}")
    return "antiGen-train " + " ".join(flags)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), action="append")
    ap.add_argument("--sh", action="store_true", help="bare commands, one per line")
    a = ap.parse_args()
    defaults = _defaults()
    for fam in a.family or list(FAMILIES):
        d, what = FAMILIES[fam]
        if not a.sh:
            print(f"\n## {fam}: {what}\n#   from {d}")
        for p in sorted(glob.glob(f"{d}/fitness_*.pt")):
            ta = torch.load(p, map_location="cpu", weights_only=False).get("train_args") or {}
            if not ta:
                print(f"# {os.path.basename(p)}: no train_args recorded (an untrained table)")
                continue
            if not a.sh:
                print(f"# {os.path.basename(os.path.realpath(p))}")
            print(command(ta, defaults))


if __name__ == "__main__":
    main()
