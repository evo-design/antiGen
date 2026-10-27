#!/usr/bin/env python
"""Background maps for the `hybridWxwPssm_s*` members, so K-L can draw their rule.

WHY A BRIDGE IS NEEDED AT ALL. `ensemble_panels.py` builds a rule's background map by
combining its MEMBERS' maps, which `background_discrimination.py` writes for anything
that is a checkpoint. The hybrid members are not checkpoints: `hybrid_baseline.py`
fits a weight per split and emits the blended map directly. It takes its query set from
a SCORE-CACHE directory (`sc.read_queries`), and the background run writes a different
layout. So this writes a score-cache view of the background queries, runs the existing
hybrid script against it unmodified, and converts its output into the background
directory's format. Nothing here re-implements the hybrid.

THREE GATES, because the whole value of the map is that it is the SAME member scored on
different queries:

1. The backbone is REBUILT and checked cell for cell against `backbone_queries.csv` --
   every query's eval_seq and its sha1. The background maps are indexed by position in
   that order, so a reordering would silently attribute one background's predictions to
   another. Rebuilding and comparing is cheaper than trusting that two runs of the same
   builder agree.
2. The selected weights must equal the ones the original hybrid run chose
   (`hybrid_weights.csv`). The weight is fitted on TRAIN splits and does not depend on
   the query set, so a different weight here means a different member, not a different
   evaluation of the same one.
3. Shape, dtype and finiteness on the way out, against the maps already in the
   directory.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from antiGen import score_cache as sc          # noqa: E402
from antiGen.background_tree import build_backbone_from_examples  # noqa: E402


def rebuilt_backbone(args, eval_range, allowed_aas):
    bb = build_backbone_from_examples(args.backbone_examples, args.seq_len, allowed_aas,
                                      args.variant, eval_range=eval_range,
                                      split_date=None)
    return bb


def gate_query_order(bb, bg_dir, eval_range):
    """Every rebuilt query must match the stored one, in the stored order."""
    path = os.path.join(bg_dir, "backbone_queries.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(bb.query_seq):
        raise SystemExit(f"{len(rows)} stored queries, {len(bb.query_seq)} rebuilt")
    bad = 0
    for i, row in enumerate(rows):
        full = bb.query_seq[i]
        ev = "".join(full[p] for p in eval_range)
        if ev != row["eval_seq"]:
            bad += 1
            continue
        if "seq_sha1" in row and row["seq_sha1"]:
            if hashlib.sha1(full.encode()).hexdigest() != row["seq_sha1"]:
                bad += 1
    if bad:
        raise SystemExit(
            f"{bad} of {len(rows)} rebuilt queries disagree with {path}. The background "
            f"maps are indexed by position in that order, so writing a map built on "
            f"this backbone would attribute one genotype's predictions to another.")
    print(f"  [gate 1] {len(rows)} queries match backbone_queries.csv, order and sha1")


def gate_weights(out_dir, reference):
    got = {r["model"]: float(r["weight"])
           for r in csv.DictReader(open(os.path.join(out_dir, "hybrid_weights.csv")))}
    want = {r["model"]: float(r["weight"]) for r in csv.DictReader(open(reference))}
    diff = {k: (want.get(k), v) for k, v in got.items() if want.get(k) != v}
    if diff:
        raise SystemExit(
            "selected weights differ from the original run: " +
            ", ".join(f"{k} {a} -> {b}" for k, (a, b) in sorted(diff.items())) +
            ". The weight is fitted on the TRAIN splits and does not depend on the "
            "query set, so this is a different member rather than the same member on "
            "different queries.")
    print(f"  [gate 2] {len(got)} weights identical to {os.path.basename(reference)}: "
          + ", ".join(f"{k.split('_')[-1]}={v}" for k, v in sorted(got.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background-dir", required=True)
    ap.add_argument("--backbone-examples", required=True)
    ap.add_argument("--train-examples", required=True)
    ap.add_argument("--root-counts", required=True)
    ap.add_argument("--reference-weights", required=True)
    ap.add_argument("--scratch", required=True, help="score-cache view of the queries")
    ap.add_argument("--alphabet", default="ACDEFGHIKLMNPQRSTVWY")
    ap.add_argument("--variant", default="BA.2")
    ap.add_argument("--name-prefix", default="hybridWxwPssm")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse a scratch dir that already holds the maps")
    args = ap.parse_args()

    bg = args.background_dir
    man = json.load(open(os.path.join(bg, "backbone_manifest.json")))
    args.seq_len = int(man["seq_len"])
    eval_range = list(range(man["eval_range_start1"] - 1, man["eval_range_end1"]))
    allowed = man["allowed_aas"]
    if allowed != args.alphabet:
        raise SystemExit(f"manifest alphabet {allowed!r} != --alphabet {args.alphabet!r}")
    n_q = int(man["n_queries"])
    print(f"background dir: {n_q} queries, {len(eval_range)} positions, |A|={len(allowed)}")

    if not args.skip_build:
        bb = rebuilt_backbone(args, eval_range, allowed)
        gate_query_order(bb, bg, eval_range)

        os.makedirs(args.scratch, exist_ok=True)
        aa_idx = np.asarray([[allowed.index(c) for c in s] for s in bb.query_seq],
                            dtype=np.int8)
        empty = (np.zeros(0, np.int32), np.zeros(0, np.int16),
                 np.zeros(0, np.int16), np.zeros(0, np.float32))
        sc.write_queries(args.scratch, aa_idx, empty)
        sha = hashlib.sha1(aa_idx.tobytes()).hexdigest()
        sc.write_query_manifest(args.scratch, dataset="sc2", variant=args.variant,
                                seq_len=args.seq_len, eval_range=eval_range,
                                list_specs=[], sha1=sha, n_queries=len(aa_idx),
                                n_events=0.0)
        print(f"  wrote a score-cache view of {len(aa_idx)} queries to {args.scratch}")

        cmd = [args.python, "antiGen/hybrid_baseline.py",
               "--maps-dir", args.scratch,
               "--train-examples", args.train_examples,
               "--root-counts", args.root_counts,
               "--alphabet", args.alphabet,
               "--variant", args.variant,
               "--name-prefix", args.name_prefix,
               "--output-dir", args.scratch]
        print("  running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    gate_weights(args.scratch, args.reference_weights)

    ref = None
    for fn in sorted(os.listdir(bg)):
        if fn.startswith("scores_") and fn.endswith(".npz"):
            with np.load(os.path.join(bg, fn)) as z:
                ref = z["scores"].shape
            break
    n_written = 0
    for i in range(10):
        name = f"{args.name_prefix}_s{i}"
        src = sc.map_path(args.scratch, name)
        if not os.path.exists(src):
            raise SystemExit(f"hybrid_baseline wrote no map for {name} at {src}")
        arr = np.load(src, mmap_mode="r")
        arr = np.asarray(arr, dtype=np.float32)
        if ref is not None and arr.shape != ref:
            raise SystemExit(f"{name}: shape {arr.shape} != the directory's {ref}")
        n_bad = int((~np.isfinite(arr)).sum())
        dest = os.path.join(bg, f"scores_{name}.npz")
        tmp = dest + ".tmp.npz"
        np.savez_compressed(tmp, scores=arr,
                            eval_range=np.asarray(eval_range, dtype=np.int32))
        os.replace(tmp, dest)
        print(f"  {name:24s} {arr.shape} non-finite={n_bad}")
        n_written += 1
    print(f"  [gate 3] {n_written} map(s) written in the background directory's format")


if __name__ == "__main__":
    main()
