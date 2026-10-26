#!/usr/bin/env python
"""Write each cached model's score map AT THE VARIANT ROOT, in the shape panel K reads.

**Why this exists.** `fig2_sc2_benchmarks.R`'s DMS-correlation panel reads
``data/sc2/<model>/variant_root_predictions_*.csv`` -- one (seq_len, 22) matrix per
model, the model's prediction on the variant root genotype. Nothing in the package
writes those files any more; the committed ones are dated 2026-06-22 and their producer
is gone. They do not need a GPU to recreate: the variant root is already one of the
queries in the list score cache, so a model's row there IS its prediction on that
genotype, and the cache holds every rung the figure draws.

**The root is located by GENOTYPE, not by a remembered index.** The cache is rebuilt
whenever the query set changes; an index that quietly moved would write a different
background's predictions under the right filename, and nothing downstream could tell.
Duplicate rows of the same genotype must agree, which is checked.

**The output keeps the published shape** -- `seq_len` rows over the 22-token frame --
so `read_model_predictions` and `mask_wildtype` work unchanged. Two things are NaN by
construction and mean "this model says nothing here": every position outside the eval
range, because the cache only holds the RBD, and the `*`/`-` columns, because the
20-letter arm has no such emission. `read_model_predictions` already maps -Inf to NA and
the correlation is taken pairwise-complete, so NaN is the representation that panel
already understands.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from antiGen import score_cache as sc
from antiGen.config import LEGACY_AAS, STANDARD_AAS
from antiGen.eval_utils import RBD_RANGE


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--maps-dir', required=True, help='the --dump-scores cache')
    ap.add_argument('--examples', required=True,
                    help='corpus holding the IS_VARIANT_ROOT record')
    ap.add_argument('--variant', default='21L (BA.2)')
    ap.add_argument('--alphabet', default=STANDARD_AAS,
                    help='the frame the cached maps were written in')
    ap.add_argument('--seq-len', type=int, default=1273)
    ap.add_argument('--output-dir', required=True,
                    help='written as <output-dir>/<model>/variant_root_predictions_<model>.csv')
    ap.add_argument('--ref-out', default=None,
                    help='also write the variant root sequence here, for mask_wildtype')
    ap.add_argument('--models', default=None, help='comma-separated substrings to keep')
    ap.add_argument('--rule-json', default=None, metavar='JSON',
                    help="Also write a directory per frozen combination rule, named "
                         "ens_<rule_id>, so a figure that draws the combination bars "
                         "can read them here on the same footing as a checkpoint. The "
                         "rule is applied to the members' maps AT THE ROOT, which is "
                         "the same arithmetic the bars themselves use.")
    args = ap.parse_args()

    # ── the variant root genotype, streamed out of the corpus ───────────────
    root, in_flags, pending = None, False, False
    with open(args.examples) as fh:
        for line in fh:
            s = line.strip()
            if s == "IS_VARIANT_ROOT:":
                in_flags, pending = True, False
                continue
            if in_flags:
                if "," in s and s.rsplit(",", 1)[-1] in ("True", "False"):
                    if s.endswith(",True") and (
                            args.variant is None or s.rsplit(",", 1)[0] == args.variant):
                        pending = True
                    continue
                in_flags = False
                if pending and s:
                    root = s
                    break
    if root is None:
        raise SystemExit(f"{args.examples}: no IS_VARIANT_ROOT record for "
                         f"{args.variant!r}")
    if len(root) != args.seq_len:
        raise SystemExit(f"variant root is {len(root)} residues, expected {args.seq_len}")
    print(f"variant root: {args.variant}, {len(root)} residues")

    aa_idx = sc.read_queries(args.maps_dir)[0]
    want = np.array([args.alphabet.index(c) for c in root], dtype=aa_idx.dtype)
    hits = np.where((aa_idx == want).all(axis=1))[0]
    if len(hits) == 0:
        raise SystemExit(f"the variant root is not among the {aa_idx.shape[0]} queries "
                         f"in {args.maps_dir}")
    print(f"  dumped query {int(hits[0])}"
          + (f" (+{len(hits) - 1} duplicate row(s))" if len(hits) > 1 else ""))

    eval_range = list(RBD_RANGE)
    keep = [m.strip() for m in args.models.split(',')] if args.models else None
    models = [m for m in sc.list_models(args.maps_dir)
              if keep is None or any(k in m for k in keep)]
    print(f"  {len(models)} model(s) in the cache")

    os.makedirs(args.output_dir, exist_ok=True)
    for name in sorted(models):
        arr, _n_neg = sc.load_map(args.maps_dir, name, mmap=True, neginf_to_nan=True)
        rows = [np.asarray(arr[int(h)], dtype=np.float64) for h in hits]
        for r in rows[1:]:
            d = np.nanmax(np.abs(np.nan_to_num(rows[0] - r, nan=0.0)))
            if d > 1e-6:
                raise SystemExit(f"{name}: duplicate rows of the variant root differ "
                                 f"by {d:.2e}; the cache and the corpus disagree about "
                                 f"this query")
        block = rows[0]                                   # (len(eval_range), E)
        if block.shape[0] != len(eval_range):
            raise SystemExit(f"{name}: map covers {block.shape[0]} positions, "
                             f"expected {len(eval_range)}")

        full = np.full((args.seq_len, len(LEGACY_AAS)), np.nan)
        for j, aa in enumerate(args.alphabet):
            full[eval_range, LEGACY_AAS.index(aa)] = block[:, j]

        d = os.path.join(args.output_dir, name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"variant_root_predictions_{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(LEGACY_AAS))
            for i in range(args.seq_len):
                w.writerow(["" if np.isnan(v) else f"{v:.8f}" for v in full[i]])
        print(f"  wrote {path}")

    if args.rule_json:
        import json
        from antiGen import ensemble_rules as er
        import torch
        specs = json.load(open(args.rule_json))
        if isinstance(specs, dict):
            specs = [specs]
        members = sorted({m for sp in specs for m in sp["members"]})
        maps = {}
        for m in members:
            arr, _ = sc.load_map(args.maps_dir, m, mmap=True, neginf_to_nan=True)
            maps[m] = torch.as_tensor(
                np.asarray(arr[int(hits[0])], dtype=np.float32)).unsqueeze(0)
        for spec in specs:
            spec = dict(spec)
            for k in ("figure_label", "figure_group", "family"):
                spec.pop(k, None)
            rule = er.Rule(family="frozen",
                           **{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in spec.items()})
            block = er.apply(maps, rule)[0].numpy().astype(np.float64)
            full = np.full((args.seq_len, len(LEGACY_AAS)), np.nan)
            for j, aa in enumerate(args.alphabet):
                full[eval_range, LEGACY_AAS.index(aa)] = block[:, j]
            name = f"ens_{rule.rule_id}"
            d = os.path.join(args.output_dir, name)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"variant_root_predictions_{name}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(list(LEGACY_AAS))
                for i in range(args.seq_len):
                    w.writerow(["" if np.isnan(v) else f"{v:.8f}" for v in full[i]])
            print(f"  wrote {path}")

    if args.ref_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.ref_out)), exist_ok=True)
        with open(args.ref_out, "w") as f:
            f.write(root + "\n")
        print(f"  wrote {args.ref_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
