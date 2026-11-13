#!/usr/bin/env python
"""antiGen-data-revised's count matrix, written in the `pssms/root.csv` shape.

WHY. `PSSMFitnessPredictor` -- the `pssm` partner of a gated model -- reads a CSV whose
header is `pos,<alphabet>` and whose rows are raw counts, then takes log1p. The revised
corpus does not ship that file: it keeps the same numbers as `counts_<D>_cur.tsv`, a
TSV with `#sequences` / `#sites` preamble and a `#site` header, which is what every
other reader in the pipeline takes via `--root-counts`.

Rather than teach the predictor a second format or write into the data repo, this
converts one into the other, beside the checkpoints that will use it.

THE CONVERSION IS CHECKED AGAINST THE PIPELINE'S OWN READER. `load_counts_matrix` is
what produces the `pssm_root` RUNG on the same panel, so the gated model's partner has
to be that same matrix -- otherwise the gate line is a blend of antiGen with something
that merely resembles the PSSM beside it. The file is read back with the predictor's
own parser and compared cell for cell.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from antiGen.load import load_counts_matrix                          # noqa: E402
from antiGen.models.pssm_fitness_predictor import _load_pssm_log_counts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-counts", required=True)
    ap.add_argument("--alphabet", default="ACDEFGHIKLMNPQRSTVWY")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    counts = load_counts_matrix(args.root_counts, args.alphabet)   # (L, E) float
    arr = counts.numpy() if isinstance(counts, torch.Tensor) else np.asarray(counts)
    L, E = arr.shape
    if E != len(args.alphabet):
        raise SystemExit(f"counts have {E} columns for a {len(args.alphabet)}-letter alphabet")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("pos," + ",".join(args.alphabet) + "\n")
        for i in range(L):
            fh.write(str(i) + "," + ",".join(f"{v:.0f}" for v in arr[i]) + "\n")
    os.replace(tmp, args.output)

    # Read back through the predictor's own parser and compare to the pipeline's.
    got = _load_pssm_log_counts(args.output, E)
    want = torch.log1p(torch.as_tensor(arr, dtype=torch.float32))
    if got.shape != want.shape:
        raise SystemExit(f"round trip changed the shape: {tuple(got.shape)} != {tuple(want.shape)}")
    worst = float((got - want).abs().max())
    print(f"{args.root_counts}")
    print(f"  -> {args.output}  ({L} positions x {E} residues)")
    print(f"  round trip vs load_counts_matrix, worst |delta| in log1p = {worst:.3e}")
    if worst > 1e-5:
        raise SystemExit("round trip FAILED: the partner would not be the pssm_root rung")


if __name__ == "__main__":
    main()
