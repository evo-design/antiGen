#!/usr/bin/env python
"""Validate an antiGen ``examples_cutoff_*.txt`` training-data file.

Parses the file with the same reader used for training (``antiGen.load.load_examples``)
and checks the invariants training/eval rely on: a single consistent sequence length,
only alphabet characters, in-range mutation positions, at least one TRAINING example, and
(if a variant is given) at least one variant root. Such files are produced by the
antiGen-data tool; see the README (Training).

Usage:
    antiGen-validate-examples antiGen-data/example/out/examples_cutoff_100.txt
"""
from __future__ import annotations

import argparse

from antiGen.config import LEGACY_AAS
from antiGen.load import load_examples

# The widest alphabet antiGen has ever written, so a file in either frame validates.
# Training now defaults to the 20 standard residues (antiGen-train --alphabet).
ALPHABET = LEGACY_AAS


def validate(path, variant=None, alphabet=ALPHABET):
    """Parse and check an examples file. Raises SystemExit with a clear message on failure."""
    try:
        examples = load_examples(path, alphabet, variant)
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"Failed to parse '{path}': {exc}\n"
                         f"Expected an antiGen-data examples_cutoff_*.txt file (see the README, Training).")
    if not examples:
        raise SystemExit(f"No examples parsed from '{path}'.")

    seq_lens = {len(e.aa_idx) for e in examples}
    if len(seq_lens) != 1:
        raise SystemExit(f"Inconsistent sequence lengths {sorted(seq_lens)} — every example must "
                         f"share one seq_len (align/pad to a common reference).")
    seq_len = seq_lens.pop()

    n = len(examples)
    n_train = sum(bool(e.is_training) for e in examples)
    n_test = n - n_train
    n_root = sum(bool(e.is_testing_root) for e in examples)
    n_desc = sum(bool(e.is_descended_from_variant) for e in examples)
    n_child = sum(int(e.mat.sum().item()) for e in examples)

    if n_train == 0:
        raise SystemExit("No training-window examples found (TRAINING / POST_CUTOFF: False) "
                         "— training needs at least one.")

    print(f"OK: {n} examples | seq_len={seq_len} | training={n_train} testing={n_test} "
          f"| variant={variant!r}: roots={n_root} descended={n_desc} | child mutations={n_child}")
    if variant is not None and n_root == 0:
        print(f"  [warn] no IS_VARIANT_ROOT=True example for variant {variant!r} "
              f"(fine for training; eval's variant-emergence needs one).")
    return examples


def main():
    ap = argparse.ArgumentParser(description="Validate an antiGen examples_cutoff_*.txt file.")
    ap.add_argument("examples", help="Path to an examples_cutoff_*.txt file")
    ap.add_argument("--variant", default=None, help="Variant name to count roots/descent for")
    ap.add_argument("--alphabet", default=ALPHABET,
                    help="Amino-acid alphabet (default: the 22-token frame, which accepts "
                         "a 20-residue file too)")
    args = ap.parse_args()
    validate(args.examples, args.variant, args.alphabet)
    print("Validation passed.")


if __name__ == "__main__":
    main()
