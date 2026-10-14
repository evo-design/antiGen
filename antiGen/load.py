"""
Load pre-computed training/testing examples and PSSMs.
"""

import os
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch


@dataclass
class Example:
    aa_idx: object       # (seq_len,) long tensor of tokenized AA indices
    mat: object          # (seq_len, E) float tensor of child mutation counts
    is_training: bool
    parent_id: int = None
    parent_mutation: tuple = None  # (pos, alt_idx)
    is_testing_root: bool = False
    is_descended_from_variant: bool = False
    source: str = None   # name of the examples file this came from (multi-source training)


def _match_variant(label, variant):
    """Check if a clade label matches the variant name (exact or in parentheses)."""
    return label == variant or f"({variant})" in label


def load_examples(filepath, allowed_aas, variant, source=None):
    """Load pre-computed training/testing examples from a text file.

    Reads is_testing_root and is_descended_from_variant directly from
    IS_VARIANT_ROOT and IS_DESCENDED_FROM_VARIANT fields, matching the
    variant name either as the full clade label or inside parentheses.

    Returns a list of Example objects.
    """
    E = len(allowed_aas)
    examples = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip blank lines
        if not line:
            i += 1
            continue

        # Expect EXAMPLE_ID line
        if not line.startswith("EXAMPLE_ID:"):
            i += 1
            continue

        # PARENT_EXAMPLE_ID
        i += 1
        parent_line = lines[i].strip()
        parent_str = parent_line.split(":")[1].strip()
        parent_id = None if parent_str == "None" else int(parent_str)

        # MUTATION_FROM_PARENT:
        i += 1  # "MUTATION_FROM_PARENT:"
        i += 1
        mut_line = lines[i].strip()
        if mut_line == "None":
            parent_mutation = None
        else:
            parts = mut_line.split(",")
            ref_aa, pos, alt_aa = parts[0], int(parts[1]), parts[2]
            parent_mutation = (pos, allowed_aas.index(alt_aa))

        # TRAINING or TESTING
        i += 1
        is_training = lines[i].strip() == "TRAINING"

        # IS_DESCENDED_FROM_VARIANT:
        i += 1  # "IS_DESCENDED_FROM_VARIANT:"
        i += 1
        is_descended_from_variant = False
        while i < len(lines) and lines[i].strip().endswith(("True", "False")):
            label, val = lines[i].strip().rsplit(",", 1)
            if _match_variant(label, variant) and val == "True":
                is_descended_from_variant = True
            i += 1

        # IS_VARIANT_ROOT:
        # i is now on the "IS_VARIANT_ROOT:" line
        i += 1
        is_testing_root = False
        while i < len(lines) and lines[i].strip().endswith(("True", "False")):
            label, val = lines[i].strip().rsplit(",", 1)
            if _match_variant(label, variant) and val == "True":
                is_testing_root = True
            i += 1

        # Full AA sequence
        seq = lines[i].strip()
        seq_len = len(seq)
        aa_idx = torch.tensor(
            [allowed_aas.index(c) for c in seq], dtype=torch.long
        )

        # MUTATIONS_TO_CHILDREN:
        i += 1  # "MUTATIONS_TO_CHILDREN:"
        i += 1
        mat = torch.zeros(seq_len, E, dtype=torch.float32)
        while i < len(lines) and lines[i].strip():
            parts = lines[i].strip().split(",")
            pos, alt_aa = int(parts[0]), parts[1]
            mat[pos, allowed_aas.index(alt_aa)] += 1.0
            i += 1

        examples.append(Example(
            aa_idx=aa_idx,
            mat=mat,
            is_training=is_training,
            parent_id=parent_id,
            parent_mutation=parent_mutation,
            is_testing_root=is_testing_root,
            is_descended_from_variant=is_descended_from_variant,
            source=source,
        ))

        i += 1

    n_testing_roots = sum(1 for ex in examples if ex.is_testing_root)
    n_variant_desc = sum(1 for ex in examples if ex.is_descended_from_variant)
    print(f"Loaded {len(examples)} examples from {filepath}")
    print(f"  {n_testing_roots} variant testing root(s), {n_variant_desc} variant-descended for '{variant}'")
    return examples


def load_pssms(pssm_dir, variant, E):
    """Load root PSSM and variant PSSMs from CSV files.

    The number of variant time steps is auto-detected from the files
    present in the variant PSSM directory.

    Returns (pssm_root, pssm_variant) where pssm_root is a (seq_len, E)
    tensor and pssm_variant is a list of n_steps (seq_len, E) tensors.
    """
    pssm_root = torch.tensor(
        np.loadtxt(os.path.join(pssm_dir, "root.csv"), delimiter=",", skiprows=1, usecols=range(1, E + 1)),
        dtype=torch.float32,
    )
    print(f"Loaded pssm_root: {pssm_root.shape}")

    variant_pssm_dir = os.path.join(pssm_dir, variant)
    # Auto-detect time steps by parsing filenames (start time may not be 0)
    import glob as glob_mod
    import re
    time_pattern = re.compile(r"time_(\d+\.\d+)\.csv$")
    times = []
    for f in glob_mod.glob(os.path.join(variant_pssm_dir, "time_*.csv")):
        m = time_pattern.search(os.path.basename(f))
        if m:
            times.append(float(m.group(1)))
    times.sort()
    pssm_variant = []
    for t in times:
        path = os.path.join(variant_pssm_dir, f"time_{t:.1f}.csv")
        pv = torch.tensor(
            np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, E + 1)),
            dtype=torch.float32,
        )
        pssm_variant.append(pv)
    if pssm_variant:
        print(f"Loaded {len(pssm_variant)} variant PSSMs from {variant_pssm_dir} "
              f"(t={times[0]:.1f} to t={times[-1]:.1f})")
    else:
        print(f"Loaded 0 variant PSSMs from {variant_pssm_dir}")

    return pssm_root, pssm_variant
