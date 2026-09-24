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
    n_mat: object = None  # (seq_len, E) float tensor: the count this file is about --
                          # descendants of each mutation for a cutoff file, descendants in
                          # the next time window for an interval file. None for files
                          # written before counts existed.
    total_mat: object = None  # (seq_len, E) all-time descendants, only in interval files
    cur_mat: object = None    # (seq_len, E) descendants collected in THIS example's own
                              # window, only in interval files. The target (n_mat) is the
                              # next window; this is the same split one year earlier, which
                              # is what the persistence baseline predicts from.
    time_interval: int = None  # window index, only in interval files
    time_range: tuple = None   # (lo, hi) absolute decimal years, only in interval files


def _match_variant(label, variant):
    """Check if a clade label matches the variant name (exact or in parentheses)."""
    return label == variant or f"({variant})" in label


def load_examples(filepath, allowed_aas, variant):
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

    n_rows = 0              # MUTATIONS_TO_CHILDREN rows seen
    n_rows_with_count = 0   # ...of which carried a descendant count

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

        # Optional TIME_INTERVAL / TIME_RANGE, present only in --intervals files. The
        # same genotype appears once per window it is relevant to, with different counts,
        # so these are what tell the rows apart.
        i += 1
        time_interval = time_range = None
        while lines[i].strip().startswith(("TIME_INTERVAL:", "TIME_RANGE:")):
            key, _, value = lines[i].strip().partition(":")
            if key == "TIME_INTERVAL":
                time_interval = int(value)
            else:
                lo, hi = value.split(",")
                time_range = (float(lo), float(hi))
            i += 1

        # PARENT_EXAMPLE_ID
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
        # Rows are "<pos>,<alt>" or, for files carrying descendant counts,
        # "<pos>,<alt>,<n>" where n is the number of observed sequences descended from
        # that substitution on this example's side of the train/test split. Both are
        # accumulated, so two children opening with the same substitution add up.
        i += 1  # "MUTATIONS_TO_CHILDREN:"
        i += 1
        mat = torch.zeros(seq_len, E, dtype=torch.float32)
        n_mat = torch.zeros(seq_len, E, dtype=torch.float32)
        total_mat = torch.zeros(seq_len, E, dtype=torch.float32)
        cur_mat = torch.zeros(seq_len, E, dtype=torch.float32)
        has_total = has_cur = False
        while i < len(lines) and lines[i].strip():
            parts = lines[i].strip().split(",")
            pos, alt_aa = int(parts[0]), parts[1]
            mat[pos, allowed_aas.index(alt_aa)] += 1.0
            if len(parts) > 2:
                n_mat[pos, allowed_aas.index(alt_aa)] += float(parts[2])
                n_rows_with_count += 1
            if len(parts) > 3:
                total_mat[pos, allowed_aas.index(alt_aa)] += float(parts[3])
                has_total = True
            if len(parts) > 4:
                cur_mat[pos, allowed_aas.index(alt_aa)] += float(parts[4])
                has_cur = True
            n_rows += 1
            i += 1

        examples.append(Example(
            aa_idx=aa_idx,
            mat=mat,
            is_training=is_training,
            parent_id=parent_id,
            parent_mutation=parent_mutation,
            is_testing_root=is_testing_root,
            is_descended_from_variant=is_descended_from_variant,
            n_mat=n_mat,
            total_mat=total_mat if has_total else None,
            cur_mat=cur_mat if has_cur else None,
            time_interval=time_interval,
            time_range=time_range,
        ))

        i += 1

    # Descendant counts are all-or-nothing: a file either predates the field or carries it
    # on every row. A partial file means the reader got out of step with the format, which
    # would silently produce zero counts for the rows it missed.
    if n_rows_with_count == 0:
        for ex in examples:
            ex.n_mat = None
    elif n_rows_with_count != n_rows:
        raise ValueError(
            f"{filepath}: {n_rows_with_count} of {n_rows} MUTATIONS_TO_CHILDREN rows carry a "
            f"descendant count. Expected either all of them or none."
        )

    n_testing_roots = sum(1 for ex in examples if ex.is_testing_root)
    n_variant_desc = sum(1 for ex in examples if ex.is_descended_from_variant)
    print(f"Loaded {len(examples)} examples from {filepath}")
    print(f"  {n_testing_roots} variant testing root(s), {n_variant_desc} variant-descended for '{variant}'")
    print(f"  descendant counts: {'present' if n_rows_with_count else 'absent (pre-count file)'}")
    windows = sorted({ex.time_interval for ex in examples if ex.time_interval is not None})
    if windows:
        print(f"  time windows: {len(windows)} ({windows[0]}..{windows[-1]}), "
              f"counts are samples in the following window")
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


def split_by_window(examples, test_window):
    """Split an ``--intervals`` examples file on its time windows.

    Window ``test_window`` becomes the test set and every earlier window the training
    set. Later windows are dropped rather than folded into training: the generator emits
    a window whenever a successor window exists at all, so the most recent one is counted
    against a partially observed year and its targets are undercounted by construction.

    Returns ``(training, testing, dropped)``.
    """
    if all(ex.time_interval is None for ex in examples):
        raise ValueError(
            "--test-window needs an examples file carrying TIME_INTERVAL lines "
            "(examples_intervals.txt, written by antiGen-data --intervals)."
        )
    training, testing, dropped = [], [], []
    for ex in examples:
        if ex.time_interval is None:
            raise ValueError("Mixed file: some examples carry TIME_INTERVAL and some do not.")
        bucket = (training if ex.time_interval < test_window
                  else testing if ex.time_interval == test_window
                  else dropped)
        bucket.append(ex)
    if not testing:
        seen = sorted({ex.time_interval for ex in examples})
        raise ValueError(f"No examples in window {test_window}; the file has {seen}.")
    print(f"Window split at {test_window}: {len(training)} training examples "
          f"(windows < {test_window}), {len(testing)} testing, "
          f"{len(dropped)} dropped from later windows")
    return training, testing, dropped
