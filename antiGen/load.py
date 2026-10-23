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


def _parse_split_label(line, filepath):
    """Return True if this record is in the training window.

    Two generators write this field. The original antiGen-data tool writes the bare
    word ``TRAINING`` or ``TESTING``; antiGen-data-revised writes
    ``POST_CUTOFF: True``/``False``, which is the same fact negated. Anything else is
    an error rather than a silent False, because reading an unrecognised label as
    "not training" empties the training pool and the run fails much later with a
    confusing message.
    """
    if line in ("TRAINING", "TESTING"):
        return line == "TRAINING"
    if line.startswith("POST_CUTOFF:"):
        value = line.split(":", 1)[1].strip()
        if value in ("True", "False"):
            return value == "False"
    raise ValueError(
        f"{filepath}: expected 'TRAINING'/'TESTING' or 'POST_CUTOFF: True|False' "
        f"where the train/test split label belongs, got {line!r}")


def _detect_position_base(records, seq_len, filepath):
    """Return 0 or 1: the index of the first residue in this file's mutation positions.

    antiGen-data writes 0-indexed positions; antiGen-data-revised writes 1-indexed
    ones, matching the 1-indexed ``#site`` column of its count matrix. The rest of
    antiGen is 0-indexed throughout (``RBD_RANGE`` is 330..530 in those coordinates),
    so one of the two has to be shifted, and guessing wrong moves every mutation one
    residue along without failing.

    It is not guessed. ``MUTATION_FROM_PARENT`` is ``from,pos,to`` against the
    record's own sequence, so each non-root record votes for whichever base puts
    ``to`` at ``pos``; a file is accepted only if one base explains every record and
    the other does not. With no such evidence -- a file of one record -- the position
    range decides, and a range that fits both bases falls back to 0.
    """
    fits = {0: 0, 1: 0}
    for seq, pos, alt in records:
        for base in (0, 1):
            j = pos - base
            if 0 <= j < seq_len and seq[j] == alt:
                fits[base] += 1
    n = len(records)
    if n:
        unambiguous = [b for b in (0, 1) if fits[b] == n]
        if len(unambiguous) == 1:
            return unambiguous[0]
        if not unambiguous:
            raise ValueError(
                f"{filepath}: MUTATION_FROM_PARENT does not match the record's own "
                f"sequence under either 0- or 1-indexed positions "
                f"({fits[0]}/{n} and {fits[1]}/{n} records fit). The file is "
                f"inconsistent; run antiGen-validate-examples on it.")
    return None


def load_examples(filepath, allowed_aas, variant, source=None):
    """Load pre-computed training/testing examples from a text file.

    Reads is_testing_root and is_descended_from_variant directly from
    IS_VARIANT_ROOT and IS_DESCENDED_FROM_VARIANT fields, matching the
    variant name either as the full clade label or inside parentheses.

    Accepts both generators' dialects of the format: ``TRAINING``/``TESTING`` or
    ``POST_CUTOFF:`` for the split label, and 0- or 1-indexed mutation positions,
    which are detected from the file and returned 0-indexed either way. Repeated
    ``(pos, alt)`` entries under MUTATIONS_TO_CHILDREN are counts: each one adds 1 to
    that cell of the mutation matrix.

    Returns a list of Example objects.
    """
    E = len(allowed_aas)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Pass 1: parse to plain Python. Positions stay as written, because which base
    # they are in is a property of the file and is only knowable once it is all read.
    raw = []
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
            parent_mutation = (pos, alt_aa)

        # TRAINING / TESTING, or POST_CUTOFF:
        i += 1
        is_training = _parse_split_label(lines[i].strip(), filepath)

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

        # MUTATIONS_TO_CHILDREN:
        i += 1  # "MUTATIONS_TO_CHILDREN:"
        i += 1
        muts = []
        while i < len(lines) and lines[i].strip():
            parts = lines[i].strip().split(",")
            muts.append((int(parts[0]), parts[1]))
            i += 1

        raw.append(dict(seq=seq, muts=muts, is_training=is_training,
                        parent_id=parent_id, parent_mutation=parent_mutation,
                        is_testing_root=is_testing_root,
                        is_descended_from_variant=is_descended_from_variant))
        i += 1

    if not raw:
        return []

    seq_lens = {len(r["seq"]) for r in raw}
    if len(seq_lens) != 1:
        raise ValueError(f"{filepath}: inconsistent sequence lengths {sorted(seq_lens)}")
    seq_len = seq_lens.pop()

    bad = sorted({c for r in raw for c in r["seq"]} - set(allowed_aas))
    if bad:
        raise ValueError(
            f"{filepath}: sequence characters {bad} are outside the alphabet "
            f"'{allowed_aas}'. Pass --alphabet to train in a wider frame.")
    bad = sorted({a for r in raw for _, a in r["muts"]} - set(allowed_aas))
    if bad:
        raise ValueError(
            f"{filepath}: mutation residues {bad} are outside the alphabet "
            f"'{allowed_aas}'. Pass --alphabet to train in a wider frame.")

    base = _detect_position_base(
        [(r["seq"], r["parent_mutation"][0], r["parent_mutation"][1])
         for r in raw if r["parent_mutation"] is not None],
        seq_len, filepath)
    all_pos = [p for r in raw for p, _ in r["muts"]]
    all_pos += [r["parent_mutation"][0] for r in raw if r["parent_mutation"] is not None]
    if base is None:
        # Nothing to check MUTATION_FROM_PARENT against; let the range decide, and
        # keep the 0-indexed reading when both are possible.
        base = 1 if all_pos and max(all_pos) >= seq_len else 0
    if all_pos and not (base <= min(all_pos) and max(all_pos) < seq_len + base):
        raise ValueError(
            f"{filepath}: mutation positions {min(all_pos)}..{max(all_pos)} fall outside "
            f"{base}..{seq_len + base - 1} for a {seq_len}-residue sequence")

    # Pass 2: tensors, with positions shifted to the 0-indexed frame the rest of
    # antiGen works in.
    aa_to_idx = {c: j for j, c in enumerate(allowed_aas)}
    examples = []
    for r in raw:
        aa_idx = torch.tensor([aa_to_idx[c] for c in r["seq"]], dtype=torch.long)
        mat = torch.zeros(seq_len, E, dtype=torch.float32)
        for pos, alt_aa in r["muts"]:
            # A repeated (pos, alt) is a count, not a duplicate: two children of this
            # node carry the same mutation.
            mat[pos - base, aa_to_idx[alt_aa]] += 1.0
        pm = r["parent_mutation"]
        examples.append(Example(
            aa_idx=aa_idx,
            mat=mat,
            is_training=r["is_training"],
            parent_id=r["parent_id"],
            parent_mutation=None if pm is None else (pm[0] - base, aa_to_idx[pm[1]]),
            is_testing_root=r["is_testing_root"],
            is_descended_from_variant=r["is_descended_from_variant"],
            source=source,
        ))

    n_testing_roots = sum(1 for ex in examples if ex.is_testing_root)
    n_variant_desc = sum(1 for ex in examples if ex.is_descended_from_variant)
    n_train = sum(1 for ex in examples if ex.is_training)
    print(f"Loaded {len(examples)} examples from {filepath} "
          f"(L={seq_len}, {base}-indexed positions)")
    print(f"  {n_train} training, {len(examples) - n_train} testing; "
          f"{n_testing_roots} variant testing root(s), {n_variant_desc} variant-descended for '{variant}'")
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


def load_counts_matrix(path, allowed_aas):
    """Read antiGen-data-revised's positional count matrix as an (seq_len, E) tensor.

    The revised generator writes ``counts_<run>.tsv`` where ``pssms/root.csv`` used to
    be: two header lines (``#sequences``, ``#sites``), then a ``#site`` column header
    naming the residues, then one row per site. Two differences from the legacy CSV,
    both handled here rather than by the caller:

    * **Sites are 1-indexed**, matching the mutation positions of the same generator
      (NOTES s53). They come back 0-indexed, which is the frame the rest of antiGen
      works in.
    * **The column order is the file's, not the alphabet's.** The header is read and
      each column placed at its index in ``allowed_aas``, so a 20-column file read
      under the 22-token frame leaves gap and stop at zero rather than shifting the
      last two residues into them.

    A residue in the file that the alphabet does not carry is an error: it is a count
    that would be silently discarded, and the de novo mask is defined by which cells
    are nonzero.
    """
    with open(path) as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    meta = {r[0]: r[1] for r in rows if r[0].startswith("#") and r[0] != "#site"}
    header = next((r for r in rows if r[0] == "#site"), None)
    if header is None:
        raise ValueError(f"{path}: no '#site' column header line")
    letters = header[1:]
    bad = sorted(set(letters) - set(allowed_aas))
    if bad:
        raise ValueError(
            f"{path}: counts for residue(s) {bad} that the alphabet "
            f"'{allowed_aas}' does not carry")
    col_of = [allowed_aas.index(a) for a in letters]

    body = [r for r in rows if not r[0].startswith("#")]
    seq_len = int(meta.get("#sites", len(body)))
    if len(body) != seq_len:
        raise ValueError(f"{path}: #sites {seq_len} but {len(body)} data rows")
    counts = torch.zeros(seq_len, len(allowed_aas), dtype=torch.float32)
    sites = []
    for r in body:
        site = int(r[0])
        sites.append(site)
        for j, v in enumerate(r[1:]):
            counts[site - 1, col_of[j]] = float(v)
    if sorted(sites) != list(range(1, seq_len + 1)):
        raise ValueError(
            f"{path}: sites are not 1..{seq_len} exactly "
            f"(min {min(sites)}, max {max(sites)}, {len(set(sites))} distinct)")
    print(f"Loaded counts matrix {path}: {tuple(counts.shape)}, "
          f"{int((counts > 0).sum())} nonzero cells over "
          f"{meta.get('#sequences', '?')} sequences")
    return counts
