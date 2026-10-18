"""Curated mutation lists as evaluation subsets.

Figure 2's metrics run over every candidate cell in the eval range. A *list* eval
restricts them to a named set of (position, alternative residue) cells -- the ECDC
variant-defining mutations, say, or the CoG-UK mutations of interest -- so the question
becomes "among the mutations somebody curated as important, how well does the model
rank them?" rather than "among all cells".

This is the same machinery as the de novo ("unseen") subset: the eval builds a
``(seq_len, E)`` boolean where True means *exclude*, and ``evaluate_full`` NaN-masks
those cells in the model's scores, in every baseline's scores, and zeroes them in the
targets. A list mask is therefore ``~in_list``.

**Only (position, alt) is used; the listed wild-type residue is deliberately ignored.**
ECDC annotates each lineage's mutations relative to *that lineage's* parent, so entries
like ``D339H`` name a residue the shared root does not carry (it has G at 339). Keying
on the wild-type letter would silently drop those. Ten of ECDC's 372 parsable entries
are of this kind; every one of CoG-UK's 323 matches the root, which is what makes the
1-based reading of the coordinates verifiable rather than assumed.

A listed cell whose alternative residue is what the query already carries is not a
mutation at all. Nothing special is done about it here because the eval already handles
it: ``compute_hit_rate`` NaN-masks each example's own residue, and the emergence path
masks the testing root's residue through ``wt_mask``. Such a cell simply never becomes
a candidate or a target.

Indels. Deletions are representable -- the alphabet carries a gap -- and are parsed to
``(position, '-')``, one cell per deleted position, so ``Δ69-70`` yields two. Insertions
are not representable as (position, residue) and are dropped, as is ``E484X``, whose
alternative is "anything". Both are counted and reported rather than passed over.
"""
from __future__ import annotations

import csv
import os
import re

# A substitution: wild-type letter, 1-based position, alternative residue. The
# wild-type letter is parsed only so it can be reported; it never selects a cell.
_SUB = re.compile(r"^([A-Z])(\d+)([A-Z*-])$")
# `483del`, `S31del`, and ranges: `Δ69-70`, `137-145del`, `69/70del`. The unicode minus
# (U+2212) and the division slash both appear in the ECDC sheet.
_DEL_ONE = re.compile(r"^([A-Z]?)(\d+)del$", re.IGNORECASE)
# `del` is sometimes truncated in the source sheet ("137-145de"), so match a `de`
# prefix rather than the full word.
_DEL_RANGE = re.compile(r"^[ΔΔ]?(\d+)\s*[-−/]\s*(\d+)\s*(?:de\w*)?$",
                        re.IGNORECASE)
_DEL_PREFIX = re.compile(r"^[ΔΔ](\d+)$")


def parse_mutation(text):
    """One raw cell -> list of (1-based position, alt residue), possibly empty.

    Returns [] for anything not expressible as (position, residue): insertions, and
    ambiguous alternatives such as ``E484X``.
    """
    s = (text or "").strip()
    if not s:
        return []
    s = s.split("(")[0].strip()          # ECDC appends footnotes: "E484X (d)"
    if not s or s.lower().startswith("ins"):
        return []

    m = _SUB.match(s)
    if m:
        alt = m.group(3)
        return [] if alt == "X" else [(int(m.group(2)), alt)]

    m = _DEL_ONE.match(s)
    if m:
        return [(int(m.group(2)), "-")]

    m = _DEL_PREFIX.match(s)
    if m:
        return [(int(m.group(1)), "-")]

    m = _DEL_RANGE.match(s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo <= hi and hi - lo < 100:
            return [(p, "-") for p in range(lo, hi + 1)]
    return []


def load_mutation_list(path, allowed_aas, seq_len, column="mutation",
                       root_seq=None, label=None):
    """Read a CSV of mutations into cells plus a report of what was and was not used.

    Returns ``(cells, report)``. ``cells`` is a set of ``(0-based position, residue
    index)``. ``report`` carries the counts worth printing: how many rows parsed, how
    many distinct cells resulted, how many were dropped as unparsable or out of range,
    and -- when ``root_seq`` is given -- how many listed wild-type letters disagree with
    the root, which is the check that the coordinates are being read 1-based.
    """
    if not os.path.exists(path):
        raise SystemExit(f"mutation list not found: {path}")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"mutation list is empty: {path}")
    if column not in rows[0]:
        raise SystemExit(f"{path}: no '{column}' column (has {list(rows[0])[:6]})")

    cells, unparsed, out_of_range, wt_mismatch = set(), [], 0, 0
    n_indel = 0
    for row in rows:
        raw = row.get(column)
        parsed = parse_mutation(raw)
        if not parsed:
            unparsed.append((raw or "").strip())
            continue
        if len(parsed) > 1 or parsed[0][1] == "-":
            n_indel += 1
        sub = _SUB.match((raw or "").strip().split("(")[0].strip())
        for pos1, alt in parsed:
            idx = pos1 - 1
            if idx < 0 or idx >= seq_len or alt not in allowed_aas:
                out_of_range += 1
                continue
            if root_seq is not None and sub is not None and root_seq[idx] != sub.group(1):
                wt_mismatch += 1
            cells.add((idx, allowed_aas.index(alt)))

    report = dict(path=path, label=label or os.path.basename(path), n_rows=len(rows),
                  n_cells=len(cells), n_unparsed=len(unparsed), n_indel_rows=n_indel,
                  n_out_of_range=out_of_range, n_wt_mismatch=wt_mismatch,
                  unparsed_examples=sorted(set(unparsed))[:8])
    return cells, report


def cells_to_mask(cells, seq_len, n_aas, eval_range=None):
    """(seq_len, E) bool, True where a cell IS in the list.

    The eval's variant masks mean True = *exclude*, so callers pass ``~mask``. With
    ``eval_range`` the mask is returned already subset to those rows, which is the form
    the emergence path needs (its PSSM and baseline maps are subset before masking);
    the recall path masks the full-length tensor and subsets afterwards.
    """
    import torch
    mask = torch.zeros(seq_len, n_aas, dtype=torch.bool)
    for pos, aa in cells:
        mask[pos, aa] = True
    if eval_range is not None:
        mask = mask[eval_range, :]
    return mask


def describe(report, in_range=None):
    """One-line summary for the eval log."""
    s = (f"  {report['label']:12s} {report['n_rows']:4d} rows -> "
         f"{report['n_cells']:4d} cells")
    if in_range is not None:
        s += f" ({in_range} in eval range)"
    bits = []
    if report["n_unparsed"]:
        bits.append(f"{report['n_unparsed']} unparsable "
                    f"({', '.join(report['unparsed_examples'][:4])})")
    if report["n_indel_rows"]:
        bits.append(f"{report['n_indel_rows']} indel rows -> gap cells")
    if report["n_out_of_range"]:
        bits.append(f"{report['n_out_of_range']} outside the sequence")
    if report["n_wt_mismatch"]:
        bits.append(f"{report['n_wt_mismatch']} wild-type letters differ from the root "
                    f"(lineage-relative annotation; position+alt still used)")
    return s + ("  [" + "; ".join(bits) + "]" if bits else "")
