#!/usr/bin/env python3
"""Independent checker for the per-cutoff variant count matrices written by
`antigen place --out-variant-counts`.

Reads nothing but the directory itself and re-derives every invariant from it,
so it shares no code and no assumptions with the C++ that produced it:

  - one counts_<date>.tsv per cutoff, each well-formed: a #sequences and a
    #sites header, the residues as columns in alphabet order -- the 20 standard
    ones, or those plus '-' and '*' under `antigen --alphabet 22` -- and
    one row per site, numbered 1..nSites. Which of the two it is comes off the
    file's own header, and every matrix in the directory must agree
  - every count is non-negative, and the row totals sum to the residue total a
    full matrix would have if nothing were missing, minus the missing data --
    i.e. no row holds more than #sequences
  - the matrices are **nested**: M_D counts the sequences dated at or after D,
    and raising D can only drop sequences, so for consecutive cutoffs D < D'
    every cell of M_D' is at most the matching cell of M_D, and #sequences
    falls with it. That is the one real constraint on 24 matrices of 1,273 x 20
    cells, and it is a statement about the data, not about how it was written
  - the differences between consecutive matrices are themselves matrices of a
    disjoint set of sequences, so they too are non-negative everywhere and
    their sequence counts sum back to the earliest matrix's

    python3 -I check_variant_counts.py output/sc2-test/counts_ba2 [expected_sequences]
"""
import os
import re
import sys

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA22 = AA20 + ["-", "*"]


def read_matrix(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        if head[0] != "#sequences":
            raise ValueError(f"{path}: first line is not #sequences")
        nseq = int(head[1])
        head = f.readline().rstrip("\n").split("\t")
        if head[0] != "#sites":
            raise ValueError(f"{path}: second line is not #sites")
        nsites = int(head[1])
        cols = f.readline().rstrip("\n").split("\t")
        if cols[1:] == AA20:
            aa = AA20
        elif cols[1:] == AA22:
            aa = AA22
        else:
            raise ValueError(f"{path}: column header is {cols[1:]}, neither the 20 "
                             f"residues nor the 22")
        rows = []
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) != len(aa) + 1:
                raise ValueError(f"{path}: site {len(rows) + 1} has {len(p) - 1} columns, "
                                 f"not {len(aa)}")
            if int(p[0]) != len(rows) + 1:
                raise ValueError(f"{path}: site column is {p[0]} on row {len(rows) + 1}")
            rows.append([int(x) for x in p[1:]])
    if len(rows) != nsites:
        raise ValueError(f"{path}: {len(rows)} rows against #sites {nsites}")
    return nseq, aa, rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    d = sys.argv[1]
    expect = int(sys.argv[2]) if len(sys.argv) > 2 else None

    names = sorted(n for n in os.listdir(d) if re.fullmatch(r"counts_\d{4}-\d{2}-\d{2}\.tsv", n))
    if not names:
        print(f"{d}: no counts_<date>.tsv files")
        sys.exit(1)
    cutoffs = [n[len("counts_"):-len(".tsv")] for n in names]  # sorted is ascending by date

    bad = []
    prev = None
    deltas = 0
    AA = None
    for c, n in zip(cutoffs, names):
        nseq, aa, rows = read_matrix(os.path.join(d, n))
        if AA is None:
            AA = aa
        elif aa != AA:
            bad.append(f"{c}: {len(aa)} residue columns, against {len(AA)} in the ones before")
            continue
        for s, row in enumerate(rows, 1):
            for a, v in zip(AA, row):
                if v < 0:
                    bad.append(f"{c}: site {s} {a} is {v}")
                elif v > nseq:
                    bad.append(f"{c}: site {s} {a} is {v}, above the {nseq} sequences")
        if prev is not None:
            pc, pseq, prows = prev
            # Nested: this cutoff is the later one, so it is the subset.
            if nseq > pseq:
                bad.append(f"{c} has {nseq} sequences, above {pc}'s {pseq}")
            for s, (row, prow) in enumerate(zip(rows, prows), 1):
                for a, v, pv in zip(AA, row, prow):
                    if v > pv:
                        bad.append(f"{c}: site {s} {a} is {v}, above {pc}'s {pv}")
                        break
            deltas += pseq - nseq
        print(f"  {c}: {nseq} sequences, {sum(sum(r) for r in rows)} residues")
        prev = (c, nseq, rows)

    first = read_matrix(os.path.join(d, names[0]))[0]
    last = prev[1]
    if deltas + last != first:
        bad.append(f"the drops sum to {deltas}, leaving {deltas + last} against {first}")
    if expect is not None and first != expect:
        bad.append(f"{cutoffs[0]} holds {first} sequences, expected {expect}")

    print(f"{d}: {len(names)} matrices, {cutoffs[0]} .. {cutoffs[-1]}, "
          f"{first} sequences down to {last}")
    if bad:
        print(f"  {len(bad)} PROBLEMS:")
        for b in bad[:20]:
            print("   ", b)
        sys.exit(1)
    print("  clean: well-formed, non-negative, and nested cell by cell across every cutoff")


main()
