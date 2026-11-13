#!/usr/bin/env python
"""CoVFit scored on EACH QUERY'S OWN BACKGROUND.

`covfit.py` builds one static (L, E) map from the single mutants of XFG and the eval
applies it to every query whatever its residues. On the 2024-10-01 fixed-split arm that
put the COG-UK RBD list in CoVFit's bottom decile 24% of the time (top decile 8%): the
sites XFG already carries an adaptive residue at (455, 456, 486, 493, 452, 484) rank
lowest, because moving AWAY from an adapted residue looks deleterious. antiGen is asked
each query; this asks CoVFit each query too.

For every query genotype the eval scores, every RBD single mutant (eval range x 19) and
the query itself go through the CoVFit CLI; a cell's score is mutant fitness minus the
query's (`fitness_mean`, the 17-country mean the CLI reports). Ranks within a query are
what the eval uses, so the subtraction is cosmetic.

  select  the query set: exactly list_macro.py's `scored` -- test examples (split at
          --split-date) carrying a listed target in the eval range -- deduplicated by
          genotype. Writes queries.npz.
  fasta   chunk K of N of those queries as a FASTA for the CLI.
  build   parse the CLI's TSVs (one or more folds, averaged) into a per-query map
          NPZ for `list_macro.py --query-maps`.

INPUT CONVENTION. CoVFit's own examples are aligned 1,274-residue spikes with `-` for a
deletion. The 20-letter sc2 corpus has no gap state -- a deletion is missing data and is
imputed (site 483 reads V in 98% of test genotypes, though BA.2.86/JN.1 carry
V483del) -- and there is no 22-letter sc2 build. Queries are therefore passed as
encoded: no deletions. The XFG map carried V483- (from the older gapped corpus).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from antiGen.eval_utils import RBD_RANGE  # noqa: E402
from antiGen.load import load_examples  # noqa: E402
from antiGen.mutation_lists import load_mutation_list  # noqa: E402

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def cmd_select(a):
    E = len(AA20)
    test = [e for p in a.test_examples
            for e in load_examples(p, AA20, a.variant, split_date=a.split_date)
            if not e.is_training]
    train = [e for e in load_examples(a.train_examples, AA20, a.variant,
                                      split_date=a.split_date) if e.is_training]
    seq_len = len(train[0].aa_idx)
    root = "".join(AA20[i] for i in train[0].aa_idx.tolist())   # list_macro's root_seq
    listed = torch.zeros(seq_len, E, dtype=torch.bool)
    for item in a.mutation_list:
        label, path = item.split("=", 1)
        cells, _ = load_mutation_list(path, AA20, seq_len, root_seq=root, label=label)
        for pos, aa in cells:
            listed[pos, aa] = True
    in_range = torch.zeros_like(listed)
    in_range[RBD_RANGE, :] = True
    listed &= in_range
    scored = [e for e in test if bool((e.mat[listed] > 0).any())]
    uniq = {}
    for e in scored:
        uniq.setdefault(bytes(e.aa_idx.to(torch.int8).numpy()), e.aa_idx.to(torch.int8).numpy())
    aa = np.stack(list(uniq.values()))
    np.savez(a.out, aa=aa, eval_range=np.array(RBD_RANGE))
    print(f"{len(test)} test examples, {len(scored)} carry a listed target, "
          f"{len(aa)} distinct genotypes -> {a.out}")


def cmd_fasta(a):
    z = np.load(a.queries)
    aa, er = z["aa"], z["eval_range"].tolist()
    idx = np.array_split(np.arange(len(aa)), a.n_chunks)[a.chunk]
    n = 0
    with open(a.out, "w") as f:
        for q in idx:
            s = "".join(AA20[i] for i in aa[q])
            f.write(f">q{q}_WT\n{s}\n")
            n += 1
            for p in er:
                for alt in AA20:
                    if alt == s[p]:
                        continue
                    f.write(f">q{q}_{p}_{alt}\n{s[:p]}{alt}{s[p + 1:]}\n")
                    n += 1
    print(f"chunk {a.chunk}/{a.n_chunks}: queries {idx[0]}-{idx[-1]}, {n} sequences -> {a.out}")


def read_tsv(path):
    out = {}
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        assert h[:2] == ["seq_name", "fitness_mean"], h[:3]
        for row in r:
            out[row[0]] = float(row[1])
    return out


def cmd_build(a):
    z = np.load(a.queries)
    aa, er = z["aa"], z["eval_range"].tolist()
    Nq, L = aa.shape
    folds = []
    for fd in sorted(glob.glob(os.path.join(a.tsv_root, "fold*"))):
        sc = {}
        for t in sorted(glob.glob(os.path.join(fd, "chunk*", "CoVFit_Predictions_fold_*.tsv"))):
            sc.update(read_tsv(t))
        folds.append((os.path.basename(fd), sc))
    if not folds:
        raise SystemExit(f"no fold*/chunk*/ TSVs under {a.tsv_root}")
    want = Nq * (1 + len(er) * (len(AA20) - 1))
    for name, sc in folds:
        if len(sc) != want:
            raise SystemExit(f"{name}: {len(sc)} scores, expected {want} -- a chunk is "
                             f"missing or partial")
    S = np.zeros((Nq, len(er), len(AA20)), dtype=np.float64)
    for _name, sc in folds:
        for q in range(Nq):
            wt = sc[f"q{q}_WT"]
            for r, p in enumerate(er):
                for j, alt in enumerate(AA20):
                    if j == aa[q, p]:
                        continue                       # the query's own residue: 0
                    S[q, r, j] += sc[f"q{q}_{p}_{alt}"] - wt
    S /= len(folds)
    np.savez(a.out, aa=aa, eval_range=np.array(er), scores=S.astype(np.float32),
             folds=np.array([n for n, _ in folds]))
    d = S.ravel()
    print(f"{Nq} queries x {len(er)} sites, folds {[n for n, _ in folds]} -> {a.out}; "
          f"delta 1/50/99%: {np.round(np.quantile(d, [.01, .5, .99]), 5)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--train-examples", required=True)
    s.add_argument("--test-examples", action="append", required=True)
    s.add_argument("--split-date", required=True)
    s.add_argument("--mutation-list", action="append", required=True)
    s.add_argument("--variant", default="BA.2")
    s.add_argument("--out", required=True)
    f = sp.add_parser("fasta")
    f.add_argument("--queries", required=True)
    f.add_argument("--chunk", type=int, required=True)
    f.add_argument("--n-chunks", type=int, required=True)
    f.add_argument("--out", required=True)
    b = sp.add_parser("build")
    b.add_argument("--queries", required=True)
    b.add_argument("--tsv-root", required=True, help="holds fold<F>/chunk<K>/ CLI outputs")
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "fasta": cmd_fasta, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
