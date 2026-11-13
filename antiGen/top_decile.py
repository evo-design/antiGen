#!/usr/bin/env python
"""Rank every substitution of a query and list the top decile, optionally in a range.

The ranking is the paper's all-cells evaluation: within the chosen coordinate range,
every (position, residue) cell except each position's wild-type residue is a candidate;
candidates are ranked by the model's score; the top decile is the ceil(q * n) best.
For the SARS-CoV-2 RBD (positions 331-531) that is 201 x 19 = 3,819 candidates and a
top decile of 382.

Usage:
    antiGen-top-decile query.fasta                          # shipped model, whole sequence
    antiGen-top-decile query.fasta --start 331 --end 531 -o top_rbd.csv
    antiGen-top-decile query.fasta --checkpoint my_model.pt --qtile 0.05

Columns: rank (1 = most likely; tied cells share their best rank), mutation (e.g.
L455S), position (1-based), ref_aa, alt_aa, score (the model's log-score), prob (the
substitution probability, renormalised over the candidates in the range) and percentile
(100 = top).
"""
from __future__ import annotations

import argparse
import csv
import math

import torch

from antiGen import pretrained
from antiGen.infer import read_fasta
from antiGen.models.factory import load_model


def rank_substitutions(scores, seq, alphabet, start=1, end=None):
    """Every non-wild-type cell in [start, end] (1-based, inclusive), best first.

    ``scores`` is one query's (L, E) score map. Returns a list of dicts with rank,
    mutation, position, ref_aa, alt_aa, score, prob and percentile.
    """
    L = len(seq)
    end = L if end is None else end
    if not 1 <= start <= end <= L:
        raise ValueError(f"range {start}-{end} is outside the sequence (1-{L})")
    cells = []
    for pos in range(start, end + 1):
        ref = seq[pos - 1]
        for j, alt in enumerate(alphabet):
            if alt != ref:
                cells.append((float(scores[pos - 1, j]), pos, ref, alt))
    n = len(cells)
    vals = torch.tensor([c[0] for c in cells], dtype=torch.float64)
    probs = torch.softmax(vals, dim=0)
    order = sorted(range(n), key=lambda i: -cells[i][0])
    out, prev, rank = [], None, 0
    for k, i in enumerate(order):
        s, pos, ref, alt = cells[i]
        if s != prev:
            rank, prev = k + 1, s          # ties share the best rank
        out.append({"rank": rank, "mutation": f"{ref}{pos}{alt}", "position": pos,
                    "ref_aa": ref, "alt_aa": alt, "score": s, "prob": float(probs[i]),
                    "percentile": 100.0 * (1.0 - (rank - 1) / n)})
    return out


def top_decile(ranked, qtile=0.1):
    """The cells ranked within the top ceil(qtile * n) (boundary ties included)."""
    k = math.ceil(qtile * len(ranked) - 1e-9)
    return [r for r in ranked if r["rank"] <= k]


def score_query(model, seq, alphabet):
    """One query's (L, E) score map."""
    x = pretrained.encode(seq, alphabet)
    with torch.no_grad():
        return model(x)[0].float()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasta", help="Query sequence(s): FASTA or a raw sequence file")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--pretrained", default=None,
                     help=f"A shipped model (default {pretrained.DEFAULT}): "
                          + ", ".join(pretrained.available()))
    src.add_argument("--checkpoint", default=None, help="Your own trained checkpoint (.pt)")
    ap.add_argument("--start", type=int, default=1, help="First position, 1-based (default 1)")
    ap.add_argument("--end", type=int, default=None, help="Last position, 1-based (default: end)")
    ap.add_argument("--qtile", type=float, default=0.1, help="Top fraction to list (default 0.1)")
    ap.add_argument("--all", action="store_true", help="Write every candidate, not just the top")
    ap.add_argument("-o", "--output", default="top_decile.csv")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.checkpoint:
        model, _ = load_model(args.checkpoint, device=args.device)
        model.eval()
        name = args.checkpoint
    else:
        model, info = pretrained.load(args.pretrained or pretrained.DEFAULT, device=args.device)
        name = info["description"]
    alphabet, seq_len = model.config.alphabet.amino_acids, model.config.seq_len
    print(f"model: {name}")

    rows = []
    for qid, seq in read_fasta(args.fasta):
        if len(seq) != seq_len:
            raise SystemExit(f"{qid}: length {len(seq)}, but the model's frame is {seq_len}")
        ranked = rank_substitutions(score_query(model, seq, alphabet), seq, alphabet,
                                    args.start, args.end)
        keep = ranked if args.all else top_decile(ranked, args.qtile)
        print(f"{qid}: {len(ranked)} candidates in {args.start}-{args.end or seq_len}; "
              f"{'all' if args.all else f'top {args.qtile:g}'}: {len(keep)} listed")
        rows += [{"query_id": qid, **r} for r in keep]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "rank", "mutation", "position", "ref_aa",
                                          "alt_aa", "score", "prob", "percentile"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "score": f"{r['score']:.6f}", "prob": f"{r['prob']:.6e}",
                        "percentile": f"{r['percentile']:.2f}"})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
