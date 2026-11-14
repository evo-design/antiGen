#!/usr/bin/env python
"""Rank the path steps under the count baselines, on the SAME pool antiGen uses.

So the case-study panel can put antiGen next to a counting model: for each step, where
does `Event Counts` (log N[i,a]) or `Smoothed Counts` (the Dirichlet cpssm) place the
substitution among all HA substitutions available from the parent? Same candidate pool,
same masking, same `rank_fractions` code path as `rank_path_cells.py` -- only the score
map changes, from the neural model to a baseline fitted on the model's training file.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import count_baselines as cb                                           # noqa: E402
from antiGen import eval_utils as eu                                   # noqa: E402
from antiGen.background_discrimination import rank_fractions, top_q_weight  # noqa: E402
from antiGen.load import load_examples                                 # noqa: E402

AAS = "ACDEFGHIKLMNPQRSTVWY*-"


def read_sequences(path, wanted):
    want, out, cur = set(wanted), {}, None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("EXAMPLE_ID:"):
                cur = int(line.split(":", 1)[1])
            elif line == "IS_VARIANT_ROOT:" and cur in want:
                out[cur] = next(fh).rstrip("\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path-csv", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--train", required=True, help="the model's training-examples file")
    ap.add_argument("--baseline", required=True,
                    help="count_baselines math key: frequency (Event Counts), "
                         "cpssm_dir (Smoothed Counts), site_prop, pssm")
    ap.add_argument("--parent-id-col", default="parent_example_id_int")
    ap.add_argument("--qtile", type=float, default=0.1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    aas = AAS
    E = len(aas)
    pid = args.parent_id_col

    steps = list(csv.DictReader(open(args.path_csv)))
    with_pos = [s for s in steps if s["position_1based"]]
    parents = [int(s[pid]) for s in with_pos]
    seqs = read_sequences(args.examples, parents)
    seq_len = len(next(iter(seqs.values())))
    eval_range = list(range(seq_len))
    R = len(eval_range)

    train = [e for e in load_examples(args.train, aas, "BA.2") if e.is_training]
    st = cb.fit(train, seq_len, E)

    aa_batch = torch.tensor([[aas.index(c) for c in seqs[p]] for p in parents],
                            dtype=torch.long)
    scores = cb.score_map(args.baseline, st, aa_batch).float()
    gi, si = aas.find("-"), aas.find("*")
    scores[:, :, gi] = float("nan")
    scores[:, :, si] = float("nan")
    scores = scores[:, eval_range, :]

    pool_mask = eu.real_substitution_mask(R, aas)
    wt_pool = eu.real_wildtype_mask(aa_batch[:, eval_range], aas)
    above, tie, n_valid, filled = rank_fractions(
        scores, aa_batch, eval_range, pool_mask=pool_mask, wt_pool_mask=wt_pool,
        return_filled=True)
    wq = top_q_weight(above, tie, n_valid, args.qtile)
    wq = torch.where(filled, torch.zeros_like(wq), wq)
    nv = int(n_valid[0])
    print(f"baseline '{args.baseline}' ({cb.DISPLAY_NAMES.get(args.baseline, args.baseline)}), "
          f"pool {nv} cells")

    by_step = {}
    for b, s in enumerate(with_pos):
        r = eval_range.index(int(s["position_0based"]))
        j = aas.index(s["to_aa"])
        a, t = float(above[b, r, j] * nv), float(tie[b, r, j] * nv)
        by_step[s["step"]] = {
            "rank_best": int(round(a)) + 1,
            "rank_mean": f"{a + (t + 1.0) / 2.0:.1f}",
            "n_tied": int(round(t)) - 1,
            "n_candidates": nv,
            "pct_from_top": f"{100.0 * (a + (t + 1.0) / 2.0) / nv:.3f}",
            "in_top_decile": f"{float(wq[b, r, j]):.4f}",
            "score": f"{float(scores[b, r, j]):.6f}",
        }

    carry = [c for c in steps[0].keys()]
    fields = carry + ["in_pool", "rank_best", "rank_mean", "n_tied", "n_candidates",
                      "pct_from_top", "in_top_decile", "score"]
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in steps:
            row = {k: s.get(k, "") for k in carry}
            row["in_pool"] = int(s["step"] in by_step)
            row.update(by_step.get(s["step"], {}))
            w.writerow(row)
    print(f"wrote {args.output}")
    for s in steps:
        if s["step"] in by_step:
            d = by_step[s["step"]]
            print(f"  {s['mutation_1based']:>8s}  rank {d['rank_best']:5d} / {nv}   "
                  f"top {d['pct_from_top']:>7s}%   decile={d['in_top_decile']}")


if __name__ == "__main__":
    main()
