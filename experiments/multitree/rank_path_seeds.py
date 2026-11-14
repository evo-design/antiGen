#!/usr/bin/env python
"""Rank the J.2->K steps across all seeds of one fig4 arm, to show the panel's numbers
are not a single-checkpoint artifact. Loads the architecture once and swaps state dicts.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from antiGen import eval_utils as eu                                   # noqa: E402
from antiGen.background_discrimination import rank_fractions, top_q_weight  # noqa: E402
from antiGen.eval_utils import build_model_gap_stop_mask, load_checkpoint  # noqa: E402
from antiGen.models.factory import create_model                        # noqa: E402

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-csv", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--arm-glob", required=True,
                    help="glob of per-seed run dirs, e.g. .../fig4_h3n2_vf02/esm650m_p4_s*")
    ap.add_argument("--ckpt-suffix", default="_best.pt")
    ap.add_argument("--parent-id-col", default="parent_example_id_int")
    ap.add_argument("--qtile", type=float, default=0.1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    steps = [s for s in csv.DictReader(open(args.path_csv)) if s["position_1based"]]
    parents = [int(s[args.parent_id_col]) for s in steps]
    seqs = read_sequences(args.examples, parents)
    seq_len = len(next(iter(seqs.values())))
    E = len(AAS)
    eval_range = list(range(seq_len))

    aa_batch = torch.tensor([[AAS.index(c) for c in seqs[p]] for p in parents],
                            dtype=torch.long)
    pool_mask = eu.real_substitution_mask(len(eval_range), AAS)
    wt_pool = eu.real_wildtype_mask(aa_batch[:, eval_range], AAS)

    seed_dirs = sorted(glob.glob(args.arm_glob))
    print(f"{len(seed_dirs)} seed dir(s)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed = {}   # seed_label -> {step: rank_best}
    model = None
    for d in seed_dirs:
        hits = sorted(glob.glob(os.path.join(d, f"*{args.ckpt_suffix}")))
        if not hits:
            print(f"  {d}: no {args.ckpt_suffix}, skipped")
            continue
        ckpt_path = hits[0]
        label = os.path.basename(d)
        ckpt, cfg = load_checkpoint(ckpt_path, map_location=device)
        if model is None:
            model = create_model(cfg, device)
            gi = AAS.find("-"); si = AAS.find("*")
            mm = build_model_gap_stop_mask(model, seq_len, E, gi, si).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            s = model(aa_batch.to(device)).float()
            s[:, mm] = float("nan")
        scores = s.cpu()[:, eval_range, :]
        above, tie, n_valid, filled = rank_fractions(
            scores, aa_batch, eval_range, pool_mask=pool_mask, wt_pool_mask=wt_pool,
            return_filled=True)
        wq = top_q_weight(above, tie, n_valid, args.qtile)
        wq = torch.where(filled, torch.zeros_like(wq), wq)
        nv = int(n_valid[0])
        row = {}
        for b, st in enumerate(steps):
            r = eval_range.index(int(st["position_0based"]))
            j = AAS.index(st["to_aa"])
            rank = int(round(float(above[b, r, j] * nv))) + 1
            row[st["mutation_1based"]] = (rank, float(wq[b, r, j]))
        per_seed[label] = (row, nv)
        print(f"  {label}: pool {nv}  " +
              "  ".join(f"{m}={row[m][0]}" for m in list(row)[:4]) + " ...")

    # Tabulate: per step, ranks across seeds + median + decile hit fraction.
    muts = [s["mutation_1based"] for s in steps]
    labels = list(per_seed)
    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mutation_1based"] + [f"rank_{l}" for l in labels]
                   + ["rank_median", "n_candidates", "decile_hits", "n_seeds"])
        for m in muts:
            ranks = [per_seed[l][0][m][0] for l in labels]
            dh = sum(per_seed[l][0][m][1] > 0 for l in labels)
            nv = per_seed[labels[0]][1]
            w.writerow([m] + ranks + [int(np.median(ranks)), nv, dh, len(labels)])
    print(f"wrote {args.output}")
    for m in muts:
        ranks = [per_seed[l][0][m][0] for l in labels]
        dh = sum(per_seed[l][0][m][1] > 0 for l in labels)
        print(f"  {m:>8s}  ranks {ranks}  median {int(np.median(ranks)):5d}  "
              f"decile {dh}/{len(labels)}")


if __name__ == "__main__":
    main()
