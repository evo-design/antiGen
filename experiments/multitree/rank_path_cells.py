#!/usr/bin/env python
"""Rank each RBD step of a named evolutionary path among all RBD substitutions.

**The question panel D asks.** Walking BA.2.86 -> KP.3 one observed substitution at a
time: given the genotype the lineage actually had when that substitution happened,
where does the model put it among every substitution it could have called instead?

**THE QUERY IS THE PARENT, not the child.** A step's mutation is scored on the
background it arose on -- the parent's full 1,273-residue genotype -- because that is
the only sequence a forecaster would have had in hand before it happened. Scoring it on
the child would be asking the model to rank a mutation the query already carries.

**THE DENOMINATOR IS THE FIGURE'S.** `rank_fractions` is the same code path
`compute_hit_rate` ranks on, with the same two pool halves the 20-letter arm passes, so
the candidate set is the eval range less the query's own residue at each position --
201 x 20 - 201 = 3,819 -- and a rank here means the same thing as a rank anywhere else
in this figure.

**Ties are reported, not broken.** `rank_best` is 1 + the number of cells scored
strictly higher; `rank_mean` is the midpoint of the tied block. They differ by hundreds
of places for a coarse predictor, and one argsort rank would hide that.

Only steps whose position falls in the RBD are scored; the rest of the path is carried
through to the output with a blank rank so the trajectory stays whole.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from antiGen import eval_utils as eu                                   # noqa: E402
from antiGen.background_discrimination import rank_fractions, top_q_weight  # noqa: E402
from antiGen.eval_utils import RBD_RANGE, build_model_gap_stop_mask, load_checkpoint  # noqa: E402
from antiGen.models.factory import create_model                        # noqa: E402


def read_sequences(path, wanted):
    """EXAMPLE_ID -> genotype, for the ids asked for.

    Parsed straight from the file rather than through `load_examples`, which drops the
    EXAMPLE_ID: its `Example` carries a parent_id but not its own, so there is no key
    to look these up by afterwards.
    """
    want, out, cur = set(wanted), {}, None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("EXAMPLE_ID:"):
                cur = int(line.split(":", 1)[1])
            elif line == "IS_VARIANT_ROOT:" and cur in want:
                seq = next(fh).rstrip("\n")
                out[cur] = seq
                if len(out) == len(want):
                    return out
    missing = sorted(want - set(out))
    if missing:
        raise SystemExit(f"{path}: no record for EXAMPLE_ID {missing}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path-csv", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--alphabet", default="ACDEFGHIKLMNPQRSTVWY")
    ap.add_argument("--qtile", type=float, default=0.1)
    ap.add_argument("--eval-range", default="whole",
                    help="the candidate pool: 'whole' (the whole protein), 'rbd' "
                         "(eval_utils.RBD_RANGE), or 'LO-HI' 1-based inclusive")
    ap.add_argument("--parent-id-col", default="parent_example_id",
                    help="the path-csv column holding the integer EXAMPLE_ID of each "
                         "step's parent genotype")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    aas = args.alphabet
    E = len(aas)
    pid = args.parent_id_col

    # Read every step's parent first, so the sequence length is known before the
    # candidate pool is built -- the whole-protein pool is range(seq_len).
    steps = list(csv.DictReader(open(args.path_csv)))
    with_pos = [s for s in steps if s["position_1based"]]
    all_parents = [int(s[pid]) for s in with_pos]
    seqs = read_sequences(args.examples, all_parents)
    seq_len = len(next(iter(seqs.values())))

    if args.eval_range == "whole":
        eval_range = list(range(seq_len))
    elif args.eval_range == "rbd":
        eval_range = RBD_RANGE
    else:
        lo1, hi1 = (int(x) for x in args.eval_range.split("-"))
        eval_range = list(range(lo1 - 1, hi1))
    R = len(eval_range)
    lo1, hi1 = eval_range[0] + 1, eval_range[-1] + 1

    scored = [s for s in with_pos if lo1 <= int(s["position_1based"]) <= hi1]
    print(f"{len(steps)} step(s), {len(scored)} in the pool "
          f"({lo1}-{hi1}, {R} sites)")
    if not scored:
        raise SystemExit("no step falls in the eval range")

    parents = [int(s[pid]) for s in scored]

    # The parent must actually carry the FROM residue, or the step is being scored
    # against a background it did not happen on.
    for s in scored:
        p, pos0, frm = int(s[pid]), int(s["position_0based"]), s["from_aa"]
        have = seqs[p][pos0]
        if have != frm:
            raise SystemExit(
                f"step {s['step']}: parent {p} has {have} at 1-based "
                f"{int(s['position_1based'])}, the path says {frm}")
    print(f"  all {len(scored)} parents carry the expected from-residue")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model_config = load_checkpoint(args.checkpoint, map_location=device)
    model = create_model(model_config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    gap_idx = aas.find("-") if "-" in aas else None
    stop_idx = aas.find("*") if "*" in aas else None
    mm = build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx).to(device)

    aa_batch = torch.tensor([[aas.index(c) for c in seqs[p]] for p in parents],
                            dtype=torch.long)
    with torch.no_grad():
        s = model(aa_batch.to(device)).float()
        s[:, mm] = float("nan")
    scores = s.cpu()[:, eval_range, :]

    pool_mask = eu.real_substitution_mask(R, aas)
    wt_pool = eu.real_wildtype_mask(aa_batch[:, eval_range], aas)
    above, tie, n_valid, filled = rank_fractions(
        scores, aa_batch, eval_range, pool_mask=pool_mask, wt_pool_mask=wt_pool,
        return_filled=True)
    wq = top_q_weight(above, tie, n_valid, args.qtile)
    wq = torch.where(filled, torch.zeros_like(wq), wq)
    nv = int(n_valid[0])
    if (n_valid != nv).any():
        raise SystemExit(f"the candidate pool differs between queries ({n_valid.tolist()})")
    print(f"  candidate pool: {nv} cells per query")

    by_step = {}
    for b, st in enumerate(scored):
        r = eval_range.index(int(st["position_0based"]))
        j = aas.index(st["to_aa"])
        a, t = float(above[b, r, j] * nv), float(tie[b, r, j] * nv)
        by_step[st["step"]] = {
            "rank_best": int(round(a)) + 1,
            "rank_mean": f"{a + (t + 1.0) / 2.0:.1f}",
            "n_tied": int(round(t)) - 1,
            "n_candidates": nv,
            "pct_from_top": f"{100.0 * (a + (t + 1.0) / 2.0) / nv:.3f}",
            "in_top_decile": f"{float(wq[b, r, j]):.4f}",
            "score": f"{float(scores[b, r, j]):.8f}",
        }

    carry = [c for c in ("step", "mutation_1based", "from_aa", "position_1based",
                         "to_aa", "gene", "gene_position", "parent_example_id",
                         "example_id", "child_subclade", "child_date")
             if c in (steps[0].keys() if steps else [])]
    fields = carry + ["in_pool", "rank_best", "rank_mean", "n_tied", "n_candidates",
                      "pct_from_top", "in_top_decile", "score"]
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for st in steps:
            row = {k: st.get(k, "") for k in carry}
            row["in_pool"] = int(st["step"] in by_step)
            row.update(by_step.get(st["step"], {}))
            w.writerow(row)
    print(f"wrote {args.output}")
    for st in steps:
        if st["step"] in by_step:
            d = by_step[st["step"]]
            print(f"  {st['mutation_1based']:>8s}  rank {d['rank_best']:5d} / {nv}"
                  f"   top {d['pct_from_top']:>7s}%   decile={d['in_top_decile']}")


if __name__ == "__main__":
    main()
