#!/usr/bin/env python
"""Forward-pass the ten val-selected antiGens on chosen TEST-corpus examples.

**Which examples.** By default the single record in ``antiGen-data-revised``'s test
corpus whose ``IS_VARIANT_ROOT`` block names the dataset's variant -- for sc2 that is
``21L (BA.2),True`` -- found by streaming the file rather than by an example id, so a
regenerated corpus cannot silently point this at another node. Exactly one record must
carry the flag; more or fewer is an error. ``--example-ids`` selects named records
instead, and every one of them must exist.

**Which predictor.** NOTES s56's bar: the ten runs of ``train_antigen_valsel10.sh``, each
stopped at the epoch its own random holdout chose, combined by the published
``logsoftmax`` / ``arith`` / ``union_min`` rule -- the mean of the members' log-probs.
The combination goes through ``antiGen.ensemble_rules`` itself, not a re-implementation
of it. The one free choice left is the SUPPORT the rule normalises over, which
``--support`` names: ``spike`` is the whole (L, E) grid, ``rbd`` is the 201-position
``RBD_RANGE`` that ``ensemble_sweep`` uses for the figure's bar. Within a query the two
differ by one constant per member -- hence by one constant after the mean -- so the
ordering inside the RBD is the same either way, and only the probability column's
conditioning changes. Under ``--support rbd`` the whole-spike columns are carried
alongside, so the two files join.

Wild-type cells are NaN, exactly as ``ensemble_sweep._wt_masked`` writes them: a query's
own residue is not a candidate substitution. With a 20-letter alphabet there is no gap or
stop column to mask, so ``union_min`` has nothing to impute and every member emits every
candidate cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch

from antiGen import ensemble_rules as er
from antiGen.config import STANDARD_AAS
from antiGen.datasets import default_variant
from antiGen.eval_utils import RBD_RANGE
from antiGen.load import _match_variant
from antiGen.models.factory import load_model

RULE = ("logsoftmax", "arith", "union_min")   # querypool_arm/rules9.json, "Mean of log-probs"


def combine(maps, support=None):
    """-> (ens, members_lp, cells, prob, rank) for one normalisation SUPPORT.

    ``maps`` is a list of K (B, L, E) score maps, already NaN on the cells a member does
    not emit and on each query's own residue. ``support`` is an (L, E) bool of the cells
    the rule may see, or None for every cell the members emit. It is not cosmetic:
    ``logsoftmax`` normalises inside it, so a cell's log-prob under the RBD support is
    conditioned on the RBD candidate pool and under the whole-spike support on the whole
    spike. The two differ by one constant per member -- hence by one constant after the
    mean -- so the ORDER inside the RBD is the same either way, which NOTES s58 checks
    against the figure's cached maps.

    Everything below is per query: the softmax, the ranks and the candidate count.
    """
    norm, pool_kind, nan_conv = RULE
    restricted = maps
    if support is not None:
        restricted = [torch.where(support, m, torch.full_like(m, float('nan'))) for m in maps]
    valid = [~torch.isnan(x) for x in restricted]
    per_q = [v.flatten(1).sum(1) for v in valid]
    if any(not torch.equal(n, per_q[0]) for n in per_q):
        raise SystemExit("members disagree about which cells they emit")
    stack = torch.stack([er.normalize(m, norm, v) for m, v in zip(restricted, valid)])
    filled, ens_valid = er.resolve_nan(stack, nan_conv)
    ens = er.pool(filled, [1.0] * len(maps), pool_kind)        # (B, L, E)
    cells = ens_valid
    # exp(mean of log-probs) is the members' geometric mean, which is sub-normalised;
    # renormalising over each query's candidate set makes the column a distribution again.
    masked = torch.where(cells, ens, torch.full_like(ens, float('-inf')))
    flat = masked.flatten(1)                                   # (B, L*E)
    prob = flat.softmax(1).reshape(ens.shape)
    order = torch.argsort(flat, dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(1, flat.shape[1] + 1).repeat(flat.shape[0], 1))
    return ens, stack, cells, prob, rank.reshape(ens.shape)


def scan_records(path, *, variant=None, want_ids=None):
    """-> [(example_id, sequence, parent_mutation)] for the records asked for.

    A streaming re-read of the block structure ``load.load_examples`` parses, kept to the
    fields this script needs. With ``want_ids`` the records are returned in the order
    asked for; otherwise the one record flagged ``IS_VARIANT_ROOT`` for ``variant``.
    ``parent_mutation`` is the raw ``(from, pos, to)`` of ``MUTATION_FROM_PARENT``, kept
    only so the caller can check which position base the file is written in.
    """
    want = None if want_ids is None else set(want_ids)
    found = {}
    with open(path) as fh:
        ex_id = parent_mut = None
        for line in fh:
            s = line.strip()
            if s.startswith("EXAMPLE_ID:"):
                ex_id, parent_mut = int(s.split(":", 1)[1]), None
            elif s == "MUTATION_FROM_PARENT:":
                nxt = next(fh).strip()
                if nxt != "None":
                    a, pos, b = nxt.split(",")
                    parent_mut = (a, int(pos), b)
            elif s == "IS_VARIANT_ROOT:":
                is_root = False
                for s in map(str.strip, fh):
                    if not s.endswith(("True", "False")):
                        break           # s is now the sequence line
                    label, val = s.rsplit(",", 1)
                    if variant is not None and _match_variant(label, variant) and val == "True":
                        is_root = True
                if (want is None and is_root) or (want is not None and ex_id in want):
                    found[ex_id] = (ex_id, s, parent_mut)
                    if want is not None and len(found) == len(want):
                        break
    if want is None:
        if len(found) != 1:
            raise SystemExit(
                f"{path}: expected exactly one IS_VARIANT_ROOT record for variant "
                f"{variant!r}, found {len(found)}" +
                (f" (example ids {sorted(found)})" if found else ""))
        return list(found.values())
    missing = [i for i in want_ids if i not in found]
    if missing:
        raise SystemExit(f"{path}: no record with EXAMPLE_ID {missing}")
    return [found[i] for i in want_ids]


def check_position_base(records, seq_len):
    """Abort unless every record's MUTATION_FROM_PARENT reads 0-indexed.

    ``load._detect_position_base`` is a vote over the whole file and returns positions
    0-indexed whichever base it finds, so the loader absorbs the difference. Nothing here
    goes through the loader -- the sequence line is all a forward pass needs -- so the
    same vote is run over the selected records alone, as a guard on the ``pos0`` column
    this script writes. s54 recorded the revised test corpus as 0-indexed; a regenerated
    one that is not would shift every position by one without failing.
    """
    votes = {0: 0, 1: 0}
    n = 0
    for _ex_id, seq, mut in records:
        if mut is None:
            continue
        n += 1
        for base in (0, 1):
            j = mut[1] - base
            if 0 <= j < seq_len and seq[j] == mut[2]:
                votes[base] += 1
    if n and votes[0] != n:
        raise SystemExit(
            f"MUTATION_FROM_PARENT does not read 0-indexed on the selected records "
            f"({votes[0]}/{n} fit base 0, {votes[1]}/{n} fit base 1). The corpus' "
            f"position convention moved; see load._detect_position_base.")
    return n


def member_checkpoints(root, n):
    """-> [(name, path)] for the N val-selected runs, one ``_best.pt`` each."""
    out = []
    for i in range(n):
        d = os.path.join(root, f"antigen_vs_s{i}")
        best = sorted(f for f in os.listdir(d) if f.startswith("fitness_") and f.endswith("_best.pt"))
        if len(best) != 1:
            raise SystemExit(f"{d}: expected one fitness_*_best.pt, found {len(best)}")
        out.append((f"s{i}", os.path.join(d, best[0])))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-examples",
                   default="/home/ivan.specht/antiGen-data-revised/output/sc2-2021/test/examples_all_cur.txt")
    p.add_argument("--example-ids", default=None,
                   help="Comma-separated EXAMPLE_IDs to score, in the order given. "
                        "Default: the one record flagged IS_VARIANT_ROOT for the variant.")
    p.add_argument("--members-root",
                   default="/large_storage/hielab/ispecht/fitness-inference/valsel10_aa20",
                   help="Directory holding antigen_vs_s0 .. antigen_vs_s<N-1>")
    p.add_argument("--n-members", type=int, default=10)
    p.add_argument("--dataset", default="sc2")
    p.add_argument("--variant", default=None, help="Default: the dataset's own variant")
    p.add_argument("--alphabet", default=STANDARD_AAS)
    p.add_argument("--support", choices=["spike", "rbd"], default="spike",
                   help="Cells the rule normalises over, and the rows written. 'spike': "
                        "the whole 1273 x E grid. 'rbd': eval_utils.RBD_RANGE only, which "
                        "is the support ensemble_sweep uses for the figure's bar.")
    p.add_argument("-o", "--output", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out", "variant_root_ensemble.csv"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    variant = args.variant or default_variant(args.dataset)
    aas = args.alphabet
    E = len(aas)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    if RULE[:2] not in er.LEGAL:
        raise SystemExit(f"rule {RULE[0]}/{RULE[1]} is not in ensemble_rules.LEGAL")

    want_ids = None
    if args.example_ids:
        want_ids = [int(x) for x in args.example_ids.replace(",", " ").split()]
        if len(set(want_ids)) != len(want_ids):
            raise SystemExit(f"--example-ids repeats an id: {want_ids}")

    print(f"=== queries: {args.test_examples} ===", flush=True)
    records = scan_records(args.test_examples, variant=variant, want_ids=want_ids)
    seq_lens = {len(seq) for _, seq, _ in records}
    if len(seq_lens) != 1:
        raise SystemExit(f"selected records have different lengths {sorted(seq_lens)}")
    L = seq_lens.pop()
    bad = sorted({c for _, seq, _ in records for c in seq} - set(aas))
    if bad:
        raise SystemExit(f"queries carry residues outside '{aas}': {bad}")
    n_voted = check_position_base(records, L)
    for ex_id, _seq, mut in records:
        tag = f"{mut[0]}{mut[1] + 1}{mut[2]}" if mut else "root of the corpus"
        print(f"  EXAMPLE_ID {ex_id}, length {L}, from parent by {tag}", flush=True)
    print(f"  position base checked 0-indexed on {n_voted}/{len(records)} record(s)", flush=True)

    aa_idx = torch.tensor([[aas.index(c) for c in seq] for _, seq, _ in records],
                          dtype=torch.long, device=device)

    members = member_checkpoints(args.members_root, args.n_members)
    maps = []
    for name, path in members:
        model, _ = load_model(path, device=device)
        model.eval()
        if model.config.seq_len != L:
            raise SystemExit(f"{name}: model seq_len {model.config.seq_len} != query length {L}")
        if model.config.alphabet.amino_acids != aas:
            raise SystemExit(f"{name}: model alphabet {model.config.alphabet.amino_acids!r} "
                             f"!= {aas!r}")
        with torch.no_grad():
            logits = model(aa_idx).float()                    # (B, L, E)
        # The query's own residue is not a candidate: ensemble_sweep._wt_masked.
        logits = logits.scatter(2, aa_idx.unsqueeze(-1), float('nan'))
        maps.append(logits.cpu())
        print(f"  [{name}] {os.path.basename(path)}", flush=True)
        del model

    spike = combine(maps)
    if args.support == "rbd":
        sup = torch.zeros(L, E, dtype=torch.bool)
        sup[RBD_RANGE] = True
        primary = combine(maps, sup)
    else:
        primary = spike
    ens, members_lp, cells, ens_prob, rank = primary

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    rbd = args.support == "rbd"
    n_rows = 0
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["example_id", "pos0", "pos1", "ref_aa", "alt_aa",
                    "ens_logprob", "ens_prob", "ens_rank"]
                   # Under the RBD support the two probability columns answer different
                   # questions -- "given a mutation in the RBD" and "given a mutation
                   # anywhere in spike" -- and the second is what joins this file to the
                   # whole-spike one. Both are carried rather than one being recoverable.
                   + (["ens_logprob_spike", "ens_prob_spike", "ens_rank_spike"] if rbd else [])
                   + [f"logprob_{n}" for n, _ in members])
        for b, (ex_id, seq, _mut) in enumerate(records):
            for pos in range(L):
                ref = seq[pos]
                for j, alt in enumerate(aas):
                    if not bool(cells[b, pos, j]):
                        continue
                    n_rows += 1
                    w.writerow([ex_id, pos, pos + 1, ref, alt,
                                f"{ens[b, pos, j].item():.8f}",
                                f"{ens_prob[b, pos, j].item():.10e}",
                                int(rank[b, pos, j])]
                               + ([f"{spike[0][b, pos, j].item():.8f}",
                                   f"{spike[3][b, pos, j].item():.10e}",
                                   int(spike[4][b, pos, j])] if rbd else [])
                               + [f"{members_lp[k, b, pos, j].item():.8f}"
                                  for k in range(len(members))])
    os.replace(tmp, args.output)

    per_q = int(cells[0].sum())
    n_pos = int(cells[0].any(dim=1).sum())
    print(f"\n=== {args.output} ===")
    print(f"  {len(records)} quer{'y' if len(records) == 1 else 'ies'} x {per_q} candidate "
          f"cells = {n_rows} rows ({n_pos} positions x {E} residues, wild type removed); "
          f"support = {args.support}")
    print(f"  rule: {'/'.join(RULE)} over {len(members)} members, equal weights")
    for b, (ex_id, seq, _mut) in enumerate(records):
        top = torch.topk(ens_prob[b].flatten(), 10)
        hits = " ".join(f"{seq[i // E]}{i // E + 1}{aas[i % E]} {v:.2e}"
                        for v, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"  [{ex_id}] top 10: {hits}")
    side = {"example_ids": [r[0] for r in records], "variant": variant, "seq_len": L,
            "alphabet": aas, "support": args.support,
            "eval_range": [RBD_RANGE[0], RBD_RANGE[-1]] if rbd else [0, L - 1],
            "rule": dict(zip(("norm", "pool", "nan"), RULE)),
            "members": {n: p for n, p in members},
            "n_candidate_cells_per_query": per_q, "n_rows": n_rows,
            "test_examples": os.path.abspath(args.test_examples)}
    with open(os.path.splitext(args.output)[0] + "_manifest.json", "w") as fh:
        json.dump(side, fh, indent=2)
    print(f"  manifest: {os.path.splitext(args.output)[0] + '_manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
