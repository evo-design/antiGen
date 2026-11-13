#!/usr/bin/env python
"""Apply the pre-registered selection protocol to a sweep, and freeze ONE rule.

This exists so that the choice of rule is a function of the sweep output rather than a
judgement made after looking at the reported numbers. It reads only the SELECT columns.
It never reads a `report` row, and it says so by refusing to run if asked for one.

**The protocol, fixed before the attached-tree sweep was run.**

1. ``Rule 0`` is the uniform average over the member family's rungs -- the ``subsets``
   stage's row for that family. Nothing is fitted in it, so it cannot be overfitted, and
   NOTES s14 found plain averaging was the only thing that worked on the other arm. The
   backbone dry run reproduced that: the uniform average scored 0.693 on select against
   greedy forward selection's 0.641.
2. A fitted rule **qualifies** only if it beats Rule 0 on the SELECT cells of *both*
   pools, at the headline quantile and on the AUC. The claim being made is that one rule
   beats all four panels, so a rule that wins on one cut and loses on another has not
   earned the claim.
3. Among qualifiers, the frozen rule maximises the mean of the four SELECT statistics
   (q0 and AUC, for each pool), with ties broken toward FEWER distinct members -- the
   simpler mixture, on the usual grounds and because a smaller mixture is cheaper to
   describe and to reproduce.
4. If nothing qualifies, **the frozen rule is Rule 0**. That is a result, not a failure:
   it says the headroom the oracle shows is not reachable by any rule in the grid, which
   is exactly s14's conclusion on the other arm.

The decision trail -- Rule 0's four numbers, how many rules qualified, and the margin of
the winner -- is printed and written beside the rule so the choice can be audited without
re-running anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

POOLS = ("fullpool", "unseen_fullpool")


def auc_sorted(xy):
    pts = sorted(set([(0.0, 0.0)] + list(xy) + [(1.0, 1.0)]))
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return float(sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2 for i in range(len(x) - 1)))


def load_select(path):
    """-> {rule_id: {'meta': row, 'select': {(pool, q): macro}}}, SELECT rows only."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["cellset"] != "select":
                continue
            d = out.setdefault(r["rule_id"], {"meta": r, "select": {}})
            d["select"][(r["pool"], round(float(r["qtile"]), 6))] = float(r["macro"])
            d["meta"] = r
    return out


def stats(entry, q0):
    """The four SELECT statistics the protocol compares on."""
    s = entry["select"]
    out = {}
    for p in POOLS:
        out[(p, "q0")] = s.get((p, round(q0, 6)), float("nan"))
        grid = [(q, v) for (pp, q), v in s.items() if pp == p]
        out[(p, "auc")] = auc_sorted(grid) if len(grid) > 1 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep-csv', required=True)
    ap.add_argument('--family', default='F3_all')
    ap.add_argument('--qtile', type=float, default=0.1)
    ap.add_argument('--require-both-pools', dest='both', action='store_true', default=True)
    ap.add_argument('--allow-single-pool', dest='both', action='store_false',
                    help='Relax rule 2 to the search pool alone. Weakens the claim from '
                         '"one rule beats all four panels" and is recorded as such.')
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    entries = load_select(args.sweep_csv)
    if not entries:
        raise SystemExit(f"{args.sweep_csv} has no `select` rows -- was it written by a "
                         f"sweep that computed both cell sets?")

    rule0_id = None
    for rid, e in entries.items():
        if e["meta"]["family"] == f"subset:{args.family}":
            rule0_id = rid
            break
    if rule0_id is None:
        fams = sorted({e["meta"]["family"] for e in entries.values()
                       if e["meta"]["family"].startswith("subset:")})
        raise SystemExit(f"no `subset:{args.family}` row in the sweep; the protocol needs "
                         f"the parameter-free rule as its baseline. Present: {fams}")

    s0 = stats(entries[rule0_id], args.qtile)
    print(f"Rule 0 (parameter-free, {args.family}): {entries[rule0_id]['meta']['label']}")
    for k in sorted(s0, key=str):
        print(f"  select {k[0]:16s} {k[1]:4s} = {s0[k]:.4f}")
    # A criterion that cannot be evaluated disqualifies EVERY rule and would silently
    # freeze Rule 0 -- the same shape as the filters that removed everything and exited 0
    # in NOTES s30/s31. Refuse instead of reporting an unearned null.
    missing = [k for k, v in s0.items() if v != v and (args.both or k[0] == "fullpool")]
    if missing:
        raise SystemExit(
            "the protocol cannot be evaluated: Rule 0 has no value for "
            + ", ".join(f"{k[0]}/{k[1]}" for k in missing)
            + ". An AUC needs more than one quantile in the sweep grid, and q0 needs "
              f"q={args.qtile} to be in it. Re-run the sweep with the full --qtiles grid; "
              "do NOT read the null this would otherwise produce, because every rule "
              "would be disqualified by the missing criterion rather than by its score.")

    qualifiers = []
    for rid, e in entries.items():
        if rid == rule0_id or e["meta"]["family"] == "verify":
            continue
        st = stats(e, args.qtile)
        keys = [k for k in st if args.both or k[0] == "fullpool"]
        if any(st[k] != st[k] for k in keys):          # NaN anywhere disqualifies
            continue
        if all(st[k] > s0[k] for k in keys):
            qualifiers.append((sum(st[k] for k in keys) / len(keys), rid, st))

    print(f"\n{len(qualifiers)} of {len(entries) - 1} rules beat Rule 0 on "
          f"{'both pools' if args.both else 'the search pool'}, "
          f"at q={args.qtile} AND on the AUC")

    if not qualifiers:
        chosen_id, chosen_st, verdict = rule0_id, s0, "rule0_unbeaten"
        print("\nFROZEN: Rule 0. Nothing in the grid beat a plain uniform average on the "
              "selection cells.\nThat is the NOTES s14 result reproduced, not a failure "
              "of the sweep.")
    else:
        qualifiers.sort(key=lambda t: (-t[0],
                                       len(set(entries[t[1]]["meta"]["members"].split("|")))))
        _score, chosen_id, chosen_st = qualifiers[0]
        verdict = "fitted_rule"
        m0 = sum(s0[k] for k in chosen_st if args.both or k[0] == "fullpool")
        print(f"\nFROZEN: {chosen_id}  {entries[chosen_id]['meta']['label']}")
        print(f"  family {entries[chosen_id]['meta']['family']}, "
              f"margin over Rule 0 on select = "
              f"{_score - m0 / len([k for k in chosen_st if args.both or k[0] == 'fullpool']):+.4f}")

    meta = entries[chosen_id]["meta"]
    members = meta["members"].split("|")
    weights = [float(x) for x in meta["weights"].split("|")] if meta["weights"] else None
    spec = {"norm": meta["norm"], "pool": meta["pooling"], "nan": meta["nan"],
            "members": members}
    if weights and len(set(weights)) > 1:
        spec["weights"] = weights
    for k, col in (("beta", "beta"), ("k", "k"), ("qprime", "qprime"), ("trim", "trim")):
        if meta.get(col):
            spec[k] = float(meta[col])
    if meta.get("gate"):
        spec["gate"] = [float(x) for x in meta["gate"].split("/")]

    with open(args.out_json, "w") as f:
        json.dump(spec, f, indent=1, sort_keys=True)
    trail = os.path.splitext(args.out_json)[0] + "_decision.json"
    with open(trail, "w") as f:
        json.dump({"verdict": verdict, "family": args.family, "qtile": args.qtile,
                   "require_both_pools": bool(args.both),
                   "rule0_id": rule0_id,
                   "rule0_select": {f"{k[0]}:{k[1]}": v for k, v in s0.items()},
                   "chosen_id": chosen_id,
                   "chosen_select": {f"{k[0]}:{k[1]}": v for k, v in chosen_st.items()},
                   "n_rules": len(entries), "n_qualifiers": len(qualifiers),
                   "sweep_csv": os.path.abspath(args.sweep_csv)},
                  f, indent=2, sort_keys=True)
    print(f"\nWrote {args.out_json} and {trail}")
    print(f"Next: antiGen/ensemble_report.py --rule-json {args.out_json} ... "
          f"(this is the first and only step that reads the reported cells)")


if __name__ == "__main__":
    main()
