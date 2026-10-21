#!/usr/bin/env python
"""The yield row (G-J) and the background row (K-L) for a frozen set of ensemble rules.

``ensemble_report.py`` covers the recall panels A-F, which are the ones the rule was
chosen on. This is the rest of the figure, and neither row needs a forward pass:

* **G-J** (``list_variant_recall.csv``) puts ONE query to the model -- the BA.2 testing
  root -- and that genotype is already in the list dump. It is found here by matching
  ``aa_idx`` against every dumped query rather than by quoting an index, because an index
  is only correct for the dump it was read off.
* **K-L** (``background_summary.csv``) ranks each cell across the 402 backbone genotypes,
  and ``evals_sc2_background/scores_*.npz`` already holds every member's map over them.
  This script writes the COMBINED map as another ``scores_*.npz`` and leaves the contrast
  itself to ``background_contrast.py``, unmodified -- the published path stays the
  published path, and the ensemble enters it as just another model.

**Why K-L is drawable at all, since this repo said for a while that it was not.** The
claim was that a within-query normalisation destroys the cross-query level information
the background contrast reads. It does not: that panel's ruler compares cells *within*
one query (``common_percentiles``: ``vals > s_c``, both from query ``q``) and holds only
the reference pool's CELL SET fixed across queries. So it is a within-query rank taken at
many queries and then contrasted host-against-control, and a strictly monotone
within-query transform leaves it unchanged to the bit. NOTES s50 has the measurement;
``ensemble_rules``' docstring has the correction and the weaker statement that survives.

**Both rows are gated the same way as s48's GATE 1 and s49's GATE 2**, and for the same
reason: a one-member ensemble is a no-op on ranks (``test_one_member_is_a_no_op``), so
each member, pushed through this script as a one-member rule, must reproduce its own
published row. Until that holds, no ensemble row is written. ``--gate-only`` runs just
that check.

**The -inf convention, reconciled rather than assumed.** ``compute_hit_rate`` folds -inf
into NaN and drops it from ``n_valid``; ``list_variant_recall.top_q_weights`` tests only
``isnan`` and so keeps -inf in the pool, despite a docstring that says otherwise. The two
therefore disagree on the top-decile denominator for any member that emits -inf. Measured
on the nine antiGen-family members: **none of them emits a single -inf cell**, so the two
conventions coincide here and the published convention (keep) is used. The check is run
every time rather than remembered, because it is a property of the members, not of the
code -- add the codon baseline to a rule and it starts to matter.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from antiGen import datasets
from antiGen import ensemble_rules as er
from antiGen import score_cache as sc
from antiGen.eval_utils import RBD_RANGE
from antiGen.list_macro import load_examples, load_mutation_list, load_pssms
from antiGen.list_variant_recall import curve

ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"
VARR_OUT = "list_variant_recall_ensemble.csv"
VARR_FIELDS = ["model", "list", "subset", "threshold", "step", "n_targets",
               "n_selected", "value"]


def load_rules(path):
    """-> [(Rule, figure_label)], rejecting two specs that are secretly one rule."""
    with open(path) as f:
        specs = json.load(f)
    if isinstance(specs, dict):
        specs = [specs]
    out, seen = [], {}
    for spec in specs:
        spec = dict(spec)
        spec.pop("family", None)
        flab = spec.pop("figure_label", None)
        r = er.Rule(family="frozen",
                    **{k: (tuple(v) if isinstance(v, list) else v)
                       for k, v in spec.items()})
        if r.rule_id in seen:
            raise SystemExit(f"two specs collapse to the same rule {r.rule_id}: {r.label}")
        seen[r.rule_id] = True
        out.append((r, flab or r.label))
    return out


def one_member_rule(member, template):
    """``template``'s algebra over a single member: the identity the gates lean on."""
    return er.Rule(family="gate", norm=template.norm, pool=template.pool,
                   nan=template.nan, members=(member,))


# ── G-J: the yield row ───────────────────────────────────────────────────────────

def build_varr_substrate(data_dir, mutation_lists, variant=None):
    """Everything ``list_variant_recall.main`` derives before it touches a model."""
    variant = variant or datasets.default_variant("sc2")
    files = sorted(glob.glob(os.path.join(data_dir, "examples_cutoff_*.txt")))
    if not files:
        raise SystemExit(f"No examples_cutoff_*.txt in {data_dir}")
    examples = load_examples(files[0], ALLOWED_AAS, variant)
    seq_len = len(examples[0].aa_idx)
    roots = [e for e in examples if e.is_testing_root]
    if not roots:
        raise SystemExit(f"No testing root for variant {variant!r}")
    root = roots[0]
    train0 = [e for e in examples if e.is_training][0]

    E = len(ALLOWED_AAS)
    pssm_root, pssm_variant = load_pssms(os.path.join(data_dir, "pssms"), variant, E)
    if not pssm_variant:
        raise SystemExit("No variant PSSMs: there is no time grid to sweep.")
    pssm_variant_sub = [pv[RBD_RANGE, :] for pv in pssm_variant]
    seen_sub = pssm_root[RBD_RANGE, :] > 0

    wt_keep = torch.ones(len(RBD_RANGE), E, dtype=torch.bool)
    wt_keep.scatter_(1, root.aa_idx[RBD_RANGE].unsqueeze(-1), False)

    thresholds = [100, 1] if variant == "BA.2" else [10, 1]
    target_sets = {}
    for item in mutation_lists:
        label, path = item.split("=", 1)
        cells, _ = load_mutation_list(
            path, ALLOWED_AAS, seq_len, label=label,
            root_seq="".join(ALLOWED_AAS[i] for i in train0.aa_idx.tolist()))
        in_full = torch.zeros(seq_len, E, dtype=torch.bool)
        for pos, aa in cells:
            in_full[pos, aa] = True
        in_sub = in_full[RBD_RANGE, :]
        for subset, base in (("all", in_sub & wt_keep),
                             ("unseen", in_sub & wt_keep & ~seen_sub)):
            for thr in thresholds:
                target_sets[(label, subset, thr)] = [base & (pv >= thr)
                                                     for pv in pssm_variant_sub]
    return root, wt_keep, target_sets


def locate_root_query(maps_dir, root):
    """The dump index of the variant root, matched on sequence. Never quoted."""
    aa_all, _ = sc.read_queries(maps_dir)
    want = root.aa_idx.numpy().astype(aa_all.dtype)
    hits = np.flatnonzero((aa_all == want[None, :]).all(axis=1))
    if len(hits) != 1:
        raise SystemExit(f"variant root matches {len(hits)} dumped queries, expected 1")
    return int(hits[0])


def varr_rows(name, scores_2d, wt_keep, target_sets, qtile):
    s = scores_2d.clone().float()
    s[~wt_keep] = float("nan")
    rows = []
    for (label, subset, thr), masks in target_sets.items():
        for i, (n_targets, n_selected, value) in enumerate(curve(s, masks, qtile)):
            rows.append(dict(model=name, list=label, subset=subset, threshold=thr,
                             step=i, n_targets=n_targets,
                             n_selected=round(n_selected, 2), value=value))
    return rows


def read_published_varr(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["model"], r["list"], r["subset"], int(r["threshold"]),
                 int(r["step"]))] = (float(r["n_selected"]), float(r["value"]))
    return out


def gate_varr(maps_dir, qi, members, template, wt_keep, target_sets, qtile,
              published, tol):
    """Each member as a one-member rule must reproduce its published G-J rows."""
    print("GATE (G-J): one-member ensembles against list_variant_recall.csv")
    worst, n = 0.0, 0
    for m in members:
        block, _ = sc.load_block(maps_dir, m, qi, qi + 1, neginf_to_nan=False)
        raw = torch.from_numpy(block[0])
        n_neginf = int(torch.isneginf(raw).sum())
        folded = torch.where(torch.isneginf(raw), torch.full_like(raw, float("nan")), raw)
        maps = {m: folded.unsqueeze(0)}
        got = er.apply(maps, one_member_rule(m, template))[0]
        for row in varr_rows(m, got, wt_keep, target_sets, qtile):
            key = (m, row["list"], row["subset"], row["threshold"], row["step"])
            if key not in published:
                raise SystemExit(f"published G-J has no row for {key}")
            d = abs(row["value"] - published[key][1])
            worst = max(worst, d)
            n += 1
        print(f"  {m:45s} -inf cells {n_neginf:5d}  max|d| so far {worst:.2e}")
    print(f"  {n} rows compared, worst |d| = {worst:.3e} -> "
          f"{'PASSED' if worst <= tol else 'FAILED'}")
    if worst > tol:
        raise SystemExit("G-J gate failed; no ensemble row written")
    return worst


# ── K-L: the background row ──────────────────────────────────────────────────────

def load_background_maps(background_dir, members):
    maps = {}
    for m in members:
        path = os.path.join(background_dir, f"scores_{m}.npz")
        if not os.path.exists(path):
            raise SystemExit(f"no background map for {m!r} in {background_dir}")
        with np.load(path) as z:
            arr = z["scores"].astype(np.float32)
            eval_range = z["eval_range"]
        arr = np.where(np.isneginf(arr), np.nan, arr)
        maps[m] = torch.from_numpy(arr)
    return maps, eval_range


def write_background_map(out_dir, name, scores, eval_range):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"scores_{name}.npz")
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, scores=scores.astype(np.float32),
                        eval_range=np.asarray(eval_range))
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule-json", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--mutation-list", action="append", required=True,
                    help="LABEL=PATH, repeatable")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--qtile", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--maps-dir", help="the list dump, for G-J")
    ap.add_argument("--published-varr", help="list_variant_recall.csv to gate against")
    ap.add_argument("--background-dir", help="the scores_*.npz dir, for K-L")
    ap.add_argument("--background-out", help="where to write combined background maps")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rules = load_rules(args.rule_json)
    members = sorted({m for r, _ in rules for m in r.members})
    print(f"{len(rules)} rule(s) over {len(members)} member(s)")
    for r, lab in rules:
        print(f"  ens_{r.rule_id}  {r.label}  -> {lab}")

    # ── G-J ──────────────────────────────────────────────────────────────────
    if args.maps_dir:
        if not args.published_varr:
            raise SystemExit("--maps-dir needs --published-varr to gate against")
        root, wt_keep, target_sets = build_varr_substrate(
            args.data_dir, args.mutation_list, args.variant)
        qi = locate_root_query(args.maps_dir, root)
        print(f"\nvariant root is dumped query {qi}")
        published = read_published_varr(args.published_varr)
        gate_varr(args.maps_dir, qi, members, rules[0][0], wt_keep, target_sets,
                  args.qtile, published, args.tol)

        if not args.gate_only:
            blocks = {}
            for m in members:
                block, _ = sc.load_block(args.maps_dir, m, qi, qi + 1,
                                         neginf_to_nan=True)
                blocks[m] = torch.from_numpy(block)
            rows = []
            for r, _lab in rules:
                if r.norm == "globalz":
                    raise SystemExit(
                        f"rule {r.rule_id} is globalz, whose moments are taken over the "
                        "whole block; a single-query slice would compute them from one "
                        "query. Score G-J for a globalz rule over the full dump instead.")
                combined = er.apply(blocks, r)[0]
                rows += varr_rows(f"ens_{r.rule_id}", combined, wt_keep,
                                  target_sets, args.qtile)
                sel = {rr["n_selected"] for rr in rows if rr["model"] == f"ens_{r.rule_id}"}
                best = max(rr["value"] for rr in rows
                           if rr["model"] == f"ens_{r.rule_id}"
                           and rr["subset"] == "all" and rr["threshold"] == 1)
                print(f"  ens_{r.rule_id}  top decile = {sel.pop():.1f} cells  "
                      f"max (all, 1+) = {best:.4f}")
            path = os.path.join(args.output_dir, VARR_OUT)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=VARR_FIELDS)
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {len(rows)} rows -> {path}")

    # ── K-L ──────────────────────────────────────────────────────────────────
    if args.background_dir:
        out_dir = args.background_out or os.path.join(args.output_dir, "background_maps")
        maps, eval_range = load_background_maps(args.background_dir, members)
        q = next(iter(maps.values())).shape[0]
        print(f"\nbackground: {q} query genotypes, {len(members)} members")
        # The gate's own maps, consumed by background_contrast and compared there.
        for m in members:
            got = er.apply(maps, one_member_rule(m, rules[0][0]))
            write_background_map(out_dir, f"gate_{m}", got.numpy(), eval_range)
        print(f"  wrote {len(members)} one-member gate maps -> {out_dir}")
        if not args.gate_only:
            for r, _lab in rules:
                combined = er.apply(maps, r)
                write_background_map(out_dir, f"ens_{r.rule_id}", combined.numpy(),
                                     eval_range)
            print(f"  wrote {len(rules)} ensemble maps -> {out_dir}")
        print("  now run background_contrast.py over that directory; the gate rows must "
              "reproduce their published lift before any ens_ row is used.")


if __name__ == "__main__":
    main()
