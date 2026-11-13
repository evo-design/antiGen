#!/usr/bin/env python
"""The combination bars on `sc2_results`' own panels, from the test-set score cache.

**Why this exists.** The five drawn rules are cached-map predictors, not checkpoints, so
`eval.py` cannot score them: its roster is whatever `discover_checkpoints` finds. The
list figure solves this with `ensemble_report.py` over `list_score_maps_aa20`, but that
cache holds the 11,523 CoG-UK-list-carrying queries and `sc2_results`' recall panels
score all 71,704 post-cutoff test records. `eval.py --dump-scores` writes the cache this
reads (NOTES s68); this script turns it into rows.

**Both metrics, on one pass of the cache.**

* recall (panels B-E): `compute_hit_rate` over the test queries, event-weighted, exactly
  the call `evaluate_full` makes -- same common pool, same per-variant masking, same
  fractional tie rule.
* precision (panels G-J, L): `variant_hit_rates` at the variant root, which is ONE query
  of the same cache, located **by genotype** rather than by index. The cache is rebuilt
  whenever the query set changes and a moved index would quietly score another
  background (the mistake s67 avoided for panel K).

**It writes `*_ensemble.csv`, and only `ens_*` rows.** `ensemble_report.py`'s module note
says why: the figure checks jobs that overlap against each other, so re-reporting a
single-model row from a second job turns a reproduction into a disagreement. Every row
here is a rule; no comparator is emitted even though computing one would be free.

**The pool is not optional here.** The members abstain on different cells -- that is what
`union_min` resolves -- and a rule's output is NaN wherever the resolved pool is empty.
Scoring that against each model's own emission set would give the ensemble a top-q of a
different size from its own members', which is the comparison the bars exist to make.

Usage:
    python experiments/multitree/ensemble_test_rows.py \
        --maps-dir  /large_storage/.../test_score_maps_aa20 \
        --rules     /large_storage/.../ensembles9_aa20/rules_drawn5.json \
        --root-counts    .../sc2-2021/train/counts_2021_cur.tsv \
        --variant-counts-dir .../sc2-test/counts_ba2 --counts-since 2021-01-01 \
        --root-seq  antiGen-figures/ref/BA.2_aa20.txt \
        --output-dir /large_storage/.../evals_sc2_aa20
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from antiGen import ensemble_rules as er
from antiGen import eval as ev
from antiGen import score_cache as sc
from antiGen.eval_utils import (compute_hit_rate, real_substitution_mask,
                                real_wildtype_mask, variant_hit_rates)
from antiGen.load import load_counts_matrix, load_counts_series

#: eval.py's own grids, imported as values rather than restated, so a change there is a
#: change here. The recall CSV's columns are top_10.0% .. top_90.0%.
N_BUCKETS = 10
EMERGENCE_QUANTILES = [round(0.1 * i, 4) for i in range(1, 11)]


def qtile_labels(n_buckets=N_BUCKETS):
    return [f"top_{(i + 1) * 100 / n_buckets:.1f}%" for i in range(n_buckets - 1)]


def load_rules(path):
    """The drawn rules, as `ensembles9_aa20/rules_drawn5.json` stores them."""
    with open(path) as f:
        specs = json.load(f)
    out = []
    for spec in specs:
        rule = er.Rule(norm=spec["norm"], pool=spec["pool"], nan=spec["nan"],
                       members=tuple(spec["members"]),
                       weights=tuple(spec.get("weights", ())) or (),
                       family=spec.get("family", ""))
        out.append((rule, spec.get("figure_label", rule.label),
                    spec.get("figure_group", "")))
    ids = [r.rule_id for r, _, _ in out]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"two specs collapse to the same rule_id: {ids}")
    return out


def _verify(args, hits, n_tgt, probe):
    """A one-member rule against that member's own row, read off eval.py's CSV."""
    path = os.path.join(args.output_dir,
                        "quantile_hit_rates_all_all-models.csv")
    if not os.path.exists(path):
        print(f"[verify] {path} is not there yet; skipping the gate")
        return
    with open(path, newline="") as f:
        rows = {r[0]: [float(x) for x in r[1:]] for r in list(csv.reader(f))[1:]}
    want = rows.get(args.verify_member)
    if want is None:
        raise SystemExit(
            f"--verify-member {args.verify_member!r} has no row in {path}. The members "
            f"of the val-selected rules are deliberately NOT drawn rungs, so pick one "
            f"of the nine mean-rule members instead.")
    d = max(n_tgt["none"], 1.0)
    got = [v / d for v in hits[(probe.rule_id, "none")]]
    worst = max(abs(a - b) for a, b in zip(got, want))
    print(f"[verify] {args.verify_member}: one-member rule vs its own row, "
          f"worst |delta| = {worst:.3e} over {len(want)} quantiles")
    if worst > 1e-6:
        raise SystemExit(
            f"the cached path does not reproduce {args.verify_member}'s row "
            f"({worst:.3e} > 1e-6). The maps, the query order or the pool disagree "
            f"with the run that wrote the CSV; the bars would not be on its ruler.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps-dir", required=True,
                    help="the test-set score cache written by eval.py --dump-scores")
    ap.add_argument("--rules", required=True, help="rules_drawn5.json")
    ap.add_argument("--root-counts", required=True)
    ap.add_argument("--variant-counts-dir", default=None,
                    help="omit to write the recall rows only")
    ap.add_argument("--counts-since", default=None)
    ap.add_argument("--root-seq", default=None,
                    help="one-letter sequence of the variant root, for locating it in "
                         "the cache BY GENOTYPE. Required with --variant-counts-dir.")
    ap.add_argument("--alphabet", default="ACDEFGHIKLMNPQRSTVWY")
    ap.add_argument("--block", type=int, default=512, help="queries per block")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="the metric is 25k nanquantile calls over 2M-cell tensors; "
                         "on CPU that is hours and on a GPU it is minutes.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--verify-member", default=None, metavar="NAME",
                    help="Also score NAME as a ONE-MEMBER rule and check it against "
                         "that model's own row in quantile_hit_rates_*_all-models.csv. "
                         "A normalisation is monotone within a query, so a one-member "
                         "rule must rank cell-for-cell with its member and the two "
                         "numbers must agree to float noise. This is the gate on the "
                         "whole cached path at once -- the dump's query order, the "
                         "-inf convention, the pool, the tie rule -- and it is the "
                         "only check that the bars are on the same ruler as the rungs "
                         "they sit beside. It costs one extra rule's worth of work.")
    args = ap.parse_args()

    aas = args.alphabet
    E = len(aas)
    man = sc.read_query_manifest(args.maps_dir)
    eval_range = [int(i) for i in man["eval_range"]]
    R, N = len(eval_range), int(man["n_queries"])
    seq_len = int(man["seq_len"])
    os.makedirs(args.output_dir, exist_ok=True)

    rules = load_rules(args.rules)
    if args.verify_member:
        # Same (norm, pool, nan) as the drawn log-prob rules, over one member.
        probe = er.Rule(norm="logsoftmax", pool="arith", nan="union_min",
                        members=(args.verify_member,))
        rules = rules + [(probe, f"VERIFY {args.verify_member}", "")]
    members = sorted({m for rule, _, _ in rules for m in rule.members})
    have = set(sc.list_models(args.maps_dir))
    missing = [m for m in members if m not in have]
    if missing:
        raise SystemExit(
            f"{len(missing)} member(s) have no map in {args.maps_dir}:\n  "
            + "\n  ".join(missing)
            + "\n\nThe nine mean-rule members come off the main eval's "
              "--dump-models; vs10Antigen_s* and hybridWxwPssm_s* need their own "
              "pass (NOTES s68).")
    print(f"{len(rules)} rule(s) over {len(members)} distinct member(s), "
          f"{N} queries x {R} positions x {E} residues")
    for rule, label, _ in rules:
        print(f"  ens_{rule.rule_id}  {label}  ({len(rule.members)} members)")

    # ── the masks, built exactly as eval.py builds them ──────────────────────
    pssm_root = load_counts_matrix(args.root_counts, aas)
    if pssm_root.shape[0] != seq_len:
        raise SystemExit(f"{args.root_counts}: {pssm_root.shape[0]} sites against a "
                         f"{seq_len}-residue sequence")
    seen_full = pssm_root > 0                      # (seq_len, E)
    seen_sub = seen_full[eval_range, :]
    common_pool = real_substitution_mask(R, aas)
    # At 20 letters `*_nogapstop` is the same question as its plain twin -- there is no
    # column to strike -- so the four CSVs the figure reads are two measurements. They
    # are written out in full rather than aliased, because `common.R` reads four files
    # and a missing one is a hole rather than an inherited value.
    recall_variants = [("all", None), ("unseen", seen_full),
                       ("all_nogapstop", None), ("unseen_nogapstop", seen_full)]
    # ...so the two distinct masks are computed once and written twice. Keyed by the
    # mask rather than by the suffix, which is what makes the aliasing a consequence of
    # the alphabet rather than a hardcoded claim about it: at 22 letters the gap/stop
    # mask is not None and these are four groups again.
    groups = {}
    for suffix, mask in recall_variants:
        key = "none" if mask is None else "seen"
        groups.setdefault(key, (mask, []))[1].append(suffix)
    print("recall variants: " + "; ".join(
        f"{'+'.join(sfx)} -> one pass" for _k, (_m, sfx) in groups.items()))

    aa_all, _tgt = sc.read_queries(args.maps_dir)
    aa_all = torch.from_numpy(aa_all.astype(np.int64))
    if aa_all.shape != (N, seq_len):
        raise SystemExit(f"queries.npz holds {tuple(aa_all.shape)}, manifest says "
                         f"{(N, seq_len)}")
    qidx, pos, alt, cnt = _tgt

    # ── recall ───────────────────────────────────────────────────────────────
    device = torch.device(args.device)
    # The precision row is one query and stays on the CPU, so keep an unmoved copy.
    common_pool_cpu = common_pool
    common_pool = common_pool.to(device)
    rule_ids = [r.rule_id for r, _, _ in rules]
    hits = {(rid, key): [0.0] * (N_BUCKETS - 1) for rid in rule_ids for key in groups}
    n_tgt = {key: 0.0 for key in groups}

    for start in range(0, N, args.block):
        stop = min(start + args.block, N)
        maps = {}
        for m in members:
            blk, _ = sc.load_block(args.maps_dir, m, start, stop, neginf_to_nan=True)
            maps[m] = torch.from_numpy(np.asarray(blk, dtype=np.float32)).to(device)
        aa_b = aa_all[start:stop].to(device)
        # Targets at FULL sequence length, which is what compute_hit_rate slices.
        sel = (qidx >= start) & (qidx < stop)
        tgt = torch.zeros(stop - start, seq_len, E, device=device)
        tgt[torch.from_numpy(qidx[sel].astype(np.int64)).to(device) - start,
            torch.from_numpy(pos[sel].astype(np.int64)).to(device),
            torch.from_numpy(alt[sel].astype(np.int64)).to(device)] = \
            torch.from_numpy(cnt[sel].astype(np.float32)).to(device)
        wt_pool = real_wildtype_mask(aa_b[:, eval_range], aas)

        per_group = {}
        for key, (mask, _sfx) in groups.items():
            if mask is None:
                per_group[key] = (common_pool, tgt)
            else:
                md = mask.to(device)
                v_tgt = tgt.clone()
                v_tgt[md.unsqueeze(0).expand_as(v_tgt)] = 0
                per_group[key] = (common_pool & ~md[eval_range, :], v_tgt)
            # Over the eval range only, which is what compute_hit_rate counts.
            # queries.npz holds nothing outside it, so this is a tautology today and
            # a guard if the dump ever widens.
            n_tgt[key] += float(per_group[key][1][:, eval_range, :].sum())

        for rule, _, _ in rules:
            scores = er.apply(maps, rule)              # (B, R, E), NaN off-pool
            for key, (pool, v_tgt) in per_group.items():
                for i in range(N_BUCKETS - 1):
                    h, _n = compute_hit_rate(scores, v_tgt, aa_b, (i + 1) / N_BUCKETS,
                                             eval_range=eval_range, pool_mask=pool,
                                             wt_pool_mask=wt_pool)
                    hits[(rule.rule_id, key)][i] += h
        if (start // args.block) % 10 == 0:
            print(f"  [recall] {stop}/{N}", flush=True)

    for suffix, mask in recall_variants:
        path = os.path.join(args.output_dir,
                            f"quantile_hit_rates_{suffix}_ensemble.csv")
        key = "none" if mask is None else "seen"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model"] + qtile_labels())
            d = max(n_tgt[key], 1.0)
            for rule, label, _ in rules:
                if label.startswith("VERIFY "):
                    continue        # a probe, not a predictor the figure draws
                w.writerow([f"ens_{rule.rule_id}"]
                           + [f"{v / d:.8f}" for v in hits[(rule.rule_id, key)]])
        print(f"[write] {path}  ({n_tgt[key]:.0f} target events)")

    if args.verify_member:
        _verify(args, hits, n_tgt, probe)

    if not args.variant_counts_dir:
        print("no --variant-counts-dir: recall rows only")
        return

    # ── precision, at the variant root ───────────────────────────────────────
    if not args.root_seq:
        raise SystemExit("--root-seq is required with --variant-counts-dir: the root "
                         "is located by genotype, never by index.")
    with open(args.root_seq) as f:
        root_str = "".join(f.read().split())
    if len(root_str) != seq_len:
        raise SystemExit(f"{args.root_seq}: {len(root_str)} residues against a "
                         f"{seq_len}-residue sequence")
    root_idx = torch.tensor([aas.index(c) for c in root_str], dtype=torch.int64)
    match = (aa_all == root_idx.unsqueeze(0)).all(dim=1).nonzero().flatten().tolist()
    if not match:
        raise SystemExit(f"no cached query matches {args.root_seq}. The cache is for a "
                         f"different corpus, or the reference is the published root "
                         f"rather than the revised one (they differ at Q493R, inside "
                         f"the RBD -- NOTES s67).")
    qi = match[0]
    print(f"variant root: query {qi}" + (f" ({len(match)} identical rows)"
                                         if len(match) > 1 else ""))

    steps, pssm_variant = load_counts_series(args.variant_counts_dir, aas,
                                             since=args.counts_since)
    pv_sub = [pv[eval_range, :] for pv in pssm_variant]
    pv_sub_unseen = []
    for pv in pv_sub:
        m = pv.clone()
        m[seen_sub] = float('nan')
        pv_sub_unseen.append(m)
    print(f"emergence grid: {len(steps)} step(s), {steps[0]} .. {steps[-1]}")

    wt_mask = torch.ones(R, E, dtype=torch.bool)
    wt_mask.scatter_(1, root_idx[eval_range].unsqueeze(-1), False)
    root_real = real_wildtype_mask(root_idx[eval_range].unsqueeze(0), aas)[0]
    base_pool = common_pool_cpu & wt_mask & root_real.unsqueeze(-1)

    root_maps = {}
    for m in members:
        blk, _ = sc.load_block(args.maps_dir, m, qi, qi + 1, neginf_to_nan=True)
        root_maps[m] = torch.from_numpy(np.asarray(blk, dtype=np.float32))

    specs = [("all", pv_sub, 100, base_pool),
             ("unseen", pv_sub_unseen, 100, base_pool & ~seen_sub),
             ("all_thresh1", pv_sub, 1, base_pool),
             ("unseen_thresh1", pv_sub_unseen, 1, base_pool & ~seen_sub),
             ("all_nogapstop", pv_sub, 100, base_pool),
             ("all_nogapstop_thresh1", pv_sub, 1, base_pool),
             ("unseen_nogapstop", pv_sub_unseen, 100, base_pool & ~seen_sub),
             ("unseen_nogapstop_thresh1", pv_sub_unseen, 1, base_pool & ~seen_sub)]

    rows = {suffix: {} for suffix, *_ in specs}
    for rule, label, _ in rules:
        if label.startswith("VERIFY "):
            continue
        s2d = er.apply(root_maps, rule)[0]             # (R, E)
        s_unseen = s2d.clone()
        s_unseen[seen_sub] = float('nan')
        for suffix, pvs, thr, pool in specs:
            rows[suffix][f"ens_{rule.rule_id}"] = variant_hit_rates(
                s_unseen if "unseen" in suffix else s2d, wt_mask, pvs,
                threshold_count=thr, quantiles=EMERGENCE_QUANTILES, pool_mask=pool)

    # The same sidecar ensemble_report.py writes, under the same name and format, so
    # the figure wrapper reads its row names from disk instead of carrying a
    # hand-copied blake2b hash -- the bar drawn is provably the rule that was scored.
    # A VERIFY probe is not a drawn rule and is kept out of it.
    rows_path = os.path.join(args.output_dir, "ensemble_figure_rows.tsv")
    with open(rows_path, "w") as f:
        for rule, label, grp in rules:
            if label.startswith("VERIFY "):
                continue
            f.write(f"ens_{rule.rule_id}\t{label}" + (f"\t{grp}" if grp else "") + "\n")
    print(f"[write] {rows_path}")

    # eval.py's labeller, imported rather than restated: the ensemble CSV is rbind-ed
    # onto the main one by the figure, which asserts the two headers are identical, so
    # a second copy of this convention is a header mismatch waiting to happen.
    step_labels = ev.emergence_step_labels(len(steps), steps)
    for suffix, *_ in specs:
        path = os.path.join(args.output_dir,
                            f"variant_emergence_{suffix}_ensemble.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "top_quantile"] + step_labels)
            for name in sorted(rows[suffix]):
                for q in EMERGENCE_QUANTILES:
                    w.writerow([name, f"{q:.1f}"]
                               + [f"{v:.8f}" for v in rows[suffix][name][q]])
        print(f"[write] {path}")


if __name__ == "__main__":
    main()
