#!/usr/bin/env python
"""Score frozen rules on the reported panels, with paired intervals, and emit the CSV.

This is the step that is allowed to look at the reported cells, and it runs once. What it
scores must already have been fixed before it runs; this script takes that choice as an
input file and does not search.

**A rule file may hold one rule or a set of them, and the distinction is about honesty
rather than batching.** One rule is the output of ``ensemble_freeze.py``: a rule chosen
on the SELECT cells, reported once. A set is for a family specified in advance and
reported *whole* -- the nine static rules of three normalisations x {mean, max, min},
say, where nothing is fitted and no member or parameter is chosen, so there is nothing
to select on and every rule in the family gets a bar. What the list mode must never be
used for is running several rules and drawing the best: that is selection on the reported
cells with the selection step hidden.

**Two comparators, because the obvious one is biased.** Panel C's 0.5125 is the maximum
over 30 rungs, taken with the reported panel in view, so "the ensemble beats the panel
leader" is a contrast against an optimistically selected comparator. The second
comparator is ``simple_d128_n2_lr0.0001`` -- antiGen, the rung ``fig_coguk_list.R``'s
``curve_panel`` keeps in every panel regardless of rank, and therefore pre-specified.
Both are reported; neither alone is the claim.

**The resampling unit is the CELL**, because the metric is an unweighted mean over cells,
and the bootstrap is paired because every rule scored the same cells. Two things are
printed with every interval:

* n is 235 and 131, not 26,491. The intervals are wide. That is s31/s32's finding about
  this metric, not a defect of this script.
* each per-cell rate is itself an average over 1 to 972 events, and a cell-level
  bootstrap does not propagate that within-cell noise. ``--boot-events`` adds the second
  level (resample cells, then events within each resampled cell) as the conservative
  reading. It models the within-cell draw as Bernoulli at the observed rate, which is an
  approximation: the stored per-cell hit mass is fractional at the top-q boundary, so the
  individual event outcomes are not recoverable from it.

**And the noise floor.** NOTES s47 measured antiGen's seed spread on these exact panels
at 0.028-0.035 (0.098 on the event-weighted de novo cut). A gain inside that band is not
a result, and the floor is printed beside every delta rather than left to be remembered.

Output: ``list_macro_summary_ensemble.csv`` in ``list_macro.py``'s exact schema, with
model names ``ens_<rule_id>`` and NOTHING else -- see the module notes on why the figure's
cross-job agreement check makes re-reporting a single-model row dangerous.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from antiGen import ensemble_rules as er
from antiGen import ensemble_sweep as es
from antiGen.background_discrimination import rank_fractions, top_q_weight

SUMMARY = "list_macro_summary_ensemble.csv"
CONTRASTS = "ensemble_contrasts.csv"
PERCELL = "ensemble_percell.csv"
ROWS = "ensemble_figure_rows.tsv"

#: NOTES s47, panels C/D: the spread of three antiGen retrains differing only in seed.
#: The only seed replicates that exist for this roster, so the only empirical floor.
SEED_FLOOR = {"fullpool": 0.035, "unseen_fullpool": 0.028}


def percell_for_rules(sub, rules, pool_name, qtiles, device="cpu"):
    """{rule_id: {q: (cell_hits, cell_events)}} over the whole substrate, in ONE pass.

    Every predictor this script reports on -- each frozen rule and each single-model
    comparator -- is a :class:`~antiGen.ensemble_rules.Rule` here, a comparator being a
    one-member rule. That is exactly the identity ``test_one_member_is_a_no_op`` pins
    through ``compute_hit_rate``, so the comparator values this returns are the
    published single-model values and not a second implementation of them.

    The loop is block-outermost and shares one normalised stack across the rules that
    use it, for the reason ``ensemble_sweep.score_rules`` gives: a rule-outermost loop
    re-reads every member's 318 MB map once per rule, which over a dozen rules and two
    pools is ~100 GB of I/O to recompute arrays that differ only after pooling.
    """
    pmask = sub.pool_mask(pool_name)
    needed = sorted({m for r in rules for m in r.members})
    norms = sorted({r.norm for r in rules})
    idx_of = {m: i for i, m in enumerate(needed)}
    seen_dev = sub.seen.to(device)
    acc = {}
    for start, stop in sub.blocks():
        aa_b = sub.aa_eval[start:stop].to(device)
        tgt_b = sub.targets_block(start, stop).to(device)
        wtm = sub.wt_pool(start, stop)
        wtm = wtm.to(device) if wtm is not None else None
        masked = [es._wt_masked(sub.raw(m, start, stop).to(device), aa_b, pmask, wtm)
                  for m in needed]
        valid = [~torch.isnan(x) for x in masked]
        nv = torch.stack([v.reshape(v.shape[0], -1).sum(dim=1).clamp(min=1)
                          for v in valid])
        stacks = {n: torch.stack([er.normalize(masked[k], n, valid[k])
                                  for k in range(len(needed))]) for n in norms}
        del masked
        for rule in rules:
            sel = [idx_of[m] for m in rule.members]
            filled, ok = er.resolve_nan(stacks[rule.norm][sel], rule.nan)
            s = er.pool(filled, rule.w, rule.pool, beta=rule.beta, k=rule.k,
                        qprime=rule.qprime, trim=rule.trim, n_valid=nv[sel])
            if rule.gate is not None:
                ws, wu = rule.gate
                g = torch.where(seen_dev.expand_as(s), torch.full_like(s, float(ws)),
                                torch.full_like(s, float(wu)))
                s = g * s + (1.0 - g) * filled.mean(dim=0)
            s = torch.where(ok, s, torch.full_like(s, float('nan')))
            above, tie, n_valid = rank_fractions(s, aa_b, list(range(s.shape[1])))
            a = acc.setdefault(rule.rule_id, {})
            for q in qtiles:
                W = top_q_weight(above, tie, n_valid, q)
                h, e = (W * tgt_b).sum(dim=0), tgt_b.sum(dim=0)
                if q not in a:
                    a[q] = [torch.zeros_like(h), torch.zeros_like(e)]
                a[q][0] += h
                a[q][1] += e
        del stacks
    return {rid: {q: (v[0].cpu(), v[1].cpu()) for q, v in a.items()}
            for rid, a in acc.items()}


def paired_bootstrap(rate_a, rate_b, events, n_boot, seed, two_level=False):
    """(delta, lo, hi) for mean(a) - mean(b), resampling CELLS with replacement.

    Paired: one resample of the cell index is applied to both predictors, which is what
    makes the interval narrower than the difference of two independent intervals and is
    legitimate because both scored the same cells.
    """
    rng = np.random.default_rng(seed)
    n = len(rate_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    a, b = rate_a[idx], rate_b[idx]
    if two_level:
        k = np.maximum(events[idx], 1.0)
        a = rng.binomial(k.astype(np.int64), np.clip(a, 0, 1)) / k
        b = rng.binomial(k.astype(np.int64), np.clip(b, 0, 1)) / k
    d = a.mean(axis=1) - b.mean(axis=1)
    return (float(rate_a.mean() - rate_b.mean()),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def auc_sorted(xs, ys):
    """Trapezoid over the sorted grid with the (0,0) and (1,1) anchors fig_coguk_list adds."""
    pts = sorted(set([(0.0, 0.0)] + list(zip(xs, ys)) + [(1.0, 1.0)]))
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return float(sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2 for i in range(len(x) - 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    es.add_substrate_args(ap)
    ap.add_argument('--rule-json', required=True,
                    help='The frozen rule, as written by ensemble_sweep (a dict with '
                         'norm/pool/nan/members and optionally weights and params), or '
                         'a JSON LIST of such dicts to report several rules at once. A '
                         'list is for a set of rules fixed in advance and reported '
                         'whole -- a family of parameter-free static rules, say -- NOT '
                         'for reporting the best of several: each rule in it gets its '
                         'own bar, so the reader sees the losers too. Each may carry a '
                         '"figure_label" for its bar.')
    ap.add_argument('--compare', default='simple_d128_n2_lr0.0001',
                    help='Comma-separated comparator members. The panel incumbents are '
                         'added automatically.')
    ap.add_argument('--qtiles', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--boot-events', action='store_true',
                    help='Two-level bootstrap: resample cells, then events within cell.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--figure-label', default=None,
                    help="The bar's label on the sibling figure. Defaults to the rule's "
                         "own compact description.")
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    qtiles = [float(q) for q in args.qtiles.split(',') if q.strip()]
    sub, _label = es.substrate_from_args(args)
    with open(args.rule_json) as f:
        specs = json.load(f)
    if isinstance(specs, dict):
        specs = [specs]
    rules, fig_labels = [], {}
    for spec in specs:
        spec = dict(spec)
        spec.pop("family", None)
        flab = spec.pop("figure_label", None)
        r = er.Rule(family="frozen", **{k: (tuple(v) if isinstance(v, list) else v)
                                        for k, v in spec.items()})
        if r.rule_id in fig_labels:
            # Two specs with the same canonical form are ONE rule. Drawing it twice
            # would put the same evidence on the figure under two names.
            raise SystemExit(f"two specs collapse to the same rule {r.rule_id}: "
                             f"{r.label}")
        fig_labels[r.rule_id] = flab or args.figure_label or r.label
        rules.append(r)
    print(f"\n{len(rules)} frozen rule(s):")
    for r in rules:
        print(f"  {r.rule_id}  {r.label}  ({len(r.members)} members) "
              f"-> {fig_labels[r.rule_id]}")
    missing = sorted({m for r in rules for m in r.members} - set(sub.members))
    if missing:
        raise SystemExit(f"a frozen rule names members with no cached map: "
                         f"{', '.join(missing)}")

    comparators = [c.strip() for c in args.compare.split(',') if c.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows, contrast_rows, percell_rows = [], [], []
    # Every member as a one-member rule, so the panel incumbent is identified from the
    # data rather than hard-coded -- which also means the comparison cannot go stale if
    # a rung moves -- and so that it is scored through the same code path as the
    # ensembles, on the identity the no-op test pins.
    mem_rule = {m: er.Rule(norm="pctile", pool="arith", nan="union_min", members=(m,),
                           family="member") for m in sub.members}

    for pool_name in es.POOLS:
        sets = sub.cellsets(pool_name,
                            position_disjoint=args.select_position_disjoint)
        print(f"\n=== {pool_name} ===", flush=True)
        acc = percell_for_rules(sub, rules + list(mem_rule.values()), pool_name,
                                qtiles, device=args.device)

        for cs_name, cs_mask in sets.items():
            ce = (acc[rules[0].rule_id][qtiles[0]][1] * cs_mask).numpy()
            obs = ce > 0
            nobs = int(obs.sum())
            if not nobs:
                continue

            def rates(a, q):
                h = (a[q][0] * cs_mask).numpy()[obs]
                e = (a[q][1] * cs_mask).numpy()[obs]
                return np.divide(h, e, out=np.zeros(nobs), where=e > 0)

            mem_q = {m: {q: rates(acc[mem_rule[m].rule_id], q) for q in qtiles}
                     for m in sub.members}
            incumbent = max(sub.members, key=lambda m: mem_q[m][qtiles[0]].mean())
            inc_auc_best = max(sub.members,
                               key=lambda m: auc_sorted(qtiles,
                                                        [mem_q[m][q].mean() for q in qtiles]))
            todo = list(dict.fromkeys(comparators + [incumbent, inc_auc_best]))
            floor = SEED_FLOOR[pool_name]
            print(f"  {cs_name}: {nobs} observed cells | best single "
                  f"{incumbent[:34]} q={qtiles[0]:.2f} "
                  f"{mem_q[incumbent][qtiles[0]].mean():.4f}")

            for rule in rules:
                ens = acc[rule.rule_id]
                ens_q = {q: rates(ens, q) for q in qtiles}
                ens_auc = auc_sorted(qtiles, [ens_q[q].mean() for q in qtiles])

                if cs_name == "report":
                    variant = (f"list_{_label}_fullpool" if pool_name == "fullpool"
                               else f"list_{_label}_unseen_fullpool")
                    for q in qtiles:
                        h = (ens[q][0] * cs_mask).numpy()
                        e = (ens[q][1] * cs_mask).numpy()
                        summary_rows.append([
                            f"ens_{rule.rule_id}", variant, f"{q:.2f}",
                            f"{ens_q[q].mean():.8f}",
                            f"{h.sum() / max(e.sum(), 1e-12):.8f}", nobs,
                            f"{e.sum():.0f}"])

                print(f"    [{fig_labels[rule.rule_id]}] q={qtiles[0]:.2f} "
                      f"{ens_q[qtiles[0]].mean():.4f}, AUC {ens_auc:.4f}")
                for m in todo:
                    if m not in mem_q:
                        print(f"      [skip] comparator {m} has no cached map")
                        continue
                    q0 = qtiles[0]
                    d, lo, hi = paired_bootstrap(ens_q[q0], mem_q[m][q0],
                                                 (ens[q0][1] * cs_mask).numpy()[obs],
                                                 args.n_boot, args.seed,
                                                 two_level=args.boot_events)
                    m_auc = auc_sorted(qtiles, [mem_q[m][q].mean() for q in qtiles])
                    kind = ("panel incumbent (biased: max over %d rungs)" % len(sub.members)
                            if m in (incumbent, inc_auc_best) else "pre-specified")
                    verdict = "clears" if lo > 0 else ("inside CI" if hi > 0 else "below")
                    beats_floor = "yes" if d > floor else "NO (inside the seed floor)"
                    print(f"      vs {m[:40]:40s} d={d:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                          f"{verdict:9s} >floor({floor:.3f}): {beats_floor}")
                    contrast_rows.append(dict(
                        rule_id=rule.rule_id, label=fig_labels[rule.rule_id],
                        pool=pool_name, cellset=cs_name,
                        comparator=m, comparator_kind=kind, qtile=q0, n_cells=nobs,
                        ensemble=ens_q[q0].mean(), comparator_value=mem_q[m][q0].mean(),
                        delta=d, lo=lo, hi=hi, seed_floor=floor,
                        beats_seed_floor=bool(d > floor),
                        ensemble_auc=ens_auc, comparator_auc=m_auc,
                        delta_auc=ens_auc - m_auc,
                        two_level=bool(args.boot_events), n_boot=args.n_boot))
                if cs_name == "report":
                    # `rates()` indexes the (R, E) arrays with the boolean `obs`, which
                    # yields the True positions in C order -- so np.nonzero(obs)
                    # enumerates the cells in exactly the order the rate vectors are
                    # laid out in.
                    rr, cc = np.nonzero(obs)
                    ce_full = (ens[qtiles[0]][1] * cs_mask).numpy()
                    for j, (r_i, c_i) in enumerate(zip(rr, cc)):
                        percell_rows.append(dict(
                            rule_id=rule.rule_id, label=fig_labels[rule.rule_id],
                            pool=pool_name,
                            pos1=sub.eval_range[int(r_i)] + 1,
                            alt=es.ALLOWED_AAS[int(c_i)],
                            k_events=float(ce_full[r_i, c_i]),
                            ensemble=float(ens_q[qtiles[0]][j]),
                            incumbent=float(mem_q[incumbent][qtiles[0]][j]),
                            incumbent_name=incumbent,
                            seen=bool(sub.seen[int(r_i), int(c_i)])))

    with open(os.path.join(args.output_dir, SUMMARY), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "qtile", "macro", "micro", "n_cells", "n_events"])
        for r in summary_rows:
            w.writerow(r)
    # ens_* rows ONLY. fig_coguk_list.R cross-checks any model scored by more than one
    # job to < 1e-6 and stops the PUBLISHED figure on a disagreement, so a single-model
    # row re-reported here -- which the verification pass naturally produces -- could
    # halt a render that has nothing to do with this arm.
    assert all(r[0].startswith("ens_") for r in summary_rows)
    for name, rows, fields in (
            (CONTRASTS, contrast_rows, list(contrast_rows[0]) if contrast_rows else []),
            (PERCELL, percell_rows, list(percell_rows[0]) if percell_rows else [])):
        if not rows:
            continue
        with open(os.path.join(args.output_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    # The figure wrapper reads its row names from here rather than carrying a
    # hand-copied hash, so the bar drawn is provably the rule that was scored.
    with open(os.path.join(args.output_dir, ROWS), "w") as f:
        for rule in rules:
            f.write(f"ens_{rule.rule_id}\t{fig_labels[rule.rule_id]}\n")
    print(f"\nWrote {SUMMARY} ({len(summary_rows)} rows), {CONTRASTS} "
          f"({len(contrast_rows)}), {PERCELL} ({len(percell_rows)}), {ROWS} "
          f"in {args.output_dir}")
    for rule in rules:
        print(f"Figure name: ens_{rule.rule_id}  {fig_labels[rule.rule_id]}")


if __name__ == "__main__":
    main()
