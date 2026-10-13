#!/usr/bin/env python
"""The mutation kernel's descent gate: measure `p_down` before any forecast is scored.

`PROPOSAL_mutation_kernel.md` section 2 settles the naive version of this round outright.
Write `phi_w = min_j (d(w,j) - z*_j)`, the c-transform of the optimal *target* potential
of the unextended transport. By Danskin the right-derivative of the convolved score at
`mu = 0` is

    dW_1/dmu |_0  =  sum_v x_v sum_{(q,b)} pi(q,b|v) [ phi_{v ⊕ (q,b)} - phi_v ]

and because the Hamming cost is integer-valued, every bracket is exactly -1, 0 or +1. So
with `p_down`, `p_flat` the `x`-weighted shares of proposal mass on each,

    dW_1/dmu |_0  =  p_up - p_down  =  1 - p_flat - 2 p_down

`W_1` is convex in `mu`, so an arm with a positive derivative at zero has its optimum
*at* zero -- not "is expected to lose", but provably cannot help at any `mu`. Setting the
derivative to zero gives the bar:

    p_down*  =  (1 - p_flat) / 2  =  0.446 on the backbone panel

A uniform kernel achieves 0.077. The required concentration is 5.8x against a candidate
set of ~311 per parent of which roughly 24 are downhill, and that is the number the round
lives or dies on.

**Why this is a gate and not a sweep.** It costs one transport solve per transition, is
fitted to nothing, and scores no forecast -- so an arm it rules out costs the
multiple-looks budget nothing. Roughly fifty configurations have already been scored on
these eight transitions; this is the first pre-scoring filter in the line of work.

**Why it is restricted to the training transitions.** `phi` reads `z*`, which reads the
transition's own targets, so `p_down` is an *oracle* diagnostic. Measuring it on a
held-out transition before that transition has been scored would choose an arm on the test
set, which is exactly what the rolling-origin protocol exists to prevent -- and
`Task.descent_gate` raises rather than allowing it. This driver therefore takes the
transitions up to `--gate-through` and never the later ones, which is the same
rolling-origin discipline every hyperparameter here already obeys.

The pre-registered rule, and it is dispositive either way:

> For each arm, measure `p_down` on the training transitions only, `x`-weighted under
> that arm's own `pi`, at each `(tau_k, m)` in the grid. An arm whose best `p_down` does
> not clear 0.446 is **not run** on the held-out transitions and is reported as gated out
> with its number. No forecast is scored for it.

Given uniform sits at 0.077, every arm being gated out is the most likely outcome, and it
is a real result: it closes the one-step convolution route on a measurement rather than on
a sweep.

Usage:
    python -m antiGen.kernel_gate --tree ~/h3n2/tree.json --ref ~/h3n2/ref.json \
        --arms kunif,kemp --tau-ks 1,0.5,0.25 --ms 0,10,3,1
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from antiGen.mutation_kernel import ARMS, BREAK_EVEN, gate, reads_slow
from antiGen.season_momentum import (Task, TipPanel, add_panel_source_args,
                                     tips_from_args)


def gate_rows(panel, transitions, arms, tau_ks, ms, tau_es=(None,),
              tau_e_slows=(None,), verbose=True):
    """One row per `(transition, arm, tau_k, m)`, plus the parent score per transition.

    The parent is whole-season persistence, the same `season` row the protocol's reference
    column reports, and it is scored *first* -- both because the table wants it and
    because `Task.descent_gate` refuses to run on a transition nothing has scored."""
    rows = []
    for s in transitions:
        task = Task(panel, s)
        x = task.baseline("season")
        w = task.score(x)                  # the reference row, and the leak guard's key
        if verbose:
            print(f"  {s}->{s + 1}: season EMD {w:7.4f}  support {len(task.sup):4d}  "
                  f"targets {len(task.tgt_g):3d}")
        for arm in arms:
          # Only the momentum arms read a slow table. Sweeping one for `kcons`, `kemp` or
          # `kunif` would emit configurations identical in every number reported here, and
          # `causal_operating_point`'s argmin among them would be settled by a tie-break
          # rather than by a measurement. So a one-timescale arm gets exactly one slow
          # entry, `None`, and its rows say so.
          slows = tau_e_slows if reads_slow(arm) else (None,)
          for tau_e in tau_es:
           for tau_slow in slows:
            # The ratio is the content of the momentum term, so a slow window at or below
            # the fast one is not a bad setting but a different model. Skipped rather than
            # raised: a rectangular sweep is the natural way to ask, and most of it is
            # valid.
            if (tau_slow is not None and tau_e is not None
                    and not float(tau_slow) > float(tau_e)):
                continue
            for tau_k in tau_ks:
                for m in ms:
                    d = task.descent_gate(x, arm, tau_k, m, tau_e, tau_slow)
                    # `tau_e` goes in the ROW, not only in the run header. A table that
                    # names its timescale once cannot hold two of them, and the causal
                    # consumer picks over (tau_e, tau_k, m) jointly -- it has to see all
                    # three on every row or it is comparing across incomparable runs.
                    rows.append(dict(season=int(s), next=int(s + 1), arm=arm,
                                     tau_e=(None if tau_e is None else float(tau_e)),
                                     tau_e_slow=(None if tau_slow is None
                                                 else float(tau_slow)),
                                     tau_k=float(tau_k), m=int(m), emd_season=w,
                                     n_support=len(task.sup),
                                     n_target=len(task.tgt_g), **d))
                    if verbose:
                        print(f"      {arm:>10} tau_e {str(tau_e):>5}/"
                              f"{str(tau_slow):<5} tau_k {tau_k:5g} m {m:>4}  "
                              f"cands/parent {d['n_cand'] / max(len(task.sup), 1):7.1f}  "
                              f"p_down {d['p_down']:.4f}  p_flat {d['p_flat']:.4f}  "
                              f"break-even {d['break_even']:.4f}  "
                              f"dW/dmu {d['deriv']:+.4f}  greedy {d['greedy']:+.4f}")
    return rows


def summarise(rows, break_even=BREAK_EVEN, verbose=True):
    """Mean `p_down` per `(arm, tau_k, m)` over the transitions, and the gate's verdict."""
    def _key(r):
        return (r["arm"], r.get("tau_e"), r.get("tau_e_slow"), r["tau_k"], r["m"])

    keys = sorted({_key(r) for r in rows},
                  key=lambda k: (k[0], str(k[1]), str(k[2]), k[3], k[4]))
    stats, table = {}, []
    for k in keys:
        sel = [r for r in rows if _key(r) == k]
        # `gate` reads `(arm, tau_k, m)` and reports an arm's BEST p_down, so when a
        # table sweeps `tau_e` as well the entry has to be the max over it rather than
        # whichever tau_e came last -- otherwise the verdict line quietly names the
        # p_down of one timescale under the label of another.
        sk = (k[0], k[3], k[4])
        p = float(np.mean([r["p_down"] for r in sel]))
        stats[sk] = max(stats.get(sk, -np.inf), p)
        table.append(dict(arm=k[0], tau_e=k[1], tau_e_slow=k[2], tau_k=k[3], m=k[4],
                          p_down=float(np.mean([r["p_down"] for r in sel])),
                          p_flat=float(np.mean([r["p_flat"] for r in sel])),
                          deriv=float(np.mean([r["deriv"] for r in sel])),
                          greedy=float(np.mean([r["greedy"] for r in sel])),
                          cands=float(np.mean([r["n_cand"] for r in sel])),
                          n=len(sel)))
    verdict = gate(stats, break_even)
    if verbose:
        print(f"\n{'arm':>10} {'tau_e':>7} {'slow':>7} {'tau_k':>7} {'m':>5} "
              f"{'p_down':>8} {'p_flat':>8} {'dW/dmu':>9} {'greedy':>8} {'cands':>9}")
        for t in sorted(table, key=lambda t: (t["arm"], -t["p_down"])):
            print(f"{t['arm']:>10} {str(t['tau_e']):>7} {str(t['tau_e_slow']):>7} "
                  f"{t['tau_k']:7g} {t['m']:5d} {t['p_down']:8.4f} "
                  f"{t['p_flat']:8.4f} {t['deriv']:+9.4f} {t['greedy']:+8.4f} "
                  f"{t['cands']:9.0f}")
        print(f"\nthe bar is p_down > {break_even:.4f}; a convex score with a positive "
              f"right-derivative at mu = 0 attains its minimum there, so an arm below it "
              f"\ncannot help at any mu and is not scored on a held-out transition:")
        for arm in sorted(verdict):
            v = verdict[arm]
            te = [t["tau_e"] for t in table
                  if t["arm"] == arm and t["tau_k"] == v["tau_k"] and t["m"] == v["m"]
                  and abs(t["p_down"] - v["p_down"]) < 1e-12]
            at_te = f"tau_e {te[0]}, " if len(set(t for t in te)) == 1 and te else ""
            print(f"  {arm:>10}  best p_down {v['p_down']:.4f} at {at_te}tau_k "
                  f"{v['tau_k']:g}, m {v['m']}  -->  "
                  f"{'CLEARS the gate' if v['passes'] else 'GATED OUT'}"
                  + ("   (leaky: reads a model pretrained on the test seasons)"
                     if v["leaky"] else ""))
    return table, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_panel_source_args(ap)
    ap.add_argument("--min-allele-freq", type=float, default=0.005)
    ap.add_argument("--lookback", type=float, default=5.0)
    ap.add_argument("--max-support", type=int, default=0)
    ap.add_argument("--min-season-seqs", type=int, default=30)
    ap.add_argument("--min-train", type=int, default=2)
    ap.add_argument("--drop-targets", default="2020")
    ap.add_argument("--gate-through", type=int, default=2022,
                    help="the last transition the gate may look at, named by the season "
                         "predicted *from* -- so 2022 is the transition predicting 2023, "
                         "and the default reproduces the proposal's training split of "
                         "predict-2018 through predict-2023 exactly. Every table in the "
                         "log is labelled by the season predicted, which is one year "
                         "later than this flag, and getting that off by one would hand "
                         "the gate a transition the protocol holds out. Everything after "
                         "it is held out and its p_down is never computed here: the "
                         "measurement reads the transition's own targets, so restricting "
                         "it is the whole of what keeps the gate honest")
    ap.add_argument("--full-table", action="store_true",
                    help="measure every transition in the panel, not only those up to "
                         "--gate-through, and stamp the output as a causal table. This "
                         "is NOT a gate verdict and must never be read as one: the "
                         "verdict rule is 'measure p_down on the training transitions, "
                         "and an arm that does not clear the bar is never scored', and a "
                         "table that covers the held-out transitions cannot deliver it. "
                         "What the table is for is `season_momentum "
                         "--kernel-gate-table`, which reads each origin only the rows "
                         "whose targets had closed by that origin's horizon and so "
                         "chooses the operating point causally. Producing the rows and "
                         "consuming them are separated precisely so that the causality "
                         "lives in one testable filter instead of in when the driver "
                         "happened to stop")
    ap.add_argument("--transitions", default="",
                    help="gate exactly these transitions instead, comma separated. Use "
                         "it to reproduce a published row, not to reach past "
                         "--gate-through")
    ap.add_argument("--arms", default=",".join(sorted(ARMS)))
    ap.add_argument("--tau-ks", default="1",
                    help="proposal temperatures to sweep, pi proportional to "
                         "(s psi)^(1/tau_k). Sweeping here is free of the exposure "
                         "problem: no forecast is scored")
    ap.add_argument("--ms", default="0",
                    help="truncations to sweep: keep the top m proposals per parent, 0 "
                         "for all of them. This is the lever -- the question is whether "
                         "the top of a proposal ranking is enriched for the right "
                         "directions, and m is how that enrichment becomes a forecast")
    ap.add_argument("--tau-es", default=None,
                    help="sweep several kernel timescales in one table, comma separated; "
                         "'none' for the arm's own default. Each row carries its own "
                         "tau_e, so one file holds the whole (tau_e, tau_k, m) surface "
                         "and the causal consumer can pick over all three jointly. "
                         "Before this the sweep lived across five separate files with "
                         "different tau_k grids in them, which is how a config could be "
                         "argmin on a panel no other config was measured on")
    ap.add_argument("--tau-e-slows", default=None,
                    help="slow site-allele windows to sweep for the consensus momentum "
                         "arm, comma separated; 'none' for the 3x-fast fallback. Until "
                         "2026-09-28 the slow window was pinned at 3x the fast one, "
                         "giving the allele-level momentum one free timescale where the "
                         "genotype-level momentum column it mirrors has two. Pairs with "
                         "slow <= fast are skipped, and arms that read no slow table "
                         "(kcons, kemp, kunif) get a single `None` entry rather than a "
                         "sweep of a field they cannot see")
    ap.add_argument("--tau-e", type=float, default=None,
                    help="discount timescale for kemp's substitution counts, in years; "
                         "the panel's own lookback by default. Declared rather than "
                         "selected")
    ap.add_argument("--plm-marginals", default=None)
    ap.add_argument("--break-even", type=float, default=BREAK_EVEN)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    tips, skipped, src, submitted = tips_from_args(ap, a)
    arms = [x for x in a.arms.split(",") if x]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        ap.error(f"unknown kernel arms {bad}; have {sorted(ARMS)}")
    marg = None
    if a.plm_marginals:
        z = np.load(a.plm_marginals, allow_pickle=True)
        marg = dict(alphabet=str(z["alphabet"]), sites=[int(x) for x in z["sites"]],
                    logp={str(g): v for g, v in zip(z["genotypes"], z["logp"])})
    needs = sorted({x for x in arms if ARMS[x]["needs_plm"]})
    if needs and marg is None:
        ap.error(f"arms {', '.join(needs)} need --plm-marginals")
    panel = TipPanel(tips, a.min_allele_freq, a.lookback, a.max_support,
                     marginals=marg, submitted=submitted)

    drop = {int(x) for x in a.drop_targets.split(",") if x.strip()}
    counts = {s: int((panel.season == s).sum()) for s in panel.seasons}
    usable = [s for s in panel.seasons if counts.get(s, 0) >= a.min_season_seqs]
    order = [s for s in usable if (s + 1) in usable and (s + 1) not in drop]
    order = order[a.min_train:]
    if a.transitions:
        want = {int(x) for x in a.transitions.split(",")}
        trans = [s for s in order if s in want]
    elif a.full_table:
        trans = list(order)
    else:
        trans = [s for s in order if s <= a.gate_through]
    held = [s for s in order if s not in trans]

    print(f"source: {src}; {len(tips)} dated sequences ({skipped} dropped/undated), "
          f"{len(panel.genotypes)} genotypes, {len(panel.sites)} varying sites")
    if a.full_table:
        print("FULL TABLE: every transition measured. This is not a gate verdict -- the "
              "verdict\n  needs a held-out half -- it is a per-transition table for "
              "causal consumption by\n  `season_momentum --kernel-gate-table`, which "
              "reads each origin only the rows\n  whose targets had closed by its own "
              "horizon.")
    print(f"gating on " + ", ".join(f"{s}->{s+1}" for s in trans))
    print(f"held out, never measured here: "
          + (", ".join(f"{s}->{s+1}" for s in held) or "(none)"))
    print(f"arms: {', '.join(arms)}\n")

    tau_ks = [float(x) for x in a.tau_ks.split(",")]
    ms = [int(x) for x in a.ms.split(",")]
    if a.tau_es:
        tau_es = [None if x.strip().lower() in ("none", "") else float(x)
                  for x in a.tau_es.split(",")]
    else:
        tau_es = [a.tau_e]
    if a.tau_e_slows:
        tau_e_slows = [None if x.strip().lower() in ("none", "") else float(x)
                       for x in a.tau_e_slows.split(",")]
    else:
        tau_e_slows = [None]
    rows = gate_rows(panel, trans, arms, tau_ks, ms, tau_es, tau_e_slows)
    table, verdict = summarise(rows, a.break_even)
    if a.output:
        with open(a.output, "w") as f:
            json.dump(dict(source=src, panel=a.panel_provenance,
                           gated_on=trans, held_out=held, arms=arms,
                           causal_table=bool(a.full_table),
                           min_train=a.min_train, drop_targets=sorted(drop),
                           tau_ks=tau_ks, ms=ms, tau_e=a.tau_e, tau_es=tau_es,
                           tau_e_slows=tau_e_slows,
                           break_even=a.break_even, per_transition=rows,
                           operating_points=table, verdict=verdict), f, indent=2)
        print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
