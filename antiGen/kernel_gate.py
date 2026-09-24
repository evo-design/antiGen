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

from antiGen.mutation_kernel import ARMS, BREAK_EVEN, gate
from antiGen.season_momentum import Task, TipPanel, load_alignment_dated, load_tips


def gate_rows(panel, transitions, arms, tau_ks, ms, tau_e=None, verbose=True):
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
            for tau_k in tau_ks:
                for m in ms:
                    d = task.descent_gate(x, arm, tau_k, m, tau_e)
                    rows.append(dict(season=int(s), next=int(s + 1), arm=arm,
                                     tau_k=float(tau_k), m=int(m), emd_season=w,
                                     n_support=len(task.sup),
                                     n_target=len(task.tgt_g), **d))
                    if verbose:
                        print(f"      {arm:>10} tau_k {tau_k:5g} m {m:>4}  "
                              f"cands/parent {d['n_cand'] / max(len(task.sup), 1):7.1f}  "
                              f"p_down {d['p_down']:.4f}  p_flat {d['p_flat']:.4f}  "
                              f"break-even {d['break_even']:.4f}  "
                              f"dW/dmu {d['deriv']:+.4f}  greedy {d['greedy']:+.4f}")
    return rows


def summarise(rows, break_even=BREAK_EVEN, verbose=True):
    """Mean `p_down` per `(arm, tau_k, m)` over the transitions, and the gate's verdict."""
    keys = sorted({(r["arm"], r["tau_k"], r["m"]) for r in rows})
    stats, table = {}, []
    for k in keys:
        sel = [r for r in rows if (r["arm"], r["tau_k"], r["m"]) == k]
        stats[k] = float(np.mean([r["p_down"] for r in sel]))
        table.append(dict(arm=k[0], tau_k=k[1], m=k[2], p_down=stats[k],
                          p_flat=float(np.mean([r["p_flat"] for r in sel])),
                          deriv=float(np.mean([r["deriv"] for r in sel])),
                          greedy=float(np.mean([r["greedy"] for r in sel])),
                          cands=float(np.mean([r["n_cand"] for r in sel])),
                          n=len(sel)))
    verdict = gate(stats, break_even)
    if verbose:
        print(f"\n{'arm':>10} {'tau_k':>7} {'m':>5} {'p_down':>8} {'p_flat':>8} "
              f"{'dW/dmu':>9} {'greedy':>8} {'cands':>9}")
        for t in sorted(table, key=lambda t: (t["arm"], -t["p_down"])):
            print(f"{t['arm']:>10} {t['tau_k']:7g} {t['m']:5d} {t['p_down']:8.4f} "
                  f"{t['p_flat']:8.4f} {t['deriv']:+9.4f} {t['greedy']:+8.4f} "
                  f"{t['cands']:9.0f}")
        print(f"\nthe bar is p_down > {break_even:.4f}; a convex score with a positive "
              f"right-derivative at mu = 0 attains its minimum there, so an arm below it "
              f"\ncannot help at any mu and is not scored on a held-out transition:")
        for arm in sorted(verdict):
            v = verdict[arm]
            print(f"  {arm:>10}  best p_down {v['p_down']:.4f} at tau_k "
                  f"{v['tau_k']:g}, m {v['m']}  -->  "
                  f"{'CLEARS the gate' if v['passes'] else 'GATED OUT'}"
                  + ("   (leaky: reads a model pretrained on the test seasons)"
                     if v["leaky"] else ""))
    return table, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--alignment", nargs="+", default=None)
    ap.add_argument("--metadata", default=None)
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
    ap.add_argument("--tau-e", type=float, default=None,
                    help="discount timescale for kemp's substitution counts, in years; "
                         "the panel's own lookback by default. Declared rather than "
                         "selected")
    ap.add_argument("--plm-marginals", default=None)
    ap.add_argument("--break-even", type=float, default=BREAK_EVEN)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    if a.alignment:
        if not a.metadata:
            ap.error("--alignment needs --metadata")
        tips, skipped = load_alignment_dated(a.alignment, a.metadata)
        src = f"alignment, {len(a.alignment)} segments"
    else:
        if not (a.tree and a.ref):
            ap.error("give either --tree/--ref or --alignment/--metadata")
        tips, skipped = load_tips(a.tree, a.ref)
        src = "guide-tree tips"
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
                     marginals=marg)

    drop = {int(x) for x in a.drop_targets.split(",") if x.strip()}
    counts = {s: int((panel.season == s).sum()) for s in panel.seasons}
    usable = [s for s in panel.seasons if counts.get(s, 0) >= a.min_season_seqs]
    order = [s for s in usable if (s + 1) in usable and (s + 1) not in drop]
    order = order[a.min_train:]
    if a.transitions:
        want = {int(x) for x in a.transitions.split(",")}
        trans = [s for s in order if s in want]
    else:
        trans = [s for s in order if s <= a.gate_through]
    held = [s for s in order if s not in trans]

    print(f"source: {src}; {len(tips)} dated sequences ({skipped} dropped/undated), "
          f"{len(panel.genotypes)} genotypes, {len(panel.sites)} varying sites")
    print(f"gating on " + ", ".join(f"{s}->{s+1}" for s in trans))
    print(f"held out, never measured here: "
          + (", ".join(f"{s}->{s+1}" for s in held) or "(none)"))
    print(f"arms: {', '.join(arms)}\n")

    tau_ks = [float(x) for x in a.tau_ks.split(",")]
    ms = [int(x) for x in a.ms.split(",")]
    rows = gate_rows(panel, trans, arms, tau_ks, ms, a.tau_e)
    table, verdict = summarise(rows, a.break_even)
    if a.output:
        with open(a.output, "w") as f:
            json.dump(dict(source=src, gated_on=trans, held_out=held, arms=arms,
                           tau_ks=tau_ks, ms=ms, tau_e=a.tau_e,
                           break_even=a.break_even, per_transition=rows,
                           operating_points=table, verdict=verdict), f, indent=2)
        print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
