#!/usr/bin/env python
"""Convolve `mlr+mom`'s forecast with antiGen's forward pass, and price the mixing weight.

The forecasting side of this repository predicts a distribution over the genotypes
already sequenced; the generative side predicts, for a given genotype, which substitution
its children will carry. They are distributions over different things, and the operation
that combines them is a convolution:

    x'  =  (1 - mu) x  +  mu K x,      (K x)_w = sum_v x_v pi(q, b | v),  w = v + (q,b)

`x` is `mlr+mom`'s forecast for next season and `pi(. | v)` is antiGen's forward pass on
`v`, the checkpoint being the one whose train/test split is this origin's own horizon.
`mu` is the only new number, and this module prices it three ways at every origin:

  **mu = 0**       the parent exactly. The family nests it, so this column must
                   reproduce the standing `mlr+mom` result to the last digit, and it is
                   checked rather than assumed.
  **mu-hat**       chosen on the rolling selector's own validation folds -- the
                   `val_folds` transitions immediately before the one being predicted,
                   never the one being predicted. This is the forecast.
  **mu-star**      the minimiser on the test transition itself, by golden section.
                   **Oracle**: it reads the season being predicted, and exists only to
                   separate "the kernel points the wrong way" from "the kernel points the
                   right way and `mu` was chosen badly", which want different responses.

The parent is held at exactly what the standing protocol selects for `mlr+mom` -- same
grid, same `top_k` averaging, same training window -- so the only thing the selector
chooses here that it did not choose before is `mu`, over a handful of candidates rather
than over a grid multiplied by a new field. `season_momentum --families mlr+mom+K` is the
other reading, in which `mu` enters the grid and is chosen jointly with the parent's
hyperparameters; the two are worth reporting side by side and this module is not a
substitute for it.

Usage:
    python -m antiGen.antigen_convolve --tree ~/h3n2/tree.json --ref ~/h3n2/ref.json \
        --antigen-checkpoints .../checkpoints --transitions 2022 -o out.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from antiGen.antigen_kernel import build as agen_candidates
from antiGen.mutation_kernel import convolve, golden_mu
from antiGen.season_momentum import (FAMILIES, MONTH, Task, TipPanel, average,
                                     load_tips, make_grid, run_family, select,
                                     train_set)

# The mixing weights the selector chooses between. `0` must be here: it is what makes the
# family nest its parent, and a selector that returns it at every origin is a clean
# negative result rather than a failed run. The upper end is 1 because the bound says the
# whole downside is `mu` -- there is no reason to cap the search below the point where the
# convolution replaces the forecast outright.
MUS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def parent_forecast(tasks, order, i, grid, family, val_folds, maxiter, gamma_pen, top_k):
    """The standing protocol's forecast for `order[i]`, and the combos behind it.

    Byte-identical to what `rolling` computes for the same family: the same `select`,
    the same `train_set`, the same `average` over the best `top_k` candidates. Returned
    rather than scored, because what happens to it afterwards is the point."""
    combos, _ = select(tasks, order, i, grid, family, val_folds, maxiter, gamma_pen,
                       top_k)
    preds = [run_family(tasks[order[i]], c, family,
                        train_set(tasks, order, i, c["window"]), maxiter, gamma_pen)
             for c in combos]
    return average(preds), combos


def sweep(task, w, cand, mus):
    """`mu -> EMD` for one transition, at a fixed parent forecast."""
    return {float(m): float(task.score(convolve(w, cand, m))) for m in mus}


def run(panel, root, order, tasks, i, grid, family, val_folds, maxiter, gamma_pen,
        top_k, mus, device=None, sites_only=False, drop_gap_stop=False, verbose=True):
    """One origin: the parent, the validation sweep, the chosen `mu`, and the oracle."""
    s = order[i]
    task = tasks[s]
    w, combos = parent_forecast(tasks, order, i, grid, family, val_folds, maxiter,
                                gamma_pen, top_k)
    kw = dict(device=device, sites_only=sites_only, drop_gap_stop=drop_gap_stop,
              verbose=verbose)
    cand = agen_candidates(panel, task.sup, task.M, task.tgt_g, task.horizon, root, **kw)

    # The validation folds are the selector's own: the `val_folds` transitions before
    # this one, each forecast by the *same* combos and convolved by *its own* origin's
    # checkpoint. Nothing here reads season s + 1.
    folds = [j for j in range(max(1, i - val_folds), i) if train_set(tasks, order, j, 0)]
    per_fold = {}
    for j in folds:
        t = tasks[order[j]]
        wj = average([run_family(t, c, family,
                                 train_set(tasks, order, j, c["window"]), maxiter,
                                 gamma_pen) for c in combos])
        cj = agen_candidates(panel, t.sup, t.M, t.tgt_g, t.horizon, root, **kw)
        per_fold[int(order[j])] = sweep(t, wj, cj, mus)
    val = {m: float(np.mean([per_fold[j][m] for j in per_fold])) for m in mus} \
        if per_fold else {}
    mu_hat = min(val, key=lambda m: val[m]) if val else 0.0

    test = sweep(task, w, cand, mus)
    mu_star, emd_star = golden_mu(lambda m: task.score(convolve(w, cand, m)))
    row = dict(
        season=int(s), next=int(s + 1), n_support=len(task.sup),
        n_target=len(task.tgt_g), n_next_seqs=task.n_next, floor=task.floor,
        floor_ext=task.floor_ext("all"),
        n_cand_rows=int(cand.n_rows), n_proposals=int(cand.n_proposals),
        inert=float(cand.inert),
        combos=[dict(tau_f_months=c["tau_f"] / MONTH, tau_s_years=c["tau_s"],
                     lam=c["lam"], window=c["window"], alpha=c.get("alpha", 0.0),
                     gamma_pen=c.get("gamma_pen")) for c in combos],
        val_folds=sorted(int(j) for j in per_fold), val_curve=val,
        val_per_fold=per_fold, test_curve=test,
        mu_hat=float(mu_hat), emd_mu0=test[0.0], emd_mu_hat=test[float(mu_hat)],
        mu_star=float(mu_star), emd_mu_star=float(emd_star),
        off_support_mu_hat=float(convolve(w, cand, mu_hat).off_support),
    )
    if verbose:
        print(f"  {s}->{s+1}  floor {task.floor:6.3f} (ext {row['floor_ext']:6.3f})  "
              f"rows {cand.n_rows:6d}  inert {cand.inert:.3f} | "
              f"mu=0 {test[0.0]:7.4f}  mu_hat {mu_hat:5.3g} -> "
              f"{row['emd_mu_hat']:7.4f} ({100 * (test[0.0] - row['emd_mu_hat']) / test[0.0]:+6.2f}%)"
              f"  | oracle mu* {mu_star:5.3f} -> {emd_star:7.4f} "
              f"({100 * (test[0.0] - emd_star) / test[0.0]:+6.2f}%)")
    return row


def report(rows, mask=()):
    """The panel means, with the masked transitions excluded from the averages only."""
    keep = [r for r in rows if r["next"] not in mask]
    if not keep:
        return {}
    out = dict(n=len(keep), masked=sorted(mask),
               mu0=float(np.mean([r["emd_mu0"] for r in keep])),
               mu_hat=float(np.mean([r["emd_mu_hat"] for r in keep])),
               mu_star=float(np.mean([r["emd_mu_star"] for r in keep])),
               floor=float(np.mean([r["floor"] for r in keep])),
               floor_ext=float(np.mean([r["floor_ext"] for r in keep])),
               wins_hat=int(sum(r["emd_mu_hat"] < r["emd_mu0"] - 1e-12 for r in keep)),
               wins_star=int(sum(r["emd_mu_star"] < r["emd_mu0"] - 1e-12 for r in keep)),
               mu_hat_zero=int(sum(r["mu_hat"] == 0.0 for r in keep)),
               mu_star_zero=int(sum(r["mu_star"] < 1e-3 for r in keep)))
    # The oracle's mean over a common mu, which is the fair "one number for the panel"
    # version of the oracle: a single mu chosen with hindsight, not one per transition.
    common = {m: float(np.mean([r["test_curve"][str(m)] if str(m) in r["test_curve"]
                                else r["test_curve"][m] for r in keep]))
              for m in keep[0]["test_curve"]}
    out["common_curve"] = common
    out["mu_common"] = float(min(common, key=lambda m: common[m]))
    out["mu_common_emd"] = float(min(common.values()))
    print(f"\n{len(keep)} transitions scored"
          + (f" (targets {sorted(mask)} masked from the averages)" if mask else ""))
    print(f"  floor {out['floor']:.3f}   one-step floor {out['floor_ext']:.3f}")
    print(f"  mu = 0      {out['mu0']:.4f}   (the standing mlr+mom forecast)")
    print(f"  mu-hat      {out['mu_hat']:.4f}   "
          f"{100 * (out['mu0'] - out['mu_hat']) / out['mu0']:+.2f}%   "
          f"{out['wins_hat']}/{len(keep)} transitions improved, "
          f"{out['mu_hat_zero']} selected mu = 0")
    print(f"  mu-star     {out['mu_star']:.4f}   "
          f"{100 * (out['mu0'] - out['mu_star']) / out['mu0']:+.2f}%   ORACLE, per "
          f"transition, {out['wins_star']}/{len(keep)} improved")
    print(f"  mu common   {out['mu_common_emd']:.4f}   "
          f"{100 * (out['mu0'] - out['mu_common_emd']) / out['mu0']:+.2f}%   ORACLE, "
          f"one mu for the panel, at mu = {out['mu_common']:g}")
    print("  the whole curve, averaged over the scored transitions:")
    for m in sorted(common, key=float):
        print(f"    mu {float(m):5.3g}  {common[m]:.4f}"
              f"  {100 * (out['mu0'] - common[m]) / out['mu0']:+7.2f}%")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--antigen-checkpoints", required=True,
                    help="directory of per-origin checkpoints, season_<S>/fitness_*.pt")
    ap.add_argument("--antigen-device", default=None)
    ap.add_argument("--antigen-sites-only", action="store_true",
                    help="renormalise the proposal onto the panel's varying sites")
    ap.add_argument("--antigen-drop-gap-stop", action="store_true")
    ap.add_argument("--family", default="mlr+mom", choices=sorted(FAMILIES),
                    help="the parent whose forecast is convolved")
    ap.add_argument("--min-allele-freq", type=float, default=0.005)
    ap.add_argument("--lookback", type=float, default=5.0)
    ap.add_argument("--min-season-seqs", type=int, default=30)
    ap.add_argument("--min-train", type=int, default=2)
    ap.add_argument("--val-folds", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--gamma-pen", type=float, default=0.01)
    ap.add_argument("--tau-f-months", default="0.5,1,2,4")
    ap.add_argument("--tau-s-years", default="1,3,inf")
    ap.add_argument("--lambdas", default="0.3,1,3,10,30")
    ap.add_argument("--windows", default="2,3,0")
    ap.add_argument("--alphas", default="0,0.05")
    ap.add_argument("--gamma-pens", default="0.01,0.3,3")
    ap.add_argument("--drop-targets", default="",
                    help="targets removed from evaluation, validation and training "
                         "alike. The standing panel removes none and masks the eval on "
                         "2021 afterwards, which is --mask")
    ap.add_argument("--mask", default="2021",
                    help="targets excluded from the reported averages only, having been "
                         "fitted and forecast like any other")
    ap.add_argument("--mus", default=",".join(f"{m:g}" for m in MUS))
    ap.add_argument("--transitions", default="",
                    help="evaluate only these transitions, by the season predicted from")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    tips, skipped = load_tips(a.tree, a.ref)
    panel = TipPanel(tips, a.min_allele_freq, a.lookback)
    print(f"{len(tips)} dated sequences ({skipped} undated), "
          f"{len(panel.genotypes)} distinct genotypes, {len(panel.features)} features "
          f"over {panel.n_sites} varying sites ({len(panel.sites)} sites carry any "
          f"variation at all)")

    grid = make_grid([float(x) for x in a.tau_f_months.split(",")],
                     [float(x) for x in a.tau_s_years.split(",")],
                     [float(x) for x in a.lambdas.split(",")],
                     [int(x) for x in a.windows.split(",")],
                     [float(x) for x in a.alphas.split(",")],
                     gamma_pens=tuple(float(x) for x in a.gamma_pens.split(",")))
    drop = {int(x) for x in a.drop_targets.split(",") if x.strip()}
    mask = {int(x) for x in a.mask.split(",") if x.strip()}
    mus = tuple(float(x) for x in a.mus.split(","))

    counts = {s: int((panel.season == s).sum()) for s in panel.seasons}
    usable = [s for s in panel.seasons if counts.get(s, 0) >= a.min_season_seqs]
    order = [s for s in usable if (s + 1) in usable and (s + 1) not in drop]
    tasks = {s: Task(panel, s) for s in order}
    only = {int(x) for x in a.transitions.split(",")} if a.transitions else None
    print(f"transitions in play: " + ", ".join(f"{s}->{s+1}" for s in order))
    print(f"{len(grid)} hyperparameter combinations for the parent; "
          f"{len(mus)} mixing weights\n")

    rows = []
    for i in range(a.min_train, len(order)):
        if only is not None and order[i] not in only:
            continue
        rows.append(run(panel, a.antigen_checkpoints, order, tasks, i, grid, a.family,
                        a.val_folds, a.maxiter, a.gamma_pen, a.top_k, mus,
                        device=a.antigen_device, sites_only=a.antigen_sites_only,
                        drop_gap_stop=a.antigen_drop_gap_stop))
    summary = report(rows, mask) if only is None else {}
    if a.output:
        with open(a.output, "w") as f:
            json.dump(dict(family=a.family, mus=list(mus), drop_targets=sorted(drop),
                           mask=sorted(mask), n_grid=len(grid), top_k=a.top_k,
                           val_folds=a.val_folds,
                           sites_only=bool(a.antigen_sites_only),
                           drop_gap_stop=bool(a.antigen_drop_gap_stop),
                           per_transition=rows, summary=summary), f, indent=2)
        print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
