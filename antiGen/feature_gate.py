"""The redundancy gate for the antiGen design columns, run before anything is scored.

`distance_from_consensus` was added to this model as a column on 2026-09-23 and
discarded. The reason was not that it forecast badly -- it was that `dfc_v` is a linear
function of the one-hot allele design, `R^2` 0.968-0.997 at the eight backbone horizons,
so `mlr` could already express it with `beta = -1` on every non-consensus allele. The
arm added a *shrinkage target*, one direction of beta exempted from `lam` and handed the
weak `gamma_pen`, rather than information. That is a real degree of freedom and it is
worth a test, but it is not what "a new feature" means, and it should be established
before a scoring array is spent rather than after.

So this module asks the same question of the antiGen columns, on the training
transitions only, and fits no forecast:

**redundancy** -- `R^2` of the column regressed on the family's own one-hot allele
    block. A column the indicators can already express comes back near 1. The bar is
    `REDUNDANCY_BAR`, set at the bottom of the range `dfc` measured so this round cannot
    pass by being marginally less redundant than the arm that already failed.

**signal** -- weighted Spearman between the column and `log(assigned / base)`, the
    realised next-season log growth on the support, where `assigned` is the
    nearest-neighbour-assigned next-season mass that the multinomial fit actually
    targets. This reads the transition's own future, so it is an oracle diagnostic and
    is restricted to the training transitions by the same discipline `kernel_gate`'s
    descent measurement obeys. A column can be non-redundant and still carry nothing,
    and those two failures want different responses -- the first says the design already
    had it, the second says nobody has it.

Neither number is a forecast and neither is fitted, so a column this module rules out
costs the multiple-looks budget nothing.

    python -m antiGen.feature_gate --tree ~/h3n2/tree.json --ref ~/h3n2/ref.json \\
        --antigen-checkpoints /path/to/checkpoints --gate-through 2022
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from antiGen.antigen_feature import FEATURES, anchors, redundancy
from antiGen.season_momentum import MONTH, Task, TipPanel, load_alignment_dated, load_tips


def _spearman(x, y, w):
    """Weighted Spearman: Pearson on the weighted ranks.

    Ties are averaged, which matters here because `arr` is exactly zero for about a
    third of the support and an arbitrary tie-break would manufacture a correlation out
    of row order."""
    def rank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=np.float64)
        r[o] = np.arange(len(v), dtype=np.float64)
        # average the ranks within each tied block
        vs = v[o]
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[j + 1] == vs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = np.mean(np.arange(i, j + 1))
            i = j + 1
        return r

    a, b = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    w = np.asarray(w, float)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(a)) / len(a)
    am, bm = float(w @ a), float(w @ b)
    va = float(w @ (a - am) ** 2)
    vb = float(w @ (b - bm) ** 2)
    if va <= 0 or vb <= 0:
        return float("nan")
    return float(w @ ((a - am) * (b - bm))) / np.sqrt(va * vb)


def gate_rows(panel, transitions, features=FEATURES, verbose=True):
    """One row per `(transition, feature)`, plus `dfc` and the two calibration anchors.

    `dfc` is here as the arm that already failed this test, and `null`/`spanned` as the
    references its number has to be read against -- the design is rank deficient, so
    neither 0 nor 1 is the right scale and both depend on the horizon."""
    rows = []
    for s in transitions:
        t = Task(panel, s)
        w = t.baseline("season")
        w = w if w.sum() > 0 else t.level(MONTH)
        base = np.asarray(w, float)
        # The realised next-season log growth on the support, against the base measure
        # the tilt multiplies. `assigned` is the nearest-neighbour-assigned next-season
        # mass -- the quantity the multinomial fit actually targets.
        growth = np.log(np.maximum(t.assigned, 1e-12)) - np.log(np.maximum(base, 1e-12))
        null, spanned = anchors(t.S, w=base)
        mid = 0.5 * (null + spanned)
        if verbose:
            print(f"\n  {s}->{s + 1}: support {len(t.sup)}  targets {len(t.tgt_g)}  "
                  f"| anchors: noise {null:+.3f}  spanned {spanned:+.3f}  "
                  f"midpoint {mid:+.3f}")
        for f in tuple(features) + ("dfc",):
            col = t.dfc() if f == "dfc" else t.agen(f)
            cv, insample, _ = redundancy(t.S, col, w=base)
            rho = _spearman(col, growth, base)
            rows.append(dict(season=int(s), next=int(s + 1), feature=f,
                             cv_r2=float(cv), in_r2=float(insample),
                             null=null, spanned=spanned, midpoint=mid,
                             redundant=bool(cv > mid), spearman=float(rho),
                             n_support=int(len(t.sup))))
            if verbose:
                print(f"    {f:>6}  CV R^2 {cv:+7.4f} (in-sample {insample:.4f})  "
                      f"{'REDUNDANT' if cv > mid else 'novel':>9}   "
                      f"Spearman vs growth {rho:+.4f}")
    return rows


def summarise(rows, verbose=True):
    feats = sorted({r["feature"] for r in rows}, key=lambda f: f == "dfc")
    table, verdict = [], {}
    for f in feats:
        sel = [r for r in rows if r["feature"] == f]
        cv = np.array([r["cv_r2"] for r in sel])
        rho = np.array([r["spearman"] for r in sel])
        red = int(sum(r["redundant"] for r in sel))
        table.append(dict(feature=f, cv_r2=float(cv.mean()),
                          in_r2=float(np.mean([r["in_r2"] for r in sel])),
                          redundant_in=red, n=len(sel),
                          spearman=float(np.nanmean(rho)),
                          pos=int(np.sum(rho > 0))))
        verdict[f] = dict(feature=f, cv_r2=float(cv.mean()),
                          passes=bool(red <= len(sel) // 2),
                          spearman=float(np.nanmean(rho)))
    if verbose:
        nl = float(np.mean([r["null"] for r in rows]))
        sp = float(np.mean([r["spanned"] for r in rows]))
        print(f"\ncalibration anchors, averaged over the gated transitions: "
              f"noise {nl:+.4f}, spanned {sp:+.4f}, midpoint {0.5 * (nl + sp):+.4f}")
        print(f"\n{'feature':>8} {'CV R^2':>8} {'in-samp':>8} {'redundant':>10} "
              f"{'Spearman':>9} {'rho>0':>7}")
        for t in table:
            print(f"{t['feature']:>8} {t['cv_r2']:+8.4f} {t['in_r2']:8.4f} "
                  f"{t['redundant_in']:5d}/{t['n']:<4d} {t['spearman']:+9.4f} "
                  f"{t['pos']:4d}/{t['n']:<3d}")
        print("\nA column is redundant at a horizon when its CV R^2 sits above the "
              "midpoint of the two\nanchors on that horizon's own design. There is no "
              "absolute bar: in-sample R^2 is mostly\noverfitting here (noise scores "
              "0.67-0.88 against a rank-124-167 design on 271-442\ngenotypes), and even "
              "cross-validated, an exactly-spanned column only reaches ~0.80.\n`dfc` is "
              "in the table as the arm already discarded for redundancy, not as a "
              "candidate.")
        for f in feats:
            if f == "dfc":
                continue
            v = verdict[f]
            print(f"  {f:>6}  CV R^2 {v['cv_r2']:+.4f}  -->  "
                  f"{'CLEARS -- carries information the design does not'
                     if v['passes'] else 'REDUNDANT, not scored'}"
                  f"   (Spearman {v['spearman']:+.4f})")
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
    ap.add_argument("--drop-targets", default="2020")
    ap.add_argument("--gate-through", type=int, default=2022,
                    help="the last transition the gate may look at, named by the season "
                         "predicted *from*, so 2022 is the transition predicting 2023. "
                         "The Spearman column reads the transition's own future, so "
                         "restricting this is what keeps the gate honest -- the same "
                         "discipline kernel_gate's descent measurement obeys")
    ap.add_argument("--transitions", default="")
    ap.add_argument("--features", default=",".join(FEATURES))
    ap.add_argument("--antigen-checkpoints", required=True)
    ap.add_argument("--antigen-device", default=None)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    if a.alignment:
        if not a.metadata:
            ap.error("--alignment needs --metadata")
        tips, _ = load_alignment_dated(a.alignment, a.metadata)
        src = f"alignment, {len(a.alignment)} segments"
    else:
        if not (a.tree and a.ref):
            ap.error("give either --tree/--ref or --alignment/--metadata")
        tips, _ = load_tips(a.tree, a.ref)
        src = "guide-tree tips"
    feats = tuple(x for x in a.features.split(",") if x)
    bad = [x for x in feats if x not in FEATURES]
    if bad:
        ap.error(f"unknown features {bad}; have {list(FEATURES)}")

    panel = TipPanel(tips, a.min_allele_freq, a.lookback, a.max_support)
    panel.agen_root = a.antigen_checkpoints
    panel.agen_device = a.antigen_device
    print(f"source: {src}; {len(panel.tips)} dated sequences, "
          f"{len(panel.genotypes)} genotypes, {len(panel.sites)} varying sites")

    drop = {int(x) for x in a.drop_targets.split(",") if x.strip()}
    counts = {s: int((panel.season == s).sum()) for s in panel.seasons}
    usable = [s for s in panel.seasons if counts.get(s, 0) >= a.min_season_seqs]
    order = [s for s in usable if (s + 1) in usable and (s + 1) not in drop]
    if a.transitions:
        sel = [int(x) for x in a.transitions.split(",")]
    else:
        sel = [s for s in order if s <= a.gate_through]
    held = [s for s in order if s not in sel]
    print("gating on " + ", ".join(f"{s}->{s + 1}" for s in sel))
    print("held out, never measured here: "
          + ", ".join(f"{s}->{s + 1}" for s in held))

    rows = gate_rows(panel, sel, feats)
    table, verdict = summarise(rows)
    if a.output:
        with open(a.output, "w") as f:
            json.dump(dict(source=src, gated_on=sel, held_out=held,
                           features=list(feats),
                           per_transition=rows, table=table, verdict=verdict),
                      f, indent=2)
        print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
