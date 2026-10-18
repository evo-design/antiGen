#!/usr/bin/env python
"""Which datasets could join the D = 2025-01-01 forecast panel?

The wide-corpus rule (`wide_corpus.py`) asks only whether a build can be shown to carry
nothing observed after the cutoff, because there it is training context. A **forecast
panel** tree has to do more: it is trained on at D and *scored* on what happened after,
so it must clear five further bars, and each one eliminates real candidates.

  1. **Route A only.** Route B -- today's build with every tip pre-cutoff -- means the
     dataset is frozen and has no post-D history at all. Legitimate training context
     (s11's `joint_all` is exactly that), useless as an evaluation tree.
  2. **The reference frame must match** between the snapshot and today's build.
     `make_forecast_testset.py` refuses otherwise: train and test examples would not
     share coordinates, so a "novel" cell's position would mean different things on
     the two sides.
  3. **It must be a time tree.** `crossing_branches` reads `node_attrs.num_date` on
     both ends of a branch and nothing else. A divergence-only build yields zero
     crossing branches however well dated its tips are -- this is strictly stronger
     than wide_corpus's tip-date reader, which falls back to date/year/submitted.
  4. **A post-D budget worth scoring.** A tree contributing two targets adds noise and
     a per-tree row that cannot clear zero.
  5. **No tip overlap with a tree already in the panel.** s8 found h5n1 and h5nx
     sharing 37% of tip names: two cuts of one build would put the same evolution in
     the panel twice and break the paired bootstrap's independence across trees.

Read-only with respect to the panel: it writes example files to a scratch directory
and reports. Nothing here changes `data/forecast/`.

Usage:
    python experiments/multitree/probe_forecast_panel.py --probe data/wide/probe@2025-01-01.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import wide_corpus as wc                                                   # noqa: E402

D = "2025-01-01"
# Already in the eleven-tree panel, by the short name the forecast files use.
IN_PANEL = {"ncov": "ncov/open/global/all-time",
            "rsv_a": "rsv/a/F/all-time", "rsv_b": "rsv/b/F/all-time",
            "h5nx": "avian-flu/h5nx/ha/all-time", "measles": "measles/genome/global",
            "denv1": "dengue/denv1/E", "denv2": "dengue/denv2/E",
            "denv3": "dengue/denv3/E", "denv4": "dengue/denv4/E",
            "h3n2_ha": "seasonal-flu/h3n2/ha/12y",
            "h1n1pdm_ha": "seasonal-flu/h1n1pdm/ha/12y"}
PANEL_PATHS = set(IN_PANEL.values())
MIN_TRAIN_EXAMPLES = 40
MIN_TARGETS = 10
## A single shared tip name is coincidence; a shared strain SET is duplication.
OVERLAP_MAX = 0.05


def tip_names(tree):
    out, stack = set(), [tree]
    while stack:
        n = stack.pop()
        kids = n.get("children")
        if kids:
            stack.extend(kids)
        else:
            out.add(n.get("name"))
    return out


def genes_for(path, info):
    g = wc.GENES.get(path)
    if g:
        return g, None
    real = list(info.get("genes") or [])
    if len(real) == 1:
        return real, None
    return None, f"ambiguous gene choice among {real[:8]}"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", default=os.path.join(HERE, "data", "wide",
                                                   f"probe@{D}.json"))
    p.add_argument("--scratch", required=True, help="where example files are written")
    p.add_argument("--out", default=None, help="JSON report path")
    args = p.parse_args(argv)

    os.makedirs(args.scratch, exist_ok=True)
    with open(args.probe) as fh:
        rows = json.load(fh)

    # Tip sets of the trees already in the panel, for the overlap check. Read from the
    # SNAPSHOT, which is the build a candidate would be trained beside.
    panel_tips = {}
    for short, path in IN_PANEL.items():
        try:
            payload, _ = wc.fetch(path, D, offline=True)
        except Exception:                                                 # noqa: BLE001
            continue
        panel_tips[short] = tip_names(payload["tree"])
        del payload

    report = []
    for row in rows:
        path = row["dataset"]
        rec = dict(dataset=path, tier=row.get("tier"), in_panel=path in PANEL_PATHS)
        route = row.get("route")
        arch = row.get("archive") or {}
        if path in PANEL_PATHS:
            rec.update(verdict="already in panel")
            report.append(rec)
            continue
        if route != "A":
            rec.update(verdict="no archived build at D",
                       why=(wc._why_not(row) if route is None
                            else "route B: frozen, no post-D history"))
            report.append(rec)
            continue

        genes, why = genes_for(path, arch)
        if genes is None:
            rec.update(verdict="gene choice", why=why)
            report.append(rec)
            continue
        rec.update(genes=genes, updated=arch.get("updated"), n_tips=arch.get("n_tips"))

        short = wc.short_name(path)
        snap, cur = wc.cache_path(path, D), wc.cache_path(path, "current")
        if not os.path.exists(cur):
            rec.update(verdict="current build not cached")
            report.append(rec)
            continue

        base = ["--genes", *genes, "--ambiguous-ref", "gap", "--strip-trailing-stop"]
        ref = []
        if not arch.get("root_sequence_embedded"):
            try:
                ref = ["--ref", wc.root_sequence_path(path, D)]
            except Exception as exc:                                      # noqa: BLE001
                rec.update(verdict="no dated reference", why=str(exc)[:60])
                report.append(rec)
                continue

        train = os.path.join(args.scratch, f"examples_{short}@{D}_train.txt")
        rc, out, err = run([sys.executable, os.path.join(HERE, "make_tree_examples.py"),
                            "--tree", snap, "--out", train] + base + ref)
        if rc != 0:
            rec.update(verdict="train generation failed",
                       why=(err.strip().splitlines() or ["?"])[-1][:90])
            report.append(rec)
            continue
        n_ex = sum(1 for l in open(train) if l.startswith("EXAMPLE_ID:"))
        n_mut = sum(1 for l in open(train) if "," in l and l.split(",")[0].isdigit())
        rec.update(train_examples=n_ex, train_mutations=n_mut)

        test = os.path.join(args.scratch, f"examples_{short}@{D}_test.txt")
        # The probe only fetches the CURRENT build for candidates route A could not
        # serve, so a route-A row carries no `current` record and its
        # root_sequence_embedded flag cannot be read from the probe. Check the
        # payload itself, or every build whose reference lives in a sidecar is
        # wrongly reported as a test failure.
        curref = []
        # Parse, do not scan: hmpv/a/G embeds its root sequence at byte 3.2M, so a
        # truncated substring search reports it missing and the tree is lost.
        with open(cur) as fh:
            cur_embedded = "root_sequence" in json.load(fh)
        if not cur_embedded:
            try:
                curref = ["--current-ref", wc.root_sequence_path(path, "current")]
            except Exception as exc:                                      # noqa: BLE001
                rec.update(verdict="no current reference", why=str(exc)[:60])
                report.append(rec)
                continue
        rc, out, err = run([sys.executable,
                            os.path.join(HERE, "make_forecast_testset.py"),
                            "--current", cur, "--snapshot", snap,
                            "--date", D, "--out", test]
                           + base + ref[:0] + curref
                           + (["--snapshot-ref", ref[1]] if ref else []))
        if rc != 0:
            msg = (err.strip().splitlines() or ["?"])[-1][:120]
            rec.update(verdict=("reference frame changed"
                                if "reference frame changed" in msg else "test failed"),
                       why=msg)
            report.append(rec)
            continue
        for line in out.splitlines():
            if "targets=" in line:
                for tok in line.split():
                    if tok.startswith(("crossing", "examples=", "targets=", "novel=")):
                        k, _, v = tok.partition("=")
                        if v.isdigit():
                            rec[{"examples": "test_examples", "targets": "targets",
                                 "novel": "novel"}.get(k, k)] = int(v)

        # Overlap against every tree already in the panel.
        try:
            cand_tips = tip_names(wc.fetch(path, D, offline=True)[0]["tree"])
            # Fraction of the SMALLER tree, so a big tree sharing a few names with
            # a small one is not called an overlap. A single shared tip name is a
            # coincidence (hmpv/a and rsv_b share exactly one); what disqualifies a
            # candidate is a substantial shared strain set, as with h5n1 vs h5nx.
            ov = {}
            for sname, t in panel_tips.items():
                k = len(cand_tips & t)
                if k:
                    ov[sname] = [k, round(k / max(min(len(cand_tips), len(t)), 1), 4)]
            rec["overlap"] = ov
        except Exception:                                                 # noqa: BLE001
            rec["overlap"] = None

        if rec.get("targets", 0) == 0:
            rec["verdict"] = "no post-D crossing branches (divergence tree?)"
        elif any(f >= OVERLAP_MAX for _, f in (rec.get("overlap") or {}).values()):
            w = max((rec["overlap"] or {}).items(), key=lambda kv: kv[1][1])
            rec["verdict"] = f"overlaps {w[0]} ({100 * w[1][1]:.0f}% of the smaller tree)"
        elif n_ex < MIN_TRAIN_EXAMPLES:
            rec["verdict"] = f"{n_ex} train examples, below floor {MIN_TRAIN_EXAMPLES}"
        elif rec.get("targets", 0) < MIN_TARGETS:
            rec["verdict"] = f"{rec.get('targets')} post-D targets, too few"
        else:
            rec["verdict"] = "ELIGIBLE"
        report.append(rec)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)

    elig = [r for r in report if r["verdict"] == "ELIGIBLE"]
    print(f"\n{'='*92}\nELIGIBLE: {len(elig)} new trees\n{'='*92}")
    print(f"{'dataset':34s} {'gene':14s} {'tr_ex':>6s} {'tr_mut':>7s} "
          f"{'targets':>8s} {'novel':>6s}")
    for r in sorted(elig, key=lambda r: -r.get("targets", 0)):
        print(f"{r['dataset']:34s} {'+'.join(r['genes'])[:14]:14s} "
              f"{r.get('train_examples', 0):6d} {r.get('train_mutations', 0):7d} "
              f"{r.get('targets', 0):8d} {r.get('novel', 0):6d}")
    print(f"\n{'='*92}\nEXCLUDED\n{'='*92}")
    for r in sorted(report, key=lambda r: r["verdict"]):
        if r["verdict"] in ("ELIGIBLE", "already in panel"):
            continue
        extra = f" -- {r['why']}" if r.get("why") else ""
        print(f"{r['dataset']:34s} {r['verdict']}{extra[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
