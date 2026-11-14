#!/usr/bin/env python
"""Score the core-surface joint arms on the 13-panel NO-ATTACH eval with the exact
fig13 metric (per-event fractional hit@10, gap/stop masked, all + de novo), so the arms
drop straight into the fig5_multitree_noattach ladder as new rungs.

Each arm is a single run trained with --save-every-epoch and NO val split, so there is no
_best.pt; we report epoch 3, matching the reporting epoch of the existing `joint` rung.

Writes, per invocation:
  <out-npz>                      per-event weights keyed  <tree>__<sub>__<arm>
  <figdir>/ladder_core_arms.csv  fig13-schema ladder rows (macro recall, tree-bootstrap CI)
  stdout                         a macro recall@10 table (all + novel) for each arm

Usage:
  PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
  $PY score_core_arms_fig13.py --epoch 3 \
      --arm joint_surface_snapshot_partial4=<run_dir> \
      --arm joint_surface_all_partial4=<run_dir> ... \
      --out-npz <dir>/perevent_core_arms.npz
"""
import os, sys, glob, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import numpy as np
import torch
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, load_esm, ALPHA_SET, D, dev

NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'denv1', 'denv2', 'denv3',
         'denv4', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
FIGDIR = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach"
# fig13 ladder schema constants (match collect_fig13_noattach.py)
UNIV, MET, RULE, EVAL, B = "standard", "hit10", "pooled", "Temporal holdout", 2000


def macro_ci(per, rng):
    per = np.asarray([p for p in per if p == p])
    if len(per) == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, len(per), size=(B, len(per)))
    bs = per[idx].mean(1)
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def find_ckpt(run_dir, epoch):
    hits = glob.glob(os.path.join(run_dir, f"*_epoch{epoch}.pt"))
    return hits[0] if hits else None


def score_arm(model, trees):
    """Return {(tree,sub): np.array(per-event weights)} for this model."""
    out = {}
    for tree in trees:
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te:
                continue
            weights = []
            for ex in te:
                ev = events_of(ex)
                if not ev:
                    continue
                ab = ex.aa_idx.unsqueeze(0).to(dev)
                with torch.no_grad():
                    scores = model(ab).float().squeeze(0).cpu()
                weights += per_event_weights(scores, ex.aa_idx, ev)
            out[(tree, sub)] = np.array(weights, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=RUN_DIR",
                    help="repeatable; the arm's rung name and its output dir")
    ap.add_argument("--epoch", type=int, default=3)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--ladder", default=os.path.join(FIGDIR, "ladder_core_arms.csv"))
    ap.add_argument("--trees", nargs="+", default=TREES)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, run_dir = spec.split("=", 1)
        arms[name] = run_dir

    rng = np.random.default_rng(0)
    npz = {}                       # <tree>__<sub>__<arm> -> weights
    ladder_rows = []               # (subset, rung, mean, lo, hi)
    summary = []                   # (arm, all_macro, novel_macro)

    for name, run_dir in arms.items():
        ck = find_ckpt(run_dir, args.epoch)
        if not ck:
            print(f"!! {name}: no epoch{args.epoch} checkpoint in {run_dir}", flush=True)
            continue
        model = load_esm(ck)
        per_event = score_arm(model, args.trees)
        del model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        macro = {}
        for sub in ('all', 'novel'):
            per_tree = [per_event[(t, sub)].mean() for t in args.trees
                        if (t, sub) in per_event and len(per_event[(t, sub)]) > 0]
            m, lo, hi = macro_ci(per_tree, rng)
            macro[sub] = m
            ladder_rows.append((sub, name, m, lo, hi))
            for t in args.trees:
                if (t, sub) in per_event:
                    npz[f"{t}__{sub}__{name}"] = per_event[(t, sub)]
        summary.append((name, macro['all'], macro['novel']))
        print(f"{name:42s} all_macro={macro['all']:.3f}  novel_macro={macro['novel']:.3f}  "
              f"(ckpt epoch{args.epoch})", flush=True)

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, **npz)

    os.makedirs(os.path.dirname(args.ladder), exist_ok=True)
    with open(args.ladder, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["universe", "metric", "rule", "eval", "subset", "rung", "value", "lo", "hi"])
        for sub, rung, m, lo, hi in ladder_rows:
            w.writerow([UNIV, MET, RULE, EVAL, sub, rung, m, lo, hi])

    print(f"\nwrote {len(npz)} per-event arrays -> {args.out_npz}")
    print(f"wrote {len(ladder_rows)} ladder rows -> {args.ladder}")
    print("\n=== macro recall@10 (fig13 metric) ===")
    print(f"{'arm':42s} {'all':>7} {'novel':>7}")
    for name, a, n in summary:
        print(f"{name:42s} {a:>7.3f} {n:>7.3f}")


if __name__ == "__main__":
    main()
