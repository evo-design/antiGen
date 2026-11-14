#!/usr/bin/env python
"""De-novo + all-substitution recall for the RETRAINED 3-seed ensemble arms (EXP1/EXP2).

Mirrors score_denovo_panels.py exactly -- same shared metric (score_perevent.
per_event_weights), same de-novo pool strike (training-seen cells pssm_root>0, built from
the FIXED per-tree train FILE, not any model's 90/10 split), same budgets {.10,.05,.01},
same event-level bootstrap CI (nboot=2000, seed 0), same _test.txt/_novel.txt targets and
the same 13 trees. The ONLY difference is the arm: each arm here is the LOGIT-LEVEL
ensemble of 3 seeds (raw logit maps averaged before ranking).

Arms:
  xformer3      from-scratch transformer, per-virus 3-seed ensemble            (EXP1)
  single650p4   ESM-2 650M partial-4, per-virus 3-seed ensemble                (EXP2 single)
  joint13p4     ESM-2 650M partial-4, 13-tree joint, val-selected, 3-seed ens  (EXP2 13)
  joint49p4     ESM-2 650M partial-4, 49-tree joint, val-selected, 3-seed ens  (EXP2 49)

Kept in a SEPARATE CSV from denovo_panels_recall.csv (different models/experiment) with
the IDENTICAL schema, so the two union by schema. Writes retrain_ensembles.csv +
retrain_ensembles.npz + retrain_ensembles_provenance.json.
"""
import os, sys, glob, json, csv, socket, subprocess, datetime
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
# shared metric: strike + arbitrary-q aware, one source of truth
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
# generic checkpoint loader (handles esm2 AND from-scratch transformer)
from ensemble_logit import load_m

REPO = "/home/aakarsh/projects/antiGen"
R = "/large_storage/hielab/aakarsh/multitree"
NA = f"{HERE}/data/forecast_noattach"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
BUDGETS = {"10": 0.10, "05": 0.05, "01": 0.01}
NBOOT = 2000
SEEDS = [0, 1, 2]

OUT_DIR = f"{R}/fc13_core_surface/retrain_ensembles"
DATA_DIR = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0"
NPZ = f"{OUT_DIR}/retrain_ensembles.npz"
CSV = f"{DATA_DIR}/retrain_ensembles.csv"
PROV = f"{DATA_DIR}/retrain_ensembles_provenance.json"

# arm -> (shared_across_trees, dir_template). {tree}/{seed} filled per call.
ARM_DIRS = {
    'xformer3':    (False, f"{R}/fc13_single_xformer_noattach/single_{{tree}}_s{{seed}}"),
    'single650p4': (False, f"{R}/fc13_single_esm650_p4_noattach/single_{{tree}}_s{{seed}}"),
    'joint13p4':   (True,  f"{R}/fc13_joint_650m/joint13_p4_val_s{{seed}}"),
    'joint49p4':   (True,  f"{R}/fc13_core_surface/joint_surface_snapshot_p4_val_s{{seed}}"),
}
ARMS = list(ARM_DIRS)


def best_ckpt(d):
    cks = [p for p in glob.glob(f"{d}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return cks[-1] if cks else None


def seed_ckpts(arm, tree):
    _, tmpl = ARM_DIRS[arm]
    paths = [best_ckpt(tmpl.format(tree=tree, seed=s)) for s in SEEDS]
    return paths if all(paths) else None


def boot_ci(w, nboot=NBOOT, seed=0):
    a = np.asarray(w, dtype=np.float64)
    if a.size == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(nboot, a.size))
    means = a[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@torch.no_grad()
def ensemble_map(models, ab):
    """Logit-level ensemble: mean of the raw (L,E) logit maps, then returned for ranking."""
    outs = [m(ab).float().squeeze(0).cpu() for m in models]
    return torch.stack(outs, 0).mean(0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)

    # --- precompute per-tree, shared across arms: L, seen_mask (FULL train file),
    #     and the test/novel examples with their events. Identical for every arm. ---
    tree_data = {}
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        seen_counts = torch.zeros(L, E)
        for ex in tr:
            seen_counts += ex.mat
        seen_mask = seen_counts > 0            # == eval.py pssm_root>0, from the FIXED file
        subs = {}
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            entries = []
            for ex in te:
                ev = events_of(ex)
                if ev:
                    entries.append((ex.aa_idx, ev))
            if entries:
                subs[sub] = entries
        tree_data[tree] = (L, seen_mask, subs)

    out = {}
    resolved = {}
    for arm in ARMS:
        shared, _ = ARM_DIRS[arm]
        shared_models = None
        if shared:
            cks = seed_ckpts(arm, TREES[0])
            if cks is None:
                print(f"[skip arm] {arm}: missing checkpoints"); continue
            resolved[arm] = cks
            shared_models = [load_m(c) for c in cks]
        else:
            resolved[arm] = {}
        for tree in TREES:
            L, seen_mask, subs = tree_data[tree]
            if shared:
                models = shared_models
            else:
                cks = seed_ckpts(arm, tree)
                if cks is None:
                    print(f"[skip] {arm} {tree}: missing checkpoints"); continue
                resolved[arm][tree] = cks
                models = [load_m(c) for c in cks]
            for sub, entries in subs.items():
                pools = [('full', None)] if sub == 'all' else [('full', None), ('denovo', seen_mask)]
                acc = {(pool, qk): [] for pool, _ in pools for qk in BUDGETS}
                for aa, ev in entries:
                    ab = aa.unsqueeze(0).to(dev)
                    mp = ensemble_map(models, ab)
                    for pool, smask in pools:
                        for qk, qv in BUDGETS.items():
                            acc[(pool, qk)] += per_event_weights(mp, aa, ev, seen_mask=smask, q=qv)
                for (pool, qk), v in acc.items():
                    out[f"{tree}__{sub}__{pool}__{arm}__q{qk}"] = np.array(v, dtype=np.float32)
            if not shared:
                del models
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
            print(f"done {arm} {tree}", flush=True)
        if shared:
            del shared_models
            if dev.type == 'cuda':
                torch.cuda.empty_cache()

    np.savez_compressed(NPZ, **out)
    print("wrote", NPZ, flush=True)

    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["virus", "subset", "pool", "arm", "q", "n_events", "recall", "ci_lo", "ci_hi"])
        for tree in TREES:
            for sub in ("all", "novel"):
                for pool in ("full", "denovo"):
                    for arm in ARMS:
                        for qk in BUDGETS:
                            key = f"{tree}__{sub}__{pool}__{arm}__q{qk}"
                            if key not in out:
                                continue
                            a = out[key]
                            lo, hi = boot_ci(a)
                            w.writerow([tree, sub, pool, arm, f"0.{qk}", len(a),
                                        f"{a.mean():.6f}" if len(a) else "nan",
                                        f"{lo:.6f}", f"{hi:.6f}"])
    print("wrote", CSV, flush=True)

    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"]).decode().strip())
    except Exception:
        sha, dirty = "unknown", None
    prov = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git_sha": sha, "git_dirty": dirty, "script": os.path.abspath(__file__),
        "metric": "score_perevent.per_event_weights (shared); de-novo pool strikes pssm_root>0 "
                  "built from the fixed per-tree train file; arms = 3-seed logit ensembles",
        "D": D, "data_dir": NA, "trees": TREES, "budgets": BUDGETS, "nboot": NBOOT,
        "seeds": SEEDS, "arms": ARMS, "resolved_checkpoints": resolved,
        "torch": torch.__version__, "device": str(dev), "npz": NPZ, "csv": CSV,
        "schema_matches": "denovo_panels_recall.csv",
    }
    with open(PROV, "w") as fh:
        json.dump(prov, fh, indent=2)
    print("wrote", PROV, flush=True)


if __name__ == '__main__':
    main()
