#!/usr/bin/env python
"""Re-score the two JOINT retrain arms with PER-VIRUS (per-source) val-recall@10 epoch
selection, instead of the single macro-over-all-sources _best.pt epoch.

For each eval virus v and seed s, pick the epoch maximizing that virus's OWN val recall@10
(hit_rate['0.1'] in <run>/*_val_metrics_by_epoch.json; tie -> earliest epoch). Then the
arm's prediction for v is the 3-seed logit ensemble of {seed s @ its best epoch for v}.

Everything else is identical to score_retrain_ensembles.py: shared per_event_weights metric,
§67 de-novo strike from the FIXED train file, pools {full, denovo}, q {.10,.05,.01},
bootstrap CI (nboot=2000, seed 0), same test/novel sets, same schema. Writes
retrain_ensembles_pervirussel.csv (+ _provenance.json).
"""
import os, sys, glob, json, csv, socket, subprocess, datetime
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
from ensemble_logit import load_m

REPO = "/home/aakarsh/projects/antiGen"
R = "/large_storage/hielab/aakarsh/multitree"
NA = f"{HERE}/data/forecast_noattach"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
BUDGETS = {"10": 0.10, "05": 0.05, "01": 0.01}
NBOOT = 2000
SEEDS = [0, 1, 2]
ARMS = {
    'joint13p4': f"{R}/fc13_joint_650m/joint13_p4_val_s{{s}}",
    'joint49p4': f"{R}/fc13_core_surface/joint_surface_snapshot_p4_val_s{{s}}",
}
DATA_DIR = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0"
CSV = f"{DATA_DIR}/retrain_ensembles_pervirussel.csv"
PROV = f"{DATA_DIR}/retrain_ensembles_pervirussel_provenance.json"


def best_epoch(tmpl, seed, virus):
    f = glob.glob(f"{tmpl.format(s=seed)}/*_val_metrics_by_epoch.json")[0]
    d = json.load(open(f))
    cur = [(rec['epoch'], rec['val']['per_source'][virus]['hit_rate']['0.1'])
           for rec in d if virus in rec['val']['per_source']]
    return max(cur, key=lambda er: (er[1], -er[0]))[0]   # max recall, tie -> earliest


def ckpt_at(tmpl, seed, epoch):
    hits = glob.glob(f"{tmpl.format(s=seed)}/*_epoch{epoch}.pt")
    return hits[0] if hits else None


def boot_ci(w, nboot=NBOOT, seed=0):
    a = np.asarray(w, dtype=np.float64)
    if a.size == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(nboot, a.size))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@torch.no_grad()
def ensemble_map(models, ab):
    return torch.stack([m(ab).float().squeeze(0).cpu() for m in models], 0).mean(0)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # precompute per-tree seen_mask (fixed train file) + test/novel events
    tree_data = {}
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        seen = torch.zeros(L, E)
        for ex in tr:
            seen += ex.mat
        seen = seen > 0
        subs = {}
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            entries = [(ex.aa_idx, events_of(ex)) for ex in te if events_of(ex)]
            if entries:
                subs[sub] = entries
        tree_data[tree] = (L, seen, subs)

    out = {}
    chosen = {}   # arm -> {virus: [ep_s0, ep_s1, ep_s2]}
    cache = {}    # (arm,seed,epoch) -> model, bounded
    def get_model(arm, seed, epoch):
        key = (arm, seed, epoch)
        if key not in cache:
            if len(cache) >= 6:
                cache.clear()
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
            ck = ckpt_at(ARMS[arm], seed, epoch)
            if ck is None:
                raise SystemExit(f"missing ckpt {arm} s{seed} epoch{epoch}")
            cache[key] = load_m(ck)
        return cache[key]

    for arm, tmpl in ARMS.items():
        chosen[arm] = {}
        for tree in TREES:
            eps = [best_epoch(tmpl, s, tree) for s in SEEDS]
            chosen[arm][tree] = eps
            models = [get_model(arm, s, eps[s]) for s in SEEDS]
            L, seen, subs = tree_data[tree]
            for sub, entries in subs.items():
                pools = [('full', None)] if sub == 'all' else [('full', None), ('denovo', seen)]
                acc = {(pool, qk): [] for pool, _ in pools for qk in BUDGETS}
                for aa, ev in entries:
                    ab = aa.unsqueeze(0).to(dev)
                    mp = ensemble_map(models, ab)
                    for pool, smask in pools:
                        for qk, qv in BUDGETS.items():
                            acc[(pool, qk)] += per_event_weights(mp, aa, ev, seen_mask=smask, q=qv)
                for (pool, qk), v in acc.items():
                    out[f"{tree}__{sub}__{pool}__{arm}__q{qk}"] = np.array(v, dtype=np.float32)
            print(f"done {arm} {tree} epochs={eps}", flush=True)

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
                            a = out[key]; lo, hi = boot_ci(a)
                            w.writerow([tree, sub, pool, arm, f"0.{qk}", len(a),
                                        f"{a.mean():.6f}" if len(a) else "nan",
                                        f"{lo:.6f}", f"{hi:.6f}"])
    print("wrote", CSV, flush=True)

    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    json.dump({
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git_sha": sha, "script": os.path.abspath(__file__),
        "selection": "per-virus (per-source) val recall@10 best epoch; tie->earliest",
        "metric": "score_perevent.per_event_weights; de-novo strike pssm_root>0 from fixed train file",
        "D": D, "trees": TREES, "budgets": BUDGETS, "nboot": NBOOT, "seeds": SEEDS,
        "chosen_epochs": chosen, "device": str(dev),
        "schema_matches": "denovo_panels_recall.csv", "csv": CSV,
    }, open(PROV, "w"), indent=2)
    print("wrote", PROV, flush=True)


if __name__ == '__main__':
    main()
