#!/usr/bin/env python
"""9-virus / 10-seed eval for the standardized 650m/p4 model arms.

New eval panel = 9 viruses (10-virus panel minus denv2; no dengue at all). Arms:
single (per-virus), joint9 (NEW: joint-trained on the 9), joint49 (13+36 surface).
joint13 dropped. 10-seed MeanLogProb ensemble, posthoc NLL-min checkpoint selection
(macro=pooled overall NLL for joint arms; per-tree NLL for single), per_event_weights
metric, §67 de-novo strike. Arm-outer loop: each joint arm's 10 seed-models are loaded
once and reused across all 9 viruses.

Writes (to multitree_noattach_j9/):
  HI_j9.npz           per-event weights, key = virus__subset__pool__arm
  HI_j9_pervirus.csv  per-virus recall+CI per arm (basis for macro averaging)
  HI_j9_ladder.csv    pooled (event-weighted) recall per arm
  HI_j9_forest.csv    per-virus (joint9-single, joint49-single) paired delta + CI
"""
import os, sys, glob, json, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
from ensemble_logit import load_m

R = "/large_storage/hielab/aakarsh/multitree"
REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
DATA_DIR = f"{REPO}/antiGen-figures/data/multitree_noattach_j9{os.environ.get('J9_SUFFIX', '')}"
os.makedirs(DATA_DIR, exist_ok=True)
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles',
         'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
SEEDS = list(range(10)); NBOOT = 2000; Q = 0.10
# joint arms: path has no {t}; single is per-virus.
DIRS = {'single': f"{R}/fc13_single_esm650_p4_noattach/single_{{t}}_s{{s}}",
        'joint9':  f"{R}/fc13_joint9/joint9_p4_val_s{{s}}",
        'joint49': f"{R}/fc13_core_surface/joint_surface_snapshot_p4_val_s{{s}}"}
JOINT_ARMS = ('joint9', 'joint49')
ARMS = ('single', 'joint9', 'joint49')


def _load_records(path):
    """Tolerant parse of a val_metrics_by_epoch.json. A clean file loads normally;
    a file whose write was truncated mid-record (disk-full during training) is
    salvaged by extracting every complete top-level {...} element by brace depth."""
    txt = open(path).read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    recs = []; depth = 0; start = None; instr = False; esc = False
    for i, c in enumerate(txt):
        if instr:
            if esc: esc = False
            elif c == '\\': esc = True
            elif c == '"': instr = False
            continue
        if c == '"': instr = True; continue
        if c == '{':
            if depth == 0: start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try: recs.append(json.loads(txt[start:i + 1]))
                except Exception: pass
                start = None
    print(f"  [salvaged {len(recs)} epoch records from truncated {os.path.basename(path)}]", flush=True)
    return recs


INTACT_MIN = 2_000_000_000  # bytes; a genuine 650M-p4 ckpt is ~2.604e9, truncated far smaller


def _intact(p):
    try:
        return p is not None and os.path.getsize(p) >= INTACT_MIN
    except OSError:
        return False


def nll_ranked_epochs(d):
    """Epochs ordered by ascending val NLL (ties -> earliest). Tolerant parse."""
    f = glob.glob(f"{d}/*_val_metrics_by_epoch.json"); rec = _load_records(f[0])
    cur = [(r['epoch'], r['val']['overall']['nll_per_mutation']) for r in rec
           if r.get('epoch', -1) >= 0 and r['val']['overall'].get('nll_per_mutation') is not None]
    return [e for e, _ in sorted(cur, key=lambda er: (er[1], er[0]))]


def macro_epoch(d):
    eps = nll_ranked_epochs(d)
    if not eps: raise ValueError(f"no NLL epochs in {d}")
    return eps[0]


def epoch_pt(d, ep):
    c = glob.glob(f"{d}/*_epoch{ep}.pt"); return c[0] if c else None


# Optional "tree:s,s;tree:s" exclusions (e.g. a prelim skipping runs still retraining,
# to avoid reading checkpoints mid-write). For single arm only (keyed by tree).
_EXCL = {}
for _grp in os.environ.get('J9_SINGLE_EXCLUDE', '').split(';'):
    if ':' in _grp:
        _t, _ss = _grp.split(':', 1)
        _EXCL[_t] = {int(x) for x in _ss.split(',') if x.strip() != ''}


def load_seed_model(arm, tree, seed):
    """Load the lowest-NLL epoch whose checkpoint is BOTH intact by size AND actually
    loadable (some disk-full-truncated .pt pass the size prefilter but fail torch.load).
    Falls back down the NLL ranking; returns (model, epoch) or None if nothing loads."""
    d = DIRS[arm].format(t=tree, s=seed)
    if not glob.glob(f"{d}/*_val_metrics_by_epoch.json"):
        return None
    try:
        ranked = nll_ranked_epochs(d)
    except Exception:
        return None
    for ep in ranked:
        p = epoch_pt(d, ep)
        if not _intact(p):
            continue
        try:
            m = load_m(p)
        except Exception as e:
            print(f"  [skip {arm} {tree} s{seed} epoch{ep}: load failed {type(e).__name__}]", flush=True)
            continue
        if ep != ranked[0]:
            print(f"  [{arm} {tree} s{seed}: NLL-min epoch{ranked[0]} unusable -> epoch{ep}]", flush=True)
        return m, ep
    print(f"  [DROP {arm} {tree} s{seed}: no loadable checkpoint]", flush=True)
    return None


def load_ensemble(arm, tree, excl=()):
    """Load all available seed-models for an arm/tree, skipping excluded and corrupt ones."""
    models, seeds = [], []
    for s in SEEDS:
        if s in excl:
            continue
        r = load_seed_model(arm, tree, s)
        if r is not None:
            models.append(r[0]); seeds.append(s)
    return models, seeds


@torch.no_grad()
def emap(models, ab):
    lps = []
    for m in models:
        o = m(ab).float().squeeze(0)
        lps.append(torch.log_softmax(o.reshape(-1), dim=-1).reshape(o.shape).cpu())
    return torch.stack(lps, 0).mean(0)


def boot_ci(vec):
    a = np.asarray(vec, float)
    if a.size == 0: return (float('nan'), float('nan'))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, a.size, (NBOOT, a.size))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def free(models):
    for m in models:
        try: del m
        except Exception: pass
    if dev.type == 'cuda': torch.cuda.empty_cache()


def main():
    # per-tree: seen mask + {subset: entries}
    tdata = {}
    for t in TREES:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        L = len(tr[0].aa_idx); seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        subs = {}
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{t}@{D}_{suf}.txt"
            if os.path.exists(p) and os.path.getsize(p) > 0:
                te = load_examples(p, ALPHA_SET, None, source=t)
                ent = [(ex.aa_idx, events_of(ex)) for ex in te if events_of(ex)]
                if ent: subs[sub] = ent
        tdata[t] = (seen, subs)

    W = {}  # virus__subset__pool__arm -> np.array (event order stable across arms)

    def score_arm_tree(arm, tree, models):
        seen, subs = tdata[tree]
        for sub, ent in subs.items():
            pools = [('full', None)] if sub == 'all' else [('full', None), ('denovo', seen)]
            acc = {pool: [] for pool, _ in pools}
            for aa, ev in ent:
                mp = emap(models, aa.unsqueeze(0).to(dev))
                for pool, smask in pools:
                    acc[pool] += per_event_weights(mp, aa, ev, seen_mask=smask, q=Q)
            for pool, v in acc.items():
                W[f"{tree}__{sub}__{pool}__{arm}"] = np.asarray(v, np.float32)

    seed_log = []  # (arm, tree, n_seeds, seeds)
    # joint arms: load seed-models once (same across trees), reuse
    for arm in JOINT_ARMS:
        models, seeds = load_ensemble(arm, None)
        seed_log.append((arm, "-", len(seeds), seeds))
        if not models: raise SystemExit(f"no loadable {arm} seed-models")
        for t in TREES:
            score_arm_tree(arm, t, models); print(f"  {arm} {t}", flush=True)
        free(models)
    # single arm: per-tree ensemble over the AVAILABLE (loadable, non-excluded) seeds
    for t in TREES:
        models, seeds = load_ensemble('single', t, excl=_EXCL.get(t, ()))
        seed_log.append(("single", t, len(seeds), seeds))
        if not models: raise SystemExit(f"no loadable single seed-models for {t}")
        score_arm_tree('single', t, models); free(models)
        print(f"  single {t}  (seeds={seeds})", flush=True)
    with open(f"{DATA_DIR}/single_seed_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "virus", "n_seeds", "seeds"])
        for arm, t, n, ss in seed_log: w.writerow([arm, t, n, " ".join(map(str, ss))])

    np.savez_compressed(f"{DATA_DIR}/HI_j9.npz", **W)

    # per-virus recall per arm (macro basis)
    with open(f"{DATA_DIR}/HI_j9_pervirus.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["virus", "subset", "pool", "arm", "n_events", "recall", "ci_lo", "ci_hi"])
        for t in TREES:
            for sub, pool in [("all", "full"), ("novel", "denovo")]:
                for a in ARMS:
                    k = f"{t}__{sub}__{pool}__{a}"
                    if k not in W: continue
                    v = W[k]; lo, hi = boot_ci(v)
                    w.writerow([t, sub, pool, a, len(v), f"{v.mean():.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    # pooled (event-weighted) recall per arm
    with open(f"{DATA_DIR}/HI_j9_ladder.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subset", "pool", "arm", "n_events", "pooled_recall", "ci_lo", "ci_hi"])
        for sub, pool in [("all", "full"), ("novel", "denovo")]:
            for a in ARMS:
                ks = [k for k in W if k.endswith(f"__{sub}__{pool}__{a}")]
                vec = np.concatenate([W[k] for k in ks]) if ks else np.array([])
                lo, hi = boot_ci(vec)
                w.writerow([sub, pool, a, len(vec),
                            f"{vec.mean():.6f}" if len(vec) else "nan", f"{lo:.6f}", f"{hi:.6f}"])

    # per-virus (joint - single) paired delta + CI
    with open(f"{DATA_DIR}/HI_j9_forest.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["virus", "subset", "pool", "contrast", "n_events",
                                        "single", "joint", "delta", "d_lo", "d_hi"])
        for t in TREES:
            for sub, pool in [("all", "full"), ("novel", "denovo")]:
                sk = f"{t}__{sub}__{pool}__single"
                if sk not in W: continue
                sv = W[sk]
                for jarm in JOINT_ARMS:
                    jk = f"{t}__{sub}__{pool}__{jarm}"
                    if jk not in W: continue
                    jv = W[jk]
                    n = min(len(sv), len(jv)); d = jv[:n] - sv[:n]
                    lo, hi = boot_ci(d)
                    w.writerow([t, sub, pool, f"{jarm}-single", n,
                                f"{sv[:n].mean():.6f}", f"{jv[:n].mean():.6f}",
                                f"{d.mean():.6f}", f"{lo:.6f}", f"{hi:.6f}"])
    print("wrote HI_j9 npz + pervirus + ladder + forest", flush=True)


if __name__ == '__main__':
    main()
