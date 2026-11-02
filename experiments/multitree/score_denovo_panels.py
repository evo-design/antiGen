#!/usr/bin/env python
"""Re-eval de-novo (and all-substitution) recall for every Figure-5 panel, ONE pipeline.

Replaces the hardcoded single-virus bars and regenerates the 13-virus ladder from the
SAME 650M no-attach arms, with the canonical de-novo pool (training-seen cells struck).
Nothing here is hardcoded: arms are resolved from checkpoint globs at run time, the
metric is the shared `score_perevent.per_event_weights` (one source of truth), and the
results land in a tidy CSV plus a per-event npz plus a provenance JSON.

Arms (the "right models", identical definitions to score_perevent_sp4e0_budgets.py):
  joint    multitree49 snapshot partial-4, epoch 0   (one ckpt, all trees)
  joint13  multitree13 partial-4, epoch 0            (one ckpt, all trees)
  single   single-virus ESM-650M no-attach, per-tree _best.pt
  zeroshot pretrained ESM-2 650M, no finetuning
  site_prop cpssm_dir pssm frequency where_times_which substitution codon_blosum where_only
           count/substitution/codon-BLOSUM baselines fit on the no-attach train corpus

For every (tree, subset, arm, q) we record recall both ways:
  pool=full    the whole-protein candidate pool (the OLD behaviour; `all` uses only this)
  pool=denovo  training-seen cells struck (the canonical de-novo pool; `novel` only)
so the effect of the strike is in the CSV rather than lost. Recall = mean per-event hit
weight; CI = event-level bootstrap (2000 resamples, seed 0).

Outputs (paths printed at the end):
  <STORE>/fc13_core_surface/denovo_panels/perevent_denovo_panels.npz   per-event weights
  antiGen-figures/data/multitree_noattach_sp4e0/denovo_panels_recall.csv   tidy summary
  antiGen-figures/data/multitree_noattach_sp4e0/denovo_panels_provenance.json
"""
import os, sys, glob, json, csv, socket, subprocess, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
from antiGen.eval_utils import (load_checkpoint, fit_substitution_baseline,
    positional_marginal_counts, frequency_baseline_scores, where_only_baseline_scores)
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
import count_baselines
# the metric is imported, never recopied: strike + arbitrary-q aware
from score_perevent import per_event_weights, events_of, load_esm, E, ALPHA_SET, D, dev

REPO = "/home/aakarsh/projects/antiGen"
R = "/large_storage/hielab/aakarsh/multitree"
NA = f"{REPO}/experiments/multitree/data/forecast_noattach"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
BUDGETS = {"10": 0.10, "05": 0.05, "01": 0.01}
NBOOT = 2000

OUT_DIR = f"{R}/fc13_core_surface/denovo_panels"
DATA_DIR = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0"
NPZ = f"{OUT_DIR}/perevent_denovo_panels.npz"
CSV = f"{DATA_DIR}/denovo_panels_recall.csv"
PROV = f"{DATA_DIR}/denovo_panels_provenance.json"

# shared-checkpoint globs (resolved once; recorded in provenance)
GLOB_JOINT   = f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt"
GLOB_JOINT13 = f"{R}/fc13_joint_650m/joint13_partial4_noattach_noval/*_epoch0.pt"
GLOB_ZCFG    = f"{R}/fc13_single_esm650_noattach/single_ncov_s0/*_best.pt"
def glob_single(tree):
    return [p for p in glob.glob(f"{R}/fc13_single_esm650_noattach/single_{tree}_s0/*_best.pt")
            if "_best_" not in p][-1]


def build_codon_blosum(seq_len):
    cfg = ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=seq_len,
                      alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                      codon_blosum=CodonBlosumConfig(kappa=4.0))
    return create_model(cfg, dev).eval()


def boot_ci(w, nboot=NBOOT, seed=0):
    """Event-level bootstrap percentile CI for the mean hit weight."""
    a = np.asarray(w, dtype=np.float64)
    if a.size == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(nboot, a.size))
    means = a[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)

    j_ck = glob.glob(GLOB_JOINT)[0]; j13_ck = glob.glob(GLOB_JOINT13)[0]
    zcfg_ck = glob.glob(GLOB_ZCFG)[0]
    resolved = {"joint": j_ck, "joint13": j13_ck, "zeroshot_config": zcfg_ck, "single": {}}
    jm = load_esm(j_ck); j13m = load_esm(j13_ck)
    _, zcfg = load_checkpoint(zcfg_ck, map_location='cpu')
    zsm = create_model(zcfg, dev).eval()   # zeroshot = pretrained ESM-2 650M

    MODEL_ARMS = ['joint', 'joint13', 'single', 'zeroshot']
    BASE_ARMS = ['site_prop','cpssm_dir','pssm','frequency','where_times_which',
                 'substitution','codon_blosum','where_only']
    ARMS = MODEL_ARMS + BASE_ARMS

    out = {}          # per-event weights, keyed tree__sub__pool__arm__q{..}
    counts = {}       # (tree, sub) -> n_novel cells reference, for the CSV
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        # training-root count matrix -> de-novo "seen" mask (== eval.py's pssm_root>0)
        seen_counts = torch.zeros(L, E)
        for ex in tr: seen_counts += ex.mat
        seen_mask = seen_counts > 0

        st = count_baselines.fit(tr, L, E)
        fitted = fit_substitution_baseline(tr, dev, ALPHA_SET)
        sub_mat = torch.log_softmax(fitted.score_matrix().detach().cpu(), dim=-1)
        pos_counts = positional_marginal_counts(tr, L, E)
        freq = frequency_baseline_scores(tr, L, E); wonly = where_only_baseline_scores(tr, L, E)
        cb = build_codon_blosum(L)
        s_ck = glob_single(tree); resolved["single"][tree] = s_ck
        sm = load_esm(s_ck)

        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            # pool=denovo only makes sense for the novel subset; all uses full only.
            pools = [('full', None)] if sub == 'all' else [('full', None), ('denovo', seen_mask)]
            acc = {(pool, r, qk): [] for pool, _ in pools for r in ARMS for qk in BUDGETS}
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js = jm(ab).float().squeeze(0).cpu()
                    j13s = j13m(ab).float().squeeze(0).cpu()
                    ss = sm(ab).float().squeeze(0).cpu()
                    zs = zsm(ab).float().squeeze(0).cpu()
                    cbs = cb(ab).float().squeeze(0).cpu()
                ab1 = aa.unsqueeze(0)
                maps = {'joint': js, 'joint13': j13s, 'single': ss, 'zeroshot': zs,
                        'site_prop': count_baselines.score_map('site_prop', st, ab1).squeeze(0),
                        'cpssm_dir': count_baselines.score_map('cpssm_dir', st, ab1).squeeze(0),
                        'pssm': count_baselines.score_map('pssm', st, ab1).squeeze(0),
                        'frequency': torch.log(freq), 'where_only': torch.log(wonly),
                        'where_times_which': sub_mat[aa] + torch.log(pos_counts.clamp(min=1e-12)).view(L, 1),
                        'substitution': sub_mat[aa], 'codon_blosum': cbs}
                for pool, smask in pools:
                    for r, mp in maps.items():
                        for qk, qv in BUDGETS.items():
                            acc[(pool, r, qk)] += per_event_weights(mp, aa, ev, seen_mask=smask, q=qv)
            for (pool, r, qk), v in acc.items():
                out[f"{tree}__{sub}__{pool}__{r}__q{qk}"] = np.array(v, dtype=np.float32)
        del sm
        if dev.type == 'cuda': torch.cuda.empty_cache()
        print(f"done {tree}", flush=True)

    np.savez_compressed(NPZ, **out)
    print("wrote", NPZ, flush=True)

    # tidy CSV
    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["virus", "subset", "pool", "arm", "q", "n_events", "recall", "ci_lo", "ci_hi"])
        for tree in TREES:
            for sub in ("all", "novel"):
                for pool in ("full", "denovo"):
                    for r in ARMS:
                        for qk in BUDGETS:
                            key = f"{tree}__{sub}__{pool}__{r}__q{qk}"
                            if key not in out: continue
                            a = out[key]
                            lo, hi = boot_ci(a)
                            w.writerow([tree, sub, pool, r, f"0.{qk}", len(a),
                                        f"{a.mean():.6f}" if len(a) else "nan",
                                        f"{lo:.6f}", f"{hi:.6f}"])
    print("wrote", CSV, flush=True)

    # provenance
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"]).decode().strip())
    except Exception:
        sha, dirty = "unknown", None
    prov = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git_sha": sha, "git_dirty": dirty,
        "script": os.path.abspath(__file__),
        "metric": "score_perevent.per_event_weights (shared); de-novo pool strikes pssm_root>0",
        "D": D, "data_dir": NA, "trees": TREES, "budgets": BUDGETS, "nboot": NBOOT,
        "arms": ARMS, "resolved_checkpoints": resolved,
        "torch": torch.__version__, "device": str(dev),
        "npz": NPZ, "csv": CSV,
    }
    with open(PROV, "w") as fh:
        json.dump(prov, fh, indent=2)
    print("wrote", PROV, flush=True)


if __name__ == '__main__':
    main()
