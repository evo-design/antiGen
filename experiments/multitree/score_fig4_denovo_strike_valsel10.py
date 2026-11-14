#!/usr/bin/env python
"""A-F de-novo recall@10% under the NOTES-§67 strike convention, VALSEL10 recipe. Same as
score_fig4_denovo_strike.py except: (a) models are the NLL-selected 10-seed ensembles from
endemic_valsel10, and (b) the two model arms are combined by MEAN-LOG-PROB (MeanLogProbEnsemble:
log_softmax over the full flattened L*E grid per seed, then mean across seeds) instead of raw
logit mean. Baselines/zero-shot are single maps, so their ranking is unchanged by the combine.
Emits antigen-b9's schema to denovo_panels_AF_recall_valsel10.csv (pool=full and pool=denovo)."""
import os, sys, glob, csv
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.eval_utils import (fit_substitution_baseline, positional_marginal_counts,
    frequency_baseline_scores, where_only_baseline_scores, load_checkpoint)
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
import count_baselines
from score_perevent import events_of, load_esm, per_event_weights, E, ALPHA_SET, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
RUNS = f"{R}/endemic_valsel10"
DST = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/denovo_panels_AF_recall_valsel10.csv"
CUTS = {"denv2": "2019-10-01", "rsv_a": "2023-10-01", "h3n2": "2022-10-01"}
SEEDS = range(10); Q = 0.10; B = 2000

def mlp_global(z):
    """MeanLogProbEnsemble: log_softmax over the flattened L*E grid per seed, mean over seeds."""
    S, L, Ee = z.shape
    return torch.log_softmax(z.reshape(S, L * Ee), dim=-1).reshape(S, L, Ee).mean(0)

def pew_pool(scores, aa_idx, events, Nmat, strike):
    """fractional top-Q hit per event; if strike, strike training-seen (N>0) cells from the
    candidate pool before the top-q cut. Delegates to the CANONICAL kernel
    score_perevent.per_event_weights (@9d09a17) so the model rungs share it bit-for-bit with
    the de-novo panel baselines; seen_mask=(N>0) on the strike pool, None on the full pool."""
    return per_event_weights(scores, aa_idx, events,
                             seen_mask=(Nmat > 0) if strike else None, q=Q)

def best_pts(run):
    """the SINGLE newest *_best.pt in the seed dir (ignoring per-source *_best_<src>.pt).
    Guards against a stale checkpoint left by the earlier early-stop valsel10 run that shared
    this output dir: the no-early-stop rerun always has the later mtime, so take the max."""
    cands = [p for p in glob.glob(f"{run}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return [max(cands, key=os.path.getmtime)] if cands else []

def ci(w, rng):
    w = np.asarray(w, float)
    if not len(w): return (np.nan, np.nan, np.nan)
    bo = w[rng.integers(0, len(w), size=(B, len(w)))].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))

def main():
    rng = np.random.default_rng(0)
    rows = []
    for tree, cut in CUTS.items():
        full = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        L = len(full[0].aa_idx); st = count_baselines.fit(full, L, E); N = st["N"]
        te = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", ALPHA_SET, None, source=tree)
        ev_all = [(ex, [(p, a) for (p, a) in events_of(ex) if a not in MASK]) for ex in te]
        ev_all = [(ex, ev) for ex, ev in ev_all if ev]
        dn_mask = np.array([float(N[p, a]) < 1 for ex, ev in ev_all for (p, a) in ev], bool)
        n_dn = int(dn_mask.sum())
        print(f"\n{tree}@{cut}: {len(dn_mask)} events, {n_dn} de-novo", flush=True)

        fitted = fit_substitution_baseline(full, dev, ALPHA_SET)
        sub_mat = torch.log_softmax(fitted.score_matrix().detach().cpu(), dim=-1)
        pos_counts = positional_marginal_counts(full, L, E)
        freq = frequency_baseline_scores(full, L, E); wonly = where_only_baseline_scores(full, L, E)
        cb = create_model(ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=L,
                 alphabet=AlphabetConfig(amino_acids=ALPHA_SET), codon_blosum=CodonBlosumConfig(kappa=4.0)), dev).eval()
        zc = glob.glob(f"{R}/fc13_single_esm650/single_denv2_s0/*_best.pt") or \
             glob.glob(f"{R}/fc13_single_esm650_noattach/single_ncov_s0/*_best.pt")
        _, zcfg = load_checkpoint(zc[0], map_location='cpu'); zsm = create_model(zcfg, dev).eval()

        def arm_map(ex, kind, models=None):
            aa = ex.aa_idx; ab = aa.unsqueeze(0); L_ = L
            if kind == 'ensemble':
                with torch.no_grad():
                    z = torch.stack([m(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu() for m in models], 0)
                return mlp_global(z)                                 # MeanLogProbEnsemble (global)
            if kind == 'zeroshot':
                with torch.no_grad(): return zsm(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
            if kind == 'codon_blosum':
                with torch.no_grad(): return cb(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
            if kind == 'where_times_which':
                return sub_mat[aa] + torch.log(pos_counts.clamp(min=1e-12)).view(L_, 1)
            if kind == 'frequency': return torch.log(freq)
            return count_baselines.score_map(kind, st, ab).squeeze(0)     # cpssm_dir/site_prop/pssm

        MODEL_ARMS = [("xformer", "Transformer (scratch)"), ("esm35m_ft", "ESM-2 35M finetune")]
        BASE_ARMS = [("cpssm_dir", "Smoothed Counts"), ("frequency", "Event Counts"),
                     ("site_prop", "Site x Substitution"), ("where_times_which", "Where x which"),
                     ("pssm", "PSSM"), ("codon_blosum", "Codon aware BLOSUM62"),
                     ("zeroshot", "ESM-2 zero-shot")]
        def emit(arm_label, wfull, wstrike):
            for pool, w in (("full", wfull), ("denovo", wstrike)):
                wd = np.asarray(w)[dn_mask]; m, lo, hi = ci(wd, rng)
                rows.append([tree, "novel", pool, arm_label, 0.10, n_dn, m, lo, hi,
                             "attach", cut, "endemic_valsel10_10seed"])

        for key, lab in MODEL_ARMS:
            models = [load_esm(p) for p in sorted(sum([best_pts(f"{RUNS}/{tree}_{key}_s{s}") for s in SEEDS], []))]
            if not models:
                print(f"  {lab}: NO CKPTS at {RUNS}/{tree}_{key}_s*", flush=True); continue
            wf, ws = [], []
            for ex, ev in ev_all:
                sm = arm_map(ex, 'ensemble', models)
                wf += pew_pool(sm, ex.aa_idx, ev, N, False)
                ws += pew_pool(sm, ex.aa_idx, ev, N, True)
            del models
            if dev.type == 'cuda': torch.cuda.empty_cache()
            emit(lab, wf, ws); print(f"  {lab} ({len(list(SEEDS))}-seed MLP): full {np.asarray(wf)[dn_mask].mean():.3f}  strike {np.asarray(ws)[dn_mask].mean():.3f}", flush=True)
        for key, lab in BASE_ARMS:
            wf, ws = [], []
            for ex, ev in ev_all:
                sm = arm_map(ex, key)
                wf += pew_pool(sm, ex.aa_idx, ev, N, False)
                ws += pew_pool(sm, ex.aa_idx, ev, N, True)
            emit(lab, wf, ws); print(f"  {lab}: full {np.asarray(wf)[dn_mask].mean():.3f}  strike {np.asarray(ws)[dn_mask].mean():.3f}", flush=True)
        for pool in ("full", "denovo"):
            rows.append([tree, "novel", pool, "Uniform", 0.10, n_dn, 0.10, 0.10, 0.10, "attach", cut, "endemic_valsel10_10seed"])

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["virus","subset","pool","arm","q","n_events","recall","ci_lo","ci_hi","dataset","cut","ckpt_set"])
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows)} rows)")

if __name__ == '__main__':
    main()
