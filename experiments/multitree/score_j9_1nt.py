#!/usr/bin/env python
"""9-virus macro ladders (Fig-5 panels H=all / I=de novo) recomputed TWICE per rung:
  * baseline = the published j9 convention (full L*E pool; de-novo = NOTES-s67 strike)
  * onent    = candidate pool AND observed targets both restricted to amino-acid
               substitutions reachable by a SINGLE nucleotide change (codon-agnostic),
               levelling the genetic code across every rung.

Rungs (same as build_fig4_j9.RUNGS): joint49-sel9, joint9, single, xformer (antiGen arms),
then cpssm_dir / site_prop / where_times_which / codon_blosum / pssm / frequency / zeroshot
baselines, + uniform. Per-virus recall@10% over the 9 forecast_noattach trees @2025-01-01,
then MACRO-averaged (equal weight per virus, CI propagated) exactly as build_fig4_j9 does.
Validates the baseline macro against ladder_fig4_j9.csv. Writes ladder_fig4_j9_1nt.csv
(+ pervirus_fig4_j9_1nt.csv) to antiGen-figures/data/multitree_noattach_j9/.
"""
import os, sys, glob, csv
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.eval_utils import (fit_substitution_baseline, positional_marginal_counts,
    frequency_baseline_scores, load_checkpoint)
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
from antiGen.models.codon_blosum_fitness_predictor import _codon_conditional
import count_baselines
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
from ensemble_logit import load_m
from score_joint49_sel9 import load_joint49_sel9
from score_HI_j9 import load_ensemble, boot_ci
from score_xformer_j9 import best_pt

R  = "/large_storage/hielab/aakarsh/multitree"
REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
J9 = f"{REPO}/antiGen-figures/data/multitree_noattach_j9"
OUT_LAD = f"{J9}/ladder_fig4_j9_1nt.csv"
OUT_PV  = f"{J9}/pervirus_fig4_j9_1nt.csv"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
XF_DIR = f"{R}/fc13_single_xformer_val_noattach/single_{{t}}_s{{s}}"
SEEDS = list(range(10)); Q = 0.10

# 1nt accessibility over the full alphabet (first 20 chars of ALPHA_SET == _AA20)
REACH = torch.zeros(E, E, dtype=torch.bool)
REACH[:20, :20] = torch.tensor(_codon_conditional(kappa=1.0, epsilon=0.0) > 0)


@torch.no_grad()
def emap(models, ab):
    lps = []
    for m in models:
        o = m(ab).float().squeeze(0)
        lps.append(torch.log_softmax(o.reshape(-1), dim=-1).reshape(o.shape).cpu())
    return torch.stack(lps, 0).mean(0)

import gc
def free(models):
    try: models.clear()          # drop the caller's references so the GPU mem is released
    except Exception:
        for m in list(models):
            try: del m
            except Exception: pass
    gc.collect()
    if dev.type == 'cuda': torch.cuda.empty_cache()

def se_of(lo, hi): return (hi - lo) / (2 * 1.96)
def macro_est_ci(rv):
    r = [x[0] for x in rv]; se = [se_of(x[1], x[2]) for x in rv]; n = len(r)
    est = sum(r) / n; s = (sum(e * e for e in se) ** 0.5) / n
    return est, max(0.0, est - 1.96 * s), min(1.0, est + 1.96 * s)


def build_tdata():
    """per tree: seen, count-state st, N, deterministic baseline maps, and per-subset
    entries carrying (aa, ev_full, ev_1nt, not1nt mask)."""
    td = {}
    for t in TREES:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        L = len(tr[0].aa_idx); st = count_baselines.fit(tr, L, E); N = st["N"]
        seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        sub_mat = torch.log_softmax(fit_substitution_baseline(tr, dev, ALPHA_SET)
                                    .score_matrix().detach().cpu(), dim=-1)
        pos_counts = positional_marginal_counts(tr, L, E)
        freq = frequency_baseline_scores(tr, L, E)
        subs = {}
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{t}@{D}_{suf}.txt"
            if not (os.path.exists(p) and os.path.getsize(p) > 0): continue
            te = load_examples(p, ALPHA_SET, None, source=t)
            ent = []
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                aa = ex.aa_idx
                not1 = ~REACH[aa]
                keep = REACH[aa[[p_ for p_, _ in ev]], [a_ for _, a_ in ev]].tolist()
                ev1 = [e for e, k in zip(ev, keep) if k]
                ent.append((aa, ev, ev1, not1))
            if ent: subs[sub] = ent
        td[t] = dict(L=L, st=st, N=N, seen=seen, sub_mat=sub_mat,
                     pos_counts=pos_counts, freq=freq, subs=subs)
    return td


def score_map(kind, aa, d, models=None, zsm=None, cb=None):
    ab = aa.unsqueeze(0)
    if kind == 'ensemble':
        return emap(models, ab.to(dev))
    if kind == 'zeroshot':
        with torch.no_grad(): return zsm(ab.to(dev)).float().squeeze(0).cpu()
    if kind == 'codon_blosum':
        with torch.no_grad(): return cb(ab.to(dev)).float().squeeze(0).cpu()
    if kind == 'where_times_which':
        return d['sub_mat'][aa] + torch.log(d['pos_counts'].clamp(min=1e-12)).view(d['L'], 1)
    if kind == 'frequency': return torch.log(d['freq'])
    return count_baselines.score_map(kind, d['st'], ab).squeeze(0)  # cpssm_dir/site_prop/pssm


# rung key -> (map-kind, needs-models)
RUNGS = [("joint", 'ensemble'), ("joint9", 'ensemble'), ("single", 'ensemble'),
         ("xformer", 'ensemble'),
         ("cpssm_dir", 'cpssm_dir'), ("site_prop", 'site_prop'),
         ("where_times_which", 'where_times_which'), ("codon_blosum", 'codon_blosum'),
         ("pssm", 'pssm'), ("frequency", 'frequency'), ("zeroshot", 'zeroshot')]


def score_rung_tree(rung, kind, t, td, models=None, zsm=None, cb=None):
    """-> {(subset, conv): per-event weight list} for this rung on tree t."""
    d = td[t]; seen = d['seen']; out = {}
    for sub, ent in d['subs'].items():
        # all -> full pool / no strike ; novel -> de-novo strike pool
        base_strike = None if sub == 'all' else seen
        wf_b, wf_1 = [], []
        for aa, ev, ev1, not1 in ent:
            mp = score_map(kind, aa, d, models=models, zsm=zsm, cb=cb)
            sm_b = base_strike
            sm_1 = not1 if base_strike is None else (base_strike | not1)
            wf_b += per_event_weights(mp, aa, ev,  seen_mask=sm_b, q=Q)
            wf_1 += per_event_weights(mp, aa, ev1, seen_mask=sm_1, q=Q)
        subkey = 'all' if sub == 'all' else 'novel'
        out[(subkey, 'baseline')] = wf_b
        out[(subkey, 'onent')]    = wf_1
    return out


def main():
    td = build_tdata()
    cb_by_L = {}
    def cb_for(L):
        if L not in cb_by_L:
            cb_by_L[L] = create_model(ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=L,
                alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                codon_blosum=CodonBlosumConfig(kappa=4.0)), dev).eval()
        return cb_by_L[L]

    # PV[(rung, tree, subset, conv)] = per-event weight np.array
    PV = {}
    def stash(rung, t, out):
        for (subset, conv), w in out.items():
            PV[(rung, t, subset, conv)] = np.asarray(w, np.float32)

    # --- antiGen ensemble arms (load once where possible, reuse across trees) ---
    print("loading joint49 sel9 (10 seeds)...", flush=True)
    j49 = [load_joint49_sel9(s)[0] for s in SEEDS]
    for t in TREES: stash("joint", t, score_rung_tree("joint", 'ensemble', t, td, models=j49)); print(f"  joint {t}", flush=True)
    free(j49)

    print("loading joint9 (10 seeds)...", flush=True)
    j9m, j9s = load_ensemble('joint9', None)
    for t in TREES: stash("joint9", t, score_rung_tree("joint9", 'ensemble', t, td, models=j9m)); print(f"  joint9 {t}", flush=True)
    free(j9m)

    for t in TREES:
        sm, ss = load_ensemble('single', t)
        stash("single", t, score_rung_tree("single", 'ensemble', t, td, models=sm)); free(sm)
        print(f"  single {t} (seeds={ss})", flush=True)

    for t in TREES:
        models = []
        for s in SEEDS:
            c = best_pt(XF_DIR.format(t=t, s=s))
            if c:
                try: models.append(load_m(c))
                except Exception: pass
        if not models: print(f"  [no xformer {t}]", flush=True); continue
        stash("xformer", t, score_rung_tree("xformer", 'ensemble', t, td, models=models)); free(models)
        print(f"  xformer {t}", flush=True)

    # --- deterministic baselines (load the zero-shot ESM-2 650M backbone now, after the
    #     heavy ensemble arms are freed, so it never shares the GPU with 10 resident models) ---
    zc = (glob.glob(f"{R}/fc13_single_esm650_p4_noattach/single_ncov_s0/*_epoch*.pt") or
          glob.glob(f"{R}/fc13_single_esm650_noattach/single_ncov_s0/*_best.pt") or
          glob.glob(f"{R}/fc13_single_esm650/single_h3n2_ha_s0/*_best.pt"))
    _, zcfg = load_checkpoint(zc[0], map_location='cpu'); zsm = create_model(zcfg, dev).eval()
    for rung, kind in RUNGS[4:]:
        for t in TREES:
            cb = cb_for(td[t]['L']) if kind == 'codon_blosum' else None
            stash(rung, t, score_rung_tree(rung, kind, t, td, zsm=zsm, cb=cb))
        print(f"  baseline {rung}", flush=True)

    # ---- per-virus recall + CI ----
    pv_rows = []
    for (rung, _k) in RUNGS:
        for t in TREES:
            for subset in ('all', 'novel'):
                for conv in ('baseline', 'onent'):
                    k = (rung, t, subset, conv)
                    if k not in PV: continue
                    v = PV[k]; lo, hi = boot_ci(v)
                    pv_rows.append([rung, t, subset, conv, len(v), f"{v.mean():.6f}", f"{lo:.6f}", f"{hi:.6f}"])
    with open(OUT_PV, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rung","virus","subset","convention","n_events","recall","ci_lo","ci_hi"])
        w.writerows(pv_rows)

    # ---- macro ladder (equal weight per virus, CI propagated) ----
    lad = []
    for subset in ('all', 'novel'):
        for conv in ('baseline', 'onent'):
            for (rung, _k) in RUNGS:
                rv = []
                for t in TREES:
                    k = (rung, t, subset, conv)
                    if k in PV and len(PV[k]):
                        v = PV[k]; lo, hi = boot_ci(v); rv.append((float(v.mean()), lo, hi))
                if not rv: continue
                est, lo, hi = macro_est_ci(rv)
                lad.append([rung, subset, conv, f"{est:.6f}", f"{lo:.6f}", f"{hi:.6f}", len(rv)])
            lad.append(["uniform", subset, conv, "0.100000", "0.100000", "0.100000", 0])
    with open(OUT_LAD, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rung","subset","convention","est","lo","hi","n_virus"])
        w.writerows(lad)
    print("\nwrote", OUT_LAD, "and", OUT_PV)

    # ---- validation: baseline macro vs published ladder_fig4_j9.csv ----
    pub = {}
    pf = f"{J9}/ladder_fig4_j9.csv"
    if os.path.exists(pf):
        for r in csv.DictReader(open(pf)):
            pub[(r['rung'], r['subset'])] = float(r['est'])
        print("\n=== VALIDATION: baseline macro vs published ladder_fig4_j9.csv ===")
        print(f"  {'rung':<20}{'subset':<7}{'mine':>9}{'pub':>9}{'Δ':>9}")
        for row in lad:
            if row[2] != 'baseline': continue
            key = (row[0], row[1]);
            if key in pub:
                mine = float(row[3]); print(f"  {row[0]:<20}{row[1]:<7}{mine:>9.4f}{pub[key]:>9.4f}{mine-pub[key]:>+9.4f}")


if __name__ == '__main__':
    main()
