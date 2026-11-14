#!/usr/bin/env python
"""Per-event fractional hit weights at q=10% for the Fig-5 bootstrap, ATTACH test set.

Same rungs and machinery as score_perevent.py, but:
  joint  = partial-4 attach @ epoch 1 (one checkpoint for all trees)
  single = single-virus ESM-650m lm-only, attached (per tree)
  count/substitution baselines fit on the ATTACH training corpus
  test/novel from forecast_attachtest, 11 attach trees (no ncov/denv4/hmpv_b_g).
Writes perevent_fig13_attach.npz consumed by collect_fig13_attach.py.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
import torch
from antiGen.load import load_examples
from antiGen.eval_utils import (fit_substitution_baseline, positional_marginal_counts,
    frequency_baseline_scores, where_only_baseline_scores)
import count_baselines
from score_perevent import per_event_weights, events_of, load_esm, E, ALPHA_SET, D, dev

R="/large_storage/hielab/aakarsh/multitree"
ATRAIN="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_attach"
ATEST="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_attachtest"
TREES=['h5nx','rsv_a','rsv_b','measles','denv1','denv2','denv3','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
OUT=f"{R}/fc13_joint_650m/joint13_partial4_attach_recallsel_s0/perevent_fig13_attach.npz"

def main():
    out={}
    j_ck=glob.glob(f"{R}/fc13_joint_650m/joint13_partial4_attach_recallsel_s0/*_epoch1.pt")[0]
    jm=load_esm(j_ck)
    for tree in TREES:
        tr=load_examples(f"{ATRAIN}/examples_{tree}@{D}_train.txt",ALPHA_SET,None,source=tree)
        L=len(tr[0].aa_idx)
        st=count_baselines.fit(tr,L,E)
        fitted=fit_substitution_baseline(tr,dev,ALPHA_SET)
        sub_mat=torch.log_softmax(fitted.score_matrix().detach().cpu(),dim=-1)
        pos_counts=positional_marginal_counts(tr,L,E)
        freq=frequency_baseline_scores(tr,L,E); wonly=where_only_baseline_scores(tr,L,E)
        s_ck=[p for p in glob.glob(f"{R}/fc13_single_esm650/single_{tree}_s0/*_best.pt") if "_best_" not in p][-1]
        sm=load_esm(s_ck)
        for sub,suf in [('all','test'),('novel','novel')]:
            p=f"{ATEST}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,ALPHA_SET,None,source=tree)
            if not te: continue
            rows={r:[] for r in ['joint','single','site_prop','cpssm_dir','pssm','frequency','where_times_which','substitution','where_only']}
            for ex in te:
                ev=events_of(ex)
                if not ev: continue
                aa=ex.aa_idx; ab=aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js=jm(ab).float().squeeze(0).cpu(); ss=sm(ab).float().squeeze(0).cpu()
                ab1=aa.unsqueeze(0)
                maps={'joint':js,'single':ss,
                  'site_prop':count_baselines.score_map('site_prop',st,ab1).squeeze(0),
                  'cpssm_dir':count_baselines.score_map('cpssm_dir',st,ab1).squeeze(0),
                  'pssm':count_baselines.score_map('pssm',st,ab1).squeeze(0),
                  'frequency':torch.log(freq),'where_only':torch.log(wonly),
                  'where_times_which':sub_mat[aa]+torch.log(pos_counts.clamp(min=1e-12)).view(L,1),
                  'substitution':sub_mat[aa]}
                for r,mp in maps.items(): rows[r]+=per_event_weights(mp,aa,ev)
            out[f"{tree}|{sub}"]={r:np.array(v,dtype=np.float32) for r,v in rows.items()}
        del sm
        if dev.type=='cuda': torch.cuda.empty_cache()
        print(f"done {tree}",flush=True)
    np.savez_compressed(OUT, **{k.replace("|","__")+"__"+r:v for k,d in out.items() for r,v in d.items()})
    print("wrote",OUT)

if __name__=='__main__': main()
