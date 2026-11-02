#!/usr/bin/env python
"""Per-event fractional hit weights at q=10% for the Fig-5 bootstrap, NO-ATTACH regime.

Same rungs/machinery as score_perevent_attach.py, but:
  joint  = 5-seed logit-level ensemble of the lm-only no-attach models @ epoch 3
           (the antiGen rung the figure shows -- averaged raw logits, then ranked)
  single = single-virus ESM-650m lm-only, no-attach (per tree, val-selected _best.pt)
  count/substitution baselines fit on the NO-ATTACH training corpus
  test/novel from forecast_noattach, 13 no-attach trees.
Writes perevent_fig13_noattach.npz consumed by collect_fig13_noattach.py.
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
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
import count_baselines
from score_perevent import per_event_weights, events_of, load_esm, E, ALPHA_SET, D, dev

def build_codon_blosum(seq_len):
    """Training-free Codon aware BLOSUM62 (kappa=4) scorer, built at this tree's length."""
    cfg=ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=seq_len,
                    alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                    codon_blosum=CodonBlosumConfig(kappa=4.0))
    return create_model(cfg, dev).eval()

R="/large_storage/hielab/aakarsh/multitree"
NA="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
TREES=['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
OUT=f"{R}/fc13_joint_650m/joint13_lmonly_noattach_recallsel_s0/perevent_fig13_noattach.npz"

def main():
    out={}
    # antiGen rung = 5-seed logit ensemble @ epoch 3 (same ckpts as ensemble_logit.py)
    ens_cks=[glob.glob(f"{R}/fc13_joint_650m/joint13_lmonly_noattach_recallsel_s{s}/*_epoch3.pt")[0] for s in (0,1,2,3,4)]
    ens=[load_esm(c) for c in ens_cks]
    for tree in TREES:
        tr=load_examples(f"{NA}/examples_{tree}@{D}_train.txt",ALPHA_SET,None,source=tree)
        L=len(tr[0].aa_idx)
        # training-root count matrix -> de-novo "seen" mask (== eval.py's pssm_root>0)
        seen_counts=torch.zeros(L,E)
        for ex in tr: seen_counts+=ex.mat
        seen_mask=seen_counts>0
        st=count_baselines.fit(tr,L,E)
        fitted=fit_substitution_baseline(tr,dev,ALPHA_SET)
        sub_mat=torch.log_softmax(fitted.score_matrix().detach().cpu(),dim=-1)
        pos_counts=positional_marginal_counts(tr,L,E)
        freq=frequency_baseline_scores(tr,L,E); wonly=where_only_baseline_scores(tr,L,E)
        cb=build_codon_blosum(L)                 # training-free Codon aware BLOSUM62
        s_ck=[p for p in glob.glob(f"{R}/fc13_single_esm650_noattach/single_{tree}_s0/*_best.pt") if "_best_" not in p][-1]
        sm=load_esm(s_ck)
        for sub,suf in [('all','test'),('novel','novel')]:
            p=f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,ALPHA_SET,None,source=tree)
            if not te: continue
            rows={r:[] for r in ['joint','single','site_prop','cpssm_dir','pssm','frequency','where_times_which','substitution','codon_blosum','where_only']}
            for ex in te:
                ev=events_of(ex)
                if not ev: continue
                aa=ex.aa_idx; ab=aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js=torch.stack([m(ab).float().squeeze(0).cpu() for m in ens],0).mean(0)  # 5-seed logit avg
                    ss=sm(ab).float().squeeze(0).cpu()
                    cbs=cb(ab).float().squeeze(0).cpu()                                      # Codon aware BLOSUM62
                ab1=aa.unsqueeze(0)
                maps={'joint':js,'single':ss,
                  'site_prop':count_baselines.score_map('site_prop',st,ab1).squeeze(0),
                  'cpssm_dir':count_baselines.score_map('cpssm_dir',st,ab1).squeeze(0),
                  'pssm':count_baselines.score_map('pssm',st,ab1).squeeze(0),
                  'frequency':torch.log(freq),'where_only':torch.log(wonly),
                  'where_times_which':sub_mat[aa]+torch.log(pos_counts.clamp(min=1e-12)).view(L,1),
                  'substitution':sub_mat[aa],'codon_blosum':cbs}
                sm_arg = seen_mask if sub=='novel' else None   # de-novo strikes seen cells
                for r,mp in maps.items(): rows[r]+=per_event_weights(mp,aa,ev,seen_mask=sm_arg)
            out[f"{tree}|{sub}"]={r:np.array(v,dtype=np.float32) for r,v in rows.items()}
        del sm
        if dev.type=='cuda': torch.cuda.empty_cache()
        print(f"done {tree}",flush=True)
    np.savez_compressed(OUT, **{k.replace("|","__")+"__"+r:v for k,d in out.items() for r,v in d.items()})
    print("wrote",OUT)

if __name__=='__main__': main()
