#!/usr/bin/env python
"""Logit-level ensemble of several checkpoints on a forecast test set, recall@10, gap/stop
masked. For each example the per-seed raw logits are AVERAGED, then ranked (so this is a
true logit-level ensemble, not a geometric mean of per-model probabilities). Scores the
ensemble and each single seed; macro + micro over trees, all + de novo.

Usage: ensemble_logit.py --data <dir> --checkpoints c1 c2 c3 --trees ... [--out j.json]
"""
import os, sys, json, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
import torch
from torch.utils.data import DataLoader
from antiGen.load import load_examples
from antiGen.train import MutationDataset
from antiGen.eval_utils import load_checkpoint, compute_hit_rate
from antiGen.models.factory import create_model

A="ACDEFGHIKLMNPQRSTVWY*-"; D="2025-01-01"; Q=0.1
GAP,STOP=A.index('-'),A.index('*'); MASK=[GAP,STOP]
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_m(ck):
    cap,cfg=load_checkpoint(ck,map_location='cpu'); m=create_model(cfg,torch.device('cpu'))
    m.load_state_dict(cap['model_state_dict'],strict=False); del cap
    return m.to(dev).eval()

@torch.no_grad()
def recall(models, te, L, combine='logit', seen_mask=None):
    """models: list; returns recall@10 over this tree's events.

    combine: how the seeds are pooled before ranking --
      'logit'   mean of raw logits (mean-field; the default/committed behaviour)
      'logprob' mean of per-position log-softmax (geometric mean of probs = product-of-experts)
      'prob'    log of the mean of per-position softmax (arithmetic mixture-of-experts)
    For a single model all three give the SAME global ranking; they differ only for an
    ensemble, and only because recall@10 ranks cells across positions (see per-position
    normalizer discussion)."""
    ld=DataLoader(MutationDataset(te),batch_size=16,shuffle=False)
    hits=0.0; n=0.0
    for ab,tb in ld:
        ab=ab.to(dev); tb=tb.to(dev)
        z=torch.stack([m(ab).float() for m in models],0)               # (S,B,L,E)
        S,Bsz,L2,E2=z.shape
        if combine=='logit':     sc=z.mean(0)
        elif combine=='logprob': sc=torch.log_softmax(z,dim=-1).mean(0)                 # per-position norm
        elif combine=='prob':    sc=torch.log(torch.softmax(z,dim=-1).mean(0).clamp_min(1e-12))
        elif combine=='logprob_global':   # normalize over the whole L*E grid, then average
            zg=torch.log_softmax(z.reshape(S,Bsz,L2*E2),dim=-1).reshape(S,Bsz,L2,E2)
            sc=zg.mean(0)
        elif combine=='prob_global':
            pg=torch.softmax(z.reshape(S,Bsz,L2*E2),dim=-1).reshape(S,Bsz,L2,E2)
            sc=torch.log(pg.mean(0).clamp_min(1e-12))
        else: raise ValueError(combine)
        sc=sc.clone()
        for c in MASK: sc[:,:,c]=float('nan')
        # is-multitree de-novo convention (NOTES §67): strike training-seen cells
        # (pssm_root>0) from the candidate pool before the top-q cut. seen_mask is a
        # (L,E) bool; passed only for the novel subset. None => full pool (unchanged).
        if seen_mask is not None:
            sc[:, seen_mask.to(sc.device)] = float('nan')
        tbm=tb.clone(); tbm[:,:,MASK]=0
        h,_=compute_hit_rate(sc, tbm, ab, qtile=Q, eval_range=list(range(L)))
        hits+=h; n+=tbm.sum().item()
    return hits, n

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True)
    ap.add_argument('--checkpoints',nargs='+',required=True)
    ap.add_argument('--trees',nargs='+',required=True); ap.add_argument('--out',default=None)
    ap.add_argument('--combine',default='logit',choices=['logit','logprob','prob','logprob_global','prob_global'])
    ap.add_argument('--denovo-strike',action='store_true',
                    help='NOTES §67 convention: on the novel subset, strike training-seen '
                         'cells (sum(ex.mat)>0) from the candidate pool before the top-q cut')
    a=ap.parse_args()
    models=[load_m(c) for c in a.checkpoints]
    arms={'ensemble':models, **{f'seed{i}':[m] for i,m in enumerate(models)}}
    res={}
    for sub,suf in [('all','test'),('novel','novel')]:
        res[sub]={}
        for t in a.trees:
            p=f"{a.data}/examples_{t}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,A,None,source=t)
            if not te: continue
            L=len(te[0].aa_idx); res[sub][t]={}
            sm=None
            if sub=='novel' and a.denovo_strike:
                tr=load_examples(f"{a.data}/examples_{t}@{D}_train.txt",A,None,source=t)
                sm=sum(ex.mat for ex in tr)>0
            for name,ms in arms.items():
                h,n=recall(ms,te,L,combine=a.combine,seen_mask=sm)
                res[sub][t][name]={'recall10':h/max(n,1),'n':n}
    def agg(sub,name):
        pv=[(res[sub][t][name]['recall10'],res[sub][t][name]['n']) for t in res[sub] if name in res[sub][t] and res[sub][t][name]['n']]
        if not pv: return None,None
        return statistics.mean(v for v,_ in pv), sum(v*n for v,n in pv)/sum(n for _,n in pv)
    for sub in res:
        for name in arms:
            ma,mi=agg(sub,name)
            print(f"{sub:6} {name:10} macro={ma:.3f} micro={mi:.3f}")
    if a.out: json.dump(res,open(a.out,'w'),indent=2); print("wrote",a.out)

if __name__=='__main__': main()
