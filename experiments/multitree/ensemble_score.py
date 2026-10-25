#!/usr/bin/env python
"""Seed-ensemble vs single-seed on the post-D test set (recall@10 + NLL, ALL + de novo).

Ensemble = geometric mean of the seeds' probability maps (average of log-probs), Ivan's
convention. Scores the ensemble AND each single checkpoint, micro-pooled over trees,
gap/stop masked.  Usage: ensemble_score.py --checkpoints c1 c2 c3 --trees ... --out j.json
"""
import os, sys, json, argparse
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
import torch
from torch.utils.data import DataLoader
from antiGen.load import load_examples
from antiGen.train import MutationDataset
from antiGen.eval_utils import load_checkpoint, compute_probs, compute_loss_full, compute_hit_rate
from antiGen.models.factory import create_model

FD="/home/ivan.specht/antiGen/experiments/multitree/data/forecast"; D="2025-01-01"
A="ACDEFGHIKLMNPQRSTVWY*-"; E=len(A); GAP,STOP=A.index('-'),A.index('*'); MASK=[GAP,STOP]
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_m(ck):
    cap,cfg=load_checkpoint(ck,map_location='cpu'); m=create_model(cfg,torch.device('cpu'))
    m.load_state_dict(cap['model_state_dict'],strict=False); del cap
    return m.to(dev).eval()

@torch.no_grad()
def score(models, trees):
    """models: list of model objs. Returns {sub: {recall10, nll}} micro-pooled (ensemble)."""
    out={}
    for sub,suf in [('all','test'),('novel','novel')]:
        hits=nmut=loss=0.0
        for t in trees:
            p=f"{FD}/examples_{t}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,A,None,source=t)
            if not te: continue
            L=len(te[0].aa_idx)
            for ab,tb in DataLoader(MutationDataset(te),batch_size=8,shuffle=False):
                ab=ab.to(dev); tb=tb.to(dev)
                tbb=tb.clone(); tbb[:,:,MASK]=0
                logps=[]
                for m in models:
                    pr=compute_probs(m(ab).float(), tbb, ab)     # (B,L,E) wt-masked, clamped
                    logps.append(torch.log(pr))
                logp=torch.stack(logps,0).mean(0)                 # geometric mean of probs
                # NLL (renormalized inside compute_loss_full)
                ens_probs=torch.exp(logp)
                bl,_=compute_loss_full(ens_probs, tbb); loss+=bl.item()
                # recall
                sc=logp.clone()
                for c in MASK: sc[:,:,c]=float('nan')
                h,_=compute_hit_rate(sc, tbb, ab, qtile=0.1, eval_range=list(range(L)))
                hits+=h; nmut+=tbb.sum().item()
        out[sub]={'recall10':hits/max(nmut,1), 'nll':loss/max(nmut,1), 'n':nmut}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoints',nargs='+',required=True)
    ap.add_argument('--trees',nargs='+',required=True); ap.add_argument('--out',default=None)
    a=ap.parse_args()
    ms=[load_m(c) for c in a.checkpoints]
    res={'singles':{}, 'ensemble':{}}
    for i,m in enumerate(ms):
        res['singles'][f"seed{i}"]=score([m], a.trees)
        print(f"single {i}: all r10={res['singles'][f'seed{i}']['all']['recall10']:.3f} "
              f"novel r10={res['singles'][f'seed{i}']['novel']['recall10']:.3f}",flush=True)
    res['ensemble']=score(ms, a.trees)
    print(f"ENSEMBLE: all r10={res['ensemble']['all']['recall10']:.3f} "
          f"novel r10={res['ensemble']['novel']['recall10']:.3f}",flush=True)
    if a.out: json.dump(res,open(a.out,'w'),indent=2); print("wrote",a.out)

if __name__=='__main__': main()
