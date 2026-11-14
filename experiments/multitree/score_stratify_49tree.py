#!/usr/bin/env python
"""Per-test-event: training count of that exact (pos,alt), and whether the 49-tree
partial-4 snapshot antiGen (single ckpt, epoch 0) and Smoothed Counts (cpssm_dir) each
caught it at q=10%.

NO-ATTACH regime, 13 eval trees. The model is the 49-snapshot partial-4 finetune
(13 eval + 36 extra core viruses) at epoch 0 -- the 'antiGen (49 trees)' bar in Fig 5.
Training counts are per eval-tree (the coordinate frame is per protein, so the 36 extra
viruses cannot contribute to a given eval tree's (pos,alt) cell) -- i.e. the SAME counts
Smoothed Counts sees. Writes stratify_49tree.npz with {tree__sub__ncount,__model,__smoothed}.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
import torch
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, per_event_weights, E, ALPHA_SET, D, dev, MASK

R="/large_storage/hielab/aakarsh/multitree"
NA="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
TREES=['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
CKPT=glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*multitree49*_epoch0.pt")[0]
OUT=f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/stratify_49tree.npz"

def main():
    print("ckpt:",CKPT,flush=True)
    model=load_esm(CKPT)
    out={}
    for tree in TREES:
        tr=load_examples(f"{NA}/examples_{tree}@{D}_train.txt",ALPHA_SET,None,source=tree)
        L=len(tr[0].aa_idx)
        st=count_baselines.fit(tr,L,E)
        N=st["N"]                                # (L,E) training EVENT counts
        for sub,suf in [('all','test'),('novel','novel')]:
            p=f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,ALPHA_SET,None,source=tree)
            if not te: continue
            nc=[]; ew=[]; sw=[]
            for ex in te:
                ev=events_of(ex)
                if not ev: continue
                aa=ex.aa_idx; ab=aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    es=model(ab).float().squeeze(0).cpu()
                ss=count_baselines.score_map('cpssm_dir',st,aa.unsqueeze(0)).squeeze(0)
                e_w=per_event_weights(es,aa,ev); s_w=per_event_weights(ss,aa,ev)
                k=0
                for (pos,a) in ev:
                    if a in MASK: continue
                    nc.append(float(N[pos,a])); ew.append(e_w[k]); sw.append(s_w[k]); k+=1
            out[f"{tree}__{sub}__ncount"]=np.array(nc,dtype=np.float32)
            out[f"{tree}__{sub}__model"]=np.array(ew,dtype=np.float32)
            out[f"{tree}__{sub}__smoothed"]=np.array(sw,dtype=np.float32)
        print(f"done {tree}",flush=True)
    np.savez_compressed(OUT,**out); print("wrote",OUT)

if __name__=='__main__': main()
