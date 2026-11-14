#!/usr/bin/env python
"""Per-test-event: training count of that exact (pos,alt) substitution, and whether
partial-4 (joint) and Smoothed Counts (cpssm_dir) each caught it in the top-10%.

Lets us stratify recall@10 and the joint-minus-SmoothedCounts gap by how recurrent the
test substitution is in the TRAINING corpus. Attach test, 11 trees. Writes stratify_attach.npz
with parallel arrays {tree__sub__ncount, __joint, __smoothed}.
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
ATRAIN="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_attach"
ATEST="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_attachtest"
TREES=['h5nx','rsv_a','rsv_b','measles','denv1','denv2','denv3','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
OUT="/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/joint13_partial4_attach_recallsel_s0/stratify_attach.npz"

def main():
    out={}
    cks=[glob.glob(f"{R}/fc13_joint_650m/joint13_partial4_attach_recallsel_s{s}/*_epoch1.pt")[0] for s in (0,1,2)]
    ens=[load_esm(c) for c in cks]
    jm=ens[0]   # seed 0 = the single "joint" model used elsewhere
    for tree in TREES:
        tr=load_examples(f"{ATRAIN}/examples_{tree}@{D}_train.txt",ALPHA_SET,None,source=tree)
        L=len(tr[0].aa_idx)
        st=count_baselines.fit(tr,L,E)          # st["N"] = (L,E) training EVENT counts
        N=st["N"]
        for sub,suf in [('all','test'),('novel','novel')]:
            p=f"{ATEST}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,ALPHA_SET,None,source=tree)
            if not te: continue
            nc=[]; jw=[]; ew=[]; sw=[]
            for ex in te:
                ev=events_of(ex)
                if not ev: continue
                aa=ex.aa_idx; ab=aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    logits=[m(ab).float().squeeze(0).cpu() for m in ens]
                js=logits[0]
                es=torch.stack(logits,0).mean(0)       # logit-level ensemble of the 3 seeds
                ss=count_baselines.score_map('cpssm_dir',st,aa.unsqueeze(0)).squeeze(0)
                j_w=per_event_weights(js,aa,ev); e_w=per_event_weights(es,aa,ev); s_w=per_event_weights(ss,aa,ev)
                k=0
                for (pos,a) in ev:
                    if a in MASK: continue
                    nc.append(float(N[pos,a])); jw.append(j_w[k]); ew.append(e_w[k]); sw.append(s_w[k]); k+=1
            out[f"{tree}__{sub}__ncount"]=np.array(nc,dtype=np.float32)
            out[f"{tree}__{sub}__joint"]=np.array(jw,dtype=np.float32)
            out[f"{tree}__{sub}__ensemble"]=np.array(ew,dtype=np.float32)
            out[f"{tree}__{sub}__smoothed"]=np.array(sw,dtype=np.float32)
        print(f"done {tree}",flush=True)
    np.savez_compressed(OUT,**out); print("wrote",OUT)

if __name__=='__main__': main()
