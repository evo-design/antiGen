#!/usr/bin/env python
"""Per-epoch val-vs-test correspondence for a run with --save-every-epoch.

For each saved epoch checkpoint, scores the post-D test set (recall@10 + NLL, ALL + de novo,
micro-pooled over the trees, gap/stop masked) and pairs it with that epoch's VALIDATION
recall@10 + NLL read from <prefix>_val_metrics_by_epoch.json. Writes epochs.json.
"""
import os, sys, json, glob, re, argparse
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
import torch
from torch.utils.data import DataLoader
from antiGen.load import load_examples
from antiGen.train import MutationDataset
from antiGen.eval_utils import load_checkpoint, validate_per_source
from antiGen.models.factory import create_model

FD="/home/ivan.specht/antiGen/experiments/multitree/data/forecast"; D="2025-01-01"
A="ACDEFGHIKLMNPQRSTVWY*-"
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def micro_test(model, trees):
    out={}
    for sub,suf in [('all','test'),('novel','novel')]:
        num_r=den=num_l=0.0
        for t in trees:
            p=f"{FD}/examples_{t}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,A,None,source=t)
            if not te: continue
            ld={t:DataLoader(MutationDataset(te),batch_size=16,shuffle=False)}
            r=validate_per_source(model, ld, dev, qtiles=(0.1,), mask_gap_stop=True)['per_source'][t]
            n=r['n_mutations']; num_r+=r['hit_rate']['0.1']*n; num_l+=r['nll_per_mutation']*n; den+=n
        out[f"test_recall10_{sub}"]=num_r/den if den else float('nan')
        out[f"test_nll_{sub}"]=num_l/den if den else float('nan')
    return out

def val_from_log(run):
    f=glob.glob(run+"/*_val_metrics_by_epoch.json")
    if not f: return {}
    d=json.load(open(f[0])); out={}
    for r in d:
        ps=r['val']['per_source']; num_r=den=num_l=0.0
        for st in ps.values():
            n=st['n_mutations']; num_r+=st['hit_rate']['0.1']*n
            num_l+=st.get('nll_per_mutation_true', st['nll_per_mutation'])*n; den+=n
        out[r['epoch']]={'val_recall10':num_r/den if den else float('nan'),
                         'val_nll':num_l/den if den else float('nan')}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('run_dir')
    ap.add_argument('--trees',nargs='+',required=True); ap.add_argument('--out',default=None)
    a=ap.parse_args()
    vlog=val_from_log(a.run_dir)
    cks=sorted(glob.glob(a.run_dir+"/fitness_*_epoch*.pt"),
               key=lambda p:int(re.search(r'_epoch(\d+)\.pt$',p).group(1)))
    res={}
    for ck in cks:
        ep=int(re.search(r'_epoch(\d+)\.pt$',ck).group(1))
        cap,cfg=load_checkpoint(ck,map_location='cpu')
        m=create_model(cfg,torch.device('cpu')); m.load_state_dict(cap['model_state_dict'],strict=False)
        m=m.to(dev).eval(); del cap
        row=micro_test(m,a.trees); row.update(vlog.get(ep,{}))
        res[ep]=row
        print(f"epoch {ep}: val_r10={row.get('val_recall10',float('nan')):.3f} "
              f"test_r10_all={row['test_recall10_all']:.3f} test_r10_novel={row['test_recall10_novel']:.3f}",flush=True)
        del m
        if dev.type=='cuda': torch.cuda.empty_cache()
    if a.out: json.dump(res,open(a.out,'w'),indent=2); print("wrote",a.out)

if __name__=='__main__': main()
