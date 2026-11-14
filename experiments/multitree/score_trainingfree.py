#!/usr/bin/env python
"""Training-free substitution-matrix baselines on a forecast test set, recall@10, gap/stop
masked, same machinery as the ESM rungs. Macro + micro over the trees, all + de novo.

  blosum62              raw BLOSUM62 exchangeability (ModelType.BLOSUM62)
  codon_aware_blosum62  codon-corrected BLOSUM62, kappa=4 (ModelType.CODON_BLOSUM) -- the
                        "Codon aware BLOSUM62" baseline; every ingredient external, no fitting.
"""
import os, sys, json, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
import torch
from torch.utils.data import DataLoader
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
from antiGen.load import load_examples
from antiGen.train import MutationDataset
from antiGen.eval_utils import validate_per_source

A="ACDEFGHIKLMNPQRSTVWY*-"; D="2025-01-01"
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NAMES={'blosum62':'BLOSUM62','codon_aware_blosum62':'Codon aware BLOSUM62'}

def build(model_name, seq_len):
    alpha=AlphabetConfig(amino_acids=A)
    if model_name=='blosum62':
        cfg=ModelConfig(model_type=ModelType.BLOSUM62, seq_len=seq_len, alphabet=alpha)
    else:
        cfg=ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=seq_len, alphabet=alpha,
                        codon_blosum=CodonBlosumConfig(kappa=4.0))
    return create_model(cfg, dev).eval()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True)
    ap.add_argument('--trees',nargs='+',required=True); ap.add_argument('--out',default=None)
    a=ap.parse_args()
    res={}
    for sub,suf in [('all','test'),('novel','novel')]:
        res[sub]={}
        for t in a.trees:
            p=f"{a.data}/examples_{t}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,A,None,source=t)
            if not te: continue
            L=len(te[0].aa_idx)
            ld={t:DataLoader(MutationDataset(te),batch_size=16,shuffle=False)}
            res[sub][t]={}
            for m in NAMES:
                model=build(m,L)
                r=validate_per_source(model, ld, dev, qtiles=(0.1,), mask_gap_stop=True)['per_source'][t]
                res[sub][t][m]={'recall10':r['hit_rate']['0.1'],'n':r['n_mutations']}
                del model
    # macro/micro summary
    for sub in res:
        for m in NAMES:
            pv=[(res[sub][t][m]['recall10'],res[sub][t][m]['n']) for t in res[sub] if m in res[sub][t] and res[sub][t][m]['n']]
            if pv:
                print(f"{sub:6} {NAMES[m]:22} macro={statistics.mean(v for v,_ in pv):.3f} "
                      f"micro={sum(v*n for v,n in pv)/sum(n for _,n in pv):.3f}")
    if a.out: json.dump(res,open(a.out,'w'),indent=2); print("wrote",a.out)

if __name__=='__main__': main()
