#!/usr/bin/env python
"""Mutation-event count per dataset for the 49-snapshot training pool, written as a CSV
for the R figure. Events are counted the SAME way as datasize.csv / plot_multi_vs_single:
sum(len(events_of(ex))) over the training examples. Two groups:
  'Train + test'  the 13 eval-panel trees (have a post-cutoff test set)       -> highlight
  'Train only'    the 36 extra core-virus snapshots (training data only)
"""
import os, sys, io, csv, json, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
from antiGen.load import load_examples
from score_perevent import events_of, ALPHA_SET, D

NA="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
MAN="/home/aakarsh/projects/antiGen/experiments/multitree/data/trees_core_surface.json"
OUT="/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/dataset_event_counts.csv"
EVAL=['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
      'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']

def count(path, source):
    with contextlib.redirect_stdout(io.StringIO()):
        ex=load_examples(path, ALPHA_SET, None, source=source)
    return sum(len(events_of(e)) for e in ex)

rows=[]
# 13 eval = Train + test
for t in EVAL:
    p=f"{NA}/examples_{t}@{D}_train.txt"
    rows.append((t, count(p,t), "Train + test"))
# 36 core snapshots = Train only
man=json.load(open(MAN))
nsnap=0
for key,v in man.items():
    if v.get("route")!="snapshot":   # the snapshot pool is route==snapshot (36); prune panels excluded
        continue
    rows.append((key, count(v["path"],key), "Train only")); nsnap+=1

rows.sort(key=lambda r:-r[1])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["key","events","group"])
    for k,e,g in rows: w.writerow([k,e,g])
print(f"wrote {OUT}  (eval={len(EVAL)}, core_snapshot={nsnap}, total={len(rows)})")
print("top5:", rows[:5]); print("bot5:", rows[-5:])
