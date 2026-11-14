#!/usr/bin/env python
"""Per-virus LEARNED GATE between the FROM-SCRATCH antiGen transformer (frozen) and antiGen ESM
(ESM-2 35M finetune), on the ATTACH per-virus-cutoff data. Dengue + RSV first ("to see what the
results look like").

Direct adaptation of gate_49tree_smoothed.py: same gate_search machinery (Block / score_of /
best_theta / canon), but the model map `lm` comes from the per-virus scratch transformer
(fig4_<tree>_vf02/xformer_s0 best.pt, frozen) instead of the 49-tree model, and `lw` is the ESM-2 35M finetune model map. The 'gated' family (w differs for seen vs unseen cells) is the hard router.

Per tree: gate theta fit on that tree's OWN pre-cutoff val split (3 random 10% holdouts),
maximizing pre-cutoff recall@10 (all + de novo), applied UNSEEN to post-cutoff test. Reports
the two inputs alone, each gate family, a global gate, and the per-genotype oracle ceiling.
"""
from __future__ import annotations
import os, sys, io, json, contextlib, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import count_baselines
from antiGen.load import load_examples
from antiGen.train import split_examples
from calibrate_forecast import collect, AL, E
from collect_figure_metrics import logp_model, seen_cells, boot_pooled
from gate_search import Block, score_of, SPEC, draws, counts
from score_perevent import load_esm
from forecast_hit_summary import per_example_hits

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R = "/large_storage/hielab/aakarsh/multitree"
DST = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
CUT = {'denv2': '2019-10-01', 'rsv_a': '2023-10-01'}
TREES = ['denv2', 'rsv_a']
CKPT_A_OF = {t: [p for p in glob.glob(f"{R}/fig4_{t}_vf02/xformer_s0/*_best.pt")
                 if "_best_" not in os.path.basename(p)][0] for t in TREES}
CKPT_B_OF = {t: [p for p in glob.glob(f"{R}/fig4_{t}_vf02/esm35m_ft_s0/*_best.pt")
                 if "_best_" not in os.path.basename(p)][0] for t in TREES}
FAMILIES = ['blend', 'gated', 'cell_gate', 'geno_gate']
PICK = 'gated'           # the seen/unseen hard router
OUT = "experiments/multitree/out/gate_xfmr_esm.json"
rng = np.random.default_rng(0)

def block_for(model_a, model_b, examples, train_ex, L):
    aa = torch.stack([e.aa_idx for e in examples]).to(dev)
    tg = torch.stack([e.mat for e in examples]).to(dev)
    lo_a, _, wt = collect(model_a, examples, dev)
    lm = torch.tensor(logp_model(lo_a, wt), device=dev, dtype=torch.float32)
    lo_b, _, wt2 = collect(model_b, examples, dev)
    lw = torch.tensor(logp_model(lo_b, wt2), device=dev, dtype=torch.float32)
    seq, ev = counts(train_ex, L)
    seen = torch.tensor(seen_cells(train_ex, L), device=dev)
    return Block(lm, lw, seen, seq, ev, aa, tg)

def best_theta(sel_blocks, kind):
    TH = draws(kind, rng); best, bth = -1.0, None
    for th in TH:
        a = n = wa = wn = 0.0
        for bk in sel_blocks:
            t = torch.tensor(th, device=dev, dtype=torch.float32)
            hw, tw, nv = bk.fast_recall(score_of(bk, kind, t))
            a += float(hw.sum()); wa += float(tw.sum())
            n += float(hw[nv].sum()); wn += float(tw[nv].sum())
        s = a / max(wa, 1e-9) + n / max(wn, 1e-9)
        if s > best: best, bth = s, th
    return bth

def canon_one(bk, kind, th, mask_stop):
    out = {}; t = torch.tensor(th, device=dev, dtype=torch.float32)
    sc = score_of(bk, kind, t).reshape(bk.shape)
    nov = (~bk.seen3).unsqueeze(0).to(bk.tg.dtype)
    for sub, tgt in (('all', bk.tg), ('novel', bk.tg * nov)):
        if float(tgt.sum()) == 0: continue
        per = per_example_hits(sc, tgt, bk.aa, mask_stop).cpu().numpy()
        n = tgt.sum((1, 2)).cpu().numpy(); out[sub] = (per, n)
    return out

def pool(parts, sub):
    xs = [parts[(t, sub)] for t in TREES if (t, sub) in parts]
    return boot_pooled(xs) if xs else (float('nan'),) * 3
def contrast(parts, base, sub):
    xs = [(parts[(t, sub)][0] - base[(t, sub)][0], parts[(t, sub)][1])
          for t in TREES if (t, sub) in parts and (t, sub) in base]
    return boot_pooled(xs) if xs else (float('nan'),) * 3

def main():
    print("A(scratch):", {t: os.path.basename(CKPT_A_OF[t]) for t in TREES})
    print("B(ESM):", {t: os.path.basename(CKPT_B_OF[t]) for t in TREES}, flush=True)
    sel, sco = {}, {}
    for tree in TREES:
        ma = load_esm(CKPT_A_OF[tree]); mb = load_esm(CKPT_B_OF[tree]); cut = CUT[tree]
        with contextlib.redirect_stdout(io.StringIO()):
            pre = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", AL, None, source=tree)
            post = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", AL, None, source=tree)
        if not post: continue
        L = len(pre[0].aa_idx); blocks = []
        for s in (0, 1, 2):
            with contextlib.redirect_stdout(io.StringIO()):
                trb, vab = split_examples({tree: pre}, 'random', 0.1, s)
            if vab[tree]: blocks.append(block_for(ma, mb, vab[tree], trb[tree], L))
        sel[tree] = blocks
        sco[tree] = block_for(ma, mb, post, pre, L)
        print(f"  built {tree}", flush=True)
    trees = [t for t in TREES if t in sco]

    ref_model, ref_smoothed = {}, {}
    for t in trees:
        for sub, d in canon_one(sco[t], 'blend', [0.0], True).items():  ref_model[(t, sub)] = d
        for sub, d in canon_one(sco[t], 'blend', [1.0], False).items(): ref_smoothed[(t, sub)] = d
    print("\n== inputs (pooled post-cutoff, canonical) ==")
    for nm, ref in (('antiGen (scratch)', ref_model), ('antiGen (ESM-2)', ref_smoothed)):
        print(f"  {nm:18s} ALL {pool(ref,'all')[0]:.3f}   DE NOVO {pool(ref,'novel')[0]:.3f}")

    results = {}
    for kind in FAMILIES:
        parts, thetas = {}, {}
        for t in trees:
            th = best_theta(sel[t], kind); thetas[t] = [float(x) for x in th]
            for sub, d in canon_one(sco[t], kind, th, True).items(): parts[(t, sub)] = d
        results[kind] = dict(parts=parts, thetas=thetas)
    th_glob = best_theta([b for t in trees for b in sel[t]], PICK)
    glob_parts = {}
    for t in trees:
        for sub, d in canon_one(sco[t], PICK, th_glob, True).items(): glob_parts[(t, sub)] = d

    orc = {}
    for t in trees:
        bk = sco[t]
        hm, tw, nv = bk.fast_recall(bk.lm); hw, _, _ = bk.fast_recall(bk.lw)
        pe_m = torch.zeros(bk.B, device=dev).index_add_(0, bk.bi, hm)
        pe_w = torch.zeros(bk.B, device=dev).index_add_(0, bk.bi, hw)
        best = torch.where((pe_w > pe_m)[bk.bi], hw, hm)
        for sub, m in (('all', torch.ones_like(bk.tnovel)), ('novel', bk.tnovel)):
            if float(tw[m].sum()) == 0: continue
            orc[(t, sub)] = (best[m].cpu().numpy(), tw[m].cpu().numpy())
    def pool_raw(parts, sub):
        xs = [parts[(t, sub)] for t in trees if (t, sub) in parts]
        return boot_pooled(xs) if xs else (float('nan'),) * 3

    print("\n== learned gates (pooled post-cutoff recall@10, canonical; delta vs scratch model) ==")
    print(f'{"gate":22s} {"ALL":>7s} {"vs model":>20s} {"DE NOVO":>8s} {"vs model":>20s}')
    for kind in FAMILIES:
        tag = f"per-virus {kind}" + ("  * (router)" if kind == PICK else "")
        parts = results[kind]['parts']
        a = pool(parts, 'all'); n = pool(parts, 'novel')
        da = contrast(parts, ref_model, 'all'); dn = contrast(parts, ref_model, 'novel')
        print(f'{tag:22s} {a[0]:7.3f} {da[0]:+6.3f} [{da[1]:+.3f},{da[2]:+.3f}] '
              f'{n[0]:8.3f} {dn[0]:+6.3f} [{dn[1]:+.3f},{dn[2]:+.3f}]')
    ga = pool(glob_parts, 'all'); gn = pool(glob_parts, 'novel')
    gda = contrast(glob_parts, ref_model, 'all'); gdn = contrast(glob_parts, ref_model, 'novel')
    print(f'{"GLOBAL "+PICK:22s} {ga[0]:7.3f} {gda[0]:+6.3f} [{gda[1]:+.3f},{gda[2]:+.3f}] '
          f'{gn[0]:8.3f} {gdn[0]:+6.3f} [{gdn[1]:+.3f},{gdn[2]:+.3f}]')
    oa = pool_raw(orc, 'all'); on = pool_raw(orc, 'novel')
    print(f'{"per-genotype ORACLE":22s} {oa[0]:7.3f} {"(ceiling)":>20s} {on[0]:8.3f} {"(ceiling)":>20s}')
    pk = results[PICK]['parts']
    sa = contrast(pk, ref_smoothed, 'all'); sn = contrast(pk, ref_smoothed, 'novel')
    print(f"\n== router ('{PICK}') vs Smoothed Counts: ALL {sa[0]:+.3f} [{sa[1]:+.3f},{sa[2]:+.3f}]   "
          f"DE NOVO {sn[0]:+.3f} [{sn[1]:+.3f},{sn[2]:+.3f}]")
    print("per-virus router thetas (w_seen, w_unseen):", results[PICK]['thetas'])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({'ckpts': CKPT_OF,
               'inputs': {'scratch': {'all': pool(ref_model,'all')[0], 'novel': pool(ref_model,'novel')[0]},
                          'esm': {'all': pool(ref_smoothed,'all')[0], 'novel': pool(ref_smoothed,'novel')[0]}},
               'gates': {k: {'all': pool(results[k]['parts'],'all'), 'novel': pool(results[k]['parts'],'novel'),
                             'thetas': results[k]['thetas']} for k in FAMILIES},
               'oracle': {'all': oa, 'novel': on}, 'pick': PICK},
              open(OUT, 'w'), indent=1, default=float)
    print("\nwrote", OUT)

if __name__ == '__main__':
    main()
