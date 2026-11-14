#!/usr/bin/env python
"""Per-virus LEARNED GATE between antiGen (49-tree partial-4, epoch 0) and Smoothed
Counts (cpssm_dir), NO-ATTACH, 13 eval trees.

Reuses the gate_search machinery (Block / score_of / fast_recall / canonical) but
substitutes the two maps being mixed:
    lm = 49-tree partial-4 epoch-0 log-prob map   (was the model ensemble)
    lw = cpssm_dir "Smoothed Counts" log-prob map (was where x which)
Both are pushed through the SAME logp_model normalizer, so a convex blend
(1-w)*lm + w*lw and its ranking are on one comparable scale.

LEARNED PER VIRUS. For each tree the gate parameters theta are chosen on that tree's
OWN pre-D validation split (random 10% held out of the pre-cutoff train), maximizing
pre-D recall@10 (all + de novo), then applied UNSEEN to that tree's post-D test. No
tree's gate sees any other tree's data and no gate sees post-D. The 13 per-virus gated
test blocks are pooled for the headline. For context we also report:
  - model alone / Smoothed alone (the two inputs),
  - a single GLOBAL gate fit across all trees' pre-D (one theta for all 13),
  - the per-genotype ORACLE (pick the better predictor per query -- an UPPER BOUND,
    never a result), to say how much complementarity a gate could in principle reach.

Families tried per virus (increasing capacity), all fit on pre-D only:
  blend      w global                      (1 param)
  gated      w depends on seen/unseen      (2 params)
  cell_gate  w = sigmoid(theta . cellfeat) (6 params)  <- the pre-registered gate
  geno_gate  w = sigmoid(theta . genofeat) (5 params)
"""
from __future__ import annotations
import os, sys, io, json, contextlib, glob
import numpy as np
import torch

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

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
D = "2025-01-01"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
CKPT = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/"
                 f"*multitree49*_epoch0.pt")[0]
FAMILIES = ['blend', 'gated', 'cell_gate', 'geno_gate']
PICK = 'cell_gate'        # the pre-registered learned gate
OUT = "experiments/multitree/out/gate_49tree_smoothed.json"
rng = np.random.default_rng(0)


def smoothed_logp(st, aa, wt):
    """cpssm_dir log-score -> the SAME normalized-log-prob space as the model map."""
    ls = count_baselines.score_map('cpssm_dir', st, aa).cpu().numpy()   # (B,L,E) log q
    return logp_model(ls, wt)                                           # renormalize+clamp


def block_for(model, examples, train_ex, L):
    """A gate_search Block with lm=model map, lw=Smoothed Counts map, both normalized."""
    aa = torch.stack([e.aa_idx for e in examples]).to(dev)
    tg = torch.stack([e.mat for e in examples]).to(dev)
    lo, _, wt = collect(model, examples, dev)
    lm = torch.tensor(logp_model(lo, wt), device=dev, dtype=torch.float32)
    st = count_baselines.fit(train_ex, L, E)
    # count_baselines keeps its stats on CPU; score_map indexes M[aa], so aa must
    # match that device.
    lw = torch.tensor(smoothed_logp(st, aa.cpu(), wt), device=dev, dtype=torch.float32)
    seq, ev = counts(train_ex, L)
    seen = torch.tensor(seen_cells(train_ex, L), device=dev)
    return Block(lm, lw, seen, seq, ev, aa, tg)


def fast_both(bk, kind, th):
    t = torch.tensor(th, device=dev, dtype=torch.float32)
    hw, tw, nv = bk.fast_recall(score_of(bk, kind, t))
    a = float(hw.sum()) / max(float(tw.sum()), 1e-9)
    n = float(hw[nv].sum()) / max(float(tw[nv].sum()), 1e-9)
    return a, n


def best_theta(sel_blocks, kind):
    """theta maximizing pre-D (all + de novo) recall over the given selection blocks."""
    TH = draws(kind, rng)
    best, bth = -1.0, None
    for th in TH:
        a = n = wa = wn = 0.0
        for bk in sel_blocks:
            t = torch.tensor(th, device=dev, dtype=torch.float32)
            hw, tw, nv = bk.fast_recall(score_of(bk, kind, t))
            a += float(hw.sum()); wa += float(tw.sum())
            n += float(hw[nv].sum()); wn += float(tw[nv].sum())
        s = a / max(wa, 1e-9) + n / max(wn, 1e-9)
        if s > best:
            best, bth = s, th
    return bth


from forecast_hit_summary import per_example_hits


def canon_one(bk, kind, th, mask_stop):
    """Per-example fractional hits for one scoring block, all + de novo."""
    out = {}
    t = torch.tensor(th, device=dev, dtype=torch.float32)
    sc = score_of(bk, kind, t).reshape(bk.shape)
    nov = (~bk.seen3).unsqueeze(0).to(bk.tg.dtype)
    for sub, tgt in (('all', bk.tg), ('novel', bk.tg * nov)):
        if float(tgt.sum()) == 0:
            continue
        per = per_example_hits(sc, tgt, bk.aa, mask_stop).cpu().numpy()
        n = tgt.sum((1, 2)).cpu().numpy()
        out[sub] = (per, n)
    return out


def pool(parts, sub):
    xs = [parts[(t, sub)] for t in TREES if (t, sub) in parts]
    return boot_pooled(xs) if xs else (float('nan'),) * 3


def contrast(parts, base, sub):
    xs = [(parts[(t, sub)][0] - base[(t, sub)][0], parts[(t, sub)][1])
          for t in TREES if (t, sub) in parts and (t, sub) in base]
    return boot_pooled(xs) if xs else (float('nan'),) * 3


def main():
    print("ckpt:", CKPT, flush=True)
    model = load_esm(CKPT)

    sel = {}     # tree -> [val selection blocks]
    sco = {}     # tree -> post-D scoring block
    for tree in TREES:
        with contextlib.redirect_stdout(io.StringIO()):
            pre = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", AL, None, source=tree)
            post = load_examples(f"{NA}/examples_{tree}@{D}_test.txt", AL, None, source=tree)
        if not post:
            continue
        L = len(pre[0].aa_idx)
        blocks = []
        for s in (0, 1, 2):                         # 3 random val splits -> steadier fit
            with contextlib.redirect_stdout(io.StringIO()):
                trb, vab = split_examples({tree: pre}, 'random', 0.1, s)
            if vab[tree]:
                blocks.append(block_for(model, vab[tree], trb[tree], L))
        sel[tree] = blocks
        sco[tree] = block_for(model, post, pre, L)
        print(f"  built {tree}", flush=True)

    trees = [t for t in TREES if t in sco]

    # ---- references: the two inputs, canonical ----
    ref_model, ref_smoothed = {}, {}
    for t in trees:
        for sub, d in canon_one(sco[t], 'blend', [0.0], True).items():
            ref_model[(t, sub)] = d                 # w=0 -> model alone (ESM: mask stop)
        for sub, d in canon_one(sco[t], 'blend', [1.0], False).items():
            ref_smoothed[(t, sub)] = d              # w=1 -> smoothed alone (baseline)

    print("\n== inputs (pooled post-D, canonical) ==")
    for nm, ref in (('antiGen 49-tree', ref_model), ('Smoothed Counts', ref_smoothed)):
        a, n = pool(ref, 'all')[0], pool(ref, 'novel')[0]
        print(f"  {nm:16s} ALL {a:.3f}   DE NOVO {n:.3f}")

    # ---- per-virus learned gates: fit per tree on its own pre-D, score its post-D ----
    results = {}
    for kind in FAMILIES:
        parts, thetas = {}, {}
        for t in trees:
            th = best_theta(sel[t], kind)
            thetas[t] = [float(x) for x in th]
            for sub, d in canon_one(sco[t], kind, th, True).items():
                parts[(t, sub)] = d
        results[kind] = dict(parts=parts, thetas=thetas)

    # ---- one GLOBAL gate (all trees share theta), for the cell_gate family ----
    glob_parts = {}
    th_glob = best_theta([b for t in trees for b in sel[t]], PICK)
    for t in trees:
        for sub, d in canon_one(sco[t], PICK, th_glob, True).items():
            glob_parts[(t, sub)] = d

    # ---- per-genotype ORACLE ceiling (upper bound, NOT a result) ----
    orc = {}
    for t in trees:
        bk = sco[t]
        hm, tw, nv = bk.fast_recall(bk.lm)
        hw, _, _ = bk.fast_recall(bk.lw)
        # per-example: take whichever input scores higher on that genotype
        pe_m = torch.zeros(bk.B, device=dev).index_add_(0, bk.bi, hm)
        pe_w = torch.zeros(bk.B, device=dev).index_add_(0, bk.bi, hw)
        pick_w = (pe_w > pe_m)[bk.bi]
        best = torch.where(pick_w, hw, hm)
        for sub, m in (('all', torch.ones_like(bk.tnovel)), ('novel', bk.tnovel)):
            if float(tw[m].sum()) == 0:
                continue
            # realize as a per-example vector aligned to canon's (per, n) shape is
            # awkward; report the pooled scalar directly from fractional hits
            orc[(t, sub)] = (best[m].cpu().numpy(), tw[m].cpu().numpy())

    def pool_raw(parts, sub):
        xs = [parts[(t, sub)] for t in trees if (t, sub) in parts]
        return boot_pooled(xs) if xs else (float('nan'),) * 3

    # ---- report ----
    print("\n== learned gates (pooled post-D recall@10, canonical; delta vs the "
          "better input) ==")
    print(f'{"gate":22s} {"ALL":>7s} {"vs model":>20s} {"DE NOVO":>8s} {"vs model":>20s}')
    for kind in FAMILIES:
        tag = f"per-virus {kind}" + ("  *" if kind == PICK else "")
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
    print(f'{"per-genotype ORACLE":22s} {oa[0]:7.3f} {"(ceiling)":>20s} '
          f'{on[0]:8.3f} {"(ceiling)":>20s}')

    # also delta of the pre-registered gate vs Smoothed Counts
    pk = results[PICK]['parts']
    sa = contrast(pk, ref_smoothed, 'all'); sn = contrast(pk, ref_smoothed, 'novel')
    print(f"\n== PRE-REGISTERED gate = per-virus '{PICK}' ==")
    print(f"  vs Smoothed Counts: ALL {sa[0]:+.3f} [{sa[1]:+.3f},{sa[2]:+.3f}]   "
          f"DE NOVO {sn[0]:+.3f} [{sn[1]:+.3f},{sn[2]:+.3f}]")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dump = {
        'ckpt': CKPT,
        'inputs': {
            'antigen_49tree': {'all': pool(ref_model, 'all')[0],
                               'novel': pool(ref_model, 'novel')[0]},
            'smoothed': {'all': pool(ref_smoothed, 'all')[0],
                         'novel': pool(ref_smoothed, 'novel')[0]},
        },
        'gates': {k: {'all': pool(results[k]['parts'], 'all'),
                      'novel': pool(results[k]['parts'], 'novel'),
                      'vs_model_all': contrast(results[k]['parts'], ref_model, 'all'),
                      'vs_model_novel': contrast(results[k]['parts'], ref_model, 'novel'),
                      'thetas': results[k]['thetas']} for k in FAMILIES},
        'global_pick': {'all': ga, 'novel': gn, 'theta': [float(x) for x in th_glob]},
        'oracle': {'all': oa, 'novel': on},
        'pick': PICK,
    }
    with open(OUT, 'w') as fh:
        json.dump(dump, fh, indent=1, default=float)
    print("\nwrote", OUT)


if __name__ == '__main__':
    main()
