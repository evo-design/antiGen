#!/usr/bin/env python
"""Search a wide family of model/`where x which` combination rules.

`gate_ceiling.py` showed an oracle per-genotype gate reaches 0.662 de novo recall
against the 3-seed ensemble's 0.566 -- +0.096 [+0.051, +0.155]. That headroom is
what this script tries to reach with rules that use only pre-D information.

Rules, in increasing capacity. `lm`/`lw` are normalized log-probability maps.

  novelty bonus   lm + c * unseen                          1 param
                  A pure budget reallocation: monotone inside the seen and unseen
                  groups, so it only moves cells ACROSS the seen/unseen boundary.
                  This is NOTES s11's alpha upweight, but measured on recall and
                  with c selected on pre-D rather than on the scored set.
  blend           (1-w) lm + w lw                          1 param
  gated blend     w depends on seen/unseen                 2 params
  bonus + gated   both of the above                        3 params
  rank blend      the same on normalized ranks             1-2 params
  cell gate       w = sigmoid(theta . cell features)       6 params
  genotype gate   w = sigmoid(theta . genotype features)   5 params
  reranker        score = theta . [lm, lw, features]       7 params

Everything is fit by direct random search on pre-D recall at top 10%, NOT on NLL.
The lesson from s11 is that an NLL-optimal blend need not move ranking at all, so
the search optimizes the reported metric directly.

**Transfer diagnostic.** Before any of this can work, the thing a gate keys on has to
behave the same way pre-D and post-D. It demonstrably does not for the raw win rate:
the model LOSES to the baseline on the pre-D de novo split (0.445 vs 0.454) and wins
post-D (0.566 vs 0.503). The script therefore reports, for every candidate feature,
its AUC for predicting "the model out-ranked the baseline on this genotype" in both
regimes. A feature whose AUC flips sign between them cannot support a gate, and
knowing that is worth more than any single rule's score.

Search uses a fast exact top-10% count (no sort); the rules that survive are then
re-scored through the canonical `per_example_hits` with fractional tie-breaking, and
only those canonical numbers are reported.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu                              # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

from antiGen.eval_utils import (fit_substitution_baseline,                 # noqa: E402
                                positional_marginal_counts,
                                WhereTimesWhichPredictor)
from antiGen.load import load_examples                                     # noqa: E402
from antiGen.train import split_examples                                   # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import boot_pooled, seen_cells                 # noqa: E402
from forecast_hit_summary import per_example_hits                          # noqa: E402
from fuse_forecast import D, DATA, PANEL, load_joint, logp_model           # noqa: E402

STOP = AL.index('*')
Q = 0.1
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def counts(train_ex, L):
    """Per-cell evidence from the training half: sequences, events, positional."""
    seq = np.zeros((L, E), dtype=np.float32)
    ev = np.zeros((L, E), dtype=np.float32)
    idx = np.arange(L)
    for ex in train_ex:
        seq[idx, ex.aa_idx.numpy()] += 1.0
        for pos, alt in (ex.mat > 0).nonzero().tolist():
            ev[pos, alt] += 1.0
    return seq, ev


class Block:
    """One (tree, phase) evaluation unit, with everything a rule needs precomputed."""

    def __init__(self, lm, lw, seen, seq, ev, aa, tg):
        B = lm.shape[0]
        self.B, self.shape = B, lm.shape
        valid = torch.ones_like(lm, dtype=torch.bool)
        valid.scatter_(2, aa.unsqueeze(-1), False)
        valid[:, :, STOP] = False
        self.valid = valid.reshape(B, -1)
        self.nvalid = self.valid.sum(1).float()
        self.lm = lm.reshape(B, -1)
        self.lw = lw.reshape(B, -1)
        self.rm = self._ranks(self.lm)
        self.rw = self._ranks(self.lw)

        s = seen.reshape(-1).float()
        pos_mut = torch.tensor(ev.sum(1), device=lm.device)
        f = [torch.ones_like(s), s,
             torch.log1p(torch.tensor(seq.reshape(-1), device=lm.device)) / 5.0,
             torch.log1p(torch.tensor(ev.reshape(-1), device=lm.device)),
             (torch.log1p(pos_mut).unsqueeze(1).expand(-1, E).reshape(-1))]
        self.cellf = torch.stack(f, 1).unsqueeze(0).expand(B, -1, -1)   # (B, LE, 5)
        self.unseen = (1.0 - s).unsqueeze(0).expand(B, -1)

        # genotype features: all computable from the query alone
        p = torch.softmax(torch.where(self.valid, self.lm,
                                      torch.full_like(self.lm, -1e9)), 1)
        ent = -(p * torch.log(p.clamp(min=1e-30))).sum(1)
        k = (Q * self.nvalid).long().clamp(min=1)
        top = self.lm.argsort(1, descending=True)
        seen_frac = torch.stack([s[top[b, :k[b]]].mean() for b in range(B)])
        zm = (self.rm - self.rm.mean(1, keepdim=True))
        zw = (self.rw - self.rw.mean(1, keepdim=True))
        agree = ((zm * zw).sum(1) / (zm.norm(dim=1) * zw.norm(dim=1)).clamp(min=1e-9))
        self.genof = torch.stack([torch.ones_like(ent), ent / 10.0, seen_frac,
                                  agree, self.nvalid.log() / 10.0], 1)   # (B, 5)

        # targets
        tf = tg.reshape(B, -1)
        bi, ci = tf.nonzero(as_tuple=True)
        self.bi, self.ci = bi, ci
        self.tw = tf[bi, ci]                       # a cell can carry weight > 1
        self.tnovel = (1.0 - s)[ci] > 0.5
        self.aa, self.tg, self.seen3 = aa, tg, seen

    @staticmethod
    def _ranks(x):
        order = x.argsort(1)
        r = torch.empty_like(x)
        ar = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        r.scatter_(1, order, ar.expand(x.shape[0], -1))
        return r / max(x.shape[1] - 1, 1)

    def fast_recall(self, score, chunk=4096):
        """(all, novel) recall at top 10%, exact, without sorting.

        A target is a hit when fewer than Q * n_valid valid cells outrank it. Counting
        is a broadcast comparison, which is far cheaper than the quantile the canonical
        routine takes -- and the search needs tens of thousands of evaluations.
        """
        s = torch.where(self.valid, score, torch.full_like(score, -float('inf')))
        hits = torch.empty(len(self.bi), device=s.device, dtype=torch.bool)
        for i in range(0, len(self.bi), chunk):
            b, c = self.bi[i:i + chunk], self.ci[i:i + chunk]
            ts = s[b, c].unsqueeze(1)
            cnt = (s[b] > ts).sum(1).float()
            hits[i:i + chunk] = cnt < Q * self.nvalid[b]
        hw = hits.float() * self.tw
        return hw, self.tw, self.tnovel


def build(models, wxw_seeds, examples, train_ex, L, ens):
    aa = torch.stack([e.aa_idx for e in examples]).to(dev)
    tg = torch.stack([e.mat for e in examples]).to(dev)
    pm = positional_marginal_counts(train_ex, L, E)
    lws = []
    for s in wxw_seeds:
        fit = fit_substitution_baseline(train_ex, dev, AL, seed=s)
        fac = WhereTimesWhichPredictor(fit.config, pm, fit).to(dev)
        wl, _, wt = collect(fac, examples, dev)
        lws.append(torch.tensor(logp_model(wl, wt), device=dev, dtype=torch.float32))
        del fit, fac
    lw = torch.stack(lws).mean(0)
    lms = [torch.tensor(logp_model(collect(models[s], examples, dev)[0], wt),
                        device=dev, dtype=torch.float32) for s in ens]
    lm = torch.stack(lms).mean(0)
    seq, ev = counts(train_ex, L)
    seen = torch.tensor(seen_cells(train_ex, L), device=dev)
    return Block(lm, lw, seen, seq, ev, aa, tg)


# ---------------------------------------------------------------- rule families
def score_of(bk, kind, th):
    """Score map for a rule family at parameter vector `th`."""
    lm, lw, un = bk.lm, bk.lw, bk.unseen
    if kind == 'bonus':
        return lm + th[0] * un
    if kind == 'blend':
        return (1 - th[0]) * lm + th[0] * lw
    if kind == 'rank_blend':
        return (1 - th[0]) * bk.rm + th[0] * bk.rw
    if kind == 'gated':
        w = th[0] + (th[1] - th[0]) * un
        return (1 - w) * lm + w * lw
    if kind == 'rank_gated':
        w = th[0] + (th[1] - th[0]) * un
        return (1 - w) * bk.rm + w * bk.rw
    if kind == 'bonus_gated':
        w = th[0] + (th[1] - th[0]) * un
        return (1 - w) * lm + w * lw + th[2] * un
    if kind == 'cell_gate':
        w = torch.sigmoid((bk.cellf * th[:5]).sum(-1))
        return (1 - w) * lm + w * lw + th[5] * un
    if kind == 'geno_gate':
        w = torch.sigmoid((bk.genof * th[:5]).sum(-1)).unsqueeze(1)
        return (1 - w) * lm + w * lw
    if kind == 'reranker':
        return (th[0] * lm + th[1] * lw + th[2] * bk.rm + th[3] * bk.rw
                + (bk.cellf[:, :, 1:] * th[4:8]).sum(-1))
    raise ValueError(kind)


SPEC = {                         # kind -> (n params, sampling scale, n draws)
    'bonus':       (1, 'bonus',  41),
    'blend':       (1, 'unit',   41),
    'rank_blend':  (1, 'unit',   41),
    'gated':       (2, 'unit',   441),
    'rank_gated':  (2, 'unit',   441),
    'bonus_gated': (3, 'mixed',  3000),
    'cell_gate':   (6, 'theta',  6000),
    'geno_gate':   (5, 'theta',  6000),
    'reranker':    (8, 'theta',  6000),
}


def draws(kind, rng):
    n, scale, m = SPEC[kind]
    if scale == 'bonus':
        return np.linspace(0, 20, m).reshape(-1, 1)
    if scale == 'unit':
        if n == 1:
            return np.linspace(0, 1, m).reshape(-1, 1)
        g = np.linspace(0, 1, int(round(np.sqrt(m))))
        return np.array([[a, b] for a in g for b in g])
    if scale == 'mixed':
        return np.column_stack([rng.uniform(0, 1, m), rng.uniform(0, 1, m),
                                rng.uniform(0, 20, m)])
    th = rng.normal(0, 2.0, (m, n))
    if kind == 'cell_gate':
        th[:, 5] = rng.uniform(0, 20, m)
    if kind == 'reranker':
        th[:, 0] = np.abs(th[:, 0]) + 0.2
    return th


def evaluate(blocks, kind, th):
    """Pooled (all, novel) recall of one parameter vector across blocks."""
    ha = hn = na = nn = 0.0
    t = torch.tensor(th, device=dev, dtype=torch.float32)
    for bk in blocks:
        hw, tw, nv = bk.fast_recall(score_of(bk, kind, t))
        ha += float(hw.sum()); na += float(tw.sum())
        hn += float(hw[nv].sum()); nn += float(tw[nv].sum())
    return ha / max(na, 1e-9), hn / max(nn, 1e-9)


def canonical(blocks, kind, th, mask_stop=True):
    """Re-score through the library routine, with fractional tie-breaking.

    `mask_stop` follows OUTPUTS_LEGITIMATE_STOP: True for anything carrying ESM-2
    output, False for a pure baseline. Getting this wrong on the wxw reference
    inflates every contrast against it by about 0.02.
    """
    per, nvec = {}, {}
    t = torch.tensor(th, device=dev, dtype=torch.float32)
    for i, bk in enumerate(blocks):
        sc = score_of(bk, kind, t).reshape(bk.shape)
        nov = (~bk.seen3).unsqueeze(0).to(bk.tg.dtype)
        for sub, tgt in (('all', bk.tg), ('novel', bk.tg * nov)):
            if float(tgt.sum()) == 0:
                continue
            per[(i, sub)] = per_example_hits(sc, tgt, bk.aa,
                                             mask_stop).cpu().numpy()
            nvec[(i, sub)] = tgt.sum((1, 2)).cpu().numpy()
    return per, nvec


def transfer_diagnostic(sel, sco):
    """Does anything predict WHICH predictor wins, consistently in both regimes?

    For each genotype feature, the AUC for 'the model out-ranked the baseline on
    this genotype'. A gate can only work if a feature's AUC is on the same side of
    0.5 in both columns.
    """
    def rows(blocks):
        F, y, w = [], [], []
        for bk in blocks:
            hm, tw, nv = bk.fast_recall(bk.lm)
            hw, _, _ = bk.fast_recall(bk.lw)
            per_ex_m = torch.zeros(bk.B, device=dev).index_add_(0, bk.bi, hm)
            per_ex_w = torch.zeros(bk.B, device=dev).index_add_(0, bk.bi, hw)
            n_ex = torch.zeros(bk.B, device=dev).index_add_(0, bk.bi, tw)
            m = n_ex > 0
            F.append(bk.genof[m].cpu().numpy())
            y.append((per_ex_m[m] > per_ex_w[m]).float().cpu().numpy())
            w.append(n_ex[m].cpu().numpy())
        return np.concatenate(F), np.concatenate(y), np.concatenate(w)

    names = ['(bias)', 'model entropy', 'seen frac of top-10%',
             'model/wxw agreement', 'log n_valid']
    out = {}
    for tag, blocks in (('pre-D', sel), ('post-D', sco)):
        F, y, _ = rows(blocks)
        out[tag] = {}
        for j, nm in enumerate(names):
            if j == 0:
                continue
            x = F[:, j]
            pos, neg = x[y > 0.5], x[y < 0.5]
            if len(pos) == 0 or len(neg) == 0:
                out[tag][nm] = float('nan'); continue
            # AUC via the rank-sum identity
            r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
            out[tag][nm] = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                                 / (len(pos) * len(neg)))
        out[tag]['_n'] = float(len(y))
        out[tag]['_base'] = float(y.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--ens', default='0,1,2', help='model seeds in the ensemble')
    ap.add_argument('--wxw-seeds', default='0,1,2')
    ap.add_argument('--root', default=None, help='override checkpoint root')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ens = [int(x) for x in args.ens.split(',')]
    wxs = [int(x) for x in args.wxw_seeds.split(',')]
    if args.root:
        import fuse_forecast
        fuse_forecast.FORECAST_PT = args.root

    models = {s: load_joint(s) for s in ens}
    sel, sco = [], []
    for tree in PANEL:
        with contextlib.redirect_stdout(io.StringIO()):
            pre = load_examples(f'{DATA}/examples_{tree}@{D}_train.txt', AL, None,
                                source=tree)
            post = load_examples(f'{DATA}/examples_{tree}@{D}_test.txt', AL, None,
                                 source=tree)
        if not post:
            continue
        L = len(pre[0].aa_idx)
        # Selection blocks: one per model seed, each on the split THAT seed held out.
        # (With --split-seed fixed across the ensemble this collapses to one clean
        # block; with the s0-s7 series it is the best available approximation and the
        # model map used is that seed's own, not the ensemble's.)
        for s in ens:
            with contextlib.redirect_stdout(io.StringIO()):
                trb, vab = split_examples({tree: pre}, 'random', 0.1, s)
            if vab[tree]:
                sel.append(build(models, wxs, vab[tree], trb[tree], L, [s]))
        sco.append(build(models, wxs, post, pre, L, ens))
        print(f'  {tree}', flush=True)

    print('\n== transfer diagnostic: AUC for "the model out-ranked wxw here" ==')
    diag = transfer_diagnostic(sel, sco)
    names = [k for k in diag['pre-D'] if not k.startswith('_')]
    print(f'{"feature":26s} {"pre-D":>8s} {"post-D":>8s}   (0.5 = no signal)')
    for nm in names:
        a, b = diag['pre-D'][nm], diag['post-D'][nm]
        flag = '' if (a - 0.5) * (b - 0.5) > 0 else '   <- flips'
        print(f'{nm:26s} {a:8.3f} {b:8.3f}{flag}')
    print(f'  model wins on {diag["pre-D"]["_base"]:.3f} of pre-D genotypes '
          f'(n={diag["pre-D"]["_n"]:.0f}), {diag["post-D"]["_base"]:.3f} of post-D '
          f'(n={diag["post-D"]["_n"]:.0f})')

    # reference rungs, canonical
    ref = {}
    for nm, kind, th, ms in (('model ens', 'blend', [0.0], True),
                             ('wxw ens', 'blend', [1.0], False)):
        ref[nm] = canonical(sco, kind, th, ms)

    def pooled(per, nvec, sub):
        ks = [k for k in per if k[1] == sub]
        return boot_pooled([(per[k], nvec[k]) for k in ks])[0]

    def contrast(per, nvec, base, sub):
        ks = [k for k in per if k[1] == sub and k in base[0]]
        return boot_pooled([(per[k] - base[0][k], nvec[k]) for k in ks])

    def pooled0(nm, sub):
        p, n = ref[nm]
        return pooled(p, n, sub)
    print('\n== reference (canonical stop convention) ==')
    for nm in ('model ens', 'wxw ens'):
        print(f'  {nm:12s} ALL {pooled0(nm, "all"):.3f}   '
              f'DE NOVO {pooled0(nm, "novel"):.3f}')

    rng = np.random.default_rng(0)
    results = []
    print('\n== search: each family, params chosen on pre-D, reported post-D '
          '(canonical) ==')
    print(f'{"family":14s} {"objective":9s} {"preD all":>9s} {"preD nov":>9s} '
          f'{"ALL":>7s} {"vs wxw":>21s} {"DE NOVO":>8s} {"vs wxw":>21s}')
    for kind in SPEC:
        TH = draws(kind, rng)
        ev = np.array([evaluate(sel, kind, th) for th in TH])
        for obj, key in (('all', 0), ('novel', 1), ('both', None)):
            score = ev[:, key] if key is not None else ev.sum(1)
            th = TH[int(np.argmax(score))]
            per, nvec = canonical(sco, kind, th)
            line = f'{kind:14s} {obj:9s} {ev[int(np.argmax(score)), 0]:9.3f} ' \
                   f'{ev[int(np.argmax(score)), 1]:9.3f}'
            rec = {'family': kind, 'objective': obj,
                   'theta': [float(x) for x in th],
                   'preD_all': float(ev[int(np.argmax(score)), 0]),
                   'preD_novel': float(ev[int(np.argmax(score)), 1]),
                   'preD_sum': float(ev[int(np.argmax(score))].sum())}
            for sub in ('all', 'novel'):
                v = pooled(per, nvec, sub)
                d, lo, hi = contrast(per, nvec, ref['wxw ens'], sub)
                line += f' {v:7.3f} {d:+6.3f} [{lo:+.3f},{hi:+.3f}]'
                rec[sub] = dict(recall=v, vs_wxw=d, lo=lo, hi=hi)
                dm, lom, him = contrast(per, nvec, ref['model ens'], sub)
                rec[sub]['vs_model'] = dm; rec[sub]['vs_model_lo'] = lom
                rec[sub]['vs_model_hi'] = him
            print(line)
            results.append(rec)

    # ---- the pre-registered rule: family AND parameters chosen by one pre-D
    # criterion (all + de novo recall), so nothing in the choice saw post-D data.
    # Reporting the best of 27 (family x objective) cells AFTER looking at post-D
    # would be a garden of forking paths; this is the honest single number.
    cand = [r for r in results if r['objective'] == 'both']
    pick = max(cand, key=lambda r: r['preD_sum'])
    print(f"\n== PRE-REGISTERED rule: '{pick['family']}', chosen on pre-D "
          f"(all + de novo), theta={[round(x, 3) for x in pick['theta']]} ==")
    for sub in ('all', 'novel'):
        v = pick[sub]
        print(f"  {sub:6s} recall {v['recall']:.3f}   vs wxw {v['vs_wxw']:+.3f} "
              f"[{v['lo']:+.3f},{v['hi']:+.3f}]   vs model ens "
              f"{v['vs_model']:+.3f} [{v['vs_model_lo']:+.3f},{v['vs_model_hi']:+.3f}]")

    print(f'\n{"family":14s} {"objective":9s} {"ALL vs model ens":>23s} '
          f'{"DE NOVO vs model ens":>23s}')
    for r in results:
        print(f'{r["family"]:14s} {r["objective"]:9s} '
              f'{r["all"]["vs_model"]:+8.3f} [{r["all"]["vs_model_lo"]:+.3f},'
              f'{r["all"]["vs_model_hi"]:+.3f}] '
              f'{r["novel"]["vs_model"]:+8.3f} [{r["novel"]["vs_model_lo"]:+.3f},'
              f'{r["novel"]["vs_model_hi"]:+.3f}]')

    print('\n== POST-D ORACLE per family -- upper bounds, NOT results ==')
    for kind in SPEC:
        TH = draws(kind, rng)
        ev = np.array([evaluate(sco, kind, th) for th in TH])
        print(f'  {kind:14s} ALL {ev[:, 0].max():.3f}   '
              f'DE NOVO {ev[:, 1].max():.3f}')

    with open(f'{args.out}/gate_search.json', 'w') as fh:
        json.dump({'results': results, 'transfer': diag}, fh, indent=1)


if __name__ == '__main__':
    main()
