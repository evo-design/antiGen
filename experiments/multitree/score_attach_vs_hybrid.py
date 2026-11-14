#!/usr/bin/env python
"""Head-to-head: from-scratch antiGen transformer trained PER-VIRUS WITH ATTACHMENT
(@2025-01-01) vs the ESM-2 49-tree NO-ATTACH hybrid (joint49-sel9), on h3n2_ha / denv2 /
rsv_a. Both arms scored on the SHARED forecast_noattach test, same metric kernel
(score_perevent.per_event_weights, q=10%), MeanLogProb-global ensembles (emap).

Three subsets, so the de-novo leak is made explicit (the attach arm trained on ~20x more
events, so "novel-relative-to-no-attach-train" overcounts what it has truly never seen):
  all        -- forecast_noattach test, full pool (no strike).  The clean comparison.
  novel_na   -- forecast_noattach novel targets, strike = NO-ATTACH train seen mask (the
                standard A7 / section-67 de-novo convention the hybrid figure uses). The
                attach arm gets "leak credit" here.
  novel_dc   -- forecast_attach decontaminated novel targets, strike = ATTACH train seen
                mask (the superset). Fair de-novo: only cells the attach training never saw.

Arms are scored on IDENTICAL event streams (same file, same events_of order) so per-event
weights align for a PAIRED bootstrap (attach minus hybrid). Writes attach_vs_hybrid_*.csv
+ attach_vs_hybrid.npz under multitree_noattach_j9.

joint49 is virus-independent (same 10-seed set for every virus) -> loaded once. The attach
arm is per-virus (own 3-seed set) -> loaded per virus. joint49 checkpoint selection is its
standard 9-eval-virus macro val NLL (load_joint49_sel9); the attach arm uses per-virus val
NLL over its own saved epochs.
"""
import os, sys, glob, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
from ensemble_logit import load_m
from score_HI_j9 import _intact, epoch_pt, emap, boot_ci, free, nll_ranked_epochs
from score_joint49_sel9 import load_joint49_sel9

R = "/large_storage/hielab/aakarsh/multitree"
REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
AT = f"{HERE}/data/forecast_attach"
DD = f"{REPO}/antiGen-figures/data/multitree_noattach_j9"
TREES = ['h3n2_ha', 'denv2', 'rsv_a']
Q = 0.10
XF_SEEDS = [0, 1, 2]; J49_SEEDS = list(range(10))
XFDIR = f"{R}/fc13_single_xformer_attach/single_{{t}}_s{{s}}"
SUBSETS = ['all', 'novel_na', 'novel_dc']       # (subset -> (target file dir+suffix, strike mask))
NBOOT = 2000


def seen_mask(train_path, L):
    tr = load_examples(train_path, ALPHA_SET, None, source='x')
    m = torch.zeros(L, E)
    for ex in tr: m += ex.mat
    return m > 0


def align_mask(m, L):
    """Match a seen-mask to a subset's example length. The attach novel files for
    rsv_a/rsv_b/measles carry one extra TRAILING stop position vs their train (known
    trailing-stop inconsistency), so pad the end with False (a stop column, masked
    anyway) or trim. Keeps positions 0..L-1 aligned frame-for-frame."""
    if m is None or m.shape[0] == L: return m
    if m.shape[0] < L:
        return torch.cat([m, torch.zeros(L - m.shape[0], m.shape[1], dtype=m.dtype)], 0)
    return m[:L]


def load_xf_seed(t, s):
    """Lowest-val-NLL intact+loadable epoch for the per-virus attach xformer (seed s)."""
    d = XFDIR.format(t=t, s=s)
    # posthoc per-virus val-NLL selection; the from-scratch xformer ckpts are ~51MB, so
    # DO NOT use the 650M _intact (2GB) guard -- use a small-file size check instead.
    for ep in nll_ranked_epochs(d):
        p = epoch_pt(d, ep)
        if p and os.path.getsize(p) > 5_000_000:
            try: return load_m(p)
            except Exception: continue
    # fallback: training-time best.pt (the score_xformer_j9 convention)
    for p in glob.glob(f"{d}/*_best.pt"):
        if "_best_" not in os.path.basename(p) and os.path.getsize(p) > 5_000_000:
            try: return load_m(p)
            except Exception: continue
    raise RuntimeError(f"no loadable xfattach ckpt {t} s{s}")


def build_tdata():
    """Per virus: for each subset, the (aa_idx, events) entries and the strike mask."""
    td = {}
    for t in TREES:
        te0 = load_examples(f"{NA}/examples_{t}@{D}_test.txt", ALPHA_SET, None, source=t)
        L = len(te0[0].aa_idx)
        m_na = seen_mask(f"{NA}/examples_{t}@{D}_train.txt", L)
        m_at = seen_mask(f"{AT}/examples_{t}@{D}_train.txt", L)
        # ALL three subsets are scored on the SAME non-attach forecast tree (forecast_noattach).
        # Only the candidate-pool strike (and, for novel_dc, an extra target filter) changes:
        #   all       -- every test event, full pool.
        #   novel_na   -- no-attach-novel targets, strike no-attach-seen cells (standard §67).
        #   novel_dc   -- the SAME non-attach test stream, but keep only targets the ATTACH
        #                 train never saw AND strike attach-seen cells (leak-free de novo).
        # spec: subset -> (target file, pool-strike mask, keep-filter mask). An event is kept
        # only where keep[pos,aa] is False (i.e. that cell is unseen by the attach train).
        spec = {
            'all':      (f"{NA}/examples_{t}@{D}_test.txt",  None, None),
            'novel_na': (f"{NA}/examples_{t}@{D}_novel.txt", m_na, None),
            'novel_dc': (f"{NA}/examples_{t}@{D}_novel.txt", m_at, m_at),
        }
        subs = {}
        for sub, (path, strike, keep) in spec.items():
            if not (os.path.exists(path) and os.path.getsize(path) > 0): continue
            te = load_examples(path, ALPHA_SET, None, source=t)
            ent = []
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                aa = ex.aa_idx
                if len(aa) > L: aa = aa[:L]; ev = [(p, a) for (p, a) in ev if p < L]
                if keep is not None: ev = [(p, a) for (p, a) in ev if not bool(keep[p, a])]
                if ev: ent.append((aa, ev))
            if ent: subs[sub] = (ent, align_mask(strike, L))
        # leak: fraction of no-attach-novel target cells that the ATTACH train has seen
        leak_n = leak_hit = 0
        if 'novel_na' in subs:
            for aa, ev in subs['novel_na'][0]:
                for (pos, a) in ev:
                    leak_n += 1; leak_hit += int(bool(m_at[pos, a]))
        td[t] = (L, subs, (leak_n, leak_hit))
    return td


def stream_score(seed_loaders, items):
    """MeanLogProb-global ensemble with ONE model resident on the GPU at a time.
    seed_loaders: list of zero-arg callables returning a model (may raise to skip a seed).
    items: list of (key, aa_idx, events, strike). Returns ({key: per-event np.array}, n_used),
    pooled in item order. Streaming caps GPU memory at a single model so it is portable
    across whatever card the job lands on (the all-10-resident version OOM'd)."""
    acc = [None] * len(items); n_used = 0
    for load in seed_loaders:
        try:
            m = load()
        except (Exception, SystemExit) as e:
            print(f"    [stream: seed load failed {type(e).__name__}; skipping]", flush=True); continue
        with torch.no_grad():
            for i, (key, aa, ev, strike) in enumerate(items):
                o = m(aa.unsqueeze(0).to(dev)).float().squeeze(0)
                lp = torch.log_softmax(o.reshape(-1), dim=-1).reshape(o.shape).cpu()
                acc[i] = lp if acc[i] is None else acc[i] + lp
        del m
        if dev.type == 'cuda': torch.cuda.empty_cache()
        n_used += 1
    if n_used == 0: raise SystemExit("stream_score: no usable seeds")
    out = {}
    for i, (key, aa, ev, strike) in enumerate(items):
        out.setdefault(key, []).extend(
            per_event_weights(acc[i] / n_used, aa, ev, seen_mask=strike, q=Q))
    return {k: np.asarray(v, np.float32) for k, v in out.items()}, n_used


def macro_boot(per_virus_arrays, nboot=NBOOT, seed=0):
    """Macro mean across viruses + CI by resampling events within each virus."""
    arrs = [a for a in per_virus_arrays if a.size]
    if not arrs: return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    pt = float(np.mean([a.mean() for a in arrs]))
    bs = np.empty(nboot)
    for b in range(nboot):
        bs[b] = np.mean([a[rng.integers(0, a.size, a.size)].mean() for a in arrs])
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    td = build_tdata()
    W = {}  # (arm, virus, subset) -> per-event array

    def items_for(trees):
        out = []
        for t in trees:
            for sub, (ent, strike) in td[t][1].items():
                for aa, ev in ent:
                    out.append(((t, sub), aa, ev, strike))
        return out

    # --- hybrid: joint49-sel9 streamed over 10 seeds (virus-independent) ---
    def j49_loader(s):
        def f():
            try: return load_joint49_sel9(s)[0]
            except SystemExit as e: raise RuntimeError(str(e))
        return f
    jout, jn = stream_score([j49_loader(s) for s in J49_SEEDS], items_for(TREES))
    for (t, sub), v in jout.items(): W[('hybrid', t, sub)] = v
    print(f"  hybrid scored ({jn} seeds)", flush=True)

    # --- attach xformer: per-virus, streamed over 3 seeds ---
    for t in TREES:
        xout, xn = stream_score([(lambda s=s: load_xf_seed(t, s)) for s in XF_SEEDS], items_for([t]))
        for (tt, sub), v in xout.items(): W[('xfattach', tt, sub)] = v
        print(f"  xfattach scored {t} ({xn} seeds)", flush=True)

    np.savez_compressed(f"{DD}/attach_vs_hybrid.npz",
                        **{f"{a}__{t}__{s}": v for (a, t, s), v in W.items()})

    # --- per-virus recall + CI ---
    with open(f"{DD}/attach_vs_hybrid_pervirus.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "virus", "subset", "n_events", "recall", "ci_lo", "ci_hi"])
        for (a, t, s), v in W.items():
            lo, hi = boot_ci(v); w.writerow([a, t, s, len(v), f"{v.mean():.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    # --- macro recall + CI, per arm/subset; paired macro delta (xfattach - hybrid) ---
    rows_macro, rows_paired = [], []
    for sub in SUBSETS:
        for a in ('hybrid', 'xfattach'):
            arrs = [W[(a, t, sub)] for t in TREES if (a, t, sub) in W]
            m, lo, hi = macro_boot(arrs)
            rows_macro.append([a, sub, len(arrs), f"{m:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
        # paired: per-virus event-aligned diff (xfattach - hybrid)
        diffs = [W[('xfattach', t, sub)] - W[('hybrid', t, sub)]
                 for t in TREES if ('xfattach', t, sub) in W and ('hybrid', t, sub) in W]
        dm, dlo, dhi = macro_boot(diffs)
        rows_paired.append(["macro", sub, f"{dm:+.6f}", f"{dlo:+.6f}", f"{dhi:+.6f}"])
        for t in TREES:
            if ('xfattach', t, sub) in W and ('hybrid', t, sub) in W:
                d = W[('xfattach', t, sub)] - W[('hybrid', t, sub)]
                lo, hi = boot_ci(d); rows_paired.append([t, sub, f"{d.mean():+.6f}", f"{lo:+.6f}", f"{hi:+.6f}"])
    with open(f"{DD}/attach_vs_hybrid_macro.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "subset", "n_virus", "macro_recall", "ci_lo", "ci_hi"]); w.writerows(rows_macro)
    with open(f"{DD}/attach_vs_hybrid_paired.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["virus", "subset", "delta_xf_minus_hybrid", "ci_lo", "ci_hi"]); w.writerows(rows_paired)

    # --- leak table ---
    with open(f"{DD}/attach_vs_hybrid_leak.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["virus", "n_noattach_novel", "n_attach_seen", "frac_attach_seen"])
        for t in TREES:
            ln, lh = td[t][2]; w.writerow([t, ln, lh, f"{(lh/ln if ln else 0):.4f}"])

    # --- console summary ---
    print("\n  === macro recall@10% (h3n2_ha / denv2 / rsv_a) ===")
    print(f"  {'subset':<10}{'hybrid':>10}{'xfattach':>10}{'delta':>12}")
    for sub in SUBSETS:
        h = np.mean([W[('hybrid', t, sub)].mean() for t in TREES if ('hybrid', t, sub) in W])
        x = np.mean([W[('xfattach', t, sub)].mean() for t in TREES if ('xfattach', t, sub) in W])
        print(f"  {sub:<10}{h:>10.4f}{x:>10.4f}{x-h:>+12.4f}")
    print("\n  === de-novo leak (noattach-novel cells the attach train has seen) ===")
    for t in TREES:
        ln, lh = td[t][2]; print(f"  {t:<10} {lh}/{ln} = {(lh/ln if ln else 0):.1%}")
    print("\n  wrote attach_vs_hybrid_{pervirus,macro,paired,leak}.csv + .npz", flush=True)


if __name__ == '__main__':
    main()
