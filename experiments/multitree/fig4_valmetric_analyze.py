#!/usr/bin/env python
"""Compare epoch-selection rules on denv2: which VAL metric picks the epoch that maximizes
TEST de-novo recall@10%? Reads perevent_fig4_valmetric_denv2.npz (per variant/seed/epoch:
val hit weights + per-val-event novelty/count; test hit weights + de-novo flags).

For each run we build per-epoch curves of:
  val_all      = mean val hit@10                       (CURRENT selection metric)
  val_novel    = mean val hit@10 over NOVEL val events (unseen in the 90% train-split)
  val_rareK    = mean val hit@10 over val events with train-split count <= K
  test_all / test_denovo = the targets.
A selection rule picks e* = argmax_e(val metric); we report the test_denovo (and test_all)
it lands on, averaged over the 3 seeds, vs the oracle (argmax test_denovo) and the current rule.
Also Spearman corr (across epochs, per run) of each val metric with test_denovo.
"""
import os
import numpy as np
NPZ = "/large_storage/hielab/aakarsh/multitree/fig4_single_attach/perevent_fig4_valmetric_denv2.npz"
VARIANTS = ["esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft", "xformer"]
SEEDS = [0, 1, 2]
LAB = {"esm35m_p4": "ESM-2 35M p4", "esm35m_ft": "ESM-2 35M ft",
       "esm650m_p4": "ESM-2 650M p4", "esm650m_ft": "ESM-2 650M ft", "xformer": "Transformer"}
RARE_K = [1, 2]

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3: return np.nan
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d else np.nan

def main():
    z = np.load(NPZ)
    testdn_flag = z["__testdn__"].astype(bool)
    def mean_sub(w, mask=None):
        w = np.asarray(w, float)
        if mask is None: return float(w.mean()) if len(w) else np.nan
        return float(w[mask].mean()) if mask.sum() else np.nan

    # selection rules -> function(curves dict for a run) -> metric array over epochs
    rules = ["val_all", "val_novel"] + [f"val_rare{k}" for k in RARE_K]
    print(f"{'variant':<16}{'rule':<12}{'selEp(mean)':>11}{'test_denovo':>12}{'test_all':>10}{'rho(vs dn)':>11}")
    summary = {}
    for v in VARIANTS:
        # gather per-seed per-epoch
        agg = {r: [] for r in rules + ["oracle_dn", "current"]}
        agg_all = {r: [] for r in rules + ["oracle_dn", "current"]}
        selep = {r: [] for r in rules}
        rhos = {r: [] for r in rules}
        for s in SEEDS:
            es = sorted(int(k.split("__e")[1].split("__")[0]) for k in z.files
                        if k.startswith(f"{v}__s{s}__e") and k.endswith("__valw"))
            if not es: continue
            curves = {r: [] for r in rules}
            t_dn, t_all = [], []
            for e in es:
                p = f"{v}__s{s}__e{e}"
                vw = z[f"{p}__valw"]; nov = z[f"{p}__valnovel"].astype(bool); cnt = z[f"{p}__valcnt"]
                tw = z[f"{p}__testw"]
                curves["val_all"].append(mean_sub(vw))
                curves["val_novel"].append(mean_sub(vw, nov))
                for k in RARE_K: curves[f"val_rare{k}"].append(mean_sub(vw, cnt <= k))
                t_dn.append(mean_sub(tw, testdn_flag)); t_all.append(mean_sub(tw))
            t_dn = np.array(t_dn); t_all = np.array(t_all); es = np.array(es)
            # current = argmax val_all (what best.pt approximates); oracle = argmax test_denovo
            for r in rules:
                c = np.array(curves[r], float)
                if np.all(np.isnan(c)): continue
                i = int(np.nanargmax(c))
                agg[r].append(t_dn[i]); agg_all[r].append(t_all[i]); selep[r].append(es[i])
                rhos[r].append(spearman(c, t_dn))
            agg["oracle_dn"].append(t_dn.max()); agg_all["oracle_dn"].append(t_all[int(t_dn.argmax())])
            ic = int(np.nanargmax(curves["val_all"]))
            agg["current"].append(t_dn[ic]); agg_all["current"].append(t_all[ic])
        for r in rules:
            if not agg[r]: continue
            print(f"{LAB[v]:<16}{r:<12}{np.mean(selep[r]):>11.1f}{np.mean(agg[r]):>12.3f}"
                  f"{np.mean(agg_all[r]):>10.3f}{np.nanmean(rhos[r]):>11.2f}")
        print(f"{LAB[v]:<16}{'oracle_dn':<12}{'':>11}{np.mean(agg['oracle_dn']):>12.3f}{np.mean(agg_all['oracle_dn']):>10.3f}{'':>11}")
        print()
        summary[v] = {r: np.mean(agg[r]) for r in rules + ["oracle_dn"]}

    # headline: pooled over the 4 ESM variants
    esm = ["esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
    print("=== ESM-pooled test de-novo recall@10% by selection rule ===")
    for r in rules + ["oracle_dn"]:
        print(f"  {r:<12}: {np.mean([summary[v][r] for v in esm if r in summary[v]]):.3f}")

if __name__ == '__main__':
    main()
