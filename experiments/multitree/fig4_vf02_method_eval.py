#!/usr/bin/env python
"""'5x val_all-selection + ensemble' eval for one tree's vf02 runs (FIG4_TREES=<tree>).

Each of the 5 seeds' best.pt (selected by val recall@10 'all' = recall10_macro) is logit-
ensembled; scored on the test set three ways (all / non-de-novo / de-novo) with event
bootstrap CIs. Baselines (fit on the full train corpus) are scored the same three ways for a
full comparison. Prints the table; the model 5-seed ensemble rows are the deployable arms.
"""
import os, sys, glob, zlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.eval_utils import (fit_substitution_baseline, positional_marginal_counts,
    frequency_baseline_scores, where_only_baseline_scores, load_checkpoint)
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
DST = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
CUTS = {"denv2": "2019-10-01", "rsv_a": "2023-10-01", "h3n2": "2022-10-01"}
TREE = os.environ.get("FIG4_TREES", "denv2").strip()
CUT = CUTS[TREE]
RUNS = f"{R}/fig4_{TREE}_vf02"
VARIANTS = ["xformer", "esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
VL = {"xformer": "Transformer (scratch)", "esm35m_p4": "ESM-2 35M partial-4",
      "esm35m_ft": "ESM-2 35M finetune", "esm650m_p4": "ESM-2 650M partial-4",
      "esm650m_ft": "ESM-2 650M finetune"}
SEEDS = [0, 1, 2, 3, 4]; Q = 0.10; B = 2000

def pew(scores, aa_idx, events):
    s = scores.clone().float()
    for c in MASK: s[:, c] = float('nan')
    s[torch.arange(len(aa_idx)), aa_idx] = float('nan')
    flat = s.flatten(); valid = ~torch.isnan(flat); nv = int(valid.sum())
    if nv == 0: return [0.0]*len([e for e in events if e[1] not in MASK])
    vb = torch.nanquantile(flat, 1.0 - Q)
    above = (flat > vb) & valid; na = int(above.sum()); tie = (flat == vb) & valid; nt = int(tie.sum())
    frac = max(0.0, (Q*nv - na))/nt if nt > 0 else 0.0
    out = []
    for (p, a) in events:
        if a in MASK: continue
        v = s[p, a]; out.append(0.0 if torch.isnan(v) else (1.0 if v > vb else (frac if v == vb else 0.0)))
    return out

def best_pt(run):
    c = [p for p in glob.glob(f"{run}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return c[-1] if c else None

def tw_three(weights, dn):
    w = np.asarray(weights); return w.mean(), w[~dn].mean(), w[dn].mean()

def main():
    full = load_examples(f"{DST}/examples_{TREE}@{CUT}_train.txt", ALPHA_SET, None, source=TREE)
    L = len(full[0].aa_idx); st = count_baselines.fit(full, L, E); N = st["N"]
    te = load_examples(f"{DST}/examples_{TREE}@{CUT}_test.txt", ALPHA_SET, None, source=TREE)
    dn = np.array([float(N[p, a]) < 1 for ex in te for (p, a) in events_of(ex) if a not in MASK], bool)
    print(f"\n==== {TREE} @ {CUT}: recall@10%  (all={len(dn)}, non-DN={int((~dn).sum())}, DN={int(dn.sum())}) ====")
    print(f"     method = 5-seed (val-frac 0.2) val_all-selected best.pt, logit ensemble\n")
    print(f"{'arm':<26}{'ALL':>8}{'non-DN':>9}{'DE-NOVO':>9}")

    # models: 5-seed best.pt ensemble
    for v in VARIANTS:
        paths = [best_pt(f"{RUNS}/{v}_s{s}") for s in SEEDS]; paths = [p for p in paths if p]
        if not paths: print(f"{VL[v]:<26}{'(no ckpts)':>8}"); continue
        models = [load_esm(p) for p in paths]
        w = []
        with torch.no_grad():
            for ex in te:
                ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                if not ev: continue
                z = torch.stack([m(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu() for m in models], 0)
                w += pew(z.mean(0), ex.aa_idx, ev)
        del models
        if dev.type == 'cuda': torch.cuda.empty_cache()
        a, nd, d = tw_three(w, dn)
        tag = "" if len(paths) == 5 else f"  ({len(paths)}/5 seeds)"
        print(f"{VL[v]:<26}{a:>8.3f}{nd:>9.3f}{d:>9.3f}{tag}")

    # baselines (fit on full train)
    print(f"{'-'*52}")
    fitted = fit_substitution_baseline(full, dev, ALPHA_SET)
    sub_mat = torch.log_softmax(fitted.score_matrix().detach().cpu(), dim=-1)
    pos_counts = positional_marginal_counts(full, L, E)
    freq = frequency_baseline_scores(full, L, E); wonly = where_only_baseline_scores(full, L, E)
    cbcfg = ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=L,
                        alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                        codon_blosum=CodonBlosumConfig(kappa=4.0))
    cb = create_model(cbcfg, dev).eval()
    BL = [("cpssm_dir", "Smoothed Counts"), ("site_prop", "Site x Substitution"),
          ("frequency", "Event Counts"), ("where_times_which", "Where x which"),
          ("substitution", "Substitution"), ("codon_blosum", "Codon aware BLOSUM62"),
          ("pssm", "PSSM"), ("where_only", "Where only")]
    rows = {k: [] for k, _ in BL}
    with torch.no_grad():
        for ex in te:
            ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
            if not ev: continue
            aa = ex.aa_idx; ab1 = aa.unsqueeze(0)
            cbs = cb(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
            m = {"cpssm_dir": count_baselines.score_map("cpssm_dir", st, ab1).squeeze(0),
                 "site_prop": count_baselines.score_map("site_prop", st, ab1).squeeze(0),
                 "pssm": count_baselines.score_map("pssm", st, ab1).squeeze(0),
                 "frequency": torch.log(freq), "where_only": torch.log(wonly),
                 "where_times_which": sub_mat[aa] + torch.log(pos_counts.clamp(min=1e-12)).view(L, 1),
                 "substitution": sub_mat[aa], "codon_blosum": cbs}
            for k, _ in BL: rows[k] += pew(m[k], aa, ev)
    for k, lab in BL:
        a, nd, d = tw_three(rows[k], dn)
        print(f"{lab:<26}{a:>8.3f}{nd:>9.3f}{d:>9.3f}")
    print(f"{'Uniform':<26}{0.10:>8.3f}{0.10:>9.3f}{0.10:>9.3f}")

if __name__ == '__main__':
    main()
