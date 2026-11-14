"""Model-free (S×M–MCM, Mixture) and EVE rungs of figure 4, full and 1nt pools."""
import csv
import os
import sys
import time

import numpy as np
import torch

from antiGen.eval_utils import fit_substitution_baseline, positional_marginal_counts
from antiGen.load import load_examples
from fig4_common import (A, CKPT, CORPORA, D, E, ENDEMIC_CORPORA, OUT, PANEL9, Q, REACH_1NT, corpus,
                         dev, events_of, per_event_weights, pool_1nt, seen_mask)

EVE_DIR = os.environ.get("FIG4_EVE", os.path.join(os.path.dirname(CKPT), "eve"))
ENDEMIC_TEST = os.environ.get("FIG4_ENDEMIC_TEST", os.path.join(os.path.dirname(CORPORA), "endemic"))
ENDEMIC = [("dengue", "denv2", "denv2", "2019-10-01"), ("rsv_a", "rsv_a", "rsv-a", "2023-10-01"),
           ("flu", "h3n2", "h3n2", "2022-10-01")]
HEADER = ["panel", "virus", "subset", "rung", "n", "recall", "ci_lo", "ci_hi"]
FLOOR = 1e-30
B = 2000


def fit(train):
    """MCM counts plus the S×M site and mixture terms off one training corpus."""
    L = len(train[0].aa_idx)
    N = sum(ex.mat for ex in train)
    sub = torch.log_softmax(fit_substitution_baseline(train, dev, A).score_matrix().detach().cpu(), dim=-1)
    pos = positional_marginal_counts(train, L, E).float()
    Pm = sub.exp().clone()
    Pm.fill_diagonal_(0.0)
    Pm = Pm / Pm.sum(dim=1, keepdim=True).clamp(min=FLOOR)
    return dict(N=N, Nt=float(N.sum()), sub=sub, sprime=pos / pos.sum(), Pm=Pm)


def rung_map(kind, aa, f):
    if kind == "sxm_mcm":
        P = f["sprime"].view(-1, 1) * f["Pm"][aa]
        return torch.log(((f["N"] + f["Nt"] * P) / (2 * f["Nt"])).clamp(min=FLOOR))
    return f["sub"][aa]


def weights(m, aa, ev, strike):
    """Per-event weights under the full pool and under the 1nt pool (1nt targets only)."""
    ev1 = [(p, a) for p, a in ev if REACH_1NT[aa[p], a]]
    return (per_event_weights(m, aa, ev, strike=strike),
            per_event_weights(m, aa, ev1, strike=strike, pool=pool_1nt(aa)), ev1)


def ci(w):
    w = np.asarray(w, float)
    if not len(w):
        return float("nan"), float("nan"), float("nan")
    bo = w[np.random.default_rng(0).integers(0, len(w), size=(B, len(w)))].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))


def macro(rv):
    """Equal weight per virus; per-virus bootstrap SEs propagated in quadrature."""
    n = len(rv)
    est = sum(r for r, _, _ in rv) / n
    s = sum(((hi - lo) / (2 * 1.96)) ** 2 for _, lo, hi in rv) ** 0.5 / n
    return est, max(0.0, est - 1.96 * s), min(1.0, est + 1.96 * s)


def row(panel, virus, sub, rung, w):
    r, lo, hi = ci(w)
    return [panel, virus, sub, rung, len(w), f"{r:.6f}", f"{lo:.6f}", f"{hi:.6f}"]


def endemic_rows():
    rows = {"full": [], "1nt": []}
    for lab, tree, build, cut in ENDEMIC:
        train = load_examples(f"{ENDEMIC_CORPORA}/{build}-{cut}/train/examples_{cut}_cur.txt", A, None, source=tree)
        f = fit(train)
        seen = f["N"] > 0
        test = load_examples(f"{ENDEMIC_TEST}/examples_{tree}@{cut}_test.txt", A, None, source=tree)
        acc = {(c, s): [] for c in rows for s in ("all", "denovo")}
        for ex in test:
            ev = events_of(ex)
            if not ev:
                continue
            m = rung_map("sxm_mcm", ex.aa_idx, f)
            wf, w1, _ = weights(m, ex.aa_idx, ev, None)
            acc["full", "all"] += wf
            acc["1nt", "all"] += w1
            df, d1, ev1 = weights(m, ex.aa_idx, ev, seen)
            acc["full", "denovo"] += [w for w, (p, a) in zip(df, ev) if not seen[p, a]]
            acc["1nt", "denovo"] += [w for w, (p, a) in zip(d1, ev1) if not seen[p, a]]
        for (conv, sub), w in acc.items():
            rows[conv].append(row("endemic", lab, sub, "sxm_mcm", w))
        print(f"[endemic {tree}] done", flush=True)
    return rows


def eve_maps(tree):
    p = f"{EVE_DIR}/scores_{tree}.pt"
    return torch.load(p) if os.path.exists(p) else None


def panel9_rows():
    rows = {"full": [], "1nt": []}
    per = {}
    for t in PANEL9:
        train = corpus(t, "train")
        f = fit(train)
        seen = seen_mask(train)
        eve = eve_maps(t)
        for sub, split in (("all", "test"), ("novel", "novel")):
            test = corpus(t, split)
            strike = None if sub == "all" else seen
            acc = {}
            for i, ex in enumerate(test):
                ev = events_of(ex)
                if not ev:
                    continue
                maps = {k: rung_map(k, ex.aa_idx, f) for k in ("sxm_mcm", "mixture")}
                if eve is not None:
                    maps["eve"] = eve[split][i]
                for k, m in maps.items():
                    wf, w1, _ = weights(m, ex.aa_idx, ev, strike)
                    acc.setdefault(("full", k), []).extend(wf)
                    acc.setdefault(("1nt", k), []).extend(w1)
            for (conv, k), w in acc.items():
                r = row("j9_pervirus", t, sub, k, w)
                rows[conv].append(r)
                per.setdefault((conv, sub, k), []).append(tuple(float(x) for x in r[5:]))
        print(f"[panel9 {t}] done{'' if eve is not None else ' (no EVE scores)'}", flush=True)
    for (conv, sub, k), rv in per.items():
        if k == "eve" and len(rv) < len(PANEL9):
            continue
        est, lo, hi = macro(rv)
        rows[conv].append(["j9_macro", "MACRO", sub, k, len(rv), f"{est:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
    return rows


def main():
    end, p9 = endemic_rows(), panel9_rows()
    os.makedirs(OUT, exist_ok=True)
    for conv, name in (("full", "fig4_extra_rungs.csv"), ("1nt", "fig4_extra_rungs_1nt.csv")):
        with open(f"{OUT}/{name}", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(end[conv] + p9[conv])
        print("wrote", f"{OUT}/{name}")


def eve_scores(tree):
    """Pretrain frozen EVE on the tree's MSA and save its score matrices for test and novel."""
    from antiGen.config import AlphabetConfig, EVEConfig, ModelConfig, ModelType
    from antiGen.models.factory import create_model

    L = len(corpus(tree, "train")[0].aa_idx)
    cfg = ModelConfig(model_type=ModelType.EVE, seq_len=L, alphabet=AlphabetConfig(amino_acids=A),
                      eve=EVEConfig(msa_path=f"{EVE_DIR}/msa/{tree}.fasta", freeze_backbone=True))
    t0 = time.time()
    model = create_model(cfg, dev).eval()
    print(f"[{tree}] EVE pretrained in {time.time() - t0:.0f}s", flush=True)
    out = {}
    for split in ("test", "novel"):
        mats = []
        for ex in corpus(tree, split):
            with torch.no_grad():
                m = model(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
            m[torch.isinf(m)] = float("nan")
            mats.append(m)
        out[split] = mats
    torch.save(out, f"{EVE_DIR}/scores_{tree}.pt")


if __name__ == "__main__":
    if sys.argv[1:2] == ["eve"]:
        eve_scores(PANEL9[int(sys.argv[2])] if sys.argv[2].isdigit() else sys.argv[2])
    else:
        main()
