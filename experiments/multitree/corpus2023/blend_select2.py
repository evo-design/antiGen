#!/usr/bin/env python
"""Five-way blend weights, chosen on the SARS-CoV-2 development holdouts (NOTES s158).

blend_select.py (s157) with two more components, both legitimately out of sample on
member i's holdout:
  A  antiGen CV x10 member i
  J  joint (SARS-CoV-2 + 14 coronaviruses) member i (--joint-epoch; 'best' = its own
     --save-best checkpoint, chosen on that same holdout by training's early stopping)
  C  the coronavirus-only finetune at --cov-epoch (round 1's dev choice: epoch 0)
  S  mcm_interpolation (SxS / MCM interpolation), its counts REFIT on member i's
     training 90% -- the published row is fit on all training examples, holdout
     included, so it cannot be weighted on the holdout as-is
  K  codon: untrained, out of sample everywhere

score = sum_k w_k log p_k, renormalised per query (sum w = 1). Weights on the full
simplex grid (--step); the PRIMARY pick is the grid minimum of the mean-over-members
holdout NLL over all mutations (the s157 criterion). Every subset lies on the grid, so
the primary pick is one choice, not a menu. The de novo criterion's minimum is reported
as a secondary pick. The panels are never read here.
"""
import argparse
import glob
import itertools
import json

import torch

from holdout_select import AA20, EX, SRC, STORE, TEMPLATE, build, logprobs, nll
from antiGen.config import AlphabetConfig, CodonConfig, ModelConfig, ModelType
from antiGen.eval_utils import CountedStatPredictor
from antiGen.load import load_examples
from antiGen.models.factory import create_model
from antiGen.statistical_baselines import fit as count_fit
from antiGen.train import split_examples

NAMES = ("A", "J", "C", "S", "K")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-epoch", default="best")
    ap.add_argument("--cov-epoch", type=int, default=0)
    ap.add_argument("--cov-dir", default=f"{STORE}/joint_cov15_2023/covonly_s42/epochs")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default=f"{STORE}/joint_cov15_2023/blend_select2.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    exs = load_examples(EX, AA20, "BA.2", source=SRC)
    L = len(exs[0].aa_idx); E = len(AA20)
    covm = build(glob.glob(f"{a.cov_dir}/*_ep{a.cov_epoch:02d}.pt")[0], dev, TEMPLATE)
    codon = create_model(ModelConfig(model_type=ModelType.CODON, seq_len=L,
                                     alphabet=AlphabetConfig(amino_acids=AA20),
                                     codon=CodonConfig()), dev).eval()
    per = []
    for i in (0, 1, 2):
        tr, va = split_examples({SRC: exs}, "random", 0.1, i)
        tr, va = tr[SRC], va[SRC]
        seen = torch.zeros(L, E, dtype=torch.bool)
        for e in tr:
            seen[torch.arange(L), e.aa_idx] = True
            seen |= e.mat > 0
        ag = glob.glob(f"{STORE}/ckpt_stage_vs10_2023-01-01/fitness_vs10Antigen_s{i}_*.pt")[0]
        jf = (glob.glob(f"{STORE}/joint_cov15_2023/cv/s{i}/fitness_*_best.pt")[0]
              if a.joint_epoch == "best" else
              glob.glob(f"{STORE}/joint_cov15_2023/cv/s{i}/epochs/*_ep{int(a.joint_epoch):02d}.pt")[0])
        jep = torch.load(jf, map_location="cpu", weights_only=False)["epoch"]
        smod = CountedStatPredictor(
            ModelConfig(model_type=ModelType.SITE_SUBSTITUTION, seq_len=L,
                        alphabet=AlphabetConfig(amino_acids=AA20)),
            count_fit(tr, L, E), "mcm_interpolation").to(dev).eval()
        lp = dict(A=logprobs(build(ag, dev), va, dev), J=logprobs(build(jf, dev, TEMPLATE), va, dev),
                  C=logprobs(covm, va, dev), S=logprobs(smod, va, dev), K=logprobs(codon, va, dev))
        tgt = torch.stack([e.mat for e in va])
        p = dict(i=i, jep=jep, tgt=tgt.to(dev), dn=(~seen).float().to(dev),
                 lp=[torch.nan_to_num(lp[n], neginf=-1e4).to(dev) for n in NAMES])
        p["solo"] = {n: (nll(lp[n], tgt)[0], nll(lp[n], tgt, (~seen).float())[0]) for n in NAMES}
        per.append(p)
        print(f"member {i}: joint epoch {jep}; solo all / de novo NLL "
              + ", ".join(f"{n} {v[0]:.3f}/{v[1]:.3f}" for n, v in p["solo"].items()), flush=True)

    @torch.no_grad()
    def score(ws):
        al = dn = 0.0
        for p in per:
            s = sum(w * x for w, x in zip(ws, p["lp"]) if w > 0)
            m = torch.log_softmax(s.flatten(1), 1).view_as(s)
            m = m.masked_fill(p["lp"][0] <= -1e4 + 1, 0.0)  # WT cells (masked in every comp)
            t = p["tgt"]
            al += float(-(t * m).sum()) / float(t.sum())
            td = t * p["dn"]
            dn += float(-(td * m).sum()) / max(float(td.sum()), 1)
        return al / 3, dn / 3

    K = int(round(1 / a.step))
    res = {}
    for c in itertools.product(range(K + 1), repeat=len(NAMES) - 1):
        if sum(c) > K:
            continue
        ws = (round((K - sum(c)) * a.step, 4),) + tuple(round(x * a.step, 4) for x in c)
        res[ws] = score(ws)
    print(f"{len(res)} grid points", flush=True)

    def best(cond, crit=0):
        k = min((k for k in res if cond(k)), key=lambda k: res[k][crit])
        return dict(zip(NAMES, k), all=res[k][0], denovo=res[k][1])

    sub = lambda keep: (lambda k: all(w == 0 for n, w in zip(NAMES, k) if n not in keep))
    out = dict(cov_epoch=a.cov_epoch, joint_epochs=[p["jep"] for p in per],
               solo={n: [p["solo"][n] for p in per] for n in NAMES},
               antiGen_only=best(lambda k: k[0] == 1.0),
               A_J_C=best(sub("AJC")), A_S=best(sub("AS")), A_K=best(sub("AK")),
               A_S_K=best(sub("ASK")),
               primary=best(lambda k: True), secondary_denovo=best(lambda k: True, crit=1))
    json.dump(out, open(a.out, "w"), indent=1)
    for n in ("antiGen_only", "A_J_C", "A_S", "A_K", "A_S_K", "primary", "secondary_denovo"):
        b = out[n]
        print(f"{n:17s} " + " ".join(f"w{c}={b[c]:.2f}" for c in NAMES)
              + f"  holdout NLL all {b['all']:.4f}  de novo {b['denovo']:.4f}")


if __name__ == "__main__":
    main()
