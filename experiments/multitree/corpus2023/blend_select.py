#!/usr/bin/env python
"""Three-way blend weights, chosen on the SARS-CoV-2 development holdouts (NOTES s157).

Components, per CV member i (i = 0, 1, 2; split seed i, the holdout antiGen CV x10 member
i and joint member i share):
  A  antiGen CV x10 member i
  J  joint (SARS-CoV-2 + 14 coronaviruses) member i, at --joint-epoch ('best' = the
     member's own --save-best checkpoint)
  C  the coronavirus-only finetune (never saw SARS-CoV-2, so every holdout is out of
     sample for it), at the epoch with the lowest SARS-CoV-2 loss pooled over the three
     holdouts.

score = wA log pA + wJ log pJ + wC log pC, renormalised per query (wA + wJ + wC = 1), the
weighted mean-of-log-probs rule. Weights on a grid, chosen by the mean over members of
the holdout NLL over ALL mutations -- the criterion that chose mixture A. The panels are
never read here.
"""
import argparse
import glob
import itertools
import json

import torch

from holdout_select import (AA20, EX, SRC, STORE, TEMPLATE, build, logprobs, nll)
from antiGen.load import load_examples
from antiGen.train import split_examples


def mix(lps, ws):
    s = sum(w * torch.nan_to_num(lp, neginf=-1e4) for w, lp in zip(ws, lps) if w > 0)
    return torch.log_softmax(s.flatten(1), 1).view_as(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-epoch", default="best")
    ap.add_argument("--cov-dir", default=f"{STORE}/joint_cov15_2023/covonly_s42/epochs")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default=f"{STORE}/joint_cov15_2023/blend_select.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exs = load_examples(EX, AA20, "BA.2", source=SRC)
    cov_eps = sorted(glob.glob(f"{a.cov_dir}/*_ep*.pt"))
    covm = {int(f.rsplit("_ep", 1)[1][:2]): build(f, device, TEMPLATE) for f in cov_eps}
    per = []
    for i in (0, 1, 2):
        tr, va = split_examples({SRC: exs}, "random", 0.1, i)
        tr, va = tr[SRC], va[SRC]
        L, E = va[0].mat.shape
        seen = torch.zeros(L, E, dtype=torch.bool)
        for e in tr:
            seen[torch.arange(L), e.aa_idx] = True
            seen |= e.mat > 0
        tgt = torch.stack([e.mat for e in va])
        ag = glob.glob(f"{STORE}/ckpt_stage_vs10_2023-01-01/fitness_vs10Antigen_s{i}_*.pt")[0]
        jf = (glob.glob(f"{STORE}/joint_cov15_2023/cv/s{i}/fitness_*_best.pt")[0]
              if a.joint_epoch == "best" else
              glob.glob(f"{STORE}/joint_cov15_2023/cv/s{i}/epochs/*_ep{int(a.joint_epoch):02d}.pt")[0])
        jep = torch.load(jf, map_location="cpu", weights_only=False)["epoch"]
        per.append(dict(i=i, tgt=tgt, dn=(~seen).float(),
                        A=logprobs(build(ag, device), va, device),
                        J=logprobs(build(jf, device, TEMPLATE), va, device), jep=jep,
                        C={ep: logprobs(m, va, device) for ep, m in covm.items()}))
        print(f"member {i}: joint epoch {jep}", flush=True)
    # coronavirus-only epoch: pooled SARS-CoV-2 holdout NLL (all mutations)
    def pooled(ep):
        num = sum(nll(p["C"][ep], p["tgt"])[0] * nll(p["C"][ep], p["tgt"])[1] for p in per)
        return num / sum(nll(p["C"][ep], p["tgt"])[1] for p in per)
    cov_curve = {ep: pooled(ep) for ep in covm}
    cep = min(cov_curve, key=cov_curve.get)
    print("coronavirus-only, pooled SARS-CoV-2 holdout NLL by epoch:",
          {k: round(v, 4) for k, v in cov_curve.items()}, "-> epoch", cep, flush=True)
    for p in per:   # keep the chosen epoch only; the grid runs on the GPU
        p["C"] = p["C"][cep].to(device)
        for k in ("A", "J", "tgt", "dn"):
            p[k] = p[k].to(device)
    grid = [round(x * a.step, 4) for x in range(int(round(1 / a.step)) + 1)]
    res = {}
    for wj, wc in itertools.product(grid, grid):
        wa = round(1 - wj - wc, 4)
        if wa < -1e-9:
            continue
        ms = [mix([p["A"], p["J"], p["C"]], (wa, wj, wc)) for p in per]
        al = [nll(m, p["tgt"])[0] for m, p in zip(ms, per)]
        dn = [nll(m, p["tgt"], p["dn"])[0] for m, p in zip(ms, per)]
        del ms
        res[(wa, wj, wc)] = (sum(al) / 3, sum(dn) / 3)
    def best(cond):
        k = min((k for k in res if cond(k)), key=lambda k: res[k][0])
        return dict(wA=k[0], wJ=k[1], wC=k[2], all=res[k][0], denovo=res[k][1])
    out = dict(cov_epoch=cep, cov_curve=cov_curve, joint_epochs=[p["jep"] for p in per],
               antiGen_only=best(lambda k: k == (1.0, 0.0, 0.0)),
               A_J=best(lambda k: k[2] == 0), A_C=best(lambda k: k[1] == 0),
               A_J_C=best(lambda k: True))
    json.dump(out, open(a.out, "w"), indent=1)
    for n in ("antiGen_only", "A_J", "A_C", "A_J_C"):
        b = out[n]
        print(f"{n:13s} wA={b['wA']:.2f} wJ={b['wJ']:.2f} wC={b['wC']:.2f}  "
              f"holdout NLL all {b['all']:.4f}  de novo {b['denovo']:.4f}")


if __name__ == "__main__":
    main()
