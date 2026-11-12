#!/usr/bin/env python
"""Choose, WITHOUT the test panels, (a) each joint CV member's epoch and (b) the weight w
of a joint / antiGen mixture -- on the SARS-CoV-2 holdouts the two CV ensembles share
(NOTES s157).

Joint member i (split seed i) and antiGen CV x10 member i hold out the SAME 10% of the
2023-01-01 SARS-CoV-2 training examples, so on that holdout both are out of sample.

Two holdout losses, per mutation:
  all     every holdout child mutation (what --save-best selected on);
  denovo  only the holdout mutations at a (site, residue) cell that occurs NOWHERE in the
          member's training split -- in no training node's sequence and in no training
          child mutation. The holdout analogue of panels E/F's de novo set
          (pssm_root == 0 over the training window).

Mixture: log p = (1 - w) log p_antiGen + w log p_joint, renormalised per query over the
non-wild-type cells -- the ranking a weighted mean-of-log-probs ensemble gives.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from antiGen.eval_utils import load_checkpoint
from antiGen.load import load_examples
from antiGen.models.factory import create_model
from antiGen.train import split_examples

AA20 = "ACDEFGHIKLMNPQRSTVWY"
SRC = "examples_2023-01-01_cur"
EX = "/home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train/examples_2023-01-01_cur.txt"
STORE = "/large_storage/hielab/ispecht/fitness-inference"
TEMPLATE = (f"{STORE}/cutoff_series/2023-01-01/esm2_35m_partial/"
            "fitness_esm2_esm2_t12_35M_UR50D_lr0.0001_20261025_001617.pt")


def build(path, device, template=None):
    """A model from a checkpoint; with `template`, its weights in the template's config
    (joint checkpoints record seq_len 1,447; ESM-2 does not depend on it)."""
    if template is None:
        ck, cfg = load_checkpoint(path, map_location=device)
        sd = ck["model_state_dict"]
    else:
        _, cfg = load_checkpoint(template, map_location=device)
        sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    m = create_model(cfg, device)
    m.load_state_dict(sd, strict=template is not None)
    return m.eval()


@torch.no_grad()
def logprobs(model, exs, device, bs=32):
    """(N, L, E) log-probabilities, WT masked (-inf), normalised per query."""
    out = []
    for i in range(0, len(exs), bs):
        aa = torch.stack([e.aa_idx for e in exs[i:i + bs]]).to(device)
        lg = model(aa).float()
        lg = lg.scatter(2, aa.unsqueeze(-1), float("-inf"))
        out.append(torch.log_softmax(lg.flatten(1), 1).view_as(lg).cpu())
    return torch.cat(out)


def nll(lp, tgt, mask=None):
    """Mean NLL per target mutation; `mask` (L, E) restricts to de novo cells."""
    t = tgt if mask is None else tgt * mask
    n = float(t.sum())
    return float(-(t * torch.nan_to_num(lp, neginf=0.0)).sum()) / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="0,1,2")
    ap.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--out", default=f"{STORE}/joint_cov15_2023/holdout_select.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exs = load_examples(EX, AA20, "BA.2", source=SRC)
    W = [float(x) for x in a.weights.split(",")]
    report = {}
    for i in [int(x) for x in a.members.split(",")]:
        tr, va = split_examples({SRC: exs}, "random", 0.1, i)
        tr, va = tr[SRC], va[SRC]
        L, E = va[0].mat.shape
        seen = torch.zeros(L, E, dtype=torch.bool)
        for e in tr:
            seen[torch.arange(L), e.aa_idx] = True
            seen |= e.mat > 0
        denovo = (~seen).float()
        tgt = torch.stack([e.mat for e in va])
        n_all, n_dn = float(tgt.sum()), float((tgt * denovo).sum())
        print(f"member {i}: holdout {len(va)} examples, {n_all:.0f} mutations, "
              f"{n_dn:.0f} at de novo cells", flush=True)
        ag = glob.glob(f"{STORE}/ckpt_stage_vs10_2023-01-01/fitness_vs10Antigen_s{i}_*.pt")[0]
        lp_ag = logprobs(build(ag, device), va, device)
        r = {"antiGen": dict(all=nll(lp_ag, tgt)[0], denovo=nll(lp_ag, tgt, denovo)[0]),
             "joint_epochs": {}, "mix": {}}
        lps = {}
        for f in sorted(glob.glob(f"{STORE}/joint_cov15_2023/cv/s{i}/epochs/*_ep*.pt")):
            ep = int(f.rsplit("_ep", 1)[1][:2])
            lps[ep] = logprobs(build(f, device, TEMPLATE), va, device)
            r["joint_epochs"][ep] = dict(all=nll(lps[ep], tgt)[0], denovo=nll(lps[ep], tgt, denovo)[0])
            print(f"  joint epoch {ep}: all {r['joint_epochs'][ep]['all']:.4f}  "
                  f"de novo {r['joint_epochs'][ep]['denovo']:.4f}", flush=True)
        print(f"  antiGen member {i}: all {r['antiGen']['all']:.4f}  de novo {r['antiGen']['denovo']:.4f}")
        for crit in ("all", "denovo"):
            ep = min(r["joint_epochs"], key=lambda k: r["joint_epochs"][k][crit])
            for w in W:
                mix = (1 - w) * torch.nan_to_num(lp_ag, neginf=-1e4) + w * torch.nan_to_num(lps[ep], neginf=-1e4)
                mix = torch.log_softmax(mix.flatten(1), 1).view_as(mix)
                r["mix"][f"{crit}|ep{ep}|w{w:.2f}"] = dict(all=nll(mix, tgt)[0], denovo=nll(mix, tgt, denovo)[0])
        report[i] = r
    json.dump(report, open(a.out, "w"), indent=1)
    # summary: the epoch each criterion picks, and the w each criterion picks, per member
    print("\nsummary")
    for i, r in report.items():
        for crit in ("all", "denovo"):
            ep = min(r["joint_epochs"], key=lambda k: r["joint_epochs"][k][crit])
            ms = {k: v for k, v in r["mix"].items() if k.startswith(f"{crit}|")}
            best = min(ms, key=lambda k: ms[k][crit])
            print(f"  member {i} [{crit:6s}] joint epoch {ep}; best w {best.split('|w')[1]} "
                  f"(mix {crit} NLL {ms[best][crit]:.4f} vs antiGen {r['antiGen'][crit]:.4f})")


if __name__ == "__main__":
    main()
