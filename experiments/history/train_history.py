#!/usr/bin/env python
"""Train the history transformer on the SARS-CoV-2 2023-01-01 corpus (NOTES s156).

One example per tree node: the query is the node's mutation history from the root
(antiGen.history), the target its child mutations -- antiGen's own targets and loss
(compute_probs + compute_loss_batch: softmax over every non-wild-type cell, NLL per
child mutation). A random 10% holdout of the training examples picks the epoch: the
SAME holdout as antiGen CV x10's member SPLIT_SEED (source name
examples_2023-01-01_cur, split_examples), so the two are comparable member for member.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from antiGen.eval_utils import compute_loss_batch, compute_probs
from antiGen.history import (HistorySiteTransformer, HistoryTransformer,
                             attach_histories, drop_tokens, make_tokens, pad_histories)
from antiGen.load import load_examples
from antiGen.train import split_examples

AA20 = "ACDEFGHIKLMNPQRSTVWY"
SRC = "examples_2023-01-01_cur"


def batches(exs, bs, shuffle, rng):
    idx = list(range(len(exs)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [exs[j] for j in idx[i:i + bs]]


TOKENS, TOKEN_DROP, GEN = "history", 0.0, torch.Generator().manual_seed(0)


def tensors(batch, device, train=False):
    aa = torch.stack([e.aa_idx for e in batch]).to(device)
    mat = torch.stack([e.mat for e in batch]).to(device)
    toks = [make_tokens(e, TOKENS) for e in batch]
    if train and TOKEN_DROP > 0:
        toks = [drop_tokens(t, TOKEN_DROP, GEN) for t in toks]
    hist, pad = pad_histories(toks, device)
    return aa, mat, hist, pad


@torch.no_grad()
def evaluate(model, exs, bs, device):
    model.eval()
    tot, n = 0.0, 0.0
    for b in batches(exs, bs, False, None):
        aa, mat, hist, pad = tensors(b, device)
        probs = compute_probs(model(aa, hist, pad).float(), mat, aa)
        m = float(mat.sum())
        tot += float(compute_loss_batch(probs, mat)) * m
        n += m
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="/home/ivan.specht/antiGen-data-revised/output/"
                    "sc2-2023-01-01/train/examples_2023-01-01_cur.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--arch", choices=["query", "site"], default="query",
                    help="query: one [QUERY] vector scores every cell (v1); site: per-site "
                         "queries cross-attend over the history (design (1))")
    ap.add_argument("--n-cross", type=int, default=2)
    ap.add_argument("--tokens", choices=["history", "set", "diff"], default="history",
                    help="history: ordered, branch-step embedded; set: the history events "
                         "unordered; diff: net differences from the root (NOTES s156)")
    ap.add_argument("--token-drop", type=float, default=0.0,
                    help="training-time dropout of whole input tokens")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global TOKENS, TOKEN_DROP
    TOKENS, TOKEN_DROP = a.tokens, a.token_drop
    GEN.manual_seed(a.seed)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exs = load_examples(a.examples, AA20, "BA.2", source=SRC)
    attach_histories(exs)
    tr, va = split_examples({SRC: exs}, "random", 0.1, a.split_seed)
    tr, va = tr[SRC], va[SRC]
    L = len(exs[0].aa_idx)
    print(f"{len(tr)} train / {len(va)} holdout examples; L={L}; history length "
          f"median {int(np.median([e.history.shape[0] for e in exs]))}, "
          f"max {max(e.history.shape[0] for e in exs)}", flush=True)

    if a.arch == "site":
        model = HistorySiteTransformer(L, len(AA20), a.d_model, a.layers, a.heads, a.d_ff,
                                       a.dropout, a.n_cross).to(device)
    else:
        model = HistoryTransformer(L, len(AA20), a.d_model, a.layers, a.heads, a.d_ff,
                                   a.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: {n_par:,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    rng = random.Random(a.seed)
    best, best_ep, bad, log = math.inf, -1, 0, []
    print(f"epoch -1 holdout NLL/mut {evaluate(model, va, 256, device):.4f}", flush=True)
    for ep in range(a.epochs):
        model.train()
        t0, tl, tn = time.time(), 0.0, 0
        for b in batches(tr, a.batch_size, True, rng):
            aa, mat, hist, pad = tensors(b, device, train=True)
            loss = compute_loss_batch(compute_probs(model(aa, hist, pad).float(), mat, aa), mat)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += float(loss); tn += 1
        v = evaluate(model, va, 256, device)
        log.append(dict(epoch=ep, train_loss=tl / tn, val_nll=v, sec=time.time() - t0))
        star = ""
        if v < best:
            best, best_ep, bad, star = v, ep, 0, "  *best"
            torch.save(dict(config=model.config(), model_state_dict=model.state_dict(), tokens=a.tokens,
                            epoch=ep, val_nll=v, args=vars(a)),
                       os.path.join(a.out, "history_best.pt"))
        else:
            bad += 1
        print(f"epoch {ep:3d}  train {tl / tn:.4f}  holdout NLL/mut {v:.4f}  "
              f"({time.time() - t0:.0f}s){star}", flush=True)
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
        if bad >= a.patience:
            print(f"early stop: no holdout improvement for {a.patience} epochs", flush=True)
            break
    print(f"best epoch {best_ep}, holdout NLL/mut {best:.4f}"
          + ("  ** BEST IS THE LAST EPOCH: budget binds" if best_ep == a.epochs - 1 else ""))


if __name__ == "__main__":
    main()
