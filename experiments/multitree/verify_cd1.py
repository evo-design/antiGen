#!/usr/bin/env python
"""Independent check of list_macro --cd1-variants (NOTES s159).

Recomputes, from scratch, top-decile recall on the all-RBD-cells arm at 2023-01-01:
  C     targets all cells,       pool every non-wild-type cell
  E     targets de novo cells,   pool the de novo cells
  C1/E1 the same, with pool AND targets restricted per query to residues one nucleotide
        change from that query's wild type
and compares with list_macro's summary. Shares only data and model LOADING with
list_macro: reachability (own genetic-code walk), the pools, the fractional top-q
selection (exact expected hits under uniform random tie-breaking) and the per-cell /
per-mutation pooling are written here independently.

Models: a constant score (must give exactly q), codon (own route count), mcm_interpolation,
simple_d128 (the single antiGen model) and the antiGen CV x10 mean of log-probs.
"""
import csv
import os
import glob
import itertools
import sys

import numpy as np
import torch

from antiGen.eval_utils import compute_hit_rate, load_checkpoint
from antiGen.load import load_counts_matrix, load_examples
from antiGen.models.factory import create_model
from antiGen import statistical_baselines as sb

AA = "ACDEFGHIKLMNPQRSTVWY"
E = 20
REV = "/home/ivan.specht/antiGen-data-revised/output"
STORE = "/large_storage/hielab/ispecht/fitness-inference"
D = "2023-01-01"
Q = 0.1
R0, R1 = 330, 531                      # RBD 331-531, 1-based -> 0-based half-open
SUMMARY = f"{STORE}/evals_cd1_2023-01-01/out/list_macro_summary.csv"

# ---- genetic code, written out independently of codon_fitness_predictor ----
bases = "TCAG"
code = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
table = {a + b + c: code[16 * i + 4 * j + k] for (i, a), (j, b), (k, c)
         in itertools.product(enumerate(bases), repeat=3)}
assert table["ATG"] == "M" and table["TGG"] == "W" and table["TAA"] == "*" and table["GGG"] == "G"
syn = {a: [c for c, x in table.items() if x == a] for a in AA}
routes = np.zeros((E, E))
for i, w in enumerate(AA):
    for c in syn[w]:
        for p in range(3):
            for b in bases:
                if b != c[p]:
                    m = table[c[:p] + b + c[p + 1:]]
                    if m in AA:
                        routes[i, AA.index(m)] += 1.0 / len(syn[w])
np.fill_diagonal(routes, 0)
reach = routes > 0
print("reachable substitutions:", int(reach.sum()))      # expect 150
assert int(reach.sum()) == 150


def expected_hits(s, t):
    """s, t: 1-D pool scores and target counts. Expected hit mass when exactly q*n cells
    are selected, ties at the boundary broken uniformly at random."""
    n = len(s)
    k = Q * n
    order = np.argsort(-s, kind="stable")
    s, t = s[order], t[order]
    sel = np.zeros(n)
    used = 0.0
    i = 0
    while i < n and used < k - 1e-12:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        g = j - i + 1
        take = min(g, k - used)
        sel[i:j + 1] = take / g
        used += take
        i = j + 1
    out_h, out_t = np.empty(n), np.empty(n)
    out_h[order], out_t[order] = sel * t, t                  # back to the pool's order
    return out_h, out_t


def main():
    pooled = []
    for p in (f"{REV}/sc2-{D}/train/examples_{D}_cur.txt", f"{REV}/sc2-test/examples_all_cur.txt"):
        pooled += load_examples(p, AA, "BA.2", split_date=D)
    train = [e for e in pooled if e.is_training]
    test = [e for e in pooled if not e.is_training]
    seen = (load_counts_matrix(f"{REV}/sc2-{D}/train/counts_{D}_cur.tsv", AA) > 0)
    seen = np.asarray(seen)[R0:R1]
    print(f"{len(train)} train, {len(test)} test; seen RBD cells {int(seen.sum())}")
    test = [e for e in test if float(e.mat[R0:R1].sum()) > 0]
    print(f"{len(test)} test queries with an RBD target")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = sb.fit(train, 1273, E)
    def load(p):
        ck, cfg = load_checkpoint(p, map_location=dev)
        m = create_model(cfg, dev); m.load_state_dict(ck["model_state_dict"], strict=False)
        return m.eval()
    simple = load(glob.glob(f"{STORE}/cutoff_stage_aa20/{D}/fitness_simple_d128_*.pt")[0])
    members = [load(p) for p in sorted(glob.glob(f"{STORE}/ckpt_stage_vs10_{D}/fitness_*.pt"))]
    logroutes = np.round(np.log(routes + 0.01), 9)     # exact ties stay exact

    @torch.no_grad()
    def scores(name, aa):                                   # aa: (B, L) -> (B, R, E) numpy
        x = aa.to(dev)
        if name == "constant":
            return np.zeros((aa.shape[0], R1 - R0, E))
        if name == "codon":
            return logroutes[aa[:, R0:R1].numpy()]
        if name == "mcm_interpolation":
            return sb.score_map("mcm_interpolation", st, aa)[:, R0:R1].numpy()
        if name == "simple_d128_n2_lr0.0001":
            return simple(x).float()[:, R0:R1].cpu().numpy()
        if name == "ens_vs10_2023":
            s = sum(torch.log_softmax(m(x).float()[:, R0:R1].flatten(1), 1) for m in members) / len(members)
            return s.view(aa.shape[0], R1 - R0, E).cpu().numpy()

    import os
    MODELS = os.environ.get("VERIFY_MODELS", "constant,codon,mcm_interpolation,"
                            "simple_d128_n2_lr0.0001,ens_vs10_2023").split(",")
    VARS = ["C", "E", "C1", "E1", "C1t", "E1t"]
    acc = {(m, v): [np.zeros((R1 - R0, E)), np.zeros((R1 - R0, E))] for m in MODELS for v in VARS}
    rr = np.arange(R1 - R0)
    CHR = bool(os.environ.get("VERIFY_CHR"))
    acc2 = {(m, v): [np.zeros((R1 - R0, E)), np.zeros((R1 - R0, E))] for m in MODELS for v in VARS}
    for b in range(0, len(test), 128):
        batch = test[b:b + 128]
        aa = torch.stack([e.aa_idx for e in batch])
        tg = torch.stack([e.mat for e in batch])[:, R0:R1].numpy()
        S = {m: np.round(scores(m, aa).astype(np.float64), int(os.environ.get("VERIFY_ROUND", "12"))) for m in MODELS}
        for qi, e in enumerate(batch):
            wt = e.aa_idx[R0:R1].numpy()
            notwt = np.ones((R1 - R0, E), bool); notwt[rr, wt] = False
            r1 = reach[wt]                                   # (R, E) reachable from THIS wt
            pools = {"C": notwt, "E": notwt & ~seen, "C1": notwt & r1, "E1": notwt & ~seen & r1,
                     # diagnostic: targets restricted to 1-nt cells, pool NOT restricted
                     "C1t": notwt, "E1t": notwt & ~seen}
            for v, pool in pools.items():
                idx = np.nonzero(pool)
                t = tg[qi][idx] * (r1[idx] if v.endswith("t") else 1)
                if t.sum() == 0:
                    continue
                for m in MODELS:
                    h, ev = expected_hits(S[m][qi][idx], t)
                    acc[(m, v)][0][idx] += h
                    acc[(m, v)][1][idx] += ev
                    if CHR:
                        sc = torch.full((1, R1 - R0, E), float("nan"))
                        sc[0] = torch.from_numpy(S[m][qi]).float()
                        tt = torch.zeros(1, 1273, E); tt[0, R0:R1] = torch.from_numpy(tg[qi] * pool)
                        a1 = torch.zeros(1, 1273, dtype=torch.long); a1[0] = e.aa_idx
                        _h, _n, ch, ce = compute_hit_rate(sc, tt, a1, Q, eval_range=list(range(R0, R1)),
                                                          per_cell=True, pool_mask=torch.from_numpy(pool).unsqueeze(0))
                        acc2[(m, v)][0] += ch.cpu().numpy(); acc2[(m, v)][1] += ce.cpu().numpy()
        if b % 2560 == 0:
            print(f"  {b}/{len(test)}", flush=True)

    ref = {}
    name = {"C": "list_allcells_fullpool", "E": "list_allcells_unseen_fullpool",
            "C1": "list_allcells_cd1_fullpool", "E1": "list_allcells_unseen_cd1_fullpool", "C1t": "-", "E1t": "-"}
    for r in csv.DictReader(open(SUMMARY)):
        if r["qtile"] == "0.10":
            ref[(r["model"], r["variant"])] = (float(r["macro"]), float(r["micro"]),
                                                int(r["n_cells"]), float(r["n_events"]))
    worst = 0.0
    print(f"\n{'model':26s} {'var':3s} {'macro':>8s} {'micro':>8s} {'cells':>6s} {'events':>7s}   list_macro")
    for m in MODELS:
        for v in VARS:
            h, ev = acc[(m, v)]
            obs = ev > 0
            mac = float((h[obs] / ev[obs]).mean()); mic = float(h.sum() / ev.sum())
            line = f"{m:26s} {v:3s} {mac:8.4f} {mic:8.4f} {int(obs.sum()):6d} {ev.sum():7.0f}"
            if (m, name[v]) in ref:
                rm, ri, rc, re_ = ref[(m, name[v])]
                d = max(abs(mac - rm), abs(mic - ri))
                worst = max(worst, d)
                line += f"   {rm:.4f} {ri:.4f} {rc:6d} {re_:7.0f}  diff {d:.1e}"
            if CHR:
                h2, e2 = acc2[(m, v)]; o2 = e2 > 0
                line += f"   | compute_hit_rate on these scores: {float((h2[o2] / e2[o2]).mean()):.4f} {float(h2.sum() / e2.sum()):.4f}"
            print(line)
    print(f"\nworst difference against list_macro: {worst:.2e}")


if __name__ == "__main__":
    sys.exit(main())
