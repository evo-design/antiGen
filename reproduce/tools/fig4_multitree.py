"""Figure 4 H-K: nine-virus macro ladders and paired forests, full and single-nucleotide pools."""
import argparse
import csv
import gc
import glob
import os

import numpy as np
import torch

from antiGen import statistical_baselines as sb
from antiGen.config import (AlphabetConfig, CodonBlosumConfig, ESM2Config, ModelConfig,
                            ModelType)
from antiGen.eval_utils import fit_substitution_baseline, frequency_baseline_scores
from antiGen.models.factory import create_model
from fig4_common import (A, CKPT, E, OUT, PANEL9, Q, REACH_1NT, SEEDS, best_ckpt, corpus, dev,
                         events_of, load_model, per_event_weights, pool_1nt, seen_mask)

SCORES = os.environ.get("FIG4_SCORES", f"{CKPT}/scores/multitree")
NBOOT = 2000
BATCH = int(os.environ.get("FIG4_BATCH", "8"))
ENSEMBLES = {"joint": "joint49_esm650/s{s}", "joint9": "joint9_esm650/s{s}",
             "single": "single9_esm650/{t}_s{s}", "xformer": "single9_xformer/{t}_s{s}"}
COUNTS = ["cpssm_dir", "site_prop", "where_times_which", "pssm", "frequency"]
LADDER = list(ENSEMBLES) + ["cpssm_dir", "site_prop", "where_times_which", "codon_blosum",
                            "pssm", "frequency", "zeroshot"]
SUBSETS = (("all", "test"), ("novel", "novel"))
CONVS = ("baseline", "onent")
FORESTS = (("single", "single"), ("smoothed", "cpssm_dir"))


def tree_data(t):
    """Train corpus, seen mask, per-subset (aa_idx, events) and the unique parent sequences."""
    train = corpus(t, "train")
    subs = {}
    for sub, split in SUBSETS:
        ent = [(ex.aa_idx, ev) for ex in corpus(t, split) if (ev := events_of(ex))]
        if ent:
            subs[sub] = ent
    index, seqs = {}, []
    for ent in subs.values():
        for aa, _ in ent:
            k = aa.numpy().tobytes()
            if k not in index:
                index[k] = len(seqs)
                seqs.append(aa)
    return dict(train=train, seen=seen_mask(train), subs=subs, index=index, seqs=seqs)


def weights(td, map_of):
    """Per-event hit weights keyed subset__convention; map_of(aa) gives the (L, E) score map."""
    out = {}
    for sub, ent in td["subs"].items():
        strike = None if sub == "all" else td["seen"]
        wb, w1 = [], []
        for aa, ev in ent:
            mp = map_of(aa)
            ev1 = [(p, a) for p, a in ev if REACH_1NT[aa[p], a]]
            wb += per_event_weights(mp, aa, ev, strike=strike)
            w1 += per_event_weights(mp, aa, ev1, strike=strike, pool=pool_1nt(aa))
        out[f"{sub}__baseline"] = np.asarray(wb, np.float32)
        out[f"{sub}__onent"] = np.asarray(w1, np.float32)
    return out


@torch.no_grad()
def add_logprobs(model, td, acc):
    """Accumulate the model's flattened log-softmax map over every unique sequence of a tree."""
    seqs = td["seqs"]
    for i in range(0, len(seqs), BATCH):
        o = model(torch.stack(seqs[i:i + BATCH]).to(dev)).float()
        acc[i:i + BATCH] += torch.log_softmax(o.flatten(1), -1).view_as(o).cpu()


def free(model):
    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def mean_logprob(td_all, jobs):
    """Mean-log-prob ensemble maps per tree; jobs maps a model builder to the trees it scores."""
    acc = {t: torch.zeros(len(td_all[t]["seqs"]), len(td_all[t]["seqs"][0]), E) for t in td_all}
    n = {t: 0 for t in td_all}
    for build, trees in jobs:
        m = build()
        for t in trees:
            add_logprobs(m, td_all[t], acc[t])
            n[t] += 1
        free(m)
    return {t: acc[t] / n[t] for t in td_all if n[t]}, n


def ensemble_jobs(rung, trees):
    found = {}
    for t in trees:
        for s in SEEDS:
            d = f"{CKPT}/{ENSEMBLES[rung].format(t=t, s=s)}"
            if glob.glob(f"{d}/fitness_*_best.pt"):
                found.setdefault(best_ckpt(d), []).append(t)
    return [(lambda p=p: load_model(p), ts) for p, ts in sorted(found.items())]


def zeroshot():
    cfg = ModelConfig(model_type=ModelType.ESM2_FITNESS_PREDICTOR, alphabet=AlphabetConfig(amino_acids=A),
                      esm2=ESM2Config(model_name="facebook/esm2_t33_650M_UR50D", hidden_size=1280))
    return create_model(cfg, dev).eval()


def codon_blosum(L):
    cfg = ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=L, alphabet=AlphabetConfig(amino_acids=A),
                      codon_blosum=CodonBlosumConfig(kappa=4.0))
    return create_model(cfg, dev).eval()


def count_maps(td, rung):
    train = td["train"]
    L = len(train[0].aa_idx)
    if rung == "frequency":
        f = torch.log(frequency_baseline_scores(train, L, E))
        return lambda aa: f
    st = sb.fit(train, L, E)
    if rung == "where_times_which":
        sub = torch.log_softmax(fit_substitution_baseline(train, dev, A).score_matrix().detach().cpu(), -1)
        return lambda aa: sb.site_mixture_map(st, aa[None], sub)[0]
    return lambda aa: sb.score_map(rung, st, aa[None])[0]


def save(rung, td_all, map_of_tree, n_models=None):
    os.makedirs(SCORES, exist_ok=True)
    W = {}
    for t in PANEL9:
        if t in td_all and (m := map_of_tree(t)) is not None:
            W.update({f"{t}__{k}": v for k, v in weights(td_all[t], m).items()})
    np.savez_compressed(f"{SCORES}/{rung}.npz", **W)
    if n_models is not None:
        with open(f"{SCORES}/{rung}_n_seeds.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["virus", "n_seeds"])
            w.writerows([t, n_models.get(t, 0)] for t in PANEL9)
    print(f"{rung}: {sorted({k.split('__')[0] for k in W})}"
          + (f" n_seeds={[n_models.get(t, 0) for t in PANEL9]}" if n_models else ""), flush=True)


def score(rungs):
    td_all = {t: tree_data(t) for t in PANEL9}
    for rung in rungs:
        if rung in ENSEMBLES:
            jobs = ensemble_jobs(rung, PANEL9)
            if not jobs:
                print(f"{rung}: no checkpoints", flush=True)
                continue
            maps, n = mean_logprob(td_all, jobs)
        elif rung == "zeroshot":
            maps, n = mean_logprob(td_all, [(zeroshot, PANEL9)])
        elif rung == "codon_blosum":
            maps, n = mean_logprob(td_all, [(lambda t=t: codon_blosum(len(td_all[t]["seqs"][0])), [t])
                                            for t in PANEL9])
        else:
            save(rung, td_all, lambda t: count_maps(td_all[t], rung))
            continue
        lookup = lambda t: (lambda aa, t=t: maps[t][td_all[t]["index"][aa.numpy().tobytes()]]) if t in maps else None
        save(rung, td_all, lookup, n if rung in ENSEMBLES else None)


def boot_ci(v):
    a = np.asarray(v, float)
    if a.size == 0:
        return float("nan"), float("nan")
    m = a[np.random.default_rng(0).integers(0, a.size, (NBOOT, a.size))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def macro(rv, clamp):
    est = sum(r[0] for r in rv) / len(rv)
    s = sum(((hi - lo) / (2 * 1.96)) ** 2 for _, lo, hi in rv) ** 0.5 / len(rv)
    lo, hi = est - 1.96 * s, est + 1.96 * s
    return (est, max(0.0, lo), min(1.0, hi)) if clamp else (est, lo, hi)


def fmt(x):
    return f"{x:.6f}"


def build():
    W = {r: dict(np.load(f"{SCORES}/{r}.npz")) for r in LADDER if os.path.exists(f"{SCORES}/{r}.npz")}
    print("missing rungs:", [r for r in LADDER if r not in W], flush=True)
    os.makedirs(OUT, exist_ok=True)

    def ladder_rows(sub, conv):
        rows = []
        for r in W:
            rv = [(float(v.mean()), *boot_ci(v)) for t in PANEL9
                  if len(v := W[r].get(f"{t}__{sub}__{conv}", np.array([])))]
            if rv:
                rows.append([r, *map(fmt, macro(rv, True)), len(rv)])
        return rows + [["uniform", fmt(0.1), fmt(0.1), fmt(0.1), 0]]

    with open(f"{OUT}/ladder_fig4_j9.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "subset", "est", "lo", "hi", "n_virus"])
        for sub, _ in SUBSETS:
            w.writerows([r[0], sub, *r[1:]] for r in ladder_rows(sub, "baseline"))
    with open(f"{OUT}/ladder_fig4_j9_1nt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "subset", "convention", "est", "lo", "hi", "n_virus"])
        for sub, _ in SUBSETS:
            for conv in CONVS:
                w.writerows([r[0], sub, conv, *r[1:]] for r in ladder_rows(sub, conv))

    for conv, name in (("baseline", "forest_fig4_j9.csv"), ("onent", "forest_fig4_j9_1nt.csv")):
        rows = []
        for contrast, comp in FORESTS:
            if "joint" not in W or comp not in W:
                continue
            pooled = {}
            for t in PANEL9:
                for sub, _ in SUBSETS:
                    k = f"{t}__{sub}__{conv}"
                    if k not in W["joint"] or k not in W[comp]:
                        continue
                    d = W["joint"][k] - W[comp][k]
                    r = (float(d.mean()), *boot_ci(d))
                    rows.append([contrast, t, sub, *map(fmt, r), 0])
                    pooled.setdefault(sub, []).append(r)
            rows += [[contrast, "POOLED", sub, *map(fmt, macro(rv, False)), 1] for sub, rv in pooled.items()]
        with open(f"{OUT}/{name}", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["contrast", "tree", "subset", "est", "lo", "hi", "summary"])
            w.writerows(rows)
    print(f"wrote {OUT}/{{ladder,forest}}_fig4_j9{{,_1nt}}.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "build"])
    ap.add_argument("--rungs", nargs="+", default=LADDER, choices=LADDER)
    a = ap.parse_args()
    score(a.rungs) if a.step == "score" else build()
