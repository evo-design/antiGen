"""Figure 4 A-F: per-virus recall@10% (all + de novo) on the full and the single-nucleotide pool."""
import csv
import os

import numpy as np
import torch

from antiGen import statistical_baselines as sb
from antiGen.config import (AlphabetConfig, CodonBlosumConfig, ESM2Config, ModelConfig,
                            ModelType)
from antiGen.eval_utils import (fit_substitution_baseline, frequency_baseline_scores,
                                positional_marginal_counts)
from antiGen.load import load_examples
from antiGen.models.factory import create_model
from fig4_common import (CKPT, CORPORA, ENDEMIC_CORPORA, OUT, REPO, SEEDS, A, E, best_ckpt, dev,
                         events_of, load_model, per_event_weights, pool_1nt, seen_mask)

TREES = {"denv2": ("denv2-2019-10-01", "2019-10-01", "dengue"),
         "rsv_a": ("rsv-a-2023-10-01", "2023-10-01", "rsv_a"),
         "h3n2": ("h3n2-2022-10-01", "2022-10-01", "flu")}
TEST = os.environ.get("FIG4_ENDEMIC_TEST", os.path.join(os.path.dirname(CORPORA), "endemic"))
EVE = os.path.join(REPO, "antiGen-figures", "data", "fig4", "eve")
ARMS = [("antiGen", "ensemble"), ("S×S–MCM Interp.", "cpssm_dir"), ("MCM", "frequency"),
        ("S×S", "site_prop"), ("S×M", "where_times_which"), ("PFM", "pssm"),
        ("Codon-aware B62", "codon_blosum"), ("ESM-2 zero-shot", "zeroshot"), ("EVE", "eve"),
        ("Codon-B62 Mixture", "mixture")]
COLS = ["virus", "tree", "subset", "convention", "method", "n_events", "recall", "ci_lo", "ci_hi"]
B = 2000


def ci(w, rng):
    w = np.asarray(w, float)
    if not len(w):
        return float("nan"), float("nan"), float("nan"), 0
    bo = w[rng.integers(0, len(w), size=(B, len(w)))].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), len(w)


def ensemble_runs(tree):
    runs = []
    for s in SEEDS:
        try:
            runs.append(best_ckpt(f"{CKPT}/endemic/{tree}_xformer_s{s}"))
        except StopIteration:
            pass
    return runs


def map_fn(kind, train, L, tree):
    """Per-example (L, E) score function for one arm; None if the arm has no checkpoints."""
    st = sb.fit(train, L, E)
    if kind == "ensemble":
        paths = ensemble_runs(tree)
        print(f"[{tree}] antiGen: {len(paths)}/{len(SEEDS)} seeds", flush=True)
        if not paths:
            return None
        models = [load_model(p) for p in paths]

        def f(aa):
            with torch.no_grad():
                z = torch.stack([m(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu() for m in models])
            return torch.log_softmax(z.reshape(len(models), -1), -1).reshape(z.shape).mean(0)
        return f
    if kind in ("zeroshot", "codon_blosum"):
        cfg = (ModelConfig(model_type=ModelType.ESM2_FITNESS_PREDICTOR, seq_len=L,
                           esm2=ESM2Config(model_name="facebook/esm2_t33_650M_UR50D",
                                           freeze_backbone=True, finetune_layers=0))
               if kind == "zeroshot" else
               ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=L,
                           alphabet=AlphabetConfig(amino_acids=A),
                           codon_blosum=CodonBlosumConfig(kappa=4.0)))
        m = create_model(cfg, dev).eval()

        def f(aa):
            with torch.no_grad():
                return m(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
        return f
    if kind in ("where_times_which", "mixture"):
        sub = torch.log_softmax(fit_substitution_baseline(train, dev, A).score_matrix().detach().cpu(), -1)
        if kind == "mixture":
            return lambda aa: sub[aa]
        where = torch.log(positional_marginal_counts(train, L, E).clamp(min=1e-12)).view(L, 1)
        return lambda aa: sub[aa] + where
    if kind == "frequency":
        freq = torch.log(frequency_baseline_scores(train, L, E))
        return lambda aa: freq
    if kind == "eve":
        eve = torch.tensor(np.genfromtxt(f"{EVE}/{tree}.csv", delimiter=",", skip_header=1),
                           dtype=torch.float32)
        assert eve.shape == (L, E)
        return lambda aa: eve
    return lambda aa: sb.score_map(kind, st, aa.unsqueeze(0)).squeeze(0)


def main():
    rng = np.random.default_rng(0)
    rows = []
    for tree, (build, cut, virus) in TREES.items():
        train = load_examples(f"{ENDEMIC_CORPORA}/{build}/train/examples_{cut}_cur.txt", A, None, source=tree)
        test = load_examples(f"{TEST}/examples_{tree}@{cut}_test.txt", A, None, source=tree)
        L = len(train[0].aa_idx)
        seen = seen_mask(train)
        recs = []
        for ex in test:
            ev = events_of(ex)
            if not ev:
                continue
            reach = pool_1nt(ex.aa_idx)
            ev1 = [(p, a) for p, a in ev if reach[p, a]]
            recs.append((ex.aa_idx, reach, ev, ev1,
                         np.array([not seen[p, a] for p, a in ev], bool),
                         np.array([not seen[p, a] for p, a in ev1], bool)))
        for lab, kind in ARMS:
            f = map_fn(kind, train, L, tree)
            if f is None:
                continue
            acc = {k: [] for k in ("all_base", "all_1nt", "dn_base", "dn_1nt")}
            for aa, reach, ev, ev1, dn, dn1 in recs:
                sm = f(aa)
                acc["all_base"] += per_event_weights(sm, aa, ev)
                acc["dn_base"] += list(np.asarray(per_event_weights(sm, aa, ev, strike=seen))[dn])
                if ev1:
                    acc["all_1nt"] += per_event_weights(sm, aa, ev1, pool=reach)
                    acc["dn_1nt"] += list(np.asarray(per_event_weights(sm, aa, ev1, strike=seen, pool=reach))[dn1])
            del f
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            for subset, conv, key in [("all", "baseline", "all_base"), ("all", "onent", "all_1nt"),
                                      ("denovo", "baseline", "dn_base"), ("denovo", "onent", "dn_1nt")]:
                m, lo, hi, n = ci(acc[key], rng)
                rows.append([virus, tree, subset, conv, lab, n, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
            print(f"[{tree}] {lab}: all {np.mean(acc['all_base']):.3f}/{np.mean(acc['all_1nt']):.3f}  "
                  f"denovo {np.mean(acc['dn_base']):.3f}/{np.mean(acc['dn_1nt']):.3f}  "
                  f"n {len(acc['all_base'])}/{len(acc['all_1nt'])} dn {len(acc['dn_base'])}/{len(acc['dn_1nt'])}",
                  flush=True)
        for subset in ("all", "denovo"):
            for conv in ("baseline", "onent"):
                rows.append([virus, tree, subset, conv, "Uniform", 0, "0.1000", "0.1000", "0.1000"])
    os.makedirs(OUT, exist_ok=True)
    for name, conv in (("fig4_af_panels.csv", "baseline"), ("fig4_1nt_panels.csv", "onent")):
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            w.writerows(r for r in rows if r[3] == conv)


if __name__ == "__main__":
    main()
