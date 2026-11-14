"""Shared paths, model loading and the per-event recall@10 kernel for the figure 4 scorers."""
import glob
import os

import torch

from antiGen.config import STANDARD_AAS
from antiGen.eval_utils import load_checkpoint
from antiGen.load import load_examples
from antiGen.models.codon_blosum_fitness_predictor import _codon_conditional
from antiGen.models.factory import create_model

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENDEMIC_CORPORA = os.environ.get("CORPORA", "/home/ivan.specht/antiGen-data-revised/output")
CORPORA = os.environ.get("FIG4_CORPORA", f"{ENDEMIC_CORPORA}/fig4-noattach")
CKPT = os.environ.get("FIG4_CKPT", os.environ.get("CKPT", "/large_storage/hielab/ispecht/fitness-inference") + "/fig4")
FIGDATA = os.environ.get("FIGDATA", os.path.join(REPO, "antiGen-figures", "data"))
OUT = os.path.join(FIGDATA, "fig4")

A = STANDARD_AAS
E = len(A)
D = "2025-01-01"
Q = 0.1
PANEL9 = ["h5nx", "ncov", "rsv_a", "rsv_b", "measles", "h3n2_ha", "h1n1pdm_ha", "vic_ha", "hmpv_a_g"]
SEEDS = range(10)
REACH_1NT = torch.tensor(_codon_conditional(kappa=1.0, epsilon=0.0) > 0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def corpus(tree, split):
    """Examples for a panel tree; split is train, test (all post-D events) or novel."""
    p = f"{CORPORA}/examples_{tree}@{D}_{split}.txt"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return []
    return load_examples(p, A, None, source=tree)


def best_ckpt(run_dir):
    return next(p for p in glob.glob(f"{run_dir}/fitness_*_best.pt") if "_best_" not in os.path.basename(p))


def load_model(path):
    ck, cfg = load_checkpoint(path, map_location="cpu")
    m = create_model(cfg, torch.device("cpu"))
    m.load_state_dict(ck["model_state_dict"], strict=False)
    return m.to(dev).eval()


def events_of(ex):
    """Observed child mutations of one record as (pos, aa) pairs, with multiplicity."""
    out = []
    for pos, aa in torch.nonzero(ex.mat > 0).tolist():
        out += [(pos, aa)] * int(round(float(ex.mat[pos, aa])))
    return out


def seen_mask(train):
    """Cells observed anywhere in training (pssm_root > 0), struck from the de-novo pool."""
    return sum(ex.mat for ex in train) > 0


def pool_1nt(aa_idx):
    return REACH_1NT[aa_idx]


def per_event_weights(scores, aa_idx, events, strike=None, pool=None, q=Q):
    """Fractional top-q hit weight of each event, with compute_hit_rate's tie-break.

    Wild-type cells, `strike` cells and cells outside `pool` are removed from the
    candidates before the top-q boundary is drawn.
    """
    s = scores.detach().float().cpu().clone()
    s[torch.arange(len(aa_idx)), aa_idx] = float("nan")
    if strike is not None:
        s[strike] = float("nan")
    if pool is not None:
        s[~pool] = float("nan")
    flat = s.flatten()
    valid = ~torch.isnan(flat)
    nv = int(valid.sum())
    if nv == 0:
        return [0.0] * len(events)
    vb = torch.nanquantile(flat, 1.0 - q, interpolation="lower")
    n_above = int(((flat > vb) & valid).sum())
    n_tie = int(((flat == vb) & valid).sum())
    frac = max(0.0, q * nv - n_above) / n_tie if n_tie else 0.0
    w = []
    for pos, aa in events:
        v = s[pos, aa]
        w.append(0.0 if torch.isnan(v) else 1.0 if v > vb else frac if v == vb else 0.0)
    return w
