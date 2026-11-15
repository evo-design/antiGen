"""Shared paths, model loading and the per-event recall@10 kernel for the figure 4 scorers."""
import glob
import os

import torch

from antiGen.config import STANDARD_AAS
from antiGen.eval_utils import load_checkpoint, top_q_weights
from antiGen.load import load_examples
from antiGen.models.codon_blosum_fitness_predictor import _codon_conditional
from antiGen.models.factory import create_model

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENDEMIC_CORPORA = os.environ.get("CORPORA", "/home/ivan.specht/antiGen-data-revised/output")
CORPORA = os.environ.get("FIG4_CORPORA", f"{ENDEMIC_CORPORA}/fig4-noattach")
TREES = os.environ.get("FIG4_TREES", f"{ENDEMIC_CORPORA}/fig4-trees")
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
    """A multi-tree corpus: the training snapshot (train) or the post-D queries (test)."""
    ex = load_examples(f"{CORPORA}/examples_{tree}@{D}_{split}.txt", A, None, source=tree,
                       split_date=D if split == "test" else None)
    return ex if split == "train" else [e for e in ex if not e.is_training]


def endemic_test(build, cut, tree):
    """The post-cutoff records of an A-F test tree."""
    ex = load_examples(f"{ENDEMIC_CORPORA}/{build}-test/examples_all_cur.txt", A, None, source=tree,
                       split_date=cut)
    return [e for e in ex if not e.is_training]


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
    """Fractional top-q hit weight of each event, by the shared rule (eval_utils.top_q_weights).

    The candidate pool is every non-wild-type cell, less `strike` and outside `pool`. A cell
    in the pool the model cannot score (NaN or -inf) ranks last, stays in the pool and is
    never credited; the boundary selects exactly q * n cells, ties split fractionally.
    """
    s = scores.detach().float().cpu()
    cand = torch.ones(s.shape, dtype=torch.bool)
    cand[torch.arange(len(aa_idx)), aa_idx] = False
    if strike is not None:
        cand &= ~strike.cpu()
    if pool is not None:
        cand &= pool.cpu()
    w, _ = top_q_weights(s, q, pool_mask=cand)
    return [float(w[pos, aa]) for pos, aa in events]
