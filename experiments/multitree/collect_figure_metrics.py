#!/usr/bin/env python
"""Score both evaluation regimes on the same trees, two metrics, two subsets.

Writes the tidy CSVs behind Figure 5. Everything is computed here rather than read
back from the per-run JSONs, so the two regimes go through byte-identical scoring
code and the same definition of "novel".

Regimes (same trees, same evaluation trees; --panel picks 9 or 11 of them):
  random    each tree's random held-out split. Not forecast-valid -- a held-out node's
            parent is usually in the training half.
  temporal  the D = 2025-01-01 holdout: trained on the build as published at D,
            scored on substitutions that did not exist then.
  What is deliberately NOT matched: the random arm trains on today's full builds and
  the temporal arm on the D snapshot. That difference is the comparison, not a confound.

Subsets: `all` post-split substitutions, and `novel` -- those whose (position, alt)
cell never occurs among the training examples the model actually saw. The same rule is
applied in both regimes, so the two columns mean the same thing. (For the temporal
regime this is very slightly different from the published sidecar count, which keyed
novelty off the snapshot *tree* rather than the generated examples; the script prints
both so the gap is visible.)

Metrics: NLL per mutation (lower better) and hit@10% (higher better), the latter with
`compute_hit_rate`'s fractional tie-breaking and the stop column NaN-masked for
predictors whose OUTPUTS_LEGITIMATE_STOP is False.

Seeds. Every seed-dependent rung is ENSEMBLED into one predictor by averaging its
log-probability maps -- the geometric mean of the maps -- rather than scored once per
seed with the metrics averaged afterwards. For the ESM-2 arms this is the largest
single effect measured on this arm (de novo recall 0.501 -> 0.586 at K = 8), so the
per-seed convention was understating both of them; it is applied to `joint` and
`single` at the SAME K, since an 8-seed joint ensemble against a 3-seed single one
would credit joint training with ensembling its control was denied. The fitted
`substitution` and `where_times_which` rungs carry a seeded minibatch sampler of
their own and are ensembled the same way over --wxw-seeds; they saturate by K = 2
(NOTES s14), so three is enough and the model's gains with K are not an artifact of
the baseline being fit once.

Ensembling is available in the TEMPORAL regime only. There every seed scores the same
post-D test set, so the maps are commensurable. In the random regime each seed carries
its own three-way split -- seed 0's test block is inside seed 1's training half -- so
there is no common test set to average over and no legitimate ensemble; that regime
keeps the per-seed convention, which is one more reason it sits in the supplement.

Epoch rules, both reported because the choice materially changes the effect size:
  pooled    the epoch with the best POOLED validation NLL -- one checkpoint for every
            tree, which is what --save-best ships and what you would actually deploy.
            Conservative for the joint model, which must compromise across trees.
  per-tree  each tree scored at ITS OWN best validation epoch (--save-best-per-source).
            Only the joint arm is affected: a single-tree run has one source, so its
            pooled best epoch already is its per-tree best, and one checkpoint serves
            both rules. Baselines do not depend on the epoch at all.
Selection uses validation data only under both rules, in both regimes.

Outputs `ladder.csv`, `deltas.csv`, `per_example.npz` and `agreement.csv`. The
last is for the overlap panels: per tree and subset, how many observed
substitutions fall in BOTH the model's and `where x which`'s top 10%, in exactly
one of them, or in neither. The ladders say which predictor scores higher and the
forests say by how much, but neither can tell a model that learned something the
baseline did not from one that rediscovered the baseline -- and on de novo
substitutions their recalls are close enough for that to matter.

Usage:
    python experiments/multitree/collect_figure_metrics.py [--out DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu                              # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

from antiGen.config import AlphabetConfig, CodonConfig, ModelConfig, ModelType  # noqa: E402
from antiGen.eval_utils import (fit_substitution_baseline,                 # noqa: E402
                                frequency_baseline_scores, load_checkpoint,
                                positional_marginal_counts,
                                where_only_baseline_scores, WhereTimesWhichPredictor)
from antiGen.load import load_examples                                     # noqa: E402
from antiGen.models.factory import create_model                            # noqa: E402
from calibrate_forecast import collect, temp_probs, AL, E                  # noqa: E402
from forecast_hit_summary import Q, STOP                                 # noqa: E402

GAP = AL.index('-')
## Selection budgets. 10% is the figure's long-standing headline; 1% is the
## actionable one -- a top-1% list is ~137 cells per genotype against ~1,370,
## which is the order of a shortlist someone would actually assay. They are
## reported side by side because a predictor can be better at the head of the
## list and worse further down: on the zero-shot panel the model leads the
## substitution matrix at 1-2% and loses from 5% on.
BUDGETS = {'hit10': 0.10, 'hit1': 0.01}

SEEDS = [0, 1, 2]               # random regime: one model per split, no ensemble
# Temporal regime: the seeds averaged into one ESM-2 predictor. `joint_panel` has 16
# and the single-tree controls have 8, so 8 is the matched number -- K = 16 is a
# scaling note for the joint arm alone and may not enter the joint-vs-single contrast.
ENS_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
WXW_SEEDS = [0, 1, 2]           # seeds of fit_substitution_baseline's sampler
D = '2025-01-01'
DATA = 'experiments/multitree/data'
MATCHED = '/large_storage/hielab/ispecht/multitree/matched'
FORECAST = '/large_storage/hielab/ispecht/multitree/forecast'
# joint_panel retrained with --save-best-per-source; the original forecast/
# runs are left untouched so the already-reported numbers still stand.
FORECAST_PT = '/large_storage/hielab/ispecht/multitree/forecast_pertree'
# The eleven-tree panel's joint arm: a separate root so the nine-tree runs
# behind the published figure are never overwritten or mixed in.
FORECAST11 = '/large_storage/hielab/ispecht/multitree/forecast11'

## Two panels, selected with --panel. `9` is the published Figure 5. `11` adds
## seasonal flu H3N2 HA and H1N1pdm HA, whose @2025-01-01 archives NOTES s8 wrongly
## ruled out (it searched `flu/seasonal`; the paths are `seasonal-flu/...`). Both are
## route A -- meta.updated 2024-12-29, every tip < 2025.0 -- and neither shares a tip
## name with the other or with h5nx HA. The panel goes 564 -> 724 post-D targets and
## 156 -> 196 novel, and h5nx falls from 41% to 32% of the test set.
##
## The joint arms differ per panel and live in their own roots, so the nine-tree runs
## behind the published figure are never read by the eleven-tree pass or vice versa.
## The nine shared single-tree controls ARE the same runs in both: a single-tree arm
## does not depend on the panel it is scored beside.
PANELS = {
    9: dict(trees=['ncov', 'rsv_a', 'rsv_b', 'h5nx', 'measles',
                   'denv1', 'denv2', 'denv3', 'denv4'],
            temporal_joint=FORECAST_PT + '/joint_panel',
            random_joint=MATCHED + '/joint9'),
    11: dict(trees=['ncov', 'rsv_a', 'rsv_b', 'h5nx', 'measles',
                    'denv1', 'denv2', 'denv3', 'denv4', 'h3n2_ha', 'h1n1pdm_ha'],
             temporal_joint=FORECAST11 + '/joint_panel11',
             random_joint=MATCHED + '/joint11'),
}
PANEL = PANELS[9]['trees']      # rebound from --panel in main()
RUNGS = ['uniform', 'codon', 'where_only', 'pssm', 'substitution', 'frequency',
         'where_times_which', 'single', 'joint']
METRICS = ['nll', 'hit10', 'hit1']
RULES = ['pooled', 'per-tree']
## Which cells every rung is scored over. The figure's headline uses `standard`.
##   standard  the 20 ordinary residues. Stop is ESM-2's <unk> logit reused, not
##             a prediction; gap is a column the ESM-2 arms put 0.00% of their
##             mass on, so including it scores them on a column they structurally
##             cannot use. 3 of 564 post-D targets are stop or degenerate and 13
##             are gap, so this is the cleanest like-for-like comparison.
##   common    standard plus the gap column, so the ESM-2 arms' gap deficit --
##             0 of 13 gap targets caught, against `where x which`'s 2/3 -- stays
##             visible. This is the supplement's version and the honest check on
##             the headline.
##   legacy    what this script did before: stop removed from the ESM-2 rungs'
##             ranking only. n_valid is counted after masking, so that left every
##             baseline ~5% more budget (1,436 cells against 1,367). Kept
##             reachable and labelled, not used by the figure.
UNIVERSES = ['standard', 'common', 'legacy']
ESM2_RUNGS = ('single', 'joint')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_dir(tree, seed):
    """Random-split single-tree control, three-way split.

    Only matched/ qualifies: the decomp/ controls used the two-way split, so their
    validation examples overlap this arm's test block and reusing them would leak.
    """
    p = f'{MATCHED}/single_{tree}_s{seed}'
    if not glob.glob(p + '/*_best.pt'):
        raise SystemExit(f'no three-way-split single run for {tree} seed {seed} ({p})')
    return p


def load_model(path, pat='*_best.pt'):
    hits = sorted(glob.glob(os.path.join(path, pat)))
    if not hits:
        raise SystemExit(f'missing {pat} in {path} -- was it trained with '
                         f'--save-best-per-source?')
    cp, cfg = load_checkpoint(hits[-1], map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def logp_model(logits, wt):
    """Normalized log-probabilities over non-wild-type cells, through the 1e-8 clamp.

    This is the space seed ensembling happens in. Averaging raw logits would be
    meaningless across independently trained networks -- their scales are not
    comparable -- while averaging probabilities is an arithmetic mixture, which is
    dominated by whichever seed is most confident and barely reorders the head of
    the list (NOTES s14). The mean of the log-probs is the geometric mean of the
    maps: every seed gets a veto, and every cell's rank moves.

    A one-element "ensemble" is a no-op downstream: softmax of a log-prob map
    returns the same map, so the single-seed numbers are unchanged by this path.
    """
    p = temp_probs(logits, wt, 1.0)
    p = p / p.reshape(p.shape[0], -1).sum(axis=1)[:, None, None]
    return np.log(np.clip(p, 1e-30, None))


def split_random(tree, seed):
    """(seen pool, test) for the random regime.

    Three-way split: the model trained on `train`, chose its epoch on `val`, and is
    reported on `test`, which nothing in training or tuning touched. Baselines are fit
    on train+val -- everything available before the test block -- which mirrors the
    temporal regime, where they are fit on the whole pre-D snapshot file.
    """
    from antiGen.train import held_out_split
    with contextlib.redirect_stdout(io.StringIO()):
        ex = load_examples(f'{DATA}/examples_{tree}_nocutoff.txt', AL, None, source=tree)
        tr, va, te = held_out_split({tree: ex}, 0.1, 0.1, seed)
    return tr[tree] + va[tree], te[tree]


def split_temporal(tree, _seed):
    with contextlib.redirect_stdout(io.StringIO()):
        tr = load_examples(f'{DATA}/forecast/examples_{tree}@{D}_train.txt', AL,
                           None, source=tree)
        te = load_examples(f'{DATA}/forecast/examples_{tree}@{D}_test.txt', AL,
                           None, source=tree)
    return tr, te


def seen_cells(train_ex, L):
    """Cells already present in training, by EITHER route.

    A cell counts as seen if some training branch mutated to that residue there,
    *or* if any training sequence simply carries that residue there.

    The second clause is not redundant. The consensus residue at a position is
    inherited from the root, not mutated into, so it leaves no event -- and a
    post-D reversion from a rare derived state back to consensus would then be
    labeled de novo despite being the single most predictable cell at that
    position. On the D = 2025-01-01 panel that was 12 of 157 event-novel
    targets, every one a reversion to a residue carried by 65-99.5% of the
    training sequences, and a pure composition baseline called all 12 correctly.
    De novo has to mean "this residue has not been seen here", not merely "no
    branch in the training window happened to mutate into it".
    """
    m = np.zeros((L, E), dtype=bool)
    idx = np.arange(L)
    for ex in train_ex:
        m[idx, ex.aa_idx.numpy()] = True                 # carried by this sequence
        for pos, alt in (ex.mat > 0).nonzero().tolist():
            m[pos, alt] = True                           # mutated to on this branch
    return m


def score_tree(model_scores, tg, aa, novel):
    """Per-example NLL and per-budget recall, for the `all` and `novel` subsets.

    Returns {subset: (loss, {metric: hits}, n)}. Excluded columns are already
    -inf in `logits`, so no mask flag is needed here any more.
    """
    out = {}
    for subset in ('all', 'novel'):
        t = tg if subset == 'all' else tg * torch.tensor(
            novel, device=tg.device, dtype=tg.dtype).unsqueeze(0)
        if float(t.sum()) == 0:
            continue
        probs = model_scores['probs']
        loss = -(t * torch.log(probs.clamp(min=1e-300))).sum((1, 2))
        hits = {m: hits_at_q(model_scores['logits'], t, aa, q).cpu().numpy()
                for m, q in BUDGETS.items()}
        out[subset] = (loss.cpu().numpy(), hits, t.sum((1, 2)).cpu().numpy())
    return out


def hits_at_q(scores, tg, aa, q):
    """`compute_hit_rate`'s fractional tie-breaking at an arbitrary budget.

    `per_example_hits` hard-codes Q = 0.1. Excluded columns arrive already set to
    -inf by the caller, which this turns into NaN, so one mechanism covers the
    universe and the budget.
    """
    s = scores.clone()
    s.scatter_(2, aa.unsqueeze(-1), float('nan'))
    s = torch.where(torch.isneginf(s), torch.full_like(s, float('nan')), s)
    B = s.shape[0]
    fl = s.reshape(B, -1).float()
    n_valid = (~torch.isnan(fl)).sum(1).clamp(min=1).float()
    vb = torch.nanquantile(fl, 1.0 - q, dim=1, interpolation='lower').view(B, 1, 1)
    above, at = s > vb, s == vb
    n_above = above.reshape(B, -1).sum(1).float()
    frac = ((q * n_valid - n_above)
            / at.reshape(B, -1).sum(1).float().clamp(min=1)).clamp(0.0, 1.0)
    return (above * tg).reshape(B, -1).sum(1) + frac * (at * tg).reshape(B, -1).sum(1)


def caught_mask(scores, aa, cols, q=Q):
    """Per-cell boolean: does this cell fall inside the predictor's top-10% budget?

    The ladder's `compute_hit_rate` splits a tied block at the budget boundary
    fractionally, which is the right answer for a rate but cannot say whether one
    particular cell was selected -- and an overlap panel has to label individual
    cells. This takes the strictly-above set instead. Where the scores are
    continuous the two decisions coincide; the marginals this produces are checked
    against the ladder's recall in main() and the gap is printed, so the panel
    cannot quietly drift from the column it sits under.
    """
    s = scores.clone()
    for c in cols:
        s[:, :, c] = float('nan')
    s.scatter_(2, aa.unsqueeze(-1), float('nan'))
    s = torch.where(torch.isneginf(s), torch.full_like(s, float('nan')), s)
    B = s.shape[0]
    fl = s.reshape(B, -1).float()
    vb = torch.nanquantile(fl, 1.0 - q, dim=1, interpolation='lower').view(B, 1, 1)
    return s > vb


def _resample(parts, nb, seed):
    """Pooled bootstrap: resample examples within each tree, then pool.

    `parts` is a list of (numerator, denominator) arrays, one per tree. Vectorized
    because the ladder alone needs 128 of these and the pure-Python version was the
    slowest step in the script by an order of magnitude.
    """
    rng = np.random.default_rng(seed)
    num = np.zeros(nb)
    den = np.zeros(nb)
    for a, n in parts:
        idx = rng.integers(0, len(a), size=(nb, len(a)))
        num += a[idx].sum(axis=1)
        den += n[idx].sum(axis=1)
    return num / np.maximum(den, 1e-9)


def boot(A, B, N, nb=10000, sd=0):
    """95% interval for a paired difference, pooled over one tree or many."""
    pt = (A.sum() - B.sum()) / max(N.sum(), 1e-9)
    d = _resample([(A - B, N)], nb, sd)
    return pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_pooled(parts, nb=10000, sd=0):
    """Same, resampling within each tree before pooling across them."""
    num = sum(float(a.sum()) for a, _ in parts)
    den = sum(float(n.sum()) for _, n in parts)
    d = _resample(parts, nb, sd)
    return num / max(den, 1e-9), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='antiGen-figures/data/multitree')
    p.add_argument('--panel', type=int, default=9, choices=sorted(PANELS),
                   help='which evaluation panel to score; see PANELS. 9 is the '
                        'published Figure 5, 11 adds the two seasonal-flu HA trees. '
                        'The joint arm comes from a different root per panel, so '
                        'the two never read each other\'s runs.')
    p.add_argument('--universes', default=','.join(UNIVERSES),
                   help='scoring universes to emit; see UNIVERSES. All three are '
                        'written to the CSVs with a `universe` column and the R '
                        'script picks, because they cost one extra masking pass '
                        'each over score maps that are already computed.')
    p.add_argument('--ens-seeds', default=','.join(map(str, ENS_SEEDS)),
                   help='seeds averaged into one ESM-2 predictor in the temporal '
                        'regime. The SAME set is used for `joint` and `single`; '
                        'K must be matched or the contrast credits joint training '
                        'with ensembling its control was denied.')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # PANEL is module-level because a dozen comprehensions below close over it; the
    # panel choice rebinds it once, here, before any of them run.
    global PANEL
    panel = PANELS[args.panel]
    PANEL = panel['trees']
    temporal_joint, random_joint = panel['temporal_joint'], panel['random_joint']
    print(f'panel: {len(PANEL)} trees ({", ".join(PANEL)})', flush=True)
    ens_seeds = [int(x) for x in args.ens_seeds.split(',')]
    universes = [u for u in args.universes.split(',') if u]
    missing = [f'{root}_s{s}' if nm is None else f'{root}/{nm}_s{s}'
               for s in ens_seeds
               for root, nm in [(temporal_joint, None)]
               + [(FORECAST, f'single_{t}') for t in PANEL]
               if not glob.glob((f'{root}_s{s}' if nm is None
                                 else f'{root}/{nm}_s{s}') + '/*_best.pt')]
    if missing:
        raise SystemExit('missing runs for the requested ensemble:\n  '
                         + '\n  '.join(missing))
    print(f'ESM-2 seed ensemble: K = {len(ens_seeds)} ({args.ens_seeds})', flush=True)

    # acc[(regime, rule, tree, subset, rung, metric)] = list of per-example arrays.
    # In the temporal regime each list holds ONE entry -- the seed ensemble is a
    # single predictor. In the random regime it holds one entry per split seed and
    # the metrics are averaged across them, the convention this script used
    # throughout before ensembling (see the module docstring for why that regime
    # cannot ensemble).
    acc, nvec = {}, {}
    # agree[(regime, rule, tree, subset)] = [both, model only, wxw only, neither],
    # summed over jobs; agree_n counts them so the random regime's three per-seed
    # splits are averaged rather than tripled.
    agree, agree_n = {}, {}
    for regime, joint_root, single_root in (
            ('Random holdout', random_joint, MATCHED),
            ('Temporal holdout', temporal_joint, FORECAST)):
        temporal = regime.startswith('Temporal')
        for tree in PANEL:
            # Each job is (seeds averaged into one predictor, the split to score
            # on). The split is built lazily: the random regime has one per seed and
            # an example carries a dense L x E target map, so materializing all
            # three of ncov's splits at once costs ~1 GB for no reason.
            jobs = ([(ens_seeds, lambda: split_temporal(tree, 0))] if temporal
                    else [([s], lambda s=s: split_random(tree, s)) for s in SEEDS])
            for mseeds, make_split in jobs:
                train_ex, eval_ex = make_split()
                if not eval_ex:
                    continue
                L = len(eval_ex[0].aa_idx)
                aa = torch.stack([e.aa_idx for e in eval_ex]).to(dev)
                tg = torch.stack([e.mat for e in eval_ex]).to(dev)
                novel = ~seen_cells(train_ex, L)

                ## The scoring universe. Under `common` every rung is scored over
                ## the same cells -- everything except the query's own residue and
                ## the stop column -- so q = 10% buys all of them the same number
                ## of cells. `legacy` masked stop for the ESM-2 rungs only, which
                ## gave each baseline ~5% more budget (1,436 cells against 1,367)
                ## over a useful universe that is identical for practical purposes.
                ##
                ## Stop goes for everyone because ESM-2's stop column is the <unk>
                ## logit reused, not a prediction, and because only 1 of 564 post-D
                ## targets is a stop. Gap STAYS for everyone: it is a real learned
                ## ESM-2 token and 2.5% of targets, and removing it would be
                ## choosing the convention that flatters the model, which is
                ## exactly the error this rule exists to prevent.
                ##
                ## Targets outside the universe are not scored at all, rather than
                ## left in to be charged at the probability floor: no predictor can
                ## reach them, so they measure nothing. That drops the 1 stop target
                ## and the 2 degenerate wild-type-to-wild-type targets (both h5nx;
                ## the training files contain none).
                base_wt = torch.ones_like(tg, dtype=torch.bool)
                base_wt.scatter_(2, aa.unsqueeze(-1), False)

                scores = {}
                ## The two fitted-substitution rungs. `fit_substitution_baseline`
                ## carries a seeded minibatch sampler, so they are seed-dependent
                ## predictors like the model and are ensembled the same way. They
                ## saturate by K = 2 (NOTES s14) -- the point is fairness, not a
                ## gain: the model's seeds must not be the only ones averaged.
                pm = positional_marginal_counts(train_ex, L, E)
                lps, wtm = {'substitution': [], 'where_times_which': []}, None
                for bs in WXW_SEEDS:
                    fit = fit_substitution_baseline(train_ex, dev, AL, seed=bs)
                    fac = WhereTimesWhichPredictor(fit.config, pm, fit).to(dev)
                    for nm, mdl in (('substitution', fit),
                                    ('where_times_which', fac)):
                        lo, _, wtm = collect(mdl, eval_ex, dev)
                        lps[nm].append(torch.tensor(logp_model(lo, wtm), device=dev))
                    del fit, fac
                for nm, maps in lps.items():
                    scores[nm] = (torch.stack(maps).mean(0), False)
                del lps

                cm = create_model(ModelConfig(
                    model_type=ModelType.CODON, seq_len=L,
                    alphabet=AlphabetConfig(amino_acids=AL), codon=CodonConfig()), dev)
                scores['codon'] = (torch.tensor(collect(cm, eval_ex, dev)[0],
                                                device=dev), False)
                del cm
                for nm, fn in (('frequency', frequency_baseline_scores),
                               ('where_only', where_only_baseline_scores)):
                    scores[nm] = (torch.log(fn(train_ex, L, E).to(dev)
                                            ).unsqueeze(0).expand(len(eval_ex), -1, -1),
                                  False)
                ## PSSM: residue composition of the training sequences,
                ## log(count + 1) -- the same construction as the repo's
                ## pssm_root baseline, but counted here rather than read from
                ## <data_dir>/pssms/root.csv. That artifact exists only for the
                ## paper's four viruses and, being precomputed over a fixed
                ## training window, has no notion of this arm's train/val/test
                ## split, so reusing it would leak the test block.
                ##
                ## Note what it counts: SEQUENCES carrying each residue, not
                ## substitution EVENTS. A mutation founding a large clade is
                ## counted once by `frequency` and once per descendant here.
                pssm = np.zeros((L, E), dtype=np.float32)
                for ex in train_ex:
                    pssm[np.arange(L), ex.aa_idx.numpy()] += 1.0
                scores['pssm'] = (
                    torch.log1p(torch.tensor(pssm, device=dev)
                                ).unsqueeze(0).expand(len(eval_ex), -1, -1), False)
                scores['uniform'] = (torch.zeros_like(scores['codon'][0]), False)

                def ensemble(dirs, pat='*_best.pt'):
                    """One predictor from a matched set of runs, pooled two ways.

                    Returns (log-linear, arithmetic). The log-linear pool -- the
                    mean of the log-prob maps, i.e. their normalized geometric mean
                    -- is the one the figure reports, because it is the pool that
                    reorders the head of the list and the ranking gain is what
                    NOTES s14 measured. The arithmetic pool, the mean of the
                    probability maps, is the textbook density ensemble: Jensen
                    guarantees its NLL is no worse than the mean of the seeds'. It
                    is carried only as a diagnostic, printed below, so the figure's
                    choice is checked against it instead of assumed. Reporting one
                    pool in the recall column and the other in the NLL column would
                    make the two columns describe different predictors.
                    """
                    lsum, psum = None, None
                    for d in dirs:
                        m = load_model(d, pat)
                        lo = collect(m, eval_ex, dev)[0]
                        del m
                        t = torch.tensor(logp_model(lo, wtm), device=dev)
                        lsum = t if lsum is None else lsum + t
                        psum = t.exp() if psum is None else psum + t.exp()
                    return (lsum / len(dirs),
                            torch.log((psum / len(dirs)).clamp(min=1e-30)))

                # A single-tree run has one source, so its pooled best epoch IS its
                # per-tree best; one checkpoint serves both rules.
                arms = [('single', '*_best.pt',
                         [single_dir(tree, s) if not temporal
                          else f'{single_root}/single_{tree}_s{s}' for s in mseeds])]
                arms += [(f'joint@{rule}', pat,
                          [f'{joint_root}_s{s}' for s in mseeds])
                         for rule, pat in (('pooled', '*_best.pt'),
                                           ('per-tree', f'*_best_{tree}.pt'))]
                for nm, pat, dirs in arms:
                    gm, am = ensemble(dirs, pat)
                    scores[nm] = (gm, True)
                    if len(dirs) > 1:
                        base, _, rl = nm.partition('@')
                        scores[f'{base}_arith' + (f'@{rl}' if rl else '')] = (am, True)

                ## Overlap of the two top-10% sets. The question this answers is
                ## not "which predictor is better" -- that is the forest rows --
                ## but whether they are the SAME predictor wearing two hats. A
                ## large `model only` / `wxw only` pair says they disagree about
                ## which amino-acid changes to flag, which is what makes their
                ## similar recalls a coincidence of totals rather than of content.
                nv = torch.tensor(novel, device=dev, dtype=tg.dtype).unsqueeze(0)
                for uni in universes:
                    ## Columns no rung may use under this universe. `legacy` is
                    ## the odd one out: it excludes stop from the ESM-2 rungs
                    ## only, which is what made the budgets differ.
                    def cols_for(rung_base):
                        if uni == 'standard':
                            return (STOP, GAP)
                        if uni == 'common':
                            return (STOP,)
                        return (STOP,) if rung_base in ESM2_RUNGS else ()

                    ## Targets outside the universe are dropped, not left to be
                    ## charged at the probability floor: no predictor can reach
                    ## them, so they measure nothing and only inflate every
                    ## rung's NLL equally. `legacy` keeps them, as it did.
                    keep = base_wt.clone()
                    if uni != 'legacy':
                        for c in cols_for('uniform'):
                            keep[:, :, c] = False
                    tg_u = tg * keep.to(tg.dtype)
                    wtm_u = wtm & keep.cpu().numpy()

                    for met, q in BUDGETS.items():
                        cw = caught_mask(scores['where_times_which'][0], aa,
                                         cols_for('where_times_which'), q)
                        for rule in RULES:
                            cm = caught_mask(scores[f'joint@{rule}'][0], aa,
                                             cols_for('joint'), q)
                            for subset, t in (('all', tg_u), ('novel', tg_u * nv)):
                                if float(t.sum()) == 0:
                                    continue
                                cell = [float((t * (cm & cw)).sum()),
                                        float((t * (cm & ~cw)).sum()),
                                        float((t * (~cm & cw)).sum()),
                                        float((t * (~cm & ~cw)).sum())]
                                k = (uni, met, regime, rule, tree, subset)
                                agree[k] = [a + b for a, b in
                                            zip(agree.get(k, [0.0] * 4), cell)]
                                agree_n[k] = agree_n.get(k, 0) + 1
                            del cm
                        del cw

                    for rung, (sc, _ms) in scores.items():
                        base, _, rule_of = rung.partition('@')
                        cols = cols_for(base)
                        ## Excluded columns are set to -inf rather than handed to
                        ## `per_example_hits`'s mask_stop flag, which only knows
                        ## about stop. The hit helper turns -inf into NaN itself,
                        ## so one mechanism covers both columns and both metrics.
                        sc_u = sc.clone()
                        for c in cols:
                            sc_u[:, :, c] = float('-inf')
                        pr = torch.tensor(temp_probs(sc_u.cpu().numpy(), wtm_u, 1.0),
                                          device=dev)
                        pr = pr / pr.reshape(pr.shape[0], -1).sum(1).clamp(
                            min=1e-30).view(-1, 1, 1)
                        got = score_tree({'probs': pr, 'logits': sc_u}, tg_u, aa,
                                         novel)
                        rules = [rule_of] if rule_of else RULES
                        for subset, (loss, hits, n) in got.items():
                            for rl in rules:
                                acc.setdefault(
                                    (uni, regime, rl, tree, subset, base, 'nll'),
                                    []).append(loss)
                                for met, hv in hits.items():
                                    acc.setdefault(
                                        (uni, regime, rl, tree, subset, base, met),
                                        []).append(hv)
                            nvec[(uni, regime, tree, subset)] = n
                        del sc_u, pr
                    del tg_u, keep
                del scores, nv
            print(f"  {regime} / {tree}", flush=True)

    mean = {k: np.mean(v, axis=0) for k, v in acc.items()}

    ## Is the figure's pooling rule the right one? Log-linear pooling is what the
    ## ranking work used; arithmetic pooling is what density theory recommends. If
    ## they disagree -- one column favouring each -- that has to be said out loud
    ## rather than settled by whichever was run first. Printed, not written: the
    ## `*_arith` rungs are a check on the choice, not rungs of the ladder.
    print('\n== seed-ensemble pooling rule: log-linear (figure) vs arithmetic ==')
    print(f'{"regime":18s} {"subset":7s} {"arm":8s} {"NLL loglin":>11s} '
          f'{"NLL arith":>10s} {"recall ll":>10s} {"recall ar":>10s}')
    for regime in ('Random holdout', 'Temporal holdout'):
        for subset in ('all', 'novel'):
            for arm in ('joint', 'single'):
                row, ok = [], True
                for rung in (arm, f'{arm}_arith'):
                    for metric in ('nll', 'hit10'):
                        u = universes[0]
                        ks = [(u, regime, 'pooled', t, subset, rung, metric)
                              for t in PANEL
                              if (u, regime, 'pooled', t, subset, rung,
                                  metric) in mean]
                        if not ks:
                            ok = False
                            break
                        num = sum(float(mean[k].sum()) for k in ks)
                        den = sum(float(nvec[(u, regime, k[3], subset)].sum())
                                  for k in ks)
                        row.append(num / max(den, 1e-9))
                if ok:
                    print(f'{regime:18s} {subset:7s} {arm:8s} {row[0]:11.3f} '
                          f'{row[2]:10.3f} {row[1]:10.3f} {row[3]:10.3f}')

    # ---- ladder.csv
    with open(f'{args.out}/ladder.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['universe', 'eval', 'rule', 'subset', 'metric', 'rung',
                    'value', 'lo', 'hi', 'n_mutations'])
        for uni in universes:
          for regime in ('Random holdout', 'Temporal holdout'):
            for rule in RULES:
                for subset in ('all', 'novel'):
                    trees = [t for t in PANEL if (uni, regime, t, subset) in nvec]
                    N = sum(nvec[(uni, regime, t, subset)].sum() for t in trees)
                    if not N:
                        continue
                    for rung in RUNGS:
                        for metric in METRICS:
                            ks = [(uni, regime, rule, t, subset, rung, metric)
                                  for t in trees]
                            if any(k not in mean for k in ks):
                                continue
                            ## NLL is reported as nats better than uniform. That
                            ## is a PAIRED difference -- uniform and the rung score
                            ## the same examples -- so the interval must be taken
                            ## on the difference. Bootstrapping the raw NLL and
                            ## then subtracting uniform's point estimate instead
                            ## carries uniform's own sampling variance, which gives
                            ## the uniform rung a wide interval around a quantity
                            ## that is identically zero.
                            if metric == 'nll':
                                parts = [
                                    (mean[(uni, regime, rule, t, subset, 'uniform',
                                           metric)] - mean[k],
                                     nvec[(uni, regime, t, subset)])
                                    for k, t in zip(ks, trees)]
                            else:
                                parts = [(mean[k], nvec[(uni, regime, t, subset)])
                                         for k, t in zip(ks, trees)]
                            est, lo, hi = boot_pooled(parts)
                            w.writerow([uni, regime, rule, subset, metric, rung,
                                        f'{est:.6f}', f'{lo:.6f}', f'{hi:.6f}',
                                        f'{N:.0f}'])

    # ---- deltas.csv
    with open(f'{args.out}/deltas.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['universe', 'eval', 'rule', 'subset', 'metric', 'contrast',
                    'tree', 'est', 'lo', 'hi', 'pooled'])
        for uni in universes:
         for regime in ('Random holdout', 'Temporal holdout'):
          for rule in RULES:
            for subset in ('all', 'novel'):
                for metric in METRICS:
                    for contrast, other in (('vs single-tree model', 'single'),
                                            ('vs where x which', 'where_times_which')):
                        A, B, N = [], [], []
                        for tree in PANEL:
                            ka = (uni, regime, rule, tree, subset, 'joint',
                                  metric)
                            kb = (uni, regime, rule, tree, subset, other, metric)
                            if ka not in mean or kb not in mean:
                                continue
                            a, b = mean[ka], mean[kb]
                            n = nvec[(uni, regime, tree, subset)]
                            est, lo, hi = boot(a, b, n)
                            w.writerow([uni, regime, rule, subset, metric, contrast,
                                        tree,
                                        f'{est:.6f}', f'{lo:.6f}', f'{hi:.6f}', 0])
                            A.append(a); B.append(b); N.append(n)
                        if not A:
                            continue
                        est, lo, hi = boot_pooled(
                            [(a - b, n) for a, b, n in zip(A, B, N)])
                        w.writerow([uni, regime, rule, subset, metric, contrast,
                                'POOLED',
                                    f'{est:.6f}', f'{lo:.6f}', f'{hi:.6f}', 1])
    # ---- agreement.csv: how much of each top-10% budget the two share
    with open(f'{args.out}/agreement.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['universe', 'budget', 'eval', 'rule', 'subset', 'tree',
                    'both', 'model_only', 'wxw_only', 'neither', 'n_mutations'])
        for uni in universes:
         for met in BUDGETS:
          for regime in ('Random holdout', 'Temporal holdout'):
            for rule in RULES:
                for subset in ('all', 'novel'):
                    tot = [0.0] * 4
                    for tree in PANEL:
                        k = (uni, met, regime, rule, tree, subset)
                        if k not in agree:
                            continue
                        v = [x / agree_n[k] for x in agree[k]]
                        tot = [a + b for a, b in zip(tot, v)]
                        w.writerow([uni, met, regime, rule, subset, tree]
                                   + [f'{x:.4f}' for x in v] + [f'{sum(v):.4f}'])
                    if sum(tot):
                        w.writerow([uni, met, regime, rule, subset, 'POOLED']
                                   + [f'{x:.4f}' for x in tot]
                                   + [f'{sum(tot):.4f}'])

    ## The panel's circles must agree with the bars above them. caught_mask uses
    ## the strictly-above set while the ladder splits ties fractionally, so the
    ## two can differ wherever scores tie at the budget boundary; print the gap
    ## rather than trusting that they do not.
    print('\n== overlap marginals vs ladder recall (should match to ~0.001) ==')
    for uni in universes:
     for met in BUDGETS:
      for regime in ('Random holdout', 'Temporal holdout'):
        for subset in ('all', 'novel'):
            tot = [0.0] * 4
            for tree in PANEL:
                k = (uni, met, regime, 'pooled', tree, subset)
                if k in agree:
                    tot = [a + b / agree_n[k] for a, b in zip(tot, agree[k])]
            n = sum(tot)
            if not n:
                continue
            for rung, got in (('joint', (tot[0] + tot[1]) / n),
                              ('where_times_which', (tot[0] + tot[2]) / n)):
                ks = [(uni, regime, 'pooled', t, subset, rung, met)
                      for t in PANEL
                      if (uni, regime, 'pooled', t, subset, rung, met) in mean]
                ref = (sum(float(mean[k].sum()) for k in ks)
                       / max(sum(float(nvec[(uni, regime, k[3], subset)].sum())
                                 for k in ks), 1e-9))
                print(f'  {uni:9s} {met:6s} {regime:18s} {subset:6s} {rung:18s} '
                      f'overlap {got:.4f}  ladder {ref:.4f}  '
                      f'delta {got - ref:+.4f}')

    np.savez_compressed(
        f'{args.out}/per_example.npz',
        **{'|'.join(k): v for k, v in mean.items()},
        **{'n|' + '|'.join(k): v for k, v in nvec.items()})
    print(f"wrote {args.out}/ladder.csv, {args.out}/deltas.csv, "
          f"{args.out}/agreement.csv and per_example.npz")


if __name__ == '__main__':
    main()
