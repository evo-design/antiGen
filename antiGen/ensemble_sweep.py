#!/usr/bin/env python
"""Score ensemble rules over cached score maps, and keep selection off the test targets.

**The question.** Panels C-F of `coguk_list_results` have three different leaders between
them -- ESM-2 35M partial FT on C, antiGen on D and F, ESM-2 35M LM-head FT on E -- and
the per-cell oracle over the roster is 0.96 against best singles of 0.51. This script asks
how much of that complementarity a training-free combination of the existing checkpoints
can actually reach.

**The prior is that the answer is "very little."** NOTES s14 fit nine families of mixing
rules by random search on the forecasting arm and *not one* beat a plain average of seeds,
while the oracle there showed +0.096 of headroom and no feature predicted which predictor
would win. Its conclusion -- "the headroom is real and is a SELECTION failure, not a
capacity one" -- is what this arm tests on a second substrate. The design therefore makes
a null legible rather than merely possible: a parameter-free rule is the pre-registered
headline, a same-seed-duplicate mixture is available as a negative control, and NOTES s47
measured the noise floor on these exact panels at 0.028-0.035.

**Selection never sees the reported targets.** The metric is a mean over *cells*, which
makes the target cell set a free axis: this script computes every rule on two disjoint
cell sets at once, off one pass.

    report  the curated list's cells            -- panels C-F, 235 and 131 cells
    select  observed eval-range cells NOT on the list -- 1,235 and 720 cells on the
            attached tree, or with ``--select-position-disjoint`` the 764/442 that also
            sit at a position carrying no listed cell

Queries, candidate pool, wild-type and gap/stop masking and the fractional tie rule are
identical between the two; only which cells are counted differs. That is a disjointness of
*targets*, not of the test window -- see the plan's limitation note -- and the
position-disjoint cut exists because the 22 residues at one position are coupled through
the shared pool, so cell-disjoint is not by itself position-disjoint.

**Both numbers are always written.** A sweep whose selection column is hidden invites
reading its maximum as a result; printing both makes the optimism visible as the gap
between them.

**No GPU.** Everything is read from maps that ``background_discrimination.py`` (backbone)
or ``list_macro.py --dump-scores`` (attached tree) already wrote.

Usage, Stage 1 (backbone, no GPU, maps already on disk):
    python antiGen/ensemble_sweep.py --substrate backbone \\
        --background-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2_background_update \\
        --data-dir /home/ivan.specht/antiGen-data/output1 \\
        --tree-json sc2_update.json --ref-json antiGen-data/ref.json --cutoff 0.989 \\
        --mutation-list coguk=coguk.csv --grid verify --output-dir /tmp/sweep
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from antiGen import ensemble_rules as er
from antiGen import score_cache as sc
from antiGen.eval_utils import real_substitution_mask
from antiGen.background_discrimination import rank_fractions, top_q_weight
from antiGen.backbone_list_eval import summarize, target_matrix
from antiGen import datasets
from antiGen.background_tree import build_backbone
from antiGen.load import load_pssms
from antiGen.mutation_lists import load_mutation_list

SWEEP = "ensemble_sweep.csv"
ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"

#: (name, score_mask attribute) -- the two candidate pools the four panels use. The
#: `unseen` pool is Figure 2's own de novo convention: rank only among cells absent from
#: the training window.
POOLS = ("fullpool", "unseen_fullpool")

#: The figure's own model grouping, mirrored from `fig_coguk_list.R`'s MODEL_GROUP so the
#: member families below are defined by the same classification the panels are coloured
#: by, rather than by a fresh judgement made here about what each model "saw".
MODEL_GROUP = {
    "simple_d128_n2_lr0.0001": "antiGen",
    "eve_ft_lr0.0001": "antiGen Finetunes",
    "tranception_ft_both_lr0.0001": "antiGen Finetunes",
    "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4": "antiGen Finetunes",
    "esm2_esm2_t12_35M_UR50D_lr0.001_frozen": "antiGen Finetunes",
    "esm2_esm2_t12_35M_UR50D_lr1e-05_full": "antiGen Finetunes",
    "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4": "antiGen Finetunes",
    "esm2_esm2_t33_650M_UR50D_lr0.001_frozen": "antiGen Finetunes",
    "esm2_esm2_t33_650M_UR50D_lr1e-05_full": "antiGen Finetunes",
    "pssm_root": "Pandemic Data in Training Window",
    "pssm_root_position": "Pandemic Data in Training Window",
    "MSA_Transformer": "Pandemic Data in Training Window",
    "MSA_Transformer_BA2": "Pandemic Data in Training Window",
    "MSA_Transformer_XFG": "Pandemic Data in Training Window",
    "eve_frozen": "Pandemic Data in Training Window",
    "tranception_frozen_both": "Pandemic Data in Training Window",
    "evcouplings": "Pandemic Data in Training Window",
    "pyr0": "Pandemic Data in Training Window",
    "CoVFit": "Pandemic Data in Training Window",
    "substitution": "Pandemic Data in Training Window",
    "where_times_which": "Pandemic Data in Training Window",
    "Omicron_BA1_bind": "Deep Mutational Scans",
    "Omicron_BA1_expr": "Deep Mutational Scans",
    "Omicron_BA2_bind": "Deep Mutational Scans",
    "Omicron_BA2_expr": "Deep Mutational Scans",
    "EVEscape": "Pre-Pandemic Data Only",
    "esm2_esm2_t12_35M_UR50D_lr0.001_noft": "Pre-Pandemic Data Only",
    "esm2_esm2_t33_650M_UR50D_lr0.001_noft": "Pre-Pandemic Data Only",
    "blosum62": "Pre-Pandemic Data Only",
    "codon": "Pre-Pandemic Data Only",
}

#: Members that READ THE QUERY, as opposed to the fixed (position x residue) maps. This
#: is the line panels K-L already draw: a fixed map cannot move its decile at all. Kept
#: as an explicit set rather than derived, because a model's group does not determine it
#: -- BLOSUM62, the codon table and the substitution matrix score a wild-type -> mutant
#: transition and so are query-dependent, while sitting in three different groups.
QUERY_DEPENDENT = {
    "simple_d128_n2_lr0.0001", "eve_ft_lr0.0001", "tranception_ft_both_lr0.0001",
    "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4",
    "esm2_esm2_t12_35M_UR50D_lr0.001_frozen",
    "esm2_esm2_t12_35M_UR50D_lr1e-05_full",
    "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4",
    "esm2_esm2_t33_650M_UR50D_lr0.001_frozen",
    "esm2_esm2_t33_650M_UR50D_lr1e-05_full",
    "esm2_esm2_t12_35M_UR50D_lr0.001_noft", "esm2_esm2_t33_650M_UR50D_lr0.001_noft",
    "eve_frozen", "tranception_frozen_both", "blosum62", "codon", "substitution",
    "where_times_which",
}


def member_families(members):
    """The three nested member families, fixed in advance (see the plan's Stage 5).

    F1 answers "a better antiGen", F2 "a forecaster using only within-window
    information", F3 is the unrestricted ceiling. Any member whose name is not in
    MODEL_GROUP -- a seed retrain, a staged mt2021 checkpoint -- is treated as belonging
    to the antiGen family when its name says so, and is otherwise only in F3, so an
    unrecognised rung can never silently join the narrow claim.
    """
    def grp(m):
        if m in MODEL_GROUP:
            return MODEL_GROUP[m]
        if m.startswith(("valselAntigen", "fix32Antigen")):
            return "antiGen"
        if m.startswith(("valselEsm2p", "fix32Esm2p", "valselEve", "fix32Eve")):
            return "antiGen Finetunes"
        return "unclassified"
    f1 = [m for m in members if grp(m) in ("antiGen", "antiGen Finetunes")]
    f2 = [m for m in members if grp(m) in ("antiGen", "antiGen Finetunes",
                                           "Pandemic Data in Training Window",
                                           "Pre-Pandemic Data Only")]
    return {"F1_antigen": f1, "F2_within_window": f2, "F3_all": list(members)}


def preregistered_subsets(members):
    """Uniform-weight member subsets named before any number is looked at.

    These are the rungs a fitted rule has to beat: a uniform average over a subset chosen
    by classification alone has no free parameters, so it cannot be overfitted, and NOTES
    s14 found plain averaging was the ONLY thing that worked on the other arm.
    """
    fams = member_families(members)
    out = dict(fams)
    out["all"] = list(members)
    for g in sorted(set(MODEL_GROUP.values())):
        sel = [m for m in members if MODEL_GROUP.get(m) == g]
        if len(sel) >= 2:
            out["group_" + g.split()[0].lower()] = sel
    qd = [m for m in members if m in QUERY_DEPENDENT]
    if qd:
        out["query_dependent"] = qd
    return {k: v for k, v in out.items() if len(v) >= 2}


class Substrate:
    """Queries, targets, masks and a block-wise reader, for one test set.

    Holds nothing model-specific. ``blocks()`` yields ``(start, stop)`` so the sweep can
    stream the attached tree's 17,978 queries without materialising a 9.5 GB stack, while
    the backbone's 318 fit in one block.
    """

    def __init__(self, *, maps_dir, aa_eval, targets, seen, in_full, observed,
                 eval_range, members, block=2048):
        self.maps_dir = maps_dir
        self.aa_eval = aa_eval            # (N, R) long, residue index per eval position
        self.targets = targets            # (N, R, E) float, target counts
        self.seen = seen                  # (R, E) bool, carried in the training window
        self.in_full = in_full            # (R, E) bool, on the curated list
        self.observed = observed          # (R, E) bool, ever a target in this test set
        self.eval_range = eval_range
        self.members = list(members)
        self.block = block
        self.n_queries = int(aa_eval.shape[0])
        self._cache = {}

    def blocks(self):
        for s in range(0, self.n_queries, self.block):
            yield s, min(s + self.block, self.n_queries)

    def raw(self, member, start, stop):
        """One member's block, with ``-inf`` already folded into NaN.

        Only the CURRENT block is cached. Keeping every block would hold the entire
        stack -- 30 members x 17,978 queries x 201 x 22 float32 is 9.5 GB -- which is the
        one thing the block-wise read exists to avoid, and it would come on top of the
        normalised stacks the sweep builds per block.
        """
        key = (member, start, stop)
        if key not in self._cache:
            stale = [k for k in self._cache if (k[1], k[2]) != (start, stop)]
            for k in stale:
                del self._cache[k]
            arr, _ = sc.load_block(self.maps_dir, member, start, stop)
            self._cache[key] = torch.from_numpy(arr)
        return self._cache[key]

    def pool_mask(self, pool_name):
        """Cells EXCLUDED from this pool, as an (R, E) bool.

        The gap and stop columns are excluded for every member, which is what makes "the
        top decile" the same size for all of them: without it ``n_valid`` is whatever a
        member happens to emit and runs 3,819 to 4,221 across this roster -- a 10% spread
        that is a property of a model's alphabet, not of its predictions. It removes no
        target, since none of the 247 listed RBD cells is a gap or a stop. See
        ``eval_utils.real_substitution_mask`` and NOTES s51.
        """
        drop = ~real_substitution_mask(self.seen.shape[0], ALLOWED_AAS,
                                       device=self.seen.device)
        return drop if pool_name == "fullpool" else (drop | self.seen)

    def cellsets(self, pool_name, *, position_disjoint=False):
        """-> {name: (R, E) bool target mask} for this pool.

        The de novo pools report only cells absent from the training window, matching the
        pool they are ranked in; the full pool reports every cell of the set.
        """
        unseen = pool_name != "fullpool"
        rep = self.in_full & (~self.seen if unseen else torch.ones_like(self.seen))
        sel = self.observed & ~self.in_full
        if position_disjoint:
            listed_pos = self.in_full.any(dim=1)
            sel = sel & ~listed_pos.unsqueeze(1)
        if unseen:
            sel = sel & ~self.seen
        return {"report": rep, "select": sel}


def _wt_masked(scores, aa_eval_block, pool_mask):
    """Apply the query's own residue and the pool mask, so ranks are over candidates only.

    Normalisation has to happen over exactly the pool the metric ranks in. Leaving the
    wild-type cells in would shift each member's percentiles by a different amount --
    harmless for one model, since only the order inside the candidate set is read, but a
    real perturbation once several members are averaged. ``rank_fractions`` masks them
    again downstream, which is idempotent.
    """
    s = scores.clone()
    s.scatter_(2, aa_eval_block.unsqueeze(-1), float('nan'))
    if pool_mask is not None:
        s[:, pool_mask] = float('nan')
    return s


def build_backbone_substrate(args):
    """The guide-tree substrate, from the maps ``background_discrimination.py`` wrote."""
    E = len(ALLOWED_AAS)
    # seq_len and the eval range come from the manifest the maps were written with, not
    # from this module's defaults: five directories under fitness-inference/ carry
    # scores_*.npz under identical names from different guide trees, and a mismatched
    # range would line the maps up against the wrong positions without any array-level
    # signature.
    with open(os.path.join(args.background_dir, "backbone_manifest.json")) as fh:
        bm = json.load(fh)
    seq_len = bm["seq_len"]
    eval_range = list(range(bm["eval_range_start1"] - 1, bm["eval_range_end1"]))
    bb = build_backbone(args.tree_json, args.ref_json, args.segment, seq_len,
                        ALLOWED_AAS, args.cutoff, eval_range=eval_range)
    if bb.report['n_wt_mismatch']:
        raise SystemExit("branch mutations disagree with the reconstruction")
    if len(bb.query_seq) != bm["n_queries"]:
        raise SystemExit(
            f"rebuilt backbone has {len(bb.query_seq)} query genotypes but "
            f"{args.background_dir}'s manifest says {bm['n_queries']}: the tree or "
            f"cutoff does not match the one the maps were written with")
    T = torch.from_numpy(target_matrix(bb, eval_range, E))
    aa_eval = torch.tensor(
        [[ALLOWED_AAS.index(c) for c in seq[eval_range[0]:eval_range[-1] + 1]]
         for seq in bb.query_seq], dtype=torch.long)
    pssm_root, _ = load_pssms(os.path.join(args.data_dir, "pssms"),
                              args.variant or datasets.default_variant("sc2"), E)
    seen = (pssm_root > 0)[eval_range, :]
    members = [os.path.basename(p)[len("scores_"):-len(".npz")]
               for p in sorted(glob.glob(os.path.join(args.background_dir, "scores_*.npz")))]
    if not members:
        raise SystemExit(f"no scores_*.npz in {args.background_dir}")
    return bb, T, aa_eval, seen, eval_range, members, seq_len


def build_list_substrate(args):
    """The attached-tree substrate, from ``list_macro.py --dump-scores``."""
    man = sc.read_query_manifest(args.maps_dir)
    eval_range = man["eval_range"]
    E = len(ALLOWED_AAS)
    aa_full, (qi, po, al, ct) = sc.read_queries(args.maps_dir)
    pos_of = {p: i for i, p in enumerate(eval_range)}
    N, R = aa_full.shape[0], len(eval_range)
    aa_eval = torch.from_numpy(aa_full[:, eval_range].astype(np.int64))
    T = torch.zeros(N, R, E)
    keep = np.array([p in pos_of for p in po])
    rows = np.array([pos_of[p] for p in po[keep]], dtype=np.int64)
    T[torch.from_numpy(qi[keep].astype(np.int64)), torch.from_numpy(rows),
      torch.from_numpy(al[keep].astype(np.int64))] += torch.from_numpy(ct[keep])
    pssm_root, _ = load_pssms(os.path.join(args.data_dir, "pssms"),
                              man.get("variant") or datasets.default_variant("sc2"), E)
    seen = (pssm_root > 0)[eval_range, :]
    members = sc.list_models(args.maps_dir)
    if not members:
        raise SystemExit(f"no cached maps in {args.maps_dir}")
    return man, T, aa_eval, seen, eval_range, members


def load_lists(specs, eval_range, seq_len):
    """-> (R, E) bool mask per --mutation-list, restricted to the eval range."""
    pos_of = {p: i for i, p in enumerate(eval_range)}
    out = []
    for item in specs:
        if '=' not in item:
            raise SystemExit(f"--mutation-list wants LABEL=PATH, got {item!r}")
        label, path = item.split('=', 1)
        cells, report = load_mutation_list(path, ALLOWED_AAS, seq_len, label=label)
        m = torch.zeros(len(eval_range), len(ALLOWED_AAS), dtype=torch.bool)
        for pos, aa in cells:
            if pos in pos_of:
                m[pos_of[pos], aa] = True
        print(f"  {label}: {report['n_cells']} parsed cells, {int(m.sum())} in the eval range")
        out.append((label, m))
    return out


def score_rules(sub, rules, qtiles, *, position_disjoint=False, device="cpu",
                progress_every=25, pools=None):
    """Evaluate every rule over every pool, cell set and quantile. -> list of row dicts.

    Normalisation is built once per (pool, normalisation, member) and shared by every
    rule that uses it. That ordering is forced by cost -- recomputing it per rule is
    ~5x the work -- and by correctness: the pool a member is normalised over must be the
    pool the metric ranks it in, so the loop cannot hoist the pool out.
    """
    by_pool = {}
    for r in rules:
        by_pool.setdefault(r.norm, []).append(r)
    norms_needed = sorted(by_pool)

    acc = {}        # (rule_id, pool, cellset, q) -> [cell_hits, cell_events]
    tie_acc = {}    # (rule_id, pool) -> [sum of n_at/n_valid, n_queries]
    nval_acc = {}
    K = len(sub.members)
    idx_of = {m: i for i, m in enumerate(sub.members)}

    for pool_name in (pools or POOLS):
        pmask = sub.pool_mask(pool_name)
        sets = sub.cellsets(pool_name, position_disjoint=position_disjoint)
        done = 0
        for start, stop in sub.blocks():
            aa_b = sub.aa_eval[start:stop].to(device)
            tgt_b = sub.targets[start:stop].to(device)
            # One masked copy per member, then one normalised stack per normalisation.
            masked = [_wt_masked(sub.raw(m, start, stop).to(device), aa_b, pmask)
                      for m in sub.members]
            valid = [~torch.isnan(x) for x in masked]
            nv = torch.stack([v.reshape(v.shape[0], -1).sum(dim=1).clamp(min=1)
                              for v in valid])                      # (K, B)
            stacks = {n: torch.stack([er.normalize(masked[k], n, valid[k])
                                      for k in range(K)]) for n in norms_needed}
            del masked

            for norm, rs in by_pool.items():
                S = stacks[norm]
                for rule in rs:
                    sel = [idx_of[m] for m in rule.members]
                    filled, ok = er.resolve_nan(S[sel], rule.nan)
                    out = er.pool(filled, rule.w, rule.pool, beta=rule.beta, k=rule.k,
                                  qprime=rule.qprime, trim=rule.trim, n_valid=nv[sel])
                    if rule.gate is not None:
                        ws, wu = rule.gate
                        g = torch.where(sub.seen.to(device).expand_as(out),
                                        torch.full_like(out, float(ws)),
                                        torch.full_like(out, float(wu)))
                        out = g * out + (1.0 - g) * filled.mean(dim=0)
                    out = torch.where(ok, out, torch.full_like(out, float('nan')))

                    above, tie, n_valid = rank_fractions(
                        out, aa_b, list(range(out.shape[1])))
                    key_t = (rule.rule_id, pool_name)
                    nval_acc.setdefault(key_t, []).append(float(n_valid.float().mean()))
                    for q in qtiles:
                        W = top_q_weight(above, tie, n_valid, q)
                        if abs(q - qtiles[0]) < 1e-12:
                            # n_at / n_valid: the share of the pool sitting in the
                            # boundary tie block. A coarse rule (vote pooling, a DMS
                            # panel) can put a large fraction there, and the fractional
                            # top-q rule then hands it a slice of a block it never
                            # really ranked -- the BLOSUM62 failure mode, so it is
                            # reported rather than assumed small.
                            partial = ((W > 1e-9) & (W < 1 - 1e-9)).reshape(
                                W.shape[0], -1).sum(dim=1).float()
                            t = tie_acc.setdefault(key_t, [0.0, 0])
                            t[0] += float((partial / n_valid.float()).sum())
                            t[1] += int(W.shape[0])
                        for cs_name, cs_mask in sets.items():
                            tgt = tgt_b * cs_mask.to(device)
                            key = (rule.rule_id, pool_name, cs_name, q)
                            ch = (W * tgt).sum(dim=0)
                            ce = tgt.sum(dim=0)
                            if key not in acc:
                                acc[key] = [torch.zeros_like(ch), torch.zeros_like(ce)]
                            acc[key][0] += ch
                            acc[key][1] += ce
            del stacks
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"    {pool_name}: block {done} ({stop}/{sub.n_queries})", flush=True)

    rows = []
    by_id = {r.rule_id: r for r in rules}
    for (rid, pool_name, cs_name, q), (ch, ce) in sorted(acc.items()):
        r = by_id[rid]
        macro, micro, n_cells, n_events = summarize(ch.cpu(), ce.cpu())
        tsum, tn = tie_acc.get((rid, pool_name), [0.0, 1])
        rows.append(dict(
            rule_id=rid, family=r.family, label=r.label, norm=r.norm, pooling=r.pool,
            nan=r.nan, beta=r.beta, k=r.k, qprime=r.qprime, trim=r.trim,
            gate=("" if r.gate is None else f"{r.gate[0]:g}/{r.gate[1]:g}"),
            n_members=len(r.members), members="|".join(r.members),
            weights="|".join(f"{w:g}" for w in r.w),
            pool=pool_name, cellset=cs_name, qtile=q, macro=macro, micro=micro,
            n_cells=n_cells, n_events=n_events,
            tie_frac=tsum / max(tn, 1),
            n_valid=float(np.mean(nval_acc.get((rid, pool_name), [float('nan')]))),
        ))
    return rows


def _select_value(rows, rule_id, pool, qtile):
    """The SELECT-set macro for one rule: the only number selection may look at."""
    for r in rows:
        if (r["rule_id"] == rule_id and r["cellset"] == "select"
                and r["pool"] == pool and abs(r["qtile"] - qtile) < 1e-12):
            return r["macro"]
    return float("nan")


def greedy_select(sub, members, rounds, qtiles, *, norm, pooling, nan, pool_name,
                  device="cpu", position_disjoint=False, log=print):
    """Caruana forward selection with replacement, scored on the SELECT cells only.

    With replacement, because a repeat IS the weight: picking a member twice doubles its
    share, which is how this family expresses a weighting without a continuous search.

    Each round is one :func:`score_rules` call over the K candidate extensions, so the
    round shares one normalised stack across its candidates rather than rebuilding it per
    candidate -- the 5x difference the plan's cost note is about.

    Returns (chosen, trace) where trace carries each round's winner and its SELECT value,
    so the selection path is on the record rather than only its endpoint.
    """
    chosen, trace, all_rows = [], [], []
    q0 = qtiles[0]
    prev = -np.inf
    for rnd in range(1, rounds + 1):
        cands = []
        for m in members:
            # Duplicating the ONLY member already chosen is a provable no-op: a weighted
            # mean of identical maps is that map. Offering it as a candidate lets a round
            # with no genuine improvement "win" by standing still, which is how the first
            # run of this spent five rounds re-adding pssm_root.
            if len(set(chosen)) == 1 and m == chosen[0]:
                continue
            cands.append(er.Rule(norm=norm, pool=pooling, nan=nan,
                                 members=tuple(chosen + [m]), family=f"greedy_r{rnd}"))
        uniq, seen_ids = [], set()
        for r in cands:
            if r.rule_id not in seen_ids:
                seen_ids.add(r.rule_id)
                uniq.append(r)
        rows = score_rules(sub, uniq, [q0], position_disjoint=position_disjoint,
                           device=device, progress_every=0, pools=(pool_name,))
        all_rows.extend(rows)

        def _key(r):
            v = _select_value(rows, r.rule_id, pool_name, q0)
            novel = 1 if set(r.members) - set(chosen) else 0
            # Ties break toward ADDING A NEW MEMBER, then toward the smaller mixture.
            # On a few hundred cells the metric is granular enough that ties are common,
            # and breaking them toward a repeat would quietly stall the search.
            return ((-np.inf if np.isnan(v) else v), novel, -len(set(r.members)))

        best = max(uniq, key=_key)
        v = _select_value(rows, best.rule_id, pool_name, q0)
        added = list(best.members)
        for m in chosen:
            if m in added:
                added.remove(m)
        chosen = list(best.members)
        improved = v > prev + 1e-12
        trace.append(dict(round=rnd, added=(added[0] if added else ""), select=v,
                          improved=bool(improved), members="|".join(chosen)))
        log(f"    greedy r{rnd}: +{(added[0] if added else '?')[:44]:44s} "
            f"select={v:.4f}{'' if improved else '   (no improvement)'}")
        if not improved and rnd > 1:
            # Stop, and say so. A greedy path that has stopped improving and keeps
            # being extended reports a K that was never earned.
            log(f"    greedy: stopped at K={len(chosen)} -- round {rnd} did not improve "
                f"on {prev:.4f}")
            break
        prev = max(prev, v)
    return chosen, trace, all_rows


def simplex_rules(members, n_draws, seed, *, norm, pooling, nan, alpha=1.0):
    """Dirichlet draws over the member simplex -- the `gate_search.py` random-search family.

    Weights are rounded to 4 decimals before the rule id is taken, so two draws that are
    the same weighting to within rounding collapse to one row instead of counting twice.
    """
    rng = np.random.default_rng(seed)
    out, seen_ids = [], set()
    for _ in range(n_draws):
        w = rng.dirichlet(np.full(len(members), alpha))
        r = er.Rule(norm=norm, pool=pooling, nan=nan, members=tuple(members),
                    weights=tuple(round(float(x), 4) for x in w), family="simplex")
        if r.rule_id not in seen_ids:
            seen_ids.add(r.rule_id)
            out.append(r)
    return out


def hierarchical_rules(members, n_draws, seed, *, norm, pooling, nan):
    """Group means, then a search over the five-group simplex.

    A member's weight is its group weight divided by the group's size, so every group
    contributes its mean rather than its sum and a large group does not dominate by
    counting.
    """
    groups = {}
    for m in members:
        groups.setdefault(MODEL_GROUP.get(m, "unclassified"), []).append(m)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    out, seen_ids = [], set()
    draws = [np.full(len(keys), 1.0 / len(keys))] + \
            [rng.dirichlet(np.ones(len(keys))) for _ in range(n_draws)]
    for gw in draws:
        w, mem = [], []
        for gi, g in enumerate(keys):
            for m in groups[g]:
                mem.append(m)
                w.append(float(gw[gi]) / len(groups[g]))
        tot = sum(w) or 1.0
        r = er.Rule(norm=norm, pool=pooling, nan=nan, members=tuple(mem),
                    weights=tuple(round(x / tot, 6) for x in w), family="hierarchical")
        if r.rule_id not in seen_ids:
            seen_ids.add(r.rule_id)
            out.append(r)
    return out


FIELDS = ["rule_id", "family", "label", "norm", "pooling", "nan", "beta", "k", "qprime",
          "trim", "gate", "n_members", "members", "weights", "pool", "cellset", "qtile",
          "macro", "micro", "n_cells", "n_events", "tie_frac", "n_valid"]


def write_rows(output_dir, rows, fname=SWEEP, append=False):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, fname)
    new = append and os.path.exists(path)
    mode = "a" if new else "w"
    with open(path + ("" if new else ".tmp"), mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not new:
            w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
    if not new:
        os.replace(path + ".tmp", path)
    return path


def add_substrate_args(ap):
    """The flags that define a substrate. Shared with ensemble_diagnostics.py."""
    ap.add_argument('--substrate', choices=['backbone', 'list'], required=True)
    ap.add_argument('--background-dir', help='backbone substrate: the scores_*.npz dir')
    ap.add_argument('--maps-dir', help='list substrate: the --dump-scores dir')
    ap.add_argument('--data-dir', required=True, help='for pssms/root.csv (the seen mask)')
    ap.add_argument('--tree-json'); ap.add_argument('--ref-json')
    ap.add_argument('--segment', default='S')
    ap.add_argument('--cutoff', type=float, default=1.0)
    ap.add_argument('--variant', default=None)
    ap.add_argument('--mutation-list', action='append', required=True,
                    help='LABEL=PATH, repeatable')
    ap.add_argument('--list', default=None,
                    help='which --mutation-list label is the report set (default: the first)')
    ap.add_argument('--members', default=None, help='comma-separated substrings to keep')
    ap.add_argument('--exclude', default=None, help='comma-separated substrings to drop')
    ap.add_argument('--select-position-disjoint', action='store_true')
    ap.add_argument('--query-block', type=int, default=2048)
    return ap


def substrate_from_args(args):
    """-> (Substrate, report-list label). One definition, two callers."""
    if args.substrate == 'backbone':
        for need in ('background_dir', 'tree_json', 'ref_json'):
            if not getattr(args, need):
                raise SystemExit(f"--substrate backbone needs --{need.replace('_', '-')}")
        _bb, T, aa_eval, seen, eval_range, members, seq_len = build_backbone_substrate(args)
        maps_dir = args.background_dir
    else:
        if not args.maps_dir:
            raise SystemExit("--substrate list needs --maps-dir")
        man, T, aa_eval, seen, eval_range, members = build_list_substrate(args)
        maps_dir, seq_len = args.maps_dir, man["seq_len"]

    if args.members:
        keep = [x.strip() for x in args.members.split(',') if x.strip()]
        members = [m for m in members if any(k in m for k in keep)]
    if args.exclude:
        drop = [x.strip() for x in args.exclude.split(',') if x.strip()]
        members = [m for m in members if not any(d in m for d in drop)]
    if len(members) < 1:
        # Discovering members and filtering every one away is a misconfiguration, not a
        # no-op (NOTES s30/s31), and a one-member "ensemble" reproduces a published row
        # exactly and so looks right.
        raise SystemExit("--members/--exclude left no members")
    print(f"Substrate {args.substrate}: {aa_eval.shape[0]} queries, "
          f"{len(eval_range)} eval positions, {len(members)} members")

    lists = load_lists(args.mutation_list, eval_range, seq_len)
    label = args.list or lists[0][0]
    in_full = dict(lists).get(label)
    if in_full is None:
        raise SystemExit(f"--list {label!r} is not among {[l for l, _ in lists]}")
    observed = T.sum(dim=0) > 0

    sub = Substrate(maps_dir=maps_dir, aa_eval=aa_eval, targets=T, seen=seen,
                    in_full=in_full, observed=observed, eval_range=eval_range,
                    members=members, block=args.query_block)
    for pool_name in POOLS:
        cs = sub.cellsets(pool_name,
                          position_disjoint=bool(getattr(args, 'select_position_disjoint',
                                                         False)))
        print(f"  {pool_name:18s} report {int(cs['report'].sum()):5d} cells "
              f"({int((cs['report'] & observed).sum())} observed) | "
              f"select {int((cs['select']).sum()):5d} cells "
              f"({int((cs['select'] & observed).sum())} observed)")
    return sub, label


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_substrate_args(ap)
    ap.add_argument('--grid', default='verify',
                    help='Comma-separated stages: verify | families | gate | nan | '
                         'subsets | simplex | hier | greedy. The adaptive ones (greedy) '
                         'are scored round by round on the SELECT cells only.')
    ap.add_argument('--family', default='F3_all',
                    choices=['F1_antigen', 'F2_within_window', 'F3_all'],
                    help='Which pre-registered member family the fitted stages search in.')
    ap.add_argument('--cell', default='pctile/arith/union_min',
                    help='The norm/pool/nan cell the fitted stages search at. The '
                         'pre-registered default is Borda count: the only cell that is '
                         'both scale-free and distribution-free over this roster.')
    ap.add_argument('--greedy-rounds', type=int, default=12)
    ap.add_argument('--simplex-draws', type=int, default=200)
    ap.add_argument('--hier-draws', type=int, default=200)
    ap.add_argument('--search-pool', default='fullpool', choices=list(POOLS),
                    help='Which pool the adaptive search optimises on.')
    ap.add_argument('--search-seed', type=int, default=7)
    ap.add_argument('--rules-json', default=None,
                    help='Score exactly the rules in this JSON list (each a dict of '
                         'norm/pool/nan/members[/weights/params]) instead of enumerating '
                         'a grid. This is phase 2 of the funnel: a wide search runs at '
                         'one quantile, its finalists are re-scored here over the full '
                         'grid so the AUC criterion can be applied to them.')
    ap.add_argument('--rescore-top', type=int, default=0, metavar='N',
                    help='After the search, write the top N rules by SELECT macro at the '
                         'headline quantile to finalists.json, for a phase-2 run. Rule 0 '
                         '(the parameter-free uniform average) is always included, '
                         'whether or not it is in the top N -- the freeze protocol needs '
                         'it as its baseline and a funnel that dropped it would silently '
                         'make every later comparison unevaluable.')
    ap.add_argument('--qtiles', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    qtiles = [float(q) for q in args.qtiles.split(',') if q.strip()]
    # The headline quantile must come first: the tie-fraction diagnostic is accumulated
    # at qtiles[0], and it is the operating point every panel is read at.
    if abs(qtiles[0] - 0.1) > 1e-9:
        print(f"[warn] qtiles[0] = {qtiles[0]}, not 0.1; the tie diagnostic reports that "
              f"quantile and the panels are read at 0.1")

    sub, label = substrate_from_args(args)
    members = sub.members

    norm, pooling, nan = args.cell.split('/')
    if (norm, pooling) not in er.LEGAL:
        raise SystemExit(f"--cell {args.cell}: ({norm}, {pooling}) is not a defined "
                         f"combination; see ensemble_rules.LEGAL")
    fams = member_families(members)
    fam_members = fams[args.family]
    if len(fam_members) < 2:
        raise SystemExit(f"member family {args.family} has {len(fam_members)} member(s) "
                         f"in this map set; a mixture needs at least two")

    rules, extra_rows, traces = [], [], {}
    if args.rules_json:
        with open(args.rules_json) as f:
            specs = json.load(f)
        for sp in specs:
            sp = dict(sp)
            fam = sp.pop("family", "rescore")
            rules.append(er.Rule(family=fam,
                                 **{k: (tuple(v) if isinstance(v, list) else v)
                                    for k, v in sp.items()}))
        print(f"  rescoring {len(rules)} rule(s) from {args.rules_json}")
        stages = []
    else:
        stages = [x.strip() for x in args.grid.split(',') if x.strip()]
    for stage in stages:
        if stage == 'verify':
            # One single-member rule per member, so the output can be checked against the
            # published single-model CSVs. This is the gate that makes every later
            # ensemble number trustworthy (see test_ensemble_rules).
            for m in members:
                rules.append(er.Rule(norm=norm, pool=pooling, nan=nan, members=(m,),
                                     family="verify"))
        elif stage == 'subsets':
            # Nothing is fitted in this stage: every rule is a uniform average over a
            # subset named by the figure's own classification. These are the rungs a
            # fitted rule has to beat on the SELECT cells before it may be reported.
            for name, sel in sorted(preregistered_subsets(members).items()):
                rules.append(er.Rule(norm=norm, pool=pooling, nan=nan,
                                     members=tuple(sel), family=f"subset:{name}"))
        elif stage == 'simplex':
            rules.extend(simplex_rules(fam_members, args.simplex_draws, args.search_seed,
                                       norm=norm, pooling=pooling, nan=nan))
        elif stage == 'hier':
            rules.extend(hierarchical_rules(fam_members, args.hier_draws, args.search_seed,
                                            norm=norm, pooling=pooling, nan=nan))
        elif stage == 'greedy':
            print(f"  greedy forward selection in {args.family} "
                  f"({len(fam_members)} members), {args.greedy_rounds} rounds, "
                  f"optimising SELECT macro on {args.search_pool} at q={qtiles[0]}")
            chosen, trace, rows_g = greedy_select(
                sub, fam_members, args.greedy_rounds, qtiles, norm=norm, pooling=pooling,
                nan=nan, pool_name=args.search_pool, device=args.device,
                position_disjoint=args.select_position_disjoint)
            traces['greedy'] = trace
            extra_rows.extend(rows_g)
            # The selected rule at every K along the path, so the sparsity curve is read
            # off the same run rather than refitted.
            for kk in range(1, len(chosen) + 1):
                rules.append(er.Rule(norm=norm, pool=pooling, nan=nan,
                                     members=tuple(chosen[:kk]), family=f"greedy_k{kk}"))
        else:
            rules.extend(er.enumerate_grid(tuple(fam_members), stage))

    seen_ids, uniq = set(), []
    for r in rules:
        if r.rule_id not in seen_ids:
            seen_ids.add(r.rule_id)
            uniq.append(r)
    print(f"{len(uniq)} distinct rules over {len(POOLS)} pools x 2 cell sets x "
          f"{len(qtiles)} quantiles")

    rows = score_rules(sub, uniq, qtiles,
                       position_disjoint=args.select_position_disjoint,
                       device=args.device) if uniq else []
    path = write_rows(args.output_dir, rows + extra_rows)

    if args.rescore_top:
        by_id = {r.rule_id: r for r in uniq}
        q0 = qtiles[0]
        score = {}
        for r in rows:
            if r['cellset'] == 'select' and abs(r['qtile'] - q0) < 1e-12:
                score[r['rule_id']] = score.get(r['rule_id'], 0.0) + r['macro']
        rule0 = [r for r in uniq if r.family.startswith("subset:")]
        keep = [rid for rid, _ in sorted(score.items(), key=lambda kv: -kv[1])
                ][:args.rescore_top]
        for r in rule0:
            if r.rule_id not in keep:
                keep.append(r.rule_id)
        specs = []
        for rid in keep:
            r = by_id[rid]
            sp = dict(norm=r.norm, pool=r.pool, nan=r.nan, members=list(r.members),
                      family=r.family)
            if len(set(r.w)) > 1:
                sp["weights"] = list(r.w)
            for k in ("beta", "k", "qprime", "trim"):
                if getattr(r, k) is not None:
                    sp[k] = getattr(r, k)
            if r.gate is not None:
                sp["gate"] = list(r.gate)
            specs.append(sp)
        fpath = os.path.join(args.output_dir, "finalists.json")
        with open(fpath, "w") as f:
            json.dump(specs, f, indent=1)
        print(f"\n  wrote {fpath}: {len(specs)} finalist(s) "
              f"(top {args.rescore_top} by SELECT at q={q0}, plus "
              f"{len(rule0)} parameter-free baseline(s))")
    # Print the honest pair for the headline cell: what selection saw, and what the
    # reported panels say. A sweep that prints only the second invites reading its
    # maximum as a result.
    q0 = qtiles[0]
    for pool_name in POOLS:
        cand = [r for r in rows if r['pool'] == pool_name and r['cellset'] == 'select'
                and abs(r['qtile'] - q0) < 1e-12]
        if not cand:
            continue
        top = sorted(cand, key=lambda r: -r['macro'])[:5]
        rep = {(r['rule_id']): r for r in rows if r['pool'] == pool_name
               and r['cellset'] == 'report' and abs(r['qtile'] - q0) < 1e-12}
        print(f"\n  {pool_name} @ q={q0}, ranked by SELECT (report shown, not used):")
        for r in top:
            rr = rep.get(r['rule_id'], {})
            print(f"    select={r['macro']:.4f}  report={rr.get('macro', float('nan')):.4f}"
                  f"  tie={100 * r['tie_frac']:.1f}%  {r['family']:16s} {r['label']}")
    if traces:
        with open(os.path.join(args.output_dir, "search_trace.json"), "w") as f:
            json.dump(traces, f, indent=2)
    with open(os.path.join(args.output_dir, "sweep_manifest.json"), "w") as f:
        json.dump(dict(substrate=args.substrate, maps_dir=sub.maps_dir, members=members,
                       family=args.family, family_members=fam_members, cell=args.cell,
                       list=label, qtiles=qtiles, n_rules=len(uniq),
                       select_position_disjoint=bool(args.select_position_disjoint),
                       args=dict(sorted(vars(args).items()))), f, indent=2, sort_keys=True)
    print(f"\nWrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
