"""One tree per forecast origin, each estimated from that origin's `S(T)` alone.

**Why.** Two of Huddleston's four sequence-only metrics need a topology -- `lbi` walks
nodes and branch lengths, `ne_star` walks a tip up to its ancestor in the prior season --
and the pair is exactly their designated best arm, `ne_star-lbi`. Everything in this
project's ledger reads both off **one** Nextstrain build dated 2026, which knows the whole
record. Censoring LBI to the horizon does not undo a topology estimated with hindsight,
and one build swap was worth 0.3866 AA to `lbi`. That has been the oldest item on the
"still exposed" list since 2026-09-24, and it is the one that most directly flatters
their side of the comparison.

`PROPOSAL_additive_panel.md` §8 says what to do about it, and says the additive panel is
what makes it cheap: at each origin take `S(T)` -- which the manifest already defines, and
which is nested by construction -- and rebuild. This module is that, with the tools that
are actually installed here:

| §8 asks for | used here |
| --- | --- |
| `augur align` | nothing: the panel is already a 566-residue nextclade translation |
| IQ-TREE | IQ-TREE 3, `--fast`, on the amino-acid alignment |
| `augur refine` (TreeTime) | TreeTime directly -- `augur refine` is a wrapper on it |
| `augur ancestral` | the same TreeTime run, which writes ancestral sequences |

**The deviation to declare.** Nextstrain builds H3N2 phylogenies from *nucleotides* and
we hold only the amino-acid translation, so the substitution model is a protein model
(JTT by default) over 566 residues rather than a nucleotide model over 1,737 sites. Fewer
informative characters means a less resolved topology. It is stated here, it belongs in
the entry that reports the first number off one of these trees, and it is a smaller
deviation than the one it removes: an amino-acid tree estimated at origin `T` is causal,
a nucleotide tree estimated in 2026 is not.

**The tree holds `S(T)` and nothing else.** No target tips. That is the point of the
`BackbonePanel` split: `task.tgt_w` is read off the *panel*, which holds every season, and
the tree is asked for `lbi` and `ne_star` over the support only. A tree with the target
in it would be the hindsight this module exists to remove, placed one season further on.

**The join back to the panel is by sequence.** The tips written here are exactly the
sequences `load_alignment_dated` hands `TipPanel` -- after its ambiguity floor and after
its ambiguous residues are resolved -- so `BackbonePanel._live` finds every one of them
in `panel.gid`. `verify` re-checks that against a built tree and reports the miss count
rather than trusting it.

Output is the subset of the auspice v2 schema `backbone_compare.load_tree` reads:
`node_attrs.num_date.value` on every node and `branch_attrs.mutations` keyed by segment.
Nothing else is written, because nothing else is read.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

from antiGen.season_momentum import (H3N2_SEGMENTS, add_panel_source_args,
                                     load_alignment_dated, tips_from_args)

# A forecast origin `T` is 1 April of season `T`, the same convention `Task` uses.
ORIGIN_OFFSET = 0.25

# The protein model. Declared, not selected: JTT is TreeTime's own default amino-acid
# model and Nextstrain's `augur ancestral --root-sequence` path uses it too, so picking
# it costs no choice that could have been made by looking at an outcome. `--model` can
# override it, and whatever is used is written into the tree's provenance block.
DEFAULT_MODEL = "JTT"

IQTREE = os.environ.get("IQTREE", os.path.expanduser("~/miniconda/envs/tpcmp/bin/iqtree"))
TREETIME = os.environ.get(
    "TREETIME", os.path.expanduser("~/miniconda/envs/tpcmp/bin/treetime"))


def origin_horizon(season):
    """The forecast origin of season `season`, as a decimal year."""
    return float(season) + ORIGIN_OFFSET


def segment_offsets(ref_json, segments=H3N2_SEGMENTS):
    """Where each segment starts in the concatenation, and its length.

    The same convention `load_tips` and `load_tree` use. A mutation is written back into
    the segment it falls in and 1-indexed within it, which is how `load_tree` reads it;
    writing them all against one pseudo-segment would parse but would silently shift
    every position by an offset the reader then applies again.
    """
    ref = json.load(open(ref_json))
    off, total = {}, 0
    for s in segments:
        off[s] = (total, len(ref[s]))
        total += len(ref[s])
    return off, total


def write_fasta(path, names, seqs):
    with open(path, "w") as f:
        for n, g in zip(names, seqs):
            f.write(f">{n}\n{g}\n")


def write_dates(path, names, dates):
    """TreeTime's date table. Decimal years, so no round-trip through a calendar."""
    with open(path, "w") as f:
        f.write("name,date\n")
        for n, d in zip(names, dates):
            f.write(f"{n},{float(d):.6f}\n")


def _run(cmd, log, cwd=None):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    with open(log, "ab") as lf:
        lf.write(("\n$ " + " ".join(str(c) for c in cmd) + "\n").encode())
        lf.flush()
        p = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {p.returncode}; see {log}")


def read_fasta_ordered(path):
    out, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].strip().split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def segment_of(pos0, offsets):
    """Which segment a 0-indexed concatenation position falls in, and its offset."""
    for seg, (start, length) in offsets.items():
        if start <= pos0 < start + length:
            return seg, start
    raise IndexError(f"position {pos0} is outside the reference concatenation")


def diff_mutations(parent_seq, child_seq, offsets):
    """The substitutions on one branch, keyed by segment and 1-indexed within it.

    Computed from the two reconstructed sequences rather than copied out of TreeTime's
    own `branch_attrs.mutations`. **That is deliberate and it was found the hard way.**
    TreeTime's auspice export lists a strict subset of the substitutions on a branch --
    the first tree built here had five on a tip that differed from its parent at
    eighteen positions -- and `load_tree` reconstructs each node's sequence by applying
    the list it is given, so an incomplete list produces a tip sequence that is not the
    tip's. Every one of the 91 tips then failed the sequence join and contributed
    nothing to `lbi` or `ne_star`, silently.

    Diffing the sequences makes `load_tree`'s reconstruction exact by construction, and
    `panel_tree verify` is what proves it rather than this docstring.
    """
    out = {}
    for i, (a, b) in enumerate(zip(parent_seq, child_seq)):
        if a != b:
            seg, start = segment_of(i, offsets)
            out.setdefault(seg, []).append(f"{a}{i - start + 1}{b}")
    return out


def convert_auspice(tt_tree, offsets, ref_seq, seq_of, tip_dates=None):
    """TreeTime's auspice tree, reduced to what `load_tree` reads.

    The topology and the *internal* nodes' `num_date` are TreeTime's -- which is what
    `augur refine` would have handed over -- and the branch mutations are recomputed
    from `seq_of`, the reconstructed sequence of every node.

    **A tip's date is the manifest's, not TreeTime's.** TreeTime re-estimates tip dates
    against its clock and will move one that does not fit; `verify` found 12 tips of the
    origin-2015 build sitting after their own origin's horizon, which would have put
    them outside `local_branching_index`'s censoring and outside `_live`'s window. The
    manifest pins each accession's date at admission precisely so that a sequence's
    position in time is not revisable, and a dating run is not an exception to that. The root's own branch carries the
    reference-to-root difference, because `load_tree` starts its reconstruction from the
    reference concatenation and a tree whose inferred root is something else would hand
    back tip sequences that are wrong everywhere the two differ. That costs `ne_star`
    nothing: `mutational_load` walks `while n.parent is not None`, so a root's own
    mutation list is never counted.
    """
    missing = []

    def node(rn, parent_seq):
        name = rn.get("name")
        seq = seq_of.get(name)
        if seq is None:
            missing.append(name)
            seq = parent_seq
        out = {"name": name}
        d = ((rn.get("node_attrs") or {}).get("num_date") or {}).get("value")
        if tip_dates is not None and name in tip_dates:
            d = tip_dates[name]
        if d is not None:
            out["node_attrs"] = {"num_date": {"value": float(d)}}
        m = diff_mutations(parent_seq, seq, offsets)
        if m:
            out["branch_attrs"] = {"mutations": m}
        kids = [node(c, seq) for c in rn.get("children") or []]
        if kids:
            out["children"] = kids
        return out

    root = node(tt_tree, ref_seq)
    if missing:
        raise KeyError(
            f"{len(missing)} nodes have no reconstructed sequence, e.g. {missing[:5]}. "
            f"`ne_star` counts the substitutions on each branch, and a branch with no "
            f"sequence at one end has no count")
    return root


def build_origin(season, names, dates, seqs, out_dir, ref_json, model=DEFAULT_MODEL,
                 threads=4, keep_work=False, iqtree=IQTREE, treetime=TREETIME,
                 provenance=None):
    """One origin's tree, from `S(T)` to the JSON `backbone_compare` reads.

    `names`, `dates` and `seqs` are the WHOLE panel; the date-prefix is taken here, so
    that a caller cannot accidentally pass a set already restricted by a lookback window
    -- `local_branching_index` passes messages over the whole tree below the horizon, not
    only over the window, and a window-restricted tree would quietly change LBI.
    """
    T = origin_horizon(season)
    sel = [i for i, d in enumerate(dates) if float(d) <= T]
    if len(sel) < 4:
        raise ValueError(f"origin {season}: S(T) holds {len(sel)} sequences, which is "
                         f"not a tree")
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, f"work_{int(season)}")
    os.makedirs(work, exist_ok=True)
    log = os.path.join(out_dir, f"build_{int(season)}.log")
    open(log, "w").close()

    aln = os.path.join(work, "aln.fasta")
    dtab = os.path.join(work, "dates.csv")
    write_fasta(aln, [names[i] for i in sel], [seqs[i] for i in sel])
    write_dates(dtab, [names[i] for i in sel], [dates[i] for i in sel])
    n_dup = len(sel) - len({seqs[i] for i in sel})
    print(f"origin {season}: S(T) = {len(sel)} sequences through {T:.2f} "
          f"({n_dup} are repeats of another's genotype)", flush=True)

    # 1. Topology. `--fast` is IQ-TREE's single-tree heuristic -- the same one
    #    `augur tree --method iqtree` reaches for on builds this size. Not a maximum
    #    thoroughness search, and that is a declared cost of doing this nine times.
    pre = os.path.join(work, "iq")
    _run([iqtree, "-s", aln, "-m", model, "--fast", "-T", str(threads),
          "--prefix", pre, "-redo", "--quiet"], log)
    nwk = pre + ".treefile"

    # 2. Dates and ancestral sequences, in one TreeTime run -- which is what
    #    `augur refine` and `augur ancestral` each wrap. `--keep-polytomies` because
    #    resolving them needs information the amino-acid alignment does not carry, and a
    #    fabricated resolution would change LBI's message passing.
    #    `--clock-filter 0` because TreeTime prunes clock outliers by default (4 IQRs)
    #    and **a tree that is missing a sequence of `S(T)` is not a tree of `S(T)`**.
    #    `verify` caught this at origin 2015, 2,373 tips against the 2,374 the manifest
    #    admits; a support genotype whose only tip was pruned would silently carry
    #    `lbi = ne_star = 0`. An H3N2 panel subsampled 90 per month is going to hold
    #    clock outliers and they are part of the panel both sides are scored on.
    tt = os.path.join(work, "tt")
    _run([treetime, "--tree", nwk, "--aln", aln, "--dates", dtab, "--aa",
          "--outdir", tt, "--keep-polytomies", "--reroot", "least-squares",
          "--clock-filter", "0", "--max-iter", "2"], log)

    # 3. TreeTime's own auspice tree already carries `num_date` on every node and the
    #    substitutions on every branch -- it is `augur export v2`'s input in all but
    #    name. Two things are wrong with it for our reader and both are fixed here
    #    rather than in the reader: every substitution is filed under the key `nuc` and
    #    indexed over the whole concatenation, and the inferred root is not the
    #    reference `load_tree` starts its reconstruction from.
    aus = os.path.join(tt, "auspice_tree.json")
    if not os.path.exists(aus):
        raise RuntimeError(f"TreeTime wrote no auspice_tree.json in {tt}; see {log}")
    tt_tree = json.load(open(aus))["tree"]
    anc = read_fasta_ordered(os.path.join(tt, "ancestral_sequences.fasta"))
    offsets, total = segment_offsets(ref_json)
    ref = json.load(open(ref_json))
    ref_seq = "".join(ref[sg] for sg in H3N2_SEGMENTS)
    # **Tips take their observed sequence, not TreeTime's reconstruction of it.** The
    # observed residue is the datum; a reconstruction that disagrees with it at an
    # ambiguous site would make the tip a genotype the panel does not carry, and the
    # join would lose it. How often they disagree is reported, because a large number
    # would mean the alignment handed to IQ-TREE was not the panel's.
    seq_of = dict(anc)
    observed = {names[i]: seqs[i] for i in sel}
    recon_differs = sum(1 for n, g in observed.items()
                        if n in seq_of and seq_of[n] != g)
    seq_of.update(observed)
    bad = sorted({len(v) for v in seq_of.values()} - {total})
    if bad:
        raise ValueError(
            f"origin {season}: TreeTime returned sequences of length {bad} and the "
            f"reference concatenation is {total}. The distance maps are indexed against "
            f"the concatenation, so a mismatch would silently score the wrong residues")

    root = convert_auspice(tt_tree, offsets, ref_seq, seq_of,
                           tip_dates={names[i]: float(dates[i]) for i in sel})
    doc = {
        "version": "v2",
        "meta": {"updated": None},
        "tree": root,
        # Provenance travels with the tree, for the same reason the panel's hash travels
        # with every scored run: a tree file that cannot name the origin it was built for
        # is one `cp` away from being read at the wrong one.
        "antigen_panel_tree": dict(
            origin=int(season), horizon=float(T), n_tips=len(sel),
            n_distinct_genotypes=len(sel) - n_dup, model=model,
            tips_reconstructed_differently=int(recon_differs),
            iqtree=iqtree, treetime=treetime, **(provenance or {})),
    }
    got = set()

    def _tips(n):
        kids = n.get("children") or []
        if not kids:
            got.add(n["name"])
        for c in kids:
            _tips(c)

    _tips(root)
    want = {names[i] for i in sel}
    if got != want:
        raise ValueError(
            f"origin {season}: the tree holds {len(got)} tips and S(T) is {len(want)}; "
            f"missing {sorted(want - got)[:5]}, extra {sorted(got - want)[:5]}. A tree "
            f"that is not over S(T) is not the panel's tree")

    out = os.path.join(out_dir, f"tree_{int(season)}.json")
    with open(out, "w") as f:
        json.dump(doc, f)
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"origin {season}: wrote {out}", flush=True)
    return out


def verify(tree_json, ref_json, panel):
    """Every tip of the tree must be a genotype the panel carries, and dated at or
    before the origin. Returns `(n_tips, n_missing, n_after_horizon)`.

    The join `BackbonePanel._live` does is by sequence, so a tip the panel cannot name is
    a tip whose `lbi` and `ne_star` reach nothing. Zero is the only acceptable number and
    this says so out loud rather than letting a silent drop show up as a flat metric.
    """
    from antiGen.backbone_compare import load_tree

    doc = json.load(open(tree_json))
    prov = doc.get("antigen_panel_tree") or {}
    T = float(prov.get("horizon", float("inf")))
    _, nodes = load_tree(tree_json, ref_json)
    tips = [n for n in nodes if n.is_tip and n.date is not None]
    missing = sum(1 for n in tips if panel.gid.get(n.seq) is None)
    after = sum(1 for n in tips if float(n.date) > T + 1e-9)
    return len(tips), missing, after


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="one tree per origin from the panel")
    add_panel_source_args(b)
    # `--ref` comes from `add_panel_source_args`, where it names the guide tree's
    # reference. Here the panel is an alignment and `--ref` is read only for the segment
    # lengths, which is the same file and the same convention.
    b.set_defaults(ref=os.path.expanduser("~/h3n2/ref.json"))
    b.add_argument("--origins", required=True,
                   help="comma-separated seasons, e.g. 2018,2020,2021")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--keep-work", action="store_true")
    b.add_argument("--skip-existing", action="store_true",
                   help="leave an origin alone if its tree is already written, so an "
                        "array that lost one task is re-runnable without rebuilding "
                        "the ones that landed")

    v = sub.add_parser("verify", help="check a built tree joins to the panel")
    add_panel_source_args(v)
    v.set_defaults(ref=os.path.expanduser("~/h3n2/ref.json"))
    v.add_argument("--trees-dir", required=True)
    v.add_argument("--origins", required=True)
    v.add_argument("--min-allele-freq", type=float, default=0.005)
    v.add_argument("--lookback", type=float, default=5.0)

    a = ap.parse_args(argv)
    origins = [int(x) for x in a.origins.split(",") if x.strip()]

    if a.cmd == "build":
        if not a.alignment:
            ap.error("panel_tree builds from --alignment/--metadata (and --manifest); "
                     "a tree is the output here, not the input")
        tips, dropped, names = load_alignment_dated(
            a.alignment, a.metadata, manifest=a.manifest, with_names=True)
        tips_from_args(ap, a)                       # the manifest preflight, and the hash
        dates = [d for d, _ in tips]
        seqs = [g for _, g in tips]
        print(f"panel: {len(tips)} dated sequences ({dropped} dropped), "
              f"{len(set(seqs))} genotypes")
        for s in origins:
            out = os.path.join(a.out_dir, f"tree_{s}.json")
            if a.skip_existing and os.path.exists(out):
                print(f"origin {s}: {out} exists, skipping")
                continue
            build_origin(s, names, dates, seqs, a.out_dir, a.ref, model=a.model,
                         threads=a.threads, keep_work=a.keep_work,
                         provenance=getattr(a, "panel_provenance", None))
        return 0

    from antiGen.season_momentum import TipPanel
    tips, _skipped, _src, submitted = tips_from_args(ap, a)
    panel = TipPanel(tips, min_allele_freq=a.min_allele_freq, lookback=a.lookback,
                     submitted=submitted)
    bad = 0
    for s in origins:
        path = os.path.join(a.trees_dir, f"tree_{s}.json")
        n, miss, after = verify(path, a.ref, panel)
        flag = "OK" if (miss == 0 and after == 0) else "FAILED"
        bad += (miss > 0) or (after > 0)
        print(f"origin {s}: {n} tips, {miss} not in the panel, {after} after the "
              f"horizon -- {flag}")
    if bad:
        print("\nA tip the panel cannot name contributes nothing to `lbi` or `ne_star`, "
              "and a tip after the horizon is the hindsight these trees remove.",
              file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
