#!/usr/bin/env python
"""The backbone phylogeny, before the attachment phase, as a set of query genotypes.

**Why this tree and not the examples file.** ``antiGen-data`` builds its training and
testing examples in two stages: it reads the guide phylogeny (``tree.json``, a
Nextstrain build whose branches carry mutation lists), then *attaches* every aligned
sequence to it by parsimony. The examples file is the merged result. For the question
this module exists to serve -- *which genetic background does a given mutation arise
from?* -- the attachment phase is a liability: it lands tens of thousands of tips on the
tree and the same mutation is then recorded against hundreds of near-identical
backgrounds, so a count of "backgrounds that host this mutation" mostly counts sampling
depth. The guide tree is one ancestral reconstruction per branch, so a mutation appears
once per branch it actually arose on.

What is reconstructed here is exactly what ``antiGen-data`` reconstructs for those same
nodes: the genotype at a node is the reference sequence with every branch mutation on
the path from the root applied in order (``assign_spike_sequences`` in
``antiGen-data/src/main.cpp``). Two checks make that claim testable rather than
asserted, and both are run by ``build_backbone``:

* every branch mutation's wild-type letter must equal the residue the reconstruction
  already has at that position (0 of 2,839 disagree on the sc2 guide tree);
* the root genotype must equal the reference sequence the data generator starts from.

**Queries are genotypes, not nodes.** Most branches of the guide tree carry no spike
mutation at all -- the sc2 tree has 8,859 nodes and 2,839 spike mutation events -- so
thousands of nodes share a genotype. A model sees only the sequence, so nodes with the
same genotype are the same query and are collapsed into one. The sc2 guide tree's 4,034
post-cutoff internal nodes collapse to 402 distinct spike genotypes.

**The cutoff.** ``cutoff_years`` is counted from the root date, matching
``antiGen-data``'s CLI argument, so a query here is post-cutoff in exactly the sense the
examples file labels ``TESTING``. Queries are restricted to post-cutoff genotypes so
that nothing a model trained on the training window has already seen is used as a
background.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Backbone:
    """The guide tree, with per-node ancestry and the post-cutoff query genotypes."""

    seq_len: int
    allowed_aas: str
    root_time: float
    cutoff: float
    root_seq: str

    # Per node (DFS pre-order over the guide tree).
    node_parent: np.ndarray            # (N,) int32, -1 at the root
    node_time: np.ndarray              # (N,) float64, decimal year
    node_in: np.ndarray                # (N,) int32, DFS entry index
    node_out: np.ndarray               # (N,) int32, DFS exit index (exclusive)
    node_clade: list                   # (N,) clade_membership, "" when absent
    node_qidx: np.ndarray              # (N,) int32, query index or -1
    branch_muts: list                  # (N,) [(pos0, aa_idx)] on the branch INTO the node

    # Per query genotype.
    query_seq: list = field(default_factory=list)       # (Q,) genotype strings
    query_nodes: list = field(default_factory=list)     # (Q,) lists of node indices
    query_branches: list = field(default_factory=list)  # (Q,) post-cutoff child branches

    # hosts[(pos0, aa_idx)][node] = post-cutoff child branches of that NODE carrying the
    # mutation. Keyed by node rather than by query genotype because the contrast has to
    # exclude a host's ancestors and descendants, which is a statement about nodes; the
    # query-level view is a groupby on ``node_qidx``.
    hosts: dict = field(default_factory=dict)

    report: dict = field(default_factory=dict)

    def is_ancestor(self, a, b):
        """Is node ``a`` an ancestor of (or equal to) node ``b``?"""
        return self.node_in[a] <= self.node_in[b] < self.node_out[a]

    def related(self, a, b):
        """Are nodes ``a`` and ``b`` on the same root-to-tip path?"""
        return self.is_ancestor(a, b) or self.is_ancestor(b, a)


def build_backbone(tree_json, ref_json, segment, seq_len, allowed_aas,
                   cutoff_years, eval_range=None):
    """Parse a Nextstrain tree into a :class:`Backbone`.

    ``eval_range`` restricts which mutations are recorded as hosted (the RBD, for sc2);
    the genotypes themselves are always reconstructed over the whole segment.
    """
    with open(ref_json) as fh:
        ref = json.load(fh)
    if segment not in ref:
        raise SystemExit(f"{ref_json}: no '{segment}' key (has {sorted(ref)[:8]})")
    root_seq = ref[segment][:seq_len]
    if len(root_seq) != seq_len:
        raise SystemExit(f"{ref_json}: '{segment}' is {len(ref[segment])} long, "
                         f"cannot supply {seq_len} residues")

    with open(tree_json) as fh:
        tree = json.load(fh)["tree"]
    root_time = tree["node_attrs"]["num_date"]["value"]
    cutoff = root_time + cutoff_years
    in_eval = None if eval_range is None else set(eval_range)

    parents, times, clades, seqs = [], [], [], []
    node_in, node_out = [], []
    branch_muts = []          # per node: [(pos0, alt_idx)] on the branch INTO it
    wt_mismatch = [0]
    unparsable = [0]
    counter = [0]

    def visit(node, parent_idx, parent_seq):
        idx = len(parents)
        parents.append(parent_idx)
        times.append(node["node_attrs"]["num_date"]["value"])
        clades.append(node["node_attrs"].get("clade_membership", {}).get("value", ""))
        node_in.append(counter[0])
        counter[0] += 1
        node_out.append(-1)

        seq = list(parent_seq)
        muts = []
        for raw in node.get("branch_attrs", {}).get("mutations", {}).get(segment, []):
            try:
                wt, pos1, alt = raw[0], int(raw[1:-1]), raw[-1]
            except ValueError:
                unparsable[0] += 1
                continue
            if pos1 < 1 or pos1 > seq_len or alt not in allowed_aas:
                unparsable[0] += 1
                continue
            if seq[pos1 - 1] != wt:
                wt_mismatch[0] += 1
            seq[pos1 - 1] = alt
            muts.append((pos1 - 1, allowed_aas.index(alt)))
        seq = "".join(seq)
        seqs.append(seq)
        branch_muts.append(muts)

        for child in node.get("children", []):
            visit(child, idx, seq)
        node_out[idx] = counter[0]
        return idx

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 100000))
    try:
        visit(tree, -1, root_seq)
    finally:
        sys.setrecursionlimit(old_limit)

    n = len(parents)
    node_parent = np.asarray(parents, dtype=np.int32)
    node_time = np.asarray(times, dtype=np.float64)
    node_in_a = np.asarray(node_in, dtype=np.int32)
    node_out_a = np.asarray(node_out, dtype=np.int32)

    if seqs[0] != root_seq:
        raise SystemExit("root genotype != reference sequence")

    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[node_parent[i]].append(i)

    # A query is a post-cutoff node with at least one post-cutoff child branch;
    # identical genotypes are one query.
    seq_to_q, query_seq, query_nodes, query_branches = {}, [], [], []
    node_qidx = np.full(n, -1, dtype=np.int32)
    hosts = {}
    for i in range(n):
        if node_time[i] <= cutoff:
            continue
        kids = [c for c in children[i] if node_time[c] > cutoff]
        if not kids:
            continue
        q = seq_to_q.get(seqs[i])
        if q is None:
            q = len(query_seq)
            seq_to_q[seqs[i]] = q
            query_seq.append(seqs[i])
            query_nodes.append([])
            query_branches.append(0)
        node_qidx[i] = q
        query_nodes[q].append(i)
        query_branches[q] += len(kids)
        for c in kids:
            for cell in branch_muts[c]:
                if in_eval is not None and cell[0] not in in_eval:
                    continue
                hosts.setdefault(cell, {})
                hosts[cell][i] = hosts[cell].get(i, 0) + 1

    report = dict(
        tree_json=tree_json, segment=segment, seq_len=seq_len,
        root_time=root_time, cutoff=cutoff, n_nodes=n,
        n_leaves=sum(1 for i in range(n) if not children[i]),
        n_mutations=sum(len(m) for m in branch_muts),
        n_wt_mismatch=wt_mismatch[0], n_unparsable=unparsable[0],
        n_query_nodes=int((node_qidx >= 0).sum()), n_queries=len(query_seq),
        n_host_cells=len(hosts),
        n_host_events=sum(sum(d.values()) for d in hosts.values()),
    )
    return Backbone(seq_len=seq_len, allowed_aas=allowed_aas, root_time=root_time,
                    cutoff=cutoff, root_seq=root_seq, node_parent=node_parent, node_time=node_time,
                    node_in=node_in_a, node_out=node_out_a, node_clade=clades,
                    node_qidx=node_qidx, branch_muts=branch_muts, query_seq=query_seq, query_nodes=query_nodes,
                    query_branches=query_branches, hosts=hosts, report=report)


def describe(report):
    return (
        f"backbone {report['tree_json']} [{report['segment']}]\n"
        f"  {report['n_nodes']} nodes ({report['n_leaves']} leaves), "
        f"{report['n_mutations']} branch mutations, "
        f"{report['n_wt_mismatch']} wild-type mismatches, "
        f"{report['n_unparsable']} unparsable\n"
        f"  root {report['root_time']:.3f}, cutoff {report['cutoff']:.3f}\n"
        f"  {report['n_query_nodes']} post-cutoff query nodes -> "
        f"{report['n_queries']} distinct genotypes\n"
        f"  {report['n_host_cells']} cells hosted in range, "
        f"{report['n_host_events']} host events"
    )
