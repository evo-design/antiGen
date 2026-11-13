#!/usr/bin/env python
"""All-lineage post-cutoff PFMs from the placed test corpus (NOTES s160).

`antigen place --out-variant-counts` writes, per cutoff D, the residue counts of the
placed sequences dated >= D -- but only INSIDE the --variant (BA.2) subtree. At
2023-01-01 that is essentially every sequence; at 2021-01-01 it misses Alpha, Delta,
BA.1. This recomputes the same quantity, by the same rules (VariantCounts in
antiGen-data-revised/src/main.cpp), with and without the subtree restriction:

  * a sequence's date is its prepared date, parsed strictly (YYYY-MM-DD, else 0 =
    before every cutoff), as parseDateCode does; "dated >= D" is upper_bound bucketing
  * each residue of the sequence's own aligned string that is one of the 20 letters
    counts once at its site
  * placement: work/matches_all.tsv (exact matches) or the tree node that carries the
    strain as a label (new leaves)

The BA.2 root is not recorded on disk; it is found as the node whose subtree holds the
counts_ba2 sequence totals at every cutoff. GATES, all exact: counts_all (no date, no
subtree), counts_ba2/counts_2021-01-01 and counts_ba2/counts_2023-01-01. Only then is the
all-lineage matrix written.
"""
import collections
import subprocess
import sys

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
import os
# antiGen-data's work/ (the placed test tree) and output/sc2-test (its counts_ba2 dates)
W = os.environ.get("CORPORA_WORK", "/home/ivan.specht/antiGen-data-revised/work")
T = os.environ.get("CORPORA", "/home/ivan.specht/antiGen-data-revised/output") + "/sc2-test"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/large_storage/hielab/ispecht/fitness-inference/postcutoff_counts"
import glob as _g
# Every cutoff counts_ba2 carries (24, ascending); each sequence goes to ONE bucket m =
# the number of cutoffs it is at or after, and M_D is the suffix sum -- the C++'s scheme.
CUTS = sorted(int(f.rsplit("_", 1)[1][:10].replace("-", ""))
              for f in _g.glob(f"{T}/counts_ba2/counts_*.tsv"))
L = 1273


def date_code(s):
    p = s.split("-")
    if len(p) != 3 or len(p[0]) != 4 or not all(x.isdigit() for x in p) or not (1 <= len(p[1]) <= 2) \
            or not (1 <= len(p[2]) <= 2):
        return 0
    y, m, d = int(p[0]), int(p[1]), int(p[2])
    return y * 10000 + m * 100 + d if 1 <= m <= 12 and 1 <= d <= 31 else 0


def read_counts(path):
    seqs, rows = None, []
    for line in open(path):
        if line.startswith("#sequences"):
            seqs = int(line.split()[1])
        elif not line.startswith("#"):
            rows.append([int(x) for x in line.split()[1:21]])
    return seqs, np.array(rows, dtype=np.int64)


# ---- tree: parents and strain -> node
parent, node_of = {}, {}
with open(f"{T}/tree_all_cur.tsv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split("\t")
        parent[int(c[0])] = int(c[1])
        if c[6] != ".":
            for lab in c[6].split(","):
                node_of[lab] = int(c[0])
print(f"tree: {len(parent)} nodes, {len(node_of)} labelled strains", flush=True)
with open(f"{W}/matches_all.tsv") as f:
    next(f)
    for line in f:
        s, n = line.rstrip("\n").split("\t")
        node_of[s] = int(n)
print(f"strains with a placement: {len(node_of)}", flush=True)

# ---- pass 1: dates
dates = {}
p = subprocess.Popen(["cut", "-f1,2", f"{W}/prepared_all.tsv"], stdout=subprocess.PIPE, text=True)
for line in p.stdout:
    if line.startswith("#"):
        continue
    s, d = line.rstrip("\n").split("\t")
    dates[s] = date_code(d)
p.wait()
print(f"prepared: {len(dates)} sequences; unplaced {sum(s not in node_of for s in dates)}", flush=True)

# ---- BA.2 root: the node whose subtree's per-cutoff sequence totals match counts_ba2
lab = lambda c: f"{c // 10000}-{c // 100 % 100:02d}-{c % 100:02d}"
ba2_files = {c: f"{T}/counts_ba2/counts_{lab(c)}.tsv" for c in CUTS}
want = [read_counts(ba2_files[c])[0] for c in CUTS]
import bisect
per_node = collections.defaultdict(lambda: [0] * len(CUTS))
for s_, n in node_of.items():
    d = dates.get(s_)
    if d is None:
        continue
    m = bisect.bisect_right(CUTS, d)          # cutoffs at or before d = cutoffs d is >= of
    b = per_node[n]
    for k in range(m):
        b[k] += 1
children = collections.defaultdict(list)
for v, u in parent.items():
    children[u].append(v)
order, stack = [], [r for r, u in parent.items() if u == -1]
while stack:
    v = stack.pop(); order.append(v); stack.extend(children[v])
tot = {}
for v in reversed(order):
    a = list(per_node.get(v, [0] * len(CUTS)))
    for c in children[v]:
        t = tot[c]
        for k in range(len(CUTS)):
            a[k] += t[k]
    tot[v] = a
cand = [v for v, a in tot.items() if a == want]
print(f"BA.2 root candidates matching all {len(CUTS)} counts_ba2 totals: {cand[:5]} ({len(cand)})", flush=True)
assert cand, "no node reproduces the counts_ba2 sequence totals"
depth = {}
for v in order:
    depth[v] = 0 if parent[v] == -1 else depth[parent[v]] + 1
root = min(cand, key=depth.get)
inside = set()
stack = [root]
while stack:
    v = stack.pop(); inside.add(v); stack.extend(children[v])
print(f"BA.2 root: node {root}; {len(inside)} nodes in its subtree", flush=True)

# ---- pass 2: residue counts, one bucket per (inside BA.2, m)
lut = np.full(256, 255, np.uint8)
for i, a in enumerate(AA):
    lut[ord(a)] = i
NB = len(CUTS) + 1
B = np.zeros((2, NB, L, 20), np.int64)
NSB = np.zeros((2, NB), np.int64)


def flush(buf, keys):
    arr = lut[np.frombuffer(b"".join(buf), np.uint8).reshape(len(buf), L)]
    keys = np.array(keys)
    for key in np.unique(keys):
        a = arr[keys == key]
        ins, m = divmod(int(key), NB)
        for i in range(20):
            B[ins, m, :, i] += (a == i).sum(0)


buf, keys, n_seen = [], [], 0
with open(f"{W}/prepared_all.tsv", "rb") as f:
    for line in f:
        if line.startswith(b"#"):
            continue
        s_, d, q = line.rstrip(b"\n").split(b"\t")
        n = node_of.get(s_.decode())
        if n is None:
            continue
        m = bisect.bisect_right(CUTS, date_code(d.decode()))
        ins = int(n in inside)
        assert len(q) == L, (s_, len(q))
        NSB[ins, m] += 1
        buf.append(q); keys.append(ins * NB + m); n_seen += 1
        if len(buf) == 50000:
            flush(buf, keys); buf, keys = [], []
            if n_seen % 1000000 == 0:
                print(f"  {n_seen} sequences", flush=True)
if buf:
    flush(buf, keys)


def M_at(k, ins=(0, 1)):   # sequences dated >= CUTS[k]: buckets m > k
    return sum(B[i, k + 1:].sum(0) for i in ins), int(sum(NSB[i, k + 1:].sum() for i in ins))


allM, alln = B.sum((0, 1)), int(NSB.sum())
n, ref = read_counts(f"{T}/counts_all_cur.tsv")
assert alln == n and int(np.abs(allM - ref).max()) == 0, ("all", alln, n)
print(f"GATE OK: all sequences reproduce counts_all_cur.tsv ({n}) exactly")
for k, c in enumerate(CUTS):
    m_, n_ = M_at(k, ins=(1,))
    n, ref = read_counts(ba2_files[c])
    assert n_ == n and int(np.abs(m_ - ref).max()) == 0, (lab(c), n_, n)
print(f"GATE OK: the BA.2 subtree reproduces all {len(CUTS)} counts_ba2 matrices exactly")

import os
os.makedirs(OUT, exist_ok=True)
for k, c in enumerate(CUTS):
    m_, n_ = M_at(k)
    with open(f"{OUT}/counts_all_lineages_{lab(c)}.tsv", "w") as f:
        f.write(f"#sequences\t{n_}\n#sites\t{L}\n#site\t" + "\t".join(AA) + "\n")
        for i in range(L):
            f.write(f"{i + 1}\t" + "\t".join(str(x) for x in m_[i]) + "\n")
    print(f"wrote counts_all_lineages_{lab(c)}.tsv ({n_} sequences)")
