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
W = "/home/ivan.specht/antiGen-data-revised/work"
T = "/home/ivan.specht/antiGen-data-revised/output/sc2-test"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/large_storage/hielab/ispecht/fitness-inference/postcutoff_counts"
CUTS = [20210101, 20230101]
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

# ---- BA.2 root: the node whose subtree's sequence totals match counts_ba2 at every cutoff
ba2_files = {c: f"{T}/counts_ba2/counts_{c // 10000}-{c // 100 % 100:02d}-{c % 100:02d}.tsv" for c in CUTS}
want = {c: read_counts(f)[0] for c, f in ba2_files.items()}
n_first = read_counts(f"{T}/counts_ba2/counts_2020-04-01.tsv")[0]
sub = collections.Counter()          # node -> sequences placed in its subtree, by bucket
per_node = collections.defaultdict(lambda: [0, 0, 0])   # all dated>=2020-04-01, >=2021, >=2023
for s, n in node_of.items():
    d = dates.get(s)
    if d is None:
        continue
    b = per_node[n]
    b[0] += d >= 20200401; b[1] += d >= CUTS[0]; b[2] += d >= CUTS[1]
children = collections.defaultdict(list)
for v, u in parent.items():
    children[u].append(v)
order, stack = [], [r for r, u in parent.items() if u == -1]
while stack:
    v = stack.pop(); order.append(v); stack.extend(children[v])
tot = {}
for v in reversed(order):
    a = list(per_node.get(v, (0, 0, 0)))
    for c in children[v]:
        t = tot[c]; a[0] += t[0]; a[1] += t[1]; a[2] += t[2]
    tot[v] = a
cand = [v for v, a in tot.items() if a[0] == n_first and a[1] == want[CUTS[0]] and a[2] == want[CUTS[1]]]
print(f"BA.2 root candidates matching counts_ba2 totals: {cand[:5]} ({len(cand)})", flush=True)
assert cand, "no node reproduces the counts_ba2 sequence totals"
# the top-most candidate (a chain of single-child nodes would all match)
depth = {}
for v in order:
    depth[v] = 0 if parent[v] == -1 else depth[parent[v]] + 1
root = min(cand, key=depth.get)
inside = set()
stack = [root]
while stack:
    v = stack.pop(); inside.add(v); stack.extend(children[v])
print(f"BA.2 root: node {root}; {len(inside)} nodes in its subtree", flush=True)

# ---- pass 2: residue counts
lut = np.full(256, 255, np.uint8)
for i, a in enumerate(AA):
    lut[ord(a)] = i
M = {k: np.zeros((L, 20), np.int64) for k in ("all", "ba2_2021", "ba2_2023", "all_2021", "all_2023")}
NS = collections.Counter()


def flush(buf, keys):
    arr = lut[np.frombuffer(b"".join(buf), np.uint8).reshape(len(buf), L)]
    for key in set(k for ks in keys for k in ks):
        sel = np.array([key in ks for ks in keys])
        a = arr[sel]
        for i in range(20):
            M[key][:, i] += (a == i).sum(0)


buf, keys = [], []
with open(f"{W}/prepared_all.tsv", "rb") as f:
    for line in f:
        if line.startswith(b"#"):
            continue
        s, d, q = line.rstrip(b"\n").split(b"\t")
        s = s.decode()
        n = node_of.get(s)
        if n is None:
            continue
        dc = date_code(d.decode())
        ks = ["all"]
        if dc >= CUTS[0]: ks.append("all_2021")
        if dc >= CUTS[1]: ks.append("all_2023")
        if n in inside:
            if dc >= CUTS[0]: ks.append("ba2_2021")
            if dc >= CUTS[1]: ks.append("ba2_2023")
        for k in ks:
            NS[k] += 1
        assert len(q) == L, (s, len(q))
        buf.append(q); keys.append(ks)
        if len(buf) == 50000:
            flush(buf, keys); buf, keys = [], []
            if NS["all"] % 1000000 == 0:
                print(f"  {NS['all']} sequences", flush=True)
if buf:
    flush(buf, keys)
print({k: v for k, v in NS.items()}, flush=True)

for key, path in (("all", f"{T}/counts_all_cur.tsv"), ("ba2_2021", ba2_files[CUTS[0]]),
                  ("ba2_2023", ba2_files[CUTS[1]])):
    n, ref = read_counts(path)
    assert NS[key] == n, (key, NS[key], n)
    diff = int(np.abs(M[key] - ref).max())
    assert diff == 0, (key, diff)
    print(f"GATE OK: {key} reproduces {path.split('/')[-1]} ({n} sequences) exactly")

import os
os.makedirs(OUT, exist_ok=True)
for key, lab in (("all_2021", "2021-01-01"), ("all_2023", "2023-01-01")):
    with open(f"{OUT}/counts_all_lineages_{lab}.tsv", "w") as f:
        f.write(f"#sequences\t{NS[key]}\n#sites\t{L}\n#site\t" + "\t".join(AA) + "\n")
        for i in range(L):
            f.write(f"{i + 1}\t" + "\t".join(str(x) for x in M[key][i]) + "\n")
    print(f"wrote {OUT}/counts_all_lineages_{lab}.tsv ({NS[key]} sequences)")
