"""Rewrite a 22-letter examples file in the 20-letter frame antiGen-data writes by default.

Gap and stop residues are filled from the parent (the column consensus at a root), targets
to gap/stop or to the residue already there are dropped, and an edge left with no change
gets MUTATION_FROM_PARENT None.

    python reproduce/tools/fill_gap_stop.py IN OUT
"""
import collections
import sys

MISSING = set("-*")


def parse(path):
    recs, cur = [], []
    for line in open(path):
        line = line.rstrip("\n")
        if line:
            cur.append(line)
        elif cur:
            recs.append(cur)
            cur = []
    if cur:
        recs.append(cur)
    return recs


def fields(rec):
    i_mut = rec.index("MUTATION_FROM_PARENT:") + 1
    i_seq = rec.index("IS_VARIANT_ROOT:") + 1
    i_tgt = rec.index("MUTATIONS_TO_CHILDREN:") + 1
    return i_mut, i_seq, i_tgt


def main(src, dst):
    recs = parse(src)
    if not recs:
        open(dst, "w").close()
        return
    by_id, parent = {}, {}
    for r in recs:
        rid = r[0].split(":", 1)[1].strip()
        pid = r[1].split(":", 1)[1].strip()
        by_id[rid] = r
        parent[rid] = None if pid == "None" else pid

    seqs = {rid: r[fields(r)[1]] for rid, r in by_id.items()}
    cols = [collections.Counter(s[i] for s in seqs.values() if s[i] not in MISSING)
            for i in range(len(next(iter(seqs.values()))))]
    consensus = [c.most_common(1)[0][0] if c else "A" for c in cols]

    filled = {}

    def fill(rid):
        stack = []
        while rid not in filled:
            stack.append(rid)
            p = parent[rid]
            if p is None or p in filled:
                break
            rid = p
        for r in reversed(stack):
            p = parent[r]
            ref = consensus if p is None else filled[p]
            filled[r] = "".join(ref[i] if c in MISSING else c for i, c in enumerate(seqs[r]))

    for rid in by_id:
        fill(rid)

    out = []
    for rid, r in by_id.items():
        i_mut, i_seq, i_tgt = fields(r)
        seq = filled[rid]
        r = list(r)
        r[i_seq] = seq
        p = parent[rid]
        if p is not None and r[i_mut] != "None":
            _, pos, _ = r[i_mut].split(",")
            pos = int(pos)
            frm, to = filled[p][pos], seq[pos]
            r[i_mut] = "None" if frm == to else f"{frm},{pos},{to}"
        tgts = [t for t in r[i_tgt:] if t[-1] not in MISSING and seq[int(t.split(",")[0])] != t[-1]]
        out.append("\n".join(r[:i_tgt] + tgts))
    with open(dst, "w") as f:
        f.write("\n\n".join(out) + "\n\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
