#!/usr/bin/env python3
"""Put one N's inputs into the order both tools will place them in.

matUtils writes samples in the order of the UCSC tree it extracts them from, so
the VCFs (and the FASTAs rebuilt from them) are phylogenetically clustered:
each sample sits next to its relatives. Placing in that order lets any tool
that exploits locality look better than it is. Instead, both tools place:

    by collection date, and within one day in a seeded pseudo-random order --
    md5(seed + strain name), so the order is fixed and reproducible

which is also our tool's own default ordering (chronological, shuffled within
a day). The realised order is computed here, once, and written into both
inputs: the FASTA records and the VCF's sample columns. Our tool reads the
FASTA with --keep-order; UShER places in VCF column order.

With --batch B the VCF is also cut into consecutive B-sample batches of that
same order, for chaining usher-sampled runs. Rows at which no sample in a batch
carries an ALT are left out of that batch's file.

    python3 -I bench/reorder_inputs.py IN_DIR OUT_DIR N [--batch 100000]

writes OUT_DIR/append_N.{fasta,vcf}, append_N_meta.tsv, order_N.txt and, with
--batch, batch_N_<i>.vcf.
"""
import argparse
import hashlib
import os

import numpy as np

SEED = "antigen-bench-chrono-1"


def key(name, date):
    return (date, hashlib.md5((SEED + name).encode()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("n", type=int)
    ap.add_argument("--batch", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    src = lambda ext: os.path.join(a.in_dir, f"append_{a.n}{ext}")
    dst = lambda ext: os.path.join(a.out_dir, f"append_{a.n}{ext}")

    # --- the order -----------------------------------------------------------
    dates = {}
    with open(src("_meta.tsv")) as f:
        next(f)
        for line in f:
            s, d = line.rstrip("\n").split("\t")
            dates[s] = d
    if any(len(d) != 10 for d in dates.values()):
        raise SystemExit("every sample needs a full YYYY-MM-DD date")
    order = sorted(dates, key=lambda s: key(s, dates[s]))
    rank = {s: i for i, s in enumerate(order)}
    with open(os.path.join(a.out_dir, f"order_{a.n}.txt"), "w") as f:
        f.write("\n".join(order) + "\n")
    with open(dst("_meta.tsv"), "w") as f:
        f.write("strain\tdate\n")
        for s in order:
            f.write(f"{s}\t{dates[s]}\n")
    print(f"{len(order)} samples, {order and dates[order[0]]} .. {order and dates[order[-1]]}")

    # --- FASTA: one header line and one sequence line per record --------------
    offs = {}
    with open(src(".fasta"), "rb") as f:
        while True:
            pos = f.tell()
            h = f.readline()
            if not h:
                break
            seq = f.readline()
            offs[h[1:].rstrip(b"\n").decode()] = (pos, len(h) + len(seq))
    if set(offs) != set(order):
        raise SystemExit("FASTA and metadata disagree about the samples")
    with open(src(".fasta"), "rb") as f, open(dst(".fasta"), "wb") as o:
        for s in order:
            p, n = offs[s]
            f.seek(p)
            o.write(f.read(n))
    print(f"wrote {dst('.fasta')}")

    # --- VCF: permute the sample columns --------------------------------------
    nb = (len(order) + a.batch - 1) // a.batch if a.batch else 0
    bouts = [open(os.path.join(a.out_dir, f"batch_{a.n}_{i}.vcf"), "wb") for i in range(nb)]
    kept = [0] * nb
    with open(src(".vcf"), "rb") as f, open(dst(".vcf"), "wb") as o:
        perm = None
        for raw in f:
            if raw.startswith(b"##"):
                o.write(raw)
                for b in bouts:
                    b.write(raw)
                continue
            line = raw.rstrip(b"\n")
            fixed, block = line.split(b"\t", 9)[:9], line.split(b"\t", 9)[9]
            if raw.startswith(b"#CHROM"):
                cols = block.decode().split("\t")
                if set(cols) != set(order):
                    raise SystemExit("VCF and metadata disagree about the samples")
                perm = np.array([cols.index(s) for s in order]) if len(cols) < 2000 else \
                    np.array([i for _, i in sorted((rank[c], i) for i, c in enumerate(cols))])
                head = b"\t".join(fixed) + b"\t"
                o.write(head + "\t".join(order).encode() + b"\n")
                for i, b in enumerate(bouts):
                    b.write(head + "\t".join(order[i * a.batch:(i + 1) * a.batch]).encode() + b"\n")
                continue
            n = len(order)
            if len(block) != 2 * n - 1:
                raise SystemExit(f"POS {fixed[1].decode()}: not one character per genotype")
            gt = np.frombuffer(block, dtype=np.uint8)[0::2][perm]
            out = np.full(2 * n - 1, 9, dtype=np.uint8)  # tabs
            out[0::2] = gt
            prefix = b"\t".join(fixed) + b"\t"
            o.write(prefix + out.tobytes() + b"\n")
            for i, b in enumerate(bouts):
                g = gt[i * a.batch:(i + 1) * a.batch]
                if (g != ord("0")).any():
                    ob = np.full(2 * len(g) - 1, 9, dtype=np.uint8)
                    ob[0::2] = g
                    b.write(prefix + ob.tobytes() + b"\n")
                    kept[i] += 1
    for b in bouts:
        b.close()
    print(f"wrote {dst('.vcf')}" + (f" and {nb} batches (rows kept: {kept})" if nb else ""))


if __name__ == "__main__":
    main()
