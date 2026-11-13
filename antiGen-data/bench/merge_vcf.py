#!/usr/bin/env python3
"""One VCF of every tip: the guide tree's tips and the appended samples.

The common scorer rebuilds each final tree's mutation annotation from tip
genotypes alone, so it needs a VCF that covers both sets of tips. The two
inputs list different positions: a position absent from one file means that
file's samples carry REF there, not that they are missing, so the absent side
is filled with 0. Where both list a position the ALT lists are merged and the
second file's genotype indices remapped onto the merged list. Missing calls
('.') are kept as missing.

Haploid, one character per genotype, as both guide.vcf (tree_to_vcf.py) and
matUtils extract write them; anything else is refused rather than guessed at.

    python3 -I bench/merge_vcf.py guide.vcf append_N.vcf out.vcf
"""
import sys
import numpy as np


def rows(path):
    """(header samples, iterator of (pos, ref, alts, gt-bytes))."""
    f = open(path, "rb")
    samples = None
    for raw in f:
        if raw.startswith(b"##"):
            continue
        if raw.startswith(b"#CHROM"):
            samples = raw.rstrip(b"\n").split(b"\t")[9:]
            break

    def it():
        n = len(samples)
        for raw in f:
            p = raw.rstrip(b"\n").split(b"\t", 9)
            g = p[9]
            if len(g) != 2 * n - 1:
                raise SystemExit(f"{path}: POS {p[1].decode()} is not one character per genotype")
            yield int(p[1]), p[3], p[4].split(b","), g
        f.close()
    return samples, it()


def main():
    a_path, b_path, out = sys.argv[1:4]
    sa, ra = rows(a_path)
    sb, rb = rows(b_path)
    if set(sa) & set(sb):
        raise SystemExit("the two VCFs share sample names")
    zeros_a = b"\t".join([b"0"] * len(sa))
    zeros_b = b"\t".join([b"0"] * len(sb))
    o = open(out, "wb")
    o.write(b"##fileformat=VCFv4.2\n")
    o.write(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + b"\t".join(sa + sb) + b"\n")

    def emit(pos, ref, alts, ga, gb):
        o.write(b"MN908947.3\t%d\t.\t%s\t%s\t.\tPASS\t.\tGT\t" % (pos, ref, b",".join(alts)))
        o.write(ga + b"\t" + gb + b"\n")

    na = next(ra, None)
    nb = next(rb, None)
    both = only_a = only_b = remapped = 0
    while na or nb:
        if nb is None or (na and na[0] < nb[0]):
            emit(na[0], na[1], na[2], na[3], zeros_b); only_a += 1; na = next(ra, None)
        elif na is None or nb[0] < na[0]:
            emit(nb[0], nb[1], nb[2], zeros_a, nb[3]); only_b += 1; nb = next(rb, None)
        else:
            pos, refa, altsa, ga = na
            _, refb, altsb, gb = nb
            if refa != refb:
                raise SystemExit(f"POS {pos}: REF {refa!r} vs {refb!r}")
            alts = list(altsa)
            for x in altsb:
                if x not in alts:
                    alts.append(x)
            if alts[:len(altsb)] != altsb:
                # b's index k (1-based) -> merged index; one byte table over the block
                lut = np.arange(256, dtype=np.uint8)
                for k, x in enumerate(altsb, start=1):
                    m = alts.index(x) + 1
                    if m > 9:
                        raise SystemExit(f"POS {pos}: more than 9 ALT alleles")
                    lut[ord("0") + k] = ord("0") + m
                arr = np.frombuffer(gb, dtype=np.uint8)
                gb = lut[arr].tobytes()
                remapped += 1
            emit(pos, refa, alts, ga, gb); both += 1
            na = next(ra, None); nb = next(rb, None)
    o.close()
    print(f"{len(sa)} + {len(sb)} samples; positions: {both} in both ({remapped} remapped), "
          f"{only_a} only in {a_path}, {only_b} only in {b_path} -> {out}")


if __name__ == "__main__":
    main()
