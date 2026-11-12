#!/usr/bin/env python
"""Split every pre-2023 coronavirus spike record into tree groups by taxonomy (NOTES s155).

Input: the one Coronaviridae fetch (fetch_group.py --name coronaviridae, SARS-CoV-2
excluded by taxon, PDB/patent excluded, 300-1600 aa: full-length spikes AND partial
ones, the latter attached but never in a backbone).

Each record's taxon comes from its GenPept source feature (db_xref taxon:N); its
lineage from NCBI Taxonomy. Groups, coarsest only when needed:

  * every sarbecovirus is ONE group, built in the SARS-CoV-2 frame;
  * a SPECIES with >= --min full-length records is its own group;
  * the records of smaller species pool at their SUBGENUS, if that pool reaches --min;
  * what is left pools at the GENUS, if that reaches --min; otherwise it is dropped
    and listed in the report.

Writes <out>/<group>/{raw.faa, records.gpc.xml, members.tsv} for build_cov_tree.py, and
<out>/groups.tsv.
"""
import argparse
import collections
import os
import re
import sys
import xml.etree.ElementTree as ET

from Bio import SeqIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_group import call  # noqa: E402


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the coronaviridae fetch directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min", type=int, default=20, help="full-length records a group needs")
    ap.add_argument("--full", type=int, default=900, help="aa length that counts as full-length")
    a = ap.parse_args()
    text = open(os.path.join(a.src, "records.gpc.xml"), encoding="utf-8").read()
    blocks = re.findall(r"<INSDSeq>.*?</INSDSeq>", text, flags=re.S)
    rec_xml, taxid = {}, {}
    for b in blocks:
        acc = re.search(r"<INSDSeq_accession-version>(.*?)</INSDSeq_accession-version>", b).group(1)
        t = re.search(r"<INSDQualifier_value>taxon:(\d+)</INSDQualifier_value>", b)
        rec_xml[acc] = b
        taxid[acc] = t.group(1) if t else None
    seqs = {r.id: r for r in SeqIO.parse(os.path.join(a.src, "raw.faa"), "fasta")}
    # lineages
    lin = {}
    uniq = sorted({t for t in taxid.values() if t})
    for i in range(0, len(uniq), 200):
        x = ET.fromstring(call("efetch.fcgi", db="taxonomy", id=",".join(uniq[i:i + 200])))
        for tx in x.findall("Taxon"):
            ranks = {l.findtext("Rank"): l.findtext("ScientificName") for l in tx.iter("Taxon")}
            ranks[tx.findtext("Rank")] = tx.findtext("ScientificName")
            lin[tx.findtext("TaxId")] = ranks
            for alias in tx.iter("AkaTaxIds"):
                for t in alias.iter("TaxId"):
                    lin[t.text] = ranks
    def rank(acc, r):
        v = lin.get(taxid.get(acc), {}).get(r)
        # NCBI's catch-all "species" ("Bat coronavirus", "Alphacoronavirus sp.") are
        # mixed bags, not species: their records pool at subgenus / genus instead.
        if r == "species" and v and (v.endswith(" sp.") or v == "Bat coronavirus"
                                     or "unclassified" in v):
            return None
        return v
    full = {acc for acc, rr in seqs.items() if len(rr.seq) >= a.full}
    # species counts over full-length records
    sp_n = collections.Counter(rank(acc, "species") or "?" for acc in seqs if acc in full)
    group = {}
    for acc in seqs:
        if rank(acc, "subgenus") == "Sarbecovirus":
            group[acc] = "sarbecovirus"
        elif rank(acc, "species") and sp_n[rank(acc, "species")] >= a.min:
            group[acc] = "sp_" + slug(rank(acc, "species"))
    for level in ("subgenus", "genus"):
        pool = collections.Counter(rank(acc, level) for acc in seqs
                                   if acc not in group and acc in full and rank(acc, level))
        for acc in seqs:
            if acc not in group and rank(acc, level) and pool[rank(acc, level)] >= a.min:
                group[acc] = f"{level[:3]}_" + slug(rank(acc, level))
    os.makedirs(a.out, exist_ok=True)
    by = collections.defaultdict(list)
    for acc, g in group.items():
        by[g].append(acc)
    rows = []
    for g, accs in sorted(by.items()):
        d = os.path.join(a.out, g)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "raw.faa"), "w") as f:
            SeqIO.write([seqs[x] for x in accs], f, "fasta")
        open(os.path.join(d, "records.gpc.xml"), "w").write(
            "<INSDSet>" + "".join(rec_xml[x] for x in accs if x in rec_xml) + "</INSDSet>")
        with open(os.path.join(d, "members.tsv"), "w") as f:
            f.write("accession\tlength\tspecies\tsubgenus\tgenus\n")
            for x in accs:
                f.write(f"{x}\t{len(seqs[x].seq)}\t{rank(x, 'species')}\t{rank(x, 'subgenus')}\t{rank(x, 'genus')}\n")
        nfull = sum(x in full for x in accs)
        sps = collections.Counter(rank(x, "species") for x in accs)
        rows.append((g, len(accs), nfull, len(accs) - nfull, rank(accs[0], "genus"),
                     "; ".join(f"{k} ({v})" for k, v in sps.most_common(4))))
    dropped = [acc for acc in seqs if acc not in group]
    with open(os.path.join(a.out, "groups.tsv"), "w") as f:
        f.write("group\trecords\tfull_length\tpartial\tgenus\ttop_species\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"{len(seqs)} records -> {len(rows)} groups; {len(dropped)} dropped "
          f"({len([x for x in dropped if x in full])} full-length)")
    dsp = collections.Counter((rank(x, "genus"), rank(x, "species")) for x in dropped)
    print("dropped, by (genus, species):", dsp.most_common(12))
    for r in rows:
        print("\t".join(map(str, r)))


if __name__ == "__main__":
    main()
