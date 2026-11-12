#!/usr/bin/env python
"""Cut one mature peptide out of polyprotein records, and merge it into a protein group
(NOTES s155).

Flaviviruses (dengue, Zika, West Nile) and alphaviruses (chikungunya) are deposited
mostly as whole polyproteins, so a title query for the envelope protein finds only the
minority deposited on their own. This reads a polyprotein fetch (fetch_group.py
output), takes the `mat_peptide` feature whose product matches --product, and writes
that subsequence under the polyprotein's own accession, so its GenPept metadata (host,
country, collection date) still applies. With --merge-into, the cut peptides are added
to an existing group's raw.faa / records.gpc.xml (a record already present by
accession is not added twice).
"""
import argparse
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poly_dir")
    ap.add_argument("--product", required=True, help="regex over the mat_peptide product")
    ap.add_argument("--merge-into", required=True, help="group directory to add the peptides to")
    a = ap.parse_args()
    pat = re.compile(a.product, re.I)
    text = open(os.path.join(a.poly_dir, "records.gpc.xml"), encoding="utf-8").read()
    blocks = re.findall(r"<INSDSeq>.*?</INSDSeq>", text, flags=re.S)
    out = []
    for b in blocks:
        acc = re.search(r"<INSDSeq_accession-version>(.*?)</", b).group(1)
        seq = re.search(r"<INSDSeq_sequence>(.*?)</", b, flags=re.S)
        if not seq:
            continue
        seq = seq.group(1).strip().upper()
        for feat in re.findall(r"<INSDFeature>.*?</INSDFeature>", b, flags=re.S):
            if "<INSDFeature_key>mat_peptide</INSDFeature_key>" not in feat:
                continue
            prod = re.search(r"<INSDQualifier_name>product</INSDQualifier_name>\s*"
                             r"<INSDQualifier_value>(.*?)</", feat)
            iv = re.findall(r"<INSDInterval_from>(\d+)</.*?<INSDInterval_to>(\d+)</", feat, flags=re.S)
            if prod and pat.search(prod.group(1)) and len(iv) == 1:
                f, t = map(int, iv[0])
                out.append((acc, seq[f - 1:t], b))
                break
    g = a.merge_into
    have = set()
    if os.path.exists(os.path.join(g, "raw.faa")):
        have = {l[1:].split()[0] for l in open(os.path.join(g, "raw.faa")) if l.startswith(">")}
    add = [(acc, s, b) for acc, s, b in out if acc not in have]
    with open(os.path.join(g, "raw.faa"), "a") as f:
        for acc, s, _ in add:
            f.write(f">{acc} {a.product} cut from polyprotein\n{s}\n")
    gx = os.path.join(g, "records.gpc.xml")
    old = open(gx, encoding="utf-8").read() if os.path.exists(gx) else "<INSDSet></INSDSet>"
    old = old.rsplit("</INSDSet>", 1)[0]
    open(gx, "w", encoding="utf-8").write(old + "".join(b for _, _, b in add) + "</INSDSet>\n")
    print(f"{os.path.basename(a.poly_dir)}: {len(blocks)} polyproteins, {len(out)} with a "
          f"matching mat_peptide, {len(add)} added to {g}")


if __name__ == "__main__":
    main()
