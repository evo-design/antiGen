#!/usr/bin/env python
"""Fetch one protein group from Entrez, PUBLISHED BEFORE 2023-01-01 (NOTES s155).

A record published before the cutoff cannot carry anything sampled after it. EVERY record is
downloaded by default: build_cov_tree.py subsamples only the BACKBONE and then attaches
every cleaned record to it with the antiGen-data-revised placer, so nothing is
discarded (NOTES s155). --cap, if set, draws uniformly within creation years.

Writes <out>/<name>/{query.txt, raw.faa, records.gpc.xml, ids.txt}.

    python fetch_group.py --name h3n2_ha --query '...' --slen 540:580 --out DIR
"""
import argparse
import collections
import os
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PDAT = "1900/01/01:2022/12/31[PDAT]"


def call(tool, retries=8, **params):
    data = urllib.parse.urlencode(dict(params, tool="antigen_corpus")).encode()
    for k in range(retries):
        try:
            time.sleep(0.4)                       # <= 3 requests/s without an API key
            with urllib.request.urlopen(EU + tool, data=data, timeout=180) as r:
                return r.read()
        except Exception as exc:                   # 429s and dropped TLS: back off
            wait = 2 * (k + 1)
            print(f"  {tool} retry {k + 1} in {wait}s: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{tool} failed after {retries} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--query", required=True, help="organism/title part; date and length are added")
    ap.add_argument("--slen", required=True, help="e.g. 540:580")
    ap.add_argument("--cap", type=int, default=0, help="0 = every record (the default)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    d = os.path.join(a.out, a.name)
    os.makedirs(d, exist_ok=True)
    term = (f"({a.query}) AND {a.slen}[SLEN] AND {PDAT} "
            f"NOT srcdb_pdb[PROP] NOT srcdb_pat[PROP]")
    open(os.path.join(d, "query.txt"), "w").write(term + "\n")
    r = ET.fromstring(call("esearch.fcgi", db="protein", term=term, retmax=0, usehistory="y"))
    n, web, key = int(r.findtext("Count")), r.findtext("WebEnv"), r.findtext("QueryKey")
    print(f"{a.name}: {n} hits", flush=True)
    # every uid + creation date, from the summaries
    uids = []
    for start in range(0, n, 5000):
        s = ET.fromstring(call("esummary.fcgi", db="protein", WebEnv=web, query_key=key,
                               retstart=start, retmax=5000))
        for doc in s.iter("DocSum"):
            uid = doc.findtext("Id")
            cd = next((i.text for i in doc.iter("Item") if i.get("Name") == "CreateDate"), "") or ""
            uids.append((uid, cd[:4]))
    if len(uids) != n:
        raise SystemExit(f"{a.name}: {len(uids)} summaries for {n} hits")
    rng = random.Random(a.seed)
    if a.cap and n > a.cap:
        by = collections.defaultdict(list)
        for u, y in uids:
            by[y].append(u)
        for v in by.values():
            rng.shuffle(v)
        pick, years = [], sorted(by)
        while len(pick) < a.cap:
            for y in years:
                if by[y] and len(pick) < a.cap:
                    pick.append(by[y].pop())
        print(f"  subsampled {len(pick)} of {n} across {len(years)} creation years", flush=True)
    else:
        pick = [u for u, _ in uids]
    open(os.path.join(d, "ids.txt"), "w").write("\n".join(pick) + "\n")
    with open(os.path.join(d, "raw.faa"), "wb") as fa, open(os.path.join(d, "records.gpc.xml"), "wb") as gx:
        gx.write(b"<INSDSet>\n")
        for i in range(0, len(pick), 500):
            ids = ",".join(pick[i:i + 500])
            fa.write(call("efetch.fcgi", db="protein", id=ids, rettype="fasta", retmode="text"))
            x = call("efetch.fcgi", db="protein", id=ids, rettype="gpc", retmode="xml")
            body = x.split(b"<INSDSet>", 1)[-1].rsplit(b"</INSDSet>", 1)[0]
            gx.write(body)
        gx.write(b"</INSDSet>\n")
    nf = open(os.path.join(d, "raw.faa")).read().count(">")
    print(f"{a.name}: wrote {nf} sequences", flush=True)
    if nf != len(pick):
        raise SystemExit(f"{a.name}: {nf} sequences for {len(pick)} ids")


if __name__ == "__main__":
    main()
