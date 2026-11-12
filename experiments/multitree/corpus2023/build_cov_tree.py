#!/usr/bin/env python
"""One coronavirus spike backbone, from Entrez records to an antiGen examples file (NOTES s155).

Input: a group directory written by fetch_cov_spikes.sh (raw.faa, records.gpc.xml), all
records published before 2023-01-01.

  1. clean     standard length window around the reference (0.9-1.1x), <= 1% ambiguous
               residues, exact duplicates collapsed (the first accession kept, the count
               recorded); for the sarbecoviruses, anything within 1% of the SARS-CoV-2
               reference is dropped -- a SARS-CoV-2 spike filed under another name.
  2. backbone  only above --max-tips, round-robin over (country, year) strata so a burst
               of sequencing in one place and year cannot dominate the backbone.
  3. align     MAFFT --add --keeplength onto the reference: the REFERENCE'S coordinates,
               insertions relative to it dropped, deletions as '-'. For the sarbecoviruses
               the reference is the SARS-CoV-2 root of the 2023-01-01 build, so this tree
               is in the same 1,273-site frame as the SARS-CoV-2 corpus.
  4. tree      IQ-TREE 3, LG+G4, -fast, --keep-ident, with marginal ancestral states;
               midpoint-rooted (no outgroup is close enough to align without distorting
               the ingroup alignment).
  5. write     a Nextstrain v2 tree JSON (branch amino-acid mutations under gene S,
               num_date where a collection date exists) and a root-sequence sidecar, the
               two files the antigen binary reads.
  6. attach   EVERY cleaned record (duplicates included) is attached to the backbone
               by `antigen prepare` + `place`, the antiGen-data-revised arm recipe
               (run_arms.sbatch), which then writes the examples (one allele drawn per
               node and site, one mutation per edge). The backbone subsample (step 2,
               applied to the distinct sequences) only makes the tree faster and more
               reliable to infer; no record is discarded by it.

Gaps: a tip's deletion is written as a mutation to '-', which the 20-letter alphabet
reads as missing. An internal node is '-' wherever none of its descendant tips has a
residue, so ancestral reconstruction cannot invent a mutation in a column that no
descendant observed.
"""
import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from Bio import Phylo, SeqIO

STD = set("ACDEFGHIKLMNPQRSTVWY")
IQTREE = "/home/ivan.specht/miniconda/envs/tpcmp/bin/iqtree3"
MAFFT = "/home/ivan.specht/miniconda/envs/usher/bin/mafft"
ANTIGEN = "/home/ivan.specht/antiGen-data-revised/work/antigen_arms"   # the general-length build (run_arms.sbatch)
CHECK = "/home/ivan.specht/antiGen-data-revised/check_examples.py"
SC2_ROOT = ("/home/ivan.specht/antiGen-data-revised/data/trees/gisaid/"
            "ncov_gisaid_global_all-time@2022-12-30_root-sequence.json")


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def parse_meta(xml_path):
    # EDirect writes one XML document per download chunk, concatenated; keep only the
    # INSDSeq elements and parse them under a single root.
    import io
    text = open(xml_path, encoding="utf-8").read()
    seqs = re.findall(r"<INSDSeq>.*?</INSDSeq>", text, flags=re.S)
    meta = {}
    for _, el in ET.iterparse(io.StringIO("<INSDSet>" + "".join(seqs) + "</INSDSet>")):
        if el.tag != "INSDSeq":
            continue
        acc = el.findtext("INSDSeq_accession-version")
        q = {}
        for qual in el.iter("INSDQualifier"):
            n, v = qual.findtext("INSDQualifier_name"), qual.findtext("INSDQualifier_value")
            if n and v and n not in q:
                q[n] = v
        meta[acc] = dict(organism=el.findtext("INSDSeq_organism") or "",
                         host=q.get("host", ""),
                         country=(q.get("geo_loc_name") or q.get("country") or "").split(":")[0],
                         date=q.get("collection_date", ""),
                         created=el.findtext("INSDSeq_create-date") or "")
        el.clear()
    return meta


MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def year_of(date):
    m = re.search(r"(19|20)\d\d", date or "")
    return int(m.group(0)) if m else None


def num_date(date):
    """Decimal year from GenBank's collection_date forms, or None."""
    d = (date or "").strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", d)
    if m:
        y, mo, da = map(int, m.groups())
        return y + (mo - 1 + (da - 0.5) / 31) / 12
    m = re.fullmatch(r"(\d{2})-(\w{3})-(\d{4})", d)
    if m and m.group(2) in MONTHS:
        return int(m.group(3)) + (MONTHS[m.group(2)] - 1 + (int(m.group(1)) - 0.5) / 31) / 12
    m = re.fullmatch(r"(\d{4})-(\d{2})", d) or re.fullmatch(r"(\w{3})-(\d{4})", d)
    if m:
        a, b = m.groups()
        y, mo = (int(a), int(b)) if a.isdigit() else (int(b), MONTHS.get(a, 6))
        return y + (mo - 0.5) / 12
    return None


def identity(a, b):
    n = same = 0
    for x, y in zip(a, b):
        if x in STD and y in STD:
            n += 1
            same += x == y
    return same / n if n else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("group_dir")
    ap.add_argument("--reference", default=None,
                    help="accession of the frame reference; 'sc2' for the SARS-CoV-2 root; "
                         "default: a RefSeq record if present, else the modal-length one")
    ap.add_argument("--max-tips", type=int, default=1500)
    ap.add_argument("--min-frag", type=int, default=300,
                    help="partial records at least this long are attached (not in the backbone)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    d = a.group_dir
    name = os.path.basename(d.rstrip("/"))
    meta = parse_meta(os.path.join(d, "records.gpc.xml"))
    allrecs = list(SeqIO.parse(os.path.join(d, "raw.faa"), "fasta"))
    report = collections.OrderedDict(group=name, fetched=len(allrecs))
    # NOT NATURAL SEQUENCES: PDB chains (engineered spikes -- proline-stabilised,
    # furin-site knockouts), patent records and synthetic constructs.
    def unnatural(r):
        desc = r.description.lower()
        return (r.id.startswith("pdb|") or "patent" in desc or "synthetic" in desc
                or "synthetic construct" in meta.get(r.id, {}).get("organism", "").lower())
    recs = [(r.id, str(r.seq).upper().rstrip("*")) for r in allrecs if not unnatural(r)]
    report["not_natural"] = len(allrecs) - len(recs)

    # ── reference ──
    if a.reference == "sc2":
        ref_id, ref = "SARS-CoV-2_root", json.load(open(SC2_ROOT))["S"].rstrip("*")
    else:
        by = dict(recs)
        if a.reference:
            ref_id = a.reference
        else:
            rs = [i for i, _ in recs if i.startswith(("YP_", "NP_"))]
            if rs:
                ref_id = rs[0]
            else:
                modal = collections.Counter(len(s) for _, s in recs).most_common(1)[0][0]
                ref_id = next(i for i, s in recs if len(s) == modal)
        ref = by[ref_id]
    report["reference"] = ref_id
    report["ref_len"] = len(ref)

    # ── clean ──
    # `pool` is every record that passes, duplicates included: all of it is attached
    # below. `uniq` is one representative per distinct sequence: the backbone is drawn
    # from it, and only it is aligned (a duplicate takes its representative's row).
    lo, hi = 0.9 * len(ref), 1.1 * len(ref)
    pool, uniq, rep = [], [], {}
    frags, frep = [], {}           # partial spikes: attached, never in the backbone
    drop = collections.Counter()
    for i, s in recs:
        if a.min_frag <= len(s) < lo and sum(c not in STD for c in s) <= 0.01 * len(s):
            frags.append((i, s))
            frep.setdefault(s, i)
            continue
        if not lo <= len(s) <= hi:
            drop["length"] += 1
            continue
        if sum(c not in STD for c in s) > 0.01 * len(s):
            drop["ambiguous"] += 1
            continue
        pool.append((i, s))
        if s not in rep:
            rep[s] = i
            uniq.append((i, s))
    report["dropped"] = dict(drop)
    report["cleaned_records"] = len(pool)
    report["distinct_sequences"] = len(uniq)

    # ── align the distinct sequences onto the reference frame ──
    safe = {i: re.sub(r"[^A-Za-z0-9_]", "_", i) for i, _ in pool + frags}
    report["partial_records"] = len(frags)
    back = {v: k for k, v in safe.items()}
    with open(os.path.join(d, "ref.faa"), "w") as f:
        f.write(f">REF\n{ref}\n")
    with open(os.path.join(d, "seqs.faa"), "w") as f:
        for i, s in uniq:
            f.write(f">{safe[i]}\n{s}\n")
    # --addfragments scales to many sequences; --6merpair because this build's
    # pairlocalalign segfaults on it.
    mode = ["--add"] if len(uniq) <= 5000 else ["--6merpair", "--addfragments"]
    with open(os.path.join(d, "aln.faa"), "w") as out:
        run([MAFFT, "--thread", str(a.threads), "--quiet", "--keeplength", "--anysymbol",
             *mode, os.path.join(d, "seqs.faa"), os.path.join(d, "ref.faa")], stdout=out)
    aln_u = {r.id: str(r.seq).upper() for r in SeqIO.parse(os.path.join(d, "aln.faa"), "fasta")}
    aln_u.pop("REF")
    if a.reference == "sc2":
        close = {k for k, s in aln_u.items() if identity(s, ref) >= 0.99}
        for k in close:
            aln_u.pop(k)
        report["dropped"]["within_1pct_of_SARS-CoV-2"] = len(close)
    # every record's aligned row, through its representative
    aln_all = {}
    for i, s in pool:
        r = safe[rep[s]]
        if r in aln_u:
            aln_all[safe[i]] = aln_u[r]
    if frags:
        with open(os.path.join(d, "frags.faa"), "w") as f:
            for sq, i in frep.items():
                f.write(f">{safe[i]}\n{sq}\n")
        with open(os.path.join(d, "aln_frags.faa"), "w") as out:
            run([MAFFT, "--thread", str(a.threads), "--quiet", "--keeplength", "--anysymbol",
                 "--6merpair", "--addfragments", os.path.join(d, "frags.faa"),
                 os.path.join(d, "ref.faa")],   # --6merpair: localpair segfaults here
                stdout=out)
        aln_f = {r.id: str(r.seq).upper() for r in SeqIO.parse(os.path.join(d, "aln_frags.faa"), "fasta")}
        aln_f.pop("REF")
        nf = 0
        for i, sq in frags:
            row = aln_f.get(safe[frep[sq]])
            if row is None or (a.reference == "sc2" and identity(row, ref) >= 0.99):
                continue
            aln_all[safe[i]] = row
            nf += 1
        report["partial_attached"] = nf
    report["attached_records"] = len(aln_all)

    # ── the backbone: a stratified subsample of the distinct sequences ──
    rng = random.Random(a.seed)
    cand = [(back[k], s) for k, s in aln_u.items()]
    if len(cand) > a.max_tips:
        strata = collections.defaultdict(list)
        for i, s in cand:
            m = meta.get(i, {})
            strata[(m.get("country", ""), year_of(m.get("date")))].append((i, s))
        for v in strata.values():
            rng.shuffle(v)
        order, keys = [], sorted(strata, key=str)
        while len(order) < a.max_tips:
            for k in keys:
                if strata[k] and len(order) < a.max_tips:
                    order.append(strata[k].pop())
        report["backbone_subsampled_from"] = len(cand)
        cand = order
    aln = {safe[i]: s for i, s in cand}
    tips_path = os.path.join(d, "aln_tips.faa")
    new = "".join(f">{k}\n{s}\n" for k, s in aln.items())
    unchanged = os.path.exists(tips_path) and open(tips_path).read() == new
    open(tips_path, "w").write(new)
    report["tips"] = len(aln)

    # ── tree + ancestral states ──
    pre = os.path.join(d, "iq")
    # Reused when a finished run covers exactly these tips (the tree is the slow step).
    def have_tree():
        if not (unchanged and os.path.exists(pre + ".state") and os.path.exists(pre + ".treefile")):
            return False
        tips = {c.name for c in Phylo.read(pre + ".treefile", "newick").get_terminals()}
        return tips == set(aln)
    if have_tree():
        print("  reusing", pre + ".treefile", flush=True)
    else:
        run([IQTREE, "-s", os.path.join(d, "aln_tips.faa"), "-m", "LG+G4", "-fast",
             "--keep-ident", "-asr", "-T", str(a.threads), "--prefix", pre, "-redo",
             "--quiet", "--seed", str(a.seed)])
    states = collections.defaultdict(dict)
    with open(pre + ".state") as f:
        for line in f:
            if line.startswith("#") or line.startswith("Node\tSite"):
                continue
            node, site, st = line.split("\t")[:3]
            states[node][int(site) - 1] = st
    L = len(ref)
    anc = {n: "".join(v.get(j, "-") for j in range(L)) for n, v in states.items()}
    tree = Phylo.read(pre + ".treefile", "newick")
    tree.root_at_midpoint()

    # observed mask, bottom-up: a node 'observes' a column if any descendant tip has a residue
    seq = {}
    def fill(cl):
        if cl.is_terminal():
            s = aln[cl.name]
            seq[id(cl)] = s
            return [c in STD for c in s]
        masks = [fill(ch) for ch in cl.clades]
        obs = [any(col) for col in zip(*masks)]
        src = anc.get(cl.name) if cl.name else None
        if src is None:   # the midpoint root has no state: take its nearer child's
            src = seq[id(min(cl.clades, key=lambda c: c.branch_length or 0))]
        seq[id(cl)] = "".join(ch if o else "-" for ch, o in zip(src, obs))
        return obs
    fill(tree.root)

    n_mut = 0
    def to_json(cl, parent_seq, k=[0]):
        s = seq[id(cl)]
        muts = [] if parent_seq is None else [
            f"{p}{j + 1}{c}" for j, (p, c) in enumerate(zip(parent_seq, s)) if p != c]
        nonlocal_count[0] += sum(1 for m in muts if m[0] in STD and m[-1] in STD)
        k[0] += 1
        node = {"name": back.get(cl.name, cl.name) if cl.is_terminal() else f"NODE_{k[0]:06d}",
                "branch_attrs": {"mutations": {"S": muts}} if muts else {}, "node_attrs": {}}
        if cl.is_terminal():
            nd = num_date(meta.get(back.get(cl.name, ""), {}).get("date"))
            if nd is not None:
                node["node_attrs"]["num_date"] = {"value": nd}
        else:
            node["children"] = [to_json(ch, s) for ch in cl.clades]
        return node
    nonlocal_count = [0]
    root_seq = seq[id(tree.root)]
    tj = {"version": "v2", "meta": {"title": f"{name} spike, records published before 2023-01-01",
                                    "updated": "2022-12-31"},
          "tree": to_json(tree.root, None)}
    json.dump(tj, open(os.path.join(d, "tree.json"), "w"))
    json.dump({"S": root_seq + "*"}, open(os.path.join(d, "root.json"), "w"))
    report["residue_mutations_on_tree"] = nonlocal_count[0]

    # ── attach EVERY record, then write the examples: the antiGen-data-revised arm
    #    recipe (run_arms.sbatch's train side), prepare + place, same binary.
    # Dates: the collection date where GenBank has one at day or month resolution,
    # else the record's creation date -- an upper bound on collection, and before the
    # cutoff by construction (PDAT filter). --no-clock, because a creation date is not
    # a collection date and the clock envelope would reject it for that alone.
    def iso(acc):
        m = meta.get(acc, {})
        cd = (m.get("date") or "").strip().title()
        for pat, f in ((r"(\d{4})-(\d{2})-(\d{2})", lambda g: f"{g[0]}-{g[1]}-{g[2]}"),
                       (r"(\d{2})-(\w{3})-(\d{4})",
                        lambda g: f"{g[2]}-{MONTHS[g[1]]:02d}-{g[0]}" if g[1] in MONTHS else None),
                       (r"(\d{4})-(\d{2})", lambda g: f"{g[0]}-{g[1]}"),
                       (r"(\w{3})-(\d{4})",
                        lambda g: f"{g[1]}-{MONTHS[g[0]]:02d}" if g[0] in MONTHS else None)):
            mm = re.fullmatch(pat, cd)
            if mm and f(mm.groups()):
                return f(mm.groups()), "collection"
        c = re.fullmatch(r"(\d{2})-(\w{3})-(\d{4})", m.get("created", "").title())
        if c and c.group(2) in MONTHS:
            return f"{c.group(3)}-{MONTHS[c.group(2)]:02d}-{c.group(1)}", "creation"
        return None, None
    src = collections.Counter()
    with open(os.path.join(d, "all_aln.faa"), "w") as fa, open(os.path.join(d, "meta.tsv"), "w") as mt:
        mt.write("strain\tdate\n")
        for k, srow in aln_all.items():
            dt, how = iso(back[k])
            src[how] += 1
            if dt is None:
                continue
            fa.write(f">{k}\n{srow}\n")
            mt.write(f"{k}\t{dt}\n")
    report["attach_dates"] = dict(src)
    P = lambda f: os.path.join(d, f)
    with open(P("place.log"), "w") as log:
        run([ANTIGEN, "prepare", "--gene", "S", "--root-seq", P("root.json"),
             "--fasta", P("all_aln.faa"), "--metadata", P("meta.tsv"),
             "--date-cutoff", "2023-01-01", "--date-impute", "month", "--no-clock",
             "--out", P("prepared.tsv")], stdout=log, stderr=log)
        run([ANTIGEN, "place", "--gene", "S", "--tree", P("tree.json"), "--root-seq", P("root.json"),
             "--prepared", P("prepared.tsv"), "--seed", "1", "--threads", str(a.threads),
             "--out-tree", P("tree_placed.tsv"), "--out", P("sets.tsv"),
             "--out-examples", P(f"examples_{name}.txt"), "--out-counts", P("counts.tsv")],
            stdout=log, stderr=log)
    for f in ("sets.tsv", "prepared.tsv"):
        os.remove(P(f))
    pl = open(P("place.log")).read()
    report["placed"] = [l.strip() for l in pl.splitlines()
                        if l.startswith(("placed", "examples", "prepared", "kept", "selected"))][:6]
    chk = subprocess.run([sys.executable, "-I", CHECK, os.path.join(d, f"examples_{name}.txt")],
                         capture_output=True, text=True)
    open(os.path.join(d, "check.txt"), "w").write(chk.stdout + chk.stderr)
    report["check"] = chk.stdout.strip().splitlines()[:3]
    hosts = collections.Counter(meta.get(back.get(k, ""), {}).get("host", "") or "?" for k in aln)
    orgs = collections.Counter(meta.get(back.get(k, ""), {}).get("organism", "") or "?" for k in aln)
    yrs = [year_of(meta.get(back.get(k, ""), {}).get("date")) for k in aln]
    yrs = [y for y in yrs if y]
    report["hosts_top"] = hosts.most_common(6)
    report["organisms_top"] = orgs.most_common(6)
    report["collection_years"] = [min(yrs), max(yrs)] if yrs else None
    json.dump(report, open(os.path.join(d, "report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
