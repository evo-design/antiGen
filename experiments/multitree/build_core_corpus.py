#!/usr/bin/env python
"""Build the two no-attach training corpora from Ivan's core Nextstrain snapshot,
all cut at the forecast date D = 2025-01-01 so nothing can leak into the fc13 eval.

Source: /home/ivan.specht/antiGen-multitree/data/core/<slug>/ — one dir per dataset,
each with tree.json.gz + root-sequence.json.gz. Every tree predates 2026-05-01, i.e.
they are 2026-04 snapshots and DO contain post-D evolution, so we never generate from
them directly for training. Instead, per (virus, protein) panel:

  * SNAPSHOT route (preferred, strictly leak-free): fetch the build as published at D
    (Charon ``<path>@2025-01-01``) — exactly how the fc13 _train.txt were made — and
    generate examples from it. The snapshot IS the cutoff: no post-D sequence, topology
    or ancestral reconstruction exists in it.
  * PRUNE route (fallback, only where D is not retained): drop every node in Ivan's
    2026-04 tree with num_date > D, then generate. Targets stay pre-D events, but the
    ancestral reconstruction saw post-D tips — a mild query-side leak, flagged per panel.

Selection is snapshot-aware: among the trees carrying a panel's genes, prefer one with a
retained D snapshot, then the most AA mutation events (diversity / example-count proxy,
NOT tip count — dense recent windows are near-clonal). One tree per (virus, protein).

Two corpora are written, differing ONLY in how many proteins per virus are kept:
  --mode single   one canonical antigenic protein per virus  (matches the fc13 corpus)
  --mode surface  every surface/antigenic protein a virus offers (more diversity+volume)

The 13 eval (virus, protein) pairs are EXCLUDED here (EVAL_REUSE): training reuses their
existing leak-free _train.txt. ebola is excluded entirely (held-out de-novo test virus).

Usage:
  PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
  $PY experiments/multitree/build_core_corpus.py --mode single --probe-cache <probe.json>
  $PY experiments/multitree/build_core_corpus.py --mode surface --probe-cache <probe.json>
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_forecast_testset import decimal_year, node_date  # noqa: E402

CORE = "/home/ivan.specht/antiGen-multitree/data/core"
DATASET_DIR = CORE  # dataset.json per slug gives the Charon path
CHARON = "https://nextstrain.org/charon/getDataset?prefix=/"
DATE = "2025-01-01"
CUTOFF = decimal_year(DATE)
SNAP_CACHE = os.path.join(HERE, "data", "core_snapshots")  # fetched @D builds
MIN_EXAMPLES = 50

# The 13 eval (virus, protein) pairs — excluded here, reused via their own _train.txt.
# hmpv:g is excluded too (it overlaps the hmpv_a G eval target on a superset tree).
EVAL_REUSE = {
    ("dengue_denv1", "e"), ("dengue_denv2", "e"), ("dengue_denv3", "e"),
    ("dengue_denv4", "e"), ("seasonal-flu_h1n1pdm", "ha"),
    ("seasonal-flu_h3n2", "ha"), ("seasonal-flu_vic", "ha"),
    ("avian-flu_h5nx", "ha"), ("measles", "h"), ("ncov", "s"),
    ("rsv_a", "f"), ("rsv_b", "f"), ("hmpv", "g"),
}

# ── Panel table ────────────────────────────────────────────────────────────────
# Per virus: panels keyed by a short label -> the tree gene list that forms that
# panel (concatenated in this order, exactly like the fc13 corpus does for HA).
# "canonical" names the single panel kept in --mode single. Everything here is kept
# in --mode surface. Genes are the *tree* gene names (naming varies by build).
#
# Edit this table to change what trains; nothing else below encodes biology.
VIRUS_PANELS = {
    # Influenza — HA (canonical antigen) + NA. Avian HA is one "HA" gene; seasonal
    # HA is split SigPep+HA1+HA2 (matches the fc13 corpus seq_len 566/585).
    "avian-flu_h5n1":      {"canonical": "ha", "panels": {"ha": ["HA"], "na": ["NA"]}},
    "avian-flu_h5nx":      {"canonical": "ha", "panels": {"ha": ["HA"], "na": ["NA"]}},
    "avian-flu_h7n9":      {"canonical": "ha", "panels": {"ha": ["HA"], "na": ["NA"]}},
    "avian-flu_h7nx":      {"canonical": "ha", "panels": {"ha": ["HA"], "na": ["NA"]}},
    "avian-flu_h9n2":      {"canonical": "ha", "panels": {"ha": ["HA"], "na": ["NA"]}},
    "seasonal-flu_h3n2":   {"canonical": "ha", "panels": {"ha": ["SigPep", "HA1", "HA2"], "na": ["NA"]}},
    "seasonal-flu_h1n1pdm":{"canonical": "ha", "panels": {"ha": ["SigPep", "HA1", "HA2"], "na": ["NA"]}},
    "seasonal-flu_vic":    {"canonical": "ha", "panels": {"ha": ["SigPep", "HA1", "HA2"], "na": ["NA"]}},
    "seasonal-flu_yam":    {"canonical": "ha", "panels": {"ha": ["SigPep", "HA1", "HA2"], "na": ["NA"]}},
    # Coronaviruses — spike only (E/M are minor, excluded to avoid non-antigen noise).
    "ncov":                {"canonical": "s", "panels": {"s": ["S"]}},
    "seasonal-cov_229e":   {"canonical": "spike", "panels": {"spike": ["Spike"]}},
    "seasonal-cov_hku1":   {"canonical": "spike", "panels": {"spike": ["Spike"]}},
    "seasonal-cov_nl63":   {"canonical": "spike", "panels": {"spike": ["Spike"]}},
    "seasonal-cov_oc43":   {"canonical": "spike", "panels": {"spike": ["Spike"]}},
    # Flaviviruses — envelope (canonical) + prM where the build exposes it.
    "dengue_denv1":        {"canonical": "e", "panels": {"e": ["E"], "prm": ["pr"]}},
    "dengue_denv2":        {"canonical": "e", "panels": {"e": ["E"], "prm": ["pr"]}},
    "dengue_denv3":        {"canonical": "e", "panels": {"e": ["E"], "prm": ["pr"]}},
    "dengue_denv4":        {"canonical": "e", "panels": {"e": ["E"], "prm": ["pr"]}},
    "WNV":                 {"canonical": "env", "panels": {"env": ["env"], "prm": ["prM"]}},
    "zika":                {"canonical": "env", "panels": {"env": ["ENV"], "prm": ["PRO"]}},
    "yellow-fever":        {"canonical": "e", "panels": {"e": ["E"], "prm": ["prM"]}},
    # Alphaviruses — E1 + E2 (both neutralizing antigens; E2 canonical).
    "chikv":               {"canonical": "e2", "panels": {"e1": ["E1"], "e2": ["E2"]}},
    "rubella":             {"canonical": "e1", "panels": {"e1": ["E1"], "e2": ["E2"]}},
    # Paramyxo/pneumoviruses — fusion/attachment glycoproteins (+ SH).
    "measles":             {"canonical": "h", "panels": {"h": ["H"], "f": ["F"]}},
    "mumps":               {"canonical": "hn", "panels": {"hn": ["HN"], "f": ["F"], "sh": ["SH"]}},
    "hmpv":                {"canonical": "f", "panels": {"f": ["F"], "g": ["G"], "sh": ["SH"]}},
    "rsv_a":               {"canonical": "f", "panels": {"f": ["F"], "g": ["G"], "sh": ["SH"]}},
    "rsv_b":               {"canonical": "f", "panels": {"f": ["F"], "g": ["G"], "sh": ["SH"]}},
    "nipah":               {"canonical": "g", "panels": {"g": ["G"], "f": ["F"]}},
    # Single-glycoprotein viruses.
    "lassa":               {"canonical": "gpc", "panels": {"gpc": ["GPC"]}},
    "hantavirus":          {"canonical": "gpc", "panels": {"gpc": ["GPC"]}},
    "oropouche":           {"canonical": "gc", "panels": {"gc": ["OROVsMgp1"]}},
    # NOTE: ebola is deliberately OMITTED. ebola-bdbv is the de-novo held-out TEST
    # virus (data/new_heldout_virus/); the mutation-event selector picks ebola_bdbv as
    # the top ebola GP build, so including ebola here would leak the held-out virus
    # into training. It is absent from the fc13 corpus for the same reason.
    "rabies":              {"canonical": "g", "panels": {"g": ["RABVgp4"]}},
    "mpox":                {"canonical": "opg210", "panels": {"opg210": ["OPG210"]}},
    # Non-enveloped — capsid proteins (VP1 canonical).
    "enterovirus":         {"canonical": "vp1", "panels": {"vp1": ["VP1"], "vp2": ["VP2"], "vp3": ["VP3"]}},
    "norovirus":           {"canonical": "vp1", "panels": {"vp1": ["VP1"], "vp2": ["VP2"]}},
}

# References carrying a trailing stop get it stripped; --ambiguous-ref gap maps stray
# X/ambiguous residues to '-'. Both are safe to pass always (they no-op otherwise) and
# make the bulk run robust.


def virus_key_of(slug):
    """Map a core slug to its virus-panel key (keeps subtype/serotype/subgroup)."""
    m = re.match(r"avian-flu_([a-z0-9]+)", slug)
    if m:
        return f"avian-flu_{m.group(1)}"
    m = re.match(r"seasonal-flu_([a-z0-9]+)", slug)
    if m:
        return f"seasonal-flu_{m.group(1)}"
    m = re.match(r"seasonal-cov_([a-z0-9]+)", slug)
    if m:
        return f"seasonal-cov_{m.group(1)}"
    m = re.match(r"dengue_(denv[1-4])", slug)
    if m:
        return f"dengue_{m.group(1)}"
    m = re.match(r"(rsv_[ab])", slug)
    if m:
        return m.group(1)
    return slug.split("_")[0]


def scan_core():
    """Return {slug: (gene_muts, n_tips, tree_path, ref_path)} for every core tree,
    where gene_muts[gene] is the number of amino-acid mutation events on branches for
    that gene. We select trees by mutation events in the panel's genes, not by tip
    count: dense recent windows (2y flu, 6m ncov) carry many near-identical tips but
    little evolutionary signal (near-clonal, low ESS), so tip count would pick the
    least diverse build. Mutation-event count tracks both diversity and the number of
    training examples the generator will emit."""
    out = {}
    for name in sorted(os.listdir(CORE)):
        d = os.path.join(CORE, name)
        tp = os.path.join(d, "tree.json.gz")
        if not os.path.isfile(tp):
            continue
        with gzip.open(tp) as f:
            tree = json.load(f)
        gene_muts, stack = {}, [tree["tree"]]
        while stack:
            n = stack.pop()
            for k, muts in n.get("branch_attrs", {}).get("mutations", {}).items():
                if k != "nuc":
                    gene_muts[k] = gene_muts.get(k, 0) + len(muts)
            stack.extend(n.get("children", []))
        ntips = 0
        ds = os.path.join(d, "dataset.json")
        if os.path.isfile(ds):
            try:
                ntips = json.load(open(ds)).get("n_tips", 0) or 0
            except Exception:
                pass
        ref = os.path.join(d, "root-sequence.json.gz")
        out[name] = (gene_muts, ntips, tp, ref if os.path.isfile(ref) else None)
    return out


def charon_path(slug):
    """Nextstrain dataset path for a core slug, from its dataset.json."""
    dj = os.path.join(DATASET_DIR, slug, "dataset.json")
    if os.path.isfile(dj):
        try:
            return json.load(open(dj)).get("path")
        except Exception:
            return None
    return None


def load_verdicts(cache_path):
    """{nextstrain_path: verdict} from a probe cache ('snapshot' means D is retained)."""
    if not cache_path or not os.path.isfile(cache_path):
        return {}
    return {r["path"]: r["verdict"] for r in json.load(open(cache_path))}


def _request(url, timeout=600, attempts=3):
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "antigen-build-core/1",
                              "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except OSError as exc:
            last = exc
        time.sleep(3 * (attempt + 1))
    raise last


def _write_atomic(dst, data):
    """Write bytes to dst via a temp file + rename, so a failed/partial fetch never
    leaves a 0-byte file that would poison the cache on a later run."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def fetch_snapshot(path, local_ref=None):
    """Fetch the @D build (+ root-sequence) for a Nextstrain path into SNAP_CACHE.
    Returns (tree_json_file, ref_path) or None if the tree is unavailable. ref_path is,
    in order of preference: the dated sidecar (if served), None (snapshot embeds its own
    root_sequence, which make_tree_examples then uses), or the frame-identical local
    sidecar from Ivan's build as a last resort."""
    os.makedirs(SNAP_CACHE, exist_ok=True)
    safe = path.strip("/").replace("/", "_")
    tree_file = os.path.join(SNAP_CACHE, f"{safe}@{DATE}.json")
    ref_file = os.path.join(SNAP_CACHE, f"{safe}@{DATE}_root-sequence.json")
    url = f"{CHARON}{path}@{DATE}"
    if not (os.path.isfile(tree_file) and os.path.getsize(tree_file) > 0):
        try:
            payload = _request(url)
            json.loads(payload)  # validate
        except Exception:
            return None
        _write_atomic(tree_file, payload)
    # dated root-sequence sidecar, only kept if it is real non-empty JSON
    if not (os.path.isfile(ref_file) and os.path.getsize(ref_file) > 0):
        try:
            data = _request(f"{url}&type=root-sequence")
            json.loads(data)
            _write_atomic(ref_file, data)
        except Exception:
            pass
    if os.path.isfile(ref_file) and os.path.getsize(ref_file) > 0:
        return tree_file, ref_file
    # no dated sidecar: use the snapshot's embedded root_sequence if present, else the
    # local frame-identical sidecar (Ivan's build, same dataset path).
    try:
        with open(tree_file) as fh:
            embeds = "root_sequence" in json.load(fh)
    except Exception:
        embeds = False
    return tree_file, (None if embeds else local_ref)


def prune_tree_file(src_gz, out_json):
    """Write a copy of a core tree with every node dated after D (and its subtree)
    removed. Nodes without a date are kept. The ancestral reconstruction is unchanged,
    so this is the mildly hindsight-informed fallback used only when no @D snapshot
    exists. Returns the local root-sequence sidecar to use as --ref."""
    with gzip.open(src_gz) as f:
        tree = json.load(f)
    stack = [tree["tree"]]
    while stack:
        n = stack.pop()
        kept = [c for c in n.get("children", [])
                if not (node_date(c) is not None and node_date(c) > CUTOFF)]
        n["children"] = kept
        stack.extend(kept)
    with open(out_json, "w") as fh:
        json.dump(tree, fh)
    ref = os.path.join(os.path.dirname(src_gz), "root-sequence.json.gz")
    return ref if os.path.isfile(ref) else None


def run_generator(tree_path, ref_path, genes, out_path):
    """make_tree_examples on a (snapshot or pruned) tree; returns (seq_len, examples)
    or (None, err). ref_path=None lets the generator use the tree's embedded root_sequence."""
    cmd = [sys.executable, os.path.join(HERE, "make_tree_examples.py"),
           "--tree", tree_path, "--genes", *genes,
           "--strip-trailing-stop", "--ambiguous-ref", "gap", "--drop-gap-stop",
           "--out", out_path]
    if ref_path:
        cmd += ["--ref", ref_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None, (res.stderr.strip().splitlines() or ["?"])[-1][:80]
    seq_len = examples = None
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("genes="):
            seq_len = int(line.split("seq_len=")[1])
        if line.startswith("node examples="):
            examples = int(line.split("total=")[1])
    return seq_len, examples


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single", "surface"], required=True)
    ap.add_argument("--out", default=None, help="example dir (default data/core_<mode>)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--probe-cache", default=None,
                    help="JSON from the @D availability probe; drives snapshot-vs-prune")
    ap.add_argument("--dry-run", action="store_true", help="select panels, don't generate")
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(HERE, "data", f"core_{args.mode}")
    manifest_path = args.manifest or os.path.join(HERE, "data", f"trees_core_{args.mode}.json")
    os.makedirs(out_dir, exist_ok=True)

    verdicts = load_verdicts(args.probe_cache)
    core = scan_core()
    by_virus = {}
    for slug, (gene_muts, ntips, tp, ref) in core.items():
        by_virus.setdefault(virus_key_of(slug), []).append((slug, gene_muts, ntips, tp, ref))

    # Select one tree per (virus, panel): prefer a candidate with a retained @D snapshot,
    # then the most AA mutation events (diversity / example proxy). Snapshot -> clean
    # cutoff; otherwise prune Ivan's tree at D (fallback). EVAL_REUSE pairs are skipped.
    selections = []  # dicts
    for virus, spec in VIRUS_PANELS.items():
        trees = by_virus.get(virus, [])
        labels = ([spec["canonical"]] if args.mode == "single"
                  else list(spec["panels"].keys()))
        for label in labels:
            if (virus, label) in EVAL_REUSE:
                continue
            pgenes = spec["panels"][label]
            cands = []
            for (slug, gene_muts, ntips, tp, ref) in trees:
                if ref and all(g in gene_muts for g in pgenes):
                    nmuts = sum(gene_muts[g] for g in pgenes)
                    cpath = charon_path(slug)
                    has_snap = verdicts.get(cpath) == "snapshot"
                    cands.append({"slug": slug, "nmuts": nmuts, "ntips": ntips,
                                  "tp": tp, "ref": ref, "cpath": cpath,
                                  "has_snap": has_snap})
            panel = f"{virus.replace('-', '_')}_{label}"
            if not cands:
                selections.append({"panel": panel, "virus": virus, "label": label,
                                   "genes": pgenes, "slug": None})
                continue
            snap_cands = [c for c in cands if c["has_snap"]]
            pick = max(snap_cands or cands, key=lambda c: c["nmuts"])
            pick.update({"panel": panel, "virus": virus, "label": label, "genes": pgenes,
                         "route": "snapshot" if pick["has_snap"] else "prune"})
            selections.append(pick)

    chosen = [s for s in selections if s.get("slug")]
    n_snap = sum(1 for s in chosen if s["route"] == "snapshot")
    print(f"=== mode={args.mode}: {len(chosen)} panels "
          f"({n_snap} snapshot / {len(chosen) - n_snap} prune-fallback); "
          f"{len(EVAL_REUSE)} eval pairs reused elsewhere ===\n")
    print(f"{'panel':26s} {'genes':20s} {'aa_muts':>7} {'route':>8}  tree-slug")
    for s in selections:
        if not s.get("slug"):
            print(f"{s['panel']:26s} {','.join(s['genes']):20s} {'-':>7} {'NONE':>8}  -- none --")
            continue
        print(f"{s['panel']:26s} {','.join(s['genes']):20s} {s['nmuts']:>7} "
              f"{s['route']:>8}  {s['slug']}")

    if args.dry_run:
        return 0

    manifest, rows = {}, []
    tmpdir = tempfile.mkdtemp(prefix="core_prune_")
    for s in chosen:
        panel, pgenes = s["panel"], s["genes"]
        out_path = os.path.join(out_dir, f"examples_{panel}@{DATE}_train.txt")
        if s["route"] == "snapshot":
            fetched = fetch_snapshot(s["cpath"], local_ref=s["ref"])
            if not fetched:
                rows.append((panel, "-", "-", f"snapshot fetch failed ({s['cpath']})"))
                continue
            tree_path, ref_path = fetched
        else:
            tree_path = os.path.join(tmpdir, f"{panel}.json")
            ref_path = prune_tree_file(s["tp"], tree_path)
        seq_len, examples = run_generator(tree_path, ref_path, pgenes, out_path)
        if seq_len is None:
            rows.append((panel, "-", "-", f"[{s['route']}] {examples}"))
            continue
        if (examples or 0) < MIN_EXAMPLES:
            rows.append((panel, seq_len, examples, f"too few (<{MIN_EXAMPLES}), skipped"))
            if os.path.exists(out_path):
                os.remove(out_path)
            continue
        rows.append((panel, seq_len, examples, f"ok [{s['route']}]"))
        manifest[panel] = {"dataset": s["slug"], "virus": s["virus"], "genes": pgenes,
                           "path": out_path, "seq_len": seq_len, "examples": examples,
                           "n_tips": s["ntips"], "source": "core", "attach_mode": "none",
                           "cutoff": DATE, "route": s["route"], "nextstrain": s["cpath"]}

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    print(f"\n{'panel':26s} {'seq_len':>7} {'examples':>9}  note")
    for panel, seq_len, examples, note in sorted(rows):
        print(f"{panel:26s} {str(seq_len):>7} {str(examples):>9}  {note}")
    total = sum(m["examples"] for m in manifest.values())
    n_snap = sum(1 for m in manifest.values() if m["route"] == "snapshot")
    print(f"\n{len(manifest)} panels ({n_snap} snapshot / {len(manifest) - n_snap} prune), "
          f"{total} examples -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
