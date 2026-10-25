"""Collate the quarterly cutoff series into one table.

    python3 -I summarize_series.py [work/series] [output]

Reads each cutoff's run log and writes <output>/series_summary.tsv plus a short
report on stdout. A cutoff with no log, or a log with no "done" line, is listed
as MISSING rather than skipped -- the failure mode this is guarding against is a
task that was held and never ran, which leaves the array looking finished.
"""
import os
import re
import sys

COLS = ["cutoff", "tree", "build_date", "in_window", "fail_coverage", "fail_clock_high",
        "fail_clock_low", "attached", "attached_pct", "tree_nodes", "tree_tips",
        "placements", "matched", "split", "final_nodes", "records", "mutations",
        "checker", "elapsed", "status"]


def parse(path):
    t = open(path, errors="replace").read()
    g = lambda rx, n=1, d="": (m.group(n) if (m := re.search(rx, t, re.M)) else d)
    r = {
        "cutoff": g(r"^cutoff : (\S+)"),
        "tree": os.path.basename(g(r"^tree   : (\S+)")),
        "in_window": g(r"^fasta.*?(\d+) in the date window"),
        "fail_coverage": g(r"^filters\s+: (\d+) fail coverage"),
        "fail_clock_high": g(r"^\s+(\d+) too diverged"),
        "fail_clock_low": g(r"too diverged for their date, (\d+) too ancestral"),
        "attached": g(r"^attached\s+: (\d+) \("),
        "attached_pct": g(r"^attached\s+: \d+ \(([\d.]+)%\)"),
        "tree_nodes": g(r"^tree\s+: (\d+) nodes"),
        "tree_tips": g(r"^tree\s+: \d+ nodes, (\d+) tips"),
        "matched": g(r"^ties\s+: (\d+) placements matched"),
        "records": g(r"^examples\s+: (\d+) records"),
        "mutations": g(r"^examples\s+: \d+ records, (\d+) mutations"),
        "checker": g(r"^checker: exit (\d+)", 1, "-"),
    }
    r["build_date"] = (m.group(1) if (m := re.search(r"@(\d{4}-\d{2}-\d{2})", r["tree"])) else "")
    r["placements"] = g(r"placed\s+(\d+) sequences") or r["attached"]
    r["split"] = g(r"(\d+) splits?\b")
    r["final_nodes"] = g(r"final tree\s*:?\s*(\d+) nodes") or g(r"^tree\s+: (\d+) nodes.*$")
    start, done = g(r"^start  : (.+)$"), g(r"^done   : (.+)$")
    r["elapsed"] = ""
    if start and done:
        try:
            import datetime as dt
            f = "%a %b %d %I:%M:%S %p %Z %Y"
            a = dt.datetime.strptime(" ".join(start.split()), f.replace("%Z ", ""))
            b = dt.datetime.strptime(" ".join(done.split()), f.replace("%Z ", ""))
            r["elapsed"] = str(b - a)
        except Exception:
            r["elapsed"] = ""
    r["status"] = "OK" if done and r["checker"] == "0" else ("NO-DONE" if not done else "CHECKER")
    return r


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else "work/series"
    out = sys.argv[2] if len(sys.argv) > 2 else "output"
    want = [l.split("\t")[0] for l in open("slurm/cutoffs.tsv")]
    logs = {}
    for f in sorted(os.listdir(work)):
        if not re.fullmatch(r"run_\d+_\d+\.log", f):
            continue
        r = parse(os.path.join(work, f))
        if r["cutoff"]:                      # later runs of a requeued task win
            logs[r["cutoff"]] = r
    rows = [logs.get(c, {"cutoff": c, "status": "MISSING"}) for c in want]
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, "series_summary.tsv")
    with open(dest, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in COLS) + "\n")
    print(f"{'cutoff':12s} {'tree build':12s} {'attached':>10s} {'%':>6s} "
          f"{'records':>8s} {'muts':>8s} {'chk':>4s}  status")
    for r in rows:
        print(f"{r['cutoff']:12s} {r.get('build_date',''):12s} {r.get('attached',''):>10s} "
              f"{r.get('attached_pct',''):>6s} {r.get('records',''):>8s} "
              f"{r.get('mutations',''):>8s} {r.get('checker',''):>4s}  {r['status']}")
    bad = [r["cutoff"] for r in rows if r["status"] != "OK"]
    print(f"\nwrote {dest}")
    print(f"{len(rows) - len(bad)}/{len(rows)} cutoffs OK; not OK: {bad or 'none'}")


if __name__ == "__main__":
    main()
