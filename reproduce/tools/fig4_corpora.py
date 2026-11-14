"""Figure 4 multi-tree corpora: each tree's 2025-01-01 snapshot as its training corpus, and the
post-cutoff edges of today's tree as the test set of the PANEL9 trees."""
import csv
import json
import os
import subprocess
import tempfile

from fig4_common import CORPORA, D, PANEL9, REPO, TREES

ANTIGEN = os.path.join(REPO, "antiGen-data", "antigen")


def root_seq(tree, tmp):
    side = tree.replace(".json", "_root-sequence.json")
    if os.path.exists(side):
        return side
    path = os.path.join(tmp, os.path.basename(side))
    with open(path, "w") as fh:
        json.dump(json.load(open(tree))["root_sequence"], fh)
    return path


def sets(tree, genes, out, tmp, *extra):
    subprocess.run([ANTIGEN, "sets", "--tree", tree, "--root-seq", root_seq(tree, tmp), "--gene", genes,
                    "--alphabet", "20", "--out-examples", out, *extra], check=True, stdout=subprocess.DEVNULL)


def main():
    os.makedirs(CORPORA, exist_ok=True)
    with open(os.path.join(REPO, "reproduce", "fig4_trees.tsv")) as fh, tempfile.TemporaryDirectory() as tmp:
        for row in csv.DictReader(fh, delimiter="\t"):
            t, genes = row["tree"], row["genes"]
            sets(f"{TREES}/{t}@{D}.json", genes, f"{CORPORA}/examples_{t}@{D}_train.txt", tmp)
            if t in PANEL9:
                sets(f"{TREES}/{t}@current.json", genes, f"{CORPORA}/examples_{t}@{D}_test.txt", tmp,
                     "--post-cutoff", D, "--post-cutoff-edges")
            print(t, flush=True)


if __name__ == "__main__":
    main()
