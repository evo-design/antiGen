#!/usr/bin/env python
"""Re-label a forecast test set's novelty against the ATTACH training corpus.

The post-D test targets are the same regardless of how training data is built, but
"novel" means "never seen in training", and that reference is arm-dependent:

- no-attach arm: training IS the snapshot tree, so make_forecast_testset.py's
  snapshot-substitution novelty is already correct.
- attach arm: training is the parsimony-attached corpus (antiGen-data-revised), which
  contains more observed substitutions than the bare snapshot tree. A target absent
  from the snapshot but present in the attach corpus was in fact seen in training, so
  it is NOT novel for this arm.

This reads the attach train examples file, collects every (pos, alt) substitution it
contains (MUTATION_FROM_PARENT and MUTATIONS_TO_CHILDREN, both 0-based), and re-emits
the attach arm's <base>_novel.txt and <base>_novelty.json from the shared _test.txt.
The _test.txt itself is unchanged and shared by both arms.

Usage:
    python relabel_novelty_attach.py \
        --train experiments/multitree/data/forecast_attach/examples_denv2@2025-01-01_train.txt \
        --test  experiments/multitree/data/forecast_noattach/examples_denv2@2025-01-01_test.txt \
        --out   experiments/multitree/data/forecast_attach/examples_denv2@2025-01-01_test.txt
"""
from __future__ import annotations

import argparse
import json
import os


def seen_substitutions(train_path):
    """Every (pos, alt) the attach training file records as a substitution event."""
    seen = set()
    in_children = False
    with open(train_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "MUTATIONS_TO_CHILDREN:":
                in_children = True
                continue
            if line.startswith("MUTATION_FROM_PARENT"):
                in_children = False
                continue
            if line == "" or line.startswith("EXAMPLE_ID"):
                in_children = False
            if in_children and "," in line:
                pos, alt = line.split(",", 1)
                try:
                    seen.add((int(pos), alt))
                except ValueError:
                    pass
            elif (not in_children) and line.count(",") == 2:
                # MUTATION_FROM_PARENT body: "from,pos,to"
                frm, pos, to = line.split(",")
                try:
                    seen.add((int(pos), to))
                except ValueError:
                    pass
    return seen


def read_test_examples(test_path):
    """Parse the shared _test.txt into [(query_seq, [(pos, alt), ...]), ...]."""
    examples, seq, targets, in_children = [], None, [], False
    with open(test_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("EXAMPLE_ID"):
                if seq is not None:
                    examples.append((seq, targets))
                seq, targets, in_children = None, [], False
            elif line == "MUTATIONS_TO_CHILDREN:":
                in_children = True
            elif in_children and "," in line:
                pos, alt = line.split(",", 1)
                targets.append((int(pos), alt))
            elif (not in_children) and line and set(line) <= set("ACDEFGHIKLMNPQRSTVWY*-") \
                    and len(line) > 20:
                seq = line
    if seq is not None:
        examples.append((seq, targets))
    return examples


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="attach training examples file")
    p.add_argument("--test", required=True, help="shared _test.txt (post-D targets)")
    p.add_argument("--out", required=True,
                   help="attach-arm _test path; _novel.txt and _novelty.json are "
                        "derived from it (the _test.txt itself is not rewritten)")
    p.add_argument("--date", default="2025-01-01")
    args = p.parse_args(argv)

    known = seen_substitutions(args.train)
    examples = read_test_examples(args.test)
    seq_len = len(examples[0][0]) if examples else 0

    n_targets = sum(len(t) for _, t in examples)
    n_novel = sum(1 for _, t in examples for m in t if m not in known)

    base = os.path.splitext(args.out)[0]
    novel_path = base.replace("_test", "") + "_novel.txt"
    written = 0
    with open(novel_path, "w") as out:
        for seq, targets in examples:
            keep = [m for m in targets if m not in known]
            if not keep:
                continue
            out.write(f"EXAMPLE_ID: {written}\n")
            out.write("PARENT_EXAMPLE_ID: None\n")
            out.write("MUTATION_FROM_PARENT:\nNone\n")
            out.write("TESTING\n")
            out.write("IS_DESCENDED_FROM_VARIANT:\n")
            out.write("IS_VARIANT_ROOT:\n")
            out.write(f"{seq}\n")
            out.write("MUTATIONS_TO_CHILDREN:\n")
            for pos, alt in keep:
                out.write(f"{pos},{alt}\n")
            out.write("\n")
            written += 1

    novelty = {i: [[pos, alt, (pos, alt) not in known] for pos, alt in targets]
               for i, (_seq, targets) in enumerate(examples)}
    side = base + "_novelty.json"
    with open(side, "w") as handle:
        json.dump({"date": args.date, "seq_len": seq_len, "window": "all",
                   "novelty_reference": "attach_training", "train_file": args.train,
                   "n_examples": len(examples), "n_targets": n_targets,
                   "n_novel": n_novel, "per_example": novelty}, handle)

    print(f"{args.train}\n  novelty vs attach training ({len(known)} distinct subs seen)")
    print(f"  examples={len(examples)} targets={n_targets} "
          f"novel={n_novel} ({100 * n_novel / max(n_targets, 1):.0f}%)  "
          f"-> {novel_path} ({written} examples), {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
