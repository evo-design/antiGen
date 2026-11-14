#!/usr/bin/env python
"""Per-virus EVE MSAs for the nine-virus forecast panel, built ONLY from the Nextstrain
snapshot trees the panel trains on (D = 2025-01-01), so EVE sees exactly the sequence
information every other rung sees -- no outside alignments.

For each tree: parse the @2025-01-01 snapshot auspice JSON with make_tree_examples.py's own
parser and settings (--genes / --ambiguous-ref gap / --strip-trailing-stop / --drop-gap-stop
and the same reference), WITHOUT collapsing zero-AA branches, and write every TIP's
reconstructed amino-acid genotype (root + the mutations on its path). Row 0 is the root
genotype (EVE's reference row). Causal by construction: every snapshot has 0 tips >= D.

The settings below are the ones that regenerate each data/forecast_noattach train file
BYTE-IDENTICALLY (checked 2026-11-13); this script re-asserts that the root row equals the
train file's first example sequence and that the length equals its seq_len.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_tree_examples as mte
import wide_corpus as wc

D = "2025-01-01"
NA = f"{HERE}/data/forecast_noattach"
OUT = f"{HERE}/data/eve_msa_j9"
NCOV_REF = f"{wc.CACHE}/ncov_open_global_all-time@2026-05-01_root-sequence.json"  # Wuhan-Hu-1, date-invariant
TREES = {'h5nx': ('avian-flu/h5nx/ha/all-time', ['HA']),
         'ncov': ('ncov/open/global/all-time', ['S']),
         'rsv_a': ('rsv/a/F/all-time', ['F']), 'rsv_b': ('rsv/b/F/all-time', ['F']),
         'measles': ('measles/genome/global', ['H']),
         'h3n2_ha': ('seasonal-flu/h3n2/ha/12y', ['SigPep', 'HA1', 'HA2']),
         'h1n1pdm_ha': ('seasonal-flu/h1n1pdm/ha/12y', ['SigPep', 'HA1', 'HA2']),
         'vic_ha': ('seasonal-flu/vic/ha/12y', ['SigPep', 'HA1', 'HA2']),
         'hmpv_a_g': ('hmpv/a/G', ['G'])}


def train_root(t):
    """First example's sequence in the train file (the root genotype) as a string."""
    sys.path.insert(0, os.path.dirname(HERE))
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False; _iu._torchvision_version = "N/A"
    from antiGen.load import load_examples
    ex = load_examples(f"{NA}/examples_{t}@{D}_train.txt", mte.ALPHABET, None, source=t)
    return "".join(mte.ALPHABET[i] for i in ex[0].aa_idx.tolist())


def main():
    os.makedirs(OUT, exist_ok=True)
    for t, (path, genes) in TREES.items():
        tj = mte.load_json_maybe_gzip(wc.cache_path(path, D))
        ref_obj = (mte.load_json_maybe_gzip(NCOV_REF) if t == 'ncov' else tj["root_sequence"])
        ref_seq, offsets = mte.build_reference(ref_obj, genes, True, ambiguous='gap')
        L = len(ref_seq)
        stats = {"malformed": 0, "ambiguous": 0, "out_of_range": 0, "from_mismatch": 0}
        root = mte.parse_tree(tj["tree"], genes, offsets, L, stats, mte.STANDARD_AA)
        root.seq = mte.apply_mutations(ref_seq, root.muts, root.name, stats)
        tips, stack = [], [root]
        while stack:
            n = stack.pop()
            for c in n.children:
                c.seq = mte.apply_mutations(n.seq, c.muts, c.name, stats)
                stack.append(c)
            if not n.children:
                tips.append((n.name, n.seq))
        tr = train_root(t)
        assert len(root.seq) == L == len(tr), f"{t}: length {len(root.seq)} vs train {len(tr)}"
        assert root.seq == tr, f"{t}: root genotype != train file's first example"
        with open(f"{OUT}/{t}@{D}_tips.fasta", "w") as fh:
            fh.write(f">ROOT_{t}\n{root.seq}\n")
            for name, seq in tips:
                fh.write(f">{name}\n{seq}\n")
        n_uniq = len({s for _, s in tips})
        print(f"{t:11} L={L:5d} tips={len(tips):5d} unique={n_uniq:5d}  root==train OK  "
              f"from-mismatch={stats['from_mismatch']}", flush=True)


if __name__ == "__main__":
    main()
