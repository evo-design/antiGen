#!/usr/bin/env python
"""Per-residue relative solvent accessibility (RSA) for SARS-CoV-2 spike, mapped to the
1273 model frame -- the isolated 'accessibility' signal behind EVEscape, computed whole-
protein from structure (no RBD coverage limit).

SASA is computed on the full CLOSED TRIMER (6VXX, all three protomers) so a residue buried
at the inter-protomer interface reads as buried, not exposed. We then read chain A's
per-residue SASA, divide by Tien-2013 theoretical maxASA -> RSA in [0,1], and place it at
0-index resnum-1. The frame is checked against the ncov reference (Wuhan-Hu-1): chain A's
residue identities must match the reference at that offset. Missing/disordered residues
(NTD loops, furin site) stay NaN -> neutral (0) gate.
Writes accessibility_sc2.npz: acc (L,), covered (L,).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from antiGen.load import load_examples

_3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
         'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
         'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def three_to_one(x): return _3TO1[x]
from score_perevent import ALPHA_SET, D

PDB = "/tmp/claude-10361/-home-aakarsh-projects-antiGen/8850eec5-d868-444b-9082-8c15bdc010c4/scratchpad/struct/6vxx.pdb"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/accessibility_sc2.npz"
# Tien et al. 2013 theoretical maxASA (Angstrom^2)
MAXASA = {'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,
          'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,
          'Y':263,'V':174}

def main():
    tr = load_examples(f"{NA}/examples_ncov@{D}_train.txt", ALPHA_SET, None, source="ncov")
    ref = "".join(ALPHA_SET[int(i)] for i in tr[0].aa_idx)      # Wuhan-Hu-1 spike, L=1273
    L = len(ref)
    st = PDBParser(QUIET=True).get_structure("s", PDB)
    model = st[0]
    ShrakeRupley().compute(model, level="R")                    # trimer-context residue SASA
    chainA = model["A"]
    acc = np.full(L, np.nan); match = tot = 0
    for res in chainA:
        if res.id[0] != " ": continue                           # skip hetero/water
        try: aa = three_to_one(res.resname)
        except Exception: continue
        i = res.id[1] - 1                                       # author numbering -> 0-index
        if not (0 <= i < L): continue
        tot += 1
        if ref[i] == aa: match += 1
        rsa = res.sasa / MAXASA.get(aa, 200.0)
        acc[i] = min(rsa, 1.0)
    print(f"chain A residues mapped: {tot}   wildtype match to ncov reference: {match}/{tot}"
          f" = {match/tot:.3f}")
    covered = ~np.isnan(acc)
    print(f"accessibility coverage: {covered.sum()}/{L} spike positions")
    for lab, lo, hi in [("NTD",13,305),("RBD",330,530),("S1-other",531,685),("S2",686,1272)]:
        seg = covered[lo:hi+1]
        print(f"  {lab:9s} {lo}-{hi}: {seg.sum()}/{hi-lo+1} covered, "
              f"mean RSA {np.nanmean(acc[lo:hi+1]):.3f}")
    np.savez(OUT, acc=acc, covered=covered)
    print("wrote", OUT)

if __name__ == '__main__':
    main()
