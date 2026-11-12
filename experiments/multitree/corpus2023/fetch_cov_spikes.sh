#!/bin/bash
# Coronavirus spike proteins for the 2023-01-01 pretraining corpus (NOTES s155).
#
# One Entrez protein query per group, RESTRICTED BY PUBLICATION DATE to before
# 2023-01-01: a record published before the cutoff cannot carry anything sampled after
# it, whatever its collection date says (and many have none). SARS-CoV-2 is excluded by
# taxon from the sarbecovirus group; build_cov_tree.py also drops anything within 1% of
# the SARS-CoV-2 reference, in case a SARS-CoV-2 spike is filed under another name.
#
# Writes $OUT/<group>/raw.faa and the GenPept records (records.gpc.xml), which
# build_cov_tree.py parses for organism, host, country and collection date.
set -euo pipefail
OUT=${OUT:-/large_storage/hielab/ispecht/fitness-inference/corpus2023/cov}
S='(spike[Title] OR "surface glycoprotein"[Title] OR "S protein"[Title] OR "spike glycoprotein"[Title] OR "S glycoprotein"[Title])'
P='1900/01/01:2022/12/31[PDAT]'
L='900:1500[SLEN]'
COV_GROUPS="sarbeco|\"Sarbecovirus\"[Organism] NOT \"Severe acute respiratory syndrome coronavirus 2\"[Organism]
merbeco|\"Merbecovirus\"[Organism]
betacov1|\"Betacoronavirus 1\"[Organism]
hku1|txid290028[Organism:exp]
mhv|\"Murine coronavirus\"[Organism]
hcov229e|txid11137[Organism:exp]
nl63|txid277944[Organism:exp]
pedv|txid28295[Organism:exp]
alphacov1|txid693997[Organism:exp]
ibv|txid11120[Organism:exp]
deltacov|\"Deltacoronavirus\"[Organism]"
mkdir -p "$OUT"
while IFS='|' read -r g q; do
    [ -n "${ONLY:-}" ] && ! printf '%s\n' $ONLY | grep -qx "$g" && continue
    d=$OUT/$g; mkdir -p "$d"
    query="($q) AND $S AND $L AND $P"
    echo "$query" > "$d/query.txt"
    esearch -db protein -query "$query" < /dev/null > "$d/search.xml"
    efetch -format fasta < "$d/search.xml" > "$d/raw.faa"
    efetch -format gpc < "$d/search.xml" > "$d/records.gpc.xml"
    echo "$g: $(grep -c '^>' "$d/raw.faa") sequences, $(grep -c '<INSDSeq>' "$d/records.gpc.xml") records"
done <<< "$COV_GROUPS"
