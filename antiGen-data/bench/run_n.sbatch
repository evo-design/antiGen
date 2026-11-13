#!/bin/bash
#SBATCH --job-name=ubench
#SBATCH --partition=cpu_high_mem
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/large_storage/hielab/ivan.specht/usher-bench/logs/run_%x_%j.out
#
# One N, every tool, back to back on one allocation -- the nodes are shared, so
# running the tools of one N together is what keeps their timings comparable.
#
#   sbatch --job-name=ubench_10000 bench/run_n.sbatch 10000 ours ushs:3 ushc
#
# Tools:
#   ours  antigen prepare + place, nucleotide alphabet, input order kept
#   ushs  usher-sampled, placement only (every optimization round disabled)
#   ushc  classic usher, placement only (strictly sequential, like ours)
#   ushs1 usher-sampled with --batch_size_per_process 1: one sample searched at a
#         time, committed before the next -- sequential, pruning kept
#   ushsb1 the same, as chained 100k batches
#   ushsb usher-sampled as chained 100k batches, each placed onto the tree the
#         last one saved, the batches cut from the same placement order. A single usher-sampled run of 1M dies with "node id
#         Mismatch" and a segfault before placing anything, in v0.6.6, with or
#         without optimization. Times are summed over batches.
#
# Every final tree is then re-scored by one judge -- UShER's own Fitch-Sankoff
# over a Newick of that tree and one VCF of every tip -- so the parsimony
# numbers compared are computed the same way whatever produced the tree.
# Each tool's self-reported score is kept alongside.
#
# Writes $BD/results/<tool>_<N>_r<rep>.csv, one row each:
#   tool,N,rep,threads,place_s,wall_s,max_rss_kb,final_nodes,parsimony_common,parsimony_self,prep_s
set -uo pipefail
N=$1; shift
TOOLS=("$@")
# Fixed, not taken from the allocation: asking for more memory makes SLURM hand
# out more CPUs, and the thread count has to be the same at every N.
THREADS=${BENCH_THREADS:-32}

BD=/large_storage/hielab/ivan.specht/usher-bench
# Placement inputs in the order both tools place them -- by date, a seeded
# random order within a day (bench/reorder_inputs.py). The scorer's tip VCFs
# do not depend on order and stay with the unordered originals.
IN=$BD/inputs_chrono
ORIG=$BD/inputs
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
BIN=${BENCH_BIN:-$REPO/work/antigen_bound3}
PY=/home/ivan.specht/miniconda/bin/python3
GUIDE_PB=/home/ivan.specht/antiGen-data/bench/guide.pb
GUIDE_VCF=/home/ivan.specht/antiGen-data/bench/guide.vcf
TREE=/home/ivan.specht/sc2-open-data/tree.json
REF=/home/ivan.specht/sc2-open-data/ref.json
source /home/ivan.specht/miniconda/etc/profile.d/conda.sh; conda activate usher
mkdir -p "$BD/results"

# Open MPI under SLURM tries to join the allocation as a multi-rank job and
# usher-sampled exits 1; with these unset it starts as a singleton.
UNSET=$(for v in $(compgen -v | grep -E '^SLURM_|^PMI_|^PMIX_'); do printf ' -u %s' "$v"; done)
quiet() { grep -v -iE "openfabric|mlx5|^-+$|local host|local device|device|btl_|note: you|lower perf|vendor|^\s*$"; }

echo "host    : $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p'))"
echo "N       : $N   threads $THREADS   tools ${TOOLS[*]}"
echo "binary  : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))"
echo "usher   : $(usher --version 2>&1 | grep -m1 UShER)"
echo "start   : $(date)"

# --- the scorer's tip VCF, built once per N -----------------------------------
TIPS=$ORIG/tips_$N.vcf
(
  flock 9
  if [ ! -s "$TIPS" ]; then
    $PY -I "$REPO/bench/merge_vcf.py" "$GUIDE_VCF" "$ORIG/append_$N.vcf" "$TIPS.tmp" && mv "$TIPS.tmp" "$TIPS"
  fi
) 9> "$TIPS.lock"
[ -s "$TIPS" ] || { echo "no tip VCF"; exit 1; }

# score <newick> <dir>: total parsimony of that topology over $TIPS. UShER and
# matUtils prefix their output directory to output paths, so every call runs
# from inside the directory with bare file names.
score() {
  local nwk=$1 d=$2
  mkdir -p "$d"
  (cd "$d" && usher -t "$nwk" -v "$TIPS" -o scored.pb -d . -T "$THREADS" > usher.log 2>&1)
  local sum; sum=$(cd "$d" && matUtils summary -i scored.pb 2>/dev/null)
  local s; s=$(echo "$sum" | sed -n 's/^Total Tree Parsimony: //p')
  local tips; tips=$(echo "$sum" | sed -n 's/^Total Samples in Tree: //p')
  echo "scored  : $nwk -> parsimony ${s:-NA} over ${tips:-NA} samples" >&2
  echo "${s:-NA}"
}
timed() {  # timed <timefile> <logfile> cmd...  -> prints "wall rss"
  local tf=$1 lf=$2; shift 2
  /usr/bin/time -f "WALL=%e RSS=%M" -o "$tf" "$@" > "$lf" 2>&1
  echo "$(sed -n 's/.*WALL=\([0-9.]*\).*/\1/p' "$tf") $(sed -n 's/.*RSS=\([0-9]*\).*/\1/p' "$tf")"
}
row() { local f="$BD/results/$1_${N}_r$REP.csv"
        echo "tool,N,rep,threads,place_s,wall_s,max_rss_kb,final_nodes,parsimony_common,parsimony_self,prep_s" > "$f"
        echo "$1,$N,$REP,$THREADS,$2,$3,$4,$5,$6,$7,$8" >> "$f"
        echo "result  : $(tail -1 "$f")"; }

# A tool may be given as tool:k to run it k times -- usher-sampled commits its
# parallel searches in completion order, so its tree and score vary run to run.
for SPEC in "${TOOLS[@]}"; do
 TOOL=${SPEC%%:*}; NREP=1; [ "$SPEC" != "$TOOL" ] && NREP=${SPEC##*:}
 for REP in $(seq 1 "$NREP"); do
  D=$BD/runs/${TOOL}_${N}_r$REP
  # ushs1/ushsb1: usher-sampled with one sample in flight at a time. Each search
  # is committed to the tree before the next one starts (place_sample.cpp:
  # update_main_tree, then retry_callback), so placement is sequential, while
  # each search keeps its pruning and its internal parallelism.
  SEQARG=()
  case $TOOL in ushs1|ushsb1) SEQARG=(--batch_size_per_process 1) ;; esac
  rm -rf "$D"; mkdir -p "$D"
  echo; echo "=== $TOOL N=$N rep $REP/$NREP  $(date) ==="
  case $TOOL in
  ours)
    read PW PR < <(timed "$D/prep.time" "$D/prepare.log" "$BIN" prepare --alphabet nuc --gene nuc \
        --root-seq "$REF" --fasta "$IN/append_$N.fasta" --metadata "$IN/append_${N}_meta.tsv" \
        --date-cutoff none --no-clock --min-covered 0 --keep-order --out "$D/prep.tsv")
    EVERY=$(( N >= 100000 ? N / 200 : (N >= 1000 ? N / 100 : 10) ))
    read W R < <(timed "$D/place.time" "$D/place.log" "$BIN" place --alphabet nuc --gene nuc \
        --tree "$TREE" --root-seq "$REF" --prepared "$D/prep.tsv" --keep-order \
        --threads "$THREADS" --progress-every "$EVERY" \
        --out-tree "$D/tree.tsv" --out-matches "$D/matches.tsv")
    grep -E "attached|filters" "$D/prepare.log"
    grep -E "^allele sets|^placed|added|split a branch|matched an existing" "$D/place.log"
    PLACE=$(sed -n 's/^placed *: [0-9]* sequences in \([0-9.]*\) s.*/\1/p' "$D/place.log")
    NODES=$(sed -n 's/.*tree now \([0-9]*\) nodes.*/\1/p' "$D/place.log")
    BASE=$(sed -n 's/^allele sets *: parsimony score \([0-9]*\).*/\1/p' "$D/place.log")
    ADDED=$(sed -n 's/.*added \([0-9]*\) to the parsimony score.*/\1/p' "$D/place.log")
    SELF=$(( BASE + ADDED ))
    rm -f "$D/prep.tsv"
    $PY -I "$REPO/bench/ours_to_newick.py" "$D/tree.tsv" "$D/matches.tsv" "$D/final.nwk"
    COMMON=$(score "$D/final.nwk" "$D/score")
    row ours "$PLACE" "$W" "$R" "$NODES" "$COMMON" "$SELF" "$PW"
    ;;
  ushs|ushs1|ushc)
    if [ "$TOOL" != ushc ]; then
      read W R < <(cd "$D" && timed place.time place.log env $UNSET usher-sampled -i "$GUIDE_PB" \
          -v "$IN/append_$N.vcf" -T "$THREADS" --optimization_radius 0 --optimization_minutes 0 \
          --last_optimization_minutes 0 "${SEQARG[@]}" -o out.pb -d .)
      MS=$(grep -oE "Took [0-9]+ msec" "$D/place.log" | grep -oE "[0-9]+" | tail -1)
      PLACE=$(awk -v m="${MS:-NA}" 'BEGIN{ if (m=="NA") print "NA"; else printf "%.3f", m/1000 }')
      if grep -qiE "optimiz.*(round|iteration)|SPR" "$D/place.log"; then echo "WARNING: optimization ran"; fi
    else
      read W R < <(cd "$D" && timed place.time place.log usher -i "$GUIDE_PB" \
          -v "$IN/append_$N.vcf" -T "$THREADS" -o out.pb -d .)
      PLACE=$W   # classic usher has no placement timer of its own: wall time
    fi
    quiet < "$D/place.log" | grep -vE "Sample name:|^Current tree size" | tail -4
    rm -f "$D/mutation-paths.txt"
    SUM=$(cd "$D" && matUtils summary -i out.pb 2>/dev/null)
    SELF=$(echo "$SUM" | sed -n 's/^Total Tree Parsimony: //p')
    NODES=$(echo "$SUM" | sed -n 's/^Total Nodes in Tree: //p')
    (cd "$D" && matUtils extract -i out.pb -t final.nwk > extract.log 2>&1)
    COMMON=$(score "$D/final.nwk" "$D/score")
    row "$TOOL" "${PLACE:-NA}" "$W" "$R" "${NODES:-NA}" "$COMMON" "${SELF:-NA}" "NA"
    ;;
  ushsb|ushsb1)
    # Consecutive 100k slices of this N's own placement order, cut by
    # reorder_inputs.py --batch 100000.
    NB=$(ls "$IN"/batch_${N}_*.vcf 2>/dev/null | wc -l)
    [ "$NB" -ge 1 ] || { echo "no batch_${N}_*.vcf in $IN"; exit 1; }
    CUR=$GUIDE_PB; PLACE=0; W=0; R=0
    for i in $(seq 0 $(( NB - 1 ))); do
      mkdir -p "$D/b$i"
      read BW BR < <(cd "$D/b$i" && timed place.time place.log env $UNSET usher-sampled -i "$CUR" \
          -v "$IN/batch_${N}_$i.vcf" -T "$THREADS" --optimization_radius 0 --optimization_minutes 0 \
          --last_optimization_minutes 0 "${SEQARG[@]}" -o out.pb -d .)
      MS=$(grep -oE "Took [0-9]+ msec" "$D/b$i/place.log" | grep -oE "[0-9]+" | tail -1)
      [ -s "$D/b$i/out.pb" ] && [ -n "$MS" ] || { echo "batch $i failed"; quiet < "$D/b$i/place.log" | tail -5; break; }
      rm -f "$D/b$i/mutation-paths.txt" "$D/b$i/final-tree.nh"
      PLACE=$(awk -v a="$PLACE" -v m="$MS" 'BEGIN{printf "%.3f", a + m/1000}')
      W=$(awk -v a="$W" -v b="$BW" 'BEGIN{printf "%.2f", a + b}')
      [ "$BR" -gt "$R" ] && R=$BR
      echo "batch $i: took $MS ms, wall $BW s, rss $BR kB, running place total $PLACE s"
      [ "$i" -gt 0 ] && [ "$CUR" != "$GUIDE_PB" ] && rm -f "$CUR"
      CUR=$D/b$i/out.pb
    done
    mv "$CUR" "$D/out.pb"
    SUM=$(cd "$D" && matUtils summary -i out.pb 2>/dev/null)
    SELF=$(echo "$SUM" | sed -n 's/^Total Tree Parsimony: //p')
    NODES=$(echo "$SUM" | sed -n 's/^Total Nodes in Tree: //p')
    TIPS_IN=$(echo "$SUM" | sed -n 's/^Total Samples in Tree: //p')
    echo "final   : $TIPS_IN samples in the tree"
    (cd "$D" && matUtils extract -i out.pb -t final.nwk > extract.log 2>&1)
    COMMON=$(score "$D/final.nwk" "$D/score")
    row "$TOOL" "$PLACE" "$W" "$R" "${NODES:-NA}" "$COMMON" "${SELF:-NA}" "NA"
    ;;
  *) echo "unknown tool $TOOL"; exit 1 ;;
  esac
  rm -f "$D/score/scored.pb"
 done
done
echo; echo "done    : $(date)"
