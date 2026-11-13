#!/bin/bash
#SBATCH --job-name=ubench-ident
#SBATCH --partition=cpu_high_mem
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/large_storage/hielab/ivan.specht/usher-bench/logs/ident_%j.out
#
# A placement-speed change must not change any placement. Old and new binary,
# same inputs, back to back on one allocation: the tree, the matches, every
# progress line (nodes and score) and every placement statistic must be
# identical. Only time and the scan's visit counters may differ.
#
#   sbatch bench/ident.sbatch <old-binary> <new-binary> N...
set -uo pipefail
OLD=$1; NEW=$2; shift 2
BD=/large_storage/hielab/ivan.specht/usher-bench
O=$BD/runs/ident_$SLURM_JOB_ID; mkdir -p "$O"
echo "old $OLD ($(md5sum "$OLD" | cut -c1-8))  new $NEW ($(md5sum "$NEW" | cut -c1-8))  host $(hostname)"
fail=0
for N in "$@"; do
  "$NEW" prepare --alphabet nuc --gene nuc --root-seq ~/sc2-open-data/ref.json \
    --fasta "$BD/inputs_chrono/append_$N.fasta" --metadata "$BD/inputs_chrono/append_${N}_meta.tsv" \
    --date-cutoff none --no-clock --min-covered 0 --keep-order --out "$O/prep_$N.tsv" > /dev/null 2>&1
  for b in old new; do
    BIN=$OLD; [ $b = new ] && BIN=$NEW
    "$BIN" place --alphabet nuc --gene nuc --tree ~/sc2-open-data/tree.json --root-seq ~/sc2-open-data/ref.json \
      --prepared "$O/prep_$N.tsv" --keep-order --threads 32 --progress-every 1000 \
      --out-tree "$O/tree_${b}_$N.tsv" --out-matches "$O/matches_${b}_$N.tsv" > "$O/place_${b}_$N.log" 2>&1
  done
  rm -f "$O/prep_$N.tsv"
  st() { grep -E "^(placed|               [0-9]+ split|               added|ties|               [0-9]+ of those|               of the|               [0-9]+ (had|tied)|               draw size|ambiguity|               spread|               set sizes)" "$1" | sed 's/in [0-9.]* s/in T s/'; }
  t=$(cmp -s "$O/tree_old_$N.tsv" "$O/tree_new_$N.tsv" && echo same || echo DIFF)
  m=$(cmp -s "$O/matches_old_$N.tsv" "$O/matches_new_$N.tsv" && echo same || echo DIFF)
  p=$(diff <(grep -oE "placed, [0-9]+ nodes, score [0-9]+" "$O/place_old_$N.log") <(grep -oE "placed, [0-9]+ nodes, score [0-9]+" "$O/place_new_$N.log") > /dev/null && echo same || echo DIFF)
  s=$(diff <(st "$O/place_old_$N.log") <(st "$O/place_new_$N.log") > /dev/null && echo same || echo DIFF)
  for x in $t $m $p $s; do [ $x = same ] || fail=1; done
  echo "N=$N tree:$t matches:$m progress:$p stats:$s"
  echo "   old: $(grep -oE 'in [0-9.]+ s' "$O/place_old_$N.log")  $(grep -oE '\([0-9]+ per placement\)' "$O/place_old_$N.log")"
  echo "   new: $(grep -oE 'in [0-9.]+ s' "$O/place_new_$N.log")  $(grep -oE '\([0-9]+ per placement\)' "$O/place_new_$N.log")  $(grep -A2 '^scan' "$O/place_new_$N.log" | tr -s ' ' | tail -2 | tr '\n' ' ')"
  rm -f "$O"/tree_*_$N.tsv
done
[ $fail = 0 ] && echo "ALL IDENTICAL" || echo "SOMETHING DIFFERS"
exit $fail
