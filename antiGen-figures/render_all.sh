#!/bin/bash
# Render every current figure in figs/ (not figs/old/) from the staged data in data/.
#
#   cd antiGen-figures && bash render_all.sh            # all figures
#   bash render_all.sh supp_cd1 supp_lm_cd1             # just these
#
# Each figure is one Rscript call; the staged data each one reads are committed under
# data/ (how they are produced from checkpoints: REPRODUCE.md). TRACE_DIR=<dir> plus
# OPENTRACE_SO=<opentrace.so> (an LD_PRELOAD shim logging every file opened for reading)
# also writes the files each render reads -- how REPRODUCE.md's dependency list was
# built. ptrace (strace) is not permitted on the cluster.
set -euo pipefail
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}

declare -A CMD=(
  [sc2_results_revised2]="figures/fig_sc2_revised2.R"
  [sc2_results_revised2_supp]="REVISED2_VARIANT=supp figures/fig_sc2_revised2.R"
  [supp_min100_2021]="FIG_ARM=2021full figures/figS_cd1_min100.R"
  [supp_fullspike]="figures/figS_fullspike.R"
  [supp_query_adapt]="figures/figS_query_adapt.R"
  [supp_dms_revised2]="figures/figS_dms_revised2.R"
  [supp_gate_units]="figures/figS_gate_units.R"
  [supp_usher_benchmark]="figures/figS_usher_benchmark.R"
  [supp_dataseed]="figures/figS_dataseed.R"
  [supp_cv10_loss]="figures/figS_cv10_loss.R"
)
ORDER=(sc2_results_revised2 sc2_results_revised2_supp supp_min100_2021 supp_fullspike
       supp_query_adapt supp_dms_revised2 supp_gate_units supp_usher_benchmark supp_dataseed
       supp_cv10_loss)
[ $# -gt 0 ] && ORDER=("$@")

for fig in "${ORDER[@]}"; do
  c=${CMD[$fig]:?unknown figure $fig}
  script=${c##* }                                   # the last word is the script
  env_part=${c%"$script"}                          # anything before it is VAR=value
  echo "== $fig"
  if [ -n "${TRACE_DIR:-}" ]; then
    mkdir -p "$TRACE_DIR"
    rm -f "$TRACE_DIR/$fig.trace"
    env $env_part OPENTRACE_LOG="$TRACE_DIR/$fig.trace" LD_PRELOAD="${OPENTRACE_SO:?}" \
      "$RSCRIPT" "$script" 2>&1 | { grep -v Fontconfig || true; } | tail -1
  else
    env $env_part "$RSCRIPT" "$script" 2>&1 | { grep -v Fontconfig || true; } | tail -1
  fi
done
