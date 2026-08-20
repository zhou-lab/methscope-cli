#!/usr/bin/env bash
# End-to-end parity test: methscope (C) vs MethScope (R) on the mouse example.
#
# Requires: a conda env with libxgboost (XGB_PREFIX / CONDA_PREFIX), and a
# working R with the MethScope light deps (magrittr,dplyr,tidyr,stringr,
# data.table,xgboost,FNN). Set MS_REPO to the MethScope checkout.
set -euo pipefail
cd "$(dirname "$0")/.."

MS_REPO="${MS_REPO:-../MethScope}"
XGB_PREFIX="${XGB_PREFIX:-${CONDA_PREFIX:-}}"
OUT="${OUT_DIR:-/tmp/ms2_parity}"
mkdir -p "$OUT"

echo "== build =="
make XGB_PREFIX="$XGB_PREFIX" >/dev/null

MODEL=models/Liu2021_MouseBrain_P1000.msm
if [ ! -f "$MODEL" ]; then
  echo "== repackage model =="
  Rscript tools/repackage_models.R "$MS_REPO" ./methscope ./models
fi

Q="$MS_REPO/inst/extdata/example.cg"
echo "== methscope classify =="
./methscope classify --probs -o "$OUT/c_pred.tsv" "$Q" "$MODEL"
# The feature-matrix leg is gone: `predict` was renamed to `classify` in
# 7716c50 and this script was never updated, so it had already been failing
# here; `matrix` has since been removed outright because the .refx it wrote
# had no reader left. Nothing in the C tree emits that TSV now, so the R
# comparison below covers the probabilities only.

echo "== R reference =="
MS_REPO="$MS_REPO" OUT_DIR="$OUT" Rscript test/r_reference.R

echo "== compare =="
OUT_DIR="$OUT" Rscript test/compare.R
