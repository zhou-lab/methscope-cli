#!/bin/bash
## The mrmp commands beyond the default build: mrmp-pool (the step that
## SELECTS across sets), mrmp-export's yame-facing forms, mrmp-build's
## calling and imputation knobs, and resolver calibration with its cache.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_ref "$d"; ms_ref_bank "$d"; cd "$d"
"$MS" mrmp-build ref.cg flat.mrmp >/dev/null 2>&1
"$MS" mrmp-build --name other --top 1 --force ref.cg top1.mrmp >/dev/null 2>&1
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all \
      --force bankref.cg bank.mrmp >/dev/null 2>&1

## ---- 1. mrmp-pool: sets compete for a budget, losers are dropped ----------
"$MS" mrmp-pool --pooled-top 4 --min-cpgs 2 -o pool.mrmp flat.mrmp bank.mrmp > pool.log 2>&1
grep -q "budget 4, floor 2 CpGs" pool.log || { echo "pool did not report its cut:"; cat pool.log; exit 1; }
grep -q "won no pattern and were dropped" pool.log || { echo "no set lost under a 4-pattern budget:"; cat pool.log; exit 1; }
"$MS" inspect pool.mrmp > pool.txt 2>&1
grep -qE "patterns +4( |$)" pool.txt || { echo "the pooled artifact does not hold exactly 4 patterns:"; cat pool.txt; exit 1; }
## budget 0 is a plain concatenation plus the row check
"$MS" mrmp-pool --pooled-top 0 --include-all-0 --include-all-1 -o cat.mrmp flat.mrmp top1.mrmp >/dev/null 2>&1
"$MS" inspect cat.mrmp > cat.txt 2>&1
grep -q "chain of 2 sets" cat.txt || { echo "budget 0 did not keep both sets"; exit 1; }
if "$MS" mrmp-pool -o x.mrmp >/dev/null 2>&1; then echo "mrmp-pool with no input exited 0"; exit 1; fi

## ---- 2. mrmp-export: --set, --top, the TSV forms, --pna-label, a chain ---
"$MS" mrmp-export --set root bank.mrmp one.cm >/dev/null 2>&1
[ "$("$YAME" info one.cm 2>/dev/null | awk -F'\t' 'NR==2 {print $2}')" = 1 ] ||
  { echo "--set root did not export one record"; exit 1; }
"$MS" mrmp-export --top 1 --pna-label BG flat.mrmp t1.cm >/dev/null 2>&1
"$YAME" info t1.cm 2>/dev/null | grep -q "P1,BG" || { echo "--top 1 --pna-label BG keys:"; "$YAME" info t1.cm; exit 1; }
"$MS" mrmp-export --patterns - flat.mrmp 2>/dev/null > pat.tsv
grep -q $'^111000\tP1\t' pat.tsv || { echo "--patterns - shape:"; head -2 pat.tsv; exit 1; }
"$MS" mrmp-export --counts - flat.mrmp 2>/dev/null > cnt.tsv
grep -q $'^229\t111000$' cnt.tsv || { echo "--counts - shape:"; head -2 cnt.tsv; exit 1; }
grep -q "222222" cnt.tsv || { echo "--counts must include the PNA row"; exit 1; }
"$MS" mrmp-export bank.mrmp chain.cm > chain.log 2>&1
grep -q "18 sets -> chain.cm (+ chain.cm.idx)" chain.log || { echo "chain export:"; cat chain.log; exit 1; }
[ -s chain.cm.idx ] || { echo "chain export wrote no .idx"; exit 1; }
if "$MS" mrmp-export --set nope bank.mrmp x.cm >/dev/null 2>&1; then echo "--set nope exited 0"; exit 1; fi

## ---- 3. mrmp-build calling/imputation knobs each run and keep 2 patterns --
for o in "--shrink-pseudocnt 1" "--feature-mindepth 2 --max-lowdepth-frac 0.5" \
         "--impute-ambiguous random --impute-seed 3" \
         "--impute-ambiguous majority --min-major-fold 2" "--max-ambig-frac 0.5" \
         "--delta-mean-top 5" "--include-all-0 --include-all-1" \
         "--beta-threshold 0.6 --call-mindepth 2" "--call-band 0.45,0.55"; do
  "$MS" mrmp-build $o --force ref.cg v.mrmp > v.log 2>&1 || { echo "mrmp-build $o failed:"; cat v.log; exit 1; }
  "$MS" inspect v.mrmp > v.txt 2>&1
  grep -qE "patterns +2( |$)" v.txt || { echo "mrmp-build $o did not find the 2 patterns"; exit 1; }
done
## a band nothing survives is an error, not an empty artifact
if "$MS" mrmp-build --call-band 0.1,0.9 --force ref.cg cb.mrmp > cb.log 2>&1; then
  echo "an empty selection exited 0"; exit 1
fi
grep -q "no CpG survived" cb.log || { echo "wrong refusal:"; cat cb.log; exit 1; }

## ---- 4. calibration: LOO resolvers from a cell store, cached and strided --
printf 'A1\tA1\nA2\tA2\nA3\tA3\nB1\tB1\nB2\tB2\nB3\tB3\n' > cl.tsv
mkdir -p cache
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all \
      --cell-store bankref.cg --cell-labels cl.tsv --calib-threads 2 \
      --resolver-cache cache --resolver-stride 1/2 --force bankref.cg cal.mrmp > cal.log 2>&1
grep -q "out-of-stride pair(s) skipped -- this artifact is PARTIAL" cal.log ||
  { echo "a strided build did not warn it is partial:"; cat cal.log; exit 1; }
n1=$(sed -n 's/^ *\([0-9]*\) resolver pair(s) built.*/\1/p' cal.log)
[ "$(ls cache/*.mrmp | wc -l)" = "$n1" ] || { echo "cache holds $(ls cache | wc -l) blocks for $n1 built"; exit 1; }
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all \
      --cell-store bankref.cg --cell-labels cl.tsv --resolver-cache cache \
      --force bankref.cg cal2.mrmp > cal2.log 2>&1
grep -q "15 resolver pair(s) built ($n1 from cache)" cal2.log ||
  { echo "the second pass did not reuse the $n1 cached blocks:"; cat cal2.log; exit 1; }
"$MS" inspect cal2.mrmp > cal2.txt 2>&1
grep -q "chain of 18 sets" cal2.txt || { echo "calibrated bank is not 18 sets"; exit 1; }
## ---- 5. real LOO calibration: it needs 20 cells behind a pair ------------
## One cell per class (above) is under ms_pair_calibrate()'s floor of 20, so
## the resolvers were built but never calibrated. Ten noisy replicates per
## class -- the bank classes with 15% of sites flipped -- put 20 cells behind
## every pair, and the leave-one-out grid runs for real.
: > cells.tsv
for c in 1 2 3 4 5 6; do for r in 1 2 3 4 5 6 7 8 9 10; do
  awk -v c="$c" -v r="$r" 'BEGIN { srand(c * 100 + r); for (i = 0; i < 400; i++) {
      g = (c <= 3); f = (c == 1 || c == 4); h = (c == 2 || c == 5)
      if (i % 7 < 4)       m = (g ? 9 : 1)
      else if (i % 11 < 3) m = (f ? 9 : 1)
      else if (i % 13 < 2) m = (h ? 9 : 1)
      else                 m = 5
      if (rand() < 0.15) m = 10 - m
      print m "\t" (10 - m) } }' > cell.txt
  "$YAME" pack -f m cell.txt > "c${c}_${r}.cg" 2>/dev/null
  printf 'c%s_%s\t%s\n' "$c" "$r" "$(sed -n "${c}p" banknames.txt)" >> cells.tsv
done; done
cat c[1-6]_*.cg > cells.cg
cut -f1 cells.tsv > cellnames.txt
"$YAME" index -s cellnames.txt cells.cg 2>/dev/null
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all \
      --cell-store cells.cg --cell-labels cells.tsv --calib-threads 2 --calib-eps 0.01 \
      --force bankref.cg loo.mrmp > loo.log 2>&1 || { echo "calibrated build failed:"; cat loo.log; exit 1; }
## the group-split pairs have CpGs to spare and calibrate; the within-group
## pairs sit on 68 CpGs, under the grid's 100-CpG admission, and say so.
## Every pair takes one branch or the other.
n_cal=$(grep -c "= calibrated root@.*LOO macro .* over 20 cells" loo.log || true)
n_skip=$(grep -c "calibrate: no grid cell admits >= 100 CpGs; skipping" loo.log || true)
[ "$n_cal" -ge 1 ] && [ "$n_skip" -ge 1 ] && [ $((n_cal + n_skip)) = 15 ] ||
  { echo "15 pairs should split between calibrated ($n_cal) and skipped ($n_skip):"; grep -i calib loo.log; exit 1; }
grep -q "\[full grid\]" loo.log || { echo "no pair reported the full grid:"; grep calibrated loo.log; exit 1; }
"$MS" inspect loo.mrmp > loo.txt 2>&1
grep -q "chain of 18 sets" loo.txt || { echo "the calibrated bank is not 18 sets"; exit 1; }
## a labels file naming a cell the store lacks is an error, not a silent skip
printf 'ghost\tA1\n' >> cells.tsv
if "$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all \
     --cell-store cells.cg --cell-labels cells.tsv --force bankref.cg x.mrmp > ghost.log 2>&1; then
  grep -qi "ghost\|missing\|not in" ghost.log || { echo "a label for a missing cell passed silently"; exit 1; }
fi
echo "ok: pool budget, export forms, build knobs, calibration cache and LOO grid"
