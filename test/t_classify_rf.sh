#!/bin/bash
## The random-forest framework end to end: classify-train --framework rf
## selects single rows inside the run, writes an "rf" bundle, and classify
## scores it -- asserted on the calls it makes on records it never saw.
##
## Fixture: 300 rows, the first 10 carrying the class (A near 0.8, B near
## 0.2), the rest noise, and the last 20 missing in every record so the
## missingness filter has something to drop. 24 training and 6 held-out
## records, half of each class.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
cd "$d"

## rec <seed> <class> -- one record as M<TAB>U text, 10 counts per row
rec () {
  awk -v s="$1" -v c="$2" 'BEGIN { srand(s); for (i = 0; i < 300; i++) {
      if (i >= 280) { print "0\t0"; continue }
      p = (i < 10) ? ((c == "A") ? 0.8 : 0.2) : rand()
      p += (rand() - 0.5) * 0.3; if (p < 0) p = 0; if (p > 1) p = 1
      m = int(p * 10 + 0.5); print m "\t" (10 - m) } }'
}
: > train.txt; : > names.txt; : > labels.txt
for k in $(seq 1 30); do
  c=$([ $((k % 2)) = 1 ] && echo A || echo B)
  rec $k $c > r$k.txt
  "$YAME" pack -f m r$k.txt > r$k.cg 2>/dev/null
done
for k in $(seq 1 24); do
  echo s$k >> names.txt; [ $((k % 2)) = 1 ] && echo A >> labels.txt || echo B >> labels.txt
done
cat $(for k in $(seq 1 24); do echo r$k.cg; done) > train.cg
"$YAME" index -s names.txt train.cg 2>/dev/null
cat $(for k in $(seq 25 30); do echo r$k.cg; done) > test.cg
printf 'q25\nq26\nq27\nq28\nq29\nq30\n' > qnames.txt
"$YAME" index -s qnames.txt test.cg 2>/dev/null

train () {   # train <out.clfx> <threads>
  "$MS" classify-train --framework rf --batch 100 --select 10 --ntree 100 \
        --seed 7 --threads "$2" -l labels.txt -o "$1" train.cg > "$1.log" 2>&1 ||
    { echo "classify-train --framework rf failed:"; cat "$1.log"; exit 1; }
}
train m1.clfx 1
train m3.clfx 3

## the filter drops exactly the 20 all-missing rows
grep -q "280 rows kept" m1.clfx.log ||
  { echo "the missingness filter did not keep 280 of 300 rows:"; cat m1.clfx.log; exit 1; }
## threads change nothing: the bundles are byte-identical
cmp -s m1.clfx m3.clfx || { echo "1 and 3 threads wrote different models"; exit 1; }

## held-out records come back as their class, with one column per class
"$MS" classify --probs -o pred.tsv m1.clfx test.cg 2>/dev/null
[ "$(head -1 pred.tsv)" = "$(printf 'cell\tprediction_label\tconfidence\tcertainty\tA\tB')" ] ||
  { echo "rf --probs header:"; head -1 pred.tsv; exit 1; }
[ "$(tail -n +2 pred.tsv | cut -f2 | tr '\n' ' ')" = "A B A B A B " ] ||
  { echo "the forest mislabels held-out records:"; cat pred.tsv; exit 1; }
## confidence is the winning vote fraction, and the two columns sum to 1
awk -F'\t' 'NR > 1 { if ($3 < 0.5 || ($5 + $6) < 0.999 || ($5 + $6) > 1.001) bad++ }
            END { exit (bad + 0) > 0 }' pred.tsv ||
  { echo "vote fractions are inconsistent:"; cat pred.tsv; exit 1; }

## inspect names the framework and the recipe
"$MS" inspect m1.clfx > insp.txt 2>&1
grep -q "random forest" insp.txt && grep -q "select 10 of 280 rows" insp.txt ||
  { echo "inspect does not describe the forest:"; cat insp.txt; exit 1; }

## refusals: --data is xgboost-only, and labels must match the records
if "$MS" classify --data x.msfm m1.clfx > o.txt 2>&1; then
  echo "classify --data accepted an rf bundle"; exit 1; fi
grep -q "xgboost-only" o.txt || { echo "--data refusal:"; cat o.txt; exit 1; }
head -5 labels.txt > short.txt
if "$MS" classify-train --framework rf -l short.txt -o x.clfx train.cg > o.txt 2>&1; then
  echo "a short label file was accepted"; exit 1; fi
grep -q "label count" o.txt || { echo "label-count refusal:"; cat o.txt; exit 1; }
exit 0
