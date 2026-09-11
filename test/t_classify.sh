#!/bin/bash
## The classifier chain end to end: mrmp-build -> classify-featurize ->
## classify-train -> classify, on fixtures small enough to pack inline.
##
## It asserts the calls are RIGHT, not merely that the commands exit 0. The six
## cells are drawn from the two classes with 15% of sites flipped, so a model
## that cannot recover their labels has learned nothing -- and a chain that
## silently trains on the wrong thing (the docs' Train tab did, for a year)
## would still exit 0 at every step.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_train "$d"

## ---- 1. the MRMP is built over CLASSES, and must see exactly two ----------
"$MS" mrmp-build --flat "$d/ref.cg" "$d/m.mrmp" > "$d/build.log" 2>&1
grep -q "2 classes" "$d/build.log" ||
  { echo "mrmp-build did not see 2 classes:"; cat "$d/build.log"; exit 1; }

## ---- 2. featurize: one row per cell, columns from the MRMP ---------------
"$MS" classify-featurize -l "$d/labels.txt" -o "$d/f.msfm" \
      "$d/train.cg" "$d/m.mrmp" > "$d/feat.log" 2>&1
grep -q "6 records" "$d/feat.log" ||
  { echo "featurize did not emit 6 records:"; cat "$d/feat.log"; exit 1; }
grep -q "2 classes" "$d/feat.log" ||
  { echo "featurize lost the class count:"; cat "$d/feat.log"; exit 1; }
[ -s "$d/f.msfm" ] || { echo "featurize wrote nothing"; exit 1; }

## the labels ride INSIDE the .msfm, which is what stops a later train being
## handed a mismatched label file -- inspect must be able to read them back
"$MS" inspect "$d/f.msfm" > "$d/msfm.txt" 2>&1
grep -qE "class|label" "$d/msfm.txt" ||
  { echo "inspect on a .msfm reports no class information:"; cat "$d/msfm.txt"; exit 1; }

## ---- 3. train ------------------------------------------------------------
"$MS" classify-train --data "$d/f.msfm" -o "$d/model.clfx" > "$d/train.log" 2>&1
[ -s "$d/model.clfx" ] || { echo "classify-train wrote no model"; exit 1; }
"$MS" inspect "$d/model.clfx" | grep -q "MSBNDL1" ||
  { echo "the trained model is not a bundle"; exit 1; }

## ---- 4. classify: the six training cells must come back as their labels ---
"$MS" classify "$d/model.clfx" "$d/train.cg" 2>/dev/null > "$d/pred.tsv"
grep -q "prediction_label" "$d/pred.tsv" || { echo "no header in the prediction table"; exit 1; }
n=$(tail -n +2 "$d/pred.tsv" | wc -l)
[ "$n" = "6" ] || { echo "expected 6 predictions, got $n"; cat "$d/pred.tsv"; exit 1; }

## cells 1-3 are class A, 4-6 are class B, in that order
paste <(tail -n +2 "$d/pred.tsv" | cut -f2) "$d/labels.txt" |
  awk -F'\t' '$1 != $2 { bad++ } END { exit (bad + 0) > 0 }' ||
  { echo "the model mislabels its own training cells:"; paste <(tail -n +2 "$d/pred.tsv" | cut -f1,2) "$d/labels.txt"; exit 1; }

## confidence must be a probability
tail -n +2 "$d/pred.tsv" | awk -F'\t' '$3 < 0 || $3 > 1 { bad++ } END { exit (bad + 0) > 0 }' ||
  { echo "confidence outside [0,1]"; cat "$d/pred.tsv"; exit 1; }

## ---- 5. refusals ---------------------------------------------------------
if "$MS" classify "$d/m.mrmp" "$d/train.cg" >/dev/null 2>&1; then
  echo "classify accepted a .mrmp where a bundle belongs"; exit 1
fi
if "$MS" classify-train --data "$d/nope.msfm" -o "$d/x.clfx" >/dev/null 2>&1; then
  echo "classify-train on a missing .msfm exited 0"; exit 1
fi
echo "ok: build -> featurize -> train -> classify, labels recovered"
