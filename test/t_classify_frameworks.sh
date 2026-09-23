#!/bin/bash
## The two classify paths the tree suite never enters: the violation rule
## (unfitted, scored straight off a .mrmp) and the logistic framework (the
## one fitted model that still featurizes a raw .cg itself), plus the -p trim
## that only the logistic trainer can reach.
##
## Both are asserted on the labels they recover, not on exit codes. The six
## cells are the same two classes with 15% of sites flipped, so a rule that
## cannot separate them has learned nothing.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_train "$d"
cd "$d"
"$MS" mrmp-build ref.cg m.mrmp > build.log 2>&1
"$MS" mrmp-export m.mrmp m.cm > /dev/null 2>&1
"$MS" classify-featurize -l labels.txt -o f.msfm train.cg m.mrmp > /dev/null 2>&1

## labels_ok <pred.tsv> -- column 2 of the body must equal labels.txt
labels_ok () {
  paste <(tail -n +2 "$1" | cut -f2) labels.txt |
    awk -F'\t' '$1 != $2 { bad++ } END { exit (bad + 0) > 0 }'
}

## ---- 1. violation: the fixture has 2 patterns, the default floor is 20 ----
"$MS" classify --framework violation m.mrmp train.cg 2>/dev/null > v20.tsv
head -1 v20.tsv | grep -q "margin" ||
  { echo "violation header does not say margin (it has no posterior):"; head -1 v20.tsv; exit 1; }
[ "$(tail -n +2 v20.tsv | cut -f2 | sort -u)" = "NA" ] ||
  { echo "below --min-patterns every call must be NA, not guessed:"; cat v20.tsv; exit 1; }

## with the floor lowered every cell comes back as its class, with --probs
## one column per class, and --threads takes the indexed path
"$MS" classify --framework violation --min-patterns 1 --probs --threads 2 \
      m.mrmp train.cg 2>/dev/null > v1.tsv
[ "$(head -1 v1.tsv)" = "$(printf 'cell\tprediction_label\tmargin\tA\tB')" ] ||
  { echo "violation --probs header:"; head -1 v1.tsv; exit 1; }
[ "$(tail -n +2 v1.tsv | wc -l)" = 6 ] || { echo "expected 6 rows:"; cat v1.tsv; exit 1; }
labels_ok v1.tsv || { echo "the violation rule mislabels the fixture:"; cat v1.tsv; exit 1; }

## every weighting must agree on a fixture this clean
for w in log1p linear flat; do
  "$MS" classify --framework violation --min-patterns 1 --pattern-weight $w \
        m.mrmp train.cg 2>/dev/null > w.tsv
  labels_ok w.tsv || { echo "--pattern-weight $w mislabels:"; cat w.tsv; exit 1; }
done

## prebuilt features: one positional, the artifact
"$MS" classify --framework violation --min-patterns 1 --data f.msfm m.mrmp \
      2>/dev/null > vd.tsv
labels_ok vd.tsv || { echo "violation --data mislabels:"; cat vd.tsv; exit 1; }

## an exported .cm has no binstrings to read the rule from
if "$MS" classify --framework violation m.cm train.cg >/dev/null 2>&1; then
  echo "violation accepted a .cm"; exit 1
fi
## and there is nothing to train: the trainer points at classify
if "$MS" classify-train --framework violation -o v.clfx m.mrmp > vt.log 2>&1; then
  echo "classify-train --framework violation exited 0"; exit 1
fi
grep -q "classify --framework violation" vt.log ||
  { echo "the refusal does not say where the rule lives:"; cat vt.log; exit 1; }

## ---- 2. logistic, trained from the .mrmp ARTIFACT ------------------------
## This is the bundle classify-train writes by default, and until 2026-09-23
## classify could not read it back ("not a readable CX stream"): the trainer
## bundled the .mrmp, the scorer expected a .cm at the bundle prefix.
"$MS" classify-train --framework logistic -l labels.txt -o lin.clfx \
      train.cg m.mrmp > lin.log 2>&1
grep -q "trained logistic model (2-class) on 6 cells x 2 feature" lin.log ||
  { echo "logistic training summary:"; cat lin.log; exit 1; }
"$MS" inspect lin.clfx > lin.txt 2>&1
grep -q '"logistic" framework mark' lin.txt ||
  { echo "the bundle carries no logistic kind mark"; exit 1; }

"$MS" classify --probs lin.clfx train.cg 2>/dev/null > lin.tsv
[ "$(head -1 lin.tsv)" = "$(printf 'cell\tprediction_label\tconfidence\tcertainty\tA\tB')" ] ||
  { echo "logistic --probs header:"; head -1 lin.tsv; exit 1; }
labels_ok lin.tsv || { echo "the logistic model mislabels its training cells:"; cat lin.tsv; exit 1; }
tail -n +2 lin.tsv | awk -F'\t' '$3 < 0.5 || $3 > 1 { bad++ } END { exit (bad + 0) > 0 }' ||
  { echo "confidence of the called class must be in [0.5,1]:"; cat lin.tsv; exit 1; }

## -o and --no-header
"$MS" classify --no-header -o out.tsv lin.clfx train.cg 2>/dev/null
[ "$(wc -l < out.tsv)" = 6 ] || { echo "--no-header -o wrote $(wc -l < out.tsv) lines"; exit 1; }

## the linear scorer featurizes its own query
if "$MS" classify --data f.msfm lin.clfx >/dev/null 2>&1; then
  echo "classify --data was accepted on a logistic bundle"; exit 1
fi

## --data at TRAIN time is fine: the features are the same matrix
"$MS" classify-train --framework logistic --data f.msfm -o lind.clfx m.mrmp \
      > /dev/null 2>&1
"$MS" classify lind.clfx train.cg 2>/dev/null > lind.tsv
labels_ok lind.tsv || { echo "logistic trained from .msfm mislabels:"; cat lind.tsv; exit 1; }

## ---- 3. -p: trim, and the two refusals around it --------------------------
## keeping 1 of 2 patterns on a .cm trims the bundled mask to that pattern
"$MS" classify-train --framework logistic -p 1 -l labels.txt -o t.clfx \
      train.cg m.cm > t.log 2>&1
grep -q "x 1 feature(s)" t.log && grep -q "trimmed mrmp" t.log ||
  { echo "-p 1 did not trim:"; cat t.log; exit 1; }
"$MS" classify t.clfx train.cg 2>/dev/null > t.tsv
[ "$(tail -n +2 t.tsv | wc -l)" = 6 ] || { echo "trimmed model scored $(tail -n +2 t.tsv | wc -l) rows"; exit 1; }
## one pattern still separates these two classes
labels_ok t.tsv || { echo "the one-pattern model mislabels:"; cat t.tsv; exit 1; }

## a .mrmp cannot be trimmed at train time (cut it with mrmp-pool instead)
if "$MS" classify-train --framework logistic -p 1 -l labels.txt -o x.clfx \
     train.cg m.mrmp > p.log 2>&1; then
  echo "-p 1 on a .mrmp exited 0"; exit 1
fi
grep -q "cannot trim a .mrmp" p.log || { echo "wrong refusal:"; cat p.log; exit 1; }
## and -p past the pattern count is an error, not a silent clamp
if "$MS" classify-train --framework logistic -p 5 -l labels.txt -o x.clfx \
     train.cg m.cm >/dev/null 2>&1; then
  echo "-p 5 of 2 exited 0"; exit 1
fi

## ---- 4. framework refusals ----------------------------------------------
printf 'A\nA\nB\nB\nC\nC\n' > l3.txt
if "$MS" classify-train --framework logistic -l l3.txt -o x.clfx train.cg m.cm \
     > l3.log 2>&1; then
  echo "logistic accepted 3 classes"; exit 1
fi
grep -q "binary" l3.log || { echo "3-class refusal:"; cat l3.log; exit 1; }
if "$MS" classify-train --framework nope -l labels.txt -o x.clfx train.cg m.cm \
     >/dev/null 2>&1; then
  echo "unknown framework exited 0"; exit 1
fi
## a .ubj out is refused: the logistic model needs its MRMP beside it
if "$MS" classify-train --framework logistic -l labels.txt -o x.ubj train.cg m.cm \
     >/dev/null 2>&1; then
  echo "logistic wrote a loose .ubj"; exit 1
fi
echo "ok: violation and logistic score the fixture; trim and refusals hold"
