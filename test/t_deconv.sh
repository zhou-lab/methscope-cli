#!/bin/bash
## deconv-build-ref and deconv on the same two-class fixture the classifier
## uses. Asserts the FRACTIONS, not just the exit code: a cell drawn from class
## A must come back mostly A, and every cell's fractions must sum to 1. A
## solver that returns a valid-looking table of wrong numbers exits 0.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_train "$d"

## ---- 1. build the reference ---------------------------------------------
"$MS" deconv-build-ref -o "$d/r.msdref" "$d/ref.cg" > "$d/ref.log" 2>&1
[ -s "$d/r.msdref" ] || { echo "deconv-build-ref wrote nothing"; exit 1; }
grep -q "2 classes" "$d/ref.log" ||
  { echo "reference did not see 2 classes:"; cat "$d/ref.log"; exit 1; }
"$MS" inspect "$d/r.msdref" >/dev/null 2>&1 ||
  { echo "inspect cannot read the reference it just built"; exit 1; }

## ---- 2. deconvolve the six cells ----------------------------------------
"$MS" deconv "$d/r.msdref" "$d/train.cg" 2>/dev/null > "$d/out.tsv"
head -1 "$d/out.tsv" | grep -q "fraction" ||
  { echo "no fraction column in the output:"; head -3 "$d/out.tsv"; exit 1; }
## long form: one row per (cell, class), so 6 cells x 2 classes
n=$(tail -n +2 "$d/out.tsv" | wc -l)
[ "$n" = "12" ] || { echo "expected 12 rows (6 cells x 2 classes), got $n"; exit 1; }

## ---- 3. every cell's fractions sum to 1 ---------------------------------
tail -n +2 "$d/out.tsv" | awk -F'\t' '
  { s[$1] += $3 }
  END { for (c in s) if (s[c] < 0.99 || s[c] > 1.01) { print c, s[c]; bad++ }
        exit (bad + 0) > 0 }' ||
  { echo "some cell's fractions do not sum to 1"; cat "$d/out.tsv"; exit 1; }

## ---- 4. each cell is called as the class it was drawn from --------------
## cells 1-3 are A, 4-6 are B. The dominant class must match, with a clear
## margin -- these are not subtle mixtures.
tail -n +2 "$d/out.tsv" | sort -k1,1 -k3,3gr | awk -F'\t' '
  !seen[$1]++ { want = ($1 == "c1" || $1 == "c2" || $1 == "c3") ? "A" : "B"
                if ($2 != want)  { print $1 " called " $2 " wanted " want; bad++ }
                if ($3 < 0.75)   { print $1 " dominant fraction only " $3;  bad++ } }
  END { exit (bad + 0) > 0 }' ||
  { echo "deconvolution does not recover the classes:"; cat "$d/out.tsv"; exit 1; }

## ---- 5. refusals --------------------------------------------------------
if "$MS" deconv "$d/ref.cg" "$d/train.cg" >/dev/null 2>&1; then
  echo "deconv accepted a plain .cg where a .msdref belongs"; exit 1
fi
if "$MS" deconv-build-ref -o "$d/x.msdref" "$d/nope.cg" >/dev/null 2>&1; then
  echo "deconv-build-ref on a missing store exited 0"; exit 1
fi
echo "ok: reference built, fractions sum to 1, classes recovered"
