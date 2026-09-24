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

## ---- 6. the rescue rebuild: a weak pair the band cannot see -------------
## Four classes over 400 CpGs. A and B each have their own block. C and D
## differ ONLY on a block where A sits at 0.35 -- inside the 0.30,0.70 no-call
## band -- so every CpG that separates C from D fails admission and the pair
## has zero segregating CpGs: weak. The rescue round (--rescue-below, default
## 2000) then re-admits those CpGs on the pair's own beta gap (0.8), with A
## judged against the wider rescue band instead. Every other row is 0.5 and
## carries nothing. A 70/30 C/D mixture must come back led by C with rescue
## on; with it off (--rescue-below 0) the pair is invisible and C cannot lead.
for c in A B C D; do
  awk -v c="$c" 'BEGIN { for (i = 0; i < 400; i++) {
    if (i % 7 < 4)       m = (c == "A") ? 9 : 1
    else if (i % 11 < 3) m = (c == "A") ? 3.5 : (c == "C") ? 9 : 1
    else if (i % 13 < 2) m = (c == "B") ? 9 : 1
    else                 m = 5
    print m * 2 "\t" (10 - m) * 2 } }' > "$d/$c.txt"
  "$YAME" pack -f m "$d/$c.txt" > "$d/$c.cg" 2>/dev/null
done
cat "$d/A.cg" "$d/B.cg" "$d/C.cg" "$d/D.cg" > "$d/r4.cg"
printf 'A\nB\nC\nD\n' > "$d/n4.txt"; "$YAME" index -s "$d/n4.txt" "$d/r4.cg" 2>/dev/null
paste "$d/C.txt" "$d/D.txt" | awk -F'\t' '{ print 7 * $1 + 3 * $3 "\t" 7 * $2 + 3 * $4 }' > "$d/mix.txt"
"$YAME" pack -f m "$d/mix.txt" > "$d/mix.cg" 2>/dev/null
printf 'cd73\n' > "$d/mn.txt"; "$YAME" index -s "$d/mn.txt" "$d/mix.cg" 2>/dev/null
"$MS" deconv-build-ref -o "$d/r4.msdref" "$d/r4.cg" > "$d/r4.log" 2>&1 ||
  { echo "4-class reference:"; cat "$d/r4.log"; exit 1; }
"$MS" deconv --rescue-min-depth 1 "$d/r4.msdref" "$d/mix.cg" 2>/dev/null > "$d/on.tsv"
"$MS" deconv --rescue-below 0     "$d/r4.msdref" "$d/mix.cg" 2>/dev/null > "$d/off.tsv"
top () { tail -n +2 "$1" | sort -t$'\t' -k3,3gr | head -1 | cut -f2,3; }
case "$(top "$d/on.tsv")" in
  C*) ;;
  *) echo "with rescue, the C/D mixture should be led by C, got: $(top "$d/on.tsv")"; cat "$d/on.tsv"; exit 1 ;;
esac
frac=$(top "$d/on.tsv" | cut -f2)
awk -v f="$frac" 'BEGIN { exit !(f >= 0.6) }' || { echo "with rescue, C leads by only $frac"; cat "$d/on.tsv"; exit 1; }
case "$(top "$d/off.tsv")" in
  C*) echo "without rescue the weak pair should be invisible, but C leads:"; cat "$d/off.tsv"; exit 1 ;;
esac
echo "ok: reference built, fractions sum to 1, classes recovered, rescue rebuild resolves a weak pair"
