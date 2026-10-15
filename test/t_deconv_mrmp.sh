#!/bin/bash
## deconv on an MRMP EMBEDDING (.msfm) instead of a .cg. Three classes over
## 600 CpGs, each high on its own block; eight mixtures of known proportions
## are featurized against an MRMP set built on the same reference and handed
## to deconv. Asserts the fractions come back, that --threads does not change
## a byte, that a count-free embedding still works, and the refusals.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT

## ---- fixture: 3 classes, 20 reads per CpG -------------------------------
for c in A B C; do
  awk -v c="$c" 'BEGIN { for (i = 0; i < 600; i++) {
    if      (i % 6 == 0) m = (c == "A") ? 18 : 2
    else if (i % 6 == 1) m = (c == "B") ? 18 : 2
    else if (i % 6 == 2) m = (c == "C") ? 18 : 2
    else if (i % 6 == 3) m = (c == "C") ? 2 : 18
    else                 m = 10
    print m "\t" 20 - m } }' > "$d/$c.txt"
  "$YAME" pack -f m "$d/$c.txt" > "$d/$c.cg" 2>/dev/null
done
cat "$d/A.cg" "$d/B.cg" "$d/C.cg" > "$d/r3.cg"
printf 'A\nB\nC\n' > "$d/n3.txt"; "$YAME" index -s "$d/n3.txt" "$d/r3.cg" 2>/dev/null
"$MS" deconv-build-ref -o "$d/r3.msdref" "$d/r3.cg" > "$d/ref.log" 2>&1 ||
  { echo "reference:"; cat "$d/ref.log"; exit 1; }
"$MS" mrmp-build --top 0 "$d/r3.cg" "$d/r3.mrmp" > "$d/mrmp.log" 2>&1 ||
  { echo "mrmp-build:"; cat "$d/mrmp.log"; exit 1; }

## eight mixtures, weights in tenths; record name carries the truth
: > "$d/mix.cg"; : > "$d/mn.txt"
for w in 6,3,1 1,8,1 2,2,6 5,5,0 0,1,9 3,3,4 7,0,3 4,4,2; do
  IFS=, read -r a b c <<< "$w"
  paste "$d/A.txt" "$d/B.txt" "$d/C.txt" | awk -F'\t' -v a="$a" -v b="$b" \
    -v c="$c" '{ print a*$1 + b*$3 + c*$5 "\t" a*$2 + b*$4 + c*$6 }' \
    > "$d/m.txt"
  "$YAME" pack -f m "$d/m.txt" >> "$d/mix.cg" 2>/dev/null
  echo "m$a$b$c" >> "$d/mn.txt"
done
"$YAME" index -s "$d/mn.txt" "$d/mix.cg" 2>/dev/null

## ---- 1. the embedding, with counts --------------------------------------
"$MS" classify-featurize --continuous-features --with-n -o "$d/mix.msfm" \
  "$d/mix.cg" "$d/r3.mrmp" > "$d/feat.log" 2>&1 ||
  { echo "classify-featurize --with-n:"; cat "$d/feat.log"; exit 1; }
"$MS" inspect "$d/mix.msfm" 2>/dev/null | grep -q "per-pattern CpG counts" ||
  { echo "inspect does not report the count block"; exit 1; }

## ---- 2. fractions within 0.05 of the truth ------------------------------
"$MS" deconv --wide "$d/r3.msdref" "$d/mix.msfm" 2>/dev/null > "$d/w1.tsv"
check() {   ## $1 = wide output; truth from the record name m<A><B><C>
  awk -F'\t' 'NR == 1 { for (j = 2; j <= NF; j++) col[$j] = j; next }
    { n = $1
      t["A"] = substr(n, 2, 1) / 10; t["B"] = substr(n, 3, 1) / 10
      t["C"] = substr(n, 4, 1) / 10
      for (k in t) { e = $(col[k]) - t[k]; if (e < 0) e = -e
        if (e > 0.05) { print $1, k, $(col[k]), "want", t[k]; bad++ } } }
    END { if (NR != 9) { print "rows:", NR; bad++ }; exit (bad + 0) > 0 }' "$1"
}
check "$d/w1.tsv" || { echo "embedding deconv misses the truth:"; cat "$d/w1.tsv"; exit 1; }

## ---- 3. --threads changes nothing ---------------------------------------
"$MS" deconv --wide --threads 4 "$d/r3.msdref" "$d/mix.msfm" 2>/dev/null > "$d/w4.tsv"
cmp -s "$d/w1.tsv" "$d/w4.tsv" ||
  { echo "--threads 4 differs from 1:"; diff "$d/w1.tsv" "$d/w4.tsv"; exit 1; }

## ---- 4. no counts: equal weights, still right ---------------------------
"$MS" classify-featurize --continuous-features -o "$d/nn.msfm" \
  "$d/mix.cg" "$d/r3.mrmp" > /dev/null 2>&1
"$MS" deconv --wide "$d/r3.msdref" "$d/nn.msfm" 2> "$d/nn.err" > "$d/nn.tsv"
grep -q "no per-pattern CpG counts" "$d/nn.err" ||
  { echo "no notice that the embedding lacks counts:"; cat "$d/nn.err"; exit 1; }
check "$d/nn.tsv" || { echo "count-free embedding misses the truth:"; cat "$d/nn.tsv"; exit 1; }

## ---- 5. refusals --------------------------------------------------------
if "$MS" deconv --panel-out "$d/p.tsv" "$d/r3.msdref" "$d/mix.msfm" >/dev/null 2>&1; then
  echo "deconv accepted --panel-out for an .msfm query"; exit 1
fi
"$MS" classify-featurize -o "$d/bin.msfm" "$d/mix.cg" "$d/r3.mrmp" >/dev/null 2>&1
if "$MS" deconv "$d/r3.msdref" "$d/bin.msfm" >/dev/null 2>&1; then
  echo "deconv accepted a binarised .msfm"; exit 1
fi
## a reference over a different row space (500 CpGs, not 600)
for c in A B C; do head -500 "$d/$c.txt" | "$YAME" pack -f m - > "$d/s$c.cg" 2>/dev/null; done
cat "$d/sA.cg" "$d/sB.cg" "$d/sC.cg" > "$d/s3.cg"
"$YAME" index -s "$d/n3.txt" "$d/s3.cg" 2>/dev/null
"$MS" deconv-build-ref -o "$d/s3.msdref" "$d/s3.cg" > /dev/null 2>&1
if "$MS" deconv "$d/s3.msdref" "$d/mix.msfm" >/dev/null 2>&1; then
  echo "deconv accepted an embedding from another row space"; exit 1
fi
echo "ok"
