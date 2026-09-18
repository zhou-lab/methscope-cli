#!/bin/bash
## mrmp-build and mrmp-export on a tiny reference: the feature foundation every
## other subcommand reads. Asserts the SHAPE the build claims, not just that it
## exits 0 -- a flat build over a separable reference must find the separation.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_ref "$d"

## ---- 1. flat: one set over all classes ------------------------------------
"$MS" mrmp-build "$d/ref.cg" "$d/flat.mrmp" > "$d/build.log" 2>&1
"$MS" inspect "$d/flat.mrmp" > "$d/flat.txt" 2>&1
grep -q "one set" "$d/flat.txt" || { echo "the default mode did not produce one set:"; cat "$d/flat.txt"; exit 1; }
## the six cells are two groups of three, so the build must see 6 classes
grep -qE "classes +6" "$d/flat.txt" || { echo "expected 6 classes:"; cat "$d/flat.txt"; exit 1; }
## and the reference it names must be the store it was built from
grep -q "ref.cg" "$d/flat.txt" || { echo "inspect does not name the reference store"; exit 1; }

## ---- 2. the fixture is separable, so patterns must be found ---------------
pat=$(sed -n 's/^ *patterns *\([0-9]*\).*/\1/p' "$d/flat.txt" | head -1)
[ -n "$pat" ] && [ "$pat" -ge 2 ] ||
  { echo "a separable 2-group reference yielded $pat patterns"; cat "$d/flat.txt"; exit 1; }

## ---- 3. a bank build: the annealed split finds the two-level hierarchy of
## ms_ref_bank and emits its binstring blocks plus one resolver per class
## pair (C(6,2) = 15); --resolvers N keeps the N thinnest pairs by hard-block
## footprint, so 3 blocks + 5 resolvers = 8 sets exactly. A build with
## neither mode refuses: the routing tree is retired. ------------------------
ms_ref_bank "$d"
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all --force \
  "$d/bankref.cg" "$d/bank.mrmp" > "$d/bank.log" 2>&1 ||
  { echo "bank build failed:"; cat "$d/bank.log"; exit 1; }
"$MS" inspect "$d/bank.mrmp" > "$d/bank.txt" 2>&1
grep -q "MRMPIDX1" "$d/bank.txt" || { echo "bank build is not an MRMPIDX1 artifact"; exit 1; }
nset=$(sed -n 's/.*chain of \([0-9]*\) sets.*/\1/p' "$d/bank.txt" | head -1)
[ "$nset" = "18" ] || { echo "bank-full on 6 classes should be 3 blocks + 15 resolvers = 18 sets, got '$nset'"; cat "$d/bank.txt"; exit 1; }
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers 5 --force \
  "$d/bankref.cg" "$d/lite.mrmp" > "$d/lite.log" 2>&1
nlite=$("$MS" inspect "$d/lite.mrmp" 2>&1 | sed -n 's/.*chain of \([0-9]*\) sets.*/\1/p' | head -1)
[ "$nlite" = "8" ] || { echo "bank --resolvers 5 on 6 classes should be 3 blocks + 5 resolvers = 8 sets, got '$nlite'"; cat "$d/lite.log"; exit 1; }
grep -q "bank resolvers: the 5 of 15 pairs" "$d/lite.log" || { echo "the build log does not report the footprint pick"; cat "$d/lite.log"; exit 1; }
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --force \
  "$d/bankref.cg" "$d/min.mrmp" > "$d/min.log" 2>&1
nmin=$("$MS" inspect "$d/min.mrmp" 2>&1 | sed -n 's/.*chain of \([0-9]*\) sets.*/\1/p' | head -1)
[ "$nmin" = "3" ] || { echo "the default (--resolvers 0, hard blocks only) should be the 3 blocks, got '$nmin'"; cat "$d/min.log"; exit 1; }
if "$MS" mrmp-build --bank --anneal-min-seg 100,5 --resolver-gate 1 --force \
  "$d/bankref.cg" "$d/gate.mrmp" >/dev/null 2>&1; then
  echo "--resolver-gate is retired but was accepted"; exit 1
fi
if "$MS" mrmp-build --anneal-min-seg 100,5 "$d/bankref.cg" "$d/none.mrmp" >/dev/null 2>&1; then
  echo "mrmp-build with --anneal-min-seg but no --bank exited 0"; exit 1
fi

## ---- 4. mrmp-export emits a runtime mask over the same row space ---------
"$MS" mrmp-export "$d/flat.mrmp" "$d/flat.cm" >/dev/null 2>&1
[ -s "$d/flat.cm" ] || { echo "mrmp-export wrote nothing"; exit 1; }
rows=$("$YAME" info "$d/flat.cm" 2>/dev/null | awk -F'\t' 'NR==2 {print $4}')
[ "$rows" = "400" ] || { echo "exported mask has $rows rows, the reference has 400"; exit 1; }
## the mask is a categorical fmt2 -- that is what makes it a runtime mask
fmt=$("$YAME" info "$d/flat.cm" 2>/dev/null | awk -F'\t' 'NR==2 {print $5}')
[ "$fmt" = "2" ] || { echo "exported mask is format $fmt, expected 2"; exit 1; }

## ---- 5. refusals: exit 0 with nothing built is never right ---------------
if "$MS" mrmp-build "$d/nope.cg" "$d/x.mrmp" >/dev/null 2>&1; then
  echo "mrmp-build on a missing store exited 0"; exit 1
fi
[ ! -f "$d/x.mrmp" ] || { echo "mrmp-build left an output after failing"; exit 1; }
if "$MS" mrmp-export "$d/nope.mrmp" "$d/y.cm" >/dev/null 2>&1; then
  echo "mrmp-export on a missing artifact exited 0"; exit 1
fi
## ---- 6. mrmp-summary IS the MRMP average -----------------------------
## Our featurizer computes per-pattern means itself rather than shelling out to
## yame (yame's kernel takes one mask per call, so a chain would cost one genome
## scan per set). That makes the two implementations independent, and this pins
## them together: the same means over the exported mask must match `yame summary
## -m`. Tolerance is 1e-3 because yame prints Beta at three decimals.
## no --with-pna: the Pna background is state 0 in the exported mask and
## `yame summary -m` does not report it, so asking for it would be an
## unmatchable row rather than a comparison.
"$MS" mrmp-summary "$d/ref.cg" "$d/flat.mrmp" > "$d/ms.tsv" 2>/dev/null
"$YAME" summary -m "$d/flat.cm" "$d/ref.cg" > "$d/yame.tsv" 2>/dev/null
awk -F'\t' '
  NR==FNR { if (FNR>1) y[$2 "\t" $4] = $10; next }
  FNR==1  { next }
  { k = $1 "\t" $3
    if (!(k in y)) { miss++; next }
    n++; dd = $4 - y[k]; if (dd < 0) dd = -dd
    if (dd > m) { m = dd; worst = k } }
  END {
    if (miss) { printf "%d sample/pattern rows have no yame counterpart\n", miss; exit 1 }
    if (n != 12) { printf "expected 6 cells x 2 patterns = 12 pairs, compared %d\n", n; exit 1 }
    if (m > 1e-3) { printf "mrmp-summary and yame summary -m disagree by %.5f at %s\n", m, worst; exit 1 }
    printf "  %d sample/pattern means match yame summary -m (max |diff| %.5f)\n", n, m }' \
  "$d/yame.tsv" "$d/ms.tsv" || exit 1

## the wide form is the same numbers transposed, so its header must name every
## set/pattern column as <set>.<pattern> and carry one row per record
"$MS" mrmp-summary --wide "$d/ref.cg" "$d/flat.mrmp" > "$d/wide.tsv" 2>/dev/null
head -1 "$d/wide.tsv" | grep -q $'\troot.P1\t' ||
  { echo "--wide header does not name columns <set>.<pattern>:"; head -1 "$d/wide.tsv"; exit 1; }
[ "$(wc -l < "$d/wide.tsv")" = "7" ] ||
  { echo "--wide should be a header plus 6 records, got $(wc -l < "$d/wide.tsv") lines"; exit 1; }

echo "ok: flat and tree builds, export shape, mrmp-summary vs yame, refusals"
