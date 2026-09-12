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
"$MS" mrmp-build --flat "$d/ref.cg" "$d/flat.mrmp" > "$d/build.log" 2>&1
"$MS" inspect "$d/flat.mrmp" > "$d/flat.txt" 2>&1
grep -q "one set" "$d/flat.txt" || { echo "--flat did not produce one set:"; cat "$d/flat.txt"; exit 1; }
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
## pair (C(6,2) = 15); --resolver-gate at the floor keeps only the pairs no
## binstring separates. A build with neither mode refuses: the routing tree
## is retired. ---------------------------------------------------------------
ms_ref_bank "$d"
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --pattern-floor 1 --force \
  "$d/bankref.cg" "$d/bank.mrmp" > "$d/bank.log" 2>&1 ||
  { echo "bank build failed:"; cat "$d/bank.log"; exit 1; }
"$MS" inspect "$d/bank.mrmp" > "$d/bank.txt" 2>&1
grep -q "MRMPIDX1" "$d/bank.txt" || { echo "bank build is not an MRMPIDX1 artifact"; exit 1; }
nset=$(sed -n 's/.*chain of \([0-9]*\) sets.*/\1/p' "$d/bank.txt" | head -1)
[ "$nset" = "18" ] || { echo "bank-full on 6 classes should be 3 blocks + 15 resolvers = 18 sets, got '$nset'"; cat "$d/bank.txt"; exit 1; }
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --pattern-floor 1 --resolver-gate 1 --force \
  "$d/bankref.cg" "$d/lite.mrmp" >/dev/null 2>&1
nlite=$("$MS" inspect "$d/lite.mrmp" 2>&1 | sed -n 's/.*chain of \([0-9]*\) sets.*/\1/p' | head -1)
[ -n "$nlite" ] && [ "$nlite" -lt "$nset" ] || { echo "bank-lite ($nlite sets) is not smaller than bank-full ($nset)"; exit 1; }
if "$MS" mrmp-build "$d/bankref.cg" "$d/none.mrmp" >/dev/null 2>&1; then
  echo "mrmp-build without --bank or --flat exited 0"; exit 1
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
echo "ok: flat and tree builds, export shape, refusals"
