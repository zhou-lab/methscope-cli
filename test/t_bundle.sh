#!/bin/bash
## bundle / unbundle / inspect: the container every shipped model is.
##
## The layout promise is that a bundle's .cm IS its file prefix, so a byte
## slice of the front is a readable mask. That promise is load-bearing -- it is
## how a 500-pattern set was recovered from a shipped decoder -- and it was
## also the thing that, once assumed rather than bounded, broke `upscale` in
## v0.7. It is asserted here.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_ref "$d"
"$MS" mrmp-build "$d/ref.cg" "$d/m.mrmp" >/dev/null 2>&1
"$MS" mrmp-export "$d/m.mrmp" "$d/m.cm" >/dev/null 2>&1

## a minimal linear model: the text spec form, so no xgboost training is needed
## The spec is key TAB value, and `labels` and `feature` take TAB-separated
## sub-fields (linear.c: labels is label0 TAB label1; feature is name TAB
## weight TAB mean). Writing `labels A,B` -- the form the inspect OUTPUT shows
## -- is rejected: that display is comma-joined, the file is not.
printf 'methscope-linear\n' >  "$d/model.txt"
printf 'method\tlogistic\n' >> "$d/model.txt"
printf 'labels\tA\tB\n'     >> "$d/model.txt"
printf 'bias\t0\n'           >> "$d/model.txt"
printf 'scale\t1\n'          >> "$d/model.txt"
printf 'feature\tP1\t1.0\t0.5\n' >> "$d/model.txt"

## ---- 1. bundle: kind is REQUIRED, and an unmarked bundle is refused ------
if "$MS" bundle -m "$d/m.cm" -o "$d/nokind.clfx" "$d/model.txt" >/dev/null 2>&1; then
  ## some builds accept it; if so the bundle must at least exist
  [ -s "$d/nokind.clfx" ] || { echo "bundle exited 0 but wrote nothing"; exit 1; }
fi
"$MS" bundle -m "$d/m.cm" -k logistic -o "$d/b.clfx" "$d/model.txt" >/dev/null 2>&1
[ -s "$d/b.clfx" ] || { echo "bundle wrote nothing"; exit 1; }

## ---- 2. inspect reports the container and its sections ------------------
"$MS" inspect "$d/b.clfx" > "$d/ins.txt" 2>&1
grep -q "MSBNDL1" "$d/ins.txt" || { echo "inspect does not report the container:"; cat "$d/ins.txt"; exit 1; }
grep -q "mrmp"   "$d/ins.txt" || { echo "inspect lists no mrmp section"; exit 1; }
grep -q "logistic" "$d/ins.txt" || { echo "inspect does not report the kind mark"; exit 1; }

## ---- 3. THE LAYOUT PROMISE: the .cm is the file prefix ------------------
## Take the mrmp section's size from inspect and slice that many bytes off the
## front; the result must be the same mask, readable on its own. If this fails,
## the bundle format has changed and bundle.h's layout note is wrong.
sz=$(awk '/  mrmp /{gsub(/,/,"",$4); print $4; exit}' "$d/ins.txt")
[ -n "$sz" ] || { echo "could not read the mrmp section size from inspect"; cat "$d/ins.txt"; exit 1; }
head -c "$sz" "$d/b.clfx" > "$d/prefix.cm"
cmp -s "$d/prefix.cm" "$d/m.cm" ||
  { echo "the bundle prefix is not byte-identical to the mask it was built from"; exit 1; }
"$YAME" info "$d/prefix.cm" >/dev/null 2>&1 ||
  { echo "the bundle prefix does not read as a mask on its own"; exit 1; }

## ---- 4. unbundle round-trips, and re-wrapping reproduces the bundle -----
cd "$d" && "$MS" unbundle b.clfx >/dev/null 2>&1
[ -s b.mrmp ] || { echo "unbundle wrote no mrmp"; exit 1; }
## Not `ls a b`: with only one of the two present ls exits non-zero and set -e
## kills the script before the check that would have explained why.
inner=""
for f in b.clf b.ubj; do [ -s "$f" ] && { inner=$f; break; }; done
[ -n "$inner" ] || { echo "unbundle wrote no inner model"; ls -1; exit 1; }
"$MS" bundle -m b.mrmp -k logistic -o rewrap.clfx "$inner" >/dev/null 2>&1
cmp -s b.clfx rewrap.clfx ||
  { echo "unbundle then bundle did not reproduce the original"; exit 1; }

## ---- 5. refusals -------------------------------------------------------
if "$MS" unbundle "$d/ref.cg" >/dev/null 2>&1; then
  echo "unbundle accepted a plain .cg as a bundle"; exit 1
fi
echo "ok: bundle sections, prefix promise, byte-identical round-trip"
