#!/bin/bash
## deconv's option surface on the two-class fixture: the v3 reference with a
## confusion trailer, the grouping it enables, the output shapes, the
## parallel path, the diagnostics and the dumps. t_deconv.sh proves the
## default answer is right; this proves every documented switch runs and
## changes what it claims to change, and nothing else.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_train "$d"; cd "$d"

## ---- 1. a v3 reference: --confusion embeds a trailer, --force overwrites --
printf '# held-out run\nA\tA\t0.9\nA\tB\t0.1\nB\tB\t0.95\nB\tA\t0.05\n' > conf.tsv
"$MS" deconv-build-ref --confusion conf.tsv --keep-all --call-mindepth 1 \
      --qfilter 0.3,0.7 --beta-threshold 0.5 -o r3.msdref ref.cg > r3.log 2>&1
"$MS" inspect r3.msdref > r3.txt 2>&1
grep -q "confusion" r3.txt || { echo "inspect shows no confusion trailer:"; cat r3.txt; exit 1; }
if "$MS" deconv-build-ref --confusion conf.tsv -o r3.msdref ref.cg > again.log 2>&1; then
  echo "deconv-build-ref overwrote an existing output without --force"; exit 1
fi
grep -q "use --force" again.log || { echo "wrong refusal:"; cat again.log; exit 1; }
"$MS" deconv-build-ref --force --confusion conf.tsv -o r3.msdref ref.cg >/dev/null 2>&1
## and a v2 one, for the refusal below
"$MS" deconv-build-ref -o r2.msdref ref.cg >/dev/null 2>&1

## ---- 2. output shapes -----------------------------------------------------
"$MS" deconv --threads 2 r3.msdref train.cg 2>/dev/null > tidy.tsv
[ "$(tail -n +2 tidy.tsv | wc -l)" = 12 ] || { echo "--threads 2 tidy rows:"; cat tidy.tsv; exit 1; }
"$MS" deconv r3.msdref train.cg 2>/dev/null > serial.tsv
cmp -s tidy.tsv serial.tsv || { echo "--threads 2 changed the answer"; diff tidy.tsv serial.tsv; exit 1; }

"$MS" deconv --wide r3.msdref train.cg 2>/dev/null > wide.tsv
[ "$(head -1 wide.tsv)" = "$(printf 'cell\tA\tB')" ] || { echo "--wide header:"; head -1 wide.tsv; exit 1; }
[ "$(tail -n +2 wide.tsv | wc -l)" = 6 ] || { echo "--wide should have 6 rows"; exit 1; }
tail -n +2 wide.tsv | awk -F'\t' '{ s = $2 + $3; if (s < 0.99 || s > 1.01) bad++ } END { exit (bad + 0) > 0 }' ||
  { echo "--wide rows do not sum to 1:"; cat wide.tsv; exit 1; }

"$MS" deconv --report r3.msdref train.cg 2>/dev/null > rep.txt
[ "$(wc -l < rep.txt)" = 6 ] || { echo "--report should be one line per record"; cat rep.txt; exit 1; }
grep -q '^c1: A [0-9.]*%; B [0-9.]*%$' rep.txt || { echo "--report line shape:"; head -1 rep.txt; exit 1; }

"$MS" deconv --min-frac 0 -o mf.tsv r3.msdref train.cg 2>/dev/null
[ "$(tail -n +2 mf.tsv | wc -l)" = 12 ] || { echo "--min-frac 0 -o rows"; exit 1; }

## ---- 3. grouping needs the trailer, and joins the pair under one label --
"$MS" deconv --group-threshold 0.05 r3.msdref train.cg 2>/dev/null > grp.tsv
grep -q "A/B" grp.tsv || { echo "--group-threshold did not join A and B:"; cat grp.tsv; exit 1; }
tail -n +2 grp.tsv | awk -F'\t' '$3 < 0.999 { bad++ } END { exit (bad + 0) > 0 }' ||
  { echo "a joined label must carry the pair's whole mass:"; cat grp.tsv; exit 1; }
## --wide never groups: the full-resolution answer is kept
"$MS" deconv --group-threshold 0.05 --wide r3.msdref train.cg 2>/dev/null > gw.tsv
cmp -s gw.tsv wide.tsv || { echo "--wide changed under --group-threshold"; exit 1; }
if "$MS" deconv --group-threshold 0.05 r2.msdref train.cg > g2.log 2>&1; then
  echo "--group-threshold ran on a reference with no trailer"; exit 1
fi
grep -q "confusion trailer" g2.log || { echo "wrong refusal:"; cat g2.log; exit 1; }

## ---- 4. tuning and diagnostic switches all run and keep the answer -------
for opt in "--no-adaptive" "--global-ref" "--weight-exponent 0.5,1" "--max-round 2" \
           "--rescue-below 0" "--rescue-target 5" "--rescue-min-depth 1" \
           "--min-cpg 1 --mass-floor 0.01 --max-segregating 5 --rescue-gap 0 --rescue-floor 0.1 --rescue-qfilter 0.4,0.6"; do
  "$MS" deconv $opt r3.msdref train.cg 2>/dev/null > o.tsv ||
    { echo "deconv $opt failed"; exit 1; }
  tail -n +2 o.tsv | sort -k1,1 -k3,3gr | awk -F'\t' '
    !seen[$1]++ { want = ($1 ~ /^c[123]$/) ? "A" : "B"; if ($2 != want) bad++ }
    END { exit (bad + 0) > 0 }' || { echo "deconv $opt lost the classes:"; cat o.tsv; exit 1; }
done
## -v narrates the scope; --pair-count and --eval-x add their reports
"$MS" deconv -v --pair-count A,B --eval-x A=0.5,B=0.5 r3.msdref train.cg 2> v.log >/dev/null
grep -q "scope 2/2" v.log || { echo "-v does not report the settled scope:"; head -5 v.log; exit 1; }
grep -qi "pair" v.log || { echo "--pair-count wrote nothing"; exit 1; }
## --delta-mean-top 3 leaves 6 CpGs per pattern on this fixture: under the
## mass floor, so every record settles an empty scope and emits no row. The
## point pinned here is that it says so rather than fabricating a call.
"$MS" deconv -v --delta-mean-top 3 r3.msdref train.cg 2> dm.log > dm.tsv
grep -q "over 6 CpGs" dm.log || { echo "--delta-mean-top 3 did not cut the panel:"; head -3 dm.log; exit 1; }
[ "$(tail -n +2 dm.tsv | wc -l)" = 0 ] || { echo "an empty scope still emitted rows:"; cat dm.tsv; exit 1; }
"$MS" deconv --force-scope A r3.msdref train.cg 2>/dev/null > fs.tsv
[ "$(tail -n +2 fs.tsv | wc -l)" -gt 0 ] || { echo "--force-scope A emitted nothing"; exit 1; }

## ---- 5. dumps: one file per switch, each non-empty ------------------------
"$MS" deconv --panel-out panel.tsv --scope-out scope.tsv --design-out design.tsv \
      r3.msdref train.cg >/dev/null 2>&1
for f in panel.tsv scope.tsv design.tsv; do
  [ -s $f ] || { echo "$f is empty"; exit 1; }
done
[ "$(head -1 scope.tsv)" = "$(printf 'cell\tA\tB')" ] || { echo "--scope-out header:"; head -1 scope.tsv; exit 1; }
[ "$(tail -n +2 scope.tsv | wc -l)" = 6 ] || { echo "--scope-out rows"; exit 1; }
echo "ok: v3 trailer, grouping, shapes, parallel, tuning, diagnostics, dumps"
