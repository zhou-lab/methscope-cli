#!/bin/bash
## The UPDEC2 training chain end to end on the two-class fixture:
## upscale-featurize (truth msur) -> upscale-set-units (processing units)
## -> upscale-train (CPU backend, resumable) -> upscale (whole-row-space
## reconstruction). Six cells x 400 CpGs train in milliseconds, which is
## what lets the trainer's option surface be exercised at all: --split,
## --threads, the feature and activation modes, --pilot-units, --dry-run,
## --force and the refusals around them.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_train "$d"; cd "$d"
"$MS" mrmp-build ref.cg m.mrmp >/dev/null 2>&1

## ---- 1. featurize: the truth msur, with its provenance manifest ----------
"$MS" upscale-featurize --reps 3 --sample 100,200 --binarize --threads 2 \
      --manifest t.tsv train.cg m.mrmp t.msur > f.log 2>&1
grep -q "wrote t.msur and t.tsv" f.log || { echo "featurize:"; cat f.log; exit 1; }
"$MS" inspect t.msur > msur.txt 2>&1
grep -qE "cells +6" msur.txt && grep -qE "replicates +6" msur.txt && grep -qE "rows +36" msur.txt ||
  { echo "3 reps x 2 levels over 6 cells should be 36 rows:"; cat msur.txt; exit 1; }
grep -q "binarized" msur.txt || { echo "--binarize not recorded"; exit 1; }
## the sampler needs the truth's .cg.idx
cp train.cg noidx.cg
if "$MS" upscale-featurize --reps 1 --sample 100 noidx.cg m.mrmp x.msur >/dev/null 2>&1; then
  echo "featurize sampled an unindexed truth"; exit 1
fi

## ---- 2. set-units: an OUTPUT partition, read from the store -------------
"$MS" upscale-set-units --unit-cpgs 100 --call-mindepth 1 --beta-threshold 0.5 ref.cg u.msui > u.log 2>&1
grep -q "wrote u.msui" u.log || { echo "set-units:"; cat u.log; exit 1; }
"$MS" inspect u.msui > msui.txt 2>&1
grep -qE "units +2 " msui.txt && grep -qE "cpgs +400 " msui.txt ||
  { echo "2 memberships over 400 CpGs expected:"; cat msui.txt; exit 1; }
if "$MS" upscale-set-units --unit-cpgs 100 m.mrmp x.msui >/dev/null 2>&1; then
  echo "set-units accepted a .mrmp where the store belongs"; exit 1
fi

## ---- 3. train: dry run, then a tiny real run with a curated split ----------
"$MS" upscale-train -i t.msur --units u.msui --mrmp m.mrmp -o mdl.updecx \
      --work-dir work --dry-run > dry.log 2>&1
grep -q "dry run complete" dry.log || { echo "--dry-run:"; cat dry.log; exit 1; }
[ ! -e mdl.updecx ] || { echo "--dry-run wrote a model"; exit 1; }
printf '0\ttrain\n1\ttrain\n2\tval\n3\ttrain\n4\tval\n5\ttest\n' > split.tsv
train () {   # $1 = out, then extra options
  local out=$1; shift
  "$MS" upscale-train -i t.msur --units u.msui --mrmp m.mrmp -o "$out" \
        --min-steps 5 --max-steps 10 --eval-every 5 --patience 2 --batch 64 \
        --eval-rows 2 --split split.tsv "$@"
}
train mdl.updecx --work-dir work --threads 2 --activation leaky --features scalar > tr.log 2>&1
grep -q "split=3/2/1 threads=2" tr.log || { echo "the curated split was not honoured:"; cat tr.log; exit 1; }
grep -q "units 2/2 (resumed=0)" tr.log || { echo "unit progress:"; cat tr.log; exit 1; }
grep -q "val MAE over 2 units: mean" tr.log || { echo "no post-join summary:"; cat tr.log; exit 1; }
grep -q "complete -> mdl.updecx" tr.log || { echo "train did not complete:"; cat tr.log; exit 1; }
"$MS" inspect mdl.updecx > mdl.txt 2>&1
grep -q '"upscale" framework mark' mdl.txt || { echo "the bundle is not marked upscale:"; cat mdl.txt; exit 1; }
## the other feature/mixed modes each train
for o in "--features beta" "--features missing" "--features count --mixed-mode direct" \
         "--activation linear --pure-bottleneck 4 --mixed-bottleneck 8"; do
  train o.updecx --work-dir w_o --force $o > o.log 2>&1 || { echo "train $o failed:"; cat o.log; exit 1; }
  rm -rf w_o
done
## --pilot-units is a CUDA-backend affordance; the CPU backend says so
printf '0\n' > pilot.txt
if train p.updecx --work-dir w_p --pilot-units pilot.txt > p.log 2>&1; then
  echo "--pilot-units ran on the CPU backend"; exit 1
fi
grep -q "CPU backend does not support --pilot-units" p.log || { echo "wrong refusal:"; cat p.log; exit 1; }

## ---- 4. refusals: existing output, a bad split ----------------------------
if train mdl.updecx --work-dir work > again.log 2>&1; then
  echo "upscale-train overwrote its output without --force"; exit 1
fi
grep -q "use --force" again.log || { echo "wrong refusal:"; cat again.log; exit 1; }
train mdl.updecx --work-dir work --force > force.log 2>&1
printf '0\ttrain\n1\tnope\n' > bad.tsv
if "$MS" upscale-train -i t.msur --units u.msui --mrmp m.mrmp -o b.updecx \
     --work-dir w_b --split bad.tsv --min-steps 5 --max-steps 10 >/dev/null 2>&1; then
  echo "a malformed --split was accepted"; exit 1
fi

## ---- 5. upscale: the model reconstructs the whole row space ---------------
"$MS" upscale -o rec.cg mdl.updecx train.cg > up.log 2>&1
grep -q "upscaled 6 sample(s) x 400 CpGs" up.log || { echo "upscale:"; cat up.log; exit 1; }
info=$("$YAME" info rec.cg 2>/dev/null | tail -1)
[ "$(echo "$info" | cut -f2,4,5)" = "$(printf '6\t400\t4')" ] || { echo "rec.cg shape: $info"; exit 1; }
"$MS" upscale --probs mdl.updecx train.cg 2>/dev/null > probs.tsv
[ "$(wc -l < probs.tsv)" = 6 ] || { echo "--probs rows"; exit 1; }
[ "$(head -1 probs.tsv | tr '\t' '\n' | wc -l)" = 400 ] || { echo "--probs should be 400 columns"; exit 1; }
"$MS" upscale --binary -o recb.cg mdl.updecx train.cg >/dev/null 2>&1
[ "$("$YAME" info recb.cg 2>/dev/null | tail -1 | cut -f5)" = 6 ] || { echo "--binary is not format 6"; exit 1; }

## ---- 6. featurize: the log-spaced coverage ladder ------------------------
## --sample-logrange is how the shipped decoders were trained: --reps is the
## TOTAL replicate count and each replicate draws its own sample size, spread
## in log space over [MIN,MAX] (skew 1 = plain log-uniform). Six cells x 4
## replicates = 24 rows, every sample size inside the range.
"$MS" upscale-featurize --reps 4 --sample-logrange 50,300 --sample-skew 1 \
      --manifest lr.tsv train.cg m.mrmp lr.msur > lr.log 2>&1
grep -q "wrote lr.msur and lr.tsv" lr.log || { echo "featurize --sample-logrange:"; cat lr.log; exit 1; }
"$MS" inspect lr.msur > lr.txt 2>&1
grep -qE "rows +24" lr.txt || { echo "4 log-range replicates over 6 cells should be 24 rows:"; cat lr.txt; exit 1; }
grep -qi "logrange\|log-range\|50.*300" lr.txt lr.tsv || { echo "the ladder is not recorded in the msur or its manifest:"; cat lr.txt; exit 1; }
if "$MS" upscale-featurize --reps 4 --sample-logrange 300,50 train.cg m.mrmp x.msur >/dev/null 2>&1; then
  echo "featurize accepted an inverted log range"; exit 1
fi

## ---- 7. featurize on a CHAIN: one column per pattern per set --------------
## A bank MRMP is a chain of sets. The featurizer maps every set's patterns
## to its own columns (ms_mrmp_group_map_chain), so a resolver's pattern is a
## column its hard block cannot express. The 6-class bank fixture builds 18
## sets; the truth here is the reference itself.
ms_ref_bank "$d"
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all --force \
      bankref.cg bank.mrmp > bank.log 2>&1 || { echo "bank build:"; cat bank.log; exit 1; }
"$MS" upscale-featurize --reps 1 --sample 100 bankref.cg bank.mrmp chain.msur > chain.log 2>&1
grep -q "chain of 18 sets -> 28 columns" chain.log ||
  { echo "chain featurize did not map 18 sets to 28 columns:"; cat chain.log; exit 1; }
"$MS" inspect chain.msur > chain.txt 2>&1
grep -qE "rows +6" chain.txt || { echo "1 rep over 6 cells should be 6 rows:"; cat chain.txt; exit 1; }
echo "ok: featurize (levels, log-range ladder, chain), set-units, train (split/threads/modes/pilot), upscale"
