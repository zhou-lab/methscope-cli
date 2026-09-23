#!/bin/bash
## The routing-tree classifier end to end on the 6-class bank fixture, with
## the featurizer options that feed it. One cell per class trains a
## degenerate booster (uniform posterior), so the training set is inflated
## with sampled replicates -- which is also what exercises the sampler.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_ref_bank "$d"; ms_train "$d"; cd "$d"
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all \
      --force bankref.cg bank.mrmp >/dev/null 2>&1
printf 'A1\nA2\nA3\nB1\nB2\nB3\n' > bl.txt

## ---- 1. featurizer options ------------------------------------------------
"$MS" classify-featurize -l bl.txt --sample 300,0 --reps 2 -b --seed 7 --threads 2 \
      -o fb.msfm bankref.cg bank.mrmp > fb.log 2>&1
grep -q "24 records" fb.log || { echo "2 levels x 2 reps x 6 cells should be 24 records:"; cat fb.log; exit 1; }
for o in "--continuous-features" "--thresh-pattern" "--counts 2" "--set root" \
         "--satellite-contrast replace" "--satellite-contrast add"; do
  "$MS" classify-featurize $o -l bl.txt -o o.msfm bankref.cg bank.mrmp > o.log 2>&1 ||
    { echo "classify-featurize $o failed:"; cat o.log; exit 1; }
  "$MS" inspect o.msfm >/dev/null 2>&1 || { echo "inspect cannot read the $o artifact"; exit 1; }
done
## no .cg.idx: the single-threaded scan, loudly; --sample then refuses
cp train.cg noidx.cg
"$MS" mrmp-build ref.cg m.mrmp >/dev/null 2>&1
"$MS" classify-featurize -l labels.txt -o ni.msfm noidx.cg m.mrmp > ni.log 2>&1
grep -q "falling back to the single-threaded scan" ni.log || { echo "unindexed query was not announced:"; cat ni.log; exit 1; }
"$MS" inspect ni.msfm > ni.txt 2>&1
grep -q "6" ni.txt || { echo "the scan path wrote a bad artifact"; exit 1; }
if "$MS" classify-featurize --sample 100 -l labels.txt -o x.msfm noidx.cg m.mrmp >/dev/null 2>&1; then
  echo "--sample ran without an index"; exit 1
fi
## --patterns below the layout is refused, not silently padded
if "$MS" classify-featurize --patterns 1 -l labels.txt -o x.msfm train.cg m.mrmp >/dev/null 2>&1; then
  echo "--patterns 1 of 2 exited 0"; exit 1
fi

## ---- 2. train: --keep-columns, and the knobs the tree parses itself ----------
"$MS" classify-train --data fb.msfm -o tree.clfx --max-depth 3 --colsample 0.8 \
      --min-child-weight 1 -n 20 --threads 2 > tt.log 2>&1
"$MS" inspect tree.clfx > ti.txt 2>&1
grep -q "MSBNDL1" ti.txt || { echo "no bundle"; exit 1; }
"$MS" inspect --tree tree.clfx > tree.txt 2>&1
grep -q "3 nodes + 15 soft, 6 classes" tree.txt || { echo "inspect --tree:"; head -4 tree.txt; exit 1; }
"$MS" inspect --tree bank.mrmp > chain.txt 2>&1
grep -q "3 nodes + 15 soft" chain.txt || { echo "inspect --tree on the chain"; exit 1; }
printf '0\n1\n2\n' > keep.txt
"$MS" classify-train --data fb.msfm --keep-columns keep.txt -o keep.clfx > kt.log 2>&1
grep -q "3 of .* columns kept (--keep-columns)" kt.log || { echo "--keep-columns:"; cat kt.log; exit 1; }
printf '2\n1\n' > bad.txt
if "$MS" classify-train --data fb.msfm --keep-columns bad.txt -o x.clfx >/dev/null 2>&1; then
  echo "a descending --keep-columns list was accepted"; exit 1
fi
## -h wins over --data: the shared usage, not the tree's own text
"$MS" classify-train --data fb.msfm -h > h.txt 2>&1
grep -q "^Usage:" h.txt ||
  { echo "classify-train --data -h printed no usage"; exit 1; }
if "$MS" classify-train --data fb.msfm --framework logistic -o x.clfx >/dev/null 2>&1; then
  echo "--data with a non-xgboost framework exited 0"; exit 1
fi

## ---- 3. classify: all six classes recovered, on every route ---------------
calls () { tail -n +2 "$1" | cut -f1,2 | awk -F'\t' '$1 != $2 { bad++ } END { exit (bad + 0) > 0 }'; }
"$MS" classify tree.clfx bankref.cg 2>/dev/null > p.tsv
calls p.tsv || { echo "the tree mislabels its own classes:"; cat p.tsv; exit 1; }
"$MS" classify --threads 2 tree.clfx bankref.cg 2>/dev/null > pt.tsv
cmp -s p.tsv pt.tsv || { echo "--threads 2 changed the calls"; exit 1; }
"$MS" classify-featurize -l bl.txt -o q.msfm bankref.cg bank.mrmp >/dev/null 2>&1
"$MS" classify --data q.msfm tree.clfx 2>/dev/null > pd.tsv
[ "$(tail -n +2 pd.tsv | wc -l)" = 6 ] || { echo "--data route rows"; exit 1; }
"$MS" classify keep.clfx bankref.cg 2>/dev/null > pk.tsv
[ "$(tail -n +2 pk.tsv | wc -l)" = 6 ] || { echo "the --keep-columns model does not score"; exit 1; }
if "$MS" classify --probs tree.clfx bankref.cg > probs.log 2>&1; then
  echo "--probs on a tree exited 0"; exit 1
fi
grep -q "not implemented on the routing-tree path" probs.log || { echo "wrong refusal:"; cat probs.log; exit 1; }

## ---- 4. relabel: the name changes, the numbers do not -----------------------
"$MS" relabel -o re.clfx 'A1=Z1' tree.clfx > re.log 2>&1
grep -q "'A1' -> 'Z1' in 1 of 1 node(s)" re.log || { echo "relabel:"; cat re.log; exit 1; }
"$MS" classify re.clfx bankref.cg 2>/dev/null > pr.tsv
[ "$(tail -n +2 pr.tsv | head -1 | cut -f2)" = "Z1" ] || { echo "A1 was not renamed:"; head -2 pr.tsv; exit 1; }
cmp -s <(cut -f3,4 p.tsv) <(cut -f3,4 pr.tsv) || { echo "relabel changed a number"; exit 1; }
if "$MS" relabel -o re.clfx 'A1=Z1' tree.clfx >/dev/null 2>&1; then echo "relabel overwrote without --force"; exit 1; fi
if "$MS" relabel --force -o re.clfx 'NOPE=Z' tree.clfx >/dev/null 2>&1; then echo "relabel of a missing label exited 0"; exit 1; fi
echo "ok: featurizer options, tree train and score, keep-columns, relabel"
