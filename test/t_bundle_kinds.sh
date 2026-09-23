#!/bin/bash
## A violation bundle: the spec is a text file, so `bundle -k violation` is
## the one way such a bundle comes to exist, and classify must read it back
## and reach the same calls as scoring the .mrmp directly.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_train "$d"; cd "$d"
"$MS" mrmp-build ref.cg m.mrmp >/dev/null 2>&1
"$MS" mrmp-export m.mrmp m.cm >/dev/null 2>&1

## the two patterns' binstrings, as mrmp-export reports them
"$MS" mrmp-export --patterns - m.mrmp 2>/dev/null > pat.tsv
b1=$(awk -F'\t' '$2=="P1"{print $1}' pat.tsv); c1=$(awk -F'\t' '$2=="P1"{print $3}' pat.tsv)
b2=$(awk -F'\t' '$2=="P2"{print $1}' pat.tsv); c2=$(awk -F'\t' '$2=="P2"{print $3}' pat.tsv)
printf 'methscope-violation\t1\nthreshold\t0.5\nweight\tsqrt\nmin_patterns\t1\nlabels\tA\tB\n' > vio.txt
printf 'pattern\tP1\t%s\t%s\npattern\tP2\t%s\t%s\n' "$b1" "$c1" "$b2" "$c2" >> vio.txt
"$MS" bundle -m m.cm -k violation -o v.clfx vio.txt >/dev/null 2>&1
"$MS" inspect v.clfx > v.txt 2>&1
grep -q '"violation" framework mark' v.txt || { echo "no violation kind mark"; exit 1; }

"$MS" classify --probs v.clfx train.cg 2>/dev/null > vb.tsv
"$MS" classify --framework violation --min-patterns 1 --probs m.mrmp train.cg 2>/dev/null > vd.tsv
cmp -s vb.tsv vd.tsv || { echo "the bundle and the direct rule disagree:"; diff vb.tsv vd.tsv; exit 1; }
paste <(tail -n +2 vb.tsv | cut -f2) labels.txt |
  awk -F'\t' '$1 != $2 { bad++ } END { exit (bad + 0) > 0 }' || { echo "mislabelled:"; cat vb.tsv; exit 1; }
## prebuilt features on the bundle route
"$MS" classify-featurize -l labels.txt -o f.msfm train.cg m.mrmp >/dev/null 2>&1
[ "$("$MS" classify --data f.msfm v.clfx 2>/dev/null | tail -n +2 | wc -l)" = 6 ] || { echo "--data on the bundle"; exit 1; }

## malformed specs are refused when read, each with its own reason
printf 'threshold\t0.5\n' > bad1.txt
"$MS" bundle -m m.cm -k violation -o b1.clfx bad1.txt >/dev/null 2>&1
"$MS" classify b1.clfx train.cg > b1.log 2>&1 && { echo "a spec with no pattern rows scored"; exit 1; }
grep -q "no pattern rows" b1.log || { echo "wrong refusal:"; cat b1.log; exit 1; }
printf 'methscope-violation\t1\nlabels\tA\tB\npattern\tP1\t100\t5\n' > bad2.txt
"$MS" bundle -m m.cm -k violation -o b2.clfx bad2.txt >/dev/null 2>&1
"$MS" classify b2.clfx train.cg > b2.log 2>&1 && { echo "a 3-wide binstring over 2 labels scored"; exit 1; }
grep -q "binstring width" b2.log || { echo "wrong refusal:"; cat b2.log; exit 1; }
printf 'pattern\tP1\t10\t5\nlabels\tA\tB\n' > bad3.txt
"$MS" bundle -m m.cm -k violation -o b3.clfx bad3.txt >/dev/null 2>&1
"$MS" classify b3.clfx train.cg > b3.log 2>&1 && { echo "a spec without the magic scored"; exit 1; }
grep -q "not a methscope-violation spec" b3.log || { echo "wrong refusal:"; cat b3.log; exit 1; }
echo "ok: violation bundle round-trips and refuses malformed specs"
