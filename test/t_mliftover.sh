#!/bin/bash
## mliftover: a model re-indexed onto an array platform's row space must
## score the array data exactly as the genome-space model scores the same
## data in genome space. The platform here is a 135-probe "array" over the
## 400-CpG fixture genome: 100 cg probes on distinct CpGs, 10 replicate pairs
## sharing a CpG, 10 cg probes that map nowhere (no coordinate, a chromosome
## the track lacks, a position that is not a CpG), and 5 non-cg probes -- rs,
## ch, ctl -- three of which sit squarely on real CpGs and must be ignored
## for what they ARE, not where they map.
##
## The oracle for the .msdref is the route the 2026-09-05 TCGA HM450 run took
## by hand: cut the class pools to the probe rows with `yame rowsub -l` and
## rebuild with deconv-build-ref. The lift must reproduce it byte for byte.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
here=$(cd "$(dirname "$0")" && pwd); . "$here/fixtures.sh"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
ms_train "$d"; ms_ref_bank "$d"; cd "$d"

## ---- 0. the genome (a fmt7 track) and the platform (a coord table) --------
## CpG i sits at chr1:1000+10i (0-based start), so beg0 = 1000+10i.
awk 'BEGIN { for (i = 0; i < 400; i++) printf "chr1\t%d\t%d\tCpG_%d\n", 1000+10*i, 1001+10*i, i }' > cpgs.bed
"$YAME" pack -f r cpgs.bed > tiny.cr 2>/dev/null
[ -s tiny.cr ] || { echo "could not pack the fmt7 track"; exit 1; }
{
  printf 'CpG_chrm\tCpG_beg\tstrand\tmapQ\n'
  ## probes 0-99: CpG 3i (0,3,6,...,297); probes 100-109: replicates of CpG 3i (i<10)
  ## probes 110-119: unmapped; probes 120-129: CpG 300+3(i-120)
  for i in $(seq 0 99);    do printf 'chr1\t%d\t+\t60\n' $((1000 + 30*i)); done
  for i in $(seq 0 9);     do printf 'chr1\t%d\t-\t60\n' $((1000 + 30*i)); done
  for i in $(seq 0 4);     do printf 'NA\tNA\t*\t0\n'; done
  for i in $(seq 0 2);     do printf 'chrX\t%d\t+\t60\n' $((1000 + 10*i)); done
  for i in $(seq 0 1);     do printf 'chr1\t%d\t+\t60\n' $((1005 + 10*i)); done   # between CpGs
  for i in $(seq 0 9);     do printf 'chr1\t%d\t+\t60\n' $((1000 + 10*(300 + 3*i))); done
  ## probes 130-134: rs on CpG 1, rs on CpG 2, ch on CpG 4, rs unmapped, ctl
  printf 'chr1\t1010\t+\t60\nchr1\t1020\t+\t60\nchr1\t1040\t+\t60\nchr7\t99\t+\t60\nNA\tNA\t*\t0\n'
} | gzip > tiny.coord.tsv.gz
{ printf 'Probe_ID\tM\tU\tcol\n'
  for i in $(seq 0 129); do printf 'cg%08d\tNA\tNA\t2\n' $i; done
  printf 'rs0000001\tNA\tNA\t2\nrs0000002\tNA\tNA\t2\nch.1.4\tNA\tNA\t2\nrs0000003\tNA\tNA\t2\nctl_norm_1\tNA\tNA\t2\n'
} | gzip > tiny.ordering.tsv.gz
LIFT="$MS mliftover --to TINY --coord tiny.coord.tsv.gz --ordering tiny.ordering.tsv.gz --cr tiny.cr"
## cpg_of_probe, 1-based rows for rowsub -l, in probe order (mapped only)
{ for i in $(seq 0 99); do echo $((3*i + 1)); done
  for i in $(seq 0 9);  do echo $((3*i + 1)); done
  for i in $(seq 0 9);  do echo $((300 + 3*i + 1)); done; } > mapped_rows.txt
[ "$(wc -l < mapped_rows.txt)" = 120 ] || { echo "fixture arithmetic"; exit 1; }

## an ARRAY-SPACE query: the six training cells re-expressed over the 130
## probes, from the same generator as ms_train (cell c, CpG i), 0/0 unmapped
gen_probe_cell () {   # $1 = cell
  awk -v c="$1" 'BEGIN { srand(c)
      for (i = 0; i < 400; i++) { b = (((c <= 3) == (i % 7 < 4)) ? 9 : 1)
        m[i] = b + ((rand() < 0.15) ? (b > 4 ? -3 : 3) : 0) }
      for (p = 0; p < 100; p++) { i = 3*p;       print m[i] "\t" (10 - m[i]) }
      for (p = 0; p < 10;  p++) { i = 3*p;       print m[i] "\t" (10 - m[i]) }
      for (p = 0; p < 10;  p++) {                 print "0\t0" }
      for (p = 0; p < 10;  p++) { i = 300 + 3*p; print m[i] "\t" (10 - m[i]) }
      for (p = 0; p < 5;   p++) {                 print "9\t1" } }'
}
for c in 1 2 3 4 5 6; do gen_probe_cell $c > pc$c.txt; "$YAME" pack -f m pc$c.txt > pc$c.cg 2>/dev/null; done
cat pc[1-6].cg > array.cg; "$YAME" index -s cellnames.txt array.cg 2>/dev/null
[ "$("$YAME" info array.cg 2>/dev/null | tail -1 | cut -f4)" = 135 ] || { echo "array query is not 135 rows"; exit 1; }

## ---- 1. a loose .mrmp: n_cpg moves, counts are recomputed, sets keep names --
"$MS" mrmp-build ref.cg m.mrmp >/dev/null 2>&1
$LIFT m.mrmp -o m.TINY.mrmp > l1.log 2>&1
grep -q "120 of 135 probes sit on a CpG row of hg38 (400 rows); 5 not lifted by type (not cg), 10 cg probes with no CpG row" l1.log ||
  { echo "map summary:"; cat l1.log; exit 1; }
"$MS" inspect m.TINY.mrmp > mi.txt 2>&1
grep -qE "reference +TINY" mi.txt || { echo "the lifted artifact does not name the platform:"; cat mi.txt; exit 1; }
grep -qE "covered +120 of 135 CpGs" mi.txt || { echo "lifted coverage:"; cat mi.txt; exit 1; }
"$MS" mrmp-export --counts - m.TINY.mrmp 2>/dev/null > cnt.tsv
## P1 = CpGs with i%7<4 among the 120 mapped; the awk below counts the truth
want=$(awk 'BEGIN { n = 0; for (p = 0; p < 100; p++) if ((3*p) % 7 < 4) n++; for (p = 0; p < 10; p++) if ((3*p) % 7 < 4) n++; for (p = 0; p < 10; p++) if ((300+3*p) % 7 < 4) n++; print n }')
got=$(awk -F'\t' '$2 == "111000" || $2 == "10" { print $1 }' cnt.tsv | head -1)
[ "$got" = "$want" ] || { echo "P1 count after the lift: got $got want $want"; cat cnt.tsv; exit 1; }

## ---- 2. a logistic bundle (.cm-front): calls identical in both spaces ------
"$MS" mrmp-export m.mrmp m.cm >/dev/null 2>&1
"$MS" classify-train --framework logistic -l labels.txt -o lin.clfx train.cg m.cm >/dev/null 2>&1
$LIFT lin.clfx -o lin.TINY.clfx > l2.log 2>&1
grep -q "mask: 3 states over 135 target rows" l2.log || { echo "mask lift summary:"; cat l2.log; exit 1; }
"$MS" inspect lin.TINY.clfx | grep -q '"logistic" framework mark' || { echo "the repacked bundle lost its kind"; exit 1; }
"$MS" classify --probs lin.clfx train.cg 2>/dev/null > g.tsv
"$MS" classify --probs lin.TINY.clfx array.cg 2>/dev/null > a.tsv
cmp -s <(cut -f1,2 g.tsv) <(cut -f1,2 a.tsv) || { echo "genome and array calls differ:"; paste g.tsv a.tsv; exit 1; }
## a mean over 120 of 400 CpGs is not the same mean; it must be close
paste <(tail -n +2 g.tsv | cut -f3) <(tail -n +2 a.tsv | cut -f3) |
  awk '{ d = $1 - $2; if (d < 0) d = -d; if (d > 0.05) bad++ } END { exit (bad + 0) > 0 }' ||
  { echo "confidences drifted by more than 0.05:"; paste g.tsv a.tsv; exit 1; }
## the genome model refuses the array query, the lifted one refuses the genome query
if "$MS" classify lin.clfx array.cg >/dev/null 2>&1; then echo "the genome model scored a 135-row query"; exit 1; fi
if "$MS" classify lin.TINY.clfx train.cg >/dev/null 2>&1; then echo "the lifted model scored a 400-row query"; exit 1; fi

## ---- 3. a tree bundle (MRMPIDX1 chain, per-node boosters): calls identical --
"$MS" mrmp-build --bank --anneal-min-seg 100,5 --min-pattern-cpgs 1 --resolvers all \
      --force bankref.cg bank.mrmp >/dev/null 2>&1
printf 'A1\nA2\nA3\nB1\nB2\nB3\n' > bl.txt
"$MS" classify-featurize -l bl.txt --sample 300,0 --reps 2 -b --seed 7 -o fb.msfm bankref.cg bank.mrmp >/dev/null 2>&1
"$MS" classify-train --data fb.msfm -o tree.clfx >/dev/null 2>&1
$LIFT tree.clfx -o tree.TINY.clfx > l3.log 2>&1
[ "$(grep -c "pattern(s) over" l3.log)" = 18 ] || { echo "18 sets should each report:"; cat l3.log; exit 1; }
"$MS" inspect --tree tree.TINY.clfx | grep -q "3 nodes + 15 soft, 6 classes" || { echo "lifted tree shape"; exit 1; }
## the six bank cells over the probes
for c in 1 2 3 4 5 6; do
  awk -v c="$c" 'BEGIN { for (i = 0; i < 400; i++) {
      g = (c <= 3); f = (c == 1 || c == 4); h = (c == 2 || c == 5)
      if (i % 7 < 4) m[i] = (g ? 9 : 1); else if (i % 11 < 3) m[i] = (f ? 9 : 1)
      else if (i % 13 < 2) m[i] = (h ? 9 : 1); else m[i] = 5 }
      for (p = 0; p < 100; p++) { i = 3*p; print m[i] "\t" (10 - m[i]) }
      for (p = 0; p < 10; p++)  { i = 3*p; print m[i] "\t" (10 - m[i]) }
      for (p = 0; p < 10; p++)  print "0\t0"
      for (p = 0; p < 10; p++)  { i = 300 + 3*p; print m[i] "\t" (10 - m[i]) }
      for (p = 0; p < 5; p++)   print "9\t1" }' > pb$c.txt
  "$YAME" pack -f m pb$c.txt > pb$c.cg 2>/dev/null
done
cat pb[1-6].cg > barray.cg; "$YAME" index -s banknames.txt barray.cg 2>/dev/null
"$MS" classify tree.clfx bankref.cg 2>/dev/null > tg.tsv
"$MS" classify tree.TINY.clfx barray.cg 2>/dev/null > ta.tsv
cmp -s <(cut -f1,2 tg.tsv) <(cut -f1,2 ta.tsv) || { echo "tree calls differ across spaces:"; paste tg.tsv ta.tsv; exit 1; }

## ---- 4. .msdref: byte-identical to the hand route (rowsub -l + build-ref) --
printf '# held-out\nA\tA\t0.9\nA\tB\t0.1\nB\tB\t0.95\nB\tA\t0.05\n' > conf.tsv
"$MS" deconv-build-ref --confusion conf.tsv -o r.msdref ref.cg >/dev/null 2>&1
$LIFT r.msdref -o r.TINY.msdref > l4.log 2>&1
grep -q "confusion trailer carried" l4.log || { echo "trailer:"; cat l4.log; exit 1; }
## the hand route lives in the 120-row space of MAPPED probes; the lift lives
## in the 130-row platform space. Compare on what both have: the kept rows'
## M/U, read back through deconv on the array query -- fractions must match.
"$YAME" rowsub -l mapped_rows.txt ref.cg > ref120.cg 2>/dev/null
"$YAME" index -s refnames.txt ref120.cg 2>/dev/null
"$MS" deconv-build-ref --confusion conf.tsv -o hand.msdref ref120.cg >/dev/null 2>&1
"$YAME" rowsub -l mapped_rows.txt train.cg > q120.cg 2>/dev/null
"$YAME" index -s cellnames.txt q120.cg 2>/dev/null
"$MS" deconv --wide hand.msdref q120.cg 2>/dev/null > hand.tsv
"$MS" deconv --wide r.TINY.msdref array.cg 2>/dev/null > lift.tsv
cmp -s hand.tsv lift.tsv || { echo "the lift and the hand route disagree:"; paste hand.tsv lift.tsv; exit 1; }
## and the lifted reference refuses the genome-space query
if "$MS" deconv r.TINY.msdref train.cg >/dev/null 2>&1; then echo "the lifted reference took a 400-row query"; exit 1; fi

## ---- 5. refusals ----------------------------------------------------------
"$MS" upscale-featurize --reps 1 --sample 100 train.cg m.mrmp t.msur >/dev/null 2>&1
"$MS" upscale-set-units --unit-cpgs 100 ref.cg u.msui >/dev/null 2>&1
printf '0\ttrain\n1\ttrain\n2\tval\n3\ttrain\n4\tval\n5\ttest\n' > split.tsv
"$MS" upscale-train -i t.msur --units u.msui --mrmp m.mrmp -o up.updecx --work-dir w \
      --min-steps 5 --max-steps 10 --eval-every 5 --batch 64 --eval-rows 2 \
      --split split.tsv > up_train.log 2>&1 || { echo "throwaway upscaler:"; cat up_train.log; exit 1; }
if $LIFT up.updecx -o x.updecx > up.log 2>&1; then
  echo "an upscaler was lifted"; exit 1
fi
grep -q "Carry the data instead" up.log || { echo "wrong refusal:"; cat up.log; exit 1; }
if $LIFT m.mrmp -o m.TINY.mrmp >/dev/null 2>&1; then
  echo "overwrote without --force"; exit 1
fi
$LIFT --force m.mrmp -o m.TINY.mrmp >/dev/null 2>&1
if $LIFT --min-retained 1000 m.mrmp -o x.mrmp >/dev/null 2>&1; then
  echo "--min-retained 1000 did not refuse"; exit 1
fi
if $LIFT train.cg -o x.out >/dev/null 2>&1; then
  echo "a plain .cg was accepted as a model"; exit 1
fi
echo "ok: .mrmp, logistic and tree bundles, .msdref lifted; calls identical; hand route matched"
