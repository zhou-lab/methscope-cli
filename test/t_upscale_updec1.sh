#!/bin/bash
## The legacy UPDEC1 inference path on a hand-built decoder small enough to
## check by hand: n_in=3, n_hidden=2, n_out=4, identity preprocessing and
## BatchNorm, ReLU hidden layer, sigmoid output. Row 1 has h = [relu(2),
## relu(-1)] = [2, 0], so the outputs are sigmoid(2), sigmoid(0), sigmoid(2),
## sigmoid(-2). Row 2's NA is imputed with the model mean (0), so every
## output is sigmoid(0) = 0.5.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
YAME=${YAME:?export YAME=/path/to/yame}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
cd "$d"
python3 - <<'PY'
import struct, array
def f32(a): return array.array('f', a).tobytes()
with open("toy.updec", "wb") as f:
    f.write(b"UPDEC1\x00\x00")
    f.write(struct.pack("<iii", 3, 2, 4)); f.write(struct.pack("<f", 1e-5))
    f.write(f32([0,0,0])); f.write(f32([0,0,0])); f.write(f32([1,1,1]))   # identity pre
    f.write(f32([1,0,0, 0,1,0])); f.write(f32([0,0]))                     # W1 -> h=[x1,x2]
    f.write(f32([1,1])); f.write(f32([0,0])); f.write(f32([0,0])); f.write(f32([1,1]))  # BN id
    f.write(f32([1,0, 0,1, 1,1, -1,0])); f.write(f32([0,0,0,0]))          # W2, b2
PY
printf 'feat_1\tfeat_2\tfeat_3\n2\t-1\t5\nNA\t0\t1\n' > toy.tsv

"$MS" upscale --probs toy.updec toy.tsv 2>/dev/null > probs.tsv
[ "$(sed -n 1p probs.tsv)" = "$(printf '0.880796\t0.5\t0.880796\t0.119204')" ] ||
  { echo "row 1 probabilities:"; cat probs.tsv; exit 1; }
[ "$(sed -n 2p probs.tsv)" = "$(printf '0.5\t0.5\t0.5\t0.5')" ] ||
  { echo "row 2 (NA imputed to the mean) probabilities:"; cat probs.tsv; exit 1; }
## the same from stdin
[ "$("$MS" upscale --probs toy.updec - < toy.tsv 2>/dev/null | wc -l)" = 2 ] ||
  { echo "stdin input did not yield 2 rows"; exit 1; }

## a dense block .cg: 2 records x 4 CpGs, format 4 (fractions)
"$MS" upscale -o out.cg toy.updec toy.tsv 2>/dev/null
info=$("$YAME" info out.cg 2>/dev/null | tail -1)
[ "$(echo "$info" | cut -f2,4,5)" = "$(printf '2\t4\t4')" ] || { echo "out.cg shape/format: $info"; exit 1; }
## --binary: format 6 calls
"$MS" upscale --binary -o outb.cg toy.updec toy.tsv 2>/dev/null
[ "$("$YAME" info outb.cg 2>/dev/null | tail -1 | cut -f5)" = 6 ] || { echo "--binary is not format 6"; exit 1; }

## refusals: a truncated model, a feature row of the wrong width
head -c 40 toy.updec > short.updec
if "$MS" upscale --probs short.updec toy.tsv >/dev/null 2>&1; then echo "a truncated UPDEC1 was accepted"; exit 1; fi
printf 'a\tb\n1\t2\n' > narrow.tsv
if "$MS" upscale --probs toy.updec narrow.tsv >/dev/null 2>&1; then echo "a 2-column row fed a 3-input model"; exit 1; fi

## ---- the bundled path: .updecx (model + MRMP + outcpg) on a query .cg ------
## This is how hg38_10k1.updecx runs, and until now only the docs gate (network,
## sbatch) exercised it. 12 CpGs: the MRMP names P1 (rows 0-3), P2 (4-7),
## P3 (8-9) and background Pna (10-11); the outcpg mask marks rows 1,4,7,10 as
## the four CpGs the decoder imputes. Record 1 has P1=1.0, P2=0.0, P3=0.5, so
## h=[1,0] and the outputs are sigmoid(1), sigmoid(0), sigmoid(1), sigmoid(-1).
## Record 2 leaves P2 uncovered (NA -> imputed 0) and has P1=0.5: h=[0.5,0].
printf 'P1\nP1\nP1\nP1\nP2\nP2\nP2\nP2\nP3\nP3\nPna\nPna\n' > mrmp.txt
"$YAME" pack -f s mrmp.txt mrmp.cm 2>/dev/null
printf '0\n1\n0\n0\n1\n0\n0\n1\n0\n0\n1\n0\n' > outcpg.txt
"$YAME" pack -f b outcpg.txt outcpg.cm 2>/dev/null
awk 'BEGIN { for (i = 0; i < 12; i++) print (i < 4 ? "10\t0" : i < 8 ? "0\t10" : "5\t5") }' > q1.txt
awk 'BEGIN { for (i = 0; i < 12; i++) print (i < 4 ? "5\t5"  : i < 8 ? "0\t0"  : "5\t5") }' > q2.txt
"$YAME" pack -f m q1.txt > q1.cg 2>/dev/null; "$YAME" pack -f m q2.txt > q2.cg 2>/dev/null
cat q1.cg q2.cg > q.cg; printf 'r1\nr2\n' > qn.txt; "$YAME" index -s qn.txt q.cg 2>/dev/null
"$MS" bundle -m mrmp.cm -O outcpg.cm -o toy.updecx toy.updec >/dev/null 2>&1
[ -s toy.updecx ] || { echo "bundle wrote no .updecx"; exit 1; }

"$MS" upscale --probs toy.updecx q.cg 2>/dev/null > bprobs.tsv
## the betas pass through float on this path, so the last printed digit can
## differ from the TSV path by one; compare to 2e-6 rather than as strings
near () { awk -v want="$2" 'NR == '"$1"' { n = split(want, w, " "); for (i = 1; i <= n; i++)
  if ($i - w[i] > 2e-6 || w[i] - $i > 2e-6) exit 1 }' bprobs.tsv; }
near 1 "0.731059 0.5 0.731059 0.268941" || { echo "bundled record 1 probabilities:"; cat bprobs.tsv; exit 1; }
near 2 "0.622459 0.5 0.622459 0.377541" || { echo "bundled record 2 (P2 uncovered -> mean) probabilities:"; cat bprobs.tsv; exit 1; }

## with an outcpg mask the .cg is genome-dimension: 12 rows, the four imputed
## CpGs carry the probabilities and every other row is NA
"$MS" upscale -o bout.cg toy.updecx q.cg 2>/dev/null
info=$("$YAME" info bout.cg 2>/dev/null | tail -1)
[ "$(echo "$info" | cut -f2,4,5)" = "$(printf '2\t12\t4')" ] || { echo "bundled out.cg shape/format: $info"; exit 1; }
"$YAME" unpack -a bout.cg 2>/dev/null > bout.tsv
## fmt4 unpacks to three decimals
[ "$(sed -n 2p bout.tsv)" = "$(printf '0.731\t0.622')" ] || { echo "row 1 (imputed) of bout.cg:"; cat bout.tsv; exit 1; }
[ "$(sed -n 11p bout.tsv)" = "$(printf '0.269\t0.378')" ] || { echo "row 10 (imputed) of bout.cg:"; cat bout.tsv; exit 1; }
nna=$(awk -F'\t' '$1 == "NA" && $2 == "NA"' bout.tsv | wc -l)
[ "$nna" = 8 ] || { echo "expected 8 NA rows outside the mask, got $nna:"; cat bout.tsv; exit 1; }
## --binary on the bundle: format 6 over the same 12 rows, 1 where p > 0.5
"$MS" upscale --binary -o boutb.cg toy.updecx q.cg 2>/dev/null
info=$("$YAME" info boutb.cg 2>/dev/null | tail -1)
[ "$(echo "$info" | cut -f2,4,5)" = "$(printf '2\t12\t6')" ] || { echo "bundled --binary shape/format: $info"; exit 1; }
"$YAME" unpack -a boutb.cg 2>/dev/null > boutb.tsv
[ "$(sed -n 2p boutb.tsv)" = "$(printf '1\t1')" ] && [ "$(sed -n 11p boutb.tsv)" = "$(printf '0\t0')" ] ||
  { echo "bundled --binary calls:"; cat boutb.tsv; exit 1; }

## without -O the bundle writes a dense block: 4 rows, not 12
"$MS" bundle -m mrmp.cm -o dense.updecx toy.updec >/dev/null 2>&1
"$MS" upscale -o dout.cg dense.updecx q.cg 2>/dev/null
[ "$("$YAME" info dout.cg 2>/dev/null | tail -1 | cut -f4)" = 4 ] || { echo "an unmasked bundle did not write a 4-row block"; exit 1; }
## refusal: an MRMP with fewer patterns than the model has inputs
printf 'P1\nP1\nP1\nP1\nP2\nP2\nP2\nP2\nPna\nPna\nPna\nPna\n' > two.txt
"$YAME" pack -f s two.txt two.cm 2>/dev/null
"$MS" bundle -m two.cm -o two.updecx toy.updec >/dev/null 2>&1
if "$MS" upscale --probs two.updecx q.cg >/dev/null 2>&1; then echo "a 2-pattern MRMP fed a 3-input model"; exit 1; fi
echo "ok: UPDEC1 forward pass matches the hand calculation; .cg, --binary, stdin, refusals; bundled .updecx with and without outcpg"
