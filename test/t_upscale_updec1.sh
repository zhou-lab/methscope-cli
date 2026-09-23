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
echo "ok: UPDEC1 forward pass matches the hand calculation; .cg, --binary, stdin, refusals"
