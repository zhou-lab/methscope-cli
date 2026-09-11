## Shared fixture builders, sourced by the t_*.sh scripts.
##
## Everything is packed from inline awk rather than committed, so the suite
## carries no data and a fixture's CONTENT is visible where it is used. The
## reference is deliberately tiny and deliberately separable: six cells in two
## groups of three, with a block of CpGs that splits them cleanly, so an MRMP
## built on it has something real to find.

## ms_ref <dir> -- writes ref.cg (+ .idx) and names.txt in $1, 6 cells x 400 CpGs
ms_ref () {
  local d=$1 c
  for c in 1 2 3 4 5 6; do
    ## cells 1-3 are methylated where 4-6 are not, over 4/7 of the rows; the
    ## rest is shared, so the store has both separating and constant CpGs.
    awk -v c="$c" 'BEGIN { for (i = 0; i < 400; i++) {
        m = (((c <= 3) == (i % 7 < 4)) ? 9 : 1); print m "\t" (10 - m) } }' > "$d/cell$c.txt"
    "$YAME" pack -f m "$d/cell$c.txt" > "$d/cell$c.cg" 2>/dev/null
  done
  cat "$d"/cell[1-6].cg > "$d/ref.cg"
  printf 'A1\nA2\nA3\nB1\nB2\nB3\n' > "$d/names.txt"
  "$YAME" index -s "$d/names.txt" "$d/ref.cg" 2>/dev/null
}

## ms_labels <dir> -- two-class labels matching ms_ref, one per line in cell order
ms_labels () { printf 'A\nA\nA\nB\nB\nB\n' > "$1/labels.txt"; }
