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

## ms_train <dir> -- the TWO stores a training run needs, and the labels.
##
## This is the shape the pipeline actually requires, and the shape the docs'
## removed example got wrong. mrmp-build takes a CLASS-level reference: one
## record per class, record name = class name (yame rejects duplicate names, so
## a class cannot be three records). classify-featurize takes a CELL-level
## store plus a labels file naming those same classes. Build the MRMP over
## cells and label by group and the root node has no training rows for its own
## classes -- the failure the Train tab shipped with.
##
## Writes: ref.cg (+idx, classes A B), train.cg (+idx, 6 cells), labels.txt
ms_train () {
  local d=$1 c
  for c in A B; do
    awk -v c="$c" 'BEGIN { for (i = 0; i < 400; i++) {
        m = (((c == "A") == (i % 7 < 4)) ? 9 : 1); print m "\t" (10 - m) } }' > "$d/$c.txt"
    "$YAME" pack -f m "$d/$c.txt" > "$d/$c.cg" 2>/dev/null
  done
  cat "$d/A.cg" "$d/B.cg" > "$d/ref.cg"
  printf 'A\nB\n' > "$d/refnames.txt"
  "$YAME" index -s "$d/refnames.txt" "$d/ref.cg" 2>/dev/null

  ## six cells, three per class, with 15% of sites flipped so the cells are
  ## drawn from the class rather than identical to it -- a classifier that
  ## cannot tell these apart has learned nothing.
  for c in 1 2 3 4 5 6; do
    awk -v c="$c" 'BEGIN { srand(c); for (i = 0; i < 400; i++) {
        b = (((c <= 3) == (i % 7 < 4)) ? 9 : 1)
        m = b + ((rand() < 0.15) ? (b > 4 ? -3 : 3) : 0); print m "\t" (10 - m) } }' > "$d/t$c.txt"
    "$YAME" pack -f m "$d/t$c.txt" > "$d/t$c.cg" 2>/dev/null
  done
  cat "$d"/t[1-6].cg > "$d/train.cg"
  printf 'c1\nc2\nc3\nc4\nc5\nc6\n' > "$d/cellnames.txt"
  "$YAME" index -s "$d/cellnames.txt" "$d/train.cg" 2>/dev/null
  printf 'A\nA\nA\nB\nB\nB\n' > "$d/labels.txt"
}
