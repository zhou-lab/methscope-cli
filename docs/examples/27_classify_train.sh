#!/bin/bash
## title: Train your own classifier
## norun: the recorded build of hg38_celltype_full.clfx (labjournal 20260904_bankfull_final.sh, SPECIES=human); needs the labelled reference store
# inputs: REF = 63 per-class pooled references, STORE = the 84,601 labelled cells,
# CLAB = cell -> label for every cell, TRIDS/TRLAB = a balanced training draw.
# The driver spreads the pairwise resolver calibration over 8 SLURM jobs with
# --resolver-stride k/8 into the same --resolver-cache; this is the one-job form.
LADDER=0,4096,16384,65536,262144; T=24; SEED=20260828

methscope mrmp-build --bank --resolvers all --min-pattern-cpgs 2000 \
  --anneal-min-seg 10000,3000,1000 --max-depth 24 \
  --resolver-cache $B/cache \
  --cell-store $STORE --cell-labels $CLAB --calib-threads $T \
  --force $REF $B/bank.mrmp                  # _lite: --min-pattern-cpgs 500 --resolvers 100

yame subset -l $TRIDS -o $B/train.cg $STORE
yame index -s $TRIDS $B/train.cg
methscope classify-featurize --threads $T --satellite-contrast replace \
  --seed $SEED -b --sample $LADDER --reps 1 \
  -l $TRLAB -o $B/train.msfm $B/train.cg $B/bank.mrmp

methscope classify-train --threads $T --data $B/train.msfm \
  -o $B/bank.clfx                            # constrained defaults: max-depth 4, colsample 0.4
                                             # _lite: add -n 100 (a 100-round booster)
