#!/bin/bash
## title: Train your own classifier
## norun: the recorded build of hg38_celltype.clfx; needs a labelled reference store
methscope mrmp-build --qfilter 0.30,0.70 --delta-mean-top 20000 \
  --min-segregating 20000 --satellite-n 5 --force $M/ref.cg $W/chain.mrmp

methscope classify-featurize --threads 16 --rank-features replace \
  --seed 20260810 -b --sample "$LADDER" --reps 1 \
  -l $M/train.labels -o $W/train.msfm $M/train.cg $W/chain.mrmp

methscope classify-train --threads 16 --data $W/train.msfm \
  -o $W/hg38_celltype_tree.clfx
