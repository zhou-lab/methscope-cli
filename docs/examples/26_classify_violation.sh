#!/bin/bash
## title: Score with no trained model
# The chr20 MRMP from the Build step above, scored against the very reference
# it was built from: same row space by construction, and every cell must come
# back as itself, so the answer is checkable without a held-out set.
methscope classify --framework violation \
  chr20_40celltypes.mrmp human_hg38_40_celltypes_chr20.cg | head -4
