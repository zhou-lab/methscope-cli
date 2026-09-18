#!/bin/bash
## title: Per-pattern averages
# the same numbers as one row per record, one column per pattern
methscope mrmp-summary --wide human_hg38_40_celltypes_chr20.cg chr20_40celltypes.mrmp \
  -o chr20_patterns_wide.tsv
head -3 chr20_patterns_wide.tsv | cut -f1-4
