#!/bin/bash
## title: Per-pattern averages
# One mean methylation value per pattern per record, every set of the chain in
# one pass over the query. Query first, set second.
methscope mrmp-summary human_hg38_40_celltypes_chr20.cg chr20_40celltypes.mrmp \
  -o chr20_patterns.tsv
head -4 chr20_patterns.tsv
