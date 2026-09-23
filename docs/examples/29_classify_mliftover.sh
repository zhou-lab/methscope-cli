#!/bin/bash
## title: Lift a model onto an array
# the platform's probe list and coordinates, and the genome's CpG rows
methscope fetch -c MSA/MSA.ordering.tsv.gz MSA/MSA.hg38.coord.tsv.gz hg38/cpg_nocontig.cr
methscope mliftover --to MSA --ordering MSA.ordering.tsv.gz \
  --coord MSA.hg38.coord.tsv.gz --cr cpg_nocontig.cr \
  hg38_sex.clfx -o hg38_sex.MSA.clfx
methscope inspect hg38_sex.MSA.clfx | head -3
