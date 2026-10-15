#!/bin/bash
## title: Deconvolve an MRMP embedding
# the same nine mixtures as an MRMP embedding, over the shipped
# classifier's MRMP chain, with each pattern's CpG count (--with-n)
methscope fetch -c hg38/models/hg38_celltype_lite.clfx
methscope classify-featurize --continuous-features --with-n \
  -o mixture.msfm human_hg38_immune_mixture.cg hg38_celltype_lite.clfx

# an .msfm query is recognised by its first bytes; no option needed
methscope deconv --report hg38_62celltypes.msdref mixture.msfm
