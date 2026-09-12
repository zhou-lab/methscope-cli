#!/bin/bash
## title: Deconvolve
# a scratch dir for the outputs
mkdir -p ~/tmp/methscope && cd ~/tmp/methscope

# nine known mixtures vs the shipped 62-type reference (1.5 GB)
methscope fetch -c hg38/data/human_hg38_immune_mixture.cg
methscope fetch -c hg38/models/hg38_62celltypes.msdref
methscope deconv hg38_62celltypes.msdref human_hg38_immune_mixture.cg
