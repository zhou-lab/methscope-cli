#!/bin/bash
## title: Build the MRMP
# a scratch dir for the outputs
mkdir -p ~/tmp/methscope && cd ~/tmp/methscope

# a 40-cell-type chr20 reference (39.5 MB)
methscope fetch -c hg38/data/human_hg38_40_celltypes_chr20.cg
methscope mrmp-build --top 500 human_hg38_40_celltypes_chr20.cg chr20_40celltypes.mrmp
