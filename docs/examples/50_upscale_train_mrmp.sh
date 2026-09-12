#!/bin/bash
## title: Train
# a scratch dir for the outputs
mkdir -p ~/tmp/methscope && cd ~/tmp/methscope

# the whole pipeline from scratch, on the 40-cell-type chr20 reference
# 1. define the features
methscope fetch -c hg38/data/human_hg38_40_celltypes_chr20.cg
methscope mrmp-build --flat --force human_hg38_40_celltypes_chr20.cg chr20.mrmp
