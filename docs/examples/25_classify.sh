#!/bin/bash
## title: Classify
# a scratch dir for the outputs
mkdir -p ~/tmp/methscope && cd ~/tmp/methscope

# 4 Loyfer cells vs the shipped 33-type classifier
methscope fetch -c hg38/data/human_hg38_celltypes.cg
methscope fetch -c hg38/models/hg38_celltype.clfx
methscope classify hg38_celltype.clfx human_hg38_celltypes.cg
