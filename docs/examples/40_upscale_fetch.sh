#!/bin/bash
## title: Impute
# a scratch dir for the outputs
mkdir -p ~/tmp/methscope && cd ~/tmp/methscope

# what goes in: a sparse methylome
methscope fetch -c hg38/data/human_hg38_test.cg
yame summary human_hg38_test.cg
