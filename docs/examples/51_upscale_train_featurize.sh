#!/bin/bash
## title: Train
# 2. featurize the truth atlas -> sampling/truth msur
methscope upscale-featurize --reps 20 --sample 8000 human_hg38_40_celltypes_chr20.cg chr20.mrmp chr20.msur
