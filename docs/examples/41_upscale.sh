#!/bin/bash
## title: Impute
# impute the whole genome from it — fractions, the default
methscope fetch -c hg38/models/hg38_10k1.updecx
methscope upscale -o human_hg38_test_reconstructed.cg hg38_10k1.updecx human_hg38_test.cg
yame summary human_hg38_test_reconstructed.cg
