#!/bin/bash
## title: Impute
# or 0/1 calls, for anything that wants a set rather than a level
methscope upscale --binary -o human_hg38_test_calls.cg hg38_10k1.updecx human_hg38_test.cg
yame summary human_hg38_test_calls.cg
