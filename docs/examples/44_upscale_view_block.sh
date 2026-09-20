#!/bin/bash
## title: Whole-block view
cat human_hg38_test.truth.cg human_hg38_test.cg human_hg38_test_reconstructed.cg > three.cg
yame index -s <(printf 'truth\ninput\nreconstructed\n') three.cg
methscope fetch hg38/cpg_nocontig.cr
yame hprint -g -r chr1:921649-1151482 -w 60 three.cg
