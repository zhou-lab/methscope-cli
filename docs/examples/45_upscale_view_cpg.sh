#!/bin/bash
## title: Single-CpG view
# name the rows, then ask for a region small enough that one column is one CpG
cat human_hg38_test.truth.cg human_hg38_test.cg human_hg38_test_calls.cg > three_calls.cg
yame index -s <(printf 'truth\ninput\ncalls\n') three_calls.cg
yame hprint -c -r chr1:957686-959018 -l 8 three_calls.cg
