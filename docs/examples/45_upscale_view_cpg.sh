#!/bin/bash
## title: Single-CpG view
cat human_hg38_test.truth.cg human_hg38_test.cg human_hg38_test_calls.cg | yame rowsub -I 1_10000 - | yame hprint -c - | cut -c1621-1680
