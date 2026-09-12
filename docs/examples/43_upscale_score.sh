#!/bin/bash
## title: Score against the truth
methscope fetch -c hg38/data/human_hg38_test.truth.cg
paste <(yame rowsub -I 1_10000 human_hg38_test_calls.cg | yame unpack -a - | cut -f4) <(yame rowsub -I 1_10000 human_hg38_test.truth.cg | yame unpack -a - | cut -f4) | awk '$2!=2{t++;if($1==$2)c++} END{printf "%d/%d (%.1f%%)\n",c,t,100*c/t}'
