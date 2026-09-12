#!/bin/bash
## title: Unbundle / re-wrap
methscope unbundle hg38_sex.clfx
methscope bundle -m hg38_sex.mrmp -k logistic -o rewrapped.clfx hg38_sex.clf
cmp hg38_sex.clfx rewrapped.clfx && echo identical
