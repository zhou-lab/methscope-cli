#!/bin/bash
## title: Install
## norun: installs packages, run it yourself
# conda — installs the `methscope` binary, and `yame` with it: methscope
# declares yame as a runtime dependency, so one install covers both. It is a
# separate package because methscope bundles YAME as a LIBRARY and ships no
# `yame` binary of its own, and the Upscale and MRMP examples below call that
# binary to index, subset and view .cg stores. (Models and the CpG coordinate
# reference come from `methscope fetch`; yame is for the stores themselves.)
conda install -c zhou-lab -c conda-forge methscope

# optional, linux-64 only: adds a `methscope-cuda` binary with the CUDA
# backend for `upscale-train`. Everything else — upscale, classify,
# deconv — is pure C and needs no GPU, so most users can skip this.
conda install -c zhou-lab -c conda-forge methscope-cuda

# or build from source (bundles YAME). libxgboost is the one external
# dependency: the Makefile reads $CONDA_PREFIX, so activate the env first
# or pass XGB_PREFIX=/path/to/env.
conda create -n methscope -c conda-forge libxgboost
conda activate methscope
git clone --recurse-submodules https://github.com/zhou-lab/methscope-cli
cd methscope-cli && make          # add CUDA=1 for the training backend
# the source build links libyame.a and emits no `yame` binary either; for the
# examples that call it, `make -C YAME` builds one at YAME/yame.
