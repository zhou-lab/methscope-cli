#!/bin/bash
set -euo pipefail

# XGB_PREFIX points the Makefile at the build env's include/ and lib/ for
# libxgboost. The Makefile bakes -Wl,-rpath,$XGB_PREFIX/lib; conda-build's
# post-processing relocates that to an $ORIGIN-relative RPATH, so the installed
# binary finds libxgboost without activating any env.
#
# The Makefile declares CFLAGS with ?=, so conda's CFLAGS wins; re-add the
# -std=gnu99 the sources expect, plus CPPFLAGS (which the Makefile never reads).
export CFLAGS="${CFLAGS:-} ${CPPFLAGS:-} -std=gnu99"

# LDFLAGS must be passed on the command line: the Makefile assigns it outright
# (LDFLAGS = -L... -Wl,-rpath,...), which would otherwise discard conda's
# sysroot and rpath settings. CFLAGS is deliberately NOT passed this way --
# command-line variables propagate into the YAME sub-make and would clobber its
# own include paths.
make -j"${CPU_COUNT:-1}" CC="${CC}" XGB_PREFIX="${PREFIX}" \
     LDFLAGS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib ${LDFLAGS:-}"

# Not `install -D`: that is a GNU coreutils extension and BSD/macOS install
# rejects it.
mkdir -p "${PREFIX}/bin"
install -m 755 methscope "${PREFIX}/bin/methscope"
