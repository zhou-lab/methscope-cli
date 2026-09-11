#!/bin/bash
## Run every test/t_*.sh against the built binary and tally.
##
## Each test is self-contained: it packs its own fixtures with yame from inline
## text, exits non-zero on failure, and prints what went wrong. Nothing here
## needs the network or the model store, so it runs in the conda build on every
## platform. Tests that DO need the shared store skip cleanly when
## YAME_DATA_HOME is unset -- a laptop without it must not fail the suite.
##
##   make test        # or: MS=/path/to/methscope bash test/run.sh
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
## Resolved to ABSOLUTE paths: a test that cd's into its own temp dir (and
## several must, to exercise the commands that write beside their input) would
## otherwise lose a relative ./methscope and fail with 127 and no message.
MS=${MS:-$here/../methscope}; YAME=${YAME:-$here/../YAME/yame}
case $MS   in /*) ;; *) MS=$(cd "$(dirname "$MS")"   && pwd)/$(basename "$MS");;   esac
case $YAME in /*) ;; *) YAME=$(cd "$(dirname "$YAME")" && pwd)/$(basename "$YAME");; esac
export MS YAME
if [ ! -x "$MS" ];   then echo "no binary at $MS (run make first)" >&2; exit 2; fi
if [ ! -x "$YAME" ]; then echo "no yame at $YAME (run make -C YAME first)" >&2; exit 2; fi

## The binary links libxgboost through an rpath into the conda env that built
## it; a test invoking it from elsewhere still needs that on the library path.
if [ -n "${XGB_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$XGB_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

pass=0; fail=0
for t in "$here"/t_*.sh; do
  name=$(basename "$t" .sh)
  if out=$(bash "$t" 2>&1); then
    pass=$((pass + 1)); echo "ok    $name"
  else
    fail=$((fail + 1)); echo "FAIL  $name"
    printf '%s\n' "$out" | sed 's/^/      /'
  fi
done
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
