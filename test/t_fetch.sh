#!/bin/bash
## `methscope fetch` runs YAME's fetch code over methscope's OWN compiled
## catalogue (src/registry.h, projected from the submodule's tag): it speaks
## as methscope, lists only methscope's directories, honours
## METHSCOPE_DATA_HOME, and says when a store is behind the binary. Offline:
## -h and -l read the compiled registry and the store manifest, never the net.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
export METHSCOPE_DATA_HOME=$d/store; unset YAME_DATA_HOME

## usage speaks as methscope and names its own store variable first
out=$("$MS" fetch -h 2>&1 || true)
case "$out" in *"methscope fetch"*) ;; *) echo "fetch -h does not say 'methscope fetch'"; exit 1;; esac
case "$out" in *'$METHSCOPE_DATA_HOME, $YAME_DATA_HOME'*) ;;
  *) echo "fetch -h store order does not put METHSCOPE_DATA_HOME first"; exit 1;; esac
case "$out" in *"yame fetch"*) echo "fetch -h leaks 'yame fetch'"; exit 1;; esac

## -l lists exactly the projected catalogue: model and example-data
## directories, no genome or array knowledgebase rows
list=$("$MS" fetch -l 2>/dev/null)
dirs=$(printf '%s\n' "$list" | tail -n +2 | cut -f1 | sort -u | tr '\n' ' ')
[ "$dirs" = "hg38/data hg38/models mm10/models " ] || { echo "fetch -l directories: '$dirs'"; exit 1; }
## A model the catalogue still carries, named exactly: the classifier the
## routing-tree pair was withdrawn IN FAVOUR of at models v10.
printf '%s\n' "$list" | grep -q "hg38_celltype_full.clfx" ||
  { echo "fetch -l lacks hg38_celltype_full.clfx"; exit 1; }
## ... and a withdrawn one must NOT reappear. `hg38_celltype.clfx` is a prefix
## of `hg38_celltype_full.clfx`, so this needs a whole-field match, not grep -q.
for gone in hg38_celltype.clfx mm10_celltype_brain.clfx; do
  printf '%s\n' "$list" | cut -f2 | grep -qx "$gone" &&
    { echo "fetch -l still offers the withdrawn $gone"; exit 1; }
done
:

## --version carries the model tag the catalogue pins
v=$("$MS" --version)
case "$v" in *"models v"[0-9]*) ;; *) echo "--version lacks the models tag: $v"; exit 1;; esac
tag=$(printf '%s\n' "$list" | awk -F'\t' 'NR>1 && $1 == "hg38/models" {print $3; exit}')
case "$v" in *"models $tag)"*) ;; *) echo "--version tag '$v' != registry tag '$tag'"; exit 1;; esac

## an empty store is silent; a store filled at a tag this binary has never
## heard of is reported as ahead, and the advice names methscope, not yame
[ -z "$("$MS" fetch -l 2>&1 >/dev/null)" ] || { echo "empty store produced a state line"; exit 1; }
mkdir -p "$d/store/hg38/models"
echo "0000000000000000000000000000000000000000000000000000000000000000  nothing.clfx" \
  > "$d/store/hg38/models/SHA256SUMS"
msg=$("$MS" fetch -l 2>&1 >/dev/null || true)
case "$msg" in *"[methscope fetch] hg38/models"*) ;; *) echo "foreign-tag store not reported: '$msg'"; exit 1;; esac
case "$msg" in *"yame fetch"*) echo "advice names yame: '$msg'"; exit 1;; esac

## a bad target is an error, not a silent exit 0
if "$MS" fetch no/such/thing >/dev/null 2>&1; then echo "fetch of a bad name exited 0"; exit 1; fi
echo "fetch: usage, catalogue, version tag, store state, bad target"
