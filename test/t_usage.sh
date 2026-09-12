#!/bin/bash
## Every subcommand answers -h, and every subcommand refuses a bad invocation
## the same way: a message on stderr and a non-zero exit. The contract is that
## exit 0 with nothing done is never a valid outcome -- a caller scripting
## methscope has only the exit code to go on.
set -euo pipefail
MS=${MS:?export MS=/path/to/methscope}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT

## the list the binary itself advertises, so a new subcommand is covered the
## day it is added rather than the day someone remembers to edit this file
## The banner groups commands under headings and pads the name column, so a
## command line is two leading spaces, the name, then two-or-more spaces and a
## capitalised description. Headings have no leading spaces, so they do not match.
## `|| true`: the bare banner is a USAGE message and exits non-zero, which
## set -e plus pipefail would otherwise treat as the script failing -- silently,
## before the next line, which is how this test first appeared to do nothing.
cmds=$("$MS" 2>&1 | awk '/^  [a-z][a-z0-9-]+  +[A-Z]/ {print $1}' | sort -u || true)
[ -n "$cmds" ] || { echo "could not read the subcommand list from the banner"; exit 1; }

n=0
for c in $cmds; do
  n=$((n + 1))
  ## -h must print usage and must not be silent
  if ! out=$("$MS" "$c" -h 2>&1); then
    case "$out" in *[Uu]sage*) ;; *) echo "$c -h failed without printing usage: $out"; exit 1;; esac
  fi
  case "$out" in
    *[Uu]sage*|*usage*) ;;
    *) echo "$c -h printed no usage line"; exit 1;;
  esac
done
[ "$n" -ge 10 ] || { echo "only $n subcommands found; the banner parse is wrong"; exit 1; }

## an unknown subcommand is an error, not a silent success
if "$MS" definitely-not-a-subcommand >/dev/null 2>&1; then
  echo "an unknown subcommand exited 0"; exit 1
fi

## a subcommand given a file that does not exist must fail, not exit 0
for c in inspect; do
  if "$MS" "$c" "$d/nope.mrmp" >/dev/null 2>&1; then
    echo "$c on a missing file exited 0"; exit 1
  fi
done

## `fetch` is live again (its own catalogue, t_fetch.sh); piped, the bare
## form dumps that catalogue as TSV and must never point users at yame
out=$("$MS" fetch 2>&1 </dev/null || true)
case "$out" in
  *retired*|*"yame fetch"*) echo "methscope fetch still reads as retired: $out"; exit 1;;
esac
printf '%s\n' "$out" | grep -q "hg38/models" || { echo "bare fetch (piped) did not list the catalogue"; exit 1; }
echo "ok: $n subcommands answer -h; bad invocations refuse"
