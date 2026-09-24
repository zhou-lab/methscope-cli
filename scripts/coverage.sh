#!/bin/sh
# scripts/coverage.sh -- line coverage of test/run.sh (+ test/nnls_check) over
# methscope's own sources.
#
#   scripts/coverage.sh            measure, print the table, rewrite docs/coverage.json
#   scripts/coverage.sh --check    measure and fail if the badge is stale by more
#                                  than $TOLERANCE points (default 2.0)
#
# gcov's own summary, counting EXECUTABLE lines. This is LINE coverage, not
# branch; branch coverage is lower and is the more telling number for the model
# codecs -- measure it before quoting it.
#
# YAME is excluded: it is a submodule with its own suite and its own badge, and
# its ~9k lines would swamp a number meant to say how well OUR code is tested.
#
# Everything happens in a scratch copy. An instrumented binary must never become
# the one in the repo: it is built -O0, writes .gcda beside itself on every run,
# and is far slower than the real thing.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
work=${TMPDIR:-$HOME/tmp}/methscope-coverage.$$
TOLERANCE=${TOLERANCE:-2.0}
XGB=${XGB_PREFIX:-${CONDA_PREFIX:-}}
[ -n "$XGB" ] || { echo "set XGB_PREFIX or activate the env with libxgboost" >&2; exit 2; }

trap 'rm -rf "$work"' EXIT
mkdir -p "$work"
( cd "$here" && git ls-files | tar -cf - -T - ) | ( cd "$work" && tar -xf - )
cp -r "$here/YAME" "$work/YAME"

cd "$work"
make -C YAME lib >/dev/null 2>&1
make CC="cc -O0 -g --coverage" LDFLAGS="--coverage" XGB_PREFIX="$XGB" -j"$(nproc)" >/dev/null 2>&1
export LD_LIBRARY_PATH="$XGB/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MS=./methscope YAME=YAME/yame XGB_PREFIX="$XGB" sh test/run.sh >/dev/null 2>&1 || true
python3 test/check_updec2.py ./methscope >/dev/null 2>&1 || true
## the NNLS harness links the instrumented src/nnls.o, so its runs count
make CC="cc -O0 -g --coverage" LDFLAGS="--coverage" XGB_PREFIX="$XGB" test/nnls_check >/dev/null 2>&1 &&
  ./test/nnls_check >/dev/null 2>&1 || true

python3 - "$TOLERANCE" "$here" "${1:-}" <<'PY'
import subprocess, re, glob, sys, json, os
tol, repo, mode = float(sys.argv[1]), sys.argv[2], sys.argv[3]
out = subprocess.run(["gcov", "-o", "src"] + glob.glob("src/*.c"),
                     capture_output=True, text=True).stdout
rows, tl, tc = [], 0, 0
for m in re.finditer(r"File '([^']+)'\nLines executed:([\d.]+)% of (\d+)", out):
    f, pct, n = m.group(1), float(m.group(2)), int(m.group(3))
    if not f.startswith("src/"):
        continue
    rows.append((pct, n, f)); tl += n; tc += n * pct / 100
pct = 100 * tc / tl
for p, n, f in sorted(rows):
    print("  %-28s %5d %6.1f%%" % (f, n, p))
print("coverage: %.1f%% of %d executable lines (gcov, line coverage: test/run.sh + nnls_check)"
      % (pct, tl))

badge = os.path.join(repo, "docs", "coverage.json")
if mode == "--check":
    if not os.path.exists(badge):
        print("no %s to check against" % badge); sys.exit(1)
    have = float(json.load(open(badge))["message"].rstrip("%"))
    if abs(have - pct) > tol:
        print("badge says %.1f%%, suite measures %.1f%% (tolerance %.1f)"
              % (have, pct, tol)); sys.exit(1)
    print("badge %.1f%% agrees with the suite within %.1f points" % (have, tol))
else:
    colour = "red" if pct < 50 else "orange" if pct < 75 else "green"
    json.dump({"schemaVersion": 1, "label": "coverage",
               "message": "%.1f%%" % pct, "color": colour},
              open(badge, "w"), indent=2)
    print("wrote %s" % badge)
PY
