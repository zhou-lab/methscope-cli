#!/usr/bin/env python3
"""Run every runnable docs/examples/*.sh, in order, as a reader would.

This is the documented-workflow gate: the proof that the page's examples run
on the binary that ships. It is NOT part of `make test` -- it needs the
network once (about 1.7 GB from HuggingFace), ~1.6 GB of memory for the
deconvolution block, and a few minutes -- and it must never run in CI or a
conda build. `make test-docs` runs it; the release SOP runs that under sbatch.

Each script runs with `bash -eo pipefail` from one working directory, so a
block can use what an earlier block built, exactly as on the page. Scripts
with a `## norun:` line are skipped and reported.

The sandbox PERSISTS between runs (METHSCOPE_DOCS_SANDBOX, default
$TMPDIR/methscope-docs): fetched inputs stay, so the second run downloads
nothing and only re-verifies digests -- `methscope fetch -c` skips a present
file whose sha256 matches. Everything the examples BUILT is deleted before a
run, so `output exists` can never be the reason a block fails. The keep-list
is the compiled registry (`methscope fetch -l`): if the catalogue lists it,
it is an input.

PATH puts the working tree first: ./methscope and the submodule's YAME/yame,
so the gate tests the code in this checkout, never an installed copy. Point
METHSCOPE_DOCS_BIN at a DIRECTORY (a conda env's bin/) to test what a reader
installs instead.
"""
import glob, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "docs", "examples")
SANDBOX = os.environ.get("METHSCOPE_DOCS_SANDBOX") or os.path.join(tempfile.gettempdir(), "methscope-docs")
HOME = os.path.join(SANDBOX, "home")
WORK = os.path.join(HOME, "tmp", "methscope")   # where every example cd's to
LOGS = os.path.join(SANDBOX, "logs")
BIN = os.environ.get("METHSCOPE_DOCS_BIN") or (ROOT + ":" + os.path.join(ROOT, "YAME"))

def header(path):
    title, norun = "", None
    for line in open(path, encoding="utf-8"):
        if line.startswith("## title:"): title = line[9:].strip()
        elif line.startswith("## norun:"): norun = line[9:].strip()
        elif not line.startswith("#"): break
    return title, norun

def keep_list(env):
    out = subprocess.run(["methscope", "fetch", "-l"], env=env, capture_output=True, text=True, check=True).stdout
    return {line.split("\t")[4] for line in out.splitlines()[1:]}

def clear_outputs(keep):
    """Delete what the examples built; keep catalogued inputs (and their .idx)."""
    for name in os.listdir(WORK):
        p = os.path.join(WORK, name)
        if name in keep or (name.endswith(".idx") and name[:-4] in keep): continue
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

def main():
    for d in (WORK, LOGS): os.makedirs(d, exist_ok=True)
    env = dict(os.environ, HOME=HOME, PATH=BIN + ":" + os.environ["PATH"])
    env.pop("METHSCOPE_DATA_HOME", None)            # a reader has no store; -c fetches into WORK
    clear_outputs(keep_list(env))
    scripts = sorted(glob.glob(os.path.join(EXAMPLES, "*.sh")))
    results, t0 = [], time.time()
    for path in scripts:
        name = os.path.basename(path)[:-3]
        title, norun = header(path)
        if norun:
            results.append((name, "SKIP", norun)); print("%-32s SKIP   %s" % (name, norun)); continue
        with open(os.path.join(LOGS, name + ".log"), "w") as log:
            t1 = time.time()
            rc = subprocess.run(["bash", "-eo", "pipefail", path], cwd=WORK, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=10800).returncode
        state = "ok" if rc == 0 else "FAIL rc=%d" % rc
        results.append((name, state, title)); print("%-32s %-10s %s  (%ds)" % (name, state, title, time.time() - t1))
    failed = [r for r in results if r[1].startswith("FAIL")]
    skipped = [r for r in results if r[1] == "SKIP"]
    print("\n%d blocks, %d failed, %d skipped  (%ds; sandbox %s)" % (len(results), len(failed), len(skipped), time.time() - t0, SANDBOX))
    for name, state, _ in failed: print("  %s: %s -- see %s/%s.log" % (name, state, LOGS, name))
    sys.exit(1 if failed else 0)

if __name__ == "__main__": main()
