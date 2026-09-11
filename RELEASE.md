# Cutting a methscope release

One checklist, top to bottom. Everything downstream of step 5 is automated by
`.github/workflows/conda-build.yml`; steps 1-4 are the ones a human gets wrong.
Recipe-level detail lives in [`conda-recipe/README.md`](conda-recipe/README.md)
and [`bioconda-recipe/README.md`](bioconda-recipe/README.md) — this file is the
order of operations, not a duplicate of those.

Versions are `vX.Y`. The tag is the single source of truth: CI derives the conda
package version from it, so nothing else may carry a version literal.

This covers the **binary**. Publishing **models** is a separate procedure with
its own tag series and its own registry pin — see [`MODELS.md`](MODELS.md). A
code release does not require a model tag, or the reverse.

## 1. Bump the YAME submodule

A methscope release pins one YAME release. Check what upstream has, then move
the submodule to that tag (not to a branch — the pin must be a tag commit):

```sh
cd YAME
git fetch --tags origin
git tag --sort=-v:refname | head -3      # what is available upstream
git log --oneline vOLD..vNEW             # what you are taking on
git diff --stat vOLD vNEW -- '*.h'       # API churn that could break the build
git checkout vNEW
cd ..
```

Read the header diff before trusting the build: YAME's public surface is
`YAME/src/cdata.h` + `cfile.h`, and a silent signature change compiles here only
because methscope includes those headers directly.

## 2. Bump `METHSCOPE_VERSION`

`src/methscope.h` holds the only version literal in the source:

```sh
sed -i 's/#define METHSCOPE_VERSION "OLD"/#define METHSCOPE_VERSION "NEW"/' src/methscope.h
```

Do **not** put a version in either `meta.yaml`. `conda-recipe/meta.yaml` reads
`METHSCOPE_VERSION`, which CI sets from the tag name; a hardcoded literal once
survived the `v0.2` tag and CI rebuilt-and-overwrote `0.1.1` instead.

## 3. Rebuild, and check both version numbers

```sh
make -j4 XGB_PREFIX=$HOME/conda_envs/methscope
./methscope --version          # must show the NEW methscope AND the NEW yame
```

That one line is the gate — if either half is stale, the tag would ship a
mislabelled binary, and nothing downstream catches it: CI takes the package
version from the tag name, so conda would publish a correctly-named package
around a binary that lies about itself.

The Makefile compiles with `-MMD -MP`, so `src/*.o` correctly depend on
`src/methscope.h` and on YAME's `yame_version.h`, and a plain `make` picks up
both bumps. That was added right after v0.7; up to and including v0.7 it was
missing, and a version bump relinked a binary still reporting the old numbers —
which is how v0.7 nearly shipped. On a tree older than that, or whenever you are
unsure, `make clean` first.

## 4. Test

```sh
make check-updec2 XGB_PREFIX=$HOME/conda_envs/methscope

## the full documented-workflow gate (~2 min): runs every runnable code block
## on docs/index.html verbatim, in a sandboxed HOME
YAME_DATA_HOME=/mnt/isilon/zhou_lab/projects/20191221_references/YAME \
  python3 ~/repo/labjournal/zhouw3/2026/20260824_docs_runthrough.py
```

The docs harness puts `~/repo/YAME` on `PATH`, so rebuild that checkout too if it
is behind the tag you just pinned, or the gate tests the wrong yame.

**Every runnable block must pass.** The page carries no fail-by-design block;
skips are declared in the markup (see 4b), so the tally follows the page. Any
failure blocks the release.

**Read the block bodies, not just the tally.** The harness scores a block by its
*last* command's exit code, so a failure anywhere earlier in a multi-command
block is invisible. That is exactly how the v0.7 `upscale` regression shipped
past a green gate: the failing `upscale` was followed by a `yame summary` that
exits 0 on the empty file `upscale` had just left behind. When a release touches
a read path, spot-check `logs3/block<NN>.err` for the blocks that exercise it.

**Skips come from the page, not the harness.** They used to be hardcoded block
indices, which shifted whenever a block was added or removed — removing the Train
tab left the list pointing at a block that no longer existed while the GPU
`sbatch` illustration became runnable. They are now `data-norun` attributes in
the HTML; see 4b.

## 4b. Illustrative blocks must still match the lab record

Blocks the harness does not run carry `data-norun="<reason>"` on their `<pre>`;
the page styles them distinctly and the harness derives its skip set from that
attribute, so the two can never disagree about what is runnable. Nothing tests
their *contents*, which is exactly why they rot — check them by hand:

For each `data-norun` block, open the labjournal driver it cites and compare the
commands token by token, not by eye. On 20260910 the page's `upscale-train`
illustration had invented `--mrmp wg.mrmp` and dropped `--patterns 500
--features scalar` — the two flags that fix the shipped decoder's shape
(`input_dim 501`) — so a reader following it would have trained something else.
A card written the same day cited the wrong driver entirely, naming the bank
pipeline as the source of a model built by the tree pipeline.

```sh
grep -n 'data-norun' docs/index.html      # every block that needs checking
```

Each such block must name its driver in the surrounding prose, by repo-relative
path, so the check has somewhere to start. If a driver changed since the model
shipped, the page follows the driver that ACTUALLY BUILT the shipped artifact,
not the current one.

## 5. Commit, tag, push

```sh
git add YAME src/methscope.h
git commit -m "release vX.Y: yame vNEW"
git tag -a vX.Y -m "methscope X.Y"
git push origin main --follow-tags
```

Pushing the tag is the publishing act — CI builds linux-64 + osx-arm64, checks
the glibc floor, uploads to the **zhou-lab** channel, and creates a GitHub
Release carrying the `make dist` tarball (GitHub's own tag archives omit the
submodule, so bioconda needs that asset).

## 6. Verify what actually got published

Do not trust the green check; ask the channel:

```sh
curl -s https://api.anaconda.org/package/zhou-lab/methscope |
  python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["latest_version"], [(f["version"], f["attrs"]["subdir"]) for f in d["files"]][-2:])'
```

Both `linux-64` and `osx-arm64` must be present at the new version. Then install
into a throwaway env and confirm the binary agrees:

```sh
conda create -y -n mstest --override-channels -c zhou-lab -c conda-forge methscope
## Run the env's binary by ABSOLUTE PATH. `conda run -n mstest methscope` is not
## a valid check here: /mnt/isilon/zhoulab/labbin is ahead of the env on PATH,
## so it reports whatever labbin currently holds -- which during a release is
## still the PREVIOUS version, making a correct package look like a failed one.
## (v0.8 hit exactly this and looked like it had published a 0.7 binary.)
E=$(conda env list | awk '$1=="mstest"{print $2}')
"$E/bin/methscope" --version      # must be the version you just tagged
conda env remove -y -n mstest
```

## 6b. Verify the CUDA package separately — CI will not tell you

`methscope-cuda` is a **second conda package**, linux-64 only, built from
[`conda-recipe-cuda/`](conda-recipe-cuda/) with `CUDA=1`. It installs its binary
as `methscope-cuda`, so it coexists with `methscope` rather than replacing it;
only `upscale-train` needs it, everything else is pure C.

**Its CI step carries `continue-on-error: true`** — deliberately, so an optional
GPU package can never block the main publish. The consequence is that it can
fail, or silently not publish, and the release still goes green. Nothing else in
this document would notice. So check the channel yourself:

```sh
curl -s https://api.anaconda.org/package/zhou-lab/methscope-cuda |
  python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["latest_version"])'

conda create -y -n mscuda --override-channels -c zhou-lab -c conda-forge methscope-cuda
E=$(conda env list | awk '$1=="mscuda"{print $2}')
"$E/bin/methscope-cuda" --version        # absolute path, per the step 6 note
conda env remove -y -n mscuda
```

It must report the version you just tagged. `--version` and `-h` need no GPU, so
this check runs on any node; only an actual `upscale-train` needs a device.

**For local GPU work, install this package** rather than building from source —
`make CUDA=1` needs a CUDA toolkit (`module load cuda12.2/toolkit/12.2.2`), an
isolated worktree so it does not clobber the CPU binary, and it reproduces what
the recipe already publishes.

## 7. Refresh the lab binary

`/mnt/isilon/zhoulab/labbin/methscope` is what everyone on the HPC gets from
`PATH`, and nothing updates it automatically — it sat at 0.1.0 for three
releases. Install by **staging then renaming**, never by writing in place:
overwriting a running binary's inode segfaults live jobs, sometimes hours later.

```sh
cp methscope /mnt/isilon/zhoulab/labbin/.methscope.new
chmod 775 /mnt/isilon/zhoulab/labbin/.methscope.new
mv -f /mnt/isilon/zhoulab/labbin/.methscope.new /mnt/isilon/zhoulab/labbin/methscope
methscope --version           # from PATH, not ./
```

`mv` within one filesystem is an atomic rename: it gives the new file a new
inode, so jobs already running keep the old one and finish cleanly.

## 8. Bioconda (only for versions meant for public release)

Not every tag goes to bioconda. When one does, take the sha256 the release job
printed (or recompute it from the uploaded asset) and update `version` +
`sha256` in `bioconda-recipe/meta.yaml`, then mirror that change into
[bioconda-recipes#66997](https://github.com/bioconda/bioconda-recipes/pull/66997).
Keep `bioconda-recipe/build.sh` byte-identical to `conda-recipe/build.sh` — and
remember the PR branch is a **third** copy that drifts on its own. Diff all
three; the in-repo pair agreeing proves nothing about what the PR will build:

```sh
diff conda-recipe/build.sh bioconda-recipe/build.sh
gh api repos/zwdzwd/bioconda-recipes/contents/recipes/methscope/build.sh?ref=add-methscope \
  --jq .content | base64 -d | diff - bioconda-recipe/build.sh
```

That third diff is not hypothetical: on 20260910 the PR branch was still carrying
the 20260708 `build.sh`, two months behind the 20260722 rework that every release
since has actually built with.

## 9. Log it

Add a dated entry to the RESEARCH_LOG of
`~/repo/labjournal/zhouw3/2025/20251216_methscope.org`: the version, the YAME
pin, what the release contains, and the docs-gate result.
