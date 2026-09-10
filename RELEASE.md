# Cutting a methscope release

One checklist, top to bottom. Everything downstream of step 5 is automated by
`.github/workflows/conda-build.yml`; steps 1-4 are the ones a human gets wrong.
Recipe-level detail lives in [`conda-recipe/README.md`](conda-recipe/README.md)
and [`bioconda-recipe/README.md`](bioconda-recipe/README.md) — this file is the
order of operations, not a duplicate of those.

Versions are `vX.Y`. The tag is the single source of truth: CI derives the conda
package version from it, so nothing else may carry a version literal.

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
is behind the tag you just pinned, or the gate tests the wrong yame. Blocks 10
and 11 (the Train tab) fail by design — chr20 toy data has no training rows for
the root node. Any other failure blocks the release.

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
conda create -y -n mstest -c zhou-lab -c conda-forge methscope
conda run -n mstest methscope --version
conda env remove -y -n mstest
```

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
Keep `bioconda-recipe/build.sh` byte-identical to `conda-recipe/build.sh`:

```sh
diff conda-recipe/build.sh bioconda-recipe/build.sh
```

## 9. Log it

Add a dated entry to the RESEARCH_LOG of
`~/repo/labjournal/zhouw3/2025/20251216_methscope.org`: the version, the YAME
pin, what the release contains, and the docs-gate result.
