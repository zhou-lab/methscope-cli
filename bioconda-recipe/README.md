# Bioconda recipe for methscope

Mirror of the recipe submitted to
[bioconda-recipes#66997](https://github.com/bioconda/bioconda-recipes/pull/66997)
(`recipes/methscope/`). Kept in-repo so the recipe is versioned alongside the
source it builds; bioconda itself builds from its own copy, so edits here must
be pushed to that PR (or a follow-up one) to take effect.

## How this differs from `../conda-recipe`

The two recipes build the same binary for two different destinations, and the
differences are forced by the destination — they are not accidental drift.

|              | `../conda-recipe` (zhou-lab)   | `bioconda-recipe` (bioconda)          |
| ------------ | ------------------------------ | ------------------------------------- |
| package name | `methscope-cli`                | `methscope`                           |
| source       | `path: ..` (checked-out tree)  | release tarball `url:` + `sha256:`    |
| built by     | `.github/workflows/conda-build.yml` | bioconda CI on the PR            |
| platforms    | linux-64, osx-arm64            | linux-64, osx-64                      |

Bioconda does not allow `path:` sources — a recipe there must fetch a fixed,
hashed artifact. Since GitHub's auto-generated tag tarballs omit the pinned
YAME submodule, the `url:` points at a **self-contained release asset** built
by `make dist` (see the Makefile), which bundles the submodule and is
byte-reproducible so the `sha256:` can be independently verified.

## Version skew

`meta.yaml` here pins **0.1.0** because that is the version the open PR was
reviewed and built against. The repo is now at 0.1.1. Bumping bioconda means:
cut a `v0.1.1` release asset with `make dist`, upload it, then update `version`
+ `sha256` in the PR.

## Name

The bioconda package is `methscope`, not `methscope-cli`, so that it reads
naturally next to the R package (installed in R, not conda). Note both packages
install the same `bin/methscope`, so they will collide if installed into one
environment — use one channel or the other, not both.
