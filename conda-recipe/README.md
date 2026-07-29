# Conda recipe for methscope

Builds the `methscope` binary and publishes it to the **zhou-lab** channel.

This is one of two recipes in the repo. See [`../bioconda-recipe`](../bioconda-recipe)
for the bioconda submission, which builds the same binary from a release
tarball under the package name `methscope`.

## Install (end users)

`libxgboost` comes from conda-forge, so both channels are needed:

```sh
conda install -c zhou-lab -c conda-forge methscope
```

## Build & upload (maintainers)

Requires `conda-build` and `anaconda-client`:

```sh
conda install -n base conda-build anaconda-client

# The recipe builds the checked-out tree, so YAME must be present
git submodule update --init --recursive

# Build (linux-64 + osx-arm64; Windows is skipped)
conda build -c zhou-lab -c conda-forge conda-recipe/

# Upload the built package to the zhou-lab channel
anaconda login
anaconda upload -u zhou-lab $(conda build -c zhou-lab -c conda-forge conda-recipe/ --output)
```

## Automated builds (CI)

`.github/workflows/conda-build.yml` runs `conda build` on every push and PR
across **linux-64** and **osx-arm64**, uploading each package as a build
artifact. On a `vX.Y.Z` tag it also publishes to the `zhou-lab` channel, which
needs an `ANACONDA_TOKEN` repository secret (an anaconda.org token with upload
rights to the org). Cutting a release is then:

```sh
# bump METHSCOPE_VERSION in src/methscope.h only, commit
git tag -a v0.4 -m "methscope 0.4" && git push origin v0.4
```

Do **not** add a version to `meta.yaml`: it reads `METHSCOPE_VERSION`, which CI
sets from the tag name, so the published version cannot drift from the release
being built. A hardcoded one silently survived the `v0.2` tag once, and CI
rebuilt-and-overwrote `0.1.1` instead of publishing `0.2`.

Before uploading, the workflow checks that a *built artifact* carries the tag's
version -- a stronger guard than comparing the tag against a version literal,
since it verifies what will actually be published rather than what was declared.

`src/methscope.h` has no Makefile dependency, so clear `src/*.o` after bumping
or the binary keeps reporting the old version.

## Notes

- The recipe uses `source: path: ..` rather than a release tarball because
  GitHub's auto-generated tag archives do **not** contain the pinned YAME
  submodule. A release build is: check out the `vX.Y.Z` tag, run
  `git submodule update --init --recursive`, then `conda build`. Keep
  `METHSCOPE_VERSION` in `src/methscope.h` (meta.yaml now reads the tag) in
  sync with the tag.
- The only external dependency is `libxgboost` (conda-forge). YAME and htslib
  are linked statically from the submodule, so they are not conda deps.
- The Makefile bakes `-Wl,-rpath,$XGB_PREFIX/lib`; conda-build rewrites that
  to an `$ORIGIN`-relative RPATH, so the installed binary resolves
  `libxgboost` without activating the build env.
