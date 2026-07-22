# Conda recipe for methscope-cli

Builds the `methscope` binary and publishes it to the **zhou-lab** channel.

## Install (end users)

`libxgboost` comes from conda-forge, so both channels are needed:

```sh
conda install -c zhou-lab -c conda-forge methscope-cli
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
# bump version in conda-recipe/meta.yaml and src/methscope.h, commit
git tag -a v0.2.0 -m "methscope-cli 0.2.0" && git push origin v0.2.0
```

The workflow guards that the tag matches the recipe version before uploading.

## Notes

- The recipe uses `source: path: ..` rather than a release tarball because
  GitHub's auto-generated tag archives do **not** contain the pinned YAME
  submodule. A release build is: check out the `vX.Y.Z` tag, run
  `git submodule update --init --recursive`, then `conda build`. Keep
  `version` in `meta.yaml` and `METHSCOPE_VERSION` in `src/methscope.h` in
  sync with the tag.
- The only external dependency is `libxgboost` (conda-forge). YAME and htslib
  are linked statically from the submodule, so they are not conda deps.
- The Makefile bakes `-Wl,-rpath,$XGB_PREFIX/lib`; conda-build rewrites that
  to an `$ORIGIN`-relative RPATH, so the installed binary resolves
  `libxgboost` without activating the build env.
