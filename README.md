<h1 align="center">MethScope</h1>

<p align="center">
<a href="https://github.com/zhou-lab/methscope-cli/actions/workflows/conda-build.yml"><img alt="build" src="https://github.com/zhou-lab/methscope-cli/actions/workflows/conda-build.yml/badge.svg"></a>
<a href="https://anaconda.org/zhou-lab/methscope"><img alt="conda" src="https://img.shields.io/conda/vn/zhou-lab/methscope?label=conda"></a>
<a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-BSD--2--Clause%20(academic)%20%2F%20commercial-blue.svg"></a>
<a href="scripts/coverage.sh"><img alt="coverage" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fzhou-lab.github.io%2Fmethscope-cli%2Fcoverage.json"></a>
<a href="https://zhou-lab.github.io/methscope-cli/"><img alt="docs" src="https://img.shields.io/badge/docs-online-blueviolet"></a>
</p>

Pure-C command-line tool for ultra-fast analysis of sparse DNA methylomes via
Most Recurrent Methylation Pattern (MRMP) encoding. MethScope runs the whole
path — query `.cg` + MRMP reference → cell×pattern feature matrix → XGBoost
cell-type prediction / NNLS deconvolution — as one self-contained binary,
with no interpreter and no runtime beyond `libxgboost`.

It builds [YAME](https://github.com/zhou-lab/YAME) as a static library
(`libyame.a`) for all `.cg/.cm` I/O and the `summary` computation, and links
`libxgboost` for inference.

**Documentation, with runnable examples for every command:
<https://zhou-lab.github.io/methscope-cli/>**

## Install

```sh
conda install -c zhou-lab -c conda-forge methscope
```

## Quick start

```sh
mkdir -p ~/tmp/methscope && cd ~/tmp/methscope
methscope fetch -c hg38/models/hg38_celltype_lite.clfx hg38/data/human_hg38_celltypes.cg
methscope classify hg38_celltype_lite.clfx human_hg38_celltypes.cg
```

`methscope fetch` with no arguments browses the catalogue. Pretrained models
live on HuggingFace ([zhou-lab/methscope](https://huggingface.co/zhou-lab/methscope));
the query `.cg` fixtures live in
[methscope_data](https://github.com/zhou-lab/methscope_data).

The catalogue and the model tag are compiled into the binary — `methscope
--version` prints <!-- version:begin -->`(yame v1.57, models v12)`<!-- version:end --> — so a release fetches exactly the
models it documents. The store is shared with the other zhou-lab tools
(`yame`, `kycg`) by convention, not by dependency.

## Build from source

```sh
git clone --recurse-submodules https://github.com/zhou-lab/methscope-cli.git
cd methscope-cli
conda create -n methscope -c conda-forge libxgboost   # the one external dependency
conda activate methscope
make                             # or: make XGB_PREFIX=/path/to/env
```

The binary records an rpath to `$XGB_PREFIX/lib`, so at runtime the conda env
that provided `libxgboost` must be on the library path (activating it is
enough). `make CUDA=1 CUDA_HOME=/path/to/cuda CUDA_ARCH=sm_80` adds the GPU
backend for `upscale-train`.

## Development

`make test` runs the test suites and the generator checks offline. The
documented-workflow gate (`make test-docs`) runs every example on the page and
needs the network once. `test/parity.sh` compares `classify` probabilities
against the R `PredictCellType`; it needs an R checkout and is not part of
`make test`.

Releases are cut from a checklist kept in the lab journal rather than here: it
covers the YAME submodule pin, the version bump, the tag, conda and the shared
lab binary, and it names internal paths that would mean nothing outside the lab.

## License

Use of this software is available to academic and non-profit institutions for
research purposes under the 2-Clause BSD License; for use or transfers to
commercial entities, inquire with Dr. Wanding Zhou at <zhouw3@chop.edu>.
See `LICENSE` for the full terms.
Copyright (C) 2025-present The Children's Hospital of Philadelphia.

Vendored: `src/nnls.c` — Lawson–Hanson NNLS (C. Lawson & R. Hanson, JPL/SIAM;
[netlib lawson-hanson](https://www.netlib.org/lawson-hanson/)), f2c-translated,
self-contained.
