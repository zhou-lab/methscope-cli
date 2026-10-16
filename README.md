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

## System requirements

**Operating systems.** Linux x86-64 (glibc 2.17 or newer) and macOS on Apple
silicon (macOS 11 or newer). Conda packages are built for `linux-64` and
`osx-arm64`. Tested on macOS 26.6 (Apple M2 Pro, 10 cores, 16 GB RAM) and
Red Hat Enterprise Linux 9.1 (glibc 2.34; Intel Xeon Platinum 8380, 4 cores and
8 GB allocated).

**Software dependencies.** All are installed by conda with the package.

| dependency   | version                         | role                                   |
|--------------|---------------------------------|----------------------------------------|
| libxgboost   | 3.4.2 tested (conda-forge)      | classifier inference, the one external runtime library |
| zlib         | 1.3.2 tested                    | `.cg` / `.cm` compression              |
| libcurl      | 8.22.0 tested                   | `methscope fetch`                      |
| yame         | >= 1.50 (1.59 tested)           | the `yame` binary used by the Upscale and MRMP examples |

YAME is compiled into the binary from a pinned submodule; `methscope --version`
prints the YAME version and the model tag; at this commit it prints
<!-- version:begin -->`(yame v1.59, models v12)`<!-- version:end -->.

**Versions tested.** methscope v1.1.1 with models v12, libxgboost 3.4.2 and
yame 1.59, on macOS 26.6 (osx-arm64) and RHEL 9.1 (linux-64).

**Hardware.** No non-standard hardware is required. Classification,
deconvolution and upscaling run on a standard laptop or desktop CPU, in pure C,
with no GPU, CUDA or BLAS. A GPU is optional and is used only by
`upscale-train`, to train a new decoder faster, through the separate
`methscope-cuda` package (linux-64 only). Whole-genome upscaling uses about
3 GB of memory; the whole-genome human decoder is a 2.7 GB download.

## Installation

```sh
conda install -c zhou-lab -c conda-forge methscope
methscope --version
```

Typical install time on a standard desktop or server: under 1 min into a
fresh conda environment with an empty package cache (15 s on a Mac M2 Pro,
43 s on a RHEL 9.1 Xeon node), not counting the installation of conda itself.

## Demo

The demo classifies four cells of known type with the shipped human
cell-type classifier; the two downloads total about 27 MB.

```sh
mkdir -p ~/tmp/methscope && cd ~/tmp/methscope
methscope fetch -c hg38/data/human_hg38_celltypes.cg
methscope fetch -c hg38/models/hg38_celltype_lite.clfx
methscope classify hg38_celltype_lite.clfx human_hg38_celltypes.cg
```

Expected output (`confidence` is the probability of the called class;
`certainty` is how decisively the evidence chose it over the alternatives):

```
cell             prediction_label  confidence  certainty
Oligodendrocyte  ODC                 0.999991   0.999971
Pancreas-Beta    Beta                0.999998   0.999992
Blood-NK         NK CD16             0.925882   0.935428
Blood-Monocytes  Mono                0.949018   0.951083
```

All four calls agree with the known cell types. Expected run time on a normal
desktop or server: under 10 s in total, with peak memory under 1 GB. On a Mac
M2 Pro the two downloads took 2.6 s and `classify` 1.3 s; on a RHEL 9.1 Xeon
node, 2.3 s and 3.3 s on one thread, with 581 MB peak memory.

The documentation page gives a runnable example with its expected output for
every command, including deconvolution of simulated mixtures with exact known
proportions (`hg38/data/human_hg38_immune_mixture.cg`) and genome-wide
upscaling; `make test-docs` runs all of them.

## Running on your own data

Convert your methylation calls to a YAME `.cg` store (see the
[YAME documentation](https://github.com/zhou-lab/YAME)), fetch the model for your task
with `methscope fetch`, and run `classify`, `deconv` or `upscale` on the store.
Each command's options are documented at
<https://zhou-lab.github.io/methscope-cli/> and by `methscope <command> -h`.

`methscope fetch` with no arguments browses the catalogue. Pretrained models
live on HuggingFace ([zhou-lab/methscope](https://huggingface.co/zhou-lab/methscope));
the query `.cg` fixtures, including the demo store, live in
[methscope_data](https://github.com/zhou-lab/methscope_data). The download
store is shared with the other zhou-lab tools (`yame`, `kycg`) by convention,
not by dependency.

## Reproducing the tests and examples

Every example on the [documentation page](https://zhou-lab.github.io/methscope-cli/)
shows its expected output, with the models pinned by tag and digest. `make test`
and `make test-docs` rerun them all (see Development).

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
The pretrained models on [HuggingFace](https://huggingface.co/zhou-lab/methscope)
are under the same academic BSD-2 terms; the test data in
[methscope_data](https://github.com/zhou-lab/methscope_data) are under CC BY 4.0.

Vendored: `src/nnls.c` — Lawson–Hanson NNLS (C. Lawson & R. Hanson, JPL/SIAM;
[netlib lawson-hanson](https://www.netlib.org/lawson-hanson/)), f2c-translated,
self-contained.
