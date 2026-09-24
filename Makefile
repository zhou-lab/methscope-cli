# methscope-cli — pure-C CLI. Links YAME (static libyame.a) + libxgboost (conda).
CC ?= gcc
CFLAGS ?= -W -Wall -O3 -std=gnu99
PROG = methscope
PYTHON ?= python3

# Optional native CUDA backend (`make CUDA=1`). The normal build remains
# CPU-only and has no CUDA runtime dependency.
CUDA ?= 0
CUDA_HOME ?= $(if $(CUDA_ROOT),$(CUDA_ROOT),/usr/local/cuda)
CUDA_ARCH ?= sm_80
# A locally built binary targets one architecture (fast to compile).  A
# DISTRIBUTED one must be a fatbin or it fails outright on anything but that
# card, so the conda recipe passes CUDA_GENCODE with several targets plus a PTX
# fallback for architectures newer than any we compiled for.
CUDA_GENCODE ?= -arch=$(CUDA_ARCH)
# Extra nvcc flags (include paths etc.). conda-build needs this: nvcc comes from
# the BUILD prefix while the CUDA headers/libs come from the HOST prefix.
NVCCFLAGS ?=
NVCC ?= $(CUDA_HOME)/bin/nvcc

# --- YAME static library (built from the pinned submodule) ---------------
YAME_DIR = YAME
YAME_LIB = $(YAME_DIR)/libyame.a
HTSLIB   = $(YAME_DIR)/htslib/libhts.a

# --- libxgboost (conda-forge): provides c_api.h + libxgboost.{so,dylib} ----
# Activate the conda env that has `libxgboost` installed, or override XGB_PREFIX.
XGB_PREFIX ?= $(CONDA_PREFIX)

CFLAGS  += -Isrc -I$(YAME_DIR)/src -I$(YAME_DIR)/htslib -I$(XGB_PREFIX)/include
LDFLAGS  = -L$(XGB_PREFIX)/lib -Wl,-rpath,$(XGB_PREFIX)/lib
# libyame + htslib + zlib/pthread/curl/rt come from `yame-config --libs` -- the
# one authoritative answer for linking libyame.a (a build product of
# `make -C YAME lib`, so it is evaluated on the link line, not at parse time,
# and picks the same libcurl YAME itself was built against). methscope adds only
# libxgboost (+ the CUDA runtime under CUDA=1).
LIBS     = -lxgboost

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
CUDA_OBJ =

ifeq ($(CUDA),1)
  CUDA_OBJ = src/upunit_cuda.o
  LDFLAGS += -L$(CUDA_HOME)/lib -Wl,-rpath,$(CUDA_HOME)/lib
  LIBS += -lcublas -lcudart -lstdc++
endif

OBJ += $(CUDA_OBJ)

# --- header dependency tracking -------------------------------------------
# Kept OUT of CFLAGS on purpose: CFLAGS is reused on the link line, where -MMD
# would drop a stray methscope.d. -MMD lists every non-system header, so an
# object depends on src/methscope.h AND on YAME/src/yame_version.h; without
# this, bumping METHSCOPE_VERSION or the YAME submodule relinked a binary that
# still reported the OLD versions, and only `make clean` saved the release.
# -MP emits a phony target per header so a deleted/renamed header does not
# wedge the build with "No rule to make target".
DEPFLAGS = -MMD -MP
DEP = $(SRC:.c=.d)

.PHONY: all clean clean-all dist yame-lib yame-bin check-xgb check-updec2 test test-docs force-link registry docs inspect-snapshots

all: $(PROG)

check-updec2: $(PROG)
	$(PYTHON) test/check_updec2.py ./$(PROG)

## Command-level tests against the built binary. Self-contained: every test
## packs its own fixtures with yame from inline text, so this needs no network
## and no model store. Tests that DO want the shared store skip cleanly when
## YAME_DATA_HOME is unset.
test: $(PROG) yame-lib yame-bin test/nnls_check
	MS=./$(PROG) YAME=$(YAME_DIR)/yame XGB_PREFIX=$(XGB_PREFIX) bash test/run.sh
	./test/nnls_check
	./tools/make_registry.sh --check
	$(PYTHON) docs/build_models.py --check
	$(PYTHON) docs/build_examples.py --check
	$(PYTHON) docs/build_help.py --check
	$(PYTHON) docs/make_llms.py ./$(PROG) --check
	$(PYTHON) docs/build_readme.py ./$(PROG) --check

## The NNLS solver behind `deconv`, tested on its own: the command tests
## never make it backtrack (their references are well conditioned), so the
## harness feeds it near-collinear problems and checks the KKT conditions and
## an enumeration oracle. Links the same src/nnls.o the binary does, so an
## instrumented build (scripts/coverage.sh) counts it toward nnls.c.
test/nnls_check: test/nnls_check.c src/nnls.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lm

## docs/index.html is prose around three generated parts: the example blocks
## (docs/examples/*.sh, build_examples.py), the model table (YAME's
## assets.tsv, build_models.py) and the Reference tab (every subcommand's -h
## from THIS binary, build_help.py). `make test` checks all three.
## docs/llms.txt is the agent-facing reference and the FOURTH generated
## file: same binary, same catalogue rows as the model cards. It was the
## one nothing ran or checked, and it rotted for three weeks -- it told an
## agent `methscope fetch` was retired, listed the withdrawn v9 models, and
## documented a --flat the binary refuses. Now `make docs` writes it and
## `make test` checks it, like the other three.
## The README quotes `--version` to explain the compiled-in catalogue; that
## quote is generated too (docs/build_readme.py), because its three numbers
## move independently and it sat two YAME releases and a model tag behind.
## The model table on the docs Models tab, generated from the submodule's
## YAME/data/assets.tsv (one row per model, compiled into every tool) plus the
## registry for tag and size. A model is described once, there; the page is a
## projection of it. `docs/build_models.py --check` says whether the committed
## page is behind the TSV, and `make test` runs that check.
docs: $(PROG)
	$(PYTHON) docs/build_models.py
	$(PYTHON) docs/build_examples.py
	$(PYTHON) docs/build_help.py
	$(PYTHON) docs/make_llms.py ./$(PROG)
	$(PYTHON) docs/build_readme.py ./$(PROG)

## Recapture `methscope inspect` for every published model into
## docs/inspect_snapshots.txt, which build_models.py renders into the Models
## tab. This is the ONLY docs step that needs the artifacts -- they are 14 GB,
## so it is deliberately outside `docs` and `test`, which read the committed
## snapshot and stay offline. Run it after a model tag:
##   make inspect-snapshots MODELS=/path/to/the/models
MODELS ?= /mnt/isilon/zhou_lab/projects/20260726_HuggingFace_Models_git
inspect-snapshots: $(PROG)
	$(PYTHON) docs/capture_inspect.py $(MODELS)
	$(PYTHON) docs/build_models.py

## The documented-workflow gate: runs every runnable docs/examples/*.sh on this
## checkout's binary, as a reader would. Needs the network ONCE (the sandbox
## persists and later runs only re-verify digests) and ~1.6 GB of memory, so
## it is deliberately not part of `make test` and never runs in CI or a conda
## build. On the HPC run it under sbatch (release SOP step 4).
test-docs: $(PROG) yame-lib yame-bin
	$(PYTHON) test/docs_gate.py

## The compiled catalogue behind `methscope fetch`. Projected from the
## submodule's tools/registry/ at the pinned tag, so it is regenerated -- and
## committed -- with every submodule bump; a bumped submodule with a stale
## registry.h silently pins the previous model tag. `--version` prints the tag.
registry:
	./tools/make_registry.sh

# Always (incrementally) rebuild libyame.a from the pinned submodule so the
# static lib can never go stale relative to the checked-out YAME source.
yame-lib:
	$(MAKE) -C $(YAME_DIR) lib

$(YAME_LIB) $(HTSLIB): yame-lib

# The tests drive the yame BINARY too: test/run.sh packs its fixtures with it,
# and the docs examples call it. `lib` builds only libyame.a, so the binary has
# to be asked for separately -- otherwise `make test` on a fresh clone dies at
# run.sh's "no yame at ..." guard. Only the test targets need this; a plain
# `make` links the static lib and never wants the binary.
yame-bin:
	$(MAKE) -C $(YAME_DIR) build

src/%.o: src/%.c | check-xgb
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

src/upunit_cuda.o: src/upunit_cuda.cu src/upunit_cuda.h src/updec2.h | check-xgb
	$(NVCC) -O3 $(CUDA_GENCODE) $(NVCCFLAGS) -Isrc -c $< -o $@

# Vendored f2c-translated Lawson-Hanson NNLS: K&R style, compile warnings off.
src/nnls.o: src/nnls.c
	$(CC) $(CFLAGS) $(DEPFLAGS) -w -c $< -o $@

# CUDA changes both the object list and link libraries, which make does not
# otherwise track as prerequisites. Relink so switching CUDA=0/1 can never
# leave a stale binary from the other build mode.
$(PROG): $(OBJ) $(YAME_LIB) $(HTSLIB) force-link
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(LIBS) $$($(YAME_DIR)/yame-config --libs)

force-link:

check-xgb:
	@test -f "$(XGB_PREFIX)/include/xgboost/c_api.h" || { \
	  echo "ERROR: xgboost/c_api.h not found under XGB_PREFIX=$(XGB_PREFIX)"; \
	  echo "  Install it: conda install -c conda-forge libxgboost"; \
	  echo "  Then activate the env (sets CONDA_PREFIX) or pass XGB_PREFIX=..."; \
	  exit 1; }

# --- release tarball (self-contained: bundles the pinned YAME submodule) ----
# The GitHub auto tag-tarball OMITS the submodule, so bioconda's source: url
# needs this self-contained asset. Reproducible: pre-sorted names + fixed
# mtime/owner + `gzip -n`, so re-running yields a BYTE-IDENTICAL tarball (an
# ad-hoc `git archive` + worktree tar is not). GNU tar required. Archiving the
# YAME worktree via `git -C YAME ls-files` (not `git archive <sha>`) also
# sidesteps a submodule whose object store can't archive (e.g. stale alternates).
# Workflow: check out the release commit/tag AND `git submodule update --init`,
# then `make dist` (override the version with `make dist DIST_VERSION=1.2.3`).
DIST_VERSION ?= $(patsubst v%,%,$(shell git describe --tags --abbrev=0 2>/dev/null || echo 0.0.0))
DIST_MTIME   ?= $(shell git log -1 --format=%ct 2>/dev/null || echo 0)
DIST_NAME     = methscope-cli-$(DIST_VERSION)
DIST_TARBALL  = dist/$(DIST_NAME).tar.gz

dist:
	@test -f YAME/Makefile || { echo "ERROR: YAME submodule not checked out; run: git submodule update --init"; exit 1; }
	@mkdir -p dist
	@{ git ls-files | grep -vx YAME; git -C YAME ls-files | sed 's,^,YAME/,'; } | LC_ALL=C sort > dist/.$(DIST_NAME).files
	@tar --create --transform 's,^,$(DIST_NAME)/,' --owner=0 --group=0 --numeric-owner \
	     --mtime=@$(DIST_MTIME) --no-recursion --files-from=dist/.$(DIST_NAME).files \
	   | gzip -n -9 > $(DIST_TARBALL)
	@rm -f dist/.$(DIST_NAME).files
	@echo "built $(DIST_TARBALL)"
	@sha256sum $(DIST_TARBALL) 2>/dev/null || shasum -a 256 $(DIST_TARBALL)

clean:
	rm -f $(OBJ) $(DEP) src/upunit_cuda.o \
	      src/updec_cuda.o src/updec_nn.o src/updec_train.o $(PROG) test/nnls_check

# Also clean the YAME submodule build artifacts.
clean-all: clean
	$(MAKE) -C $(YAME_DIR) clean

# Generated by DEPFLAGS. Absent on a fresh tree, hence -include, not include.
-include $(DEP)
