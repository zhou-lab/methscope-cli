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
# Order matters: our objects, then libyame, then htslib, then xgboost, then libc.
LIBS     = $(YAME_LIB) $(HTSLIB) -lxgboost -lpthread -lz -lm

OS := $(shell uname)
ifneq ($(OS),Darwin)
  LIBS += -lrt
endif

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
CUDA_OBJ =

ifeq ($(CUDA),1)
  CUDA_OBJ = src/upfactor_cuda.o src/upresidual_cuda.o src/uphybrid_eval_cuda.o src/upunit_cuda.o
  LDFLAGS += -L$(CUDA_HOME)/lib -Wl,-rpath,$(CUDA_HOME)/lib
  LIBS += -lcublas -lcudart -lstdc++
endif

OBJ += $(CUDA_OBJ)

.PHONY: all clean clean-all dist yame-lib check-xgb check-updec2 force-link

all: $(PROG)

check-updec2: $(PROG)
	$(PYTHON) test/check_updec2.py ./$(PROG)

# Always (incrementally) rebuild libyame.a from the pinned submodule so the
# static lib can never go stale relative to the checked-out YAME source.
yame-lib:
	$(MAKE) -C $(YAME_DIR) lib

$(YAME_LIB) $(HTSLIB): yame-lib

src/%.o: src/%.c | check-xgb
	$(CC) $(CFLAGS) -c $< -o $@

src/upfactor_cuda.o: src/upfactor_cuda.cu src/upfactor_cuda.h | check-xgb
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -Isrc -c $< -o $@

src/upresidual_cuda.o: src/upresidual_cuda.cu src/upresidual_cuda.h | check-xgb
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -Isrc -c $< -o $@

src/uphybrid_eval_cuda.o: src/uphybrid_eval_cuda.cu src/uphybrid_eval_cuda.h | check-xgb
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -Isrc -c $< -o $@

src/upunit_cuda.o: src/upunit_cuda.cu src/upunit_cuda.h src/updec2.h | check-xgb
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -Isrc -c $< -o $@

# Vendored f2c-translated Lawson-Hanson NNLS: K&R style, compile warnings off.
src/nnls.o: src/nnls.c
	$(CC) $(CFLAGS) -w -c $< -o $@

# CUDA changes both the object list and link libraries, which make does not
# otherwise track as prerequisites. Relink so switching CUDA=0/1 can never
# leave a stale binary from the other build mode.
$(PROG): $(OBJ) $(YAME_LIB) $(HTSLIB) force-link
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(LIBS)

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
	rm -f $(OBJ) src/upfactor_cuda.o src/upresidual_cuda.o src/uphybrid_eval_cuda.o src/upunit_cuda.o \
	      src/updec_cuda.o src/updec_nn.o src/updec_train.o \
	      src/upfactor_train.o src/upresidual_train.o $(PROG)

# Also clean the YAME submodule build artifacts.
clean-all: clean
	$(MAKE) -C $(YAME_DIR) clean
