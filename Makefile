# methscope-cli — pure-C CLI. Links YAME (static libyame.a) + libxgboost (conda).
CC ?= gcc
CFLAGS ?= -W -Wall -O3 -std=gnu99
PROG = methscope

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

.PHONY: all clean clean-all dist yame-lib check-xgb check check-upscale-grad regen-upscale-grad

all: $(PROG)

check: check-upscale-grad

test/upscale_grad_check: test/upscale_grad_check.c src/updec_nn.c src/updec_nn.h
	$(CC) $(CFLAGS) -Isrc test/upscale_grad_check.c src/updec_nn.c -lm -o $@

check-upscale-grad: test/upscale_grad_check test/upscale_grad_reference.bin
	./test/upscale_grad_check test/upscale_grad_reference.bin

PYTORCH_PYTHON ?= python3
regen-upscale-grad:
	$(PYTORCH_PYTHON) test/make_upscale_grad_reference.py -o test/upscale_grad_reference.bin

# Always (incrementally) rebuild libyame.a from the pinned submodule so the
# static lib can never go stale relative to the checked-out YAME source.
yame-lib:
	$(MAKE) -C $(YAME_DIR) lib

$(YAME_LIB) $(HTSLIB): yame-lib

src/%.o: src/%.c | check-xgb
	$(CC) $(CFLAGS) -c $< -o $@

# Vendored f2c-translated Lawson-Hanson NNLS: K&R style, compile warnings off.
src/nnls.o: src/nnls.c
	$(CC) $(CFLAGS) -w -c $< -o $@

$(PROG): $(OBJ) $(YAME_LIB) $(HTSLIB)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(LIBS)

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
	rm -f $(OBJ) $(PROG) test/upscale_grad_check

# Also clean the YAME submodule build artifacts.
clean-all: clean
	$(MAKE) -C $(YAME_DIR) clean
