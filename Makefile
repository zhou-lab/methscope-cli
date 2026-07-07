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

.PHONY: all clean yame-lib check-xgb

all: $(PROG)

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

clean:
	rm -f $(OBJ) $(PROG)

# Also clean the YAME submodule build artifacts.
clean-all: clean
	$(MAKE) -C $(YAME_DIR) clean
