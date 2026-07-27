#!/bin/bash
set -euo pipefail

# GPU variant of build.sh. Kept SEPARATE because conda-recipe/build.sh is
# byte-identical to bioconda-recipe/build.sh by design (see the note there) and
# must not gain CUDA branches. Everything below matches build.sh except the
# CUDA=1 flags and the installed binary name.

export CFLAGS="${CFLAGS:-} ${CPPFLAGS:-} -std=gnu99"
export C_INCLUDE_PATH="${PREFIX}/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"

# A distributed binary must run on more than the card it was built on, so emit a
# fatbin spanning Volta..Hopper plus a PTX fallback that the driver JITs for
# anything newer. CUDA_HOME is the build prefix: conda ships nvcc there.
# nvcc is a BUILD dependency so it lives in $BUILD_PREFIX; the CUDA runtime and
# headers are HOST dependencies in $PREFIX. Pointing NVCC at $PREFIX gave
# "Error 127" (command not found).
make -j"${CPU_COUNT:-1}" CUDA=1 CC="${CC}" XGB_PREFIX="${PREFIX}" \
     CUDA_HOME="${PREFIX}" NVCC="${BUILD_PREFIX}/bin/nvcc" \
     NVCCFLAGS="-I${PREFIX}/include" \
     CUDA_GENCODE="-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_90,code=compute_90" \
     LDFLAGS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib ${LDFLAGS:-}"

# Installed under a distinct name so methscope and methscope-cuda can coexist:
# only training needs CUDA, and inference is pure C with no GPU dependency.
mkdir -p "${PREFIX}/bin"
install -m 755 methscope "${PREFIX}/bin/methscope-cuda"
