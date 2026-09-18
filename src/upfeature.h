// SPDX-License-Identifier: AGPL-3.0-or-later
/* Encoder feature modes, shared by the CPU and CUDA unit trainers and by the
 * .updecx reader. Split out of the deleted upfactor_cuda.h on 2026-09-17: that
 * header also declared the frozen-trunk trainer, which went with the trunk, but
 * these constants describe what every encoder input IS and outlive it. */
#ifndef METHSCOPE_UPFEATURE_H
#define METHSCOPE_UPFEATURE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MS_UPFEATURE_MISSING 0u
#define MS_UPFEATURE_COUNT 1u
#define MS_UPFEATURE_BETA 2u
/* beta + ONE scalar (log1p of total covered CpGs) instead of P missing bits.
 * The missing vector is near-constant above ~10k observed CpGs -- all zero at
 * p100, 4 of 1000 active at 0.5% -- so it costs half the encoder input to carry
 * almost nothing there. The scalar conveys the coverage regime in 1 dimension;
 * what it cannot do is say WHICH patterns are unobserved, which matters only in
 * the sparse regime where many are. */
#define MS_UPFEATURE_SCALAR 3u

/* Encoder input width for a feature mode. */
static inline uint32_t ms_upfeature_dim(uint32_t mode, uint32_t patterns) {
  if (mode == MS_UPFEATURE_BETA) return patterns;
  if (mode == MS_UPFEATURE_SCALAR) return patterns + 1u;
  return 2u * patterns;
}

#ifdef __cplusplus
}
#endif

#endif
