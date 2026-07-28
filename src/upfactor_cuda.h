// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPFACTOR_CUDA_H
#define METHSCOPE_UPFACTOR_CUDA_H

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

typedef struct {
  const char *data_path;
  const char *model_path;
  uint32_t patterns;
  uint32_t rank;
  uint32_t hidden;
  uint32_t steps;
  uint32_t batch;
  uint32_t eval_batches;
  uint32_t log_every;
  uint64_t seed;
  int device;
  double learning_rate;
  double weight_decay;
  const char *split_path; /* optional curated cell split; see upsplit.h */
  const char *homogeneous_groups;
  double homogeneous_fraction;
  uint32_t feature_mode;
} ms_upfactor_config_t;

int ms_upfactor_cuda_available(void);
int ms_upfactor_train_cuda(const ms_upfactor_config_t *cfg);

#ifdef __cplusplus
}
#endif
#endif
