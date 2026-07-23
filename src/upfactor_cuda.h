// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPFACTOR_CUDA_H
#define METHSCOPE_UPFACTOR_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
  const char *homogeneous_groups;
  double homogeneous_fraction;
} ms_upfactor_config_t;

int ms_upfactor_cuda_available(void);
int ms_upfactor_train_cuda(const ms_upfactor_config_t *cfg);

#ifdef __cplusplus
}
#endif
#endif
