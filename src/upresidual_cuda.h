// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPRESIDUAL_CUDA_H
#define METHSCOPE_UPRESIDUAL_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *data_path;
  const char *encoder_path;
  const char *index_path;
  const char *model_path;
  uint32_t rank;
  uint32_t steps;
  uint32_t batch;
  uint32_t eval_rows_per_head;
  uint32_t log_every;
  uint64_t seed;
  int device;
  double learning_rate;
  double weight_decay;
} ms_upresidual_config_t;

int ms_upresidual_cuda_available(void);
int ms_upresidual_train_cuda(const ms_upresidual_config_t *cfg);

#ifdef __cplusplus
}
#endif
#endif
