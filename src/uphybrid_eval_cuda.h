// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPHYBRID_EVAL_CUDA_H
#define METHSCOPE_UPHYBRID_EVAL_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *data_path;
  const char *encoder_path;
  const char *residual_path;
  const char *metrics_path;
  uint32_t max_cells;
  uint32_t max_reps;
  uint32_t log_every_cells;
  int device;
} ms_uphybrid_eval_config_t;

int ms_uphybrid_eval_cuda_available(void);
int ms_uphybrid_eval_cuda(const ms_uphybrid_eval_config_t *cfg);

#ifdef __cplusplus
}
#endif
#endif
