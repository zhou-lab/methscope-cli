// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPDEC_CUDA_H
#define METHSCOPE_UPDEC_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include "updec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const float   *train_x, *val_x, *test_x;
  const uint8_t *train_target, *val_target, *test_target;
  size_t         n_train, n_val, n_test;
  const uint32_t *epoch_indices;  /* epochs x n_train, one permutation per epoch */
  int             epochs, batch, device;
  double          lr, weight_decay, bn_momentum;
} ms_updec_cuda_config_t;

typedef struct {
  double   best_val, test_accuracy, test_mae;
  uint64_t test_valid, test_missing;
} ms_updec_cuda_result_t;

/* The CPU-only build supplies weak stubs; a CUDA=1 build overrides them with
 * the native CUDA/cuBLAS backend. */
int ms_updec_cuda_available(void);
int ms_updec_train_cuda(const ms_updec_cuda_config_t *cfg,
                        const ms_updec_t *initial, ms_updec_t *best,
                        ms_updec_cuda_result_t *result);

#ifdef __cplusplus
}
#endif
#endif
