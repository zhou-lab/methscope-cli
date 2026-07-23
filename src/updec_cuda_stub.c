// SPDX-License-Identifier: AGPL-3.0-or-later
#include "updec_cuda.h"

#if defined(__GNUC__) || defined(__clang__)
#define MS_WEAK __attribute__((weak))
#else
#define MS_WEAK
#endif

MS_WEAK int ms_updec_cuda_available(void) {
  return 0;
}

MS_WEAK int ms_updec_train_cuda(const ms_updec_cuda_config_t *cfg,
                                const ms_updec_t *initial, ms_updec_t *best,
                                ms_updec_cuda_result_t *result) {
  (void)cfg; (void)initial; (void)best; (void)result;
  return -1;
}
