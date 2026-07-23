// SPDX-License-Identifier: AGPL-3.0-or-later
#include "upfactor_cuda.h"

#if defined(__GNUC__) || defined(__clang__)
#define MS_WEAK __attribute__((weak))
#else
#define MS_WEAK
#endif

MS_WEAK int ms_upfactor_cuda_available(void) { return 0; }
MS_WEAK int ms_upfactor_train_cuda(const ms_upfactor_config_t *cfg) {
  (void)cfg;
  return -1;
}
