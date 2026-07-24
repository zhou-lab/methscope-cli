// SPDX-License-Identifier: AGPL-3.0-or-later
#include "upunit_cuda.h"

#if defined(__GNUC__) || defined(__clang__)
#define MS_WEAK __attribute__((weak))
#else
#define MS_WEAK
#endif

MS_WEAK int ms_upunit_cuda_available(void) { return 0; }
MS_WEAK int ms_upunit_train_cuda(const ms_upunit_config_t *cfg) {
  (void)cfg;
  return -1;
}
