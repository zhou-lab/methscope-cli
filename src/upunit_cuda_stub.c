// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
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
