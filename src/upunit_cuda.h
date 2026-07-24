// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPUNIT_CUDA_H
#define METHSCOPE_UPUNIT_CUDA_H

#include <stdint.h>
#include "upfactor_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *data_path;
  const char *index_path;
  const char *model_path; /* bare UPDEC2; caller may bundle it */
  const char *work_dir;
  const char *pilot_units_path; /* optional unit IDs; train/checkpoint only */
  const char *trunk_path; /* optional frozen UPFAC3 shared trunk */
  uint32_t patterns;
  uint32_t feature_mode; /* MS_UPFEATURE_* */
  uint32_t pure_bottleneck;
  uint32_t mixed_bottleneck;
  uint32_t mixed_direct;
  uint32_t activation;
  /* Simple homogeneity-aware per-unit rank. When adaptive_rank is set, a pure
   * unit whose single MRMP pattern is all-0 or all-1 (constitutively un/
   * methylated -> trivially predictable) uses homogeneous_rank; every other
   * factor unit keeps its base pure/mixed bottleneck. This reclaims the wasted
   * capacity on the homogeneous giants at no external-accuracy cost. Promoting
   * variable units was tried and dropped: it overfits single-cohort training. */
  uint32_t adaptive_rank;
  uint32_t homogeneous_rank; /* rank for all-0/all-1 units (default 8) */
  uint32_t min_steps;
  uint32_t max_steps;
  uint32_t eval_every;
  uint32_t patience;
  uint32_t batch;
  uint32_t eval_rows;
  uint64_t seed;
  int device;
  double learning_rate;
  double weight_decay;
} ms_upunit_config_t;

int ms_upunit_cuda_available(void);
int ms_upunit_train_cuda(const ms_upunit_config_t *cfg);

#ifdef __cplusplus
}
#endif
#endif
