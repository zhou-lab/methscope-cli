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
  /* Complexity-aware per-unit rank (factor units only). When adaptive_rank is
   * set, a unit's bottleneck is chosen from its cardinality and within-unit
   * methylation variability rather than the flat pure/mixed constants:
   * quantitatively homogeneous units (all-0/all-1 pattern, or per-CpG variance
   * below var_floor) collapse to homogeneous_rank; large variable units are
   * promoted toward max_rank. */
  uint32_t adaptive_rank;
  uint32_t max_rank;        /* cap for large variable units (e.g. 64) */
  uint32_t homogeneous_rank;/* flat units (all-0/all-1 or low variance) */
  uint32_t large_cpgs;      /* >= this many CpGs -> max_rank */
  uint32_t medium_cpgs;     /* >= this many CpGs -> 32 */
  double   var_floor;       /* mean per-CpG variance below this => homogeneous */
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
