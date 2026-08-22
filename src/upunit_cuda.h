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
  const char *split_path; /* optional curated cell split; see upsplit.h */
  uint32_t patterns;
  uint32_t feature_mode; /* MS_UPFEATURE_* */
  uint32_t pure_bottleneck;
  uint32_t mixed_bottleneck;
  uint32_t mixed_direct;
  uint32_t activation;
  uint32_t min_steps;
  uint32_t max_steps;
  uint32_t eval_every;
  uint32_t patience;
  uint32_t batch;
  uint32_t eval_rows;
  /* Which cells the early-stop signal is measured on. VAL is the default and
   * the honest one. TRAIN exists to answer "is a val split worth its samples"
   * -- with val folded into training there is nothing held out to stop on, so
   * the epsilon/patience test runs against a fixed sample of TRAINING rows
   * instead. That detects convergence, never overfitting; use it knowing so. */
  int stop_on_train;
  /* Improvement threshold for the early-stop test. An eval counts as progress
   * only if it beats the best by more than this. Fixed at 1e-7 historically,
   * which is ~1.7e-6 of a typical 0.06 MAE -- fine when a step touches half a
   * unit's CpGs, wrong when it touches 0.08% of them, because per-eval progress
   * then shrinks toward the threshold and genuine improvement reads as a
   * plateau. Scale it down for large units. */
  double stop_eps;
  uint32_t threads;   /* CPU backend: worker threads over units */
  uint64_t seed;
  int device;
  double learning_rate;
  double weight_decay;
} ms_upunit_config_t;

int ms_upunit_cuda_available(void);
int ms_upunit_train_cuda(const ms_upunit_config_t *cfg);
/* Portable fallback: same numerics, threaded over units. Checkpoints and the
 * emitted UPDEC2 are byte-compatible with the CUDA backend. */
int ms_upunit_train_cpu(const ms_upunit_config_t *cfg);

#ifdef __cplusplus
}
#endif
#endif
