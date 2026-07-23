// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPDEC_H
#define METHSCOPE_UPDEC_H

#include <stddef.h>

/* Portable upscale decoder shared by `upscale` inference and
 * `upscale-train`. Arrays are float32 exactly as stored in UPDEC1. */
typedef struct ms_updec_t {
  int     n_in, n_hidden, n_out;
  float   bn_eps;
  float  *imp_mean;
  float  *sc_mean, *sc_scale;
  float  *W1, *b1;
  float  *bn_g, *bn_b, *bn_m, *bn_v;
  float  *W2, *b2;
} ms_updec_t;

ms_updec_t *ms_updec_alloc(int n_in, int n_hidden, int n_out);
ms_updec_t *ms_updec_load(const char *path);
ms_updec_t *ms_updec_load_buf(const void *buf, size_t len);
void        ms_updec_write(const char *path, const ms_updec_t *m);
void        ms_updec_free(ms_updec_t *m);

void ms_updec_standardize(const ms_updec_t *m, const double *feat, double *xs);
void ms_updec_forward_eval(const ms_updec_t *m, const double *xs,
                           double *probs, double *hidden);

#endif
