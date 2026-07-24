// SPDX-License-Identifier: AGPL-3.0-or-later
/* Shared UPDEC1 model I/O and eval-mode inference. */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "updec.h"

static void xdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] %s\n", msg);
  exit(1);
}

static size_t checked_mul(size_t a, size_t b, const char *what) {
  if (a && b > SIZE_MAX / a) xdie("decoder dimensions overflow", what);
  return a * b;
}

static float *falloc(size_t n, const char *what) {
  float *p = calloc(n ? n : 1, sizeof(*p));
  if (!p) xdie("out of memory allocating decoder", what);
  return p;
}

ms_updec_t *ms_updec_alloc(int n_in, int n_hidden, int n_out) {
  if (n_in <= 0 || n_hidden <= 0 || n_out <= 0)
    xdie("bad dimensions for .updec", NULL);
  ms_updec_t *m = calloc(1, sizeof(*m));
  if (!m) xdie("out of memory allocating decoder", NULL);
  m->n_in = n_in; m->n_hidden = n_hidden; m->n_out = n_out;
  size_t I = (size_t)n_in, H = (size_t)n_hidden, O = (size_t)n_out;
  m->imp_mean = falloc(I, "imputer_mean");
  m->sc_mean  = falloc(I, "scaler_mean");
  m->sc_scale = falloc(I, "scaler_scale");
  m->W1 = falloc(checked_mul(H, I, "W1"), "W1");
  m->b1 = falloc(H, "b1");
  m->bn_g = falloc(H, "bn_gamma");
  m->bn_b = falloc(H, "bn_beta");
  m->bn_m = falloc(H, "bn_mean");
  m->bn_v = falloc(H, "bn_var");
  m->W2 = falloc(checked_mul(O, H, "W2"), "W2");
  m->b2 = falloc(O, "b2");
  return m;
}

void ms_updec_free(ms_updec_t *m) {
  if (!m) return;
  free(m->imp_mean); free(m->sc_mean); free(m->sc_scale);
  free(m->W1); free(m->b1); free(m->bn_g); free(m->bn_b);
  free(m->bn_m); free(m->bn_v); free(m->W2); free(m->b2); free(m);
}

static uint32_t read_u32(FILE *fp, const char *what) {
  unsigned char b[4];
  if (fread(b, 1, 4, fp) != 4) xdie("truncated .updec at", what);
  return (uint32_t)b[0] | (uint32_t)b[1] << 8 |
         (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static float read_f32_1(FILE *fp, const char *what) {
  uint32_t u = read_u32(fp, what); float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

static void read_f32(FILE *fp, float *a, size_t n, const char *what) {
  for (size_t i = 0; i < n; ++i) a[i] = read_f32_1(fp, what);
}

static ms_updec_t *updec_read(FILE *fp) {
  unsigned char magic[8];
  if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "UPDEC1\0\0", 8) != 0)
    xdie("not a .updec (bad magic)", NULL);
  uint32_t ui = read_u32(fp, "n_in"), uh = read_u32(fp, "n_hidden");
  uint32_t uo = read_u32(fp, "n_out");
  if (!ui || !uh || !uo || ui > INT32_MAX || uh > INT32_MAX || uo > INT32_MAX)
    xdie("bad dimensions in .updec", NULL);
  ms_updec_t *m = ms_updec_alloc((int)ui, (int)uh, (int)uo);
  m->bn_eps = read_f32_1(fp, "bn_eps");
  if (!(m->bn_eps > 0.0f) || !isfinite(m->bn_eps)) xdie("bad bn_eps in .updec", NULL);
  size_t I = ui, H = uh, O = uo;
  read_f32(fp, m->imp_mean, I, "imputer_mean");
  read_f32(fp, m->sc_mean, I, "scaler_mean");
  read_f32(fp, m->sc_scale, I, "scaler_scale");
  read_f32(fp, m->W1, checked_mul(H, I, "W1"), "W1");
  read_f32(fp, m->b1, H, "b1");
  read_f32(fp, m->bn_g, H, "bn_gamma");
  read_f32(fp, m->bn_b, H, "bn_beta");
  read_f32(fp, m->bn_m, H, "bn_mean");
  read_f32(fp, m->bn_v, H, "bn_var");
  read_f32(fp, m->W2, checked_mul(O, H, "W2"), "W2");
  read_f32(fp, m->b2, O, "b2");
  if (fgetc(fp) != EOF) xdie("trailing bytes after .updec model", NULL);
  for (size_t j = 0; j < I; ++j)
    if (!isfinite(m->imp_mean[j]) || !isfinite(m->sc_mean[j]) ||
        !(m->sc_scale[j] > 0.0f) || !isfinite(m->sc_scale[j]))
      xdie("invalid preprocessing value in .updec", NULL);
  return m;
}

ms_updec_t *ms_updec_load(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) xdie("cannot open .updec", path);
  ms_updec_t *m = updec_read(fp);
  fclose(fp);
  return m;
}

ms_updec_t *ms_updec_load_buf(const void *buf, size_t len) {
  FILE *fp = fmemopen((void *)buf, len, "rb");
  if (!fp) xdie("fmemopen failed for embedded model", NULL);
  ms_updec_t *m = updec_read(fp);
  fclose(fp);
  return m;
}

void ms_updec_standardize(const ms_updec_t *m, const double *feat, double *xs) {
  for (int j = 0; j < m->n_in; ++j) {
    double x = feat[j];
    if (isnan(x)) x = m->imp_mean[j];
    xs[j] = (x - m->sc_mean[j]) / m->sc_scale[j];
  }
}

static double sigmoid_stable(double x) {
  if (x >= 0.0) { double z = exp(-x); return 1.0 / (1.0 + z); }
  double z = exp(x); return z / (1.0 + z);
}

void ms_updec_forward_eval(const ms_updec_t *m, const double *xs,
                           double *probs, double *hidden) {
  int I = m->n_in, H = m->n_hidden, O = m->n_out;
  for (int i = 0; i < H; ++i) {
    const float *w = m->W1 + (size_t)i * I;
    double acc = m->b1[i];
    for (int j = 0; j < I; ++j) acc += (double)w[j] * xs[j];
    if (acc < 0.0) acc = 0.0;
    hidden[i] = m->bn_g[i] * (acc - m->bn_m[i]) /
                sqrt((double)m->bn_v[i] + m->bn_eps) + m->bn_b[i];
  }
  for (int k = 0; k < O; ++k) {
    const float *w = m->W2 + (size_t)k * H;
    double acc = m->b2[k];
    for (int i = 0; i < H; ++i) acc += (double)w[i] * hidden[i];
    probs[k] = sigmoid_stable(acc);
  }
}
