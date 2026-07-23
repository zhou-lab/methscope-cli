// SPDX-License-Identifier: AGPL-3.0-or-later
/* Dimension-generic numerical kernels for the upscale decoder trainer. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "updec_nn.h"

void ms_nn_linear_fwd_f32(double *out, const float *x, const float *W,
                          const float *bias, int B, int in, int outdim) {
  for (int b = 0; b < B; ++b)
    for (int o = 0; o < outdim; ++o) {
      double s = bias[o];
      const float *xp = x + (size_t)b * in;
      const float *wp = W + (size_t)o * in;
      for (int j = 0; j < in; ++j) s += (double)xp[j] * wp[j];
      out[(size_t)b * outdim + o] = s;
    }
}

void ms_nn_linear_fwd_f64(double *out, const double *x, const float *W,
                          const float *bias, int B, int in, int outdim) {
  for (int b = 0; b < B; ++b)
    for (int o = 0; o < outdim; ++o) {
      double s = bias[o];
      const double *xp = x + (size_t)b * in;
      const float *wp = W + (size_t)o * in;
      for (int j = 0; j < in; ++j) s += xp[j] * wp[j];
      out[(size_t)b * outdim + o] = s;
    }
}

#define LINEAR_BWD(NAME, XTYPE) \
void NAME(const double *dout, const XTYPE *x, const float *W, \
          double *dW, double *db, double *dx, int B, int in, int outdim) { \
  memset(dW, 0, (size_t)outdim * in * sizeof(*dW)); \
  memset(db, 0, (size_t)outdim * sizeof(*db)); \
  if (dx) memset(dx, 0, (size_t)B * in * sizeof(*dx)); \
  for (int b = 0; b < B; ++b) { \
    const XTYPE *xp = x + (size_t)b * in; \
    for (int o = 0; o < outdim; ++o) { \
      double d = dout[(size_t)b * outdim + o]; \
      double *gwp = dW + (size_t)o * in; \
      const float *wp = W + (size_t)o * in; \
      db[o] += d; \
      for (int j = 0; j < in; ++j) { \
        gwp[j] += d * xp[j]; \
        if (dx) dx[(size_t)b * in + j] += d * wp[j]; \
      } \
    } \
  } \
}

LINEAR_BWD(ms_nn_linear_bwd_f32, float)
LINEAR_BWD(ms_nn_linear_bwd_f64, double)

void ms_nn_relu(double *x, size_t n) {
  for (size_t i = 0; i < n; ++i) if (x[i] < 0.0) x[i] = 0.0;
}

void ms_nn_relu_bwd(double *dx, const double *relu_out, size_t n) {
  for (size_t i = 0; i < n; ++i) if (!(relu_out[i] > 0.0)) dx[i] = 0.0;
}

void ms_nn_bn_train(double *out, double *xhat, double *invstd,
                    double *mean, double *var, const double *x,
                    const float *gamma, const float *beta,
                    float *running_mean, float *running_var,
                    int B, int H, double eps, double momentum) {
  for (int h = 0; h < H; ++h) {
    double mu = 0.0;
    for (int b = 0; b < B; ++b) mu += x[(size_t)b * H + h];
    mu /= B;
    double vv = 0.0;
    for (int b = 0; b < B; ++b) {
      double d = x[(size_t)b * H + h] - mu; vv += d * d;
    }
    vv /= B;
    double is = 1.0 / sqrt(vv + eps);
    mean[h] = mu; var[h] = vv; invstd[h] = is;
    running_mean[h] = (float)((1.0 - momentum) * running_mean[h] + momentum * mu);
    running_var[h] = (float)((1.0 - momentum) * running_var[h] +
                            momentum * vv * (double)B / (B - 1));
    for (int b = 0; b < B; ++b) {
      size_t q = (size_t)b * H + h;
      double z = (x[q] - mu) * is;
      xhat[q] = z; out[q] = gamma[h] * z + beta[h];
    }
  }
}

void ms_nn_bn_eval(double *out, const double *x, const float *gamma,
                   const float *beta, const float *running_mean,
                   const float *running_var, int B, int H, double eps) {
  for (int h = 0; h < H; ++h) {
    double is = 1.0 / sqrt((double)running_var[h] + eps);
    for (int b = 0; b < B; ++b) {
      size_t q = (size_t)b * H + h;
      out[q] = gamma[h] * (x[q] - running_mean[h]) * is + beta[h];
    }
  }
}

void ms_nn_bn_bwd(double *dx, double *dgamma, double *dbeta,
                  const double *dout, const double *xhat,
                  const double *invstd, const float *gamma, int B, int H) {
  for (int h = 0; h < H; ++h) {
    double dg = 0.0, db = 0.0, sum_dxhat = 0.0, sum_xdx = 0.0;
    for (int b = 0; b < B; ++b) {
      size_t q = (size_t)b * H + h;
      dg += dout[q] * xhat[q]; db += dout[q];
      double dh = dout[q] * gamma[h];
      sum_dxhat += dh; sum_xdx += dh * xhat[q];
    }
    dgamma[h] = dg; dbeta[h] = db;
    double scale = invstd[h] / B;
    for (int b = 0; b < B; ++b) {
      size_t q = (size_t)b * H + h;
      double dh = dout[q] * gamma[h];
      dx[q] = scale * (B * dh - sum_dxhat - xhat[q] * sum_xdx);
    }
  }
}

double ms_nn_sigmoid(double x) {
  if (x >= 0.0) { double z = exp(-x); return 1.0 / (1.0 + z); }
  double z = exp(x); return z / (1.0 + z);
}

double ms_nn_bce_logits(const double *logits, const uint8_t *target,
                        double *grad, size_t n) {
  size_t valid = 0;
  for (size_t i = 0; i < n; ++i) valid += target[i] != 2;
  double denom = (double)valid + 1e-8, loss = 0.0;
  for (size_t i = 0; i < n; ++i) {
    int keep = target[i] != 2;
    double y = keep ? target[i] : 0.0, z = logits[i];
    if (keep) loss += fmax(z, 0.0) - z * y + log1p(exp(-fabs(z)));
    if (grad) grad[i] = keep ? (ms_nn_sigmoid(z) - y) / denom : 0.0;
  }
  return loss / denom;
}

void ms_nn_adam_step(float *theta, const double *grad, double *m, double *v,
                     size_t n, double lr, double weight_decay,
                     double inv_bias1, double inv_bias2) {
  for (size_t i = 0; i < n; ++i) {
    double g = grad[i] + weight_decay * theta[i];
    m[i] = 0.9 * m[i] + 0.1 * g;
    v[i] = 0.999 * v[i] + 0.001 * g * g;
    double mh = m[i] * inv_bias1, vh = v[i] * inv_bias2;
    theta[i] = (float)(theta[i] - lr * mh / (sqrt(vh) + 1e-8));
  }
}

int ms_nn_plateau_step(double val, double *best, int *bad_epochs, double *lr) {
  if (val < *best * (1.0 - 1e-4)) {
    *best = val; *bad_epochs = 0;
  } else if (++*bad_epochs > 2) {
    *lr *= 0.5; *bad_epochs = 0; return 1;
  }
  return 0;
}
