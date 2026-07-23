// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPDEC_NN_H
#define METHSCOPE_UPDEC_NN_H

#include <stddef.h>
#include <stdint.h>

void ms_nn_linear_fwd_f32(double *out, const float *x, const float *W,
                          const float *bias, int B, int in, int outdim);
void ms_nn_linear_fwd_f64(double *out, const double *x, const float *W,
                          const float *bias, int B, int in, int outdim);

void ms_nn_linear_bwd_f32(const double *dout, const float *x, const float *W,
                          double *dW, double *db, double *dx,
                          int B, int in, int outdim);
void ms_nn_linear_bwd_f64(const double *dout, const double *x, const float *W,
                          double *dW, double *db, double *dx,
                          int B, int in, int outdim);

void ms_nn_relu(double *x, size_t n);
void ms_nn_relu_bwd(double *dx, const double *relu_out, size_t n);

void ms_nn_bn_train(double *out, double *xhat, double *invstd,
                    double *mean, double *var, const double *x,
                    const float *gamma, const float *beta,
                    float *running_mean, float *running_var,
                    int B, int H, double eps, double momentum);
void ms_nn_bn_eval(double *out, const double *x, const float *gamma,
                   const float *beta, const float *running_mean,
                   const float *running_var, int B, int H, double eps);
void ms_nn_bn_bwd(double *dx, double *dgamma, double *dbeta,
                  const double *dout, const double *xhat,
                  const double *invstd, const float *gamma, int B, int H);

/* Returns the masked mean. If grad is non-NULL it receives dloss/dlogit. */
double ms_nn_bce_logits(const double *logits, const uint8_t *target,
                        double *grad, size_t n);

void ms_nn_adam_step(float *theta, const double *grad, double *m, double *v,
                     size_t n, double lr, double weight_decay,
                     double inv_bias1, double inv_bias2);

/* PyTorch ReduceLROnPlateau defaults used by upscale training. */
int ms_nn_plateau_step(double val, double *best, int *bad_epochs, double *lr);

double ms_nn_sigmoid(double x);

#endif
