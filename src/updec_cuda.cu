// SPDX-License-Identifier: AGPL-3.0-or-later
/* Native CUDA/cuBLAS backend for `upscale-train`.
 *
 * Matrices retain the row-major UPDEC1 layout on host and device. cuBLAS sees
 * those buffers as their column-major transposes, avoiding layout conversions.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include "updec_cuda.h"

enum { P_W1, P_B1, P_GAMMA, P_BETA, P_W2, P_B2, P_COUNT };
static const int KTHREAD = 256;
static const int RTHREAD = 128;

static void fail_cuda(const char *call, const char *msg, const char *file, int line) {
  std::fprintf(stderr, "[methscope] upscale-train CUDA: %s failed at %s:%d: %s\n",
               call, file, line, msg);
  std::exit(1);
}

#define CUDA_OK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) fail_cuda(#call, cudaGetErrorString(_e), __FILE__, __LINE__); \
} while (0)

static const char *cublas_message(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
    default: return "unknown cuBLAS error";
  }
}

#define CUBLAS_OK(call) do { \
  cublasStatus_t _s = (call); \
  if (_s != CUBLAS_STATUS_SUCCESS) fail_cuda(#call, cublas_message(_s), __FILE__, __LINE__); \
} while (0)

static float *alloc_f(size_t n) {
  float *p = NULL; CUDA_OK(cudaMalloc((void **)&p, n * sizeof(*p))); return p;
}
static uint8_t *alloc_u8(size_t n) {
  uint8_t *p = NULL; CUDA_OK(cudaMalloc((void **)&p, n * sizeof(*p))); return p;
}
static uint32_t *alloc_u32(size_t n) {
  uint32_t *p = NULL; CUDA_OK(cudaMalloc((void **)&p, n * sizeof(*p))); return p;
}
static uint64_t *alloc_u64(size_t n) {
  uint64_t *p = NULL; CUDA_OK(cudaMalloc((void **)&p, n * sizeof(*p))); return p;
}
static double *alloc_d(size_t n) {
  double *p = NULL; CUDA_OK(cudaMalloc((void **)&p, n * sizeof(*p))); return p;
}

__device__ static float sigmoidf_stable(float x) {
  if (x >= 0.0f) { float z = expf(-x); return 1.0f / (1.0f + z); }
  float z = expf(x); return z / (1.0f + z);
}

__global__ static void gather_x_kernel(float *dst, const float *src,
                                       const uint32_t *idx, int B, int I) {
  size_t n = (size_t)B * I;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x) {
    int b = (int)(q / I), j = (int)(q % I);
    dst[q] = src[(size_t)idx[b] * I + j];
  }
}

__global__ static void bias_relu_kernel(float *x, const float *bias, int B, int D) {
  size_t n = (size_t)B * D;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x) {
    float v = x[q] + bias[q % D];
    x[q] = v > 0.0f ? v : 0.0f;
  }
}

__global__ static void bias_kernel(float *x, const float *bias, int B, int D) {
  size_t n = (size_t)B * D;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x)
    x[q] += bias[q % D];
}

__global__ static void bn_train_kernel(const float *x, float *xhat, float *out,
                                       const float *gamma, const float *beta,
                                       float *running_mean, float *running_var,
                                       float *invstd, int B, int H,
                                       float eps, float momentum) {
  int h = blockIdx.x;
  if (h >= H) return;
  __shared__ float s[RTHREAD];
  float sum = 0.0f;
  for (int b = threadIdx.x; b < B; b += blockDim.x) sum += x[(size_t)b * H + h];
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  float mean = s[0] / B;
  float ss = 0.0f;
  for (int b = threadIdx.x; b < B; b += blockDim.x) {
    float d = x[(size_t)b * H + h] - mean; ss += d * d;
  }
  s[threadIdx.x] = ss;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  float var = s[0] / B;
  if (threadIdx.x == 0) {
    invstd[h] = rsqrtf(var + eps);
    running_mean[h] = (1.0f - momentum) * running_mean[h] + momentum * mean;
    running_var[h] = (1.0f - momentum) * running_var[h] +
                     momentum * var * ((float)B / (B - 1));
  }
  __syncthreads();
  float is = invstd[h];
  for (int b = threadIdx.x; b < B; b += blockDim.x) {
    size_t q = (size_t)b * H + h;
    float z = (x[q] - mean) * is;
    xhat[q] = z; out[q] = gamma[h] * z + beta[h];
  }
}

__global__ static void bn_eval_kernel(const float *x, float *out,
                                      const float *gamma, const float *beta,
                                      const float *mean, const float *var,
                                      int B, int H, float eps) {
  size_t n = (size_t)B * H;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x) {
    int h = q % H;
    out[q] = gamma[h] * (x[q] - mean[h]) * rsqrtf(var[h] + eps) + beta[h];
  }
}

__global__ static void bias_grad_kernel(const float *dout, float *db, int B, int D) {
  int j = blockIdx.x;
  if (j >= D) return;
  __shared__ float s[RTHREAD];
  float sum = 0.0f;
  for (int b = threadIdx.x; b < B; b += blockDim.x) sum += dout[(size_t)b * D + j];
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0) db[j] = s[0];
}

__global__ static void bn_backward_kernel(float *dx, float *dgamma, float *dbeta,
                                          const float *dout, const float *xhat,
                                          const float *invstd, const float *gamma,
                                          int B, int H) {
  int h = blockIdx.x;
  if (h >= H) return;
  __shared__ float sdg[RTHREAD], sdb[RTHREAD], sdx[RTHREAD], sxh[RTHREAD];
  float dg = 0.0f, db = 0.0f, ds = 0.0f, xs = 0.0f;
  for (int b = threadIdx.x; b < B; b += blockDim.x) {
    size_t q = (size_t)b * H + h;
    float d = dout[q], dh = d * gamma[h];
    dg += d * xhat[q]; db += d; ds += dh; xs += dh * xhat[q];
  }
  sdg[threadIdx.x] = dg; sdb[threadIdx.x] = db;
  sdx[threadIdx.x] = ds; sxh[threadIdx.x] = xs;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) {
      sdg[threadIdx.x] += sdg[threadIdx.x + d];
      sdb[threadIdx.x] += sdb[threadIdx.x + d];
      sdx[threadIdx.x] += sdx[threadIdx.x + d];
      sxh[threadIdx.x] += sxh[threadIdx.x + d];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { dgamma[h] = sdg[0]; dbeta[h] = sdb[0]; }
  float scale = invstd[h] / B, sumd = sdx[0], sumxd = sxh[0];
  for (int b = threadIdx.x; b < B; b += blockDim.x) {
    size_t q = (size_t)b * H + h;
    float dh = dout[q] * gamma[h];
    dx[q] = scale * (B * dh - sumd - xhat[q] * sumxd);
  }
}

__global__ static void relu_backward_kernel(float *dx, const float *relu, size_t n) {
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x)
    if (!(relu[q] > 0.0f)) dx[q] = 0.0f;
}

__global__ static void masked_grad_kernel(float *logits,
                                          const uint8_t *target,
                                          const uint32_t *idx,
                                          int B, int O, float denom) {
  size_t n = (size_t)B * O;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x) {
    int b = q / O, k = q % O;
    uint8_t y = target[(size_t)idx[b] * O + k];
    logits[q] = y == 2 ? 0.0f : (sigmoidf_stable(logits[q]) - y) / denom;
  }
}

__global__ static void loss_partial_kernel(const float *logits,
                                           const uint8_t *target,
                                           size_t n, double *partial) {
  __shared__ double s[KTHREAD];
  double sum = 0.0;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x) {
    uint8_t y = target[q];
    if (y != 2) {
      double z = logits[q];
      sum += fmax(z, 0.0) - z * y + log1p(exp(-fabs(z)));
    }
  }
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

__global__ static void metrics_partial_kernel(const float *logits,
                                              const uint8_t *target,
                                              size_t n, uint64_t *correct,
                                              double *abs_sum) {
  __shared__ uint64_t sc[KTHREAD];
  __shared__ double sa[KTHREAD];
  uint64_t c = 0; double a = 0.0;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x) {
    uint8_t y = target[q];
    if (y != 2) {
      float p = sigmoidf_stable(logits[q]);
      c += ((p > 0.5f) ? 1 : 0) == y;
      a += fabs((double)p - y);
    }
  }
  sc[threadIdx.x] = c; sa[threadIdx.x] = a;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) {
      sc[threadIdx.x] += sc[threadIdx.x + d];
      sa[threadIdx.x] += sa[threadIdx.x + d];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { correct[blockIdx.x] = sc[0]; abs_sum[blockIdx.x] = sa[0]; }
}

__global__ static void adam_kernel(float *theta, const float *grad,
                                   float *m, float *v, size_t n,
                                   float lr, float wd, float ib1, float ib2) {
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < n; q += (size_t)blockDim.x * gridDim.x) {
    float g = grad[q] + wd * theta[q];
    float mm = 0.9f * m[q] + 0.1f * g;
    float vv = 0.999f * v[q] + 0.001f * g * g;
    m[q] = mm; v[q] = vv;
    theta[q] -= lr * (mm * ib1) / (sqrtf(vv * ib2) + 1e-8f);
  }
}

struct DeviceModel {
  float *theta[P_COUNT], *grad[P_COUNT], *mom[P_COUNT], *vel[P_COUNT];
  size_t n[P_COUNT];
  float *run_mean, *run_var;
};

struct DeviceData {
  float *x;
  uint8_t *target;
  size_t rows;
};

struct Workspace {
  float *x, *a1, *xhat, *bn, *logits, *dh, *invstd;
  uint32_t *idx;
  double *loss_partial, *abs_partial;
  uint64_t *correct_partial;
  double *h_partial;
  uint64_t *h_correct;
  int maxB, partial_blocks;
};

static int blocks_for(size_t n, int threads = KTHREAD) {
  size_t b = (n + threads - 1) / threads;
  if (b > 4096) b = 4096;
  return (int)(b ? b : 1);
}

static void upload(float *dst, const float *src, size_t n) {
  CUDA_OK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
}
static void download(float *dst, const float *src, size_t n) {
  CUDA_OK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void init_device_model(DeviceModel *d, const ms_updec_t *m) {
  d->n[P_W1] = (size_t)m->n_hidden * m->n_in;
  d->n[P_B1] = m->n_hidden;
  d->n[P_GAMMA] = m->n_hidden;
  d->n[P_BETA] = m->n_hidden;
  d->n[P_W2] = (size_t)m->n_out * m->n_hidden;
  d->n[P_B2] = m->n_out;
  const float *host[P_COUNT] = {m->W1,m->b1,m->bn_g,m->bn_b,m->W2,m->b2};
  for (int p = 0; p < P_COUNT; ++p) {
    d->theta[p] = alloc_f(d->n[p]); d->grad[p] = alloc_f(d->n[p]);
    d->mom[p] = alloc_f(d->n[p]); d->vel[p] = alloc_f(d->n[p]);
    upload(d->theta[p], host[p], d->n[p]);
    CUDA_OK(cudaMemset(d->mom[p], 0, d->n[p] * sizeof(float)));
    CUDA_OK(cudaMemset(d->vel[p], 0, d->n[p] * sizeof(float)));
  }
  d->run_mean = alloc_f(m->n_hidden); d->run_var = alloc_f(m->n_hidden);
  upload(d->run_mean, m->bn_m, m->n_hidden);
  upload(d->run_var, m->bn_v, m->n_hidden);
}

static void upload_model(DeviceModel *d, const ms_updec_t *m) {
  const float *host[P_COUNT] = {m->W1,m->b1,m->bn_g,m->bn_b,m->W2,m->b2};
  for (int p = 0; p < P_COUNT; ++p) upload(d->theta[p], host[p], d->n[p]);
  upload(d->run_mean, m->bn_m, m->n_hidden);
  upload(d->run_var, m->bn_v, m->n_hidden);
}

static void download_model(ms_updec_t *m, const DeviceModel *d) {
  float *host[P_COUNT] = {m->W1,m->b1,m->bn_g,m->bn_b,m->W2,m->b2};
  for (int p = 0; p < P_COUNT; ++p) download(host[p], d->theta[p], d->n[p]);
  download(m->bn_m, d->run_mean, m->n_hidden);
  download(m->bn_v, d->run_var, m->n_hidden);
}

static void free_device_model(DeviceModel *d) {
  for (int p = 0; p < P_COUNT; ++p) {
    cudaFree(d->theta[p]); cudaFree(d->grad[p]); cudaFree(d->mom[p]); cudaFree(d->vel[p]);
  }
  cudaFree(d->run_mean); cudaFree(d->run_var);
}

static DeviceData upload_data(const float *x, const uint8_t *target,
                              size_t rows, int I, int O) {
  DeviceData d = {NULL,NULL,rows};
  d.x = alloc_f(rows * I); d.target = alloc_u8(rows * O);
  CUDA_OK(cudaMemcpy(d.x, x, rows * I * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d.target, target, rows * O, cudaMemcpyHostToDevice));
  return d;
}

static Workspace init_workspace(int maxB, int I, int H, int O) {
  Workspace w = {};
  w.maxB = maxB; w.partial_blocks = 1024;
  w.x = alloc_f((size_t)maxB * I);
  w.a1 = alloc_f((size_t)maxB * H); w.xhat = alloc_f((size_t)maxB * H);
  w.bn = alloc_f((size_t)maxB * H); w.dh = alloc_f((size_t)maxB * H);
  w.logits = alloc_f((size_t)maxB * O); w.invstd = alloc_f(H);
  w.idx = alloc_u32(maxB);
  w.loss_partial = alloc_d(w.partial_blocks);
  w.abs_partial = alloc_d(w.partial_blocks);
  w.correct_partial = alloc_u64(w.partial_blocks);
  w.h_partial = (double *)std::malloc(w.partial_blocks * sizeof(double));
  w.h_correct = (uint64_t *)std::malloc(w.partial_blocks * sizeof(uint64_t));
  if (!w.h_partial || !w.h_correct) {
    std::fprintf(stderr, "[methscope] upscale-train CUDA: out of host memory\n");
    std::exit(1);
  }
  return w;
}

static void free_workspace(Workspace *w) {
  cudaFree(w->x); cudaFree(w->a1); cudaFree(w->xhat); cudaFree(w->bn);
  cudaFree(w->logits); cudaFree(w->dh); cudaFree(w->invstd); cudaFree(w->idx);
  cudaFree(w->loss_partial); cudaFree(w->abs_partial); cudaFree(w->correct_partial);
  std::free(w->h_partial); std::free(w->h_correct);
}

static uint32_t *row_valid_counts(const uint8_t *target, size_t rows, int O,
                                  uint64_t *total, uint64_t *missing) {
  uint32_t *count = (uint32_t *)std::malloc(rows * sizeof(*count));
  if (!count) { std::fprintf(stderr, "[methscope] CUDA: out of host memory\n"); std::exit(1); }
  uint64_t tv = 0, tm = 0;
  for (size_t r = 0; r < rows; ++r) {
    uint32_t n = 0;
    const uint8_t *p = target + r * O;
    for (int k = 0; k < O; ++k) n += p[k] != 2;
    count[r] = n; tv += n; tm += (uint64_t)O - n;
  }
  if (total) *total = tv; if (missing) *missing = tm;
  return count;
}

static void copy_model_metadata(ms_updec_t *dst, const ms_updec_t *src) {
  size_t I = src->n_in;
  dst->bn_eps = src->bn_eps;
  std::memcpy(dst->imp_mean, src->imp_mean, I * sizeof(float));
  std::memcpy(dst->sc_mean, src->sc_mean, I * sizeof(float));
  std::memcpy(dst->sc_scale, src->sc_scale, I * sizeof(float));
}

static void gemm_forward1(cublasHandle_t h, const DeviceModel &m,
                          Workspace &w, int B, int I, int H) {
  const float one = 1.0f, zero = 0.0f;
  CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, H, B, I,
                        &one, m.theta[P_W1], I, w.x, I, &zero, w.a1, H));
  bias_relu_kernel<<<blocks_for((size_t)B*H),KTHREAD>>>(w.a1,m.theta[P_B1],B,H);
}

static void gemm_forward2(cublasHandle_t h, const DeviceModel &m,
                          Workspace &w, int B, int H, int O) {
  const float one = 1.0f, zero = 0.0f;
  CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, O, B, H,
                        &one, m.theta[P_W2], H, w.bn, H, &zero, w.logits, O));
  bias_kernel<<<blocks_for((size_t)B*O),KTHREAD>>>(w.logits,m.theta[P_B2],B,O);
}

static void forward_eval(cublasHandle_t h, const DeviceModel &m, Workspace &w,
                         int B, int I, int H, int O, float eps) {
  gemm_forward1(h,m,w,B,I,H);
  bn_eval_kernel<<<blocks_for((size_t)B*H),KTHREAD>>>(
      w.a1,w.bn,m.theta[P_GAMMA],m.theta[P_BETA],
      m.run_mean,m.run_var,B,H,eps);
  gemm_forward2(h,m,w,B,H,O);
}

static double validation_loss(cublasHandle_t h, const DeviceModel &m,
                              Workspace &w, const DeviceData &data,
                              const uint32_t *row_valid, int batch,
                              int I, int H, int O, float eps) {
  double total_loss = 0.0;
  for (size_t pos = 0; pos < data.rows; pos += batch) {
    int B = (int)((data.rows-pos < (size_t)batch) ? data.rows-pos : (size_t)batch);
    CUDA_OK(cudaMemcpy(w.x, data.x + pos*I, (size_t)B*I*sizeof(float),
                       cudaMemcpyDeviceToDevice));
    forward_eval(h,m,w,B,I,H,O,eps);
    uint64_t valid = 0;
    for (int b = 0; b < B; ++b) valid += row_valid[pos+b];
    size_t n = (size_t)B*O;
    int nb = blocks_for(n); if (nb > w.partial_blocks) nb = w.partial_blocks;
    loss_partial_kernel<<<nb,KTHREAD>>>(w.logits,data.target+pos*O,n,w.loss_partial);
    CUDA_OK(cudaMemcpy(w.h_partial,w.loss_partial,nb*sizeof(double),cudaMemcpyDeviceToHost));
    double raw = 0.0; for (int q = 0; q < nb; ++q) raw += w.h_partial[q];
    total_loss += raw / ((double)valid + 1e-8);
  }
  return total_loss;
}

static void test_metrics(cublasHandle_t h, const DeviceModel &m, Workspace &w,
                         const DeviceData &data, uint64_t valid, uint64_t missing,
                         int batch, int I, int H, int O, float eps,
                         ms_updec_cuda_result_t *result) {
  uint64_t correct = 0; double abs_sum = 0.0;
  for (size_t pos = 0; pos < data.rows; pos += batch) {
    int B = (int)((data.rows-pos < (size_t)batch) ? data.rows-pos : (size_t)batch);
    CUDA_OK(cudaMemcpy(w.x,data.x+pos*I,(size_t)B*I*sizeof(float),cudaMemcpyDeviceToDevice));
    forward_eval(h,m,w,B,I,H,O,eps);
    size_t n = (size_t)B*O;
    int nb = blocks_for(n); if (nb > w.partial_blocks) nb = w.partial_blocks;
    metrics_partial_kernel<<<nb,KTHREAD>>>(w.logits,data.target+pos*O,n,
                                           w.correct_partial,w.abs_partial);
    CUDA_OK(cudaMemcpy(w.h_correct,w.correct_partial,nb*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(w.h_partial,w.abs_partial,nb*sizeof(double),cudaMemcpyDeviceToHost));
    for (int q = 0; q < nb; ++q) { correct += w.h_correct[q]; abs_sum += w.h_partial[q]; }
  }
  result->test_valid = valid; result->test_missing = missing;
  result->test_accuracy = valid ? (double)correct / valid : NAN;
  result->test_mae = valid ? abs_sum / valid : NAN;
}

static double monotonic_seconds(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

extern "C" int ms_updec_cuda_available(void) {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

extern "C" int ms_updec_train_cuda(const ms_updec_cuda_config_t *cfg,
                                    const ms_updec_t *initial, ms_updec_t *best,
                                    ms_updec_cuda_result_t *result) {
  CUDA_OK(cudaSetDevice(cfg->device));
  cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop,cfg->device));
  std::fprintf(stderr,"[methscope] upscale-train: CUDA device %d: %s (%.0f MiB, cc %d.%d)\n",
               cfg->device,prop.name,(double)prop.totalGlobalMem/(1024*1024),prop.major,prop.minor);
  int I=initial->n_in,H=initial->n_hidden,O=initial->n_out;
  size_t maxBn=cfg->n_train<(size_t)cfg->batch+1?cfg->n_train:(size_t)cfg->batch+1;
  size_t vb=cfg->n_val<(size_t)cfg->batch?cfg->n_val:(size_t)cfg->batch;
  if(vb>maxBn)maxBn=vb;
  if(cfg->n_test){size_t tb=cfg->n_test<(size_t)cfg->batch?cfg->n_test:(size_t)cfg->batch;if(tb>maxBn)maxBn=tb;}

  DeviceModel dm = {}; init_device_model(&dm,initial);
  DeviceData train=upload_data(cfg->train_x,cfg->train_target,cfg->n_train,I,O);
  DeviceData val=upload_data(cfg->val_x,cfg->val_target,cfg->n_val,I,O);
  DeviceData test={NULL,NULL,0};
  if(cfg->n_test)test=upload_data(cfg->test_x,cfg->test_target,cfg->n_test,I,O);
  Workspace w=init_workspace((int)maxBn,I,H,O);
  uint64_t train_valid_total=0,val_valid_total=0,test_valid=0,test_missing=0;
  uint32_t *train_valid=row_valid_counts(cfg->train_target,cfg->n_train,O,&train_valid_total,NULL);
  uint32_t *val_valid=row_valid_counts(cfg->val_target,cfg->n_val,O,&val_valid_total,NULL);
  uint32_t *test_counts=NULL;
  if(cfg->n_test)test_counts=row_valid_counts(cfg->test_target,cfg->n_test,O,&test_valid,&test_missing);

  cublasHandle_t handle; CUBLAS_OK(cublasCreate(&handle));
  CUBLAS_OK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
  CUBLAS_OK(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
  copy_model_metadata(best,initial);
  double best_val=std::numeric_limits<double>::infinity();
  double sched_best=best_val, lr=cfg->lr, started=monotonic_seconds();
  int bad=0; uint64_t step=0;
  const float one=1.0f,zero=0.0f;

  for(int ep=0;ep<cfg->epochs;++ep) {
    double epoch_start=monotonic_seconds();
    const uint32_t *perm=cfg->epoch_indices+(size_t)ep*cfg->n_train;
    for(size_t pos=0;pos<cfg->n_train;) {
      size_t remain=cfg->n_train-pos; int B=remain<(size_t)cfg->batch?(int)remain:cfg->batch;
      if(remain==(size_t)cfg->batch+1)B=cfg->batch+1;
      const uint32_t *hidx=perm+pos;
      CUDA_OK(cudaMemcpy(w.idx,hidx,(size_t)B*sizeof(uint32_t),cudaMemcpyHostToDevice));
      gather_x_kernel<<<blocks_for((size_t)B*I),KTHREAD>>>(w.x,train.x,w.idx,B,I);
      gemm_forward1(handle,dm,w,B,I,H);
      bn_train_kernel<<<H,RTHREAD>>>(w.a1,w.xhat,w.bn,dm.theta[P_GAMMA],
          dm.theta[P_BETA],dm.run_mean,dm.run_var,w.invstd,B,H,
          initial->bn_eps,(float)cfg->bn_momentum);
      gemm_forward2(handle,dm,w,B,H,O);
      uint64_t valid=0;for(int b=0;b<B;++b)valid+=train_valid[hidx[b]];
      masked_grad_kernel<<<blocks_for((size_t)B*O),KTHREAD>>>(
          w.logits,train.target,w.idx,B,O,(float)((double)valid+1e-8));

      /* dW2^T = BN^T x dlogits; bytes are row-major dW2. */
      CUBLAS_OK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,H,O,B,
          &one,w.bn,H,w.logits,O,&zero,dm.grad[P_W2],H));
      bias_grad_kernel<<<O,RTHREAD>>>(w.logits,dm.grad[P_B2],B,O);
      /* dBN^T = W2^T x dlogits^T. */
      CUBLAS_OK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,H,B,O,
          &one,dm.theta[P_W2],H,w.logits,O,&zero,w.dh,H));
      bn_backward_kernel<<<H,RTHREAD>>>(w.dh,dm.grad[P_GAMMA],dm.grad[P_BETA],
          w.dh,w.xhat,w.invstd,dm.theta[P_GAMMA],B,H);
      relu_backward_kernel<<<blocks_for((size_t)B*H),KTHREAD>>>(w.dh,w.a1,(size_t)B*H);
      /* dW1^T = X^T x dz1. */
      CUBLAS_OK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,I,H,B,
          &one,w.x,I,w.dh,H,&zero,dm.grad[P_W1],I));
      bias_grad_kernel<<<H,RTHREAD>>>(w.dh,dm.grad[P_B1],B,H);

      ++step;
      float ib1=(float)(1.0/(1.0-std::pow(0.9,(double)step)));
      float ib2=(float)(1.0/(1.0-std::pow(0.999,(double)step)));
      for(int p=0;p<P_COUNT;++p)adam_kernel<<<blocks_for(dm.n[p]),KTHREAD>>>(
          dm.theta[p],dm.grad[p],dm.mom[p],dm.vel[p],dm.n[p],
          (float)lr,(float)cfg->weight_decay,ib1,ib2);
      pos+=B;
    }
    double vl=validation_loss(handle,dm,w,val,val_valid,cfg->batch,I,H,O,initial->bn_eps);
    if(!std::isfinite(vl)) {
      std::fprintf(stderr,"[methscope] upscale-train CUDA: non-finite validation loss\n");
      std::exit(1);
    }
    if(vl<sched_best*(1.0-1e-4)){sched_best=vl;bad=0;}
    else if(++bad>2){lr*=0.5;bad=0;}
    if(vl<best_val){best_val=vl;download_model(best,&dm);}
    std::fprintf(stderr,
      "[methscope] upscale-train: epoch %d/%d  val_loss %.9g  lr %.9g  gpu_seconds %.3f\n",
      ep+1,cfg->epochs,vl,lr,monotonic_seconds()-epoch_start);
  }
  result->best_val=best_val; result->test_accuracy=NAN; result->test_mae=NAN;
  result->test_valid=result->test_missing=0;
  if(cfg->n_test){
    upload_model(&dm,best);
    test_metrics(handle,dm,w,test,test_valid,test_missing,cfg->batch,I,H,O,
                 initial->bn_eps,result);
  }
  CUDA_OK(cudaDeviceSynchronize());
  std::fprintf(stderr,"[methscope] upscale-train: CUDA total seconds %.3f\n",
               monotonic_seconds()-started);

  cublasDestroy(handle);
  free_device_model(&dm);
  cudaFree(train.x);cudaFree(train.target);cudaFree(val.x);cudaFree(val.target);
  if(cfg->n_test){cudaFree(test.x);cudaFree(test.target);}
  free_workspace(&w);
  std::free(train_valid);std::free(val_valid);std::free(test_counts);
  return 0;
}
