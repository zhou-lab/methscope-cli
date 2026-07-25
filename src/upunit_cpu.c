// SPDX-License-Identifier: AGPL-3.0-or-later
/* CPU backend for `upscale-train`, mirroring src/upunit_cuda.cu.
 *
 * WHY THIS EXISTS
 *   Training was CUDA-only, so the one pipeline the docs describe end to end
 *   could not actually be finished by a reader without a GPU. The work per
 *   step is small -- one rank x input gemv plus a decode over the batch, on
 *   the order of half a megaflop -- so the GPU was never buying arithmetic. It
 *   was buying throughput across the ~700 units, and units train completely
 *   independently (that is what makes the run resumable), so a thread pool
 *   over units recovers most of it.
 *
 * WHAT IS SHARED WITH THE CUDA PATH
 *   The on-disk contracts: MSURAW2 / MSUIDX1 layouts, the UPUCK1 checkpoint
 *   (including index_checksum and run_checksum), and the emitted UPDEC2. A
 *   checkpoint written here resumes there and vice versa, which is the point
 *   -- you can featurize and start on CPU, then finish on a GPU node.
 *
 * WHAT IS NOT BIT-IDENTICAL
 *   Nothing can be. The CUDA factor_back accumulates dz with atomicAdd, whose
 *   summation order is not fixed, so two GPU runs of the same seed already
 *   differ in the last bits. The contract here is the same arithmetic in the
 *   same order per step -- same PCG stream, same Adam, same early stopping --
 *   so results agree to floating-point noise rather than exactly.
 *
 * NOT SUPPORTED
 *   --trunk. The frozen shared trunk is a research path (`_upscale
 *   trunk-train`) and is rejected here rather than silently ignored. */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "updec2.h"
#include "upsplit.h"
#include "upunit_cuda.h"

#define THREADS_MAX 64u

static void cdie(const char *m) {
  fprintf(stderr, "[methscope] upscale-train: %s\n", m); exit(1);
}
static void cdiep(const char *m, const char *p) {
  fprintf(stderr, "[methscope] upscale-train: %s: %s\n", m, p); exit(1);
}
static void *xmal(size_t n) { void *p = malloc(n ? n : 1); if (!p) cdie("out of memory"); return p; }
static void *xcal(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) cdie("out of memory"); return p; }

/* ---- on-disk structs, identical to the CUDA backend ---------------------- */
typedef struct { char magic[8]; uint32_t version, n_cells, n_reps, n_patterns;
  uint64_t n_cpg; uint32_t sampled_per_cell, flags;
  uint64_t groups_offset, truth_offset, records_offset, record_bytes; } MsurHeader;
#pragma pack(push,1)
typedef struct { char magic[8]; uint32_t version, flags, pattern_length, target_unit_cpgs;
  uint32_t n_units, n_real_memberships, n_pna_units, reserved32;
  uint64_t n_cpg, n_real_cpg, n_pna_cpg;
  uint64_t unit_offset, cpg_offset, membership_offset, file_bytes;
  uint64_t pattern_checksum, reserved0, reserved1, reserved2; } MsuiHeader;
typedef struct { uint64_t output_offset; uint32_t first_membership, membership_count, cpg_count, flags; } MsuiUnit;
typedef struct { uint64_t pattern_key, output_offset; uint32_t count, unit; } MsuiMembership;
typedef struct { char magic[8]; uint32_t version, unit, mode, rank, activation, best_step, cpg_count, input_dim;
  uint64_t index_checksum, run_checksum, param_floats; float best_mae; uint32_t reserved; } CheckHeader;
#pragma pack(pop)

/* ---- the same PCG and hashes, so the split and sampling match ------------- */
typedef struct { uint64_t state, inc; } Pcg;
static uint32_t rnd(Pcg *p) {
  uint64_t o = p->state;
  p->state = o * UINT64_C(6364136223846793005) + p->inc;
  uint32_t x = (uint32_t)(((o >> 18) ^ o) >> 27), r = (uint32_t)(o >> 59);
  return (x >> r) | (x << ((-r) & 31));
}
static void pseed(Pcg *p, uint64_t s) {
  p->state = 0; p->inc = UINT64_C(1442695040888963407);
  rnd(p); p->state += s; rnd(p);
}
static uint64_t hash64(uint64_t x) {
  x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}
static uint64_t fnv(uint64_t h, const void *p, size_t n) {
  const uint8_t *q = (const uint8_t *)p;
  for (size_t i = 0; i < n; ++i) { h ^= q[i]; h *= UINT64_C(1099511628211); }
  return h;
}
static float sigf(float x) {
  if (x >= 0) { float z = expf(-x); return 1.0f / (1.0f + z); }
  float z = expf(x); return z / (1.0f + z);
}

/* ---- mmap ---------------------------------------------------------------- */
typedef struct { int fd; size_t n; uint8_t *p; } Map;
static Map mapread(const char *path) {
  Map m; m.fd = open(path, O_RDONLY);
  if (m.fd < 0) cdiep("cannot open", path);
  struct stat st;
  if (fstat(m.fd, &st) || st.st_size <= 0) cdiep("cannot stat", path);
  m.n = (size_t)st.st_size;
  m.p = mmap(NULL, m.n, PROT_READ, MAP_SHARED, m.fd, 0);
  if (m.p == MAP_FAILED) cdiep("cannot mmap", path);
  return m;
}
static void unmap(Map *m) { if (m->p) munmap(m->p, m->n); if (m->fd >= 0) close(m->fd); }
static int range(uint64_t off, uint64_t len, uint64_t total) {
  return off <= total && len <= total - off;
}

/* ---- target sampling, same arithmetic as the CUDA host code -------------- */
static int observed(const uint32_t *a, uint32_t n, uint32_t x) {
  uint32_t lo = 0, hi = n;
  while (lo < hi) { uint32_t mid = lo + (hi - lo) / 2;
    if (a[mid] < x) lo = mid + 1; else hi = mid; }
  return lo < n && a[lo] == x;
}
static uint32_t targets(const MsurHeader *h, const uint8_t *base, const uint32_t *cpg,
                        uint64_t begin, uint32_t O, uint32_t cell, uint32_t rep,
                        uint64_t start, uint32_t want, uint32_t *id, float *y) {
  size_t row = (size_t)rep * h->n_cells + cell;
  const uint8_t *rec = base + h->records_offset + row * h->record_bytes;
  const uint32_t *sel = (const uint32_t *)(rec + (size_t)h->n_patterns * 8);
  const uint16_t *truth = (const uint16_t *)(base + h->truth_offset) + (size_t)cell * h->n_cpg;
  uint32_t n = 0, seen = 0, q = (uint32_t)(start % (O ? O : 1));
  while (n < want && seen < O) {
    uint32_t pos = cpg[begin + q];
    uint16_t v = truth[pos];
    if (v != UINT16_MAX && !observed(sel, h->sampled_per_cell, pos)) {
      id[n] = q; y[n] = (float)v / 65534.0f; ++n;
    }
    ++seen; if (++q >= O) q = 0;
  }
  return n;
}

/* ---- one processing unit's parameters ------------------------------------ */
typedef struct { float *t, *m, *v; size_t n; } Param;
static void prand(Param *p, size_t n, float scale, uint64_t s) {
  p->n = n; p->t = xmal(n * 4); p->m = xcal(n, 4); p->v = xcal(n, 4);
  for (size_t q = 0; q < n; ++q) {
    uint64_t z = hash64(s + q * UINT64_C(0x9e3779b97f4a7c15));
    float u = (float)((z >> 40) + .5) * (1.0f / 16777216.0f);
    p->t[q] = (2 * u - 1) * scale;
  }
}
static void pzero(Param *p, size_t n) {
  p->n = n; p->t = xcal(n, 4); p->m = xcal(n, 4); p->v = xcal(n, 4);
}
static void pcopy(Param *p, const float *src, size_t n) {
  p->n = n; p->t = xmal(n * 4); memcpy(p->t, src, n * 4);
  p->m = xcal(n, 4); p->v = xcal(n, 4);
}
static void pfree(Param *p) { free(p->t); free(p->m); free(p->v); }

typedef struct { Param A, a, E, b; int direct, I, R, O; } Net;

static void net_make(Net *n, int direct, int I, int R, int O, const float *bias, uint64_t s) {
  memset(n, 0, sizeof(*n));
  n->direct = direct; n->I = I; n->R = R; n->O = O;
  if (direct) {
    prand(&n->E, (size_t)O * I, sqrtf(6.0f / (I + 1)), s ^ 11);
    pcopy(&n->b, bias, (size_t)O);
  } else {
    prand(&n->A, (size_t)R * I, sqrtf(6.0f / (I + R)), s ^ 13);
    pzero(&n->a, (size_t)R);
    prand(&n->E, (size_t)O * R, .02f, s ^ 17);
    pcopy(&n->b, bias, (size_t)O);
  }
}
static void net_free(Net *n) {
  if (!n->direct) { pfree(&n->A); pfree(&n->a); }
  pfree(&n->E); pfree(&n->b);
}
static size_t net_floats(const Net *n) {
  return (n->direct ? (size_t)n->O * n->I
                    : (size_t)n->R * n->I + n->R + (size_t)n->O * n->R) + n->O;
}
static void net_flatten(const Net *n, float *out) {
  size_t q = 0;
  if (!n->direct) {
    memcpy(out + q, n->A.t, n->A.n * 4); q += n->A.n;
    memcpy(out + q, n->a.t, n->a.n * 4); q += n->a.n;
  }
  memcpy(out + q, n->E.t, n->E.n * 4); q += n->E.n;
  memcpy(out + q, n->b.t, n->b.n * 4);
}

/* z = act(A.x + a) */
static void hidden(const Net *n, const float *x, float *z, int act) {
  for (int r = 0; r < n->R; ++r) {
    const float *row = n->A.t + (size_t)r * n->I;
    float s = 0;
    for (int j = 0; j < n->I; ++j) s += row[j] * x[j];
    s += n->a.t[r];
    z[r] = (act && s < 0) ? .01f * s : s;
  }
}

static double eval_rows(const Net *n, const float *X, int act,
                        const uint32_t *const *ids, const float *const *ys,
                        const uint32_t *lens, const size_t *rows, uint32_t nrow,
                        float *z) {
  double ae = 0; uint64_t cnt = 0;
  for (uint32_t k = 0; k < nrow; ++k) {
    const float *x = X + rows[k] * (size_t)n->I;
    if (!n->direct) hidden(n, x, z, act);
    for (uint32_t q = 0; q < lens[k]; ++q) {
      uint32_t c = ids[k][q];
      float s = n->b.t[c];
      if (n->direct) { const float *W = n->E.t + (size_t)c * n->I;
        for (int j = 0; j < n->I; ++j) s += W[j] * x[j]; }
      else { const float *E = n->E.t + (size_t)c * n->R;
        for (int r = 0; r < n->R; ++r) s += E[r] * z[r]; }
      float d = sigf(s) - ys[k][q];
      ae += fabs((double)d); ++cnt;
    }
  }
  return cnt ? ae / (double)cnt : NAN;
}

static void adam(float *t, float *m, float *v, float g, float lr, float ib1, float ib2) {
  float mm = .9f * *m + .1f * g, vv = .999f * *v + .001f * g * g;
  *m = mm; *v = vv;
  *t -= lr * (mm * ib1) / (sqrtf(vv * ib2) + 1e-8f);
}

/* One optimizer step over a batch of targets; mirrors factor_back/direct_back. */
static void step_batch(Net *n, const float *x, const uint32_t *id, const float *y,
                       uint32_t B, float lr, float wd, float ib1, float ib2,
                       int act, float *z, float *dz) {
  if (n->direct) {
    for (uint32_t q = 0; q < B; ++q) {
      uint32_t c = id[q];
      float *W = n->E.t + (size_t)c * n->I;
      float *Wm = n->E.m + (size_t)c * n->I, *Wv = n->E.v + (size_t)c * n->I;
      float s = n->b.t[c];
      for (int j = 0; j < n->I; ++j) s += W[j] * x[j];
      float p = sigf(s), dl = 2 * (p - y[q]) * p * (1 - p) / B;
      for (int j = 0; j < n->I; ++j) adam(&W[j], &Wm[j], &Wv[j], dl * x[j] + wd * W[j], lr, ib1, ib2);
      adam(&n->b.t[c], &n->b.m[c], &n->b.v[c], dl, lr, ib1, ib2);
    }
    return;
  }
  hidden(n, x, z, act);
  memset(dz, 0, (size_t)n->R * 4);
  for (uint32_t q = 0; q < B; ++q) {
    uint32_t c = id[q];
    float *E = n->E.t + (size_t)c * n->R;
    float *Em = n->E.m + (size_t)c * n->R, *Ev = n->E.v + (size_t)c * n->R;
    float s = n->b.t[c];
    for (int r = 0; r < n->R; ++r) s += E[r] * z[r];
    float p = sigf(s), dl = 2 * (p - y[q]) * p * (1 - p) / B;
    for (int r = 0; r < n->R; ++r) {
      float old = E[r];              /* dz uses the pre-update weight, as on GPU */
      dz[r] += dl * old;
      adam(&E[r], &Em[r], &Ev[r], dl * z[r] + wd * old, lr, ib1, ib2);
    }
    adam(&n->b.t[c], &n->b.m[c], &n->b.v[c], dl, lr, ib1, ib2);
  }
  if (act) for (int r = 0; r < n->R; ++r) if (!(z[r] > 0)) dz[r] *= .01f;
  for (int r = 0; r < n->R; ++r) {
    float *Ar = n->A.t + (size_t)r * n->I;
    float *Am = n->A.m + (size_t)r * n->I, *Av = n->A.v + (size_t)r * n->I;
    for (int j = 0; j < n->I; ++j) adam(&Ar[j], &Am[j], &Av[j], dz[r] * x[j] + wd * Ar[j], lr, ib1, ib2);
    adam(&n->a.t[r], &n->a.m[r], &n->a.v[r], dz[r], lr, ib1, ib2);
  }
}

/* ---- checkpoints (UPUCK1, byte-compatible with the CUDA backend) ---------- */
static void ckpath(char *p, size_t n, const char *d, uint32_t u) {
  if (snprintf(p, n, "%s/unit_%06u.upuck", d, u) >= (int)n) cdie("checkpoint path too long");
}
static int ck_load(const char *path, uint32_t ui, uint64_t sum, uint64_t runsum,
                   int mode, int rank, int act, int O, int I,
                   float **par, size_t *npar, float *mae, uint32_t *step) {
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  CheckHeader h;
  if (fread(&h, 1, sizeof(h), f) != sizeof(h) || memcmp(h.magic, "UPUCK1", 6) ||
      h.version != 1 || h.unit != ui || h.index_checksum != sum ||
      h.run_checksum != runsum || h.mode != (uint32_t)mode || h.rank != (uint32_t)rank ||
      h.activation != (uint32_t)act || h.cpg_count != (uint32_t)O ||
      h.input_dim != (uint32_t)I) { fclose(f); return 0; }
  float *p = xmal((size_t)h.param_floats * 4);
  if (fread(p, 4, h.param_floats, f) != h.param_floats || fgetc(f) != EOF) {
    fclose(f); free(p); return 0;
  }
  fclose(f);
  *par = p; *npar = h.param_floats; *mae = h.best_mae; *step = h.best_step;
  return 1;
}
static void ck_save(const char *path, uint32_t ui, uint64_t sum, uint64_t runsum,
                    int mode, int rank, int act, int O, int I,
                    const float *par, size_t npar, float mae, uint32_t step) {
  char tmp[4096];
  if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) cdie("checkpoint path too long");
  FILE *f = fopen(tmp, "wb");
  if (!f) cdiep("cannot create checkpoint", tmp);
  CheckHeader h; memset(&h, 0, sizeof(h));
  memcpy(h.magic, "UPUCK1", 6);
  h.version = 1; h.unit = ui; h.mode = mode; h.rank = rank; h.activation = act;
  h.best_step = step; h.cpg_count = O; h.input_dim = I;
  h.index_checksum = sum; h.run_checksum = runsum; h.param_floats = npar; h.best_mae = mae;
  if (fwrite(&h, 1, sizeof(h), f) != sizeof(h) ||
      (npar && fwrite(par, 4, npar, f) != npar)) cdiep("write failed", tmp);
  if (fclose(f) || rename(tmp, path)) cdiep("cannot finalize checkpoint", path);
}

/* ---- per-unit work, run by the thread pool ------------------------------- */
typedef struct {
  const ms_upunit_config_t *c;
  const MsurHeader *h; const MsuiHeader *ih;
  const uint8_t *data; const MsuiUnit *units; const uint32_t *cpg;
  const float *X; const float *all_bias;
  const uint32_t *train; uint32_t n_train;
  const uint32_t *val; uint32_t n_val;
  int I; uint64_t isum, runsum;
  float *unit_mae; uint32_t *unit_step;
  uint32_t next;                       /* shared cursor */
  uint32_t done, resumed;
  pthread_mutex_t lock;
} Job;

static void *worker(void *arg) {
  Job *J = (Job *)arg;
  const ms_upunit_config_t *c = J->c;
  const MsurHeader *h = J->h;
  uint32_t *bid = xmal((size_t)c->batch * 4);
  float *by = xmal((size_t)c->batch * 4);
  float *z = xmal((size_t)(c->mixed_bottleneck > c->pure_bottleneck ?
                           c->mixed_bottleneck : c->pure_bottleneck) * 4 + 4);
  float *dz = xmal((size_t)(c->mixed_bottleneck > c->pure_bottleneck ?
                            c->mixed_bottleneck : c->pure_bottleneck) * 4 + 4);
  for (;;) {
    pthread_mutex_lock(&J->lock);
    uint32_t ui = J->next < J->ih->n_units ? J->next++ : UINT32_MAX;
    pthread_mutex_unlock(&J->lock);
    if (ui == UINT32_MAX) break;

    const MsuiUnit *u = &J->units[ui];
    int pure = (u->flags & 1) != 0;
    int direct = c->mixed_direct && !pure;
    int R = direct ? 0 : (int)(pure ? c->pure_bottleneck : c->mixed_bottleneck);
    int O = (int)u->cpg_count;
    char path[4096]; ckpath(path, sizeof(path), c->work_dir, ui);

    /* validation rows, drawn exactly as the CUDA backend draws them */
    Pcg er; pseed(&er, c->seed + 101 + ui);
    uint32_t nrow = 0, tries = 0, maxtries = c->eval_rows * 32 > 32 ? c->eval_rows * 32 : 32;
    uint32_t **eids = xcal(c->eval_rows, sizeof(uint32_t *));
    float **eys = xcal(c->eval_rows, sizeof(float *));
    uint32_t *elen = xcal(c->eval_rows, 4);
    size_t *erow = xcal(c->eval_rows, sizeof(size_t));
    while (nrow < c->eval_rows && tries++ < maxtries) {
      uint32_t cell = J->val[rnd(&er) % J->n_val], rep = rnd(&er) % h->n_reps;
      uint32_t want = c->batch < u->cpg_count ? c->batch : u->cpg_count;
      uint32_t *idv = xmal((size_t)want * 4); float *yv = xmal((size_t)want * 4);
      uint32_t B = targets(h, J->data, J->cpg, u->output_offset, u->cpg_count,
                           cell, rep, rnd(&er), want, idv, yv);
      if (!B) { free(idv); free(yv); continue; }
      eids[nrow] = idv; eys[nrow] = yv; elen[nrow] = B;
      erow[nrow] = (size_t)rep * h->n_cells + cell; ++nrow;
    }
    if (!nrow) cdie("could not form validation rows");

    float *par = NULL; size_t npar = 0; float bestmae = 0; uint32_t beststep = 0;
    if (ck_load(path, ui, J->isum, J->runsum, direct ? 0 : 1, R, c->activation, O, J->I,
                &par, &npar, &bestmae, &beststep)) {
      free(par);
      pthread_mutex_lock(&J->lock);
      J->unit_mae[ui] = bestmae; J->unit_step[ui] = beststep; ++J->resumed; ++J->done;
      pthread_mutex_unlock(&J->lock);
      goto cleanup;
    }

    float *bias = xmal((size_t)O * 4);
    for (int o = 0; o < O; ++o) bias[o] = J->all_bias[J->cpg[u->output_offset + o]];
    Net net; net_make(&net, direct, J->I, R, O, bias, c->seed ^ ((uint64_t)ui << 32));
    free(bias);

    size_t nf = net_floats(&net);
    float *best = xmal(nf * 4);
    double v = eval_rows(&net, J->X, c->activation,
                         (const uint32_t *const *)eids, (const float *const *)eys,
                         elen, erow, nrow, z);
    bestmae = (float)v; net_flatten(&net, best);
    Pcg rr; pseed(&rr, c->seed ^ ui ^ UINT64_C(0xd1b54a32d192ed03));
    uint32_t bad = 0, step;
    for (step = 1; step <= c->max_steps; ++step) {
      uint32_t cell = J->train[rnd(&rr) % J->n_train], rep = rnd(&rr) % h->n_reps;
      uint32_t want = c->batch < u->cpg_count ? c->batch : u->cpg_count;
      uint32_t B = targets(h, J->data, J->cpg, u->output_offset, u->cpg_count,
                           cell, rep, rnd(&rr), want, bid, by);
      if (!B) cdie("could not form unit target batch");
      const float *x = J->X + ((size_t)rep * h->n_cells + cell) * (size_t)J->I;
      float ib1 = 1.0f / (1 - powf(.9f, (float)step));
      float ib2 = 1.0f / (1 - powf(.999f, (float)step));
      step_batch(&net, x, bid, by, B, (float)c->learning_rate, (float)c->weight_decay,
                 ib1, ib2, c->activation, z, dz);
      if (step % c->eval_every == 0) {
        v = eval_rows(&net, J->X, c->activation,
                      (const uint32_t *const *)eids, (const float *const *)eys,
                      elen, erow, nrow, z);
        if (v + 1e-7 < bestmae) { bestmae = (float)v; beststep = step; net_flatten(&net, best); bad = 0; }
        else if (step >= c->min_steps) ++bad;
        if (step >= c->min_steps && bad >= c->patience) break;
      }
    }
    ck_save(path, ui, J->isum, J->runsum, direct ? 0 : 1, R, c->activation, O, J->I,
            best, nf, bestmae, beststep);
    net_free(&net); free(best);
    pthread_mutex_lock(&J->lock);
    J->unit_mae[ui] = bestmae; J->unit_step[ui] = beststep; ++J->done;
    if (J->done % 10 == 0 || J->done == J->ih->n_units)
      fprintf(stderr, "[methscope] upscale-train: units %u/%u (resumed=%u) last_val_mae=%.6f\n",
              J->done, J->ih->n_units, J->resumed, bestmae);
    pthread_mutex_unlock(&J->lock);

  cleanup:
    for (uint32_t k = 0; k < nrow; ++k) { free(eids[k]); free(eys[k]); }
    free(eids); free(eys); free(elen); free(erow);
  }
  free(bid); free(by); free(z); free(dz);
  return NULL;
}

int ms_upunit_train_cpu(const ms_upunit_config_t *c) {
  if (c->trunk_path) cdie("the CPU backend does not support --trunk; build with make CUDA=1");
  if (c->pilot_units_path) cdie("the CPU backend does not support --pilot-units");

  Map data = mapread(c->data_path), idx = mapread(c->index_path);
  const MsurHeader *h = (const MsurHeader *)data.p;
  const MsuiHeader *ih = (const MsuiHeader *)idx.p;
  if (data.n < sizeof(MsurHeader) || memcmp(h->magic, "MSURAW2\0", 8) || h->version != 2 ||
      !(h->flags & 1) || !h->truth_offset)
    cdie("training requires embedded-truth MSURAW2");
  if (idx.n < sizeof(MsuiHeader) || memcmp(ih->magic, "MSUIDX1", 7) || ih->version != 1 ||
      h->n_cpg != ih->n_cpg)
    cdie("training requires matching MSUIDX1");
  uint32_t P = c->patterns ? c->patterns : h->n_patterns;
  if (!P || P > h->n_patterns) cdie("invalid pattern count");
  int F = (c->feature_mode == MS_UPFEATURE_BETA ? 1 : 2) * (int)P, I = F;

  uint64_t rows = (uint64_t)h->n_cells * h->n_reps;
  uint64_t recend = h->records_offset + rows * h->record_bytes;
  uint64_t truthbytes = (uint64_t)h->n_cells * h->n_cpg * 2;
  if (recend > data.n || !range(h->truth_offset, truthbytes, data.n) ||
      ih->file_bytes > idx.n ||
      !range(ih->unit_offset, (uint64_t)ih->n_units * sizeof(MsuiUnit), idx.n) ||
      !range(ih->cpg_offset, ih->n_cpg * 4, idx.n))
    cdie("truncated training payload");
  const MsuiUnit *units = (const MsuiUnit *)(idx.p + ih->unit_offset);
  const uint32_t *cpg = (const uint32_t *)(idx.p + ih->cpg_offset);
  const MsuiMembership *members = (const MsuiMembership *)(idx.p + ih->membership_offset);

  /* checksums exactly as the CUDA backend computes them, so checkpoints interop */
  uint64_t isum = fnv(UINT64_C(1469598103934665603), idx.p, ih->file_bytes);
  uint64_t trunk_checksum = 0;
  uint64_t runsum = fnv(UINT64_C(1469598103934665603), h, sizeof(*h));
  runsum = fnv(runsum, &data.n, sizeof(data.n));
  runsum = fnv(runsum, &c->patterns, sizeof(c->patterns));
  runsum = fnv(runsum, &c->feature_mode, sizeof(c->feature_mode));
  runsum = fnv(runsum, &trunk_checksum, sizeof(trunk_checksum));
  runsum = fnv(runsum, &c->pure_bottleneck, sizeof(c->pure_bottleneck));
  runsum = fnv(runsum, &c->mixed_bottleneck, sizeof(c->mixed_bottleneck));
  runsum = fnv(runsum, &c->mixed_direct, sizeof(c->mixed_direct));
  runsum = fnv(runsum, &c->activation, sizeof(c->activation));
  runsum = fnv(runsum, &c->min_steps, sizeof(c->min_steps));
  runsum = fnv(runsum, &c->max_steps, sizeof(c->max_steps));
  runsum = fnv(runsum, &c->eval_every, sizeof(c->eval_every));
  runsum = fnv(runsum, &c->patience, sizeof(c->patience));
  runsum = fnv(runsum, &c->batch, sizeof(c->batch));
  runsum = fnv(runsum, &c->eval_rows, sizeof(c->eval_rows));
  runsum = fnv(runsum, &c->seed, sizeof(c->seed));
  runsum = fnv(runsum, &c->learning_rate, sizeof(c->learning_rate));
  runsum = fnv(runsum, &c->weight_decay, sizeof(c->weight_decay));

  /* cell split: curated file, else the seeded 70/15/15 shuffle */
  uint32_t *train = xmal((size_t)h->n_cells * 4), *val = xmal((size_t)h->n_cells * 4);
  uint32_t n_train = 0, n_val = 0, n_test = 0;
  if (c->split_path) {
    uint8_t *lab = xmal(h->n_cells);
    ms_upsplit_load("upscale-train", c->split_path, h->n_cells, lab);
    for (uint32_t i = 0; i < h->n_cells; ++i) {
      if (lab[i] == MS_UPSPLIT_TRAIN) train[n_train++] = i;
      else if (lab[i] == MS_UPSPLIT_VAL) val[n_val++] = i;
      else ++n_test;
    }
    runsum = fnv(runsum, lab, h->n_cells);
    free(lab);
  } else {
    uint32_t *cells = xmal((size_t)h->n_cells * 4);
    for (uint32_t i = 0; i < h->n_cells; ++i) cells[i] = i;
    Pcg sr; pseed(&sr, c->seed);
    for (uint32_t q = h->n_cells; q > 1; --q) {
      uint32_t j = rnd(&sr) % q, t = cells[q - 1]; cells[q - 1] = cells[j]; cells[j] = t;
    }
    size_t nt = (size_t)h->n_cells * 70 / 100, nv = (size_t)h->n_cells * 15 / 100;
    if (!nt || !nv || nt + nv >= h->n_cells) cdie("too few source cells for split");
    for (size_t i = 0; i < nt; ++i) train[n_train++] = cells[i];
    for (size_t i = nt; i < nt + nv; ++i) val[n_val++] = cells[i];
    n_test = h->n_cells - (uint32_t)(nt + nv);
    free(cells);
  }

  /* feature standardization from the training cells only */
  float *mean = xcal((size_t)F, 4), *scale = xcal((size_t)F, 4);
  {
    double *sum = xcal((size_t)F, sizeof(double)), *ss = xcal((size_t)F, sizeof(double));
    uint64_t *cnt = xcal((size_t)F, sizeof(uint64_t));
    uint64_t tr = (uint64_t)n_train * h->n_reps;
    for (uint32_t r = 0; r < h->n_reps; ++r) for (uint32_t k = 0; k < n_train; ++k) {
      const uint8_t *rec = data.p + h->records_offset +
        ((size_t)r * h->n_cells + train[k]) * h->record_bytes;
      const float *b = (const float *)rec;
      const uint32_t *nn = (const uint32_t *)(rec + (size_t)h->n_patterns * 4);
      for (uint32_t p = 0; p < P; ++p) {
        uint32_t j = c->feature_mode == MS_UPFEATURE_BETA ? p : 2 * p;
        if (isfinite(b[p])) { sum[j] += b[p]; cnt[j]++; }
        if (c->feature_mode != MS_UPFEATURE_BETA) {
          double a = c->feature_mode == MS_UPFEATURE_COUNT ? log1p((double)nn[p])
                                                           : (isfinite(b[p]) ? 0.0 : 1.0);
          sum[j + 1] += a; cnt[j + 1]++;
        }
      }
    }
    for (uint32_t p = 0; p < P; ++p) {
      uint32_t j = c->feature_mode == MS_UPFEATURE_BETA ? p : 2 * p;
      if (!cnt[j]) cdie("one MRMP is always missing in training");
      mean[j] = (float)(sum[j] / cnt[j]);
      if (c->feature_mode != MS_UPFEATURE_BETA) mean[j + 1] = (float)(sum[j + 1] / cnt[j + 1]);
    }
    for (uint32_t r = 0; r < h->n_reps; ++r) for (uint32_t k = 0; k < n_train; ++k) {
      const uint8_t *rec = data.p + h->records_offset +
        ((size_t)r * h->n_cells + train[k]) * h->record_bytes;
      const float *b = (const float *)rec;
      const uint32_t *nn = (const uint32_t *)(rec + (size_t)h->n_patterns * 4);
      for (uint32_t p = 0; p < P; ++p) {
        uint32_t j = c->feature_mode == MS_UPFEATURE_BETA ? p : 2 * p;
        if (isfinite(b[p])) { double d = b[p] - mean[j]; ss[j] += d * d; }
        if (c->feature_mode != MS_UPFEATURE_BETA) {
          double a = c->feature_mode == MS_UPFEATURE_COUNT ? log1p((double)nn[p])
                                                           : (isfinite(b[p]) ? 0.0 : 1.0);
          double d = a - mean[j + 1]; ss[j + 1] += d * d;
        }
      }
    }
    for (int j = 0; j < F; ++j) {
      double q = ss[j] / (double)tr;
      scale[j] = (float)sqrt(q > 1e-12 ? q : 1e-12);
      if (!isfinite(mean[j]) || !isfinite(scale[j]) || !(scale[j] > 0))
        cdie("invalid feature preprocessing");
    }
    free(sum); free(ss); free(cnt);
  }

  /* the standardized feature matrix for every (cell, replicate) row */
  float *X = xmal((size_t)rows * F * 4);
  for (uint64_t r = 0; r < rows; ++r) {
    const uint8_t *rec = data.p + h->records_offset + r * h->record_bytes;
    const float *b = (const float *)rec;
    const uint32_t *nn = (const uint32_t *)(rec + (size_t)h->n_patterns * 4);
    float *x = X + r * F;
    for (uint32_t p = 0; p < P; ++p) {
      uint32_t j = c->feature_mode == MS_UPFEATURE_BETA ? p : 2 * p;
      x[j] = isfinite(b[p]) ? (b[p] - mean[j]) / scale[j] : 0;
      if (c->feature_mode != MS_UPFEATURE_BETA) {
        double a = c->feature_mode == MS_UPFEATURE_COUNT ? log1p((double)nn[p])
                                                         : (isfinite(b[p]) ? 0.0 : 1.0);
        x[j + 1] = (float)((a - mean[j + 1]) / scale[j + 1]);
      }
    }
  }

  unsigned nth = c->threads ? c->threads : 1;
  if (nth > THREADS_MAX) nth = THREADS_MAX;
  fprintf(stderr, "[methscope] upscale-train: CPU backend; cells=%u reps=%u patterns=%u "
          "input=%d units=%u split=%u/%u/%u threads=%u\n",
          h->n_cells, h->n_reps, P, I, ih->n_units, n_train, n_val, n_test, nth);

  /* genome-wide train-cell bias priors */
  float *all_bias = xcal(ih->n_cpg, 4);
  {
    uint16_t *acount = xcal(ih->n_cpg, 2);
    const uint16_t *truth = (const uint16_t *)(data.p + h->truth_offset);
    for (uint32_t k = 0; k < n_train; ++k) {
      const uint16_t *t = truth + (size_t)train[k] * h->n_cpg;
      for (uint64_t pos = 0; pos < ih->n_cpg; ++pos) {
        uint16_t v = t[pos];
        if (v != UINT16_MAX) { all_bias[pos] += (float)v / 65534.0f; acount[pos]++; }
      }
    }
    for (uint64_t pos = 0; pos < ih->n_cpg; ++pos) {
      double p = acount[pos] ? all_bias[pos] / acount[pos] : .5;
      if (p < .01) p = .01;
      if (p > .99) p = .99;
      all_bias[pos] = (float)log(p / (1 - p));
    }
    free(acount);
  }

  Job J; memset(&J, 0, sizeof(J));
  J.c = c; J.h = h; J.ih = ih; J.data = data.p; J.units = units; J.cpg = cpg;
  J.X = X; J.all_bias = all_bias; J.train = train; J.n_train = n_train;
  J.val = val; J.n_val = n_val; J.I = I; J.isum = isum; J.runsum = runsum;
  J.unit_mae = xmal((size_t)ih->n_units * 4);
  J.unit_step = xcal(ih->n_units, 4);
  pthread_mutex_init(&J.lock, NULL);
  pthread_t *tid = xmal(nth * sizeof(pthread_t));
  for (unsigned t = 0; t < nth; ++t)
    if (pthread_create(&tid[t], NULL, worker, &J)) cdie("cannot start worker thread");
  for (unsigned t = 0; t < nth; ++t) pthread_join(tid[t], NULL);
  free(tid); pthread_mutex_destroy(&J.lock);
  free(all_bias); free(X);

  /* assemble the bare UPDEC2 from the checkpoints */
  uint64_t metadata_bytes = sizeof(ms_updec2_header_t) + (uint64_t)F * 8 +
    (uint64_t)ih->n_units * sizeof(ms_updec2_unit_t) + ih->n_cpg * 4 +
    (uint64_t)ih->n_real_memberships * sizeof(ms_updec2_membership_t);
  ms_updec2_unit_t *od = xmal((size_t)ih->n_units * sizeof(*od));
  uint64_t po = metadata_bytes;
  for (uint32_t ui = 0; ui < ih->n_units; ++ui) {
    const MsuiUnit *u = &units[ui];
    int direct = c->mixed_direct && !(u->flags & 1);
    uint32_t R = direct ? 0 : ((u->flags & 1) ? c->pure_bottleneck : c->mixed_bottleneck);
    uint64_t nf = direct ? (uint64_t)u->cpg_count * I + u->cpg_count
                         : (uint64_t)R * I + R + (uint64_t)u->cpg_count * R + u->cpg_count;
    od[ui].output_offset = u->output_offset; od[ui].param_offset = po;
    od[ui].param_bytes = nf * 4; od[ui].cpg_count = u->cpg_count;
    od[ui].membership_count = u->membership_count;
    od[ui].mode = (uint16_t)(direct ? 0 : 1);
    od[ui].bottleneck_dim = (uint16_t)R; od[ui].flags = u->flags;
    po += nf * 4;
  }
  ms_updec2_header_t oh; memset(&oh, 0, sizeof(oh));
  memcpy(oh.magic, MS_UPDEC2_MAGIC, 8);
  oh.version = 3;
  oh.flags = MS_UPDEC2_FLAG_GENOMIC |
    (c->feature_mode == MS_UPFEATURE_COUNT ? MS_UPDEC2_FLAG_COUNT : 0) |
    (c->feature_mode == MS_UPFEATURE_BETA ? MS_UPDEC2_FLAG_BETA_ONLY : 0);
  oh.patterns = P; oh.input_dim = F; oh.n_units = ih->n_units;
  oh.n_memberships = ih->n_real_memberships; oh.target_unit_cpgs = ih->target_unit_cpgs;
  oh.activation = c->activation; oh.n_cpg = ih->n_cpg;
  oh.mean_offset = sizeof(oh);
  oh.scale_offset = oh.mean_offset + (uint64_t)F * 4;
  oh.unit_offset = oh.scale_offset + (uint64_t)F * 4;
  oh.cpg_offset = oh.unit_offset + (uint64_t)ih->n_units * sizeof(ms_updec2_unit_t);
  oh.membership_offset = oh.cpg_offset + ih->n_cpg * 4;
  oh.param_offset = metadata_bytes; oh.file_bytes = po; oh.index_checksum = isum;

  FILE *out = fopen(c->model_path, "wb");
  if (!out) cdiep("cannot create UPDEC2", c->model_path);
  if (fwrite(&oh, 1, sizeof(oh), out) != sizeof(oh) ||
      fwrite(mean, 4, (size_t)F, out) != (size_t)F ||
      fwrite(scale, 4, (size_t)F, out) != (size_t)F ||
      fwrite(od, sizeof(*od), ih->n_units, out) != ih->n_units ||
      fwrite(cpg, 4, ih->n_cpg, out) != ih->n_cpg ||
      fwrite(members, sizeof(*members), ih->n_real_memberships, out) != ih->n_real_memberships)
    cdiep("write failed", c->model_path);
  uint64_t psum = UINT64_C(1469598103934665603);
  for (uint32_t ui = 0; ui < ih->n_units; ++ui) {
    char path[4096]; ckpath(path, sizeof(path), c->work_dir, ui);
    FILE *f = fopen(path, "rb");
    if (!f) cdiep("missing unit checkpoint", path);
    CheckHeader ch;
    if (fread(&ch, 1, sizeof(ch), f) != sizeof(ch)) cdiep("truncated unit checkpoint", path);
    float *p = xmal((size_t)ch.param_floats * 4);
    if (fread(p, 4, ch.param_floats, f) != ch.param_floats) cdiep("truncated unit checkpoint", path);
    fclose(f);
    if ((uint64_t)ch.param_floats * 4 != od[ui].param_bytes) cdiep("checkpoint size mismatch", path);
    if (fwrite(p, 4, ch.param_floats, out) != ch.param_floats) cdiep("write failed", c->model_path);
    psum = fnv(psum, p, (size_t)ch.param_floats * 4);
    free(p);
  }
  oh.parameter_checksum = psum;
  if (fseek(out, 0, SEEK_SET) || fwrite(&oh, 1, sizeof(oh), out) != sizeof(oh))
    cdiep("cannot rewrite header", c->model_path);
  if (fclose(out)) cdiep("cannot close UPDEC2", c->model_path);
  fprintf(stderr, "[methscope] upscale-train: wrote bare UPDEC2 %s (%llu bytes), resumed=%u\n",
          c->model_path, (unsigned long long)oh.file_bytes, J.resumed);

  free(od); free(mean); free(scale); free(train); free(val);
  free(J.unit_mae); free(J.unit_step);
  unmap(&idx); unmap(&data);
  return 0;
}
