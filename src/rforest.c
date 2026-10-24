// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
/**
 * Random forest for classification -- see rforest.h for what is computed.
 *
 * Sized for array cohorts (n in the hundreds, p in the tens of thousands per
 * call): a node's rows are re-sorted per drawn feature, which at n ~ 100
 * costs less than keeping p presorted index lists per tree.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "rforest.h"

static void pdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] %s\n", msg);
  exit(1);
}

/* splitmix64, one stream per tree, the same generator msfm_build.c uses for
 * its per-cell draws. */
typedef struct { uint64_t s; } rf_rng_t;

static inline uint64_t rng_next(rf_rng_t *r) {
  uint64_t z = (r->s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
/* The splitmix64 finalizer, used to seed each tree. A splitmix stream only
 * adds a constant to its state, so seeding tree t with seed ^ (gamma * t)
 * made tree t+1's stream tree t's shifted by about one draw: the bootstraps
 * overlapped, and 39 of 478 biopsy rows were never out of bag in 500 trees.
 * Hashing puts every tree's state at an unrelated point of the cycle. */
static inline uint64_t rng_mix(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static inline uint32_t rng_below(rf_rng_t *r, uint32_t n) {
  return (uint32_t)(rng_next(r) % n);
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) pdie("out of memory (random forest)", NULL);
  return p;
}
static void *xcalloc(size_t n, size_t s) {
  void *p = calloc(n ? n : 1, s ? s : 1);
  if (!p) pdie("out of memory (random forest)", NULL);
  return p;
}

/* ------------------------------------------------------------- one tree --- */

typedef struct {
  rf_tree_t t;
  uint32_t cap;
} tbuf_t;

static int32_t tree_add(tbuf_t *b) {
  if (b->t.n_node == b->cap) {
    b->cap = b->cap ? 2 * b->cap : 64;
    b->t.feat  = realloc(b->t.feat,  b->cap * sizeof(int32_t));
    b->t.thr   = realloc(b->t.thr,   b->cap * sizeof(float));
    b->t.left  = realloc(b->t.left,  b->cap * sizeof(int32_t));
    b->t.right = realloc(b->t.right, b->cap * sizeof(int32_t));
    b->t.cls   = realloc(b->t.cls,   b->cap * sizeof(int32_t));
    if (!b->t.feat || !b->t.thr || !b->t.left || !b->t.right || !b->t.cls)
      pdie("out of memory (random forest)", NULL);
  }
  int32_t k = (int32_t)b->t.n_node++;
  b->t.feat[k] = -1; b->t.thr[k] = 0.f;
  b->t.left[k] = b->t.right[k] = -1; b->t.cls[k] = 0;
  return k;
}

/* Walk one row; jf >= 0 substitutes value v for feature jf (the permuted
 * column of the importance pass). */
static inline int32_t tree_class(const rf_tree_t *t, const float *x,
                                 int32_t jf, float v) {
  int32_t k = 0;
  while (t->feat[k] >= 0) {
    float xv = (t->feat[k] == jf) ? v : x[t->feat[k]];
    k = (xv <= t->thr[k]) ? t->left[k] : t->right[k];
  }
  return t->cls[k];
}

typedef struct { float x; uint32_t r; } xr_t;
static int cmp_xr(const void *a, const void *b) {
  float u = ((const xr_t *)a)->x, v = ((const xr_t *)b)->x;
  return (u > v) - (u < v);
}

/* Shared, read-only inputs plus the per-worker scratch. */
typedef struct {
  const float *X;
  const int   *y;
  uint32_t n, p, nclass, mtry;
  uint64_t seed;
} job_t;

typedef struct {
  uint32_t *rows;      /* in-bag rows (distinct), partitioned per node */
  uint32_t *w;         /* bootstrap multiplicity per row */
  uint32_t *fperm;     /* feature draw order, p */
  xr_t     *xr;        /* n */
  double   *cl, *cr, *cp;   /* nclass each */
  uint32_t *stk;       /* node stack: (node, lo, hi) triples */
} scratch_t;

/* Per-tree result, reduced in tree order after the workers finish. */
typedef struct {
  tbuf_t    b;
  uint32_t  n_used;    /* features this tree splits on */
  int32_t  *used;
  double   *dmda;      /* per used feature: (right - right_perm) / n_oob */
  double   *dgini;     /* per used feature: summed Gini decrease */
  int32_t  *oob_cls;   /* n: this tree's call, -1 when in bag */
} tres_t;

static void grow_tree(const job_t *J, scratch_t *S, uint32_t ti, tres_t *R) {
  const uint32_t n = J->n, p = J->p, C = J->nclass;
  rf_rng_t rng = { rng_mix(rng_mix(J->seed) ^ (ti + 1ULL)) };

  /* Bootstrap: n draws with replacement, kept as distinct rows + weights. */
  memset(S->w, 0, n * sizeof(uint32_t));
  for (uint32_t i = 0; i < n; ++i) S->w[rng_below(&rng, n)]++;
  uint32_t nin = 0;
  for (uint32_t i = 0; i < n; ++i) if (S->w[i]) S->rows[nin++] = i;
  for (uint32_t j = 0; j < p; ++j) S->fperm[j] = j;

  /* Gini decrease per feature, dense while growing, compacted below. */
  double *gdec = xcalloc(p, sizeof(double));
  uint8_t *isused = xcalloc(p, 1);

  tbuf_t *b = &R->b;
  memset(b, 0, sizeof(*b));
  uint32_t sp = 0;
  S->stk[sp++] = (uint32_t)tree_add(b); S->stk[sp++] = 0; S->stk[sp++] = nin;
  while (sp) {
    uint32_t hi = S->stk[--sp], lo = S->stk[--sp], nd = S->stk[--sp];
    memset(S->cp, 0, C * sizeof(double));
    double W = 0;
    for (uint32_t k = lo; k < hi; ++k) {
      S->cp[J->y[S->rows[k]]] += S->w[S->rows[k]];
      W += S->w[S->rows[k]];
    }
    /* Majority class, ties broken at random as R does. */
    int32_t best_c = 0; uint32_t nties = 1;
    for (uint32_t c = 1; c < C; ++c) {
      if (S->cp[c] > S->cp[best_c]) { best_c = (int32_t)c; nties = 1; }
      else if (S->cp[c] == S->cp[best_c] && rng_below(&rng, ++nties) == 0)
        best_c = (int32_t)c;
    }
    b->t.cls[nd] = best_c;
    if (S->cp[best_c] == W || W <= 1) continue;          /* pure: a leaf */

    double crit0 = 0;
    for (uint32_t c = 0; c < C; ++c) crit0 += S->cp[c] * S->cp[c];
    crit0 /= W;

    /* mtry features without replacement; the best Gini split among them.
     * Like R, any split between distinct values is taken even with no
     * decrease; a node is a leaf only when every drawn feature is constant
     * across it. */
    double best = -1e300; int32_t bf = -1; float bthr = 0.f;
    uint32_t sties = 1;            /* tie count, R's bookkeeping */
    const uint32_t m = J->mtry < p ? J->mtry : p;
    for (uint32_t d = 0; d < m; ++d) {
      uint32_t q = d + rng_below(&rng, p - d);
      uint32_t f = S->fperm[q]; S->fperm[q] = S->fperm[d]; S->fperm[d] = f;
      uint32_t nn = hi - lo;
      for (uint32_t k = 0; k < nn; ++k) {
        uint32_t r = S->rows[lo + k];
        S->xr[k].x = J->X[(size_t)r * p + f]; S->xr[k].r = r;
      }
      qsort(S->xr, nn, sizeof(xr_t), cmp_xr);
      if (S->xr[0].x == S->xr[nn - 1].x) continue;
      memset(S->cl, 0, C * sizeof(double));
      memcpy(S->cr, S->cp, C * sizeof(double));
      sties = 1;
      double wl = 0, wr = W, sl = 0, sr = crit0 * W;    /* sums of squares */
      for (uint32_t k = 0; k + 1 < nn; ++k) {
        uint32_t r = S->xr[k].r, c = (uint32_t)J->y[r];
        double wt = S->w[r];
        sl += wt * (2 * S->cl[c] + wt); S->cl[c] += wt; wl += wt;
        sr += wt * (wt - 2 * S->cr[c]); S->cr[c] -= wt; wr -= wt;
        if (S->xr[k].x == S->xr[k + 1].x) continue;
        double crit = sl / wl + sr / wr;
        /* Ties are broken R's way, which is not uniform: the count restarts
         * at each feature, and a new best is counted twice, so of two tied
         * cuts in one feature the first stays with odds 2:1. Keeping the
         * first always, or a uniform draw, both left iris (x10, no rounding)
         * at OOB 0.030 against R's 0.0286; this rule gives 0.0290. */
        int take = 0;
        if (crit > best) { take = 1; sties = 1; best = crit; }
        if (crit == best && rng_below(&rng, ++sties) == 0) take = 1;
        if (take) {
          best = crit; bf = (int32_t)f;
          double mid = 0.5 * ((double)S->xr[k].x + (double)S->xr[k + 1].x);
          bthr = (float)mid;
          /* rounding must not send the upper value left */
          if (!(bthr < S->xr[k + 1].x)) bthr = S->xr[k].x;
        }
      }
    }
    if (bf < 0) continue;                       /* all drawn constant: leaf */

    gdec[bf] += best - crit0;
    isused[bf] = 1;
    /* Partition rows[lo, hi) on the split. */
    uint32_t a = lo, z = hi;
    while (a < z) {
      if (J->X[(size_t)S->rows[a] * p + bf] <= bthr) ++a;
      else { uint32_t t = S->rows[a]; S->rows[a] = S->rows[--z]; S->rows[z] = t; }
    }
    int32_t L = tree_add(b), Rn = tree_add(b);
    b->t.feat[nd] = bf; b->t.thr[nd] = bthr;
    b->t.left[nd] = L; b->t.right[nd] = Rn;
    S->stk[sp++] = (uint32_t)Rn; S->stk[sp++] = a;  S->stk[sp++] = hi;
    S->stk[sp++] = (uint32_t)L;  S->stk[sp++] = lo; S->stk[sp++] = a;
  }

  /* OOB calls, then the permutation importance of every feature used. */
  uint32_t *oob = xmalloc(n * sizeof(uint32_t)), noob = 0, right = 0;
  R->oob_cls = xmalloc(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    R->oob_cls[i] = -1;
    if (S->w[i]) continue;
    oob[noob++] = i;
    R->oob_cls[i] = tree_class(&b->t, J->X + (size_t)i * p, -1, 0.f);
    right += (R->oob_cls[i] == J->y[i]);
  }
  R->n_used = 0;
  for (uint32_t j = 0; j < p; ++j) R->n_used += isused[j];
  R->used  = xmalloc(R->n_used * sizeof(int32_t));
  R->dmda  = xcalloc(R->n_used, sizeof(double));
  R->dgini = xmalloc(R->n_used * sizeof(double));
  uint32_t *perm = xmalloc(n * sizeof(uint32_t));
  for (uint32_t j = 0, u = 0; j < p; ++j) {
    if (!isused[j]) continue;
    R->used[u] = (int32_t)j; R->dgini[u] = gdec[j];
    if (noob) {
      for (uint32_t k = 0; k < noob; ++k) perm[k] = oob[k];
      for (uint32_t k = noob; k > 1; --k) {             /* Fisher-Yates */
        uint32_t q = rng_below(&rng, k);
        uint32_t t = perm[k - 1]; perm[k - 1] = perm[q]; perm[q] = t;
      }
      uint32_t rp = 0;
      for (uint32_t k = 0; k < noob; ++k) {
        uint32_t i = oob[k];
        float v = J->X[(size_t)perm[k] * p + j];
        rp += (tree_class(&b->t, J->X + (size_t)i * p, (int32_t)j, v) == J->y[i]);
      }
      R->dmda[u] = ((double)right - (double)rp) / noob;
    }
    ++u;
  }
  free(perm); free(oob); free(gdec); free(isused);
}

typedef struct {
  const job_t *J;
  tres_t *res;
  uint32_t ntree, t0, step;
} worker_t;

static void *worker(void *arg) {
  worker_t *w = arg;
  const job_t *J = w->J;
  scratch_t S;
  S.rows  = xmalloc(J->n * sizeof(uint32_t));
  S.w     = xmalloc(J->n * sizeof(uint32_t));
  S.fperm = xmalloc(J->p * sizeof(uint32_t));
  S.xr    = xmalloc(J->n * sizeof(xr_t));
  S.cl    = xmalloc(J->nclass * sizeof(double));
  S.cr    = xmalloc(J->nclass * sizeof(double));
  S.cp    = xmalloc(J->nclass * sizeof(double));
  /* Depth-first, each split pushes two triples: 3 * (2n + 1) bounds it. */
  S.stk   = xmalloc(3 * (2 * (size_t)J->n + 2) * sizeof(uint32_t));
  for (uint32_t t = w->t0; t < w->ntree; t += w->step)
    grow_tree(J, &S, t, &w->res[t]);
  free(S.rows); free(S.w); free(S.fperm); free(S.xr);
  free(S.cl); free(S.cr); free(S.cp); free(S.stk);
  return NULL;
}

rf_forest_t *rf_fit(const float *X, uint32_t n, uint32_t p, const int *y,
                    uint32_t nclass, const rf_param_t *par,
                    double *mda, double *gini) {
  if (!n || !p || nclass < 2) pdie("random forest: empty input", NULL);
  for (uint32_t i = 0; i < n; ++i)
    if (y[i] < 0 || (uint32_t)y[i] >= nclass)
      pdie("random forest: label out of range", NULL);
  for (size_t k = 0; k < (size_t)n * p; ++k)
    if (isnan(X[k])) pdie("random forest: NaN in the feature matrix "
                          "(impute first)", NULL);

  job_t J = { X, y, n, p, nclass,
              par->mtry ? par->mtry : (uint32_t)floor(sqrt((double)p)),
              par->seed };
  if (!J.mtry) J.mtry = 1;
  const uint32_t ntree = par->ntree ? par->ntree : 500;
  tres_t *res = xcalloc(ntree, sizeof(tres_t));

  int T = par->threads > 0 ? par->threads : 1;
  if ((uint32_t)T > ntree) T = (int)ntree;
  pthread_t *th = xmalloc(T * sizeof(pthread_t));
  worker_t *wk = xmalloc(T * sizeof(worker_t));
  for (int k = 0; k < T; ++k) {
    wk[k] = (worker_t){ &J, res, ntree, (uint32_t)k, (uint32_t)T };
    if (pthread_create(&th[k], NULL, worker, &wk[k]))
      pdie("random forest: cannot start a thread", NULL);
  }
  for (int k = 0; k < T; ++k) pthread_join(th[k], NULL);
  free(th); free(wk);

  /* Reduce in tree order, so the sums do not depend on scheduling. */
  if (mda)  memset(mda,  0, p * sizeof(double));
  if (gini) memset(gini, 0, p * sizeof(double));
  uint32_t *votes = xcalloc((size_t)n * nclass, sizeof(uint32_t));
  rf_forest_t *f = xcalloc(1, sizeof(rf_forest_t));
  f->ntree = ntree; f->p = p; f->nclass = nclass;
  f->tree = xmalloc(ntree * sizeof(rf_tree_t));
  for (uint32_t t = 0; t < ntree; ++t) {
    tres_t *r = &res[t];
    for (uint32_t u = 0; u < r->n_used; ++u) {
      if (mda)  mda[r->used[u]]  += r->dmda[u];
      if (gini) gini[r->used[u]] += r->dgini[u];
    }
    for (uint32_t i = 0; i < n; ++i)
      if (r->oob_cls[i] >= 0) votes[(size_t)i * nclass + r->oob_cls[i]]++;
    f->tree[t] = r->b.t;
    free(r->used); free(r->dmda); free(r->dgini); free(r->oob_cls);
  }
  free(res);
  for (uint32_t j = 0; j < p; ++j) {
    if (mda)  mda[j]  /= ntree;
    if (gini) gini[j] /= ntree;
  }
  /* A tied vote is broken at random, as R does, not given to the first
   * class. */
  rf_rng_t rng = { rng_mix(par->seed) };
  uint32_t nv = 0, wrong = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t *v = votes + (size_t)i * nclass;
    uint32_t c = 0, tot = v[0], nties = 1;
    for (uint32_t k = 1; k < nclass; ++k) {
      tot += v[k];
      if (v[k] > v[c]) { c = k; nties = 1; }
      else if (v[k] == v[c] && rng_below(&rng, ++nties) == 0) c = k;
    }
    if (!tot) continue;
    ++nv; wrong += ((int)c != y[i]);
  }
  f->oob_err = nv ? (double)wrong / nv : NAN;
  f->oob_vote = xcalloc((size_t)n * nclass, sizeof(double));
  for (uint32_t i = 0; i < n; ++i) {
    double tot = 0;
    for (uint32_t k = 0; k < nclass; ++k) tot += votes[(size_t)i * nclass + k];
    for (uint32_t k = 0; tot > 0 && k < nclass; ++k)
      f->oob_vote[(size_t)i * nclass + k] = votes[(size_t)i * nclass + k] / tot;
  }
  free(votes);
  return f;
}

void rf_predict(const rf_forest_t *f, const float *X, uint32_t n,
                double *prob) {
  memset(prob, 0, (size_t)n * f->nclass * sizeof(double));
  for (uint32_t i = 0; i < n; ++i) {
    const float *x = X + (size_t)i * f->p;
    for (uint32_t t = 0; t < f->ntree; ++t)
      prob[(size_t)i * f->nclass + tree_class(&f->tree[t], x, -1, 0.f)] += 1;
    for (uint32_t k = 0; k < f->nclass; ++k)
      prob[(size_t)i * f->nclass + k] /= f->ntree;
  }
}

void rf_free(rf_forest_t *f) {
  if (!f) return;
  for (uint32_t t = 0; t < f->ntree; ++t) {
    rf_tree_t *r = &f->tree[t];
    free(r->feat); free(r->thr); free(r->left); free(r->right); free(r->cls);
  }
  free(f->tree);
  free(f->oob_vote);
  free(f);
}

/* -------------------------------------------------------- serialization --- */

#define RF_MAGIC "MSRF0001"

static void put32(uint8_t **w, uint32_t v) { memcpy(*w, &v, 4); *w += 4; }

uint8_t *rf_serialize(const rf_forest_t *f, size_t *len) {
  size_t n = 8 + 12;
  for (uint32_t t = 0; t < f->ntree; ++t) n += 4 + 20 * (size_t)f->tree[t].n_node;
  uint8_t *buf = xmalloc(n), *w = buf;
  memcpy(w, RF_MAGIC, 8); w += 8;
  put32(&w, f->ntree); put32(&w, f->p); put32(&w, f->nclass);
  for (uint32_t t = 0; t < f->ntree; ++t) {
    const rf_tree_t *r = &f->tree[t];
    size_t k = r->n_node;
    put32(&w, r->n_node);
    memcpy(w, r->feat,  4 * k); w += 4 * k;
    memcpy(w, r->thr,   4 * k); w += 4 * k;
    memcpy(w, r->left,  4 * k); w += 4 * k;
    memcpy(w, r->right, 4 * k); w += 4 * k;
    memcpy(w, r->cls,   4 * k); w += 4 * k;
  }
  *len = n;
  return buf;
}

rf_forest_t *rf_deserialize(const uint8_t *buf, size_t len, size_t *used) {
  if (len < 20 || memcmp(buf, RF_MAGIC, 8)) return NULL;
  const uint8_t *r = buf + 8, *end = buf + len;
  uint32_t hdr[3];
  memcpy(hdr, r, 12); r += 12;
  if (!hdr[0] || !hdr[1] || hdr[2] < 2) return NULL;
  rf_forest_t *f = xcalloc(1, sizeof(rf_forest_t));
  f->ntree = hdr[0]; f->p = hdr[1]; f->nclass = hdr[2];
  f->oob_err = NAN;
  f->tree = xcalloc(f->ntree, sizeof(rf_tree_t));
  for (uint32_t t = 0; t < f->ntree; ++t) {
    uint32_t k;
    if (end - r < 4) goto bad;
    memcpy(&k, r, 4); r += 4;
    if (!k || (size_t)(end - r) < 20 * (size_t)k) goto bad;
    rf_tree_t *tr = &f->tree[t];
    tr->n_node = k;
    tr->feat  = xmalloc(4 * k); memcpy(tr->feat,  r, 4 * k); r += 4 * k;
    tr->thr   = xmalloc(4 * k); memcpy(tr->thr,   r, 4 * k); r += 4 * k;
    tr->left  = xmalloc(4 * k); memcpy(tr->left,  r, 4 * k); r += 4 * k;
    tr->right = xmalloc(4 * k); memcpy(tr->right, r, 4 * k); r += 4 * k;
    tr->cls   = xmalloc(4 * k); memcpy(tr->cls,   r, 4 * k); r += 4 * k;
    /* a node must point inside its tree, forward, and a leaf's class in
     * range, or a crafted model walks off the arrays at scoring time */
    for (uint32_t j = 0; j < k; ++j) {
      if (tr->feat[j] >= 0) {
        if ((uint32_t)tr->feat[j] >= f->p ||
            tr->left[j] <= (int32_t)j || (uint32_t)tr->left[j] >= k ||
            tr->right[j] <= (int32_t)j || (uint32_t)tr->right[j] >= k)
          goto bad;
      } else if (tr->cls[j] < 0 || (uint32_t)tr->cls[j] >= f->nclass) goto bad;
    }
  }
  if (used) *used = (size_t)(r - buf);
  return f;
bad:
  rf_free(f);
  return NULL;
}
