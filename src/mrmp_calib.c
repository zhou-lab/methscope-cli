// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
/* Per-pair LOO calibration of (shrink-pseudocnt, min-sbeta-gap) for 2-class
 * selections, from single-cell evidence.
 *
 * Why this exists: a fixed (a, G) cannot serve every pair -- the honest
 * operating point moves with pool depth and each pair's biology, and picking
 * it from training-set performance alone fails on both ends (train is blind
 * to the winner's-curse plateau; a superset pre-screen leaks -- the
 * Ambroise-McLachlan trap, measured at +37 points of optimism on
 * Ent Co/Gob Co before the harness was fixed). The validated recipe
 * (fold-0, 13 pairs, mean 2.8 points off the per-pair oracle, worst 10.7):
 *
 *   leave-one-CELL-out over the training cells, selection pools = all cells
 *   minus self (the subtract-self trick: every cell is a fold at no extra
 *   cost, and the inner pool depth n-1 matches the final build), evaluated
 *   over the FULL row space (no pre-screen -- that is what makes it
 *   leak-free), then the LEAST RESTRICTIVE grid cell within eps of the best
 *   validation macro: looser admission keeps more CpGs for sparse queries,
 *   so the boundary is approached from the loose side, never crossed.
 *
 * Efficiency: each cell's record is inflated ONCE (threaded pass 1) into a
 * sparse (row, M, U) cache; pass 2 (threaded over cells, no shared writes)
 * streams each cache once, bucketing every read by (side, G-bin) of its
 * leave-self-out gap at each of the 11 pseudocounts -- the whole 11x11
 * (a, G) grid then falls out of suffix sums over the G-bins, with no
 * per-grid-cell rescan of the data.
 *
 * The classifier proxy is the anchored sign feature the featurizer emits:
 * pooled beta of the A-hyper side vs the B-hyper side over the cell's own
 * reads, a side under SIDE_FLOOR observed reads contributing the neutral
 * 0.5 (both under: no call) -- mirroring msfm's --side-floor semantics. */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mrmp.h"
#include "cfile.h"
#include "cdata.h"
#include "index.h"

#define CAL_NA 11
#define CAL_NG 11
#define CAL_SIDE_FLOOR 3           /* matches msfm's default --side-floor */
#define CAL_MIN_ADMIT 100          /* a pick must admit at least this many
                                    * CpGs from the full pools: below it the
                                    * feature is too thin to pool at query
                                    * coverage, whatever LOO says */
static const double CAL_AS[CAL_NA] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14};
#define CAL_G0    0.28             /* gap grid: 0.28 .. 0.53 by 0.025; the
                                    * harness swept this span and every pick
                                    * across 13 pairs landed inside it */
#define CAL_GSTEP 0.025

static void cdie(const char *msg, const char *det) {
  fprintf(stderr, "[methscope] calibrate: %s%s%s\n", msg,
          det ? ": " : "", det ? det : "");
  exit(1);
}

typedef struct { uint32_t row; uint16_t m, u; } cal_read_t;

/* ---- extracted-cell cache -------------------------------------------------
 * A cell participates in one calibration per pair its class joins (~60 per
 * fold), and inflating + scanning its record dominates pass 1. This LRU
 * keeps the extracted sparse (row, M, U) arrays across ms_pair_calibrate
 * calls, keyed by the record's BGZF virtual offset (unique within a store).
 * Entries in use by the current pair are pinned; eviction only touches
 * unpinned entries. Capacity: METHSCOPE_CALCACHE_MB (default 8192; 0
 * disables). The bank build sorts resolver pairs by class, so the A-side
 * class stays hot across its run of pairs. */
typedef struct centry {
  int64_t voff;
  cal_read_t *reads;
  uint32_t n;
  uint32_t pinned;
  struct centry *newer, *older;  /* LRU list, head = newest */
  struct centry *hnext;          /* hash chain */
} centry_t;

#define CCH_BITS 16
static uint64_t cch_ncpg;                /* row space, fixed by first record */
static centry_t *cch_hash[1 << CCH_BITS];
static centry_t *cch_new, *cch_old;      /* LRU ends */
static size_t cch_bytes, cch_cap = (size_t)-1;
static const char *cch_store;            /* cache is valid for ONE store */
static pthread_mutex_t cch_lock = PTHREAD_MUTEX_INITIALIZER;

static size_t cch_capacity(void) {
  if (cch_cap != (size_t)-1) return cch_cap;
  const char *e = getenv("METHSCOPE_CALCACHE_MB");
  long mb = e && *e ? atol(e) : 8192;
  cch_cap = mb > 0 ? (size_t)mb << 20 : 0;
  return cch_cap;
}

static uint32_t cch_slot(int64_t voff) {
  uint64_t h = (uint64_t)voff * 0x9e3779b97f4a7c15ull;
  return (uint32_t)(h >> (64 - CCH_BITS));
}

static void cch_unlink(centry_t *e) {
  if (e->newer) e->newer->older = e->older; else cch_new = e->older;
  if (e->older) e->older->newer = e->newer; else cch_old = e->newer;
  e->newer = e->older = NULL;
}

static void cch_push(centry_t *e) {       /* to the newest end */
  e->newer = NULL; e->older = cch_new;
  if (cch_new) cch_new->newer = e;
  cch_new = e;
  if (!cch_old) cch_old = e;
}

static void cch_evict_locked(void) {
  centry_t *e = cch_old;
  while (cch_bytes > cch_capacity() && e) {
    centry_t *nx = e->newer;
    if (!e->pinned) {
      cch_unlink(e);
      centry_t **pp = &cch_hash[cch_slot(e->voff)];
      while (*pp && *pp != e) pp = &(*pp)->hnext;
      if (*pp) *pp = e->hnext;
      cch_bytes -= (size_t)e->n * sizeof(cal_read_t) + sizeof(centry_t);
      free(e->reads); free(e);
    }
    e = nx;
  }
}

static void cch_reset_locked(void) {      /* store changed: drop everything */
  for (centry_t *e = cch_old; e; ) {
    centry_t *nx = e->newer;
    free(e->reads); free(e); e = nx;
  }
  memset(cch_hash, 0, sizeof(cch_hash));
  cch_new = cch_old = NULL; cch_bytes = 0;
}

/* returns a pinned entry, or NULL on miss */
static centry_t *cch_get(const char *store, int64_t voff) {
  if (!cch_capacity()) return NULL;
  pthread_mutex_lock(&cch_lock);
  if (cch_store && strcmp(cch_store, store)) { cch_reset_locked(); cch_store = NULL; }
  if (!cch_store) { cch_store = strdup(store); }
  centry_t *e = cch_hash[cch_slot(voff)];
  while (e && e->voff != voff) e = e->hnext;
  if (e) { ++e->pinned; cch_unlink(e); cch_push(e); }
  pthread_mutex_unlock(&cch_lock);
  return e;
}

/* takes ownership of reads[]; returns the pinned entry, or NULL if the
 * cache is off (caller keeps ownership then) */
static centry_t *cch_put(const char *store, int64_t voff,
                         cal_read_t *reads, uint32_t n) {
  if (!cch_capacity()) return NULL;
  pthread_mutex_lock(&cch_lock);
  if (cch_store && strcmp(cch_store, store)) { cch_reset_locked(); cch_store = NULL; }
  if (!cch_store) { cch_store = strdup(store); }
  centry_t *e = calloc(1, sizeof(centry_t));
  if (!e) { pthread_mutex_unlock(&cch_lock); return NULL; }
  e->voff = voff; e->reads = reads; e->n = n; e->pinned = 1;
  e->hnext = cch_hash[cch_slot(voff)];
  cch_hash[cch_slot(voff)] = e;
  cch_push(e);
  cch_bytes += (size_t)n * sizeof(cal_read_t) + sizeof(centry_t);
  cch_evict_locked();
  pthread_mutex_unlock(&cch_lock);
  return e;
}

static void cch_unpin(centry_t *e) {
  pthread_mutex_lock(&cch_lock);
  if (e->pinned) --e->pinned;
  cch_evict_locked();
  pthread_mutex_unlock(&cch_lock);
}

typedef struct {
  const char *store;
  const int64_t *voff;           /* per cell, BGZF virtual offset */
  const uint8_t *cls;            /* per cell, 0 = class A, 1 = class B */
  uint32_t n_cells;
  uint64_t n_cpg;                /* 0 until the first record fixes it */
  cal_read_t **reads;            /* per cell, sparse cache (owned or cached) */
  uint32_t *n_reads;
  centry_t **cent;               /* per cell, pinned cache entry or NULL */
  uint32_t a_lo, a_hi;           /* pass-2 pseudocount range (inclusive) */
  /* pass 2 inputs (pass 1 leaves them NULL) */
  const uint32_t *pm, *pu;       /* full pools, [2][n_cpg] */
  double *bm, *bu;               /* buckets, [cell][a][side][G-bin] */
  uint32_t cursor;
  pthread_mutex_t lock;
  int failed;
} cal_job_t;

/* pass 1: inflate one cell per grab into its sparse cache (or take it
 * pinned from the cross-pair cache) */
static void *cal_load(void *arg) {
  cal_job_t *J = (cal_job_t *)arg;
  cfile_t cf = open_cfile((char *)J->store);
  for (;;) {
    pthread_mutex_lock(&J->lock);
    uint32_t k = J->cursor < J->n_cells ? J->cursor++ : UINT32_MAX;
    pthread_mutex_unlock(&J->lock);
    if (k == UINT32_MAX) break;
    centry_t *ce = cch_get(J->store, J->voff[k]);
    if (ce) {
      J->reads[k] = ce->reads; J->n_reads[k] = ce->n; J->cent[k] = ce;
      /* a cached cell cannot fix n_cpg; a fresh cell in this batch will */
      continue;
    }
    if (bgzf_seek(cf.fh, J->voff[k], SEEK_SET) != 0) { J->failed = 1; break; }
    cdata_t c = read_cdata1(&cf);
    if (!c.n) { J->failed = 1; break; }
    decompress_in_situ(&c);
    pthread_mutex_lock(&J->lock);
    if (!J->n_cpg) J->n_cpg = c.n;   /* first record fixes the row space */
    if (!cch_ncpg) cch_ncpg = c.n;   /* remembered for all-cached batches */
    uint64_t n_cpg = J->n_cpg;
    pthread_mutex_unlock(&J->lock);
    if (c.fmt != '3' || c.n != n_cpg) { free_cdata(&c); J->failed = 1; break; }
    uint32_t cap = 1 << 16, n = 0;
    cal_read_t *rr = malloc((size_t)cap * sizeof(cal_read_t));
    if (!rr) { free_cdata(&c); J->failed = 1; break; }
    for (uint64_t i = 0; i < n_cpg; ++i) {
      uint64_t mu = f3_get_mu(&c, i);
      if (!mu) continue;
      if (n == cap) {
        cap <<= 1;
        rr = realloc(rr, (size_t)cap * sizeof(cal_read_t));
        if (!rr) { J->failed = 1; break; }
      }
      uint64_t m = mu >> 32, u = mu & 0xffffffffu;
      rr[n].row = (uint32_t)i;
      rr[n].m = (uint16_t)(m > 0xFFFF ? 0xFFFF : m);
      rr[n].u = (uint16_t)(u > 0xFFFF ? 0xFFFF : u);
      ++n;
    }
    free_cdata(&c);
    J->reads[k] = rr; J->n_reads[k] = n;
    J->cent[k] = cch_put(J->store, J->voff[k], rr, n);  /* NULL: we own rr */
  }
  bgzf_close(cf.fh);
  return NULL;
}

/* release a pair's cells: unpin cached entries, free owned ones */
static void cal_release(cal_job_t *J) {
  for (uint32_t k = 0; k < J->n_cells; ++k) {
    if (!J->reads[k]) continue;
    if (J->cent[k]) cch_unpin(J->cent[k]);
    else free(J->reads[k]);
    J->reads[k] = NULL;
  }
}

/* pass 2: bucket one cell's reads by (a, side, G-bin) of the
 * leave-self-out gap, then free the cache. No shared writes: each cell owns
 * its bucket slab, and the pools are read-only here. */
static void *cal_bucket(void *arg) {
  cal_job_t *J = (cal_job_t *)arg;
  const uint64_t n_cpg = J->n_cpg;
  for (;;) {
    pthread_mutex_lock(&J->lock);
    uint32_t k = J->cursor < J->n_cells ? J->cursor++ : UINT32_MAX;
    pthread_mutex_unlock(&J->lock);
    if (k == UINT32_MAX) break;
    const uint32_t self = J->cls[k];
    double *km0 = J->bm + ((size_t)k * CAL_NA * 2) * CAL_NG;
    double *ku0 = J->bu + ((size_t)k * CAL_NA * 2) * CAL_NG;
    for (uint32_t r = 0; r < J->n_reads[k]; ++r) {
      const cal_read_t *rd = &J->reads[k][r];
      const uint64_t i = rd->row;
      double am = J->pm[i], au = J->pu[i];
      double bm = J->pm[n_cpg + i], bu = J->pu[n_cpg + i];
      if (self == 0) { am -= rd->m; au -= rd->u; }
      else { bm -= rd->m; bu -= rd->u; }
      const double da = am + au, db = bm + bu;
      if (da == 0 || db == 0) continue;
      for (uint32_t ai = J->a_lo; ai <= J->a_hi; ++ai) {
        const double a = CAL_AS[ai];
        double g = (am + a) / (da + 2 * a) - (bm + a) / (db + 2 * a);
        double ag = g < 0 ? -g : g;
        if (ag < CAL_G0) continue;
        int b = (int)((ag - CAL_G0) / CAL_GSTEP);
        if (b >= CAL_NG) b = CAL_NG - 1;
        const int side = g > 0 ? 0 : 1;      /* 0 = A-hyper */
        km0[(ai * 2 + side) * CAL_NG + b] += rd->m;
        ku0[(ai * 2 + side) * CAL_NG + b] += rd->u;
      }
    }
    /* caches are NOT freed here: the a=1-only pass may be followed by the
     * full-grid pass on the same cells, and cached cells are shared across
     * pairs -- cal_release() at the end settles ownership. */
  }
  return NULL;
}

static void cal_run(cal_job_t *J, uint32_t threads, void *(*fn)(void *)) {
  J->cursor = 0;
  if (threads < 1) threads = 1;
  if (threads > J->n_cells) threads = J->n_cells;
  pthread_t *tid = calloc(threads, sizeof(pthread_t));
  if (!tid) cdie("out of memory (threads)", NULL);
  for (uint32_t t = 0; t < threads; ++t)
    if (pthread_create(&tid[t], NULL, fn, J))
      cdie("cannot start calibration worker", NULL);
  for (uint32_t t = 0; t < threads; ++t) pthread_join(tid[t], NULL);
  free(tid);
}

/* Calibrate one class pair. cellsA/cellsB are cell names resolved against
 * the store's .idx. Returns 1 and fills out_* on success; 0 (with a stderr
 * note) when calibration is impossible -- missing index, too few cells, no
 * grid cell clearing the admission floor -- leaving the caller's defaults
 * in force. */
/* the LOO macro of every G threshold at ONE pseudocount, from the buckets */
static void cal_macro_row(const cal_job_t *J, const uint8_t *cls, uint32_t nc,
                          uint32_t ai, double *row) {
  for (int gj = 0; gj < CAL_NG; ++gj) {
    uint32_t okA = 0, okB = 0, tA = 0, tB = 0;
    for (uint32_t k = 0; k < nc; ++k) {
      const double *km = J->bm + (((size_t)k * CAL_NA + ai) * 2) * CAL_NG;
      const double *ku = J->bu + (((size_t)k * CAL_NA + ai) * 2) * CAL_NG;
      double m0 = 0, u0 = 0, m1 = 0, u1 = 0;
      for (int j = gj; j < CAL_NG; ++j) {
        m0 += km[j];          u0 += ku[j];           /* A-hyper side */
        m1 += km[CAL_NG + j]; u1 += ku[CAL_NG + j];  /* B-hyper side */
      }
      if (cls[k] == 0) ++tA; else ++tB;
      const double o0 = m0 + u0, o1 = m1 + u1;
      if (o0 < CAL_SIDE_FLOOR && o1 < CAL_SIDE_FLOOR) continue;
      const double b0 = o0 >= CAL_SIDE_FLOOR ? m0 / o0 : 0.5;
      const double b1 = o1 >= CAL_SIDE_FLOOR ? m1 / o1 : 0.5;
      if (b0 == b1) continue;
      /* an A cell reads methylated on the A-hyper side */
      const int pred = b0 > b1 ? 0 : 1;
      if (pred == 0 && cls[k] == 0) ++okA;
      if (pred == 1 && cls[k] == 1) ++okB;
    }
    row[gj] = 0.5 * ((double)okA / (tA ? tA : 1)
                   + (double)okB / (tB ? tB : 1));
  }
}

int ms_pair_calibrate(const char *cellstore, char *const *cellsA, uint32_t nA,
                      char *const *cellsB, uint32_t nB,
                      float eps, uint32_t threads,
                      double *out_a, double *out_G, double *out_valid,
                      uint32_t *out_ncell, uint64_t *out_nadm, int *out_grid) {
  char *fidx = get_fname_index((char *)cellstore);
  index_t *idx = fidx ? loadIndex(fidx) : NULL;
  free(fidx);
  if (!idx) {
    fprintf(stderr, "[methscope] calibrate: no .idx beside the cell store; "
            "skipping\n");
    return 0;
  }
  uint32_t nc = 0, miss = 0;
  int64_t *voff = calloc((size_t)nA + nB, sizeof(int64_t));
  uint8_t *cls = calloc((size_t)nA + nB, 1);
  if (!voff || !cls) cdie("out of memory (cell table)", NULL);
  for (uint32_t k = 0; k < nA + nB; ++k) {
    char *nm = k < nA ? cellsA[k] : cellsB[k - nA];
    int64_t off = getIndex(idx, nm);
    if (off < 0) { ++miss; continue; }
    voff[nc] = off; cls[nc] = k >= nA; ++nc;
  }
  cleanIndex(idx);
  if (miss)
    fprintf(stderr, "[methscope] calibrate: %u cell(s) not in the store "
            "index (skipped)\n", miss);
  if (nc < 20) { free(voff); free(cls); return 0; }

  cal_job_t J = {0};
  J.store = cellstore; J.voff = voff; J.cls = cls; J.n_cells = nc;
  J.reads = calloc(nc, sizeof(cal_read_t *));
  J.n_reads = calloc(nc, sizeof(uint32_t));
  J.cent = calloc(nc, sizeof(centry_t *));
  if (!J.reads || !J.n_reads || !J.cent)
    cdie("out of memory (cell caches)", NULL);
  pthread_mutex_init(&J.lock, NULL);
  cal_run(&J, threads, cal_load);
  if (J.failed) cdie("cell store read failed (need fmt3 records over one "
                     "row space)", cellstore);
  if (!J.n_cpg) J.n_cpg = cch_ncpg;      /* every cell came from the cache */
  const uint64_t n_cpg = J.n_cpg;
  if (!n_cpg) cdie("no readable cell records", cellstore);

  /* full pools: raw M/U sums per class, the pools the final build sees */
  uint32_t *pm = calloc((size_t)2 * n_cpg, sizeof(uint32_t));
  uint32_t *pu = calloc((size_t)2 * n_cpg, sizeof(uint32_t));
  if (!pm || !pu) cdie("out of memory (pools)", NULL);
  for (uint32_t k = 0; k < nc; ++k) {
    const size_t off = (size_t)cls[k] * n_cpg;
    for (uint32_t r = 0; r < J.n_reads[k]; ++r) {
      pm[off + J.reads[k][r].row] += J.reads[k][r].m;
      pu[off + J.reads[k][r].row] += J.reads[k][r].u;
    }
  }

  /* admitted-CpG counts per (a, G) on the FULL pools: what the final build
   * would select at that grid cell, and the restrictiveness ranking */
  uint64_t nadm[CAL_NA][CAL_NG];
  memset(nadm, 0, sizeof(nadm));
  for (uint64_t i = 0; i < n_cpg; ++i) {
    const double da = (double)pm[i] + pu[i];
    const double db = (double)pm[n_cpg + i] + pu[n_cpg + i];
    if (da == 0 || db == 0) continue;
    for (uint32_t ai = 0; ai < CAL_NA; ++ai) {
      const double a = CAL_AS[ai];
      double g = (pm[i] + a) / (da + 2 * a)
               - (pm[n_cpg + i] + a) / (db + 2 * a);
      double ag = g < 0 ? -g : g;
      if (ag < CAL_G0) continue;
      int b = (int)((ag - CAL_G0) / CAL_GSTEP);
      if (b >= CAL_NG) b = CAL_NG - 1;
      for (int j = 0; j <= b; ++j) ++nadm[ai][j];
    }
  }

  /* TIERED pass 2. Tier A: bucket the DEFAULT pseudocount only (a=1) and
   * test the loosest grid cell. The picker takes the least-restrictive
   * cell within eps of the best macro; when (1, 0.28) scores >= 1 - eps
   * AND its admission count is the grid maximum, no other cell can win --
   * the full grid provably returns the default, so skip it. Measured on
   * 4,120 completed calibrations, 94-97%% of pairs exit here. Tier B (the
   * true confusers): bucket the remaining pseudocounts over the SAME held
   * cell caches (no re-inflation) and evaluate the full grid. */
  J.pm = pm; J.pu = pu;
  J.bm = calloc((size_t)nc * CAL_NA * 2 * CAL_NG, sizeof(double));
  J.bu = calloc((size_t)nc * CAL_NA * 2 * CAL_NG, sizeof(double));
  if (!J.bm || !J.bu) cdie("out of memory (buckets)", NULL);
  uint64_t nadm_max = 0;
  for (uint32_t ai = 0; ai < CAL_NA; ++ai)
    for (int gj = 0; gj < CAL_NG; ++gj)
      if (nadm[ai][gj] > nadm_max) nadm_max = nadm[ai][gj];

  double macro[CAL_NA][CAL_NG];
  int bi = -1, bj = -1; uint64_t bn = 0; int grid = 0;

  J.a_lo = 0; J.a_hi = 0;
  cal_run(&J, threads, cal_bucket);
  cal_macro_row(&J, cls, nc, 0, macro[0]);
  if (macro[0][0] >= 1.0 - eps && nadm[0][0] >= CAL_MIN_ADMIT &&
      nadm[0][0] >= nadm_max) {
    bi = 0; bj = 0; bn = nadm[0][0];     /* exact early exit */
  } else {
    grid = 1;
    J.a_lo = 1; J.a_hi = CAL_NA - 1;
    cal_run(&J, threads, cal_bucket);
    for (uint32_t ai = 1; ai < CAL_NA; ++ai)
      cal_macro_row(&J, cls, nc, ai, macro[ai]);
    double best = -1;
    for (uint32_t ai = 0; ai < CAL_NA; ++ai)
      for (int gj = 0; gj < CAL_NG; ++gj)
        if (macro[ai][gj] > best) best = macro[ai][gj];
    /* least restrictive within eps of the best, above the admission floor */
    for (uint32_t ai = 0; ai < CAL_NA; ++ai)
      for (int gj = 0; gj < CAL_NG; ++gj)
        if (macro[ai][gj] >= best - eps && nadm[ai][gj] >= CAL_MIN_ADMIT &&
            (bi < 0 || nadm[ai][gj] > bn)) {
          bi = ai; bj = gj; bn = nadm[ai][gj];
        }
  }
  cal_release(&J);
  pthread_mutex_destroy(&J.lock);
  free(J.reads); free(J.n_reads); free(J.cent); free(pm); free(pu);
  free(J.bm); free(J.bu); free(voff); free(cls);

  if (bi < 0) {
    fprintf(stderr, "[methscope] calibrate: no grid cell admits >= %u CpGs; "
            "skipping\n", CAL_MIN_ADMIT);
    return 0;
  }
  *out_a = CAL_AS[bi];
  *out_G = CAL_G0 + bj * CAL_GSTEP;
  *out_valid = macro[bi][bj];
  if (out_ncell) *out_ncell = nc;
  if (out_nadm) *out_nadm = bn;
  if (out_grid) *out_grid = grid;
  return 1;
}
