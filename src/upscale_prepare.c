// SPDX-License-Identifier: AGPL-3.0-or-later
/* Build the compact raw-CG msur used by the global upscale trainer.
 *
 * The input .cg remains the immutable truth store.  This command writes only
 * deterministic YAME-compatible sampled CpG positions and MRMP summaries, so
 * it deliberately does not create downsampled .cg files or replicated TSVs. */
/* random_r/initstate_r are GNU extensions; needed before any header. */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>

#include "methscope.h"
#include "cfile.h"
#include "summary.h"
#include "mrmp.h"

#include "msur.h"

#define MSUR_MAX_LEVELS 16

static void pdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] upscale-featurize: %s: %s\n", msg, arg);
  else fprintf(stderr, "[methscope] upscale-featurize: %s\n", msg);
  exit(1);
}

static void *xmalloc(size_t n, const char *what) {
  void *p = malloc(n ? n : 1);
  if (!p) pdie("out of memory", what);
  return p;
}

static uint64_t parse_u64(const char *s, const char *what) {
  errno = 0; char *end = NULL; unsigned long long v = strtoull(s, &end, 10);
  if (errno || end == s || *end) pdie("invalid integer", what);
  return (uint64_t)v;
}

/* Explicit portable RNG state used by the sampling loop. A private state is
 * required because the replicate loop may be parallelised. The generator is
 * deliberately independent of libc, so Linux and macOS produce the same
 * sequence for a given seed. */
typedef struct { uint32_t state; } ms_rng_t;

static void ms_rng_seed(ms_rng_t *r, unsigned seed) {
  r->state = seed ? seed : 1u;
}

static double yame_rand01(ms_rng_t *r) {
  r->state = r->state * 1664525u + 1013904223u;
  return (double)(r->state >> 1) / 2147483648.0;
}

/* Copy-on-write view of the eligible array.
 *
 * The shuffle mutates `a` and restore_partial_shuffle() puts it back: fine
 * serially, a data race the moment two replicates run at once, since every
 * replicate shuffles the SAME per-cell array. Copying is no better -- a
 * well-covered cell has ~26 M eligible CpGs, so a per-thread copy per
 * (replicate, cell) would cost more than the work it parallelises.
 *
 * At most `k` swaps happen, touching at most 2k of n positions, so an overlay
 * of the changed entries is tiny (~116 k slots against 26 M) and leaves `a`
 * read-only. Open addressing, power-of-two capacity, no deletes. */
typedef struct { uint64_t *pos; uint32_t *val; uint64_t cap, mask; } cow_t;

static void cow_init(cow_t *c, uint64_t k) {
  uint64_t cap = 16; while (cap < 4 * (k + 1)) cap <<= 1;
  c->cap = cap; c->mask = cap - 1;
  c->pos = xmalloc(cap * sizeof(*c->pos), "shuffle overlay keys");
  c->val = xmalloc(cap * sizeof(*c->val), "shuffle overlay values");
  for (uint64_t i = 0; i < cap; ++i) c->pos[i] = UINT64_MAX;
}
static void cow_free(cow_t *c) { free(c->pos); free(c->val); }
static void cow_reset(cow_t *c) {
  for (uint64_t i = 0; i < c->cap; ++i) c->pos[i] = UINT64_MAX;
}
static inline uint64_t cow_slot(const cow_t *c, uint64_t p) {
  uint64_t h = (p * 1099511628211ULL) & c->mask;
  while (c->pos[h] != UINT64_MAX && c->pos[h] != p) h = (h + 1) & c->mask;
  return h;
}
static inline uint32_t cow_get(const cow_t *c, const uint32_t *a, uint64_t p) {
  uint64_t h = cow_slot(c, p);
  return c->pos[h] == p ? c->val[h] : a[p];
}
static inline void cow_set(cow_t *c, uint64_t p, uint32_t v) {
  uint64_t h = cow_slot(c, p);
  c->pos[h] = p; c->val[h] = v;
}

/* The swap sequence a plain in-place Fisher-Yates would perform -- same
 * randoms, same j, same resulting a[0..k) -- written to the overlay instead of
 * the array, so `a` stays read-only and no restore pass is needed. */
static void partial_fisher_yates_cow(const uint32_t *a, uint64_t n, uint64_t k,
                                     ms_rng_t *rng, cow_t *c, uint32_t *out) {
  if (k > n) k = n;
  cow_reset(c);
  for (uint64_t i = 0; i < k; ++i) {
    uint64_t j = i + (uint64_t)(yame_rand01(rng) * (double)(n - i));
    uint32_t ti = cow_get(c, a, i), tj = cow_get(c, a, j);
    cow_set(c, i, tj); cow_set(c, j, ti);
    out[i] = tj;
  }
}


static int pnum(const char *s) {
  if (!s || (s[0] != 'P' && s[0] != 'p')) return 0;
  char *end = NULL; long n = strtol(s + 1, &end, 10);
  return end != s + 1 && *end == '\0' && n > 0 && n <= INT_MAX ? (int)n : 0;
}

static int cmp_u32(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return (x > y) - (x < y);
}

static void write_or_die(FILE *fp, const void *p, size_t n, const char *path) {
  if (n && fwrite(p, 1, n, fp) != n) pdie("write failed", path);
}

/* ---------------- worker pool over samples -------------------------------- */

/* Profiling the serial version put 52% of cycles in zlib's inflate_fast, inside
 * read_cdata1 -> bgzf_read. Threading everything AROUND that returned 1.4x at
 * best, because the inflate stayed on one handle.
 *
 * The store's .idx carries a BGZF virtual offset per record, so a worker can
 * seek straight to a sample. A POOL rather than a wave: each worker opens its
 * handle ONCE and pulls cell indices off a shared counter until they run out.
 * That is n_threads opens instead of n_cells, and no barrier -- a worker that
 * finishes a sparse cell early takes the next one instead of idling. */
typedef struct {
  const char *truth; uint64_t n_cpg;
  const int64_t *voff; uint32_t n_cells;
  cdata_t *cells; uint32_t **eligible; uint64_t *n_eligible;
  /* the truth row is converted here too, from the record this worker just
   * inflated, so the store is never inflated a second time for it */
  int embed; uint16_t *row; FILE *fp; const char *out;
  pthread_mutex_t *mu; pthread_cond_t *cv; uint32_t *next, *turn;
} pool_ctx_t;

static void *pool_worker(void *arg) {
  pool_ctx_t *w = arg;
  cfile_t cf = open_cfile((char *)w->truth);       /* once per worker */
  for (;;) {
    pthread_mutex_lock(w->mu);
    uint32_t cell = (*w->next)++;
    pthread_mutex_unlock(w->mu);
    if (cell >= w->n_cells) break;

    if (bgzf_seek(cf.fh, w->voff[cell], SEEK_SET) != 0)
      pdie("cannot seek truth store", w->truth);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) pdie("truth store record is empty", w->truth);
    decompress_in_situ(&c);
    if (c.fmt != '3' || c.n != w->n_cpg) pdie("inconsistent truth record", w->truth);

    /* TWO passes, deliberately: sizing exactly costs a second walk, but growing
     * a ~100 MB array instead measured 1.7x slower (161.6 s against 95.1 s) --
     * a realloc that cannot extend in place copies the whole thing. */
    uint64_t ne = 0;
    for (uint64_t i = 0; i < w->n_cpg; ++i) ne += f3_get_mu(&c, i) != 0;
    uint32_t *e = xmalloc((size_t)ne * sizeof(uint32_t), "eligible CpGs");
    uint64_t q = 0;
    for (uint64_t i = 0; i < w->n_cpg; ++i)
      if (f3_get_mu(&c, i)) e[q++] = (uint32_t)i;

    w->cells[cell] = c; w->eligible[cell] = e; w->n_eligible[cell] = ne;

    if (!w->embed) continue;
    for (uint64_t i = 0; i < w->n_cpg; ++i) {
      uint64_t mu = f3_get_mu(&c, i);
      /* u16 code = beta*65534; 65535 (UINT16_MAX) = missing */
      w->row[i] = mu ? (uint16_t)llround(MU2beta(mu) * 65534.0) : UINT16_MAX;
    }
    /* the truth section is laid out by cell, so writes are ordered; cells are
     * handed out in order, so a worker rarely waits long for its turn */
    pthread_mutex_lock(w->mu);
    while (*w->turn != cell) pthread_cond_wait(w->cv, w->mu);
    write_or_die(w->fp, w->row, (size_t)w->n_cpg * sizeof(*w->row), w->out);
    ++*w->turn;
    pthread_cond_broadcast(w->cv);
    pthread_mutex_unlock(w->mu);
  }
  bgzf_close(cf.fh);
  return NULL;
}


/* ---------------- one replicate ------------------------------------------- */

/* srand(r+1) was always per replicate, so replicates never depended on each
 * other; with the RNG state explicit (ms_rng_t) and the eligible arrays made
 * read-only by the copy-on-write shuffle, they simply run concurrently. Records
 * go to a per-worker buffer and are flushed in replicate order, so the file is
 * byte-identical at any thread count. */
typedef struct {
  uint32_t n_cells, ncol, n_sets;
  uint64_t n_cpg, bitmap_bytes;
  const uint16_t *group;
  const cdata_t *cells;
  uint32_t *const *eligible;
  const uint64_t *n_eligible;
  const uint32_t *rep_sample;
  const msur_rep_t *reptab;
  int binarize, v3;
  double *sum; uint32_t *count; float *beta; uint32_t *selected;
  cow_t cow; unsigned char *bitmap;
  unsigned char *buf; size_t len, cap;
  pthread_mutex_t *mu; uint32_t *next, *total_reps;
  FILE *fp; const char *out;
  pthread_cond_t *cv; uint32_t *turn;
} rep_ctx_t;

static void rbuf_put(rep_ctx_t *w, const void *p, size_t n) {
  if (w->len + n > w->cap) {
    while (w->cap < w->len + n) w->cap = w->cap ? w->cap * 2 : (1u << 20);
    w->buf = realloc(w->buf, w->cap);
    if (!w->buf) pdie("out of memory (replicate buffer)", NULL);
  }
  memcpy(w->buf + w->len, p, n); w->len += n;
}

static void *rep_worker(void *arg) {
  rep_ctx_t *w = arg;
  for (;;) {
    pthread_mutex_lock(w->mu);
    uint32_t r = (*w->next)++;
    pthread_mutex_unlock(w->mu);
    if (r >= *w->total_reps) break;

    const uint32_t sample = w->rep_sample[r];
    const int rep_bitmap = w->v3 && w->reptab[r].flags == MSUR_ENC_BITMAP;
    ms_rng_t rng; ms_rng_seed(&rng, r + 1);
    w->len = 0;
    for (uint32_t cell = 0; cell < w->n_cells; ++cell) {
      cdata_t c = w->cells[cell];
      partial_fisher_yates_cow(w->eligible[cell], w->n_eligible[cell], sample,
                               &rng, &w->cow, w->selected);
      memset(w->sum, 0, (size_t)w->ncol * sizeof(*w->sum));
      memset(w->count, 0, (size_t)w->ncol * sizeof(*w->count));
      for (uint32_t k = 0; k < sample; ++k) {
        uint64_t pos = w->selected[k];
        /* the binarise draw is per CpG, not per set: one read, one call */
        double b = MU2beta(f3_get_mu(&c, pos));
        int drawn = 0;
        for (uint32_t sx = 0; sx < w->n_sets; ++sx) {
          uint16_t g = w->group[(uint64_t)sx * w->n_cpg + pos];
          if (!g) continue;
          if (w->binarize && !drawn) { b = yame_rand01(&rng) < b ? 1.0 : 0.0; drawn = 1; }
          w->sum[g - 1] += b; ++w->count[g - 1];
        }
      }
      for (uint32_t g = 0; g < w->ncol; ++g)
        w->beta[g] = w->count[g] ? (float)(w->sum[g] / w->count[g]) : NAN;
      qsort(w->selected, sample, sizeof(*w->selected), cmp_u32);
      rbuf_put(w, w->beta, (size_t)w->ncol * sizeof(*w->beta));
      rbuf_put(w, w->count, (size_t)w->ncol * sizeof(*w->count));
      if (rep_bitmap) {
        memset(w->bitmap, 0, (size_t)w->bitmap_bytes);
        for (uint32_t k = 0; k < sample; ++k)
          w->bitmap[w->selected[k] >> 3] |= (unsigned char)(1u << (w->selected[k] & 7));
        rbuf_put(w, w->bitmap, (size_t)w->bitmap_bytes);
      } else {
        rbuf_put(w, w->selected, (size_t)sample * sizeof(*w->selected));
      }
    }
    pthread_mutex_lock(w->mu);
    while (*w->turn != r) pthread_cond_wait(w->cv, w->mu);
    write_or_die(w->fp, w->buf, w->len, w->out);
    ++*w->turn;
    pthread_cond_broadcast(w->cv);
    pthread_mutex_unlock(w->mu);
  }
  return NULL;
}

static int usage(void) {
  ms_help(stderr,
    "Usage: methscope upscale-featurize [options] TRUTH.cg IN.mrmp OUT.msur\n\n"
    "Create a compact exact-YAME sampling msur for global upscale training.\n"
    "The original TRUTH.cg remains the truth store and is never copied.\n\n"
    "  TRUTH.cg         continuous format-3 YAME .cg truth store\n"
    "  IN.mrmp          MRMPIDX1 artifact (preferred) or an exported .cm\n"
    "  OUT.msur         output msur\n\n"
    "Options:\n"
    "  --reps N         deterministic simulations per --sample level (default 100)\n"
    "  --sample N[,N..] CpGs retained per cell/replicate (default 29000). Give a\n"
    "                   comma list to mix coverages: --reps applies to EACH level,\n"
    "                   so --reps 100 --sample 14701,29000,147009 writes 300\n"
    "                   replicates. Each level stores exactly its own record size.\n"
    "  --sample-logrange MIN,MAX\n"
    "                   continuous ladder instead of --sample: --reps is the TOTAL\n"
    "                   replicate count, each getting its own sample size spread\n"
    "                   in log space over [MIN,MAX]\n"
    "  --sample-skew A  bend that draw (default 0.5). A<1 puts more replicates at\n"
    "                   the DENSE end, where finer differences must be resolved;\n"
    "                   A=1 is plain log-uniform\n"
    "  --binarize       one read per observed CpG: replace each sampled beta with a\n"
    "                   Bernoulli(beta) draw, as `yame dsample -b` does\n"
    "  --in-memory      inflate the truth store once and reuse it for all replicates\n"
    "  --threads N      inflate and scan N samples at once (default 1). Needs\n"
    "                   --in-memory and a <store>.idx, since each worker opens\n"
    "                   its own handle and seeks to its sample. Output is\n"
    "                   byte-identical at any N.\n"
    "  --manifest PATH  write provenance TSV (default OUT.msur.tsv)\n"
    "  -h, --help       show this help\n"
    "\n"
    "EVERY pattern in IN.mrmp becomes a column -- there is no width knob here.\n"
    "Pattern SELECTION belongs to `mrmp-pool --pooled-top N`, which ranks every\n"
    "pattern of every set on CpG count and prunes to a shared budget. A width\n"
    "flag here was worse than redundant: it RESERVED slots per set, so a chain\n"
    "handed its root the first N columns whether or not the root's Nth pattern\n"
    "outweighed a child's 1st. Pool first, then featurize what survives.\n"
    "\n"
    "One msur serves many models. What is stored per (cell, replicate) is the raw\n"
    "summary -- beta and covered-CpG count for every pattern, plus the observed\n"
    "set -- NOT the encoder input. Which projection the encoder sees is chosen at\n"
    "TRAINING time, so one msur covers every `upscale-train --features` mode\n"
    "(missing / count / beta / scalar). Only the *simulation* is fixed here: the\n"
    "cells, the replicate count, the coverage ladder and --binarize.\n");
  return 1;
}


int main_upscale_prepare(int argc, char *argv[]) {
  const char *out = NULL, *truth = NULL, *mrmp = NULL, *manifest = NULL;
  const char *pos[3] = {NULL, NULL, NULL};
  int npos = 0;
  uint32_t reps = 100;
  uint32_t nthreads = 1;
  uint32_t levels[MSUR_MAX_LEVELS] = {29000}, n_levels = 1, max_sample = 29000;
  uint32_t log_min = 0, log_max = 0;   /* --sample-logrange */
  double log_skew = 0.5;               /* <1 leans dense; 1 = plain log-uniform */
  int in_memory = 0, binarize = 0;
  int embed_truth = 1;   /* always embed truth -- upscale-train requires it */
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage();
      return 0;
    }
    if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (uint32_t)parse_u64(argv[++i], "--reps");
    else if (!strcmp(argv[i], "--sample") && i + 1 < argc) {
      const char *s = argv[++i];
      n_levels = 0; max_sample = 0;
      for (;;) {
        const char *comma = strchr(s, ',');
        char buf[32];
        size_t len = comma ? (size_t)(comma - s) : strlen(s);
        if (!len || len >= sizeof(buf)) pdie("invalid --sample level", argv[i]);
        memcpy(buf, s, len); buf[len] = '\0';
        if (n_levels == MSUR_MAX_LEVELS) pdie("too many --sample levels", argv[i]);
        levels[n_levels] = (uint32_t)parse_u64(buf, "--sample");
        if (!levels[n_levels]) pdie("--sample level must be positive", argv[i]);
        if (levels[n_levels] > max_sample) max_sample = levels[n_levels];
        ++n_levels;
        if (!comma) break;
        s = comma + 1;
      }
    }
    else if (!strcmp(argv[i], "--sample-logrange") && i + 1 < argc) {
      const char *v = argv[++i], *comma = strchr(v, ',');
      char buf[32];
      if (!comma || (size_t)(comma - v) >= sizeof(buf))
        pdie("--sample-logrange wants MIN,MAX", v);
      memcpy(buf, v, (size_t)(comma - v)); buf[comma - v] = '\0';
      log_min = (uint32_t)parse_u64(buf, "--sample-logrange MIN");
      log_max = (uint32_t)parse_u64(comma + 1, "--sample-logrange MAX");
      if (!log_min || log_max < log_min)
        pdie("--sample-logrange needs 0 < MIN <= MAX", v);
    }
    else if (!strcmp(argv[i], "--sample-skew") && i + 1 < argc) {
      char *end = NULL; log_skew = strtod(argv[++i], &end);
      if (!end || *end || !(log_skew > 0)) pdie("--sample-skew must be > 0", argv[i]);
    }
    else if (!strcmp(argv[i], "--binarize")) binarize = 1;
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
      nthreads = (uint32_t)parse_u64(argv[++i], "--threads");
    else if (!strcmp(argv[i], "--in-memory")) in_memory = 1;
    else if (argv[i][0] == '-') { usage(); pdie("unrecognized or incomplete option", argv[i]); }
    else if (npos < 3) pos[npos++] = argv[i];
    else pdie("too many arguments", argv[i]);
  }
  if (npos != 3) {
    usage();
    pdie("need TRUTH.cg, IN.mrmp, and OUT.msur", NULL);
  }
  if (!reps || (!log_min && !n_levels)) return usage();
  /* Both pools read the inflated records, so without --in-memory memory_cells
   * is NULL and the replicate pool dereferences it. Refuse up front: this used
   * to SEGFAULT, because the guard lived in the replicate loop and was lost
   * when the pool moved inside the in_memory block. */
  if (nthreads > 1 && !in_memory)
    pdie("--threads needs --in-memory (the streaming path is sequential)", NULL);
  if ((uint64_t)reps * n_levels > UINT32_MAX) pdie("too many replicates", NULL);

  /* Per-replicate sample sizes.  Either the small fixed ladder (--sample, reps
   * per level) or a continuous one (--sample-logrange, reps total).
   *
   * The continuous draw is stratified in LOG space -- coverage effects are
   * log-scaled, which is why every ladder we have used halves -- so the range
   * is covered evenly and the msur stays reproducible (no RNG).  --sample-skew
   * bends it: u^skew with skew < 1 pushes replicates toward the DENSE end,
   * where the model must resolve finer differences and an even ladder therefore
   * under-samples.  skew = 1 is plain log-uniform. */
  uint32_t total_reps = log_min ? reps : reps * n_levels;
  uint32_t *rep_sample = xmalloc((size_t)total_reps * sizeof(*rep_sample),
                                 "per-replicate sample sizes");
  if (log_min) {
    double lo = log((double)log_min), span = log((double)log_max) - lo;
    max_sample = 0;
    for (uint32_t r = 0; r < total_reps; ++r) {
      double u = total_reps > 1 ? (double)r / (double)(total_reps - 1) : 1.0;
      double n = exp(lo + span * pow(u, log_skew));
      rep_sample[r] = (uint32_t)(n + 0.5);
      if (rep_sample[r] < 1) rep_sample[r] = 1;
      if (rep_sample[r] > max_sample) max_sample = rep_sample[r];
    }
  } else {
    for (uint32_t r = 0; r < total_reps; ++r) rep_sample[r] = levels[r / reps];
  }
  truth = pos[0]; mrmp = pos[1]; out = pos[2];
  if (sizeof(msur_header_t) != 72 || sizeof(msur_header3_t) != 80 ||
      sizeof(msur_rep_t) != 24) pdie("unexpected msur header layout", NULL);

  /* n_cells comes from the .idx, NOT from walking the store. The old loop read
   * and BGZF-inflated every record just to count them -- a second full inflate
   * of the store, ~20 s of a 40 s run, to produce one integer. The index has a
   * line per record and the pool needs it anyway for its seek offsets. Only the
   * FIRST record is still inflated, for n_cpg and the format check. */
  uint32_t n_cells = 0;
  { int64_t *vo = NULL; char **nm = ms_read_store_index(truth, &n_cells, &vo);
    for (uint32_t k = 0; k < n_cells; ++k) free(nm[k]);
    free(nm); free(vo); }
  if (!n_cells) pdie("truth store index is empty", truth);
  cfile_t check = open_cfile((char *)truth);
  cdata_t first = read_cdata1(&check);
  if (!first.n) pdie("truth store is empty", truth);
  decompress_in_situ(&first);
  if (first.fmt != '3') pdie("truth store must be continuous format 3", truth);
  uint64_t n_cpg = first.n;
  free_cdata(&first);
  bgzf_close(check.fh);
  if (n_cpg > UINT32_MAX) pdie("msur supports at most 2^32-1 CpGs", truth);

  /* Feature group of CpG i; PNA/background stays group 0 and is never a
   * feature.  Prefer the MRMPIDX1 artifact, so this map and the mask the model
   * ends up bundling derive from one source and cannot disagree; an
   * already-exported categorical .cm is still accepted. */
  /* A CHAIN featurizes as the concatenation of its sets: a CpG lands in one
   * pattern PER SET, so a tree's leaf contributes columns its root cannot
   * express. n_sets * n_cpg uint16 is 823 MB for a 14-set chain over hg38 --
   * paid once, then reused for every replicate. Constant binstrings never
   * appear: mrmp-build routes all-0 and all-1 to PNA unless asked otherwise,
   * so a node contributes only what separates something in its own subset. */
  uint32_t n_sets = 1, *col0 = NULL, ncol = 0;
  if (ms_mrmp_is_artifact(mrmp)) {
    ms_mrmpset_t *probe = ms_mrmpset_open(mrmp);
    n_sets = probe->n_sets;
    ms_mrmpset_free(probe);
  }
  uint16_t *group = xmalloc((size_t)(n_sets > 1 ? n_sets : 1) * n_cpg * sizeof(*group),
                            "MRMP group map");
  if (n_sets > 1) {
    col0 = xmalloc((size_t)(n_sets + 1) * sizeof(*col0), "chain column offsets");
    ncol = ms_mrmp_group_map_chain(mrmp, group, n_cpg, UINT32_MAX, col0, &n_sets);
    fprintf(stderr, "[methscope] upscale-featurize: chain of %u sets -> %u columns\n",
            n_sets, ncol);
  } else if (ms_mrmp_is_artifact(mrmp)) {
    ms_mrmp_group_map(mrmp, group, n_cpg, UINT32_MAX);
    for (uint64_t i = 0; i < n_cpg; ++i) if (group[i] > ncol) ncol = group[i];
  } else {
    cfile_t cmf = open_cfile((char *)mrmp);
    cdata_t cm = read_cdata1(&cmf);
    cdata_t extra = read_cdata1(&cmf);
    bgzf_close(cmf.fh);
    if (!cm.n || extra.n) pdie("MRMP must contain exactly one record", mrmp);
    free_cdata(&extra);
    decompress_in_situ(&cm);
    if (cm.fmt != '2') pdie("MRMP must be categorical format 2", mrmp);
    fmt2_set_aux(&cm);
    if (cm.n != n_cpg) pdie("MRMP and truth CpG counts differ", mrmp);
    for (uint64_t i = 0; i < n_cpg; ++i) {
      int p = pnum(f2_get_string(&cm, i));
      if (p > UINT16_MAX - 1) pdie("MRMP pattern number exceeds uint16 group format", mrmp);
      group[i] = p > 0 ? (uint16_t)p : 0;
      if (group[i] > ncol) ncol = group[i];
    }
    free_cdata(&cm);
  }

  /* A full continuous hg38 x 207-cell store inflates to about 46 GiB.  On the
   * 120-GiB training node this trades ~750 GiB of repeated filesystem reads
   * for one read and makes the 100-replicate cache build CPU-bound. */
  cdata_t *memory_cells = NULL;
  uint32_t **memory_eligible = NULL;
  uint64_t *memory_n_eligible = NULL;

  /* One level keeps the v2 fixed-width layout, so a single-level msur stays
   * byte-identical to one built before --sample took a list.  Mixed levels get
   * v3: per-replicate tables instead of padding every short record out to the
   * widest level (a 143-CpG replicate padded to 147,009 is 99.9% filler). */
  /* ncol, NEVER a separate width. These two disagreed while --patterns existed:
   * a chain's ncol is the concatenation width but feat_bytes was sized from
   * --patterns, so the replicate table understated every record by
   * (ncol - patterns) * 8 bytes and a reader landed mid-record. */
  const uint64_t feat_bytes =
    (uint64_t)ncol * (sizeof(float) + sizeof(uint32_t));
  int v3 = 0;
  for (uint32_t r = 1; r < total_reps && !v3; ++r) v3 = rep_sample[r] != rep_sample[0];

  FILE *fp = fopen(out, "wb");
  if (!fp) pdie("cannot create output", out);
  msur_header3_t h3;
  memset(&h3, 0, sizeof(h3));
  msur_header_t *hp = &h3.h;
  memcpy(hp->magic, v3 ? MSUR3_MAGIC : MSUR2_MAGIC, 8);
  hp->version = v3 ? 3u : 2u;
  /* n_patterns is the COLUMN count -- for a chain, the concatenation width.
   * upscale-train reads its feature width from this header field verbatim, so
   * recording the flag instead would let it read 200 columns of a 658-wide
   * record and silently mis-slice every row. */
  hp->n_cells = n_cells; hp->n_reps = total_reps; hp->n_patterns = ncol;
  hp->n_cpg = n_cpg; hp->sampled_per_cell = max_sample;
  hp->flags = (embed_truth ? MSUR_F_TRUTH_U16 : 0)
    | (binarize ? MSUR_F_BINARIZED : 0)
    | (v3 ? MSUR_F_MIXED_SAMPLE : 0);
  const uint64_t head_bytes = v3 ? sizeof(h3) : sizeof(*hp);
  hp->groups_offset = head_bytes;
  hp->truth_offset = embed_truth ? hp->groups_offset + n_cpg * sizeof(uint16_t) : 0;
  uint64_t after_truth = hp->groups_offset + n_cpg * sizeof(uint16_t)
    + (embed_truth ? (uint64_t)n_cells * n_cpg * sizeof(uint16_t) : 0);

  msur_rep_t *reptab = NULL;
  if (v3) {
    /* record_bytes varies by level, so the whole table is computable up front
     * and every record stays O(1) addressable. */
    h3.rep_table_offset = after_truth;
    hp->records_offset = after_truth + (uint64_t)total_reps * sizeof(msur_rep_t);
    hp->record_bytes = 0;          /* sentinel: variable, consult the table */
    reptab = xmalloc((size_t)total_reps * sizeof(*reptab), "replicate table");
    uint64_t at = hp->records_offset;
    uint64_t bmb = msur_bitmap_bytes(n_cpg);
    uint32_t n_bitmap = 0;
    /* Escape hatch: pin the encoding instead of choosing by size.  "list"
     * reproduces pre-bitmap output byte for byte (and is how the two encodings
     * are checked for equivalence); "bitmap" forces it everywhere. */
    const char *enc_env = getenv("MSUR_OBSERVED_ENCODING");
    int force_list = enc_env && !strcmp(enc_env, "list");
    int force_bitmap = enc_env && !strcmp(enc_env, "bitmap");
    if (enc_env && !force_list && !force_bitmap && strcmp(enc_env, "auto"))
      pdie("MSUR_OBSERVED_ENCODING must be list, bitmap, or auto", enc_env);
    for (uint32_t r = 0; r < total_reps; ++r) {
      uint32_t s = rep_sample[r];
      /* Whichever observed-set encoding is smaller for THIS replicate. */
      uint64_t list_bytes = (uint64_t)s * sizeof(uint32_t);
      int bitmap = force_bitmap || (!force_list && bmb < list_bytes);
      reptab[r].sample = s;
      reptab[r].flags = bitmap ? MSUR_ENC_BITMAP : MSUR_ENC_LIST;
      reptab[r].record_bytes = feat_bytes + (bitmap ? bmb : list_bytes);
      reptab[r].offset = at;
      at += (uint64_t)n_cells * reptab[r].record_bytes;
      n_bitmap += bitmap != 0;
    }
    if (n_bitmap)
      fprintf(stderr, "[methscope] upscale-featurize: bitmap observed-set for "
              "%u/%u replicates (>= %" PRIu64 " CpGs); records %.1f GiB\n",
              n_bitmap, total_reps, bmb / sizeof(uint32_t),
              (double)(at - hp->records_offset) / (1024.0 * 1024.0 * 1024.0));
  } else {
    hp->records_offset = after_truth;
    hp->record_bytes = feat_bytes + (uint64_t)max_sample * sizeof(uint32_t);
  }
  write_or_die(fp, &h3, (size_t)head_bytes, out);
  write_or_die(fp, group, (size_t)n_cpg * sizeof(*group), out);

  /* ONE pooled pass over the samples: read, inflate, scan for eligible CpGs,
   * and convert the truth row -- all on the record the worker just inflated, so
   * the store is inflated exactly once. Placed here, after the header, because
   * the truth section is written from inside the workers in cell order.
   *
   * Pool, not waves: each worker opens its handle once and pulls the next cell
   * off a shared counter, so a worker finishing a sparse cell takes another
   * instead of idling at a barrier. */
  if (in_memory) {
    memory_cells = calloc(n_cells, sizeof(*memory_cells));
    memory_eligible = calloc(n_cells, sizeof(*memory_eligible));
    memory_n_eligible = calloc(n_cells, sizeof(*memory_n_eligible));
    if (!memory_cells || !memory_eligible || !memory_n_eligible)
      pdie("out of memory allocating truth-cell table", truth);
    uint32_t nidx = 0; int64_t *voff = NULL;
    char **snames = ms_read_store_index(truth, &nidx, &voff);
    if (nidx != n_cells) pdie("store index disagrees with the record count", truth);
    uint32_t nw = nthreads ? nthreads : 1; if (nw > 64) nw = 64;
    if (nw > n_cells) nw = n_cells;
    fprintf(stderr, "[methscope] upscale-featurize: inflating %u truth cells "
            "(%u worker%s)\n", n_cells, nw, nw == 1 ? "" : "s");
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
    uint32_t next = 0, turn = 0;
    pool_ctx_t *pc = xmalloc((size_t)nw * sizeof(*pc), "pool workers");
    pthread_t *pth = xmalloc((size_t)nw * sizeof(*pth), "pool threads");
    for (uint32_t t = 0; t < nw; ++t) {
      pc[t] = (pool_ctx_t){ .truth = truth, .n_cpg = n_cpg, .voff = voff,
        .n_cells = n_cells, .cells = memory_cells, .eligible = memory_eligible,
        .n_eligible = memory_n_eligible, .mu = &mu, .cv = &cv,
        .next = &next, .turn = &turn, .fp = fp, .out = out,
        .embed = embed_truth,
        .row = embed_truth ? xmalloc((size_t)n_cpg * sizeof(uint16_t),
                                     "truth u16 row") : NULL };
    }
    if (nw == 1) pool_worker(&pc[0]);
    else {
      for (uint32_t t = 0; t < nw; ++t)
        if (pthread_create(&pth[t], NULL, pool_worker, &pc[t]))
          pdie("cannot create pool worker", NULL);
      for (uint32_t t = 0; t < nw; ++t) pthread_join(pth[t], NULL);
    }
    for (uint32_t t = 0; t < nw; ++t) free(pc[t].row);
    for (uint32_t k = 0; k < nidx; ++k) free(snames[k]);
    free(snames); free(voff); free(pc); free(pth);
  }

  uint32_t *eligible = memory_cells ? NULL : xmalloc((size_t)n_cpg * sizeof(*eligible), "sampling workspace");
  double *sum = xmalloc((size_t)ncol * sizeof(*sum), "MRMP sums");
  uint32_t *count = xmalloc((size_t)ncol * sizeof(*count), "MRMP counts");
  float *beta = xmalloc((size_t)ncol * sizeof(*beta), "feature beta");
  uint32_t *selected = xmalloc((size_t)max_sample * sizeof(*selected), "sampled positions");
  cow_t cow; cow_init(&cow, max_sample);
  /* Only allocated when some replicate is dense enough to prefer a bitmap. */
  uint64_t bitmap_bytes = msur_bitmap_bytes(n_cpg);
  unsigned char *bitmap = NULL;
  if (v3) for (uint32_t r = 0; r < total_reps && !bitmap; ++r)
    if (reptab[r].flags == MSUR_ENC_BITMAP)
      bitmap = xmalloc((size_t)bitmap_bytes, "observed-set bitmap");

  if (v3) write_or_die(fp, reptab, (size_t)total_reps * sizeof(*reptab), out);

  /* Pooled over replicates, same shape as the sample pool: a worker takes the
   * next replicate off a shared counter, builds every cell's record into its own
   * buffer, then waits its turn to flush so the file stays replicate-major. */
  {
    uint32_t rw = nthreads ? nthreads : 1;
    if (rw > total_reps) rw = total_reps;
    if (rw > 64) rw = 64;
    pthread_mutex_t rmu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t  rcv = PTHREAD_COND_INITIALIZER;
    uint32_t rnext = 0, rturn = 0, treps = total_reps;
    rep_ctx_t *rc = xmalloc((size_t)rw * sizeof(*rc), "replicate workers");
    pthread_t *rth = xmalloc((size_t)rw * sizeof(*rth), "replicate threads");
    for (uint32_t t = 0; t < rw; ++t) {
      memset(&rc[t], 0, sizeof(rc[t]));
      rc[t].n_cells = n_cells; rc[t].ncol = ncol; rc[t].n_sets = n_sets;
      rc[t].n_cpg = n_cpg; rc[t].bitmap_bytes = bitmap_bytes;
      rc[t].group = group; rc[t].cells = memory_cells;
      rc[t].eligible = memory_eligible; rc[t].n_eligible = memory_n_eligible;
      rc[t].rep_sample = rep_sample; rc[t].reptab = reptab;
      rc[t].binarize = binarize; rc[t].v3 = v3;
      rc[t].sum = xmalloc((size_t)ncol * sizeof(double), "worker sums");
      rc[t].count = xmalloc((size_t)ncol * sizeof(uint32_t), "worker counts");
      rc[t].beta = xmalloc((size_t)ncol * sizeof(float), "worker beta");
      rc[t].selected = xmalloc((size_t)max_sample * sizeof(uint32_t), "worker positions");
      cow_init(&rc[t].cow, max_sample);
      rc[t].bitmap = bitmap ? xmalloc((size_t)bitmap_bytes, "worker bitmap") : NULL;
      rc[t].mu = &rmu; rc[t].cv = &rcv; rc[t].next = &rnext; rc[t].turn = &rturn;
      rc[t].total_reps = &treps; rc[t].fp = fp; rc[t].out = out;
    }
    if (rw == 1) rep_worker(&rc[0]);
    else {
      for (uint32_t t = 0; t < rw; ++t)
        if (pthread_create(&rth[t], NULL, rep_worker, &rc[t]))
          pdie("cannot create replicate worker", NULL);
      for (uint32_t t = 0; t < rw; ++t) pthread_join(rth[t], NULL);
    }
    for (uint32_t t = 0; t < rw; ++t) {
      free(rc[t].sum); free(rc[t].count); free(rc[t].beta); free(rc[t].selected);
      cow_free(&rc[t].cow); free(rc[t].bitmap); free(rc[t].buf);
    }
    free(rc); free(rth);
  }
  if (fclose(fp)) pdie("error closing output", out);

  char auto_manifest[PATH_MAX];
  if (!manifest) { if (snprintf(auto_manifest, sizeof(auto_manifest), "%s.tsv", out) >= (int)sizeof(auto_manifest)) pdie("output path too long", out); manifest = auto_manifest; }
  FILE *mf = fopen(manifest, "w"); if (!mf) pdie("cannot create manifest", manifest);
  char level_list[MSUR_MAX_LEVELS * 12];
  if (log_min) {
    snprintf(level_list, sizeof(level_list), "logrange %u-%u skew %.3g",
             log_min, log_max, log_skew);
  } else {
    int at = 0;
    for (uint32_t k = 0; k < n_levels; ++k)
      at += snprintf(level_list + at, sizeof(level_list) - (size_t)at,
                     k ? ",%u" : "%u", levels[k]);
  }
  fprintf(mf, "format\t%s\nversion\t%u\ntruth_cg\t%s\nmrmp\t%s\nn_cells\t%u\nn_cpg\t%" PRIu64 "\nn_reps\t%u\nreps_per_level\t%u\nsample_levels\t%s\nsampled_per_cell\t%u\nbinarized\t%s\nn_patterns\t%u\ntruth_encoding\t%s\ngroups_offset\t%" PRIu64 "\ntruth_offset\t%" PRIu64 "\nrecords_offset\t%" PRIu64 "\nrecord_bytes\t%" PRIu64 "\nrandom_protocol\tYAME_dsample_partial_fisher_yates_rand_seed_1_to_n\nfeature_columns\tP1..P%u (PNA excluded; a CHAIN concatenates its sets, so this is the total across them)\n", v3 ? "MSURAW3" : "MSURAW2", hp->version, truth, mrmp, n_cells, n_cpg, total_reps, log_min ? total_reps : reps, level_list, max_sample, binarize ? "yes (one read per observed CpG)" : "no", ncol, embed_truth ? "uint16_beta_0_65534_missing_65535" : "external_cg", hp->groups_offset, hp->truth_offset, hp->records_offset, hp->record_bytes, ncol);
  if (fclose(mf)) pdie("error closing manifest", manifest);
  if (memory_cells) {
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
      free_cdata(&memory_cells[cell]); free(memory_eligible[cell]);
    }
    free(memory_cells);
    free(memory_eligible); free(memory_n_eligible);
  }
  free(group); free(eligible); free(sum); free(count); free(beta); free(selected); cow_free(&cow);
  fprintf(stderr, "[methscope] upscale-featurize: wrote %s and %s\n", out, manifest);
  return 0;
}

/* ---- `methscope inspect DATA.msur` -------------------------------------- */

void ms_msur_report(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) pdie("cannot open", path);
  msur_header3_t h3;
  if (fread(&h3, 1, sizeof(h3), f) != sizeof(h3)) pdie("truncated", path);
  const msur_header_t *h = &h3.h;
  int v3 = msur_is_v3(h);
  if (!v3 && !msur_is_v2(h)) pdie("not a MSURAW2/3 training msur", path);
  msur_rep_t *reps = NULL;
  if (v3) {
    reps = xmalloc((size_t)h->n_reps * sizeof(*reps), "replicate table");
    if (fseek(f, (long)h3.rep_table_offset, SEEK_SET) ||
        fread(reps, sizeof(*reps), h->n_reps, f) != h->n_reps)
      pdie("truncated replicate table", path);
  }
  if (fclose(f)) pdie("error closing", path);
  int truth = (h->flags & MSUR_F_TRUTH_U16) && h->truth_offset;
  printf("format\t%s v%u\n", v3 ? "MSURAW3" : "MSURAW2", h->version);
  printf("cells\t%u\n", h->n_cells);
  printf("replicates\t%u\n", h->n_reps);
  printf("rows\t%" PRIu64 "\t(cells x replicates)\n",
         (uint64_t)h->n_cells * h->n_reps);
  printf("cpgs\t%" PRIu64 "\n", h->n_cpg);
  printf("patterns\t%u\n", h->n_patterns);
  printf("observed_beta\t%s\n", (h->flags & MSUR_F_BINARIZED)
         ? "binarized (one read per CpG)" : "continuous (full-depth truth)");
  if (v3) {
    uint32_t nb = 0;
    for (uint32_t r = 0; r < h->n_reps; ++r) nb += reps[r].flags == MSUR_ENC_BITMAP;
    if (nb) printf("observed_set\tbitmap for %u/%u replicates, sorted list for the rest\n",
                   nb, h->n_reps);
    else    printf("observed_set\tsorted list (all replicates)\n");
  } else {
    printf("observed_set\tsorted list (all replicates)\n");
  }
  printf("embedded_truth\t%s\n", truth ? "yes (trainable)" :
         "no (upscale-train will reject it)");
  printf("groups_bytes\t%" PRIu64 "\n", h->n_cpg * 2);
  if (truth)
    printf("truth_bytes\t%" PRIu64 "\n", (uint64_t)h->n_cells * h->n_cpg * 2);
  if (v3) {
    /* Collapse the per-replicate table to one line per distinct level. */
    uint64_t total = 0;
    printf("sampled_per_cell\tvariable (widest %u)\n", h->sampled_per_cell);
    for (uint32_t i = 0; i < h->n_reps; ++i) {
      total += (uint64_t)h->n_cells * reps[i].record_bytes;
      if (i && reps[i].sample == reps[i - 1].sample) continue;
      uint32_t n = 0;
      for (uint32_t j = i; j < h->n_reps && reps[j].sample == reps[i].sample; ++j) ++n;
      printf("  level\t%u CpGs\t%u replicates\trecord %" PRIu64 " B\n",
             reps[i].sample, n, reps[i].record_bytes);
    }
    printf("records_bytes\t%" PRIu64 "\n", total);
    free(reps);
  } else {
    printf("sampled_per_cell\t%u\n", h->sampled_per_cell);
    printf("record_bytes\t%" PRIu64 "\n", h->record_bytes);
    printf("records_bytes\t%" PRIu64 "\n",
           (uint64_t)h->n_cells * h->n_reps * h->record_bytes);
  }
}
