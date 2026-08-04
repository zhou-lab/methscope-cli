// SPDX-License-Identifier: AGPL-3.0-or-later
/* Sampled featurization: downsample and summarize in one pass, in memory.
 *
 * The shape this replaces was `yame dsample | classify-featurize` per replicate,
 * which wastes work three ways. The source store is decompressed once PER
 * REPLICATE (20 replicates = 20 passes over the same 6 GB, at ~3 MB/s the
 * dominant cost); an intermediate .cg is compressed and immediately
 * decompressed again; and the summary runs `summarize1` per (cell, mask), which
 * walks all 29.4M genomic CpGs whether the record covers 1.7M of them or 4,096.
 *
 * Here each cell is decompressed ONCE and every replicate is drawn from that one
 * inflated copy, and the summary is a scatter-add over only the sampled
 * positions via the CpG->pattern group map -- O(sampled) instead of O(genome),
 * which is what makes the sparse end of a coverage ladder cheap.
 *
 * Threading is per cell: the .cg index carries a BGZF virtual offset per record
 * (the same mechanism `yame subset -l` uses), so a worker seeks straight to its
 * own cells with its own file handle. Output rows are fixed stride, so workers
 * write disjoint slices and nothing needs a lock but the cell cursor.
 *
 * Determinism does NOT depend on --threads: each (cell, replicate) derives its
 * own seed from the run seed, so the draw is a pure function of that integer and
 * cannot depend on which worker got the cell or in what order. A global RNG
 * (which is what the single-threaded upscale featurizer uses) would silently
 * break that.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "methscope.h"
#include "msfm.h"
#include "mrmp.h"
#include "cfile.h"
#include "index.h"

static void bdie(const char *msg, const char *det) __attribute__((noreturn));
static void bdie(const char *msg, const char *det) {
  fprintf(stderr, "[methscope] classify-featurize: %s%s%s\n", msg,
          det ? ": " : "", det ? det : "");
  exit(1);
}

static void *bmal(size_t n, const char *what) {
  void *p = malloc(n ? n : 1);
  if (!p) bdie("out of memory", what);
  return p;
}

/* ---- deterministic per-(cell, replicate) RNG ---------------------------- */

/* splitmix64: tiny, thread-local, and seedable per draw, which is what keeps
 * the result independent of thread scheduling. */
typedef struct { uint64_t s; } rng_t;

static inline uint64_t rng_next(rng_t *r) {
  uint64_t z = (r->s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static inline void rng_seed(rng_t *r, uint64_t base, uint32_t cell, uint32_t rep) {
  /* mix the three so neighbouring (cell, rep) pairs do not give correlated
   * streams -- adding them would make (c+1,r) and (c,r+1) identical */
  r->s = base ^ (0x9e3779b97f4a7c15ULL * (cell + 1)) ^
         (0xc2b2ae3d27d4eb4fULL * (rep + 1));
  (void)rng_next(r);
}
static inline uint32_t rng_below(rng_t *r, uint32_t n) {   /* unbiased enough */
  return (uint32_t)(rng_next(r) % n);
}
static inline double rng_01(rng_t *r) {
  return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Draw `want` of `n` without replacement, leaving them in a[0..want-1].
 * Partial Fisher-Yates, same algorithm the upscale featurizer uses. */
static void partial_shuffle(uint32_t *a, uint32_t n, uint32_t want, rng_t *r) {
  for (uint32_t i = 0; i < want; ++i) {
    uint32_t j = i + rng_below(r, n - i);
    uint32_t t = a[i]; a[i] = a[j]; a[j] = t;
  }
}

/* ---- shared job state --------------------------------------------------- */

typedef struct {
  const char     *query;
  const int64_t  *offset;      /* BGZF virtual offset per cell */
  uint32_t        n_cells;
  const uint16_t *group;       /* n_cpg -> 1-based pattern, 0 = PNA/background */
  uint64_t        n_cpg;
  uint32_t        patterns;    /* feature patterns; column `patterns` is PNA */
  uint32_t        ncol;        /* patterns + 1 */
  const uint32_t *rep_sample;  /* target covered CpGs per replicate; 0 = native */
  uint32_t        n_reps;
  int             binarize;
  uint32_t        min_cpgs;    /* a beta from fewer measured CpGs is recorded NA */
  uint64_t        seed;
  uint16_t       *beta;        /* (n_reps * n_cells) x ncol */
  uint32_t       *levels;      /* n_reps * n_cells */
  uint32_t        cursor;      /* next cell to claim */
  pthread_mutex_t lock;
  int             progress;
} job_t;

static void *worker(void *arg) {
  job_t *J = (job_t *)arg;
  cfile_t cf = open_cfile((char *)J->query);

  uint32_t elig_cap = 1u << 21;                     /* grows if a cell exceeds it */
  uint32_t *elig = bmal((size_t)elig_cap * 4, "eligible positions");
  double   *sum  = bmal((size_t)J->ncol * sizeof(double), "beta sums");
  uint32_t *cnt  = bmal((size_t)J->ncol * 4, "beta counts");

  for (;;) {
    pthread_mutex_lock(&J->lock);
    uint32_t cell = J->cursor < J->n_cells ? J->cursor++ : UINT32_MAX;
    pthread_mutex_unlock(&J->lock);
    if (cell == UINT32_MAX) break;

    if (bgzf_seek(cf.fh, J->offset[cell], SEEK_SET) != 0)
      bdie("cannot seek to record", J->query);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) bdie("short read on query record", J->query);
    decompress_in_situ(&c);
    if (c.fmt != '3' || c.n != J->n_cpg)
      bdie("query record is not format 3 over the reference CpG set", J->query);

    /* Covered CpGs, once per cell: every replicate samples from this list. */
    uint32_t ne = 0;
    for (uint64_t i = 0; i < J->n_cpg; ++i) {
      if (!f3_get_mu(&c, i)) continue;
      if (ne == elig_cap) {
        elig_cap <<= 1;
        elig = realloc(elig, (size_t)elig_cap * 4);
        if (!elig) bdie("out of memory", "eligible grow");
      }
      elig[ne++] = (uint32_t)i;
    }

    for (uint32_t rep = 0; rep < J->n_reps; ++rep) {
      rng_t rng; rng_seed(&rng, J->seed, cell, rep);
      uint32_t want = J->rep_sample[rep];
      /* A target above what the cell holds is not an error: it means native
       * coverage, which is exactly how `dsample -N` behaves and why the ladder
       * saturates at the top rather than failing. */
      if (!want || want > ne) want = ne;
      /* Keeping every CpG is not a sample, so skip the shuffle: it would only
       * permute the summation order, and float addition is not associative, so
       * the native level would stop being bit-comparable with the summarize1
       * path for no gain. Ascending order also matches how YAME accumulates. */
      if (want < ne) partial_shuffle(elig, ne, want, &rng);

      memset(sum, 0, (size_t)J->ncol * sizeof(double));
      memset(cnt, 0, (size_t)J->ncol * 4);
      for (uint32_t k = 0; k < want; ++k) {
        uint64_t pos = elig[k];
        double b = MU2beta(f3_get_mu(&c, pos));
        /* One read at this CpG: the call is 0 or 1, never a fraction. */
        if (J->binarize) b = rng_01(&rng) < b ? 1.0 : 0.0;
        uint16_t g = J->group[pos];
        uint32_t col = g ? (uint32_t)g - 1 : J->patterns;   /* PNA last */
        sum[col] += b; ++cnt[col];
      }

      uint64_t row = (uint64_t)rep * J->n_cells + cell;
      uint16_t *out = J->beta + row * J->ncol;
      /* Thin evidence is recorded as MISSING, not as a confident beta. A beta
       * from one CpG can only be 0 or 1, so after the 0.5 threshold it is always
       * a maximally-confident call and never a borderline one -- and it carries
       * the same pattern weight as a beta backed by hundreds of CpGs. In mouse
       * single cells 46.5% of observed pattern-betas are exactly 0, 0.5 or 1,
       * i.e. rest on one or two measurements. Dropping them needs no format
       * change: every reader already treats MSFM_NA as unobserved. */
      for (uint32_t g = 0; g < J->ncol; ++g)
        out[g] = cnt[g] >= J->min_cpgs ? msfm_encode(sum[g] / cnt[g]) : MSFM_NA;
      J->levels[row] = want;
    }
    free_cdata(&c);
    if (J->progress && (cell % 500) == 0)
      fprintf(stderr, "[methscope] classify-featurize: cell %u/%u\n", cell, J->n_cells);
  }
  bgzf_close(cf.fh);
  free(elig); free(sum); free(cnt);
  return NULL;
}

/* ---- entry point -------------------------------------------------------- */

ms_matrix_t *ms_matrix_build_threaded(const char *query, const char *mrmp,
                                      uint32_t patterns, unsigned threads,
                                      uint32_t **levels_out) {
  uint32_t one = 0;                       /* one replicate, native coverage */
  uint16_t *beta; uint32_t *levels; char **names; uint32_t n_cells, ncol;
  ms_msfm_build_sampled(query, mrmp, patterns, &one, 1, 0, 1, 1, threads,
                        &beta, &levels, &names, &n_cells, &ncol);

  ms_matrix_t *m = bmal(sizeof(*m), "matrix");
  memset(m, 0, sizeof(*m));
  m->n_cells = (int)n_cells; m->n_patterns = (int)ncol;
  m->cell_names = names;                  /* take ownership */
  m->pattern_names = bmal((size_t)ncol * sizeof(char *), "pattern names");
  for (uint32_t c = 0; c + 1 < ncol; ++c) {
    char b[16]; snprintf(b, sizeof(b), "P%u", c + 1);
    m->pattern_names[c] = strdup(b);
  }
  m->pattern_names[ncol - 1] = strdup("Pna");   /* canonical: Pna last */
  m->M = bmal((size_t)n_cells * ncol * sizeof(double), "betas");
  m->N = bmal((size_t)n_cells * ncol * sizeof(int), "counts");
  for (uint64_t k = 0; k < (uint64_t)n_cells * ncol; ++k) {
    m->M[k] = msfm_decode(beta[k]);
    /* no per-pattern counts here; N is only ever asked "observed?", and the
     * exact per-record total travels in levels_out */
    m->N[k] = beta[k] == MSFM_NA ? 0 : 1;
  }
  free(beta);
  if (levels_out) *levels_out = levels; else free(levels);
  return m;
}

void ms_msfm_build_sampled(const char *query, const char *mrmp, uint32_t patterns,
                           const uint32_t *rep_sample, uint32_t n_reps,
                           int binarize, uint32_t min_cpgs,
                           uint64_t seed, unsigned threads,
                           uint16_t **beta_out, uint32_t **levels_out,
                           char ***names_out, uint32_t *n_cells_out,
                           uint32_t *ncol_out) {
  /* record offsets from the .cg index -- the same mechanism `yame subset -l`
   * uses, and what lets workers seek instead of sharing one stream */
  int npairs = 0;
  char *fname_index = get_fname_index((char *)query);
  if (!fname_index) bdie("cannot derive the index name", query);
  index_t *idx = loadIndex(fname_index);
  if (!idx) bdie("query needs an index (.cg.idx) for sampled featurization", query);
  index_pair_t *pairs = index_pairs(idx, &npairs);
  free(fname_index);
  if (!pairs || npairs <= 0) bdie("query index is empty", query);
  uint32_t n_cells = (uint32_t)npairs;

  /* CpG count from the first record */
  cfile_t cf0 = open_cfile((char *)query);
  cdata_t c0 = read_cdata1(&cf0);
  if (!c0.n) bdie("query is empty", query);
  decompress_in_situ(&c0);
  uint64_t n_cpg = c0.n;
  if (c0.fmt != '3') bdie("query must be format 3 (M/U counts)", query);
  free_cdata(&c0);
  bgzf_close(cf0.fh);

  /* CpG -> pattern. Prefer the MRMPIDX1 artifact so this map and the mask a
   * model bundles come from one source; an exported .cm is still accepted. */
  uint16_t *group = bmal((size_t)n_cpg * sizeof(uint16_t), "group map");
  if (ms_mrmp_is_artifact(mrmp)) {
    ms_mrmp_group_map(mrmp, group, n_cpg, patterns);
  } else {
    cfile_t cmf = open_cfile((char *)mrmp);
    cdata_t cm = read_cdata1(&cmf);
    bgzf_close(cmf.fh);
    if (!cm.n) bdie("MRMP mask is empty", mrmp);
    decompress_in_situ(&cm);
    if (cm.fmt != '2') bdie("MRMP mask must be categorical format 2", mrmp);
    fmt2_set_aux(&cm);
    if (cm.n != n_cpg) bdie("MRMP and query CpG counts differ", mrmp);
    if (!patterns) {          /* default: every non-Pna state in the mask */
      uint32_t hi = 0;
      for (uint64_t i = 0; i < n_cpg; ++i) {
        const char *s2 = f2_get_string(&cm, i);
        if (s2 && s2[0] == 'P' && s2[1] >= '0' && s2[1] <= '9') {
          uint32_t v = (uint32_t)atoi(s2 + 1);
          if (v > hi) hi = v;
        }
      }
      patterns = hi;
      if (!patterns) bdie("mask has no P<n> states", mrmp);
    }
    for (uint64_t i = 0; i < n_cpg; ++i) {
      const char *s = f2_get_string(&cm, i);
      int p = (s && s[0] == 'P' && s[1] >= '0' && s[1] <= '9') ? atoi(s + 1) : 0;
      group[i] = (p > 0 && p <= (int)patterns) ? (uint16_t)p : 0;
    }
    free_cdata(&cm);
  }

  uint32_t ncol = patterns + 1;                   /* + PNA, kept as the last column */
  uint64_t n_rows = (uint64_t)n_reps * n_cells;
  uint16_t *beta = bmal(n_rows * ncol * sizeof(uint16_t), "beta matrix");
  uint32_t *levels = bmal(n_rows * sizeof(uint32_t), "levels");

  job_t J = {0};
  J.query = query; J.n_cells = n_cells; J.group = group; J.n_cpg = n_cpg;
  J.patterns = patterns; J.ncol = ncol; J.rep_sample = rep_sample;
  J.n_reps = n_reps; J.binarize = binarize; J.seed = seed;
  /* 0 means "unset": keep any observed CpG, i.e. the old behaviour. */
  J.min_cpgs = min_cpgs ? min_cpgs : 1;
  J.beta = beta; J.levels = levels; J.cursor = 0; J.progress = 1;
  J.offset = bmal((size_t)n_cells * sizeof(int64_t), "offsets");
  char **names = bmal((size_t)n_cells * sizeof(char *), "cell names");
  for (uint32_t i = 0; i < n_cells; ++i) {
    ((int64_t *)J.offset)[i] = pairs[i].value;
    names[i] = strdup(pairs[i].key);
  }
  clean_index_pairs(pairs, npairs);
  pthread_mutex_init(&J.lock, NULL);

  if (threads < 1) threads = 1;
  if (threads > n_cells) threads = n_cells;
  pthread_t *tid = bmal(threads * sizeof(pthread_t), "threads");
  fprintf(stderr, "[methscope] classify-featurize: %u cells x %u replicate(s) "
          "x %u patterns, %u thread(s)%s\n",
          n_cells, n_reps, patterns, threads, binarize ? ", binarized" : "");
  for (unsigned t = 0; t < threads; ++t)
    if (pthread_create(&tid[t], NULL, worker, &J)) bdie("cannot create thread", NULL);
  for (unsigned t = 0; t < threads; ++t) pthread_join(tid[t], NULL);
  pthread_mutex_destroy(&J.lock);

  free((void *)J.offset); free(tid); free(group);
  *beta_out = beta; *levels_out = levels; *names_out = names;
  *n_cells_out = n_cells; *ncol_out = ncol;
}
