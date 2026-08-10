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
  /* One CpG->column map per MRMP set. An entry is a 1-based OUTPUT column, or
   * 0 meaning this set calls the CpG background (which lands in pna_col[s]).
   *
   * Holding N maps rather than merging the sets into one mask is the point: a
   * .cm assigns each CpG to exactly ONE pattern, so merging would force
   * overlapping sets to fight over the CpGs they share -- and the informative
   * CpGs are precisely the shared ones. With N maps every set keeps every CpG
   * it selected, and the cell is still inflated only once. The cost is N
   * scatter-adds per sampled CpG against a decompress that dominates. */
  /* INVERTED INDEX over CpGs, not N dense maps.
   *
   * 96.5% of genomic CpGs carry no pattern in ANY set, so scoring a CpG against
   * every set meant ~99 of 100 lookups landing on a background column. Here
   * cpg_off[pos]..cpg_off[pos+1] lists only the columns that CpG really belongs
   * to -- on a 100-set artifact that is 0 entries for almost every CpG.
   *
   * It is also far smaller: 100 dense uint16 maps over 21.8M CpGs is 4.4 GB,
   * where the offsets plus the ~774k real memberships are under 100 MB. */
  const uint32_t *cpg_off;     /* n_cpg + 1 */
  const uint16_t *cpg_col;     /* global output column per membership */
  const uint32_t *pna_col;     /* n_sets: each set's background column */
  /* Per-column binarisation cut, or NULL to leave betas continuous.
   *
   * A pattern's beta is thresholded against ITS OWN midpoint -- the point
   * halfway between the mean reference beta of the classes the pattern calls 1
   * and of those it calls 0 -- not against a flat 0.5. Inside a satellite the
   * two member classes are close relatives and a pattern can sit entirely above
   * 0.5, where 0.5 separates nothing. Binarising here is what makes a feature
   * mean "this cell is on the methylated side of this contrast" rather than
   * "this cell reads 0.918", which is the form that survives a global shift in
   * methylation -- mitotic hypomethylation moves every beta, and a cut defined
   * relative to the pattern's own two groups moves with it.
   *
   * COL_CONTINUOUS leaves a column alone (the per-set PNA background, which is
   * not a contrast); NaN marks a pattern whose groups are not both populated,
   * and those become MSFM_NA. */
  const float *col_thresh;
#define COL_CONTINUOUS (-1.0f)     /* leave this column as a fraction */
  const uint32_t *set_col0;    /* n_sets: first pattern column of each set */
  uint32_t        n_sets;
  uint64_t        n_cpg;
  uint32_t        ncol;        /* total output columns across every set */
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

  /* Phase timers. Four clock reads per cell is noise against a multi-second
   * cell, and guessing which phase dominates has already been wrong twice. */
  double t_io=0, t_elig=0, t_draw=0, t_scatter=0, t_emit=0;
  struct timespec ta, tb;
  #define TICK() clock_gettime(CLOCK_MONOTONIC, &ta)
  #define TOCK(acc) do { clock_gettime(CLOCK_MONOTONIC, &tb); \
    (acc) += (tb.tv_sec-ta.tv_sec) + 1e-9*(tb.tv_nsec-ta.tv_nsec); } while (0)

  for (;;) {
    pthread_mutex_lock(&J->lock);
    uint32_t cell = J->cursor < J->n_cells ? J->cursor++ : UINT32_MAX;
    pthread_mutex_unlock(&J->lock);
    if (cell == UINT32_MAX) break;

    TICK();
    if (bgzf_seek(cf.fh, J->offset[cell], SEEK_SET) != 0)
      bdie("cannot seek to record", J->query);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) bdie("short read on query record", J->query);
    decompress_in_situ(&c);
    TOCK(t_io);
    if (c.fmt != '3' || c.n != J->n_cpg)
      bdie("query record is not format 3 over the reference CpG set", J->query);

    /* Covered CpGs, once per cell: every replicate samples from this list. */
    TICK();
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
    TOCK(t_elig);

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
      TICK();
      if (want < ne) partial_shuffle(elig, ne, want, &rng);
      TOCK(t_draw);

      TICK();
      memset(sum, 0, (size_t)J->ncol * sizeof(double));
      memset(cnt, 0, (size_t)J->ncol * 4);
      double tot_sum = 0;
      for (uint32_t k = 0; k < want; ++k) {
        uint64_t pos = elig[k];
        double b = MU2beta(f3_get_mu(&c, pos));
        /* One read at this CpG: the call is 0 or 1, never a fraction. */
        if (J->binarize) b = rng_01(&rng) < b ? 1.0 : 0.0;
        tot_sum += b;
        for (uint32_t e = J->cpg_off[pos]; e < J->cpg_off[pos + 1]; ++e) {
          uint32_t col = J->cpg_col[e];
          sum[col] += b; ++cnt[col];
        }
      }
      /* Each set's background is what its patterns did NOT take. Deriving it
       * costs O(columns) once instead of one increment per CpG per set, and is
       * exact: every sampled CpG lands in exactly one column of every set. */
      for (uint32_t si2 = 0; si2 < J->n_sets; ++si2) {
        double ss = 0; uint32_t cc = 0;
        for (uint32_t col = J->set_col0[si2]; col < J->pna_col[si2]; ++col) {
          ss += sum[col]; cc += cnt[col];
        }
        sum[J->pna_col[si2]] = tot_sum - ss;
        cnt[J->pna_col[si2]] = want - cc;
      }
      TOCK(t_scatter);

      uint64_t row = (uint64_t)rep * J->n_cells + cell;
      uint16_t *out = J->beta + row * J->ncol;
      /* Thin evidence is recorded as MISSING, not as a confident beta. A beta
       * from one CpG can only be 0 or 1, so after the 0.5 threshold it is always
       * a maximally-confident call and never a borderline one -- and it carries
       * the same pattern weight as a beta backed by hundreds of CpGs. In mouse
       * single cells 46.5% of observed pattern-betas are exactly 0, 0.5 or 1,
       * i.e. rest on one or two measurements. Dropping them needs no format
       * change: every reader already treats MSFM_NA as unobserved. */
      TICK();
      for (uint32_t g = 0; g < J->ncol; ++g) {
        if (cnt[g] < J->min_cpgs) { out[g] = MSFM_NA; continue; }
        double b = sum[g] / (double)cnt[g];
        if (J->col_thresh) {
          float t = J->col_thresh[g];
          if (t != t) { out[g] = MSFM_NA; continue; }   /* unusable pattern */
          if (t >= 0.0f) {
            /* Landing exactly on the cut is not evidence for either side, and
             * `b > t` used to resolve it to 0 -- silently, and always against
             * the expected-1 class. It is not rare: a beta is k/n over the
             * CpGs a cell happened to observe, so any even n split down the
             * middle ties, and at one read per CpG that is mostly n=2. Across
             * mouse single cells 2.60% of observed pattern-betas sit exactly on
             * 0.5, and 99.9% of records carry at least one (median 13, max 68).
             *
             * NA is both honest and better handled: xgboost routes missing down
             * a learned default direction per node, so the model works out which
             * way a tie should fall instead of us hardcoding it wrong. */
            if (b == (double)t) { out[g] = MSFM_NA; continue; }
            b = (b > (double)t) ? 1.0 : 0.0;
          }
        }
        out[g] = msfm_encode(b);
      }
      J->levels[row] = want;
      TOCK(t_emit);
    }
    free_cdata(&c);
    if (J->progress && (cell % 500) == 0)
      fprintf(stderr, "[methscope] classify-featurize: cell %u/%u\n", cell, J->n_cells);
  }
  bgzf_close(cf.fh);
  if (getenv("METHSCOPE_PROFILE"))
    fprintf(stderr, "[profile] io %.2fs  eligible-scan %.2fs  downsample %.2fs  "
            "scatter %.2fs  emit %.2fs\n", t_io, t_elig, t_draw, t_scatter, t_emit);
  free(elig); free(sum); free(cnt);
  #undef TICK
  #undef TOCK
  return NULL;
}

/* Two passes over a block's membership runs: count memberships per CpG, then
 * fill the flat column array. `base` is the set's first global column so the
 * fill writes the final column id and the worker needs no per-set arithmetic. */
typedef struct {
  uint32_t *cnt;        /* count pass */
  uint16_t *col;        /* fill pass */
  uint32_t *cur;        /* fill pass: per-CpG write cursor */
  const uint32_t *off;  /* fill pass: per-CpG start */
  uint32_t base, patterns;
} runacc_t;

static void run_count(void *ctx, uint64_t start, uint64_t len, uint32_t rank) {
  runacc_t *a = ctx;
  if (rank == MRMP_PNA_MEMBERSHIP || rank >= a->patterns) return;
  for (uint64_t i = start; i < start + len; ++i) ++a->cnt[i];
}
static void run_fill(void *ctx, uint64_t start, uint64_t len, uint32_t rank) {
  runacc_t *a = ctx;
  if (rank == MRMP_PNA_MEMBERSHIP || rank >= a->patterns) return;
  uint16_t col = (uint16_t)(a->base + rank);
  for (uint64_t i = start; i < start + len; ++i)
    a->col[a->off[i] + a->cur[i]++] = col;
}

/* ---- entry point -------------------------------------------------------- */

ms_matrix_t *ms_matrix_build_threaded(const char *query, const char *mrmp,
                                      uint32_t patterns, unsigned threads,
                                      uint32_t **levels_out) {
  uint32_t one = 0;                       /* one replicate, native coverage */
  uint16_t *beta; uint32_t *levels; char **names; uint32_t n_cells, ncol;
  /* deconv's reference matrix wants fractions, not calls: NNLS solves for
   * proportions against continuous signatures. */
  ms_msfm_build_sampled(query, mrmp, patterns, &one, 1, 0, 1, 1, threads, 0,
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
                           uint64_t seed, unsigned threads, int binarize_feat,
                           uint16_t **beta_out, uint32_t **levels_out,
                           char ***names_out, uint32_t *n_cells_out,
                           uint32_t *ncol_out) {
  const char *one[1] = {mrmp};
  uint32_t np[1] = {patterns};
  ms_msfm_build_sampled_multi(query, one, NULL, NULL, np, 1, rep_sample, n_reps, binarize,
                              min_cpgs, seed, threads, binarize_feat,
                              beta_out, levels_out,
                              names_out, n_cells_out, ncol_out, NULL);
}

/* N MRMP sets, one pass over the query. Column layout is set-major: set s owns
 * [set_col0[s], set_col0[s] + patterns_s], the last of those being its PNA.
 * With n_sets == 1 the layout and the arithmetic are identical to the old
 * single-mask path, which is what lets the wrapper above stay byte-compatible. */
void ms_msfm_build_sampled_multi(const char *query, const char *const *mrmps,
                           const uint64_t *mrmp_base, const uint64_t *mrmp_len,
                           const uint32_t *patterns_in, uint32_t n_sets,
                           const uint32_t *rep_sample, uint32_t n_reps,
                           int binarize, uint32_t min_cpgs,
                           uint64_t seed, unsigned threads, int binarize_feat,
                           uint16_t **beta_out, uint32_t **levels_out,
                           char ***names_out, uint32_t *n_cells_out,
                           uint32_t *ncol_out, uint32_t *set_col0_out) {
  const char *mrmp = mrmps[0];
  uint32_t patterns = patterns_in[0];
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

  /* CpG -> pattern, once per set. Prefer the MRMPIDX1 artifact so this map and
   * the mask a model bundles come from one source; an exported .cm is still
   * accepted. Memory is n_sets * n_cpg * 2 B -- 44 MB per set at mm10 scale,
   * shared read-only across workers, against an output matrix that is already
   * larger. Chunking, if it is ever needed, belongs on the CELL axis: that
   * keeps the single decompress, which is the whole point. */
  /* Built by walking each set's membership RUNS, never expanding it. A run of
   * background is skipped by advancing a position, so setup costs one visit per
   * real membership (~865k on a 100-set artifact) instead of one per CpG per set
   * (2.18 billion). The dense .cm path below is unchanged -- an exported mask has
   * no runs to walk. */
  uint32_t *cpg_cnt = bmal((size_t)(n_cpg + 1) * sizeof(uint32_t), "cpg membership counts");
  memset(cpg_cnt, 0, (size_t)(n_cpg + 1) * sizeof(uint32_t));
  uint16_t **stage = bmal((size_t)n_sets * sizeof(uint16_t *), "staged maps");
  uint8_t *ra_base = bmal((size_t)n_sets, "run-walk flags");
  for (uint32_t si = 0; si < n_sets; ++si) { stage[si] = NULL; ra_base[si] = 0; }
  uint32_t *pna_col = bmal((size_t)n_sets * sizeof(uint32_t), "pna cols");
  uint32_t *set_col0 = bmal((size_t)n_sets * sizeof(uint32_t), "set offsets");
  uint32_t ncol = 0;
  for (uint32_t si = 0; si < n_sets; ++si) {
  mrmp = mrmps[si];
  patterns = patterns_in[si];
  uint16_t *group = NULL;
  if (ms_mrmp_is_artifact(mrmp)) {
    /* A .mrmp block is walked as runs in place -- no dense map, and no temp .cm
     * materialised on the way in. Exporting one per block used to inflate a
     * membership, walk it twice and recompress a 175 MB buffer, ~0.86 s per set
     * before a single cell was read. */
    if (!patterns) bdie("artifact needs an explicit pattern count", mrmp);
    runacc_t ra; memset(&ra, 0, sizeof ra);
    ra.cnt = cpg_cnt; ra.patterns = patterns;
    ms_mrmp_membership_runs(mrmp, mrmp_base ? mrmp_base[si] : 0,
                            mrmp_len ? mrmp_len[si] : 0, run_count, &ra);
    ra_base[si] = 1;                       /* mark: fill from runs, not a map */
  } else {
    group = bmal((size_t)n_cpg * sizeof(uint16_t), "group map");
    cfile_t cmf = open_cfile((char *)mrmp);
    cdata_t cm = read_cdata1(&cmf);
    bgzf_close(cmf.fh);
    if (!cm.n) bdie("MRMP mask is empty", mrmp);
    /* Walk the mask's RUNS rather than expanding it.
     *
     * A fmt2 .cm is [keys][NUL][value-width][(key-id, uint16 run) ...], and a
     * satellite mask is overwhelmingly one enormous background run: a 34-class
     * global is ~20k runs over 21.8M CpGs. Expanding to a dense array cost one
     * visit per CpG per set -- 2.18 billion on a 100-set artifact -- where the
     * runs carry the same information in a few tens of thousands of steps, and
     * background runs are skipped by advancing a counter.
     *
     * The value in a run is a KEY ID, not a pattern number, so the key table is
     * decoded first and each id mapped through it to P<n>. */
    if (cm.compressed) {
      const uint8_t *p = cm.s; uint64_t n = cm.n;
      uint64_t d = 0;
      while (d + 1 < n && !(p[d] == 0 && p[d + 1] == 0)) ++d;
      if (d + 2 >= n) bdie("mask has no data section", mrmp);
      /* key id -> pattern number, via the table that precedes the runs */
      uint32_t nk = 0;
      for (uint64_t i = 0; i < d; ++i) if (!p[i]) ++nk;
      if (p[d - 1]) ++nk;
      uint32_t *kp = bmal((size_t)(nk ? nk : 1) * 4, "key->pattern");
      { uint32_t k = 0; uint64_t i = 0;
        while (i < d && k < nk) {
          const char *ks = (const char *)p + i;
          kp[k++] = (ks[0] == 'P' && ks[1] >= '0' && ks[1] <= '9')
                  ? (uint32_t)atoi(ks + 1) : 0;
          while (i < d && p[i]) ++i;
          ++i;
        } }
      if (!patterns) { for (uint32_t k = 0; k < nk; ++k) if (kp[k] > patterns) patterns = kp[k]; }
      if (!patterns) bdie("mask has no P<n> states", mrmp);
      memset(group, 0, (size_t)n_cpg * sizeof(uint16_t));
      uint64_t q = d + 2; uint8_t vb = p[q++]; uint64_t pos = 0;
      if (!vb || vb > 8) bdie("mask has a bad value width", mrmp);
      while (q + vb + 2 <= n) {
        uint64_t v = 0;
        for (uint8_t k = 0; k < vb; ++k) v |= (uint64_t)p[q + k] << (8 * k);
        q += vb;
        uint64_t len = (uint64_t)p[q] | ((uint64_t)p[q + 1] << 8); q += 2;
        uint32_t pat = (v < nk) ? kp[v] : 0;
        if (pat && pat <= patterns && pos + len <= n_cpg)
          for (uint64_t i = pos; i < pos + len; ++i) group[i] = (uint16_t)pat;
        pos += len;
      }
      if (pos != n_cpg) bdie("MRMP and query CpG counts differ", mrmp);
      free(kp); free_cdata(&cm);
      goto have_group;
    }
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
  have_group:
  /* set s occupies [set_col0[s], set_col0[s]+patterns], PNA last within it, and
   * the map is rewritten to address the GLOBAL column so the worker needs no
   * per-set offset arithmetic in its inner loop */
  set_col0[si] = ncol;
  /* Only the dense path needs rebasing here; the run walk writes the final
   * column directly, and `group` is NULL for it. */
  if (group)
    for (uint64_t i = 0; i < n_cpg; ++i)
      if (group[i]) group[i] = (uint16_t)(set_col0[si] + group[i]);
  stage[si] = group;
  if (group) for (uint64_t i = 0; i < n_cpg; ++i) if (group[i]) ++cpg_cnt[i];
  pna_col[si] = ncol + patterns;
  ncol += patterns + 1;
  if (ncol > UINT16_MAX) bdie("fused column count exceeds uint16", mrmp);
  }

  /* prefix-sum the per-CpG counts into offsets, then fill */
  uint32_t *cpg_off = bmal((size_t)(n_cpg + 1) * sizeof(uint32_t), "cpg offsets");
  uint64_t acc = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) { cpg_off[i] = (uint32_t)acc; acc += cpg_cnt[i]; }
  cpg_off[n_cpg] = (uint32_t)acc;
  if (acc > UINT32_MAX) bdie("too many pattern memberships to index", mrmps[0]);
  uint16_t *cpg_col = bmal((acc ? acc : 1) * sizeof(uint16_t), "cpg columns");
  memset(cpg_cnt, 0, (size_t)(n_cpg + 1) * sizeof(uint32_t));   /* reuse as cursor */
  for (uint32_t si = 0; si < n_sets; ++si) {
    if (ra_base[si]) {
      runacc_t ra; memset(&ra, 0, sizeof ra);
      ra.col = cpg_col; ra.cur = cpg_cnt; ra.off = cpg_off;
      ra.base = set_col0[si]; ra.patterns = pna_col[si] - set_col0[si];
      ms_mrmp_membership_runs(mrmps[si], mrmp_base ? mrmp_base[si] : 0,
                              mrmp_len ? mrmp_len[si] : 0, run_fill, &ra);
      continue;
    }
    const uint16_t *g = stage[si];
    for (uint64_t i = 0; i < n_cpg; ++i)
      if (g[i]) cpg_col[cpg_off[i] + cpg_cnt[i]++] = (uint16_t)(g[i] - 1);
    free(stage[si]);
  }
  free(stage); free(cpg_cnt); free(ra_base);
  fprintf(stderr, "[methscope] classify-featurize: %" PRIu64 " pattern membership(s) "
          "over %" PRIu64 " CpGs (%.2f%% carry any)\n", acc, n_cpg,
          100.0 * (double)acc / (double)n_cpg);

  /* Per-column cut. 0.5 by default: it is an ABSOLUTE call ("this cell is
   * methylated here"), so it means the same thing on a cohort this reference
   * never saw. The per-pattern midpoint (--thresh-pattern) is fitted to THIS
   * reference's two groups, which separates a close pair better but travels
   * worse -- and travelling is the point when a global shift in methylation,
   * mitotic or otherwise, moves every beta at once. Background columns stay
   * continuous; they are not a contrast. */
  float *col_thresh = NULL;
  if (binarize_feat) {
    col_thresh = bmal((size_t)ncol * sizeof(float), "column cuts");
    for (uint32_t g = 0; g < ncol; ++g) col_thresh[g] = 0.5f;
    for (uint32_t si = 0; si < n_sets; ++si) col_thresh[pna_col[si]] = COL_CONTINUOUS;
    if (binarize_feat == 2) {                 /* --thresh-pattern */
      uint32_t got = 0;
      for (uint32_t si = 0; si < n_sets; ++si) {
        uint32_t np_s = pna_col[si] - set_col0[si];
        if (ms_mrmp_is_artifact(mrmps[si]))
          got += ms_mrmp_thresholds_at(mrmps[si], mrmp_base ? mrmp_base[si] : 0,
                                       mrmp_len ? mrmp_len[si] : 0, np_s,
                                       col_thresh + set_col0[si]);
      }
      if (!got) fprintf(stderr, "[methscope] classify-featurize: no per-pattern "
                        "thresholds in the input; falling back to 0.5\n");
    }
  }

  uint64_t n_rows = (uint64_t)n_reps * n_cells;
  uint16_t *beta = bmal(n_rows * ncol * sizeof(uint16_t), "beta matrix");
  uint32_t *levels = bmal(n_rows * sizeof(uint32_t), "levels");

  job_t J = {0};
  J.query = query; J.n_cells = n_cells; J.n_cpg = n_cpg;
  J.col_thresh = col_thresh;
  J.cpg_off = cpg_off; J.cpg_col = cpg_col;
  J.pna_col = pna_col; J.set_col0 = set_col0;
  J.n_sets = n_sets;
  J.ncol = ncol; J.rep_sample = rep_sample;
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
          n_cells, n_reps, ncol - n_sets, threads, binarize ? ", binarized" : "");
  for (unsigned t = 0; t < threads; ++t)
    if (pthread_create(&tid[t], NULL, worker, &J)) bdie("cannot create thread", NULL);
  for (unsigned t = 0; t < threads; ++t) pthread_join(tid[t], NULL);
  pthread_mutex_destroy(&J.lock);

  free((void *)J.offset); free(tid);
  free(cpg_off); free(cpg_col); free(pna_col); free(col_thresh);
  if (set_col0_out) memcpy(set_col0_out, set_col0, n_sets * sizeof(uint32_t));
  free(set_col0);
  *beta_out = beta; *levels_out = levels; *names_out = names;
  *n_cells_out = n_cells; *ncol_out = ncol;
}
