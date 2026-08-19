// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Adaptive deconvolution: pick the cell types from the sample, rebuild the
 * pattern set over just those, and re-solve -- all in one process, with the
 * reference resident in memory.
 *
 * WHY REBUILD RATHER THAN JUST RESTRICT
 *
 * A pattern must hold across every class in its set, so the q-filter is a
 * conjunction that tightens as classes are added. Monocyte vs Macrophage
 * carries 300 segregating CpGs inside a 33-class build and 11,935 as its own
 * 2-class set, and 92% of that gap is CpGs the wider filter REJECTED, not CpGs
 * it merely scattered across patterns. Measured on 200 known mixtures at 2^16
 * CpGs: dropping classes while REUSING the 33-class patterns is worth nothing
 * (eleven rule families, all at the baseline's 6-group r^2 0.168), while
 * dropping them and REBUILDING reaches 0.601 given the true class set and
 * 0.218 given one chosen from the data. The rebuild is the whole effect.
 *
 * WHY THE REFERENCE IS HELD IN MEMORY
 *
 * The rebuild is per sample, so the reference is read over and over: the
 * out-of-process form costs `yame subset` (a 2.6 GB read plus a write), then
 * mrmp-build's three store passes, then deconv-build-ref's, ~6 minutes per
 * class set. Held resident it is one 46 s load and then roughly a second per
 * iteration. Two facts make the resident form small:
 *
 *   - a pattern's beta is the MEAN OF PER-CpG BETAS (YAME format3.c,
 *     st[0].beta = sum_beta / n_o), never sum(M)/sum(M+U), so M and U are
 *     never needed apart -- only beta.
 *   - at the settings this supports, the only coverage fact needed is
 *     "covered at all", which a sentinel value carries.
 *
 * so beta as uint16 with 0xFFFF for uncovered costs 2 B per (class, CpG):
 * 1.94 GB for 33 classes x 29.4 M CpGs, against 7.76 GB for the store's own
 * 8-byte M|U packing. Quantisation error is 7.6e-6, far below anything the
 * 0.30/0.70 bands can resolve. (uint8 would halve it again, but its 2e-3 error
 * can flip a CpG sitting exactly on a band edge, which would make this build
 * disagree with mrmp-build's -- not worth 1 GB.)
 *
 * MEMORY. This must not run on a login node: 1.94 GB resident for hg38/33
 * classes, 2.41 GB for mouse/41, 3.82 GB for a 65-type atlas. Use srun/sbatch.
 *
 * WHAT ONE ITERATION DOES
 *
 * Per (class, CpG) two facts, and both are properties of that class ALONE --
 * neither depends on which other classes are in the set:
 *
 *     side   the binstring bit, beta > --beta-threshold
 *     ok     covered AND clearing the q-filter ON ITS OWN SIDE, i.e.
 *            side ? beta >= qfilter_hi : beta <= qfilter_lo
 *
 * so for a class set S: admissible ⟺ every k in S is ok, the binstring is the
 * side bits of S, and constant binstrings are dropped (no class on one side is
 * no contrast). Per-pattern reference betas accumulate in the SAME pass, which
 * is what removes deconv-build-ref entirely -- the betas are already in hand.
 *
 * Because admissibility requires every class in S covered, the reference comes
 * out NaN-free by construction and there is nothing to impute.
 *
 * VALID ONLY FOR the shipped selection rule: --qfilter admits and nothing else
 * cuts. --delta-mean-top (rank by delta_mean), --depth-floor-frac and
 * --min-cg-depth all need per-CpG facts a resident beta does not carry, and
 * --max-frac-na > 0 would let the sweeping-majority FILL fire, which is the one
 * genuinely set-dependent step in resolve_cpg. Those are refused, not ignored.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include "methscope.h"
#include "nnls.h"
#include <pthread.h>
#include "cfile.h"
#include "cdata.h"

#define BETA_NA 0xFFFFu            /* uncovered; 0..65534 encode beta 0..1 */

static void adie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] deconv --adaptive: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] deconv --adaptive: %s\n", msg);
  exit(1);
}

static void *axc(size_t n, size_t sz, const char *what) {
  void *p = calloc(n ? n : 1, sz);
  if (!p) adie("out of memory", what);
  return p;
}

/* ------------------------------------------------------------------ */
/* The resident reference: names + a uint16 beta per (class, CpG).      */
/* ------------------------------------------------------------------ */
typedef struct {
  uint32_t  n_class;
  uint64_t  n_cpg;
  char    **name;
  uint16_t *beta;            /* n_class * n_cpg, class-major */
} refmem_t;

static uint16_t beta_encode(double b) {
  if (b < 0) b = 0; else if (b > 1) b = 1;
  return (uint16_t)(b * 65534.0 + 0.5);
}
static double beta_decode(uint16_t v) { return (double)v / 65534.0; }

/* Names in FILE order, which IS the binstring digit order -- a permuted walk
 * would silently relabel every pattern. */
static char **read_idx(const char *ref, uint32_t *n_out, int64_t **off_out) {
  char idx[PATH_MAX];
  if (snprintf(idx, sizeof idx, "%s.idx", ref) >= (int)sizeof idx)
    adie("reference path too long", ref);
  FILE *f = fopen(idx, "r");
  if (!f) adie("cannot open reference index (expected <ref>.idx)", idx);
  size_t cap = 64, n = 0;
  char **names = axc(cap, sizeof(char *), "names");
  int64_t *off = axc(cap, sizeof(int64_t), "offsets");
  char *line = NULL; size_t lc = 0; ssize_t len;
  while ((len = getline(&line, &lc, f)) > 0) {
    char *tab = strpbrk(line, "\t\n");
    size_t nl = tab ? (size_t)(tab - line) : (size_t)len;
    if (!nl) continue;
    if (n == cap) {
      cap <<= 1;
      names = realloc(names, cap * sizeof(char *));
      off = realloc(off, cap * sizeof(int64_t));
      if (!names || !off) adie("out of memory", "index grow");
    }
    names[n] = axc(nl + 1, 1, "name");
    memcpy(names[n], line, nl);
    off[n] = (tab && *tab == '\t') ? (int64_t)strtoll(tab + 1, NULL, 10) : -1;
    ++n;
  }
  free(line); fclose(f);
  if (!n) adie("reference index is empty", idx);
  *n_out = (uint32_t)n; *off_out = off;
  return names;
}

static void refmem_load(const char *ref, uint32_t mincov, refmem_t *R) {
  int64_t *voff = NULL;
  R->name = read_idx(ref, &R->n_class, &voff);
  R->n_cpg = 0; R->beta = NULL;

  cfile_t cf = open_cfile((char *)ref);
  for (uint32_t k = 0; k < R->n_class; ++k) {
    if (voff[k] >= 0 && bgzf_seek(cf.fh, voff[k], SEEK_SET) != 0)
      adie("cannot seek to record", R->name[k]);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) adie("store record is empty", R->name[k]);
    decompress_in_situ(&c);
    if (c.fmt != '3') adie("reference must be format-3 (M/U) .cg", R->name[k]);
    if (!k) {
      R->n_cpg = c.n;
      R->beta = axc((size_t)R->n_class * R->n_cpg, sizeof(uint16_t), "beta");
      for (size_t j = 0; j < (size_t)R->n_class * R->n_cpg; ++j)
        R->beta[j] = BETA_NA;
    } else if (c.n != R->n_cpg) {
      adie("reference records disagree on CpG count", R->name[k]);
    }
    uint16_t *row = R->beta + (size_t)k * R->n_cpg;
    for (uint64_t i = 0; i < R->n_cpg; ++i) {
      uint64_t mu = f3_get_mu(&c, i);
      if (!mu || MU2cov(mu) < mincov) continue;
      row[i] = beta_encode(MU2beta(mu));
    }
    free_cdata(&c);
    fprintf(stderr, "\r[methscope] deconv --adaptive: loading reference %u/%u",
            k + 1, R->n_class);
    fflush(stderr);
  }
  bgzf_close(cf.fh);
  fprintf(stderr, "\r[methscope] deconv --adaptive: reference %u classes x "
                  "%llu CpGs, %.2f GB resident\n",
          R->n_class, (unsigned long long)R->n_cpg,
          (double)R->n_class * R->n_cpg * 2.0 / 1e9);
  free(voff);
}

/* ------------------------------------------------------------------ */
/* The query: beta + covered flag per CpG, one record.                 */
/* ------------------------------------------------------------------ */
typedef struct { uint64_t n_cpg; uint16_t *beta; } query_t;

/* Convert ONE already-read record. Records are streamed rather than all held:
 * a multi-record query is the natural cohort form (`cat`ted .cg + .idx, YAME's
 * own idiom), and 200 mixtures resident would be 11.8 GB against the 59 MB one
 * at a time actually needs. */
static void query_from_cdata(cdata_t *c, uint32_t mincov, query_t *Q) {
  Q->n_cpg = c->n;
  Q->beta = axc(Q->n_cpg, sizeof(uint16_t), "query beta");
  for (uint64_t i = 0; i < Q->n_cpg; ++i) {
    uint64_t mu = f3_get_mu(c, i);
    Q->beta[i] = (!mu || MU2cov(mu) < mincov) ? BETA_NA
                                              : beta_encode(MU2beta(mu));
  }
}

/* Record names from <query>.idx when there is one; otherwise 1,2,3... exactly
 * as `deconv` names the records of a stream. */
static char **query_names(const char *path, uint32_t *n_out,
                          int64_t **off_out) {
  char idx[PATH_MAX];
  if (snprintf(idx, sizeof idx, "%s.idx", path) >= (int)sizeof idx) return NULL;
  FILE *f = fopen(idx, "r");
  if (!f) { *n_out = 0; if (off_out) *off_out = NULL; return NULL; }
  size_t cap = 64, n = 0;
  char **nm = axc(cap, sizeof(char *), "query names");
  int64_t *off = axc(cap, sizeof(int64_t), "query offsets");
  char *line = NULL; size_t lc = 0; ssize_t len;
  while ((len = getline(&line, &lc, f)) > 0) {
    char *tab = strpbrk(line, "\t\n");
    size_t nl = tab ? (size_t)(tab - line) : (size_t)len;
    if (!nl) continue;
    if (n == cap) {
      cap <<= 1;
      nm = realloc(nm, cap * sizeof(char *));
      off = realloc(off, cap * sizeof(int64_t));
      if (!nm || !off) adie("out of memory", "query index grow");
    }
    nm[n] = axc(nl + 1, 1, "query name");
    memcpy(nm[n], line, nl);
    off[n] = (tab && *tab == '\t') ? (int64_t)strtoll(tab + 1, NULL, 10) : -1;
    ++n;
  }
  free(line); fclose(f);
  *n_out = (uint32_t)n;
  if (off_out) *off_out = off; else free(off);
  return nm;
}

/* ------------------------------------------------------------------ */
/* One iteration: patterns + reference betas + query betas for a set S. */
/* ------------------------------------------------------------------ */
/* A pattern key is the side bits of S packed into words, so a set of up to 64
 * classes is one word and the common case needs no hashing of variable-length
 * keys. Sets larger than 64 spill to a second word and so on. */
typedef struct {
  uint64_t *key;       /* n_pat * kw */
  uint64_t *count;     /* n_pat, CpGs carrying it */
  double   *rsum;      /* n_pat * ns, running sum of reference beta */
  double   *qsum;      /* n_pat, running sum of query beta */
  uint64_t *qn;        /* n_pat, query CpGs observed */
  uint32_t  n_pat, kw, ns;   /* ns = classes DEFINING the patterns (the scope) */
  uint32_t  nall;            /* rows in rsum: every class, scope or not */
} panel_t;

static void panel_free(panel_t *p) {
  free(p->key); free(p->count); free(p->rsum); free(p->qsum); free(p->qn);
}

/* Open-addressed key -> pattern index. */
typedef struct { uint32_t *slot; uint64_t mask; uint32_t kw; } khash_t2;

static uint64_t key_hash(const uint64_t *k, uint32_t kw) {
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t w = 0; w < kw; ++w) { h ^= k[w]; h *= 1099511628211ULL; }
  return h ? h : 1;
}

static uint32_t panel_intern(panel_t *p, khash_t2 *H, const uint64_t *k,
                             uint32_t *cap) {
  uint64_t i = key_hash(k, p->kw) & H->mask;
  while (H->slot[i] != UINT32_MAX) {
    uint32_t r = H->slot[i];
    if (!memcmp(p->key + (size_t)r * p->kw, k, p->kw * sizeof(uint64_t)))
      return r;
    i = (i + 1) & H->mask;
  }
  if (p->n_pat == *cap) {
    *cap *= 2;
    p->key   = realloc(p->key, (size_t)*cap * p->kw * sizeof(uint64_t));
    p->count = realloc(p->count, (size_t)*cap * sizeof(uint64_t));
    p->rsum  = realloc(p->rsum, (size_t)*cap * p->nall * sizeof(double));
    p->qsum  = realloc(p->qsum, (size_t)*cap * sizeof(double));
    p->qn    = realloc(p->qn, (size_t)*cap * sizeof(uint64_t));
    if (!p->key || !p->count || !p->rsum || !p->qsum || !p->qn)
      adie("out of memory", "panel grow");
    memset(p->count + p->n_pat, 0, (*cap - p->n_pat) * sizeof(uint64_t));
    memset(p->rsum + (size_t)p->n_pat * p->nall, 0,
           (size_t)(*cap - p->n_pat) * p->nall * sizeof(double));
    memset(p->qsum + p->n_pat, 0, (*cap - p->n_pat) * sizeof(double));
    memset(p->qn + p->n_pat, 0, (*cap - p->n_pat) * sizeof(uint64_t));
  }
  uint32_t r = p->n_pat++;
  memcpy(p->key + (size_t)r * p->kw, k, p->kw * sizeof(uint64_t));
  H->slot[i] = r;
  return r;
}

/* Build the pattern set for class subset `sel` (indices into R), accumulating
 * per-pattern reference and query sums in the same pass. */
/* `keep_scope_constant`: retain a CpG whose scope-restricted binstring is
 * constant, provided it still varies across the FULL class set.
 *
 * Dropping constant patterns is right when the scope IS the design -- a
 * pattern with no class on one side separates nothing. But under columns-only
 * the design keeps every class while the scope only picks the CpGs, and then
 * the exclusion is actively harmful: a CpG where all 5 scoped classes agree
 * may discriminate the other 28 perfectly, and throwing it away is what leaves
 * the out-of-scope classes unconstrained while they remain free to absorb
 * mass. Those CpGs pool into the all-0 and all-1 columns, which is exactly
 * where out-group information belongs. */
/* One admitted CpG, pending the per-binstring --delta-mean-top selection.
 * `gap` is the class gap in ENCODED beta units: the lowest 1-side class minus
 * the highest 0-side class, so a bigger gap is a cleaner separator. */
typedef struct { uint32_t r; uint64_t i; float gap; } dment_t;

static int dment_cmp(const void *a, const void *b) {
  const dment_t *x = a, *y = b;
  if (x->r != y->r) return x->r < y->r ? -1 : 1;
  if (x->gap != y->gap) return x->gap > y->gap ? -1 : 1;   /* best first */
  return 0;
}

/* Fold one CpG's reference and query betas into pattern `r`. */
static void panel_accum(panel_t *out, uint32_t r, uint64_t i,
                        const refmem_t *R, const query_t *Q) {
  out->count[r]++;
  double *rs = out->rsum + (size_t)r * out->nall;
  for (uint32_t c = 0; c < out->nall; ++c) {
    uint16_t b = R->beta[(size_t)c * R->n_cpg + i];
    if (b != BETA_NA) rs[c] += beta_decode(b);
  }
  if (i < Q->n_cpg && Q->beta[i] != BETA_NA) {
    out->qsum[r] += beta_decode(Q->beta[i]);
    out->qn[r]++;
  }
}

static void panel_build(const refmem_t *R, const query_t *Q,
                        const uint32_t *sel, uint32_t ns,
                        double thr, double qlo, double qhi,
                        int keep_scope_constant, uint64_t dm_top,
                        panel_t *out) {
  const uint32_t kw = (ns + 63) >> 6;
  memset(out, 0, sizeof *out);
  out->kw = kw; out->ns = ns; out->nall = R->n_class;
  uint32_t cap = 4096;
  out->key   = axc((size_t)cap * kw, sizeof(uint64_t), "keys");
  out->count = axc(cap, sizeof(uint64_t), "counts");
  out->rsum  = axc((size_t)cap * R->n_class, sizeof(double), "ref sums");
  out->qsum  = axc(cap, sizeof(double), "query sums");
  out->qn    = axc(cap, sizeof(uint64_t), "query counts");

  khash_t2 H;
  H.kw = kw; H.mask = (1ull << 18) - 1;
  H.slot = malloc((H.mask + 1) * sizeof(uint32_t));
  if (!H.slot) adie("out of memory", "pattern hash");
  memset(H.slot, 0xFF, (H.mask + 1) * sizeof(uint32_t));

  uint64_t *k = axc(kw, sizeof(uint64_t), "key scratch");
  const uint16_t lo = beta_encode(qlo), hi = beta_encode(qhi),
                 bt = beta_encode(thr);

  dment_t *ent = NULL; size_t n_ent = 0, cap_ent = 0;

  for (uint64_t i = 0; i < R->n_cpg; ++i) {
    /* RESTRICT THE REFERENCE TO WHAT THIS QUERY MEASURED, before anything else.
     * A pattern's reference beta is the mean over the CpGs that DEFINE it and
     * its observed beta the mean over the CpGs the query COVERED; if those are
     * different sets the two sides of the equation are not comparable, and at
     * 2^16 they differ by a factor of ~300. Restricting first makes every
     * downstream fact -- admissibility, the binstring, constancy, the CpG count
     * that sets the weight, and the --delta-mean-top ranking -- a statement
     * about CpGs this query actually saw. count[r] == qn[r] by construction. */
    if (i >= Q->n_cpg || Q->beta[i] == BETA_NA) continue;
    memset(k, 0, kw * sizeof(uint64_t));
    uint32_t n1 = 0;
    int admissible = 1;
    uint32_t min1 = 0xFFFFFFFFu, max0 = 0;      /* for the --delta-mean-top gap */
    for (uint32_t s = 0; s < ns; ++s) {
      uint16_t b = R->beta[(size_t)sel[s] * R->n_cpg + i];
      if (b == BETA_NA) { admissible = 0; break; }   /* absent: reject */
      if (b > bt) {
        if (b < hi) { admissible = 0; break; }       /* 1-side, fails HI */
        k[s >> 6] |= 1ull << (s & 63);
        ++n1;
        if (b < min1) min1 = b;
      } else {
        if (b > max0) max0 = b;
        /* b == bt is M==U, not a call; mrmp.c treats it as ambiguous, and an
         * ambiguous class is absent for our purposes. */
        if (b == bt || b > lo) { admissible = 0; break; }
      }
    }
    if (!admissible) continue;
    if (!n1 || n1 == ns) {
      if (!keep_scope_constant) continue;   /* no contrast within the scope */
      /* keep it only if some OTHER class disagrees -- otherwise the CpG is
       * constant over every class and informs nothing at all */
      int varies = 0;
      for (uint32_t c = 0; c < R->n_class && !varies; ++c) {
        uint16_t bc = R->beta[(size_t)c * R->n_cpg + i];
        if (bc == BETA_NA) continue;
        int side_c = (bc > bt);
        if (side_c != (n1 != 0)) varies = 1;
      }
      if (!varies) continue;
    }

    uint32_t r = panel_intern(out, &H, k, &cap);
    if (dm_top) {
      /* defer: the per-binstring cap keeps only the N cleanest separators, and
       * which those are is not known until the whole genome has been seen */
      if (n_ent == cap_ent) {
        cap_ent = cap_ent ? cap_ent * 2 : 65536;
        ent = realloc(ent, cap_ent * sizeof(dment_t));
        if (!ent) adie("out of memory", "delta-mean entries");
      }
      ent[n_ent].r = r; ent[n_ent].i = i;
      ent[n_ent].gap = (n1 && n1 != ns)
                       ? (float)((double)min1 - (double)max0) : 0.0f;
      ++n_ent;
      continue;
    }
    out->count[r]++;
    /* Accumulate the reference for EVERY class, not just the scope. A class's
     * beta over a pattern is well defined whatever chose that pattern, so the
     * scope can pick the COLUMNS while the design keeps all the CLASSES --
     * which is what makes a wrong scope cost precision instead of a lost
     * component. Classes outside the scope contribute a NaN-free row because
     * admissibility already required coverage only within the scope, so an
     * out-of-scope class may be uncovered here and is skipped for that CpG. */
    double *rs = out->rsum + (size_t)r * out->nall;
    for (uint32_t c = 0; c < out->nall; ++c) {
      uint16_t b = R->beta[(size_t)c * R->n_cpg + i];
      if (b != BETA_NA) rs[c] += beta_decode(b);
    }
    if (i < Q->n_cpg && Q->beta[i] != BETA_NA) {
      out->qsum[r] += beta_decode(Q->beta[i]);
      out->qn[r]++;
    }
  }

  /* --delta-mean-top: per BINSTRING, keep only the N CpGs with the cleanest
   * class separation. Without it one binstring can swamp the panel -- in the
   * 8-class immune rebuild `11111011` (every class but Neuron methylated,
   * separating nothing in an immune mixture) held 124,546 CpGs and the top
   * five held 91.3% of the panel, while the 29 patterns that actually split
   * T.Cell.CD4 from T.Cell.CD8 held 0.41%. Influence is linear in CpG count,
   * so that is the fit. Capping equalises the binstrings and keeps each one's
   * best evidence rather than all of its mediocre evidence. */
  if (dm_top) {
    qsort(ent, n_ent, sizeof(dment_t), dment_cmp);
    uint64_t kept = 0;
    for (size_t a = 0; a < n_ent; ) {
      size_t b = a;
      while (b < n_ent && ent[b].r == ent[a].r) ++b;
      size_t take = (b - a) < dm_top ? (b - a) : dm_top;
      for (size_t t = 0; t < take; ++t)
        panel_accum(out, ent[a + t].r, ent[a + t].i, R, Q);
      kept += take;
      a = b;
    }
    free(ent);
  }
  free(k); free(H.slot);
}

/* ------------------------------------------------------------------ */
/* Solve one panel: complete-case NNLS over the patterns the query saw. */
/* ------------------------------------------------------------------ */
/* `sel` + `n_sel` select which rows of the panel's reference matrix form the
 * design. Passing NULL uses EVERY class (columns-only scoping): the scope then
 * decides which CpGs define the patterns, and nothing is removed from the
 * solve, so a mis-scoped round costs precision rather than a component. */
/* The weight one pattern's equation carries, given n observed CpGs.
 *
 * sqrt(n) is the textbook precision weight: a pattern's value is a mean over n
 * covered CpGs, so its variance is sigma_s^2/n and sqrt(n) on the row is
 * ordinary weighted least squares. Because NNLS then SQUARES the row, a
 * pattern's influence on the objective is linear in n -- a 124,546-CpG pattern
 * counts 124,546 times a 1-CpG one. That is right when sampling noise is the
 * only error.
 *
 * It is not the only error. A pattern also carries MODEL error -- reference vs
 * query donors, a mixture that is not exactly linear, a reference profile that
 * is slightly off -- and that floor does not shrink with n. Past the point
 * where model error dominates, sqrt(n) keeps promoting a pattern that stopped
 * becoming more trustworthy, and it crowds everything else out: in the 8-class
 * immune rebuild five patterns hold 91.3% of the panel's CpGs and NONE of them
 * separates T.Cell.CD4 from T.Cell.CD8, whose 29 patterns hold 0.41%. The pair
 * goes effectively collinear and NNLS hands its mass to whichever column comes
 * first -- CD4, which then absorbs CD8 wholesale (0.285 against a true 0.000).
 *
 * Two independent knobs bound how much one pattern can claim:
 *
 *     w(n) = pow(min(n, cap), expo)
 *
 *   cap   "no pattern counts for more than `cap` CpGs of evidence, however
 *         many it holds". Below it nothing changes, so patterns the cap does
 *         not bind on keep exactly the weight they had; above it they tie
 *         instead of competing. cap = 0 is no cap.
 *   expo  0.5 is sqrt(n), the precision weight, giving influence linear in n.
 *         0.25 is sqrt(sqrt(n)), giving influence proportional to sqrt(n) --
 *         it does not truncate anyone, it compresses the whole range, which
 *         moves the CD4/CD8 splitters from 0.41% of the 8-class panel's
 *         influence to 5.84%. 0 is unweighted.
 *
 * cap = 0, expo = 0.5 is the rule every number before 20260816 was measured
 * under, so the old behaviour is a point in this family rather than a branch. */
static double pattern_weight(uint64_t n, uint64_t cap, double expo) {
  if (!n) return 0.0;
  if (expo != 0.5) {
    if (cap && n > cap) n = cap;
    return pow((double)n, expo);
  }
  if (cap && n > cap) n = cap;
  return sqrt((double)n);
}

static void panel_solve(const panel_t *p, uint32_t min_cov, double *x,
                        const uint32_t *sel, uint32_t n_sel, int weighted,
                        uint64_t cap, double expo) {
  const uint32_t ns = sel ? n_sel : p->nall;
  uint32_t nu = 0;
  for (uint32_t r = 0; r < p->n_pat; ++r)
    if (p->qn[r] >= min_cov) ++nu;
  for (uint32_t s = 0; s < ns; ++s) x[s] = 0;
  if (!nu) return;

  double *A = axc((size_t)nu * ns, sizeof(double), "design");
  double *b = axc(nu, sizeof(double), "observed");
  uint32_t q = 0;
  for (uint32_t r = 0; r < p->n_pat; ++r) {
    if (p->qn[r] < min_cov) continue;
    const double *rs = p->rsum + (size_t)r * p->nall;
    for (uint32_t s = 0; s < ns; ++s)                 /* COLUMN-major for nnls */
      A[(size_t)s * nu + q] = rs[sel ? sel[s] : s] / (double)p->count[r];
    b[q] = p->qsum[r] / (double)p->qn[r];
    /* WEIGHT BY CpG SUPPORT. A pattern's observed value is a mean over the
     * CpGs the query actually covered, so its variance goes as 1/n and the
     * right weight is n -- applied as sqrt(n) on the row, which is ordinary
     * weighted least squares. Unweighted, a pattern resting on ONE observed
     * CpG counts as much as one resting on 500, and at 2^16 the median
     * pattern holds 2 CpGs while a pair-specific add-on holds hundreds. */
    if (weighted) {
      double wq = pattern_weight(p->qn[r], cap, expo);
      for (uint32_t s = 0; s < ns; ++s) A[(size_t)s * nu + q] *= wq;
      b[q] *= wq;
    }
    ++q;
  }
  double *aw = axc((size_t)nu * ns, sizeof(double), "nnls A");
  double *bw = axc(nu, sizeof(double), "nnls b");
  double *w  = axc(ns, sizeof(double), "nnls w");
  double *zz = axc(nu, sizeof(double), "nnls zz");
  int    *ix = axc(ns, sizeof(int), "nnls index");
  memcpy(aw, A, (size_t)nu * ns * sizeof(double));
  memcpy(bw, b, (size_t)nu * sizeof(double));
  int mda = (int)nu, m = (int)nu, n = (int)ns, mode = 0;
  double rnorm = 0;
  nnls_c(aw, &mda, &m, &n, bw, x, &rnorm, w, zz, ix, &mode);
  if (mode != 1)
    fprintf(stderr, "[methscope] warning: NNLS mode %d (%u patterns)\n",
            mode, nu);
  double sum = 0;
  for (uint32_t s = 0; s < ns; ++s) { if (x[s] < 0) x[s] = 0; sum += x[s]; }
  if (sum > 0) for (uint32_t s = 0; s < ns; ++s) x[s] /= sum;
  free(A); free(b); free(aw); free(bw); free(w); free(zz); free(ix);
}


typedef struct {
  int      max_round;
  uint64_t group_thresh;
  double   drop_below, qlo, qhi, beta_thr;
  uint32_t min_cov_pat;
  int      verbose;                   /* per-round lines; off when threaded */
  int      columns_only;              /* scope picks CpGs, never the design */
  int      pairwise;                  /* settle close pairs on their own add-on */
  uint64_t pair_min_cpg;              /* refuse a verdict under this many CpGs */
  int      weighted;                  /* weight each pattern by observed CpGs */
  uint64_t dm_top;                    /* per-binstring cap on measured CpGs */
  uint64_t pattern_cap;               /* max CpGs one pattern may claim; 0=off */
  double   weight_expo;               /* 0.5 = sqrt(n); 0.25 = sqrt(sqrt(n)) */
} adapt_opt_t;

/* ------------------------------------------------------------------ */
/* Pairwise adjudication: decide a confusable pair on ITS OWN evidence. */
/* ------------------------------------------------------------------ */
/* The defect this fixes is that the scope decision was made at round 0 on the
 * WEAKEST panel there is. A pattern must hold across every class in its set, so
 * inside a 33-class conjunction Monocyte and Macrophage are separated by 300
 * CpGs, of which a 2^16 query observes 0.7 -- a zero there is NNLS's active set
 * choosing arbitrarily between two collinear columns, not evidence of absence.
 * Built as its own 2-class set the same pair carries 11,935 CpGs, ~27 observed.
 *
 * The columns are the UNION of the working panel and a pair-specific add-on,
 * and the design keeps every class in scope. Both halves are load-bearing and
 * each was measured failing alone:
 *
 *   - add-on columns, pair-only design: the add-on's CpGs are selected to split
 *     a from b with NO constraint on the other classes, so in a mixture their
 *     methylation lands on exactly those CpGs and a 2-class design has nowhere
 *     to put it. The verdict then reflects the out-group, not the pair.
 *   - narrow columns, full design (--columns-only): the panel says nothing
 *     about out-of-scope classes, which stay in the design and absorb mass
 *     freely. Measured worse than both baselines (TVD 0.497 vs flat 0.465 at
 *     2^16, 0.405 vs 0.227 at 2^20).
 *
 * Union of columns plus full design is the combination that peels the
 * out-group off via the fit while giving the pair evidence proportionate to
 * how hard the call is. Reading a and b off the FULL solve also keeps them on
 * the mixture's scale, so a genuine minor component is not forced into a
 * two-way split against its own confuser.
 *
 * The add-on is TRANSIENT: it settles one question and is discarded, so the
 * working panel never accumulates pair columns.
 *
 * Returns 0 = both hold, 1 = `a` absent, 2 = `b` absent, 3 = neither holds.
 * Case 3 is the one most likely to be a false negative, so the caller decides. */
static int adjudicate_pair(const refmem_t *R, const query_t *Q,
                           const adapt_opt_t *o, const panel_t *base,
                           const uint32_t *sel, uint32_t n_sel,
                           uint32_t a, uint32_t b, double *pa, double *pb,
                           uint64_t *n_obs) {
  /* The add-on is the pair ALONE. Building it over the pair plus the classes
   * carrying mass was tried -- the idea being that a 2-class selection picks
   * CpGs with no constraint on the abundant classes, whose methylation then
   * lands on exactly those CpGs -- and it lost. Head to head on 200 mixtures
   * at 0.30,0.70 the 2-class add-on was ahead at every rung where pairwise
   * acts (TVD 0.105/0.043/0.023 vs 0.109/0.044/0.026 at 2^18/20/22) and tied
   * at 2^16, where the --pair-min-cpg floor makes both abstain. Widening the
   * conjunction costs more CpGs than the extra constraint is worth: the pair
   * carries 11,935 alone against 2,211-6,651 with 3-5 classes in the build.
   * The out-group is peeled off by the full design below instead, which is
   * where it belongs -- the fit knows the other classes' proportions, whereas
   * the selection can only demand they sit confidently on a side. */
  uint32_t aset[2] = { a, b };
  panel_t add;
  panel_build(R, Q, aset, 2, o->beta_thr, o->qlo, o->qhi, 0, o->dm_top, &add);

  /* Every pattern of a 2-class panel splits the pair by construction -- one
   * class is '0' and the other '1', or the CpG was not admitted at all. */
  uint64_t seen = 0, cpg = 0;
  for (uint32_t r = 0; r < add.n_pat; ++r) { seen += add.qn[r]; cpg += add.count[r]; }
  *n_obs = seen;

  /* Concatenate the two column sets. The union carries NO keys: they are bit
   * patterns over DIFFERENT class sets and so are not comparable across the two
   * panels, and panel_solve reads only counts/sums. Leaving key NULL and kw 0
   * says that, where allocating a zeroed key array would leave something that
   * looks addressable and would silently mis-answer group_of(). */
  panel_t U;
  memset(&U, 0, sizeof U);
  U.nall = R->n_class; U.ns = n_sel; U.kw = 0; U.key = NULL;
  U.n_pat = base->n_pat + add.n_pat;
  U.count = axc(U.n_pat, sizeof(uint64_t), "union counts");
  U.qsum  = axc(U.n_pat, sizeof(double), "union qsum");
  U.qn    = axc(U.n_pat, sizeof(uint64_t), "union qn");
  U.rsum  = axc((size_t)U.n_pat * U.nall, sizeof(double), "union rsum");
  for (uint32_t r = 0; r < base->n_pat; ++r) {
    U.count[r] = base->count[r]; U.qsum[r] = base->qsum[r]; U.qn[r] = base->qn[r];
    memcpy(U.rsum + (size_t)r * U.nall, base->rsum + (size_t)r * base->nall,
           U.nall * sizeof(double));
  }
  for (uint32_t r = 0; r < add.n_pat; ++r) {
    uint32_t d = base->n_pat + r;
    U.count[d] = add.count[r]; U.qsum[d] = add.qsum[r]; U.qn[d] = add.qn[r];
    memcpy(U.rsum + (size_t)d * U.nall, add.rsum + (size_t)r * add.nall,
           U.nall * sizeof(double));
  }
  double *x = axc(R->n_class, sizeof(double), "pair solve");
  panel_solve(&U, o->min_cov_pat, x, sel, n_sel, o->weighted, o->pattern_cap, o->weight_expo);
  *pa = 0; *pb = 0;
  for (uint32_t s = 0; s < n_sel; ++s) {
    if (sel[s] == a) *pa = x[s];
    if (sel[s] == b) *pb = x[s];
  }
  free(x); panel_free(&U); panel_free(&add);   /* base is the caller's */

  if (o->verbose)
    fprintf(stderr, "[methscope]   %s vs %s: %llu pair CpGs, %llu observed"
                    " -> %.3f / %.3f\n", R->name[a], R->name[b],
            (unsigned long long)cpg, (unsigned long long)seen, *pa, *pb);
  /* No discriminating CpG observed: undecidable here, and saying so beats
   * inventing a verdict from an empty add-on. */
  if (!seen) return 0;
  int za = (*pa <= o->drop_below), zb = (*pb <= o->drop_below);
  if (za && zb) return 3;
  if (za) return 1;
  if (zb) return 2;
  return 0;
}

/* ------------------------------------------------------------------ */
/* Confusable groups, from the round-0 panel's own binstrings.          */
/* ------------------------------------------------------------------ */
/* Two classes are grouped when the panel separates them by <= thresh CpGs.
 * NNLS hands out exact zeros from its KKT conditions and which of two
 * near-collinear columns gets zeroed is close to arbitrary -- Monocyte and
 * Macrophage are separated by 300 CpGs here, of which a 2^16 query observes
 * 0.7 -- so a single class's zero cannot be believed while a GROUP's can.
 * Counted per PATTERN (a few thousand) rather than per CpG, so this is
 * n_pat * ns^2/2 rather than n_cpg * ns^2/2. */
static void group_of(const panel_t *p, uint64_t thresh, uint32_t *grp) {
  const uint32_t ns = p->ns;
  uint64_t *seg = axc((size_t)ns * ns, sizeof(uint64_t), "segregating");
  for (uint32_t r = 0; r < p->n_pat; ++r) {
    const uint64_t *k = p->key + (size_t)r * p->kw;
    for (uint32_t a = 0; a < ns; ++a) {
      int ba = (k[a >> 6] >> (a & 63)) & 1;
      for (uint32_t b = a + 1; b < ns; ++b) {
        int bb = (k[b >> 6] >> (b & 63)) & 1;
        if (ba != bb) seg[(size_t)a * ns + b] += p->count[r];
      }
    }
  }
  for (uint32_t s = 0; s < ns; ++s) grp[s] = s;         /* union-find */
  for (uint32_t a = 0; a < ns; ++a)
    for (uint32_t b = a + 1; b < ns; ++b) {
      if (seg[(size_t)a * ns + b] > thresh) continue;
      uint32_t ra = a, rb = b;
      while (grp[ra] != ra) ra = grp[ra];
      while (grp[rb] != rb) rb = grp[rb];
      if (ra != rb) grp[ra > rb ? ra : rb] = ra < rb ? ra : rb;
    }
  for (uint32_t s = 0; s < ns; ++s) {
    uint32_t r = s;
    while (grp[r] != r) r = grp[r];
    grp[s] = r;
  }
  free(seg);
}

/* ------------------------------------------------------------------ */
/* One resident reference, every RECORD of the query. Each record gets its own
 * class set -- that is the point -- but the reference load is paid once, which
 * over a cohort is the difference between minutes and hours. */
/* ------------------------------------------------------------------ */
/* One record, start to finish. Pure apart from the (read-only) reference,  */
/* which is what lets records run on separate threads.                      */
/* ------------------------------------------------------------------ */

static void solve_record(const refmem_t *R, const query_t *Q,
                         const adapt_opt_t *o, double *full, uint8_t *scope) {
  /* A class printed as 0.000 is ambiguous on its own: it may have been DROPPED
   * by the rule, or kept in the design and zeroed by NNLS's active set. Those
   * are different failures -- the first is ours, the second is collinearity --
   * so --verbose names what each round removes. */
  const uint32_t ns = R->n_class;
  uint32_t *sel = axc(ns, sizeof(uint32_t), "class selection");
  for (uint32_t s = 0; s < ns; ++s) sel[s] = s;
  uint32_t n_sel = ns;

  double *x = axc(ns, sizeof(double), "proportions");
  uint32_t *grp = axc(ns, sizeof(uint32_t), "groups");

  panel_t P;
  panel_build(R, Q, sel, n_sel, o->beta_thr, o->qlo, o->qhi,
              o->columns_only, o->dm_top, &P);
  if (o->verbose)
    fprintf(stderr, "[methscope] round 0: %u classes, %u patterns\n",
            n_sel, P.n_pat);
  panel_solve(&P, o->min_cov_pat, x, NULL, 0, o->weighted, o->pattern_cap, o->weight_expo); /* round 0 */
  group_of(&P, o->group_thresh, grp);        /* groups come from ROUND 0 */
  panel_free(&P);
  for (uint32_t s = 0; s < ns; ++s) full[s] = x[s];

  for (int round = 1; round <= o->max_round; ++round) {
    /* Keep a group if ANY member is above the floor: a class zeroed only
     * because a collinear partner absorbed it is protected by that partner.
     * The drop is NOT reversible, which is where this loses when it loses --
     * on 200 known mixtures at 2^16 it discarded a component carrying >5% of
     * the mixture in 85% of samples. */
    uint32_t keep_n = 0;
    uint32_t *keep = axc(n_sel, sizeof(uint32_t), "keep");
    uint8_t *cut = axc(n_sel, 1, "pairwise verdicts");

    /* PAIRWISE ADJUDICATION. A group's members are exactly the classes the
     * working panel cannot separate, so a zero among them is not evidence.
     * Rather than protect the whole group (which keeps absent classes -- with
     * Monocyte truly present, its partners Macrophage and Dendritic.Cell held
     * 0.034 and 0.081 of mass they should not have had), settle each close pair
     * on its own add-on and cut the loser. Closest first, so the most damaging
     * confusion is resolved while both candidates are still in play. */
    if (o->pairwise) {
      /* The base panel does not depend on the pair, so build it ONCE per round
       * rather than once per pair -- with three confusable pairs that was
       * three redundant full-scope genome scans. */
      panel_t base;
      panel_build(R, Q, sel, n_sel, o->beta_thr, o->qlo, o->qhi,
                  o->columns_only, o->dm_top, &base);
      for (uint32_t i = 0; i < n_sel; ++i) {
        if (cut[i]) continue;
        for (uint32_t j = i + 1; j < n_sel; ++j) {
          if (cut[j] || grp[sel[i]] != grp[sel[j]]) continue;
          double pa = 0, pb = 0; uint64_t seen = 0;
          int v = adjudicate_pair(R, Q, o, &base, sel, n_sel,
                                  sel[i], sel[j], &pa, &pb, &seen);
          if (seen < o->pair_min_cpg) continue;   /* too little to decide on */
          if (v == 1) cut[i] = 1;
          else if (v == 2) cut[j] = 1;
          /* v == 3 (neither holds) is left to the group rule below: it is the
           * case most likely to be a false negative, and cutting both on one
           * under-powered comparison is how a real component disappears. */
          if (cut[i]) break;
        }
      }
      panel_free(&base);
    }

    for (uint32_t s = 0; s < n_sel; ++s) {
      if (cut[s]) continue;                    /* lost its pairwise contest */
      int any = 0;
      for (uint32_t t = 0; t < n_sel; ++t)
        if (!cut[t] && grp[sel[t]] == grp[sel[s]] &&
            (o->columns_only ? x[sel[t]] : x[t]) > o->drop_below) { any = 1; break; }
      if (any) keep[keep_n++] = sel[s];
    }
    free(cut);
    if (!keep_n) { free(keep); break; }          /* never solve on nothing */
    if (keep_n == n_sel) {                        /* fixed point: the usual exit */
      if (o->verbose)
        fprintf(stderr, "[methscope] round %d: class set unchanged, stopping\n",
                round);
      free(keep);
      break;
    }
    if (o->verbose) {
      fprintf(stderr, "[methscope] round %d drops:", round);
      for (uint32_t a = 0; a < n_sel; ++a) {
        int still = 0;
        for (uint32_t b = 0; b < keep_n; ++b)
          if (keep[b] == sel[a]) { still = 1; break; }
        if (!still) fprintf(stderr, " %s(%.3f)", R->name[sel[a]],
                            o->columns_only ? x[sel[a]] : x[a]);
      }
      fputc('\n', stderr);
    }
    memcpy(sel, keep, keep_n * sizeof(uint32_t));
    n_sel = keep_n;
    free(keep);

    panel_build(R, Q, sel, n_sel, o->beta_thr, o->qlo, o->qhi,
              o->columns_only, o->dm_top, &P);
    if (o->verbose)
      fprintf(stderr, "[methscope] round %d: %u classes, %u patterns\n",
              round, n_sel, P.n_pat);
    panel_solve(&P, o->min_cov_pat, x,
                o->columns_only ? NULL : sel, n_sel, o->weighted, o->pattern_cap, o->weight_expo);
    panel_free(&P);
    if (o->columns_only) {
      for (uint32_t s = 0; s < ns; ++s) full[s] = x[s];
    } else {
      for (uint32_t s = 0; s < ns; ++s) full[s] = 0;
      for (uint32_t s = 0; s < n_sel; ++s) full[sel[s]] = x[s];
    }
  }
  if (o->verbose) {
    fprintf(stderr, "[methscope] final set (%u):", n_sel);
    for (uint32_t a = 0; a < n_sel; ++a)
      fprintf(stderr, " %s=%.3f", R->name[sel[a]],
              o->columns_only ? x[sel[a]] : x[a]);
    fputc('\n', stderr);
  }
  /* Report the settled SCOPE. Under columns-only it is invisible in the output
   * (every class can be non-zero), and it is the thing whose quality decides
   * whether the whole scheme works, so it has to be measurable. */
  if (scope) {
    for (uint32_t s = 0; s < ns; ++s) scope[s] = 0;
    for (uint32_t a = 0; a < n_sel; ++a) scope[sel[a]] = 1;
  }
  free(sel); free(x); free(grp);
}

/* ------------------------------------------------------------------ */
/* Threading: over RECORDS, not within one.                             */
/* ------------------------------------------------------------------ */
/* Each record has its own class set and its own round count, so there is no
 * shared mutable state and no barrier -- and the 1.94 GB reference is read-only
 * and shared, so memory grows by ~64 MB per thread rather than linearly.
 * Workers claim record indices under one mutex and each opens its OWN cfile,
 * seeking by the .idx offset (the same idiom msfm_build uses). Results land in
 * a per-record slot and are written in index order at the end, so the output is
 * byte-identical whatever --nthreads is. */
typedef struct {
  const refmem_t  *R;
  const adapt_opt_t *opt;
  const char      *query;
  const int64_t   *off;
  uint32_t         n_rec, cursor;
  uint32_t         mincov;
  double          *result;            /* n_rec * n_class */
  uint8_t         *scope;             /* n_rec * n_class, or NULL */
  pthread_mutex_t  lock;
} ajob_t;

static void *adapt_worker(void *arg) {
  ajob_t *J = (ajob_t *)arg;
  cfile_t cf = open_cfile((char *)J->query);
  for (;;) {
    pthread_mutex_lock(&J->lock);
    uint32_t i = J->cursor < J->n_rec ? J->cursor++ : UINT32_MAX;
    pthread_mutex_unlock(&J->lock);
    if (i == UINT32_MAX) break;

    if (bgzf_seek(cf.fh, J->off[i], SEEK_SET) != 0)
      adie("cannot seek to query record", J->query);
    cdata_t qc = read_cdata1(&cf);
    if (!qc.n) adie("short read on query record", J->query);
    decompress_in_situ(&qc);
    if (qc.fmt != '3') adie("query must be format-3 (M/U) .cg", J->query);
    query_t Q; query_from_cdata(&qc, J->mincov, &Q);
    free_cdata(&qc);
    if (Q.n_cpg != J->R->n_cpg)
      adie("query and reference disagree on CpG count (different row space?)",
           J->query);
    solve_record(J->R, &Q, J->opt, J->result + (size_t)i * J->R->n_class,
                 J->scope ? J->scope + (size_t)i * J->R->n_class : NULL);
    free(Q.beta);
  }
  bgzf_close(cf.fh);
  return NULL;
}

int ms_deconv_adaptive(const char *ref_cg, const char *query_cg,
                       const char *out_path, int max_round,
                       uint64_t group_thresh, double drop_below,
                       double qlo, double qhi, double beta_thr,
                       uint32_t mincov, uint32_t min_cov_pat, int no_header,
                       int nthreads, int columns_only, const char *scope_out,
                       int pairwise, unsigned long pair_min_cpg,
                       int weighted, uint64_t pattern_cap,
                       double weight_expo, uint64_t dm_top) {
  refmem_t R; refmem_load(ref_cg, mincov, &R);
  const uint32_t ns = R.n_class;

  uint32_t n_qname = 0;
  int64_t *qoff = NULL;
  char **qname = query_names(query_cg, &n_qname, &qoff);

  adapt_opt_t opt = { max_round, group_thresh, drop_below, qlo, qhi, beta_thr,
                      min_cov_pat, 1, columns_only, pairwise,
                      (uint64_t)pair_min_cpg, weighted, dm_top, pattern_cap,
                      weight_expo };
  FILE *sf = NULL;
  if (scope_out) {
    sf = fopen(scope_out, "w");
    if (!sf) adie("cannot open --scope-out", scope_out);
    fputs("cell", sf);
    for (uint32_t s = 0; s < ns; ++s) fprintf(sf, "\t%s", R.name[s]);
    fputc('\n', sf);
  }

  FILE *out = (out_path && strcmp(out_path, "-")) ? fopen(out_path, "w")
                                                  : stdout;
  if (!out) adie("cannot open output", out_path);
  if (!no_header) {
    fputs("cell", out);
    for (uint32_t s = 0; s < ns; ++s) fprintf(out, "\t%s", R.name[s]);
    fputc('\n', out);
  }

  /* Threading needs the .idx to seek by: without one the records can only be
   * read in order, so fall back rather than pretend. */
  if (nthreads > 1 && (!qoff || !n_qname)) {
    fprintf(stderr, "[methscope] deconv --adaptive: no <query>.idx, so records "
                    "can only be streamed in order; running single-threaded\n");
    nthreads = 1;
  }

  if (nthreads > 1) {
    ajob_t J;
    J.R = &R; J.opt = &opt; J.query = query_cg; J.off = qoff;
    J.n_rec = n_qname; J.cursor = 0; J.mincov = mincov;
    J.result = axc((size_t)n_qname * ns, sizeof(double), "results");
    J.scope = sf ? axc((size_t)n_qname * ns, 1, "scopes") : NULL;
    opt.verbose = 0;             /* interleaved round lines would be unreadable */
    pthread_mutex_init(&J.lock, NULL);
    pthread_t *tid = axc(nthreads, sizeof(pthread_t), "threads");
    for (int t = 0; t < nthreads; ++t)
      if (pthread_create(&tid[t], NULL, adapt_worker, &J))
        adie("cannot create thread", NULL);
    for (int t = 0; t < nthreads; ++t) pthread_join(tid[t], NULL);
    pthread_mutex_destroy(&J.lock);
    free(tid);
    for (uint32_t i = 0; i < n_qname; ++i) {      /* written in RECORD order */
      fputs(qname[i], out);
      for (uint32_t s = 0; s < ns; ++s)
        fprintf(out, "\t%.6f", J.result[(size_t)i * ns + s]);
      fputc('\n', out);
      if (sf) {
        fputs(qname[i], sf);
        for (uint32_t s = 0; s < ns; ++s)
          fprintf(sf, "\t%u", J.scope[(size_t)i * ns + s]);
        fputc('\n', sf);
      }
    }
    free(J.result); free(J.scope);
  } else {
    double *full = axc(ns, sizeof(double), "full proportions");
    uint8_t *sc = sf ? axc(ns, 1, "scope") : NULL;
    cfile_t qf = open_cfile((char *)query_cg);
    for (uint32_t qi = 0; ; ++qi) {
      cdata_t qc = read_cdata1(&qf);
      if (!qc.n) { free_cdata(&qc); break; }
      decompress_in_situ(&qc);
      if (qc.fmt != '3') adie("query must be format-3 (M/U) .cg", query_cg);
      query_t Q; query_from_cdata(&qc, mincov, &Q);
      free_cdata(&qc);
      if (Q.n_cpg != R.n_cpg)
        adie("query and reference disagree on CpG count (different row space?)",
             query_cg);
      solve_record(&R, &Q, &opt, full, sc);
      if (qi < n_qname) fputs(qname[qi], out);
      else              fprintf(out, "%u", qi + 1);
      for (uint32_t s = 0; s < ns; ++s) fprintf(out, "\t%.6f", full[s]);
      fputc('\n', out);
      fflush(out);
      if (sf) {
        if (qi < n_qname) fputs(qname[qi], sf); else fprintf(sf, "%u", qi + 1);
        for (uint32_t s = 0; s < ns; ++s) fprintf(sf, "\t%u", sc[s]);
        fputc('\n', sf);
      }
      free(Q.beta);
    }
    bgzf_close(qf.fh);
    free(full); free(sc);
  }

  if (out != stdout) fclose(out);
  if (sf) fclose(sf);
  for (uint32_t q = 0; q < n_qname; ++q) free(qname[q]);
  free(qname); free(qoff);
  for (uint32_t s = 0; s < ns; ++s) free(R.name[s]);
  free(R.name); free(R.beta);
  return 0;
}
