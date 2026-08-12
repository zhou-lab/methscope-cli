// SPDX-License-Identifier: AGPL-3.0-or-later
/* Per-binstring CpG selection. See mrmp_select.h for what the rule is and why.
 *
 * Memory: the per-CpG accumulators are kept as separate arrays rather than a
 * struct, so no padding is spent -- 21 bytes per CpG, ~460 MB at hg38/mm10
 * scale. A struct-of-arrays also lets the hot streaming loop touch only the
 * three arrays it updates for a given class. */
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "methscope.h"
#include "mrmp.h"
#include "mrmp_select.h"
#include "cfile.h"
#include "cdata.h"

void ms_select_defaults(ms_select_opt_t *o) {
  o->qfilter_lo = 0.25f; o->qfilter_hi = 0.60f;
  o->strict_lo  = 0.10f; o->strict_hi  = 0.80f;
  o->delta_mean_top = 1000;
  o->min_cg_depth = 0;
  o->max_frac_na = 0.0f;
  o->depth_floor_frac = 0.0f;      /* off; satellites turn it on */
  o->depth_floor_cap = 20;
  o->inc_all0 = 0; o->inc_all1 = 0;
  o->quiet = 0;
  o->p01_on = 1;                   /* the default rule; legacy flags switch off */
  o->p01_top = 0;                  /* no cap: --p01-min alone selects */
  o->p01_min = 0.60f;
}

static void *xc(size_t n, size_t sz, const char *what) {
  void *p = calloc(n ? n : 1, sz);
  if (!p) { fprintf(stderr, "[methscope] out of memory: %s\n", what); exit(1); }
  return p;
}

/* Seek to class k's record when the caller gave per-class offsets. Fatal on a
 * bad offset, because reading the wrong record would silently score one class
 * against another's betas. */
static void seek_record(cfile_t *cf, const int64_t *rec_off, uint32_t k) {
  if (!rec_off) return;
  if (bgzf_seek(cf->fh, rec_off[k], SEEK_SET) != 0) {
    fprintf(stderr, "[methscope] mrmp_select: cannot seek to record %u\n", k);
    exit(1);
  }
}

/* Per-class genome-wide mean depth, over CpGs the class actually covers.
 * Needed only for the RELATIVE floor: an absolute floor cannot serve both ends
 * of the class range -- 10 deletes a depth-5 class entirely while never binding
 * on a depth-112 one. */
static double *class_mean_depth(const char *ref, uint32_t ns, uint32_t mincov,
                                uint64_t n_cpg, const int64_t *rec_off) {
  double *mean = xc(ns, sizeof(double), "class mean depth");
  cfile_t cf = open_cfile((char *)ref);
  for (uint32_t k = 0; k < ns; ++k) {
    seek_record(&cf, rec_off, k);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) { free_cdata(&c); break; }
    decompress_in_situ(&c);
    uint64_t tot = 0, seen = 0;
    for (uint64_t i = 0; i < n_cpg; ++i) {
      uint64_t mu = f3_get_mu(&c, i);
      if (!mu) continue;
      uint64_t cov = MU2cov(mu);
      if (cov < mincov) continue;
      tot += cov; ++seen;
    }
    mean[k] = seen ? (double)tot / (double)seen : 0.0;
    free_cdata(&c);
  }
  bgzf_close(cf.fh);
  return mean;
}

/* qsort comparator: CpG indices, descending by whichever statistic ranks them
 * -- delta_mean by default, P(01) under --p01-top. */
static const float *g_rank;
static int by_rank_desc(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  float dx = g_rank[x], dy = g_rank[y];
  if (dx > dy) return -1;
  if (dx < dy) return 1;
  return (x < y) ? -1 : (x > y);        /* stable, so runs reproduce */
}

uint8_t *ms_mrmp_select(const char *ref, uint32_t ns, uint32_t mincov,
                        uint64_t n_cpg, const uint32_t *memb,
                        uint64_t n_cand, const char *const *binstr,
                        const ms_select_opt_t *o, uint64_t *n_kept,
                        const int64_t *rec_off) {
  /* --p01-top replaces the whole rule, so the strict union leg goes with it:
   * the point of P(01) is one number deciding both admission and rank. */
  const int use_p01 = o->p01_on;
  const int use_strict =
      !use_p01 && (o->strict_lo >= 0.0f && o->strict_hi >= 0.0f);
  /* The q-filter is a RULE, not merely a gate on the top-N leg. It used to be
   * the latter: with the strict leg off and no cap, every leg was inactive and
   * the selector returned NULL -- no selection -- so `--qfilter 0.1,0.9` kept
   * all 1,214,550 CpGs of a 2-class node instead of its 21,224. Only P(01) had
   * an uncapped form. Every rule now has one, so there is no longer a
   * combination of options that means "select nothing"; the disable check that
   * used to sit here could only fire for the case it got wrong. */

  /* per-class relative-depth targets: min(frac * own mean, cap) */
  double *target = NULL;
  if (o->depth_floor_frac > 0.0f) {
    double *mean = class_mean_depth(ref, ns, mincov, n_cpg, rec_off);
    target = xc(ns, sizeof(double), "depth targets");
    for (uint32_t k = 0; k < ns; ++k) {
      double t = o->depth_floor_frac * mean[k];
      if (t > (double)o->depth_floor_cap) t = (double)o->depth_floor_cap;
      target[k] = t;
    }
    free(mean);
  }

  float *min1 = xc(n_cpg, sizeof(float), "min1");
  float *max0 = xc(n_cpg, sizeof(float), "max0");
  float *sum1 = xc(n_cpg, sizeof(float), "sum1");
  float *sum0 = xc(n_cpg, sizeof(float), "sum0");
  uint8_t *n1 = xc(n_cpg, 1, "n1"), *n0 = xc(n_cpg, 1, "n0");
  uint8_t *npres = xc(n_cpg, 1, "npres");
  uint8_t *floor_ok = xc(n_cpg, 1, "floor_ok");
  uint16_t *mincv = xc(n_cpg, sizeof(uint16_t), "min coverage");
  /* Worst class on each side, as a Jeffreys-shrunk predictive probability.
   * Allocated only under --p01-top: two more floats per CpG is ~175 MB at
   * mm10 scale, which the delta path has no use for. */
  float *w1 = use_p01 ? xc(n_cpg, sizeof(float), "worst P(1)") : NULL;
  float *w0 = use_p01 ? xc(n_cpg, sizeof(float), "worst P(0)") : NULL;
  for (uint64_t i = 0; i < n_cpg; ++i) {
    min1[i] = 2.0f; max0[i] = -1.0f; mincv[i] = 0xFFFF; floor_ok[i] = 1;
    if (use_p01) { w1[i] = 2.0f; w0[i] = 2.0f; }
  }

  /* One streaming pass. For each class we know, per CpG, whether its binstring
   * calls that class 1 or 0, so the expected-high and expected-low groups can
   * be accumulated without ever holding all betas. */
  cfile_t cf = open_cfile((char *)ref);
  for (uint32_t k = 0; k < ns; ++k) {
    seek_record(&cf, rec_off, k);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) { free_cdata(&c); break; }
    decompress_in_situ(&c);
    const double tk = target ? target[k] : 0.0;
    for (uint64_t i = 0; i < n_cpg; ++i) {
      uint32_t r = memb[i];
      if (r == MRMP_PNA_MEMBERSHIP) continue;
      uint64_t mu = f3_get_mu(&c, i);
      uint64_t cov = mu ? MU2cov(mu) : 0;
      if (!mu || cov < mincov) { floor_ok[i] = 0; continue; }   /* absent */
      ++npres[i];
      if (cov < mincv[i]) mincv[i] = (uint16_t)(cov > 0xFFFF ? 0xFFFF : cov);
      if (target && (double)cov < tk) floor_ok[i] = 0;
      float b = (float)MU2beta(mu);
      /* P(a future single-cell draw from class k reads 1), Jeffreys-shrunk.
       * At depth 1 per cell, cov is the number of cells covering this CpG and
       * mu>>32 the number of them reading 1, so this is a shrunk cell count. */
      const float p1k =
          use_p01 ? (float)(((double)(mu >> 32) + 0.5) / ((double)cov + 1.0))
                  : 0.0f;
      if (binstr[r][k] == '1') {
        if (b < min1[i]) min1[i] = b;
        sum1[i] += b; ++n1[i];
        if (use_p01 && p1k < w1[i]) w1[i] = p1k;
      } else if (binstr[r][k] == '0') {
        if (b > max0[i]) max0[i] = b;
        sum0[i] += b; ++n0[i];
        if (use_p01 && 1.0f - p1k < w0[i]) w0[i] = 1.0f - p1k;
      }
    }
    free_cdata(&c);
  }
  bgzf_close(cf.fh);
  free(target);

  /* Per-CpG verdicts. A CpG with an empty side carries no contrast, so it can
   * pass neither leg however extreme the other side looks. */
  const uint32_t na_allow = (uint32_t)(o->max_frac_na * (float)ns);
  uint8_t *keep = xc(n_cpg, 1, "keep");
  uint8_t *qok = xc(n_cpg, 1, "qok");
  float *rank = xc(n_cpg, sizeof(float), "rank statistic");
  double p01_sum = 0.0;
  uint64_t n_strict = 0, n_admit = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) {
    /* Same relaxation as mrmp-build's --qfilter: an empty side is normally no
     * contrast and therefore skipped, but --include-all-0/-1 ask for exactly
     * those, so the gate must yield to the flag rather than silently undo it. */
    if (memb[i] == MRMP_PNA_MEMBERSHIP) continue;
    if (!n1[i] && !o->inc_all0) continue;
    if (!n0[i] && !o->inc_all1) continue;
    if ((uint32_t)(ns - npres[i]) > na_allow) continue;
    if (o->min_cg_depth && mincv[i] < o->min_cg_depth) continue;
    if (!floor_ok[i]) continue;
    if (use_p01) {
      rank[i] = w1[i] * w0[i];
      if (rank[i] >= o->p01_min) { qok[i] = 1; ++n_admit; p01_sum += rank[i]; }
      continue;                       /* P(01) is the gate and the rank both */
    }
    rank[i] = sum1[i] / n1[i] - sum0[i] / n0[i];
    if (max0[i] <= o->qfilter_lo && min1[i] >= o->qfilter_hi) qok[i] = 1;
    if (use_strict && max0[i] <= o->strict_lo && min1[i] >= o->strict_hi) {
      keep[i] = 1; ++n_strict;                     /* self-sizing leg */
    }
  }
  free(min1); free(max0); free(sum1); free(sum0);
  free(n1); free(n0); free(npres); free(floor_ok); free(mincv);
  free(w1); free(w0);

  /* Floor leg: per binstring, the top delta_mean_top among q-filter passers.
   * Counting sort by pattern rank, so this is O(n_cpg) plus a per-pattern sort
   * rather than one global sort of 21.9M keys. */
  const uint32_t top = use_p01 ? o->p01_top : o->delta_mean_top;
  uint64_t n_floor = 0;
  if (!top) {
    /* Uncapped: admission IS the selection, so there is nothing left to rank.
     * True of P(01) by design (P(01) >= --p01-min is the rule) and of an
     * uncapped q-filter for the same reason. */
    for (uint64_t i = 0; i < n_cpg; ++i)
      if (qok[i]) { keep[i] = 1; ++n_floor; }
  } else if (top) {
    uint64_t *cnt = xc(n_cand + 1, sizeof(uint64_t), "per-pattern counts");
    for (uint64_t i = 0; i < n_cpg; ++i)
      if (qok[i]) ++cnt[memb[i]];
    uint64_t *off = xc(n_cand + 1, sizeof(uint64_t), "pattern offsets");
    uint64_t run = 0;
    for (uint64_t r = 0; r < n_cand; ++r) { off[r] = run; run += cnt[r]; }
    off[n_cand] = run;
    uint32_t *idx = xc(run, sizeof(uint32_t), "pattern member lists");
    uint64_t *fill = xc(n_cand, sizeof(uint64_t), "fill cursor");
    for (uint64_t i = 0; i < n_cpg; ++i)
      if (qok[i]) idx[off[memb[i]] + fill[memb[i]]++] = (uint32_t)i;
    g_rank = rank;
    for (uint64_t r = 0; r < n_cand; ++r) {
      uint64_t m = cnt[r];
      if (!m) continue;
      uint32_t *v = idx + off[r];
      if (m > top) qsort(v, m, sizeof(uint32_t), by_rank_desc);
      uint64_t take = m < top ? m : top;
      for (uint64_t t = 0; t < take; ++t)
        if (!keep[v[t]]) { keep[v[t]] = 1; ++n_floor; }
    }
    free(cnt); free(off); free(idx); free(fill);
  }
  free(qok); free(rank);

  uint64_t tot = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) tot += keep[i];
  if (o->quiet) { /* caller reports */ }
  else if (use_p01) {
    char cap[48] = "uncapped";
    if (o->p01_top) snprintf(cap, sizeof(cap), "capped at %u/binstring",
                             o->p01_top);
    fprintf(stderr, "  select: %" PRIu64 " CpGs kept by P(01) >= %.3f, %s "
            "(%" PRIu64 " admitted, mean P(01) %.3f)\n",
            tot, (double)o->p01_min, cap, n_admit,
            n_admit ? p01_sum / (double)n_admit : 0.0);
  }
  else
    fprintf(stderr, "  select: %" PRIu64 " CpGs kept (%" PRIu64 " strict, %"
            PRIu64 " added by the top-%u floor)\n",
            tot, n_strict, n_floor, o->delta_mean_top);
  if (n_kept) *n_kept = tot;
  return keep;
}
