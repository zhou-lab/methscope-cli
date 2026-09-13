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
  /* 0.30/0.70, not the old 0.25/0.60: ONE admission rule across the classifier
   * and the deconvolver, so a CpG is admissible or not for the reference as a
   * whole rather than per consumer. The tree command list already passed
   * 0.30,0.70 explicitly, so this only moves the default to meet it. On the
   * 200-mixture Tier-1 set the two filters are a wash for deconvolution --
   * 0.30/0.70 wins at 2^18 (0.105 vs 0.114 TVD) and loses at 2^20 (0.043 vs
   * 0.034), rung-averaged 0.101 vs 0.098 -- which is far too small a gap to
   * justify the two halves of the pipeline disagreeing about what a
   * segregating CpG is. */
  o->qfilter_lo = 0.30f; o->qfilter_hi = 0.70f;
  o->min_sbeta_gap = 0.0f;           /* 0 = band mode */
  /* 20000, not the old 1000: the budget is per BINSTRING, and a 2-class
   * satellite holds two of them, so 1000 capped a focused set at 2,000 CpGs --
   * far below what the same pair carries once its filter is a 2-way rather
   * than a 41-way conjunction. Every set built since has passed 20000. */
  o->delta_mean_top = 20000;
  /* a=3: a unanimous site needs depth >= 4 to clear the 0.70 band leg, one
   * dissenting read pushes that to ~8, and a depth-26 site at raw 0.69
   * shrinks to 0.66 and fails -- quality-first selection, measured on the
   * fold-0 stringency arms (a=3 band: native macro 0.9353 with the specific
   * confuser pairs at their best). Set 0 for the old unshrunk behaviour. */
  o->shrink_pseudocnt = 3.0f;
  o->feature_mindepth = 0;         /* 0 = follow --call-mindepth */
  o->max_lowdepth_frac = 0.0f;
  o->inc_all0 = 0; o->inc_all1 = 0;
  o->quiet = 0;
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

/* qsort comparator: CpG indices, descending by whichever statistic ranks them
 * by delta_mean, the mean beta of the classes the pattern calls 1 minus that
 * of the ones it calls 0. */
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
  /* The q-filter is a RULE, not merely a gate on the top-N leg. It used to be
   * the latter: with no cap every leg was inactive and the selector returned
   * NULL -- no selection -- so `--call-band 0.1,0.9` kept all 1,214,550 CpGs of a
   * 2-class node instead of its 21,224. It now has an uncapped form of its own
   * (--delta-mean-top 0), so no combination of options means "select nothing". */

  /* The depth a class needs for its beta to count as evidence here. Defaults
   * to the call depth: a class the binstring could not even call is not
   * evidence about anything. */
  const uint32_t mindepth = o->feature_mindepth ? o->feature_mindepth : mincov;

  float *min1 = xc(n_cpg, sizeof(float), "min1");
  float *max0 = xc(n_cpg, sizeof(float), "max0");
  float *sum1 = xc(n_cpg, sizeof(float), "sum1");
  float *sum0 = xc(n_cpg, sizeof(float), "sum0");
  uint8_t *n1 = xc(n_cpg, 1, "n1"), *n0 = xc(n_cpg, 1, "n0");
  uint8_t *nlow = xc(n_cpg, 1, "nlow");        /* classes below mindepth */
  for (uint64_t i = 0; i < n_cpg; ++i) { min1[i] = 2.0f; max0[i] = -1.0f; }

  /* One streaming pass. For each class we know, per CpG, whether its binstring
   * calls that class 1 or 0, so the expected-high and expected-low groups can
   * be accumulated without ever holding all betas. */
  const float pc = o->shrink_pseudocnt;
  cfile_t cf = open_cfile((char *)ref);
  for (uint32_t k = 0; k < ns; ++k) {
    seek_record(&cf, rec_off, k);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) { free_cdata(&c); break; }
    decompress_in_situ(&c);
    for (uint64_t i = 0; i < n_cpg; ++i) {
      uint32_t r = memb[i];
      if (r == MRMP_PNA_MEMBERSHIP) continue;
      uint64_t mu = f3_get_mu(&c, i);
      uint64_t cov = mu ? MU2cov(mu) : 0;
      /* Low-depth class: no usable measurement, so it is not tested -- an
       * untested class can neither pass nor fail the band. How many such
       * classes a feature may carry is --max-lowdepth-frac's call, below.
       * (An earlier version cleared a depth-floor veto here unconditionally,
       * which dropped every CpG with one uncovered class before the tolerance
       * was consulted and cost 13.4% of the mouse genome; 20260912 entry.) */
      if (cov < mindepth) { ++nlow[i]; continue; }
      /* A class exactly at the call threshold is NOT skipped, though its
       * binstring digit was imputed. It has reads: the measurement says
       * "intermediate", which is a real answer to the band's question -- does
       * every 1-class look methylated and every 0-class unmethylated -- and
       * the answer is no, whichever way the fill went. Under any band that
       * straddles the threshold it fails as a filled 0 and as a filled 1
       * alike, so imputation does not decide the kept set. Only a class with
       * NO measurement is skipped (above), and how many of those a CpG may
       * have is --max-lowdepth-frac's call. */
      float b = (float)MU2beta(mu);
      /* The RANK gets a shrunk beta, the q-filter keeps the raw one. Without
       * shrinkage delta_mean is MAXIMIZED by depth-1 CpGs: one read gives beta
       * exactly 0 or 1, so a 1-vs-1 site scores the theoretical maximum of 1.0
       * and outranks a real difference measured over 200 cells scoring 0.85.
       * The selection then fills up on the noisiest sites -- measured, the
       * Tmem CD4/CD8 leaf's 40,000 CpGs sat at mean reference depth 1.3 with
       * 97.9% backed by <= 3 cells, and its feature scored AUC 1.0000 on the
       * cells that built the reference and 0.5002 on held-out ones.
       * (M+a)/(M+U+2a) demotes them without discarding anything: at a=1 the
       * 1-vs-1 site scores 0.667-0.333 = 0.333 while 200-cell evidence keeps
       * ~0.99. Set --shrink-pseudocnt 0 for the old, unshrunk ranking. */
      float bs = pc > 0.0f
               ? (float)(((double)(mu >> 32) + pc) / ((double)cov + 2.0 * pc))
               : b;
      (void)b;
      /* The BAND tests the shrunk beta too. On the raw beta a depth-1 class
       * reads exactly 0 or 1 -- the most extreme value possible -- so it
       * passed the band MORE easily than a well-measured class and the
       * admitted pool filled with unfalsifiable calls. Shrunk, a lone read
       * lands at 0.333/0.667 and can never clear a 0.30/0.70 band; unanimous
       * depth-2 can, and a site with dissent needs ~depth 5. A soft,
       * self-scaling depth floor instead of --feature-mindepth's hard veto. */
      if (binstr[r][k] == '1') {
        if (bs < min1[i]) min1[i] = bs;
        sum1[i] += bs; ++n1[i];
      } else if (binstr[r][k] == '0') {
        if (bs > max0[i]) max0[i] = bs;
        sum0[i] += bs; ++n0[i];
      }
    }
    free_cdata(&c);
  }
  bgzf_close(cf.fh);

  /* Per-CpG verdicts. A CpG with an empty side carries no contrast, so it can
   * pass neither leg however extreme the other side looks. */
  const uint32_t low_allow = (uint32_t)(o->max_lowdepth_frac * (float)ns);
  uint8_t *keep = xc(n_cpg, 1, "keep");
  uint8_t *qok = xc(n_cpg, 1, "qok");
  float *rank = xc(n_cpg, sizeof(float), "rank statistic");
  for (uint64_t i = 0; i < n_cpg; ++i) {
    /* Same relaxation as mrmp-build's --call-band: an empty side is normally no
     * contrast and therefore skipped, but --include-all-0/-1 ask for exactly
     * those, so the gate must yield to the flag rather than silently undo it. */
    if (memb[i] == MRMP_PNA_MEMBERSHIP) continue;
    if (!n1[i] && !o->inc_all0) continue;
    if (!n0[i] && !o->inc_all1) continue;
    if ((uint32_t)nlow[i] > low_allow) continue;
    rank[i] = sum1[i] / n1[i] - sum0[i] / n0[i];
    if (o->min_sbeta_gap > 0.0f) {
      if (min1[i] - max0[i] >= o->min_sbeta_gap) qok[i] = 1;
    } else if (max0[i] <= o->qfilter_lo && min1[i] >= o->qfilter_hi) qok[i] = 1;
  }
  free(min1); free(max0); free(sum1); free(sum0);
  free(n1); free(n0); free(nlow);

  /* Floor leg: per binstring, the top delta_mean_top among q-filter passers.
   * Counting sort by pattern rank, so this is O(n_cpg) plus a per-pattern sort
   * rather than one global sort of 21.9M keys. */
  const uint32_t top = o->delta_mean_top;
  uint64_t n_floor = 0;
  {
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
      uint64_t take = (!top || m < top) ? m : top;
      if (take < m) qsort(v, m, sizeof(uint32_t), by_rank_desc);
      for (uint64_t t = 0; t < take; ++t)
        if (!keep[v[t]]) { keep[v[t]] = 1; ++n_floor; }
    }
    free(cnt); free(off); free(idx); free(fill);
  }
  free(qok); free(rank);

  uint64_t tot = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) tot += keep[i];
  if (o->quiet) { /* caller reports */ }
  else
    fprintf(stderr, "  select: %" PRIu64 " CpGs kept (%" PRIu64 " passed the "
            "q-filter, top-%u per binstring)\n", tot, n_floor,
            o->delta_mean_top);
  if (n_kept) *n_kept = tot;
  return keep;
}
