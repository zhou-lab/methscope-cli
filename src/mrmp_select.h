// SPDX-License-Identifier: AGPL-3.0-or-later
/* Per-CpG selection for one MRMP set, applied PER BINSTRING.
 *
 * This is the cut that used to live outside the binary, as a q-filter in shell
 * plus a top-K-by-depth pass in Python. It is one cut, not two: binstrings are
 * built first over every CpG, then each binstring independently decides which
 * of its CpGs to keep.
 *
 * THE DEFAULT RULE IS P(01) >= 0.60, documented in its own section below. The
 * q-filter rule described first is the legacy one, still selectable with
 * --qfilter / --delta-mean-top, and kept because it is what
 * every artifact before 2026-08-09 was built with.
 *
 *     legacy: keep = q_filter_strict  UNION  (q_filter AND top-N by delta_mean)
 *
 * The two legs do different jobs. STRICT self-sizes: a set whose classes have
 * genuinely clean positions contributes all of them, however many that is. The
 * top-N leg is a FLOOR, so a hard set is never starved. Both are needed because
 * the strict yield is violently depth-dependent -- the same class pair gave
 * 6,453 CpGs at reference depth 7.9 and 204 at 4.54, a 30x collapse from a 2.4x
 * depth change -- so a strict-only rule would swing between folds, and a
 * top-N-only rule would discard clean CpGs past the budget.
 *
 * Measured on ANP/ASC, where the two patterns that oppose one class against two
 * neighbours were EMPTY under strict alone: the union takes them from 0 CpGs to
 * 1,000 each at +0.43 held-out margin.
 *
 * Ranking is by delta_mean (mean gap between the expected-1 and expected-0
 * groups), not delta_beta (the worst-case margin), because the q-filter already
 * bounds the worst case and ranking need not re-police it. The two scored
 * within 0.007 of each other on held-out margin while picking 17-28% different
 * CpGs, and mean degrades more gracefully as a set grows. They are identical
 * for 2 classes and diverge only at 3+.
 *
 * Ranking by delta beats the old ranking by depth: +0.586 against +0.446 at the
 * same 10,000-CpG budget. Depth and contrast are near-orthogonal, so depth
 * ranking took an unbiased sample of whatever the filter admitted rather than
 * concentrating the discriminating positions.
 *
 * ---------------------------------------------------------------------------
 * P(01): ONE STATISTIC INSTEAD OF A GATE PLUS A RANK
 *
 * The default, and what any of the legacy flags switches away from:
 *
 *     keep = every CpG with  P(01) >= --p01-min   (default 0.60)
 *            (optionally capped at the --p01-top N best PER BINSTRING)
 *
 * The cap is off by default and is a budget, not part of the rule: a threshold
 * on a calibrated probability already says which CpGs are usable, so imposing
 * top-N on top of it would silently discard CpGs that met the bar. Reach for it
 * only to bound artifact size.
 *
 * where P(01) is the probability that a future single-cell draw reads 0 in the
 * expected-0 group AND 1 in the expected-1 group. At the depth single-cell data
 * actually has -- one read per cell per CpG -- a class's beta at a CpG IS the
 * fraction of its cells reading 1, so that probability is a shrunk count. Under
 * a Jeffreys Beta(1/2,1/2) prior, for class k with M methylated and U
 * unmethylated cells,
 *
 *     P(1 | k) = (M + 1/2) / (M + U + 1)        P(0 | k) = 1 - P(1 | k)
 *
 * and, keeping the worst class on each side exactly as the q-filter's min1/max0
 * already do,
 *
 *     P(01) = min over 1-classes P(1|k)  *  min over 0-classes P(0|k)
 *
 * This folds together what the old rule tuned separately. DEPTH enters through
 * the prior: a CpG seen in 3 cells is pulled toward 1/2 and cannot outrank one
 * seen in 300. DELTA enters through the product, which is largest when the two
 * sides sit at opposite ends. And LO,HI disappear, because "is this CpG usable"
 * and "how good is it" become one number rather than a gate feeding a rank.
 *
 * What it does NOT replace: the coverage gates above it -- max_frac_na,
 * min_cg_depth and the relative depth floor -- still run first and are
 * unchanged. The prior makes them less load-bearing, but the benchmark below
 * was measured with them on, so they stay on.
 *
 * It is also the statistic matched to what the classifier does. Features are
 * binarised at a flat 0.5, so what decides whether a CpG helps is its distance
 * from 0.5, not the gap between the classes -- delta_mean ranks by the
 * CONTINUOUS-feature objective while the model trains on the binary one. P(01)
 * ranks by the binary objective directly, and being a product it prefers a
 * balanced split (0.25/0.75) over a lopsided one (0.45/0.95) at equal delta,
 * which is what a symmetric 0.5 threshold wants.
 *
 * End to end on the mouse cohort, binary features, 400 cells/class train and
 * 50/class test, at a fixed 1,000-pattern budget -- native-coverage accuracy
 * 0.9533 / 0.9561 / 0.9533 / 0.9544 at P(01) >= 0.50 / 0.60 / 0.70 / 0.75,
 * against 0.9516 for the legacy rule. Paired McNemar puts every one of those
 * inside noise (p = 0.28 to 0.75), so 0.60 is the default because it topped the
 * sweep, not because it is separable from its neighbours. The case for the rule
 * is that a 4x swing in CpG count moves accuracy by 0.0028: one threshold that
 * barely needs tuning, in place of two q-filters, a rank budget and two depth
 * floors. It also carries 1.3-2.5x more CpGs at the same pattern budget
 * (763k -> 1.68M covered rows at 0.60).
 *
 * Measured on the IT-L5/IT-L6 neighbour pair, the hardest in the mouse cohort
 * (distance 0.101): the 1,000 CpGs the delta rule selected have mean P(0)=0.836
 * and P(1)=0.791, and Spearman rho between P(01) and delta_mean among them is
 * 0.987 -- so among survivors the two agree almost perfectly, and the change
 * has to earn its keep at ADMISSION (shallow CpGs with a large apparent delta
 * that shrinkage demotes) rather than by reordering winners.
 *
 * Do not expect it to close the binary-vs-continuous gap. On that same pair a
 * cell observes ~70 of the 1,000 CpGs, which under independence would put its
 * pattern beta 8-9 SE clear of 0.5; measured across 400 training cells the
 * per-cell SD is 0.143 (IT-L5) and 0.188 (IT-L6) against an independent 0.046 --
 * a 10-15x design effect, so the pattern carries ~5-7 independent observations
 * rather than 70. Per-CpG quality is not what limits that pair; CpG-CpG
 * correlation within a cell is, and no per-CpG statistic can see it. */
#ifndef MS_MRMP_SELECT_H
#define MS_MRMP_SELECT_H

#include <stdint.h>

typedef struct {
  float    qfilter_lo, qfilter_hi;   /* expected-0 ceiling, expected-1 floor */
  uint32_t delta_mean_top;           /* per binstring; 0 disables the floor leg */
  uint32_t min_cg_depth;             /* absolute, required of EVERY class */
  float    max_frac_na;              /* fraction of classes allowed absent */
  float    depth_floor_frac;         /* relative to each class's OWN mean */
  uint32_t depth_floor_cap;          /* ceiling on the relative target */
  int      p01_on;                   /* either P(01) flag switches the rule over */
  int      inc_all0, inc_all1;       /* keep patterns no class calls 1 / 0 */
  int      quiet;                    /* suppress the per-call select line: the
                                      * satellite builders run this once PER
                                      * PAIR and fold the numbers into their own
                                      * one-line-per-pair report instead */
  uint32_t p01_top;                  /* OPTIONAL cap per binstring; 0 = uncapped */
  float    p01_min;                  /* floor on P(01); the selector when uncapped */
} ms_select_opt_t;

/* Returns a malloc'd 0/1 byte per CpG, or NULL if selection is disabled.
 *
 * `binstr[r]` is the length-ns binstring of rank r, `memb[i]` the rank of CpG i
 * (MRMP_PNA_MEMBERSHIP for PNA, which is never selected). Streams `ref` twice:
 * once for per-class mean depth (only when depth_floor_frac > 0), once for the
 * per-CpG statistics.
 *
 * `rec_off`, when non-NULL, is the BGZF virtual offset of each class's record
 * (the second column of `<ref>.idx`), so a set over a SUBSET of a store reads
 * only its own classes rather than the first ns records. NULL reads records
 * 0..ns-1 in file order, which is what a whole-store set wants. The satellite
 * builders need the subset form: writing each 2-class set out as its own .cg
 * first would cost a full recompress per pair for data already in the store. */
uint8_t *ms_mrmp_select(const char *ref, uint32_t ns, uint32_t mincov,
                        uint64_t n_cpg, const uint32_t *memb,
                        uint64_t n_cand, const char *const *binstr,
                        const ms_select_opt_t *opt, uint64_t *n_kept,
                        const int64_t *rec_off);

/* Fill `opt` with the defaults the pipeline was measured at. */
void ms_select_defaults(ms_select_opt_t *opt);

#endif
