// SPDX-License-Identifier: AGPL-3.0-or-later
/* Per-CpG selection for one MRMP set, applied PER BINSTRING.
 *
 * This is the cut that used to live outside the binary, as a q-filter in shell
 * plus a top-K-by-depth pass in Python. It is one cut, not two: binstrings are
 * built first over every CpG, then each binstring independently decides which
 * of its CpGs to keep.
 *
 * THERE IS ONE RULE: --qfilter admits, delta_mean ranks, --delta-mean-top
 * budgets (0 = uncapped). A second rule, P(01), was tried and removed -- see
 * the section at the end for the measurement that retired it.
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
 * P(01) WAS TRIED AS A SECOND RULE, AND REMOVED
 *
 * A single statistic replacing the gate-plus-rank pair: keep every CpG whose
 * P(01) -- the Jeffreys-shrunk probability that a future single cell reads 1 in
 * the worst expected-1 class AND 0 in the worst expected-0 class -- cleared a
 * floor. It was attractive because depth enters through the prior, so a CpG
 * seen in 3 cells cannot outrank one seen in 300 without a separate coverage
 * gate.
 *
 * Measured against the q-filter on the 41-class mouse CV, satellites over the
 * same 135 class pairs, same tree, same seed, both rules run capped and
 * uncapped (McNemar on 1,798 held-out cells, five coverage rungs):
 *
 *     capped at 20,000/binstring    net -3..+4 cells, every rung p >= 0.61
 *     uncapped                      net -6..+14 cells, every rung p >= 0.34
 *
 * Indistinguishable at matched budget in both regimes. Uncapped P(01) DID beat
 * the capped q-filter at the sparsest rung (+56 cells, p = 0.001) -- but that
 * was the missing budget, not the statistic: uncapping the q-filter recovers
 * the same gain (L2 0.7358 -> 0.7592) at the same 9x CpG cost. Raising the
 * floor to 0.65 was worse, not better (-32 cells at the sparsest rung).
 *
 * So the rule was carrying nothing the budget was not, and two selection rules
 * that silently switched on flag ORDER -- --p01-min after --qfilter voided both
 * --qfilter and --delta-mean-top -- cost more in confusion than the option was
 * worth. One rule now: q-filter admits, delta_mean ranks, --delta-mean-top
 * budgets, and --delta-mean-top 0 is the uncapped form.
 *
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
  int      inc_all0, inc_all1;       /* keep patterns no class calls 1 / 0 */
  int      quiet;                    /* suppress the per-call select line: the
                                      * satellite builders run this once PER
                                      * PAIR and fold the numbers into their own
                                      * one-line-per-pair report instead */
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
