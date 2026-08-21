// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * methscope — pure-C analysis of sparse DNA methylomes via MRMP encoding.
 *
 * Copyright (C) 2025 Hongxiang Fu and Wanding Zhou
 *
 * This program is free software under the GNU Affero General Public License
 * v3.0 or later. See LICENSE.
 */
#ifndef METHSCOPE_H
#define METHSCOPE_H

#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Cell x pattern feature matrix (replaces the R GenerateInput output) */
/* ------------------------------------------------------------------ */
typedef struct ms_matrix_t {
  int      n_cells;        /* rows    */
  /* Columns. The order depends on WHERE the matrix came from, and the
   * difference is load-bearing:
   *
   *   ms_matrix_build()           sorts by numeric pattern id, "Pna" last
   *   ms_matrix_build_threaded()  one set, so "Pna" is simply the last column
   *   ms_msfm_to_matrix()         VERBATIM FILE ORDER -- no sort
   *
   * A single-set .msfm is P1..PN followed by Pna, so the last case looks like
   * the other two and the old blanket claim of "Pna last" seemed to hold. A
   * FUSED multi-set .msfm breaks it: that layout is set-major, so each set's
   * "Pna.<set>" sits immediately after that set's own patterns and the
   * backgrounds are INTERLEAVED rather than trailing.
   *
   * So never assume the first k columns are the real patterns. That read is
   * what fed 66 background columns into the classifier as features and dropped
   * 66 real patterns off the end of a 100-set artifact, silently and in both
   * training and scoring. Test names with ms_is_pna_name() and rebuild the
   * matrix around the selection with ms_matrix_select(). */
  int      n_patterns;
  char   **cell_names;     /* length n_cells    */
  char   **pattern_names;  /* length n_patterns */
  double  *M;              /* row-major n_cells x n_patterns; NAN = missing */
  int     *N;              /* row-major n_cells x n_patterns; per-pattern    */
                           /* covered-CpG count (N_overlap) used for beta.   */
                           /* 0 <=> M is NAN. Enables coverage filtering.    */
} ms_matrix_t;

/**
 * Build the cell x pattern beta matrix by summarizing each query record
 * (cell) against every mask record (MRMP pattern) in reference.cm, using the
 * YAME summary core. Missing overlap -> NAN. Columns are ordered by numeric
 * pattern id ("Pna" last). Caller frees with ms_matrix_free().
 */
/* Keep only columns idx[0..n-1], in that order, compacting M/N/pattern_names in
 * place and setting n_patterns = n. Every consumer indexes columns positionally,
 * so selecting a subset means REBUILDING the matrix around it rather than
 * carrying an index alongside -- otherwise each consumer has to remember, and
 * ms_msfm_to_matrix's set-major layout (Pna interleaved, not trailing) means
 * the one that forgets silently trains on background columns. */
void         ms_matrix_select(ms_matrix_t *m, const int *idx, int n);

ms_matrix_t *ms_matrix_build(const char *query_cg, const char *ref_cm);
void         ms_matrix_free(ms_matrix_t *m);

/**
 * Write a trimmed copy of an MRMP reference (.cm) keeping only the states named
 * in keep_names[], folding every other state (incl. the original "Pna") into a
 * single "Pna" background. Used by `classify-train` to bundle exactly the mrmp the model
 * uses. Matching is by name (see matrix.c).
 */
void ms_mrmp_trim(const char *in_cm, char *const *keep_names, int n_keep,
                  const char *out_cm);

/* ------------------------------------------------------------------ */
/* Model artifacts                                                    */
/* ------------------------------------------------------------------ */
/* A methscope model ships as a self-contained bundle (`.ubjx`/`.updecx`/`.refx`,
 * see bundle.h) wrapping an inner model + its MRMP. The loose parts are:
 *   <mrmp>.mrmp            the MRMP pattern definition (a YAME .cm)
 *   <mrmp>-<panel>.ubj     an XGBoost booster with class labels embedded as
 *                          attributes (see bmeta.h)
 * Deconvolution does not use a bundle: its reference is a standalone .msdref
 * (see deconv.c), built by `deconv-build-ref`.
 * Booster-attribute helpers live in bmeta.h (they need <xgboost/c_api.h>). */

/* ------------------------------------------------------------------ */
/* Linear framework models (threshold / logistic) — see linear.c       */
/* ------------------------------------------------------------------ */
/* A binary linear classifier over the MRMP feature betas: score =
 * bias + sum_i w_i * beta_i ; p(class1) = sigmoid(score/scale) ; missing beta_i
 * is imputed with the stored training mean_i. `method` records how it was fit
 * ("threshold" = mean-difference + centroid-midpoint cutoff; "logistic" =
 * logistic regression). Serialized as a small "methscope-linear" text spec that
 * lives in the bundle's `model` section (kind = the method). */
typedef struct linmodel_t {
  int      n_feat;
  char   **names;                 /* feature (pattern) names, model order */
  double  *w;                     /* weights (length n_feat) */
  double  *mean;                  /* training means for imputation (n_feat) */
  double   bias, scale;
  char    *label0, *label1;       /* class0 = low score, class1 = high score */
  char    *method;                /* "threshold" | "logistic" */
} linmodel_t;

/* Fit over the first n_feat columns of m, given per-record class index y01 (0/1)
 * and the two label strings. method = "threshold" or "logistic". */
linmodel_t *ms_linmodel_fit(const ms_matrix_t *m, int n_feat, const int *y01,
                            const char *label0, const char *label1,
                            const char *method);
void        ms_linmodel_write(const linmodel_t *lm, const char *path);
linmodel_t *ms_linmodel_parse(const char *buf, size_t len);
/* Score one record's betas[n_feat]; returns class index (0/1), sets *p1 (prob of
 * class1) and *conf (Shannon confidence, matching predict's xgboost path). */
int  ms_linmodel_score(const linmodel_t *lm, const double *betas,
                       double *p1, double *conf);
void ms_linmodel_free(linmodel_t *lm);

/* ------------------------------------------------------------------ */
/* Violation framework (multi-class, unfitted) — see violation.c       */
/* ------------------------------------------------------------------ */
/* Calls the class a query contradicts least, using only the MRMP's own
 * binstrings: for each class, the weighted fraction of its expected-methylated
 * patterns that came back unmethylated, plus the converse, minimized. There is
 * no fitted parameter — the model IS the reference definition — so it is built
 * by transcription from the MRMP artifact rather than by training. Confidence
 * is the margin to the runner-up, which collapses toward zero when no class
 * fits, i.e. when the query's true type is absent from the reference. */
typedef struct viomodel_t {
  int      n_label, n_feat;
  char   **labels;                /* n_label, binstring position order */
  char   **names;                 /* n_feat pattern names ("P1"...) */
  char   **bin;                   /* n_feat binstrings, n_label chars each */
  double  *w;                     /* n_feat pattern weights (derived from n_cpg) */
  double   threshold;             /* beta call cutoff (0.5) */
  int      min_patterns;          /* required observed patterns per side */
  char    *weighting;             /* "sqrt" | "log1p" | "linear" | "flat" */
} viomodel_t;

/* Transcribe a classifier out of an MRMP artifact — no training data. */
viomodel_t *ms_viomodel_from_mrmp(const char *artifact, uint32_t top_k,
                                  double threshold, const char *weighting,
                                  int min_patterns);
void        ms_viomodel_write(const viomodel_t *vm, const char *path);
viomodel_t *ms_viomodel_parse(const char *buf, size_t len);
/* Per-label violation totals into out[n_label]; INFINITY where too sparse. */
void ms_viomodel_scores(const viomodel_t *vm, const double *betas, double *out);
/* Score betas[n_feat] (NaN = unobserved: skipped, never imputed). Returns the
 * winning label index, or -1 when too sparse to call; sets *score and *margin. */
int  ms_viomodel_score(const viomodel_t *vm, const double *betas,
                       double *score, double *margin);
void ms_viomodel_free(viomodel_t *vm);

/* Render subcommand help text, ANSI-styled on a TTY, plain when redirected. */
void ms_help(FILE *out, const char *text);

/* subcommand entry points */
int main_inspect(int argc, char *argv[]);
int main_predict(int argc, char *argv[]);
int main_train(int argc, char *argv[]);
int main_train_tree(int argc, char *argv[]);
int main_deconv_build_ref(int argc, char *argv[]);
int main_deconv(int argc, char *argv[]);
/* Embed labels from a meta.tsv into a raw booster (used by `bundle -l`). */
void ms_annotate_booster(const char *in_ubj, const char *meta_tsv, const char *out_ubj);
int main_upscale(int argc, char *argv[]);
int main_upscale_train(int argc, char *argv[]);
int main_upscale_internal(int argc, char *argv[]);
int main_upscale_prepare(int argc, char *argv[]);

/* `methscope inspect` reporters for the two upscale build artifacts. */
void ms_msur_report(const char *path);
void ms_msui_report(const char *path);
int main_upscale_set_units(int argc, char *argv[]);
int main_upscale_trunk_train(int argc, char *argv[]);
int main_bundle(int argc, char *argv[]);
int main_unbundle(int argc, char *argv[]);
int main_relabel(int argc, char *argv[]);

#define METHSCOPE_VERSION "0.5"


/* True for a per-set NA-BACKGROUND column: the CpGs a set has no pattern for.
 *
 * "Pna" alone is the single-set name kept for compatibility; "Pna.<set>" is what
 * a FUSED matrix writes, one per set. Fusing used to number every column
 * P1..P(n-1) and name only the very last one "Pna", so 99 of 100 background
 * columns were indistinguishable from real patterns and reached the classifier
 * as features -- --include-pna could not exclude what it could not name.
 *
 * A background column carries no class contrast by construction, which is the
 * same reason homogeneous binstrings never survive the q-filter. Test with this
 * rather than strcmp so all three consumers (train, deconv, the matrix column
 * sort) agree, and so the digit-run sort key never mistakes "Pna.bio_itl23" for
 * pattern number 23. */
static inline int ms_is_pna_name(const char *n) {
  return n && (strcmp(n, "Pna") == 0 || strncmp(n, "Pna.", 4) == 0);
}

#endif /* METHSCOPE_H */
