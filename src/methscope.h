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

#include <stdio.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Cell x pattern feature matrix (replaces the R GenerateInput output) */
/* ------------------------------------------------------------------ */
typedef struct ms_matrix_t {
  int      n_cells;        /* rows    */
  int      n_patterns;     /* columns, ordered by numeric pattern id (Pna last) */
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
ms_matrix_t *ms_matrix_build(const char *query_cg, const char *ref_cm);
void         ms_matrix_free(ms_matrix_t *m);
void         ms_matrix_write_tsv(const ms_matrix_t *m, FILE *out, int header);

/**
 * Write a trimmed copy of an MRMP reference (.cm) keeping only the states named
 * in keep_names[], folding every other state (incl. the original "Pna") into a
 * single "Pna" background. Used by `train` to bundle exactly the mrmp the model
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
 * For deconvolution the reference is a `.refx` whose `model` section is a
 * celltype x pattern signature TSV (built by `matrix --refx`).
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

/* subcommand entry points */
int main_matrix(int argc, char *argv[]);
int main_inspect(int argc, char *argv[]);
int main_predict(int argc, char *argv[]);
int main_train(int argc, char *argv[]);
int main_deconv(int argc, char *argv[]);
/* Embed labels from a meta.tsv into a raw booster (used by `bundle -l`). */
void ms_annotate_booster(const char *in_ubj, const char *meta_tsv, const char *out_ubj);
int main_upscale(int argc, char *argv[]);
int main_upscale_train(int argc, char *argv[]);
int main_upscale_prepare(int argc, char *argv[]);
int main_upscale_factor_train(int argc, char *argv[]);
int main_upscale_residual_index(int argc, char *argv[]);
int main_upscale_residual_train(int argc, char *argv[]);
int main_upscale_hybrid_eval(int argc, char *argv[]);
int main_bundle(int argc, char *argv[]);
int main_unbundle(int argc, char *argv[]);

#define METHSCOPE_VERSION "0.1.1"

#endif /* METHSCOPE_H */
