// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Booster metadata embedded inside a .ubj as XGBoost string attributes
 * (XGBoosterSetAttr/GetAttr — these persist through a UBJ save/load round-trip).
 *
 * methscope stores the human-readable class label names here so a trained `.ubj`
 * is self-describing: `predict` needs only the booster and the matching `.mrmp`,
 * with no separate labels msur. The pattern count is NOT stored — it equals
 * the booster's num_feature (XGBoosterGetNumFeature).
 */
#ifndef METHSCOPE_BMETA_H
#define METHSCOPE_BMETA_H

#include <xgboost/c_api.h>

#define MS_ATTR_LABELS "methscope_labels"   /* comma-separated, class-index order */

/* Set when the model carries ONE extra feature after the pattern columns:
 * log1p(covered CpGs) for the record. `classify` has to append the same column
 * or every prediction silently shifts by a feature, so this lives in the model
 * rather than relying on the caller to remember. Value is the transform name. */
#define MS_ATTR_SCALARCOV "methscope_scalar_cov"

/* The feature columns the booster was trained on, newline-separated, in model
 * column order (the --scalar-coverage column, if any, is NOT listed -- it is
 * appended after these and marked by MS_ATTR_SCALARCOV).
 *
 * Needed because a fused multi-set .msfm is laid out SET-MAJOR: each set's Pna
 * background sits immediately after that set's own patterns, so the background
 * columns are interleaved rather than trailing. Selecting "the first N columns"
 * therefore takes some backgrounds as features and drops an equal number of
 * real patterns off the end. Recording the names lets `classify` gather exactly
 * the columns `classify-train` used, whatever order the query artifact has. */
#define MS_ATTR_FEATURES "methscope_features"

/* How the pattern features were coded when the model was trained: "0.5" for the
 * flat cut, "pattern" for --thresh-pattern. Absent means continuous, which is
 * what every model trained before 2026-08-09 was.
 *
 * `classify --data` reads a .msfm that is already coded, but `classify` on a
 * raw .cg builds features through ms_matrix_build(), which returns a continuous
 * mean. Without this, a model trained on {0,1,NA} silently scores against betas
 * in [0,1] -- measured at 2 of 42 cells correct, with 39 of 43 collapsing onto
 * one class. The coding travels with the model so the .cg path reproduces it
 * rather than guessing. */
#define MS_ATTR_BINARIZE "methscope_binarize"

/* The label hierarchy, so a prediction is self-describing: one row per class,
 * "label\tcompartment\tlineage\tgroup\tsubtype", rows newline-separated.
 * Lets `classify --levels` report the taxonomy path without a side table, and
 * lets an external cohort be scored at whatever granularity BOTH sides resolve
 * -- a coarse "T cell" truth meets a "Blood.T.Mem.CD4" prediction at `group`.
 * "NA" at a level means the source does not resolve that far. */
#define MS_ATTR_HIERARCHY "methscope_hierarchy"

/* Embed the class labels (class-index order) into the booster's attributes.
 * labels[] has num_class entries. Exits on XGBoost error. */
void ms_booster_set_meta(BoosterHandle b, char *const *labels, int num_class);

/* Embed / read the label hierarchy (see MS_ATTR_HIERARCHY). ms_booster_get_hier
 * returns the raw TSV block or NULL; caller frees. */
void  ms_booster_set_hier(BoosterHandle b, const char *tsv);
char *ms_booster_get_hier(BoosterHandle b);

/* Embed / read the feature column names (see MS_ATTR_FEATURES). The getter
 * returns a malloc'd array of malloc'd strings and sets *n_feat, or NULL when
 * the attribute is absent -- which is how a pre-2026-08 model is recognised.
 * Caller frees each string and the array. */
void   ms_booster_set_features(BoosterHandle b, char *const *names, int n_feat);
char **ms_booster_get_features(BoosterHandle b, int *n_feat);

/* Record / read the feature coding (see MS_ATTR_BINARIZE). The getter returns
 * a malloc'd string or NULL when the attribute is absent. */
void  ms_booster_set_binarize(BoosterHandle b, const char *how);
char *ms_booster_get_binarize(BoosterHandle b);

/* Mark / test the scalar coverage feature (see MS_ATTR_SCALARCOV). */
void ms_booster_set_scalar_cov(BoosterHandle b);
int  ms_booster_has_scalar_cov(BoosterHandle b);

/* Read the embedded labels. Returns a malloc'd array of malloc'd strings and
 * sets *num_class, or NULL if the attribute is absent (caller then falls back
 * to numeric class names). Caller frees each string and the array. */
char **ms_booster_get_labels(BoosterHandle b, int *num_class);

#endif /* METHSCOPE_BMETA_H */
