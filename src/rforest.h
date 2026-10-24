// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
/**
 * Breiman's random forest for classification, written from the algorithm
 * (Breiman 2001; the R randomForest package documents the same choices) so
 * methscope can rank and select features the way the R forests in the lab's
 * array classifiers do, with no GPL code linked in.
 *
 * Per tree: a bootstrap of n rows drawn with replacement; at every node mtry
 * features drawn without replacement and the best Gini split among them,
 * cut at the midpoint between neighbouring values; nodes split until pure
 * (nodesize 1) or until no drawn feature improves Gini, which makes a leaf.
 *
 * Importance, both as R reports them in rf$importance:
 *   mda   MeanDecreaseAccuracy, unscaled: per tree, (OOB right - OOB right
 *         with feature j permuted among the OOB rows) / n OOB, averaged over
 *         all trees (a tree that never splits on j adds 0).
 *   gini  MeanDecreaseGini: total Gini decrease of j's splits, weighted by
 *         node size, averaged over trees.
 *
 * Results do not depend on the thread count: each tree draws from its own
 * stream, seeded from (seed, tree index), and trees are reduced in order.
 * The numbers will not match R draw for draw (different RNG); rankings and
 * OOB error should agree within Monte Carlo noise.
 */
#ifndef METHSCOPE_RFOREST_H
#define METHSCOPE_RFOREST_H

#include <stdint.h>

typedef struct {
  uint32_t ntree;     /* R default 500 */
  uint32_t mtry;      /* 0 = floor(sqrt(p)), R's classification default */
  uint64_t seed;
  int      threads;   /* <= 0 = 1 */
} rf_param_t;

/* One tree, nodes in creation order, node 0 the root. A leaf has
 * feat == -1 and carries its class in cls; an internal node sends
 * x[feat] <= thr to left, the rest to right. */
typedef struct {
  uint32_t n_node;
  int32_t *feat;
  float   *thr;
  int32_t *left, *right;
  int32_t *cls;
} rf_tree_t;

typedef struct {
  uint32_t   ntree, p, nclass;
  rf_tree_t *tree;
  double     oob_err;   /* OOB misclassification rate of the forest vote */
  double    *oob_vote;  /* n x nclass OOB vote fractions (R's rf$votes);
                         * a row never out of bag is all 0 */
} rf_forest_t;

/* Fit on X (n x p, row-major, no NaN) with labels y in [0, nclass).
 * mda and gini, when not NULL, receive p importances each. */
rf_forest_t *rf_fit(const float *X, uint32_t n, uint32_t p, const int *y,
                    uint32_t nclass, const rf_param_t *par,
                    double *mda, double *gini);

/* Vote fractions, n x nclass row-major: R's predict(type = "prob"). */
void rf_predict(const rf_forest_t *f, const float *X, uint32_t n,
                double *prob);

void rf_free(rf_forest_t *f);

/* The trees as bytes, for a model file: "MSRF0001", then ntree, p, nclass
 * (uint32), then per tree n_node and its five node arrays, little-endian.
 * OOB results are fit-time diagnostics and are not stored. rf_serialize
 * returns a malloc'd buffer and its length; rf_deserialize returns NULL on
 * a malformed buffer, and sets *used to the bytes it read. */
uint8_t *rf_serialize(const rf_forest_t *f, size_t *len);
rf_forest_t *rf_deserialize(const uint8_t *buf, size_t len, size_t *used);

#endif /* METHSCOPE_RFOREST_H */
