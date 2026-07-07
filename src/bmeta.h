// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Booster metadata embedded inside a .ubj as XGBoost string attributes
 * (XGBoosterSetAttr/GetAttr — these persist through a UBJ save/load round-trip).
 *
 * methscope-cli stores the human-readable class label names here so a trained `.ubj`
 * is self-describing: `predict` needs only the booster and the matching `.mrmp`,
 * with no separate labels sidecar. The pattern count is NOT stored — it equals
 * the booster's num_feature (XGBoosterGetNumFeature).
 */
#ifndef METHSCOPE_BMETA_H
#define METHSCOPE_BMETA_H

#include <xgboost/c_api.h>

#define MS_ATTR_LABELS "methscope_labels"   /* comma-separated, class-index order */

/* Embed the class labels (class-index order) into the booster's attributes.
 * labels[] has num_class entries. Exits on XGBoost error. */
void ms_booster_set_meta(BoosterHandle b, char *const *labels, int num_class);

/* Read the embedded labels. Returns a malloc'd array of malloc'd strings and
 * sets *num_class, or NULL if the attribute is absent (caller then falls back
 * to numeric class names). Caller frees each string and the array. */
char **ms_booster_get_labels(BoosterHandle b, int *num_class);

#endif /* METHSCOPE_BMETA_H */
