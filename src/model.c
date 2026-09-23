// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
/**
 * Booster metadata: embed/read the methscope class labels inside a `.ubj`
 * so the booster is self-describing.
 *
 * The booster (`.ubj`) and the MRMP reference (`.mrmp`, a YAME `.cm`) are separate
 * loose files, named so the booster declares which MRMP it belongs to, e.g.
 *   mouse_brain.mrmp                 (the MRMP pattern definition)
 *   mouse_brain-celltypes.ubj        (a booster trained on those patterns)
 */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "methscope.h"
#include "bmeta.h"
#include "bundle.h"    /* ms_bundle_pack / ms_path_is_bundle_ext */

#define XGCHK(call) do {                                            \
    if ((call) != 0) {                                             \
      fprintf(stderr, "[methscope] xgboost error: %s\n",          \
              XGBGetLastError());                                  \
      exit(1);                                                     \
    }                                                             \
  } while (0)

static void die(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] model: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] model: %s\n", msg);
  exit(1);
}

/* ------------------------------------------------------------------ */
/* Embed / read booster attributes                                    */
/* ------------------------------------------------------------------ */
void ms_booster_set_meta(BoosterHandle b, char *const *labels, int num_class) {
  /* labels joined comma-separated, in class-index order */
  size_t cap = 1;
  for (int c = 0; c < num_class; ++c) cap += strlen(labels[c]) + 1;
  char *csv = malloc(cap);
  if (!csv) die("out of memory (labels csv)", NULL);
  size_t off = 0;
  for (int c = 0; c < num_class; ++c)
    off += (size_t)snprintf(csv + off, cap - off, "%s%s", c ? "," : "", labels[c]);
  XGCHK(XGBoosterSetAttr(b, MS_ATTR_LABELS, csv));
  free(csv);
}


void ms_booster_set_features(BoosterHandle b, char *const *names, int n_feat) {
  size_t n = 1;
  for (int i = 0; i < n_feat; ++i) n += strlen(names[i]) + 1;
  char *buf = malloc(n);
  if (!buf) die("out of memory (feature names)", NULL);
  char *w = buf;
  for (int i = 0; i < n_feat; ++i) {
    if (i) *w++ = '\n';
    size_t L = strlen(names[i]);
    memcpy(w, names[i], L); w += L;
  }
  *w = '\0';
  XGCHK(XGBoosterSetAttr(b, MS_ATTR_FEATURES, buf));
  free(buf);
}

void ms_booster_set_binarize(BoosterHandle b, const char *how) {
  XGCHK(XGBoosterSetAttr(b, MS_ATTR_BINARIZE, how));
}

char *ms_booster_get_binarize(BoosterHandle b) {
  const char *val = NULL;
  int success = 0;
  if (XGBoosterGetAttr(b, MS_ATTR_BINARIZE, &val, &success) != 0 || !success || !val)
    return NULL;
  char *out = strdup(val);
  if (!out) die("out of memory (binarize)", NULL);
  return out;
}


void ms_booster_set_pooled(BoosterHandle b) {
  XGCHK(XGBoosterSetAttr(b, MS_ATTR_POOLED, "1"));
}

int ms_booster_get_pooled(BoosterHandle b) {
  const char *val = NULL;
  int success = 0;
  if (XGBoosterGetAttr(b, MS_ATTR_POOLED, &val, &success) != 0 || !success || !val)
    return 0;
  return 1;
}

void ms_booster_set_colsel(BoosterHandle b, const uint32_t *idx, uint32_t n) {
  /* comma list; 11 bytes/index is generous */
  char *buf = malloc((size_t)n * 12 + 1);
  if (!buf) die("out of memory (colsel)", NULL);
  size_t at = 0;
  for (uint32_t i = 0; i < n; ++i)
    at += (size_t)sprintf(buf + at, i ? ",%u" : "%u", idx[i]);
  XGCHK(XGBoosterSetAttr(b, MS_ATTR_COLSEL, buf));
  free(buf);
}

uint32_t *ms_booster_get_colsel(BoosterHandle b, uint32_t *n_out) {
  const char *val = NULL;
  int success = 0;
  *n_out = 0;
  if (XGBoosterGetAttr(b, MS_ATTR_COLSEL, &val, &success) != 0 || !success || !val)
    return NULL;
  uint32_t n = 1;
  for (const char *p = val; *p; ++p) if (*p == ',') ++n;
  uint32_t *idx = malloc((size_t)n * sizeof(uint32_t));
  if (!idx) die("out of memory (colsel)", NULL);
  uint32_t k = 0;
  const char *p = val;
  while (k < n) {
    idx[k++] = (uint32_t)strtoul(p, (char **)&p, 10);
    if (*p == ',') ++p; else break;
  }
  *n_out = k;
  return idx;
}

char **ms_booster_get_labels(BoosterHandle b, int *num_class) {
  const char *val = NULL;
  int success = 0;
  if (XGBoosterGetAttr(b, MS_ATTR_LABELS, &val, &success) != 0 || !success || !val) {
    *num_class = 0;
    return NULL;
  }
  /* split comma-separated copy */
  char *buf = strdup(val);
  if (!buf) die("out of memory (labels)", NULL);
  int cap = 8, k = 0;
  char **out = malloc(sizeof(char *) * cap);
  if (!out) die("out of memory", NULL);
  char *save = NULL;
  for (char *t = strtok_r(buf, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
    if (k == cap) {
      cap *= 2;
      char **tmp = realloc(out, sizeof(char *) * cap);
      if (!tmp) die("out of memory", NULL);
      out = tmp;
    }
    out[k++] = strdup(t);
    if (!out[k-1]) die("out of memory", NULL);
  }
  free(buf);
  *num_class = k;
  if (k == 0) { free(out); return NULL; }
  return out;
}

