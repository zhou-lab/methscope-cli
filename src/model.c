// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Booster metadata: embed/read the methscope class labels inside a `.ubj` as
 * XGBoost attributes. `ms_annotate_booster()` injects those attributes into a raw
 * (e.g. R-trained) booster so it becomes self-describing; it is the reusable core
 * behind `bundle -l` (which labels a booster while packing it into a `.ubjx`).
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
  char *save = NULL;
  for (char *t = strtok_r(buf, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
    if (k == cap) { cap *= 2; out = realloc(out, sizeof(char *) * cap); }
    out[k++] = strdup(t);
  }
  free(buf);
  *num_class = k;
  if (k == 0) { free(out); return NULL; }
  return out;
}

/* ------------------------------------------------------------------ */
/* ms_annotate_booster: inject labels from a meta.tsv into a raw .ubj   */
/* (the reusable core of `bundle -l`)                                   */
/* ------------------------------------------------------------------ */
static char *trim(char *s) {
  while (*s == ' ' || *s == '\t') s++;
  size_t n = strlen(s);
  while (n && (s[n-1] == '\n' || s[n-1] == '\r' || s[n-1] == ' ' || s[n-1] == '\t'))
    s[--n] = '\0';
  return s;
}

/* Parse meta.tsv (key<TAB>value lines) for the labels (comma-sep, class order).
 * Returns labels array + count. Exits if the labels line is missing. Any other
 * key (e.g. a stray 'genome' line) is ignored. */
static char **read_meta_labels(const char *path, int *num_class) {
  FILE *fp = fopen(path, "r");
  if (!fp) die("cannot open meta.tsv", path);
  char  *line = NULL; size_t cap = 0; ssize_t len;
  char **labels = NULL; int K = 0;
  while ((len = getline(&line, &cap, fp)) != -1) {
    char *tab = strchr(line, '\t');
    if (!tab) continue;
    *tab = '\0';
    char *key = trim(line);
    char *val = trim(tab + 1);
    if (strcmp(key, "labels") == 0) {
      int lcap = 8;
      labels = malloc(sizeof(char *) * lcap);
      char *save = NULL;
      for (char *t = strtok_r(val, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
        if (K == lcap) { lcap *= 2; labels = realloc(labels, sizeof(char *) * lcap); }
        labels[K++] = strdup(trim(t));
      }
    }
  }
  free(line); fclose(fp);
  if (!labels || K == 0) die("meta.tsv has no 'labels' line", path);
  *num_class = K;
  return labels;
}

/* Embed the class labels from meta.tsv (a 'labels<TAB>l1,l2,...' line) into the
 * raw booster in_ubj and save the annotated booster to out_ubj. Used by
 * `bundle -l` to prepare a labeled booster before bundling. */
void ms_annotate_booster(const char *in_ubj, const char *meta_tsv, const char *out_ubj) {
  int K = 0;
  char **labels = read_meta_labels(meta_tsv, &K);
  BoosterHandle b;
  XGCHK(XGBoosterCreate(NULL, 0, &b));
  XGCHK(XGBoosterLoadModel(b, in_ubj));
  ms_booster_set_meta(b, labels, K);
  XGCHK(XGBoosterSaveModel(b, out_ubj));
  XGBoosterFree(b);
  fprintf(stderr, "[methscope] embedded %d labels into the booster\n", K);
  for (int c = 0; c < K; ++c) free(labels[c]);
  free(labels);
}
