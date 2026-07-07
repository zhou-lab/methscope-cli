// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Binary linear classifiers over the MRMP feature betas — an interpretable
 * alternative to the XGBoost booster. Two fit methods share ONE spec + inference
 * (only the weights/bias/scale differ), and the model's `kind` mark records which:
 *
 *   "threshold" — mean-difference direction (w_i = mean1_i - mean0_i) with the
 *                 decision boundary at the MIDPOINT of the two class score
 *                 centroids (robust when the classes are well separated, e.g. sex).
 *   "logistic"  — L2-regularized logistic regression (gradient descent).
 *
 * Inference: s = bias + sum_i w_i * (beta_i or mean_i if NaN);
 *            p1 = sigmoid(s/scale); label = p1>=0.5 ? class1 : class0.
 * Missing features fall back to the stored training mean, so the rule degrades
 * gracefully on sparse / downsampled input.
 *
 * On-disk spec (the bundle's `model` section):
 *   methscope-linear <TAB> 1
 *   method  <TAB> threshold|logistic
 *   labels  <TAB> <class0> <TAB> <class1>
 *   bias    <TAB> <b>
 *   scale   <TAB> <s>
 *   feature <TAB> <name> <TAB> <weight> <TAB> <mean>      (one per feature)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "methscope.h"

static void ldie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] %s\n", msg);
  exit(1);
}

static linmodel_t *lm_alloc(int n_feat) {
  linmodel_t *lm = calloc(1, sizeof(*lm));
  if (!lm) ldie("out of memory (linmodel)", NULL);
  lm->n_feat = n_feat;
  lm->names  = calloc(n_feat, sizeof(char *));
  lm->w      = calloc(n_feat, sizeof(double));
  lm->mean   = calloc(n_feat, sizeof(double));
  lm->scale  = 1.0;
  if (!lm->names || !lm->w || !lm->mean) ldie("out of memory (linmodel)", NULL);
  return lm;
}

void ms_linmodel_free(linmodel_t *lm) {
  if (!lm) return;
  for (int i = 0; i < lm->n_feat; ++i) free(lm->names[i]);
  free(lm->names); free(lm->w); free(lm->mean);
  free(lm->label0); free(lm->label1); free(lm->method);
  free(lm);
}

/* per-feature training means (NaN-ignored); all-NaN column -> mean 0. */
static void feature_means(const ms_matrix_t *m, int n_feat, double *mean) {
  for (int j = 0; j < n_feat; ++j) {
    double sum = 0; int cnt = 0;
    for (int r = 0; r < m->n_cells; ++r) {
      double v = m->M[(size_t)r * m->n_patterns + j];
      if (!isnan(v)) { sum += v; cnt++; }
    }
    mean[j] = cnt ? sum / cnt : 0.0;
  }
}

/* --------------------------- threshold fit --------------------------- */
static void fit_threshold(const ms_matrix_t *m, int n_feat, const int *y01,
                          linmodel_t *lm) {
  /* class means per feature (NaN-ignored) -> weights = mean1 - mean0 */
  for (int j = 0; j < n_feat; ++j) {
    double s0 = 0, s1 = 0; int c0 = 0, c1 = 0;
    for (int r = 0; r < m->n_cells; ++r) {
      double v = m->M[(size_t)r * m->n_patterns + j];
      if (isnan(v)) continue;
      if (y01[r]) { s1 += v; c1++; } else { s0 += v; c0++; }
    }
    double mu0 = c0 ? s0 / c0 : lm->mean[j];
    double mu1 = c1 ? s1 / c1 : lm->mean[j];
    lm->w[j] = mu1 - mu0;
  }
  /* project training records -> score centroids (impute NaN with mean) */
  double C0 = 0, C1 = 0; int n0 = 0, n1 = 0;
  for (int r = 0; r < m->n_cells; ++r) {
    double s = 0;
    for (int j = 0; j < n_feat; ++j) {
      double v = m->M[(size_t)r * m->n_patterns + j];
      s += lm->w[j] * (isnan(v) ? lm->mean[j] : v);
    }
    if (y01[r]) { C1 += s; n1++; } else { C0 += s; n0++; }
  }
  C0 = n0 ? C0 / n0 : 0; C1 = n1 ? C1 / n1 : 0;   /* C1 >= C0 by construction */
  double cutoff = 0.5 * (C0 + C1);
  double half   = 0.5 * (C1 - C0);
  lm->bias  = -cutoff;
  lm->scale = (half > 1e-9) ? half : 1.0;         /* logit=1 at a centroid */
}

/* --------------------------- logistic fit ---------------------------- */
static void fit_logistic(const ms_matrix_t *m, int n_feat, const int *y01,
                         linmodel_t *lm) {
  int N = m->n_cells;
  /* standardize on mean-imputed features: z = (x - mean)/sd */
  double *sd = calloc(n_feat, sizeof(double));
  if (!sd) ldie("out of memory (logistic sd)", NULL);
  for (int j = 0; j < n_feat; ++j) {
    double s2 = 0;
    for (int r = 0; r < N; ++r) {
      double v = m->M[(size_t)r * m->n_patterns + j];
      double x = isnan(v) ? lm->mean[j] : v;
      double d = x - lm->mean[j]; s2 += d * d;
    }
    double var = N ? s2 / N : 0.0;
    sd[j] = var > 1e-12 ? sqrt(var) : 1.0;
  }
  double *ws = calloc(n_feat, sizeof(double));    /* standardized-space weights */
  if (!ws) ldie("out of memory (logistic w)", NULL);
  double b0 = 0.0;
  const double lr = 0.5, lambda = 1.0;
  const int    iters = 4000;
  double *gw = calloc(n_feat, sizeof(double));
  if (!gw) ldie("out of memory (logistic grad)", NULL);
  for (int it = 0; it < iters; ++it) {
    double gb = 0.0;
    for (int j = 0; j < n_feat; ++j) gw[j] = 0.0;
    for (int r = 0; r < N; ++r) {
      double logit = b0;
      for (int j = 0; j < n_feat; ++j) {
        double v = m->M[(size_t)r * m->n_patterns + j];
        double x = isnan(v) ? lm->mean[j] : v;
        logit += ws[j] * (x - lm->mean[j]) / sd[j];
      }
      double p = 1.0 / (1.0 + exp(-logit));
      double e = p - (double)y01[r];
      gb += e;
      for (int j = 0; j < n_feat; ++j) {
        double v = m->M[(size_t)r * m->n_patterns + j];
        double x = isnan(v) ? lm->mean[j] : v;
        gw[j] += e * (x - lm->mean[j]) / sd[j];
      }
    }
    b0 -= lr * gb / N;
    for (int j = 0; j < n_feat; ++j)
      ws[j] -= lr * (gw[j] / N + lambda * ws[j] / N);
  }
  /* fold standardization back into raw-beta space:
   * logit = b0 + sum ws_j (x_j - mean_j)/sd_j
   *       = [b0 - sum ws_j mean_j/sd_j] + sum (ws_j/sd_j) x_j  */
  double bias = b0;
  for (int j = 0; j < n_feat; ++j) {
    lm->w[j] = ws[j] / sd[j];
    bias    -= lm->w[j] * lm->mean[j];
  }
  lm->bias  = bias;
  lm->scale = 1.0;
  free(sd); free(ws); free(gw);
}

linmodel_t *ms_linmodel_fit(const ms_matrix_t *m, int n_feat, const int *y01,
                            const char *label0, const char *label1,
                            const char *method) {
  linmodel_t *lm = lm_alloc(n_feat);
  lm->method = strdup(method);
  lm->label0 = strdup(label0);
  lm->label1 = strdup(label1);
  for (int j = 0; j < n_feat; ++j) lm->names[j] = strdup(m->pattern_names[j]);
  feature_means(m, n_feat, lm->mean);
  if      (strcmp(method, "threshold") == 0) fit_threshold(m, n_feat, y01, lm);
  else if (strcmp(method, "logistic")  == 0) fit_logistic(m, n_feat, y01, lm);
  else ldie("unknown linear method", method);
  return lm;
}

/* ----------------------------- serialize ----------------------------- */
void ms_linmodel_write(const linmodel_t *lm, const char *path) {
  FILE *fp = fopen(path, "w");
  if (!fp) ldie("cannot open linear model output", path);
  fprintf(fp, "methscope-linear\t1\n");
  fprintf(fp, "method\t%s\n", lm->method);
  fprintf(fp, "labels\t%s\t%s\n", lm->label0, lm->label1);
  fprintf(fp, "bias\t%.9g\n", lm->bias);
  fprintf(fp, "scale\t%.9g\n", lm->scale);
  for (int j = 0; j < lm->n_feat; ++j)
    fprintf(fp, "feature\t%s\t%.9g\t%.9g\n", lm->names[j], lm->w[j], lm->mean[j]);
  fclose(fp);
}

/* ------------------------------- parse ------------------------------- */
linmodel_t *ms_linmodel_parse(const char *buf, size_t len) {
  char *copy = malloc(len + 1);
  if (!copy) ldie("out of memory (parse)", NULL);
  memcpy(copy, buf, len); copy[len] = '\0';

  /* first pass: count feature lines */
  int n_feat = 0;
  for (const char *p = copy; (p = strstr(p, "feature\t")); p += 8)
    if (p == copy || p[-1] == '\n') n_feat++;

  linmodel_t *lm = lm_alloc(n_feat > 0 ? n_feat : 1);
  lm->n_feat = 0;                    /* filled as we read feature lines */
  char *save = NULL;
  int header_ok = 0;
  for (char *line = strtok_r(copy, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
    char *tab = strchr(line, '\t');
    if (!tab) { if (strncmp(line, "methscope-linear", 16) == 0) header_ok = 1; continue; }
    *tab = '\0'; char *key = line; char *val = tab + 1;
    if      (strncmp(key, "methscope-linear", 16) == 0) header_ok = 1;
    else if (strcmp(key, "method") == 0) lm->method = strdup(val);
    else if (strcmp(key, "labels") == 0) {
      char *t2 = strchr(val, '\t');
      if (!t2) ldie("linear spec: bad labels line", NULL);
      *t2 = '\0'; lm->label0 = strdup(val); lm->label1 = strdup(t2 + 1);
    }
    else if (strcmp(key, "bias")  == 0) lm->bias  = atof(val);
    else if (strcmp(key, "scale") == 0) lm->scale = atof(val);
    else if (strcmp(key, "feature") == 0) {
      /* val = name \t weight \t mean */
      char *t2 = strchr(val, '\t'); if (!t2) ldie("linear spec: bad feature", NULL);
      *t2 = '\0'; char *wtxt = t2 + 1;
      char *t3 = strchr(wtxt, '\t'); if (!t3) ldie("linear spec: bad feature", NULL);
      *t3 = '\0'; char *mtxt = t3 + 1;
      int j = lm->n_feat++;
      lm->names[j] = strdup(val);
      lm->w[j]     = atof(wtxt);
      lm->mean[j]  = atof(mtxt);
    }
  }
  free(copy);
  if (!header_ok || !lm->method || !lm->label0)
    ldie("not a methscope-linear model spec", NULL);
  if (lm->scale == 0.0) lm->scale = 1.0;
  return lm;
}

/* ------------------------------- score ------------------------------- */
int ms_linmodel_score(const linmodel_t *lm, const double *betas,
                      double *p1_out, double *conf_out) {
  double s = lm->bias;
  for (int j = 0; j < lm->n_feat; ++j) {
    double v = betas[j];
    s += lm->w[j] * (isnan(v) ? lm->mean[j] : v);
  }
  double p1 = 1.0 / (1.0 + exp(-s / lm->scale));
  double p0 = 1.0 - p1;
  /* Shannon confidence (matches predict's xgboost path, K=2). */
  double ent = -(p0 * log(p0 + 1e-10) + p1 * log(p1 + 1e-10));
  double conf = 1.0 - ent / log(2.0);
  if (conf < 0.0) conf = 0.0;
  if (conf > 1.0) conf = 1.0;
  if (p1_out)   *p1_out   = p1;
  if (conf_out) *conf_out = conf;
  return p1 >= 0.5 ? 1 : 0;
}
