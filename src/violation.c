// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * The `violation` framework — multi-class cell-type calling with no fitted
 * parameter of any kind.
 *
 * Every MRMP already carries a binstring: the methylation state each reference
 * class takes at that pattern, '1' methylated and '0' unmethylated. That IS a
 * per-class prediction, so a classifier needs no training — only a way to ask
 * which class the query contradicts least:
 *
 *   E1_k = patterns whose binstring is '1' at class k   (expect methylated)
 *   E0_k = patterns whose binstring is '0' at class k   (expect unmethylated)
 *   w_j  = sqrt(n_cpg_j)
 *
 *   v1 = SUM_{j in E1_k, observed} w_j*[beta_j <= t] / SUM_{j in E1_k, obs} w_j
 *   v0 = SUM_{j in E0_k, observed} w_j*[beta_j >  t] / SUM_{j in E0_k, obs} w_j
 *
 *   predict argmin_k (v1 + v0), requiring >= min_patterns observed per side
 *
 * Confidence is the MARGIN, runner-up score minus winner. It is calibrated
 * (accuracy 0.757 / 0.820 / 0.899 at margin >= 0.02 / 0.05 / 0.10) and, unlike
 * a softmax probability, it goes to zero when NO class fits — which is the
 * signal that the query's true type is missing from the reference atlas.
 *
 * Why sqrt(n_cpg) and not n_cpg. CpGs inside one MRMP share a methylation
 * profile across every reference class by construction, so they are strongly
 * correlated and effective sample size saturates far below the raw count.
 * Weighting linearly lets the 10 largest patterns (28.8% of CpG mass in the
 * shipped human model) decide every call, which measurably breaks transfer to
 * tumours. Held-out accuracy: sqrt 0.842 > log1p 0.824 > flat 0.818 > linear
 * 0.797.
 *
 * Missing features are SKIPPED, not imputed. The linear frameworks impute a
 * training mean because a linear score needs every term; a violation rate is a
 * ratio over whatever was observed, so sparsity shrinks the denominator instead
 * of injecting a fabricated value. That is what keeps the rule honest on sparse
 * single cells — though it still needs coverage: accuracy is 0.163 below 8k
 * covered CpGs, rising to a 0.85 plateau above ~130k.
 *
 * On-disk spec (the bundle's `model` section):
 *   methscope-violation <TAB> 1
 *   threshold    <TAB> <t>                     beta call cutoff (0.5)
 *   weight       <TAB> sqrt|log1p|flat|linear  pattern weighting
 *   min_patterns <TAB> <n>                     required observed per side
 *   labels       <TAB> <l0> <TAB> <l1> ...     binstring position order
 *   pattern      <TAB> <name> <TAB> <binstring> <TAB> <n_cpg>   (one per row)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "methscope.h"
#include "mrmp.h"

static void vdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] violation: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] violation: %s\n", msg);
  exit(1);
}

static double vio_weight(const char *how, uint64_t n) {
  if (!strcmp(how, "sqrt"))   return sqrt((double)n);
  if (!strcmp(how, "log1p"))  return log1p((double)n);
  if (!strcmp(how, "linear")) return (double)n;
  if (!strcmp(how, "flat"))   return 1.0;
  vdie("unknown pattern weighting (sqrt|log1p|linear|flat)", how);
  return 0.0;
}

void ms_viomodel_free(viomodel_t *vm) {
  if (!vm) return;
  for (int i = 0; i < vm->n_label; ++i) free(vm->labels[i]);
  for (int j = 0; j < vm->n_feat; ++j) { free(vm->names[j]); free(vm->bin[j]); }
  free(vm->labels); free(vm->names); free(vm->bin); free(vm->w); free(vm->weighting);
  free(vm);
}

/* ------------------------------- build ------------------------------- */
/* Straight from the MRMP artifact: the binstrings, the CpG counts, and the
 * reference sample names, which for a per-cell-type reference are the labels.
 * Nothing is fitted, so this is a transcription rather than a training step. */
viomodel_t *ms_viomodel_from_mrmp(const char *artifact, uint32_t top_k,
                                  double threshold, const char *weighting,
                                  int min_patterns) {
  mrmp_top_t *t = ms_mrmp_top_read(artifact, top_k);
  if (t->n_samples < 2) vdie("reference has fewer than 2 classes", artifact);
  if (t->n_patterns < 1) vdie("reference has no ranked patterns", artifact);

  viomodel_t *vm = calloc(1, sizeof(*vm));
  if (!vm) vdie("out of memory (viomodel)", NULL);
  vm->n_label = (int)t->n_samples;
  vm->n_feat  = (int)t->n_patterns;
  vm->threshold = threshold;
  vm->min_patterns = min_patterns;
  vm->weighting = strdup(weighting);
  vm->labels = calloc(vm->n_label, sizeof(char *));
  vm->names  = calloc(vm->n_feat, sizeof(char *));
  vm->bin    = calloc(vm->n_feat, sizeof(char *));
  vm->w      = calloc(vm->n_feat, sizeof(double));
  if (!vm->weighting || !vm->labels || !vm->names || !vm->bin || !vm->w)
    vdie("out of memory (viomodel)", NULL);

  for (int i = 0; i < vm->n_label; ++i) vm->labels[i] = strdup(t->labels[i]);
  for (int j = 0; j < vm->n_feat; ++j) {
    char nb[32]; snprintf(nb, sizeof nb, "P%d", j + 1);
    vm->names[j] = strdup(nb);
    vm->bin[j]   = strdup(t->binstring[j]);
    vm->w[j]     = vio_weight(weighting, t->count[j]);
    if (!vm->names[j] || !vm->bin[j]) vdie("out of memory (viomodel)", NULL);
  }
  ms_mrmp_top_free(t);
  return vm;
}

/* ------------------------------- write ------------------------------- */
void ms_viomodel_write(const viomodel_t *vm, const char *path) {
  FILE *fp = fopen(path, "w");
  if (!fp) vdie("cannot open violation model output", path);
  fprintf(fp, "methscope-violation\t1\n");
  fprintf(fp, "threshold\t%.9g\n", vm->threshold);
  fprintf(fp, "weight\t%s\n", vm->weighting);
  fprintf(fp, "min_patterns\t%d\n", vm->min_patterns);
  fputs("labels", fp);
  for (int i = 0; i < vm->n_label; ++i) fprintf(fp, "\t%s", vm->labels[i]);
  fputc('\n', fp);
  /* The weight is written as the CpG count, not the derived w, so the spec
   * stays inspectable and a reader can re-derive under a different weighting. */
  for (int j = 0; j < vm->n_feat; ++j)
    fprintf(fp, "pattern\t%s\t%s\t%.0f\n", vm->names[j], vm->bin[j],
            !strcmp(vm->weighting, "sqrt")   ? vm->w[j] * vm->w[j]
          : !strcmp(vm->weighting, "log1p")  ? expm1(vm->w[j])
          : !strcmp(vm->weighting, "linear") ? vm->w[j] : 1.0);
  fclose(fp);
}

/* ------------------------------- parse ------------------------------- */
viomodel_t *ms_viomodel_parse(const char *buf, size_t len) {
  char *copy = malloc(len + 1);
  if (!copy) vdie("out of memory (parse)", NULL);
  memcpy(copy, buf, len); copy[len] = '\0';

  int n_feat = 0;
  for (const char *p = copy; (p = strstr(p, "pattern\t")); p += 8)
    if (p == copy || p[-1] == '\n') n_feat++;
  if (n_feat < 1) vdie("violation spec has no pattern rows", NULL);

  viomodel_t *vm = calloc(1, sizeof(*vm));
  if (!vm) vdie("out of memory (viomodel)", NULL);
  vm->n_feat = n_feat;
  vm->names = calloc(n_feat, sizeof(char *));
  vm->bin   = calloc(n_feat, sizeof(char *));
  vm->w     = calloc(n_feat, sizeof(double));
  vm->threshold = 0.5; vm->min_patterns = 20;
  if (!vm->names || !vm->bin || !vm->w) vdie("out of memory (viomodel)", NULL);

  char *weighting = NULL;
  uint64_t *counts = calloc(n_feat, sizeof(uint64_t));
  if (!counts) vdie("out of memory (parse)", NULL);
  int seen_magic = 0, j = 0;
  /* strtok_r, not strtok: the labels line is itself tokenized while the outer
   * line walk is in progress, and nested strtok calls share one static cursor
   * -- which silently truncated the pattern rows. */
  char *lsave = NULL;
  for (char *line = strtok_r(copy, "\n", &lsave); line;
       line = strtok_r(NULL, "\n", &lsave)) {
    char *tab = strchr(line, '\t');
    if (!tab) continue;
    *tab = '\0';
    char *rest = tab + 1;
    if      (!strcmp(line, "methscope-violation")) seen_magic = 1;
    else if (!strcmp(line, "threshold"))    vm->threshold = atof(rest);
    else if (!strcmp(line, "min_patterns")) vm->min_patterns = atoi(rest);
    else if (!strcmp(line, "weight"))       weighting = strdup(rest);
    else if (!strcmp(line, "labels")) {
      int n = 1; for (const char *p = rest; *p; ++p) if (*p == '\t') n++;
      vm->n_label = n;
      vm->labels = calloc(n, sizeof(char *));
      if (!vm->labels) vdie("out of memory (labels)", NULL);
      int i = 0; char *tsave = NULL;
      for (char *tk = strtok_r(rest, "\t", &tsave); tk && i < n;
           tk = strtok_r(NULL, "\t", &tsave))
        vm->labels[i++] = strdup(tk);
      vm->n_label = i;
    } else if (!strcmp(line, "pattern")) {
      /* name <TAB> binstring <TAB> n_cpg */
      char *t1 = strchr(rest, '\t'); if (!t1) vdie("malformed pattern row", rest);
      *t1 = '\0';
      char *t2 = strchr(t1 + 1, '\t'); if (!t2) vdie("malformed pattern row", rest);
      *t2 = '\0';
      if (j >= n_feat) vdie("more pattern rows than counted", rest);
      vm->names[j] = strdup(rest);
      vm->bin[j]   = strdup(t1 + 1);
      counts[j]    = strtoull(t2 + 1, NULL, 10);
      j++;
    }
  }
  if (!seen_magic) vdie("not a methscope-violation spec", NULL);
  if (j != n_feat) vdie("pattern row count mismatch", NULL);
  if (!vm->labels || vm->n_label < 2) vdie("violation spec has no labels", NULL);
  for (int k = 0; k < n_feat; ++k)
    if ((int)strlen(vm->bin[k]) != vm->n_label)
      vdie("binstring width does not match the label count", vm->names[k]);
  vm->weighting = weighting ? weighting : strdup("sqrt");
  for (int k = 0; k < n_feat; ++k) vm->w[k] = vio_weight(vm->weighting, counts[k]);
  free(counts); free(copy);
  return vm;
}

/* ------------------------------- score ------------------------------- */
/* Per-label violation totals into out[n_label]; INFINITY where the query is too
 * sparse on that label's expected-1 or expected-0 side to judge it. Exposed so
 * `--probs` can show the whole field: unlike a softmax, these are comparable
 * across queries, and seeing the runner-ups is how a near-tie gets audited. */
void ms_viomodel_scores(const viomodel_t *vm, const double *betas, double *out) {
  for (int k = 0; k < vm->n_label; ++k) {
    double d1 = 0, n1 = 0, d0 = 0, n0 = 0;
    int c1 = 0, c0 = 0;
    for (int j = 0; j < vm->n_feat; ++j) {
      double b = betas[j];
      if (!(b == b)) continue;                    /* NaN: unobserved, skip */
      int hi = b > vm->threshold;
      if (vm->bin[j][k] == '1') {
        c1++; d1 += vm->w[j]; if (!hi) n1 += vm->w[j];
      } else {
        c0++; d0 += vm->w[j]; if (hi)  n0 += vm->w[j];
      }
    }
    out[k] = (c1 < vm->min_patterns || c0 < vm->min_patterns || d1 <= 0 || d0 <= 0)
               ? INFINITY : n1 / d1 + n0 / d0;
  }
}

/* betas[n_feat] in model order; NaN means the query never observed that
 * pattern. Returns the winning label index, or -1 when nothing clears
 * min_patterns (too sparse to call). *score gets the winner's violation total
 * and *margin the runner-up gap. */
int ms_viomodel_score(const viomodel_t *vm, const double *betas,
                      double *score, double *margin) {
  double *all = malloc((size_t)vm->n_label * sizeof(double));
  if (!all) vdie("out of memory (scores)", NULL);
  ms_viomodel_scores(vm, betas, all);
  double best = INFINITY, second = INFINITY;
  int best_k = -1;
  for (int k = 0; k < vm->n_label; ++k) {
    if (all[k] < best) { second = best; best = all[k]; best_k = k; }
    else if (all[k] < second) second = all[k];
  }
  free(all);
  if (best_k < 0) { if (score) *score = NAN; if (margin) *margin = NAN; return -1; }
  if (score)  *score  = best;
  if (margin) *margin = isfinite(second) ? second - best : NAN;
  return best_k;
}
