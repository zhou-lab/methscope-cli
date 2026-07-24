// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Minimal model training (the C counterpart of R Input_training() without the
 * caret grid search). Builds the record x pattern matrix from query.cg +
 * <ref.mrmp>, reads a per-record label list (any label: cell type, sex, ...),
 * trains an XGBoost multiclass booster with fixed hyperparameters, embeds the
 * class labels as booster attributes, and writes a self-describing <out.ubjx>.
 *
 * Defaults mirror Input_training(): objective multi:softprob, eval_metric
 * mlogloss, gbtree, nrounds = round(sqrt(n_cells)).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "methscope.h"
#include "bmeta.h"
#include "bundle.h"    /* ms_mrmp_resolve / ms_bundle_pack / ms_path_is_bundle_ext */
#include <xgboost/c_api.h>

#define XGCHK(call) do {                                          \
    if ((call) != 0) {                                            \
      fprintf(stderr, "[methscope] xgboost error: %s\n",         \
              XGBGetLastError());                                 \
      exit(1);                                                    \
    }                                                             \
  } while (0)

static void tdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] train: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] train: %s\n", msg);
  exit(1);
}

/* read one label per line (trimmed); returns array of n strings */
static char **read_labels(const char *path, int *n_out) {
  FILE *fp = fopen(path, "r");
  if (!fp) tdie("cannot open labels file", path);
  int   cap = 256, n = 0;
  char **v = malloc(cap * sizeof(char *));
  char  *line = NULL; size_t cap2 = 0; ssize_t len;
  while ((len = getline(&line, &cap2, fp)) != -1) {
    while (len && (line[len-1] == '\n' || line[len-1] == '\r' || line[len-1] == ' ' || line[len-1] == '\t'))
      line[--len] = '\0';
    char *s = line; while (*s == ' ' || *s == '\t') s++;
    if (*s == '\0') continue;                 /* skip blank lines */
    if (n == cap) { cap *= 2; v = realloc(v, cap * sizeof(char *)); }
    v[n++] = strdup(s);
  }
  free(line); fclose(fp);
  *n_out = n;
  return v;
}

static int cmp_str(const void *a, const void *b) {
  return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Bundle an inner model file (booster .ubj or linear spec) into out (.ubjx),
 * marked with `kind`, alongside the MRMP. The MRMP is TRIMMED to exactly the
 * npattern feature states (others folded into "Pna") when trimming drops real
 * patterns, so predict featurizes the same set. Returns 1 if trimmed. */
static int bundle_model(const char *out, const char *kind, const char *inner_tmp,
                        const char *ref_mrmp, char *const *keep_names,
                        int npattern, int n_nonpna) {
  char *trim_tmp = NULL;
  const char *bundle_mrmp = ref_mrmp;
  int trimmed = 0;
  if (npattern < n_nonpna) {                   /* real patterns dropped -> trim */
    char tmpl[] = "/tmp/methscope_trim_XXXXXX.cm";
    int fd = mkstemps(tmpl, 3);                /* keep the .cm suffix */
    if (fd < 0) tdie("cannot create temp trimmed mrmp", NULL);
    close(fd);
    ms_mrmp_trim(ref_mrmp, keep_names, npattern, tmpl);
    trim_tmp = strdup(tmpl); bundle_mrmp = trim_tmp; trimmed = 1;
  }
  ms_bundle_pack(out, kind, inner_tmp, bundle_mrmp, NULL);
  if (trim_tmp) { unlink(trim_tmp); free(trim_tmp); }
  return trimmed;
}

static int train_usage(void) {
  fprintf(stderr,
    "\n"
    "Usage:\n"
    "  methscope train -l <labels.txt> -o <out.ubjx> [options] <query.cg> <ref.mrmp>\n"
    "\n"
    "Purpose:\n"
    "  Train a multiclass classifier for any per-record label (cell type, sex,\n"
    "  ...; fixed hyperparameters, no grid search) and write a self-describing\n"
    "  model with the class labels embedded.\n"
    "\n"
    "Arguments:\n"
    "  <query.cg>   Training methylome(s), one record per sample.\n"
    "  <ref.mrmp>   MRMP pattern definition (a YAME .cm) to featurize the query.\n"
    "               A bundle (.ubjx/.updecx) also works; its MRMP is used.\n"
    "               Features are the MRMP states; the 'Pna' NA-background state is\n"
    "               excluded by default (see --include-pna).\n"
    "\n"
    "Options:\n"
    "  -l <labels.txt>  One label per query record, in query order (required).\n"
    "  -o <out.ubjx>    Output model path (required). A '.ubjx' name writes a\n"
    "                   self-contained bundle (booster + MRMP) that `predict` can\n"
    "                   run directly; a plain '.ubj' writes just the loose booster.\n"
    "                   The bundled MRMP is TRIMMED to exactly the patterns used\n"
    "                   (others folded into 'Pna'), and `predict` uses that same set.\n"
    "  -p <npattern>    Use the first N patterns by RECURRENCE RANK. Only meaningful\n"
    "                   when patterns are named P1,P2,... (auto MRMPs); for curated\n"
    "                   named markers (e.g. Xa_hi/Xa_lo) leave it unset. Default:\n"
    "                   all non-'Pna' states.\n"
    "  --include-pna    Also use the 'Pna' NA-background state as a feature\n"
    "                   (default: excluded).\n"
    "  --framework <f>  Model framework (default: xgboost):\n"
    "                     xgboost    gradient-boosted trees (multiclass).\n"
    "                     threshold  interpretable binary linear rule; per-feature\n"
    "                                weight = class-mean difference, decision at the\n"
    "                                midpoint of the two class score-centroids.\n"
    "                     logistic   binary L2-regularized logistic regression.\n"
    "                   threshold/logistic are binary-only and require a .ubjx output.\n"
    "  -n <nrounds>     Boosting rounds, xgboost only (default: round(sqrt(n_cells))).\n"
    "  -h               Show this help message.\n"
    "\n");
  return 1;
}

int main_train(int argc, char *argv[]) {
  const char *labels_path = NULL, *out_path = NULL, *framework = "xgboost";
  int npattern = 0, nrounds = 0, include_pna = 0;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-l") == 0 && i+1 < argc) labels_path = argv[++i];
    else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_path    = argv[++i];
    else if (strcmp(argv[i], "-p") == 0 && i+1 < argc) npattern    = atoi(argv[++i]);
    else if (strcmp(argv[i], "--include-pna") == 0)    include_pna = 1;
    else if (strcmp(argv[i], "--framework") == 0 && i+1 < argc) framework = argv[++i];
    else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) nrounds     = atoi(argv[++i]);
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      train_usage(); return 0;
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      tdie("unrecognized or incomplete option", argv[i]);
    else break;
  }
  if (!labels_path || !out_path || argc - i != 2) return train_usage();
  int fw_xgb = strcmp(framework, "xgboost") == 0;
  int fw_lin = (strcmp(framework, "threshold") == 0) || (strcmp(framework, "logistic") == 0);
  if (!fw_xgb && !fw_lin) tdie("unknown --framework (xgboost|threshold|logistic)", framework);
  if (fw_lin && !ms_path_is_bundle_ext(out_path))
    tdie("threshold/logistic frameworks require a .ubjx output (bundled with the MRMP)", out_path);
  const char *query_cg = argv[i];
  char *tmp_mrmp = NULL;
  const char *ref_mrmp = ms_mrmp_resolve(argv[i + 1], &tmp_mrmp);

  /* ---- features ---- */
  ms_matrix_t *m = ms_matrix_build(query_cg, ref_mrmp);

  /* "Pna" is the NA-background state (matrix sorts it last). Count it so we can
   * exclude it from features by default. */
  int n_pna = 0;
  for (int c = 0; c < m->n_patterns; ++c)
    if (strcmp(m->pattern_names[c], "Pna") == 0) n_pna++;
  int n_nonpna = m->n_patterns - n_pna;

  /* Default npattern = all feature states: every non-"Pna" state, or all states
   * with --include-pna. A user -p N selects the first N columns by recurrence
   * rank (P1,P2,...); for curated named markers the default is what you want. */
  if (npattern <= 0)
    npattern = include_pna ? m->n_patterns : n_nonpna;
  if (npattern <= 0 || npattern > m->n_patterns)
    tdie("invalid npattern", NULL);

  /* ---- labels ---- */
  int n_lab = 0;
  char **lab = read_labels(labels_path, &n_lab);
  if (n_lab != m->n_cells)
    tdie("label count does not match number of query cells", labels_path);

  /* class set = sorted unique labels (deterministic class indices) */
  char **uniq = malloc(n_lab * sizeof(char *));
  for (int j = 0; j < n_lab; ++j) uniq[j] = lab[j];
  qsort(uniq, n_lab, sizeof(char *), cmp_str);
  int K = 0;
  for (int j = 0; j < n_lab; ++j)
    if (j == 0 || strcmp(uniq[j], uniq[K-1]) != 0) uniq[K++] = uniq[j];
  if (K < 2) tdie("need at least two classes", NULL);

  if (fw_lin && K != 2)
    tdie("threshold/logistic frameworks are binary (need exactly 2 classes)", NULL);

  /* per-record class index (0..K-1) = position in the sorted unique set */
  int *yidx = malloc((size_t)m->n_cells * sizeof(int));
  for (int r = 0; r < m->n_cells; ++r) {
    int lo = 0, hi = K - 1, idx = -1;
    while (lo <= hi) { int mid = (lo+hi)/2; int cmp = strcmp(lab[r], uniq[mid]);
      if (cmp == 0) { idx = mid; break; } else if (cmp < 0) hi = mid-1; else lo = mid+1; }
    if (idx < 0) tdie("label not found in class set (internal)", lab[r]);
    yidx[r] = idx;
  }

  int bundled = ms_path_is_bundle_ext(out_path);
  int trimmed = 0;

  if (fw_lin) {
    /* ---- linear framework (threshold / logistic): interpretable binary rule ---- */
    linmodel_t *lm = ms_linmodel_fit(m, npattern, yidx, uniq[0], uniq[1], framework);
    char tmpl[] = "/tmp/methscope_lin_XXXXXX.lin";
    int fd = mkstemps(tmpl, 4);
    if (fd < 0) tdie("cannot create temp linear model file", NULL);
    close(fd);
    ms_linmodel_write(lm, tmpl);
    trimmed = bundle_model(out_path, framework, tmpl, ref_mrmp,
                           m->pattern_names, npattern, n_nonpna);
    unlink(tmpl);
    ms_linmodel_free(lm);
    fprintf(stderr, "[methscope] trained %s model (2-class) on %d cells x %d feature(s) "
                    "-> %s (linear+MRMP bundle%s)\n", framework, m->n_cells, npattern,
                    out_path, trimmed ? ", trimmed mrmp" : "");
  } else {
    /* ---- xgboost framework ---- */
    float *ylab = malloc((size_t)m->n_cells * sizeof(float));
    for (int r = 0; r < m->n_cells; ++r) ylab[r] = (float)yidx[r];
    bst_ulong nrow = (bst_ulong)m->n_cells, ncol = (bst_ulong)npattern;
    float *data = malloc((size_t)nrow * ncol * sizeof(float));
    for (int r = 0; r < m->n_cells; ++r) {
      const double *src = m->M + (size_t)r * m->n_patterns;
      float        *dst = data + (size_t)r * npattern;
      for (int c = 0; c < npattern; ++c) dst[c] = (float)src[c];
    }
    if (nrounds <= 0) nrounds = (int)(sqrt((double)m->n_cells) + 0.5);
    if (nrounds < 1) nrounds = 1;

    DMatrixHandle dtrain; BoosterHandle booster;
    XGCHK(XGDMatrixCreateFromMat(data, nrow, ncol, NAN, &dtrain));
    XGCHK(XGDMatrixSetFloatInfo(dtrain, "label", ylab, nrow));
    XGCHK(XGBoosterCreate(&dtrain, 1, &booster));
    char kbuf[16]; snprintf(kbuf, sizeof(kbuf), "%d", K);
    XGCHK(XGBoosterSetParam(booster, "booster", "gbtree"));
    XGCHK(XGBoosterSetParam(booster, "objective", "multi:softprob"));
    XGCHK(XGBoosterSetParam(booster, "eval_metric", "mlogloss"));
    XGCHK(XGBoosterSetParam(booster, "num_class", kbuf));
    for (int it = 0; it < nrounds; ++it)
      XGCHK(XGBoosterUpdateOneIter(booster, it, dtrain));

    /* embed labels; save as loose .ubj or a (trimmed-mrmp) .ubjx bundle */
    ms_booster_set_meta(booster, uniq, K);
    if (bundled) {
      char tmpl[] = "/tmp/methscope_ubj_XXXXXX.ubj";
      int fd = mkstemps(tmpl, 4);
      if (fd < 0) tdie("cannot create temp booster file", NULL);
      close(fd);
      XGCHK(XGBoosterSaveModel(booster, tmpl));
      trimmed = bundle_model(out_path, "xgboost", tmpl, ref_mrmp,
                             m->pattern_names, npattern, n_nonpna);
      unlink(tmpl);
    } else {
      XGCHK(XGBoosterSaveModel(booster, out_path));
    }
    fprintf(stderr, "[methscope] trained %d-class xgboost model on %d cells x %d feature(s), "
                    "%d rounds -> %s%s%s\n", K, m->n_cells, npattern, nrounds, out_path,
                    bundled ? " (booster+MRMP bundle" : "",
                    bundled ? (trimmed ? ", trimmed mrmp)" : ")") : "");
    XGDMatrixFree(dtrain); XGBoosterFree(booster);
    free(data); free(ylab);
  }

  free(yidx); free(uniq);
  for (int j = 0; j < n_lab; ++j) free(lab[j]);
  free(lab);
  ms_matrix_free(m);
  ms_mrmp_cleanup(tmp_mrmp);
  return 0;
}
