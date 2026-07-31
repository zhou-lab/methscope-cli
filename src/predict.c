// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Label prediction (the C replacement for the R PredictCellType()). The label
 * can be any class the model was trained on (cell type, sex, ...). Builds the
 * record x pattern matrix from the query and the MRMP reference (<ref.mrmp>, a
 * YAME .cm), runs the XGBoost booster (with class labels embedded as
 * attributes), and reports a per-record label plus a Shannon-entropy confidence.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "methscope.h"
#include "bmeta.h"
#include "bundle.h"
#include "msfm.h"    /* ms_msfm_to_matrix -- the --data feature path */
#include "index.h"   /* get_fname_index -- --threads needs the .cg index */
#include <xgboost/c_api.h>

#define XGCHK(call) do {                                            \
    if ((call) != 0) {                                              \
      fprintf(stderr, "[methscope] xgboost error: %s\n",           \
              XGBGetLastError());                                   \
      exit(1);                                                      \
    }                                                               \
  } while (0)

static void pdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] classify: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] classify: %s\n", msg);
  exit(1);
}

/* Shannon-entropy confidence, matching R confidence_score():
 * 1 - (-sum p*log(p+1e-10)) / log(K), clamped to [0,1]. */
static double confidence_score(const float *p, int K) {
  if (K <= 1) return 1.0;
  double entropy = 0.0;
  for (int c = 0; c < K; ++c) entropy -= p[c] * log(p[c] + 1e-10);
  double conf = 1.0 - entropy / log((double)K);
  if (conf < 0.0) conf = 0.0;
  if (conf > 1.0) conf = 1.0;
  return conf;
}

static int predict_usage(void) {
  ms_help(stderr,
    "\n"
    "Usage:\n"
    "  methscope classify [options] <query.cg> <model.ubjx>\n"
    "  methscope classify [options] <query.cg> <ref.mrmp> <booster.ubj>\n"
    "  methscope classify --data <in.msfm> [options] <model.ubjx>\n"
    "\n"
    "Purpose:\n"
    "  Predict a label (cell type, sex, ... — whatever the model was trained on)\n"
    "  and a confidence score for each query record, by featurizing the query\n"
    "  against the MRMP reference and running the booster.\n"
    "\n"
    "Arguments:\n"
    "  <query.cg>     Query methylome(s); '-' reads a .cg stream from stdin\n"
    "                 (cells are then named 1,2,3,... as a stream has no index).\n"
    "  <model.ubjx>   A self-contained bundle of the booster + its MRMP (from\n"
    "                 `train -o model.ubjx` or `bundle`) — the recommended\n"
    "                 single-file form.\n"
    "  <ref.mrmp>     MRMP pattern definition (a YAME .cm) to featurize the query\n"
    "                 (loose form). A bundle (.ubjx/.updecx) also works here.\n"
    "  <booster.ubj>  Loose booster with class labels embedded (see train / bundle -l).\n"
    "\n"
    "Options:\n"
    "  -o <out.tsv>   Write output to a file instead of stdout.\n"
    "  --probs        Append one column per class with its predicted probability.\n"
    "  --levels       Append the predicted label's whole taxonomy path\n"
    "                 (compartment, lineage, group, subtype).\n"
    "  --level NAME   Append just one level -- compartment | lineage | group |\n"
    "                 subtype. Use this to score against a cohort labelled only\n"
    "                 that coarsely: collapse the prediction to the level the\n"
    "                 truth resolves to, and compare there.\n"
    "                 Both need a model trained with `classify-train --hierarchy`.\n"
    "  --no-header    Suppress the header line.\n"
    "  --threads T    Featurize with T workers (default 1). Cells are partitioned\n"
    "                 across workers, each seeking its own records, so this needs\n"
    "                 the query's .cg index; a stream keeps the serial path.\n"
    "  --data <in.msfm>  Score a prebuilt feature artifact (classify-featurize)\n"
    "                 instead of featurizing <query.cg>. <query.cg> is then omitted.\n"
    "                 Featurization dominates the cost, so this is how to score many\n"
    "                 models on one test set without repeating it. xgboost only.\n"
    "  -h             Show this help message.\n"
    "\n"
    "Output columns:\n"
    "  cell  prediction_label  confidence  [<class1> <class2> ... with --probs]\n"
    "\n");
  return 1;
}

/* Inference for the linear frameworks (threshold / logistic): featurize the
 * query against the bundled mrmp, then score each record with the linear model. */
static int predict_linear(const char *query_cg, const char *ref_mrmp,
                          void *model_buf, size_t model_len,
                          const char *out_path, int with_probs, int no_header) {
  linmodel_t *lm = ms_linmodel_parse(model_buf, model_len);
  ms_matrix_t *m = ms_matrix_build(query_cg, ref_mrmp);
  if (m->n_patterns < lm->n_feat)
    pdie("reference .mrmp has fewer patterns than the model expects", ref_mrmp);
  FILE *fout = (out_path && strcmp(out_path, "-") != 0)
                 ? fopen(out_path, "w") : stdout;
  if (!fout) pdie("cannot open output", out_path);
  if (!no_header) {
    fputs("cell\tprediction_label\tconfidence", fout);
    if (with_probs) fprintf(fout, "\t%s\t%s", lm->label0, lm->label1);
    fputc('\n', fout);
  }
  double *betas = malloc((size_t)lm->n_feat * sizeof(double));
  if (!betas) pdie("out of memory (linear betas)", NULL);
  for (int r = 0; r < m->n_cells; ++r) {
    const double *src = m->M + (size_t)r * m->n_patterns;
    for (int c = 0; c < lm->n_feat; ++c) betas[c] = src[c];
    double p1, conf;
    int cls = ms_linmodel_score(lm, betas, &p1, &conf);
    fprintf(fout, "%s\t%s\t%.6f", m->cell_names[r], cls ? lm->label1 : lm->label0, conf);
    if (with_probs) fprintf(fout, "\t%.6f\t%.6f", 1.0 - p1, p1);
    fputc('\n', fout);
  }
  free(betas);
  if (fout != stdout) fclose(fout);
  ms_matrix_free(m); ms_linmodel_free(lm);
  return 0;
}

int main_predict(int argc, char *argv[]) {
  const char *out_path = NULL, *data_path = NULL;
  unsigned threads = 1;
  int with_probs = 0, with_levels = 0;
  const char *one_level = NULL;
  int no_header = 0;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) data_path = argv[++i];
    else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
      threads = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--probs") == 0) with_probs = 1;
    else if (strcmp(argv[i], "--levels") == 0) with_levels = 1;
    else if (strcmp(argv[i], "--level") == 0 && i + 1 < argc) one_level = argv[++i];
    else if (strcmp(argv[i], "--no-header") == 0) no_header = 1;
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      predict_usage(); return 0;
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      pdie("unrecognized or incomplete option", argv[i]);
    else break;
  }
  /* With --data the features are prebuilt, so <query.cg> drops out of the
   * positional list; the model forms are otherwise unchanged. This is what
   * makes scoring many models on one test set cheap -- the featurization,
   * which dominates, happens once in classify-featurize instead of per run. */
  if (data_path) { if (argc - i != 1 && argc - i != 2) return predict_usage(); }
  else           { if (argc - i != 2 && argc - i != 3) return predict_usage(); }
  if (data_path) --i;                 /* so argv[i+1], argv[i+2] stay the model */
  const char *query_cg  = data_path ? NULL : argv[i];
  const char *ref_mrmp  = NULL;     /* mrmp path (loose arg, or the bundle path itself) */
  const char *model_name;           /* for error messages */
  char *tmp_mrmp = NULL;            /* ms_mrmp_resolve temp (always NULL now; kept for API) */

  BoosterHandle booster = NULL;

  if (argc - i == 2) {
    /* bundle form: query.cg model.ubjx (model + mrmp in one file) */
    model_name = argv[i + 1];
    if (!ms_bundle_is(model_name))
      pdie("expected a .ubjx bundle; for a bare booster give <ref.mrmp> <booster.ubj>", model_name);
    char *kind = ms_bundle_kind(model_name);      /* framework mark is REQUIRED */
    if (!kind)
      pdie("bundle has no framework 'kind' mark — regenerate it with a current "
           "train (or stamp one with 'bundle -k')", model_name);
    ref_mrmp = model_name;   /* the bundle's front bytes ARE the mrmp .cm */

    if (strcmp(kind, "threshold") == 0 || strcmp(kind, "logistic") == 0) {
      /* linear frameworks: score + return */
      if (data_path)
        pdie("--data is xgboost-only; the linear frameworks featurize their own "
             "query", model_name);
      size_t blen; void *bbuf = ms_bundle_section(model_name, "model", &blen);
      int rc = predict_linear(query_cg, ref_mrmp, bbuf, blen, out_path, with_probs, no_header);
      free(bbuf); free(kind);
      return rc;
    }
    if (strcmp(kind, "xgboost") != 0)
      pdie("unknown model framework 'kind' (expected xgboost/threshold/logistic)", kind);
    free(kind);
    size_t blen; void *bbuf = ms_bundle_section(model_name, "model", &blen);
    XGCHK(XGBoosterCreate(NULL, 0, &booster));
    XGCHK(XGBoosterLoadModelFromBuffer(booster, bbuf, blen));
    free(bbuf);
  } else {
    /* loose form: query.cg ref.mrmp booster.ubj (ref.mrmp may itself be a
     * bundle, e.g. reuse a .ubjx's mrmp with a different booster). */
    ref_mrmp   = ms_mrmp_resolve(argv[i + 1], &tmp_mrmp);
    model_name = argv[i + 2];
    if (ms_bundle_is(model_name))
      pdie("got a bundle where a bare booster.ubj was expected; use: predict query.cg model.ubjx", model_name);
    XGCHK(XGBoosterCreate(NULL, 0, &booster));
    XGCHK(XGBoosterLoadModel(booster, model_name));
  }

  bst_ulong num_feature = 0;
  XGCHK(XGBoosterGetNumFeature(booster, &num_feature));
  int P = (int)num_feature;                 /* booster's total feature count */

  /* Models trained with --scalar-coverage carry ONE extra input after the
   * pattern columns, so the pattern count is one less than num_feature. The
   * model says so itself; guessing here would shift every column by one. */
  int scalar_cov = ms_booster_has_scalar_cov(booster);
  int n_pat = scalar_cov ? P - 1 : P;
  if (n_pat < 1) pdie("booster has no pattern features", model_name);

  int K = 0;
  char **labels = ms_booster_get_labels(booster, &K);  /* NULL if not annotated */

  /* Featurizing the query is the whole cost of scoring a .cg, and it is per
   * record, so it threads. The threaded builder needs the .cg index to seek, so
   * a stream ('-') or an unindexed file keeps the serial path -- and says so,
   * rather than silently running 16x slower than asked. */
  uint32_t *levels = NULL;
  ms_matrix_t *m;
  if (data_path) {
    m = ms_msfm_to_matrix(data_path, NULL, &levels);
  } else if (threads > 1) {
    char *fidx = get_fname_index((char *)query_cg);
    int have_idx = fidx && access(fidx, R_OK) == 0;
    free(fidx);
    if (have_idx) {
      m = ms_matrix_build_threaded(query_cg, ref_mrmp, (uint32_t)n_pat, threads, &levels);
    } else {
      fprintf(stderr, "[methscope] classify: --threads needs a .cg index; "
              "falling back to the single-threaded scan\n");
      m = ms_matrix_build(query_cg, ref_mrmp);
    }
  } else {
    m = ms_matrix_build(query_cg, ref_mrmp);
  }
  if (m->n_patterns < n_pat)
    pdie("reference .mrmp has fewer patterns than the booster expects", ref_mrmp);

  /* Pack the first n_pat columns into a float matrix; NaN stays missing. */
  bst_ulong nrow = (bst_ulong)m->n_cells;
  bst_ulong ncol = (bst_ulong)P;
  float *data = malloc((size_t)nrow * ncol * sizeof(float));
  if (!data) pdie("out of memory (predict matrix)", NULL);
  for (int r = 0; r < m->n_cells; ++r) {
    const double *src = m->M + (size_t)r * m->n_patterns;
    float        *dst = data + (size_t)r * P;
    for (int c = 0; c < n_pat; ++c) dst[c] = (float)src[c]; /* NaN -> NaN */
    if (scalar_cov) {
      /* Same quantity classify-featurize stores: covered CpGs over every
       * pattern. Recomputed from N when scoring a .cg directly, so the two
       * input paths agree. */
      double total = 0;
      if (levels) total = (double)levels[r];
      else {
        const int *n = m->N + (size_t)r * m->n_patterns;
        for (int c = 0; c < m->n_patterns; ++c) if (n[c] > 0) total += n[c];
      }
      dst[n_pat] = (float)log1p(total);
    }
  }

  DMatrixHandle  dmat;
  XGCHK(XGDMatrixCreateFromMat(data, nrow, ncol, NAN, &dmat));

  bst_ulong    out_len;
  const float *out;
  XGCHK(XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out));
  int K_pred = (int)(out_len / nrow);       /* classes the booster actually emits */
  if (out_len != nrow * (bst_ulong)K_pred || K_pred < 1)
    pdie("unexpected prediction length", model_name);
  if (labels && K != K_pred)
    pdie("embedded label count does not match the booster's num_class", model_name);
  K = K_pred;

  /* Taxonomy path per class, parsed from the model's own attribute so a
   * prediction is self-describing and needs no side table. */
  static const char *LEVEL_NAME[4] = {"compartment", "lineage", "group", "subtype"};
  int lvl_col = -1;
  char **hier = NULL;              /* K * 4 entries, class-index order */
  if (with_levels || one_level) {
    if (one_level) {
      for (int c = 0; c < 4; ++c)
        if (!strcmp(one_level, LEVEL_NAME[c])) lvl_col = c;
      if (lvl_col < 0)
        pdie("--level must be compartment, lineage, group or subtype", one_level);
    }
    char *raw = ms_booster_get_hier(booster);
    if (!raw)
      pdie("model carries no hierarchy; retrain with `classify-train --hierarchy`",
           model_name);
    hier = calloc((size_t)K * 4, sizeof(char *));
    if (!hier) pdie("out of memory (hierarchy)", NULL);
    char *save = NULL;
    for (char *line = strtok_r(raw, "\n", &save); line;
         line = strtok_r(NULL, "\n", &save)) {
      char *f[5] = {0}; int nf = 0;
      for (char *p2 = line; nf < 5; ) {
        f[nf++] = p2;
        char *t = strchr(p2, '\t');
        if (!t) break;
        *t = 0; p2 = t + 1;
      }
      if (nf < 5 || !labels) continue;
      for (int c = 0; c < K; ++c)
        if (!strcmp(labels[c], f[0]))
          for (int j = 0; j < 4; ++j) hier[(size_t)c * 4 + j] = strdup(f[j + 1]);
    }
    free(raw);
    /* a class the table omits reports NA rather than an empty column */
    for (size_t k = 0; k < (size_t)K * 4; ++k)
      if (!hier[k]) hier[k] = strdup("NA");
  }

  /* Without embedded labels (un-annotated booster), fall back to numeric names. */
  char numbuf[16];
  #define LABEL(c) (labels ? labels[c] : (snprintf(numbuf, sizeof(numbuf), "%d", (c)), numbuf))

  FILE *fout = (out_path && strcmp(out_path, "-") != 0)
                 ? fopen(out_path, "w") : stdout;
  if (!fout) pdie("cannot open output", out_path);

  if (!no_header) {
    fputs("cell\tprediction_label\tconfidence", fout);
    if (with_levels)
      for (int c = 0; c < 4; ++c) fprintf(fout, "\t%s", LEVEL_NAME[c]);
    else if (lvl_col >= 0) fprintf(fout, "\t%s", LEVEL_NAME[lvl_col]);
    if (with_probs)
      for (int c = 0; c < K; ++c) fprintf(fout, "\t%s", LABEL(c));
    fputc('\n', fout);
  }

  for (int r = 0; r < m->n_cells; ++r) {
    const float *p = out + (size_t)r * K;
    int   arg  = 0;
    float best = -1.0f;
    for (int c = 0; c < K; ++c) if (p[c] >= best) { best = p[c]; arg = c; } /* ties -> last */
    double conf = confidence_score(p, K);
    fprintf(fout, "%s\t%s\t%.6f", m->cell_names[r], LABEL(arg), conf);
    if (with_levels)
      for (int c = 0; c < 4; ++c) fprintf(fout, "\t%s", hier[(size_t)arg * 4 + c]);
    else if (lvl_col >= 0) fprintf(fout, "\t%s", hier[(size_t)arg * 4 + lvl_col]);
    if (with_probs)
      for (int c = 0; c < K; ++c) fprintf(fout, "\t%.6f", p[c]);
    fputc('\n', fout);
  }
  #undef LABEL

  if (fout != stdout) fclose(fout);
  free(data); free(levels);
  XGDMatrixFree(dmat);
  XGBoosterFree(booster);
  if (labels) { for (int c = 0; c < K; ++c) free(labels[c]); free(labels); }
  ms_matrix_free(m);
  if (tmp_mrmp) { unlink(tmp_mrmp); free(tmp_mrmp); }
  return 0;
}
