// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Minimal model training (the C counterpart of R Input_training() without the
 * caret grid search). Builds the record x pattern matrix from query.cg +
 * <ref.mrmp>, reads a per-record label list (any label: cell type, sex, ...),
 * trains an XGBoost multiclass booster with fixed hyperparameters, embeds the
 * class labels as booster attributes, and writes a self-describing <out.clfx>.
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
#include "msfm.h"      /* ms_msfm_to_matrix -- the --data feature path */
#include "mrmp.h"      /* ms_mrmp_is_artifact / ms_mrmp_write_mask (violation) */
#include <xgboost/c_api.h>

#define XGCHK(call) do {                                          \
    if ((call) != 0) {                                            \
      fprintf(stderr, "[methscope] xgboost error: %s\n",         \
              XGBGetLastError());                                 \
      exit(1);                                                    \
    }                                                             \
  } while (0)

static void tdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] classify-train: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] classify-train: %s\n", msg);
  exit(1);
}

/* Parse a non-negative integer flag value; reject negatives, non-numeric
 * tokens, and trailing garbage (mirrors the validated parse in src/ui.c). */
static int parse_nonneg_int(const char *s, const char *errmsg) {
  char *e = NULL;
  long v = strtol(s, &e, 10);
  if (e == s || *e != '\0' || v < 0) tdie(errmsg, s);
  return (int)v;
}

/* read one label per line (trimmed); returns array of n strings */
static char **read_labels(const char *path, int *n_out) {
  FILE *fp = fopen(path, "r");
  if (!fp) tdie("cannot open labels file", path);
  int   cap = 256, n = 0;
  char **v = malloc(cap * sizeof(char *));
  if (!v) tdie("out of memory", NULL);
  char  *line = NULL; size_t line_cap = 0; ssize_t len;
  while ((len = getline(&line, &line_cap, fp)) != -1) {
    while (len && (line[len-1] == '\n' || line[len-1] == '\r' || line[len-1] == ' ' || line[len-1] == '\t'))
      line[--len] = '\0';
    char *s = line; while (*s == ' ' || *s == '\t') s++;
    if (*s == '\0') continue;                 /* skip blank lines */
    if (n == cap) {
      cap *= 2;
      char **tmp = realloc(v, cap * sizeof(char *));
      if (!tmp) tdie("out of memory", NULL);
      v = tmp;
    }
    v[n++] = strdup(s);
  }
  free(line); fclose(fp);
  *n_out = n;
  return v;
}

static int cmp_str(const void *a, const void *b) {
  return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Bundle an inner model file (booster .ubj or linear spec) into out (.clfx),
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
  ms_help(stderr,
    "\n"
    "Usage:\n"
    "  methscope classify-train -l <labels.txt> -o <out.clfx> [options] <query.cg> <ref.cm>\n"
    "  methscope classify-train --data <in.msfm> -o <out.clfx> [options] <ref.cm>\n"
    "  methscope classify-train --framework violation -o <out.clfx> [options] <ref.mrmp>\n"
    "\n"
    "Purpose:\n"
    "  Train a multiclass classifier for any per-record label (cell type, sex,\n"
    "  ...; fixed hyperparameters, no grid search) and write a self-describing\n"
    "  model with the class labels embedded.\n"
    "\n"
    "Arguments:\n"
    "  <query.cg>   Training methylome(s), one record per sample.\n"
    "  <ref.cm>     MRMP pattern definition, the runtime .cm from mrmp-export.\n"
    "               A bundle (.clfx/.updecx) also works; its MRMP is used.\n"
    "               Features are the MRMP states; the 'Pna' NA-background state is\n"
    "               excluded by default (see --include-pna).\n"
    "\n"
    "Options:\n"
    "  -l <labels.txt>  One label per query record, in query order (required).\n"
    "  -o <out.clfx>    Output model path (required). A '.clfx' name writes a\n"
    "                   self-contained bundle (model + MRMP) that `classify` can\n"
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
    "                     violation  multiclass, UNFITTED. Calls the class the query\n"
    "                                contradicts least, straight from the MRMP's own\n"
    "                                binstrings. Takes ONE positional, <ref.mrmp>:\n"
    "                                no training data, no -l, nothing to overfit.\n"
    "                   threshold/logistic are binary-only; all three need a .clfx out.\n"
    "  --call-threshold <t>    violation only: beta cutoff for calling a pattern\n"
    "                   methylated (default 0.5).\n"
    "  --pattern-weight <w>    violation only: sqrt|log1p|linear|flat weighting of each\n"
    "                   pattern by its CpG count (default sqrt). CpGs inside one MRMP\n"
    "                   are strongly correlated, so effective sample size saturates and\n"
    "                   'linear' lets the largest patterns decide every call.\n"
    "  --min-patterns <n>      violation only: observed patterns required on each of\n"
    "                   the expected-1 and expected-0 sides before a call (default 20);\n"
    "                   below it the record is reported NA rather than guessed.\n"
    "  -n <nrounds>     Boosting rounds, xgboost only (default: round(sqrt(n_cells))).\n"
    "  --max-depth <d>  Cap tree depth (xgboost default 6). Lower it when the\n"
    "                   training set is redundant -- e.g. one pseudobulk repeated\n"
    "                   across a coverage ladder -- since the repeats inflate\n"
    "                   split confidence and let one feature decide a class.\n"
    "  --min-child-weight <w>  Minimum child weight (xgboost default 1).\n"
    "  --colsample <f>  Per-tree feature subsample fraction, 0-1 (default 1).\n"
    "  --data <in.msfm> Train from a prebuilt feature artifact (classify-featurize)\n"
    "                   instead of featurizing <query.cg> here. Only <ref.cm> is then\n"
    "                   positional, and labels come from the artifact unless -l is\n"
    "                   given. Featurization is single-threaded, so this is how to\n"
    "                   train repeatedly on the same cells without repeating it.\n"
    "  --hierarchy TSV  Embed a label taxonomy in the model (label, compartment,\n"
    "                   lineage, group, subtype). `classify --levels` then reports\n"
    "                   the path, and an external cohort labelled only coarsely can\n"
    "                   still be scored at the level both sides resolve.\n"
    "  --scalar-coverage  Append ONE extra feature, log1p(covered CpGs) for the\n"
    "                   record, after the pattern columns -- the classifier analogue\n"
    "                   of the upscale model's `--features scalar`. Worth it when\n"
    "                   training spans a coverage ladder, so the model can tell which\n"
    "                   sparsity regime it is in. Requires --data (the artifact\n"
    "                   carries the exact per-record total).\n"
    "  -h               Show this help message.\n"
    "\n");
  return 1;
}

int main_train(int argc, char *argv[]) {
  const char *labels_path = NULL, *out_path = NULL, *framework = "xgboost";
  const char *data_path = NULL, *hier_path = NULL;
  int npattern = 0, nrounds = 0, include_pna = 0, scalar_cov = 0;
  /* xgboost tree-shape knobs. Default max_depth stays xgboost's 6; the point of
   * exposing it is that a redundant training set -- one pseudobulk replicated
   * across a coverage ladder -- makes every split look far more confident than
   * its effective sample size warrants, so trees grow deep and a class can end
   * up decided by a single feature. Capping depth is the direct remedy. */
  int max_depth = 0, min_child = 0; double colsample = 0.0;
  /* violation-framework constants. The defaults are the validated ones: a 0.5
   * call cutoff (in held-out tumour single cells, 0 of 1106 colorectal cancer
   * cells put an expected-1 pattern below it), sqrt(n_cpg) weighting, and 20
   * observed patterns per side before a call is allowed. */
  double vio_threshold = 0.5; const char *vio_weight = "sqrt";
  int vio_min_patterns = 20;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-l") == 0 && i+1 < argc) labels_path = argv[++i];
    else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_path = argv[++i];
    else if (strcmp(argv[i], "--scalar-coverage") == 0) scalar_cov = 1;
    else if (strcmp(argv[i], "--hierarchy") == 0 && i+1 < argc) hier_path = argv[++i];
    else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_path    = argv[++i];
    else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
      npattern = parse_nonneg_int(argv[++i], "-p expects a non-negative integer");
    else if (strcmp(argv[i], "--include-pna") == 0)    include_pna = 1;
    else if (strcmp(argv[i], "--framework") == 0 && i+1 < argc) framework = argv[++i];
    else if (strcmp(argv[i], "--max-depth") == 0 && i+1 < argc)
      max_depth = parse_nonneg_int(argv[++i], "--max-depth expects a non-negative integer");
    else if (strcmp(argv[i], "--min-child-weight") == 0 && i+1 < argc)
      min_child = parse_nonneg_int(argv[++i], "--min-child-weight expects a non-negative integer");
    else if (strcmp(argv[i], "--colsample") == 0 && i+1 < argc)
      colsample = atof(argv[++i]);
    else if (strcmp(argv[i], "--call-threshold") == 0 && i+1 < argc)
      vio_threshold = atof(argv[++i]);
    else if (strcmp(argv[i], "--pattern-weight") == 0 && i+1 < argc)
      vio_weight = argv[++i];
    else if (strcmp(argv[i], "--min-patterns") == 0 && i+1 < argc)
      vio_min_patterns = parse_nonneg_int(argv[++i],
        "--min-patterns expects a non-negative integer");
    else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
      nrounds = parse_nonneg_int(argv[++i], "-n expects a non-negative integer");
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      train_usage(); return 0;
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      tdie("unrecognized or incomplete option", argv[i]);
    else break;
  }
  int fw_xgb = strcmp(framework, "xgboost") == 0;
  int fw_lin = (strcmp(framework, "threshold") == 0) || (strcmp(framework, "logistic") == 0);
  int fw_vio = strcmp(framework, "violation") == 0;
  if (!fw_xgb && !fw_lin && !fw_vio)
    tdie("unknown --framework (xgboost|violation|threshold|logistic)", framework);

  /* ---- violation framework: transcribe, do not train ----
   * The MRMP artifact already states each class's expected methylation at every
   * pattern, so the classifier is fully determined by the reference. There is
   * nothing to fit, hence no training data, no labels file, and one positional:
   * the artifact itself (a .cm cannot serve -- it carries the per-CpG mask but
   * not the binstrings). Writing the mask at exactly the model's pattern count
   * is what keeps the shipped mask and the model dimension from drifting. */
  if (fw_vio) {
    if (!out_path || argc - i != 1) return train_usage();
    if (!ms_path_is_bundle_ext(out_path))
      tdie("the violation framework requires a .clfx output (bundled with the MRMP)",
           out_path);
    const char *artifact = argv[i];
    if (!ms_mrmp_is_artifact(artifact))
      tdie("the violation framework needs the MRMPIDX1 artifact (.mrmp), not an "
           "exported .cm -- the .cm has no binstrings", artifact);
    uint32_t top_k = npattern > 0 ? (uint32_t)npattern : 1000u;
    viomodel_t *vm = ms_viomodel_from_mrmp(artifact, top_k, vio_threshold,
                                           vio_weight, vio_min_patterns);
    char mtmp[] = "/tmp/methscope_vio_XXXXXX.vio";
    int mfd = mkstemps(mtmp, 4);
    if (mfd < 0) tdie("cannot create temp violation model file", NULL);
    close(mfd);
    ms_viomodel_write(vm, mtmp);
    char ctmp[] = "/tmp/methscope_viomask_XXXXXX.cm";
    int cfd = mkstemps(ctmp, 3);
    if (cfd < 0) tdie("cannot create temp mask file", NULL);
    close(cfd);
    ms_mrmp_write_mask(artifact, ctmp, "Pna", (uint32_t)vm->n_feat);
    ms_bundle_pack(out_path, "violation", mtmp, ctmp, NULL);
    unlink(mtmp); unlink(ctmp);
    fprintf(stderr, "[methscope] transcribed violation model: %d class(es) x %d "
            "pattern(s), t=%g, weight=%s, min_patterns=%d -> %s\n",
            vm->n_label, vm->n_feat, vm->threshold, vm->weighting,
            vm->min_patterns, out_path);
    ms_viomodel_free(vm);
    return 0;
  }

  /* With --data the features are already built, so <query.cg> drops out and
   * only <ref.cm> stays positional -- the bundle still has to carry the MRMP. */
  int want_pos = data_path ? 1 : 2;
  if (!out_path || argc - i != want_pos) return train_usage();
  if (!data_path && !labels_path) return train_usage();
  if (fw_lin && !ms_path_is_bundle_ext(out_path))
    tdie("threshold/logistic frameworks require a .clfx output (bundled with the MRMP)", out_path);
  if (scalar_cov && !data_path)
    tdie("--scalar-coverage needs --data (the artifact carries the covered-CpG total)", NULL);
  if (scalar_cov && fw_lin)
    tdie("--scalar-coverage is xgboost-only", framework);
  const char *query_cg = data_path ? NULL : argv[i];
  char *tmp_mrmp = NULL;
  const char *ref_mrmp = ms_mrmp_resolve(argv[i + (data_path ? 0 : 1)], &tmp_mrmp);

  /* ---- features ----
   * Featurize now, or reuse a .msfm built earlier by `classify-featurize`.
   * Both yield the same ms_matrix_t, so nothing below this point differs. */
  char **data_lab = NULL;
  uint32_t *levels = NULL;
  ms_matrix_t *m = data_path
    ? ms_msfm_to_matrix(data_path, &data_lab, &levels)
    : ms_matrix_build(query_cg, ref_mrmp);
  if (data_path && !labels_path && !data_lab)
    tdie("the artifact carries no labels; pass -l", data_path);

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

  /* ---- labels ----
   * Embedded in the artifact by default: a .msfm cannot be paired with the
   * wrong label file, which is the failure this used to die on. An explicit
   * -l still wins, so a label set can be revised without re-featurizing. */
  int n_lab = 0;
  char **lab;
  if (labels_path) {
    lab = read_labels(labels_path, &n_lab);
    if (data_lab) { for (int j = 0; j < m->n_cells; ++j) free(data_lab[j]); free(data_lab); }
  } else {
    lab = data_lab; n_lab = m->n_cells;
  }
  if (n_lab != m->n_cells)
    tdie("label count does not match number of query cells", labels_path);

  /* class set = sorted unique labels (deterministic class indices) */
  char **uniq = malloc(n_lab * sizeof(char *));
  if (!uniq) tdie("out of memory", NULL);
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
  if (!yidx) tdie("out of memory", NULL);
  for (int r = 0; r < m->n_cells; ++r) {
    /* binary search into the sorted class set */
    int lo = 0, hi = K - 1, idx = -1;
    while (lo <= hi) {
      int mid = (lo + hi) / 2;
      int cmp = strcmp(lab[r], uniq[mid]);
      if (cmp == 0) { idx = mid; break; }
      else if (cmp < 0) hi = mid - 1;
      else lo = mid + 1;
    }
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
    if (!ylab) tdie("out of memory", NULL);
    for (int r = 0; r < m->n_cells; ++r) ylab[r] = (float)yidx[r];
    /* One extra column when --scalar-coverage: log1p(covered CpGs), appended
     * after the patterns. Trees are scale-invariant, so unlike UPDEC2 this
     * needs no mean/scale standardization -- but `classify` must build the
     * column the same way, which is what the booster attribute below records. */
    bst_ulong nrow = (bst_ulong)m->n_cells;
    bst_ulong ncol = (bst_ulong)npattern + (scalar_cov ? 1 : 0);
    float *data = malloc((size_t)nrow * ncol * sizeof(float));
    if (!data) tdie("out of memory", NULL);
    for (int r = 0; r < m->n_cells; ++r) {
      const double *src = m->M + (size_t)r * m->n_patterns;
      float        *dst = data + (size_t)r * ncol;
      for (int c = 0; c < npattern; ++c) dst[c] = (float)src[c];
      if (scalar_cov) dst[npattern] = (float)log1p((double)levels[r]);
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
    char pbuf[32];
    if (max_depth > 0) {
      snprintf(pbuf, sizeof(pbuf), "%d", max_depth);
      XGCHK(XGBoosterSetParam(booster, "max_depth", pbuf));
    }
    if (min_child > 0) {
      snprintf(pbuf, sizeof(pbuf), "%d", min_child);
      XGCHK(XGBoosterSetParam(booster, "min_child_weight", pbuf));
    }
    if (colsample > 0.0) {
      snprintf(pbuf, sizeof(pbuf), "%g", colsample);
      XGCHK(XGBoosterSetParam(booster, "colsample_bytree", pbuf));
    }
    for (int it = 0; it < nrounds; ++it)
      XGCHK(XGBoosterUpdateOneIter(booster, it, dtrain));

    /* embed labels; save as loose .ubj or a (trimmed-mrmp) .ubjx bundle */
    ms_booster_set_meta(booster, uniq, K);
    if (scalar_cov) ms_booster_set_scalar_cov(booster);
    if (hier_path) {                    /* travel with the model, not beside it */
      FILE *hf = fopen(hier_path, "r");
      if (!hf) tdie("cannot open --hierarchy", hier_path);
      size_t cap = 1 << 16, n = 0;
      char *buf = malloc(cap);
      if (!buf) tdie("out of memory", NULL);
      size_t got;
      while ((got = fread(buf + n, 1, cap - n - 1, hf)) > 0) {
        n += got;
        if (n + 1 >= cap) { cap <<= 1; buf = realloc(buf, cap);
                            if (!buf) tdie("out of memory", NULL); }
      }
      buf[n] = 0; fclose(hf);
      ms_booster_set_hier(booster, buf);
      free(buf);
    }
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
