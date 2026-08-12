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
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sched.h>
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
  /* A .mrmp goes into the bundle verbatim -- it is already exactly its patterns,
   * because mrmp-pool --pooled-top prunes. Trimming is a .cm-era operation for
   * an artifact that carried more than it used; asking for it here means the cut
   * was made at the wrong step. */
  int ref_is_mrmp = 0;
  { FILE *f = fopen(ref_mrmp, "rb");
    if (f) { char mg[8];
      ref_is_mrmp = fread(mg, 1, 8, f) == 8 && !memcmp(mg, "MRMPIDX1", 8);
      fclose(f); } }
  if (ref_is_mrmp && npattern < n_nonpna)
    tdie("cannot trim a .mrmp at train time; re-run mrmp-pool --pooled-top "
         "to cut it, or pass an exported .cm", ref_mrmp);
  if (!ref_is_mrmp && npattern < n_nonpna) {    /* real patterns dropped -> trim */
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

/* thousands separators; buf >= 32 bytes */
static const char *commafmt_tr(uint64_t v, char *buf) {
  char tmp[24]; int n = snprintf(tmp, sizeof tmp, "%llu", (unsigned long long)v);
  int o = 0;
  for (int i = 0; i < n; ++i) {
    if (i && (n - i) % 3 == 0) buf[o++] = ',';
    buf[o++] = tmp[i];
  }
  buf[o] = '\0';
  return buf;
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
    "  -p <npattern>    Use the first N patterns in the artifact's own order.\n"
    "                   after the 'Pna' backgrounds have been excluded, in the\n"
    "                   For an auto MRMP that order\n"
    "                   is recurrence rank (P1,P2,...), so N means the N most\n"
    "                   recurrent; for curated named markers it is definition\n"
    "                   order, and leaving it unset is what you want.\n"
    "                   Note this cuts across a FUSED multi-set artifact rather\n"
    "                   than within each set, so a value below the total will\n"
    "                   drop whole trailing sets. Cut with mrmp-pool --pooled-top\n"
    "                   instead if you want a per-set budget.\n"
    "                   Default: every non-'Pna' state.\n"

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
    "  --eval-every <N> Report training mlogloss every N rounds, as a rolling\n"
    "                   window of the last 5. Default: nrounds/20, about 5%%\n"
    "                   overhead -- each evaluation predicts over the whole\n"
    "                   training matrix. 0 disables, 1 evaluates every round.\n"
    "  --threads <N>    xgboost nthread (default: the CPUs this process may\n"
    "                   actually use). Left to xgboost it sizes its pool from\n"
    "                   the machine, which on a shared batch node is not what\n"
    "                   the job was allocated -- so the same matrix and the\n"
    "                   same rounds have timed 76 s on one node and 46 min on\n"
    "                   another. Pinning it makes runs comparable.\n"
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
    "                   record, after the patterns -- the classifier analogue\n"
    "                   of the upscale model's `--features scalar`. Worth it when\n"
    "                   training spans a coverage ladder, so the model can tell which\n"
    "                   sparsity regime it is in. Requires --data (the artifact\n"
    "                   carries the exact per-record total).\n"
    "  -h               Show this help message.\n"
    "\n");
  return 1;
}

/* --------------------------------------------------------- classify-train-tree
 * Train every node of a routing tree from ONE shared feature matrix.
 *
 * The shell loop this replaces built a per-node .cg with `yame subset` and
 * featurized each separately -- a full store copy and a full pass per node, to
 * arrive at columns that are bit-identical to the shared ones (verified: 0 of
 * 1,201 rows differ). Here the store is featurized once against the chain, and
 * a node is (its rows) x (its columns): rows whose label is one of its classes,
 * columns from ms_msfm_layout(). Nothing is subset on disk.
 *
 * One behavioural change from the shell loop, and it is deliberate: the
 * coverage ladder is now drawn ONCE per sample rather than independently per
 * node, so every node sees the same draw. More consistent, but it does mean
 * per-node models are no longer reproducible from the old per-node commands. */
typedef struct { int max_depth, min_child; double colsample; } tune_t;

static BoosterHandle train_one(const float *X, const float *y, uint32_t nrow,
                               uint32_t ncol, int K, int nrounds, int nt,
                               const tune_t *tn, const char *tag) {
  DMatrixHandle dtrain; BoosterHandle b;
  XGCHK(XGDMatrixCreateFromMat(X, nrow, ncol, NAN, &dtrain));
  XGCHK(XGDMatrixSetFloatInfo(dtrain, "label", y, nrow));
  XGCHK(XGBoosterCreate(&dtrain, 1, &b));
  char buf[16];
  snprintf(buf, sizeof buf, "%d", K);
  XGCHK(XGBoosterSetParam(b, "booster", "gbtree"));
  XGCHK(XGBoosterSetParam(b, "objective", "multi:softprob"));
  XGCHK(XGBoosterSetParam(b, "eval_metric", "mlogloss"));
  XGCHK(XGBoosterSetParam(b, "num_class", buf));
  snprintf(buf, sizeof buf, "%d", nt > 0 ? nt : 1);
  XGCHK(XGBoosterSetParam(b, "nthread", buf));
  if (tn && tn->max_depth > 0)
    { snprintf(buf, sizeof buf, "%d", tn->max_depth);
      XGCHK(XGBoosterSetParam(b, "max_depth", buf)); }
  if (tn && tn->min_child > 0)
    { snprintf(buf, sizeof buf, "%d", tn->min_child);
      XGCHK(XGBoosterSetParam(b, "min_child_weight", buf)); }
  if (tn && tn->colsample > 0.0)
    { snprintf(buf, sizeof buf, "%g", tn->colsample);
      XGCHK(XGBoosterSetParam(b, "colsample_bytree", buf)); }
  for (int it = 0; it < nrounds; ++it)
    XGCHK(XGBoosterUpdateOneIter(b, it, dtrain));
  const char *ev = NULL, *nm = "train";
  double loss = 0.0 / 0.0;
  if (!XGBoosterEvalOneIter(b, nrounds - 1, &dtrain, &nm, 1, &ev) && ev) {
    const char *c = strrchr(ev, ':');
    if (c) loss = atof(c + 1);
  }
  fprintf(stderr, "  %-12s %5u rows x %-5u col, %2d classes, %3d rounds"
          "   mlogloss %.4f\n", tag, nrow, ncol, K, nrounds, loss);
  XGDMatrixFree(dtrain);
  return b;
}

int main_train_tree(int argc, char *argv[]) {
  const char *data_path = NULL, *out = NULL;
  int nrounds = 0, nthread = 0;
  tune_t tn = {0, 0, 0.0};
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "--data") && i + 1 < argc) data_path = argv[++i];
    else if (!strcmp(a, "--max-depth") && i + 1 < argc) tn.max_depth = atoi(argv[++i]);
    else if (!strcmp(a, "--min-child-weight") && i + 1 < argc) tn.min_child = atoi(argv[++i]);
    else if (!strcmp(a, "--colsample") && i + 1 < argc) tn.colsample = atof(argv[++i]);
    /* accepted and ignored: the .msfm carries its own labels and every column */
    else if ((!strcmp(a, "-l") || !strcmp(a, "-p") || !strcmp(a, "--eval-every"))
             && i + 1 < argc) ++i;
    else if (!strcmp(a, "--framework") && i + 1 < argc) {
      if (strcmp(argv[++i], "xgboost"))
        tdie("only --framework xgboost trains per node; the others take a .mrmp "
             "and are unfitted, so they stay on the flat path", argv[i]);
    }
    else if (!strcmp(a, "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(a, "-n") && i + 1 < argc) nrounds = atoi(argv[++i]);
    else if (!strcmp(a, "--threads") && i + 1 < argc) nthread = atoi(argv[++i]);
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stderr,
        "Usage: methscope classify-train-tree --data TRAIN.msfm -o TREE.clfx\n\n"
        "  Trains one model per node of a routing tree and writes them, with the\n"
        "  chain, as a single scorable bundle.\n\n"
        "  The chain comes from the .msfm, which carries the MRMP it was\n"
        "  featurized against -- so there is no second argument to mismatch. A\n"
        "  node is its\n"
        "  rows (samples labelled with one of its classes) by its columns (from\n"
        "  the chain's layout), so nothing is subset on disk and every node sees\n"
        "  the same coverage draw.\n\n"
        "  -n N        boosting rounds per node (default round(sqrt(rows)))\n"
        "  --threads T thread pool\n\n"
        "  Score the result with: methscope classify query.cg TREE.clfx\n");
      return 0;
    }
    else if (a[0] == '-') tdie("unrecognized or incomplete option", a);
    else tdie("unexpected argument -- the chain comes from the .msfm", a);
  }
  if (!data_path || !out)
    tdie("need --data TRAIN.msfm -o TREE.clfx (see -h)", NULL);
  /* The matrix carries the chain it was featurized against, so there is no
   * second argument to get wrong: a mismatched pairing is not expressible. */
  char *chain = ms_msfm_chain(data_path);
  if (!chain)
    tdie("this .msfm carries no MRMP -- re-run classify-featurize against a "
         "chain (artifacts written before embedding have none)", data_path);

  char **row_lab = NULL; uint32_t *levels = NULL;
  ms_matrix_t *m = ms_msfm_to_matrix(data_path, &row_lab, &levels);
  if (!row_lab) tdie("training artifact carries no labels", data_path);
  uint32_t flags = ms_msfm_flags(data_path);
  ms_msfm_layout_t *lay = ms_msfm_layout(chain, flags);
  if (lay->total != (uint32_t)m->n_patterns)
    tdie("the .msfm was not featurized against this chain (column count differs)",
         data_path);

  ms_mrmpset_t *ch = ms_mrmpset_open(chain);
  const uint32_t n = ch->n_sets;
  fprintf(stderr, "\n[methscope] classify-train-tree\n\n");
  fprintf(stderr, "  %-12s %d records x %d columns, %u node(s)\n", "data",
          m->n_cells, m->n_patterns, n);

  void **bl = calloc(n, sizeof(void *));
  uint64_t *bn = calloc(n, sizeof(uint64_t));
  char **nm = calloc(n, sizeof(char *));
  if (!bl || !bn || !nm) tdie("out of memory", NULL);

  for (uint32_t k = 0; k < n; ++k) {
    mrmp_top_t *t = ms_mrmp_top_read_at(chain, ch->block_off[k], 1);
    nm[k] = strdup(ch->name[k]);
    /* rows: samples whose label is one of THIS node's classes. The class order
     * is the block's, so the booster's class ids line up with the chain. */
    uint32_t *row = malloc((size_t)m->n_cells * sizeof(uint32_t));
    float *y = malloc((size_t)m->n_cells * sizeof(float));
    if (!row || !y) tdie("out of memory", NULL);
    uint32_t nr = 0;
    for (int r = 0; r < m->n_cells; ++r)
      for (uint32_t c = 0; c < t->n_samples; ++c)
        if (!strcmp(row_lab[r], t->labels[c])) { row[nr] = (uint32_t)r; y[nr++] = (float)c; break; }
    if (!nr) tdie("no training rows for a node's classes", nm[k]);
    uint32_t ncol = lay->ncol[k], col0 = lay->col0[k];
    float *X = malloc((size_t)nr * ncol * sizeof(float));
    if (!X) tdie("out of memory", NULL);
    for (uint32_t r = 0; r < nr; ++r)
      for (uint32_t c = 0; c < ncol; ++c)
        X[(size_t)r * ncol + c] =
          (float)m->M[(size_t)row[r] * m->n_patterns + col0 + c];
    int nrd = nrounds > 0 ? nrounds : (int)(sqrt((double)nr) + 0.5);
    if (nrd < 1) nrd = 1;
    BoosterHandle b = train_one(X, y, nr, ncol, (int)t->n_samples, nrd,
                                nthread, &tn, nm[k]);
    ms_booster_set_meta(b, t->labels, (int)t->n_samples);
    if (flags & MSFM_FLAG_BIN_FLAT)     ms_booster_set_binarize(b, "0.5");
    else if (flags & MSFM_FLAG_BIN_PAT) ms_booster_set_binarize(b, "pattern");
    {  /* the exact columns, by name -- classify selects by LAYOUT, but the
        * names keep a node's model self-describing if it is pulled out */
      char **fn = malloc((size_t)ncol * sizeof(char *));
      for (uint32_t c = 0; c < ncol; ++c) fn[c] = m->pattern_names[col0 + c];
      ms_booster_set_features(b, fn, (int)ncol);
      free(fn);
    }
    { char tmp[] = "/tmp/methscope_tree_XXXXXX.ubj";
      int fd = mkstemps(tmp, 4);
      if (fd < 0) tdie("cannot create temporary model", tmp);
      close(fd);
      XGCHK(XGBoosterSaveModel(b, tmp));
      FILE *f = fopen(tmp, "rb");
      if (!f) tdie("cannot read temporary model", tmp);
      fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
      bl[k] = malloc((size_t)sz); bn[k] = (uint64_t)sz;
      if (!bl[k] || fread(bl[k], 1, (size_t)sz, f) != (size_t)sz)
        tdie("short read on temporary model", tmp);
      fclose(f); unlink(tmp); }
    XGBoosterFree(b);
    free(X); free(row); free(y);
    ms_mrmp_top_free(t);
  }
  ms_bundle_pack_tree(out, chain, n, nm, bl, bn);
  fprintf(stderr, "\n  %-12s %u node(s) + the chain -> %s\n", "wrote", n, out);
  fprintf(stderr, "  %-12s methscope classify query.cg %s\n", "score with", out);
  for (uint32_t k = 0; k < n; ++k) { free(nm[k]); free(bl[k]); }
  free(nm); free(bl); free(bn);
  ms_msfm_layout_free(lay); ms_mrmpset_free(ch);
  unlink(chain); free(chain);
  return 0;
}

int main_train(int argc, char *argv[]) {
  /* xgboost + a prebuilt matrix is the tree path: the .msfm carries the chain
   * it was featurized against, so training walks its nodes and emits one
   * scorable bundle. The other frameworks are UNFITTED -- they transcribe a
   * .mrmp and take no training data at all -- so they stay here. */
  { const char *fw = "xgboost"; int has_data = 0;
    for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--framework") && i + 1 < argc) fw = argv[i + 1];
      else if (!strcmp(argv[i], "--data") && i + 1 < argc) has_data = 1;
    }
    if (has_data && !strcmp(fw, "xgboost")) return main_train_tree(argc, argv);
  }

  const char *labels_path = NULL, *out_path = NULL, *framework = "xgboost";
  const char *data_path = NULL, *hier_path = NULL;
  int npattern = 0, nrounds = 0, include_pna = 0, scalar_cov = 0;
  int nthread = 0;                    /* 0 = derive; see ms_train_threads() */
  int eval_every = -1;                /* -1 = auto (nrounds/20); 0 = off */
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
    else if (strcmp(argv[i], "--eval-every") == 0 && i+1 < argc)
      eval_every = parse_nonneg_int(argv[++i], "--eval-every expects a non-negative integer");
    else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc)
      nthread = parse_nonneg_int(argv[++i], "--threads expects a non-negative integer");
    else if (strcmp(argv[i], "--scalar-coverage") == 0) scalar_cov = 1;
    else if (strcmp(argv[i], "--hierarchy") == 0 && i+1 < argc) hier_path = argv[++i];
    else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_path    = argv[++i];
    else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
      npattern = parse_nonneg_int(argv[++i], "-p expects a non-negative integer");
    else if (strcmp(argv[i], "--include-pna") == 0)
      tdie("--include-pna is gone: featurize no longer emits background "
           "patterns, so there is nothing to include", NULL);
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
  /* Keep the ARTIFACT path as given. ms_mrmp_resolve() materialises a runtime
   * mask for featurizing, but the bundle should carry the .mrmp itself: it is
   * self-describing, keeps set names and per-set structure, and needs no
   * sibling .idx -- which ms_bundle_pack cannot store anyway, so bundling a
   * resolved multi-record mask silently lost every record's name. The chain
   * walker stops at MSBNDL1, so a bundled .mrmp is readable straight off the
   * bundle prefix, exactly as a bundled .cm was. */
  const char *ref_arg  = argv[i + (data_path ? 0 : 1)];
  const char *ref_mrmp = ms_mrmp_resolve(ref_arg, &tmp_mrmp);

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

  /* NA-background states -- "Pna", or "Pna.<set>" once several sets are fused,
   * one per set. Counted so they can be excluded from features by default. */
  int n_pna = 0;
  for (int c = 0; c < m->n_patterns; ++c)
    if (ms_is_pna_name(m->pattern_names[c])) n_pna++;
  int n_nonpna = m->n_patterns - n_pna;

  /* Which COLUMNS become features, resolved BY NAME.
   *
   * This used to take the first npattern columns by position, on the reasoning
   * that npattern = n_nonpna means "the non-Pna ones". That is false for a
   * fused multi-set artifact: the layout is SET-MAJOR, so each set's Pna sits
   * right after that set's own patterns and the backgrounds are interleaved,
   * not trailing. On a 100-set / 1,100-column artifact the positional rule fed
   * 66 background columns in as features and dropped 66 real patterns -- the
   * whole tail of the satellite block -- off the end. Gather through an
   * explicit index instead, and record the names in the booster so `classify`
   * reproduces the choice rather than re-deriving it. */
  int *feat_idx = malloc((size_t)m->n_patterns * sizeof(int));
  if (!feat_idx) tdie("out of memory", NULL);
  int n_feat = 0;
  for (int c = 0; c < m->n_patterns; ++c)
    if (include_pna || !ms_is_pna_name(m->pattern_names[c]))
      feat_idx[n_feat++] = c;
  /* -p N keeps the first N of that selection, so it can no longer truncate a
   * satellite by landing mid-set the way a positional cut did. */
  if (npattern > 0) {
    if (npattern > n_feat)
      tdie("-p exceeds the number of patterns", NULL);
    n_feat = npattern;
  }
  if (n_feat <= 0) tdie("the artifact has no patterns", NULL);
  /* Rebuild the matrix around the selection so every consumer below -- xgboost,
   * the linear frameworks, the violation model, and the trim path that reads
   * pattern_names -- indexes features correctly without knowing any of this. */
  ms_matrix_select(m, feat_idx, n_feat);
  free(feat_idx);
  npattern = n_feat;

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
    trimmed = bundle_model(out_path, framework, tmpl, ref_arg,
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
    const int nrounds_auto = (nrounds <= 0);
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
    /* Thread pool. Prefer what the batch system granted, then what the kernel
     * will actually schedule us on, and only then the machine's core count --
     * an unpinned pool sized from a 94-core node while holding 24 of them is a
     * confound in every timing taken from a shared queue. */
    int nt = nthread;
    if (nt <= 0) {
      const char *e = getenv("SLURM_CPUS_PER_TASK");
      if (e && *e) nt = atoi(e);
    }
    if (nt <= 0) {
      cpu_set_t set;
      if (sched_getaffinity(0, sizeof(set), &set) == 0) nt = CPU_COUNT(&set);
    }
    if (nt <= 0) nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nt > 0) {
      char tbuf[16]; snprintf(tbuf, sizeof(tbuf), "%d", nt);
      XGCHK(XGBoosterSetParam(booster, "nthread", tbuf));
      { char c1[32], c2[32];
        fprintf(stderr, "\n[methscope] classify-train\n\n");
        fprintf(stderr, "  %-14s %s records x %s patterns, %d classes\n", "data",
                commafmt_tr((uint64_t)m->n_cells, c1),
                commafmt_tr((uint64_t)npattern, c2), K);
        fprintf(stderr, "  %-14s gbtree, multi:softprob, mlogloss\n", "booster");
        fprintf(stderr, "  %-14s %d%s\n", "rounds", nrounds,
                nrounds_auto ? " (round(sqrt(records)))" : "");
        fprintf(stderr, "  %-14s %d\n", "threads", nt); }
    }
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
    /* Periodic training mlogloss, so a run that is not converging is visible
     * while it happens rather than after.
     *
     * XGBoosterEvalOneIter predicts over the whole training matrix, so it is
     * not free -- evaluating every round would roughly double the cost. Every
     * nrounds/20 keeps the overhead near 5% and still shows the shape of the
     * curve. --eval-every 0 turns it off, 1 evaluates every round.
     *
     * The rolling window is the point: a single number says nothing, five in a
     * row say whether it is still descending. */
    int estep = eval_every >= 0 ? eval_every : (nrounds >= 20 ? nrounds / 20 : 1);
    const int tty = isatty(STDERR_FILENO);
    double hist[5]; int nh = 0;
    for (int it = 0; it < nrounds; ++it) {
      XGCHK(XGBoosterUpdateOneIter(booster, it, dtrain));
      if (!estep || (it + 1) % estep != 0) continue;
      const char *ev = NULL;
      const char *nm = "train";
      if (XGBoosterEvalOneIter(booster, it, &dtrain, &nm, 1, &ev) || !ev) continue;
      const char *c = strrchr(ev, ':');
      if (!c) continue;
      if (nh < 5) hist[nh++] = atof(c + 1);
      else { for (int k = 0; k < 4; ++k) hist[k] = hist[k + 1]; hist[4] = atof(c + 1); }
      char buf[160]; int at = 0;
      for (int k = 0; k < nh; ++k)
        at += snprintf(buf + at, sizeof buf - at, "%s%.4f", k ? " " : "", hist[k]);
      fprintf(stderr, "%s  %-14s round %d/%d  mlogloss %s%s",
              tty ? "\r\033[K" : "", "training", it + 1, nrounds, buf,
              tty ? "" : "\n");
      if (tty) fflush(stderr);
    }
    if (tty) fprintf(stderr, "\r\033[K");
    /* One final evaluation regardless of --eval-every: the number a reader
     * actually wants is where it ended up, and it costs one pass. */
    { const char *ev = NULL, *nm = "train";
      if (!XGBoosterEvalOneIter(booster, nrounds - 1, &dtrain, &nm, 1, &ev) && ev) {
        const char *c = strrchr(ev, ':');
        if (c) fprintf(stderr, "  %-14s %.4f  (train mlogloss after %d rounds)\n",
                       "final loss", atof(c + 1), nrounds);
      } }

    /* embed labels; save as loose .ubj or a (trimmed-mrmp) .ubjx bundle */
    ms_booster_set_meta(booster, uniq, K);
    /* Carry the feature coding, taken from the artifact rather than from a CLI
     * flag: the coding is decided at featurize time and baked into the .msfm,
     * so the artifact is the only thing that actually knows. */
    if (data_path) {
      uint32_t ff = ms_msfm_flags(data_path);
      if (ff & MSFM_FLAG_BIN_FLAT)      ms_booster_set_binarize(booster, "0.5");
      else if (ff & MSFM_FLAG_BIN_PAT)  ms_booster_set_binarize(booster, "pattern");
    }
    {   /* the exact columns, by name, so `classify` gathers the same ones */
      char **fn = malloc((size_t)npattern * sizeof(char *));
      if (!fn) tdie("out of memory", NULL);
      for (int c = 0; c < npattern; ++c) fn[c] = m->pattern_names[c];
      ms_booster_set_features(booster, fn, npattern);
      free(fn);
    }
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
      trimmed = bundle_model(out_path, "xgboost", tmpl, ref_arg,
                             m->pattern_names, npattern, n_nonpna);
      unlink(tmpl);
    } else {
      XGCHK(XGBoosterSaveModel(booster, out_path));
    }
    fprintf(stderr, "  %-14s %s%s%s\n\n", "wrote", out_path,
            bundled ? " (booster + MRMP bundle" : "",
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
