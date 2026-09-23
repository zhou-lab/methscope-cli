// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
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
#ifdef __linux__
#include <sched.h>
#endif
#include "methscope.h"
#include "bmeta.h"
#include "bundle.h"    /* ms_mrmp_resolve / ms_bundle_pack / ms_path_is_bundle_ext */
#include "msfm.h"      /* ms_msfm_to_matrix -- the --data feature path */
#include "mrmp.h"
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
    char tmpl[4096];
    const char *td = getenv("TMPDIR");
    snprintf(tmpl, sizeof tmpl, "%s/methscope_trim_XXXXXX.cm",
             td && *td ? td : "/tmp");
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

static int train_usage(FILE *out) {
  ms_help(out,
    "\n"
    "Usage:\n"
    "  methscope classify-train -l <labels.txt> -o <out.clfx> [options] <query.cg> <ref.cm>\n"
    "  methscope classify-train --data <in.msfm> -o <out.clfx> [options] <ref.cm>\n"

    "\n"
    "Purpose:\n"
    "  Train a multiclass classifier for any per-record label (cell type, sex,\n"
    "  ...; fixed hyperparameters, no grid search) and write a self-describing\n"
    "  model with the class labels embedded.\n"
    "\n"
    "Arguments:\n"
    "  <query.cg>   Training methylome(s), one record per sample.\n"
    "  <ref.cm>     MRMP pattern definition. A bundle (.clfx/.updecx) also\n"
    "               works; its MRMP is used. Features are the MRMP states; the\n"
    "               'Pna' NA-background state is always excluded.\n"
    "               NOTE the default framework (xgboost) does not take this form\n"
    "               at all -- it needs --data TRAIN.msfm, which carries its own\n"
    "               chain. This positional serves the unfitted frameworks.\n"
    "\n"
    "Options:\n"
    "  -l <labels.txt>  One label per query record, in query order (required).\n"
    "  -o <out.clfx>    Output model path (required). A '.clfx' name writes a\n"
    "                   self-contained bundle (model + MRMP) that `classify` can\n"
    "                   run directly; a plain '.ubj' writes just the loose booster.\n"
    "                   The bundled MRMP is TRIMMED to exactly the patterns used\n"
    "                   (others folded into 'Pna'), and `classify` uses that same set.\n"
    "  -p <npattern>    Use the first N patterns, in the artifact's own order and\n"
    "                   after the 'Pna' backgrounds have been excluded. For an\n"
    "                   auto MRMP that order\n"
    "                   is recurrence rank (P1,P2,...), so N means the N most\n"
    "                   recurrent; for curated named markers it is definition\n"
    "                   order, and leaving it unset is what you want.\n"
    "                   Note this cuts across a FUSED multi-set artifact rather\n"
    "                   than within each set, so a value below the total will\n"
    "                   drop whole trailing sets. Cut with mrmp-pool --pooled-top\n"
    "                   instead if you want a per-set budget.\n"
    "                   Default: every non-'Pna' state.\n"
    "  --framework <f>  Model framework (default: xgboost):\n"
    "                     xgboost    gradient-boosted trees (multiclass).\n"
    "                     logistic   binary L2-regularized logistic regression;\n"
    "                                needs a .clfx out.\n"
    "                   The violation rule is unfitted, so it is a scoring mode:\n"
    "                   `classify --framework violation ref.mrmp query.cg`.\n"
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
    "  -h               Show this help message.\n"
    "\n");
  return out == stdout ? 0 : 1;
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
  const char *keep_path = NULL;
  int nrounds = 0, nthread = 0;
  tune_t tn = {0, 0, 0.0};
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "--data") && i + 1 < argc) data_path = argv[++i];
    else if (!strcmp(a, "--keep-columns") && i + 1 < argc) keep_path = argv[++i];
    else if (!strcmp(a, "--max-depth") && i + 1 < argc) tn.max_depth = atoi(argv[++i]);
    else if (!strcmp(a, "--min-child-weight") && i + 1 < argc) tn.min_child = atoi(argv[++i]);
    else if (!strcmp(a, "--colsample") && i + 1 < argc) tn.colsample = atof(argv[++i]);
    /* accepted and ignored: the .msfm carries its own labels and every column */
    else if ((!strcmp(a, "-l") || !strcmp(a, "-p") || !strcmp(a, "--eval-every"))
             && i + 1 < argc) ++i;
    else if (!strcmp(a, "--framework") && i + 1 < argc) {
      if (strcmp(argv[++i], "xgboost"))
        tdie("only --framework xgboost trains from a .msfm; the others take a .mrmp "
             "and are unfitted, so they stay on the flat path", argv[i]);
    }
    else if (!strcmp(a, "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(a, "-n") && i + 1 < argc) nrounds = atoi(argv[++i]);
    else if (!strcmp(a, "--threads") && i + 1 < argc) nthread = atoi(argv[++i]);
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stderr,
        "Usage: methscope classify-train --data TRAIN.msfm -o MODEL.clfx\n\n"
        "  Trains ONE pooled booster over every column of the bank chain the\n"
        "  .msfm was featurized against, and writes it with the chain as a\n"
        "  single scorable bundle. Class list and order come from the root set.\n"
        "  Defaults to a constrained booster (--max-depth 4, --colsample 0.4):\n"
        "  bank columns are selected on the training pools, and an unconstrained\n"
        "  booster memorizes their optimism (give the flags to override).\n\n"
        "  -n N        boosting rounds (default round(sqrt(rows)))\n"
        "  --threads T thread pool\n"
        "\n"
        "  Score the result with: methscope classify MODEL.clfx query.cg\n");
      return 0;
    }
    else if (a[0] == '-') tdie("unrecognized or incomplete option", a);
    else tdie("unexpected argument -- the chain comes from the .msfm", a);
  }
  if (!data_path || !out)
    tdie("need --data TRAIN.msfm -o TREE.clfx (see -h)", NULL);
  /* Pooled training defaults to a CONSTRAINED booster (depth 4, 40% column
   * subsample per tree). With a full pair bank every column was selected on
   * the training pools, so each is slightly train-optimistic; the default
   * deep, all-columns booster memorizes those separations (fold-0 human 63-
   * class: train mlogloss < 0.01, held-out confusers degrade with coverage).
   * Hiding 60% of the columns per tree and capping depth forces the ensemble
   * to spread trust across redundant contrasts; measured fold-0 this lifts
   * every human rung (native 0.9338 -> 0.9592) and is neutral on the mouse
   * 41-class. Explicit --max-depth / --colsample still override. */
  if (tn.max_depth == 0) tn.max_depth = 4;
  if (tn.colsample == 0.0) tn.colsample = 0.4;
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
  /* A satellite is not a node: it feeds an existing node's booster and gets no
   * model of its own. Checking its owner here turns a typo into a refusal
   * rather than a silently ignored set. */
  uint32_t *nodeof = malloc((size_t)ch->n_sets * sizeof(uint32_t));
  if (!nodeof) tdie("out of memory", NULL);
  uint32_t n = 0, nsat = 0;
  for (uint32_t s = 0; s < ch->n_sets; ++s) {
    char ob[512];
    if (!ms_set_is_satellite(ch->name[s])) { nodeof[n++] = s; continue; }
    ++nsat;
    if (!ms_set_owner(ch->name[s], ob, sizeof ob))
      tdie("malformed satellite name (nothing before '@')", ch->name[s]);
    uint32_t j = 0;
    for (; j < ch->n_sets; ++j)
      if (!ms_set_is_satellite(ch->name[j]) && !strcmp(ch->name[j], ob)) break;
    if (j == ch->n_sets)
      tdie("satellite names a node that is not in this chain", ch->name[s]);
    /* Its classes must be a SUBSET of the node's -- it lends that node's
     * booster columns, so a class the node never decides is evidence about
     * nothing. Siblings may overlap, which is the difference from a hard
     * child, and is not checked. inspect --tree enforces the same rule; this
     * is the copy that guards a hand-assembled chain at the point a model is
     * actually built from it. */
    mrmp_top_t *ts = ms_mrmp_top_read_at(chain, ch->block_off[s], 1);
    mrmp_top_t *tn = ms_mrmp_top_read_at(chain, ch->block_off[j], 1);
    for (uint32_t c = 0; c < ts->n_samples; ++c) {
      uint32_t seen = 0;
      for (uint32_t e = 0; e < tn->n_samples && !seen; ++e)
        seen = !strcmp(tn->labels[e], ts->labels[c]);
      if (!seen) tdie("satellite carries a class its node does not",
                      ts->labels[c]);
    }
    ms_mrmp_top_free(ts); ms_mrmp_top_free(tn);
  }
  /* Read once; every node gets a copy. A tree has no single booster to hang
   * this on, and `classify` reads it back from whichever node made the call,
   * so the taxonomy has to travel with all of them -- the same reason the
   * class-label list is set per node. */

  fprintf(stderr, "\n[methscope] classify-train\n\n");
  fprintf(stderr, "  %-12s %d records x %d columns, %u node(s)", "data",
          m->n_cells, m->n_patterns, n);
  if (nsat) fprintf(stderr, " + %u satellite(s)", nsat);
  fprintf(stderr, ", one pooled booster");
  fputc('\n', stderr);
  { /* column provenance: how much of the feature space each origin
     * contributes. Sparse-coverage behaviour tracks the per-column CpG
     * support, and that differs by origin -- wide-node rank columns carry
     * tens of CpGs while pair/satellite columns carry tens of thousands --
     * so the split is worth one line every run. */
    uint32_t c_node = 0, c_pair = 0, c_sat = 0;
    for (uint32_t s = 0; s < lay->n_sets; ++s) {
      if (ms_set_is_satellite(lay->name[s])) { c_sat += lay->ncol[s]; continue; }
      uint32_t j = 0;
      for (; j < ch->n_sets; ++j)
        if (!strcmp(ch->name[j], lay->name[s])) break;
      mrmp_top_t *t2 = j < ch->n_sets
        ? ms_mrmp_top_read_at(chain, ch->block_off[j], 1) : NULL;
      if (t2 && t2->n_samples == 2) c_pair += lay->ncol[s];
      else                          c_node += lay->ncol[s];
      if (t2) ms_mrmp_top_free(t2);
    }
    fprintf(stderr, "  %-12s %u multi-class node, %u pair node, "
            "%u satellite\n", "columns", c_node, c_pair, c_sat);
  }

  {
    /* SOFT ROUTING: one booster over the full layout width. The class list
     * and its order come from the ROOT (the hard set without a parent),
     * which covers every class by construction; rows are every labelled
     * cell; columns are ALL of them, in layout order, which is exactly how
     * the .msfm stores the matrix -- no gathering. */
    uint32_t rk = UINT32_MAX;
    for (uint32_t i = 0; i < n; ++i)
      if (!strchr(ch->name[nodeof[i]], '.')) {
        if (rk != UINT32_MAX)
          tdie("two root-level sets; a bank chain has exactly one root",
               ch->name[nodeof[i]]);
        rk = nodeof[i];
      }
    if (rk == UINT32_MAX) tdie("no root set in this chain", data_path);
    mrmp_top_t *t = ms_mrmp_top_read_at(chain, ch->block_off[rk], 1);
    /* --keep-columns: a BANK is a curated subset of the layout's columns
     * (e.g. one LCA pair contrast per class pair), given as 0-based global
     * indices, one per line, ascending. The kept set is recorded in the
     * booster (MS_ATTR_COLSEL) so classify gathers exactly these. */
    uint32_t *keep = NULL, nkeep = 0;
    if (keep_path) {
      FILE *kf = fopen(keep_path, "r");
      if (!kf) tdie("cannot open --keep-columns", keep_path);
      uint32_t cap = 4096;
      keep = malloc((size_t)cap * sizeof(uint32_t));
      if (!keep) tdie("out of memory", NULL);
      char lb[64];
      while (fgets(lb, sizeof lb, kf)) {
        char *e = NULL;
        unsigned long v = strtoul(lb, &e, 10);
        if (e == lb) continue;
        if (v >= (unsigned long)m->n_patterns)
          tdie("--keep-columns index past the matrix width", lb);
        if (nkeep && keep[nkeep - 1] >= (uint32_t)v)
          tdie("--keep-columns must be ascending and unique", lb);
        if (nkeep == cap) {
          cap *= 2;
          keep = realloc(keep, (size_t)cap * sizeof(uint32_t));
          if (!keep) tdie("out of memory", NULL);
        }
        keep[nkeep++] = (uint32_t)v;
      }
      fclose(kf);
      if (!nkeep) tdie("--keep-columns kept nothing", keep_path);
      fprintf(stderr, "  %-12s %u of %d columns kept (--keep-columns)\n",
              "bank", nkeep, m->n_patterns);
    }
    const uint32_t ncol = keep ? nkeep : (uint32_t)m->n_patterns;
    uint32_t *row = malloc((size_t)m->n_cells * sizeof(uint32_t));
    float *y = malloc((size_t)m->n_cells * sizeof(float));
    if (!row || !y) tdie("out of memory", NULL);
    uint32_t nr = 0;
    for (int r = 0; r < m->n_cells; ++r)
      for (uint32_t c = 0; c < t->n_samples; ++c)
        if (!strcmp(row_lab[r], t->labels[c])) {
          row[nr] = (uint32_t)r; y[nr++] = (float)c; break;
        }
    if (!nr) tdie("no training rows match the root's classes", data_path);
    float *X = malloc((size_t)nr * ncol * sizeof(float));
    if (!X) tdie("out of memory", NULL);
    for (uint32_t r = 0; r < nr; ++r)
      for (uint32_t c = 0; c < ncol; ++c)
        X[(size_t)r * ncol + c] =
          (float)m->M[(size_t)row[r] * m->n_patterns + (keep ? keep[c] : c)];
    int nrd = nrounds > 0 ? nrounds : (int)(sqrt((double)nr) + 0.5);
    if (nrd < 1) nrd = 1;
    BoosterHandle b = train_one(X, y, nr, ncol, (int)t->n_samples, nrd,
                                nthread, &tn, ch->name[rk]);
    ms_booster_set_meta(b, t->labels, (int)t->n_samples);
    if (flags & MSFM_FLAG_BIN_FLAT)     ms_booster_set_binarize(b, "0.5");
    else if (flags & MSFM_FLAG_BIN_PAT) ms_booster_set_binarize(b, "pattern");
    if (keep) {
      char **fn = malloc((size_t)ncol * sizeof(char *));
      if (!fn) tdie("out of memory", NULL);
      for (uint32_t c = 0; c < ncol; ++c) fn[c] = m->pattern_names[keep[c]];
      ms_booster_set_features(b, fn, (int)ncol);
      free(fn);
      ms_booster_set_colsel(b, keep, nkeep);
    } else ms_booster_set_features(b, m->pattern_names, (int)ncol);
    ms_booster_set_pooled(b);
    void *blob = NULL; uint64_t blen = 0;
    { char tmp[4096];
      const char *td = getenv("TMPDIR");
      snprintf(tmp, sizeof tmp, "%s/methscope_tree_XXXXXX.ubj",
               td && *td ? td : "/tmp");
      int fd = mkstemps(tmp, 4);
      if (fd < 0) tdie("cannot create temporary model", tmp);
      close(fd);
      XGCHK(XGBoosterSaveModel(b, tmp));
      FILE *f = fopen(tmp, "rb");
      if (!f) tdie("cannot read temporary model", tmp);
      fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
      blob = malloc((size_t)sz); blen = (uint64_t)sz;
      if (!blob || fread(blob, 1, (size_t)sz, f) != (size_t)sz)
        tdie("short read on temporary model", tmp);
      fclose(f); unlink(tmp); }
    XGBoosterFree(b);
    { char *nm1 = strdup(ch->name[rk]);
      void *bl1[1] = {blob}; uint64_t bn1[1] = {blen}; char *nms[1] = {nm1};
      ms_bundle_pack_tree(out, chain, 1, nms, bl1, bn1);
      free(nm1); }
    fprintf(stderr, "\n  %-12s 1 pooled booster + the chain -> %s\n",
            "wrote", out);
    fprintf(stderr, "  %-12s methscope classify %s query.cg\n", "score with",
            out);
    free(blob); free(X); free(row); free(y); free(keep);
    ms_mrmp_top_free(t); free(nodeof);
    ms_msfm_layout_free(lay); ms_mrmpset_free(ch);
    unlink(chain); free(chain);
    return 0;
  }

  free(nodeof);
  ms_msfm_layout_free(lay); ms_mrmpset_free(ch);
  unlink(chain); free(chain);
  return 0;
}

int main_train(int argc, char *argv[]) {
  const char *pos[4]; int npos = 0;
  /* xgboost + a prebuilt matrix is the tree path: the .msfm carries the chain
   * it was featurized against, so training walks its nodes and emits one
   * scorable bundle. The other frameworks are UNFITTED -- they transcribe a
   * .mrmp and take no training data at all -- so they stay here. */
  { const char *fw = "xgboost"; int has_data = 0, want_help = 0;
    for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--framework") && i + 1 < argc) fw = argv[i + 1];
      else if (!strcmp(argv[i], "--data") && i + 1 < argc) has_data = 1;
      else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) want_help = 1;
    }
    /* -h before the --data check, or `classify-train -h` -- the commonest way
     * to ask what the flags ARE -- dies telling you to pass one. */
    if (want_help) { return train_usage(stdout); }
    if (!strcmp(fw, "xgboost")) {
      /* xgboost is tree-only now. The flat single-booster format is gone --
       * scoring it was deleted with the second featurizer it depended on, so
       * emitting one would produce a bundle nothing can read. --data is what
       * pins the column layout to the chain the model is trained against. */
      if (!has_data)
        tdie("xgboost training needs --data <in.msfm> (classify-featurize): the "
             "flat single-booster format was removed, and a tree's columns are "
             "defined by the artifact its features were built from", NULL);
      return main_train_tree(argc, argv);
    }
  }

  const char *labels_path = NULL, *out_path = NULL, *framework = "xgboost";
  const char *data_path = NULL;
  int npattern = 0;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-l") == 0 && i+1 < argc) labels_path = argv[++i];
    else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_path = argv[++i];
    else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_path    = argv[++i];
    /* xgboost knobs. Every xgboost run is routed to main_train_tree() above,
     * which parses these itself; here they are accepted and ignored so a
     * flag the shared -h documents never dies as "unrecognized" on the
     * logistic path. */
    else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--threads")
              || !strcmp(argv[i], "--eval-every") || !strcmp(argv[i], "--max-depth")
              || !strcmp(argv[i], "--min-child-weight") || !strcmp(argv[i], "--colsample"))
             && i+1 < argc) ++i;
    else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
      npattern = parse_nonneg_int(argv[++i], "-p expects a non-negative integer");
    else if (strcmp(argv[i], "--framework") == 0 && i+1 < argc) framework = argv[++i];
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      return train_usage(stdout);
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      tdie("unrecognized or incomplete option", argv[i]);
    else if (npos < (int)(sizeof pos / sizeof *pos)) pos[npos++] = argv[i];
    else break;   /* too many positionals: the tail's own check reports it */
  }
  if (!strcmp(framework, "violation"))
    tdie("the violation rule is UNFITTED -- it transcribes a .mrmp and consumes "
         "no training data, so it is a scoring mode now: "
         "`classify --framework violation ref.mrmp query.cg` "
         "(verified identical to transcribe-then-score, 0 of 1,201 cells)",
         framework);
  /* xgboost never arrives here (routed to the tree above), so from this point
   * the framework is logistic: the one fitted model that still takes a raw
   * <query.cg> <ref.cm> and features it itself. */
  if (strcmp(framework, "logistic"))
    tdie("unknown --framework (xgboost|logistic)", framework);

  /* With --data the features are already built, so <query.cg> drops out and
   * only <ref.cm> stays positional -- the bundle still has to carry the MRMP. */
  int want_pos = data_path ? 1 : 2;
  if (!out_path || npos != want_pos) return train_usage(stderr);
  if (!data_path && !labels_path) return train_usage(stderr);
  if (!ms_path_is_bundle_ext(out_path))
    tdie("the logistic framework requires a .clfx output (bundled with the MRMP)", out_path);
  const char *query_cg = data_path ? NULL : pos[0];
  char *tmp_mrmp = NULL;
  /* Keep the ARTIFACT path as given. ms_mrmp_resolve() materialises a runtime
   * mask for featurizing, but the bundle should carry the .mrmp itself: it is
   * self-describing, keeps set names and per-set structure, and needs no
   * sibling .idx -- which ms_bundle_pack cannot store anyway, so bundling a
   * resolved multi-record mask silently lost every record's name. The chain
   * walker stops at MSBNDL1, so a bundled .mrmp is readable straight off the
   * bundle prefix, exactly as a bundled .cm was. */
  const char *ref_arg  = pos[data_path ? 0 : 1];
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
    if (!ms_is_pna_name(m->pattern_names[c]))
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
   * the linear framework and the trim path that reads
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

  if (K != 2)
    tdie("the logistic framework is binary (needs exactly 2 classes)", NULL);

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

  int trimmed = 0;

  {
    /* ---- linear framework (logistic): interpretable binary rule ---- */
    linmodel_t *lm = ms_linmodel_fit(m, npattern, yidx, uniq[0], uniq[1], framework);
    char tmpl[4096];
    const char *td = getenv("TMPDIR");
    snprintf(tmpl, sizeof tmpl, "%s/methscope_lin_XXXXXX.lin",
             td && *td ? td : "/tmp");
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
  }

  free(yidx); free(uniq);
  for (int j = 0; j < n_lab; ++j) free(lab[j]);
  free(lab);
  ms_matrix_free(m);
  ms_mrmp_cleanup(tmp_mrmp);
  return 0;
}
