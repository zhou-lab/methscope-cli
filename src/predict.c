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
#include <inttypes.h>
#include <math.h>
#include <unistd.h>
#include "methscope.h"
#include "bmeta.h"
#include "bundle.h"
#include "msfm.h"
#include "mrmp.h"    /* ms_msfm_to_matrix -- the --data feature path */
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
    "  methscope classify [options] <query.cg> <model.clfx>\n"
    "  methscope classify [options] <query.cg> <ref.mrmp> <booster.ubj>\n"
    "  methscope classify --data <in.msfm> [options] <model.clfx>\n"
    "\n"
    "Purpose:\n"
    "  Predict a label (cell type, sex, ... — whatever the model was trained on)\n"
    "  and a confidence score for each query record, by featurizing the query\n"
    "  against the MRMP reference and running the booster.\n"
    "\n"
    "Arguments:\n"
    "  <query.cg>     Query methylome(s); '-' reads a .cg stream from stdin\n"
    "                 (cells are then named 1,2,3,... as a stream has no index).\n"
    "  <model.clfx>   A self-contained bundle of the model + its MRMP (from\n"
    "                 `classify-train -o model.clfx` or `bundle`) — the recommended\n"
    "                 single-file form.\n"
    "  <ref.mrmp>     MRMP pattern definition (a YAME .cm) to featurize the query\n"
    "                 (loose form). A bundle (.clfx/.updecx) also works here.\n"
    "  <booster.ubj>  Loose booster with class labels embedded (see train / bundle -l).\n"
    "\n"
    "Options:\n"
    "  -o <out.tsv>   Write output to a file instead of stdout.\n"
    "  --framework violation   Score straight off a .mrmp with the violation\n"
    "                 rule instead of a fitted model: give <query.cg> <ref.mrmp>.\n"
    "                 The rule is UNFITTED -- a pure function of the artifact and\n"
    "                 the three knobs below -- so there is nothing to train and\n"
    "                 nothing worth storing in between.\n"
    "  --call-threshold t      violation: beta cutoff for calling a pattern (0.5)\n"
    "  --pattern-weight w      violation: sqrt|log1p|linear|flat (sqrt)\n"
    "  --min-patterns n        violation: patterns required on each side (20)\n"
    "  --top K                 violation: patterns to use, by rank (1000)\n"
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

/* Inference for the `violation` framework: featurize (or take prebuilt
 * features), then call the class the query contradicts least. Unlike the linear
 * path this accepts --data and --threads, because it consumes plain betas
 * exactly as the booster does -- which is also how the C and reference
 * implementations get validated against one identical feature matrix. */
/* ------------------------------------------------------------------- tree ---
 * Score a routing tree: featurize the query ONCE against the whole chain, then
 * walk root -> child, each node deciding only the cells its parent sent it.
 *
 * One featurize, not one per level. A node's columns are bit-identical computed
 * in-chain and alone (verified on 1,201 Bian cells against the 7-node human
 * tree, 0 rows differing), and decompression dominates the cost, so the driver
 * this replaces was paying for a second pass plus a multi-GB `yame subset` copy
 * at every level to get the same numbers.
 *
 * Columns come from ms_msfm_layout(), never from the booster's feature NAMES.
 * Names would appear to work and be silently wrong: a node's booster stores
 * P1..Pk, and in a fused matrix those match the first k columns, which belong
 * to the ROOT. */

typedef struct {
  char         *name;
  uint32_t      col0, ncol;      /* its slice of the fused matrix */
  uint32_t      n_class;
  char        **cls;             /* n_class, borrowed from the chain view */
  BoosterHandle bst;
  char        **lab;             /* K labels, class-index order */
  int           K;
  int           parent;          /* index, -1 for the root */
} tnode_t;

/* The featurize mode is not recorded anywhere, so recover it: only one mode
 * makes every node's layout width equal its booster's feature count. Trying
 * them is also a check -- a bundle whose boosters disagree with every mode is
 * mispaired with its chain, which is exactly what must not score silently. */
static uint32_t tree_mode(const char *bundle, tnode_t *nd, uint32_t n) {
  static const uint32_t TRY[] = { MSFM_FLAG_RANK_ONLY, 0u, MSFM_FLAG_RANK_ADD,
                                  MSFM_FLAG_CONTRAST_ONLY, MSFM_FLAG_CONTRAST_ADD };
  for (uint32_t t = 0; t < sizeof TRY / sizeof *TRY; ++t) {
    ms_msfm_layout_t *l = ms_msfm_layout(bundle, TRY[t]);
    int ok = (l->n_sets == n);
    for (uint32_t k = 0; ok && k < n; ++k) {
      bst_ulong nf = 0;
      XGCHK(XGBoosterGetNumFeature(nd[k].bst, &nf));
      ok = ((uint32_t)nf == l->ncol[k]);
    }
    if (ok) {
      for (uint32_t k = 0; k < n; ++k)
        { nd[k].col0 = l->col0[k]; nd[k].ncol = l->ncol[k]; }
      ms_msfm_layout_free(l);
      return TRY[t];
    }
    ms_msfm_layout_free(l);
  }
  pdie("no feature mode makes this tree's boosters match its chain -- the "
       "bundle's models and mrmp disagree", bundle);
  return 0;
}

/* argmax of one node's booster over the rows given, into out_lab/out_conf. */
static void tree_score(tnode_t *nd, const uint16_t *beta, uint32_t ncol_all,
                       const uint32_t *row, uint32_t nrow,
                       char **out_lab, double *out_conf, const char *bundle) {
  if (!nrow) return;
  float *data = malloc((size_t)nrow * nd->ncol * sizeof(float));
  if (!data) pdie("out of memory (tree scoring)", NULL);
  for (uint32_t r = 0; r < nrow; ++r)
    for (uint32_t c = 0; c < nd->ncol; ++c) {
      uint16_t v = beta[(size_t)row[r] * ncol_all + nd->col0 + c];
      data[(size_t)r * nd->ncol + c] = (v == MSFM_NA) ? NAN : (float)msfm_decode(v);
    }
  DMatrixHandle dm;
  XGCHK(XGDMatrixCreateFromMat(data, nrow, nd->ncol, NAN, &dm));
  bst_ulong olen; const float *o;
  XGCHK(XGBoosterPredict(nd->bst, dm, 0, 0, 0, &olen, &o));
  int K = (int)(olen / nrow);
  if (K < 1 || olen != (bst_ulong)nrow * K) pdie("unexpected prediction length", bundle);
  for (uint32_t r = 0; r < nrow; ++r) {
    int best = 0;
    for (int c = 1; c < K; ++c) if (o[(size_t)r * K + c] > o[(size_t)r * K + best]) best = c;
    out_lab[row[r]]  = (nd->lab && best < nd->K) ? nd->lab[best] : nd->cls[best];
    out_conf[row[r]] = o[(size_t)r * K + best];
  }
  XGDMatrixFree(dm);
  free(data);
}

static int predict_tree(const char *query_cg, const char *bundle,
                        unsigned threads, const char *out_path, int no_header) {
  ms_mrmpset_t *ch = ms_mrmpset_open(bundle);
  const uint32_t n = ch->n_sets;
  tnode_t *nd = calloc(n, sizeof(tnode_t));
  mrmp_top_t **top = calloc(n, sizeof(mrmp_top_t *));
  if (!nd || !top) pdie("out of memory (tree)", NULL);
  for (uint32_t k = 0; k < n; ++k) {
    nd[k].name = ch->name[k];
    top[k] = ms_mrmp_top_read_at(bundle, ch->block_off[k], 1);
    nd[k].n_class = top[k]->n_samples;
    nd[k].cls = top[k]->labels;
    size_t blen; void *bb = ms_bundle_section(bundle, nd[k].name, &blen);
    XGCHK(XGBoosterCreate(NULL, 0, &nd[k].bst));
    XGCHK(XGBoosterLoadModelFromBuffer(nd[k].bst, bb, blen));
    free(bb);
    nd[k].lab = ms_booster_get_labels(nd[k].bst, &nd[k].K);
  }
  /* Parent from the NAME, minus its last dotted component -- the tree's only
   * structural record, since the 128-byte MRMP header has no room for a
   * pointer. Checked rather than assumed: a missing parent, overlapping
   * siblings or a child straying outside its parent's classes would each route
   * cells somewhere unrecoverable. */
  for (uint32_t k = 0; k < n; ++k) {
    nd[k].parent = -1;
    const char *dot = strrchr(nd[k].name, '.');
    if (!dot) continue;
    size_t plen = (size_t)(dot - nd[k].name);
    for (uint32_t j = 0; j < n; ++j)
      if (strlen(nd[j].name) == plen && !strncmp(nd[j].name, nd[k].name, plen))
        { nd[k].parent = (int)j; break; }
    if (nd[k].parent < 0) pdie("node's parent is missing from the bundle", nd[k].name);
    for (uint32_t c = 0; c < nd[k].n_class; ++c) {
      uint32_t p = (uint32_t)nd[k].parent, seen = 0;
      for (uint32_t d = 0; d < nd[p].n_class; ++d)
        seen |= !strcmp(nd[p].cls[d], nd[k].cls[c]);
      if (!seen) pdie("node has a class its parent does not", nd[k].cls[c]);
    }
  }
  for (uint32_t a = 0; a < n; ++a)
    for (uint32_t b = a + 1; b < n; ++b) {
      if (nd[a].parent != nd[b].parent) continue;
      for (uint32_t x = 0; x < nd[a].n_class; ++x)
        for (uint32_t y = 0; y < nd[b].n_class; ++y)
          if (!strcmp(nd[a].cls[x], nd[b].cls[y]))
            pdie("sibling nodes share a class, so routing is ambiguous", nd[a].cls[x]);
    }
  uint32_t root = 0;
  { int found = 0;
    for (uint32_t k = 0; k < n; ++k) if (nd[k].parent < 0) { root = k; ++found; }
    if (found != 1) pdie("tree must have exactly one root", bundle); }

  const uint32_t mode = tree_mode(bundle, nd, n);

  /* one pass over the query, every node's columns for every cell */
  uint32_t one = 0, ncells = 0, ncol = 0, *levels = NULL;
  uint16_t *beta = NULL; char **cellname = NULL;
  uint64_t *base = malloc(n * sizeof(uint64_t)), *blen2 = malloc(n * sizeof(uint64_t));
  uint32_t *np = malloc(n * sizeof(uint32_t));
  if (!base || !blen2 || !np) pdie("out of memory (tree)", NULL);
  const char **rr = malloc(n * sizeof(char *));
  for (uint32_t k = 0; k < n; ++k) {
    rr[k] = bundle; base[k] = ch->block_off[k]; blen2[k] = ch->block_bytes[k];
    mrmp_top_t *t = ms_mrmp_top_read_at(bundle, ch->block_off[k], UINT32_MAX);
    np[k] = t->n_patterns; ms_mrmp_top_free(t);
  }
  uint32_t *col0 = malloc(n * sizeof(uint32_t));
  ms_msfm_build_sampled_multi(query_cg, rr, base, blen2, np, n, &one, 1,
                              1 /* -b */, 0, 20260812, threads, 1 /* 0.5 cut */,
                              (mode & MSFM_FLAG_CONTRAST_ONLY) ? 2 :
                              (mode & MSFM_FLAG_CONTRAST_ADD)  ? 1 : 0,
                              (mode & MSFM_FLAG_RANK_ONLY) ? 2 :
                              (mode & MSFM_FLAG_RANK_ADD)  ? 1 : 0,
                              &beta, &levels, &cellname, &ncells, &ncol, col0);

  char  **lab  = calloc(ncells, sizeof(char *));
  double *conf = calloc(ncells, sizeof(double));
  uint32_t *cur = malloc((size_t)ncells * sizeof(uint32_t));
  uint32_t *nxt = malloc((size_t)ncells * sizeof(uint32_t));
  if (!lab || !conf || !cur || !nxt) pdie("out of memory (tree)", NULL);

  /* Hard routing: a node decides, the cell descends to the child holding that
   * call, and a cell whose call no child covers is finished. Breadth-first by
   * node so each booster runs once over all the cells that reached it. */
  uint32_t *at = calloc(ncells, sizeof(uint32_t));
  for (uint32_t r = 0; r < ncells; ++r) at[r] = root;
  for (uint32_t pass = 0; pass < n; ++pass) {
    int moved = 0;
    for (uint32_t k = 0; k < n; ++k) {
      uint32_t m2 = 0;
      for (uint32_t r = 0; r < ncells; ++r) if (at[r] == k && !lab[r]) cur[m2++] = r;
      if (!m2) continue;
      tree_score(&nd[k], beta, ncol, cur, m2, lab, conf, bundle);
      for (uint32_t j = 0; j < m2; ++j) {
        uint32_t r = cur[j]; int dest = -1;
        for (uint32_t c = 0; c < n && dest < 0; ++c) {
          if (nd[c].parent != (int)k) continue;
          for (uint32_t x = 0; x < nd[c].n_class; ++x)
            if (!strcmp(nd[c].cls[x], lab[r])) { dest = (int)c; break; }
        }
        if (dest >= 0) { at[r] = (uint32_t)dest; lab[r] = NULL; moved = 1; }
      }
    }
    if (!moved) break;
  }

  FILE *fo = out_path ? fopen(out_path, "w") : stdout;
  if (!fo) pdie("cannot open output", out_path);
  if (!no_header) fprintf(fo, "cell\tprediction_label\tconfidence\n");
  for (uint32_t r = 0; r < ncells; ++r)
    fprintf(fo, "%s\t%s\t%.6f\n", cellname[r], lab[r] ? lab[r] : "NA", conf[r]);
  if (fo != stdout) fclose(fo);

  for (uint32_t k = 0; k < n; ++k) { XGBoosterFree(nd[k].bst); ms_mrmp_top_free(top[k]); }
  free(nd); free(top); free(beta); free(levels); free(lab); free(conf);
  free(cur); free(nxt); free(at); free(base); free(blen2); free(np);
  free((void *)rr); free(col0);
  for (uint32_t r = 0; r < ncells; ++r) free(cellname[r]);
  free(cellname);
  ms_mrmpset_free(ch);
  return 0;
}

/* Takes the model OBJECT, so it serves both a transcribed bundle and a .mrmp
 * scored directly -- the violation model is a pure function of the .mrmp plus
 * three parameters, so there is nothing to fit and nothing worth storing. */
static int predict_violation(const char *query_cg, const char *ref_mrmp,
                             const char *data_path, unsigned threads,
                             viomodel_t *vm,
                             const char *out_path, int with_probs, int no_header) {
  uint32_t *levels = NULL;
  ms_matrix_t *m;
  if (data_path) {
    m = ms_msfm_to_matrix(data_path, NULL, &levels);
  } else if (threads > 1) {
    char *fidx = get_fname_index((char *)query_cg);
    int have_idx = fidx && access(fidx, R_OK) == 0;
    free(fidx);
    if (have_idx) {
      m = ms_matrix_build_threaded(query_cg, ref_mrmp, (uint32_t)vm->n_feat,
                                   threads, &levels);
    } else {
      fprintf(stderr, "[methscope] classify: --threads needs a .cg index; "
              "falling back to the single-threaded scan\n");
      m = ms_matrix_build(query_cg, ref_mrmp);
    }
  } else {
    m = ms_matrix_build(query_cg, ref_mrmp);
  }
  if (m->n_patterns < vm->n_feat)
    pdie("reference .mrmp has fewer patterns than the model expects", ref_mrmp);

  FILE *fout = (out_path && strcmp(out_path, "-") != 0) ? fopen(out_path, "w") : stdout;
  if (!fout) pdie("cannot open output", out_path);
  if (!no_header) {
    fputs("cell\tprediction_label\tconfidence", fout);
    if (with_probs)
      for (int k = 0; k < vm->n_label; ++k) fprintf(fout, "\t%s", vm->labels[k]);
    fputc('\n', fout);
  }
  double *betas = malloc((size_t)vm->n_feat * sizeof(double));
  double *all   = malloc((size_t)vm->n_label * sizeof(double));
  if (!betas || !all) pdie("out of memory (violation scoring)", NULL);
  for (int r = 0; r < m->n_cells; ++r) {
    const double *src = m->M + (size_t)r * m->n_patterns;
    for (int c = 0; c < vm->n_feat; ++c) betas[c] = src[c];
    double score, margin;
    int k = ms_viomodel_score(vm, betas, &score, &margin);
    /* A record too sparse to clear min_patterns is reported as NA rather than
     * given a fabricated call -- the whole point of the margin is to admit
     * when the evidence is not there. */
    if (k < 0) fprintf(fout, "%s\tNA\tNA", m->cell_names[r]);
    else fprintf(fout, "%s\t%s\t%.6f", m->cell_names[r], vm->labels[k],
                 margin == margin ? margin : 0.0);
    if (with_probs) {
      ms_viomodel_scores(vm, betas, all);
      for (int c = 0; c < vm->n_label; ++c)
        if (all[c] == all[c] && all[c] != (double)INFINITY)
          fprintf(fout, "\t%.6f", all[c]);
        else fputs("\tNA", fout);
    }
    fputc('\n', fout);
  }
  free(betas); free(all); free(levels);
  if (fout != stdout) fclose(fout);
  ms_matrix_free(m); ms_viomodel_free(vm);
  return 0;
}

int main_predict(int argc, char *argv[]) {
  const char *out_path = NULL, *data_path = NULL;
  unsigned threads = 1;
  int with_probs = 0, with_levels = 0;
  const char *one_level = NULL;
  int no_header = 0;
  /* violation scored straight off a .mrmp: it is UNFITTED, a pure function of
   * the artifact and these three numbers, so transcribing it to a bundle first
   * stored nothing and made changing a parameter a rebuild. */
  int fw_violation = 0, vio_min_patterns = 20;
  double vio_threshold = 0.5;
  const char *vio_weight = "sqrt";
  uint32_t vio_top = 1000;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) data_path = argv[++i];
    else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
      threads = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--framework") == 0 && i + 1 < argc) {
      if (strcmp(argv[++i], "violation"))
        pdie("classify --framework takes only 'violation'; a fitted model "
             "carries its own framework mark", argv[i]);
      fw_violation = 1;
    }
    else if (strcmp(argv[i], "--call-threshold") == 0 && i + 1 < argc)
      vio_threshold = atof(argv[++i]);
    else if (strcmp(argv[i], "--pattern-weight") == 0 && i + 1 < argc)
      vio_weight = argv[++i];
    else if (strcmp(argv[i], "--min-patterns") == 0 && i + 1 < argc)
      vio_min_patterns = atoi(argv[++i]);
    else if (strcmp(argv[i], "--top") == 0 && i + 1 < argc)
      vio_top = (uint32_t)strtoul(argv[++i], NULL, 10);
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

  /* --framework violation: the second argument is the .mrmp itself, not a
   * bundle. Nothing is trained, so there is no model file in between. */
  if (fw_violation) {
    if (argc - i != 2) return predict_usage();
    const char *art = argv[i + 1];
    if (!ms_mrmp_is_artifact(art))
      pdie("--framework violation needs the MRMPIDX1 artifact (.mrmp); an "
           "exported .cm has no binstrings to read the rule from", art);
    viomodel_t *vm = ms_viomodel_from_mrmp(art, vio_top, vio_threshold,
                                           vio_weight, vio_min_patterns);
    /* The matrix builders take a runtime .cm, and ms_mrmp_resolve() would give
     * only the chain's FIRST block -- the rule is pooled across every set, so
     * materialise the whole chain, exactly as the transcribe step used to. */
    char ctmp[] = "/tmp/methscope_viomask_XXXXXX.cm";
    int cfd = mkstemps(ctmp, 3);
    if (cfd < 0) pdie("cannot create temp mask file", NULL);
    close(cfd);
    ms_mrmp_write_mask(art, ctmp, "Pna", (uint32_t)vm->n_feat);
    int rc = predict_violation(query_cg, ctmp, data_path, threads, vm,
                               out_path, with_probs, no_header);
    unlink(ctmp);
    { char idx[64]; snprintf(idx, sizeof idx, "%s.idx", ctmp); unlink(idx); }
    return rc;
  }

  if (argc - i == 2) {
    /* bundle form: query.cg model.ubjx (model + mrmp in one file) */
    model_name = argv[i + 1];
    if (!ms_bundle_is(model_name))
      pdie("expected a .clfx bundle; for a bare booster give <ref.mrmp> <booster.ubj>", model_name);
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
    if (strcmp(kind, "tree") == 0) {
      if (data_path)
        pdie("--data is not supported for a tree: it featurizes the whole chain "
             "itself, in one pass", model_name);
      int rc = predict_tree(query_cg, model_name, threads, out_path, no_header);
      free(kind);
      return rc;
    }
    if (strcmp(kind, "violation") == 0) {
      size_t blen; void *bbuf = ms_bundle_section(model_name, "model", &blen);
      int rc = predict_violation(query_cg, ref_mrmp, data_path, threads,
                                 ms_viomodel_parse(bbuf, blen),
                                 out_path, with_probs, no_header);
      free(bbuf); free(kind);
      return rc;
    }
    if (strcmp(kind, "xgboost") != 0)
      pdie("unknown model framework 'kind' "
           "(expected xgboost/violation/threshold/logistic)", kind);
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
      pdie("got a bundle where a bare booster.ubj was expected; use: classify query.cg model.clfx", model_name);
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
  /* Resolve the model's feature columns BY NAME.
   *
   * Taking the first n_pat columns positionally is wrong whenever the query
   * artifact is a fused multi-set .msfm: that layout is set-major, so each
   * set's Pna background sits between pattern blocks rather than after them,
   * and a positional cut both admits backgrounds and drops the tail sets. The
   * booster records the names it was trained on, so gather exactly those. */
  {
    int n_stored = 0;
    char **fn = ms_booster_get_features(booster, &n_stored);
    if (fn) {
      if (n_stored != n_pat)
        pdie("model feature list disagrees with its own feature count", model_name);
      int *idx = malloc((size_t)n_stored * sizeof(int));
      if (!idx) pdie("out of memory", NULL);
      for (int c = 0; c < n_stored; ++c) {
        idx[c] = -1;
        for (int q = 0; q < m->n_patterns; ++q)
          if (!strcmp(fn[c], m->pattern_names[q])) { idx[c] = q; break; }
        if (idx[c] < 0)
          pdie("query is missing a feature the model needs", fn[c]);
      }
      ms_matrix_select(m, idx, n_stored);
      free(idx);
      for (int c = 0; c < n_stored; ++c) free(fn[c]);
      free(fn);
    } else if (m->n_patterns != n_pat) {
      /* Pre-2026-08 model: no name list, so the columns can only be taken
       * positionally, which is exactly the case that used to go wrong quietly.
       * Say so rather than scoring on a guess. */
      fprintf(stderr, "[methscope] classify: model predates feature-name "
              "recording and the query has %d patterns for %d the model wants; "
              "taking the first %d BY POSITION, which is only correct if the "
              "artifact is single-set. Retrain to remove this ambiguity.\n",
              m->n_patterns, n_pat, n_pat);
    }
  }
  if (m->n_patterns < n_pat)
    pdie("reference .mrmp has fewer patterns than the booster expects", ref_mrmp);

  /* Reproduce the training feature coding.
   *
   * classify --data reads a .msfm whose pattern columns are already coded, but
   * this path built them with ms_matrix_build(), which returns a continuous
   * mean. Scoring a model trained on {0,1,NA} against betas in [0,1] is a
   * silent feature-space mismatch, and it does not degrade gracefully: measured
   * at 2 of 42 cells correct, 39 of 43 collapsed onto a single class. */
  {
    char *how = ms_booster_get_binarize(booster);
    if (how && !data_path)
      /* Refuse rather than answer confidently and wrongly. Coding is only half
       * of it: ms_matrix_build_threaded() also passes `patterns = n_pat` to the
       * featurizer, so the .cg route RE-SELECTS the top-N patterns by rank
       * instead of using the model's own columns -- measured at 1.20% of CpGs
       * carrying a pattern against the artifact's 7.69%. Both selections are
       * named P1..PN, so the MS_ATTR_FEATURES check matches and nothing errors:
       * right names, wrong CpGs, 0 of 40 cells correct. Until scoring
       * featurizes against the model's exact artifact, this path cannot honour
       * a model trained through a .msfm. */
      pdie("this model was trained on binarised features and scoring a .cg "
           "cannot yet reproduce that feature space (it re-selects patterns by "
           "rank); featurize first with classify-featurize and pass --data",
           model_name);
    if (how && !strcmp(how, "0.5")) {
      if (data_path) {
        /* already coded by the featurizer -- re-cutting is a no-op on {0,1} but
         * would turn a 0.5-valued CONTINUOUS artifact into NA, so don't. */
      } else {
        uint64_t nbin = 0, ntie = 0;
        for (size_t k = 0; k < (size_t)m->n_cells * m->n_patterns; ++k) {
          double b = m->M[k];
          if (b != b) continue;                       /* already missing */
          if (b == 0.5) { m->M[k] = 0.0 / 0.0; m->N[k] = 0; ++ntie; ++nbin; continue; }
          m->M[k] = b > 0.5 ? 1.0 : 0.0; ++nbin;
        }
        fprintf(stderr, "[methscope] classify: binarised %" PRIu64 " feature(s) "
                "at 0.5 to match the model (%" PRIu64 " tie(s) -> NA)\n", nbin, ntie);
      }
    } else if (how && !strcmp(how, "pattern")) {
      if (!data_path)
        pdie("model was trained with --thresh-pattern, whose cuts are per-pattern "
             "and derived from the reference; score it with --data <.msfm> built "
             "the same way, not from a .cg", model_name);
    } else if (!how && !data_path) {
      /* Pre-2026-08-09 model: continuous features, and this path is continuous
       * too, so they agree. Nothing to do, and nothing to warn about. */
    }
    free(how);
  }

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
