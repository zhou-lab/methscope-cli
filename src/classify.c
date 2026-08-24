// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Label prediction (the C replacement for the R PredictCellType()). The label
 * can be any class the model was trained on (cell type, sex, ...). Builds the
 * record x pattern matrix from the query and the MRMP reference (<ref.mrmp>, a
 * YAME .cm), runs the XGBoost booster (with class labels embedded as
 * attributes), and reports a per-record label with both P(called class) and
 * a normalized-entropy certainty.
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

static int predict_usage(FILE *out) {
  ms_help(out,
    "\n"
    "Usage:\n"
    "  methscope classify [options] <model.clfx> <query.cg>\n"
    "  methscope classify --data <in.msfm> [options] <model.clfx>\n"
    "\n"
    "Purpose:\n"
    "  Predict a label (cell type, sex, ... — whatever the model was trained on)\n"
    "  and a confidence score for each query record, by featurizing the query\n"
    "  against the MRMP reference and running the booster.\n"
    "\n"
    "Arguments:\n"
    "  <model.clfx>   A self-contained bundle of the model + its MRMP (from\n"
    "                 `classify-train -o model.clfx` or `bundle`) — the recommended\n"
    "                 single-file form.\n"
    "  <query.cg>     Query methylome(s); '-' reads a .cg stream from stdin\n"
    "                 (cells are then named 1,2,3,... as a stream has no index).\n"
    "                 Query records may be YAME format 3 (M/U counts) or\n"
    "                 format 6 (universe bit plus binary 0/1 call); positions\n"
    "                 outside the format-6 universe are treated as missing.\n"
    "\n"
    "Options:\n"
    "  -o <out.tsv>             Write output to a file instead of stdout.\n"
    "  --framework violation   Score directly from a .mrmp with the violation\n"
    "                          rule. Give <query.cg> <ref.mrmp>. The rule is an\n"
    "                          unfitted function of the artifact and the three\n"
    "                          options below.\n"
    "  --call-threshold T      Violation beta cutoff. Default: 0.5.\n"
    "  --pattern-weight W      Violation weighting: sqrt, log1p, linear, or flat.\n"
    "                          Default: sqrt.\n"
    "  --min-patterns N        Patterns required on each side. Default: 20.\n"
    "  --top K                 Patterns to use by rank. Default: 1000.\n"
    "  --probs                 Append one predicted-probability column per class.\n"
    "                          This is unavailable for routing trees because each\n"
    "                          node scores a different class subset.\n"
    "  --levels                Append the predicted label's full taxonomy path:\n"
    "                          compartment, lineage, group, and subtype.\n"
    "  --level NAME            Append one taxonomy level: compartment, lineage,\n"
    "                          group, or subtype. Taxonomy options require a model\n"
    "                          trained with `classify-train --hierarchy`.\n"
    "  --no-header             Suppress the output header.\n"
    "  --threads T             Featurize with T workers. Default: 1. Workers seek\n"
    "                          indexed query records; streams use the serial path.\n"
    "  --data <in.msfm>        Score prebuilt features from classify-featurize.\n"
    "                          Omit <query.cg> when this option is used.\n"
    "  -h                      Show this help message.\n"
    "\n"
    "Output columns:\n"
    "  cell  prediction_label  confidence  certainty  [<class1> ... with --probs]\n"
    "  confidence = P(called class); certainty = 1 - H(p)/log(K), 0 at a\n"
    "  uniform posterior and 1 at a decided one. `violation` has no posterior\n"
    "  and reports a margin instead.\n"
    "\n");
  return out == stdout ? 0 : 1;
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
    fputs("cell\tprediction_label\tconfidence\tcertainty", fout);
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
    fprintf(fout, "%s\t%s\t%.6f\t%.6f", m->cell_names[r],
            cls ? lm->label1 : lm->label0, p1 >= 0.5 ? p1 : 1.0 - p1, conf);
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
  ms_colspan_t *cs;              /* its slice, plus one per satellite */
  uint32_t      n_class;
  char        **cls;             /* n_class, borrowed from the chain view */
  BoosterHandle bst;
  char        **lab;             /* K labels, class-index order */
  int           K;
  int           parent;          /* index, -1 for the root */
} tnode_t;

/* The LAYOUT mode is not recorded anywhere, so recover it: only one mode makes
 * every node's layout width equal its booster's feature count. Trying them is
 * also a check -- a bundle whose boosters disagree with every mode is mispaired
 * with its chain, which is exactly what must not score silently.
 *
 * Only the layout. The feature CODING is recorded and is read separately by
 * tree_binarize(); the width test cannot see it. */
static uint32_t tree_mode(const char *bundle, tnode_t *nd, uint32_t n) {
  static const uint32_t TRY[] = { MSFM_FLAG_RANK_ONLY, 0u, MSFM_FLAG_RANK_ADD,
                                  MSFM_FLAG_CONTRAST_ONLY, MSFM_FLAG_CONTRAST_ADD };
  for (uint32_t t = 0; t < sizeof TRY / sizeof *TRY; ++t) {
    ms_msfm_layout_t *l = ms_msfm_layout(bundle, TRY[t]);
    ms_colspan_t **cs = calloc(n, sizeof(*cs));
    if (!cs) pdie("out of memory (tree)", NULL);
    int ok = 1;
    for (uint32_t k = 0; ok && k < n; ++k) {
      bst_ulong nf = 0;
      XGCHK(XGBoosterGetNumFeature(nd[k].bst, &nf));
      cs[k] = ms_msfm_colspan(l, nd[k].name);
      ok = cs[k] && ((uint32_t)nf == cs[k]->total);
    }
    if (ok) {
      for (uint32_t k = 0; k < n; ++k) nd[k].cs = cs[k];
      free(cs);
      ms_msfm_layout_free(l);
      return TRY[t];
    }
    for (uint32_t k = 0; k < n; ++k) ms_colspan_free(cs[k]);
    free(cs);
    ms_msfm_layout_free(l);
  }
  pdie("no feature mode makes this tree's boosters match its chain -- the "
       "bundle's models and mrmp disagree", bundle);
  return 0;
}

/* argmax of one node's booster over the rows given.
 *
 * TWO numbers, because they answer different questions and were previously
 * conflated under one column name across the three frameworks. out_conf is
 * P(called class) -- how likely the winner is. out_cert is 1 - H(p)/log(K),
 * which is 0 at a uniform posterior and 1 at certainty -- how decisive the
 * call was against the alternatives it was chosen over. A 2-class node at
 * p=0.6 gives conf 0.600 and cert 0.029; the second says "barely decided",
 * which the first cannot. K is this NODE's class count, since that is the
 * decision actually being described. */
static void tree_score(tnode_t *nd, const uint16_t *beta, uint32_t ncol_all,
                       const uint32_t *row, uint32_t nrow,
                       char **out_lab, double *out_conf, double *out_cert,
                       const char *bundle) {
  if (!nrow) return;
  const uint32_t W = nd->cs->total;
  float *data = malloc((size_t)nrow * W * sizeof(float));
  if (!data) pdie("out of memory (tree scoring)", NULL);
  for (uint32_t r = 0; r < nrow; ++r) {
    uint32_t c = 0;
    for (uint32_t g = 0; g < nd->cs->n_seg; ++g)
      for (uint32_t j = 0; j < nd->cs->ncol[g]; ++j, ++c) {
        uint16_t v = beta[(size_t)row[r] * ncol_all + nd->cs->col0[g] + j];
        data[(size_t)r * W + c] = (v == MSFM_NA) ? NAN : (float)msfm_decode(v);
      }
  }
  DMatrixHandle dm;
  XGCHK(XGDMatrixCreateFromMat(data, nrow, W, NAN, &dm));
  bst_ulong olen; const float *o;
  XGCHK(XGBoosterPredict(nd->bst, dm, 0, 0, 0, &olen, &o));
  int K = (int)(olen / nrow);
  if (K < 1 || olen != (bst_ulong)nrow * K) pdie("unexpected prediction length", bundle);
  for (uint32_t r = 0; r < nrow; ++r) {
    int best = 0;
    for (int c = 1; c < K; ++c) if (o[(size_t)r * K + c] > o[(size_t)r * K + best]) best = c;
    out_lab[row[r]]  = (nd->lab && best < nd->K) ? nd->lab[best] : nd->cls[best];
    out_conf[row[r]] = o[(size_t)r * K + best];
    double ent = 0.0;
    for (int c = 0; c < K; ++c) {
      double q = o[(size_t)r * K + c];
      if (q > 0.0) ent -= q * log(q);
    }
    double cert = (K > 1) ? 1.0 - ent / log((double)K) : 1.0;
    out_cert[row[r]] = cert < 0.0 ? 0.0 : (cert > 1.0 ? 1.0 : cert);
  }
  XGDMatrixFree(dm);
  free(data);
}

/* The four taxonomy levels, in the order --hierarchy's TSV lists them. */
static const char *LEVEL_NAME[4] = {"compartment","lineage","group","subtype"};

/* label -> its four levels, parsed from a booster's embedded --hierarchy TSV.
 * Keyed by label rather than class index, because a routed cell's call comes
 * from whichever node decided it and class ids are per node. A label the table
 * omits reports NA rather than an empty column. */
typedef struct { char **lab; char **lvl; uint32_t n; } taxo_t;

static void taxo_parse(taxo_t *tx, char *raw) {
  memset(tx, 0, sizeof *tx);
  uint32_t cap = 64;
  tx->lab = calloc(cap, sizeof(char *));
  tx->lvl = calloc((size_t)cap * 4, sizeof(char *));
  if (!tx->lab || !tx->lvl) pdie("out of memory (hierarchy)", NULL);
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
    if (nf < 5) continue;                 /* header or short row */
    if (tx->n == cap) {
      cap <<= 1;
      tx->lab = realloc(tx->lab, cap * sizeof(char *));
      tx->lvl = realloc(tx->lvl, (size_t)cap * 4 * sizeof(char *));
      if (!tx->lab || !tx->lvl) pdie("out of memory (hierarchy)", NULL);
    }
    tx->lab[tx->n] = strdup(f[0]);
    for (int j = 0; j < 4; ++j) tx->lvl[(size_t)tx->n * 4 + j] = strdup(f[j + 1]);
    ++tx->n;
  }
}

static const char *taxo_get(const taxo_t *tx, const char *label, int lvl) {
  if (!label) return "NA";
  for (uint32_t i = 0; i < tx->n; ++i)
    if (!strcmp(tx->lab[i], label)) return tx->lvl[(size_t)i * 4 + lvl];
  return "NA";
}

static void taxo_free(taxo_t *tx) {
  for (uint32_t i = 0; i < tx->n; ++i) {
    free(tx->lab[i]);
    for (int j = 0; j < 4; ++j) free(tx->lvl[(size_t)i * 4 + j]);
  }
  free(tx->lab); free(tx->lvl);
}

/* The feature CODING, unlike the layout above, IS recorded: classify-train
 * writes it onto every node (bmeta.h, MS_ATTR_BINARIZE). Read it rather than
 * assume, because binarising does not change a node's column count, so
 * tree_mode()'s width test is blind to it -- a model trained with
 * --thresh-pattern or --continuous-features would be scored against a flat 0.5
 * cut, silently. Measured in the other direction, that mismatch put 2 of 42
 * cells right, with 39 of 43 collapsing onto one class.
 *
 * Absent means continuous: every model trained before 2026-08-09 predates the
 * attribute. Both shipped models record "0.5", which is what the .cg path used
 * to hardcode, so this changes nothing for them.
 *
 * Nodes must AGREE. One query pass codes the columns every node then reads, so
 * a bundle whose boosters were trained under different codings cannot be scored
 * coherently -- refuse rather than pick one. */
static int tree_binarize(const tnode_t *nd, uint32_t n, const char *bundle) {
  int mode = -1;
  for (uint32_t k = 0; k < n; ++k) {
    char *how = ms_booster_get_binarize(nd[k].bst);
    int m = !how ? 0
          : !strcmp(how, "0.5") ? 1
          : !strcmp(how, "pattern") ? 2 : -2;
    char msg[320];
    if (m == -2) {
      snprintf(msg, sizeof msg, "node %.120s records feature coding '%.60s', "
               "which this build does not know (expected '0.5' or 'pattern')",
               nd[k].name, how);
      free(how); pdie(msg, bundle);
    }
    free(how);
    if (mode < 0) { mode = m; continue; }
    if (m != mode) {
      snprintf(msg, sizeof msg, "node %.120s was trained with a different "
               "feature coding than an earlier node; one query pass cannot code "
               "the columns both ways", nd[k].name);
      pdie(msg, bundle);
    }
  }
  return mode < 0 ? 0 : mode;
}

static int predict_tree(const char *query_cg, const char *bundle,
                        const char *data_path, unsigned threads,
                        const char *out_path, int no_header,
                        int with_levels, int lvl_col) {
  ms_mrmpset_t *ch = ms_mrmpset_open(bundle);
  /* Satellites are feature providers, not nodes: they have no booster, no
   * children and no place in routing. Their columns reach a node through its
   * colspan, so everything below counts NODES only. */
  uint32_t n = 0;
  for (uint32_t s = 0; s < ch->n_sets; ++s)
    if (!ms_set_is_satellite(ch->name[s])) ++n;
  tnode_t *nd = calloc(n, sizeof(tnode_t));
  mrmp_top_t **top = calloc(n, sizeof(mrmp_top_t *));
  uint32_t *setof = calloc(n, sizeof(uint32_t));
  if (!nd || !top || !setof) pdie("out of memory (tree)", NULL);
  for (uint32_t s = 0, k = 0; s < ch->n_sets; ++s) {
    if (ms_set_is_satellite(ch->name[s])) continue;
    setof[k++] = s;
  }
  for (uint32_t k = 0; k < n; ++k) {
    nd[k].name = ch->name[setof[k]];
    top[k] = ms_mrmp_top_read_at(bundle, ch->block_off[setof[k]], 1);
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
  const int bin_feat = tree_binarize(nd, n, bundle);

  /* one pass over the query, every node's columns for every cell -- or none at
   * all when a prebuilt matrix is handed over. A .msfm featurized against this
   * chain already HOLDS every node's columns, so routing inside it needs no
   * query pass: the refusal that used to sit here was written when classify had
   * to featurize the chain itself. */
  uint32_t one = 0, ncells = 0, ncol = 0, *levels = NULL;
  uint16_t *beta = NULL; char **cellname = NULL;
  ms_msfm_t mf; int mf_open = 0;
  if (data_path) {
    char err[256];
    if (!ms_msfm_open(&mf, data_path, err, sizeof err)) pdie(err, data_path);
    mf_open = 1;
    ncells = mf.header->n_records; ncol = mf.header->n_patterns;
    beta = (uint16_t *)mf.beta;          /* borrowed; freed with the artifact */
    cellname = mf.record_names;
    /* The width IS the check: the layout is (chain + flags), so a matrix built
     * against a different tree cannot have this one's column count. Cheap, and
     * it turns a mispairing into a refusal rather than a confident wrong call
     * on the root's columns. */
    ms_msfm_layout_t *lay = ms_msfm_layout(bundle, mode);
    if (lay->total != ncol) {
      char m[192];
      snprintf(m, sizeof m, "this .msfm has %u columns, the tree's layout wants "
               "%u -- it was featurized against a different artifact", ncol,
               lay->total);
      ms_msfm_layout_free(lay);
      pdie(m, data_path);
    }
    ms_msfm_layout_free(lay);
  } else {
  /* EVERY set in the chain, not just the nodes: the layout the colspans index
   * into counts satellites too, so featurizing only `n` blocks builds a matrix
   * that is both too narrow AND misaligned -- `n` here would also index the
   * chain by POSITION, silently taking the first n sets rather than the node
   * ones. Human at 11 nodes over a 15-set chain lost root.4, the Enterocyte
   * leaf, and read its column out of a matrix that never had it: 766 of 1,201
   * Bian cells flipped colon -> small intestine. Invisible via --data, which
   * takes a matrix built by classify-featurize over the whole chain. */
  const uint32_t nall = ch->n_sets;
  uint64_t *base = malloc(nall * sizeof(uint64_t));
  uint64_t *blen2 = malloc(nall * sizeof(uint64_t));
  uint32_t *np = malloc(nall * sizeof(uint32_t));
  if (!base || !blen2 || !np) pdie("out of memory (tree)", NULL);
  const char **rr = malloc(nall * sizeof(char *));
  for (uint32_t k = 0; k < nall; ++k) {
    rr[k] = bundle; base[k] = ch->block_off[k]; blen2[k] = ch->block_bytes[k];
    mrmp_top_t *t = ms_mrmp_top_read_at(bundle, ch->block_off[k], UINT32_MAX);
    np[k] = t->n_patterns; ms_mrmp_top_free(t);
  }
  uint32_t *col0 = malloc(nall * sizeof(uint32_t));
  ms_msfm_build_sampled_multi(query_cg, rr, base, blen2, np, nall, &one, 1,
                              1 /* -b */, 0, 20260812, threads, bin_feat,
                              (mode & MSFM_FLAG_CONTRAST_ONLY) ? 2 :
                              (mode & MSFM_FLAG_CONTRAST_ADD)  ? 1 : 0,
                              (mode & MSFM_FLAG_RANK_ONLY) ? 2 :
                              (mode & MSFM_FLAG_RANK_ADD)  ? 1 : 0,
                              &beta, &levels, &cellname, &ncells, &ncol, col0);
  free(base); free(blen2); free(np); free((void *)rr); free(col0);
  /* The --data path checks this; the .cg path did not, which is how a matrix
   * missing whole sets scored 766 cells wrong without a word. Both paths feed
   * the same colspans, so both must agree with the layout. */
  { ms_msfm_layout_t *lay = ms_msfm_layout(bundle, mode);
    if (lay->total != ncol) {
      char m[192];
      snprintf(m, sizeof m, "featurized %u columns but the tree's layout wants "
               "%u -- a chain set was missed", ncol, lay->total);
      ms_msfm_layout_free(lay); pdie(m, query_cg);
    }
    ms_msfm_layout_free(lay); }
  }

  char  **lab  = calloc(ncells, sizeof(char *));
  double *conf = calloc(ncells, sizeof(double));
  double *cert = calloc(ncells, sizeof(double));
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
      tree_score(&nd[k], beta, ncol, cur, m2, lab, conf, cert, bundle);
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

  /* Every node carries the same taxonomy (classify-train --hierarchy writes it
   * to all of them), so the first one that has it answers for the tree. */
  taxo_t tx; memset(&tx, 0, sizeof tx);
  if (with_levels || lvl_col >= 0) {
    char *raw = NULL;
    for (uint32_t k = 0; k < n && !raw; ++k) raw = ms_booster_get_hier(nd[k].bst);
    if (!raw)
      pdie("model carries no hierarchy; retrain with "
           "`classify-train --hierarchy`", bundle);
    taxo_parse(&tx, raw);
    free(raw);
  }

  FILE *fo = out_path ? fopen(out_path, "w") : stdout;
  if (!fo) pdie("cannot open output", out_path);
  if (!no_header) {
    fputs("cell\tprediction_label\tconfidence\tcertainty", fo);
    if (with_levels)      for (int c = 0; c < 4; ++c) fprintf(fo, "\t%s", LEVEL_NAME[c]);
    else if (lvl_col >= 0) fprintf(fo, "\t%s", LEVEL_NAME[lvl_col]);
    fputc('\n', fo);
  }
  for (uint32_t r = 0; r < ncells; ++r) {
    fprintf(fo, "%s\t%s\t%.6f\t%.6f", cellname[r], lab[r] ? lab[r] : "NA",
            conf[r], cert[r]);
    if (with_levels)
      for (int c = 0; c < 4; ++c) fprintf(fo, "\t%s", taxo_get(&tx, lab[r], c));
    else if (lvl_col >= 0) fprintf(fo, "\t%s", taxo_get(&tx, lab[r], lvl_col));
    fputc('\n', fo);
  }
  taxo_free(&tx);
  if (fo != stdout) fclose(fo);

  for (uint32_t k = 0; k < n; ++k) { XGBoosterFree(nd[k].bst);
    ms_colspan_free(nd[k].cs); ms_mrmp_top_free(top[k]); }
  free(setof);
  free(nd); free(top); free(lab); free(conf);
  if (!mf_open) { free(beta); free(levels); }
  free(cur); free(nxt); free(at);
  if (mf_open) ms_msfm_close(&mf);
  else { for (uint32_t r = 0; r < ncells; ++r) free(cellname[r]); free(cellname); }
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
    /* `margin`, not `confidence`: the violation rule has no posterior, and
     * naming it confidence put a third unrelated quantity in that column. */
    fputs("cell\tprediction_label\tmargin", fout);
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
  int with_probs = 0;
  int no_header = 0;
  /* violation scored straight off a .mrmp: it is UNFITTED, a pure function of
   * the artifact and these three numbers, so transcribing it to a bundle first
   * stored nothing and made changing a parameter a rebuild. */
  int with_levels = 0, lvl_col = -1;
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
    else if (strcmp(argv[i], "--level") == 0 && i + 1 < argc) {
      const char *want = argv[++i];
      for (int c = 0; c < 4; ++c)
        if (!strcmp(want, LEVEL_NAME[c])) lvl_col = c;
      if (lvl_col < 0)
        pdie("--level must be compartment, lineage, group or subtype", want);
    }
    else if (strcmp(argv[i], "--no-header") == 0) no_header = 1;
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      return predict_usage(stdout);
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      pdie("unrecognized or incomplete option", argv[i]);
    else break;
  }
  /* With --data the features are prebuilt, so <query.cg> drops out of the
   * positional list; the model forms are otherwise unchanged. This is what
   * makes scoring many models on one test set cheap -- the featurization,
   * which dominates, happens once in classify-featurize instead of per run. */
  /* Exactly one model argument. The loose <ref.mrmp> <booster.ubj> form went
   * with the flat format: a bare booster has no framework mark and no chain, so
   * there is nothing to route and nothing to check the featurization against. */
  if (data_path) { if (argc - i != 1) return predict_usage(stderr); }
  else           { if (argc - i != 2) return predict_usage(stderr); }
  if (data_path) --i;                 /* so argv[i+1], argv[i+2] stay the model */
  const char *model_arg = argv[i];
  const char *query_cg  = data_path ? NULL : argv[i + 1];
  const char *ref_mrmp  = NULL;     /* mrmp path (loose arg, or the bundle path itself) */
  const char *model_name;           /* for error messages */

  /* --framework violation: the second argument is the .mrmp itself, not a
   * bundle. Nothing is trained, so there is no model file in between. */
  if (fw_violation) {
    if (argc - i != 2) return predict_usage(stderr);
    const char *art = argv[i];
    if (!ms_mrmp_is_artifact(art))
      pdie("--framework violation needs the MRMPIDX1 artifact (.mrmp); an "
           "exported .cm has no binstrings to read the rule from", art);
    viomodel_t *vm = ms_viomodel_from_mrmp(art, vio_top, vio_threshold,
                                           vio_weight, vio_min_patterns);
    /* The matrix builders take a runtime .cm, and ms_mrmp_resolve() would give
     * only the chain's FIRST block -- the rule is pooled across every set, so
     * materialise the whole chain, exactly as the transcribe step used to. */
    /* Honors $TMPDIR as upscale's section_to_tmp does: the materialized chain
     * mask is genome-wide (tens of MB) and the cluster's shared /tmp is small. */
    const char *tdir = getenv("TMPDIR");
    if (!tdir || !*tdir) tdir = "/tmp";
    char ctmp[4096];
    if (snprintf(ctmp, sizeof(ctmp), "%s/methscope_viomask_XXXXXX.cm", tdir)
        >= (int)sizeof(ctmp)) pdie("TMPDIR path too long", tdir);
    int cfd = mkstemps(ctmp, 3);
    if (cfd < 0) pdie("cannot create temp mask file", NULL);
    close(cfd);
    ms_mrmp_write_mask(art, ctmp, "Pna", (uint32_t)vm->n_feat);
    int rc = predict_violation(query_cg, ctmp, data_path, threads, vm,
                               out_path, with_probs, no_header);
    unlink(ctmp);
    { char idx[4104]; snprintf(idx, sizeof idx, "%s.idx", ctmp); unlink(idx); }
    return rc;
  }

  {
    /* bundle form, now the only form: model.clfx query.cg (model + mrmp in one
     * file, so the artifact scoring featurizes against is never in doubt). */
    model_name = model_arg;
    if (!ms_bundle_is(model_name))
      pdie("expected a .clfx bundle", model_name);
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
      /* --probs has no meaning here yet and was being accepted silently: a tree
       * has no single softmax to report, since each node emits probabilities
       * over its OWN class subset and a routed cell is scored by several. Say
       * so rather than emit a header the caller reads as "no classes". */
      if (with_probs)
        pdie("--probs is not implemented on the routing-tree path; each node "
             "scores a different class subset, so there is no one distribution "
             "to report", model_name);
      int rc = predict_tree(query_cg, model_name, data_path, threads,
                            out_path, no_header, with_levels, lvl_col);
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
    /* The flat single-booster format is gone. It was one booster over one MRMP
     * set, and everything it did the routing tree does with a 1-node tree --
     * except that the flat scoring path never learned to featurize a .cg
     * against the model's own artifact. It called ms_matrix_build_threaded(),
     * which hardcoded continuous coding and RE-SELECTED the top-N patterns by
     * rank, so a binarised model had to refuse the .cg route and demand a
     * prebuilt .msfm. Two featurizers implementing one rule is the failure this
     * codebase keeps hitting; deleting the second is cheaper than teaching it
     * what predict_tree() already knows. */
    if (strcmp(kind, "xgboost") == 0)
      pdie("this is a flat single-booster model — a format methscope no longer "
           "scores. Retrain with `classify-train --data <msfm>`, which builds a "
           "routing-tree bundle (a single-node tree if nothing splits)",
           model_name);
    pdie("unknown model framework 'kind' "
         "(expected tree/violation/threshold/logistic)", kind);
  }
  return 1;   /* not reached */
}
