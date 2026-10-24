// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
/**
 * The random-forest framework: `classify-train --framework rf` and the
 * scoring of the "rf" bundles it writes.
 *
 * Features are single CpGs (array probes), not MRMP patterns, chosen the way
 * the lab's R array classifiers chose them (pediatric thyroid, CCR 2026):
 *   1. rows with more than --max-na of the training records missing are
 *      dropped; the rest are filled with the row's training mean;
 *   2. a forest per batch of --batch rows ranks every row by
 *      MeanDecreaseAccuracy; the top --select over all batches are kept;
 *   3. the final forest is fit on those rows.
 * Everything is fit on the records given, so a cross-validation that calls
 * this once per fold does its selection inside the fold.
 *
 * The bundle's MRMP is a one-row-per-state .cm: state P<k> is the k-th
 * selected row, every other row is Pna. `classify` featurizes the query
 * against it like any MRMP -- a one-row pattern's mean beta is that row's
 * beta, read continuously, never cut or drawn -- so the feature map travels
 * with the model and `mliftover` re-indexes it like any other.
 *
 * The model section ("MSRFMOD1"): nclass, the class labels (class-index
 * order, NUL-terminated), K, the K training means used to fill a missing
 * feature at scoring, a NUL-terminated recipe line for `inspect`, then the
 * trees (rf_serialize).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif
#include "methscope.h"
#include "bundle.h"
#include "rforest.h"
#include "cfile.h"
#include "index.h"

#define RFMOD_MAGIC "MSRFMOD1"

static void rdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] classify-train: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] classify-train: %s\n", msg);
  exit(1);
}
static void sdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] classify: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] classify: %s\n", msg);
  exit(1);
}

static int ncpu(void) {
#ifdef __linux__
  cpu_set_t cs;
  if (!sched_getaffinity(0, sizeof cs, &cs)) return CPU_COUNT(&cs);
#endif
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
}

static char **read_lines(const char *path, int *n_out) {
  FILE *fp = fopen(path, "r");
  if (!fp) rdie("cannot open labels file", path);
  int cap = 256, n = 0;
  char **v = malloc(cap * sizeof(char *));
  char *line = NULL; size_t lcap = 0; ssize_t len;
  if (!v) rdie("out of memory", NULL);
  while ((len = getline(&line, &lcap, fp)) != -1) {
    while (len && (line[len-1] == '\n' || line[len-1] == '\r' ||
                   line[len-1] == ' ' || line[len-1] == '\t'))
      line[--len] = '\0';
    char *s = line; while (*s == ' ' || *s == '\t') s++;
    if (!*s) continue;
    if (n == cap) {
      v = realloc(v, (cap *= 2) * sizeof(char *));
      if (!v) rdie("out of memory", NULL);
    }
    v[n++] = strdup(s);
  }
  free(line); fclose(fp);
  *n_out = n;
  return v;
}

static int cmp_str(const void *a, const void *b) {
  return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Every record's beta at every row, records x rows, NaN where missing.
 * Format 3 gives M / (M + U), NaN at M + U = 0; format 4 its value, NaN
 * where negative (YAME's NA). */
static float *read_betas(const char *cg, uint32_t *n_rec, uint64_t *n_row) {
  cfile_t cf = open_cfile((char *)cg);
  float *X = NULL;
  size_t cap = 0;
  uint32_t n = 0;
  uint64_t P = 0;
  for (;;) {
    cdata_t c = read_cdata1(&cf);
    if (c.n == 0) break;
    decompress_in_situ(&c);
    if (c.fmt != '3' && c.fmt != '4')
      rdie("the rf framework reads format 3 (M/U) or 4 (beta) records", cg);
    if (!n) P = c.n;
    else if (c.n != P) rdie("records differ in row count", cg);
    if ((size_t)(n + 1) * P > cap) {
      cap = cap ? 2 * cap : 16 * P;
      X = realloc(X, cap * sizeof(float));
      if (!X) rdie("out of memory (betas)", NULL);
    }
    float *x = X + (size_t)n * P;
    if (c.fmt == '3')
      for (uint64_t i = 0; i < P; ++i) {
        uint64_t mu = f3_get_mu(&c, i);
        x[i] = mu ? (float)MU2beta(mu) : NAN;
      }
    else
      for (uint64_t i = 0; i < P; ++i) {
        float v = ((float *)c.s)[i];
        x[i] = v < 0 ? NAN : v;
      }
    free_cdata(&c);
    ++n;
  }
  bgzf_close(cf.fh);
  if (!n) rdie("no records in the query", cg);
  *n_rec = n; *n_row = P;
  return X;
}

/* The bundle's MRMP: row sel[k] is state P<k+1>, every other row Pna. */
static void write_probe_cm(const char *out, uint64_t P, const uint64_t *sel,
                           uint32_t K) {
  size_t keys_nb = 4;                                   /* "Pna\0" */
  for (uint32_t k = 0; k < K; ++k) {
    char b[16]; keys_nb += (size_t)snprintf(b, sizeof b, "P%u", k + 1) + 1;
  }
  uint8_t unit = (K + 1 <= 0xff) ? 1 : (K + 1 <= 0xffff) ? 2 : 4;
  size_t nbytes = keys_nb + 1 + (size_t)P * unit;
  uint8_t *buf = calloc(nbytes, 1);
  if (!buf) rdie("out of memory (feature map)", NULL);
  size_t off = 0;
  for (uint32_t k = 0; k < K; ++k)
    off += (size_t)sprintf((char *)buf + off, "P%u", k + 1) + 1;
  memcpy(buf + off, "Pna", 4);
  buf[keys_nb] = '\0';                                  /* double NUL ends keys */
  uint8_t *data = buf + keys_nb + 1;
  const uint64_t pna = K;
  for (uint64_t i = 0; i < P; ++i)
    for (uint8_t j = 0; j < unit; ++j) data[i * unit + j] = (pna >> (8 * j)) & 0xff;
  for (uint32_t k = 0; k < K; ++k)
    for (uint8_t j = 0; j < unit; ++j)
      data[sel[k] * unit + j] = ((uint64_t)k >> (8 * j)) & 0xff;
  cdata_t c = {0};
  c.s = buf; c.n = P; c.unit = unit; c.compressed = 0; c.fmt = '2';
  cdata_write((char *)out, &c, "w", 0);
  free_cdata(&c);
}

static char *tmp_path(const char *tag, const char *ext) {
  char t[4096];
  const char *td = getenv("TMPDIR");
  snprintf(t, sizeof t, "%s/methscope_%s_XXXXXX%s", td && *td ? td : "/tmp",
           tag, ext);
  int fd = mkstemps(t, (int)strlen(ext));
  if (fd < 0) rdie("cannot create a temporary file", t);
  close(fd);
  return strdup(t);
}

typedef struct { double v; uint64_t i; } rank_t;
static int cmp_rank(const void *a, const void *b) {
  const rank_t *x = a, *y = b;
  if (x->v != y->v) return x->v > y->v ? -1 : 1;     /* importance, high first */
  return (x->i > y->i) - (x->i < y->i);              /* then row order */
}

static int rf_usage(FILE *out) {
  ms_help(out,
    "Usage:\n"
    "  methscope classify-train --framework rf -l <labels.txt> -o <out.clfx>\n"
    "                           [options] <query.cg>\n"
    "\n"
    "Purpose:\n"
    "  Train a random forest on single CpGs (array probes), selecting them as\n"
    "  the lab's R array classifiers do: rows over --max-na missing dropped,\n"
    "  the rest filled with the row mean; a forest per --batch rows ranks every\n"
    "  row by MeanDecreaseAccuracy; the top --select are kept; the final forest\n"
    "  is fit on them. All of it uses only the records given, so a cross-\n"
    "  validation that trains once per fold selects inside each fold.\n"
    "\n"
    "Arguments:\n"
    "  <query.cg>   Training records, format 3 (M/U) or 4 (beta), one per\n"
    "               sample, all in one row space (e.g. an EPICv2 store).\n"
    "\n"
    "Options:\n"
    "  -l <labels.txt>  One label per record, in record order (required).\n"
    "  -o <out.clfx>    Output bundle (required): the forest plus a one-row-\n"
    "                   per-state .cm naming the selected rows, so `classify`\n"
    "                   and `mliftover` take it like any other model.\n"
    "  --select <K>     Rows kept for the final forest (default 3000).\n"
    "  --batch <B>      Rows per ranking forest (default 10000).\n"
    "  --ntree <T>      Trees per forest (default 500).\n"
    "  --mtry <m>       Rows tried per split (default floor(sqrt(width))).\n"
    "  --max-na <f>     Drop rows missing in more than this fraction of\n"
    "                   records (default 0.05).\n"
    "  --seed <s>       Seed (default 1). Results do not depend on --threads.\n"
    "  --threads <N>    Threads (default: the CPUs this process may use).\n"
    "  -h               Show this help message.\n"
    "\n"
    "Score with `methscope classify model.clfx query.cg`; confidence is the\n"
    "fraction of trees voting for the call.\n");
  return out == stdout ? 0 : 1;
}

int main_train_rf(int argc, char *argv[]) {
  const char *labels = NULL, *out = NULL, *query = NULL;
  uint32_t K = 3000, B = 10000, ntree = 500, mtry = 0;
  double max_na = 0.05;
  uint64_t seed = 1;
  int threads = 0;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-l") && i + 1 < argc) labels = argv[++i];
    else if (!strcmp(a, "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(a, "--framework") && i + 1 < argc) ++i;
    else if (!strcmp(a, "--select") && i + 1 < argc) K = (uint32_t)atol(argv[++i]);
    else if (!strcmp(a, "--batch") && i + 1 < argc) B = (uint32_t)atol(argv[++i]);
    else if (!strcmp(a, "--ntree") && i + 1 < argc) ntree = (uint32_t)atol(argv[++i]);
    else if (!strcmp(a, "--mtry") && i + 1 < argc) mtry = (uint32_t)atol(argv[++i]);
    else if (!strcmp(a, "--max-na") && i + 1 < argc) max_na = atof(argv[++i]);
    else if (!strcmp(a, "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) return rf_usage(stdout);
    else if (a[0] == '-') rdie("unrecognized or incomplete option", a);
    else if (!query) query = a;
    else rdie("unexpected argument (rf takes one <query.cg>)", a);
  }
  if (!labels || !out || !query) return rf_usage(stderr);
  if (!ms_path_is_bundle_ext(out)) rdie("the rf framework writes a .clfx", out);
  if (!K || !B || !ntree) rdie("--select, --batch and --ntree must be positive", NULL);
  if (!(max_na >= 0 && max_na <= 1)) rdie("--max-na is a fraction, 0-1", NULL);
  if (threads <= 0) threads = ncpu();

  /* ---- records and labels ---- */
  uint32_t n; uint64_t P;
  float *X = read_betas(query, &n, &P);
  int nl;
  char **lab = read_lines(labels, &nl);
  if ((uint32_t)nl != n) rdie("label count does not match the record count", labels);
  char **cls = malloc(n * sizeof(char *));
  if (!cls) rdie("out of memory", NULL);
  memcpy(cls, lab, n * sizeof(char *));
  qsort(cls, n, sizeof(char *), cmp_str);
  uint32_t C = 0;
  for (uint32_t i = 0; i < n; ++i)
    if (!C || strcmp(cls[i], cls[C - 1])) cls[C++] = cls[i];
  if (C < 2) rdie("need at least two classes", labels);
  int *y = malloc(n * sizeof(int));
  if (!y) rdie("out of memory", NULL);
  for (uint32_t i = 0; i < n; ++i) {
    char **hit = bsearch(&lab[i], cls, C, sizeof(char *), cmp_str);
    y[i] = (int)(hit - cls);
  }
  fprintf(stderr, "\n[methscope] classify-train --framework rf\n\n");
  fprintf(stderr, "  %-12s %u records x %" PRIu64 " rows, %u classes\n",
          "input", n, P, C);

  /* ---- 1. missingness filter, mean fill, compacted in place ----
   * Record r's kept values move from r*P + j to r*nk + c, never forward of
   * where they are read, so one pass compacts without a second matrix. */
  uint64_t *row = malloc(P * sizeof(uint64_t));
  float *mean = malloc(P * sizeof(float));
  if (!row || !mean) rdie("out of memory", NULL);
  uint64_t nk = 0;
  for (uint64_t j = 0; j < P; ++j) {
    uint32_t na = 0; double s = 0;
    for (uint32_t r = 0; r < n; ++r) {
      float v = X[(size_t)r * P + j];
      if (isnan(v)) ++na; else s += v;
    }
    if (na > max_na * n || na == n) continue;
    row[nk] = j; mean[nk] = (float)(s / (n - na)); ++nk;
  }
  if (!nk) rdie("every row is missing in too many records (--max-na)", NULL);
  for (uint32_t r = 0; r < n; ++r)
    for (uint64_t c = 0; c < nk; ++c) {
      float v = X[(size_t)r * P + row[c]];
      X[(size_t)r * nk + c] = isnan(v) ? mean[c] : v;
    }
  fprintf(stderr, "  %-12s %" PRIu64 " rows kept (<= %g missing), the rest "
          "mean-filled\n", "filter", nk, max_na);

  /* ---- 2. rank every kept row by MDA, batch by batch ---- */
  rank_t *rk = malloc(nk * sizeof(rank_t));
  float *xb = malloc((size_t)n * (B < nk ? B : nk) * sizeof(float));
  double *mda = malloc((B < nk ? B : nk) * sizeof(double));
  if (!rk || !xb || !mda) rdie("out of memory", NULL);
  uint64_t nb = (nk + B - 1) / B;
  for (uint64_t b = 0; b < nb; ++b) {
    uint64_t c0 = b * B, w = (c0 + B < nk) ? B : nk - c0;
    for (uint32_t r = 0; r < n; ++r)
      memcpy(xb + (size_t)r * w, X + (size_t)r * nk + c0, w * sizeof(float));
    rf_param_t par = { ntree, mtry, seed + b, threads };
    rf_forest_t *f = rf_fit(xb, n, (uint32_t)w, y, C, &par, mda, NULL);
    rf_free(f);
    for (uint64_t c = 0; c < w; ++c) { rk[c0 + c].v = mda[c]; rk[c0 + c].i = c0 + c; }
    if ((b + 1) % 10 == 0 || b + 1 == nb)
      fprintf(stderr, "  %-12s %" PRIu64 " of %" PRIu64 " batches ranked\n",
              "rank", b + 1, nb);
  }
  free(xb); free(mda);
  qsort(rk, nk, sizeof(rank_t), cmp_rank);
  if (K > nk) K = (uint32_t)nk;

  /* ---- 3. the final forest on the top K ---- */
  float *xs = malloc((size_t)n * K * sizeof(float));
  float *ms = malloc(K * sizeof(float));
  uint64_t *sel = malloc(K * sizeof(uint64_t));
  if (!xs || !ms || !sel) rdie("out of memory", NULL);
  for (uint32_t k = 0; k < K; ++k) {
    uint64_t c = rk[k].i;
    sel[k] = row[c]; ms[k] = mean[c];
    for (uint32_t r = 0; r < n; ++r) xs[(size_t)r * K + k] = X[(size_t)r * nk + c];
  }
  free(X); free(rk);
  rf_param_t par = { ntree, mtry, seed, threads };
  rf_forest_t *f = rf_fit(xs, n, K, y, C, &par, NULL, NULL);
  fprintf(stderr, "  %-12s %u rows, %u trees, OOB error %.4f\n", "forest", K,
          f->ntree, f->oob_err);

  /* ---- the bundle ---- */
  char recipe[256];
  snprintf(recipe, sizeof recipe, "select %u of %" PRIu64 " rows (batch %u, "
           "max-na %g), %u trees, seed %" PRIu64 ", trained on %u records",
           K, nk, B, max_na, ntree, seed, n);
  size_t flen;
  uint8_t *fb = rf_serialize(f, &flen);
  char *mtmp = tmp_path("rf", ".rfm"), *ctmp = tmp_path("rfmap", ".cm");
  FILE *fo = fopen(mtmp, "wb");
  if (!fo) rdie("cannot write the temporary model", mtmp);
  fwrite(RFMOD_MAGIC, 1, 8, fo);
  fwrite(&C, 4, 1, fo);
  for (uint32_t c = 0; c < C; ++c) fwrite(cls[c], 1, strlen(cls[c]) + 1, fo);
  fwrite(&K, 4, 1, fo);
  fwrite(ms, sizeof(float), K, fo);
  fwrite(recipe, 1, strlen(recipe) + 1, fo);
  fwrite(fb, 1, flen, fo);
  if (fclose(fo)) rdie("cannot write the temporary model", mtmp);
  write_probe_cm(ctmp, P, sel, K);
  ms_bundle_pack(out, "rf", mtmp, ctmp, NULL);
  unlink(mtmp); unlink(ctmp);
  { char idx[4200]; snprintf(idx, sizeof idx, "%s.idx", ctmp); unlink(idx); }
  fprintf(stderr, "\n  %-12s %s\n  %-12s methscope classify %s query.cg\n",
          "wrote", out, "score with", out);
  free(mtmp); free(ctmp); free(fb); rf_free(f);
  free(xs); free(ms); free(sel); free(row); free(mean); free(y); free(cls);
  for (int i = 0; i < nl; ++i) free(lab[i]);
  free(lab);
  return 0;
}

/* ------------------------------------------------------------- scoring --- */

typedef struct {
  uint32_t      nclass, K;
  char        **label;
  float        *mean;
  char         *recipe;
  rf_forest_t  *forest;
} rfmod_t;

static rfmod_t *rfmod_parse(const uint8_t *b, size_t len, const char *src) {
  const uint8_t *r = b, *end = b + len;
  if (len < 16 || memcmp(r, RFMOD_MAGIC, 8)) sdie("not a random-forest model", src);
  r += 8;
  rfmod_t *m = calloc(1, sizeof(rfmod_t));
  if (!m) sdie("out of memory", NULL);
  memcpy(&m->nclass, r, 4); r += 4;
  if (m->nclass < 2 || m->nclass > 100000) sdie("malformed rf model", src);
  m->label = calloc(m->nclass, sizeof(char *));
  for (uint32_t c = 0; c < m->nclass; ++c) {
    const uint8_t *z = memchr(r, 0, (size_t)(end - r));
    if (!z) sdie("malformed rf model (labels)", src);
    m->label[c] = strdup((const char *)r); r = z + 1;
  }
  if (end - r < 4) sdie("malformed rf model", src);
  memcpy(&m->K, r, 4); r += 4;
  if ((size_t)(end - r) < 4 * (size_t)m->K) sdie("malformed rf model (means)", src);
  m->mean = malloc(4 * (size_t)m->K + 1);
  memcpy(m->mean, r, 4 * (size_t)m->K); r += 4 * (size_t)m->K;
  const uint8_t *z = memchr(r, 0, (size_t)(end - r));
  if (!z) sdie("malformed rf model (recipe)", src);
  m->recipe = strdup((const char *)r); r = z + 1;
  m->forest = rf_deserialize(r, (size_t)(end - r), NULL);
  if (!m->forest || m->forest->p != m->K || m->forest->nclass != m->nclass)
    sdie("malformed rf model (trees)", src);
  return m;
}

static void rfmod_free(rfmod_t *m) {
  for (uint32_t c = 0; c < m->nclass; ++c) free(m->label[c]);
  free(m->label); free(m->mean); free(m->recipe);
  rf_free(m->forest); free(m);
}

void ms_rf_describe(const void *buf, size_t len, const char *src, FILE *out) {
  rfmod_t *m = rfmod_parse(buf, len, src);
  fprintf(out, "      %-10s random forest\n", "method");
  fprintf(out, "      %-10s ", "labels");
  for (uint32_t c = 0; c < m->nclass; ++c)
    fprintf(out, "%s%s", c ? ", " : "", m->label[c]);
  fprintf(out, "\n      %-10s %u single-row\n", "features", m->K);
  fprintf(out, "      %-10s %u\n", "trees", m->forest->ntree);
  fprintf(out, "      %-10s %s\n", "recipe", m->recipe);
  rfmod_free(m);
}

int ms_classify_rf(const char *query_cg, const char *bundle,
                   const char *out_path, int with_probs, int no_header) {
  size_t len; void *b = ms_bundle_section(bundle, "model", &len);
  rfmod_t *md = rfmod_parse(b, len, bundle);
  free(b);
  char *tmp_cm = NULL;
  const char *cm = ms_mrmp_resolve(bundle, &tmp_cm);
  ms_matrix_t *m = ms_matrix_build(query_cg, cm);
  ms_mrmp_cleanup(tmp_cm);
  /* columns come back sorted by pattern number, Pna last: P1..PK first */
  if (m->n_patterns < (int)md->K) sdie("the bundle's feature map is short", bundle);
  for (uint32_t k = 0; k < md->K; ++k) {
    char nm[16]; snprintf(nm, sizeof nm, "P%u", k + 1);
    if (strcmp(m->pattern_names[k], nm)) sdie("the bundle's feature map is not P1..PK", bundle);
  }
  const uint32_t K = md->K, C = md->nclass;
  float *x = malloc((size_t)m->n_cells * K * sizeof(float));
  double *prob = malloc((size_t)m->n_cells * C * sizeof(double));
  if (!x || !prob) sdie("out of memory", NULL);
  uint64_t nfill = 0;
  for (int r = 0; r < m->n_cells; ++r)
    for (uint32_t k = 0; k < K; ++k) {
      double v = m->M[(size_t)r * m->n_patterns + k];
      if (isnan(v)) { v = md->mean[k]; ++nfill; }
      x[(size_t)r * K + k] = (float)v;
    }
  if (nfill)
    fprintf(stderr, "[methscope] classify: %" PRIu64 " of %" PRIu64 " feature "
            "values missing, filled with the training mean\n", nfill,
            (uint64_t)m->n_cells * K);
  rf_predict(md->forest, x, (uint32_t)m->n_cells, prob);

  FILE *fo = (out_path && strcmp(out_path, "-")) ? fopen(out_path, "w") : stdout;
  if (!fo) sdie("cannot open output", out_path);
  if (!no_header) {
    fputs("cell\tprediction_label\tconfidence\tcertainty", fo);
    if (with_probs) for (uint32_t c = 0; c < C; ++c) fprintf(fo, "\t%s", md->label[c]);
    fputc('\n', fo);
  }
  for (int r = 0; r < m->n_cells; ++r) {
    const double *p = prob + (size_t)r * C;
    uint32_t best = 0;
    double ent = 0;
    for (uint32_t c = 0; c < C; ++c) {
      if (p[c] > p[best]) best = c;            /* a tie goes to the first class */
      if (p[c] > 0) ent -= p[c] * log(p[c]);
    }
    double cert = 1.0 - ent / log((double)C);
    fprintf(fo, "%s\t%s\t%.6f\t%.6f", m->cell_names[r], md->label[best],
            p[best], cert < 0 ? 0 : cert);
    if (with_probs) for (uint32_t c = 0; c < C; ++c) fprintf(fo, "\t%.6f", p[c]);
    fputc('\n', fo);
  }
  if (fo != stdout) fclose(fo);
  free(x); free(prob);
  ms_matrix_free(m); rfmod_free(md);
  return 0;
}
