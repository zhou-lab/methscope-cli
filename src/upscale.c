// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * `upscale`: pure-C inference of a methscope "upscale" block decoder, the
 * MLP that maps an MRMP embedding to CpG-level methylation within a 10k-CpG
 * block. Reimplements (no torch, no BLAS -- only libm) the forward pass of the
 * PyTorch BottleneckDecoder + its sklearn impute/standardize preprocessing,
 * exactly as the reference evaluate_model()/BinaryDecoderDataset do:
 *
 *   x   = (impute_mean(x) - scaler_mean) / scaler_scale          (per feature)
 *   h   = relu(W1 x + b1)                                        (n_in  -> n_hidden)
 *   bn  = bn_gamma * (h - bn_mean) / sqrt(bn_var + eps) + bn_beta
 *   y   = sigmoid(W2 bn + b2)                                    (n_hidden -> n_out)
 *   call = y > 0.5
 *
 * Weights come from a legacy portable .updec file produced by the
 * tools/export_upscale_model.py converter. New training uses the unified,
 * membership-first whole-genome UPDEC2 architecture.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <unistd.h>
#include "methscope.h"
#include "bundle.h"
#include "updec.h"
#include "updec2.h"
#include "cfile.h"     /* YAME: cdata_t, cdata_write1, open_cfile, read_cdata1, FMT6_/FMT0_ macros */

static void udie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] %s\n", msg);
  exit(1);
}

/* ------------------------------------------------------------------ */
/* Feature TSV: one row per sample, n_in numeric columns.              */
/* Optional header (auto-detected: first row has a non-numeric field). */
/* "", "NA", "nan" -> NaN. Returns malloc'd row, or 0 fields at EOF.    */
/* ------------------------------------------------------------------ */
/* Split on TAB preserving empty fields (NaN values are written as ""); a plain
 * strtok would collapse adjacent tabs and silently drop missing features. */
static int parse_row(char *line, double *vals, int n_in) {
  int c = 0;
  char *p = line;
  while (c < n_in) {
    char *start = p;
    while (*p && *p != '\t' && *p != '\n' && *p != '\r') p++;
    char term = *p;
    *p = '\0';
    if (start[0] == '\0' || strcmp(start, "NA") == 0 ||
        strcmp(start, "nan") == 0 || strcmp(start, "NaN") == 0)
      vals[c] = NAN;
    else vals[c] = strtod(start, NULL);
    c++;
    if (term == '\0' || term == '\n' || term == '\r') break;
    p++;                              /* consume the tab, continue to next field */
  }
  return c;
}

/* Does a line look like a header (any field non-numeric and not NA)? */
static int looks_like_header(const char *line) {
  char *dup = strdup(line);
  char *save = NULL;
  int header = 0;
  for (char *t = strtok_r(dup, "\t", &save); t; t = strtok_r(NULL, "\t", &save)) {
    size_t L = strlen(t);
    while (L && (t[L-1] == '\r' || t[L-1] == '\n')) t[--L] = '\0';
    if (L == 0 || strcmp(t, "NA") == 0 || strcmp(t, "nan") == 0) continue;
    char *end = NULL;
    strtod(t, &end);
    if (end == t || *end != '\0') { header = 1; break; }   /* not a pure number */
  }
  free(dup);
  return header;
}

static int upscale_usage(void) {
  fprintf(stderr,
    "\n"
    "Usage:\n"
    "  methscope upscale [options] <model.updec|.updecx> [input]\n"
    "\n"
    "Purpose:\n"
    "  Impute CpG methylation from a sparse methylome. Unified UPDEC2 models\n"
    "  predict the whole genome; legacy UPDEC1 block models remain readable.\n"
    "  Both inference paths are pure C and require neither CUDA nor BLAS.\n"
    "\n"
    "Arguments:\n"
    "  <model>   A bare decoder (.updec/.updec2) OR a bundle with the MRMP attached\n"
    "            (.updecx, from `bundle`). The form selects the input below.\n"
    "  [input]   For .updec : a feature TSV (one row per sample, n_in numeric cols).\n"
    "            For .updecx: a query .cg, featurized internally against the bundled\n"
    "            MRMP. '-' or omitted reads from stdin; for .updec a header row is\n"
    "            auto-detected and empty/NA fields are imputed with the model means.\n"
    "\n"
    "Options:\n"
    "  -o <out>   Write output to a file instead of stdout (.cg is binary -- use -o).\n"
    "  --probs    Emit per-CpG probabilities as TSV instead of a .cg of 0/1 calls.\n"
    "  -h         Show this help message.\n"
    "\n"
    "Output:\n"
    "  A YAME .cg (format 6), one record per input sample. UPDEC2 output is always\n"
    "  in whole-genome CpG order. For legacy UPDEC1, if the bundle carries an\n"
    "  outcpg.cm (the imputed-CpG locations), the .cg spans the whole genome with the\n"
    "  block's CpGs called (no NA) and the rest NA; otherwise a dense block of n_out\n"
    "  CpGs. With --probs, a TSV of per-CpG probabilities (one row per sample).\n"
    "\n");
  return 1;
}

/* Output sink: either a binary .cg stream (default; 0/1 calls as format 6) or a
 * TSV of per-CpG probabilities (--probs). For .cg, `mask` (when non-NULL) is an
 * uncompressed format-0 mask over the whole genome whose set bits are this
 * model's output CpGs, in order; the n_out calls land there and everything else
 * is NA. Without `mask`, a dense block .cg of n_out CpGs (all in-universe). */
typedef struct {
  int      as_cg;      /* 1: write .cg (format 6); 0: write TSV probs */
  BGZF    *cg;         /* when as_cg */
  FILE    *tsv;        /* when !as_cg */
  cdata_t *mask;       /* outcpg (decompressed fmt0); NULL => dense block .cg */
} usink_t;

static void write_pred_cg(BGZF *fp, const ms_updec_t *m, const double *outv, cdata_t *mask) {
  cdata_t c = {0};
  c.fmt = '6'; c.unit = 2; c.compressed = 1;   /* fmt6: on-disk == in-memory bytes */
  if (mask) {
    c.n = mask->n;                              /* whole-genome dimension */
    c.s = calloc((c.n + 3) >> 2, 1);            /* all 00 = NA */
    if (!c.s) udie("out of memory (.cg)", NULL);
    uint64_t k = 0;
    for (uint64_t pos = 0; pos < mask->n && k < (uint64_t)m->n_out; ++pos)
      if (FMT0_IN_SET(*mask, pos)) {            /* k-th output CpG */
        if (outv[k] > 0.5) FMT6_SET1(c, pos); else FMT6_SET0(c, pos);
        ++k;
      }
  } else {
    c.n = m->n_out;                             /* dense block */
    c.s = calloc((c.n + 3) >> 2, 1);
    if (!c.s) udie("out of memory (.cg)", NULL);
    for (int j = 0; j < m->n_out; ++j) {
      if (outv[j] > 0.5) FMT6_SET1(c, j); else FMT6_SET0(c, j);
    }
  }
  cdata_write1(fp, &c);
  free(c.s);
}

static void sink_emit(usink_t *sk, const ms_updec_t *m, const double *outv) {
  if (sk->as_cg) { write_pred_cg(sk->cg, m, outv, sk->mask); return; }
  for (int k = 0; k < m->n_out; ++k) {          /* --probs TSV */
    if (k) fputc('\t', sk->tsv);
    fprintf(sk->tsv, "%.6g", outv[k]);
  }
  fputc('\n', sk->tsv);
}

/* Mode A: feature TSV in (bare .updec). */
static long run_from_tsv(const ms_updec_t *m, const char *feat_path, usink_t *sk) {
  FILE *in = (strcmp(feat_path, "-") == 0) ? stdin : fopen(feat_path, "r");
  if (!in) udie("cannot open features", feat_path);
  double *feat = malloc((size_t)m->n_in * sizeof(double));
  double *xs   = malloc((size_t)m->n_in * sizeof(double));
  double *hbuf = malloc((size_t)m->n_hidden * sizeof(double));
  double *outv = malloc((size_t)m->n_out * sizeof(double));
  if (!feat || !xs || !hbuf || !outv) udie("out of memory (buffers)", NULL);
  char *line = NULL; size_t cap = 0; ssize_t len; long row = 0, checked = 0;
  while ((len = getline(&line, &cap, in)) != -1) {
    if (len == 0 || line[0] == '\n' || line[0] == '\r') continue;
    if (!checked) { checked = 1; if (looks_like_header(line)) continue; }
    char *copy = strdup(line);
    int nc = parse_row(copy, feat, m->n_in);
    free(copy);
    if (nc == 0) continue;
    if (nc < m->n_in) udie("feature row has fewer columns than the model expects", feat_path);
    ms_updec_standardize(m, feat, xs);
    ms_updec_forward_eval(m, xs, outv, hbuf);
    sink_emit(sk, m, outv);
    row++;
  }
  free(line);
  if (in != stdin) fclose(in);
  free(feat); free(xs); free(hbuf); free(outv);
  return row;
}

/* Mode B: query .cg in, featurized against an MRMP (.updecx bundle).
 * The MRMP's numeric pattern order (P1..Pn, Pna last) IS the decoder's
 * feat_1..feat_n_in order, so the first n_in matrix columns map directly. */
static long run_from_cg(const ms_updec_t *m, const char *mrmp_path,
                        const char *query_cg, usink_t *sk) {
  ms_matrix_t *mx = ms_matrix_build(query_cg, mrmp_path);
  if (mx->n_patterns < m->n_in)
    udie("bundled MRMP produced fewer patterns than the model expects", mrmp_path);
  double *feat = malloc((size_t)m->n_in * sizeof(double));
  double *xs   = malloc((size_t)m->n_in * sizeof(double));
  double *hbuf = malloc((size_t)m->n_hidden * sizeof(double));
  double *outv = malloc((size_t)m->n_out * sizeof(double));
  if (!feat || !xs || !hbuf || !outv) udie("out of memory (buffers)", NULL);
  for (int r = 0; r < mx->n_cells; ++r) {
    const double *src = mx->M + (size_t)r * mx->n_patterns;
    for (int j = 0; j < m->n_in; ++j) feat[j] = src[j];   /* NaN stays; imputed in standardize */
    ms_updec_standardize(m, feat, xs);
    ms_updec_forward_eval(m, xs, outv, hbuf);
    sink_emit(sk, m, outv);
  }
  long row = mx->n_cells;
  free(feat); free(xs); free(hbuf); free(outv);
  ms_matrix_free(mx);
  return row;
}

/* extract a bundle section to a temp file; returns the path (caller unlink+free) */
static char *section_to_tmp(const void *buf, size_t len, const char *tag) {
  char tmpl[64]; snprintf(tmpl, sizeof(tmpl), "/tmp/methscope_%s_XXXXXX", tag);
  int fd = mkstemp(tmpl);
  if (fd < 0) udie("cannot create temp file", tag);
  for (size_t off = 0; off < len; ) {
    ssize_t w = write(fd, (const char *)buf + off, len - off);
    if (w <= 0) udie("error writing temp file", tag);
    off += (size_t)w;
  }
  close(fd);
  return strdup(tmpl);
}

static void sink_emit_updec2(usink_t *sk, const ms_updec2_t *m, const float *prob) {
  uint64_t n = m->header->n_cpg;
  if (!sk->as_cg) {
    for (uint64_t j = 0; j < n; ++j) {
      if (j) fputc('\t', sk->tsv);
      fprintf(sk->tsv, "%.6g", prob[j]);
    }
    fputc('\n', sk->tsv);
    return;
  }
  cdata_t c = {0};
  c.fmt = '6'; c.unit = 2; c.compressed = 1; c.n = n;
  c.s = calloc((n + 3) >> 2, 1);
  if (!c.s) udie("out of memory writing UPDEC2 .cg", NULL);
  for (uint64_t j = 0; j < n; ++j) {
    if (prob[j] > 0.5f) FMT6_SET1(c, j); else FMT6_SET0(c, j);
  }
  cdata_write1(sk->cg, &c);
  free(c.s);
}

static long run_updec2(const char *path, uint64_t offset, uint64_t length,
                       int bundled, const char *input_path, usink_t *sk,
                       uint64_t *n_cpg_out) {
  ms_updec2_t m;
  char err[256];
  if (!ms_updec2_open(&m, path, offset, length, err, sizeof(err)))
    udie(err, path);
  const uint32_t P = m.header->patterns;
  float *x = malloc((size_t)m.header->input_dim * sizeof(*x));
  float *prob = malloc((size_t)m.header->n_cpg * sizeof(*prob));
  uint32_t max_rank = 1;
  for (uint32_t u = 0; u < m.header->n_units; ++u)
    if (m.units[u].bottleneck_dim > max_rank) max_rank = m.units[u].bottleneck_dim;
  size_t work_n = max_rank + (size_t)2 * m.trunk_dim;
  float *z = malloc(work_n * sizeof(*z));
  double *beta = malloc((size_t)2 * P * sizeof(*beta));
  int *count = malloc((size_t)P * sizeof(*count));
  if (!x || !prob || !z || !beta || !count)
    udie("out of memory for UPDEC2 inference", NULL);

  long rows = 0;
  if (bundled) {
    ms_matrix_t *mx = ms_matrix_build(input_path, path);
    if (mx->n_patterns < (int)P)
      udie("bundled MRMP produced fewer patterns than UPDEC2 expects", path);
    for (int r = 0; r < mx->n_cells; ++r) {
      const double *src = mx->M + (size_t)r * mx->n_patterns;
      memcpy(beta, src, (size_t)P * sizeof(*beta));
      if (mx->N) memcpy(count, mx->N + (size_t)r * mx->n_patterns,
                        (size_t)P * sizeof(*count));
      else for (uint32_t p = 0; p < P; ++p) count[p] = isnan(beta[p]) ? 0 : 1;
      ms_updec2_prepare_input(&m, beta, count, x);
      if (!ms_updec2_forward(&m, x, prob, z, work_n))
        udie("UPDEC2 bottleneck buffer is too small", path);
      sink_emit_updec2(sk, &m, prob);
      ++rows;
    }
    ms_matrix_free(mx);
  } else {
    FILE *in = !strcmp(input_path, "-") ? stdin : fopen(input_path, "r");
    if (!in) udie("cannot open features", input_path);
    char *line = NULL; size_t cap = 0; ssize_t len; int checked = 0;
    while ((len = getline(&line, &cap, in)) != -1) {
      if (!len || line[0] == '\n' || line[0] == '\r') continue;
      if (!checked) { checked = 1; if (looks_like_header(line)) continue; }
      char *copy = strdup(line);
      int need_count = m.header->version >= 3 &&
                       (m.header->flags & MS_UPDEC2_FLAG_COUNT);
      int nc = parse_row(copy, beta, (int)(need_count ? 2 * P : P));
      free(copy);
      if (nc < (int)P) udie("feature row has fewer columns than UPDEC2 expects", input_path);
      for (uint32_t p = 0; p < P; ++p) {
        if (need_count) {
          double n = beta[P + p];
          if (!isfinite(n) || n < 0.0 || n > INT_MAX || floor(n) != n)
            udie("UPDEC2 count columns must be nonnegative integers", input_path);
          count[p] = (int)n;
        } else count[p] = isnan(beta[p]) ? 0 : 1;
      }
      if (need_count && nc < (int)(2 * P))
        udie("count model requires P beta columns followed by P count columns",
             input_path);
      ms_updec2_prepare_input(&m, beta, count, x);
      if (!ms_updec2_forward(&m, x, prob, z, work_n))
        udie("UPDEC2 bottleneck buffer is too small", path);
      sink_emit_updec2(sk, &m, prob);
      ++rows;
    }
    free(line);
    if (in != stdin) fclose(in);
  }
  *n_cpg_out = m.header->n_cpg;
  free(count); free(beta); free(z); free(prob); free(x);
  ms_updec2_close(&m);
  return rows;
}

static int read_magic_at(const char *path, uint64_t offset, char magic[8]) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return 0;
  int ok = !fseeko(fp, (off_t)offset, SEEK_SET) && fread(magic, 1, 8, fp) == 8;
  fclose(fp);
  return ok;
}

int main_upscale(int argc, char *argv[]) {
  const char *out_path = NULL;
  int with_probs = 0;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--probs") == 0) with_probs = 1;
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      upscale_usage();
      return 0;
    }
    else break;
  }
  if (argc - i < 1 || argc - i > 2) return upscale_usage();
  const char *model_path = argv[i];
  const char *input_path = (argc - i == 2) ? argv[i + 1] : "-";
  int bundled = ms_bundle_is(model_path);
  ms_bundle_entry_t model_entry = {0};
  uint64_t model_offset = 0, model_length = 0;
  if (bundled) {
    if (!ms_bundle_find(model_path, "model", &model_entry))
      udie("model section not found in bundle", model_path);
    model_offset = model_entry.offset; model_length = model_entry.length;
  }
  char inner_magic[8];
  if (!read_magic_at(model_path, model_offset, inner_magic))
    udie("cannot read model magic", model_path);
  int is_updec2 = !memcmp(inner_magic, MS_UPDEC2_MAGIC, 8);

  /* ---- output sink: .cg by default, TSV of probabilities with --probs ---- */
  usink_t sk = {0};
  sk.as_cg = !with_probs;
  if (sk.as_cg) {
    sk.cg = out_path ? bgzf_open(out_path, "w") : bgzf_dopen(fileno(stdout), "w");
    if (!sk.cg) udie("cannot open .cg output", out_path);
  } else {
    sk.tsv = out_path ? fopen(out_path, "w") : stdout;
    if (!sk.tsv) udie("cannot open output", out_path);
  }

  cdata_t mask = {0};
  long row; int n_out;

  if (is_updec2) {
    uint64_t n_cpg2 = 0;
    row = run_updec2(model_path, model_offset, model_length, bundled,
                     input_path, &sk, &n_cpg2);
    if (n_cpg2 > INT32_MAX) udie("UPDEC2 CpG count exceeds reporting range", model_path);
    n_out = (int)n_cpg2;
  } else if (bundled) {
    /* .updecx: model + MRMP (+ optional outcpg) bundled; input is a query .cg. */
    size_t mlen, olen;
    void *mbuf = ms_bundle_section(model_path, "model", &mlen);
    ms_updec_t *m = ms_updec_load_buf(mbuf, mlen);
    free(mbuf);

    /* optional output-CpG mask -> genome-dimension .cg */
    void *obuf = ms_bundle_section_opt(model_path, "outcpg", &olen);
    if (obuf && sk.as_cg) {
      char *otmp = section_to_tmp(obuf, olen, "outcpg"); free(obuf);
      cfile_t cf = open_cfile(otmp);
      cdata_t raw = read_cdata1(&cf);
      bgzf_close(cf.fh);
      unlink(otmp); free(otmp);
      if (raw.n == 0) udie("empty outcpg in bundle", model_path);
      mask = decompress(raw); free_cdata(&raw);
      sk.mask = &mask;
    } else { free(obuf); }

    n_out = m->n_out;
    row = run_from_cg(m, model_path, input_path, &sk);   /* front bytes are the mrmp .cm */
    ms_updec_free(m);
  } else {
    ms_updec_t *m = ms_updec_load(model_path);
    n_out = m->n_out;
    row = run_from_tsv(m, input_path, &sk);
    ms_updec_free(m);
  }

  if (sk.as_cg) bgzf_close(sk.cg);
  else if (sk.tsv != stdout) fclose(sk.tsv);
  if (sk.mask) free_cdata(&mask);
  fprintf(stderr, "[methscope] upscaled %ld sample(s) x %d CpGs%s\n",
          row, sk.mask ? (int)mask.n : n_out, sk.mask ? " (genome .cg)" : "");
  return 0;
}
