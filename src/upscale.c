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
 * Weights come from a portable .updec file produced by
 * tools/export_upscale_model.py (the only step that needs torch/sklearn).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include "methscope.h"
#include "bundle.h"
#include "cfile.h"     /* YAME: cdata_t, cdata_write1, open_cfile, read_cdata1, FMT6_/FMT0_ macros */

static void udie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] %s\n", msg);
  exit(1);
}

/* ------------------------------------------------------------------ */
/* .updec model (see tools/export_upscale_model.py for the byte layout) */
/* ------------------------------------------------------------------ */
typedef struct {
  int     n_in, n_hidden, n_out;
  float   bn_eps;
  float  *imp_mean;                 /* n_in  */
  float  *sc_mean, *sc_scale;       /* n_in  */
  float  *W1, *b1;                  /* n_hidden*n_in, n_hidden */
  float  *bn_g, *bn_b, *bn_m, *bn_v;/* n_hidden each */
  float  *W2, *b2;                  /* n_out*n_hidden, n_out */
} updec_t;

static float *read_f32(FILE *fp, size_t n, const char *what) {
  float *a = malloc(n * sizeof(float));
  if (!a) udie("out of memory reading model", what);
  if (fread(a, sizeof(float), n, fp) != n) udie("truncated .updec at", what);
  return a;
}

static updec_t *updec_read(FILE *fp) {
  char magic[8];
  if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "UPDEC1", 6) != 0)
    udie("not a .updec (bad magic)", NULL);
  updec_t *m = calloc(1, sizeof(updec_t));
  int32_t dims[3];
  if (fread(dims, sizeof(int32_t), 3, fp) != 3) udie("truncated .updec header", NULL);
  m->n_in = dims[0]; m->n_hidden = dims[1]; m->n_out = dims[2];
  if (m->n_in <= 0 || m->n_hidden <= 0 || m->n_out <= 0)
    udie("bad dimensions in .updec", NULL);
  if (fread(&m->bn_eps, sizeof(float), 1, fp) != 1) udie("truncated .updec eps", NULL);
  size_t I = m->n_in, H = m->n_hidden, O = m->n_out;
  m->imp_mean = read_f32(fp, I, "imputer_mean");
  m->sc_mean  = read_f32(fp, I, "scaler_mean");
  m->sc_scale = read_f32(fp, I, "scaler_scale");
  m->W1 = read_f32(fp, H * I, "W1"); m->b1 = read_f32(fp, H, "b1");
  m->bn_g = read_f32(fp, H, "bn_gamma"); m->bn_b = read_f32(fp, H, "bn_beta");
  m->bn_m = read_f32(fp, H, "bn_mean");  m->bn_v = read_f32(fp, H, "bn_var");
  m->W2 = read_f32(fp, O * H, "W2"); m->b2 = read_f32(fp, O, "b2");
  return m;
}

static updec_t *updec_load(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) udie("cannot open .updec", path);
  updec_t *m = updec_read(fp);
  fclose(fp);
  return m;
}

static updec_t *updec_load_buf(const void *buf, size_t len) {
  FILE *fp = fmemopen((void *)buf, len, "rb");
  if (!fp) udie("fmemopen failed for embedded model", NULL);
  updec_t *m = updec_read(fp);
  fclose(fp);
  return m;
}

static void updec_free(updec_t *m) {
  if (!m) return;
  free(m->imp_mean); free(m->sc_mean); free(m->sc_scale);
  free(m->W1); free(m->b1); free(m->bn_g); free(m->bn_b); free(m->bn_m); free(m->bn_v);
  free(m->W2); free(m->b2); free(m);
}

/* Standardize one sample's features once (impute NaN with the model mean, then
 * (x - scaler_mean)/scaler_scale), so the Linear1 loop need not repeat it. */
static void standardize(const updec_t *m, const double *feat, double *xs) {
  for (int j = 0; j < m->n_in; ++j) {
    double x = feat[j];
    if (isnan(x)) x = m->imp_mean[j];
    xs[j] = (x - m->sc_mean[j]) / m->sc_scale[j];
  }
}

static void forward_xs(const updec_t *m, const double *xs, double *out, double *hbuf) {
  int I = m->n_in, H = m->n_hidden, O = m->n_out;
  for (int i = 0; i < H; ++i) {
    const float *w = m->W1 + (size_t)i * I;
    double acc = m->b1[i];
    for (int j = 0; j < I; ++j) acc += (double)w[j] * xs[j];
    if (acc < 0) acc = 0;
    hbuf[i] = m->bn_g[i] * (acc - m->bn_m[i]) / sqrt((double)m->bn_v[i] + m->bn_eps) + m->bn_b[i];
  }
  for (int k = 0; k < O; ++k) {
    const float *w = m->W2 + (size_t)k * H;
    double acc = m->b2[k];
    for (int i = 0; i < H; ++i) acc += (double)w[i] * hbuf[i];
    out[k] = 1.0 / (1.0 + exp(-acc));
  }
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
    "  Upscale an MRMP embedding to CpG-level methylation for one genomic block,\n"
    "  by running a trained MLP block decoder (pure C inference, no torch).\n"
    "\n"
    "Arguments:\n"
    "  <model>   A bare decoder (.updec) OR a bundle with the MRMP attached\n"
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
    "  A YAME .cg (format 6), one record per input sample. If the bundle carries an\n"
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

static void write_pred_cg(BGZF *fp, const updec_t *m, const double *outv, cdata_t *mask) {
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

static void sink_emit(usink_t *sk, const updec_t *m, const double *outv) {
  if (sk->as_cg) { write_pred_cg(sk->cg, m, outv, sk->mask); return; }
  for (int k = 0; k < m->n_out; ++k) {          /* --probs TSV */
    if (k) fputc('\t', sk->tsv);
    fprintf(sk->tsv, "%.6g", outv[k]);
  }
  fputc('\n', sk->tsv);
}

/* Mode A: feature TSV in (bare .updec). */
static long run_from_tsv(const updec_t *m, const char *feat_path, usink_t *sk) {
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
    standardize(m, feat, xs);
    forward_xs(m, xs, outv, hbuf);
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
static long run_from_cg(const updec_t *m, const char *mrmp_path,
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
    standardize(m, feat, xs);
    forward_xs(m, xs, outv, hbuf);
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

int main_upscale(int argc, char *argv[]) {
  const char *out_path = NULL;
  int with_probs = 0;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--probs") == 0) with_probs = 1;
    else if (strcmp(argv[i], "-h") == 0) return upscale_usage();
    else break;
  }
  if (argc - i < 1 || argc - i > 2) return upscale_usage();
  const char *model_path = argv[i];
  const char *input_path = (argc - i == 2) ? argv[i + 1] : "-";

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

  if (ms_bundle_is(model_path)) {
    /* .updecx: model + MRMP (+ optional outcpg) bundled; input is a query .cg. */
    size_t mlen, olen;
    void *mbuf = ms_bundle_section(model_path, "model", &mlen);
    updec_t *m = updec_load_buf(mbuf, mlen);
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
    updec_free(m);
  } else {
    updec_t *m = updec_load(model_path);
    n_out = m->n_out;
    row = run_from_tsv(m, input_path, &sk);
    updec_free(m);
  }

  if (sk.as_cg) bgzf_close(sk.cg);
  else if (sk.tsv != stdout) fclose(sk.tsv);
  if (sk.mask) free_cdata(&mask);
  fprintf(stderr, "[methscope] upscaled %ld sample(s) x %d CpGs%s\n",
          row, sk.mask ? (int)mask.n : n_out, sk.mask ? " (genome .cg)" : "");
  return 0;
}
