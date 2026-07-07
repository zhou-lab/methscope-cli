// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Cell-type deconvolution (the C replacement for the R nnls_deconv()).
 *
 * Given a mixture query (.cg) and a self-contained deconvolution reference
 * (<panel.refx> from `matrix --refx`: a celltype x pattern signature + its MRMP),
 * estimate each mixture cell's non-negative cell-type proportions.
 *
 * Algorithm:
 *   1. Featurize the mixture against the reference's MRMP -> cell x pattern betas
 *      (NaN = no overlap), keeping per-pattern coverage counts (N_overlap).
 *   2. Intersect ref and mixture by pattern name over ALL patterns -- every non-Pna
 *      state is used (no rank cutoff, no variance filter). A cell-type-specific
 *      marker has LOW cross-cell-type variance, so a variance filter would drop
 *      exactly the most informative patterns; and low-recurrence patterns carry
 *      real signal, so a leading-N cutoff discards it. No imputation here: the
 *      reference is imputed NaN-free at .refx build time (`matrix --refx`).
 *   3. Per-sample complete-case NNLS: for each mixture cell solve
 *      min ||ref * x - cell||  s.t. x >= 0 over ONLY the patterns that cell
 *      observed (non-NaN) and that pass --min-cov (>= k covered CpGs), then
 *      normalize x / sum(x). A cell deconvolves identically alone or batched.
 *
 * Output: one row per mixture cell, columns = cell types (proportions).
 * (R returns the transpose; we keep row=cell to match predict/matrix.)
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "methscope.h"
#include "nnls.h"
#include "bundle.h"    /* ms_mrmp_resolve + .refx bundle sections */

static void ddie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] %s\n", msg);
  exit(1);
}

/* Extract a bundle section to a fresh temp file; returns the path (caller
 * unlinks + frees). Used to unpack a .refx into its MRMP + signature TSV. */
static char *extract_section_tmp(const char *bundle, const char *sec, const char *suffix) {
  size_t len; void *buf = ms_bundle_section(bundle, sec, &len);
  char tmpl[64];
  snprintf(tmpl, sizeof tmpl, "/tmp/methscope_refx_XXXXXX%s", suffix);
  int fd = mkstemps(tmpl, (int)strlen(suffix));
  if (fd < 0) ddie("cannot create temp file for .refx section", sec);
  for (size_t off = 0; off < len; ) {
    ssize_t w = write(fd, (char *)buf + off, len - off);
    if (w <= 0) ddie("error writing temp .refx section", sec);
    off += (size_t)w;
  }
  close(fd); free(buf);
  return strdup(tmpl);
}

/* ------------------------------------------------------------------ */
/* A dense labeled matrix (row-major), NaN = missing.                  */
/* ------------------------------------------------------------------ */
typedef struct {
  int     nrow, ncol;
  char  **row_names;   /* nrow */
  char  **col_names;   /* ncol */
  double *v;           /* nrow x ncol, row-major */
  int    *N;           /* nrow x ncol N_overlap (covered CpG count), or NULL */
} dmat_t;

static void dm_free(dmat_t *m) {
  if (!m) return;
  for (int i = 0; i < m->nrow; ++i) free(m->row_names[i]);
  for (int j = 0; j < m->ncol; ++j) free(m->col_names[j]);
  free(m->row_names); free(m->col_names); free(m->v); free(m->N);
}

/* Convert an ms_matrix_t (cell x pattern) into a dmat_t (borrows nothing). */
static dmat_t dm_from_matrix(const ms_matrix_t *m) {
  dmat_t d;
  d.nrow = m->n_cells; d.ncol = m->n_patterns;
  d.row_names = malloc((size_t)d.nrow * sizeof(char *));
  d.col_names = malloc((size_t)d.ncol * sizeof(char *));
  d.v = malloc((size_t)d.nrow * d.ncol * sizeof(double));
  d.N = malloc((size_t)d.nrow * d.ncol * sizeof(int));
  if (!d.row_names || !d.col_names || !d.v || !d.N) ddie("out of memory (mixture)", NULL);
  for (int i = 0; i < d.nrow; ++i) d.row_names[i] = strdup(m->cell_names[i]);
  for (int j = 0; j < d.ncol; ++j) d.col_names[j] = strdup(m->pattern_names[j]);
  memcpy(d.v, m->M, (size_t)d.nrow * d.ncol * sizeof(double));
  memcpy(d.N, m->N, (size_t)d.nrow * d.ncol * sizeof(int));
  return d;
}

/* ------------------------------------------------------------------ */
/* Read a celltype x pattern TSV (header: label<TAB>P1<TAB>...; "NA"->NaN). */
/* ------------------------------------------------------------------ */
static dmat_t dm_read_tsv(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) ddie("cannot open signature .ref", path);
  char *line = NULL; size_t cap = 0; ssize_t len;

  /* header */
  if ((len = getline(&line, &cap, fp)) == -1) ddie("empty signature .ref", path);
  if (len && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
  if (len && line[len-1] == '\r') line[--len] = '\0';
  int ccap = 16, ncol = 0;
  char **cols = malloc(ccap * sizeof(char *));
  char *save = NULL;
  int first = 1;
  for (char *t = strtok_r(line, "\t", &save); t; t = strtok_r(NULL, "\t", &save)) {
    if (first) { first = 0; continue; }           /* skip the row-label header cell */
    if (ncol == ccap) { ccap *= 2; cols = realloc(cols, ccap * sizeof(char *)); }
    cols[ncol++] = strdup(t);
  }
  if (ncol == 0) ddie("signature .ref has no pattern columns", path);

  /* rows */
  int   rcap = 64, nrow = 0;
  char **rows = malloc(rcap * sizeof(char *));
  double *v   = malloc((size_t)rcap * ncol * sizeof(double));
  while ((len = getline(&line, &cap, fp)) != -1) {
    if (len && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
    if (len && line[len-1] == '\r') line[--len] = '\0';
    if (len == 0) continue;
    if (nrow == rcap) { rcap *= 2; rows = realloc(rows, rcap * sizeof(char *));
                        v = realloc(v, (size_t)rcap * ncol * sizeof(double)); }
    int col = -1;                                  /* -1 = the row-label field */
    char *sv = NULL;
    for (char *t = strtok_r(line, "\t", &sv); t; t = strtok_r(NULL, "\t", &sv)) {
      if (col == -1) { rows[nrow] = strdup(t); col = 0; continue; }
      if (col >= ncol) break;                      /* ignore extra fields */
      double val;
      if (strcmp(t, "NA") == 0 || t[0] == '\0') val = NAN;
      else val = strtod(t, NULL);
      v[(size_t)nrow * ncol + col] = val;
      col++;
    }
    for (; col >= 0 && col < ncol; ++col) v[(size_t)nrow * ncol + col] = NAN; /* short row */
    nrow++;
  }
  free(line); fclose(fp);
  if (nrow < 2) ddie("signature .ref needs >=2 cell types (rows)", path);

  dmat_t d = { nrow, ncol, rows, cols, v, NULL };  /* signature TSV has no coverage */
  return d;
}

/* find a column index by name, or -1 */
static int col_index(const dmat_t *m, const char *name, int limit) {
  for (int j = 0; j < limit; ++j)
    if (strcmp(m->col_names[j], name) == 0) return j;
  return -1;
}

/* ------------------------------------------------------------------ */
/* NNLS wrapper: solve min||A x - b||, x>=0. A is m x n COLUMN-major.    */
/* Uses caller-provided scratch (a_work m*n, b_work m, w n, zz m, idx n).*/
/* A and b are NOT modified (copied into scratch). Returns nnls mode.    */
/* ------------------------------------------------------------------ */
static int solve_nnls(const double *A, int m, int n, const double *b,
                      double *x, double *a_work, double *b_work,
                      double *w, double *zz, int *idx) {
  memcpy(a_work, A, (size_t)m * n * sizeof(double));
  memcpy(b_work, b, (size_t)m * sizeof(double));
  int mda = m, mm = m, nn = n, mode = 0;
  double rnorm = 0;
  nnls_c(a_work, &mda, &mm, &nn, b_work, x, &rnorm, w, zz, idx, &mode);
  return mode;
}

static int deconv_usage(void) {
  fprintf(stderr,
    "\n"
    "Usage:\n"
    "  methscope deconv [options] <mixture.cg> <panel.refx>\n"
    "\n"
    "Purpose:\n"
    "  Estimate per-cell cell-type proportions by non-negative least squares\n"
    "  (NNLS) deconvolution against a cell-type signature reference.\n"
    "\n"
    "Arguments:\n"
    "  <mixture.cg>     Mixture methylome(s); '-' reads a .cg stream from stdin\n"
    "                   (cells are then named 1,2,3,... as a stream has no index).\n"
    "  <panel.refx>     A self-contained reference bundle (signature + MRMP) from\n"
    "                   `matrix --refx`. '-' reads the .refx from stdin, so you can\n"
    "                   build + deconvolve in one pipe:\n"
    "                     matrix --refx -o - pseudobulk.cg ref.mrmp \\\n"
    "                       | deconv mixture.cg -\n"
    "\n"
    "Options:\n"
    "  -o <out.tsv>     Write output to a file instead of stdout.\n"
    "  --min-cov <k>    Per cell, only use patterns with >= k covered CpGs in the\n"
    "                   mixture (default: 1). Higher k (e.g. 3) drops thinly-\n"
    "                   covered, noisy patterns -- helps on sparse input.\n"
    "  --no-header      Suppress the header line.\n"
    "  -h               Show this help message.\n"
    "\n"
    "Output:\n"
    "  One row per mixture cell; columns are cell types (proportions summing to 1).\n"
    "\n");
  return 1;
}

/* Read all of stdin into a fresh temp file; returns the path (caller unlinks). */
static char *slurp_stdin_tmp(void) {
  char tmpl[] = "/tmp/methscope_refx_XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) ddie("cannot create temp file for stdin reference", NULL);
  char buf[1 << 16]; size_t n;
  while ((n = fread(buf, 1, sizeof buf, stdin)) > 0)
    for (size_t off = 0; off < n; ) {
      ssize_t w = write(fd, buf + off, n - off);
      if (w <= 0) ddie("error writing temp reference", NULL);
      off += (size_t)w;
    }
  close(fd);
  return strdup(tmpl);
}

int main_deconv(int argc, char *argv[]) {
  const char *out_path = NULL;
  int no_header = 0, min_cov = 1;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--no-header") == 0) no_header = 1;
    else if (strcmp(argv[i], "--min-cov") == 0 && i+1 < argc) min_cov = atoi(argv[++i]);
    else if (strcmp(argv[i], "-h") == 0) return deconv_usage();
    else break;
  }
  if (argc - i != 2) return deconv_usage();
  const char *mixture_cg = argv[i];
  const char *refx = argv[i + 1];

  /* `deconv mixture.cg -` reads the .refx bundle from stdin (e.g. piped from
   * `matrix --refx -o -`); bundle reading needs a seekable file, so slurp it. */
  char *tmp_refx = NULL;
  if (strcmp(refx, "-") == 0) {
    if (strcmp(mixture_cg, "-") == 0)
      ddie("both mixture and reference read stdin; give at least one as a file", NULL);
    tmp_refx = slurp_stdin_tmp();
    refx = tmp_refx;
  }

  if (!ms_bundle_is(refx))
    ddie("expected a .refx reference bundle (build one with `matrix --refx`)", refx);
  char *kind = ms_bundle_kind(refx);
  if (!kind || strcmp(kind, "refx") != 0)
    ddie("bundle is not a deconvolution reference (kind != refx)", kind ? kind : "(unmarked)");
  free(kind);
  char *tmp_sig = extract_section_tmp(refx, "model", ".ref");
  const char *ref_mrmp = refx;      /* the bundle's front bytes ARE the mrmp .cm */
  const char *sig_ref  = tmp_sig;

  /* ---- featurize the mixture, load the signature ---- */
  ms_matrix_t *mm = ms_matrix_build(mixture_cg, ref_mrmp);
  dmat_t mix = dm_from_matrix(mm);
  ms_matrix_free(mm);
  if (tmp_refx) { unlink(tmp_refx); free(tmp_refx); }   /* mrmp read; safe to drop */
  dmat_t ref = dm_read_tsv(sig_ref);
  if (tmp_sig) { unlink(tmp_sig); free(tmp_sig); }

  /* No imputation here: the reference is imputed NaN-free at .refx build time
   * (`matrix --refx`, column-median), and the mixture keeps its NAs so each cell
   * fits complete-case in the NNLS loop below -- a single mixture sample then
   * deconvolves identically whether run alone or batched with others. */

  int n_ct = ref.nrow;                          /* cell types  */

  /* ---- used patterns: ALL patterns common to reference and mixture (by name).
   * No leading-N cutoff and no variance filter -- every non-Pna state is a
   * candidate. Per-cell complete-case (below) selects which of these a given
   * sparse cell actually observed. ---- */
  int   *mix_col = malloc(ref.ncol * sizeof(int));   /* per used pattern */
  int   *ref_col = malloc(ref.ncol * sizeof(int));
  int    n_used  = 0;
  for (int rc = 0; rc < ref.ncol; ++rc) {
    const char *pname = ref.col_names[rc];
    if (strcmp(pname, "Pna") == 0) continue;      /* NA background, never a signature feature */
    int mc = col_index(&mix, pname, mix.ncol);
    if (mc < 0) continue;                         /* pattern not present in the mixture */
    mix_col[n_used] = mc; ref_col[n_used] = rc; n_used++;
  }
  if (n_used == 0)
    ddie("no usable patterns after intersecting reference and mixture by name", NULL);

  /* ---- reference design matrix A: (n_used x n_ct), COLUMN-major ---- */
  double *A = malloc((size_t)n_used * n_ct * sizeof(double));
  for (int t = 0; t < n_ct; ++t)
    for (int p = 0; p < n_used; ++p)
      A[(size_t)t * n_used + p] = ref.v[(size_t)t * ref.ncol + ref_col[p]];

  /* ---- per-cell NNLS ---- */
  double *b  = malloc((size_t)n_used * sizeof(double));
  double *x  = malloc((size_t)n_ct  * sizeof(double));
  double *aw = malloc((size_t)n_used * n_ct * sizeof(double));
  double *bw = malloc((size_t)n_used * sizeof(double));
  double *w  = malloc((size_t)n_ct  * sizeof(double));
  double *zz = malloc((size_t)n_used * sizeof(double));
  int    *ix = malloc((size_t)n_ct  * sizeof(int));
  double *Ar = malloc((size_t)n_used * n_ct * sizeof(double));  /* per-cell design */
  int    *use = malloc((size_t)n_used * sizeof(int));           /* per-cell pattern idx */
  if (!A || !b || !x || !aw || !bw || !w || !zz || !ix || !Ar || !use)
    ddie("out of memory (nnls)", NULL);

  FILE *out = out_path ? fopen(out_path, "w") : stdout;
  if (!out) ddie("cannot open output", out_path);
  if (!no_header) {
    fputs("cell", out);
    for (int t = 0; t < n_ct; ++t) fprintf(out, "\t%s", ref.row_names[t]);
    fputc('\n', out);
  }

  for (int r = 0; r < mix.nrow; ++r) {
    /* Per-cell (per-sample) complete-case: use only patterns THIS cell actually
     * observed (non-NA beta), and with --min-cov k>1 also require >= k covered CpGs.
     * Selection depends solely on this cell + the reference, so a sample deconvolves
     * identically alone or batched. */
    int nu = 0;
    for (int p = 0; p < n_used; ++p) {
      if (isnan(mix.v[(size_t)r * mix.ncol + mix_col[p]])) continue;   /* not measured here */
      if (min_cov > 1) {
        int cov = mix.N ? mix.N[(size_t)r * mix.ncol + mix_col[p]] : 0;
        if (cov < min_cov) continue;
      }
      use[nu++] = p;
    }
    fputs(mix.row_names[r], out);
    if (nu == 0) {
      for (int t = 0; t < n_ct; ++t) fprintf(out, "\t%.6f", 0.0);
      fputc('\n', out);
      fprintf(stderr, "[methscope] warning: no patterns with >=%d covered CpGs for cell %s\n",
              min_cov, mix.row_names[r]);
      continue;
    }
    /* compact per-cell design A_r (nu x n_ct, column-major) + b_r */
    for (int t = 0; t < n_ct; ++t)
      for (int q = 0; q < nu; ++q)
        Ar[(size_t)t * nu + q] = A[(size_t)t * n_used + use[q]];
    for (int q = 0; q < nu; ++q)
      b[q] = mix.v[(size_t)r * mix.ncol + mix_col[use[q]]];
    int mode = solve_nnls(Ar, nu, n_ct, b, x, aw, bw, w, zz, ix);
    double s = 0;
    for (int t = 0; t < n_ct; ++t) { if (x[t] < 0) x[t] = 0; s += x[t]; }
    for (int t = 0; t < n_ct; ++t)
      fprintf(out, "\t%.6f", s > 0 ? x[t] / s : 0.0);
    fputc('\n', out);
    if (mode != 1)
      fprintf(stderr, "[methscope] warning: NNLS did not converge for cell %s (mode=%d)\n",
              mix.row_names[r], mode);
  }

  if (out != stdout) fclose(out);
  free(A); free(b); free(x); free(aw); free(bw); free(w); free(zz); free(ix);
  free(Ar); free(use);
  free(mix_col); free(ref_col);
  dm_free(&mix); dm_free(&ref);
  return 0;
}
