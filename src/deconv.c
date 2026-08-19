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
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include "methscope.h"
#include "nnls.h"
#include "bundle.h"    /* ms_mrmp_resolve + .refx bundle sections */

/* Adaptive mode (src/deconv_adaptive.c): pick the cell types from the sample,
 * rebuild the pattern set over just those, re-solve. Needs the REFERENCE STORE
 * rather than a prebuilt .refx, because a rebuilt pattern set has no betas
 * until they are computed from the data. */
int ms_deconv_adaptive(const char *ref_cg, const char *query_cg,
                       const char *out_path, int max_round,
                       uint64_t group_thresh, double drop_below,
                       double qlo, double qhi, double beta_thr,
                       uint32_t mincov, uint32_t min_cov_pat, int no_header,
                       int nthreads, int columns_only, const char *scope_out,
                       int pairwise, unsigned long pair_min_cpg,
                       int weighted, uint64_t pattern_cap, double weight_expo,
                       uint64_t dm_top);

static void ddie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] deconv: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] deconv: %s\n", msg);
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
  if (!cols) ddie("out of memory", NULL);
  char *save = NULL;
  int first = 1;
  for (char *t = strtok_r(line, "\t", &save); t; t = strtok_r(NULL, "\t", &save)) {
    if (first) { first = 0; continue; }           /* skip the row-label header cell */
    if (ncol == ccap) {
      ccap *= 2;
      char **tmp = realloc(cols, ccap * sizeof(char *));
      if (!tmp) ddie("out of memory", NULL);
      cols = tmp;
    }
    cols[ncol++] = strdup(t);
  }
  if (ncol == 0) ddie("signature .ref has no pattern columns", path);

  /* rows */
  int   rcap = 64, nrow = 0;
  char **rows = malloc(rcap * sizeof(char *));
  double *v   = malloc((size_t)rcap * ncol * sizeof(double));
  if (!rows || !v) ddie("out of memory", NULL);
  while ((len = getline(&line, &cap, fp)) != -1) {
    if (len && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
    if (len && line[len-1] == '\r') line[--len] = '\0';
    if (len == 0) continue;
    if (nrow == rcap) {
      rcap *= 2;
      char **rtmp = realloc(rows, rcap * sizeof(char *));
      if (!rtmp) ddie("out of memory", NULL);
      rows = rtmp;
      double *vtmp = realloc(v, (size_t)rcap * ncol * sizeof(double));
      if (!vtmp) ddie("out of memory", NULL);
      v = vtmp;
    }
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
  ms_help(stderr,
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
    "  --unweighted     Do NOT weight patterns by observed CpG count. The\n"
    "                   default weights each pattern by sqrt(n): its value is a\n"
    "                   mean over the CpGs the query covered, so precision goes\n"
    "                   as n. Unweighted, a 1-CpG pattern counts as much as a\n"
    "                   500-CpG one, which at 2^16 is most of them.\n"
    "  --min-cov <k>    Per cell, only use patterns with >= k covered CpGs in the\n"
    "                   mixture (default: 1). Higher k (e.g. 3) drops thinly-\n"
    "                   covered, noisy patterns -- helps on sparse input.\n"
    "  --no-header      Suppress the header line.\n"
    "  -h               Show this help message.\n"
    "\n"
    "Adaptive mode -- pick the cell types from the sample, then re-solve:\n"
    "  --adaptive       Take the REFERENCE STORE (.cg) instead of a .refx, solve\n"
    "                   over every class, drop the ones the sample does not\n"
    "                   contain, REBUILD the pattern set over the survivors, and\n"
    "                   solve again. The rebuild is the point: a pattern must\n"
    "                   hold across every class in its set, so dropping classes\n"
    "                   admits CpGs the wider filter rejected -- Monocyte vs\n"
    "                   Macrophage carries 300 segregating CpGs in a 33-class\n"
    "                   build and 11,935 as its own 2-class set. Restricting the\n"
    "                   classes WITHOUT rebuilding is worth nothing.\n"
    "                   Holds the reference in memory (2 B per class per CpG:\n"
    "                   ~1.9 GB for 33 classes on hg38), so run it on a compute\n"
    "                   node, never a login node.\n"
    "                   A multi-record .cg is a cohort: every record gets its\n"
    "                   own class set and the reference load is paid once.\n"
    "  --nthreads N     Run N records in parallel (default 1). The reference is\n"
    "                   read-only and shared, so memory grows by roughly 64 MB\n"
    "                   per thread rather than linearly. Needs a <query>.idx to\n"
    "                   seek by; without one it falls back to one thread. Output\n"
    "                   is written in record order, so it does not depend on N.\n"
    "  --columns-only   Let the inferred scope pick the CpGs but keep EVERY\n"
    "                   class in the solve. The scope then costs precision when\n"
    "                   it is wrong instead of losing a component outright,\n"
    "                   which is the failure mode of dropping classes outright.\n"
    "  --no-pairwise    Turn OFF the pairwise adjudication, which is ON by\n"
    "                   default under --adaptive. On it settles each confusable\n"
    "                   pair on a 2-class add-on built for that pair, then cuts\n"
    "                   the loser. The scope decision is otherwise taken on the\n"
    "                   round-0 panel, where Monocyte and Macrophage are 300\n"
    "                   CpGs apart and a 2^16 query sees 0.7 of them: a zero\n"
    "                   there is NNLS breaking a tie, not absence. The pair\n"
    "                   alone carries 11,935. Measured on 200 mixtures it cuts\n"
    "                   TVD 6-19%% at 2^18-2^22 and lifts scope precision 3-5\n"
    "                   points for at most 2 points of recall; at 2^16 it\n"
    "                   abstains and is bit-identical to off.\n"
    "  --pattern-cap N  Cap the evidence one pattern may claim at N observed\n"
    "                   CpGs: its weight becomes sqrt(min(n, N)) (default 0,\n"
    "                   no cap). sqrt(n) weighting is squared by the solve, so\n"
    "                   influence is LINEAR in n and a few huge patterns can\n"
    "                   own the fit -- in the 8-class immune rebuild five hold\n"
    "                   91.3%% of the panel and none separates T.Cell.CD4 from\n"
    "                   T.Cell.CD8, whose 29 patterns hold 0.41%%. The pair goes\n"
    "                   collinear and CD4 absorbs CD8 wholesale. Below the cap\n"
    "                   nothing changes; above it patterns tie instead of\n"
    "                   competing. Applies to the flat path too.\n"
    "  --weight-exponent E  Row weight is n^E (default 0.5 = sqrt(n), the\n"
    "                   precision weight). The solve squares it, so influence\n"
    "                   is n^2E: 0.5 gives influence linear in n, 0.25 gives\n"
    "                   sqrt(n). 0.25 compresses the range instead of\n"
    "                   truncating it, lifting the 8-class panel's CD4/CD8\n"
    "                   splitters from 0.41%% of the fit to 5.84%%. 0 is\n"
    "                   --unweighted. Applies to the flat path too.\n"
    "  --delta-mean-top N  Per BINSTRING, keep only the N MEASURED CpGs with\n"
    "                   the cleanest class gap (default 1000; 0 = keep all,\n"
    "                   and fewer than N if the q-filter admits fewer). The\n"
    "                   rebuild otherwise lets one binstring swamp the panel:\n"
    "                   in the 8-class immune rebuild `11111011` held 124,546\n"
    "                   CpGs and separates nothing in an immune mixture, while\n"
    "                   the 29 patterns splitting T.Cell.CD4 from T.Cell.CD8\n"
    "                   held 0.41%% of the panel. Influence is linear in CpG\n"
    "                   count, so that WAS the fit. Mirrors mrmp-build.\n"
    "  --pair-min-cpg N Refuse a pairwise verdict on fewer than N observed\n"
    "                   discriminating CpGs (default 50). Below that the call is\n"
    "                   guesswork: at 2^16 a floor of 50 makes pairwise abstain\n"
    "                   and match classic exactly, while at 2^20 it never binds.\n"
    "  --scope-out F    Write the settled scope per class (1/0 per class).\n"
    "                   Under --columns-only the scope is invisible in the\n"
    "                   proportions, and its quality is what decides whether\n"
    "                   adaptivity pays, so it has to be measurable.\n"
    "  --max-round N    Cap on drop-and-resolve rounds (default 8). It stops\n"
    "                   early when the class set stops changing, and normally\n"
    "                   does so well before the cap -- a 1M-CpG mixture peeled\n"
    "                   33 -> 11 -> 8 -> 7 and then held. The cap exists because\n"
    "                   reinstating a class would break monotonicity and could\n"
    "                   otherwise cycle; it is not the usual stopping rule.\n"
    "  --drop-below F   Drop a GROUP whose every member is <= F (default 0, i.e.\n"
    "                   only exact zeros). Groups, not classes: NNLS zeroes one\n"
    "                   of two near-collinear columns arbitrarily, so a class\n"
    "                   zeroed only because its partner absorbed it is protected\n"
    "                   by that partner. A drop is not reversible.\n"
    "  --group-thresh N Two classes are one group when the round-0 panel\n"
    "                   separates them by <= N CpGs (default 2000; the human\n"
    "                   reference has a clean gap, 4 pairs at 300-822 and the\n"
    "                   next at 4,192).\n"
    "  --qfilter LO,HI  Admission band for the rebuild (default 0.30,0.70).\n"
    "  --beta-threshold B  Binstring cut (default 0.50).\n"
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
  /* adaptive mode */
  /* pairwise ON by default under --adaptive: at a matched 0.30,0.70 filter it
   * cuts TVD 6-19% (0.112/0.053/0.028 -> 0.105/0.043/0.023 at 2^18/20/22) and
   * lifts scope precision 3-5 points, costing at most 2 points of recall. At
   * 2^16 it abstains under --pair-min-cpg and is bit-identical to leaving it
   * off, so there is no rung where it has to be argued for. --no-pairwise. */
  int adaptive = 0, max_round = 8, nthreads = 1, columns_only = 0, pairwise = 1;
  /* 50, not 5: below that the add-on's verdict is guesswork. Swept on 200
   * mixtures -- at 2^16 a floor of 50 makes pairwise abstain and fall back to
   * classic exactly (TVD 0.278 both), while at 2^20 every add-on clears even
   * 100 observed CpGs so the floor never binds and the full gain is kept. */
  unsigned long pair_min_cpg = 50;
  int weighted = 1;   /* weight patterns by observed CpGs; --unweighted opts out */
  /* 0 = uncapped, the rule every number before 20260816 was measured under */
  unsigned long pattern_cap = 0;
  double weight_expo = 0.5;   /* sqrt(n); 0.25 = sqrt(sqrt(n)), 0 = unweighted */
  /* 1000 measured CpGs per binstring, or fewer if the q-filter admits fewer.
   * Keeps each binstring's best evidence instead of all of its mediocre
   * evidence, and stops one binstring owning the fit. 0 disables. */
  unsigned long dm_top = 1000;
  const char *scope_out = NULL;
  uint64_t group_thresh = 2000;
  /* 0.30/0.70 matches ms_select_defaults() so the rebuild admits the same CpGs
   * the classifier's MRMP does; see the note there for the measurement. */
  double drop_below = 0.0, qlo = 0.30, qhi = 0.70, beta_thr = 0.5;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--no-header") == 0) no_header = 1;
    else if (strcmp(argv[i], "--adaptive") == 0) adaptive = 1;
    else if (strcmp(argv[i], "--columns-only") == 0) columns_only = 1;
    else if (strcmp(argv[i], "--pairwise") == 0) pairwise = 1;
    else if (strcmp(argv[i], "--no-pairwise") == 0) pairwise = 0;
    else if (strcmp(argv[i], "--unweighted") == 0) weighted = 0;
    else if (strcmp(argv[i], "--pattern-cap") == 0 && i+1 < argc)
      pattern_cap = strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--weight-exponent") == 0 && i+1 < argc)
      weight_expo = atof(argv[++i]);
    else if (strcmp(argv[i], "--delta-mean-top") == 0 && i+1 < argc)
      dm_top = strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--pair-min-cpg") == 0 && i+1 < argc)
      pair_min_cpg = strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--scope-out") == 0 && i+1 < argc)
      scope_out = argv[++i];
    else if ((strcmp(argv[i], "--nthreads") == 0 ||
              strcmp(argv[i], "--threads") == 0) && i+1 < argc)
      nthreads = atoi(argv[++i]);
    else if (strcmp(argv[i], "--max-round") == 0 && i+1 < argc)
      max_round = atoi(argv[++i]);
    else if (strcmp(argv[i], "--group-thresh") == 0 && i+1 < argc)
      group_thresh = strtoull(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--drop-below") == 0 && i+1 < argc)
      drop_below = atof(argv[++i]);
    else if (strcmp(argv[i], "--qfilter") == 0 && i+1 < argc) {
      if (sscanf(argv[++i], "%lf,%lf", &qlo, &qhi) != 2)
        ddie("--qfilter expects LO,HI", argv[i]);
    }
    else if (strcmp(argv[i], "--beta-threshold") == 0 && i+1 < argc)
      beta_thr = atof(argv[++i]);
    else if (strcmp(argv[i], "--min-cov") == 0 && i+1 < argc) {
      const char *s = argv[++i];
      char *end; errno = 0;
      long v = strtol(s, &end, 10);
      if (errno || end == s || *end != '\0' || v < 0 || v > INT_MAX)
        ddie("--min-cov expects a non-negative integer", s);
      min_cov = (int)v;
    }
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      deconv_usage(); return 0;
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      ddie("unrecognized or incomplete option", argv[i]);
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

  if (adaptive) {
    /* The second argument is the REFERENCE STORE, not a .refx: rebuilding the
     * pattern set per round leaves the prebuilt betas meaningless. */
    if (ms_bundle_is(refx))
      ddie("--adaptive takes the reference .cg, not a .refx "
           "(a rebuilt pattern set has no prebuilt betas)", refx);
    if (max_round < 0) ddie("--max-round must be >= 0", NULL);
    /* Every RECORD of the query gets its own class set; the reference load is
     * paid once. A cohort is therefore just a cat'ted multi-record .cg, which
     * is YAME's own idiom -- no list flag of our own. */
    return ms_deconv_adaptive(refx, mixture_cg, out_path, max_round,
                              group_thresh, drop_below, qlo, qhi, beta_thr,
                              1, (uint32_t)min_cov, no_header, nthreads,
                              columns_only, scope_out, pairwise, pair_min_cpg,
                              weighted, (uint64_t)pattern_cap, weight_expo,
                              (uint64_t)dm_top);
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
  if (!mix_col || !ref_col) ddie("out of memory", NULL);
  int    n_used  = 0;
  for (int rc = 0; rc < ref.ncol; ++rc) {
    const char *pname = ref.col_names[rc];
    if (ms_is_pna_name(pname)) continue;          /* NA background, never a signature feature */
    int mc = col_index(&mix, pname, mix.ncol);
    if (mc < 0) continue;                         /* pattern not present in the mixture */
    mix_col[n_used] = mc; ref_col[n_used] = rc; n_used++;
  }
  if (n_used == 0)
    ddie("no usable patterns after intersecting reference and mixture by name", NULL);

  /* ---- reference design matrix A: (n_used x n_ct), COLUMN-major ---- */
  double *A = malloc((size_t)n_used * n_ct * sizeof(double));
  if (!A) ddie("out of memory", NULL);
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

  FILE *out = (out_path && strcmp(out_path, "-") != 0)
              ? fopen(out_path, "w") : stdout;
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
    /* Weight each pattern by the CpGs this cell actually observed: the value
     * is a mean over them, so precision scales with n and sqrt(n) on the row
     * is ordinary weighted least squares. Unweighted, a 1-CpG pattern counts
     * as much as a 500-CpG one. */
    if (weighted && mix.N) {
      for (int q = 0; q < nu; ++q) {
        double nq = (double)mix.N[(size_t)r * mix.ncol + mix_col[use[q]]];
        /* same rule as the adaptive path -- see pattern_weight() there */
        if (pattern_cap && nq > (double)pattern_cap) nq = (double)pattern_cap;
        double wq = (nq > 0) ? pow(nq, weight_expo) : 1.0;
        if (wq <= 0) wq = 1;
        for (int t = 0; t < n_ct; ++t) Ar[(size_t)t * nu + q] *= wq;
        b[q] *= wq;
      }
    }
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
