// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Build the cell x pattern beta matrix (the C replacement for the R
 * GenerateInput()). Each query record (cell/pixel) is summarized against every
 * mask record (MRMP pattern) in reference.cm using YAME's summary core; the
 * per-(cell,pattern) Beta becomes a matrix entry, NaN where there is no overlap.
 *
 * Column order = by numeric pattern id (first run of digits in the state name),
 * with "Pna" always last (the R GenerateInput() order); the booster's matching
 * .ubj was trained on that same order.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include "methscope.h"

/* YAME (libyame.a) public headers */
#include "cfile.h"     /* open_cfile, read_cdata1, cdata_t, BGZF, snames_t ... */
#include "summary.h"   /* prepare_mask, summarize1, stats_t, config_t        */
#include "bundle.h"    /* ms_mrmp_resolve: accept a bundle where a .mrmp fits */

static void mdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] matrix: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] matrix: %s\n", msg);
  exit(1);
}

/* Numeric sort key for a pattern name, matching R's
 * order(as.numeric(str_extract(name, "\\d+"))): the first run of digits as a
 * number; names without digits (e.g. "Pna") sort last. */
static long pattern_numeric_key(const char *s) {
  for (const char *p = s; *p; ++p)
    if (*p >= '0' && *p <= '9') return atol(p);
  /* Non-numeric names have no recurrence rank. The NA background "Pna" always
   * sorts strictly last; other curated marker names (e.g. "Xa_hi") sort after
   * the numeric patterns but before "Pna", keeping their definition order via
   * the stable tie-break in cmp_colkey. */
  if (strcmp(s, "Pna") == 0) return LONG_MAX;
  return LONG_MAX - 1;
}

typedef struct { long key; int idx; } colkey_t;
static int cmp_colkey(const void *a, const void *b) {
  const colkey_t *x = a, *y = b;
  if (x->key < y->key) return -1;
  if (x->key > y->key) return 1;
  return x->idx - y->idx;            /* stable on ties */
}

/*
 * The MRMP reference (.cm) is a categorical (fmt2) track whose *states* are the
 * patterns (P1..Pn, Pna). summarize1() therefore returns one stats_t per state
 * (st[j].sm = pattern name, st[j].beta = the cell's mean methylation over that
 * pattern's CpGs). We gather every state into a column, then order columns by
 * numeric pattern id exactly as the R GenerateInput() did, so the first
 * npattern columns are the model's features and "Pna" lands last.
 */
ms_matrix_t *ms_matrix_build(const char *query_cg, const char *ref_cm) {
  config_t config = {0};
  config.fname_mask = (char *)ref_cm;

  /* ---- load all mask records into memory, once ---- */
  cfile_t  cf_mask     = open_cfile((char *)ref_cm);
  snames_t snames_mask = loadSampleNamesFromIndex((char *)ref_cm);
  cdata_t *c_masks = NULL;
  size_t   n_masks = 0, mcap = 0;
  for (;;) {
    cdata_t cm = read_cdata1(&cf_mask);
    if (cm.n == 0) break;
    prepare_mask(&cm);
    if (n_masks == mcap) {
      mcap = mcap ? mcap * 2 : 8;
      c_masks = realloc(c_masks, mcap * sizeof(cdata_t));
      if (!c_masks) mdie("out of memory (masks)", NULL);
    }
    c_masks[n_masks++] = cm;
  }
  bgzf_close(cf_mask.fh);
  if (n_masks == 0) mdie("reference contains no records", ref_cm);

  char **mask_names = malloc(n_masks * sizeof(char *));
  for (size_t k = 0; k < n_masks; ++k) {
    char b[32];
    if (snames_mask.n > (int)k) mask_names[k] = strdup(snames_mask.s[k]);
    else { snprintf(b, sizeof(b), "%zu", k + 1); mask_names[k] = strdup(b); }
  }

  /* ---- stream query cells; each summarize1() yields all pattern states ---- */
  cfile_t  cf_qry     = open_cfile((char *)query_cg);
  snames_t snames_qry = loadSampleNamesFromIndex((char *)query_cg);

  size_t   n_raw = 0, rawcap = 0;     /* number of state-columns (set on cell 0) */
  char   **raw_names = NULL;          /* length n_raw, summarize order           */
  double  *raw_row = NULL;            /* scratch row, length rawcap              */
  int     *raw_Ncnt = NULL;           /* scratch N_overlap row, length rawcap    */
  double  *Mraw = NULL;               /* n_cells x n_raw, raw column order        */
  int     *Nraw = NULL;               /* n_cells x n_raw N_overlap, raw order     */
  char   **cell_names = NULL;
  size_t   n_cells = 0, rcap = 0;
  int      first = 1;

  for (size_t iq = 0;; ++iq) {
    cdata_t cq = read_cdata1(&cf_qry);
    if (cq.n == 0) break;
    prepare_mask(&cq);

    size_t col = 0;
    for (size_t k = 0; k < n_masks; ++k) {
      uint64_t n_st = 0;
      stats_t *st = summarize1(&cq, &c_masks[k], &n_st, mask_names[k], "", &config);
      for (uint64_t j = 0; j < n_st; ++j) {
        double v = (st[j].beta >= 0) ? st[j].beta : NAN;
        if (first) {
          if (col == rawcap) {
            rawcap = rawcap ? rawcap * 2 : 1024;
            raw_names = realloc(raw_names, rawcap * sizeof(char *));
            raw_row   = realloc(raw_row,   rawcap * sizeof(double));
            raw_Ncnt  = realloc(raw_Ncnt,  rawcap * sizeof(int));
            if (!raw_names || !raw_row || !raw_Ncnt) mdie("out of memory (columns)", NULL);
          }
          raw_names[col] = strdup(st[j].sm);
        }
        raw_row[col]  = v;
        raw_Ncnt[col] = (st[j].n_o > (uint64_t)INT_MAX) ? INT_MAX : (int)st[j].n_o;
        col++;
      }
      for (uint64_t j = 0; j < n_st; ++j) { free(st[j].sm); free(st[j].sq); }
      if (n_st) free(st);
    }
    if (first) { n_raw = col; first = 0; }
    else if (col != n_raw) mdie("inconsistent pattern count across cells", NULL);

    if (n_cells == rcap) {
      rcap = rcap ? rcap * 2 : 256;
      cell_names = realloc(cell_names, rcap * sizeof(char *));
      Mraw = realloc(Mraw, rcap * n_raw * sizeof(double));
      Nraw = realloc(Nraw, rcap * n_raw * sizeof(int));
      if (!cell_names || !Mraw || !Nraw) mdie("out of memory (cells)", NULL);
    }
    char b[32];
    if (snames_qry.n > (int)iq) cell_names[n_cells] = strdup(snames_qry.s[iq]);
    else { snprintf(b, sizeof(b), "%zu", iq + 1); cell_names[n_cells] = strdup(b); }
    memcpy(Mraw + n_cells * n_raw, raw_row,  n_raw * sizeof(double));
    memcpy(Nraw + n_cells * n_raw, raw_Ncnt, n_raw * sizeof(int));
    n_cells++;
    free_cdata(&cq);
  }
  bgzf_close(cf_qry.fh);
  for (size_t k = 0; k < n_masks; ++k) { free_cdata(&c_masks[k]); free(mask_names[k]); }
  free(c_masks); free(mask_names);
  cleanSampleNames2(snames_mask);
  cleanSampleNames2(snames_qry);
  free(raw_row); free(raw_Ncnt);
  if (n_cells == 0 || n_raw == 0) mdie("no data produced", query_cg);

  /* ---- order columns by numeric pattern id (R parity) ---- */
  colkey_t *ck = malloc(n_raw * sizeof(colkey_t));
  for (size_t i = 0; i < n_raw; ++i) { ck[i].key = pattern_numeric_key(raw_names[i]); ck[i].idx = (int)i; }
  qsort(ck, n_raw, sizeof(colkey_t), cmp_colkey);

  char  **pattern_names = malloc(n_raw * sizeof(char *));
  double *M = malloc(n_cells * n_raw * sizeof(double));
  int    *N = malloc(n_cells * n_raw * sizeof(int));
  if (!pattern_names || !M || !N) mdie("out of memory (reorder)", NULL);
  for (size_t newc = 0; newc < n_raw; ++newc) {
    int oldc = ck[newc].idx;
    pattern_names[newc] = raw_names[oldc];           /* transfer ownership */
    for (size_t r = 0; r < n_cells; ++r) {
      M[r * n_raw + newc] = Mraw[r * n_raw + oldc];
      N[r * n_raw + newc] = Nraw[r * n_raw + oldc];
    }
  }
  free(ck); free(Mraw); free(Nraw); free(raw_names);

  ms_matrix_t *m = malloc(sizeof(ms_matrix_t));
  if (!m) mdie("out of memory", NULL);
  m->n_cells       = (int)n_cells;
  m->n_patterns    = (int)n_raw;
  m->cell_names    = cell_names;
  m->pattern_names = pattern_names;
  m->M             = M;
  m->N             = N;
  return m;
}

void ms_matrix_free(ms_matrix_t *m) {
  if (!m) return;
  for (int i = 0; i < m->n_cells; ++i) free(m->cell_names[i]);
  for (int j = 0; j < m->n_patterns; ++j) free(m->pattern_names[j]);
  free(m->cell_names);
  free(m->pattern_names);
  free(m->M);
  free(m->N);
  free(m);
}

/*
 * Write a TRIMMED copy of an MRMP reference (.cm, a single fmt2 categorical
 * record): keep the states named in keep_names[], and fold EVERY other state
 * (including the original "Pna" and any pattern not kept) into a single "Pna"
 * background state. The result has exactly n_keep (+1 for "Pna" unless it is
 * already among keep_names) states. Used by `train` to bundle only the mrmp the
 * model actually uses, so `predict` featurizes against that same trimmed set.
 *
 * Matching is BY NAME: the .cm's internal state indices follow definition order,
 * not the sorted feature order, so a positional "< N" test would be wrong.
 * (matrix re-sorts states on read, so the new dictionary order is irrelevant.)
 */
void ms_mrmp_trim(const char *in_cm, char *const *keep_names, int n_keep,
                  const char *out_cm) {
  cfile_t cf = open_cfile((char *)in_cm);
  cdata_t c  = read_cdata1(&cf);
  bgzf_close(cf.fh);
  if (c.fmt != '2') mdie("mrmp to trim is not a fmt2 categorical .cm", in_cm);
  cdata_t d = decompress(c);
  free_cdata(&c);
  fmt2_set_aux(&d);
  f2_aux_t *aux = (f2_aux_t *)d.aux;

  /* new state list = keep_names[], with "Pna" appended as the fold target
   * unless it is already listed. */
  int pna_in_keep = 0;
  for (int k = 0; k < n_keep; ++k)
    if (strcmp(keep_names[k], "Pna") == 0) { pna_in_keep = 1; break; }
  int new_nk = n_keep + (pna_in_keep ? 0 : 1);
  const char **new_names = malloc((size_t)new_nk * sizeof(char *));
  if (!new_names) mdie("out of memory (trim names)", NULL);
  for (int k = 0; k < n_keep; ++k) new_names[k] = keep_names[k];
  int pna_new = pna_in_keep ? -1 : n_keep;
  if (!pna_in_keep) new_names[n_keep] = "Pna";
  else for (int k = 0; k < n_keep; ++k)
    if (strcmp(keep_names[k], "Pna") == 0) { pna_new = k; break; }

  /* map each old state index -> new index (fold non-kept -> Pna) */
  uint64_t *old2new = malloc((size_t)aux->nk * sizeof(uint64_t));
  if (!old2new) mdie("out of memory (trim map)", NULL);
  for (uint64_t i = 0; i < aux->nk; ++i) {
    long hit = pna_new;
    for (int k = 0; k < n_keep; ++k)
      if (strcmp(aux->keys[i], keep_names[k]) == 0) { hit = k; break; }
    old2new[i] = (uint64_t)hit;
  }

  /* assemble a fresh decompressed fmt2 buffer: [keys\0...][ '\0' ][ indices ] */
  size_t keys_nb = 0;
  for (int k = 0; k < new_nk; ++k) keys_nb += strlen(new_names[k]) + 1;
  size_t nbytes = keys_nb + 1 + (size_t)d.n * d.unit;
  uint8_t *buf = malloc(nbytes);
  if (!buf) mdie("out of memory (trim buffer)", NULL);
  size_t off = 0;
  for (int k = 0; k < new_nk; ++k) {
    size_t len = strlen(new_names[k]) + 1;   /* include the '\0' */
    memcpy(buf + off, new_names[k], len);
    off += len;
  }
  buf[keys_nb] = '\0';                        /* double-null separator */
  uint8_t *data = buf + keys_nb + 1;
  for (uint64_t i = 0; i < d.n; ++i) {
    uint64_t nv = old2new[f2_get_uint64(&d, i)];
    for (uint8_t j = 0; j < d.unit; ++j) data[i * d.unit + j] = (nv >> (8 * j)) & 0xff;
  }

  cdata_t out = {0};
  out.s = buf; out.n = d.n; out.unit = d.unit; out.compressed = 0; out.fmt = '2';
  cdata_write((char *)out_cm, &out, "w", 0);

  free(old2new); free(new_names);
  free_cdata(&out); free_cdata(&d);
}

void ms_matrix_write_tsv(const ms_matrix_t *m, FILE *out, int header) {
  if (header) {
    fputs("cell", out);
    for (int j = 0; j < m->n_patterns; ++j) fprintf(out, "\t%s", m->pattern_names[j]);
    fputc('\n', out);
  }
  for (int i = 0; i < m->n_cells; ++i) {
    fputs(m->cell_names[i], out);
    const double *row = m->M + (size_t)i * m->n_patterns;
    for (int j = 0; j < m->n_patterns; ++j) {
      if (isnan(row[j])) fputs("\tNA", out);
      else               fprintf(out, "\t%.6g", row[j]);
    }
    fputc('\n', out);
  }
}

/* ------------------------------------------------------------------ */
/* `methscope matrix [-o out.tsv] <query.cg> <ref.mrmp>`               */
/* ------------------------------------------------------------------ */
static int matrix_usage(void) {
  fprintf(stderr,
    "\n"
    "Usage:\n"
    "  methscope matrix [options] <query.cg> <ref.mrmp>\n"
    "\n"
    "Purpose:\n"
    "  Build the cell x pattern beta matrix -- the featurization step shared by\n"
    "  predict/deconv/train -- summarizing each query record against every pattern.\n"
    "\n"
    "Arguments:\n"
    "  <query.cg>   Query methylome(s); '-' reads a .cg stream from stdin\n"
    "               (cells are then named 1,2,3,... as a stream has no index).\n"
    "  <ref.mrmp>   MRMP pattern definition (a YAME .cm). A bundle (.ubjx/.updecx)\n"
    "               is also accepted here; its attached MRMP is used automatically.\n"
    "\n"
    "Options:\n"
    "  -o <out>       Write output to a file instead of stdout.\n"
    "  --refx         Bundle the matrix + this MRMP into a self-contained .refx\n"
    "                 deconvolution reference (requires -o <out.refx>); use it as\n"
    "                 `deconv mixture.cg out.refx`. Build it on per-cell-type\n"
    "                 pseudobulks so rows are cell types. The reference is imputed\n"
    "                 NaN-free: NA betas take the per-pattern median across cell\n"
    "                 types, and patterns >30%% NA across cell types are dropped.\n"
    "  --min-cov <k>  Blank (NA) any beta backed by < k covered CpGs (N_overlap),\n"
    "                 reporting only well-covered patterns. Default 1 (off). Mirrors\n"
    "                 deconv's --min-cov, but here it just NA-masks the output.\n"
    "  --no-header    Suppress the header line (plain TSV output only).\n"
    "  -h             Show this help message.\n"
    "\n"
    "Output:\n"
    "  TSV: first column 'cell', then one column per pattern (NA where no overlap);\n"
    "  or, with --refx, a .refx bundle (matrix + MRMP).\n"
    "\n");
  return 1;
}

static int dcmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

/* Median of vals[0..n) (sorts in place); n > 0. */
static double median_of(double *vals, int n) {
  qsort(vals, n, sizeof(double), dcmp);
  return (n & 1) ? vals[n/2] : 0.5 * (vals[n/2 - 1] + vals[n/2]);
}

/* Reference imputation for `matrix --refx`: per pattern column, drop it if more than
 * max_na_frac of the cell types are NA; otherwise fill NA cells with the column
 * median (the pattern's median beta across cell types). Rebuilds M / pattern_names /
 * n_patterns NaN-free; N is discarded (not used by --refx). */
static void impute_col_median(ms_matrix_t *m, double max_na_frac) {
  int nc = m->n_cells, np = m->n_patterns;
  int    *keep = malloc(np * sizeof(int));
  double *med  = malloc(np * sizeof(double));
  double *col  = malloc((nc ? nc : 1) * sizeof(double));
  if (!keep || !med || !col) mdie("out of memory (impute)", NULL);
  int nkeep = 0;
  for (int j = 0; j < np; ++j) {
    int nn = 0;
    for (int i = 0; i < nc; ++i) {
      double v = m->M[(size_t)i * np + j];
      if (!isnan(v)) col[nn++] = v;
    }
    double na_frac = nc ? 1.0 - (double)nn / nc : 1.0;
    keep[j] = (nn > 0) && (na_frac <= max_na_frac);
    if (keep[j]) { med[j] = median_of(col, nn); nkeep++; }
  }
  if (nkeep == 0) mdie("all reference patterns dropped (too sparse)", NULL);

  double *nM  = malloc((size_t)nc * nkeep * sizeof(double));
  char  **npn = malloc(nkeep * sizeof(char *));
  if (!nM || !npn) mdie("out of memory (impute compact)", NULL);
  int kc = 0;
  for (int j = 0; j < np; ++j) {
    if (!keep[j]) { free(m->pattern_names[j]); continue; }
    npn[kc] = m->pattern_names[j];
    for (int i = 0; i < nc; ++i) {
      double v = m->M[(size_t)i * np + j];
      nM[(size_t)i * nkeep + kc] = isnan(v) ? med[j] : v;
    }
    kc++;
  }
  free(m->M); free(m->pattern_names); free(m->N); free(keep); free(med); free(col);
  m->M = nM; m->pattern_names = npn; m->n_patterns = nkeep; m->N = NULL;
}

int main_matrix(int argc, char *argv[]) {
  const char *out_path = NULL;
  int no_header = 0, refx = 0, min_cov = 1;
  int i = 1;
  for (; i < argc; ++i) {
    if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--no-header") == 0) no_header = 1;
    else if (strcmp(argv[i], "--refx") == 0) refx = 1;
    else if (strcmp(argv[i], "--min-cov") == 0 && i + 1 < argc) min_cov = atoi(argv[++i]);
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      matrix_usage(); return 0;
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      mdie("unrecognized or incomplete option", argv[i]);
    else break;
  }
  if (argc - i != 2) return matrix_usage();
  const char *query_cg = argv[i];
  char *tmp_mrmp = NULL;
  const char *ref_mrmp = ms_mrmp_resolve(argv[i + 1], &tmp_mrmp);

  ms_matrix_t *m = ms_matrix_build(query_cg, ref_mrmp);

  /* --min-cov k: blank (NA) any beta backed by fewer than k covered CpGs, so the
   * matrix reports only well-supported patterns (N_overlap is computed by build). */
  if (min_cov > 1) {
    size_t tot = (size_t)m->n_cells * m->n_patterns;
    for (size_t k = 0; k < tot; ++k)
      if (m->N[k] < min_cov) m->M[k] = NAN;
  }

  if (refx) {
    /* Bundle the signature matrix + its MRMP into a .refx (kind=refx). The TSV
     * needs its header (pattern names) so deconv can match patterns by name. */
    if (!out_path) mdie("--refx requires -o <out.refx>", NULL);
    /* impute the reference NaN-free at build time: per-pattern median across cell
     * types; drop patterns >30% NA. deconv then does per-sample complete-case. */
    impute_col_median(m, 0.30);
    char tmpl[] = "/tmp/methscope_refx_XXXXXX.tsv";
    int fd = mkstemps(tmpl, 4);
    if (fd < 0) mdie("cannot create temp refx tsv", NULL);
    FILE *tf = fdopen(fd, "w");
    if (!tf) mdie("fdopen temp refx tsv", NULL);
    ms_matrix_write_tsv(m, tf, 1);           /* header always for a reference */
    fclose(tf);
    ms_bundle_pack(out_path, "refx", tmpl, ref_mrmp, NULL);
    unlink(tmpl);
    fprintf(stderr, "[methscope] wrote reference bundle (%d cells x %d patterns) -> %s\n",
            m->n_cells, m->n_patterns, out_path);
  } else {
    FILE *out = out_path ? fopen(out_path, "w") : stdout;
    if (!out) mdie("cannot open output", out_path);
    ms_matrix_write_tsv(m, out, !no_header);
    if (out != stdout) fclose(out);
  }

  ms_matrix_free(m);
  ms_mrmp_cleanup(tmp_mrmp);
  return 0;
}
