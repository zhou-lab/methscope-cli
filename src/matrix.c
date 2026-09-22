// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
/**
 * Build the cell x pattern beta matrix (the C replacement for the R
 * GenerateInput()). Each query record (cell/pixel) is scored against every
 * state of every mask record (MRMP pattern) in reference.cm in one pass over
 * the CpGs, via an inverted CpG -> column index; the per-(cell,pattern) mean
 * beta becomes a matrix entry, NaN where there is no overlap. The numbers are
 * those of YAME's summary core (summarize1), which remains the fallback for a
 * non-fmt3 query.
 *
 * Column order = by numeric pattern id (first run of digits in the state name),
 * with "Pna" always last (the R GenerateInput() order); the booster's matching
 * .ubj was trained on that same order.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
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
  /* Checked BEFORE the digit scan: a fused background is "Pna.<set>" and a set
   * name may carry digits ("Pna.bio_itl23"), which would otherwise be read as
   * pattern number 23 and sorted among the real patterns. */
  /* NULL sorts with the backgrounds. ms_is_pna_name() deliberately tolerates
   * NULL and returns 0, which used to hand a NULL straight to the scan below --
   * an unnamed column is a malformed artifact, not a reason to segfault. */
  if (!s || ms_is_pna_name(s)) return LONG_MAX;
  for (const char *p = s; *p; ++p)
    if (*p >= '0' && *p <= '9') return atol(p);
  /* Non-numeric names have no recurrence rank. Curated marker names (e.g.
   * "Xa_hi") sort after the numeric patterns but before the backgrounds,
   * keeping their definition order via the stable tie-break in cmp_colkey.
   * ("Pna" itself never reaches here -- it returned above.) */
  return LONG_MAX - 1;
}

typedef struct { long key; int set; int idx; } colkey_t;
static int cmp_colkey(const void *a, const void *b) {
  const colkey_t *x = a, *y = b;
  if (x->key < y->key) return -1;
  if (x->key > y->key) return 1;
  return x->idx - y->idx;            /* stable on ties */
}

/* SET-MAJOR: every column of set k before any column of set k+1, and within a
 * set by pattern rank. This is the layout ms_mrmp_group_map_chain() produces at
 * featurize time, so an upscaler trained on a chain must be fed this order.
 *
 * It is NOT the default, because classify-train and classify both call
 * ms_matrix_build() and then ms_matrix_select() with indices resolved against
 * whatever order it returned -- every shipped .clfx stores those indices, so
 * changing the default order would silently repoint them. */
static int cmp_colkey_setmajor(const void *a, const void *b) {
  const colkey_t *x = a, *y = b;
  if (x->set != y->set) return x->set - y->set;
  if (x->key < y->key) return -1;
  if (x->key > y->key) return 1;
  return x->idx - y->idx;
}

/*
 * The MRMP reference (.cm) is a categorical (fmt2) track whose *states* are the
 * patterns (P1..Pn, Pna). summarize1() therefore returns one stats_t per state
 * (st[j].sm = pattern name, st[j].beta = the cell's mean methylation over that
 * pattern's CpGs). We gather every state into a column, then order columns by
 * numeric pattern id exactly as the R GenerateInput() did, so the first
 * npattern columns are the model's features and "Pna" lands last.
 */
void ms_matrix_select(ms_matrix_t *m, const int *idx, int n) {
  if (n <= 0 || n > m->n_patterns) mdie("bad column selection", NULL);
  /* Forward pass is safe in place: the destination index r*n + c never exceeds
   * the source r*n_patterns + idx[c], because n <= n_patterns and idx is
   * ascending, so the write cursor always trails the read cursor. */
  const int old = m->n_patterns;
  for (int r = 0; r < m->n_cells; ++r) {
    const size_t src = (size_t)r * old, dst = (size_t)r * n;
    for (int c = 0; c < n; ++c) {
      m->M[dst + c] = m->M[src + idx[c]];
      m->N[dst + c] = m->N[src + idx[c]];
    }
  }
  char **kept = malloc((size_t)n * sizeof(char *));
  if (!kept) mdie("out of memory (column selection)", NULL);
  char *taken = calloc((size_t)old, 1);
  if (!taken) mdie("out of memory (column selection)", NULL);
  for (int c = 0; c < n; ++c) { kept[c] = m->pattern_names[idx[c]]; taken[idx[c]] = 1; }
  for (int c = 0; c < old; ++c) if (!taken[c]) free(m->pattern_names[c]);
  free(taken);
  free(m->pattern_names);
  m->pattern_names = kept;
  m->n_patterns = n;
}

static ms_matrix_t *matrix_build(const char *query_cg, const char *ref_cm,
                                 int set_major) {
  config_t config = {0};
  config.fname_mask = (char *)ref_cm;

  /* ---- load all mask records into memory, once ---- */
  cfile_t  cf_mask     = open_cfile((char *)ref_cm);
  snames_t snames_mask = loadSampleNamesFromIndex((char *)ref_cm);
  /* ref_cm may be a BUNDLE (.updecx / .ubjx), whose .cm is only the file
     prefix. Bound the walk at the MSBNDL1 container or it reads into the
     trailer, which YAME has treated as a fatal since v1.40 -- the v0.7
     `upscale` regression. -1 for a plain .cm, i.e. read to end of stream. */
  const int64_t cx_limit = ms_bundle_cx_limit(ref_cm);
  cdata_t *c_masks = NULL;
  size_t   n_masks = 0, mcap = 0;
  for (;;) {
    cdata_t cm = {0};
    char rerr[CX_ERRBUF];
    cx_read_t st = cx_read_record(&cf_mask, &cm, cx_limit, rerr, sizeof rerr);
    if (st == CX_READ_END) break;
    if (st != CX_READ_OK) mdie(rerr, ref_cm);
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
  if (!mask_names) mdie("out of memory (mask names)", NULL);
  for (size_t k = 0; k < n_masks; ++k) {
    char b[32];
    if (snames_mask.n > (int)k) mask_names[k] = strdup(snames_mask.s[k]);
    else { snprintf(b, sizeof(b), "%zu", k + 1); mask_names[k] = strdup(b); }
  }

  /* ---- raw column layout, from the masks' state tables ----
   * One column per (mask record, state) in state-table order, background
   * dropped: exactly the order summarize1() returns its stats_t[] in, so the
   * layout is fixed before a cell is read instead of discovered on cell 0.
   * classify-featurize emits no background either; emitting one here would
   * make the two feature paths different widths -- the divergence that scored
   * 2 of 42 cells correct. */
  size_t   n_raw = 0;
  char   **raw_names = NULL;          /* length n_raw, summarize order           */
  int     *raw_set = NULL;            /* which mask record each column came from */
  int    **key2col = malloc(n_masks * sizeof(int *));   /* state -> raw col, -1 = Pna */
  int      fast = 1;                  /* every mask a fmt2 state track           */
  if (!key2col) mdie("out of memory (state map)", NULL);
  for (size_t k = 0; k < n_masks; ++k) {
    if (c_masks[k].fmt != '2') { fast = 0; key2col[k] = NULL; continue; }
    if (!c_masks[k].aux) fmt2_set_aux(&c_masks[k]);
    f2_aux_t *aux = (f2_aux_t *)c_masks[k].aux;
    key2col[k] = malloc((aux->nk ? aux->nk : 1) * sizeof(int));
    if (!key2col[k]) mdie("out of memory (state map)", NULL);
    for (uint64_t j = 0; j < aux->nk; ++j) {
      const char *nm = aux->keys[j];
      if (ms_is_pna_name(nm)) { key2col[k][j] = -1; continue; }
      raw_names = realloc(raw_names, (n_raw + 1) * sizeof(char *));
      raw_set   = realloc(raw_set,   (n_raw + 1) * sizeof(int));
      if (!raw_names || !raw_set) mdie("out of memory (columns)", NULL);
      raw_names[n_raw] = strdup(nm ? nm : "");
      if (!raw_names[n_raw]) mdie("out of memory (column name)", NULL);
      raw_set[n_raw] = (int)k;
      key2col[k][j] = (int)n_raw++;
    }
  }
  if (fast && n_raw == 0) mdie("reference has no pattern states", ref_cm);

  /* ---- inverted index over CpGs: the columns each CpG belongs to ----
   * summarize1() walks the whole genome once per (cell, mask), so a 124-set
   * chain cost 124 genome scans per cell. Almost every CpG is background in
   * almost every set, so list only the real memberships per CpG and each cell
   * costs one scan plus one add per membership. Beta is still the mean of
   * per-CpG betas over covered CpGs, accumulated in CpG order, so the sums are
   * bit-identical to summarize1's. Only a fmt3 (M/U) query takes this path;
   * anything else keeps the per-mask summary below. */
  uint64_t n_cpg = c_masks[0].n;
  uint32_t *cpg_off = NULL, *cpg_col = NULL;
  if (fast) {
    for (size_t k = 1; k < n_masks; ++k)
      if (c_masks[k].n != n_cpg) mdie("mask records span different CpG counts", ref_cm);
    cpg_off = calloc(n_cpg + 1, sizeof(uint32_t));
    if (!cpg_off) mdie("out of memory (CpG offsets)", NULL);
    for (size_t k = 0; k < n_masks; ++k) {
      f2_aux_t *aux = (f2_aux_t *)c_masks[k].aux;
      for (uint64_t i = 0; i < n_cpg; ++i) {
        uint64_t s = f2_get_uint64(&c_masks[k], i);
        if (s >= aux->nk) mdie("mask state data is corrupted", mask_names[k]);
        if (key2col[k][s] >= 0) ++cpg_off[i + 1];
      }
    }
    uint64_t acc = 0;
    for (uint64_t i = 0; i <= n_cpg; ++i) { acc += cpg_off[i]; cpg_off[i] = (uint32_t)acc; }
    if (acc > UINT32_MAX) mdie("too many pattern memberships to index", ref_cm);
    cpg_col = malloc((acc ? acc : 1) * sizeof(uint32_t));
    uint32_t *cur = calloc(n_cpg, sizeof(uint32_t));
    if (!cpg_col || !cur) mdie("out of memory (CpG memberships)", NULL);
    for (size_t k = 0; k < n_masks; ++k)
      for (uint64_t i = 0; i < n_cpg; ++i) {
        int col = key2col[k][f2_get_uint64(&c_masks[k], i)];
        if (col >= 0) cpg_col[cpg_off[i] + cur[i]++] = (uint32_t)col;
      }
    free(cur);
  }

  /* ---- stream query cells ---- */
  cfile_t  cf_qry     = open_cfile((char *)query_cg);
  snames_t snames_qry = loadSampleNamesFromIndex((char *)query_cg);

  double  *raw_row  = malloc(n_raw * sizeof(double));   /* scratch row          */
  int     *raw_Ncnt = malloc(n_raw * sizeof(int));      /* scratch N_overlap row */
  double  *sum      = fast ? malloc(n_raw * sizeof(double)) : NULL;
  uint64_t *cnt     = fast ? malloc(n_raw * sizeof(uint64_t)) : NULL;
  double  *Mraw = NULL;               /* n_cells x n_raw, raw column order        */
  int     *Nraw = NULL;               /* n_cells x n_raw N_overlap, raw order     */
  char   **cell_names = NULL;
  size_t   n_cells = 0, rcap = 0;
  if (!raw_row || !raw_Ncnt || (fast && (!sum || !cnt)))
    mdie("out of memory (row buffers)", NULL);

  for (size_t iq = 0;; ++iq) {
    cdata_t cq = read_cdata1(&cf_qry);
    if (cq.n == 0) break;
    prepare_mask(&cq);

    if (fast && cq.fmt == '3') {
      if (cq.n != n_cpg) mdie("query and mask CpG counts differ", query_cg);
      memset(sum, 0, n_raw * sizeof(double));
      memset(cnt, 0, n_raw * sizeof(uint64_t));
      for (uint64_t i = 0; i < n_cpg; ++i) {
        uint32_t e = cpg_off[i], e1 = cpg_off[i + 1];
        if (e == e1) continue;
        uint64_t mu = f3_get_mu(&cq, i);
        if (!mu) continue;
        double b = MU2beta(mu);
        for (; e < e1; ++e) { sum[cpg_col[e]] += b; ++cnt[cpg_col[e]]; }
      }
      for (size_t c = 0; c < n_raw; ++c) {
        raw_row[c]  = cnt[c] ? sum[c] / (double)cnt[c] : NAN;
        raw_Ncnt[c] = (cnt[c] > (uint64_t)INT_MAX) ? INT_MAX : (int)cnt[c];
      }
    } else {
      /* summarize1() takes char*, not const char*, so a string literal here is
       * only safe as long as it never writes. Hand it a writable empty buffer
       * instead of betting on that. */
      char no_query_name[] = "";
      size_t col = 0;
      for (size_t k = 0; k < n_masks; ++k) {
        uint64_t n_st = 0;
        stats_t *st = summarize1(&cq, &c_masks[k], &n_st, mask_names[k],
                                 no_query_name, &config);
        for (uint64_t j = 0; j < n_st; ++j) {
          if (ms_is_pna_name(st[j].sm)) continue;
          if (col >= n_raw || strcmp(st[j].sm ? st[j].sm : "", raw_names[col]))
            mdie("summary states do not match the mask state table", mask_names[k]);
          raw_row[col]  = (st[j].beta >= 0) ? st[j].beta : NAN;
          raw_Ncnt[col] = (st[j].n_o > (uint64_t)INT_MAX) ? INT_MAX : (int)st[j].n_o;
          col++;
        }
        for (uint64_t j = 0; j < n_st; ++j) { free(st[j].sm); free(st[j].sq); }
        if (n_st) free(st);
      }
      if (col != n_raw) mdie("inconsistent pattern count across cells", NULL);
    }

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
  for (size_t k = 0; k < n_masks; ++k) {
    free_cdata(&c_masks[k]); free(mask_names[k]); free(key2col[k]);
  }
  free(c_masks); free(mask_names); free(key2col);
  free(cpg_off); free(cpg_col); free(sum); free(cnt);
  cleanSampleNames2(snames_mask);
  cleanSampleNames2(snames_qry);
  free(raw_row); free(raw_Ncnt);   /* raw_set is read by the sort below */
  if (n_cells == 0 || n_raw == 0) mdie("no data produced", query_cg);

  /* ---- order columns by numeric pattern id (R parity) ---- */
  colkey_t *ck = malloc(n_raw * sizeof(colkey_t));
  if (!ck) mdie("out of memory (colkey)", NULL);
  for (size_t i = 0; i < n_raw; ++i) {
    ck[i].key = pattern_numeric_key(raw_names[i]);
    ck[i].set = raw_set ? raw_set[i] : 0;
    ck[i].idx = (int)i;
  }
  qsort(ck, n_raw, sizeof(colkey_t),
        set_major ? cmp_colkey_setmajor : cmp_colkey);
  free(raw_set);

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
 * already among keep_names) states. Used by `classify-train` to bundle only the mrmp the
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

/* ------------------------------------------------------------------ */

/* The default order: numeric pattern id across every record, which is what the
 * classifier's stored feature indices were resolved against. */
ms_matrix_t *ms_matrix_build(const char *query_cg, const char *ref_cm) {
  return matrix_build(query_cg, ref_cm, 0);
}

/* Set-major, for a chain-featurized upscaler -- see cmp_colkey_setmajor(). */
ms_matrix_t *ms_matrix_build_sets(const char *query_cg, const char *ref_cm) {
  return matrix_build(query_cg, ref_cm, 1);
}
