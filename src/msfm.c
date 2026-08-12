// SPDX-License-Identifier: AGPL-3.0-or-later
/* `classify-featurize` -- build a MSFMAT1 record x pattern matrix once, so
 * training and scoring never re-run the featurization.
 *
 * `classify-train` and `classify` both call ms_matrix_build() internally and
 * single-threaded, so every arm and every baseline re-featurizes the same
 * cells. Splitting featurization out is what the upscale side already does
 * (`upscale-featurize` / `upscale-train`); this is that split for label
 * models. The win is on the scoring side: featurize a test set once, then
 * score every model against it for free.
 *
 * ALL patterns are stored, not a -p prefix, so one artifact serves any
 * `--patterns N` at training time. With a `mrmp-export --top 1000` mask that
 * is 1001 columns -- 2 KB per record.
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "methscope.h"
#include "msfm.h"
#include "mrmp.h"   /* a .mrmp chain is a valid mask argument */
#include "bundle.h"
#include "index.h"   /* get_fname_index -- the fast path needs the .cg index */

/* noreturn so the compiler knows an open failure never falls through --
 * otherwise it cannot see that the input array is fully initialized. */
static void fdie(const char *msg, const char *det) __attribute__((noreturn));
static void fdie(const char *msg, const char *det) {
  fprintf(stderr, "[methscope] classify-featurize: %s%s%s\n", msg,
          det ? ": " : "", det ? det : "");
  exit(1);
}

static void *xmal(size_t n, const char *what) {
  void *p = malloc(n ? n : 1);
  if (!p) fdie("out of memory", what);
  return p;
}

static void wr(FILE *f, const void *p, size_t n, const char *path) {
  if (n && fwrite(p, 1, n, f) != n) fdie("short write", path);
}

/* ---- reading ----------------------------------------------------------- */

/* Split a NUL-separated blob into `n` pointers (into the mapping itself). */
static char **blob_index(const char *base, uint64_t off, uint32_t n,
                         uint64_t limit, char *err, size_t errn) {
  char **v = malloc((size_t)(n ? n : 1) * sizeof(*v));
  if (!v) return NULL;
  const char *p = base + off, *end = base + limit;
  for (uint32_t i = 0; i < n; ++i) {
    if (p >= end) { snprintf(err, errn, "name table overruns the file"); free(v); return NULL; }
    v[i] = (char *)p;
    p += strlen(p) + 1;
  }
  return v;
}

int ms_msfm_open(ms_msfm_t *f, const char *path, char *err, size_t errn) {
  memset(f, 0, sizeof(*f));
  int fd = open(path, O_RDONLY);
  if (fd < 0) { snprintf(err, errn, "cannot open %s", path); return 0; }
  struct stat st;
  if (fstat(fd, &st) || (uint64_t)st.st_size < sizeof(msfm_header_t)) {
    snprintf(err, errn, "truncated msfm"); close(fd); return 0;
  }
  void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { snprintf(err, errn, "cannot mmap"); close(fd); return 0; }
  const msfm_header_t *h = (const msfm_header_t *)map;
  uint64_t len = (uint64_t)st.st_size;
  if (!msfm_is(h) || h->file_bytes != len) {
    snprintf(err, errn, "not a MSFMAT1 artifact (or size mismatch)");
    munmap(map, (size_t)len); close(fd); return 0;
  }
  uint64_t nr = h->n_records, np = h->n_patterns;
  uint64_t beta_bytes = nr * np * 2;
  if (h->beta_offset + beta_bytes > len ||
      h->levels_offset + nr * 4 > len ||
      h->labels_offset + nr * 2 > len) {
    snprintf(err, errn, "truncated msfm payload");
    munmap(map, (size_t)len); close(fd); return 0;
  }
  f->fd = fd; f->map = map; f->length = len; f->header = h;
  f->beta   = (const uint16_t *)((const char *)map + h->beta_offset);
  f->levels = (const uint32_t *)((const char *)map + h->levels_offset);
  f->class_id = (const uint16_t *)((const char *)map + h->labels_offset);
  f->pattern_names = blob_index((const char *)map, h->names_offset, h->n_patterns, len, err, errn);
  f->record_names  = blob_index((const char *)map, h->rows_offset, h->n_records, len, err, errn);
  f->class_names   = blob_index((const char *)map, h->labels_offset + nr * 2,
                                h->n_classes, len, err, errn);
  if (!f->pattern_names || !f->record_names || !f->class_names) {
    ms_msfm_close(f); return 0;
  }
  return 1;
}

/* thousands separators, as mrmp.c's commafmt_local does; buf >= 32 bytes */
static const char *commafmt_msfm(uint64_t v, char *buf) {
  char tmp[24]; int n = snprintf(tmp, sizeof tmp, "%" PRIu64, v);
  int o = 0;
  for (int i = 0; i < n; ++i) {
    if (i && (n - i) % 3 == 0) buf[o++] = ',';
    buf[o++] = tmp[i];
  }
  buf[o] = '\0';
  return buf;
}

void ms_msfm_close(ms_msfm_t *f) {
  free(f->pattern_names); free(f->record_names); free(f->class_names);
  if (f->map) munmap(f->map, (size_t)f->length);
  if (f->fd > 0) close(f->fd);
  memset(f, 0, sizeof(*f));
}

uint32_t ms_msfm_flags(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) fdie("cannot open", path);
  msfm_header_t h;
  if (fread(&h, 1, sizeof h, f) != sizeof h) { fclose(f); fdie("short header", path); }
  fclose(f);
  if (!msfm_is(&h)) fdie("not an MSFMAT1 artifact", path);
  return h.flags;
}

ms_matrix_t *ms_msfm_to_matrix(const char *path, char ***labels_out,
                               uint32_t **levels_out) {
  ms_msfm_t f; char err[256];
  if (!ms_msfm_open(&f, path, err, sizeof(err))) fdie(err, path);
  const msfm_header_t *h = f.header;
  uint32_t nr = h->n_records, np = h->n_patterns;

  ms_matrix_t *m = xmal(sizeof(*m), "matrix");
  memset(m, 0, sizeof(*m));
  m->n_cells = (int)nr; m->n_patterns = (int)np;
  m->cell_names    = xmal((size_t)nr * sizeof(char *), "cell names");
  m->pattern_names = xmal((size_t)np * sizeof(char *), "pattern names");
  for (uint32_t r = 0; r < nr; ++r) m->cell_names[r] = strdup(f.record_names[r]);
  for (uint32_t c = 0; c < np; ++c) m->pattern_names[c] = strdup(f.pattern_names[c]);
  m->M = xmal((size_t)nr * np * sizeof(double), "betas");
  m->N = xmal((size_t)nr * np * sizeof(int), "counts");
  for (uint64_t k = 0; k < (uint64_t)nr * np; ++k) {
    uint16_t v = f.beta[k];
    m->M[k] = msfm_decode(v);
    /* Without a stored count block, N is only ever asked "is this observed?",
     * so a 1/0 indicator is faithful for that use and the per-record coverage
     * total is carried separately and exactly. */
    m->N[k] = (v == MSFM_NA ? 0 : 1);
  }
  if (labels_out) {
    char **lab = NULL;
    if (h->n_classes) {
      lab = xmal((size_t)nr * sizeof(char *), "labels");
      for (uint32_t r = 0; r < nr; ++r) {
        if (f.class_id[r] >= h->n_classes) fdie("class id out of range", path);
        lab[r] = strdup(f.class_names[f.class_id[r]]);
      }
    }
    *labels_out = lab;
  }
  if (levels_out) {
    uint32_t *lv = xmal((size_t)nr * 4, "levels");
    memcpy(lv, f.levels, (size_t)nr * 4);
    *levels_out = lv;
  }
  ms_msfm_close(&f);
  return m;
}

void ms_msfm_report(const char *path) {
  ms_msfm_t f; char err[256];
  if (!ms_msfm_open(&f, path, err, sizeof(err))) {
    fprintf(stderr, "[methscope] inspect: %s\n", err); exit(1);
  }
  const msfm_header_t *h = f.header;
  /* Same shape as `inspect` on a .mrmp: banner, aligned keys, thousands
   * separators, blank-line groups. */
  char cb[32], cb2[32], cb3[32];
  printf("\nMSFM  %s\n", path);
  { const char *sat = (h->flags & MSFM_FLAG_RANK_ONLY)     ? ", per-class rank only"
                    : (h->flags & MSFM_FLAG_RANK_ADD)      ? ", per-class rank added"
                    : (h->flags & MSFM_FLAG_CONTRAST_ONLY) ? ", satellite contrast only"
                    : (h->flags & MSFM_FLAG_CONTRAST_ADD)  ? ", satellite contrast added" : "";
    const char *coding =
      (h->flags & MSFM_FLAG_BIN_FLAT) ? "patterns binarised at 0.5"
    : (h->flags & MSFM_FLAG_BIN_PAT)  ? "patterns cut at per-pattern midpoints"
                                      : "continuous pattern betas";
    printf("  %-14s MSFMAT1 v%u, %s%s\n", "format", h->version, coding, sat); }
  printf("\n");
  printf("  %-14s %s\n", "records", commafmt_msfm(h->n_records, cb));
  printf("  %-14s %s\n", "patterns", commafmt_msfm(h->n_patterns, cb));
  printf("  %-14s %s\n", "classes", commafmt_msfm(h->n_classes, cb));
  if (h->n_classes) {
    printf("  %-14s ", "labels");
    for (uint32_t k = 0; k < h->n_classes && k < 8; ++k)
      printf("%s%s", k ? ", " : "", f.class_names[k]);
    if (h->n_classes > 8) printf(", ... (%u total)", h->n_classes);
    printf("\n");
  }
  printf("\n");
  printf("  %-14s u16 fixed point (code/65534; 65535 = NA)\n", "beta");
  uint64_t lo = UINT64_MAX, hi = 0, sum = 0;
  for (uint32_t r = 0; r < h->n_records; ++r) {
    uint32_t v = f.levels[r];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
  }
  if (h->n_records)
    printf("  %-14s min %s   mean %s   max %s\n", "covered CpGs",
           commafmt_msfm(lo, cb), commafmt_msfm(sum / h->n_records, cb2),
           commafmt_msfm(hi, cb3));
  printf("  %-14s %s bytes\n\n", "on disk", commafmt_msfm(h->file_bytes, cb));
  ms_msfm_close(&f);
}

/* ---- writing ----------------------------------------------------------- */

static int cmp_str(const void *a, const void *b) {
  return strcmp(*(char *const *)a, *(char *const *)b);
}

/* One label per record, in query order (the same contract as classify-train -l). */
static char **read_labels(const char *path, int expect) {
  FILE *fp = fopen(path, "r");
  if (!fp) fdie("cannot open labels", path);
  char **v = xmal((size_t)expect * sizeof(*v), "labels");
  int n = 0;
  char *line = NULL; size_t cap = 0; ssize_t len;
  while ((len = getline(&line, &cap, fp)) != -1) {
    while (len && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
    if (!len) continue;
    if (n == expect) fdie("more labels than query records", path);
    v[n++] = strdup(line);
  }
  free(line); fclose(fp);
  if (n != expect) fdie("label count does not match query records", path);
  return v;
}

static void write_msfm(const char *out, const ms_matrix_t *m, char **lab) {
  uint32_t nr = (uint32_t)m->n_cells, np = (uint32_t)m->n_patterns;

  /* class table = sorted unique labels, so the stored id IS the class index a
   * model will use (train.c derives the same order independently). */
  uint32_t nk = 0;
  char **uniq = NULL;
  uint16_t *cid = xmal((size_t)nr * 2, "class ids");
  memset(cid, 0, (size_t)nr * 2);
  if (lab) {
    uniq = xmal((size_t)nr * sizeof(*uniq), "class set");
    for (uint32_t r = 0; r < nr; ++r) uniq[r] = lab[r];
    qsort(uniq, nr, sizeof(*uniq), cmp_str);
    for (uint32_t r = 0; r < nr; ++r)
      if (!r || strcmp(uniq[r], uniq[nk-1]) != 0) uniq[nk++] = uniq[r];
    if (nk > UINT16_MAX) fdie("too many classes", NULL);
    for (uint32_t r = 0; r < nr; ++r) {
      int lo = 0, hi = (int)nk - 1, at = -1;
      while (lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(lab[r], uniq[mid]);
        if (!c) { at = mid; break; } else if (c < 0) hi = mid - 1; else lo = mid + 1;
      }
      if (at < 0) fdie("label not in class set (internal)", lab[r]);
      cid[r] = (uint16_t)at;
    }
  }

  /* section sizes */
  uint64_t names_b = 0, rows_b = 0, class_b = 0;
  for (uint32_t c = 0; c < np; ++c) names_b += strlen(m->pattern_names[c]) + 1;
  for (uint32_t r = 0; r < nr; ++r) rows_b += strlen(m->cell_names[r]) + 1;
  for (uint32_t k = 0; k < nk; ++k) class_b += strlen(uniq[k]) + 1;

  msfm_header_t h;
  memset(&h, 0, sizeof(h));
  memcpy(h.magic, MSFM_MAGIC, 7);
  h.version = 1; h.n_records = nr; h.n_patterns = np; h.n_classes = nk;
  h.flags = 0;
  h.names_offset  = sizeof(h);
  h.rows_offset   = h.names_offset + names_b;
  h.labels_offset = h.rows_offset + rows_b;
  h.levels_offset = h.labels_offset + (uint64_t)nr * 2 + class_b;
  h.beta_offset   = h.levels_offset + (uint64_t)nr * 4;
  h.count_offset  = 0;                  /* reserved; see msfm.h */
  h.file_bytes    = h.beta_offset + (uint64_t)nr * np * 2;

  FILE *f = fopen(out, "wb");
  if (!f) fdie("cannot create output", out);
  wr(f, &h, sizeof(h), out);
  for (uint32_t c = 0; c < np; ++c) wr(f, m->pattern_names[c], strlen(m->pattern_names[c]) + 1, out);
  for (uint32_t r = 0; r < nr; ++r) wr(f, m->cell_names[r], strlen(m->cell_names[r]) + 1, out);
  wr(f, cid, (size_t)nr * 2, out);
  for (uint32_t k = 0; k < nk; ++k) wr(f, uniq[k], strlen(uniq[k]) + 1, out);

  /* levels: covered CpGs summed over EVERY pattern including Pna, i.e. the
   * record's true coverage. This is the raw input for the scalar coverage
   * feature; log1p and any scaling belong to the model, not the artifact, so
   * one test artifact serves every arm. */
  uint32_t *lev = xmal((size_t)nr * 4, "levels");
  for (uint32_t r = 0; r < nr; ++r) {
    uint64_t t = 0;
    const int *n = m->N + (size_t)r * np;
    for (uint32_t c = 0; c < np; ++c) if (n[c] > 0) t += (uint64_t)n[c];
    lev[r] = t > UINT32_MAX ? UINT32_MAX : (uint32_t)t;
  }
  wr(f, lev, (size_t)nr * 4, out);
  free(lev);

  uint16_t *row = xmal((size_t)np * 2, "beta row");
  for (uint32_t r = 0; r < nr; ++r) {
    const double *src = m->M + (size_t)r * np;
    for (uint32_t c = 0; c < np; ++c) row[c] = msfm_encode(src[c]);
    wr(f, row, (size_t)np * 2, out);
  }
  free(row);
  if (fclose(f)) fdie("cannot finalize output", out);

  fprintf(stderr, "[methscope] classify-featurize: %u records x %u patterns"
          ", %u classes -> %s (%.1f MB)\n", nr, np, nk, out,
          (double)h.file_bytes / 1048576.0);
  free(cid); free(uniq);
}

/* Write from raw arrays (the sampled path), where rows are (replicate, cell)
 * and there is no ms_matrix_t. Record names get the replicate tag appended so
 * every row is uniquely identifiable. */
static void write_msfm_raw(const char *out, const uint16_t *beta,
                           const uint32_t *levels, char *const *cell_names,
                           uint32_t n_cells, uint32_t n_reps, uint32_t ncol,
                           char *const *lab, const uint32_t *rep_sample,
                           uint32_t n_sets, const uint32_t *col0,
                           char *const *set_names, uint32_t flags) {
  uint64_t nr = (uint64_t)n_reps * n_cells;
  if (nr > UINT32_MAX) fdie("too many records", NULL);

  /* class table = sorted unique labels, matching write_msfm() */
  uint32_t nk = 0; char **uniq = NULL;
  uint16_t *cid = xmal(nr * 2, "class ids");
  memset(cid, 0, (size_t)nr * 2);
  if (lab) {
    uniq = xmal((size_t)n_cells * sizeof(*uniq), "class set");
    for (uint32_t r = 0; r < n_cells; ++r) uniq[r] = lab[r];
    qsort(uniq, n_cells, sizeof(*uniq), cmp_str);
    for (uint32_t r = 0; r < n_cells; ++r)
      if (!r || strcmp(uniq[r], uniq[nk-1]) != 0) uniq[nk++] = uniq[r];
    for (uint64_t row = 0; row < nr; ++row) {
      const char *want = lab[row % n_cells];        /* replicate-major rows */
      int lo = 0, hi = (int)nk - 1, at = -1;
      while (lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(want, uniq[mid]);
        if (!c) { at = mid; break; } else if (c < 0) hi = mid - 1; else lo = mid + 1;
      }
      if (at < 0) fdie("label not in class set (internal)", want);
      cid[row] = (uint16_t)at;
    }
  }

  /* Pattern names: one running P1..PN across all sets, in set order.
   *
   * No background columns any more. They used to be emitted one per set, named
   * "Pna.<set>" so a consumer could drop them again -- which made "column" and
   * "pattern" different units and put the backgrounds BETWEEN pattern blocks,
   * the layout that let a positional feature cut swallow backgrounds and drop
   * real patterns off the end. The matrix is now exactly its patterns, so
   * ncol == mrmp-pool's --pooled-top and there is nothing to exclude. */
  const size_t PN = 64;
  char (*pn)[64] = xmal((size_t)ncol * PN, "pattern names");
  for (uint32_t g = 0; g < ncol; ++g) snprintf(pn[g], PN, "P%u", g + 1);
  (void)col0; (void)set_names; (void)n_sets;

  uint64_t names_b = 0, rows_b = 0, class_b = 0;
  for (uint32_t g = 0; g < ncol; ++g) names_b += strlen(pn[g]) + 1;
  for (uint32_t rep = 0; rep < n_reps; ++rep)
    for (uint32_t c = 0; c < n_cells; ++c) {
      char nm[512];
      snprintf(nm, sizeof(nm), "%s-R%u", cell_names[c], rep + 1);
      rows_b += strlen(nm) + 1;
    }
  for (uint32_t k = 0; k < nk; ++k) class_b += strlen(uniq[k]) + 1;

  msfm_header_t h; memset(&h, 0, sizeof(h));
  memcpy(h.magic, MSFM_MAGIC, 7);
  h.version = 1; h.n_records = (uint32_t)nr; h.n_patterns = ncol; h.n_classes = nk;
  h.flags = flags;
  h.names_offset  = sizeof(h);
  h.rows_offset   = h.names_offset + names_b;
  h.labels_offset = h.rows_offset + rows_b;
  h.levels_offset = h.labels_offset + nr * 2 + class_b;
  h.beta_offset   = h.levels_offset + nr * 4;
  h.count_offset  = 0;
  h.file_bytes    = h.beta_offset + nr * ncol * 2;

  FILE *f = fopen(out, "wb");
  if (!f) fdie("cannot create output", out);
  wr(f, &h, sizeof(h), out);
  for (uint32_t g = 0; g < ncol; ++g) wr(f, pn[g], strlen(pn[g]) + 1, out);
  for (uint32_t rep = 0; rep < n_reps; ++rep)
    for (uint32_t c = 0; c < n_cells; ++c) {
      char nm[512];
      snprintf(nm, sizeof(nm), "%s-R%u", cell_names[c], rep + 1);
      wr(f, nm, strlen(nm) + 1, out);
    }
  wr(f, cid, (size_t)nr * 2, out);
  for (uint32_t k = 0; k < nk; ++k) wr(f, uniq[k], strlen(uniq[k]) + 1, out);
  wr(f, levels, (size_t)nr * 4, out);
  wr(f, beta, (size_t)nr * ncol * 2, out);
  if (fclose(f)) fdie("cannot finalize output", out);

  if (isatty(STDERR_FILENO)) fprintf(stderr, "\r\033[K");
  { char c1[32], c2[32];
    fprintf(stderr, "  %-14s %s records (%u cells x %u draw(s)) x %s "
            "patterns, %u classes\n", "output",
            commafmt_msfm(nr, c1), n_cells, n_reps, commafmt_msfm(ncol, c2), nk);
    fprintf(stderr, "  %-14s %s (%.1f MB)\n\n", "wrote", out,
            (double)h.file_bytes / 1048576.0); }
  (void)rep_sample;
  free(cid); free(uniq); free(pn);
}

/* ---- merge ------------------------------------------------------------- */

/* Chunks cannot be byte-concatenated -- the header carries absolute offsets --
 * so merging re-lays the sections out, exactly as `yame index -s` has to run
 * after `cat`. Pattern vocabularies must match; class tables are unioned and
 * ids remapped, so chunks that happen to miss a rare class still merge. */
static int merge_msfm(const char *out, char **in, int n_in) {
  ms_msfm_t *f = xmal((size_t)n_in * sizeof(*f), "inputs");
  memset(f, 0, (size_t)n_in * sizeof(*f));
  char err[256];
  uint64_t nr_tot = 0;
  for (int i = 0; i < n_in; ++i)
    if (!ms_msfm_open(&f[i], in[i], err, sizeof(err))) fdie(err, in[i]);
  /* Validate only once every input is open, so chunk 0 is unambiguously the
   * reference. Mismatched pattern vocabularies mean different MRMPs, which
   * would silently merge into a matrix whose columns do not line up. */
  for (int i = 0; i < n_in; ++i) {
    if (f[i].header->n_patterns != f[0].header->n_patterns)
      fdie("chunks disagree on pattern count", in[i]);
    for (uint32_t c = 0; c < f[0].header->n_patterns; ++c)
      if (strcmp(f[i].pattern_names[c], f[0].pattern_names[c]))
        fdie("chunks disagree on pattern names (different MRMP?)", in[i]);
    nr_tot += f[i].header->n_records;
  }
  if (nr_tot > UINT32_MAX) fdie("too many records", NULL);

  uint32_t np = f[0].header->n_patterns;

  /* union the class tables */
  uint32_t cap = 0;
  for (int i = 0; i < n_in; ++i) cap += f[i].header->n_classes;
  char **all = xmal((size_t)(cap ? cap : 1) * sizeof(*all), "class union");
  uint32_t na = 0;
  for (int i = 0; i < n_in; ++i)
    for (uint32_t k = 0; k < f[i].header->n_classes; ++k) all[na++] = f[i].class_names[k];
  qsort(all, na, sizeof(*all), cmp_str);
  uint32_t nk = 0;
  for (uint32_t k = 0; k < na; ++k)
    if (!k || strcmp(all[k], all[nk-1])) all[nk++] = all[k];

  uint64_t names_b = 0, rows_b = 0, class_b = 0;
  for (uint32_t c = 0; c < np; ++c) names_b += strlen(f[0].pattern_names[c]) + 1;
  for (int i = 0; i < n_in; ++i)
    for (uint32_t r = 0; r < f[i].header->n_records; ++r)
      rows_b += strlen(f[i].record_names[r]) + 1;
  for (uint32_t k = 0; k < nk; ++k) class_b += strlen(all[k]) + 1;

  msfm_header_t h;
  memset(&h, 0, sizeof(h));
  memcpy(h.magic, MSFM_MAGIC, 7);
  h.version = 1; h.n_records = (uint32_t)nr_tot; h.n_patterns = np; h.n_classes = nk;
  h.flags = 0;
  h.names_offset  = sizeof(h);
  h.rows_offset   = h.names_offset + names_b;
  h.labels_offset = h.rows_offset + rows_b;
  h.levels_offset = h.labels_offset + nr_tot * 2 + class_b;
  h.beta_offset   = h.levels_offset + nr_tot * 4;
  h.count_offset  = 0;                  /* reserved; see msfm.h */
  h.file_bytes    = h.beta_offset + nr_tot * np * 2;

  FILE *o = fopen(out, "wb");
  if (!o) fdie("cannot create output", out);
  wr(o, &h, sizeof(h), out);
  for (uint32_t c = 0; c < np; ++c)
    wr(o, f[0].pattern_names[c], strlen(f[0].pattern_names[c]) + 1, out);
  for (int i = 0; i < n_in; ++i)
    for (uint32_t r = 0; r < f[i].header->n_records; ++r)
      wr(o, f[i].record_names[r], strlen(f[i].record_names[r]) + 1, out);
  for (int i = 0; i < n_in; ++i) {
    uint32_t n = f[i].header->n_records;
    uint16_t *cid = xmal((size_t)n * 2, "remapped ids");
    for (uint32_t r = 0; r < n; ++r) {
      const char *name = f[i].header->n_classes
                           ? f[i].class_names[f[i].class_id[r]] : NULL;
      uint32_t at = 0;
      if (name) {
        int lo = 0, hi = (int)nk - 1;
        while (lo <= hi) {
          int mid = (lo + hi) / 2, c = strcmp(name, all[mid]);
          if (!c) { at = (uint32_t)mid; break; } else if (c < 0) hi = mid - 1; else lo = mid + 1;
        }
      }
      cid[r] = (uint16_t)at;
    }
    wr(o, cid, (size_t)n * 2, out);
    free(cid);
  }
  for (uint32_t k = 0; k < nk; ++k) wr(o, all[k], strlen(all[k]) + 1, out);
  for (int i = 0; i < n_in; ++i)
    wr(o, f[i].levels, (size_t)f[i].header->n_records * 4, out);
  for (int i = 0; i < n_in; ++i)
    wr(o, f[i].beta, (size_t)f[i].header->n_records * np * 2, out);
  if (fclose(o)) fdie("cannot finalize output", out);

  fprintf(stderr, "[methscope] classify-featurize --merge: %d chunks -> "
          "%u records x %u patterns, %u classes -> %s (%.1f MB)\n",
          n_in, h.n_records, np, nk, out, (double)h.file_bytes / 1048576.0);
  for (int i = 0; i < n_in; ++i) ms_msfm_close(&f[i]);
  free(all); free(f);
  return 0;
}

/* ---- CLI --------------------------------------------------------------- */

static int usage(void) {
  ms_help(stderr,
    "\n"
    "Usage:\n"
    "  methscope classify-featurize [options] -o <out.msfm> <query.cg> <ref.mrmp>\n"
    "  methscope classify-featurize [options] -o <out.msfm> <query.cg> <ref.cm> [ref2.cm ...]\n"
    "  methscope classify-featurize --merge  -o <out.msfm> <in1.msfm> [in2.msfm ...]\n"
    "\n"
    "Purpose:\n"
    "  Summarize each query record against the MRMP once and store the result,\n"
    "  so training and scoring never repeat the featurization. `classify-train`\n"
    "  and `classify` both accept the artifact with --data.\n"
    "\n"
    "  Featurization is single-threaded, so the way to go faster is to split the\n"
    "  query by record, featurize the chunks in parallel, and --merge them.\n"
    "\n"
    "Arguments:\n"
    "  <query.cg>   Query methylome(s), one record per sample or cell.\n"
    "  <ref.mrmp>   A pooled MRMP. A .mrmp is a chain, so ONE argument carries\n"
    "               every set, and set names come from the blocks -- no loose\n"
    "               .cm files and no order.txt to keep in step with it. This is\n"
    "               the normal input.\n"
    "  <ref.cm>     Exported masks, one per set; a bundle also works. Equivalent,\n"
    "               but the caller owns keeping them together and in order.\n"
    "\n"
    "Options:\n"
    "  -o <out>       Output .msfm (required).\n"
    "  -l <labels>    One label per query record, in query order. Embedded in the\n"
    "                 artifact, so a downstream train cannot be handed a\n"
    "                 mismatched label file. Omit for an unlabeled query.\n"
    "  --sample LIST  Comma list of target covered-CpG counts, one replicate each\n"
    "                 (0 = keep native coverage). Every cell is inflated ONCE and\n"
    "                 all replicates are drawn from that copy, so a coverage ladder\n"
    "                 costs one pass rather than one pass per level.\n"
    "  --reps N       Replicates per --sample level (default 1).\n"
    "  --patterns N   Feature patterns P1..PN. Default: every non-Pna mask state.\n"
    "  -b, --binarize Draw one read per sampled CpG, so the call is 0/1 not a\n"
    "                 fraction -- what a real sparse methylome delivers.\n"
    "  --continuous-features\n"
    "                 Emit each pattern's mean beta instead of a call. By DEFAULT\n"
    "                 a pattern beta is cut at 0.5, so a feature says \"this cell\n"
    "                 is methylated across this pattern\" rather than \"this cell\n"
    "                 reads 0.918\". The call is absolute, so it means the same\n"
    "                 thing on a cohort this reference never saw -- which a raw\n"
    "                 beta does not once a global shift, mitotic or otherwise,\n"
    "                 moves every value at once.\n"
    "                 continuous either way; they are not a contrast.\n"
    "  --rank-features off|add|replace   (default off)\n"
    "                 One column per class PAIR of a set: the mean over the\n"
    "                 patterns calling a 1 and b 0, against the mean over those\n"
    "                 calling a 0 and b 1. Asks \"is this cell more a than b\"\n"
    "                 -- relative,\n"
    "                 so a global shift enters both means and cancels. This is\n"
    "                 --satellite-contrast generalised past k=2, and it needs a\n"
    "                 .mrmp: a .cm carries no binstrings to read the sides from.\n"
    "                 A class the set never contrasts gets no column, and a cell\n"
    "                 observing nothing on either side gets NA -- a real\n"
    "                 abstention, since the THINNER side sets the floor.\n"
    "                 replace: k rank columns instead of the set's patterns;\n"
    "                 add: both.\n"
    "  --satellite-contrast off|add|replace   (default off)\n"
    "                 A 2-class satellite's two patterns have opposite polarity,\n"
    "                 so \"is P1 above P2\" asks the pair's question RELATIVELY --\n"
    "                 anything shifting both patterns together cancels, which is\n"
    "                 the violation rule's argmin cancellation as one feature.\n"
    "                 replace: emit the contrast INSTEAD of the two patterns.\n"
    "                 add: emit it alongside them -- but note the model then has\n"
    "                 no reason to prefer it, since in training the absolute\n"
    "                 columns work just as well; that is why replace is the one\n"
    "                 that removes the fragility rather than offering an\n"
    "                 alternative to it.\n"
    "  --thresh-pattern\n"
    "                 Cut at each pattern's OWN midpoint (stored in the .mrmp)\n"
    "                 rather than 0.5. Separates a close pair better, since\n"
    "                 inside a satellite both classes can sit above 0.5 -- but it\n"
    "                 is fitted to this reference and travels worse.\n"
    "  --seed S       Sampling seed (default 1). The draw is a pure function of it\n"
    "                 and is NOT affected by --threads.\n"
    "  --threads T    Worker threads (default 1). Cells are partitioned across\n"
    "                 workers, each seeking its own records via the .cg index.\n"
    "  --legacy-summarize  Use the old genome-scan featurizer (single-threaded, no\n"
    "                 --sample). Kept for A/B checks; output is bit-identical.\n"
    "  --counts <N>   Require N measured CpGs behind a pattern's beta; below that\n"
    "                 the beta is recorded MISSING rather than kept. A beta from a\n"
    "                 single CpG can only be 0 or 1, so after the 0.5 call it is\n"
    "                 always maximally confident and never borderline -- yet it\n"
    "                 carries the same weight as one backed by hundreds of CpGs.\n"
    "                 In mouse single cells 46.5% of observed pattern-betas rest\n"
    "                 on one or two CpGs. Default 1 (keep every observed pattern).\n"
    "  --merge        Concatenate .msfm chunks (pattern sets must match).\n"
    "  -h, --help     Show this help message.\n"
    "\n"
    "Notes:\n"
    "  Every pattern is stored, never a -p prefix, so one artifact serves any\n"
    "  `classify-train --patterns N`. Betas are u16 fixed point (code/65534,\n"
    "  65535 = NA), the same encoding the upscale msur uses for truth.\n"
    "\n");
  return 1;
}

/* Expand the mask arguments into one .cm per set.
 *
 * A single .mrmp argument is a CHAIN and expands into all its sets, each written
 * to a temp .cm. That is the point of pruning at mrmp-pool: the pooled artifact
 * holds exactly the patterns it claims, so it can be handed here directly and
 * the caller never has to keep 100 loose .cm files (or an order.txt) in step
 * with it. Set names then come from the BLOCKS rather than from mask basenames,
 * which makes the Pna.<set> columns exact rather than filename-derived.
 *
 * N .cm arguments still work unchanged -- that is what mrmp-export produces and
 * what YAME tooling reads. */
static void expand_mask_args(int n_arg, char *const *arg, uint32_t *n_out,
                             const char ***refs_out, char ***tmps_out,
                             char ***names_out, uint64_t **base_out,
                             uint64_t **len_out, uint32_t **pat_out) {
  int is_chain = 0;
  if (n_arg == 1) {
    FILE *f = fopen(arg[0], "rb");
    if (f) {
      char magic[8];
      is_chain = fread(magic, 1, 8, f) == 8 && !memcmp(magic, "MRMPIDX1", 8);
      fclose(f);
    }
  }
  if (!is_chain) {
    uint32_t n = (uint32_t)n_arg;
    const char **refs = xmal(n * sizeof(char *), "mask list");
    char **tmps = xmal(n * sizeof(char *), "mask temps");
    char **nms = xmal(n * sizeof(char *), "set names");
    for (uint32_t s = 0; s < n; ++s) {
      tmps[s] = NULL;
      refs[s] = ms_mrmp_resolve(arg[s], &tmps[s]);
      const char *b = strrchr(arg[s], '/'); b = b ? b + 1 : arg[s];
      size_t ln = strlen(b);
      const char *dot = strrchr(b, '.');
      if (dot && dot != b) ln = (size_t)(dot - b);
      nms[s] = xmal(ln + 1, "set name");
      memcpy(nms[s], b, ln); nms[s][ln] = '\0';
    }
    *n_out = n; *refs_out = refs; *tmps_out = tmps; *names_out = nms;
    *base_out = NULL; *len_out = NULL; *pat_out = NULL;
    return;
  }

  /* A chain is handed over BY REFERENCE: same path, one (offset, length) per
   * block. Nothing is materialised. Exporting a temp .cm per set used to inflate
   * each membership, walk it twice and recompress a 175 MB buffer -- ~0.86 s per
   * set, which at 100 sets was most of a featurize run before any cell was
   * read. The builder walks the block's runs in place instead. */
  ms_mrmpset_t *ch = ms_mrmpset_open(arg[0]);
  uint32_t n = ch->n_sets;
  const char **refs = xmal(n * sizeof(char *), "mask list");
  char **tmps = xmal(n * sizeof(char *), "mask temps");
  char **nms = xmal(n * sizeof(char *), "set names");
  uint64_t *base = xmal(n * sizeof(uint64_t), "block offsets");
  uint64_t *blen = xmal(n * sizeof(uint64_t), "block lengths");
  uint32_t *pat = xmal(n * sizeof(uint32_t), "per-set patterns");
  for (uint32_t s = 0; s < n; ++s) {
    tmps[s] = NULL; refs[s] = arg[0];
    base[s] = ch->block_off[s]; blen[s] = ch->block_bytes[s];
    nms[s] = strdup(ch->name[s]);
    mrmp_top_t *t = ms_mrmp_top_read_at(arg[0], ch->block_off[s], UINT32_MAX);
    pat[s] = t->n_patterns;                 /* the block IS its patterns now */
    ms_mrmp_top_free(t);
  }
  *base_out = base; *len_out = blen; *pat_out = pat;
  fprintf(stderr, "\n[methscope] classify-featurize\n\n");
  fprintf(stderr, "  %-14s %s, %u set(s)\n", "artifact", arg[0], n);
  ms_mrmpset_free(ch);
  *n_out = n; *refs_out = refs; *tmps_out = tmps; *names_out = nms;
}

/* Parse a comma list of coverage targets into rep_sample, repeated `reps` times
 * each (so --sample A,B --reps 3 gives 3 replicates at A then 3 at B). */
static uint32_t *parse_levels(const char *spec, uint32_t reps, uint32_t *n_out) {
  uint32_t cap = 8, n = 0;
  uint32_t *lv = xmal(cap * 4, "levels");
  const char *p = spec;
  while (*p) {
    char *end;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p || v > UINT32_MAX) fdie("--sample expects integers", spec);
    if (n == cap) { cap <<= 1; lv = realloc(lv, cap * 4); if (!lv) fdie("out of memory", NULL); }
    lv[n++] = (uint32_t)v;
    p = (*end == ',') ? end + 1 : end;
    if (*end && *end != ',') fdie("--sample expects a comma list", spec);
  }
  if (!n) fdie("--sample is empty", spec);
  uint32_t total = n * reps;
  uint32_t *out = xmal((size_t)total * 4, "rep sample");
  for (uint32_t k = 0, w = 0; k < n; ++k)
    for (uint32_t r = 0; r < reps; ++r) out[w++] = lv[k];
  free(lv);
  *n_out = total;
  return out;
}

int main_classify_featurize(int argc, char *argv[]) {
  const char *out = NULL, *labels = NULL, *sample_spec = NULL;
  uint32_t reps = 1, patterns = 0;
  uint64_t seed = 1;
  unsigned threads = 1;
  int binarize = 0, legacy = 0;
  /* 1 = cut at 0.5 (default), 2 = per-pattern midpoints, 0 = leave continuous */
  int binarize_feat = 1;
  int contrast = 0;                 /* 0 off, 1 alongside, 2 replacing */
  int rank = 0;                     /* same three modes, generalised to any k */
  int merge = 0, i = 1;
  uint32_t min_cpgs = 0;   /* --counts N: below N measured CpGs -> NA */
  for (; i < argc; ++i) {
    if      (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "-l") && i + 1 < argc) labels = argv[++i];
    else if (!strcmp(argv[i], "--sample") && i + 1 < argc) sample_spec = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--patterns") && i + 1 < argc) patterns = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--binarize")) binarize = 1;
    else if (!strcmp(argv[i], "--counts") && i + 1 < argc)
      min_cpgs = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--continuous-features")) binarize_feat = 0;
    else if (!strcmp(argv[i], "--thresh-pattern")) binarize_feat = 2;
    else if (!strcmp(argv[i], "--rank-features") && i + 1 < argc) {
      const char *v = argv[++i];
      if      (!strcmp(v, "off"))     rank = 0;
      else if (!strcmp(v, "add"))     rank = 1;
      else if (!strcmp(v, "replace")) rank = 2;
      else fdie("--rank-features wants off|add|replace", v);
    }
    else if (!strcmp(argv[i], "--satellite-contrast") && i + 1 < argc) {
      const char *v = argv[++i];
      if      (!strcmp(v, "off"))     contrast = 0;
      else if (!strcmp(v, "add"))     contrast = 1;
      else if (!strcmp(v, "replace")) contrast = 2;
      else fdie("--satellite-contrast wants off|add|replace", v);
    }
    else if (!strcmp(argv[i], "--legacy-summarize")) legacy = 1;
    else if (!strcmp(argv[i], "--merge")) merge = 1;
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-")) fdie("unrecognized option", argv[i]);
    else break;
  }
  if (!out) return usage();

  if (merge) {
    if (argc - i < 1) return usage();
    return merge_msfm(out, argv + i, argc - i);
  }

  if (argc - i < 2) return usage();
  const char *query = argv[i];
  uint32_t n_sets; const char **refs; char **tmps; char **snames;
  uint64_t *mbase, *mlen; uint32_t *mpat;
  expand_mask_args(argc - i - 1, argv + i + 1, &n_sets, &refs, &tmps, &snames,
                   &mbase, &mlen, &mpat);
  const char *ref = refs[0];

  /* The sampled builder is the DEFAULT: it reproduces the summarize1 path
   * bit-for-bit (verified on the full pattern set) and is the only one that can
   * amortize an inflate across replicates or use more than one core. It needs
   * the .cg index to seek per record, so a query without one falls back --
   * loudly, because silently taking a 20x slower path is how the fast path ends
   * up never running. --legacy-summarize forces the old path for A/B checks. */
  char *fidx = get_fname_index((char *)query);
  int have_idx = fidx && access(fidx, R_OK) == 0;
  free(fidx);
  if (sample_spec && !have_idx)
    fdie("--sample needs a .cg.idx to seek per record; run `yame index` first", query);
  if (!legacy && !have_idx)
    fprintf(stderr, "[methscope] classify-featurize: no .cg.idx for %s -- falling "
            "back to the single-threaded scan (run `yame index` to enable the "
            "fast path)\n", query);

  if (!legacy && have_idx) {
    uint32_t n_reps = 1;
    uint32_t *rep_sample;
    if (sample_spec) {
      rep_sample = parse_levels(sample_spec, reps ? reps : 1, &n_reps);
      /* n_reps is RUNGS x --reps. Reporting only the product invited reading it
       * as the thread count, which it has nothing to do with: a 24-rung ladder
       * at --reps 1 is 24 replicates whether you run 1 thread or 64. */
      uint32_t rungs = 1;
      for (const char *q = sample_spec; *q; ++q) if (*q == ',') ++rungs;
      fprintf(stderr, "  %-14s %u rung(s) x %u replicate(s) = %u draw(s)/cell\n",
              "sampling", rungs, reps ? reps : 1, n_reps);
    }
    else { rep_sample = xmal(4, "rep sample"); rep_sample[0] = 0; n_reps = 1; }

    uint16_t *beta; uint32_t *levels; char **names; uint32_t n_cells, ncol;
    uint32_t n_sets_out = 0, *col0_out = NULL; char **set_names = NULL;
    if (n_sets == 1) {
      /* ONE set goes through the same multi-set builder, which for n_sets == 1
       * lays the columns out identically (see its header) -- the old dedicated
       * single-mask call was a fast path that quietly lacked what the multi path
       * had grown. Two features went missing on a one-block chain that worked on
       * a two-block one: the block's own pattern count (so a single-set .mrmp,
       * which is what every mrmp-tree node is, was rejected for "needs an
       * explicit pattern count"), and --satellite-contrast, which was accepted
       * and then ignored. Both are the SAME bug, and one call site cannot grow a
       * third instance of it. --patterns still overrides the block's count. */
      uint32_t np0[1]; np0[0] = patterns ? patterns : (mpat ? mpat[0] : 0);
      uint32_t col0_1[1];
      ms_msfm_build_sampled_multi(query, refs, mbase, mlen, np0, 1, rep_sample,
                                  n_reps, binarize, min_cpgs, seed, threads,
                                  binarize_feat, contrast, rank,
                                  &beta, &levels, &names, &n_cells, &ncol, col0_1);
    } else {
      /* Several MRMP sets in ONE pass over the query. Each cell is inflated
       * once and scored against every set, so N sets cost N scatter-adds per
       * sampled CpG rather than N decompressions -- and decompression is the
       * dominant cost (see the msfm_build.c header). Merging the masks instead
       * is not an option: a .cm gives each CpG exactly one pattern, so
       * overlapping sets would fight over the CpGs they share, which are
       * precisely the informative ones. */
      uint32_t *np = xmal(n_sets * sizeof(uint32_t), "pattern counts");
      /* --patterns caps the FIRST set only. The others auto-detect from their
       * own mask (np = 0), because a satellite holds 2-30 patterns and giving
       * it the global's cap would pad it with thousands of empty columns --
       * 15 satellites x 6001 would be 90k columns of which ~90 are real. */
      /* From a chain each block reports its own pattern count; --patterns still
       * caps the first set for the loose .cm form. */
      if (mpat) for (uint32_t s = 0; s < n_sets; ++s) np[s] = mpat[s];
      else { np[0] = patterns; for (uint32_t s = 1; s < n_sets; ++s) np[s] = 0; }
      uint32_t *col0 = xmal(n_sets * sizeof(uint32_t), "set offsets");
      ms_msfm_build_sampled_multi(query, refs, mbase, mlen, np, n_sets, rep_sample, n_reps,
                                  binarize, min_cpgs, seed, threads, binarize_feat,
                                  contrast, rank,
                                  &beta, &levels, &names, &n_cells, &ncol, col0);
      /* The per-set start offsets used to be dumped here -- 100 numbers on one
       * line, for a layout `inspect` reports properly. */
      fprintf(stderr, "  %-14s %u set(s) fused, %u patterns\n",
              "layout", n_sets, ncol);
      set_names = snames; snames = NULL;       /* ownership moves to the writer */
      n_sets_out = n_sets; col0_out = col0;
      free(np);
    }
    char **lab = labels ? read_labels(labels, (int)n_cells) : NULL;
    /* Explicit rather than passing binarize_feat straight through: the CLI
     * encoding and the on-disk flag are separate vocabularies. */
    uint32_t fflags = binarize_feat == 1 ? MSFM_FLAG_BIN_FLAT
                    : binarize_feat == 2 ? MSFM_FLAG_BIN_PAT : 0u;
    if (contrast == 1) fflags |= MSFM_FLAG_CONTRAST_ADD;
    if (contrast == 2) fflags |= MSFM_FLAG_CONTRAST_ONLY;
    if (rank == 1)     fflags |= MSFM_FLAG_RANK_ADD;
    if (rank == 2)     fflags |= MSFM_FLAG_RANK_ONLY;
    write_msfm_raw(out, beta, levels, names, n_cells, n_reps, ncol, lab,
                   rep_sample, n_sets_out, col0_out, set_names, fflags);
    if (set_names) { for (uint32_t s = 0; s < n_sets_out; ++s) free(set_names[s]); free(set_names); }
    free(col0_out);
    if (snames) { for (uint32_t s = 0; s < n_sets; ++s) free(snames[s]); free(snames); }
    for (uint32_t s = 0; s < n_sets; ++s) ms_mrmp_cleanup(tmps[s]);
    free(refs); free(tmps); free(mbase); free(mlen); free(mpat);
    if (lab) { for (uint32_t r = 0; r < n_cells; ++r) free(lab[r]); free(lab); }
    for (uint32_t r = 0; r < n_cells; ++r) free(names[r]);
    free(names); free(beta); free(levels); free(rep_sample);
    return 0;
  }

  ms_matrix_t *m = ms_matrix_build(query, ref);
  char **lab = labels ? read_labels(labels, m->n_cells) : NULL;
  write_msfm(out, m, lab);
  if (lab) { for (int r = 0; r < m->n_cells; ++r) free(lab[r]); free(lab); }
  ms_matrix_free(m);
  /* the legacy scan path uses only the first mask; release the rest */
  for (uint32_t s = 0; s < n_sets; ++s) ms_mrmp_cleanup(tmps[s]);
  for (uint32_t s = 0; s < n_sets; ++s) free(snames[s]);
  free(refs); free(tmps); free(snames);
  return 0;
}
