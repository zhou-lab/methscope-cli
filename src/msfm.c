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

void ms_msfm_close(ms_msfm_t *f) {
  free(f->pattern_names); free(f->record_names); free(f->class_names);
  if (f->map) munmap(f->map, (size_t)f->length);
  if (f->fd > 0) close(f->fd);
  memset(f, 0, sizeof(*f));
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
  printf("format\tMSFMAT1 v%u\n", h->version);
  printf("records\t%u\n", h->n_records);
  printf("patterns\t%u\n", h->n_patterns);
  printf("beta\tu16 fixed point (code/65534; 65535 = NA)\n");
  printf("classes\t%u\n", h->n_classes);
  if (h->n_classes) {
    printf("labels\t");
    for (uint32_t k = 0; k < h->n_classes && k < 8; ++k)
      printf("%s%s", k ? ", " : "", f.class_names[k]);
    if (h->n_classes > 8) printf(", ... (%u total)", h->n_classes);
    printf("\n");
  }
  /* the coverage scalar's raw input, summarized so a bad ladder is obvious */
  uint64_t lo = UINT64_MAX, hi = 0, sum = 0;
  for (uint32_t r = 0; r < h->n_records; ++r) {
    uint32_t v = f.levels[r];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
  }
  if (h->n_records)
    printf("covered_cpgs\tmin %" PRIu64 "  mean %" PRIu64 "  max %" PRIu64 "\n",
           lo, sum / h->n_records, hi);
  printf("file_bytes\t%" PRIu64 "\n", h->file_bytes);
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
                           char *const *lab, const uint32_t *rep_sample) {
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

  /* pattern names: P1..P(ncol-1) then Pna, the canonical column order */
  char (*pn)[16] = xmal((size_t)ncol * 16, "pattern names");
  for (uint32_t g = 0; g + 1 < ncol; ++g) snprintf(pn[g], 16, "P%u", g + 1);
  snprintf(pn[ncol - 1], 16, "Pna");

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
  h.flags = 0;
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

  fprintf(stderr, "[methscope] classify-featurize: %" PRIu64 " records "
          "(%u cells x %u replicates) x %u patterns, %u classes -> %s (%.1f MB)\n",
          nr, n_cells, n_reps, ncol, nk, out, (double)h.file_bytes / 1048576.0);
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
    "  methscope classify-featurize [options] -o <out.msfm> <query.cg> <ref.cm>\n"
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
    "  <ref.cm>     MRMP pattern definition (a YAME .cm); a bundle also works.\n"
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

  if (argc - i != 2) return usage();
  const char *query = argv[i];
  char *tmp_mrmp = NULL;
  const char *ref = ms_mrmp_resolve(argv[i + 1], &tmp_mrmp);

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
    if (sample_spec) rep_sample = parse_levels(sample_spec, reps ? reps : 1, &n_reps);
    else { rep_sample = xmal(4, "rep sample"); rep_sample[0] = 0; n_reps = 1; }

    uint16_t *beta; uint32_t *levels; char **names; uint32_t n_cells, ncol;
    ms_msfm_build_sampled(query, ref, patterns, rep_sample, n_reps, binarize,
                          min_cpgs, seed, threads,
                          &beta, &levels, &names, &n_cells, &ncol);
    char **lab = labels ? read_labels(labels, (int)n_cells) : NULL;
    write_msfm_raw(out, beta, levels, names, n_cells, n_reps, ncol, lab, rep_sample);
    if (lab) { for (uint32_t r = 0; r < n_cells; ++r) free(lab[r]); free(lab); }
    for (uint32_t r = 0; r < n_cells; ++r) free(names[r]);
    free(names); free(beta); free(levels); free(rep_sample);
    ms_mrmp_cleanup(tmp_mrmp);
    return 0;
  }

  ms_matrix_t *m = ms_matrix_build(query, ref);
  char **lab = labels ? read_labels(labels, m->n_cells) : NULL;
  write_msfm(out, m, lab);
  if (lab) { for (int r = 0; r < m->n_cells; ++r) free(lab[r]); free(lab); }
  ms_matrix_free(m);
  ms_mrmp_cleanup(tmp_mrmp);
  return 0;
}
