// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * deconv: a packed reference for per-query deconvolution.
 *
 * WHAT THIS REPLACES
 *
 * The legacy .refx froze one pattern set at build time and stored a beta TSV
 * over it. Anything that wants to CHOOSE the pattern set per query -- which is
 * the whole point of adaptive deconvolution, since the q-filter is a
 * conjunction that loosens as classes drop out -- cannot use it, and has had to
 * hold the raw M/U store in memory instead: 1.94 GB for 33 classes x 29.4 M
 * CpGs, re-derived on every run.
 *
 * So this artifact stores what a per-query rebuild actually needs, and nothing
 * else: one uint16 beta per (class, CpG), over only the CpGs that could ever
 * matter.
 *
 * WHY uint16
 *
 * A pattern's beta is the MEAN OF PER-CpG BETAS (YAME format3.c,
 * st[0].beta = sum_beta / n_o), never sum(M)/sum(M+U), so M and U are never
 * needed apart -- only beta. And at the settings this supports the only
 * coverage fact needed is "covered at all", which a sentinel carries. uint16
 * with 0xFFFF for uncovered costs 2 B per (class, CpG) against the store's
 * 8-byte M|U packing, and its quantisation error is 7.6e-6 -- far below
 * anything a 0.30/0.70 band can resolve. (uint8 would halve it again, but its
 * 2e-3 error can flip a CpG sitting exactly on a band edge, which would make
 * this disagree with mrmp-build.)
 *
 * WHICH CpGs ARE NEVER USEFUL
 *
 * Call a class CALLABLE at a CpG when it is covered and sits outside the
 * ambiguous band -- beta <= qlo or beta >= qhi. A class inside the band makes
 * no call, so any scope containing it is inadmissible there; an uncovered class
 * likewise. A pattern needs at least one class on each side or its binstring is
 * constant and separates nothing.
 *
 * Therefore a CpG can contribute to SOME scope only if at least one callable
 * class is unmethylated AND at least one callable class is methylated. If they
 * all sit on one side -- or fewer than two are callable at all -- no subset of
 * classes can ever form a non-constant admissible binstring there, whatever the
 * query. Those rows are dropped here once, rather than re-tested per query.
 *
 * This is scope-INDEPENDENT, which is what makes it safe to bake in: it is a
 * necessary condition for usefulness under every class subset, so dropping a
 * row can never remove a pattern a later rebuild would have found. It does
 * depend on (qlo, qhi, beta_thr, mincov), so those are recorded in the header
 * and a consumer must refuse a mismatch rather than silently reinterpret.
 *
 * FORMAT (MSDREF1), little-endian, all offsets from the file start:
 *
 *   magic     8   "MSDREF1\0"
 *   header   64   version, n_class, n_row (the reference row space), n_kept,
 *                 qlo, qhi, beta_thr, mincov
 *   names     -   n_class NUL-terminated class names, in FILE order, which IS
 *                 the binstring digit order
 *   rows      4 * n_kept   uint32 row index into the full row space, ascending
 *   mu        2 * n_class * n_kept   uint16 (M<<8 | U), CLASS-MAJOR.
 *             Each byte is a read count, the pair scaled down together
 *             whenever either exceeds 255 so the ratio survives; 0 means
 *             uncovered. beta = M/(M+U), depth = M+U saturating at 510.
 *             Version 1 stored a uint16 beta here and no depth at all.
 *
 * Class-major so a scope's rows are contiguous per class, matching how the
 * rebuild scans; ascending row index so a query can be walked against it with
 * a merge rather than a lookup.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include "methscope.h"
#include "cfile.h"
#include "cdata.h"
#include <pthread.h>
#include "nnls.h"
#include <math.h>

#define D2_MAGIC   "MSDREF1"
#define D2_BETA_NA 0xFFFFu

/* Set by whichever subcommand is running, so an error names the command the
 * user typed rather than whichever one happens to own the helper. */
static const char *d2_cmd = "deconv";

static void d2die(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s: %s\n", d2_cmd, msg, arg);
  else     fprintf(stderr, "[methscope] %s: %s\n", d2_cmd, msg);
  exit(1);
}

static void *d2alloc(size_t n, size_t sz, const char *what) {
  void *p = calloc(n ? n : 1, sz);
  if (!p) d2die("out of memory", what);
  return p;
}

static uint16_t d2_encode(double b) {
  if (b < 0) b = 0; else if (b > 1) b = 1;
  return (uint16_t)(b * 65534.0 + 0.5);
}

/* Class names in FILE order -- a permuted walk would silently relabel every
 * binstring downstream. */
static char **d2_read_idx(const char *ref, uint32_t *n_out, int64_t **off_out,
                           int optional) {
  char idx[PATH_MAX];
  if (snprintf(idx, sizeof idx, "%s.idx", ref) >= (int)sizeof idx)
    d2die("reference path too long", ref);
  FILE *f = fopen(idx, "r");
  if (!f) {
    if (optional) {
      *n_out = 0; *off_out = NULL;
      return NULL;
    }
    d2die("cannot open reference index (expected <ref>.idx)", idx);
  }
  size_t cap = 64, n = 0;
  char **names = d2alloc(cap, sizeof(char *), "names");
  int64_t *off = d2alloc(cap, sizeof(int64_t), "offsets");
  char *line = NULL; size_t lc = 0; ssize_t len;
  while ((len = getline(&line, &lc, f)) > 0) {
    char *tab = strpbrk(line, "\t\n");
    size_t nl = tab ? (size_t)(tab - line) : (size_t)len;
    if (!nl) continue;
    if (n == cap) {
      cap <<= 1;
      names = realloc(names, cap * sizeof(char *));
      off   = realloc(off, cap * sizeof(int64_t));
      if (!names || !off) d2die("out of memory", "index grow");
    }
    names[n] = d2alloc(nl + 1, 1, "name");
    memcpy(names[n], line, nl);
    off[n] = (tab && *tab == '\t') ? (int64_t)strtoll(tab + 1, NULL, 10) : -1;
    ++n;
  }
  free(line); fclose(f);
  if (!n) d2die("reference index is empty", idx);
  *n_out = (uint32_t)n; *off_out = off;
  return names;
}

static void put_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void put_f64(FILE *f, double v)   { fwrite(&v, 8, 1, f); }

int main_deconv_build_ref(int argc, char *argv[]) {
  d2_cmd = "deconv-build-ref";
  const char *out_path = NULL;
  double qlo = 0.30, qhi = 0.70, beta_thr = 0.5;
  uint32_t mincov = 1;
  int force = 0, keep_all = 0, i = 1;

  for (; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-o") && i + 1 < argc) out_path = argv[++i];
    else if (!strcmp(a, "--force")) force = 1;
    else if (!strcmp(a, "--keep-all")) keep_all = 1;
    else if (!strcmp(a, "--mincov") && i + 1 < argc)
      mincov = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(a, "--beta-threshold") && i + 1 < argc)
      beta_thr = atof(argv[++i]);
    else if (!strcmp(a, "--qfilter") && i + 1 < argc) {
      char *end = NULL;
      double lo = strtod(argv[++i], &end);
      if (!end || *end != ',') d2die("--qfilter wants LO,HI", argv[i]);
      double hi = strtod(end + 1, NULL);
      if (!(lo >= 0 && lo < hi && hi <= 1))
        d2die("--qfilter needs 0 <= LO < HI <= 1", argv[i]);
      qlo = lo; qhi = hi;
    }
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stdout,
"Usage:\n"
"  methscope deconv-build-ref [options] <celltypes.cg> -o <out.msdref>\n"
"\n"
"Purpose:\n"
"  Pack a per-cell-type M/U store into the uint16 M/U reference that `deconv`\n"
"  rebuilds its pattern set from, keeping only the CpGs that could ever\n"
"  contribute to some class subset.\n"
"\n"
"  A class is CALLABLE at a CpG when it is covered and outside the ambiguous\n"
"  band (beta <= LO or beta >= HI). A row is kept only when at least one\n"
"  callable class is unmethylated AND at least one is methylated -- otherwise\n"
"  no subset of classes can form a non-constant admissible binstring there,\n"
"  whatever the query, so the row is dead weight. The test is a necessary\n"
"  condition under EVERY scope, so dropping a row cannot remove a pattern a\n"
"  later rebuild would have found.\n"
"\n"
"Arguments:\n"
"  <celltypes.cg>     One record per cell type, YAME format 3 (M/U) or format 6\n"
"                     (universe 0/1), with a matching <celltypes.cg>.idx whose\n"
"                     index order IS the binstring digit order. Format-6\n"
"                     positions outside the universe are treated as missing.\n"
"\n"
"Options:\n"
"  -o <out.msdref>    Write the reference here. Required.\n"
"  --qfilter LO,HI    The admission band baked into the row test. Default:\n"
"                     0.30,0.70. Must match the solver's.\n"
"  --beta-threshold B Call a class methylated above B. Default: 0.5.\n"
"  --mincov N         A class is covered at N reads or more. Default: 1.\n"
"  --keep-all         Skip the never-useful row test, keeping every row so a\n"
"                     consumer can re-apply any band in memory. Costs 1.94 GB\n"
"                     for 33 classes x 29.4M rows.\n"
"  --force            Overwrite an existing output.\n"
"  -h                 Show this help message.\n");
      return 0;
    }
    else if (a[0] == '-' && a[1]) d2die("unrecognized option", a);
    else break;
  }
  if (argc - i != 1 || !out_path) {
    fprintf(stderr,
      "Usage: methscope deconv-build-ref <celltypes.cg> -o <out.msdref>\n");
    return 1;
  }
  const char *ref = argv[i];
  if (!force) {
    FILE *t = fopen(out_path, "rb");
    if (t) { fclose(t); d2die("output exists (use --force)", out_path); }
  }

  int64_t *voff = NULL;
  uint32_t n_class = 0;
  char **name = d2_read_idx(ref, &n_class, &voff, 0);

  /* Load every class as uint16 betas over the full row space. This is the same
   * 1.94 GB the solver used to carry; it is transient here, and the point of
   * the command is that nothing downstream has to pay it again. */
  uint64_t n_row = 0;
  uint16_t *beta = NULL;
  /* M and U as one byte each, the pair scaled down together whenever either
   * exceeds 255. Same two bytes per (class, CpG) the beta plane already costs,
   * so the artifact does not grow -- and it carries DEPTH, which a beta alone
   * cannot. Per-class mean depth here runs 119-477, so most rows scale; the
   * ratio survives to within half a unit of 255, i.e. a beta error under
   * 0.002, and total depth saturates at 510. Both are far from anything the
   * 0.30/0.70 band or a depth floor near 20 can notice. */
  uint16_t *mup = NULL;
  cfile_t cf = open_cfile((char *)ref);
  for (uint32_t k = 0; k < n_class; ++k) {
    if (voff[k] >= 0 && bgzf_seek(cf.fh, voff[k], SEEK_SET) != 0)
      d2die("cannot seek to record", name[k]);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) d2die("store record is empty", name[k]);
    decompress_in_situ(&c);
    if (c.fmt != '3' && c.fmt != '6')
      d2die("reference must be format-3 (M/U) or format-6 (universe 0/1) .cg",
            name[k]);
    if (!k) {
      n_row = c.n;
      beta = d2alloc((size_t)n_class * n_row, sizeof(uint16_t), "betas");
      for (size_t j = 0; j < (size_t)n_class * n_row; ++j) beta[j] = D2_BETA_NA;
      mup = d2alloc((size_t)n_class * n_row, sizeof(uint16_t), "M/U");
    } else if (c.n != n_row) {
      d2die("reference records disagree on CpG count", name[k]);
    }
    uint16_t *row = beta + (size_t)k * n_row;
    uint16_t *mrow = mup + (size_t)k * n_row;
    for (uint64_t r = 0; r < n_row; ++r) {
      uint64_t mu = c.fmt == '3' ? f3_get_mu(&c, r)
                     : (FMT6_IN_UNI(c, r) ?
                         (FMT6_IN_SET(c, r) ? (1ull << 32) : 1ull) : 0);
      if (!mu || MU2cov(mu) < mincov) continue;
      row[r] = d2_encode(MU2beta(mu));
      uint64_t m = mu >> 32, u = mu & 0xffffffffu, mx = m > u ? m : u;
      if (mx > 255) {                      /* scale the pair, keep the ratio */
        uint32_t m8 = (uint32_t)((m * 255 + mx / 2) / mx);
        uint32_t u8 = (uint32_t)((u * 255 + mx / 2) / mx);
        if (m && !m8) m8 = 1;              /* a covered side never rounds away */
        if (u && !u8) u8 = 1;
        m = m8; u = u8;
      }
      mrow[r] = (uint16_t)((m << 8) | u);
    }
    free_cdata(&c);
    fprintf(stderr, "\r[methscope] deconv-build-ref: reading %u/%u",
            k + 1, n_class);
    fflush(stderr);
  }
  bgzf_close(cf.fh);
  fputc('\n', stderr);
  if (n_row > 0xFFFFFFFFull) d2die("row space exceeds uint32", ref);

  /* Which rows survive. One pass, and the counters are the build's whole
   * report -- a reader should be able to see what was dropped and why. */
  const uint16_t lo = d2_encode(qlo), hi = d2_encode(qhi),
                 bt = d2_encode(beta_thr);
  uint32_t *keep = d2alloc(n_row, sizeof(uint32_t), "kept rows");
  uint64_t n_keep = 0, n_uncov = 0, n_amb = 0, n_const = 0;
  for (uint64_t r = 0; r < n_row; ++r) {
    uint32_t n_call = 0, n_hi = 0, n_lo = 0, n_cov = 0;
    for (uint32_t k = 0; k < n_class; ++k) {
      uint16_t b = beta[(size_t)k * n_row + r];
      if (b == D2_BETA_NA) continue;
      ++n_cov;
      if (b >= hi && b > bt)      { ++n_call; ++n_hi; }
      else if (b <= lo && b < bt) { ++n_call; ++n_lo; }
    }
    /* --keep-all: skip the never-useful test and keep every row. The filter is
     * band-dependent, so an artifact built with it can only ever be read at its
     * own band; an unfiltered one (1.94 GB for 33 classes) lets a harness
     * explore ANY band in memory without a rebuild, which is what makes
     * iteration minutes rather than seconds. */
    if (!keep_all) {
      if (!n_cov)        { ++n_uncov; continue; }
      if (n_call < 2)    { ++n_amb;   continue; }
      if (!n_hi || !n_lo){ ++n_const; continue; }
    }
    keep[n_keep++] = (uint32_t)r;
  }

  FILE *out = fopen(out_path, "wb");
  if (!out) d2die("cannot open output", out_path);
  fwrite(D2_MAGIC, 1, 8, out);                  /* 7 chars + NUL */
  put_u32(out, 2);                              /* version: adds M/U */
  put_u32(out, n_class);
  put_u64(out, n_row);
  put_u64(out, n_keep);
  put_f64(out, qlo); put_f64(out, qhi); put_f64(out, beta_thr);
  put_u32(out, mincov);
  put_u32(out, 0);                              /* reserved */
  for (uint32_t k = 0; k < n_class; ++k)
    fwrite(name[k], 1, strlen(name[k]) + 1, out);
  fwrite(keep, sizeof(uint32_t), n_keep, out);
  /* class-major: one contiguous run of kept betas per class */
  uint16_t *buf = d2alloc(n_keep ? n_keep : 1, sizeof(uint16_t), "row buffer");
  for (uint32_t k = 0; k < n_class; ++k) {
    const uint16_t *src = mup + (size_t)k * n_row;
    for (uint64_t j = 0; j < n_keep; ++j) buf[j] = src[keep[j]];
    fwrite(buf, sizeof(uint16_t), n_keep, out);
  }
  if (ferror(out)) d2die("error writing output", out_path);
  fclose(out);

  double mb = (double)((size_t)n_class * n_keep * 2 + (size_t)n_keep * 4) / 1e6;
  fprintf(stderr,
    "[methscope] deconv-build-ref: %u classes x %llu rows -> %llu kept "
    "(%.2f%%)\n"
    "                               dropped %llu uncovered, %llu with <2 "
    "callable classes, %llu one-sided\n"
    "                               %.1f MB (was %.2f GB as resident M/U)"
    " -> %s\n",
    n_class, (unsigned long long)n_row, (unsigned long long)n_keep,
    n_row ? 100.0 * (double)n_keep / (double)n_row : 0.0,
    (unsigned long long)n_uncov, (unsigned long long)n_amb,
    (unsigned long long)n_const,
    mb, (double)n_class * n_row * 8.0 / 1e9, out_path);

  free(buf); free(keep); free(beta); free(mup); free(voff);
  for (uint32_t k = 0; k < n_class; ++k) free(name[k]);
  free(name);
  return 0;
}

/* ==================================================================== */
/* deconv: the solver.                                                  */
/* ==================================================================== */
/*
 * THE MEASURED-ROW INVARIANT
 *
 * Every step below sees ONLY the CpGs this query measured. That is not an
 * optimisation, it is what makes the equations well posed: a pattern's
 * reference beta is a mean over the CpGs that DEFINE it and its observed beta
 * a mean over the CpGs the query COVERED, so if those are different sets the
 * two sides of the equation are not comparable. The old path averaged the
 * reference over every admissible CpG while averaging the query over the ~3%
 * it actually saw, and at 2^16 those sets differ by a factor of a few hundred.
 *
 * So the query is intersected with the reference ONCE, up front, into a dense
 * measured-row list; admissibility, the binstring, the constancy test, the CpG
 * count that sets the weight, and any per-binstring ranking all run on that
 * list and nothing else. count[r] == qn[r] holds by construction rather than
 * by discipline, and is asserted.
 */

typedef struct {
  uint32_t  n_class;
  uint64_t  n_row, n_keep;
  double    qlo, qhi, beta_thr;
  uint32_t  mincov;
  char    **name;
  uint32_t *row;              /* n_keep, ascending, into the full row space */
  int       has_depth;        /* v2 carries counts; v1 does not */
  uint16_t *mu;               /* n_class * n_keep, class-major: M<<8 | U.
                               * beta and depth are DERIVED (d2_b / d2_dep), so
                               * neither is stored -- two bytes carry both. */
} d2ref_t;

/* mu word -> encoded beta, and -> depth. A 128 KB table beats a division in a
 * loop that runs 33 x 7.9 M times per query, and it costs less than the depth
 * plane it replaces. Filled once. */
static uint16_t d2_blut[65536];
static uint8_t  d2_dlut[65536];
static void d2_lut_init(void) {
  for (uint32_t v = 0; v < 65536; ++v) {
    uint32_t m = v >> 8, u = v & 0xFF, cov = m + u;
    d2_blut[v] = cov ? d2_encode((double)m / (double)cov) : D2_BETA_NA;
    d2_dlut[v] = cov > 255 ? 255 : (uint8_t)cov;
  }
}
#define d2_b(v)   d2_blut[(v)]
#define d2_dep(v) d2_dlut[(v)]

static void d2ref_load(const char *path, d2ref_t *R) {
  FILE *f = fopen(path, "rb");
  if (!f) d2die("cannot open reference", path);
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, D2_MAGIC, 8))
    d2die("not a .msdref reference (bad magic)", path);
  uint32_t ver = 0, rsv = 0;
  if (fread(&ver, 4, 1, f) != 1 || fread(&R->n_class, 4, 1, f) != 1 ||
      fread(&R->n_row, 8, 1, f) != 1 || fread(&R->n_keep, 8, 1, f) != 1 ||
      fread(&R->qlo, 8, 1, f) != 1 || fread(&R->qhi, 8, 1, f) != 1 ||
      fread(&R->beta_thr, 8, 1, f) != 1 || fread(&R->mincov, 4, 1, f) != 1 ||
      fread(&rsv, 4, 1, f) != 1)
    d2die("truncated header", path);
  if (ver != 1 && ver != 2) d2die("unsupported .msdref version", path);

  R->name = d2alloc(R->n_class, sizeof(char *), "class names");
  for (uint32_t k = 0; k < R->n_class; ++k) {
    char buf[512]; size_t n = 0; int c;
    while ((c = fgetc(f)) > 0) {
      if (n + 1 >= sizeof buf) d2die("class name too long", path);
      buf[n++] = (char)c;
    }
    if (c < 0) d2die("truncated class names", path);
    buf[n] = '\0';
    R->name[k] = strdup(buf);
  }
  R->row  = d2alloc(R->n_keep, sizeof(uint32_t), "row index");
  R->mu = d2alloc((size_t)R->n_class * R->n_keep, sizeof(uint16_t), "M/U");
  if (fread(R->row, sizeof(uint32_t), R->n_keep, f) != R->n_keep)
    d2die("truncated row index", path);
  if (fread(R->mu, sizeof(uint16_t), (size_t)R->n_class * R->n_keep, f)
      != (size_t)R->n_class * R->n_keep)
    d2die("truncated M/U block", path);
  fclose(f);
  d2_lut_init();
  R->has_depth = (ver == 2);
  if (ver == 1) {
    /* A v1 artifact stored a bare beta and no counts. Re-encode it as M/U so
     * one code path serves both -- but the pair sums to 255 whatever the beta,
     * so d2_dep() reports 255 everywhere and any depth test would PASS
     * unconditionally. That is the opposite of honest, so has_depth records
     * the truth and every depth-dependent rule consults it rather than
     * believing the reconstructed counts. */
    size_t n = (size_t)R->n_class * R->n_keep;
    for (size_t j = 0; j < n; ++j) {
      uint16_t b = R->mu[j];
      if (b == D2_BETA_NA) { R->mu[j] = 0; continue; }
      uint32_t m = (uint32_t)((double)b / 65534.0 * 255.0 + 0.5);
      R->mu[j] = (uint16_t)((m << 8) | (255 - m));
    }
  }

  /* A version-2 artifact stores M/U rather than a beta, so unpack it into the
   * beta plane every rule below already reads, plus a depth plane that did not
   * exist before. The extra byte per (class, CpG) is memory only -- on disk the
   * artifact is unchanged in size, which is the point of packing two counts
   * into the two bytes a beta used to occupy. */
  fprintf(stderr,
    "[methscope] deconv: reference v%u, %u classes x %llu kept rows "
    "(%.0f MB%s), qfilter %.2f,%.2f\n", ver,
    R->n_class, (unsigned long long)R->n_keep,
    (double)((size_t)R->n_class * R->n_keep * 2) / 1e6,
    ver == 2 ? " incl. depth" : ", NO depth: --rescue-min-depth inactive",
    R->qlo, R->qhi);
}

static void d2ref_free(d2ref_t *R) {
  for (uint32_t k = 0; k < R->n_class; ++k) free(R->name[k]);
  free(R->name); free(R->row); free(R->mu);
}

/* What THIS query measured, as a bit per kept row.
 *
 * The reference betas are never copied or rewritten -- they are read-only and
 * shared across every query and (later) every thread, so a per-query view has
 * to be a separate, small thing. A bit per kept row is 990 KB for 7.9 M rows
 * against 47 MB for an index-plus-beta pair, and it is allocated ONCE and
 * cleared per record rather than churned. Word-at-a-time skipping makes the
 * sparse case cheap: at 2^16 coverage almost every word is zero.
 *
 * `qb[j]` is only meaningful where bit j is set. */
typedef struct {
  uint64_t  n;          /* measured rows */
  uint64_t *mask;       /* (n_keep + 63) / 64 words */
  uint16_t *qb;         /* n_keep, valid where the bit is set */
  uint64_t  nw;
} d2q_t;

static void d2q_init(const d2ref_t *R, d2q_t *Q) {
  Q->nw   = (R->n_keep + 63) / 64;
  Q->mask = d2alloc(Q->nw, sizeof(uint64_t), "measured mask");
  Q->qb   = d2alloc(R->n_keep, sizeof(uint16_t), "query betas");
  Q->n    = 0;
}

static void d2q_load(const d2ref_t *R, const cdata_t *c, d2q_t *Q) {
  memset(Q->mask, 0, Q->nw * sizeof(uint64_t));
  Q->n = 0;
  for (uint64_t j = 0; j < R->n_keep; ++j) {
    uint64_t mu = c->fmt == '3' ? f3_get_mu((cdata_t *)c, R->row[j])
                   : (FMT6_IN_UNI(*c, R->row[j]) ?
                       (FMT6_IN_SET(*c, R->row[j]) ? (1ull << 32) : 1ull)
                       : 0);
    if (!mu || !MU2cov(mu)) continue;
    Q->mask[j >> 6] |= 1ull << (j & 63);
    Q->qb[j] = d2_encode(MU2beta(mu));
    ++Q->n;
  }
}

static void d2q_free(d2q_t *Q) { free(Q->mask); free(Q->qb); }

/* ------------------------------------------------------------------ */
/* The panel, over measured rows only.                                 */
/* ------------------------------------------------------------------ */
typedef struct {
  uint64_t *key;              /* n_pat * kw, the binstring over the scope */
  uint64_t *n;                /* measured CpGs: the support of n and qsum */
  uint64_t *rn;               /* CpGs behind rsum (== n unless --global-ref) */
  double   *wscale;           /* per-pattern influence multiplier, 1 by default */
  double   *rsum;             /* n_pat * nall, reference beta sums */
  double   *qsum;             /* n_pat, query beta sums */
  uint32_t  n_pat, kw, nall;
} d2pan_t;

static void d2pan_free(d2pan_t *p) {
  free(p->key); free(p->n); free(p->rn); free(p->wscale);
  free(p->rsum); free(p->qsum);
}

typedef struct { uint32_t *slot; uint64_t mask; uint32_t kw; } d2hash_t;

static uint64_t d2_hash(const uint64_t *k, uint32_t kw) {
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t w = 0; w < kw; ++w) { h ^= k[w]; h *= 1099511628211ULL; }
  return h;
}

static uint32_t d2_intern(d2pan_t *p, d2hash_t *H, const uint64_t *k,
                          uint32_t *cap) {
  uint64_t i = d2_hash(k, p->kw) & H->mask;
  while (H->slot[i] != UINT32_MAX) {
    uint32_t r = H->slot[i];
    if (!memcmp(p->key + (size_t)r * p->kw, k, p->kw * sizeof(uint64_t)))
      return r;
    i = (i + 1) & H->mask;
  }
  if (p->n_pat == *cap) {
    *cap *= 2;
    p->key  = realloc(p->key, (size_t)*cap * p->kw * sizeof(uint64_t));
    p->n    = realloc(p->n, (size_t)*cap * sizeof(uint64_t));
    p->rn   = realloc(p->rn, (size_t)*cap * sizeof(uint64_t));
    p->wscale = realloc(p->wscale, (size_t)*cap * sizeof(double));
    p->rsum = realloc(p->rsum, (size_t)*cap * p->nall * sizeof(double));
    p->qsum = realloc(p->qsum, (size_t)*cap * sizeof(double));
    if (!p->key || !p->n || !p->rn || !p->rsum || !p->qsum)
      d2die("out of memory", "panel grow");
    memset(p->n + p->n_pat, 0, (*cap - p->n_pat) * sizeof(uint64_t));
    memset(p->rn + p->n_pat, 0, (*cap - p->n_pat) * sizeof(uint64_t));
    for (uint32_t z = p->n_pat; z < *cap; ++z) p->wscale[z] = 1.0;
    memset(p->rsum + (size_t)p->n_pat * p->nall, 0,
           (size_t)(*cap - p->n_pat) * p->nall * sizeof(double));
    memset(p->qsum + p->n_pat, 0, (*cap - p->n_pat) * sizeof(double));
  }
  uint32_t r = p->n_pat++;
  memcpy(p->key + (size_t)r * p->kw, k, p->kw * sizeof(uint64_t));
  H->slot[i] = r;
  return r;
}

/* Build the pattern set for class subset `sel` over the query's MEASURED rows.
 *
 * There is no unmeasured branch: the loop runs over Q->n, which is already the
 * intersection. So a pattern's reference mean and its observed mean are means
 * over the SAME CpGs, and `n` is the honest support of both. */
/* One admitted CpG awaiting the per-binstring --delta-mean-top selection.
 * `gap` is the class gap in ENCODED beta units -- the lowest 1-side class minus
 * the highest 0-side class -- so a larger gap is a cleaner separator. */
typedef struct { uint32_t r; uint32_t j; float gap; } d2ent_t;

static int d2ent_cmp(const void *a, const void *b) {
  const d2ent_t *x = a, *y = b;
  if (x->r != y->r) return x->r < y->r ? -1 : 1;
  if (x->gap != y->gap) return x->gap > y->gap ? -1 : 1;   /* best first */
  return 0;
}

/* `weak` (NULL = off) marks class pairs the panel could not separate, as an
 * n_class x n_class flag map; `rlo`/`rhi` is the band those CpGs are rescued
 * under. See the rescue block in the scan for what it does and why. */
static void d2pan_build(const d2ref_t *R, const d2q_t *Q,
                        const uint32_t *sel, uint32_t ns, uint64_t dm_top,
                        int global_ref, const uint8_t *weak,
                        double rlo, double rhi, const float *rgap_of,
                        double rgap, uint32_t rdepth, d2pan_t *out) {
  const uint32_t kw = (ns + 63) >> 6;
  memset(out, 0, sizeof *out);
  out->kw = kw; out->nall = R->n_class;
  uint32_t cap = 4096;
  out->key  = d2alloc((size_t)cap * kw, sizeof(uint64_t), "keys");
  out->n    = d2alloc(cap, sizeof(uint64_t), "counts");
  out->rn   = d2alloc(cap, sizeof(uint64_t), "reference counts");
  out->wscale = d2alloc(cap, sizeof(double), "weight scale");
  for (uint32_t z = 0; z < cap; ++z) out->wscale[z] = 1.0;
  out->rsum = d2alloc((size_t)cap * R->n_class, sizeof(double), "ref sums");
  out->qsum = d2alloc(cap, sizeof(double), "query sums");

  d2hash_t H;
  H.kw = kw; H.mask = (1ull << 18) - 1;
  H.slot = malloc((H.mask + 1) * sizeof(uint32_t));
  if (!H.slot) d2die("out of memory", "pattern hash");
  memset(H.slot, 0xFF, (H.mask + 1) * sizeof(uint32_t));

  uint64_t *k = d2alloc(kw, sizeof(uint64_t), "key scratch");
  const uint16_t lo = d2_encode(R->qlo), hi = d2_encode(R->qhi),
                 bt = d2_encode(R->beta_thr);
  const uint16_t rlo_e = d2_encode(rlo), rhi_e = d2_encode(rhi);
  const int32_t  rgap_e = (int32_t)(rgap * 65534.0 + 0.5);
  const int gap_mode = (rgap_e > 0) || (rgap_of != NULL);
  d2ent_t *ent = NULL; size_t n_ent = 0, cap_ent = 0;

  for (uint64_t w = 0; w < Q->nw; ++w) {
    uint64_t bits = Q->mask[w];
    while (bits) {                       /* only measured rows are visited */
    const uint64_t j = (w << 6) + (uint64_t)__builtin_ctzll(bits);
    bits &= bits - 1;
    memset(k, 0, kw * sizeof(uint64_t));
    uint32_t n1 = 0;
    int ok = 1;
    uint32_t min1 = 0xFFFFFFFFu, max0 = 0;      /* for the delta-mean gap */
    for (uint32_t s = 0; s < ns; ++s) {
      uint16_t b = d2_b(R->mu[(size_t)sel[s] * R->n_keep + j]);
      if (b == D2_BETA_NA) { ok = 0; break; }        /* uncovered: reject */
      if (b > bt) {
        if (b < hi) { ok = 0; break; }               /* 1-side, fails HI */
        k[s >> 6] |= 1ull << (s & 63);
        ++n1;
        if (b < min1) min1 = b;
      } else {
        if (b > max0) max0 = b;
        /* b == bt is M == U, not a call; an ambiguous class is absent here. */
        if (b == bt || b > lo) { ok = 0; break; }
      }
    }
    /* PER-PAIR RESCUE.
     *
     * A CpG that cleanly splits a starved pair is usually rejected for a reason
     * that has nothing to do with that pair: some THIRD class sits in the
     * ambiguous band there. On this 2^22 mixture T.Cell.CD4 vs T.Cell.CD8 has
     * 948 measured CpGs of its own and the 6-class conjunction admits 419 --
     * the other 529 are lost to a third class, not to any doubt about the pair.
     * Those 419 then carry 1.5% of the panel's CpGs but 53% of the residual,
     * and the fit buys a lower SSE by inventing 0.099 of an absent CD4.
     *
     * So when a CpG confidently splits a weak pair under the NORMAL band, the
     * bar is lowered for the REST of the conjunction only. Every class still
     * gets a digit, so the pattern keeps its full context -- this stays one
     * MRMP set, not a 2-class satellite bolted alongside it. */
    if (!ok && weak) {
      /* Does this CpG carry evidence about a weak pair?
       *
       * With --rescue-gap the test is |beta_a - beta_b| alone, with no
       * reference to the 0/1 band. That is the quantity the SOLVE uses: two
       * classes are separable in the fit when their profile means differ, not
       * when their binstring digits differ. A CpG at CD4 0.10 / CD8 0.50
       * separates the pair by 0.40 and the band discards it because 0.50 is
       * "no call" -- a rejection that says nothing about the pair. The pair's
       * own classes are then exempt from the band on the retest, since their
       * admission is justified by the gap; every other class must still clear
       * the wider band, so the pattern keeps its context. */
      int splits = 0;
      for (uint32_t a = 0; a < ns && !splits; ++a) {
        uint16_t ba = d2_b(R->mu[(size_t)sel[a] * R->n_keep + j]);
        if (ba == D2_BETA_NA) continue;
        for (uint32_t b2 = a + 1; b2 < ns; ++b2) {
          if (!weak[(size_t)sel[a] * R->n_class + sel[b2]]) continue;
          uint16_t bb = d2_b(R->mu[(size_t)sel[b2] * R->n_keep + j]);
          if (bb == D2_BETA_NA) continue;
          if (gap_mode) {
            /* per-pair threshold when one was fitted, else the flat one */
            int32_t thr = rgap_e;
            if (rgap_of) {
              float g = rgap_of[(size_t)sel[a] * R->n_class + sel[b2]];
              if (g <= 0) continue;         /* this pair was not rescued */
              thr = (int32_t)(g * 65534.0f + 0.5f);
            }
            /* The gap must be BELIEVABLE, not merely large: a 0.4 difference
             * read off two reads is noise, off thirty it is real. The floor
             * applies to the PAIR only -- the other classes are along for
             * context and a thin one there costs nothing. This is the whole
             * reason the artifact carries M/U instead of a bare beta. */
            if (rdepth && R->has_depth &&
                ((uint32_t)d2_dep(R->mu[(size_t)sel[a] * R->n_keep + j]) < rdepth ||
                 (uint32_t)d2_dep(R->mu[(size_t)sel[b2] * R->n_keep + j]) < rdepth))
              continue;
            int32_t d = (int32_t)ba - (int32_t)bb;
            if (d < 0) d = -d;
            if (d >= thr) { splits = 1; break; }
          } else {
            int ca = (ba >= hi && ba > bt) ? 1 : ((ba <= lo && ba < bt) ? 0 : -1);
            int cb = (bb >= hi && bb > bt) ? 1 : ((bb <= lo && bb < bt) ? 0 : -1);
            if (ca >= 0 && cb >= 0 && ca != cb) { splits = 1; break; }
          }
        }
      }
      if (splits) {
        memset(k, 0, kw * sizeof(uint64_t));
        n1 = 0; ok = 1;
        for (uint32_t s = 0; s < ns; ++s) {
          uint16_t b = d2_b(R->mu[(size_t)sel[s] * R->n_keep + j]);
          if (b == D2_BETA_NA) { ok = 0; break; }
          int exempt = 0;                 /* a member of some weak pair */
          if (gap_mode)
            for (uint32_t t2 = 0; t2 < ns && !exempt; ++t2)
              if (weak[(size_t)sel[s] * R->n_class + sel[t2]]) exempt = 1;
          if (b > bt) {
            if (!exempt && b < rhi_e) { ok = 0; break; }
            k[s >> 6] |= 1ull << (s & 63);
            ++n1;
          } else if (!exempt && (b == bt || b > rlo_e)) { ok = 0; break; }
        }
      }
    }
    if (!ok) continue;
    if (!n1 || n1 == ns) continue;   /* constant over the scope: separates nothing */


    uint32_t r = d2_intern(out, &H, k, &cap);
    if (dm_top) {
      /* defer: which CpGs are this binstring's cleanest is not known until the
       * whole measured set has been seen */
      if (n_ent == cap_ent) {
        cap_ent = cap_ent ? cap_ent * 2 : 65536;
        ent = realloc(ent, cap_ent * sizeof(d2ent_t));
        if (!ent) d2die("out of memory", "delta-mean entries");
      }
      ent[n_ent].r = r; ent[n_ent].j = (uint32_t)j;
      ent[n_ent].gap = (float)((double)min1 - (double)max0);
      ++n_ent;
      continue;
    }
    out->n[r]++;
    if (!global_ref) {
      out->rn[r]++;
      double *rs = out->rsum + (size_t)r * out->nall;
      for (uint32_t c = 0; c < out->nall; ++c) {
        uint16_t b = d2_b(R->mu[(size_t)c * R->n_keep + j]);
        if (b != D2_BETA_NA) rs[c] += (double)b / 65534.0;
      }
    }
    out->qsum[r] += (double)Q->qb[j] / 65534.0;
    }
  }

  /* GENOME-WIDE REFERENCE PROFILE.
   *
   * n and qsum must come from the CpGs the query measured -- that is what the
   * observed value IS. But the reference side need not, and should not: the
   * mean over ~200 measured CpGs is a noisy estimate of the pattern's true
   * profile, and noise in the DESIGN MATRIX attenuates coefficients toward zero
   * rather than averaging out, so a class whose column is noisier loses mass to
   * a correlated class whose column is cleaner. That is the shape of the
   * one-directional T.Cell.CD8 -> T.Cell.CD4 leak: present in the plain global
   * fit, shrinking as more pair CpGs are admitted, and untouched by every
 * downstream knob.
   *
   * The legacy .refx has no such noise -- its design values were computed once,
   * at build time, over every reference CpG in the pattern. This second pass
   * reproduces that: same binstrings, but the profile is the genome-wide mean.
   * Patterns with no measured CpG were never interned, so they stay absent. */
  if (global_ref) {
    for (uint64_t j = 0; j < R->n_keep; ++j) {
      memset(k, 0, kw * sizeof(uint64_t));
      uint32_t n1 = 0; int ok = 1;
      for (uint32_t s = 0; s < ns; ++s) {
        uint16_t b = d2_b(R->mu[(size_t)sel[s] * R->n_keep + j]);
        if (b == D2_BETA_NA) { ok = 0; break; }
        if (b > bt) {
          if (b < hi) { ok = 0; break; }
          k[s >> 6] |= 1ull << (s & 63);
          ++n1;
        } else if (b == bt || b > lo) { ok = 0; break; }
      }
      if (!ok || !n1 || n1 == ns) continue;
      uint64_t h = d2_hash(k, kw) & H.mask;
      uint32_t r = UINT32_MAX;
      while (H.slot[h] != UINT32_MAX) {
        uint32_t cand = H.slot[h];
        if (!memcmp(out->key + (size_t)cand * kw, k, kw * sizeof(uint64_t))) {
          r = cand; break;
        }
        h = (h + 1) & H.mask;
      }
      if (r == UINT32_MAX) continue;      /* no measured CpG: not in the panel */
      out->rn[r]++;
      double *rs = out->rsum + (size_t)r * out->nall;
      for (uint32_t c = 0; c < out->nall; ++c) {
        uint16_t b = d2_b(R->mu[(size_t)c * R->n_keep + j]);
        if (b != D2_BETA_NA) rs[c] += (double)b / 65534.0;
      }
    }
  }

  /* --delta-mean-top: per BINSTRING keep only the N measured CpGs with the
   * cleanest class separation. This caps a pattern's influence -- which is
   * linear in its CpG count -- AND improves the pattern itself, since its beta
   * then averages its best separators instead of everything admissible. A
   * weight cap would have done only the first. */
  if (dm_top) {
    qsort(ent, n_ent, sizeof(d2ent_t), d2ent_cmp);
    for (size_t a = 0; a < n_ent; ) {
      size_t b = a;
      while (b < n_ent && ent[b].r == ent[a].r) ++b;
      size_t take = (b - a) < dm_top ? (b - a) : dm_top;
      for (size_t z = 0; z < take; ++z) {
        uint32_t r = ent[a + z].r; uint32_t j = ent[a + z].j;
        out->n[r]++; out->rn[r]++;
        double *rs = out->rsum + (size_t)r * out->nall;
        for (uint32_t c = 0; c < out->nall; ++c) {
          uint16_t bb = d2_b(R->mu[(size_t)c * R->n_keep + j]);
          if (bb != D2_BETA_NA) rs[c] += (double)bb / 65534.0;
        }
        out->qsum[r] += (double)Q->qb[j] / 65534.0;
      }
      a = b;
    }
    free(ent);
  }
  free(k); free(H.slot);
}

/* Weighted NNLS over the panel. A pattern's observed beta is a mean over `n`
 * measured CpGs, so its variance goes as 1/n and the row weight is sqrt(n) --
 * which the solve then squares, making influence linear in n. That is correct
 * for sampling noise and is the whole reason weighting doubled accuracy; the
 * per-binstring cap below is what stops it running away. */
static void d2_solve(const d2pan_t *p, const uint32_t *sel, uint32_t ns,
                     uint64_t min_n, double expo, double *x) {
  uint32_t nu = 0;
  for (uint32_t r = 0; r < p->n_pat; ++r) if (p->n[r] >= min_n) ++nu;
  for (uint32_t s = 0; s < ns; ++s) x[s] = 0;
  if (!nu) return;

  double *A = d2alloc((size_t)nu * ns, sizeof(double), "design");
  double *b = d2alloc(nu, sizeof(double), "observed");
  uint32_t q = 0;
  for (uint32_t r = 0; r < p->n_pat; ++r) {
    if (p->n[r] < min_n) continue;
    const double *rs = p->rsum + (size_t)r * p->nall;
    /* n^expo on the row; the solve squares it, so influence is n^(2*expo).
     * 0.5 is the precision weight (influence linear in n), 0 is unweighted. */
    double w = (expo == 0.5) ? sqrt((double)p->n[r])
             : (expo == 0.0) ? 1.0 : pow((double)p->n[r], expo);
    if (p->wscale && p->wscale[r] != 1.0) w *= sqrt(p->wscale[r]);
    for (uint32_t s = 0; s < ns; ++s)              /* column-major for nnls */
      A[(size_t)s * nu + q] = rs[sel[s]] / (double)(p->rn[r] ? p->rn[r] : p->n[r]) * w;
    b[q] = p->qsum[r] / (double)p->n[r] * w;
    ++q;
  }
  double *aw = d2alloc((size_t)nu * ns, sizeof(double), "nnls A");
  double *bw = d2alloc(nu, sizeof(double), "nnls b");
  double *w  = d2alloc(ns, sizeof(double), "nnls w");
  double *zz = d2alloc(nu, sizeof(double), "nnls zz");
  int    *ix = d2alloc(ns, sizeof(int), "nnls index");
  memcpy(aw, A, (size_t)nu * ns * sizeof(double));
  memcpy(bw, b, (size_t)nu * sizeof(double));
  int mda = (int)nu, m = (int)nu, n = (int)ns, mode = 0;
  double rnorm = 0;
  nnls_c(aw, &mda, &m, &n, bw, x, &rnorm, w, zz, ix, &mode);
  if (mode != 1)
    fprintf(stderr, "[methscope] deconv: NNLS mode %d (%u patterns)\n",
            mode, nu);
  double sum = 0;
  for (uint32_t s = 0; s < ns; ++s) { if (x[s] < 0) x[s] = 0; sum += x[s]; }
  if (sum > 0) for (uint32_t s = 0; s < ns; ++s) x[s] /= sum;
  free(A); free(b); free(aw); free(bw); free(w); free(zz); free(ix);
}

/* seg[a*n+b] = measured CpGs the GLOBAL PANEL offers for telling a from b:
 * the CpG counts of every pattern whose binstring puts them on opposite sides.
 *
 * Judged on the panel, NOT on the raw pair. The pair on its own may be richly
 * separated -- CD4 vs CD8 has 217 measured CpGs on t1_4_32 -- while the global
 * panel admits almost none of them, because admission needs all 33 classes
 * callable at the same CpG and 92% of the pair evidence fails that. The scope
 * exists precisely to catch that case, so it has to ask what the GLOBAL fit
 * could see; the narrower rebuild is what then recovers the rest. Asking the
 * raw pair instead would report the pair as well separated and never widen.
 *
 * Cheap: n_pat * n_class^2 over a few hundred patterns, no genome pass. */
static void d2_seg_matrix_min(const d2ref_t *R, const d2pan_t *P,
                              const uint32_t *sel, uint32_t ns,
                              uint64_t min_n, uint32_t *seg) {
  const uint32_t n = R->n_class;
  memset(seg, 0, (size_t)n * n * sizeof(uint32_t));
  for (uint32_t r = 0; r < P->n_pat; ++r) {
    if (P->n[r] < min_n) continue;      /* the solve ignores these */
    const uint64_t *k = P->key + (size_t)r * P->kw;
    for (uint32_t a = 0; a < ns; ++a) {
      int ba = (k[a >> 6] >> (a & 63)) & 1;
      for (uint32_t b = a + 1; b < ns; ++b) {
        int bb = (k[b >> 6] >> (b & 63)) & 1;
        if (ba == bb) continue;
        /* `a`/`b` are BIT POSITIONS in this panel's scope; seg is indexed by
         * global class id, and the two only coincide when the scope is every
         * class in order. */
        seg[(size_t)sel[a] * n + sel[b]] += (uint32_t)P->n[r];
        seg[(size_t)sel[b] * n + sel[a]] += (uint32_t)P->n[r];
      }
    }
  }
}

static void d2_seg_matrix(const d2ref_t *R, const d2pan_t *P,
                          const uint32_t *sel, uint32_t ns, uint32_t *seg) {
  d2_seg_matrix_min(R, P, sel, ns, 0, seg);
}

/* Scope = classes carrying mass, closed under "this query cannot separate
 * them". A class sitting at exactly 0.000 must stay if it is inseparable from
 * one that has mass: NNLS hands out arbitrary exact zeros among collinear
 * columns, so on t1_4_32 CD8 came out 0.000 only because CD4 took its mass.
 * Closing the relation makes the scope a union of connected components. */
/* `allowed` (NULL = every class) bounds which classes may enter. Round 2 must
 * pass the round-1 scope: its seg matrix is built from a panel over the scope
 * alone, so every pair touching a dropped class is 0 -- meaning NOT MEASURED,
 * not "inseparable". Without the bound the rule reads those zeros as perfect
 * confusion and re-admits all 33 classes. The scope narrows monotonically,
 * which is also the premise: a rebuild only ever adds evidence. */
static uint32_t d2_scope(const d2ref_t *R, const double *x, const uint32_t *seg,
                         double mass_floor, uint64_t max_seg,
                         const uint8_t *allowed, uint8_t *in) {
  const uint32_t n = R->n_class;
  uint32_t n_in = 0;
  for (uint32_t c = 0; c < n; ++c) {
    in[c] = (x[c] >= mass_floor && (!allowed || allowed[c])) ? 1 : 0;
    n_in += in[c];
  }
  int grew = 1;
  while (grew) {
    grew = 0;
    for (uint32_t a = 0; a < n; ++a) {
      if (!in[a]) continue;
      for (uint32_t b = 0; b < n; ++b) {
        if (in[b] || a == b) continue;
        if (allowed && !allowed[b]) continue;
        if ((uint64_t)seg[(size_t)a * n + b] <= max_seg) {
          in[b] = 1; ++n_in; grew = 1;
        }
      }
    }
  }
  return n_in;
}

/* Everything one record needs beyond the read-only reference. One workspace
 * per THREAD, reused across the records that thread handles -- the buffers are
 * sized by the reference (the query beta array alone is 2 bytes x 7.9 M kept
 * rows), so allocating them per record would dominate the runtime. */
typedef struct {
  d2q_t     Q;
  uint32_t *sel, *sel2, *seg, *seg2, *segu;
  uint8_t  *inscope, *inscope2, *weak;
  float    *gap_of;
  double   *x, *x2, *xf, *xadj;
} d2ws_t;

static void d2ws_init(const d2ref_t *R, d2ws_t *w) {
  d2q_init(R, &w->Q);
  w->sel  = d2alloc(R->n_class, sizeof(uint32_t), "scope");
  for (uint32_t s = 0; s < R->n_class; ++s) w->sel[s] = s;
  w->sel2 = d2alloc(R->n_class, sizeof(uint32_t), "scope classes");
  w->seg  = d2alloc((size_t)R->n_class * R->n_class, sizeof(uint32_t), "seg");
  w->seg2 = d2alloc((size_t)R->n_class * R->n_class, sizeof(uint32_t), "seg2");
  w->segu = d2alloc((size_t)R->n_class * R->n_class, sizeof(uint32_t), "segu");
  w->inscope  = d2alloc(R->n_class, 1, "scope");
  w->inscope2 = d2alloc(R->n_class, 1, "scope2");
  w->weak     = d2alloc((size_t)R->n_class * R->n_class, 1, "weak pairs");
  w->gap_of   = d2alloc((size_t)R->n_class * R->n_class, sizeof(float), "per-pair gaps");
  w->x    = d2alloc(R->n_class, sizeof(double), "proportions");
  w->x2   = d2alloc(R->n_class, sizeof(double), "round proportions");
  w->xf   = d2alloc(R->n_class, sizeof(double), "full vector");
  w->xadj = d2alloc(R->n_class, sizeof(double), "final solve");
}

static void d2ws_free(d2ws_t *w) {
  d2q_free(&w->Q);
  free(w->sel); free(w->sel2); free(w->seg); free(w->seg2); free(w->segu);
  free(w->inscope); free(w->inscope2); free(w->weak);
  free(w->gap_of);
  free(w->x); free(w->x2); free(w->xf); free(w->xadj);
}

typedef struct {
  uint64_t min_n, dm_top, max_seg, rescue_below;
  double   rescue_lo, rescue_hi, rescue_gap, rescue_floor;
  uint64_t rescue_target;
  uint32_t rescue_depth;
  double   mass_floor;
  uint32_t max_round;
  int      narrow, verbose, global_ref;
  double   wexp[16];
  int      n_wexp;
  const char *pair_spec, *panel_out, *scope_out, *force_scope, *eval_x;
  const char *design_out;
} d2opt_t;

/* The weighted least-squares objective at an ARBITRARY composition.
 *
 * Exactly what d2_solve minimises: row r contributes w^2 (A_r.x - b_r)^2 with
 * w = n^expo, so influence is linear in n at the default expo. Evaluating it at
 * a hand-specified x answers the question NNLS cannot: is the fitted answer
 * wrong because the solver missed the optimum, or because the panel genuinely
 * scores the wrong composition better? Only the second is a modelling defect. */
static double d2_sse(const d2pan_t *p, const uint32_t *sel, uint32_t ns,
                     uint64_t min_n, double expo, const double *xfull) {
  double sse = 0;
  for (uint32_t r = 0; r < p->n_pat; ++r) {
    if (p->n[r] < min_n) continue;
    const double *rs = p->rsum + (size_t)r * p->nall;
    double den = (double)(p->rn[r] ? p->rn[r] : p->n[r]);
    double pred = 0;
    for (uint32_t s = 0; s < ns; ++s) pred += rs[sel[s]] / den * xfull[sel[s]];
    double obs = p->qsum[r] / (double)p->n[r];
    double w = (expo == 0.5) ? sqrt((double)p->n[r])
             : (expo == 0.0) ? 1.0 : pow((double)p->n[r], expo);
    if (p->wscale && p->wscale[r] != 1.0) w *= sqrt(p->wscale[r]);
    double d = w * (pred - obs);
    sse += d * d;
  }
  return sse;
}

/* Choose each weak pair's gap threshold from the evidence it actually has.
 *
 * A single threshold cannot serve every pair: loosening from the 0.30/0.70
 * band to no band at all gained Macrophage|Monocyte 5.6x (244 -> 1,361 CpGs)
 * but T.Cell.CD4|T.Cell.CD8 only 1.9x (413 -> 784). One number therefore
 * over-relaxes the pairs that respond quickly long before the starved ones
 * have enough, which is why the flat band sweep turned over.
 *
 * So: histogram |beta_a - beta_b| per weak pair in ONE pass, then read each
 * pair's threshold off the cumulative-from-the-top -- the count-versus-cutoff
 * curve is exactly that cumulative, so no search is needed. Relaxation stops
 * at whichever comes first, the target count or the floor. The floor matters:
 * without it a pair with no real evidence would relax until it was admitting
 * 0.02-gap noise, and poison the panel it was meant to help. */
#define D2_GBINS 64
static void d2_fit_gaps(const d2ref_t *R, const d2q_t *Q,
                        const uint32_t *sel, uint32_t ns, const uint8_t *weak,
                        uint64_t target, double floor_gap, uint32_t rdepth,
                        float *gap_of, int verbose) {
  uint32_t np = 0;
  uint32_t pa[64], pb[64];
  for (uint32_t a = 0; a < ns && np < 64; ++a)
    for (uint32_t b = a + 1; b < ns && np < 64; ++b)
      if (weak[(size_t)sel[a] * R->n_class + sel[b]]) {
        pa[np] = a; pb[np] = b; ++np;
      }
  if (!np) return;
  uint64_t *hist = d2alloc((size_t)np * D2_GBINS, sizeof(uint64_t), "gap hist");
  for (uint64_t w = 0; w < Q->nw; ++w) {
    uint64_t bits = Q->mask[w];
    while (bits) {
      const uint64_t j = (w << 6) + (uint64_t)__builtin_ctzll(bits);
      bits &= bits - 1;
      for (uint32_t p = 0; p < np; ++p) {
        uint16_t va = R->mu[(size_t)sel[pa[p]] * R->n_keep + j];
        uint16_t vb = R->mu[(size_t)sel[pb[p]] * R->n_keep + j];
        if (!va || !vb) continue;
        if (rdepth && R->has_depth &&
            ((uint32_t)d2_dep(va) < rdepth ||
             (uint32_t)d2_dep(vb) < rdepth)) continue;
        int32_t d = (int32_t)d2_b(va) - (int32_t)d2_b(vb);
        if (d < 0) d = -d;
        uint32_t bin = (uint32_t)((double)d / 65534.0 * D2_GBINS);
        if (bin >= D2_GBINS) bin = D2_GBINS - 1;
        hist[(size_t)p * D2_GBINS + bin]++;
      }
    }
  }
  const uint32_t floor_bin = (uint32_t)(floor_gap * D2_GBINS);
  for (uint32_t p = 0; p < np; ++p) {
    uint64_t cum = 0; uint32_t chosen = D2_GBINS - 1;
    for (int32_t bin = D2_GBINS - 1; bin >= (int32_t)floor_bin; --bin) {
      cum += hist[(size_t)p * D2_GBINS + bin];
      chosen = (uint32_t)bin;
      if (cum >= target) break;
    }
    float g = (float)chosen / (float)D2_GBINS;
    gap_of[(size_t)sel[pa[p]] * R->n_class + sel[pb[p]]] = g;
    gap_of[(size_t)sel[pb[p]] * R->n_class + sel[pa[p]]] = g;
    if (verbose)
      fprintf(stderr, "[methscope]     %s|%s gap %.3f -> %llu CpGs%s\n",
              R->name[sel[pa[p]]], R->name[sel[pb[p]]], g,
              (unsigned long long)cum, cum < target ? " (floor reached)" : "");
  }
  free(hist);
}


/* Deconvolve ONE record. Reads nothing global: the reference is const and
 * shared, everything mutable lives in `w`. */
static void d2_record(const d2ref_t *Rr, const d2opt_t *o, d2ws_t *w,
                      const cdata_t *cc, uint32_t rec, char **qname,
                      uint32_t n_qname, double *xout) {
  const d2ref_t R = *Rr;
  d2q_t Q = w->Q;                 /* shares the mask/beta buffers */
  const cdata_t c = *cc;
  uint32_t *sel = w->sel, *sel2 = w->sel2;
  uint32_t *seg = w->seg, *seg2 = w->seg2, *segu = w->segu;
  uint8_t *inscope = w->inscope, *inscope2 = w->inscope2;
  uint8_t *weak = w->weak;
  float *gap_of = w->gap_of;
  double *x = w->x, *x2 = w->x2, *xf = w->xf, *xadj = w->xadj;
  const uint64_t min_n = o->min_n, dm_top = o->dm_top, max_seg = o->max_seg,
                 rescue_below = o->rescue_below;
  const double rescue_lo = o->rescue_lo, rescue_hi = o->rescue_hi,
               rescue_gap = o->rescue_gap;
  const uint32_t rescue_depth = o->rescue_depth;
  const uint64_t rescue_target = o->rescue_target;
  const double rescue_floor = o->rescue_floor;
  const double mass_floor = o->mass_floor;
  const uint32_t max_round = o->max_round;
  const int narrow = o->narrow, verbose = o->verbose,
            global_ref = o->global_ref;
  const double *wexp = o->wexp; const int n_wexp = o->n_wexp;
  const char *pair_spec = o->pair_spec;
  (void)segu;

    d2q_load(&R, &c, &Q);

    /* How much evidence does this query hold about ONE pair, before and after
     * the all-classes admission conjunction? The first number is the pair's
     * own segregating power -- the two classes on opposite sides of the band,
     * every other class ignored. The second is what survives requiring all
     * n_class classes to make a call at the same CpG. The gap is what the
     * conjunction costs, and it is not the pair's fault. */
    if (pair_spec) {
      char spec[256]; snprintf(spec, sizeof spec, "%s", pair_spec);
      char *comma = strchr(spec, ',');
      if (!comma) d2die("--pair-count wants CLASS_A,CLASS_B", pair_spec);
      *comma = '\0';
      uint32_t ia = R.n_class, ib = R.n_class;
      for (uint32_t s2 = 0; s2 < R.n_class; ++s2) {
        if (!strcmp(R.name[s2], spec))     ia = s2;
        if (!strcmp(R.name[s2], comma + 1)) ib = s2;
      }
      if (ia == R.n_class) d2die("no such class", spec);
      if (ib == R.n_class) d2die("no such class", comma + 1);
      const uint16_t plo = d2_encode(R.qlo), phi = d2_encode(R.qhi),
                     pbt = d2_encode(R.beta_thr);
      uint64_t pair_only = 0, pair_admitted = 0;
      for (uint64_t w = 0; w < Q.nw; ++w) {
        uint64_t bits = Q.mask[w];
        while (bits) {
          const uint64_t j = (w << 6) + (uint64_t)__builtin_ctzll(bits);
          bits &= bits - 1;
          uint16_t ba = d2_b(R.mu[(size_t)ia * R.n_keep + j]);
          uint16_t bb = d2_b(R.mu[(size_t)ib * R.n_keep + j]);
          if (ba == D2_BETA_NA || bb == D2_BETA_NA) continue;
          int ca = (ba >= phi && ba > pbt) ? 1 : (ba <= plo && ba < pbt) ? 0 : -1;
          int cb = (bb >= phi && bb > pbt) ? 1 : (bb <= plo && bb < pbt) ? 0 : -1;
          if (ca < 0 || cb < 0 || ca == cb) continue;
          ++pair_only;
          int all_ok = 1;
          for (uint32_t s2 = 0; s2 < R.n_class && all_ok; ++s2) {
            uint16_t b2 = d2_b(R.mu[(size_t)s2 * R.n_keep + j]);
            if (b2 == D2_BETA_NA) all_ok = 0;
            else if (!((b2 >= phi && b2 > pbt) || (b2 <= plo && b2 < pbt)))
              all_ok = 0;
          }
          if (all_ok) ++pair_admitted;
        }
      }
      fprintf(stderr,
        "[methscope] deconv: %s: %s vs %s -- %llu measured CpGs segregate the "
        "PAIR; %llu of them also pass the %u-class conjunction (%.1f%% lost)\n",
        rec < n_qname ? qname[rec] : "record", spec, comma + 1,
        (unsigned long long)pair_only, (unsigned long long)pair_admitted,
        R.n_class,
        pair_only ? 100.0 * (double)(pair_only - pair_admitted) / (double)pair_only : 0.0);
    }
    d2pan_t P; d2pan_build(&R, &Q, sel, R.n_class, dm_top, global_ref, NULL, 0, 0, NULL, 0, 0, &P);
    d2_seg_matrix(&R, &P, sel, R.n_class, seg);
    d2_solve(&P, sel, R.n_class, min_n, wexp[0], x);

    if (verbose) {
      uint64_t tot = 0;
      for (uint32_t r = 0; r < P.n_pat; ++r) tot += P.n[r];
      fprintf(stderr,
        "[methscope] deconv: %s: %llu measured rows -> %u patterns over "
        "%llu CpGs\n",
        rec < n_qname ? qname[rec] : "record",
        (unsigned long long)Q.n, P.n_pat, (unsigned long long)tot);
    }
    {
      uint32_t n_in = d2_scope(&R, x, seg, mass_floor, max_seg, NULL, inscope);
      /* --force-scope: skip discovery and use exactly these classes. A
       * diagnostic, not a mode -- it answers whether the narrow panel is weak
       * because the scope is wrong or because a narrow panel is weak. */
      if (o->force_scope) {
        char fs[1024]; snprintf(fs, sizeof fs, "%s", o->force_scope);
        for (uint32_t c = 0; c < R.n_class; ++c) inscope[c] = 0;
        n_in = 0;
        for (char *tok = strtok(fs, ","); tok; tok = strtok(NULL, ",")) {
          uint32_t hit = R.n_class;
          for (uint32_t c = 0; c < R.n_class; ++c)
            if (!strcmp(R.name[c], tok)) hit = c;
          if (hit == R.n_class) d2die("--force-scope: no such class", tok);
          if (!inscope[hit]) { inscope[hit] = 1; ++n_in; }
        }
      }
      if (!narrow) {
        for (uint32_t c = 0; c < R.n_class; ++c) inscope[c] = 1;
        n_in = R.n_class;
      }
      if (verbose) {
        fprintf(stderr, "[methscope] deconv: %s: scope %u/%u ->",
                rec < n_qname ? qname[rec] : "record", n_in, R.n_class);
        for (uint32_t c = 0; c < R.n_class; ++c)
          if (inscope[c]) fprintf(stderr, " %s(%.3f)", R.name[c], x[c]);
        fputc('\n', stderr);
      }
      /* ITERATE. Each round rebuilds over the current scope, re-solves, and
       * re-derives the scope from the new mass and that panel's OWN segregating
       * counts. The q-filter is a conjunction, so dropping classes can only
       * ADMIT more CpGs -- every CpG that cleared the wider test still clears a
       * subset test, plus the ones that failed only because a now-dropped class
       * was ambiguous or uncovered there. So each round has strictly more
       * evidence than the last, and the scope narrows monotonically. Stops when
       * the class set stops changing. */
      uint32_t ia = R.n_class, ib = R.n_class;
      if (pair_spec) {
        char sp[256]; snprintf(sp, sizeof sp, "%s", pair_spec);
        char *cm = strchr(sp, ',');
        if (cm) {
          *cm = '\0';
          for (uint32_t c = 0; c < R.n_class; ++c) {
            if (!strcmp(R.name[c], sp))     ia = c;
            if (!strcmp(R.name[c], cm + 1)) ib = c;
          }
        }
      }
      /* --no-narrow: skip the rebuild rounds entirely and keep every class, so
       * the answer is the global fit. Isolates what the narrowing is worth. */
      for (uint32_t round = 2; narrow && !o->force_scope && round <= max_round;
           ++round) {
        uint32_t n_sel2 = 0;
        for (uint32_t c = 0; c < R.n_class; ++c)
          if (inscope[c]) sel2[n_sel2++] = c;
        if (n_sel2 < 2) break;

        d2pan_t P2; d2pan_build(&R, &Q, sel2, n_sel2, dm_top, global_ref,
                                NULL, 0, 0, NULL, 0, 0, &P2);
        d2_seg_matrix_min(&R, &P2, sel2, n_sel2, min_n, seg2);

        /* TWO-PASS REBUILD. The first pass is the ordinary q-filtered panel;
         * it also tells us which pairs it starved. Those pairs then get a
         * second pass in which a CpG that splits one of them is admitted under
         * a wider band for the OTHER classes -- one MRMP set, full context,
         * rather than a 2-class satellite alongside. */
        if (rescue_below) {
          memcpy(segu, seg2, (size_t)R.n_class * R.n_class * sizeof(uint32_t));
          memset(weak, 0, (size_t)R.n_class * R.n_class);
          uint32_t n_weak = 0;
          for (uint32_t a = 0; a < n_sel2; ++a)
            for (uint32_t b2 = a + 1; b2 < n_sel2; ++b2)
              if (seg2[(size_t)sel2[a] * R.n_class + sel2[b2]] < rescue_below) {
                weak[(size_t)sel2[a] * R.n_class + sel2[b2]] = 1;
                weak[(size_t)sel2[b2] * R.n_class + sel2[a]] = 1;
                ++n_weak;
              }
          if (n_weak) {
            memset(gap_of, 0, (size_t)R.n_class * R.n_class * sizeof(float));
            if (rescue_target)
              d2_fit_gaps(&R, &Q, sel2, n_sel2, weak, rescue_target,
                          rescue_floor, rescue_depth, gap_of, verbose);
            uint32_t was_pat = P2.n_pat;
            uint64_t was_cpg = 0;
            for (uint32_t r = 0; r < P2.n_pat; ++r) was_cpg += P2.n[r];
            d2pan_free(&P2);
            d2pan_build(&R, &Q, sel2, n_sel2, dm_top, global_ref,
                        weak, rescue_lo, rescue_hi,
                        rescue_target ? gap_of : NULL, rescue_gap, rescue_depth, &P2);
            d2_seg_matrix_min(&R, &P2, sel2, n_sel2, min_n, seg2);
            if (verbose) {
              uint64_t now_cpg = 0;
              for (uint32_t r = 0; r < P2.n_pat; ++r) now_cpg += P2.n[r];
              fprintf(stderr, "[methscope]   rescue %.2f/%.2f on %u weak pair(s)"
                              ": %u -> %u patterns, %llu -> %llu CpGs\n",
                      rescue_lo, rescue_hi, n_weak, was_pat, P2.n_pat,
                      (unsigned long long)was_cpg, (unsigned long long)now_cpg);
              for (uint32_t a = 0; a < n_sel2; ++a)
                for (uint32_t b2 = a + 1; b2 < n_sel2; ++b2) {
                  uint32_t ga = sel2[a], gb = sel2[b2];
                  if (!weak[(size_t)ga * R.n_class + gb]) continue;
                  fprintf(stderr, "[methscope]     %s|%s seg %u -> %u\n",
                          R.name[ga], R.name[gb],
                          segu[(size_t)ga * R.n_class + gb],
                          seg2[(size_t)ga * R.n_class + gb]);
                }
            }
          }
        }
        double we = wexp[(round - 1) < (uint32_t)n_wexp
                         ? (round - 1) : (uint32_t)(n_wexp - 1)];
          if (o->design_out) {      /* same dump, base panel only */
            FILE *df = fopen(o->design_out, round == 2 ? "w" : "a");
            if (!df) d2die("cannot open --design-out", o->design_out);
            if (round == 2) {
              fprintf(df, "round\tsource\tn\trn\tweight\tq");
              for (uint32_t c = 0; c < R.n_class; ++c)
                if (inscope[c]) fprintf(df, "\t%s", R.name[c]);
              fputc('\n', df);
            }
            for (uint32_t r = 0; r < P2.n_pat; ++r) {
              if (P2.n[r] < min_n) continue;
              double den = (double)(P2.rn[r] ? P2.rn[r] : P2.n[r]);
              double wv = (we == 0.5) ? sqrt((double)P2.n[r])
                        : (we == 0.0) ? 1.0 : pow((double)P2.n[r], we);
              fprintf(df, "%u\tpanel\t%llu\t%llu\t%.4f\t%.6f", round,
                      (unsigned long long)P2.n[r], (unsigned long long)P2.rn[r],
                      wv, P2.qsum[r] / (double)P2.n[r]);
              for (uint32_t s2 = 0; s2 < n_sel2; ++s2)
                fprintf(df, "\t%.6f",
                        P2.rsum[(size_t)r * P2.nall + sel2[s2]] / den);
              fputc('\n', df);
            }
            fclose(df);
          }
          d2_solve(&P2, sel2, n_sel2, min_n, we, x2);
          if (o->eval_x && verbose) {
            /* The objective at a HAND-GIVEN composition, beside the fitted
             * one. It answers the question NNLS cannot: is the answer wrong
             * because the solver missed the optimum, or because the panel
             * genuinely scores the wrong composition better? On t1_2_23 the
             * fitted answer scored 3.83 against the truth's 5.56, which is how
             * the starved-pair diagnosis was pinned down. */
            for (uint32_t c = 0; c < R.n_class; ++c) xf[c] = 0;
            for (uint32_t s2 = 0; s2 < n_sel2; ++s2) xf[sel2[s2]] = x2[s2];
            fprintf(stderr, "[methscope]   SSE fitted            %.6g\n",
                    d2_sse(&P2, sel2, n_sel2, min_n, we, xf));
            char spec[512]; snprintf(spec, sizeof spec, "%s", o->eval_x);
            char *outer = NULL;
            for (char *cand = strtok_r(spec, ";", &outer); cand;
                 cand = strtok_r(NULL, ";", &outer)) {
              double xv[64]; memset(xv, 0, sizeof xv);
              char one[256]; snprintf(one, sizeof one, "%s", cand);
              for (char *tok = strtok(one, ","); tok; tok = strtok(NULL, ",")) {
                char *eq = strchr(tok, '=');
                if (!eq) continue;
                *eq = '\0';
                for (uint32_t c = 0; c < R.n_class && c < 64; ++c)
                  if (!strcmp(R.name[c], tok)) xv[c] = atof(eq + 1);
              }
              fprintf(stderr, "[methscope]   SSE %-20s %.6g\n", cand,
                      d2_sse(&P2, sel2, n_sel2, min_n, we, xv));
            }
          }
        for (uint32_t c = 0; c < R.n_class; ++c) xf[c] = 0;
        for (uint32_t s2 = 0; s2 < n_sel2; ++s2) xf[sel2[s2]] = x2[s2];
        uint32_t n_in2 = d2_scope(&R, xf, seg2, mass_floor, max_seg,
                                  inscope, inscope2);
        if (verbose) {
          uint64_t tot = 0;
          for (uint32_t r = 0; r < P2.n_pat; ++r) tot += P2.n[r];
          fprintf(stderr,
            "[methscope] deconv: %s: round %u on %u classes -> %u patterns "
            "over %llu CpGs (w=n^%.2f)", rec < n_qname ? qname[rec] : "record",
            round, n_sel2, P2.n_pat, (unsigned long long)tot, we);
          {   /* how concentrated is this panel? */
            uint64_t big[5] = {0,0,0,0,0};
            for (uint32_t r = 0; r < P2.n_pat; ++r) {
              uint64_t v = P2.n[r];
              for (int q3 = 0; q3 < 5; ++q3)
                if (v > big[q3]) { uint64_t t3 = big[q3]; big[q3] = v; v = t3; }
            }
            uint64_t top5 = big[0]+big[1]+big[2]+big[3]+big[4];
            fprintf(stderr, " [largest %llu, top5 %.1f%% of CpGs]",
                    (unsigned long long)big[0],
                    tot ? 100.0 * (double)top5 / (double)tot : 0.0);
          }
          if (ia < R.n_class && ib < R.n_class)
            fprintf(stderr, ", %s|%s seg %u", R.name[ia], R.name[ib],
                    seg2[(size_t)ia * R.n_class + ib]);
          d2_seg_matrix_min(&R, &P2, sel2, n_sel2, min_n, segu);
          fprintf(stderr, "\n[methscope]   confusable pairs (seg total / usable"
                          " at --min-cpg %llu):", (unsigned long long)min_n);
          for (uint32_t a2 = 0; a2 < n_sel2; ++a2)
            for (uint32_t b2 = a2 + 1; b2 < n_sel2; ++b2) {
              uint32_t ga = sel2[a2], gb = sel2[b2];
              uint32_t v = seg2[(size_t)ga * R.n_class + gb];
              fprintf(stderr, " %s|%s %u/%u;", R.name[ga], R.name[gb], v,
                      segu[(size_t)ga * R.n_class + gb]);
            }
          if (ia < R.n_class && ib < R.n_class) {
            /* how is the pair's evidence distributed -- one pattern or fifty? */
            uint32_t pa = n_sel2, pb = n_sel2;
            for (uint32_t s3 = 0; s3 < n_sel2; ++s3) {
              if (sel2[s3] == ia) pa = s3;
              if (sel2[s3] == ib) pb = s3;
            }
            if (pa < n_sel2 && pb < n_sel2) {
              fprintf(stderr, "\n[methscope]   %s|%s split by:",
                      R.name[ia], R.name[ib]);
              uint32_t np = 0;
              for (uint32_t r = 0; r < P2.n_pat; ++r) {
                const uint64_t *kk = P2.key + (size_t)r * P2.kw;
                int va = (kk[pa >> 6] >> (pa & 63)) & 1;
                int vb = (kk[pb >> 6] >> (pb & 63)) & 1;
                if (va == vb) continue;
                fprintf(stderr, " [n=%llu%s q=%.3f %s=%.3f %s=%.3f]",
                        (unsigned long long)P2.n[r],
                        P2.n[r] < min_n ? "*" : "",
                        P2.qsum[r] / (double)P2.n[r],
                        R.name[ia],
                        P2.rsum[(size_t)r * P2.nall + ia] / (double)P2.n[r],
                        R.name[ib],
                        P2.rsum[(size_t)r * P2.nall + ib] / (double)P2.n[r]);
                ++np;
              }
              fprintf(stderr, "  (%u patterns; * = below --min-cpg)", np);
            }
          }
          fprintf(stderr, "\n[methscope]   scope -> %u:", n_in2);
          for (uint32_t c = 0; c < R.n_class; ++c)
            if (inscope2[c]) fprintf(stderr, " %s(%.3f)", R.name[c], xf[c]);
          fputc('\n', stderr);
        }
        d2pan_free(&P2);

        int same = 1;
        for (uint32_t c = 0; c < R.n_class; ++c)
          if (inscope[c] != inscope2[c]) { same = 0; break; }
        memcpy(inscope, inscope2, R.n_class * sizeof(uint8_t));
        memcpy(x, xf, R.n_class * sizeof(double));
        if (same) break;
      }
      if (o->scope_out) {
        FILE *sf = fopen(o->scope_out, rec ? "a" : "w");
        if (!sf) d2die("cannot open --scope-out", o->scope_out);
        if (!rec) {
          fputs("cell", sf);
          for (uint32_t c = 0; c < R.n_class; ++c) fprintf(sf, "\t%s", R.name[c]);
          fputc('\n', sf);
        }
        fprintf(sf, "%s", rec < n_qname ? qname[rec] : "record");
        for (uint32_t c = 0; c < R.n_class; ++c) fprintf(sf, "\t%u", inscope[c]);
        fputc('\n', sf);
        fclose(sf);
      }
    }
    /* FINAL SOLVE on the settled scope.
     *
     * Not redundant with the in-loop solves, though it looks it: each round
     * solves the panel it was HANDED (the previous round's scope) and its
     * result is only used to decide the next scope. This rebuilds on the scope
     * that actually survived and solves that. Removing it cost 0.062 -> 0.118
     * TVD on the immune cohort, with the pair split going 0.035 -> 0.102.
     *
     *
     * It runs unconditionally. It was once gated on "a satellite was built",
     * which meant any record whose settled scope held no confusable pair
     * silently kept the round-1 GLOBAL fit as its answer. */
    {
      uint32_t n_fin = 0;
      for (uint32_t c = 0; c < R.n_class; ++c) if (inscope[c]) sel2[n_fin++] = c;
      if (n_fin >= 2) {
        d2pan_t PF; d2pan_build(&R, &Q, sel2, n_fin, dm_top, global_ref,
                                    weak, rescue_lo, rescue_hi,
                                    rescue_target ? gap_of : NULL,
                                    rescue_gap, rescue_depth, &PF);
        d2_seg_matrix_min(&R, &PF, sel2, n_fin, min_n, seg2);
        d2_solve(&PF, sel2, n_fin, min_n, wexp[n_wexp - 1], xadj);
        for (uint32_t c = 0; c < R.n_class; ++c) x[c] = 0;
        for (uint32_t s3 = 0; s3 < n_fin; ++s3) x[sel2[s3]] = xadj[s3];
        d2pan_free(&PF);
      }
    }

    if (o->panel_out) {
      FILE *pf = fopen(o->panel_out, rec ? "a" : "w");
      if (!pf) d2die("cannot open --panel-out", o->panel_out);
      if (!rec) fputs("record\tbinstring\tn_cpg\tq_beta\n", pf);
      for (uint32_t r = 0; r < P.n_pat; ++r) {
        fprintf(pf, "%s\t", rec < n_qname ? qname[rec] : "record");
        for (uint32_t s2 = 0; s2 < R.n_class; ++s2)
          fputc((P.key[(size_t)r * P.kw + (s2 >> 6)] >> (s2 & 63)) & 1
                ? '1' : '0', pf);
        fprintf(pf, "\t%llu\t%.4f\n", (unsigned long long)P.n[r],
                P.qsum[r] / (double)P.n[r]);
      }
      fclose(pf);
    }
    d2pan_free(&P);
  memcpy(xout, x, R.n_class * sizeof(double));
}

/* One thread per worker, records handed out by an atomic counter.
 *
 * The reference is const and shared -- 522 MB for 33 classes, so the whole
 * point is that threads do NOT each carry a copy. Everything mutable is in the
 * per-thread workspace. Each thread opens its OWN handle on the query and seeks
 * by the .idx offset, because a cfile carries a decompression position that
 * cannot be shared. Results land in a per-record slot, so the output stays in
 * input order however the work is interleaved.
 *
 * nnls_c is reentrant (its f2c statics were removed); if that ever regresses,
 * this races silently rather than crashing. */
typedef struct {
  const d2ref_t *R; const d2opt_t *o;
  const char *qpath; const int64_t *qoff;
  char **qname; uint32_t n_qname, n_rec;
  double *xall;                       /* n_rec * n_class, in input order */
  uint32_t next; pthread_mutex_t lock;
} d2job_t;

static void *d2_worker(void *arg) {
  d2job_t *J = arg;
  d2ws_t ws; d2ws_init(J->R, &ws);
  cfile_t qf = open_cfile((char *)J->qpath);
  for (;;) {
    pthread_mutex_lock(&J->lock);
    uint32_t rec = J->next < J->n_rec ? J->next++ : UINT32_MAX;
    pthread_mutex_unlock(&J->lock);
    if (rec == UINT32_MAX) break;
    if (J->qoff[rec] >= 0 && bgzf_seek(qf.fh, J->qoff[rec], SEEK_SET) != 0)
      d2die("cannot seek to query record", J->qname[rec]);
    cdata_t c = read_cdata1(&qf);
    if (!c.n) break;
    decompress_in_situ(&c);
    if (c.fmt != '3' && c.fmt != '6')
    d2die("query must be format-3 (M/U) or format-6 (universe 0/1) .cg",
          J->qpath);
    if (c.n != J->R->n_row)
      d2die("query and reference disagree on the CpG row space", J->qpath);
    d2_record(J->R, J->o, &ws, &c, rec, J->qname, J->n_qname,
              J->xall + (size_t)rec * J->R->n_class);
    free_cdata(&c);
  }
  bgzf_close(qf.fh);
  d2ws_free(&ws);
  return NULL;
}

/* Output shapes. LONG is the default because the answer to "what is in this
 * sample" is a handful of classes, and a 62-column row makes the reader hunt
 * for them; WIDE is the matrix a downstream join wants. */
enum { D2_LONG = 0, D2_WIDE, D2_REPORT };

typedef struct { uint32_t s; double v; } d2sh_t;

static int d2sh_cmp(const void *a, const void *b) {
  const d2sh_t *x = a, *y = b;
  if (x->v > y->v) return -1;
  if (x->v < y->v) return 1;
  return x->s < y->s ? -1 : x->s > y->s;      /* ties in index order */
}

static void d2_emit_header(FILE *out, const d2ref_t *R, int mode) {
  if (mode == D2_REPORT) return;
  if (mode == D2_LONG) { fputs("cell\tclass\tfraction\n", out); return; }
  fputs("cell", out);
  for (uint32_t s = 0; s < R->n_class; ++s) fprintf(out, "\t%s", R->name[s]);
  fputc('\n', out);
}

/* One record's composition. LONG and REPORT drop classes below min_frac;
 * REPORT then names the dropped mass as "Others" rather than rescaling what
 * is left back up to 100%, so a percentage always means what it says. */
static void d2_emit(FILE *out, const d2ref_t *R, const char *name,
                    const double *x, int mode, double min_frac) {
  if (mode == D2_WIDE) {
    fprintf(out, "%s", name);
    for (uint32_t s = 0; s < R->n_class; ++s) fprintf(out, "\t%.6f", x[s]);
    fputc('\n', out);
    return;
  }

  d2sh_t *o = d2alloc(R->n_class, sizeof(d2sh_t), "emit order");
  double total = 0;
  for (uint32_t s = 0; s < R->n_class; ++s) {
    o[s].s = s; o[s].v = x[s]; total += x[s];
  }
  qsort(o, R->n_class, sizeof(d2sh_t), d2sh_cmp);

  double shown = 0;
  uint32_t n = 0;
  if (mode == D2_REPORT) fprintf(out, "%s:", name);
  for (uint32_t k = 0; k < R->n_class; ++k) {
    if (o[k].v < min_frac || o[k].v <= 0) break;
    if (mode == D2_LONG)
      fprintf(out, "%s\t%s\t%.6f\n", name, R->name[o[k].s], o[k].v);
    else
      fprintf(out, "%s %s %.1f%%", n ? ";" : "", R->name[o[k].s],
              o[k].v * 100.0);
    shown += o[k].v; ++n;
  }
  if (mode == D2_REPORT) {
    double rest = total - shown;
    if (!n && rest * 100.0 < 0.05) fputs(" nothing above the floor", out);
    if (rest * 100.0 >= 0.05)
      fprintf(out, "%s Others %.1f%%", n ? ";" : "", rest * 100.0);
    fputc('\n', out);
  }
  free(o);
}

int main_deconv(int argc, char *argv[]) {
  const char *out_path = NULL, *panel_out = NULL, *pair_spec = NULL;
  const char *force_scope = NULL, *eval_x = NULL, *design_out = NULL;
  const char *scope_out = NULL;
  /* 5, not 1. A pattern resting on 1-4 measured CpGs has an observed beta that
   * is essentially 0 or 1 whatever the truth. Measured ALONE on a global fit a
   * floor of 5 costs accuracy (TVD 0.261 vs 0.195), because it deletes the
   * small patterns that carry the confusable-pair contrast -- but that contrast
   * is not the global stage's job; the scope rebuild recovers it over far more
   * CpGs. The floor also feeds the scope: dropping thin pair patterns lowers
   * seg(), so a pair the global panel barely separates is more readily kept. */
  /* OFF. A per-binstring cap was meant to stop one pattern owning the fit, but
   * it does so by discarding that pattern's own CpGs, and the amount discarded
   * grows with coverage: at 2^20 the largest pattern holds 4,682 measured CpGs
   * and at 2^22 it holds 14,253, so a cap of 500 is a 9x and then a 28x cut.
   * That made accuracy DEGRADE with depth (0.059 at 2^18, 0.088 at 2^20). The
   * 500 that was briefly the default came from a sweep on the global-only fit
   * where it was worth 0.194 vs 0.195 -- noise -- before narrowing and
   * satellites existed. Dominance is real but wants a different instrument. */
  uint64_t min_n = 5, dm_top = 0;
  /* The rescue is ON by default. Without it a pair the panel cannot separate
   * is decided by a handful of CpGs carrying most of the residual, and the fit
   * buys a lower SSE by inventing an absent class -- the un-rescued rebuild is
   * WORSE than the frozen-panel path it replaced at 2^16 (0.466 vs 0.200 on
   * immune mixtures, 0.431 vs 0.359 on the 33-class set), and better than it
   * everywhere once the rescue is on. Shipping it off would ship the worst
   * configuration measured.
   *
   * 2000: a pair the panel separates by fewer than this many measured CpGs is
   * treated as starved. 0.20: admit a CpG when the pair's betas differ by that
   * much, whatever the band says about either one. Both were chosen on one
   * record and then held up unchanged on the 33-class cohort, which they were
   * never tuned against; --rescue-below 0 turns the whole thing off. */
  uint64_t rescue_below = 2000;
  double rescue_lo = 0.40, rescue_hi = 0.60, rescue_gap = 0.20;
  double rescue_floor = 0.15;
  unsigned long rescue_target = 0;
  /* 10, not 60: a gap read off a couple of reads is noise, but the sweep
   * showed 60 costs real pair evidence at every rung (0.031 -> 0.036 at
   * 2^20). 10 is the smallest floor that still excludes a coin flip. */
  unsigned long rescue_depth = 10;
  uint32_t max_round = 8;
  int narrow = 1, nthreads = 1, global_ref = 0;
  int outmode = D2_LONG;
  /* Display only: it decides what LONG and REPORT list, never what is fitted.
   * 0.005 because NNLS leaves dust among collinear columns that is not a
   * claim about the sample; --min-frac 0 shows every non-zero class. */
  double min_frac = 0.005;
  /* One value per ROUND, last repeating: "0.5,0.5,0" is 0.5 for the global fit
   * and the first rebuild, unweighted from round 3 on. A single value applies
   * everywhere. Per-round because the panel changes character as the scope
   * narrows -- at 7 classes five patterns hold 80% of the CpGs, which is not
   * true of the 33-class panel. */
  double wexp[16]; int n_wexp = 1; wexp[0] = 0.5;
  /* Absolute, in MEASURED CpGs, because that is what reaches the solve.
   *
   * 200. Measured on the global panel: on t1_4_32 it offers ~17 CpGs to
   * tell T.Cell.CD8 from T.Cell.CD4 -- 18 survive the 33-class conjunction out
   * of the 217 the pair actually holds -- and CD8 was fitted at exactly 0.000
   * because CD4 took its mass. A floor of 20 is the edge and 50 retains it;
   * 200 sits well clear, since the count moves with coverage and with
   * --delta-mean-top, and 100 vs 200 gave byte-identical trajectories. */
  uint64_t max_seg = 200;
  double mass_floor = 0.005;
  int verbose = 0, i = 1;
  for (; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-o") && i + 1 < argc) out_path = argv[++i];
    else if (!strcmp(a, "--min-cpg") && i + 1 < argc)
      min_n = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--panel-out") && i + 1 < argc) panel_out = argv[++i];
    else if (!strcmp(a, "--pair-count") && i + 1 < argc) pair_spec = argv[++i];
    else if (!strcmp(a, "--delta-mean-top") && i + 1 < argc)
      dm_top = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--max-segregating") && i + 1 < argc)
      max_seg = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--weight-exponent") && i + 1 < argc) {
      const char *v = argv[++i]; n_wexp = 0;
      while (*v && n_wexp < 16) {
        wexp[n_wexp++] = atof(v);
        const char *c = strchr(v, ','); if (!c) break; v = c + 1;
      }
      if (!n_wexp) d2die("--weight-exponent wants E or E1,E2,...", argv[i]);
    }
    else if (!strcmp(a, "--wide")) outmode = D2_WIDE;
    else if (!strcmp(a, "--report")) outmode = D2_REPORT;
    else if (!strcmp(a, "--min-frac") && i + 1 < argc)
      min_frac = atof(argv[++i]);
    else if (!strcmp(a, "--no-narrow")) narrow = 0;
    else if (!strcmp(a, "--rescue-below") && i + 1 < argc)
      rescue_below = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--rescue-gap") && i + 1 < argc)
      rescue_gap = atof(argv[++i]);
    else if (!strcmp(a, "--rescue-min-depth") && i + 1 < argc)
      rescue_depth = strtoul(argv[++i], NULL, 10);
    else if (!strcmp(a, "--rescue-target") && i + 1 < argc)
      rescue_target = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--rescue-floor") && i + 1 < argc)
      rescue_floor = atof(argv[++i]);
    else if (!strcmp(a, "--rescue-qfilter") && i + 1 < argc) {
      char *end = NULL;
      double v = strtod(argv[++i], &end);
      if (!end || *end != ',') d2die("--rescue-qfilter wants LO,HI", argv[i]);
      rescue_lo = v; rescue_hi = strtod(end + 1, NULL);
    }
    else if (!strcmp(a, "--global-ref")) global_ref = 1;
    else if ((!strcmp(a, "--nthreads") || !strcmp(a, "--threads")) && i + 1 < argc)
      nthreads = atoi(argv[++i]);
    else if (!strcmp(a, "--max-round") && i + 1 < argc)
      max_round = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(a, "--mass-floor") && i + 1 < argc)
      mass_floor = atof(argv[++i]);
    else if (!strcmp(a, "--scope-out") && i + 1 < argc) scope_out = argv[++i];
    else if (!strcmp(a, "--force-scope") && i + 1 < argc) force_scope = argv[++i];
    else if (!strcmp(a, "--eval-x") && i + 1 < argc) eval_x = argv[++i];
    else if (!strcmp(a, "--design-out") && i + 1 < argc) design_out = argv[++i];
    else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) verbose = 1;
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stdout,
"Usage:\n"
"  methscope deconv [options] <ref.msdref> <query.cg>\n"
"\n"
"Purpose:\n"
"  Estimate the cell-type composition of each query record by non-negative\n"
"  least squares against a .msdref reference. The pattern set is rebuilt per\n"
"  query from the CpGs that query actually measured, so no pattern budget is\n"
"  baked in and a sparse query is scored on its own evidence.\n"
"\n"
"  The fit is ADAPTIVE. Round 1 solves over every class. Each round after it\n"
"  keeps the classes holding at least --mass-floor, then pulls back in any\n"
"  class the panel separates from a survivor by --max-segregating measured\n"
"  CpGs or fewer -- repeating until nothing new enters, so a scope is a union\n"
"  of connected components rather than a mass cutoff. The pattern set is then\n"
"  rebuilt over that scope and refit: dropping classes relaxes the admission\n"
"  conjunction, so a narrower scope admits CpGs the full one could not. The\n"
"  scope only ever narrows, and rounds stop once it stops changing or at\n"
"  --max-round. --no-narrow keeps the round-1 fit over every class.\n"
"\n"
"Arguments:\n"
"  <ref.msdref>   Deconvolution reference from `deconv-build-ref`. Its index\n"
"                 order is the binstring digit order.\n"
"  <query.cg>     Query methylome(s), YAME format 3 (M/U counts) or format 6\n"
"                 (universe bit plus binary call). Record names come from\n"
"                 <query.cg>.idx; without one they are 1,2,3,...\n"
"\n"
"Options:\n"
"  -o <out.tsv>            Write output to a file instead of stdout.\n"
"  --wide                  Emit the full composition matrix instead: one row\n"
"                          per record, one column per reference class, in the\n"
"                          reference's index order. Every class appears, so\n"
"                          --min-frac does not apply.\n"
"  --report                Emit a readable one-line summary per record --\n"
"                          \"cell: Class 69.6%%; Class 30.4%%\" -- instead of a\n"
"                          TSV.\n"
"  --min-frac F            Hide classes below this fraction. Default: 0.005.\n"
"                          Display only: it never changes what is fitted. 0\n"
"                          shows every non-zero class.\n"
"  --threads N             Deconvolve N records in parallel. Default: 1.\n"
"                          --nthreads is an alias. The reference is shared, so\n"
"                          memory grows by the per-thread workspace (~16 MB),\n"
"                          not linearly. Needs a <query.cg>.idx to seek by; -v\n"
"                          and the dump options force a single thread.\n"
"  --no-narrow             Fit the global panel over every class, skipping the\n"
"                          per-query rebuild rounds.\n"
"  --max-round N           Cap on rebuild rounds. Default: 8. Rounds stop early\n"
"                          once the class set stops changing.\n"
"  -v                      Report per-record panel size and measured-row count.\n"
"                          --verbose is an alias.\n"
"  -h                      Show this help message.\n"
"\n"
"Tuning:\n"
"  Every default below was chosen on the 200-mixture benchmarks, where moving it\n"
"  scored worse or made no measurable difference. Change one to reproduce an\n"
"  experiment, not to improve an answer.\n"
"  --min-cpg N             Drop a pattern seen on fewer than N measured CpGs.\n"
"                          Default: 5.\n"
"  --mass-floor F          A class seeds the scope at this mass or above.\n"
"                          Default: 0.005. Not 0, because NNLS hands out\n"
"                          arbitrary exact zeros among collinear columns.\n"
"  --max-segregating N     Two classes the panel separates by this many\n"
"                          measured CpGs or fewer are inseparable for this\n"
"                          query, so both enter the scope when either carries\n"
"                          mass. Absolute, because an absolute CpG count is\n"
"                          what reaches the solve. Default: 200.\n"
"  --weight-exponent E     Row weight is n^E, so influence is n^2E. Default:\n"
"                          0.5, the precision weight. One value per round is\n"
"                          accepted as E1,E2,... with the last repeating.\n"
"  --delta-mean-top N      Per binstring, keep only the N measured CpGs with\n"
"                          the cleanest class gap, bounding how much one\n"
"                          pattern can weigh. Default: 0, keep all.\n"
"  --global-ref            Compute each pattern's reference profile over every\n"
"                          reference CpG carrying that binstring, not only the\n"
"                          measured ones. Costs a second pass over kept rows.\n"
"  --rescue-below N        A pair the rebuilt panel separates by fewer than N\n"
"                          measured CpGs is weak; the panel is then rebuilt\n"
"                          admitting CpGs that split a weak pair under a wider\n"
"                          rule, every class still getting a digit. Default:\n"
"                          2000. 0 turns the rescue off, which was the worst\n"
"                          configuration measured.\n"
"  --rescue-gap G          Rescue on |beta_a - beta_b| >= G instead of on the\n"
"                          band, exempting the pair's own classes. Default:\n"
"                          0.20. 0 uses the band rule.\n"
"  --rescue-target N       Fit each weak pair its own gap, loosening until it\n"
"                          has N measured CpGs or hits the floor. Default: 0,\n"
"                          use the flat --rescue-gap.\n"
"  --rescue-floor G        Never relax a pair past this gap. Default: 0.15.\n"
"  --rescue-min-depth N    Reads required on both classes of a pair before its\n"
"                          gap is believed. Default: 10. Needs a v2 .msdref,\n"
"                          which carries M/U rather than a bare beta.\n"
"  --rescue-qfilter LO,HI  The wider admission band. Default: 0.40,0.60.\n"
"\n"
"Diagnostics:\n"
"  --panel-out F           Dump the rebuilt panel: binstring, measured CpGs and\n"
"                          observed beta, one row per pattern per record.\n"
"  --scope-out F           Write the settled scope, 1/0 per class per record.\n"
"  --design-out F          Dump the exact system handed to the solve -- one row\n"
"                          per pattern, its weight, the observed value and every\n"
"                          scope class's profile -- so the fit can be redone\n"
"                          outside.\n"
"  --force-scope C[,C..]   Use exactly these classes, skipping discovery. It\n"
"                          answers whether a narrow panel is weak because the\n"
"                          scope is wrong or because it is narrow.\n"
"  --pair-count A,B        Report how many measured CpGs segregate classes A and\n"
"                          B on their own, and how many survive the all-classes\n"
"                          admission conjunction.\n"
"  --eval-x SPEC           Print the objective at hand-given compositions beside\n"
"                          the fitted one, as \"Cls=frac,Cls=frac;Cls=frac,...\".\n"
"                          Answers whether a wrong answer means the solver\n"
"                          missed the optimum or the panel scores the wrong\n"
"                          composition better. Needs -v.\n"
"\n"
"Output:\n"
"  By default a tidy TSV -- cell, class, fraction -- carrying only the classes\n"
"  at or above --min-frac, largest first. --wide gives the full matrix and\n"
"  --report a one-line summary per record.\n"
"\n"
"  A record's fractions are proportions of the mass the reference could\n"
"  explain: the fit is renormalized to sum to 1, so a query whose true\n"
"  composition lies outside the reference still sums to 1, spread over its\n"
"  nearest classes. --report is the one shape that does NOT rescale -- what\n"
"  --min-frac hides is named as \"Others\", so a printed percentage always\n"
"  means the same thing.\n");
      return 0;
    }
    else if (a[0] == '-' && a[1]) d2die("unrecognized option", a);
    else break;
  }
  if (argc - i != 2) {
    fprintf(stderr,
      "Usage: methscope deconv <ref.msdref> <query.cg> -o <out.tsv>\n");
    return 1;
  }
  const char *rpath = argv[i], *qpath = argv[i + 1];

  d2ref_t R;
  d2ref_load(rpath, &R);

  uint32_t n_qname = 0;
  int64_t *qoff = NULL;
  char **qname = d2_read_idx(qpath, &n_qname, &qoff, 1);

  FILE *out = out_path ? fopen(out_path, "w") : stdout;
  if (!out) d2die("cannot open output", out_path);
  d2_emit_header(out, &R, outmode);

  d2opt_t opt;
  memset(&opt, 0, sizeof opt);
  opt.min_n = min_n; opt.dm_top = dm_top; opt.max_seg = max_seg;
  opt.mass_floor = mass_floor;
  opt.rescue_below = rescue_below;
  opt.rescue_lo = rescue_lo; opt.rescue_hi = rescue_hi;
  opt.rescue_gap = rescue_gap;
  opt.rescue_depth = (uint32_t)rescue_depth;
  opt.rescue_target = rescue_target; opt.rescue_floor = rescue_floor;
  opt.narrow = narrow; opt.verbose = verbose; opt.global_ref = global_ref;
  opt.max_round = max_round;
  memcpy(opt.wexp, wexp, sizeof wexp); opt.n_wexp = n_wexp;
  opt.pair_spec = pair_spec; opt.panel_out = panel_out;
  opt.scope_out = scope_out; opt.force_scope = force_scope;
  opt.eval_x = eval_x; opt.design_out = design_out;

  if (nthreads > 1 && (verbose || panel_out || scope_out)) {
    fprintf(stderr, "[methscope] deconv: -v/--panel-out/--scope-out are "
                    "single-threaded; using 1 thread\n");
    nthreads = 1;
  }
  if (nthreads > 1 && !n_qname) {
    fprintf(stderr, "[methscope] deconv: no <query>.idx to seek by; "
                    "using 1 thread\n");
    nthreads = 1;
  }

  if (nthreads > 1) {
    d2job_t J;
    memset(&J, 0, sizeof J);
    J.R = &R; J.o = &opt; J.qpath = qpath; J.qoff = qoff;
    J.qname = qname; J.n_qname = n_qname; J.n_rec = n_qname;
    J.xall = d2alloc((size_t)n_qname * R.n_class, sizeof(double), "results");
    pthread_mutex_init(&J.lock, NULL);
    pthread_t *th = d2alloc(nthreads, sizeof(pthread_t), "threads");
    for (int t = 0; t < nthreads; ++t)
      if (pthread_create(&th[t], NULL, d2_worker, &J) != 0)
        d2die("cannot create thread", NULL);
    for (int t = 0; t < nthreads; ++t) pthread_join(th[t], NULL);
    pthread_mutex_destroy(&J.lock);
    for (uint32_t r2 = 0; r2 < n_qname; ++r2)
      d2_emit(out, &R, qname[r2], J.xall + (size_t)r2 * R.n_class,
              outmode, min_frac);
    free(J.xall); free(th);
    if (out != stdout) fclose(out);
    free(qoff);
    for (uint32_t q = 0; q < n_qname; ++q) free(qname[q]);
    free(qname); d2ref_free(&R);
    return 0;
  }

  d2ws_t ws; d2ws_init(&R, &ws);
  double *xrec = d2alloc(R.n_class, sizeof(double), "record proportions");
  cfile_t qf = open_cfile((char *)qpath);
  for (uint32_t rec = 0; ; ++rec) {
    cdata_t c = read_cdata1(&qf);
    if (!c.n) break;
    decompress_in_situ(&c);
    if (c.fmt != '3' && c.fmt != '6')
      d2die("query must be format-3 (M/U) or format-6 (universe 0/1) .cg",
            qpath);
    if (c.n != R.n_row)
      d2die("query and reference disagree on the CpG row space", qpath);

    d2_record(&R, &opt, &ws, &c, rec, qname, n_qname, xrec);
    d2_emit(out, &R, rec < n_qname ? qname[rec] : "record", xrec,
            outmode, min_frac);
    free_cdata(&c);
  }
  bgzf_close(qf.fh);
  if (out != stdout) fclose(out);
  d2ws_free(&ws); free(xrec); free(qoff);
  for (uint32_t q = 0; q < n_qname; ++q) free(qname[q]);
  free(qname); d2ref_free(&R);
  return 0;
}
