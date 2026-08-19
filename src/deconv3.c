// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * deconv3: legacy's deconvolution reference, built in one pass.
 *
 * WHY THIS EXISTS
 *
 * The shipped reference is not a flat MRMP, though nothing said so. It is a
 * POOL built in three steps -- `mrmp-build --flat` over every class, then
 * `mrmp-build-neighbor` for the pairs the global set cannot separate, then
 * `mrmp-pool` to a shared pattern budget -- and the satellites are where its
 * accuracy lives. Measured on 200 immune mixtures, dropping them roughly
 * TRIPLES the error at every depth (2^20: 0.044 -> 0.135), while the q-filter
 * band is worth 0.01-0.05 and --delta-mean-top is worth nothing at all.
 *
 * deconv2 has the same three steps relocated -- per query, over the CpGs the
 * query measured -- and is better in the sparse regime (2^16: 0.119 vs 0.200)
 * and worse everywhere else. Which relocation helps is a question that cannot
 * be answered while the two implementations ALSO differ in the reference
 * format, the row pre-filter, the design-matrix support, the class set and the
 * pattern budget. Every comparison we ran across the two moved several of those
 * at once, and three separate conclusions reversed on closer inspection.
 *
 * So this rebuilds legacy's construction inside the new code, off the same
 * packed uint16 betas, with the explicit goal of REPRODUCING it: 0.044 TVD at
 * 2^20 and 0.028 at 2^22 on the immune cohort. Parity is the acceptance test --
 * until it holds, any difference is a bug rather than a finding. Once it does,
 * each deconv2 idea becomes one switch against a known-good baseline instead of
 * a whole second pipeline.
 *
 * WHAT ONE BUILD DOES
 *
 *   1. GLOBAL   one binstring per CpG over every class: admissible when every
 *               class is callable (beta <= LO or >= HI), dropped when constant.
 *   2. SATELLITE for each pair the global set separates by <= --max-segregating
 *               CpGs, a 2-class set over the same rule restricted to that pair.
 *               A pair needs only two classes callable, so it admits CpGs the
 *               33-way conjunction rejected -- which is the whole point.
 *   3. POOL     every pattern ranked by CpG count, the top --pooled-top kept.
 *               Sets compete; nothing is reserved a slot.
 *   4. PROFILE  per-pattern mean beta for every class, computed HERE, over all
 *               the reference CpGs in the pattern. The design matrix is then a
 *               build-time quantity with no per-query sampling noise, which is
 *               what legacy has and what deconv2 re-derives per query.
 *
 * Rows are assigned to at most one pattern, global first: a .cm mask carries
 * one state per CpG, and legacy's own set sizes sum exactly (462,024 +
 * 1,540,637 = 2,002,661), so its sets are disjoint too.
 *
 * FORMAT (MSD3REF1), little-endian:
 *   magic     8   "MSD3REF"
 *   header       version, n_class, n_row, n_pat, qlo, qhi, beta_thr, mincov
 *   names     -   n_class NUL-terminated, in .cg.idx order
 *   n_ent     8              total (pattern, row) memberships
 *   poff      8 * (n_pat+1)  CSR offsets into rows
 *   rows      4 * n_ent      row ids, grouped by pattern; a row MAY
 *                            appear under several patterns
 *   count     8 * n_pat      uint64 CpGs behind each pattern
 *   beta      2 * n_pat * n_class   uint16, pattern-major
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "methscope.h"
#include "cfile.h"
#include "cdata.h"
#include "mrmp.h"
#include "nnls.h"

#define D3_MAGIC   "MSD3REF"
#define D3_NA      0xFFFFu
#define D3_NOPAT   0xFFFFFFFFu

static void d3die(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] deconv3-build-ref: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] deconv3-build-ref: %s\n", msg);
  exit(1);
}

static void *d3alloc(size_t n, size_t sz, const char *what) {
  void *p = calloc(n ? n : 1, sz);
  if (!p) d3die("out of memory", what);
  return p;
}

static uint16_t d3_enc(double b) {
  if (b < 0) b = 0; else if (b > 1) b = 1;
  return (uint16_t)(b * 65534.0 + 0.5);
}

static char **d3_read_idx(const char *ref, uint32_t *n_out, int64_t **off_out) {
  char idx[PATH_MAX];
  if (snprintf(idx, sizeof idx, "%s.idx", ref) >= (int)sizeof idx)
    d3die("reference path too long", ref);
  FILE *f = fopen(idx, "r");
  if (!f) d3die("cannot open reference index (expected <ref>.idx)", idx);
  size_t cap = 64, n = 0;
  char **names = d3alloc(cap, sizeof(char *), "names");
  int64_t *off = d3alloc(cap, sizeof(int64_t), "offsets");
  char *line = NULL; size_t lc = 0; ssize_t len;
  while ((len = getline(&line, &lc, f)) > 0) {
    char *tab = strpbrk(line, "\t\n");
    size_t nl = tab ? (size_t)(tab - line) : (size_t)len;
    if (!nl) continue;
    if (n == cap) {
      cap <<= 1;
      names = realloc(names, cap * sizeof(char *));
      off   = realloc(off, cap * sizeof(int64_t));
      if (!names || !off) d3die("out of memory", "index grow");
    }
    names[n] = d3alloc(nl + 1, 1, "name");
    memcpy(names[n], line, nl);
    off[n] = (tab && *tab == '\t') ? (int64_t)strtoll(tab + 1, NULL, 10) : -1;
    ++n;
  }
  free(line); fclose(f);
  if (!n) d3die("reference index is empty", idx);
  *n_out = (uint32_t)n; *off_out = off;
  return names;
}

/* ---- pattern table: binstring -> id, open addressed ------------------ */
typedef struct {
  uint64_t *key; uint32_t kw, n_pat, cap;
  uint64_t *count;
  uint32_t *set;            /* which set a pattern came from (0 = global) */
  uint32_t *slot; uint64_t mask;
} d3tab_t;

static uint64_t d3_hash(const uint64_t *k, uint32_t kw) {
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t w = 0; w < kw; ++w) { h ^= k[w]; h *= 1099511628211ULL; }
  return h;
}

static void d3tab_init(d3tab_t *T, uint32_t kw) {
  memset(T, 0, sizeof *T);
  T->kw = kw; T->cap = 4096; T->mask = (1ull << 20) - 1;
  T->key   = d3alloc((size_t)T->cap * kw, sizeof(uint64_t), "keys");
  T->count = d3alloc(T->cap, sizeof(uint64_t), "counts");
  T->set   = d3alloc(T->cap, sizeof(uint32_t), "sets");
  T->slot  = malloc((T->mask + 1) * sizeof(uint32_t));
  if (!T->slot) d3die("out of memory", "hash");
  memset(T->slot, 0xFF, (T->mask + 1) * sizeof(uint32_t));
}

static uint32_t d3tab_intern(d3tab_t *T, const uint64_t *k, uint32_t set) {
  uint64_t i = d3_hash(k, T->kw) & T->mask;
  while (T->slot[i] != D3_NOPAT) {
    uint32_t r = T->slot[i];
    if (!memcmp(T->key + (size_t)r * T->kw, k, T->kw * sizeof(uint64_t)))
      return r;
    i = (i + 1) & T->mask;
  }
  if (T->n_pat == T->cap) {
    T->cap *= 2;
    T->key   = realloc(T->key, (size_t)T->cap * T->kw * sizeof(uint64_t));
    T->count = realloc(T->count, (size_t)T->cap * sizeof(uint64_t));
    T->set   = realloc(T->set, (size_t)T->cap * sizeof(uint32_t));
    if (!T->key || !T->count || !T->set) d3die("out of memory", "table grow");
    memset(T->count + T->n_pat, 0, (T->cap - T->n_pat) * sizeof(uint64_t));
  }
  uint32_t r = T->n_pat++;
  memcpy(T->key + (size_t)r * T->kw, k, T->kw * sizeof(uint64_t));
  T->count[r] = 0; T->set[r] = set;
  T->slot[i] = r;
  return r;
}

static void d3tab_free(d3tab_t *T) {
  free(T->key); free(T->count); free(T->set); free(T->slot);
}

/* One assigned CpG, pending the per-binstring --delta-mean-top selection.
 * `gap` is the class gap in ENCODED beta units over the pattern's OWN class
 * set -- all classes for a global pattern, just the two for a satellite -- so
 * a larger gap is a cleaner separator. */
typedef struct { uint32_t p; uint32_t r; float gap; } d3ent_t;

static void d3_select(d3ent_t *ent, size_t n_ent, uint64_t dm_top,
                      uint32_t *member, uint64_t *count);

/* Order purely by the negated rank carried in `gap`, so the largest pattern's
 * rows are offered first and ties keep input order. */
static int d3ent_cmp_rank(const void *a, const void *b) {
  const d3ent_t *x = a, *y = b;
  if (x->gap != y->gap) return x->gap > y->gap ? -1 : 1;
  return 0;
}

static int d3ent_cmp(const void *a, const void *b) {
  const d3ent_t *x = a, *y = b;
  if (x->p != y->p) return x->p < y->p ? -1 : 1;
  if (x->gap != y->gap) return x->gap > y->gap ? -1 : 1;   /* cleanest first */
  return 0;
}

/* Keep the dm_top cleanest CpGs of each binstring; unassign the rest. Counts
 * are recomputed here, so callers must run this before anything reads them --
 * the satellite gate measures pair distances on the CAPPED global set, which is
 * the order legacy uses (mrmp-build caps, then mrmp-build-neighbor measures). */
static void d3_select(d3ent_t *ent, size_t n_ent, uint64_t dm_top,
                      uint32_t *member, uint64_t *count) {
  if (!dm_top) {
    for (size_t z = 0; z < n_ent; ++z) count[ent[z].p]++;
    return;
  }
  qsort(ent, n_ent, sizeof(d3ent_t), d3ent_cmp);
  for (size_t a = 0; a < n_ent; ) {
    size_t b = a;
    while (b < n_ent && ent[b].p == ent[a].p) ++b;
    size_t take = (b - a) < dm_top ? (b - a) : dm_top;
    for (size_t z = 0; z < take; ++z) {
      member[ent[a + z].r] = ent[a + z].p; count[ent[a + z].p]++;
    }
    for (size_t z = take; z < b - a; ++z) member[ent[a + z].r] = D3_NOPAT;
    a = b;
  }
}

static void put32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void putf(FILE *f, double v)    { fwrite(&v, 8, 1, f); }

int main_deconv3_build_ref(int argc, char *argv[]) {
  const char *out_path = NULL, *nb_path = NULL;
  double qlo = 0.25, qhi = 0.60, beta_thr = 0.5;
  uint32_t mincov = 1;
  /* 0, not 20000: the shipped global.mrmp covers 470,860 CpGs, exactly what an
   * UNCAPPED build produces here, so its cap never bound. Applying 20000 cuts
   * 44k CpGs and moves away from the reference being reproduced. */
  uint64_t max_seg = 10000, pooled_top = 1000, dm_top = 0;
  int force = 0, resolve = 0, i = 1;

  for (; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-o") && i + 1 < argc) out_path = argv[++i];
    else if (!strcmp(a, "--force")) force = 1;
    else if (!strcmp(a, "--neighbors") && i + 1 < argc) nb_path = argv[++i];
    else if (!strcmp(a, "--resolve")) resolve = 1;
    else if (!strcmp(a, "--mincov") && i + 1 < argc)
      mincov = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(a, "--max-segregating") && i + 1 < argc)
      max_seg = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--pooled-top") && i + 1 < argc)
      pooled_top = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--delta-mean-top") && i + 1 < argc)
      dm_top = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(a, "--qfilter") && i + 1 < argc) {
      char *end = NULL;
      double lo = strtod(argv[++i], &end);
      if (!end || *end != ',') d3die("--qfilter wants LO,HI", argv[i]);
      double hi = strtod(end + 1, NULL);
      if (!(lo >= 0 && lo < hi && hi <= 1))
        d3die("--qfilter needs 0 <= LO < HI <= 1", argv[i]);
      qlo = lo; qhi = hi;
    }
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      printf(
"Usage: methscope deconv3-build-ref [options] <celltypes.cg> -o <out.d3ref>\n"
"\n"
"  Build the deconvolution reference the way the shipped one was built --\n"
"  a global flat MRMP, 2-class satellites for the pairs it cannot separate,\n"
"  and a pooled pattern budget -- in one pass instead of three commands.\n"
"\n"
"  The satellites are the part that matters: removing them roughly TRIPLES\n"
"  the error on the immune benchmark at every depth. The band is worth\n"
"  0.01-0.05 TVD and --delta-mean-top is worth nothing measurable.\n"
"\n"
"  Options\n"
"    -o FILE              output .d3ref (required)\n"
"    --qfilter LO,HI      admission band (default 0.25,0.60, the shipped one)\n"
"    --neighbors F        take the satellites from an existing chain built by\n"
"                         mrmp-build-neighbor, instead of deriving them here.\n"
"                         Its selection applies a per-class relative depth\n"
"                         floor AND something further that bites hardest on the\n"
"                         closest pairs -- CD4|CD8 comes out 2.75x too large\n"
"                         when re-derived -- so for PARITY the satellites are\n"
"                         consumed rather than reproduced.\n"
"    --max-segregating N  a pair the global set separates by <= N CpGs gets a\n"
"                         satellite, when deriving them (default 10000)\n"
"    --pooled-top N       total pattern budget; sets compete (default 1000)\n"
"    --delta-mean-top N   per binstring, keep the N cleanest separators\n"
"                         (default 0 = keep all; the shipped reference\n"
"                         was built with no binding cap)\n"
"    --beta-threshold B   call a class methylated above B (default 0.5)\n"
"    --mincov N           a class is covered at N reads or more (default 1)\n"
"    --resolve            collapse the pool to ONE pattern per CpG, largest\n"
"                         claiming first, as a single-mask .refx does; without\n"
"                         it a CpG may serve several patterns, as a chain\n"
"                         written per set does\n"
"    --force              overwrite an existing output\n");
      return 0;
    }
    else if (a[0] == '-' && a[1]) d3die("unrecognized option", a);
    else break;
  }
  if (argc - i != 1 || !out_path) {
    fprintf(stderr,
      "Usage: methscope deconv3-build-ref <celltypes.cg> -o <out.d3ref>\n");
    return 1;
  }
  const char *ref = argv[i];
  if (!force) {
    FILE *t = fopen(out_path, "rb");
    if (t) { fclose(t); d3die("output exists (use --force)", out_path); }
  }

  /* ---- load the reference as packed uint16 betas --------------------- */
  int64_t *voff = NULL; uint32_t nc = 0;
  char **name = d3_read_idx(ref, &nc, &voff);
  uint64_t n_row = 0; uint16_t *beta = NULL;
  /* Satellites apply a per-class RELATIVE depth floor that the global build
   * does not: mrmp-build-neighbor sets depth_floor_frac = 1.0, giving
   * target[k] = min(mean depth of class k, 20), and a CpG is admitted only if
   * every class in the SET covers it at its own target. That is why legacy's
   * satellites are smaller than a plain q-filter admits, by a factor that
   * varies per pair (CD4|CD8 0.37x, Endo|EpiLung 0.88x) -- it tracks each
   * class's coverage, not the pair. A beta plus a covered flag cannot express
   * it, so depth is kept too; the target caps at 20, so a byte is enough. */
  uint8_t *dep = NULL;
  double *dsum = d3alloc(nc, sizeof(double), "depth sums");
  uint64_t *dn = d3alloc(nc, sizeof(uint64_t), "depth counts");
  cfile_t cf = open_cfile((char *)ref);
  for (uint32_t k = 0; k < nc; ++k) {
    if (voff[k] >= 0 && bgzf_seek(cf.fh, voff[k], SEEK_SET) != 0)
      d3die("cannot seek to record", name[k]);
    cdata_t c = read_cdata1(&cf);
    if (!c.n) d3die("store record is empty", name[k]);
    decompress_in_situ(&c);
    if (c.fmt != '3') d3die("reference must be format-3 (M/U) .cg", name[k]);
    if (!k) {
      n_row = c.n;
      beta = d3alloc((size_t)nc * n_row, sizeof(uint16_t), "betas");
      for (size_t j = 0; j < (size_t)nc * n_row; ++j) beta[j] = D3_NA;
      dep = d3alloc((size_t)nc * n_row, 1, "depths");
    } else if (c.n != n_row) d3die("records disagree on CpG count", name[k]);
    uint16_t *row = beta + (size_t)k * n_row;
    for (uint64_t r = 0; r < n_row; ++r) {
      uint64_t mu = f3_get_mu(&c, r);
      if (!mu || MU2cov(mu) < mincov) continue;
      row[r] = d3_enc(MU2beta(mu));
      uint64_t cv = MU2cov(mu);
      dep[(size_t)k * n_row + r] = cv > 255 ? 255 : (uint8_t)cv;
      dsum[k] += (double)cv; dn[k]++;
    }
    free_cdata(&c);
    fprintf(stderr, "\r[methscope] deconv3-build-ref: reading %u/%u", k + 1, nc);
    fflush(stderr);
  }
  bgzf_close(cf.fh); fputc('\n', stderr);
  if (n_row > 0xFFFFFFFFull) d3die("row space exceeds uint32", ref);

  double *target = d3alloc(nc, sizeof(double), "depth targets");
  for (uint32_t c = 0; c < nc; ++c) {
    double m = dn[c] ? dsum[c] / (double)dn[c] : 0.0;
    target[c] = m > 20.0 ? 20.0 : m;      /* depth_floor_frac 1.0, cap 20 */
  }
  free(dsum); free(dn);

  const uint16_t LO = d3_enc(qlo), HI = d3_enc(qhi), BT = d3_enc(beta_thr);
  /* One extra key word holds the SET id. Without it a satellite's binstring --
   * which records only which of its two classes is '1' -- collides with every
   * other satellite sharing that class: Dendritic|Macrophage's "{DC}" and
   * Dendritic|Monocyte's "{DC}" are the same 33-bit string. Legacy avoids this
   * by keeping each set in its own block; a flat table has to say so. */
  const uint32_t kw = ((nc + 63) >> 6) + 1;
  uint32_t *member = d3alloc(n_row, sizeof(uint32_t), "membership");
  for (uint64_t r = 0; r < n_row; ++r) member[r] = D3_NOPAT;

  /* ---- 1. GLOBAL: one binstring per CpG over every class ------------- */
  d3tab_t T; d3tab_init(&T, kw);
  uint64_t *k = d3alloc(kw, sizeof(uint64_t), "key");
  uint64_t n_glob_cpg = 0;
  d3ent_t *ent = NULL; size_t n_ent = 0, cap_ent = 0;
  #define D3_PUSH(P, R, G) do {                                              \
    if (n_ent == cap_ent) {                                                  \
      cap_ent = cap_ent ? cap_ent * 2 : (1u << 20);                          \
      ent = realloc(ent, cap_ent * sizeof(d3ent_t));                         \
      if (!ent) d3die("out of memory", "assignment entries");                \
    }                                                                        \
    ent[n_ent].p = (P); ent[n_ent].r = (uint32_t)(R); ent[n_ent].gap = (G);  \
    ++n_ent;                                                                 \
  } while (0)
  for (uint64_t r = 0; r < n_row; ++r) {
    memset(k, 0, kw * sizeof(uint64_t));
    uint32_t n1 = 0; int ok = 1;
    uint32_t mn1 = 0xFFFFFFFFu, mx0 = 0;
    for (uint32_t c = 0; c < nc; ++c) {
      uint16_t b = beta[(size_t)c * n_row + r];
      if (b == D3_NA) { ok = 0; break; }
      if (b > BT) {
        if (b < HI) { ok = 0; break; }
        k[c >> 6] |= 1ull << (c & 63); ++n1;
        if (b < mn1) mn1 = b;
      } else if (b == BT || b > LO) { ok = 0; break; }
      else if (b > mx0) mx0 = b;
    }
    if (!ok || !n1 || n1 == nc) continue;
    k[kw - 1] = 0;                      /* set 0 = global */
    uint32_t p = d3tab_intern(&T, k, 0);
    ++n_glob_cpg;
    D3_PUSH(p, r, (float)((double)mn1 - (double)mx0));
  }
  uint32_t n_glob_pat = T.n_pat;
  d3_select(ent, n_ent, dm_top, member, T.count);
  uint64_t glob_kept = 0;
  for (uint32_t p = 0; p < n_glob_pat; ++p) glob_kept += T.count[p];
  fprintf(stderr, "[methscope] deconv3-build-ref: global %u patterns, "
                  "%llu CpGs admitted -> %llu after --delta-mean-top %llu\n",
          n_glob_pat, (unsigned long long)n_glob_cpg,
          (unsigned long long)glob_kept, (unsigned long long)dm_top);

  /* ---- 2. SATELLITES for pairs the global set cannot separate -------- */
  uint64_t *seg = d3alloc((size_t)nc * nc, sizeof(uint64_t), "seg");
  for (uint32_t p = 0; p < n_glob_pat; ++p) {
    const uint64_t *kk = T.key + (size_t)p * kw;
    for (uint32_t a = 0; a < nc; ++a) {
      int ba = (kk[a >> 6] >> (a & 63)) & 1;
      for (uint32_t b = a + 1; b < nc; ++b)
        if (((kk[b >> 6] >> (b & 63)) & 1) != ba) {
          seg[(size_t)a * nc + b] += T.count[p];
          seg[(size_t)b * nc + a] += T.count[p];
        }
    }
  }
  uint32_t n_sat = 0;
  if (nb_path) {
    /* Satellites from mrmp-build-neighbor. Each set's membership is walked as
     * runs and its ranks become patterns here; a satellite claims the row even
     * where the global already did, which is how legacy's pool resolves the
     * overlap (its pooled global holds 462,024 CpGs against global.mrmp's
     * 470,860 -- exactly what the satellites took). */
    ms_mrmpset_t *ss = ms_mrmpset_open(nb_path);
    if (!ss || !ss->n_sets) d3die("no sets in --neighbors chain", nb_path);
    for (uint32_t q = 0; q < ss->n_sets; ++q) {
      uint16_t *grp = d3alloc(n_row, sizeof(uint16_t), "satellite membership");
      ms_mrmp_group_map_at(nb_path, ss->block_off[q], grp, n_row, 4096);
      uint32_t set = ++n_sat;
      uint64_t got = 0;
      uint32_t base_pat = T.n_pat;
      for (uint64_t r = 0; r < n_row; ++r) {
        if (!grp[r]) continue;                 /* 0 = PNA */
        memset(k, 0, kw * sizeof(uint64_t));
        k[0] = grp[r];                         /* rank within this set */
        k[kw - 1] = set;                       /* its own namespace */
        uint32_t p = d3tab_intern(&T, k, set);
        T.count[p]++; ++got;
        D3_PUSH(p, r, 0.0f);
      }
      fprintf(stderr, "[methscope]   satellite %-46s %u patterns / %llu CpGs\n",
              ss->name[q], T.n_pat - base_pat, (unsigned long long)got);
      free(grp);
    }
    ms_mrmpset_free(ss);
  } else
  for (uint32_t a = 0; a < nc; ++a)
    for (uint32_t b = a + 1; b < nc; ++b) {
      if (seg[(size_t)a * nc + b] > max_seg) continue;
      uint32_t set = ++n_sat;
      uint64_t got = 0;
      for (uint64_t r = 0; r < n_row; ++r) {
        /* No exclusion: legacy builds each satellite over the WHOLE row space,
         * independently of the global, and the pool resolves the overlap in the
         * satellite's favour -- its pooled global holds 462,024 CpGs against
         * global.mrmp's 470,860, exactly the 8,836 the satellites took. So a
         * contested row moves; it does not stay with whoever saw it first. */
        uint16_t ba = beta[(size_t)a * n_row + r];
        uint16_t bb = beta[(size_t)b * n_row + r];
        if (ba == D3_NA || bb == D3_NA) continue;
        if ((double)dep[(size_t)a * n_row + r] < target[a] ||
            (double)dep[(size_t)b * n_row + r] < target[b]) continue;
        int sa = (ba > BT) ? (ba >= HI ? 1 : -1) : ((ba == BT || ba > LO) ? -1 : 0);
        int sb = (bb > BT) ? (bb >= HI ? 1 : -1) : ((bb == BT || bb > LO) ? -1 : 0);
        if (sa < 0 || sb < 0 || sa == sb) continue;
        memset(k, 0, kw * sizeof(uint64_t));
        if (sa) k[a >> 6] |= 1ull << (a & 63);
        if (sb) k[b >> 6] |= 1ull << (b & 63);
        k[kw - 1] = set;                /* this satellite's own namespace */
        uint32_t p = d3tab_intern(&T, k, set);
        member[r] = p; ++got;      /* overwrites a global claim on purpose */
        D3_PUSH(p, r, (float)(sa ? ((double)ba - (double)bb)
                                 : ((double)bb - (double)ba)));
      }
      fprintf(stderr, "[methscope]   satellite %s|%s: global separates by %llu"
                      " -> %llu CpGs\n", name[a], name[b],
              (unsigned long long)seg[(size_t)a * nc + b],
              (unsigned long long)got);
    }
  fprintf(stderr, "[methscope] deconv3-build-ref: %u satellites, %u patterns "
                  "total\n", n_sat, T.n_pat);


  /* ---- 3. POOL: rank every pattern by CpG count, keep the top N ------ */
  uint32_t n_keep = T.n_pat < pooled_top ? T.n_pat : (uint32_t)pooled_top;
  uint32_t *order = d3alloc(T.n_pat, sizeof(uint32_t), "order");
  for (uint32_t p = 0; p < T.n_pat; ++p) order[p] = p;
  /* selection by count, descending -- partial is enough but n_pat is small */
  for (uint32_t x = 0; x < n_keep; ++x) {
    uint32_t best = x;
    for (uint32_t y = x + 1; y < T.n_pat; ++y)
      if (T.count[order[y]] > T.count[order[best]]) best = y;
    uint32_t t = order[x]; order[x] = order[best]; order[best] = t;
  }
  uint32_t *remap = d3alloc(T.n_pat, sizeof(uint32_t), "remap");
  for (uint32_t p = 0; p < T.n_pat; ++p) remap[p] = D3_NOPAT;
  for (uint32_t x = 0; x < n_keep; ++x) remap[order[x]] = x;

  /* NO COLLISION RESOLUTION. A chain writes each set as its own block with its
   * own membership, so a CpG may belong to a global pattern AND to a satellite
   * -- they are different features, not competing states for one mask. Legacy's
   * pooled total is the plain sum of its sets (470,860 + 1,540,637 =
   * 2,002,661), which is only possible if nothing is resolved away. Forcing one
   * pattern per row cost 458,324 CpGs and stripped the global set to ~3,700.
   *
   * So membership is stored as a CSR row list per pattern. It is also far
   * smaller than a per-row array: 2.0 M entries at 4 bytes against 29.4 M. */
  uint64_t *poff = d3alloc((size_t)n_keep + 1, sizeof(uint64_t), "pattern offsets");
  uint32_t *rank_of = d3alloc(T.n_pat, sizeof(uint32_t), "rank");
  for (uint32_t p = 0; p < T.n_pat; ++p) rank_of[p] = D3_NOPAT;
  for (uint32_t x = 0; x < n_keep; ++x) rank_of[order[x]] = x;
  if (resolve) {   /* visit entries largest-pattern-first */
    for (size_t z = 0; z < n_ent; ++z)
      ent[z].gap = (rank_of[ent[z].p] == D3_NOPAT)
                 ? -1e9f : -(float)rank_of[ent[z].p];
    qsort(ent, n_ent, sizeof(d3ent_t), d3ent_cmp_rank);
  }
  /* --resolve: one pattern per CpG, the largest claiming first. The bundled
   * .cm in a .refx carries 1001 states -- 1000 patterns plus PNA -- which is a
   * single resolved mask, whereas a chain written as a multi-record mask keeps
   * each set separate and lets a CpG serve several patterns. The two give
   * different design matrices and only one matches legacy; this switch decides
   * it by measurement rather than by reading more of the writer. */
  uint8_t *taken = NULL;
  if (resolve) taken = d3alloc(n_row, 1, "resolved rows");
  uint64_t kept_cpg = 0, drop_cpg = 0;
  for (size_t z = 0; z < n_ent; ++z) {
    uint32_t nid = rank_of[ent[z].p];
    if (nid == D3_NOPAT) { ++drop_cpg; continue; }
    if (resolve) {
      if (taken[ent[z].r]) { ++drop_cpg; continue; }
      taken[ent[z].r] = 1;
    }
    poff[nid + 1]++; ++kept_cpg;
  }
  if (resolve) memset(taken, 0, n_row);
  for (uint32_t x = 0; x < n_keep; ++x) poff[x + 1] += poff[x];
  uint32_t *prow = d3alloc(kept_cpg ? kept_cpg : 1, sizeof(uint32_t), "rows");
  uint64_t *fill = d3alloc((size_t)n_keep, sizeof(uint64_t), "fill");
  for (size_t z = 0; z < n_ent; ++z) {
    uint32_t nid = rank_of[ent[z].p];
    if (nid == D3_NOPAT) continue;
    if (resolve) { if (taken[ent[z].r]) continue; taken[ent[z].r] = 1; }
    prow[poff[nid] + fill[nid]++] = ent[z].r;
  }
  free(taken);
  free(fill); free(rank_of); free(ent);

  /* ---- 4. PROFILE: per-pattern beta for every class ------------------ */
  double *sum = d3alloc((size_t)n_keep * nc, sizeof(double), "profile sums");
  uint64_t *cnt = d3alloc((size_t)n_keep * nc, sizeof(uint64_t), "profile n");
  uint64_t *pcnt = d3alloc(n_keep, sizeof(uint64_t), "pattern counts");
  for (uint32_t p = 0; p < n_keep; ++p) {
    pcnt[p] = poff[p + 1] - poff[p];
    for (uint64_t z = poff[p]; z < poff[p + 1]; ++z) {
      uint32_t r = prow[z];
      for (uint32_t c = 0; c < nc; ++c) {
        uint16_t b = beta[(size_t)c * n_row + r];
        if (b == D3_NA) continue;
        sum[(size_t)p * nc + c] += (double)b / 65534.0;
        cnt[(size_t)p * nc + c]++;
      }
    }
  }

  FILE *out = fopen(out_path, "wb");
  if (!out) d3die("cannot open output", out_path);
  fwrite(D3_MAGIC, 1, 8, out);
  put32(out, 1); put32(out, nc);
  put64(out, n_row); put64(out, n_keep);
  putf(out, qlo); putf(out, qhi); putf(out, beta_thr);
  put32(out, mincov); put32(out, 0);
  for (uint32_t c = 0; c < nc; ++c)
    fwrite(name[c], 1, strlen(name[c]) + 1, out);
  put64(out, kept_cpg);
  fwrite(poff, sizeof(uint64_t), (size_t)n_keep + 1, out);
  fwrite(prow, sizeof(uint32_t), kept_cpg, out);
  fwrite(pcnt, sizeof(uint64_t), n_keep, out);
  uint16_t *brow = d3alloc(nc, sizeof(uint16_t), "beta row");
  for (uint32_t p = 0; p < n_keep; ++p) {
    for (uint32_t c = 0; c < nc; ++c)
      brow[c] = cnt[(size_t)p * nc + c]
              ? d3_enc(sum[(size_t)p * nc + c] / (double)cnt[(size_t)p * nc + c])
              : D3_NA;
    fwrite(brow, sizeof(uint16_t), nc, out);
  }
  if (ferror(out)) d3die("error writing output", out_path);
  fclose(out);

  fprintf(stderr,
    "[methscope] deconv3-build-ref: pooled %u of %u patterns "
    "(%llu CpGs kept, %llu dropped)\n"
    "                               %u classes, band %.2f/%.2f -> %s\n",
    n_keep, T.n_pat, (unsigned long long)kept_cpg,
    (unsigned long long)drop_cpg, nc, qlo, qhi, out_path);

  #undef D3_PUSH
  free(brow); free(sum); free(cnt); free(pcnt); free(order); free(remap);
  free(poff); free(prow);
  free(seg); free(k); free(member); free(beta); free(dep);
  free(target); free(voff);
  d3tab_free(&T);
  for (uint32_t c = 0; c < nc; ++c) free(name[c]);
  free(name);
  return 0;
}

/* ==================================================================== */
/* deconv3: the solver.                                                  */
/* ==================================================================== */
/*
 * The same fit legacy does, off the .d3ref instead of a .refx: per pattern,
 * the query's mean beta over the CpGs it measured there; per class, that
 * pattern's reference profile from the build; then complete-case NNLS over the
 * patterns this record actually observed, each row weighted by sqrt(n).
 *
 * The design matrix is a BUILD-TIME quantity -- profiles were computed over
 * every reference CpG in the pattern -- so unlike deconv2 it carries no
 * per-query sampling noise. Whether that matters is one of the questions this
 * pipeline exists to answer, and it can only be asked once parity holds.
 */
typedef struct {
  uint32_t  n_class;
  uint64_t  n_pat;        /* written as 8 bytes by the builder */
  uint64_t  n_row, n_ent;
  double    qlo, qhi, beta_thr;
  uint32_t  mincov;
  char    **name;
  uint64_t *poff;      /* n_pat + 1 */
  uint32_t *prow;      /* n_ent */
  uint64_t *pcnt;      /* n_pat */
  uint16_t *pbeta;     /* n_pat * n_class, pattern-major */
} d3ref_t;

static void d3ref_load(const char *path, d3ref_t *R) {
  FILE *f = fopen(path, "rb");
  if (!f) d3die("cannot open reference", path);
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, D3_MAGIC, 7))
    d3die("not a .d3ref reference (bad magic)", path);
  uint32_t ver = 0, rsv = 0;
  if (fread(&ver, 4, 1, f) != 1 || fread(&R->n_class, 4, 1, f) != 1 ||
      fread(&R->n_row, 8, 1, f) != 1 || fread(&R->n_pat, 8, 1, f) != 1)
    d3die("truncated header", path);
  if (fread(&R->qlo, 8, 1, f) != 1 || fread(&R->qhi, 8, 1, f) != 1 ||
      fread(&R->beta_thr, 8, 1, f) != 1 || fread(&R->mincov, 4, 1, f) != 1 ||
      fread(&rsv, 4, 1, f) != 1)
    d3die("truncated header", path);
  if (ver != 1) d3die("unsupported .d3ref version", path);
  R->name = d3alloc(R->n_class, sizeof(char *), "class names");
  for (uint32_t c = 0; c < R->n_class; ++c) {
    char buf[512]; size_t n = 0; int ch;
    while ((ch = fgetc(f)) > 0) {
      if (n + 1 >= sizeof buf) d3die("class name too long", path);
      buf[n++] = (char)ch;
    }
    if (ch < 0) d3die("truncated class names", path);
    buf[n] = '\0'; R->name[c] = strdup(buf);
  }
  if (fread(&R->n_ent, 8, 1, f) != 1) d3die("truncated membership", path);
  R->poff  = d3alloc((size_t)R->n_pat + 1, sizeof(uint64_t), "offsets");
  R->prow  = d3alloc(R->n_ent ? R->n_ent : 1, sizeof(uint32_t), "rows");
  R->pcnt  = d3alloc((size_t)R->n_pat, sizeof(uint64_t), "counts");
  R->pbeta = d3alloc((size_t)R->n_pat * R->n_class, sizeof(uint16_t), "profiles");
  if (fread(R->poff, sizeof(uint64_t), (size_t)R->n_pat + 1, f) != (size_t)R->n_pat + 1 ||
      fread(R->prow, sizeof(uint32_t), R->n_ent, f) != R->n_ent ||
      fread(R->pcnt, sizeof(uint64_t), R->n_pat, f) != R->n_pat ||
      fread(R->pbeta, sizeof(uint16_t), (size_t)R->n_pat * R->n_class, f)
        != (size_t)R->n_pat * R->n_class)
    d3die("truncated body", path);
  fclose(f);
  fprintf(stderr, "[methscope] deconv3: reference %u classes x %llu patterns "
                  "over %llu memberships (%.0f MB)\n",
          R->n_class, (unsigned long long)R->n_pat,
          (unsigned long long)R->n_ent,
          (double)(R->n_ent * 4 + (size_t)R->n_pat * R->n_class * 2) / 1e6);
}

static void d3ref_free(d3ref_t *R) {
  for (uint32_t c = 0; c < R->n_class; ++c) free(R->name[c]);
  free(R->name); free(R->poff); free(R->prow); free(R->pcnt); free(R->pbeta);
}

int main_deconv3(int argc, char *argv[]) {
  const char *out_path = NULL;
  uint32_t min_cov = 1;
  int no_header = 0, i = 1;
  for (; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-o") && i + 1 < argc) out_path = argv[++i];
    else if (!strcmp(a, "--min-cov") && i + 1 < argc)
      min_cov = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(a, "--no-header")) no_header = 1;
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      printf(
"Usage: methscope deconv3 [options] <query.cg> <ref.d3ref> -o <out.tsv>\n"
"\n"
"  Deconvolve each query record against a .d3ref. Per pattern, the query's\n"
"  mean beta over the CpGs it measured there; per class, the pattern's\n"
"  reference profile from the build; then complete-case NNLS weighted by\n"
"  sqrt(n), which is the fit the shipped `deconv` performs.\n"
"\n"
"  Options\n"
"    -o FILE        output TSV (default stdout)\n"
"    --min-cov N    only use patterns with >= N covered CpGs (default 1)\n"
"    --no-header    suppress the header line\n");
      return 0;
    }
    else if (a[0] == '-' && a[1]) d3die("unrecognized option", a);
    else break;
  }
  if (argc - i != 2) {
    fprintf(stderr, "Usage: methscope deconv3 <query.cg> <ref.d3ref> -o <out>\n");
    return 1;
  }
  const char *qpath = argv[i], *rpath = argv[i + 1];

  d3ref_t R; d3ref_load(rpath, &R);
  uint32_t n_qname = 0; int64_t *qoff = NULL;
  char **qname = d3_read_idx(qpath, &n_qname, &qoff);

  FILE *out = out_path ? fopen(out_path, "w") : stdout;
  if (!out) d3die("cannot open output", out_path);
  if (!no_header) {
    fputs("cell", out);
    for (uint32_t c = 0; c < R.n_class; ++c) fprintf(out, "\t%s", R.name[c]);
    fputc('\n', out);
  }

  const uint32_t ns = R.n_class;
  double *qsum = d3alloc((size_t)R.n_pat, sizeof(double), "query sums");
  uint64_t *qn = d3alloc((size_t)R.n_pat, sizeof(uint64_t), "query counts");
  double *A = d3alloc((size_t)R.n_pat * ns, sizeof(double), "design");
  double *b = d3alloc((size_t)R.n_pat, sizeof(double), "observed");
  double *x = d3alloc(ns, sizeof(double), "proportions");
  double *aw = d3alloc((size_t)R.n_pat * ns, sizeof(double), "nnls A");
  double *bw = d3alloc((size_t)R.n_pat, sizeof(double), "nnls b");
  double *w  = d3alloc(ns, sizeof(double), "nnls w");
  double *zz = d3alloc((size_t)R.n_pat, sizeof(double), "nnls zz");
  int    *ix = d3alloc(ns, sizeof(int), "nnls index");

  cfile_t qf = open_cfile((char *)qpath);
  for (uint32_t rec = 0; ; ++rec) {
    cdata_t c = read_cdata1(&qf);
    if (!c.n) break;
    decompress_in_situ(&c);
    if (c.fmt != '3') d3die("query must be format-3 (M/U) .cg", qpath);
    if (c.n != R.n_row)
      d3die("query and reference disagree on the CpG row space", qpath);

    memset(qsum, 0, (size_t)R.n_pat * sizeof(double));
    memset(qn, 0, (size_t)R.n_pat * sizeof(uint64_t));
    for (uint64_t p = 0; p < R.n_pat; ++p)
      for (uint64_t z = R.poff[p]; z < R.poff[p + 1]; ++z) {
        uint64_t mu = f3_get_mu(&c, R.prow[z]);
        if (!mu || !MU2cov(mu)) continue;
        qsum[p] += MU2beta(mu); qn[p]++;
      }

    uint32_t nu = 0;
    for (uint64_t p = 0; p < R.n_pat; ++p) {
      if (qn[p] < min_cov) continue;
      double wq = sqrt((double)qn[p]);
      for (uint32_t s = 0; s < ns; ++s) {
        uint16_t rb = R.pbeta[(size_t)p * ns + s];
        A[(size_t)s * (size_t)R.n_pat + nu] =        /* column-major for nnls */
          (rb == D3_NA ? 0.0 : (double)rb / 65534.0) * wq;
      }
      b[nu] = qsum[p] / (double)qn[p] * wq;
      ++nu;
    }
    for (uint32_t s = 0; s < ns; ++s) x[s] = 0;
    if (nu) {
      /* nnls wants A packed at the actual row count, so restride the columns */
      for (uint32_t s = 1; s < ns; ++s)
        memmove(aw + (size_t)s * nu, A + (size_t)s * (size_t)R.n_pat,
                nu * sizeof(double));
      memcpy(aw, A, nu * sizeof(double));
      memcpy(bw, b, nu * sizeof(double));
      int mda = (int)nu, m = (int)nu, n = (int)ns, mode = 0;
      double rnorm = 0;
      nnls_c(aw, &mda, &m, &n, bw, x, &rnorm, w, zz, ix, &mode);
      if (mode != 1)
        fprintf(stderr, "[methscope] deconv3: NNLS mode %d (%u patterns)\n",
                mode, nu);
      double sum = 0;
      for (uint32_t s = 0; s < ns; ++s) { if (x[s] < 0) x[s] = 0; sum += x[s]; }
      if (sum > 0) for (uint32_t s = 0; s < ns; ++s) x[s] /= sum;
    }
    fprintf(out, "%s", rec < n_qname ? qname[rec] : "record");
    for (uint32_t s = 0; s < ns; ++s) fprintf(out, "\t%.6f", x[s]);
    fputc('\n', out);
    free_cdata(&c);
  }
  bgzf_close(qf.fh);
  if (out != stdout) fclose(out);
  free(qsum); free(qn); free(A); free(b); free(x);
  free(aw); free(bw); free(w); free(zz); free(ix); free(qoff);
  for (uint32_t q = 0; q < n_qname; ++q) free(qname[q]);
  free(qname); d3ref_free(&R);
  return 0;
}
