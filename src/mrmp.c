// SPDX-License-Identifier: AGPL-3.0-or-later
/* Native MRMP construction: `methscope mrmp build/inspect/export`.
 * See mrmp.h for the artifact format and the binstring semantics reproduced. */
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "methscope.h"
#include "mrmp.h"
#include "cfile.h"
#include "cdata.h"

/* binstring defaults, matching YAME rowop.c (main_rowop getopt defaults). */
#define MRMP_DEF_MINCOV        1u
#define MRMP_DEF_BETA_THRESH   0.5f
#define MRMP_DEF_MAX_AMBIG     1.0f   /* 1.0 == off */
#define MRMP_DEF_MIN_FOLD      10.0f

static void die(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] mrmp: %s: %s\n", msg, arg);
  else fprintf(stderr, "[methscope] mrmp: %s\n", msg);
  exit(1);
}

static void *xcalloc(size_t n, size_t sz, const char *what) {
  void *p = calloc(n ? n : 1, sz ? sz : 1);
  if (!p) die("out of memory", what);
  return p;
}

static uint64_t parse_u64(const char *s, const char *what) {
  errno = 0; char *e = NULL; unsigned long long v = strtoull(s, &e, 10);
  if (errno || e == s || *e) die("invalid integer", what);
  return (uint64_t)v;
}

/* ---------------- base-3 pattern key <-> string ------------------------- */

/* Decode a base-3 key into a length-`len` string of '0'/'1'/'2' (sample 0 is
 * the most-significant digit; matches the encode loop in build). */
static void key_to_string(uint64_t key, uint32_t len, char *out) {
  for (uint32_t i = 0; i < len; ++i) out[i] = '0';
  out[len] = '\0';
  for (uint32_t i = 0; i < len; ++i) {
    out[len - 1 - i] = (char)('0' + (int)(key % 3));
    key /= 3;
  }
}

/* ---------------- open-addressing hash: u64 key -> pattern slot ---------- */

typedef struct {
  uint64_t *keys;      /* slot -> key (0 empty; real all-0 key IS 0, tracked) */
  uint32_t *slot;      /* slot -> pattern index + 1 (0 == empty) */
  uint64_t cap;        /* power of two */
  uint64_t mask;
} phash_t;

static void phash_init(phash_t *h, uint64_t expect) {
  uint64_t cap = 1024;
  while (cap < expect * 2) cap <<= 1;   /* < 0.5 load */
  h->cap = cap; h->mask = cap - 1;
  h->keys = xcalloc(cap, sizeof(uint64_t), "phash keys");
  h->slot = xcalloc(cap, sizeof(uint32_t), "phash slot");
}

static uint64_t mix64(uint64_t x) {   /* splitmix64 finalizer */
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x ^= x >> 31; return x;
}

/* Return existing pattern index for key, or add via *n_pat and pat[]. */
static uint32_t phash_intern(phash_t *h, uint64_t key,
                             mrmp_pattern_t *pat, uint64_t *n_pat) {
  uint64_t i = mix64(key) & h->mask;
  for (;;) {
    if (!h->slot[i]) {
      uint32_t idx = (uint32_t)(*n_pat)++;
      pat[idx].key = key; pat[idx].count = 0;
      h->keys[i] = key; h->slot[i] = idx + 1;
      return idx;
    }
    if (h->keys[i] == key) return h->slot[i] - 1;
    i = (i + 1) & h->mask;
  }
}

static uint32_t phash_find(const phash_t *h, uint64_t key) {
  uint64_t i = mix64(key) & h->mask;
  for (;;) {
    if (!h->slot[i]) return MRMP_PNA_MEMBERSHIP;
    if (h->keys[i] == key) return h->slot[i] - 1;
    i = (i + 1) & h->mask;
  }
}

/* ---------------- reference sample names (from <ref>.cg.idx) ------------- */

static char **read_sample_names(const char *ref, uint32_t *n_out) {
  char idx[PATH_MAX];
  if (snprintf(idx, sizeof(idx), "%s.idx", ref) >= (int)sizeof(idx))
    die("reference path too long", ref);
  FILE *f = fopen(idx, "r");
  if (!f) die("cannot open reference index (expected <ref>.idx)", idx);
  size_t cap = 64, n = 0;
  char **names = xcalloc(cap, sizeof(char *), "sample names");
  char *line = NULL; size_t lcap = 0; ssize_t len;
  while ((len = getline(&line, &lcap, f)) > 0) {
    char *tab = strpbrk(line, "\t\n");
    size_t nl = tab ? (size_t)(tab - line) : (size_t)len;
    if (!nl) continue;
    if (n == cap) { cap <<= 1; names = realloc(names, cap * sizeof(char *)); }
    names[n] = xcalloc(nl + 1, 1, "sample name");
    memcpy(names[n], line, nl);
    ++n;
  }
  free(line); fclose(f);
  if (!n) die("reference index is empty", idx);
  if (n > 40) die("more than 40 samples exceeds the base-3 uint64 key", idx);
  *n_out = (uint32_t)n;
  return names;
}

/* ---------------- binstring per-CpG resolution -------------------------- */

/* Resolve CpG i from the meth/ambig bit-planes to its base-3 pattern key,
 * reproducing YAME rowop_binstring: ambiguous cells are filled with the CpG's
 * confident majority; a CpG becomes the all-'2' sentinel when its ambiguous
 * fraction exceeds max_ambig or its confident majority is not sweeping. */
static uint64_t resolve_cpg(const uint8_t *meth, const uint8_t *ambig,
                            uint64_t i, uint32_t ns, uint32_t stride,
                            uint64_t n_cpg, float min_fold, float max_ambig,
                            uint64_t pna_key, int *is_pna) {
  uint32_t n1 = 0, namb = 0;
  for (uint32_t g = 0; g < stride; ++g) {
    n1   += (uint32_t)__builtin_popcount(meth[(uint64_t)g * n_cpg + i]);
    namb += (uint32_t)__builtin_popcount(ambig[(uint64_t)g * n_cpg + i]);
  }
  uint32_t n0 = ns - n1 - namb;
  int fill_one = (n1 > n0);                       /* exact tie -> '0' */
  uint32_t hi = fill_one ? n1 : n0, lo = fill_one ? n0 : n1;
  int sweeping = (hi > 0) && (lo == 0 || (double)hi >= min_fold * (double)lo);
  if ((ns && (double)namb > max_ambig * (double)ns) || (namb > 0 && !sweeping)) {
    *is_pna = 1;
    return pna_key;
  }
  *is_pna = 0;
  uint64_t key = 0;
  for (uint32_t s = 0; s < ns; ++s) {
    uint64_t off = (uint64_t)(s >> 3) * n_cpg + i;
    int digit;
    if ((ambig[off] >> (s & 7)) & 1) digit = fill_one ? 1 : 0;
    else digit = (meth[off] >> (s & 7)) & 1;
    key = key * 3 + (uint64_t)digit;
  }
  return key;
}

/* ---------------- ranking (count desc, key asc) ------------------------- */

static mrmp_pattern_t *g_pat;
static int rank_cmp(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  if (g_pat[x].count != g_pat[y].count)
    return g_pat[x].count > g_pat[y].count ? -1 : 1;   /* count desc */
  if (g_pat[x].key != g_pat[y].key)
    return g_pat[x].key < g_pat[y].key ? -1 : 1;        /* key asc */
  return 0;
}

/* ---------------- build -------------------------------------------------- */

static void write_or_die(FILE *fp, const void *p, size_t n, const char *path) {
  if (n && fwrite(p, 1, n, fp) != n) die("write failed", path);
}

static int mrmp_build(int argc, char *argv[]) {
  const char *ref = NULL, *out = NULL;
  uint32_t K = 1000, mincov = MRMP_DEF_MINCOV;
  float beta_thr = MRMP_DEF_BETA_THRESH, max_ambig = MRMP_DEF_MAX_AMBIG,
        min_fold = MRMP_DEF_MIN_FOLD;
  int include_homogeneous = 1, force = 0;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      fprintf(stderr,
        "Usage: methscope mrmp build --reference REF.cg -o OUT.mrmp [options]\n\n"
        "Reproduce YAME `rowop -o binstring` per CpG over the reference samples,\n"
        "count exact membership patterns, rank them (count desc, key asc), and\n"
        "select the top -K as P1..PK. The all-'2' sentinel is always PNA.\n\n"
        "  --reference PATH   discretized format-3 reference .cg (needs <ref>.idx)\n"
        "  -o PATH            output MRMPIDX1 artifact (required)\n"
        "  --patterns K       encoder patterns to select (default 1000)\n"
        "  --[no-]include-homogeneous  keep all-0/all-1 candidates (default on)\n"
        "  --mincov N         binstring -c (default 1)\n"
        "  --beta-threshold X binstring -b (default 0.5)\n"
        "  --max-ambig-frac X binstring -m (default 1.0 = off)\n"
        "  --min-major-fold X binstring -M (default 10)\n"
        "  --force            overwrite an existing output\n");
      return 0;
    }
    else if (!strcmp(a, "--reference") && i + 1 < argc) ref = argv[++i];
    else if (!strcmp(a, "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(a, "--patterns") && i + 1 < argc) K = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--include-homogeneous")) include_homogeneous = 1;
    else if (!strcmp(a, "--no-include-homogeneous")) include_homogeneous = 0;
    else if (!strcmp(a, "--mincov") && i + 1 < argc) mincov = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--beta-threshold") && i + 1 < argc) beta_thr = (float)atof(argv[++i]);
    else if (!strcmp(a, "--max-ambig-frac") && i + 1 < argc) max_ambig = (float)atof(argv[++i]);
    else if (!strcmp(a, "--min-major-fold") && i + 1 < argc) min_fold = (float)atof(argv[++i]);
    else if (!strcmp(a, "--force")) force = 1;
    else die("unrecognized or incomplete option", a);
  }
  if (!ref || !out) die("need --reference and -o (see mrmp build -h)", NULL);
  if (!K) die("--patterns must be positive", NULL);
  if (!force) { struct stat st; if (!stat(out, &st)) die("output exists (use --force)", out); }

  uint32_t ns = 0;
  char **names = read_sample_names(ref, &ns);
  const uint32_t stride = (ns + 7) >> 3;        /* bytes per sample byte-group */

  /* Pass 1: stream each sample record into two bit-planes over all CpGs.
   *   meth  bit set  <=> confident methylated (beta > threshold, covered)
   *   ambig bit set  <=> zero/low coverage OR beta == threshold (M==U tie)
   * confident unmethylated == neither bit. Layout matches rowop_binstring:
   * byte index (g*n_cpg + i), bit (k & 7) for sample k, group g = k>>3. */
  uint64_t n_cpg = 0;
  uint8_t *meth = NULL, *ambig = NULL;
  cfile_t cf = open_cfile((char *)ref);
  uint32_t k = 0;
  for (;; ++k) {
    cdata_t c = read_cdata1(&cf);
    if (!c.n) { free_cdata(&c); break; }
    decompress_in_situ(&c);
    if (c.fmt != '3') die("reference must be format-3 (M/U) .cg", ref);
    if (k == 0) {
      n_cpg = c.n;
      meth  = xcalloc((size_t)stride * n_cpg, 1, "meth plane");
      ambig = xcalloc((size_t)stride * n_cpg, 1, "ambig plane");
    } else if (c.n != n_cpg) {
      die("reference samples disagree on CpG count", ref);
    }
    const uint64_t base = (uint64_t)(k >> 3) * n_cpg;
    const uint8_t bit = (uint8_t)(1u << (k & 7));
    for (uint64_t i = 0; i < n_cpg; ++i) {
      uint64_t mu = f3_get_mu(&c, i);
      if (!mu || MU2cov(mu) < mincov) {
        ambig[base + i] |= bit;
      } else {
        double beta = MU2beta(mu);
        if (beta > beta_thr) meth[base + i] |= bit;
        else if (beta == beta_thr) ambig[base + i] |= bit;
        /* else confident unmethylated: leave both clear */
      }
    }
    free_cdata(&c);
  }
  bgzf_close(cf.fh);
  if (k != ns) die("reference record count differs from <ref>.idx", ref);
  if (!n_cpg) die("reference is empty", ref);

  /* Pass 2: resolve every CpG to a pattern key and count patterns. The keys
   * are not stored; the membership pass below re-derives them from the planes,
   * trading a cheap recompute for ~n_cpg*8 bytes of RAM. */
  uint64_t pna_key = 0;
  for (uint32_t s = 0; s < ns; ++s) pna_key = pna_key * 3 + 2;  /* all-'2' */

  phash_t h; phash_init(&h, 1u << 21);          /* ~2.4M patterns, < 0.6 load */
  uint64_t pat_cap = 1u << 20, n_pat = 0;
  mrmp_pattern_t *pat = xcalloc(pat_cap, sizeof(mrmp_pattern_t), "pattern table");
  uint64_t pna_cpg = 0, checksum = 1469598103934665603ULL;  /* FNV-1a offset */

  for (uint64_t i = 0; i < n_cpg; ++i) {
    int is_pna;
    uint64_t key = resolve_cpg(meth, ambig, i, ns, stride, n_cpg,
                               min_fold, max_ambig, pna_key, &is_pna);
    if (is_pna) {
      ++pna_cpg;
    } else {
      if (n_pat == pat_cap) {                   /* grow (rare: > 2^20 patterns) */
        pat_cap <<= 1; pat = realloc(pat, pat_cap * sizeof(*pat));
        if (!pat) die("out of memory", "pattern table grow");
      }
      uint32_t idx = phash_intern(&h, key, pat, &n_pat);
      ++pat[idx].count;
    }
    checksum = (checksum ^ key) * 1099511628211ULL;
  }

  /* Candidate set: all {0,1} patterns. Homogeneous all-0/all-1 are the two
   * patterns with a single distinct digit; drop them when not included. */
  uint64_t all0_key = 0, all1_key = 0;
  for (uint32_t s = 0; s < ns; ++s) all1_key = all1_key * 3 + 1;
  /* rank candidates */
  uint32_t *order = xcalloc(n_pat, sizeof(uint32_t), "rank order");
  uint64_t n_cand = 0;
  for (uint64_t p = 0; p < n_pat; ++p) {
    if (!include_homogeneous &&
        (pat[p].key == all0_key || pat[p].key == all1_key)) continue;
    order[n_cand++] = (uint32_t)p;
  }
  g_pat = pat;
  qsort(order, n_cand, sizeof(uint32_t), rank_cmp);

  /* rank_of[pattern index] = 0-based rank among candidates (or PNA). */
  uint32_t *rank_of = xcalloc(n_pat, sizeof(uint32_t), "rank_of");
  for (uint64_t p = 0; p < n_pat; ++p) rank_of[p] = MRMP_PNA_MEMBERSHIP;
  for (uint64_t r = 0; r < n_cand; ++r) rank_of[order[r]] = (uint32_t)r;

  if (K > n_cand) {
    fprintf(stderr, "[methscope] mrmp: only %" PRIu64 " candidates < K=%u; "
            "selecting all\n", n_cand, K);
    K = (uint32_t)n_cand;
  }

  /* Serialize MRMPIDX1. Sections are laid out in header order. */
  FILE *fp = fopen(out, "wb");
  if (!fp) die("cannot create output", out);
  mrmp_header_t hd; memset(&hd, 0, sizeof(hd));
  if (sizeof(hd) != 128) die("MRMPIDX1 header is not 128 bytes", NULL);
  memcpy(hd.magic, MRMPIDX_MAGIC, 8);
  hd.version = MRMPIDX_VERSION; hd.n_samples = ns; hd.n_selected = K;
  hd.flags = include_homogeneous ? MRMP_FLAG_INCLUDE_HOMOGENEOUS : 0;
  hd.n_cpg = n_cpg; hd.n_candidates = n_cand;
  hd.pna_key = pna_key; hd.pna_cpg = pna_cpg;
  hd.mincov = mincov; hd.beta_threshold = beta_thr;
  hd.max_ambig_frac = max_ambig; hd.min_major_fold = min_fold;
  hd.content_checksum = checksum;

  uint64_t off = sizeof(hd);
  hd.refname_offset = off;   off += strlen(ref) + 1;
  hd.names_offset = off;     for (uint32_t s = 0; s < ns; ++s) off += strlen(names[s]) + 1;
  hd.patterns_offset = off;  off += n_cand * sizeof(mrmp_pattern_t);
  hd.membership_offset = off;

  write_or_die(fp, &hd, sizeof(hd), out);
  write_or_die(fp, ref, strlen(ref) + 1, out);
  for (uint32_t s = 0; s < ns; ++s) write_or_die(fp, names[s], strlen(names[s]) + 1, out);
  for (uint64_t r = 0; r < n_cand; ++r)
    write_or_die(fp, &pat[order[r]], sizeof(mrmp_pattern_t), out);
  /* per-CpG membership rank (re-derive keys from the planes; PNA sentinel
   * where the CpG is the all-'2' pattern). */
  {
    const uint64_t CHUNK = 1u << 20;
    uint32_t *buf = xcalloc(CHUNK, sizeof(uint32_t), "membership buffer");
    for (uint64_t i = 0; i < n_cpg; ) {
      uint64_t m = n_cpg - i < CHUNK ? n_cpg - i : CHUNK;
      for (uint64_t j = 0; j < m; ++j) {
        int is_pna;
        uint64_t key = resolve_cpg(meth, ambig, i + j, ns, stride, n_cpg,
                                   min_fold, max_ambig, pna_key, &is_pna);
        uint32_t p = is_pna ? MRMP_PNA_MEMBERSHIP : phash_find(&h, key);
        buf[j] = (p == MRMP_PNA_MEMBERSHIP) ? MRMP_PNA_MEMBERSHIP : rank_of[p];
      }
      write_or_die(fp, buf, (size_t)m * sizeof(uint32_t), out);
      i += m;
    }
    free(buf);
  }
  free(meth); free(ambig);
  if (fclose(fp)) die("error closing output", out);

  fprintf(stderr,
    "[methscope] mrmp build: %s\n"
    "  samples=%u  CpGs=%" PRIu64 "  distinct patterns=%" PRIu64
    " (+PNA)  candidates=%" PRIu64 "\n"
    "  selected K=%u  PNA CpGs=%" PRIu64 " (%.2f%%)  checksum=%016" PRIx64 "\n",
    out, ns, n_cpg, n_pat, n_cand, K, pna_cpg,
    100.0 * (double)pna_cpg / (double)n_cpg, checksum);

  free(pat); free(order); free(rank_of);
  free(h.keys); free(h.slot);
  for (uint32_t s = 0; s < ns; ++s) free(names[s]);
  free(names);
  return 0;
}

/* ---------------- shared reader for inspect / export -------------------- */

typedef struct {
  void *map; size_t bytes; int fd;
  const mrmp_header_t *h;
  const char *refname;
  const char **names;           /* n_samples */
  const mrmp_pattern_t *pat;    /* n_candidates (rank order) */
  const uint32_t *membership;   /* n_cpg */
} mrmp_reader_t;

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static void mrmp_open(mrmp_reader_t *r, const char *path) {
  memset(r, 0, sizeof(*r));
  int fd = open(path, O_RDONLY);
  if (fd < 0) die("cannot open MRMP artifact", path);
  struct stat st;
  if (fstat(fd, &st) || (uint64_t)st.st_size < sizeof(mrmp_header_t))
    die("MRMP artifact is truncated", path);
  void *m = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) die("cannot mmap MRMP artifact", path);
  const mrmp_header_t *h = (const mrmp_header_t *)m;
  if (memcmp(h->magic, MRMPIDX_MAGIC, 8) || h->version != MRMPIDX_VERSION)
    die("bad MRMPIDX1 magic or version", path);
  uint64_t need = h->membership_offset + h->n_cpg * sizeof(uint32_t);
  if (h->patterns_offset + h->n_candidates * sizeof(mrmp_pattern_t) >
        h->membership_offset || need > (uint64_t)st.st_size)
    die("MRMP artifact offsets are out of bounds", path);
  r->map = m; r->bytes = st.st_size; r->fd = fd; r->h = h;
  r->refname = (const char *)m + h->refname_offset;
  r->names = xcalloc(h->n_samples, sizeof(char *), "names index");
  const char *p = (const char *)m + h->names_offset;
  for (uint32_t s = 0; s < h->n_samples; ++s) {
    ((const char **)r->names)[s] = p; p += strlen(p) + 1;
  }
  r->pat = (const mrmp_pattern_t *)((const char *)m + h->patterns_offset);
  r->membership = (const uint32_t *)((const char *)m + h->membership_offset);
}

static void mrmp_close(mrmp_reader_t *r) {
  free((void *)r->names);
  if (r->map) munmap(r->map, r->bytes);
  if (r->fd >= 0) close(r->fd);
}

/* ---------------- inspect ----------------------------------------------- */

static int mrmp_inspect(int argc, char *argv[]) {
  const char *path = NULL; int show_patterns = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      fprintf(stderr, "Usage: methscope mrmp inspect FILE.mrmp [--patterns]\n");
      return 0;
    } else if (!strcmp(argv[i], "--patterns")) show_patterns = 1;
    else if (argv[i][0] != '-') path = argv[i];
    else die("unrecognized option", argv[i]);
  }
  if (!path) die("need a FILE.mrmp", NULL);
  mrmp_reader_t r; mrmp_open(&r, path);
  const mrmp_header_t *h = r.h;
  printf("format\tMRMPIDX1 v%u\n", h->version);
  printf("reference\t%s\n", r.refname);
  printf("samples\t%u\n", h->n_samples);
  printf("pattern_length\t%u\n", h->n_samples);
  printf("cpgs\t%" PRIu64 "\n", h->n_cpg);
  printf("distinct_candidates\t%" PRIu64 "\n", h->n_candidates);
  printf("selected_patterns\t%u\n", h->n_selected);
  printf("include_homogeneous\t%s\n",
         (h->flags & MRMP_FLAG_INCLUDE_HOMOGENEOUS) ? "yes" : "no");
  printf("pna_cpgs\t%" PRIu64 "\t%.4f%%\n", h->pna_cpg,
         100.0 * (double)h->pna_cpg / (double)h->n_cpg);
  printf("binstring\tmincov=%u beta=%.3f max_ambig_frac=%.3f min_major_fold=%.3f\n",
         h->mincov, h->beta_threshold, h->max_ambig_frac, h->min_major_fold);
  printf("content_checksum\t%016" PRIx64 "\n", h->content_checksum);
  char *buf = xcalloc(h->n_samples + 1, 1, "string buffer");
  key_to_string(h->pna_key, h->n_samples, buf);
  printf("pna_pattern\t%s\n", buf);
  if (show_patterns) {
    uint64_t lim = h->n_selected < h->n_candidates ? h->n_selected : h->n_candidates;
    printf("#pattern\tlabel\tcount\n");
    for (uint64_t p = 0; p < lim; ++p) {
      key_to_string(r.pat[p].key, h->n_samples, buf);
      printf("%s\tP%" PRIu64 "\t%" PRIu64 "\n", buf, p + 1, r.pat[p].count);
    }
  }
  free(buf);
  mrmp_close(&r);
  return 0;
}

/* ---------------- export ------------------------------------------------ */

static int mrmp_export(int argc, char *argv[]) {
  const char *path = NULL, *mask = NULL, *patterns = NULL, *counts = NULL,
             *pna_label = "Pna";
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      fprintf(stderr,
        "Usage: methscope mrmp export FILE.mrmp [--mask CM] [--patterns TSV]\n"
        "                             [--counts TSV] [--pna-label NAME]\n\n"
        "  --mask CM        per-CpG P1..PK/Pna labels as a YAME format-2 .cm\n"
        "  --patterns TSV   selected patterns: string<tab>P<rank><tab>count\n"
        "  --counts TSV     every pattern (incl. PNA): count<tab>string\n"
        "  --pna-label NAME background label in --mask (default Pna)\n");
      return 0;
    } else if (!strcmp(a, "--mask") && i + 1 < argc) mask = argv[++i];
    else if (!strcmp(a, "--patterns") && i + 1 < argc) patterns = argv[++i];
    else if (!strcmp(a, "--counts") && i + 1 < argc) counts = argv[++i];
    else if (!strcmp(a, "--pna-label") && i + 1 < argc) pna_label = argv[++i];
    else if (a[0] != '-') path = a;
    else die("unrecognized option", a);
  }
  if (!path) die("need a FILE.mrmp", NULL);
  if (!mask && !patterns && !counts) die("nothing to export (see mrmp export -h)", NULL);
  mrmp_reader_t r; mrmp_open(&r, path);
  const mrmp_header_t *h = r.h;
  const uint32_t ns = h->n_samples;
  char *buf = xcalloc(ns + 1, 1, "string buffer");

  if (patterns) {
    FILE *f = fopen(patterns, "w");
    if (!f) die("cannot create --patterns", patterns);
    uint64_t lim = h->n_selected < h->n_candidates ? h->n_selected : h->n_candidates;
    for (uint64_t p = 0; p < lim; ++p) {
      key_to_string(r.pat[p].key, ns, buf);
      fprintf(f, "%s\tP%" PRIu64 "\t%" PRIu64 "\n", buf, p + 1, r.pat[p].count);
    }
    if (fclose(f)) die("error closing --patterns", patterns);
  }

  if (counts) {
    /* Every candidate in rank order, then the PNA sentinel. */
    FILE *f = fopen(counts, "w");
    if (!f) die("cannot create --counts", counts);
    for (uint64_t p = 0; p < h->n_candidates; ++p) {
      key_to_string(r.pat[p].key, ns, buf);
      fprintf(f, "%" PRIu64 "\t%s\n", r.pat[p].count, buf);
    }
    key_to_string(h->pna_key, ns, buf);
    fprintf(f, "%" PRIu64 "\t%s\n", h->pna_cpg, buf);
    if (fclose(f)) die("error closing --counts", counts);
  }

  if (mask) {
    /* Build a raw YAME format-2 cdata directly (no genome-sized text file),
     * mirroring fmt2_read_raw: first-seen key order over genomic CpGs, then
     * cdata_compress (RLE) + cdata_write. Labels: P(rank+1) or the PNA label. */
    const uint64_t n = h->n_cpg, K = h->n_selected;
    /* distinct label ids in first-seen order */
    uint32_t *label_id = xcalloc(n, sizeof(uint32_t), "label ids");
    /* map rank(<K) -> key id; PNA and below-K share the PNA label. */
    int32_t *key_of_rank = xcalloc(K, sizeof(int32_t), "rank->keyid");
    for (uint64_t j = 0; j < K; ++j) key_of_rank[j] = -1;
    int32_t pna_id = -1;
    uint64_t n_keys = 0;
    /* worst case: K selected labels + 1 PNA */
    char **keys = xcalloc(K + 1, sizeof(char *), "label keys");
    size_t keys_bytes = 0;
    for (uint64_t i = 0; i < n; ++i) {
      uint32_t rank = r.membership[i];
      int is_sel = (rank != MRMP_PNA_MEMBERSHIP && rank < K);
      int32_t id;
      if (is_sel) {
        if (key_of_rank[rank] < 0) {
          char lbl[32]; int m = snprintf(lbl, sizeof(lbl), "P%u", rank + 1);
          keys[n_keys] = xcalloc((size_t)m + 1, 1, "label");
          memcpy(keys[n_keys], lbl, m);
          keys_bytes += (size_t)m + 1;
          key_of_rank[rank] = (int32_t)n_keys++;
        }
        id = key_of_rank[rank];
      } else {
        if (pna_id < 0) {
          size_t m = strlen(pna_label);
          keys[n_keys] = xcalloc(m + 1, 1, "pna label");
          memcpy(keys[n_keys], pna_label, m);
          keys_bytes += m + 1;
          pna_id = (int32_t)n_keys++;
        }
        id = pna_id;
      }
      label_id[i] = (uint32_t)id;
    }
    /* assemble raw fmt2 cdata: [keys \0-joined][\0][data: n * 8 LE] */
    cdata_t c; memset(&c, 0, sizeof(c));
    c.fmt = '2'; c.compressed = 0; c.unit = 8; c.n = n;
    c.aux = calloc(1, sizeof(f2_aux_t));
    c.s = xcalloc(keys_bytes + 1 + (size_t)n * 8, 1, "fmt2 raw buffer");
    f2_aux_t *aux = (f2_aux_t *)c.aux;
    aux->nk = n_keys;
    aux->keys = xcalloc(n_keys, sizeof(char *), "aux keys");
    size_t pos = 0;
    for (uint64_t kk = 0; kk < n_keys; ++kk) {
      size_t m = strlen(keys[kk]);
      memcpy(c.s + pos, keys[kk], m);
      aux->keys[kk] = (char *)(c.s + pos);
      pos += m; c.s[pos++] = '\0';
    }
    c.s[pos++] = '\0';                 /* key/data separator (double NUL) */
    aux->data = c.s + pos;
    for (uint64_t i = 0; i < n; ++i) {
      uint8_t *d = c.s + pos + i * 8;
      for (int b = 0; b < 8; ++b) d[b] = (uint8_t)(label_id[i] >> (8 * b));
    }
    cdata_compress(&c);               /* -> RLE fmt2 */
    cdata_write((char *)mask, &c, "w", 0);
    free_cdata(&c);
    for (uint64_t kk = 0; kk < n_keys; ++kk) free(keys[kk]);
    free(keys); free(label_id); free(key_of_rank);
    fprintf(stderr, "[methscope] mrmp export: wrote %s (%" PRIu64 " labels)\n",
            mask, n_keys);
  }

  free(buf);
  mrmp_close(&r);
  return 0;
}

/* ---------------- dispatch ---------------------------------------------- */

int main_mrmp(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr,
      "Usage: methscope mrmp <build|inspect|export> ...\n\n"
      "  build    construct an MRMPIDX1 mask from a reference .cg\n"
      "  inspect  report dimensions and selected patterns\n"
      "  export   emit a YAME .cm mask and/or pattern/count TSVs\n");
    return 1;
  }
  if (!strcmp(argv[1], "build"))   return mrmp_build(argc - 1, argv + 1);
  if (!strcmp(argv[1], "inspect")) return mrmp_inspect(argc - 1, argv + 1);
  if (!strcmp(argv[1], "export"))  return mrmp_export(argc - 1, argv + 1);
  fprintf(stderr, "[methscope] mrmp: unknown subcommand '%s'\n", argv[1]);
  return 1;
}
