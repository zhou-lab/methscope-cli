// SPDX-License-Identifier: AGPL-3.0-or-later
/* Native MRMP construction: `methscope mrmp-build` / `mrmp-export`, plus the
 * MRMPIDX1 arm of `methscope inspect`.
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

/* Message prefix: the running subcommand, or plain "mrmp" when the ms_mrmp_*
 * entry points are called as a library from another command. */
static const char *g_cmd = "mrmp";

static void die(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s: %s\n", g_cmd, msg, arg);
  else fprintf(stderr, "[methscope] %s: %s\n", g_cmd, msg);
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

/* A key is mrmp_key_words(ns) words; word w holds samples [40w, 40w+40), most
 * significant digit first, so comparing words in order compares the sample
 * string in order. With ns <= 40 this is one word packed exactly as before. */

/* Decode a key into a length-`len` string of '0'/'1'/'2'. */
static void key_to_string(const uint64_t *key, uint32_t len, char *out) {
  for (uint32_t i = 0; i < len; ++i) out[i] = '0';
  out[len] = '\0';
  uint32_t nw = mrmp_key_words(len);
  for (uint32_t w = 0; w < nw; ++w) {
    uint32_t lo = w * MRMP_TRITS_PER_WORD;
    uint32_t n  = len - lo < MRMP_TRITS_PER_WORD ? len - lo : MRMP_TRITS_PER_WORD;
    uint64_t k  = key[w];
    for (uint32_t i = 0; i < n; ++i) {         /* fill this word right to left */
      out[lo + n - 1 - i] = (char)('0' + (int)(k % 3));
      k /= 3;
    }
  }
}

/* Lexicographic by sample order, which is word order (see the packing note). */
static int key_cmp(const uint64_t *a, const uint64_t *b, uint32_t nw) {
  for (uint32_t w = 0; w < nw; ++w)
    if (a[w] != b[w]) return a[w] < b[w] ? -1 : 1;
  return 0;
}

/* ---------------- open-addressing hash: key -> pattern slot -------------- */

typedef struct {
  uint64_t *keys;      /* cap * nw words; emptiness is slot[i] == 0, not the key */
  uint32_t *slot;      /* slot -> pattern index + 1 (0 == empty) */
  uint64_t cap;        /* power of two */
  uint64_t mask;
  uint64_t used;       /* occupied slots; drives the grow-at-70% rebuild */
  uint32_t nw;         /* key words */
} phash_t;

static void phash_init(phash_t *h, uint64_t expect, uint32_t nw) {
  uint64_t cap = 1024;
  while (cap < expect * 2) cap <<= 1;   /* headroom below the 0.7 load cap */
  h->cap = cap; h->mask = cap - 1; h->used = 0; h->nw = nw;
  h->keys = xcalloc(cap * nw, sizeof(uint64_t), "phash keys");
  h->slot = xcalloc(cap, sizeof(uint32_t), "phash slot");
}

static uint64_t mix64(uint64_t x) {   /* splitmix64 finalizer */
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x ^= x >> 31; return x;
}

/* Fold every word in, so two patterns differing only past word 0 do not
 * collide into the same probe sequence. */
static uint64_t key_hash(const uint64_t *key, uint32_t nw) {
  uint64_t x = 0;
  for (uint32_t w = 0; w < nw; ++w) x = mix64(x ^ key[w]);
  return x;
}

/* Double the table and re-key every occupied slot. Without this a reference
 * with more distinct patterns than the initial table could hold would spin
 * forever (open addressing never finds an empty slot once full), so a denser-
 * than-expected reference must trigger a grow, not a silent hang. */
static void phash_rebuild(phash_t *h) {
  uint64_t ncap = h->cap << 1, nmask = ncap - 1;
  uint32_t nw = h->nw;
  uint64_t *nkeys = xcalloc(ncap * nw, sizeof(uint64_t), "phash keys");
  uint32_t *nslot = xcalloc(ncap, sizeof(uint32_t), "phash slot");
  for (uint64_t i = 0; i < h->cap; ++i) {
    if (!h->slot[i]) continue;
    uint64_t j = key_hash(h->keys + i * nw, nw) & nmask;
    while (nslot[j]) j = (j + 1) & nmask;
    memcpy(nkeys + j * nw, h->keys + i * nw, nw * sizeof(uint64_t));
    nslot[j] = h->slot[i];
  }
  free(h->keys); free(h->slot);
  h->keys = nkeys; h->slot = nslot; h->cap = ncap; h->mask = nmask;
}

/* Return existing pattern index for key, or append to keys[]/counts[]. */
static uint32_t phash_intern(phash_t *h, const uint64_t *key,
                             uint64_t *pkeys, uint64_t *pcount, uint64_t *n_pat) {
  if ((h->used + 1) * 10 >= h->cap * 7) phash_rebuild(h); /* grow at 70% */
  uint32_t nw = h->nw;
  uint64_t i = key_hash(key, nw) & h->mask;
  for (;;) {
    if (!h->slot[i]) {
      uint32_t idx = (uint32_t)(*n_pat)++;
      memcpy(pkeys + (uint64_t)idx * nw, key, nw * sizeof(uint64_t));
      pcount[idx] = 0;
      memcpy(h->keys + i * nw, key, nw * sizeof(uint64_t));
      h->slot[i] = idx + 1; ++h->used;
      return idx;
    }
    if (!key_cmp(h->keys + i * nw, key, nw)) return h->slot[i] - 1;
    i = (i + 1) & h->mask;
  }
}

static uint32_t phash_find(const phash_t *h, const uint64_t *key) {
  uint32_t nw = h->nw;
  uint64_t i = key_hash(key, nw) & h->mask;
  for (;;) {
    if (!h->slot[i]) return MRMP_PNA_MEMBERSHIP;
    if (!key_cmp(h->keys + i * nw, key, nw)) return h->slot[i] - 1;
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
  /* No sample cap: the key widens to ceil(n/40) words (see mrmp.h). */
  *n_out = (uint32_t)n;
  return names;
}

/* ---------------- binstring per-CpG resolution -------------------------- */

/* Resolve CpG i from the meth/ambig bit-planes to its base-3 pattern key,
 * reproducing YAME rowop_binstring: ambiguous cells are filled with the CpG's
 * confident majority; a CpG becomes the all-'2' sentinel when its ambiguous
 * fraction exceeds max_ambig or its confident majority is not sweeping. */
static void resolve_cpg(const uint8_t *meth, const uint8_t *ambig,
                        uint64_t i, uint32_t ns, uint32_t stride,
                        uint64_t n_cpg, float min_fold, float max_ambig,
                        const uint64_t *pna_key, int *is_pna, uint64_t *key) {
  uint32_t n1 = 0, namb = 0;
  for (uint32_t g = 0; g < stride; ++g) {
    n1   += (uint32_t)__builtin_popcount(meth[(uint64_t)g * n_cpg + i]);
    namb += (uint32_t)__builtin_popcount(ambig[(uint64_t)g * n_cpg + i]);
  }
  uint32_t n0 = ns - n1 - namb;
  int fill_one = (n1 > n0);                       /* exact tie -> '0' */
  uint32_t hi = fill_one ? n1 : n0, lo = fill_one ? n0 : n1;
  int sweeping = (hi > 0) && (lo == 0 || (double)hi >= min_fold * (double)lo);
  uint32_t nw = mrmp_key_words(ns);
  if ((ns && (double)namb > max_ambig * (double)ns) || (namb > 0 && !sweeping)) {
    *is_pna = 1;
    memcpy(key, pna_key, nw * sizeof(uint64_t));
    return;
  }
  *is_pna = 0;
  for (uint32_t w = 0; w < nw; ++w) key[w] = 0;
  for (uint32_t s = 0; s < ns; ++s) {
    uint64_t off = (uint64_t)(s >> 3) * n_cpg + i;
    int digit;
    if ((ambig[off] >> (s & 7)) & 1) digit = fill_one ? 1 : 0;
    else digit = (meth[off] >> (s & 7)) & 1;
    key[s / MRMP_TRITS_PER_WORD] = key[s / MRMP_TRITS_PER_WORD] * 3 + (uint64_t)digit;
  }
}

/* ---------------- ranking (count desc, key asc) ------------------------- */

static const uint64_t *g_keys;   /* n_pat * g_nw */
static const uint64_t *g_count;
static uint32_t g_nw;
static int rank_cmp(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  if (g_count[x] != g_count[y])
    return g_count[x] > g_count[y] ? -1 : 1;            /* count desc */
  return key_cmp(g_keys + (uint64_t)x * g_nw,           /* key asc */
                 g_keys + (uint64_t)y * g_nw, g_nw);
}

/* ---------------- build -------------------------------------------------- */

static void write_or_die(FILE *fp, const void *p, size_t n, const char *path) {
  if (n && fwrite(p, 1, n, fp) != n) die("write failed", path);
}

int main_mrmp_build(int argc, char *argv[]) {
  g_cmd = "mrmp-build";
  const char *pos[2] = {NULL, NULL};
  int npos = 0;
  uint32_t mincov = MRMP_DEF_MINCOV;
  float beta_thr = MRMP_DEF_BETA_THRESH, max_ambig = MRMP_DEF_MAX_AMBIG,
        min_fold = MRMP_DEF_MIN_FOLD;
  int force = 0;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stderr,
        "Usage: methscope mrmp-build [options] REF.cg OUT.mrmp\n\n"
        "Reproduce YAME `rowop -o binstring` per CpG over the reference samples,\n"
        "count exact membership patterns, and rank them (count desc, key asc).\n"
        "Every candidate is ranked and every CpG keeps its exact rank, so the\n"
        "top-K cut belongs to the consumer, not here. The all-'2' sentinel is PNA.\n\n"
        "  REF.cg             discretized format-3 reference .cg (needs <ref>.idx)\n"
        "  OUT.mrmp           output MRMPIDX1 artifact\n\n"
        "  --mincov N         binstring -c (default 1)\n"
        "  --beta-threshold X binstring -b (default 0.5)\n"
        "  --max-ambig-frac X binstring -m (default 1.0 = off)\n"
        "  --min-major-fold X binstring -M (default 10)\n"
        "  --force            overwrite an existing output\n");
      return 0;
    }
    else if (!strcmp(a, "--mincov") && i + 1 < argc) mincov = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--beta-threshold") && i + 1 < argc) beta_thr = (float)atof(argv[++i]);
    else if (!strcmp(a, "--max-ambig-frac") && i + 1 < argc) max_ambig = (float)atof(argv[++i]);
    else if (!strcmp(a, "--min-major-fold") && i + 1 < argc) min_fold = (float)atof(argv[++i]);
    else if (!strcmp(a, "--force")) force = 1;
    else if (a[0] == '-') die("unrecognized or incomplete option", a);
    else if (npos < 2) pos[npos++] = a;
    else die("too many arguments", a);
  }
  if (npos != 2) die("need REF.cg and OUT.mrmp (see mrmp-build -h)", NULL);
  const char *ref = pos[0], *out = pos[1];
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
  uint32_t nw = mrmp_key_words(ns);
  uint64_t *pna_key = xcalloc(nw, sizeof(uint64_t), "pna key");
  for (uint32_t s = 0; s < ns; ++s)
    pna_key[s / MRMP_TRITS_PER_WORD] = pna_key[s / MRMP_TRITS_PER_WORD] * 3 + 2;

  phash_t h; phash_init(&h, 1u << 21, nw); /* ~2.4M patterns; grows past 0.7 load */
  uint64_t pat_cap = 1u << 20, n_pat = 0;
  /* Keys and counts are separate arrays because a record is 8*nw+8 bytes, not a
   * fixed struct; they are interleaved back into that layout on write. */
  uint64_t *pkeys = xcalloc(pat_cap * nw, sizeof(uint64_t), "pattern keys");
  uint64_t *pcount = xcalloc(pat_cap, sizeof(uint64_t), "pattern counts");
  uint64_t pna_cpg = 0, checksum = 1469598103934665603ULL;  /* FNV-1a offset */
  uint64_t *key = xcalloc(nw, sizeof(uint64_t), "cpg key");

  for (uint64_t i = 0; i < n_cpg; ++i) {
    int is_pna;
    resolve_cpg(meth, ambig, i, ns, stride, n_cpg,
                min_fold, max_ambig, pna_key, &is_pna, key);
    if (is_pna) {
      ++pna_cpg;
    } else {
      if (n_pat == pat_cap) {                   /* grow (rare: > 2^20 patterns) */
        pat_cap <<= 1;
        pkeys = realloc(pkeys, pat_cap * nw * sizeof(*pkeys));
        pcount = realloc(pcount, pat_cap * sizeof(*pcount));
        if (!pkeys || !pcount) die("out of memory", "pattern table grow");
      }
      uint32_t idx = phash_intern(&h, key, pkeys, pcount, &n_pat);
      ++pcount[idx];
    }
    for (uint32_t w = 0; w < nw; ++w)
      checksum = (checksum ^ key[w]) * 1099511628211ULL;
  }

  /* Candidate set: every {0,1} pattern, homogeneous all-0/all-1 included.
   * Those two are the largest units on hg38 (all-1 ~10.7M CpGs, all-0 ~2.2M);
   * excluding them would strand ~44% of the genome in PNA. */
  uint32_t *order = xcalloc(n_pat, sizeof(uint32_t), "rank order");
  uint64_t n_cand = 0;
  for (uint64_t p = 0; p < n_pat; ++p) order[n_cand++] = (uint32_t)p;
  g_keys = pkeys; g_count = pcount; g_nw = nw;
  qsort(order, n_cand, sizeof(uint32_t), rank_cmp);

  /* rank_of[pattern index] = 0-based rank among candidates (or PNA). */
  uint32_t *rank_of = xcalloc(n_pat, sizeof(uint32_t), "rank_of");
  for (uint64_t p = 0; p < n_pat; ++p) rank_of[p] = MRMP_PNA_MEMBERSHIP;
  for (uint64_t r = 0; r < n_cand; ++r) rank_of[order[r]] = (uint32_t)r;

  /* Serialize MRMPIDX1. Sections are laid out in header order. */
  FILE *fp = fopen(out, "wb");
  if (!fp) die("cannot create output", out);
  mrmp_header_t hd; memset(&hd, 0, sizeof(hd));
  if (sizeof(hd) != 128) die("MRMPIDX1 header is not 128 bytes", NULL);
  memcpy(hd.magic, MRMPIDX_MAGIC, 8);
  hd.version = MRMPIDX_VERSION; hd.n_samples = ns;
  hd.n_selected = (uint32_t)n_cand;   /* the cut is the consumer's */
  hd.flags = MRMP_FLAG_INCLUDE_HOMOGENEOUS;   /* always: see the candidate loop */
  hd.n_cpg = n_cpg; hd.n_candidates = n_cand;
  hd.pna_key = pna_key[0]; hd.pna_cpg = pna_cpg;
  hd.mincov = mincov; hd.beta_threshold = beta_thr;
  hd.max_ambig_frac = max_ambig; hd.min_major_fold = min_fold;
  hd.content_checksum = checksum;

  uint64_t off = sizeof(hd);
  hd.refname_offset = off;   off += strlen(ref) + 1;
  hd.names_offset = off;     for (uint32_t s = 0; s < ns; ++s) off += strlen(names[s]) + 1;
  hd.patterns_offset = off;  off += n_cand * mrmp_pattern_stride(ns);
  hd.membership_offset = off;

  write_or_die(fp, &hd, sizeof(hd), out);
  write_or_die(fp, ref, strlen(ref) + 1, out);
  for (uint32_t s = 0; s < ns; ++s) write_or_die(fp, names[s], strlen(names[s]) + 1, out);
  for (uint64_t r = 0; r < n_cand; ++r) {   /* nw key words, then the count */
    uint32_t x = order[r];
    write_or_die(fp, pkeys + (uint64_t)x * nw, nw * sizeof(uint64_t), out);
    write_or_die(fp, &pcount[x], sizeof(uint64_t), out);
  }
  /* per-CpG membership rank (re-derive keys from the planes; PNA sentinel
   * where the CpG is the all-'2' pattern). */
  {
    const uint64_t CHUNK = 1u << 20;
    uint32_t *buf = xcalloc(CHUNK, sizeof(uint32_t), "membership buffer");
    for (uint64_t i = 0; i < n_cpg; ) {
      uint64_t m = n_cpg - i < CHUNK ? n_cpg - i : CHUNK;
      for (uint64_t j = 0; j < m; ++j) {
        int is_pna;
        resolve_cpg(meth, ambig, i + j, ns, stride, n_cpg,
                    min_fold, max_ambig, pna_key, &is_pna, key);
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
    "[methscope] mrmp-build: %s\n"
    "  samples=%u  CpGs=%" PRIu64 "  distinct patterns=%" PRIu64
    " (+PNA)  candidates=%" PRIu64 "\n"
    "  PNA CpGs=%" PRIu64 " (%.2f%%)  checksum=%016" PRIx64 "\n",
    out, ns, n_cpg, n_pat, n_cand, pna_cpg,
    100.0 * (double)pna_cpg / (double)n_cpg, checksum);

  free(pkeys); free(pcount); free(pna_key); free(key);
  free(order); free(rank_of);
  free(h.keys); free(h.slot);
  for (uint32_t s = 0; s < ns; ++s) free(names[s]);
  free(names);
  return 0;
}

/* ---------------- shared reader for inspect / export -------------------- */

typedef struct {
  void *map; size_t bytes; int fd;      /* the whole mapping, for munmap */
  const char *blk; uint64_t blk_bytes;  /* this MRMPIDX1 within the mapping */
  const mrmp_header_t *h;
  const char *refname;
  const char **names;           /* n_samples */
  const char *pat;              /* n_candidates records, `stride` apart */
  uint32_t stride;              /* mrmp_pattern_stride(n_samples) */
  uint32_t nw;                  /* mrmp_key_words(n_samples) */
  const uint32_t *membership;   /* n_cpg */
} mrmp_reader_t;

/* A pattern record is nw key words followed by the count, so it is addressed by
 * stride rather than indexed as a struct (which only fits the one-word case). */
static const uint64_t *pat_key(const mrmp_reader_t *r, uint64_t p) {
  return (const uint64_t *)(const void *)(r->pat + p * r->stride);
}
static uint64_t pat_count(const mrmp_reader_t *r, uint64_t p) {
  uint64_t v;
  memcpy(&v, r->pat + p * r->stride + (uint64_t)r->nw * sizeof(uint64_t), sizeof v);
  return v;
}

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

/* True if [offset, offset + count*size) fits within `bytes`, evaluated without
 * overflowing uint64 (offset and count come straight from the file header). */
static int region_ok(uint64_t offset, uint64_t count, uint64_t size,
                     uint64_t bytes) {
  if (offset > bytes) return 0;
  return size == 0 || count <= (bytes - offset) / size;
}

/* Open the MRMPIDX1 block that begins at `base`. base == 0 is a bare artifact;
 * a nonzero base addresses one block inside a MRMPSET1 container. Every offset
 * in a block header is relative to that block, so the only change from the
 * single-artifact case is which pointer the offsets are added to, and which
 * length they are bounds-checked against. `blk_bytes` == 0 means "to the end of
 * the file", which is what a bare artifact wants. */
static void mrmp_open_at(mrmp_reader_t *r, const char *path, uint64_t base,
                         uint64_t blk_bytes) {
  memset(r, 0, sizeof(*r));
  int fd = open(path, O_RDONLY);
  if (fd < 0) die("cannot open MRMP artifact", path);
  struct stat st;
  if (fstat(fd, &st) || (uint64_t)st.st_size < sizeof(mrmp_header_t))
    die("MRMP artifact is truncated", path);
  uint64_t fsz = (uint64_t)st.st_size;
  if (base > fsz || fsz - base < sizeof(mrmp_header_t))
    die("MRMP block starts past the end of the file", path);
  uint64_t sz = blk_bytes ? blk_bytes : fsz - base;
  if (sz > fsz - base) die("MRMP block extends past the end of the file", path);
  void *m = mmap(NULL, fsz, PROT_READ, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) die("cannot mmap MRMP artifact", path);
  const char *blk = (const char *)m + base;
  const mrmp_header_t *h = (const mrmp_header_t *)(const void *)blk;
  if (memcmp(h->magic, MRMPIDX_MAGIC, 8) || h->version != MRMPIDX_VERSION)
    die("bad MRMPIDX1 magic or version", path);
  /* Header counts are attacker-controlled; validate every region (overflow-safe)
   * before dereferencing, and require the pattern block to end at or before the
   * membership block, matching the writer's layout. Bounds are the BLOCK's, not
   * the file's, so a corrupt block cannot read into its neighbour. */
  if (h->n_cpg > UINT32_MAX)
    die("MRMP artifact CpG count is implausible", path);
  uint64_t pstride = mrmp_pattern_stride(h->n_samples);
  if (!region_ok(h->membership_offset, h->n_cpg, sizeof(uint32_t), sz) ||
      !region_ok(h->patterns_offset, h->n_candidates, pstride, sz) ||
      h->patterns_offset + h->n_candidates * pstride > h->membership_offset ||
      h->refname_offset >= sz || h->names_offset >= sz)
    die("MRMP artifact offsets are out of bounds", path);
  r->map = m; r->bytes = fsz; r->fd = fd; r->h = h;
  r->blk = blk; r->blk_bytes = sz;
  /* refname and each sample name must be NUL-terminated inside the block, or the
   * strlen walk below would run off the end of a truncated/crafted file. */
  const char *blk_end = blk + sz;
  r->refname = blk + h->refname_offset;
  if (!memchr(r->refname, '\0', (size_t)(blk_end - r->refname)))
    die("MRMP artifact refname is not terminated", path);
  r->names = xcalloc(h->n_samples, sizeof(char *), "names index");
  const char *p = blk + h->names_offset;
  for (uint32_t s = 0; s < h->n_samples; ++s) {
    if (p >= blk_end || !memchr(p, '\0', (size_t)(blk_end - p)))
      die("MRMP artifact sample names are truncated", path);
    ((const char **)r->names)[s] = p; p += strlen(p) + 1;
  }
  r->pat = blk + h->patterns_offset;
  r->stride = (uint32_t)pstride;
  r->nw = mrmp_key_words(h->n_samples);
  r->membership = (const uint32_t *)(const void *)(blk + h->membership_offset);
}

static void mrmp_open(mrmp_reader_t *r, const char *path) {
  mrmp_open_at(r, path, 0, 0);
}

static void mrmp_close(mrmp_reader_t *r) {
  free((void *)r->names);
  if (r->map) munmap(r->map, r->bytes);
  if (r->fd >= 0) close(r->fd);
}

/* ---------------- top-K decoded view ------------------------------------ */

/* Same decode `inspect --patterns` and `export --patterns` print, materialized
 * for callers that need the patterns as data rather than as text. Copies out of
 * the mapping so the artifact can be closed immediately. */
mrmp_top_t *ms_mrmp_top_read_at(const char *artifact, uint64_t base,
                                uint32_t top_k) {
  mrmp_reader_t r; mrmp_open_at(&r, artifact, base, 0);
  const mrmp_header_t *h = r.h;
  uint64_t lim = top_k < h->n_candidates ? top_k : h->n_candidates;
  if (lim > UINT32_MAX) die("top_k is implausible", artifact);

  mrmp_top_t *t = xcalloc(1, sizeof(*t), "mrmp top view");
  t->n_samples  = h->n_samples;
  t->n_patterns = (uint32_t)lim;
  t->labels     = xcalloc(h->n_samples, sizeof(char *), "top labels");
  t->binstring  = xcalloc(lim ? lim : 1, sizeof(char *), "top binstrings");
  t->count      = xcalloc(lim ? lim : 1, sizeof(uint64_t), "top counts");
  for (uint32_t s = 0; s < h->n_samples; ++s) {
    size_t n = strlen(r.names[s]) + 1;
    t->labels[s] = xcalloc(n, 1, "top label");
    memcpy(t->labels[s], r.names[s], n);
  }
  for (uint64_t p = 0; p < lim; ++p) {
    t->binstring[p] = xcalloc((size_t)h->n_samples + 1, 1, "top binstring");
    key_to_string(pat_key(&r, p), h->n_samples, t->binstring[p]);
    t->count[p] = pat_count(&r, p);
  }
  mrmp_close(&r);
  return t;
}

mrmp_top_t *ms_mrmp_top_read(const char *artifact, uint32_t top_k) {
  return ms_mrmp_top_read_at(artifact, 0, top_k);
}

void ms_mrmp_top_free(mrmp_top_t *t) {
  if (!t) return;
  for (uint32_t s = 0; s < t->n_samples; ++s) free(t->labels[s]);
  for (uint32_t p = 0; p < t->n_patterns; ++p) free(t->binstring[p]);
  free(t->labels); free(t->binstring); free(t->count); free(t);
}

/* ---------------- MRMPSET1 container ------------------------------------ */

int ms_mrmpset_is(const char *path) {
  char magic[8];
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  size_t got = fread(magic, 1, sizeof(magic), f);
  fclose(f);
  return got == sizeof(magic) && !memcmp(magic, MRMPSET_MAGIC, 8);
}

ms_mrmpset_t *ms_mrmpset_open(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open MRMP set", path);
  mrmpset_header_t h;
  if (fread(&h, 1, sizeof h, f) != sizeof h) die("MRMP set is truncated", path);
  if (memcmp(h.magic, MRMPSET_MAGIC, 8) || h.version != MRMPSET_VERSION)
    die("bad MRMPSET1 magic or version", path);
  if (!h.n_sets || h.n_sets > 4096) die("MRMP set count is implausible", path);
  if (fseek(f, 0, SEEK_END)) die("cannot size MRMP set", path);
  uint64_t fsz = (uint64_t)ftell(f);
  if (!region_ok(h.table_offset, h.n_sets, sizeof(mrmpset_entry_t), fsz) ||
      h.names_offset >= fsz)
    die("MRMP set offsets are out of bounds", path);

  ms_mrmpset_t *s = xcalloc(1, sizeof(*s), "mrmp set");
  s->n_sets      = h.n_sets;
  s->name        = xcalloc(h.n_sets, sizeof(char *), "set names");
  s->block_off   = xcalloc(h.n_sets, sizeof(uint64_t), "block offsets");
  s->block_bytes = xcalloc(h.n_sets, sizeof(uint64_t), "block sizes");

  mrmpset_entry_t *ent = xcalloc(h.n_sets, sizeof(*ent), "set table");
  if (fseek(f, (long)h.table_offset, SEEK_SET) ||
      fread(ent, sizeof(*ent), h.n_sets, f) != h.n_sets)
    die("cannot read the MRMP set table", path);
  for (uint32_t i = 0; i < h.n_sets; ++i) {
    /* every block must lie wholly inside the file, so a bad entry fails here
     * rather than when a reader dereferences into another set's bytes */
    if (ent[i].block_offset > fsz || ent[i].block_bytes > fsz - ent[i].block_offset ||
        ent[i].block_bytes < sizeof(mrmp_header_t))
      die("MRMP set block extends past the end of the file", path);
    s->block_off[i]   = ent[i].block_offset;
    s->block_bytes[i] = ent[i].block_bytes;
  }
  free(ent);

  /* names: n_sets NUL-terminated strings from names_offset to the first block */
  uint64_t nend = fsz;
  for (uint32_t i = 0; i < h.n_sets; ++i)
    if (s->block_off[i] > h.names_offset && s->block_off[i] < nend)
      nend = s->block_off[i];
  uint64_t nbytes = nend - h.names_offset;
  char *blob = xcalloc(nbytes + 1, 1, "set name blob");
  if (fseek(f, (long)h.names_offset, SEEK_SET) ||
      fread(blob, 1, nbytes, f) != nbytes)
    die("cannot read MRMP set names", path);
  blob[nbytes] = '\0';
  const char *p = blob;
  for (uint32_t i = 0; i < h.n_sets; ++i) {
    if ((uint64_t)(p - blob) >= nbytes) die("MRMP set names are truncated", path);
    size_t n = strlen(p) + 1;
    s->name[i] = xcalloc(n, 1, "set name");
    memcpy(s->name[i], p, n);
    p += n;
  }
  free(blob);
  fclose(f);
  return s;
}

void ms_mrmpset_free(ms_mrmpset_t *s) {
  if (!s) return;
  for (uint32_t i = 0; i < s->n_sets; ++i) free(s->name[i]);
  free(s->name); free(s->block_off); free(s->block_bytes); free(s);
}

void ms_mrmpset_write(const char *out, uint32_t n_sets, const char *const *name,
                      const void *const *block, const uint64_t *block_bytes) {
  if (!n_sets) die("a MRMP set needs at least one set", out);
  uint64_t names_bytes = 0;
  for (uint32_t i = 0; i < n_sets; ++i) names_bytes += strlen(name[i]) + 1;

  mrmpset_header_t h;
  memset(&h, 0, sizeof h);
  memcpy(h.magic, MRMPSET_MAGIC, 8);
  h.version      = MRMPSET_VERSION;
  h.n_sets       = n_sets;
  h.table_offset = sizeof h;
  h.names_offset = h.table_offset + (uint64_t)n_sets * sizeof(mrmpset_entry_t);

  /* Blocks are 8-byte aligned so a reader can cast a block header in place
   * rather than memcpy it out; the header is a multiple of 8 already. */
  mrmpset_entry_t *ent = xcalloc(n_sets, sizeof(*ent), "set table");
  uint64_t off = h.names_offset + names_bytes;
  for (uint32_t i = 0; i < n_sets; ++i) {
    off = (off + 7u) & ~7ull;
    ent[i].block_offset = off;
    ent[i].block_bytes  = block_bytes[i];
    off += block_bytes[i];
  }
  h.file_bytes = off;

  FILE *f = fopen(out, "wb");
  if (!f) die("cannot write MRMP set", out);
  write_or_die(f, &h, sizeof h, out);
  write_or_die(f, ent, (size_t)n_sets * sizeof(*ent), out);
  for (uint32_t i = 0; i < n_sets; ++i)
    write_or_die(f, name[i], strlen(name[i]) + 1, out);
  uint64_t at = h.names_offset + names_bytes;
  for (uint32_t i = 0; i < n_sets; ++i) {
    static const char pad[8] = {0};
    uint64_t want = ent[i].block_offset;
    if (want > at) { write_or_die(f, pad, (size_t)(want - at), out); at = want; }
    write_or_die(f, block[i], (size_t)block_bytes[i], out);
    at += block_bytes[i];
  }
  fclose(f);
  free(ent);
}

/* ---------------- inspect ----------------------------------------------- */

int main_mrmp_inspect(int argc, char *argv[]) {
  g_cmd = "inspect";
  const char *path = NULL; int show_patterns = 0; uint32_t top_k = 20;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      ms_help(stderr, "Usage: methscope inspect [options] IN.mrmp\n\n"
        "  IN.mrmp      MRMPIDX1 artifact to report on\n\n"
        "  --patterns   list the top-ranked patterns after the header\n"
        "  --top K      how many to list (default 20)\n");
      return 0;
    } else if (!strcmp(argv[i], "--patterns")) show_patterns = 1;
    else if (!strcmp(argv[i], "--top") && i + 1 < argc)
      top_k = (uint32_t)parse_u64(argv[++i], "--top");
    else if (argv[i][0] == '-') die("unrecognized option", argv[i]);
    else if (!path) path = argv[i];
    else die("too many arguments", argv[i]);
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
  printf("selectable_patterns\t%u\n", h->n_selected);
  printf("include_homogeneous\t%s\n",
         (h->flags & MRMP_FLAG_INCLUDE_HOMOGENEOUS) ? "yes" : "no");
  printf("pna_cpgs\t%" PRIu64 "\t%.4f%%\n", h->pna_cpg,
         100.0 * (double)h->pna_cpg / (double)h->n_cpg);
  printf("binstring\tmincov=%u beta=%.3f max_ambig_frac=%.3f min_major_fold=%.3f\n",
         h->mincov, h->beta_threshold, h->max_ambig_frac, h->min_major_fold);
  printf("content_checksum\t%016" PRIx64 "\n", h->content_checksum);
  char *buf = xcalloc(h->n_samples + 1, 1, "string buffer");
  /* The header stores only word 0 of the sentinel, so rebuild the full key --
   * it is all-'2' and therefore fully determined by n_samples -- and check the
   * stored word agrees, which catches a header/sample-count mismatch. */
  uint32_t pna_nw = mrmp_key_words(h->n_samples);
  uint64_t *pna = xcalloc(pna_nw, sizeof(uint64_t), "pna key");
  for (uint32_t s = 0; s < h->n_samples; ++s)
    pna[s / MRMP_TRITS_PER_WORD] = pna[s / MRMP_TRITS_PER_WORD] * 3 + 2;
  if (pna[0] != h->pna_key) die("PNA key disagrees with n_samples", path);
  key_to_string(pna, h->n_samples, buf);
  free(pna);
  printf("pna_pattern\t%s\n", buf);
  if (show_patterns) {
    uint64_t lim = top_k < h->n_candidates ? top_k : h->n_candidates;
    printf("#pattern\tlabel\tcount\t(top %" PRIu64 " of %" PRIu64 ")\n",
           lim, h->n_candidates);
    for (uint64_t p = 0; p < lim; ++p) {
      key_to_string(pat_key(&r, p), h->n_samples, buf);
      printf("%s\tP%" PRIu64 "\t%" PRIu64 "\n", buf, p + 1, pat_count(&r, p));
    }
  }
  free(buf);
  mrmp_close(&r);
  return 0;
}

/* ---------------- export ------------------------------------------------ */

int main_mrmp_export(int argc, char *argv[]) {
  g_cmd = "mrmp-export";
  const char *pos[2] = {NULL, NULL}, *patterns = NULL, *counts = NULL,
             *pna_label = "Pna";
  int npos = 0;
  uint32_t top_k = 1000;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stderr,
        "Usage: methscope mrmp-export [options] IN.mrmp OUT.cm\n\n"
        "  IN.mrmp          MRMPIDX1 artifact\n"
        "  OUT.cm           per-CpG P1..PK/Pna labels as a YAME format-2 mask\n\n"
        "  --top K          rank cut for the mask and --patterns (default 1000)\n"
        "  --patterns TSV   also write top-K patterns: string<tab>P<rank><tab>count\n"
        "  --counts TSV     also write every pattern (incl. PNA): count<tab>string\n"
        "  --pna-label NAME background label in the mask (default Pna)\n");
      return 0;
    } else if (!strcmp(a, "--patterns") && i + 1 < argc) patterns = argv[++i];
    else if (!strcmp(a, "--counts") && i + 1 < argc) counts = argv[++i];
    else if (!strcmp(a, "--top") && i + 1 < argc) top_k = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--pna-label") && i + 1 < argc) pna_label = argv[++i];
    else if (a[0] == '-') die("unrecognized option", a);
    else if (npos < 2) pos[npos++] = a;
    else die("too many arguments", a);
  }
  if (npos != 2) die("need IN.mrmp and OUT.cm (see mrmp-export -h)", NULL);
  const char *path = pos[0], *mask = pos[1];
  mrmp_reader_t r; mrmp_open(&r, path);
  const mrmp_header_t *h = r.h;
  const uint32_t ns = h->n_samples;
  char *buf = xcalloc(ns + 1, 1, "string buffer");

  if (patterns) {
    FILE *f = fopen(patterns, "w");
    if (!f) die("cannot create --patterns", patterns);
    uint64_t lim = top_k < h->n_candidates ? top_k : h->n_candidates;
    for (uint64_t p = 0; p < lim; ++p) {
      key_to_string(pat_key(&r, p), ns, buf);
      fprintf(f, "%s\tP%" PRIu64 "\t%" PRIu64 "\n", buf, p + 1, pat_count(&r, p));
    }
    if (fclose(f)) die("error closing --patterns", patterns);
  }

  if (counts) {
    /* Every candidate in rank order, then the PNA sentinel. */
    FILE *f = fopen(counts, "w");
    if (!f) die("cannot create --counts", counts);
    for (uint64_t p = 0; p < h->n_candidates; ++p) {
      key_to_string(pat_key(&r, p), ns, buf);
      fprintf(f, "%" PRIu64 "\t%s\n", pat_count(&r, p), buf);
    }
    /* PNA is all-'2', so its key follows from ns; the header keeps only word 0. */
    uint32_t pna_nw = mrmp_key_words(ns);
    uint64_t *pna = xcalloc(pna_nw, sizeof(uint64_t), "pna key");
    for (uint32_t s2 = 0; s2 < ns; ++s2)
      pna[s2 / MRMP_TRITS_PER_WORD] = pna[s2 / MRMP_TRITS_PER_WORD] * 3 + 2;
    key_to_string(pna, ns, buf);
    free(pna);
    fprintf(f, "%" PRIu64 "\t%s\n", h->pna_cpg, buf);
    if (fclose(f)) die("error closing --counts", counts);
  }

  if (mask) ms_mrmp_write_mask(path, mask, pna_label, top_k);

  free(buf);
  mrmp_close(&r);
  return 0;
}

/* ---------------- artifact -> runtime forms ------------------------------ */

int ms_mrmp_is_artifact(const char *path) {
  char magic[8];
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open MRMP", path);
  size_t got = fread(magic, 1, sizeof(magic), f);
  fclose(f);
  return got == sizeof(magic) && !memcmp(magic, MRMPIDX_MAGIC, 8);
}

void ms_mrmp_group_map_at(const char *artifact, uint64_t base, uint16_t *group,
                          uint64_t n_cpg, uint32_t patterns) {
  mrmp_reader_t r; mrmp_open_at(&r, artifact, base, 0);
  if (r.h->n_cpg != n_cpg) die("MRMP artifact CpG count disagrees", artifact);
  uint32_t K = r.h->n_selected < patterns ? r.h->n_selected : patterns;
  /* group[] is uint16 (1-based rank, 0 = PNA), so a selectable rank must fit in
   * 15 usable bits; guard against a caller/artifact that would alias rank+1. */
  if (K > UINT16_MAX - 1) die("MRMP selectable pattern count exceeds uint16", artifact);
  for (uint64_t i = 0; i < n_cpg; ++i) {
    uint32_t rank = r.membership[i];
    group[i] = (rank != MRMP_PNA_MEMBERSHIP && rank < K)
             ? (uint16_t)(rank + 1) : 0;
  }
  mrmp_close(&r);
}

void ms_mrmp_group_map(const char *artifact, uint16_t *group, uint64_t n_cpg,
                       uint32_t patterns) {
  ms_mrmp_group_map_at(artifact, 0, group, n_cpg, patterns);
}

void ms_mrmp_write_mask(const char *artifact, const char *out_cm,
                        const char *pna_label, uint32_t top_k) {
  mrmp_reader_t r; mrmp_open(&r, artifact);
  const mrmp_header_t *h = r.h;
  if (!pna_label) pna_label = "Pna";
  if (!top_k) die("mask needs a positive rank cut", out_cm);
  if (top_k > h->n_selected) top_k = h->n_selected;
  {
    /* Build a raw YAME format-2 cdata directly (no genome-sized text file),
     * mirroring fmt2_read_raw: first-seen key order over genomic CpGs, then
     * cdata_compress (RLE) + cdata_write. Labels: P(rank+1) or the PNA label. */
    const uint64_t n = h->n_cpg, K = top_k;
    const char *mask = out_cm;
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
    fprintf(stderr, "[methscope] mrmp: wrote mask %s (%" PRIu64 " labels)\n",
            mask, n_keys);
  }
  mrmp_close(&r);
}


/* ---------------- mrmp-pack: combine artifacts into a container ---------- */

/* Exists because the sets in a container are ordinary MRMPIDX1 artifacts, so
 * combining ones already on disk is just concatenation plus a table. That makes
 * the container testable on its own, and lets a pipeline that already built its
 * sets separately adopt the fused featurizer without rebuilding them. */
int main_mrmp_pack(int argc, char *argv[]) {
  g_cmd = "mrmp-pack";
  const char *out = NULL;
  int i = 1;
  for (; i < argc; ++i) {
    if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      fprintf(stderr,
        "Usage: methscope mrmp-pack -o OUT.mrmpset NAME:IN.mrmp [NAME:IN.mrmp ...]\n\n"
        "  Combine MRMPIDX1 artifacts into one MRMPSET1 container, in the order\n"
        "  given. Set 0 is conventionally the global set and the rest satellites;\n"
        "  the featurizer pools patterns across all of them.\n");
      return 0;
    }
    else if (argv[i][0] == '-') die("unrecognized option", argv[i]);
    else break;
  }
  if (!out || argc - i < 1) { die("need -o OUT and at least one NAME:IN.mrmp", NULL); }

  uint32_t n = (uint32_t)(argc - i);
  char **name = xcalloc(n, sizeof(char *), "set names");
  void **blk  = xcalloc(n, sizeof(void *), "blocks");
  uint64_t *len = xcalloc(n, sizeof(uint64_t), "block sizes");
  for (uint32_t k = 0; k < n; ++k) {
    char *spec = argv[i + k];
    char *colon = strchr(spec, ':');
    if (!colon) die("expected NAME:IN.mrmp", spec);
    *colon = '\0';
    name[k] = spec;
    const char *path = colon + 1;
    if (!ms_mrmp_is_artifact(path)) die("not a MRMPIDX1 artifact", path);
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open", path);
    if (fseek(f, 0, SEEK_END)) die("cannot size", path);
    len[k] = (uint64_t)ftell(f);
    rewind(f);
    blk[k] = xcalloc(len[k], 1, "block");
    if (fread(blk[k], 1, len[k], f) != len[k]) die("short read", path);
    fclose(f);
  }
  ms_mrmpset_write(out, n, (const char *const *)name,
                   (const void *const *)blk, len);
  fprintf(stderr, "[methscope] mrmp-pack: %u sets -> %s\n", n, out);
  for (uint32_t k = 0; k < n; ++k) free(blk[k]);
  free(name); free(blk); free(len);
  return 0;
}

/* ---------------- inspect: the MRMPSET1 arm ------------------------------ */

/* Per-set dimensions, then the POOLED view -- which sets would actually
 * contribute at a given rank cut. That second table is the one that matters in
 * practice: a satellite holding 2-30 patterns is unaffected by a per-set
 * `--top 1000`, so the only cut that means anything is the pooled one, and this
 * is where you see whether a set earns its place or is crowded out. */
int main_mrmpset_inspect(const char *path) {
  g_cmd = "inspect";
  ms_mrmpset_t *s = ms_mrmpset_open(path);
  printf("MRMPSET1 container: %s\n", path);
  printf("  sets: %u\n\n", s->n_sets);

  /* gather every pattern from every set for the pooled table */
  uint64_t total = 0;
  mrmp_top_t **top = xcalloc(s->n_sets, sizeof(*top), "per-set tops");
  for (uint32_t i = 0; i < s->n_sets; ++i) {
    top[i] = ms_mrmp_top_read_at(path, s->block_off[i], UINT32_MAX);
    total += top[i]->n_patterns;
  }

  printf("  %-12s %7s %9s %12s  %s\n", "set", "classes", "patterns", "CpGs", "members");
  for (uint32_t i = 0; i < s->n_sets; ++i) {
    mrmp_top_t *t = top[i];
    uint64_t cpg = 0;
    for (uint32_t p = 0; p < t->n_patterns; ++p) cpg += t->count[p];
    printf("  %-12s %7u %9u %12" PRIu64 "  ", s->name[i], t->n_samples,
           t->n_patterns, cpg);
    /* naming every class is the point for a satellite; for a big global set it
     * would be noise, so elide past a handful */
    if (t->n_samples <= 8) {
      for (uint32_t k = 0; k < t->n_samples; ++k)
        printf("%s%s", k ? ", " : "", t->labels[k]);
    } else {
      printf("%s, %s, ... (%u more)", t->labels[0], t->labels[1],
             t->n_samples - 2);
    }
    putchar('\n');
  }

  /* pooled rank by CpG count, the same order the featurizer selects in */
  typedef struct { uint64_t cnt; uint32_t set; } pc_t;
  pc_t *all = xcalloc(total ? total : 1, sizeof(*all), "pooled");
  uint64_t n = 0;
  for (uint32_t i = 0; i < s->n_sets; ++i)
    for (uint32_t p = 0; p < top[i]->n_patterns; ++p) {
      all[n].cnt = top[i]->count[p]; all[n].set = i; ++n;
    }
  for (uint64_t a = 1; a < n; ++a) {            /* insertion sort by -cnt */
    pc_t v = all[a]; uint64_t b = a;
    while (b && all[b-1].cnt < v.cnt) { all[b] = all[b-1]; --b; }
    all[b] = v;
  }
  static const uint64_t CUTS[] = {100, 500, 1000, 2000};
  printf("\n  pooled selection by CpG count (what --top actually keeps)\n");
  printf("  %-8s %10s  %s\n", "--top", "CpG floor", "columns per set");
  uint32_t *per = xcalloc(s->n_sets, sizeof(uint32_t), "per-set counts");
  for (unsigned c = 0; c < sizeof(CUTS)/sizeof(*CUTS); ++c) {
    uint64_t k = CUTS[c] < n ? CUTS[c] : n;
    if (!k) continue;
    memset(per, 0, s->n_sets * sizeof(uint32_t));
    for (uint64_t a = 0; a < k; ++a) per[all[a].set]++;
    printf("  %-8" PRIu64 " %10" PRIu64 "  ", CUTS[c], all[k-1].cnt);
    for (uint32_t i = 0; i < s->n_sets; ++i)
      printf("%s%s=%u", i ? " " : "", s->name[i], per[i]);
    putchar('\n');
  }
  printf("\n  %" PRIu64 " patterns available across all sets\n", n);

  free(per); free(all);
  for (uint32_t i = 0; i < s->n_sets; ++i) ms_mrmp_top_free(top[i]);
  free(top);
  ms_mrmpset_free(s);
  return 0;
}
