// SPDX-License-Identifier: AGPL-3.0-or-later
/* Native MRMP construction: `methscope mrmp-build` / `mrmp-export`, plus the
 * MRMPIDX1 arm of `methscope inspect`.
 * See mrmp.h for the artifact format and the binstring semantics reproduced. */
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "methscope.h"
#include "mrmp.h"
#include "mrmp_select.h"
#include "msfm.h"
#include "cfile.h"
#include "cdata.h"
#include "index.h"
#include <zlib.h>

/* BGZF blocks are framed here, on zlib, rather than through htslib's
 * bgzf_compress(): libyame.a carries its own bgzf.o which shadows htslib's, so
 * referencing the htslib symbol pulls in a second definition and the link fails
 * on `multiple definition of bgzf_close`. YAME's own bgzf exports no
 * compress-to-buffer entry point. See the note in YAME/tmp. */

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

/* Names in FILE order, and optionally each record's BGZF virtual offset (the
 * .idx second column). Parsed here rather than through YAME's loadIndex()
 * because that returns a khash whose iteration order is arbitrary, and file
 * order IS the binstring's digit order -- a permuted walk would silently
 * relabel every pattern. */
static char **read_store_index(const char *ref, uint32_t *n_out,
                               int64_t **off_out) {
  char idx[PATH_MAX];
  if (snprintf(idx, sizeof(idx), "%s.idx", ref) >= (int)sizeof(idx))
    die("reference path too long", ref);
  FILE *f = fopen(idx, "r");
  if (!f) die("cannot open reference index (expected <ref>.idx)", idx);
  size_t cap = 64, n = 0;
  char **names = xcalloc(cap, sizeof(char *), "sample names");
  int64_t *off = xcalloc(cap, sizeof(int64_t), "sample offsets");
  char *line = NULL; size_t lcap = 0; ssize_t len;
  while ((len = getline(&line, &lcap, f)) > 0) {
    char *tab = strpbrk(line, "\t\n");
    size_t nl = tab ? (size_t)(tab - line) : (size_t)len;
    if (!nl) continue;
    if (n == cap) {
      cap <<= 1;
      names = realloc(names, cap * sizeof(char *));
      off   = realloc(off, cap * sizeof(int64_t));
      if (!names || !off) die("out of memory", "sample index grow");
    }
    names[n] = xcalloc(nl + 1, 1, "sample name");
    memcpy(names[n], line, nl);
    off[n] = (tab && *tab == '\t') ? (int64_t)strtoll(tab + 1, NULL, 10) : -1;
    ++n;
  }
  free(line); fclose(f);
  if (!n) die("reference index is empty", idx);
  /* No sample cap: the key widens to ceil(n/40) words (see mrmp.h). */
  *n_out = (uint32_t)n;
  if (off_out) *off_out = off; else free(off);
  return names;
}

static char **read_sample_names(const char *ref, uint32_t *n_out) {
  return read_store_index(ref, n_out, NULL);
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

/* Membership RLE, defined with the reader further down since inflating is where
 * the format is interpreted; declared here because mrmp-build deflates. */
/* ---- BGZF framing for the membership payload ----------------------------
 *
 * htslib exports bgzf_compress() for ONE block and no in-memory inflate, so the
 * blocks are emitted in a loop here and walked on the way back -- the same thing
 * htslib does internally, and the reason the reader parses BSIZE rather than
 * handing the buffer to zlib as a multi-member gzip stream. */

static uint8_t *bgzf_deflate_buf(const uint8_t *src, uint64_t slen,
                                 uint64_t *out_n) {
  uint64_t cap = slen + slen / 8 + 4096, n = 0;
  uint8_t *out = xcalloc(cap, 1, "bgzf buffer");
  uint8_t *tmp = xcalloc(BGZF_MAX_BLOCK_SIZE, 1, "bgzf block");
  uint64_t at = 0;
  do {                                   /* do/while so slen == 0 still frames */
    uint32_t want = slen - at < BGZF_BLOCK_SIZE ? (uint32_t)(slen - at)
                                                : (uint32_t)BGZF_BLOCK_SIZE;
    z_stream zs; memset(&zs, 0, sizeof zs);
    if (deflateInit2(&zs, 6, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK)
      die("deflateInit2 failed", "membership");
    zs.next_in = (Bytef *)(uintptr_t)(src + at); zs.avail_in = want;
    zs.next_out = tmp; zs.avail_out = BGZF_MAX_BLOCK_SIZE;
    if (deflate(&zs, Z_FINISH) != Z_STREAM_END) die("deflate failed", "membership");
    uint32_t clen = (uint32_t)zs.total_out;
    deflateEnd(&zs);

    uint32_t bsize = 18 + clen + 8;      /* header + payload + CRC32 + ISIZE */
    if (n + bsize > cap) { cap = (n + bsize) * 2; out = realloc(out, cap);
                           if (!out) die("out of memory", "bgzf buffer"); }
    uint8_t *b = out + n;
    static const uint8_t hdr[12] = {31,139,8,4,0,0,0,0,0,255,6,0};
    memcpy(b, hdr, 12);
    b[12] = 'B'; b[13] = 'C'; b[14] = 2; b[15] = 0;
    b[16] = (uint8_t)((bsize - 1) & 0xff);
    b[17] = (uint8_t)(((bsize - 1) >> 8) & 0xff);
    memcpy(b + 18, tmp, clen);
    uint32_t crc = (uint32_t)crc32(crc32(0L, NULL, 0), src + at, want);
    for (int k = 0; k < 4; ++k) b[18 + clen + k] = (uint8_t)(crc >> (8 * k));
    for (int k = 0; k < 4; ++k) b[22 + clen + k] = (uint8_t)(want >> (8 * k));
    n += bsize; at += want;
  } while (at < slen);
  free(tmp);
  *out_n = n;
  return out;
}

/* Walk the framed blocks, inflating each with raw deflate. A BGZF block is a
 * gzip member whose BC extra subfield carries BSIZE-1 at byte 16, so the block
 * extent is known before inflating and the payload is [18, bsize-8). */
static void bgzf_inflate_buf(const uint8_t *src, uint64_t slen,
                             uint8_t *dst, uint64_t dlen, const char *what) {
  uint64_t at = 0, out = 0;
  while (at < slen) {
    if (slen - at < 18) die("truncated BGZF block header", what);
    if (src[at] != 31 || src[at + 1] != 139) die("not a BGZF block", what);
    uint32_t bsize = (uint32_t)src[at + 16] | ((uint32_t)src[at + 17] << 8);
    ++bsize;
    if (bsize < 26 || bsize > slen - at) die("bad BGZF block size", what);
    z_stream zs; memset(&zs, 0, sizeof zs);
    zs.next_in = (Bytef *)(uintptr_t)(src + at + 18);
    zs.avail_in = bsize - 18 - 8;
    zs.next_out = dst + out;
    zs.avail_out = (uInt)(dlen - out);
    if (inflateInit2(&zs, -15) != Z_OK) die("inflateInit2 failed", what);
    int rc = inflate(&zs, Z_FINISH);
    uint64_t got = zs.total_out;
    inflateEnd(&zs);
    if (rc != Z_STREAM_END) die("BGZF block did not inflate", what);
    out += got; at += bsize;
  }
  if (out != dlen) die("membership inflated to the wrong size", what);
}

static uint8_t *memb_compress(const uint32_t *memb, uint64_t n_cpg,
                              uint64_t n_cand, uint64_t *out_n);

int main_mrmp_build(int argc, char *argv[]) {
  g_cmd = "mrmp-build";
  const char *pos[2] = {NULL, NULL}, *set_name = NULL;
  int npos = 0;
  uint32_t mincov = MRMP_DEF_MINCOV;
  float beta_thr = MRMP_DEF_BETA_THRESH, max_ambig = MRMP_DEF_MAX_AMBIG,
        min_fold = MRMP_DEF_MIN_FOLD;
  int force = 0;
  /* Per-CpG selection, evaluated inline as each CpG is resolved. Every one of
   * these is a property of the CpG alone, so none needs the binstring to be
   * known first -- which is why they can run in the same pass and a CpG that
   * fails never enters the pattern hash at all. Counts, and therefore ranks,
   * are then post-filter by construction. (A per-BINSTRING rank such as
   * --delta-mean-top cannot work this way; that lives in the satellite
   * builders, which group by pattern first.) */
  float qf_lo = -1.0f, qf_hi = -1.0f;   /* <0 == q-filter off */
  float max_frac_na = 0.0f;
  uint32_t min_cg_depth = 0;
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
        "  --force            overwrite an existing output\n"
        "\n"
        " Per-CpG selection (off unless given). Applied as each CpG is\n"
        " resolved, so a CpG that fails never enters the pattern table and\n"
        " the pattern counts -- hence the ranking, hence --top downstream --\n"
        " describe the CpGs that actually survived.\n"
        "  --qfilter LO,HI    keep a CpG only if every MEASURED class calling\n"
        "                     0 has beta <= LO and every one calling 1 has\n"
        "                     beta >= HI. Both sides are required: bounding\n"
        "                     only the low side leaves the expected-1 classes\n"
        "                     free to sit near 0.5, where a cell of such a\n"
        "                     class reads ambiguously and falls to the wrong\n"
        "                     side.\n"
        "                     EXACT betas, not quantiles. The shell pipeline\n"
        "                     this replaces read yame `rowop -o stat`, whose\n"
        "                     q95_0/q05_1 come off a 16-BIN histogram -- only\n"
        "                     8 distinct values on a k/16 grid. Its tuned\n"
        "                     0.25,0.6 therefore enforced beta >= 0.625 on the\n"
        "                     expected-1 side, not 0.60. So 0.25,0.625 is the\n"
        "                     honest translation of what shipped; 0.25,0.6\n"
        "                     here is genuinely more permissive than it was\n"
        "                     there. Re-tune against held-out margin rather\n"
        "                     than inheriting a number fitted to a binning\n"
        "                     artifact of another tool.\n"
        "  --max-frac-na F    fraction of classes allowed to be UNCOVERED at a\n"
        "                     CpG (default 0 = none). An uncovered class is\n"
        "                     imputed the majority digit by the binstring\n"
        "                     rule, so it is not a measurement and the\n"
        "                     q-filter never tests it; this bounds how much\n"
        "                     imputation is tolerated. Note --min-major-fold\n"
        "                     already discards most such CpGs as PNA.\n"
        "  --min-cg-depth N   minimum coverage required of EVERY measured\n"
        "                     class. Absolute, which is safe only when thin\n"
        "                     classes are excluded from this set; satellites\n"
        "                     need a floor relative to each class's own mean\n");
      return 0;
    }
    else if (!strcmp(a, "--mincov") && i + 1 < argc) mincov = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--beta-threshold") && i + 1 < argc) beta_thr = (float)atof(argv[++i]);
    else if (!strcmp(a, "--max-ambig-frac") && i + 1 < argc) max_ambig = (float)atof(argv[++i]);
    else if (!strcmp(a, "--min-major-fold") && i + 1 < argc) min_fold = (float)atof(argv[++i]);
    else if (!strcmp(a, "--qfilter") && i + 1 < argc) {
      const char *v = argv[++i]; char *end = NULL;
      qf_lo = strtof(v, &end);
      if (!end || *end != ',') die("--qfilter wants LO,HI", v);
      qf_hi = strtof(end + 1, NULL);
      if (!(qf_lo >= 0.0f && qf_hi <= 1.0f && qf_lo < qf_hi))
        die("--qfilter needs 0 <= LO < HI <= 1", v);
    }
    else if (!strcmp(a, "--max-frac-na") && i + 1 < argc) max_frac_na = (float)atof(argv[++i]);
    else if (!strcmp(a, "--min-cg-depth") && i + 1 < argc) min_cg_depth = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--name") && i + 1 < argc) set_name = argv[++i];
    else if (!strcmp(a, "--force")) force = 1;
    else if (a[0] == '-') die("unrecognized or incomplete option", a);
    else if (npos < 2) pos[npos++] = a;
    else die("too many arguments", a);
  }
  if (npos != 2) die("need REF.cg and OUT.mrmp (see mrmp-build -h)", NULL);
  const char *ref = pos[0], *out = pos[1];
  if (!force) { struct stat st; if (!stat(out, &st)) die("output exists (use --force)", out); }

  /* The set's name travels INSIDE the block, so it survives being cat'd into a
   * chain. Defaulting to the output's basename means the common case needs no
   * flag and still produces a readable pooled table. */
  char *name_buf = NULL;
  if (!set_name) {
    const char *b = strrchr(out, '/'); b = b ? b + 1 : out;
    size_t n = strlen(b);
    const char *dot = strrchr(b, '.');
    if (dot && dot != b) n = (size_t)(dot - b);
    name_buf = xcalloc(n + 1, 1, "set name");
    memcpy(name_buf, b, n);
    set_name = name_buf;
  }

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
  /* Selection state, accumulated from MEASURED classes only. A class with no
   * coverage is imputed the majority digit by resolve_cpg, so its binstring
   * bit is not a measurement and must not be tested. */
  const int sel_on = (qf_lo >= 0.0f) || min_cg_depth || (max_frac_na > 0.0f);
  float *sel_max0 = NULL, *sel_min1 = NULL;
  uint16_t *sel_cov = NULL;
  uint8_t *sel_pres = NULL;
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
      if (sel_on) {
        sel_max0 = xcalloc(n_cpg, sizeof(float), "sel max0");
        sel_min1 = xcalloc(n_cpg, sizeof(float), "sel min1");
        sel_cov  = xcalloc(n_cpg, sizeof(uint16_t), "sel min coverage");
        sel_pres = xcalloc(n_cpg, 1, "sel n present");
        for (uint64_t i = 0; i < n_cpg; ++i) {
          sel_max0[i] = -1.0f; sel_min1[i] = 2.0f; sel_cov[i] = 0xFFFF;
        }
      }
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
        if (sel_on && beta != beta_thr) {          /* a tie is not a call */
          uint64_t cov = MU2cov(mu);
          ++sel_pres[i];
          if (cov < sel_cov[i]) sel_cov[i] = (uint16_t)(cov > 0xFFFF ? 0xFFFF : cov);
          if (beta > beta_thr) { if ((float)beta < sel_min1[i]) sel_min1[i] = (float)beta; }
          else                 { if ((float)beta > sel_max0[i]) sel_max0[i] = (float)beta; }
        }
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

  /* keep[i]: this CpG survived selection and was interned. Recorded rather
   * than recomputed, so the membership pass below cannot disagree with what
   * was actually counted -- a mismatch there would look up a pattern that was
   * never interned. */
  uint8_t *keep = sel_on ? xcalloc(n_cpg, 1, "selection keep") : NULL;
  const uint32_t na_allow = (uint32_t)(max_frac_na * (float)ns);
  uint64_t n_filtered = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) {
    int is_pna;
    resolve_cpg(meth, ambig, i, ns, stride, n_cpg,
                min_fold, max_ambig, pna_key, &is_pna, key);
    if (!is_pna && sel_on) {
      uint32_t nmeas = sel_pres[i];
      int ok = ((uint32_t)(ns - nmeas) <= na_allow)
            && (!min_cg_depth || (nmeas && sel_cov[i] >= min_cg_depth));
      /* both sides must be populated: a CpG with no expected-0 or no
       * expected-1 measured class carries no contrast to threshold */
      if (ok && qf_lo >= 0.0f)
        ok = (sel_max0[i] >= 0.0f) && (sel_min1[i] <= 1.0f)
          && (sel_max0[i] <= qf_lo) && (sel_min1[i] >= qf_hi);
      if (!ok) { is_pna = 1; ++n_filtered; }
      else keep[i] = 1;
    }
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

  /* per-CpG membership rank (re-derive keys from the planes; PNA sentinel
   * where the CpG is the all-'2' pattern). Materialised rather than streamed,
   * because the threshold pass below needs the same mapping and re-deriving it
   * twice would double the resolve_cpg work.
   *
   * Built BEFORE the header is laid out, because the section is compressed and
   * the layout cannot place what follows it until its size is known. */
  uint32_t *memb = xcalloc(n_cpg, sizeof(uint32_t), "membership");
  for (uint64_t i = 0; i < n_cpg; ++i) {
    if (keep && !keep[i]) { memb[i] = MRMP_PNA_MEMBERSHIP; continue; }
    int is_pna;
    resolve_cpg(meth, ambig, i, ns, stride, n_cpg,
                min_fold, max_ambig, pna_key, &is_pna, key);
    uint32_t p = is_pna ? MRMP_PNA_MEMBERSHIP : phash_find(&h, key);
    memb[i] = (p == MRMP_PNA_MEMBERSHIP) ? MRMP_PNA_MEMBERSHIP : rank_of[p];
  }
  uint64_t memb_n = 0;
  uint8_t *memb_rle = memb_compress(memb, n_cpg, n_cand, &memb_n);
  if (memb_n > UINT32_MAX) die("compressed membership exceeds 4 GB", out);
  fprintf(stderr, "  membership: %" PRIu64 " -> %" PRIu64 " bytes (%.0fx)\n",
          n_cpg * sizeof(uint32_t), memb_n,
          (double)(n_cpg * sizeof(uint32_t)) / (double)(memb_n ? memb_n : 1));

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
  hd.membership_bytes = (uint32_t)memb_n;

  uint64_t off = sizeof(hd);
  hd.refname_offset = off;   off += strlen(ref) + 1;
  hd.name_offset = (uint32_t)off; off += strlen(set_name) + 1;
  hd.names_offset = off;     for (uint32_t s = 0; s < ns; ++s) off += strlen(names[s]) + 1;
  hd.patterns_offset = off;  off += n_cand * mrmp_pattern_stride(ns);
  hd.membership_offset = off; off += memb_n;
  hd.thresh_offset = off;    off += (uint64_t)n_cand * sizeof(float);
  hd.flags |= MRMP_FLAG_THRESH | MRMP_FLAG_MEMB_RLE | MRMP_FLAG_MEMB_BGZF;
  /* Pad to 8 so this file can be cat'd in front of another one and leave its
   * header aligned. Every MRMPIDX1 is self-padding for exactly this reason. */
  const uint64_t pad_bytes = (((off + 7u) & ~7ull) - off);

  write_or_die(fp, &hd, sizeof(hd), out);
  write_or_die(fp, ref, strlen(ref) + 1, out);
  write_or_die(fp, set_name, strlen(set_name) + 1, out);
  for (uint32_t s = 0; s < ns; ++s) write_or_die(fp, names[s], strlen(names[s]) + 1, out);
  for (uint64_t r = 0; r < n_cand; ++r) {   /* nw key words, then the count */
    uint32_t x = order[r];
    write_or_die(fp, pkeys + (uint64_t)x * nw, nw * sizeof(uint64_t), out);
    write_or_die(fp, &pcount[x], sizeof(uint64_t), out);
  }
  write_or_die(fp, memb_rle, (size_t)memb_n, out);
  free(memb_rle);

  /* ---- per-pattern binarisation midpoints --------------------------------
   *
   * A second pass over the reference, because the ingestion above kept only the
   * meth/ambig bit planes and the midpoint needs the actual betas. For each
   * pattern, average the reference beta over its CpGs separately for the
   * classes it calls 1 and those it calls 0, then take the midpoint. That is
   * the cut which best separates the two groups AS THE REFERENCE SEES THEM --
   * and it must not be a fixed 0.5, because inside a satellite the members are
   * close relatives and a pattern can sit entirely above 0.5, where 0.5
   * carries no information at all.
   *
   * NaN marks a pattern that cannot be binarised: a homogeneous binstring (no
   * contrast to threshold), or a reference that puts the expected-0 group at or
   * above the expected-1 group (the pattern does not hold up on its own
   * reference). Consumers drop those rather than guess. */
  {
    double *sum = xcalloc(n_cand * ns, sizeof(double), "threshold sums");
    uint64_t *cnt = xcalloc(n_cand * ns, sizeof(uint64_t), "threshold counts");
    cfile_t cf2 = open_cfile((char *)ref);
    for (uint32_t kk = 0; kk < ns; ++kk) {
      cdata_t c = read_cdata1(&cf2);
      if (!c.n) { free_cdata(&c); break; }
      decompress_in_situ(&c);
      for (uint64_t i = 0; i < n_cpg; ++i) {
        uint32_t r = memb[i];
        if (r == MRMP_PNA_MEMBERSHIP) continue;
        uint64_t mu = f3_get_mu(&c, i);
        if (!mu || MU2cov(mu) < mincov) continue;
        sum[(uint64_t)r * ns + kk] += MU2beta(mu);
        cnt[(uint64_t)r * ns + kk] += 1;
      }
      free_cdata(&c);
    }
    bgzf_close(cf2.fh);

    float *thr = xcalloc(n_cand, sizeof(float), "thresholds");
    uint64_t n_usable = 0;
    char *bs = xcalloc((size_t)ns + 1, 1, "binstring");
    for (uint64_t r = 0; r < n_cand; ++r) {
      key_to_string(pkeys + (uint64_t)order[r] * nw, ns, bs);
      double s1 = 0, s0 = 0; uint32_t n1 = 0, n0 = 0;
      for (uint32_t kk = 0; kk < ns; ++kk) {
        uint64_t c2 = cnt[(uint64_t)r * ns + kk];
        if (!c2) continue;
        double m = sum[(uint64_t)r * ns + kk] / (double)c2;
        if (bs[kk] == '1') { s1 += m; ++n1; }
        else if (bs[kk] == '0') { s0 += m; ++n0; }
      }
      if (!n1 || !n0) { thr[r] = (float)(0.0 / 0.0); continue; }
      double hi = s1 / n1, lo = s0 / n0;
      if (!(hi > lo)) { thr[r] = (float)(0.0 / 0.0); continue; }
      thr[r] = (float)(0.5 * (hi + lo));
      ++n_usable;
    }
    write_or_die(fp, thr, (size_t)n_cand * sizeof(float), out);
    fprintf(stderr, "  thresholds: %" PRIu64 " of %" PRIu64
            " patterns binarisable\n", n_usable, n_cand);
    free(sum); free(cnt); free(thr); free(bs);
  }
  if (pad_bytes) {
    static const char pad[8] = {0};
    write_or_die(fp, pad, (size_t)pad_bytes, out);
  }
  free(memb);
  free(meth); free(ambig);
  free(keep); free(sel_max0); free(sel_min1); free(sel_cov); free(sel_pres);
  if (sel_on)
    fprintf(stderr, "  select: %" PRIu64 " CpGs dropped by --qfilter/"
            "--min-cg-depth/--max-frac-na (%.2f%% of non-PNA candidates)\n",
            n_filtered, 100.0 * (double)n_filtered /
            (double)(n_filtered + n_cpg - pna_cpg ? n_filtered + n_cpg - pna_cpg : 1));
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

/* ---------------- membership RLE (YAME format-2 payload) ---------------- */

/* The membership array is one pattern rank per genomic CpG and ~99% of it is
 * the PNA sentinel, so it is exactly what YAME's format-2 run-length coder is
 * for -- the same coder mrmp-export already runs when it writes a .cm, applied
 * in the artifact instead of one step later.
 *
 * It goes through a fmt2 cdata_t rather than a bare codec because
 * compressDataToRLE() is static in YAME's format2.c and reads its values via
 * f2_get_uint64(), so the public way in is to hand it a fmt2. That costs a
 * synthesized key table we never read back; the note in YAME/tmp asks for the
 * codec to be exported over plain values instead.
 *
 * Key id == rank, with n_cand standing for PNA. That identity is the point: the
 * decoder takes ranks straight out of f2_get_uint64 and never parses a label,
 * unlike ms_mrmp_write_mask, which assigns ids in first-seen order because a .cm
 * is meant to be read BY name. */

static uint8_t *memb_compress(const uint32_t *memb, uint64_t n_cpg,
                              uint64_t n_cand, uint64_t *out_n) {
  size_t keys_bytes = 0;
  for (uint64_t r = 0; r < n_cand; ++r) {
    char lbl[32];
    keys_bytes += (size_t)snprintf(lbl, sizeof lbl, "P%" PRIu64, r + 1) + 1;
  }
  keys_bytes += 4;                                   /* "Pna\0" */

  cdata_t c; memset(&c, 0, sizeof c);
  c.fmt = '2'; c.compressed = 0; c.unit = 8; c.n = n_cpg;
  c.s = xcalloc(keys_bytes + 1 + (size_t)n_cpg * 8, 1, "membership fmt2 buffer");
  size_t pos = 0;
  for (uint64_t r = 0; r < n_cand; ++r)
    pos += (size_t)snprintf((char *)c.s + pos, keys_bytes + 1 - pos,
                            "P%" PRIu64, r + 1) + 1;
  memcpy(c.s + pos, "Pna", 4); pos += 4;
  c.s[pos++] = '\0';                                 /* key/data double NUL */
  for (uint64_t i = 0; i < n_cpg; ++i) {
    uint64_t v = memb[i] == MRMP_PNA_MEMBERSHIP ? n_cand : memb[i];
    uint8_t *d = c.s + pos + i * 8;
    for (int b = 0; b < 8; ++b) d[b] = (uint8_t)(v >> (8 * b));
  }
  cdata_compress(&c);                                /* -> RLE fmt2 */

  /* Then deflate. The RLE is where the shape is, but the deflate is where the
   * bytes are -- 2.74x measured on a 34-class global. Section layout is
   * [uint64 rle_bytes][BGZF blocks]; the length rides inline because the header
   * has no field left to hold it. */
  uint64_t blob_n = 0;
  uint8_t *blob = bgzf_deflate_buf(c.s, c.n, &blob_n);
  uint8_t *sec = xcalloc(8 + blob_n, 1, "membership section");
  uint64_t rle_n = c.n;
  memcpy(sec, &rle_n, 8);
  memcpy(sec + 8, blob, blob_n);
  free(blob); free(c.s);
  *out_n = 8 + blob_n;
  return sec;                                        /* caller owns */
}

/* The RLE payload of a membership section, inflated but NOT expanded: the fmt2
 * key table, then a value-width byte, then (value, uint16 run length) records.
 * Caller frees. */
static uint8_t *memb_rle_payload(const uint8_t *buf, uint64_t nbytes, int bgzf,
                                 uint64_t *out_n, const char *what) {
  if (!bgzf) { uint8_t *p = xcalloc(nbytes ? nbytes : 1, 1, "rle"); 
               memcpy(p, buf, nbytes); *out_n = nbytes; return p; }
  if (nbytes < 8) die("membership section is truncated", what);
  uint64_t rle_n = 0;
  memcpy(&rle_n, buf, 8);
  uint8_t *p = xcalloc(rle_n ? rle_n : 1, 1, "rle");
  bgzf_inflate_buf(buf + 8, nbytes - 8, p, rle_n, what);
  *out_n = rle_n;
  return p;
}

static uint32_t *memb_decompress(const uint8_t *buf, uint64_t nbytes,
                                 uint64_t n_cpg, uint64_t n_cand, int bgzf,
                                 const char *what) {
  cdata_t c; memset(&c, 0, sizeof c);
  c.fmt = '2'; c.compressed = 1; c.unit = 8;
  if (bgzf) {
    if (nbytes < 8) die("membership section is truncated", what);
    uint64_t rle_n = 0;
    memcpy(&rle_n, buf, 8);
    c.n = rle_n;
    c.s = xcalloc(rle_n ? rle_n : 1, 1, "membership rle");
    bgzf_inflate_buf(buf + 8, nbytes - 8, c.s, rle_n, what);
  } else {
    c.n = nbytes;
    c.s = xcalloc(nbytes ? nbytes : 1, 1, "membership rle");
    memcpy(c.s, buf, nbytes);
  }
  cdata_t d = decompress(c);
  free(c.s);
  if (cdata_n(&d) != n_cpg) die("membership inflates to the wrong CpG count", what);
  uint32_t *memb = xcalloc(n_cpg, sizeof(uint32_t), "membership");
  for (uint64_t i = 0; i < n_cpg; ++i) {
    uint64_t v = f2_get_uint64(&d, i);
    memb[i] = v >= n_cand ? MRMP_PNA_MEMBERSHIP : (uint32_t)v;
  }
  free_cdata(&d);
  return memb;
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
  const uint32_t *membership;   /* n_cpg; NULL until mrmp_membership() if RLE */
  uint32_t *memb_owned;         /* non-NULL once inflated, freed by mrmp_close */
  int memb_rle;                 /* section is compressed, inflate on demand */
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
 * a nonzero base addresses a later block in the chain. Every offset
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
  /* The membership section is n_cpg * 4 dense, or membership_bytes when RLE. */
  const int memb_rle = (h->flags & MRMP_FLAG_MEMB_RLE) != 0;
  const uint64_t memb_bytes = memb_rle ? h->membership_bytes
                                       : h->n_cpg * sizeof(uint32_t);
  if (memb_rle && !h->membership_bytes)
    die("MRMP artifact claims RLE membership but records no size", path);
  if (!region_ok(h->membership_offset, memb_bytes, 1, sz) ||
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
  /* Inflated LAZILY, by mrmp_membership(). Most openers -- inspect, the pattern
   * decoder, both satellite builders reading the global's header -- never touch
   * membership at all, and `inspect` on a container opens every block in turn,
   * so inflating here would cost one full array per set to answer a question
   * about pattern counts. */
  r->memb_rle = memb_rle;
  r->membership = memb_rle
    ? NULL : (const uint32_t *)(const void *)(blk + h->membership_offset);
}

/* The membership array, inflating it on first use when the section is RLE.
 * Both consumers walk it once and linearly, and each opens one block at a time,
 * so the peak is one array live rather than one per set in a container. */
static const uint32_t *mrmp_membership(mrmp_reader_t *r) {
  if (!r->membership && r->memb_rle)
    r->membership = r->memb_owned =
      memb_decompress((const uint8_t *)(r->blk + r->h->membership_offset),
                      r->h->membership_bytes, r->h->n_cpg, r->h->n_candidates,
                      (r->h->flags & MRMP_FLAG_MEMB_BGZF) != 0, "membership");
  return r->membership;
}

static void mrmp_open(mrmp_reader_t *r, const char *path) {
  mrmp_open_at(r, path, 0, 0);
}

static void mrmp_close(mrmp_reader_t *r) {
  free((void *)r->names);
  free(r->memb_owned);
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

/* ---------------- the chain: sets concatenated, walked ------------------- */

/* Walk the file's blocks. Each MRMPIDX1 carries everything needed to find the
 * next one -- its sections are at known offsets and its size is derivable -- so
 * the chain needs no table and no wrapper, and `cat a.mrmp b.mrmp` is an exact
 * combine. This is the whole reader side of that format. */
ms_mrmpset_t *ms_mrmpset_open(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open MRMP", path);
  if (fseeko(f, 0, SEEK_END)) die("cannot size MRMP", path);
  uint64_t fsz = (uint64_t)ftello(f);

  ms_mrmpset_t *s = xcalloc(1, sizeof(*s), "mrmp chain");
  uint32_t cap = 8;
  s->name        = xcalloc(cap, sizeof(char *), "set names");
  s->block_off   = xcalloc(cap, sizeof(uint64_t), "block offsets");
  s->block_bytes = xcalloc(cap, sizeof(uint64_t), "block sizes");

  uint64_t at = 0;
  while (at < fsz) {
    /* Peek the magic before anything else, however few bytes remain. A chain
     * ends at EOF or at the start of something deliberately not another block:
     * a bundle puts the .mrmp at offset 0 and its MSBNDL1 container right after,
     * exactly as it used to rely on a .cm's BGZF EOF marker to stop yame. The
     * container is far SHORTER than a block header, so this has to be checked
     * before the truncation test rather than after it. Anything else still
     * dies -- stopping only on a known magic is what keeps the bounds check a
     * real corruption test. */
    char mg[8] = {0};
    uint64_t left = fsz - at;
    if (fseeko(f, (off_t)at, SEEK_SET)) die("cannot seek MRMP", path);
    size_t got = fread(mg, 1, left < 8 ? (size_t)left : 8, f);
    if (got >= 7 && !memcmp(mg, "MSBNDL1", 7)) break;
    if (left < sizeof(mrmp_header_t))
      die("MRMP chain ends mid-header -- the file is truncated", path);
    mrmp_header_t h;
    if (fseeko(f, (off_t)at, SEEK_SET) || fread(&h, 1, sizeof h, f) != sizeof h)
      die("cannot read MRMP block header", path);
    if (memcmp(h.magic, MRMPIDX_MAGIC, 8) || h.version != MRMPIDX_VERSION)
      die("bad MRMPIDX1 magic or version in chain", path);
    /* Bound on the UNPADDED end: an artifact written before blocks were padded
     * stops exactly at its last section, so testing the padded stride would
     * reject its final block as one byte past EOF. Advance by the padded
     * stride, clamped, so a chain of new blocks still steps correctly. */
    uint64_t end = ms_mrmp_block_end(&h), nb = ms_mrmp_block_bytes(&h);
    if (end < sizeof(mrmp_header_t) || end > fsz - at)
      die("MRMP block extends past the end of the file", path);
    if (nb > fsz - at) nb = fsz - at;

    if (s->n_sets == cap) {
      cap <<= 1;
      s->name        = realloc(s->name, cap * sizeof(char *));
      s->block_off   = realloc(s->block_off, cap * sizeof(uint64_t));
      s->block_bytes = realloc(s->block_bytes, cap * sizeof(uint64_t));
      if (!s->name || !s->block_off || !s->block_bytes)
        die("out of memory", "mrmp chain grow");
    }
    s->block_off[s->n_sets]   = at;
    s->block_bytes[s->n_sets] = nb;

    /* The name rides in the block, so it survives a cat. An older artifact has
     * name_offset 0 and gets a positional label instead. */
    char *nm = NULL;
    if (h.name_offset && h.name_offset < nb) {
      char buf[256];
      if (fseeko(f, (off_t)(at + h.name_offset), SEEK_SET)) die("cannot seek", path);
      size_t got = fread(buf, 1, sizeof buf - 1, f);
      buf[got] = '\0';
      if (got) { nm = xcalloc(strlen(buf) + 1, 1, "set name"); strcpy(nm, buf); }
    }
    if (!nm) {
      char buf[32]; snprintf(buf, sizeof buf, "set%u", s->n_sets);
      nm = xcalloc(strlen(buf) + 1, 1, "set name"); strcpy(nm, buf);
    }
    s->name[s->n_sets] = nm;

    ++s->n_sets;
    at += nb;
  }
  fclose(f);
  if (!s->n_sets) die("MRMP file holds no sets", path);
  return s;
}

void ms_mrmpset_free(ms_mrmpset_t *s) {
  if (!s) return;
  for (uint32_t i = 0; i < s->n_sets; ++i) free(s->name[i]);
  free(s->name); free(s->block_off); free(s->block_bytes); free(s);
}

void ms_mrmp_chain_write(const char *out, uint32_t n_sets,
                         const void *const *block, const uint64_t *block_bytes) {
  if (!n_sets) die("a MRMP file needs at least one set", out);
  FILE *f = fopen(out, "wb");
  if (!f) die("cannot write MRMP", out);
  for (uint32_t i = 0; i < n_sets; ++i) {
    /* Each block is already padded to a multiple of 8 by whoever built it, so
     * nothing is inserted between them. That is what makes this identical to
     * `cat` of the same blocks, and what keeps the next header 8-aligned for a
     * reader that casts it in place. */
    if (block_bytes[i] & 7u) die("MRMP block is not 8-aligned", out);
    write_or_die(f, block[i], (size_t)block_bytes[i], out);
  }
  fclose(f);
}

/* Write a chain by COPYING each block from its source file, patching only its
 * n_selected on the way past.
 *
 * mrmp-pool needs this and an in-memory writer cannot serve it: a block carries
 * a membership section per genomic CpG, so holding every input at once costs the
 * SUM of the inputs -- 8.7 GB for ~100 dense sets, which OOMed a login node.
 * Streaming makes the peak one 8 MB buffer regardless of set count.
 *
 * The pooled cut is the only mutation, and n_selected lives in the block's first
 * 128 bytes, so patching the first chunk is enough; no block is rewritten, and
 * the output is byte-for-byte a concatenation of the (patched) inputs. */
static void img_put(char *img, uint64_t *at, const void *p, size_t n);

/* Rewrite a block to hold ONLY its first `keep_n` patterns, folding the CpGs of
 * every dropped pattern into PNA.
 *
 * This is what makes --pooled-top a real cut rather than a view. It used to be
 * expressed by shrinking n_selected while every pattern stayed on disk, so a
 * file that said 1000 held 5,005 and a consumer had to be told which prefix was
 * live. Now the artifact IS its patterns, n_selected == n_candidates always, and
 * the pooled .mrmp can travel to a model unchanged.
 *
 * Winners are a PREFIX of each set's own ranking -- both rankings are by CpG
 * count descending -- so keeping the first keep_n is exactly keeping the pooled
 * winners, and no pattern has to be dropped from the middle.
 *
 * content_checksum is carried over deliberately. It hashes the per-CpG key
 * stream, which is a property of how the reference RESOLVED, and pruning drops
 * patterns without re-resolving any CpG -- so it still identifies the build this
 * came from, and two differently-pruned files share it correctly. */
static void prune_block(const char *path, uint64_t base, uint64_t blk_bytes,
                        uint32_t keep_n, void **img_out, uint64_t *bytes_out) {
  mrmp_reader_t r; mrmp_open_at(&r, path, base, blk_bytes);
  const mrmp_header_t *h = r.h;
  const uint32_t ns = h->n_samples, nw = r.nw;
  const uint64_t n_cpg = h->n_cpg;
  if (keep_n > h->n_candidates) keep_n = (uint32_t)h->n_candidates;

  const uint32_t *memb = mrmp_membership(&r);
  uint32_t *memb2 = xcalloc(n_cpg, sizeof(uint32_t), "pruned membership");
  uint64_t pna_cpg = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) {
    uint32_t rank = memb[i];
    if (rank == MRMP_PNA_MEMBERSHIP || rank >= keep_n) {
      memb2[i] = MRMP_PNA_MEMBERSHIP; ++pna_cpg;
    } else memb2[i] = rank;
  }
  uint64_t memb_n = 0;
  uint8_t *memb_rle = memb_compress(memb2, n_cpg, keep_n, &memb_n);
  if (memb_n > UINT32_MAX) die("compressed membership exceeds 4 GB", path);

  const char *set_name = h->name_offset ? r.blk + h->name_offset : "set";
  const float *thr = (h->flags & MRMP_FLAG_THRESH)
                   ? (const float *)(const void *)(r.blk + h->thresh_offset) : NULL;

  mrmp_header_t hd = *h;
  hd.n_candidates = keep_n; hd.n_selected = keep_n;
  hd.pna_cpg = pna_cpg;
  hd.membership_bytes = (uint32_t)memb_n;
  hd.flags |= MRMP_FLAG_MEMB_RLE | MRMP_FLAG_MEMB_BGZF;

  uint64_t off = sizeof(hd);
  hd.refname_offset = off;    off += strlen(r.refname) + 1;
  hd.name_offset = (uint32_t)off; off += strlen(set_name) + 1;
  hd.names_offset = off;      for (uint32_t k = 0; k < ns; ++k) off += strlen(r.names[k]) + 1;
  hd.patterns_offset = off;   off += (uint64_t)keep_n * mrmp_pattern_stride(ns);
  hd.membership_offset = off; off += memb_n;
  if (thr) { hd.thresh_offset = off; off += (uint64_t)keep_n * sizeof(float); }
  const uint64_t img_bytes = (off + 7u) & ~7ull;

  char *img = xcalloc(img_bytes, 1, "pruned block");
  uint64_t at = 0;
  img_put(img, &at, &hd, sizeof(hd));
  img_put(img, &at, r.refname, strlen(r.refname) + 1);
  img_put(img, &at, set_name, strlen(set_name) + 1);
  for (uint32_t k = 0; k < ns; ++k)
    img_put(img, &at, r.names[k], strlen(r.names[k]) + 1);
  for (uint32_t p = 0; p < keep_n; ++p) {
    img_put(img, &at, pat_key(&r, p), (size_t)nw * sizeof(uint64_t));
    uint64_t cnt = pat_count(&r, p);
    img_put(img, &at, &cnt, sizeof(uint64_t));
  }
  img_put(img, &at, memb_rle, (size_t)memb_n);
  if (thr) img_put(img, &at, thr, (size_t)keep_n * sizeof(float));

  free(memb_rle); free(memb2);
  mrmp_close(&r);
  *img_out = img; *bytes_out = img_bytes;
}

static void chain_write_streamed(const char *out, uint32_t n_sets,
                                 const char *const *src, const uint64_t *src_off,
                                 const uint64_t *block_bytes,
                                 const uint32_t *keep) {
  if (!n_sets) die("a MRMP file needs at least one set", out);
  FILE *f = fopen(out, "wb");
  if (!f) die("cannot write MRMP", out);
  const size_t CH = 8u << 20;
  char *buf = xcalloc(CH, 1, "block copy buffer");
  for (uint32_t i = 0; i < n_sets; ++i) {
    if (block_bytes[i] & 7u) die("MRMP block is not 8-aligned", src[i]);
    FILE *in = fopen(src[i], "rb");
    if (!in) die("cannot open", src[i]);
    if (fseeko(in, (off_t)src_off[i], SEEK_SET)) die("cannot seek", src[i]);
    uint64_t left = block_bytes[i], done = 0;
    while (left) {
      size_t want = left < (uint64_t)CH ? (size_t)left : CH;
      if (fread(buf, 1, want, in) != want) die("short read", src[i]);
      if (!done) {
        if (want < sizeof(mrmp_header_t)) die("block is smaller than its header", src[i]);
        memcpy(buf + offsetof(mrmp_header_t, n_selected), &keep[i], sizeof(uint32_t));
      }
      write_or_die(f, buf, want, out);
      left -= want; done += want;
    }
    fclose(in);
  }
  free(buf);
  fclose(f);
}

/* ---------------- inspect ----------------------------------------------- */

static const char *commafmt_local(uint64_t v, char *buf);

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
  char cb[32], cb2[32];
  const char *nm = h->name_offset ? r.blk + h->name_offset : NULL;
  uint64_t covered = h->n_cpg - h->pna_cpg;
  uint64_t memb_b = (h->flags & MRMP_FLAG_MEMB_RLE)
                  ? h->membership_bytes : h->n_cpg * sizeof(uint32_t);
  uint64_t blk_b = ms_mrmp_block_bytes(h);

  printf("\nMRMP  %s\n", path);
  printf("  %-14s MRMPIDX1 v%u, one set\n", "format", h->version);
  if (nm) printf("  %-14s %s\n", "name", nm);
  printf("  %-14s %s\n", "reference", r.refname);
  printf("\n");
  printf("  %-14s %u\n", "classes", h->n_samples);
  printf("  %-14s %s\n", "patterns", commafmt_local(h->n_candidates, cb));
  printf("  %-14s %s of %s CpGs (%.2f%%)\n", "covered",
         commafmt_local(covered, cb), commafmt_local(h->n_cpg, cb2),
         h->n_cpg ? 100.0 * covered / (double)h->n_cpg : 0.0);
  printf("  %-14s %s CpGs (%.2f%%) match no pattern\n", "PNA",
         commafmt_local(h->pna_cpg, cb),
         h->n_cpg ? 100.0 * h->pna_cpg / (double)h->n_cpg : 0.0);
  printf("\n");

  /* Storage, because it is the thing that changed most and the thing a reader
   * most often wants to check: membership dominates a block, so its encoding
   * and its ratio explain the file size on their own. */
  printf("  %-14s %s bytes\n", "block", commafmt_local(blk_b, cb));
  printf("  %-14s %s bytes, %s (%.0fx vs dense)\n", "membership",
         commafmt_local(memb_b, cb),
         (h->flags & MRMP_FLAG_MEMB_BGZF) ? "RLE + BGZF"
           : (h->flags & MRMP_FLAG_MEMB_RLE) ? "RLE" : "dense uint32",
         memb_b ? (double)(h->n_cpg * 4) / (double)memb_b : 1.0);
  printf("  %-14s %s\n", "thresholds",
         (h->flags & MRMP_FLAG_THRESH) ? "present (per-pattern midpoints)"
                                       : "absent (consumers assume 0.5)");
  printf("\n");
  printf("  %-14s mincov=%u beta=%.3f max_ambig=%.3f min_major_fold=%.3f\n",
         "resolution", h->mincov, h->beta_threshold, h->max_ambig_frac,
         h->min_major_fold);
  printf("  %-14s %016" PRIx64 "  (over the per-CpG key stream)\n",
         "checksum", h->content_checksum);
  if (h->n_selected != h->n_candidates)
    printf("  %-14s only the first %u pattern(s) are live -- this file predates\n"
           "  %-14s mrmp-pool's prune and carries more than it uses\n",
           "NOTE", h->n_selected, "");
  printf("\n");

  /* The header stores only word 0 of the sentinel, so rebuild the full key --
   * it is all-'2' and therefore fully determined by n_samples -- and check the
   * stored word agrees, which catches a header/sample-count mismatch. */
  char *buf = xcalloc(h->n_samples + 1, 1, "string buffer");
  uint32_t pna_nw = mrmp_key_words(h->n_samples);
  uint64_t *pna = xcalloc(pna_nw, sizeof(uint64_t), "pna key");
  for (uint32_t sIdx = 0; sIdx < h->n_samples; ++sIdx)
    pna[sIdx / MRMP_TRITS_PER_WORD] = pna[sIdx / MRMP_TRITS_PER_WORD] * 3 + 2;
  if (pna[0] != h->pna_key) die("PNA key disagrees with n_samples", path);
  key_to_string(pna, h->n_samples, buf);
  free(pna);

  if (show_patterns) {
    printf("  PNA sentinel  %s\n\n", buf);
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

/* Export every set of a container as <dir>/<name>.cm, plus an order.txt naming
 * them in set order.
 *
 * This is the step that CLOSES the four-command workflow. mrmp-pool emits a
 * a chain and classify-featurize consumes .cm masks, so without a per-block
 * export there is no path from a pooled artifact to a feature vector, and the
 * pooling is unusable however correct it is.
 *
 * One file PER SET rather than one fused mask, because a set's pattern ranks are
 * its own: P1 names a different pattern in each, so they cannot share a label
 * space in one .cm. The featurizer already takes N masks and pools their columns
 * itself, which is the same reason mrmp-pool does not have to.
 *
 * order.txt is the one thing a reader cannot recover from the .cm files alone --
 * a directory listing is alphabetical, while the pooled feature vector is laid
 * out in SET order, and a classifier handed its columns in the wrong order fails
 * silently rather than loudly. */
static void mrmp_block_mask_cdata(const char *artifact, uint64_t base,
                                  uint64_t blk_bytes, const char *pna_label,
                                  uint32_t top_k, cdata_t *out,
                                  uint64_t *n_keys_out);

static int export_container(const char *path, const char *out_cm,
                            const char *pna_label, uint32_t top_k) {
  ms_mrmpset_t *s = ms_mrmpset_open(path);

  /* One .cm holding every set as a record, plus the .idx naming them -- YAME's
   * own multi-record store convention, written with YAME's own index writer so
   * the offsets are the BGZF VIRTUAL offsets it expects rather than byte
   * offsets. (Hand-writing those is the classic way to scramble sample->data.)
   *
   * This used to be a directory of <set>.cm files plus an order.txt. Nothing in
   * methscope reads either any more -- classify-featurize takes the .mrmp and
   * the bundle carries one -- so export exists purely to hand sets to YAME, and
   * it should speak YAME's idiom. The .idx is also strictly better than
   * order.txt was: it names records rather than ordering them, so
   * `yame subset -s <set>` works. */
  BGZF *fp = bgzf_open2(out_cm, "w");
  if (!fp) die("cannot create", out_cm);
  index_t *idx = kh_init(index);   /* insert_index fills a table, never makes one */
  for (uint32_t k = 0; k < s->n_sets; ++k) {
    cdata_t c; uint64_t n_keys = 0;
    mrmp_block_mask_cdata(path, s->block_off[k], s->block_bytes[k],
                          pna_label, top_k, &c, &n_keys);
    int64_t voff = bgzf_tell(fp);      /* before the record, not after */
    cdata_write1(fp, &c);
    free_cdata(&c);
    idx = insert_index(idx, s->name[k], voff);   /* borrows the name string */
  }
  if (bgzf_close(fp) < 0) die("error closing", out_cm);

  char *ipath = get_fname_index(out_cm);
  FILE *ifp = fopen(ipath, "w");
  if (!ifp) die("cannot create index", ipath);
  writeIndex(ifp, idx);
  fclose(ifp);
  freeIndex(idx);                  /* the keys belong to the walked chain */
  fprintf(stderr, "[methscope] mrmp-export: %u sets -> %s (+ %s)\n",
          s->n_sets, out_cm, ipath);
  free(ipath);
  ms_mrmpset_free(s);
  return 0;
}

int main_mrmp_export(int argc, char *argv[]) {
  g_cmd = "mrmp-export";
  const char *pos[2] = {NULL, NULL}, *patterns = NULL, *counts = NULL,
             *pna_label = "Pna", *set_name = NULL;
  int npos = 0;
  uint32_t top_k = 1000;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stderr,
        "Usage: methscope mrmp-export [options] IN.mrmp OUT.cm\n\n"
        "  IN               a .mrmp: one set, or a chain of several\n"
        "  OUT.cm           per-CpG P1..PK/Pna labels as a YAME format-2 mask\n"
        "  OUT.cm           a multi-set input writes ONE .cm holding every set as\n"
        "                   a record, plus OUT.cm.idx naming them -- YAME's own\n"
        "                   store convention, so `yame subset -s <set>` works.\n\n"
        "  This command is an INTERFACE TO YAME, not part of the pipeline.\n"
        "  classify-featurize reads a .mrmp directly and a bundle carries one, so\n"
        "  nothing in methscope consumes a .cm any more; export exists to hand\n"
        "  sets to yame and should speak yame's idiom rather than ours.\n\n"
        "  --set NAME       export just this set of a container, to OUT.cm\n"
        "  --top K          rank cut for the mask and --patterns (default 1000)\n"
        "  --patterns TSV   also write top-K patterns: string<tab>P<rank><tab>count\n"
        "  --counts TSV     also write every pattern (incl. PNA): count<tab>string\n"
        "  --pna-label NAME background label in the mask (default Pna)\n");
      return 0;
    } else if (!strcmp(a, "--set") && i + 1 < argc) set_name = argv[++i];
    else if (!strcmp(a, "--patterns") && i + 1 < argc) patterns = argv[++i];
    else if (!strcmp(a, "--counts") && i + 1 < argc) counts = argv[++i];
    else if (!strcmp(a, "--top") && i + 1 < argc) top_k = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--pna-label") && i + 1 < argc) pna_label = argv[++i];
    else if (a[0] == '-') die("unrecognized option", a);
    else if (npos < 2) pos[npos++] = a;
    else die("too many arguments", a);
  }
  if (npos != 2) die("need IN.mrmp and OUT.cm (see mrmp-export -h)", NULL);
  const char *path = pos[0], *mask = pos[1];

  /* One set or many is the same format now, so the arity decides: a multi-set
   * chain with no --set goes to a directory, a single-set file to one .cm. */
  ms_mrmpset_t *s = ms_mrmpset_open(path);
  if (s->n_sets > 1 && !set_name) {
    if (patterns || counts)
      die("--patterns/--counts describe ONE set; add --set NAME", path);
    ms_mrmpset_free(s);
    return export_container(path, mask, pna_label, top_k);
  }
  uint64_t base = 0, blk_bytes = 0;
  if (set_name) {
    uint32_t k = 0;
    for (; k < s->n_sets && strcmp(s->name[k], set_name); ++k) {}
    if (k == s->n_sets) die("no such set in the file", set_name);
    base = s->block_off[k]; blk_bytes = s->block_bytes[k];
  } else {
    base = s->block_off[0]; blk_bytes = s->block_bytes[0];
  }
  mrmp_reader_t r; mrmp_open_at(&r, path, base, blk_bytes);
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

  if (mask) ms_mrmp_write_mask_at(path, base, blk_bytes, mask, pna_label, top_k);

  free(buf);
  mrmp_close(&r);
  if (s) ms_mrmpset_free(s);
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
  const uint32_t *memb = mrmp_membership(&r);
  for (uint64_t i = 0; i < n_cpg; ++i) {
    uint32_t rank = memb[i];
    group[i] = (rank != MRMP_PNA_MEMBERSHIP && rank < K)
             ? (uint16_t)(rank + 1) : 0;
  }
  mrmp_close(&r);
}

void ms_mrmp_membership_runs(const char *path, uint64_t base, uint64_t blk_bytes,
                             ms_mrmp_run_cb cb, void *ctx) {
  mrmp_reader_t r; mrmp_open_at(&r, path, base, blk_bytes);
  const mrmp_header_t *h = r.h;
  if (!(h->flags & MRMP_FLAG_MEMB_RLE)) {
    /* Pre-RLE artifact: synthesize runs so callers need only one path. */
    const uint32_t *m = (const uint32_t *)(const void *)(r.blk + h->membership_offset);
    uint64_t i = 0;
    while (i < h->n_cpg) {
      uint64_t j = i + 1;
      while (j < h->n_cpg && m[j] == m[i]) ++j;
      cb(ctx, i, j - i, m[i]);
      i = j;
    }
    mrmp_close(&r);
    return;
  }
  uint64_t n = 0;
  uint8_t *p = memb_rle_payload((const uint8_t *)(r.blk + h->membership_offset),
                                h->membership_bytes,
                                (h->flags & MRMP_FLAG_MEMB_BGZF) != 0, &n, path);
  /* Keys are NUL-separated and closed by an extra NUL, so the first double-NUL
   * ends the table; no key is ever empty (they are P1..PN and Pna). */
  uint64_t d = 0;
  while (d + 1 < n && !(p[d] == 0 && p[d + 1] == 0)) ++d;
  if (d + 2 >= n) die("membership RLE has no data section", path);
  d += 2;
  uint8_t vb = p[d++];
  if (!vb || vb > 8) die("membership RLE has a bad value width", path);
  uint64_t pos = 0;
  while (d + vb + 2 <= n) {
    uint64_t v = 0;
    for (uint8_t k = 0; k < vb; ++k) v |= (uint64_t)p[d + k] << (8 * k);
    d += vb;
    uint64_t len = (uint64_t)p[d] | ((uint64_t)p[d + 1] << 8);
    d += 2;
    if (len) cb(ctx, pos, len, v >= h->n_candidates ? MRMP_PNA_MEMBERSHIP : (uint32_t)v);
    pos += len;
  }
  if (pos != h->n_cpg) die("membership runs do not cover the CpG count", path);
  free(p);
  mrmp_close(&r);
}

void ms_mrmp_group_map(const char *artifact, uint16_t *group, uint64_t n_cpg,
                       uint32_t patterns) {
  ms_mrmp_group_map_at(artifact, 0, group, n_cpg, patterns);
}

/* Build the fmt2 mask cdata for ONE block, without deciding where it goes.
 *
 * Split out so a block's mask can be either a standalone .cm or one record of a
 * multi-record store: mrmp-export's only remaining job is handing sets to YAME,
 * and YAME's idiom for many records is one file plus a .idx of names, not a
 * directory of files plus a hand-rolled order list. */
static void mrmp_block_mask_cdata(const char *artifact, uint64_t base,
                                  uint64_t blk_bytes, const char *pna_label,
                                  uint32_t top_k, cdata_t *out,
                                  uint64_t *n_keys_out) {
  mrmp_reader_t r; mrmp_open_at(&r, artifact, base, blk_bytes);
  const mrmp_header_t *h = r.h;
  if (!pna_label) pna_label = "Pna";
  if (!top_k) die("mask needs a positive rank cut", artifact);
  if (top_k > h->n_selected) top_k = h->n_selected;
  {
    /* Build a raw YAME format-2 cdata directly (no genome-sized text file),
     * mirroring fmt2_read_raw: first-seen key order over genomic CpGs, then
     * cdata_compress (RLE) + cdata_write. Labels: P(rank+1) or the PNA label. */
    const uint64_t n = h->n_cpg, K = top_k;
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
    const uint32_t *memb = mrmp_membership(&r);
    for (uint64_t i = 0; i < n; ++i) {
      uint32_t rank = memb[i];
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
    for (uint64_t kk = 0; kk < n_keys; ++kk) free(keys[kk]);
    free(keys); free(label_id); free(key_of_rank);
    *out = c; *n_keys_out = n_keys;
  }
  mrmp_close(&r);
}

void ms_mrmp_write_mask_at(const char *artifact, uint64_t base,
                           uint64_t blk_bytes, const char *out_cm,
                           const char *pna_label, uint32_t top_k) {
  cdata_t c; uint64_t n_keys = 0;
  mrmp_block_mask_cdata(artifact, base, blk_bytes, pna_label, top_k, &c, &n_keys);
  cdata_write((char *)out_cm, &c, "w", 0);
  free_cdata(&c);
  fprintf(stderr, "[methscope] mrmp: wrote mask %s (%" PRIu64 " labels)\n",
          out_cm, n_keys);
}

void ms_mrmp_write_mask(const char *artifact, const char *out_cm,
                        const char *pna_label, uint32_t top_k) {
  ms_mrmp_write_mask_at(artifact, 0, 0, out_cm, pna_label, top_k);
}


/* mrmp-pack is gone: a chain IS a concatenation, so `cat a.mrmp b.mrmp > c.mrmp`
 * is the exact combine it used to perform. mrmp-pool remains, because applying
 * a shared column budget is selection rather than merging and cat cannot do it. */

/* One pooled candidate: which set it came from and how many CpGs carry it.
 * File scope, with a plain comparator -- a nested function would be a GCC
 * extension and would force an executable stack for the trampoline. */
typedef struct { uint64_t count; uint32_t set; } ent_t;
static int pooled_cmp(const void *a, const void *b) {
  const ent_t *x = a, *y = b;
  if (x->count > y->count) return -1;
  if (x->count < y->count) return 1;
  return (x->set < y->set) ? -1 : (x->set > y->set);
}

int main_mrmp_pool(int argc, char *argv[]) {
  g_cmd = "mrmp-pool";
  const char *out = NULL;
  uint32_t pooled_top = 1000;
  int i = 1;
  for (; i < argc; ++i) {
    if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--pooled-top") && i + 1 < argc)
      pooled_top = (uint32_t)parse_u64(argv[++i], "--pooled-top");
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      ms_help(stderr,
        "Usage: methscope mrmp-pool [options] -o OUT.mrmp IN.mrmp [IN.mrmp ...]\n\n"
        "Pool several MRMP sets into one chain and cut them to a\n"
        "shared column budget. Needs no store and no reference -- pattern CpG\n"
        "counts are already in each artifact -- so re-pooling at a different\n"
        "budget costs seconds. Plain concatenation is just `cat`; this is the\n"
        "step that SELECTS.\n\n"
        "A set is a set: the inputs may come from any generator and from\n"
        "DIFFERENT stores, as long as they share a row space. Nothing here\n"
        "distinguishes a global from a satellite.\n\n"
        "Each input is a CHAIN of one or more sets and expands into all of\n"
        "them under their own names, so a mrmp-build-thin output competes for\n"
        "slots as the many 2-class sets it is, not as one blob. A set carries\n"
        "its name in its own block (mrmp-build --name), so pooling needs no\n"
        "NAME: prefix and cannot mislabel anything.\n\n"
        "  --pooled-top N   total pattern budget across every input, ranked by\n"
        "                   CpG count (default 1000). Sets COMPETE for these\n"
        "                   slots rather than being reserved any, so a set that\n"
        "                   cannot field well-covered patterns loses -- the right\n"
        "                   verdict, since a pattern too thin to rank is too thin\n"
        "                   to trust. The cut PRUNES: the output holds exactly N\n"
        "                   patterns, with the CpGs of the rest folded into PNA\n"
        "                   and any set winning nothing dropped, so the file IS\n"
        "                   what it claims and can go to a model unchanged. To\n"
        "                   re-pool at another budget, re-run this on the same\n"
        "                   generator outputs -- they are untouched. 0 disables\n"
        "                   the cut, leaving a plain `cat` plus the row check.\n"
        "  -o OUT           output container\n");
      return 0;
    }
    else if (argv[i][0] == '-') die("unrecognized option", argv[i]);
    else break;
  }
  if (!out || argc - i < 1) die("need -o OUT and at least one IN.mrmp", NULL);

  /* Every input is a chain of one or more sets and expands into all of them. That is what makes the four-command workflow
   * close: mrmp-build-thin emits ONE file holding many 2-class sets, and pooling
   * has to see them as the separate competitors they are, not as one blob.
   * Expanded blocks keep the container's own set names -- those came from the
   * generator that knows what each set is, and are what makes `inspect`'s pooled
   * table readable. So the input count is not the set count. */
  uint32_t cap = (uint32_t)(argc - i), n = 0;
  char **name = xcalloc(cap, sizeof(char *), "set names");
  uint64_t *len = xcalloc(cap, sizeof(uint64_t), "block sizes");
  uint64_t *soff = xcalloc(cap, sizeof(uint64_t), "block offsets");
  const char **path = xcalloc(cap, sizeof(char *), "paths");
  for (uint32_t k = 0; k < (uint32_t)(argc - i); ++k) {
    const char *p = argv[i + k];
    ms_mrmpset_t *s = ms_mrmpset_open(p);
    uint32_t take = s->n_sets;
    if (n + take > cap) {
      cap = n + take + 8;
      name = realloc(name, cap * sizeof(char *));
      len  = realloc(len, cap * sizeof(uint64_t));
      soff = realloc(soff, cap * sizeof(uint64_t));
      path = realloc(path, cap * sizeof(char *));
      if (!name || !len || !soff || !path) die("out of memory", "pool input grow");
    }
    for (uint32_t j = 0; j < take; ++j) {
      soff[n] = s->block_off[j]; len[n] = s->block_bytes[j];
      name[n] = s->name[j]; path[n] = p;
      ++n;
    }
    /* the name STRINGS are now owned by name[], so release the walk's arrays
     * but not ms_mrmpset_free, which would take the strings with them */
    free(s->block_off); free(s->block_bytes); free(s->name); free(s);
  }

  /* Expansion can yield nothing even though inputs were given: a container
   * holding no blocks. The walker rejects an empty file, but this keeps the
   * guard local rather than depending on that. */
  if (!n) die("inputs expanded to no sets at all", NULL);

  /* Headers only, never whole blocks. A block carries one membership entry per
   * genomic CpG -- 87 MB at 21.8 M CpGs -- so reading them all in costs the SUM
   * of the inputs, which was ~8.7 GB for the ~100 sets the pair generators now
   * produce and OOMed a login node. Everything pooling needs (n_cpg to check the
   * row space, n_candidates and the counts to rank) sits in the header and the
   * pattern records, which together are kilobytes. */
  mrmp_header_t *hd = xcalloc(n, sizeof(mrmp_header_t), "block headers");
  for (uint32_t k = 0; k < n; ++k) {
    FILE *f = fopen(path[k], "rb");
    if (!f) die("cannot open", path[k]);
    if (fseeko(f, (off_t)soff[k], SEEK_SET) ||
        fread(&hd[k], 1, sizeof hd[k], f) != sizeof hd[k])
      die("cannot read MRMP block header", path[k]);
    fclose(f);
    if (memcmp(hd[k].magic, MRMPIDX_MAGIC, 8)) die("not a MRMPIDX1 block", path[k]);
  }

  /* Same row space, ENFORCED not assumed. Membership arrays are indexed by CpG
   * row, so mixing references would scramble pattern-to-CpG assignment exactly
   * the way a hand-concatenated .cg scrambles sample-to-data. n_cpg is in the
   * header, so the check is free. */
  for (uint32_t k = 1; k < n; ++k) {
    if (hd[k].n_cpg != hd[0].n_cpg) {
      fprintf(stderr, "[methscope] mrmp-pool: %s has %" PRIu64 " CpGs but %s "
              "has %" PRIu64 " -- different row spaces cannot be pooled\n",
              path[k], hd[k].n_cpg, path[0], hd[0].n_cpg);
      exit(1);
    }
  }

  uint32_t *won = xcalloc(n, sizeof(uint32_t), "per-set winners");
  for (uint32_t k = 0; k < n; ++k) won[k] = hd[k].n_selected;

  if (pooled_top) {
    /* Both the within-set ranking and the pooled ranking are by CpG count
     * descending, so a set's pooled winners are a PREFIX of its own ranking.
     * That is what lets the cut be expressed by shrinking each block's
     * n_selected: no pattern has to be dropped from the middle, and every
     * consumer already takes min(n_selected, its own K). */
    uint64_t n_all = 0;
    for (uint32_t k = 0; k < n; ++k) n_all += hd[k].n_candidates;
    ent_t *e = xcalloc(n_all ? n_all : 1, sizeof(ent_t), "pooled entries");
    uint64_t m = 0;
    /* Just the pattern records of each block -- n_candidates * stride bytes,
     * kilobytes -- read at the block's own base, since an input may be one set
     * inside a container rather than a whole file. */
    for (uint32_t k = 0; k < n; ++k) {
      uint64_t st = mrmp_pattern_stride(hd[k].n_samples);
      uint64_t koff = (uint64_t)mrmp_key_words(hd[k].n_samples) * sizeof(uint64_t);
      uint64_t nb = hd[k].n_candidates * st;
      if (!nb) continue;
      char *rec = xcalloc(nb, 1, "pattern records");
      FILE *f = fopen(path[k], "rb");
      if (!f) die("cannot open", path[k]);
      if (fseeko(f, (off_t)(soff[k] + hd[k].patterns_offset), SEEK_SET) ||
          fread(rec, 1, nb, f) != nb)
        die("cannot read MRMP pattern records", path[k]);
      fclose(f);
      for (uint64_t r = 0; r < hd[k].n_candidates; ++r) {
        memcpy(&e[m].count, rec + r * st + koff, sizeof(uint64_t));
        e[m].set = k; ++m;
      }
      free(rec);
    }
    qsort(e, m, sizeof(ent_t), pooled_cmp);
    uint64_t take = m < pooled_top ? m : pooled_top;
    memset(won, 0, n * sizeof(uint32_t));
    for (uint64_t j = 0; j < take; ++j) ++won[e[j].set];
    for (uint32_t k = 0; k < n; ++k)
      fprintf(stderr, "  %-22s %6u of %" PRIu64 " patterns keep a column\n",
              name[k], won[k], hd[k].n_candidates);
    free(e);
  }

  if (!pooled_top) {
    /* No cut: nothing to prune, so this is a byte-for-byte concatenation --
     * exactly what `cat` of the same inputs produces. */
    chain_write_streamed(out, n, path, soff, len, won);
  } else {
    /* PRUNE. Each block is rewritten to hold only the patterns it won, with the
     * CpGs of the rest folded into PNA, so the output holds exactly --pooled-top
     * patterns and can travel to a model as-is.
     *
     * A set that won NOTHING is dropped entirely rather than kept as an empty
     * block: it would otherwise export a mask whose only label is background and
     * contribute a dead all-PNA column to every fused feature matrix. */
    uint32_t m = 0;
    void **img = xcalloc(n, sizeof(void *), "pruned blocks");
    uint64_t *ilen = xcalloc(n, sizeof(uint64_t), "pruned sizes");
    uint32_t dropped = 0;
    for (uint32_t k = 0; k < n; ++k) {
      if (!won[k]) { ++dropped; continue; }
      prune_block(path[k], soff[k], len[k], won[k], &img[m], &ilen[m]);
      ++m;
    }
    if (!m) die("every set was cut to nothing; raise --pooled-top", out);
    if (dropped)
      fprintf(stderr, "[methscope] mrmp-pool: %u set(s) won no column and were "
              "dropped\n", dropped);
    ms_mrmp_chain_write(out, m, (const void *const *)img, ilen);
    for (uint32_t k = 0; k < m; ++k) free(img[k]);
    free(img); free(ilen);
    n = m;
  }
  fprintf(stderr, "[methscope] mrmp-pool: %u sets, budget %u -> %s\n",
          n, pooled_top, out);
  free(name); free(len); free(soff); free(path); free(hd); free(won);
  return 0;
}

/* ---------------- mrmp-build-thin --------------------------------------- */

/* One 2-class satellite per (thin class, partner). A class is thin because it
 * LACKS THE CELLS to define a pattern, not because it resembles anything, so
 * the fix is a set small enough that its few cells can still hold an opinion.
 *
 * Three pieces, in order: the thin list (store labels minus the global's), the
 * partner search (projection over reference pattern-average vectors), and one
 * MRMPIDX1 per surviving pair. They are all here rather than in a new file
 * because the per-pair build IS mrmp-build's resolution -- resolve_cpg, the
 * pattern hash, rank_cmp -- over a subset of the store's records. */

/* Read one record of a store by BGZF virtual offset. Random access, not a
 * sequential scan, so a 2-class set costs two record inflates rather than a
 * pass over all 41. */
static cdata_t read_record_at(cfile_t *cf, int64_t voff, const char *what) {
  if (voff < 0) die("store index carries no offset for sample", what);
  if (bgzf_seek(cf->fh, voff, SEEK_SET) != 0) die("cannot seek store", what);
  cdata_t c = read_cdata1(cf);
  if (!c.n) die("store record is empty", what);
  decompress_in_situ(&c);
  if (c.fmt != '3') die("store must be format-3 (M/U) .cg", what);
  return c;
}

static void img_put(char *img, uint64_t *at, const void *p, size_t n) {
  memcpy(img + *at, p, n); *at += n;
}

typedef struct {
  void    *img;      /* complete MRMPIDX1 image, malloc'd */
  uint64_t bytes;
  uint64_t n_pat;    /* patterns still holding a CpG after selection */
  uint64_t n_kept;   /* CpGs surviving selection */
} subset_block_t;

/* Build a complete MRMPIDX1 for `ns` named classes of `store`, in memory.
 *
 * The binstring parameters come from the GLOBAL artifact's header rather than
 * from flags of their own. A satellite that resolved CpGs on a different rule
 * than the global it supplements would put two incompatible pattern definitions
 * in one pooled feature vector, and there is no flag combination a user could
 * pass that makes that a good idea.
 *
 * The pass order differs from mrmp-build. There, every cut is a property of the
 * CpG alone, so a failing CpG is dropped before it is ever interned and the
 * counts are post-filter by construction. The union rule is not: top-N by
 * delta_mean is PER BINSTRING, so the binstrings must exist first. Hence
 * build -> select -> RECOUNT. The recount is not bookkeeping: mrmp-pool ranks
 * columns by exactly these counts, so a pattern still advertising the CpGs
 * selection took away would win slots it cannot fill. */
static void build_subset_block(const char *store, uint32_t ns,
                               char *const *label, const int64_t *voff,
                               const mrmp_header_t *gh,
                               const ms_select_opt_t *sel,
                               const char *set_name,
                               subset_block_t *out) {
  const uint32_t mincov = gh->mincov;
  const float beta_thr = gh->beta_threshold, max_ambig = gh->max_ambig_frac,
              min_fold = gh->min_major_fold;
  const uint32_t stride = (ns + 7) >> 3, nw = mrmp_key_words(ns);

  /* Pass 1: the meth/ambig bit planes, over the chosen records only. */
  uint64_t n_cpg = 0;
  uint8_t *meth = NULL, *ambig = NULL;
  cfile_t cf = open_cfile((char *)store);
  for (uint32_t k = 0; k < ns; ++k) {
    cdata_t c = read_record_at(&cf, voff[k], label[k]);
    if (!k) {
      n_cpg = c.n;
      meth  = xcalloc((size_t)stride * n_cpg, 1, "meth plane");
      ambig = xcalloc((size_t)stride * n_cpg, 1, "ambig plane");
    } else if (c.n != n_cpg) {
      die("store records disagree on CpG count", label[k]);
    }
    const uint64_t base = (uint64_t)(k >> 3) * n_cpg;
    const uint8_t bit = (uint8_t)(1u << (k & 7));
    for (uint64_t i = 0; i < n_cpg; ++i) {
      uint64_t mu = f3_get_mu(&c, i);
      if (!mu || MU2cov(mu) < mincov) { ambig[base + i] |= bit; continue; }
      double beta = MU2beta(mu);
      if (beta > beta_thr) meth[base + i] |= bit;
      else if (beta == beta_thr) ambig[base + i] |= bit;   /* M==U is not a call */
    }
    free_cdata(&c);
  }
  bgzf_close(cf.fh);

  /* Pass 2: resolve, intern, and remember each CpG's pattern. Kept rather than
   * re-derived (as mrmp-build does) because selection rewrites it twice and a
   * third resolve pass would cost more than the 4 bytes per CpG. */
  uint64_t *pna_key = xcalloc(nw, sizeof(uint64_t), "pna key");
  for (uint32_t s = 0; s < ns; ++s)
    pna_key[s / MRMP_TRITS_PER_WORD] = pna_key[s / MRMP_TRITS_PER_WORD] * 3 + 2;
  phash_t h; phash_init(&h, 1u << 12, nw);
  uint64_t pat_cap = 1u << 12, n_pat = 0;
  uint64_t *pkeys = xcalloc(pat_cap * nw, sizeof(uint64_t), "pattern keys");
  uint64_t *pcount = xcalloc(pat_cap, sizeof(uint64_t), "pattern counts");
  uint32_t *pidx = xcalloc(n_cpg, sizeof(uint32_t), "cpg -> pattern");
  uint64_t *key = xcalloc(nw, sizeof(uint64_t), "cpg key");
  uint64_t checksum = 1469598103934665603ULL;       /* FNV-1a offset */
  for (uint64_t i = 0; i < n_cpg; ++i) {
    int is_pna;
    resolve_cpg(meth, ambig, i, ns, stride, n_cpg,
                min_fold, max_ambig, pna_key, &is_pna, key);
    if (is_pna) {
      pidx[i] = MRMP_PNA_MEMBERSHIP;
    } else {
      if (n_pat == pat_cap) {
        pat_cap <<= 1;
        pkeys = realloc(pkeys, pat_cap * nw * sizeof(*pkeys));
        pcount = realloc(pcount, pat_cap * sizeof(*pcount));
        if (!pkeys || !pcount) die("out of memory", "pattern table grow");
      }
      uint32_t p = phash_intern(&h, key, pkeys, pcount, &n_pat);
      ++pcount[p]; pidx[i] = p;
    }
    for (uint32_t w = 0; w < nw; ++w)
      checksum = (checksum ^ key[w]) * 1099511628211ULL;
  }
  free(meth); free(ambig); free(h.keys); free(h.slot); free(key);

  /* Provisional ranking, needed only so ms_mrmp_select can address a binstring
   * by rank; the ranking that ships is recomputed below on the kept counts. */
  uint32_t *order = xcalloc(n_pat ? n_pat : 1, sizeof(uint32_t), "rank order");
  for (uint64_t p = 0; p < n_pat; ++p) order[p] = (uint32_t)p;
  g_keys = pkeys; g_count = pcount; g_nw = nw;
  qsort(order, n_pat, sizeof(uint32_t), rank_cmp);
  uint32_t *rank_of = xcalloc(n_pat ? n_pat : 1, sizeof(uint32_t), "rank_of");
  for (uint64_t r = 0; r < n_pat; ++r) rank_of[order[r]] = (uint32_t)r;

  uint32_t *memb = xcalloc(n_cpg, sizeof(uint32_t), "membership");
  for (uint64_t i = 0; i < n_cpg; ++i)
    memb[i] = pidx[i] == MRMP_PNA_MEMBERSHIP ? MRMP_PNA_MEMBERSHIP
                                             : rank_of[pidx[i]];
  char **binstr = xcalloc(n_pat ? n_pat : 1, sizeof(char *), "binstrings");
  for (uint64_t r = 0; r < n_pat; ++r) {
    binstr[r] = xcalloc((size_t)ns + 1, 1, "binstring");
    key_to_string(pkeys + (uint64_t)order[r] * nw, ns, binstr[r]);
  }

  /* n_kept is not taken here: the count that matters is the one AFTER empty
   * patterns are dropped, which the recount below produces anyway. */
  uint8_t *keep = ms_mrmp_select(store, ns, mincov, n_cpg, memb, n_pat,
                                 (const char *const *)binstr, sel, NULL, voff);
  for (uint64_t r = 0; r < n_pat; ++r) free(binstr[r]);
  free(binstr); free(memb); free(order);

  /* Recount over the survivors. A homogeneous binstring passes NEITHER leg --
   * one of its two groups is empty, so there is no contrast to threshold -- so
   * in a 2-class set all-0 and all-1, which between them carry nearly the whole
   * genome, come out with zero CpGs. Dropping empty patterns is what keeps the
   * artifact's pattern list the patterns it can actually score. */
  uint64_t *ncount = xcalloc(n_pat ? n_pat : 1, sizeof(uint64_t), "kept counts");
  for (uint64_t i = 0; i < n_cpg; ++i) {
    if (pidx[i] == MRMP_PNA_MEMBERSHIP) continue;
    if (keep && !keep[i]) { pidx[i] = MRMP_PNA_MEMBERSHIP; continue; }
    ++ncount[pidx[i]];
  }
  free(keep);

  uint64_t n_cand = 0;
  uint32_t *ord2 = xcalloc(n_pat ? n_pat : 1, sizeof(uint32_t), "rank order");
  for (uint64_t p = 0; p < n_pat; ++p)
    if (ncount[p]) ord2[n_cand++] = (uint32_t)p;
  if (!n_cand) die("no CpG survived selection for this set", label[0]);
  g_keys = pkeys; g_count = ncount; g_nw = nw;
  qsort(ord2, n_cand, sizeof(uint32_t), rank_cmp);
  for (uint64_t p = 0; p < n_pat; ++p) rank_of[p] = MRMP_PNA_MEMBERSHIP;
  for (uint64_t r = 0; r < n_cand; ++r) rank_of[ord2[r]] = (uint32_t)r;

  uint32_t *memb2 = xcalloc(n_cpg, sizeof(uint32_t), "membership");
  uint64_t pna_cpg = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) {
    memb2[i] = pidx[i] == MRMP_PNA_MEMBERSHIP ? MRMP_PNA_MEMBERSHIP
                                              : rank_of[pidx[i]];
    if (memb2[i] == MRMP_PNA_MEMBERSHIP) ++pna_cpg;
  }
  free(pidx); free(rank_of); free(pcount);

  /* Per-pattern binarisation midpoints (MRMP_FLAG_THRESH in mrmp.h). This is
   * where a satellite most needs them: two close relatives can put a whole
   * pattern above 0.5, where the fixed cut carries no information at all. */
  double *sum = xcalloc(n_cand * ns, sizeof(double), "threshold sums");
  uint64_t *cnt = xcalloc(n_cand * ns, sizeof(uint64_t), "threshold counts");
  cf = open_cfile((char *)store);
  for (uint32_t k = 0; k < ns; ++k) {
    cdata_t c = read_record_at(&cf, voff[k], label[k]);
    for (uint64_t i = 0; i < n_cpg; ++i) {
      uint32_t r = memb2[i];
      if (r == MRMP_PNA_MEMBERSHIP) continue;
      uint64_t mu = f3_get_mu(&c, i);
      if (!mu || MU2cov(mu) < mincov) continue;
      sum[(uint64_t)r * ns + k] += MU2beta(mu);
      cnt[(uint64_t)r * ns + k] += 1;
    }
    free_cdata(&c);
  }
  bgzf_close(cf.fh);

  float *thr = xcalloc(n_cand, sizeof(float), "thresholds");
  char *bs = xcalloc((size_t)ns + 1, 1, "binstring");
  for (uint64_t r = 0; r < n_cand; ++r) {
    key_to_string(pkeys + (uint64_t)ord2[r] * nw, ns, bs);
    double s1 = 0, s0 = 0; uint32_t n1 = 0, n0 = 0;
    for (uint32_t k = 0; k < ns; ++k) {
      uint64_t c2 = cnt[(uint64_t)r * ns + k];
      if (!c2) continue;
      double m = sum[(uint64_t)r * ns + k] / (double)c2;
      if (bs[k] == '1') { s1 += m; ++n1; }
      else if (bs[k] == '0') { s0 += m; ++n0; }
    }
    if (!n1 || !n0) { thr[r] = (float)(0.0 / 0.0); continue; }
    double hi = s1 / n1, lo = s0 / n0;
    thr[r] = (hi > lo) ? (float)(0.5 * (hi + lo)) : (float)(0.0 / 0.0);
  }
  free(sum); free(cnt); free(bs);

  /* Serialize into memory, in header order, exactly as mrmp-build lays a file
   * out -- a block in a chain is a byte-identical standalone MRMPIDX1. */
  mrmp_header_t hd; memset(&hd, 0, sizeof(hd));
  memcpy(hd.magic, MRMPIDX_MAGIC, 8);
  hd.version = MRMPIDX_VERSION; hd.n_samples = ns;
  hd.n_selected = (uint32_t)n_cand;
  /* the candidate policy is mrmp-build's, homogeneous included; those patterns
   * simply end with no CpGs and are dropped by the empty-pattern rule above */
  hd.flags = MRMP_FLAG_INCLUDE_HOMOGENEOUS | MRMP_FLAG_THRESH;
  hd.n_cpg = n_cpg; hd.n_candidates = n_cand;
  hd.pna_key = pna_key[0]; hd.pna_cpg = pna_cpg;
  hd.mincov = mincov; hd.beta_threshold = beta_thr;
  hd.max_ambig_frac = max_ambig; hd.min_major_fold = min_fold;
  hd.content_checksum = checksum;

  /* This is where compression earns its keep: a 2-class satellite describes a
   * few thousand CpGs and the dense array would still be one uint32 for every
   * one of the ~21.8 M in the genome. */
  uint64_t memb_n = 0;
  uint8_t *memb_rle = memb_compress(memb2, n_cpg, n_cand, &memb_n);
  if (memb_n > UINT32_MAX) die("compressed membership exceeds 4 GB", label[0]);
  hd.membership_bytes = (uint32_t)memb_n;
  hd.flags |= MRMP_FLAG_MEMB_RLE | MRMP_FLAG_MEMB_BGZF;

  uint64_t off = sizeof(hd);
  hd.refname_offset = off;    off += strlen(store) + 1;
  hd.name_offset = (uint32_t)off; off += strlen(set_name) + 1;
  hd.names_offset = off;      for (uint32_t s = 0; s < ns; ++s) off += strlen(label[s]) + 1;
  hd.patterns_offset = off;   off += n_cand * mrmp_pattern_stride(ns);
  hd.membership_offset = off; off += memb_n;
  hd.thresh_offset = off;     off += n_cand * sizeof(float);
  /* Pad to 8 so the NEXT block's header is aligned when this one is
   * concatenated after it -- readers cast the header in place. */
  const uint64_t img_bytes = (off + 7u) & ~7ull;

  char *img = xcalloc(img_bytes, 1, "mrmp block");
  uint64_t at = 0;
  img_put(img, &at, &hd, sizeof(hd));
  img_put(img, &at, store, strlen(store) + 1);
  img_put(img, &at, set_name, strlen(set_name) + 1);
  for (uint32_t s = 0; s < ns; ++s)
    img_put(img, &at, label[s], strlen(label[s]) + 1);
  for (uint64_t r = 0; r < n_cand; ++r) {
    img_put(img, &at, pkeys + (uint64_t)ord2[r] * nw, nw * sizeof(uint64_t));
    img_put(img, &at, &ncount[ord2[r]], sizeof(uint64_t));
  }
  img_put(img, &at, memb_rle, (size_t)memb_n);
  img_put(img, &at, thr, (size_t)n_cand * sizeof(float));
  at = img_bytes;                        /* the pad is part of the block */
  free(memb_rle);

  free(pkeys); free(ncount); free(ord2); free(memb2); free(thr); free(pna_key);
  out->img = img; out->bytes = at;
  out->n_pat = n_cand; out->n_kept = n_cpg - pna_cpg;
}

/* Length of a record name minus the `-R<digits>` replicate suffix the sampled
 * featurizer appends (`ASC-R1`). Matching on that key rather than on record
 * POSITION is the one place this differs from the Python driver, which indexed
 * the matrix by the class's line number in classes.txt -- correct only while
 * the .msfm rows and the store records happen to be in the same order. */
static size_t class_key_len(const char *name) {
  size_t n = strlen(name), i = n;
  while (i > 0 && name[i - 1] >= '0' && name[i - 1] <= '9') --i;
  if (i < n && i >= 2 && name[i - 1] == 'R' && name[i - 2] == '-') return i - 2;
  return n;
}

static uint32_t msfm_row_of(char *const *rec_names, uint32_t n_rec,
                            const char *cls) {
  size_t n = strlen(cls);
  for (uint32_t r = 0; r < n_rec; ++r) {
    const char *nm = rec_names[r];
    size_t k = class_key_len(nm);
    if (k == n && !memcmp(nm, cls, n)) return r;   /* first replicate wins */
  }
  return UINT32_MAX;
}

/* Projection distance: mean |v - w| over the pattern-average vectors, on the
 * columns both classes actually have. The u16 codes are differenced directly
 * and scaled once at the end, which is the same number as differencing betas. */
static double proj_dist(const uint16_t *beta, uint32_t stride,
                        uint32_t ra, uint32_t rb,
                        uint32_t K, uint32_t *n_shared) {
  const uint16_t *va = beta + (uint64_t)ra * stride;
  const uint16_t *vb = beta + (uint64_t)rb * stride;
  double s = 0; uint32_t m = 0;
  for (uint32_t j = 0; j < K; ++j) {
    if (va[j] == MSFM_NA || vb[j] == MSFM_NA) continue;
    s += fabs((double)va[j] - (double)vb[j]);
    ++m;
  }
  *n_shared = m;
  return m ? s / (double)m / 65534.0 : 1.0 / 0.0;
}

typedef struct { double d; const char *name; uint32_t k; } cand_t;
static int cand_cmp(const void *a, const void *b) {
  const cand_t *x = a, *y = b;
  if (x->d < y->d) return -1;
  if (x->d > y->d) return 1;
  return strcmp(x->name, y->name);          /* ties resolve by name, not by luck */
}

/* `<generator>_<a>_<b>` -- e.g. thin_ANP_ASC, neighbor_IT-L5_IT-L6.
 *
 * The generator is spelled out because the name is the only place it survives:
 * a set is a set, and nothing downstream distinguishes them, so "which
 * generator proposed this" is readable in a pooled `inspect` table or nowhere.
 * It was `th_`/`nb_` to match the Python driver's names; that mattered only
 * while runs were being compared across the two implementations.
 *
 * Class labels go in VERBATIM rather than lowercased with '-' stripped. `IT-L5`
 * is what the label actually is, and `itl5` was a second decoding step for a
 * reader. None of the vocabulary contains '_', so the separator stays
 * unambiguous. The name also becomes a .cm record name and the `Pna.<set>`
 * column name, so it avoids '.' (already the Pna separator) and ':'. */
static char *pair_set_name(const char *prefix, const char *a, const char *b) {
  size_t n = strlen(prefix) + strlen(a) + strlen(b) + 2;
  char *s = xcalloc(n, 1, "set name");
  snprintf(s, n, "%s%s_%s", prefix, a, b);
  return s;
}

int main_mrmp_build_thin(int argc, char *argv[]) {
  g_cmd = "mrmp-build-thin";
  const char *pos[3] = {NULL, NULL, NULL};
  int npos = 0, force = 0;
  uint32_t n_partner = 3, proj_top = 6000;
  ms_select_opt_t sel; ms_select_defaults(&sel);
  sel.depth_floor_frac = 1.0f;   /* satellites turn the RELATIVE floor on */
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stderr,
        "Usage: methscope mrmp-build-thin [options] STORE.cg GLOBAL.mrmp OUT.mrmp\n\n"
        "  One 2-class satellite per (thin class, partner) pair, written as one\n"
        "  chain. Thin classes are derived as (store labels -\n"
        "  GLOBAL.mrmp labels). A class is thin because it LACKS THE CELLS to\n"
        "  define a pattern, not because it resembles anything -- no distance\n"
        "  metric finds ANP/ASC (196th of 820 by weighted Hamming, correctly,\n"
        "  since their confident evidence really does separate), yet 52% of\n"
        "  held-out ANP cells were called ASC. This generator and\n"
        "  mrmp-build-neighbor therefore find different things and neither\n"
        "  subsumes the other.\n\n"
        "  Binstring resolution (mincov, beta threshold, ambiguity, majority\n"
        "  fold) is read from GLOBAL.mrmp's header rather than re-declared here:\n"
        "  a satellite resolved on a different rule than the global it\n"
        "  supplements would put two incompatible pattern definitions in one\n"
        "  pooled feature vector.\n\n"
        "  The partner search needs the reference featurized against\n"
        "  GLOBAL.mrmp, and does that itself. It used to be a required\n"
        "  --ref-msfm argument; nothing checked that the matrix given had been\n"
        "  built against the global given beside it, and a mismatched pair chose\n"
        "  partners in the wrong feature space silently. None of the\n"
        "  featurizer's options were free choices here either -- it must be this\n"
        "  reference, this global, every pattern, native coverage -- so there was\n"
        "  nothing for a caller to decide.\n\n"
        "  --n-partner N             (default 3)\n"
        "        Nearest classes each thin class is given a satellite against,\n"
        "        found by projection of reference pattern-average vectors. Each\n"
        "        partner gets its OWN 2-class satellite; they are NOT combined\n"
        "        into one N-class set. A joint binstring forces the thin class\n"
        "        to oppose every neighbour at once, and that joint requirement\n"
        "        is where its noise concentrates: held-out separation 0.228 for\n"
        "        a 3-class set against 0.317 and 0.487 for the two pairs it\n"
        "        replaced. Partner candidates include other THIN classes, not\n"
        "        only deep ones -- PC loses more held-out cells to VLMC-Pia\n"
        "        (17.1%) than to its deep partner VLMC (15.9%). A thin-thin pair\n"
        "        is reachable from both ends and is emitted once.\n\n"
        "  --projection-top N        (default 6000)\n"
        "        Pattern columns of --ref-msfm the distance is taken over. The\n"
        "        featurizer's trailing PNA column is always excluded: it is the\n"
        "        unpatterned background, so it carries no contrast between two\n"
        "        classes and only dilutes the mean.\n\n"
        "  --qfilter LO,HI           (default 0.25,0.6)\n"
        "        As mrmp-build. Forms the FLOOR leg of the selection rule.\n\n"
        "  --qfilter-strict LO,HI    (default 0.1,0.8)\n"
        "        Tighter gate forming the SELF-SIZING leg: every CpG passing it\n"
        "        is kept regardless of budget, so a pair with genuinely clean\n"
        "        positions contributes all of them. Its yield is steeply\n"
        "        depth-dependent by design -- the same pair gave 6,453 CpGs at\n"
        "        reference depth 7.9 and 204 at 4.54 -- which is exactly why it\n"
        "        cannot be used alone.\n\n"
        "  --delta-mean-top N        (default 1000)\n"
        "        PER BINSTRING, keep the N highest by delta_mean among --qfilter\n"
        "        passers. The floor leg: it guarantees a pattern is never\n"
        "        starved when --qfilter-strict returns almost nothing.\n"
        "        delta_mean (mean gap between the expected-1 and expected-0\n"
        "        groups) rather than delta_beta (worst-case margin) because\n"
        "        --qfilter already bounds the worst case; the two scored within\n"
        "        0.007 of each other on held-out margin. Identical for 2\n"
        "        classes, diverging only at 3+.\n\n"
        "  --depth-floor-frac F      (default 1.0)\n"
        "        Per class, require coverage of at least F times THAT CLASS'S\n"
        "        OWN genome-wide mean depth. Relative, so one number serves\n"
        "        classes spanning depth 5 to 112. Purely relative is too harsh\n"
        "        when many classes must agree at once, hence the cap below.\n\n"
        "  --depth-floor-cap N       (default 20)\n"
        "        Cap the above at N cells: \"enough to be reliable, never more\n"
        "        than the class can give\". At depth 20 a beta carries SE ~ 0.11,\n"
        "        and a class at 112 gains nothing from being asked for 112.\n\n"
        "  --force                   overwrite an existing output\n\n"
        "  Output is a chain of 2-class sets, which mrmp-pool takes\n"
        "  directly (it expands a chain into its sets). Sets are sets: a\n"
        "  satellite here is the same object mrmp-build writes, and nothing\n"
        "  downstream distinguishes them.\n");
      return 0;
    }
    else if (!strcmp(a, "--n-partner") && i + 1 < argc)
      n_partner = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--projection-top") && i + 1 < argc)
      proj_top = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--delta-mean-top") && i + 1 < argc)
      sel.delta_mean_top = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--depth-floor-frac") && i + 1 < argc)
      sel.depth_floor_frac = (float)atof(argv[++i]);
    else if (!strcmp(a, "--depth-floor-cap") && i + 1 < argc)
      sel.depth_floor_cap = (uint32_t)parse_u64(argv[++i], a);
    else if ((!strcmp(a, "--qfilter") || !strcmp(a, "--qfilter-strict"))
             && i + 1 < argc) {
      int strict = a[9] != '\0';
      const char *v = argv[++i]; char *end = NULL;
      float lo = strtof(v, &end);
      if (!end || *end != ',') die("wants LO,HI", a);
      float hi = strtof(end + 1, NULL);
      if (!(lo >= 0.0f && hi <= 1.0f && lo < hi)) die("needs 0 <= LO < HI <= 1", a);
      if (strict) { sel.strict_lo = lo; sel.strict_hi = hi; }
      else        { sel.qfilter_lo = lo; sel.qfilter_hi = hi; }
    }
    else if (!strcmp(a, "--force")) force = 1;
    else if (a[0] == '-') die("unrecognized or incomplete option", a);
    else if (npos < 3) pos[npos++] = a;
    else die("too many arguments", a);
  }
  if (npos != 3) die("need STORE.cg GLOBAL.mrmp OUT.mrmp (see -h)", NULL);
  const char *store = pos[0], *global = pos[1], *out = pos[2];
  if (!n_partner) die("--n-partner must be at least 1", NULL);
  if (!force) { struct stat st; if (!stat(out, &st)) die("output exists (use --force)", out); }

  uint32_t nstore = 0; int64_t *voff = NULL;
  char **slab = read_store_index(store, &nstore, &voff);

  {  /* the global must be ONE set: its binstring columns define the classes */
    ms_mrmpset_t *gs = ms_mrmpset_open(global);
    uint32_t gn = gs->n_sets;
    ms_mrmpset_free(gs);
    if (gn != 1) die("GLOBAL must hold exactly one set", global);
  }
  mrmp_reader_t gr; mrmp_open(&gr, global);
  const mrmp_header_t gh = *gr.h;
  const uint32_t ngl = gr.h->n_samples;
  char **glab = xcalloc(ngl, sizeof(char *), "global labels");
  for (uint32_t s = 0; s < ngl; ++s) {
    size_t n = strlen(gr.names[s]) + 1;
    glab[s] = xcalloc(n, 1, "global label");
    memcpy(glab[s], gr.names[s], n);
  }
  mrmp_close(&gr);

  /* Thin == in the store but not in the global. The artifact carries its own
   * sample names, so the thin/deep split is recoverable from it and needs no
   * side file and no depth threshold repeated here. */
  uint8_t *is_thin = xcalloc(nstore, 1, "thin flags");
  uint32_t n_thin = 0;
  for (uint32_t s = 0; s < nstore; ++s) {
    int in_g = 0;
    for (uint32_t g = 0; g < ngl && !in_g; ++g) in_g = !strcmp(slab[s], glab[g]);
    if (!in_g) { is_thin[s] = 1; ++n_thin; }
  }
  for (uint32_t g = 0; g < ngl; ++g) {
    int in_s = 0;
    for (uint32_t s = 0; s < nstore && !in_s; ++s) in_s = !strcmp(slab[s], glab[g]);
    if (!in_s)
      fprintf(stderr, "[methscope] %s: warning: global class '%s' is not in the "
              "store -- the two were built from different references\n",
              g_cmd, glab[g]);
  }
  if (!n_thin)
    die("every store class is already in the global; no thin class to cover", global);

  /* Featurize the reference against THIS global, here, rather than taking a
   * .msfm as an argument.
   *
   * It used to be required, on the grounds that featurizing is a command in its
   * own right and one matrix could serve several callers. There are no several
   * callers -- mrmp-build-thin was the only consumer -- and nothing verified
   * that the matrix passed had been built against the global passed beside it.
   * A mismatched pair computed the projection in the wrong feature space and
   * chose the wrong partners, silently. Building it here makes that
   * unrepresentable rather than merely documented.
   *
   * None of the featurizer's knobs are free choices either: it must be this
   * reference against this global, every pattern, native coverage, unbinarised.
   * So there was nothing for a caller to decide. */
  char tpl[] = "/tmp/methscope_thinref_XXXXXX.cm";
  int tfd = mkstemps(tpl, 3);
  if (tfd < 0) die("cannot create a temporary mask", tpl);
  close(tfd);
  ms_mrmp_write_mask(global, tpl, "Pna", UINT32_MAX);
  uint16_t *beta = NULL; uint32_t *levels = NULL; char **rec = NULL;
  uint32_t n_rec = 0, ncol = 0, native = 0;
  ms_msfm_build_sampled(store, tpl, 0, &native, 1, 0, 0, 1, 1,
                        &beta, &levels, &rec, &n_rec, &ncol);
  unlink(tpl);
  free(levels);
  /* last column is the PNA background: no contrast, so it only dilutes a mean */
  uint32_t np = ncol ? ncol - 1 : 0;
  const uint32_t K = proj_top < np ? proj_top : np;
  uint32_t *row = xcalloc(nstore, sizeof(uint32_t), "msfm rows");
  for (uint32_t s = 0; s < nstore; ++s) {
    row[s] = msfm_row_of(rec, n_rec, slab[s]);
    if (row[s] == UINT32_MAX)
      die("the reference has no record for store class", slab[s]);
  }

  /* Partner search, then dedup: a thin-thin pair is reachable from both ends. */
  uint32_t cap = n_thin * n_partner, npair = 0;
  uint32_t *pa = xcalloc(cap, sizeof(uint32_t), "pair a");
  uint32_t *pb = xcalloc(cap, sizeof(uint32_t), "pair b");
  double   *pd = xcalloc(cap, sizeof(double), "pair distance");
  cand_t *cand = xcalloc(nstore, sizeof(cand_t), "partner candidates");
  for (uint32_t s = 0; s < nstore; ++s) {
    if (!is_thin[s]) continue;
    uint32_t nc = 0;
    for (uint32_t c = 0; c < nstore; ++c) {
      if (c == s) continue;
      uint32_t m; double d = proj_dist(beta, ncol, row[s], row[c], K, &m);
      if (m <= 50) continue;    /* too few shared columns for a mean to mean much */
      cand[nc].d = d; cand[nc].name = slab[c]; cand[nc].k = c; ++nc;
    }
    qsort(cand, nc, sizeof(cand_t), cand_cmp);
    uint32_t take = n_partner < nc ? n_partner : nc;
    if (!take)
      fprintf(stderr, "[methscope] %s: warning: no partner for thin class '%s'\n",
              g_cmd, slab[s]);
    for (uint32_t j = 0; j < take; ++j) {
      uint32_t b = cand[j].k, dup = 0;
      for (uint32_t q = 0; q < npair && !dup; ++q)
        dup = (pa[q] == s && pb[q] == b) || (pa[q] == b && pb[q] == s);
      if (dup) continue;
      pa[npair] = s; pb[npair] = b; pd[npair] = cand[j].d; ++npair;
    }
  }
  free(cand); free(row); free(beta);
  for (uint32_t r = 0; r < n_rec; ++r) free(rec[r]);
  free(rec);
  if (!npair) die("no (thin, partner) pair survived the projection", global);

  fprintf(stderr, "[methscope] %s: %u thin of %u store classes, %u satellites\n",
          g_cmd, n_thin, nstore, npair);

  char **name = xcalloc(npair, sizeof(char *), "set names");
  void **blk  = xcalloc(npair, sizeof(void *), "blocks");
  uint64_t *len = xcalloc(npair, sizeof(uint64_t), "block sizes");
  for (uint32_t q = 0; q < npair; ++q) {
    /* members in STORE order, so a set's digit order is the store's */
    uint32_t k0 = pa[q] < pb[q] ? pa[q] : pb[q];
    uint32_t k1 = pa[q] < pb[q] ? pb[q] : pa[q];
    char *lab[2] = {slab[k0], slab[k1]};
    int64_t vo[2] = {voff[k0], voff[k1]};
    name[q] = pair_set_name("thin_", slab[pa[q]], slab[pb[q]]);
    fprintf(stderr, "  [%u/%u] %-24s %s + %s  projection %.5f\n",
            q + 1, npair, name[q], slab[pa[q]], slab[pb[q]], pd[q]);
    subset_block_t sb;
    build_subset_block(store, 2, lab, vo, &gh, &sel, name[q], &sb);
    blk[q] = sb.img; len[q] = sb.bytes;
    fprintf(stderr, "      %" PRIu64 " patterns over %" PRIu64 " CpGs\n",
            sb.n_pat, sb.n_kept);
  }
  ms_mrmp_chain_write(out, npair, (const void *const *)blk, len);
  fprintf(stderr, "[methscope] %s: %u sets -> %s\n", g_cmd, npair, out);

  for (uint32_t q = 0; q < npair; ++q) { free(blk[q]); free(name[q]); }
  free(name); free(blk); free(len); free(pa); free(pb); free(pd);
  for (uint32_t s = 0; s < nstore; ++s) free(slab[s]);
  for (uint32_t g = 0; g < ngl; ++g) free(glab[g]);
  free(slab); free(glab); free(voff); free(is_thin);
  return 0;
}

/* ---------------- mrmp-build-neighbor ----------------------------------- */

/* Satellites for classes that GENUINELY RESEMBLE each other, so their evidence
 * overlaps and a distance metric finds them. Complementary to mrmp-build-thin,
 * which addresses a measurement artifact rather than a real similarity, and
 * neither subsumes the other.
 *
 * Deliberately NOT a port of the WPGMA partition this replaces. WPGMA
 * PARTITIONS -- each class lands in at most one block, and the 3-class cap then
 * truncates -- and that is where three quarters of the remaining error was
 * living: measured on the 10-fold arm, 73.5% of it sat in confusion pairs no
 * satellite covered, because IT-L4 and MGE-Pvalb fell out of the vocabulary
 * entirely and PAL-Inh <-> LSX-Inh was split across two blocks. Overlapping
 * closest-N pairs fix that by construction: a class appears in as many sets as
 * it has close neighbours, and every pair that matters is reachable.
 *
 * Pairs rather than blocks is also the better trade per column. Pattern count
 * grows as 2^N-2 while covered error does not -- a 2-class set spent 2 pooled
 * columns for 7.4% of error where a 6-class one spent 62 for 4.6%. */

/* Distance between two class columns of the global: the CpG-weighted share of
 * DISAGREEMENTS among the CpGs where at least one of the two is called 0. A
 * small distance is a predicted confusion, since this is the evidence the
 * classifier scores with.
 *
 * Weighted by CpG count rather than counting patterns, because a pattern
 * carrying 10,000 CpGs and one carrying 3 are not equal evidence -- the global's
 * counts are violently skewed (median 17).
 *
 * The denominator is the point, and plain Hamming (dividing by ALL weight) is
 * wrong here. Every class is called 1 on 81-93% of weighted CpGs, so the
 * disagreement weight collapses to roughly (1 - frac1(a)) + (1 - frac1(b)) and
 * the nearest neighbour of EVERY class comes out as whichever class is called 1
 * most often -- PAL-Inh for 19 of 34 classes, with ASC-PAL-Inh ranked nearest of
 * all. That is the binstring-centroid artifact that got the A-score partner gate
 * dropped, reproduced exactly. Normalising by the union of the ZERO calls
 * removes the marginal-frequency term, because a class that is almost never 0
 * lands far from everything instead of close to it.
 *
 * Measured against the 10-fold confusion matrix, over the pairs each metric
 * proposes at --n-partner 3: same share of deep-deep error covered (68.0% vs
 * 67.4%) from 81 sets rather than 85, and a less degenerate partner list. A
 * disagreement implies at least one 0, so num <= den and the result is in [0,1]. */
static double column_dist(const mrmp_top_t *t, uint32_t a, uint32_t b) {
  double num = 0, den = 0;
  for (uint32_t p = 0; p < t->n_patterns; ++p) {
    const char ca = t->binstring[p][a], cb = t->binstring[p][b];
    const double w = (double)t->count[p];
    if (ca != cb) num += w;
    if (ca == '0' || cb == '0') den += w;
  }
  return den > 0 ? num / den : 1.0 / 0.0;
}

int main_mrmp_build_neighbor(int argc, char *argv[]) {
  g_cmd = "mrmp-build-neighbor";
  const char *pos[3] = {NULL, NULL, NULL};
  int npos = 0, force = 0, dry_run = 0;
  uint32_t n_partner = 3, dist_top = 6000;
  double max_dist = 0.0;                  /* 0 == no distance gate; see help */
  ms_select_opt_t sel; ms_select_defaults(&sel);
  sel.depth_floor_frac = 1.0f;   /* satellites turn the RELATIVE floor on */
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stderr,
        "Usage: methscope mrmp-build-neighbor [options] STORE.cg GLOBAL.mrmp OUT.mrmp\n\n"
        "  One 2-class satellite per (class, near neighbour) pair over the\n"
        "  classes GLOBAL.mrmp already covers, written as one chain.\n"
        "   Complementary to mrmp-build-thin: this one finds classes\n"
        "  that GENUINELY RESEMBLE each other, where thin covers classes too\n"
        "  shallow to define a pattern at all. Thin classes are absent from the\n"
        "  global by construction, so the two generators cannot propose the same\n"
        "  pair and neither subsumes the other.\n\n"
        "  Needs no reference featurization: the binstring columns and their CpG\n"
        "  counts are already in GLOBAL.mrmp, so the distance is a pure artifact\n"
        "  computation. The store is read only to BUILD the chosen pairs.\n\n"
        "  OVERLAPPING PAIRS, NOT A PARTITION. The WPGMA generator this replaces\n"
        "  put each class in at most one block and then truncated at 3, which is\n"
        "  where 73.5% of all remaining error was living -- IT-L4 and MGE-Pvalb\n"
        "  fell out of the vocabulary entirely and PAL-Inh <-> LSX-Inh was split\n"
        "  across two blocks. A class here appears in as many sets as it has\n"
        "  close neighbours. Pairs also buy more per pooled column than blocks:\n"
        "  pattern count grows as 2^N-2 while covered error does not.\n\n"
        "  --n-partner N             (default 3)\n"
        "        Nearest classes each class is given a satellite against. A pair\n"
        "        reachable from both ends is emitted ONCE, so the set count is\n"
        "        well below N x classes. Raising it buys coverage at a rising\n"
        "        price in pooled columns -- measured on a 34-class global against\n"
        "        the share of deep-deep confusion error the proposed pairs cover:\n"
        "          N=1  28 sets 37.8%   N=2  55 sets 59.6%   N=3  81 sets 68.0%\n"
        "          N=4 105 sets 76.2%   N=5 130 sets 79.9%\n"
        "        against an 80.9% ceiling for the 85 worst pairs chosen with the\n"
        "        confusion matrix in hand. Each 2-class set spends ~2 of\n"
        "        mrmp-pool's columns, so N=5 puts a quarter of a 1,000 budget\n"
        "        here. 3 is the default for the same reason it is in\n"
        "        mrmp-build-thin, not because 3 is special.\n\n"
        "  --max-dist D              (default 0 = no gate)\n"
        "        Skip a partner farther than D, so an isolated class is not given\n"
        "        arbitrary neighbours just to fill its quota. Deliberately OFF by\n"
        "        default: the tempting number is the old --wpgma-height 0.011,\n"
        "        but that was a WPGMA MERGE HEIGHT -- an average over cluster\n"
        "        members -- not a pairwise distance, and it was measured on a\n"
        "        different metric besides. Use --dry-run to see the scale on your\n"
        "        own reference before setting this.\n\n"
        "  --dist-top K              (default 6000)\n"
        "        Pattern columns of GLOBAL.mrmp the distance is taken over,\n"
        "        highest CpG count first. Matches mrmp-build's --top so the\n"
        "        distance is computed over the patterns that actually ship.\n\n"
        "  --qfilter LO,HI           (default 0.25,0.6)\n"
        "        As mrmp-build. Forms the FLOOR leg of the selection rule.\n\n"
        "  --qfilter-strict LO,HI    (default 0.1,0.8)\n"
        "        Tighter gate forming the SELF-SIZING leg: every CpG passing it\n"
        "        is kept regardless of budget, so a pair with genuinely clean\n"
        "        positions contributes all of them.\n\n"
        "  --delta-mean-top N        (default 1000)\n"
        "        PER BINSTRING, keep the N highest by delta_mean among --qfilter\n"
        "        passers. The floor leg: it guarantees a pattern is never starved\n"
        "        when --qfilter-strict returns almost nothing.\n\n"
        "  --depth-floor-frac F      (default 1.0)\n"
        "        Per class, require coverage of at least F times THAT CLASS'S OWN\n"
        "        genome-wide mean depth. Relative, so one number serves classes\n"
        "        spanning depth 5 to 112.\n\n"
        "  --depth-floor-cap N       (default 20)\n"
        "        Cap the above at N cells: \"enough to be reliable, never more\n"
        "        than the class can give\".\n\n"
        "  --dry-run                 print the chosen pairs and their distances,\n"
        "        then stop without building anything. The pairing is seconds of\n"
        "        artifact arithmetic while the build is hours of store passes, so\n"
        "        this is how --max-dist and --n-partner get tuned: the absolute\n"
        "        distance scale is not knowable in advance, and committing to a\n"
        "        full build to discover it is the expensive way to find out.\n\n"
        "  --force                   overwrite an existing output\n\n"
        "  Binstring resolution (mincov, beta threshold, ambiguity, majority\n"
        "  fold) is read from GLOBAL.mrmp's header rather than re-declared here,\n"
        "  for the same reason as mrmp-build-thin: a satellite resolved on a\n"
        "  different rule than the global it supplements would put two\n"
        "  incompatible pattern definitions in one pooled feature vector.\n");
      return 0;
    }
    else if (!strcmp(a, "--n-partner") && i + 1 < argc)
      n_partner = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--dist-top") && i + 1 < argc)
      dist_top = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--max-dist") && i + 1 < argc) max_dist = atof(argv[++i]);
    else if (!strcmp(a, "--delta-mean-top") && i + 1 < argc)
      sel.delta_mean_top = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--depth-floor-frac") && i + 1 < argc)
      sel.depth_floor_frac = (float)atof(argv[++i]);
    else if (!strcmp(a, "--depth-floor-cap") && i + 1 < argc)
      sel.depth_floor_cap = (uint32_t)parse_u64(argv[++i], a);
    else if ((!strcmp(a, "--qfilter") || !strcmp(a, "--qfilter-strict"))
             && i + 1 < argc) {
      int strict = a[9] != '\0';
      const char *v = argv[++i]; char *end = NULL;
      float lo = strtof(v, &end);
      if (!end || *end != ',') die("wants LO,HI", a);
      float hi = strtof(end + 1, NULL);
      if (!(lo >= 0.0f && hi <= 1.0f && lo < hi)) die("needs 0 <= LO < HI <= 1", a);
      if (strict) { sel.strict_lo = lo; sel.strict_hi = hi; }
      else        { sel.qfilter_lo = lo; sel.qfilter_hi = hi; }
    }
    else if (!strcmp(a, "--dry-run")) dry_run = 1;
    else if (!strcmp(a, "--force")) force = 1;
    else if (a[0] == '-') die("unrecognized or incomplete option", a);
    else if (npos < 3) pos[npos++] = a;
    else die("too many arguments", a);
  }
  if (npos != 3) die("need STORE.cg GLOBAL.mrmp OUT.mrmp (see -h)", NULL);
  const char *store = pos[0], *global = pos[1], *out = pos[2];
  if (!n_partner) die("--n-partner must be at least 1", NULL);
  if (!force && !dry_run) { struct stat st; if (!stat(out, &st)) die("output exists (use --force)", out); }

  uint32_t nstore = 0; int64_t *voff = NULL;
  char **slab = read_store_index(store, &nstore, &voff);

  {  /* the global must be ONE set: its binstring columns define the classes */
    ms_mrmpset_t *gs = ms_mrmpset_open(global);
    uint32_t gn = gs->n_sets;
    ms_mrmpset_free(gs);
    if (gn != 1) die("GLOBAL must hold exactly one set", global);
  }
  mrmp_reader_t gr; mrmp_open(&gr, global);
  const mrmp_header_t gh = *gr.h;
  mrmp_close(&gr);

  /* The binstring columns ARE the input: labels, per-pattern calls and CpG
   * counts all come out of the global, so no store pass and no .msfm. */
  mrmp_top_t *t = ms_mrmp_top_read(global, dist_top);
  const uint32_t ngl = t->n_samples;
  if (ngl < 2) die("global has fewer than two classes; no pair to build", global);

  /* Each global class must be findable in the store, since that is where the
   * pair is actually built from. */
  uint32_t *srow = xcalloc(ngl, sizeof(uint32_t), "store rows");
  for (uint32_t g = 0; g < ngl; ++g) {
    uint32_t s = 0;
    for (; s < nstore && strcmp(slab[s], t->labels[g]); ++s) {}
    if (s == nstore)
      die("global class is not in the store -- the two were built from "
          "different references", t->labels[g]);
    srow[g] = s;
  }

  /* Closest-N per class, then dedup: a pair is reachable from both ends. */
  uint32_t cap = ngl * n_partner, npair = 0;
  uint32_t *pa = xcalloc(cap, sizeof(uint32_t), "pair a");
  uint32_t *pb = xcalloc(cap, sizeof(uint32_t), "pair b");
  double   *pd = xcalloc(cap, sizeof(double), "pair distance");
  cand_t *cand = xcalloc(ngl, sizeof(cand_t), "partner candidates");
  for (uint32_t a = 0; a < ngl; ++a) {
    uint32_t nc = 0;
    for (uint32_t b = 0; b < ngl; ++b) {
      if (b == a) continue;
      double d = column_dist(t, a, b);
      if (max_dist > 0.0 && d > max_dist) continue;
      cand[nc].d = d; cand[nc].name = t->labels[b]; cand[nc].k = b; ++nc;
    }
    qsort(cand, nc, sizeof(cand_t), cand_cmp);
    uint32_t take = n_partner < nc ? n_partner : nc;
    if (!take)
      fprintf(stderr, "[methscope] %s: warning: no partner within --max-dist "
              "for '%s'\n", g_cmd, t->labels[a]);
    for (uint32_t j = 0; j < take; ++j) {
      uint32_t b = cand[j].k, dup = 0;
      for (uint32_t q = 0; q < npair && !dup; ++q)
        dup = (pa[q] == a && pb[q] == b) || (pa[q] == b && pb[q] == a);
      if (dup) continue;
      pa[npair] = a; pb[npair] = b; pd[npair] = cand[j].d; ++npair;
    }
  }
  free(cand);
  if (!npair) die("no (class, neighbour) pair survived --max-dist", global);

  fprintf(stderr, "[methscope] %s: %u classes, %u satellites\n",
          g_cmd, ngl, npair);

  if (dry_run) {
    printf("#set\tclass_a\tclass_b\tdistance\n");
    for (uint32_t q = 0; q < npair; ++q) {
      char *nm = pair_set_name("neighbor_", t->labels[pa[q]], t->labels[pb[q]]);
      printf("%s\t%s\t%s\t%.6f\n", nm, t->labels[pa[q]], t->labels[pb[q]], pd[q]);
      free(nm);
    }
    free(pa); free(pb); free(pd);
    for (uint32_t s = 0; s < nstore; ++s) free(slab[s]);
    free(slab); free(voff); free(srow);
    ms_mrmp_top_free(t);
    return 0;
  }

  char **name = xcalloc(npair, sizeof(char *), "set names");
  void **blk  = xcalloc(npair, sizeof(void *), "blocks");
  uint64_t *len = xcalloc(npair, sizeof(uint64_t), "block sizes");
  for (uint32_t q = 0; q < npair; ++q) {
    /* members in STORE order, so a set's digit order is the store's */
    uint32_t s0 = srow[pa[q]], s1 = srow[pb[q]];
    uint32_t k0 = s0 < s1 ? s0 : s1, k1 = s0 < s1 ? s1 : s0;
    char *lab[2] = {slab[k0], slab[k1]};
    int64_t vo[2] = {voff[k0], voff[k1]};
    name[q] = pair_set_name("neighbor_", t->labels[pa[q]], t->labels[pb[q]]);
    fprintf(stderr, "  [%u/%u] %-24s %s + %s  distance %.5f\n",
            q + 1, npair, name[q], t->labels[pa[q]], t->labels[pb[q]], pd[q]);
    subset_block_t sb;
    build_subset_block(store, 2, lab, vo, &gh, &sel, name[q], &sb);
    blk[q] = sb.img; len[q] = sb.bytes;
    fprintf(stderr, "      %" PRIu64 " patterns over %" PRIu64 " CpGs\n",
            sb.n_pat, sb.n_kept);
  }
  ms_mrmp_chain_write(out, npair, (const void *const *)blk, len);
  fprintf(stderr, "[methscope] %s: %u sets -> %s\n", g_cmd, npair, out);

  for (uint32_t q = 0; q < npair; ++q) { free(blk[q]); free(name[q]); }
  free(name); free(blk); free(len); free(pa); free(pb); free(pd);
  for (uint32_t s = 0; s < nstore; ++s) free(slab[s]);
  free(slab); free(voff); free(srow);
  ms_mrmp_top_free(t);
  return 0;
}

/* Thousands separators. inspect.c has its own; this one keeps mrmp.c's reports
 * self-contained rather than exporting that one for two call sites. */
static const char *commafmt_local(uint64_t v, char *buf) {
  char tmp[24]; int n = snprintf(tmp, sizeof tmp, "%" PRIu64, v);
  int commas = (n - 1) / 3, len = n + commas;
  buf[len] = '\0';
  int bi = len - 1, oi = n - 1, cnt = 0;
  while (oi >= 0) { buf[bi--] = tmp[oi--]; if (++cnt % 3 == 0 && oi >= 0) buf[bi--] = ','; }
  return buf;
}

/* ---------------- inspect: the multi-set arm ----------------------------- */

/* Per-set dimensions, then the POOLED view -- which sets would actually
 * contribute at a given rank cut. That second table is the one that matters in
 * practice: a satellite holding 2-30 patterns is unaffected by a per-set
 * `--top 1000`, so the only cut that means anything is the pooled one, and this
 * is where you see whether a set earns its place or is crowded out. */
int main_mrmpset_inspect(const char *path) {
  g_cmd = "inspect";
  ms_mrmpset_t *s = ms_mrmpset_open(path);

  mrmp_top_t **top = xcalloc(s->n_sets, sizeof(*top), "per-set tops");
  uint64_t total_pat = 0, total_cpg = 0, n_cpg_rows = 0, file_bytes = 0;
  uint32_t wname = 3;
  for (uint32_t i = 0; i < s->n_sets; ++i) {
    top[i] = ms_mrmp_top_read_at(path, s->block_off[i], UINT32_MAX);
    total_pat += top[i]->n_patterns;
    file_bytes += s->block_bytes[i];
    size_t ln = strlen(s->name[i]);
    if (ln > wname) wname = (uint32_t)ln;          /* names decide the width */
  }
  { mrmp_reader_t r; mrmp_open_at(&r, path, s->block_off[0], s->block_bytes[0]);
    n_cpg_rows = r.h->n_cpg; mrmp_close(&r); }

  char cb[32], cb2[32];
  /* Per-set CpG totals first: the summary is the headline, so it has to be
   * computed before anything is printed rather than tallied under the table. */
  uint64_t *set_cpg = xcalloc(s->n_sets, sizeof(uint64_t), "per-set CpGs");
  for (uint32_t i = 0; i < s->n_sets; ++i) {
    for (uint32_t p = 0; p < top[i]->n_patterns; ++p) set_cpg[i] += top[i]->count[p];
    total_cpg += set_cpg[i];
  }
  const double pct = n_cpg_rows ? 100.0 * total_cpg / (double)n_cpg_rows : 0.0;

  printf("\nMRMP  %s\n", path);
  printf("  %-14s %u\n", "sets", s->n_sets);
  printf("  %-14s %s over %s CpGs (%.2f%% of the row space)\n", "patterns",
         commafmt_local(total_pat, cb), commafmt_local(total_cpg, cb2), pct);
  printf("  %-14s the remaining %.2f%% -- no set has a pattern there\n",
         "PNA", 100.0 - pct);
  printf("  %-14s %s CpG rows\n", "row space", commafmt_local(n_cpg_rows, cb));
  printf("  %-14s %s bytes\n", "on disk", commafmt_local(file_bytes, cb));
  printf("\n");

  /* Width follows the longest name, so a 24-char satellite cannot shove the
   * numeric columns out of line the way a fixed %-12s did. */
  printf("  %-*s  %7s  %8s  %12s  %6s  %s\n", wname, "set",
         "classes", "patterns", "CpGs", "share", "members");
  for (uint32_t i = 0; i < s->n_sets; ++i) {
    mrmp_top_t *t = top[i];
    printf("  %-*s  %7u  %8u  %12s  %5.1f%%  ", wname, s->name[i],
           t->n_samples, t->n_patterns, commafmt_local(set_cpg[i], cb),
           total_pat ? 100.0 * t->n_patterns / (double)total_pat : 0.0);
    /* naming every class is the point for a satellite; for a big global set it
     * would be noise, so elide past a handful */
    if (t->n_samples <= 6) {
      for (uint32_t k = 0; k < t->n_samples; ++k)
        printf("%s%s", k ? ", " : "", t->labels[k]);
    } else {
      printf("%s, %s, ... (%u more)", t->labels[0], t->labels[1],
             t->n_samples - 2);
    }
    putchar('\n');
  }

  /* The per-set totals ARE the budget now. There used to be a table here
   * simulating what a further --top K would keep, back when the cut was a view
   * over patterns that all stayed on disk; mrmp-pool prunes, so a set holds
   * exactly its columns and the simulation answered a question that no longer
   * exists -- while printing 100 name=count pairs on one line. */
  printf("\n");
  for (uint32_t i = 0; i < s->n_sets; ++i) ms_mrmp_top_free(top[i]);
  free(top); free(set_cpg);
  ms_mrmpset_free(s);
  return 0;
}
