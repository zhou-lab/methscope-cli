// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
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

static const char *commafmt_local(uint64_t v, char *buf);   /* defined below */
static uint32_t g_floor_active;      /* --bank type-1 pattern floor; 0 = off */
static void spin_start(int tty, const char *msg);           /* defined below */
static void spin_stop(void);

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

/* A binstring with no class on one side separates nothing: every class called
 * 1, or every class called 0, discriminates no pair however many CpGs it
 * carries. mrmp-build's --call-band happens to exclude these (its both-sides test
 * is max0 >= 0 && min1 <= 1), but only when a filter is given, and pool accepts
 * blocks from any generator built with any flags -- so the check belongs where
 * every pattern passes through. '2' is the ambiguous call and counts as
 * neither side. Returns 1 for all-1, 2 for all-0, 0 when both sides are
 * present. The two are separate because they are different situations: all-1
 * is a CpG methylated in every class, all-0 one methylated in none. */
static int key_flatness(const uint64_t *key, uint32_t ns, char *scratch) {
  key_to_string(key, ns, scratch);
  int n0 = 0, n1 = 0;
  for (uint32_t i = 0; i < ns; ++i) {
    if (scratch[i] == '0') ++n0;
    else if (scratch[i] == '1') ++n1;
  }
  if (!n0) return 1;        /* no class called 0 -> all-1 */
  if (!n1) return 2;        /* no class called 1 -> all-0 */
  return 0;
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

/* ---------------- reference sample names (from <ref>.cg.idx) ------------- */

/* Names in FILE order, and optionally each record's BGZF virtual offset (the
 * .idx second column). Parsed here rather than through YAME's loadIndex()
 * because that returns a khash whose iteration order is arbitrary, and file
 * order IS the binstring's digit order -- a permuted walk would silently
 * relabel every pattern. */
char **ms_read_store_index(const char *ref, uint32_t *n_out,
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

/* ---------------- binstring per-CpG resolution -------------------------- */

/* Resolve CpG i from the meth/ambig bit-planes to its base-3 pattern key,
 * reproducing YAME rowop_binstring: ambiguous cells are filled with the CpG's
 * confident majority; a CpG becomes the all-'2' sentinel when its ambiguous
 * fraction exceeds max_ambig or its confident majority is not sweeping. */
static void resolve_cpg(const uint8_t *meth, const uint8_t *ambig,
                        uint64_t i, uint32_t ns, uint32_t stride,
                        uint64_t n_cpg, float min_fold, float max_ambig,
                        int impute, uint64_t impute_seed,
                        const uint64_t *pna_key, int *is_pna, uint64_t *key,
                        int inc_all0, int inc_all1) {
  uint32_t n1 = 0, namb = 0;
  for (uint32_t g = 0; g < stride; ++g) {
    n1   += (uint32_t)__builtin_popcount(meth[(uint64_t)g * n_cpg + i]);
    namb += (uint32_t)__builtin_popcount(ambig[(uint64_t)g * n_cpg + i]);
  }
  uint32_t n0 = ns - n1 - namb;
  /* Default (--impute-ambiguous none): a binstring is a statement about EVERY
   * class, so one class that is not confidently called makes the CpG PNA. The
   * other strategies fill the ambiguous classes instead, under the guards
   * below (--min-major-fold for majority, --max-ambig-frac for all of them,
   * and --max-lowdepth-frac in selection). Imputation is a claim about missing
   * evidence, so it is made only when asked for. */
  if (namb > 0 && impute == MRMP_IMPUTE_NONE) {
    *is_pna = 1;
    memcpy(key, pna_key, mrmp_key_words(ns) * sizeof(uint64_t));
    return;
  }
  int fill_one = (n1 > n0);                       /* exact tie -> '0' */
  uint32_t hi = fill_one ? n1 : n0, lo = fill_one ? n0 : n1;
  /* "Sweeping" gates the MAJORITY strategy only: it asks whether there is a
   * majority worth copying. A fixed or random fill makes no claim about the
   * confident classes, so nothing to gate. */
  int sweeping = (impute != MRMP_IMPUTE_MAJORITY) ||
                 ((hi > 0) && (lo == 0 || (double)hi >= min_fold * (double)lo));
  uint32_t nw = mrmp_key_words(ns);
  if ((ns && (double)namb > max_ambig * (double)ns) || (namb > 0 && !sweeping)) {
    *is_pna = 1;
    memcpy(key, pna_key, nw * sizeof(uint64_t));
    return;
  }
  /* No class on one side separates nothing. Done HERE rather than left to
   * --call-band's both-sides test, which only runs when a filter is given -- an
   * unfiltered build would otherwise carry these as ordinary patterns, ranked
   * by CpG count like anything else, able to win pooled budget while
   * discriminating no pair. */
  if ((!n0 && !inc_all1) || (!n1 && !inc_all0)) {
    *is_pna = 1;
    memcpy(key, pna_key, nw * sizeof(uint64_t));
    return;
  }
  *is_pna = 0;
  for (uint32_t w = 0; w < nw; ++w) key[w] = 0;
  for (uint32_t s = 0; s < ns; ++s) {
    uint64_t off = (uint64_t)(s >> 3) * n_cpg + i;
    int digit;
    if ((ambig[off] >> (s & 7)) & 1) {
      switch (impute) {
        case MRMP_IMPUTE_ZERO: digit = 0; break;
        case MRMP_IMPUTE_ONE:  digit = 1; break;
        case MRMP_IMPUTE_RANDOM: {
          /* A hash of (seed, CpG, class), not a stream from one generator:
           * the draw must not depend on how many CpGs or classes came first,
           * so it is the same under any thread count or row order. splitmix64
           * finaliser. */
          uint64_t x = impute_seed ^ (i * 0x9E3779B97F4A7C15ULL) ^
                       ((uint64_t)s * 0xBF58476D1CE4E5B9ULL);
          x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
          x ^= x >> 27; x *= 0x94D049BB133111EBULL;
          x ^= x >> 31;
          digit = (int)(x & 1ULL);
          break;
        }
        default: digit = fill_one ? 1 : 0; break;   /* MRMP_IMPUTE_MAJORITY */
      }
    } else digit = (meth[off] >> (s & 7)) & 1;
    key[s / MRMP_TRITS_PER_WORD] = key[s / MRMP_TRITS_PER_WORD] * 3 + (uint64_t)digit;
  }
}

/* ---------------- ranking (count desc, key asc) ------------------------- */

/* --impute-seed, for MRMP_IMPUTE_RANDOM. Not in the header (no room), so a
 * random-filled set is reproducible from the build command, not from itself. */
static uint64_t g_impute_seed = 1;

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
  do {                                  /* do/while so slen == 0 still frames */
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
  keys_bytes += 4;                                  /* "Pna\0" */

  cdata_t c; memset(&c, 0, sizeof c);
  c.fmt = '2'; c.compressed = 0; c.unit = 8; c.n = n_cpg;
  c.s = xcalloc(keys_bytes + 1 + (size_t)n_cpg * 8, 1, "membership fmt2 buffer");
  size_t pos = 0;
  for (uint64_t r = 0; r < n_cand; ++r)
    pos += (size_t)snprintf((char *)c.s + pos, keys_bytes + 1 - pos,
                           "P%" PRIu64, r + 1) + 1;
  memcpy(c.s + pos, "Pna", 4); pos += 4;
  c.s[pos++] = '\0';                                /* key/data double NUL */
  for (uint64_t i = 0; i < n_cpg; ++i) {
    uint64_t v = memb[i] == MRMP_PNA_MEMBERSHIP ? n_cand : memb[i];
    uint8_t *d = c.s + pos + i * 8;
    for (int b = 0; b < 8; ++b) d[b] = (uint8_t)(v >> (8 * b));
  }
  cdata_compress(&c);                               /* -> RLE fmt2 */

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
  return sec;                                       /* caller owns */
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
#include <pthread.h>
#include <signal.h>

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
  if (h->flags & MRMP_FLAG_MINSEG) {
    t->split_minseg = h->split_minseg; t->has_minseg = 1;
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
      size_t nread = fread(buf, 1, sizeof buf - 1, f);
      buf[nread] = '\0';
      if (nread) { nm = xcalloc(strlen(buf) + 1, 1, "set name"); strcpy(nm, buf); }
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
/* keep_ranks (ascending, length keep_n) names the patterns to retain; NULL
 * keeps the prefix 0..keep_n-1, which is what a pure top-N cut wants. Anything
 * not listed has its CpGs folded into PNA and its rank remapped away, so the
 * output is dense again. */
static void prune_block(const char *path, uint64_t base, uint64_t blk_bytes,
                        uint32_t keep_n, const uint32_t *keep_ranks,
                        void **img_out, uint64_t *bytes_out) {
  mrmp_reader_t r; mrmp_open_at(&r, path, base, blk_bytes);
  const mrmp_header_t *h = r.h;
  const uint32_t ns = h->n_samples, nw = r.nw;
  const uint64_t n_cpg = h->n_cpg;
  if (keep_n > h->n_candidates) keep_n = (uint32_t)h->n_candidates;

  /* old rank -> new rank, PNA for everything dropped */
  uint32_t *newrank = xcalloc(h->n_candidates ? h->n_candidates : 1,
                             sizeof(uint32_t), "rank remap");
  for (uint64_t p = 0; p < h->n_candidates; ++p) newrank[p] = MRMP_PNA_MEMBERSHIP;
  for (uint32_t j = 0; j < keep_n; ++j)
    newrank[keep_ranks ? keep_ranks[j] : j] = j;

  const uint32_t *memb = mrmp_membership(&r);
  uint32_t *memb2 = xcalloc(n_cpg, sizeof(uint32_t), "pruned membership");
  uint64_t pna_cpg = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) {
    uint32_t rank = memb[i];
    uint32_t nr = (rank == MRMP_PNA_MEMBERSHIP || rank >= h->n_candidates)
                ? MRMP_PNA_MEMBERSHIP : newrank[rank];
    if (nr == MRMP_PNA_MEMBERSHIP) { memb2[i] = MRMP_PNA_MEMBERSHIP; ++pna_cpg; }
    else memb2[i] = nr;
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
    uint32_t src = keep_ranks ? keep_ranks[p] : p;
    img_put(img, &at, pat_key(&r, src), (size_t)nw * sizeof(uint64_t));
    uint64_t cnt = pat_count(&r, src);
    img_put(img, &at, &cnt, sizeof(uint64_t));
  }
  img_put(img, &at, memb_rle, (size_t)memb_n);
  if (thr) {
    for (uint32_t p = 0; p < keep_n; ++p) {
      float v = thr[keep_ranks ? keep_ranks[p] : p];
      img_put(img, &at, &v, sizeof(float));
    }
  }

  free(newrank); free(memb_rle); free(memb2);
  mrmp_close(&r);
  *img_out = img; *bytes_out = img_bytes;
}

/* One block, re-indexed onto a target row space. The image is built the way
 * prune_block builds one -- same layout, same writer -- because a lift is the
 * same operation with the pattern table left alone: no rank moves, so the
 * booster's columns and the binstrings still mean what they meant. Only the
 * membership is rewritten (target row t takes source row cpg_of_row[t]) and
 * the counts are recomputed over it.
 *
 * content_checksum is carried over on purpose, as prune_block carries it:
 * it identifies how the reference RESOLVED, and no CpG is re-resolved here.
 * The new reference name is what says the row space changed. */
static void lift_block(const char *path, uint64_t base, uint64_t blk_bytes,
                       const int64_t *cpg_of_row, uint64_t n_rows,
                       const char *refname, uint64_t min_retained,
                       uint64_t *n_below, void **img_out, uint64_t *bytes_out) {
  mrmp_reader_t r; mrmp_open_at(&r, path, base, blk_bytes);
  const mrmp_header_t *h = r.h;
  const uint32_t ns = h->n_samples, nw = r.nw;
  const uint64_t n_cand = h->n_candidates, n_src = h->n_cpg;
  const char *set_name = h->name_offset ? r.blk + h->name_offset : "set";

  const uint32_t *memb = mrmp_membership(&r);
  uint32_t *memb2 = xcalloc(n_rows ? n_rows : 1, sizeof(uint32_t), "lifted membership");
  uint64_t *count = xcalloc(n_cand ? n_cand : 1, sizeof(uint64_t), "lifted counts");
  uint64_t pna_cpg = 0;
  for (uint64_t t = 0; t < n_rows; ++t) {
    int64_t src = cpg_of_row[t];
    uint32_t rank = (src < 0 || (uint64_t)src >= n_src) ? MRMP_PNA_MEMBERSHIP
                                                        : memb[src];
    if (rank == MRMP_PNA_MEMBERSHIP || rank >= n_cand) {
      memb2[t] = MRMP_PNA_MEMBERSHIP; ++pna_cpg;
    } else { memb2[t] = rank; ++count[rank]; }
  }
  uint64_t memb_n = 0;
  uint8_t *memb_rle = memb_compress(memb2, n_rows, n_cand, &memb_n);
  if (memb_n > UINT32_MAX) die("compressed membership exceeds 4 GB", path);

  /* what the lift kept, per pattern, so a reader sees the footprint shrink
   * before trusting a call made on it */
  uint64_t kept_min = UINT64_MAX, kept_max = 0, kept_sum = 0, empty = 0, below = 0;
  for (uint64_t p = 0; p < n_cand; ++p) {
    if (count[p] < kept_min) kept_min = count[p];
    if (count[p] > kept_max) kept_max = count[p];
    kept_sum += count[p];
    if (!count[p]) ++empty;
    if (min_retained && count[p] < min_retained) ++below;
  }
  if (!n_cand) kept_min = 0;
  fprintf(stderr, "[methscope] mliftover: %s: %" PRIu64 " pattern(s) over %" PRIu64
          " of %" PRIu64 " target rows (was %" PRIu64 " of %" PRIu64 " CpGs); "
          "CpGs per pattern min %" PRIu64 " mean %.1f max %" PRIu64 "%s\n",
          set_name, n_cand, kept_sum, n_rows, n_src - h->pna_cpg, n_src, kept_min,
          n_cand ? (double)kept_sum / (double)n_cand : 0.0, kept_max,
          empty ? " -- EMPTY patterns present" : "");
  if (empty)
    fprintf(stderr, "[methscope] mliftover: %s: %" PRIu64 " pattern(s) keep no "
            "CpG on this platform; they stay as columns and read as missing\n",
            set_name, empty);
  *n_below += below;

  const float *thr = (h->flags & MRMP_FLAG_THRESH)
                   ? (const float *)(const void *)(r.blk + h->thresh_offset) : NULL;
  mrmp_header_t hd = *h;
  hd.n_cpg = n_rows;
  hd.pna_cpg = pna_cpg;
  hd.membership_bytes = (uint32_t)memb_n;
  hd.flags |= MRMP_FLAG_MEMB_RLE | MRMP_FLAG_MEMB_BGZF;

  uint64_t off = sizeof(hd);
  hd.refname_offset = off;    off += strlen(refname) + 1;
  hd.name_offset = (uint32_t)off; off += strlen(set_name) + 1;
  hd.names_offset = off;      for (uint32_t k = 0; k < ns; ++k) off += strlen(r.names[k]) + 1;
  hd.patterns_offset = off;   off += n_cand * mrmp_pattern_stride(ns);
  hd.membership_offset = off; off += memb_n;
  if (thr) { hd.thresh_offset = off; off += n_cand * sizeof(float); }
  const uint64_t img_bytes = (off + 7u) & ~7ull;

  char *img = xcalloc(img_bytes, 1, "lifted block");
  uint64_t at = 0;
  img_put(img, &at, &hd, sizeof(hd));
  img_put(img, &at, refname, strlen(refname) + 1);
  img_put(img, &at, set_name, strlen(set_name) + 1);
  for (uint32_t k = 0; k < ns; ++k)
    img_put(img, &at, r.names[k], strlen(r.names[k]) + 1);
  for (uint64_t p = 0; p < n_cand; ++p) {
    img_put(img, &at, pat_key(&r, p), (size_t)nw * sizeof(uint64_t));
    img_put(img, &at, &count[p], sizeof(uint64_t));
  }
  img_put(img, &at, memb_rle, (size_t)memb_n);
  if (thr) img_put(img, &at, thr, (size_t)n_cand * sizeof(float));

  free(memb_rle); free(memb2); free(count);
  mrmp_close(&r);
  *img_out = img; *bytes_out = img_bytes;
}

uint64_t ms_mrmp_lift(const char *in, const char *out, const int64_t *cpg_of_row,
                      uint64_t n_rows, const char *refname, uint64_t min_retained) {
  ms_mrmpset_t *s = ms_mrmpset_open(in);
  const uint32_t n = s->n_sets;
  void **img = xcalloc(n, sizeof(void *), "lifted images");
  uint64_t *bytes = xcalloc(n, sizeof(uint64_t), "lifted sizes");
  uint64_t below = 0;
  for (uint32_t k = 0; k < n; ++k)
    lift_block(in, s->block_off[k], s->block_bytes[k], cpg_of_row, n_rows,
               refname, min_retained, &below, &img[k], &bytes[k]);
  ms_mrmp_chain_write(out, n, (const void *const *)img, bytes);
  for (uint32_t k = 0; k < n; ++k) free(img[k]);
  free(img); free(bytes);
  ms_mrmpset_free(s);
  return below;
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
  /* No arguments at all is a question, not an error: print the help rather
   * than a one-line complaint that tells the reader to go ask for it. */
  if (argc == 1) { char *h[2]; h[0] = argv[0]; h[1] = (char *)"-h";
                   (void)main_mrmp_inspect(2, h); return 1; }
  const char *path = NULL; int show_patterns = 0; uint32_t top_k = 20;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      ms_help(stdout, "Usage: methscope inspect [options] IN.mrmp\n\n"
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
  /* Spelled as the flags that set them: "beta=0.500" reads like a measured
   * value rather than the knob it is. */
  printf("  %-14s --call-mindepth %u  --beta-threshold %.3f\n",
         "resolution", h->mincov, h->beta_threshold);
  printf("  %-14s --max-ambig-frac %.3f  --min-major-fold %.3f\n",
         "", h->max_ambig_frac, h->min_major_fold);
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
                           const char *pna_label, uint32_t top_k,
                           const char *who) {
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
  fprintf(stderr, "[methscope] %s: %u sets -> %s (+ %s)\n",
          who, s->n_sets, out_cm, ipath);
  free(ipath);
  ms_mrmpset_free(s);
  return 0;
}

int main_mrmp_export(int argc, char *argv[]) {
  g_cmd = "mrmp-export";
  /* No arguments at all is a question, not an error: print the help rather
   * than a one-line complaint that tells the reader to go ask for it. */
  if (argc == 1) { char *h[2]; h[0] = argv[0]; h[1] = (char *)"-h";
                   (void)main_mrmp_export(2, h); return 1; }
  const char *pos[2] = {NULL, NULL}, *patterns = NULL, *counts = NULL,
             *pna_label = "Pna", *set_name = NULL;
  int npos = 0;
  uint32_t top_k = 1000;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stdout,
        "Usage: methscope mrmp-export [options] IN.mrmp OUT.cm\n\n"
        "  IN               a .mrmp: one set, or a chain of several\n"
        "  OUT.cm           per-CpG P1..PK/Pna labels as a YAME format-2 mask\n"
        "  OUT.cm           OPTIONAL when --patterns/--counts is given\n"
        "  OUT.cm           a multi-set input writes ONE .cm holding every set as\n"
        "                   a record, plus OUT.cm.idx naming them -- YAME's own\n"
        "                   store convention, so `yame subset -s <set>` works.\n\n"
        "  This command is an INTERFACE TO YAME, not part of the pipeline.\n"
        "  classify-featurize reads a .mrmp directly and a bundle carries one, so\n"
        "  nothing in methscope consumes a .cm any more; export exists to hand\n"
        "  sets to yame and should speak yame's idiom rather than ours.\n\n"
        "  --set NAME       export just this set of a container, to OUT.cm\n"
        "  --top K          rank cut for the mask and --patterns (default 1000)\n"
        "  --patterns TSV   top-K patterns: string<tab>P<rank><tab>count (- = stdout)\n"
        "  --counts TSV     every pattern (incl. PNA): count<tab>string (- = stdout)\n"
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
  /* OUT.cm is required only when a mask is actually wanted. --patterns and
   * --counts are the way to read a set's binstrings from outside the binary
   * (see the trit-order note in mrmp.h -- decoding keys by hand is a trap), and
   * demanding an unwanted OUT.cm made every such call pass /dev/null. */
  if (npos < 1) die("need IN.mrmp (see mrmp-export -h)", NULL);
  if (npos < 2 && !patterns && !counts)
    die("need OUT.cm, or --patterns/--counts (see mrmp-export -h)", NULL);
  const char *path = pos[0], *mask = npos > 1 ? pos[1] : NULL;

  /* One set or many is the same format now, so the arity decides: a multi-set
   * chain with no --set goes to a directory, a single-set file to one .cm. */
  ms_mrmpset_t *s = ms_mrmpset_open(path);
  if (s->n_sets > 1 && !set_name) {
    if (patterns || counts)
      die("--patterns/--counts describe ONE set; add --set NAME", path);
    if (!mask) die("a whole chain needs OUT.cm; --set NAME reads one set", path);
    ms_mrmpset_free(s);
    return export_container(path, mask, pna_label, top_k, "mrmp-export");
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
    FILE *f = !strcmp(patterns, "-") ? stdout : fopen(patterns, "w");
    if (!f) die("cannot create --patterns", patterns);
    uint64_t lim = top_k < h->n_candidates ? top_k : h->n_candidates;
    for (uint64_t p = 0; p < lim; ++p) {
      key_to_string(pat_key(&r, p), ns, buf);
      fprintf(f, "%s\tP%" PRIu64 "\t%" PRIu64 "\n", buf, p + 1, pat_count(&r, p));
    }
    if (f != stdout && fclose(f)) die("error closing --patterns", patterns);
  }

  if (counts) {
    /* Every candidate in rank order, then the PNA sentinel. */
    FILE *f = !strcmp(counts, "-") ? stdout : fopen(counts, "w");
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
    if (f != stdout && fclose(f)) die("error closing --counts", counts);
  }

  if (mask) ms_mrmp_write_mask_at(path, base, blk_bytes, mask, pna_label, top_k);

  free(buf);
  mrmp_close(&r);
  if (s) ms_mrmpset_free(s);
  return 0;
}

/* ---------------- artifact -> runtime forms ------------------------------ */

uint64_t ms_mrmp_n_cpg_at(const char *path, uint64_t base) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open", path);
  mrmp_header_t h;
  if (fseeko(f, (off_t)base, SEEK_SET) || fread(&h, 1, sizeof h, f) != sizeof h) {
    fclose(f); die("cannot read MRMP header", path);
  }
  fclose(f);
  if (memcmp(h.magic, MRMPIDX_MAGIC, 8)) die("not an MRMPIDX1 block", path);
  return h.n_cpg;
}

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

uint32_t ms_mrmp_group_map_chain(const char *chain, uint16_t *group,
                                uint64_t n_cpg, uint32_t patterns_per_set,
                                uint32_t *col0, uint32_t *n_sets_out) {
  ms_mrmpset_t *ch = ms_mrmpset_open(chain);
  uint32_t total = 0;
  for (uint32_t s = 0; s < ch->n_sets; ++s) {
    col0[s] = total;
    uint16_t *g = group + (uint64_t)s * n_cpg;
    ms_mrmp_group_map_at(chain, ch->block_off[s], g, n_cpg, patterns_per_set);
    /* the set's own width: how many distinct ids it actually emitted */
    uint32_t k = 0;
    for (uint64_t i = 0; i < n_cpg; ++i) if (g[i] > k) k = g[i];
    /* shift into the global column space so the caller can just concatenate */
    if (total)
      for (uint64_t i = 0; i < n_cpg; ++i) if (g[i]) g[i] = (uint16_t)(g[i] + total);
    total += k;
    if (total > UINT16_MAX - 1)
      die("chain feature columns exceed the uint16 group format", chain);
  }
  col0[ch->n_sets] = total;
  *n_sets_out = ch->n_sets;
  ms_mrmpset_free(ch);
  return total;
}

uint32_t ms_mrmp_thresholds_at(const char *path, uint64_t base, uint64_t blk_bytes,
                              uint32_t n_want, float *out) {
  mrmp_reader_t r; mrmp_open_at(&r, path, base, blk_bytes);
  uint32_t n = 0;
  if (r.h->flags & MRMP_FLAG_THRESH) {
    n = r.h->n_candidates < n_want ? (uint32_t)r.h->n_candidates : n_want;
    memcpy(out, r.blk + r.h->thresh_offset, (size_t)n * sizeof(float));
  }
  mrmp_close(&r);
  return n;
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
  /* Unlike the mask above there is no multi-record escape here: a group map is
   * one pattern id per CpG, so overlapping sets are not representable at all.
   * Refuse rather than return block 0 as if it were the whole artifact. */
  ms_mrmpset_t *s = ms_mrmpset_open(artifact);
  const uint32_t n_sets = s->n_sets;
  ms_mrmpset_free(s);
  if (n_sets > 1)
    die("artifact is a chain of several sets and a group map holds one pattern "
        "per CpG, so overlapping sets cannot be represented -- pass a "
        "single-set .mrmp", artifact);
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
  /* A chain does NOT collapse into one mask record. Sets overlap on CpGs and a
   * .cm gives each CpG exactly one pattern, so the CpGs two sets share would
   * fight -- and those are precisely the informative ones. Write YAME's
   * multi-record form instead, one record per set plus a .idx, which is what
   * ms_matrix_build already reads.
   *
   * This used to take block 0 unconditionally. Since a chain's first block is
   * itself a valid MRMPIDX1 that succeeded silently, handing every caller the
   * GLOBAL set alone: a 100-set artifact resolved to 803 labels rather than
   * 1,100. That is why deconvolution "only used the global MRMP". */
  ms_mrmpset_t *s = ms_mrmpset_open(artifact);
  const uint32_t n_sets = s->n_sets;
  ms_mrmpset_free(s);
  if (n_sets > 1) { export_container(artifact, out_cm, pna_label, top_k, "mrmp"); return; }
  ms_mrmp_write_mask_at(artifact, 0, 0, out_cm, pna_label, top_k);
}


/* mrmp-pack is gone: a chain IS a concatenation, so `cat a.mrmp b.mrmp > c.mrmp`
 * is the exact combine it used to perform. mrmp-pool remains, because applying
 * a shared column budget is selection rather than merging and cat cannot do it. */

/* One pooled candidate: which set it came from and how many CpGs carry it.
 * File scope, with a plain comparator -- a nested function would be a GCC
 * extension and would force an executable stack for the trampoline. */
typedef struct { uint64_t count; uint32_t set; uint32_t rank; } ent_t;
static int pooled_cmp(const void *a, const void *b) {
  const ent_t *x = a, *y = b;
  if (x->count > y->count) return -1;
  if (x->count < y->count) return 1;
  return (x->set < y->set) ? -1 : (x->set > y->set);
}

int main_mrmp_pool(int argc, char *argv[]) {
  const char **pos = xcalloc((size_t)argc, sizeof(char *), "positional args");
  int npos = 0;
  g_cmd = "mrmp-pool";
  int inc_all0 = 0, inc_all1 = 0;
  /* No arguments at all is a question, not an error: print the help rather
   * than a one-line complaint that tells the reader to go ask for it. */
  if (argc == 1) { char *h[2]; h[0] = argv[0]; h[1] = (char *)"-h";
                   (void)main_mrmp_pool(2, h); return 1; }
  const char *out = NULL;
  uint32_t pooled_top = 1000, min_cpgs = 0;
  int i = 1;
  for (; i < argc; ++i) {
    if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--pooled-top") && i + 1 < argc)
      pooled_top = (uint32_t)parse_u64(argv[++i], "--pooled-top");
    else if (!strcmp(argv[i], "--min-cpgs") && i + 1 < argc)
      min_cpgs = (uint32_t)parse_u64(argv[++i], "--min-cpgs");
    else if (!strcmp(argv[i], "--include-all-0")) inc_all0 = 1;
    else if (!strcmp(argv[i], "--include-all-1")) inc_all1 = 1;
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      ms_help(stdout,
        "Usage: methscope mrmp-pool [options] -o OUT.mrmp IN.mrmp [IN.mrmp ...]\n\n"
        "Pool several MRMP sets into one chain and cut them to a\n"
        "shared pattern budget. Needs no store and no reference -- pattern CpG\n"
        "counts are already in each artifact -- so re-pooling at a different\n"
        "budget costs seconds. Plain concatenation is just `cat`; this is the\n"
        "step that SELECTS.\n\n"
        "A set is a set: the inputs may come from any generator and from\n"
        "DIFFERENT stores, as long as they share a row space. Nothing here\n"
        "distinguishes a global from a satellite.\n\n"
        "Each input is a CHAIN of one or more sets and expands into all of\n"
        "them under their own names, so a satellite-bearing chain competes for\n"
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
        "  -o OUT           output container\n"
        "  --min-cpgs N     drop any pattern carrying fewer than N CpGs,\n"
        "                   BEFORE the budget ranks what is left. A pattern is\n"
        "                   a feature, and a feature averages the reads a cell\n"
        "                   happens to have in it -- so a 1-CpG pattern is one\n"
        "                   binarized read, which is noise wearing a column.\n"
        "                   The tail is nearly free to cut: on the 7-set Zhou\n"
        "                   tree, --min-cpgs 10 drops 82% of the patterns and\n"
        "                   1.6% of the CpGs (4,626 of 9,965 are singletons).\n"
        "                   Gate and budget compose: --min-cpgs alone prunes to\n"
        "                   everything that clears the floor.\n"
        "  --include-all-0            keep patterns no class calls 1\n"
        "  --include-all-1            keep patterns no class calls 0\n"
        "                             Both are folded into PNA by default: a\n"
        "                             binstring with no class on one side\n"
        "                             separates nothing however many CpGs it\n"
        "                             carries, so it should not consume budget.\n"
        "                             mrmp-build --call-band also excludes them,\n"
        "                             but only when a filter is given, and pool\n"
        "                             takes blocks from any generator.\n");
      return 0;
    }
    else if (argv[i][0] == '-') die("unrecognized option", argv[i]);
    else pos[npos++] = argv[i];   /* a positional never stops the scan */
  }
  if (!out || npos < 1) die("need -o OUT and at least one IN.mrmp", NULL);

  /* Every input is a chain of one or more sets and expands into all of them.
   * That is what makes the four-command workflow close: one `mrmp-build
   * --satellite-n` emits ONE file holding many 2-class sets, and pooling has to
   * see them as the separate competitors they are, not as one blob.
   * Expanded blocks keep the container's own set names -- those came from the
   * generator that knows what each set is, and are what makes `inspect`'s pooled
   * table readable. So the input count is not the set count. */
  uint32_t cap = (uint32_t)npos, n = 0;
  char **name = xcalloc(cap, sizeof(char *), "set names");
  uint64_t *len = xcalloc(cap, sizeof(uint64_t), "block sizes");
  uint64_t *soff = xcalloc(cap, sizeof(uint64_t), "block offsets");
  const char **path = xcalloc(cap, sizeof(char *), "paths");
  for (uint32_t k = 0; k < (uint32_t)npos; ++k) {
    const char *p = pos[k];
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
  uint32_t **keep_list = NULL;      /* per-set winning ranks; NULL == prefix */
  uint64_t n_all1 = 0, n_all0 = 0, n_thin = 0;
  for (uint32_t k = 0; k < n; ++k) won[k] = hd[k].n_selected;

  if (pooled_top || min_cpgs) {
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
      char *scr = xcalloc((size_t)hd[k].n_samples + 1, 1, "binstring scratch");
      for (uint64_t r = 0; r < hd[k].n_candidates; ++r) {
        { int fl = key_flatness((const uint64_t *)(const void *)(rec + r * st),
                               hd[k].n_samples, scr);
          if (fl == 1 && !inc_all1) { ++n_all1; continue; }
          if (fl == 2 && !inc_all0) { ++n_all0; continue; } }
        uint64_t cnt; memcpy(&cnt, rec + r * st + koff, sizeof(cnt));
        if (cnt < min_cpgs) { ++n_thin; continue; }
        e[m].count = cnt; e[m].set = k; e[m].rank = (uint32_t)r; ++m;
      }
      free(scr);
      free(rec);
    }
    qsort(e, m, sizeof(ent_t), pooled_cmp);
    uint64_t take = (pooled_top && m > pooled_top) ? pooled_top : m;
    if (!take) die("--min-cpgs left no pattern anywhere; lower it", out);
    memset(won, 0, n * sizeof(uint32_t));
    for (uint64_t j = 0; j < take; ++j) ++won[e[j].set];
    /* Which ranks won, per set, in count-descending order -- which is ascending
     * rank order, so the pruned block keeps recurrence rank meaning what it
     * meant. Needed because the winners are no longer a prefix once flat
     * patterns are skipped. */
    keep_list = xcalloc(n, sizeof(uint32_t *), "per-set keep lists");
    { uint32_t *fill = xcalloc(n, sizeof(uint32_t), "keep cursors");
      for (uint32_t k = 0; k < n; ++k)
        if (won[k]) keep_list[k] = xcalloc(won[k], sizeof(uint32_t), "keep list");
      for (uint64_t j = 0; j < take; ++j) {
        uint32_t k = e[j].set;
        keep_list[k][fill[k]++] = e[j].rank;
      }
      free(fill); }
    if (n_thin)
      fprintf(stderr, "  gated by --min-cpgs %u: %" PRIu64 " pattern(s) below "
              "the floor, folded into PNA\n", min_cpgs, n_thin);
    if (n_all1 || n_all0)
      fprintf(stderr, "  folded into PNA: %" PRIu64 " all-1 pattern(s), %"
              PRIu64 " all-0 -- they separate no class "
              "(--include-all-1 / --include-all-0 retain them)\n",
              n_all1, n_all0);
    /* Width follows the longest name, as `inspect` does: at a fixed %-22s a
     * satellite name overran the field and shoved every number out of line.
     * "patterns keep a column" was also repeated once per row; it is a column
     * heading, not a sentence. */
    { uint32_t wn = 3;
      for (uint32_t k = 0; k < n; ++k) {
        size_t ln = strlen(name[k]);
        if (ln > wn) wn = (uint32_t)ln;
      }
      fprintf(stderr, "  %-*s  %7s  %9s\n", wn, "set", "pooled", "of");
      char c1[32], c2[32];
      for (uint32_t k = 0; k < n; ++k)
        fprintf(stderr, "  %-*s  %7s  %9s\n", wn, name[k],
                commafmt_local(won[k], c1),
                commafmt_local(hd[k].n_candidates, c2)); }
    free(e);
  }

  if (!pooled_top && !min_cpgs) {
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
      prune_block(path[k], soff[k], len[k], won[k],
                  keep_list ? keep_list[k] : NULL, &img[m], &ilen[m]);
      ++m;
    }
    if (!m) die("every set was cut to nothing; raise --pooled-top", out);
    if (dropped)
      fprintf(stderr, "[methscope] mrmp-pool: %u set(s) won no pattern and were "
              "dropped\n", dropped);
    ms_mrmp_chain_write(out, m, (const void *const *)img, ilen);
    for (uint32_t k = 0; k < m; ++k) free(img[k]);
    free(img); free(ilen);
    n = m;
  }
  if (min_cpgs)
    fprintf(stderr, "[methscope] mrmp-pool: %u sets, budget %u, floor %u CpGs "
            "-> %s\n", n, pooled_top, min_cpgs, out);
  else
    fprintf(stderr, "[methscope] mrmp-pool: %u sets, budget %u -> %s\n",
            n, pooled_top, out);
  free(name); free(len); free(soff); free(path); free(hd); free(won);
  return 0;
}

/* ---------------- subset-block builder (shared) -------------------------- */

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

/* ---------------- store -> per-CpG binstring (shared) --------------------- */

/* mrmp-build's first two passes, exported so upscale-set-units resolves CpGs by
 * the same rule rather than a second copy of it. See mrmp.h. */
void ms_binstring_map(const char *store, uint32_t ns, char *const *label,
                      const int64_t *voff, uint32_t mincov, float beta_thr,
                      float max_ambig, float min_fold, int impute,
                      uint64_t impute_seed,
                      int inc_all0, int inc_all1, ms_binstring_map_t *out) {
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

  /* Pass 2: resolve, intern, remember. */
  uint64_t *pna_key = xcalloc(nw, sizeof(uint64_t), "pna key");
  for (uint32_t s = 0; s < ns; ++s)
    pna_key[s / MRMP_TRITS_PER_WORD] = pna_key[s / MRMP_TRITS_PER_WORD] * 3 + 2;
  phash_t h; phash_init(&h, 1u << 12, nw);
  uint64_t pat_cap = 1u << 12, n_pat = 0, pna_cpg = 0;
  uint64_t *pkeys = xcalloc(pat_cap * nw, sizeof(uint64_t), "pattern keys");
  uint64_t *pcount = xcalloc(pat_cap, sizeof(uint64_t), "pattern counts");
  uint32_t *pidx = xcalloc(n_cpg, sizeof(uint32_t), "cpg -> pattern");
  uint64_t *key = xcalloc(nw, sizeof(uint64_t), "cpg key");
  uint64_t checksum = 1469598103934665603ULL;       /* FNV-1a offset */
  for (uint64_t i = 0; i < n_cpg; ++i) {
    int is_pna;
    resolve_cpg(meth, ambig, i, ns, stride, n_cpg,
                min_fold, max_ambig, impute, impute_seed, pna_key, &is_pna,
                key, inc_all0, inc_all1);
    if (is_pna) {
      pidx[i] = MRMP_PNA_MEMBERSHIP; ++pna_cpg;
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
  free(meth); free(ambig); free(h.keys); free(h.slot); free(key); free(pna_key);

  out->n_samples = ns; out->n_cpg = n_cpg; out->n_pat = n_pat;
  out->keys = pkeys;   out->count = pcount; out->cpg_pat = pidx;
  out->pna_cpg = pna_cpg; out->checksum = checksum;
}

void ms_binstring_map_free(ms_binstring_map_t *m) {
  if (!m) return;
  free(m->keys); free(m->count); free(m->cpg_pat);
  m->keys = NULL; m->count = NULL; m->cpg_pat = NULL;
  m->n_pat = 0; m->n_cpg = 0;
}

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
  const uint32_t nw = mrmp_key_words(ns);

  /* Passes 1-2 are ms_binstring_map(): planes, then resolve/intern. Shared with
   * upscale-set-units so the two cannot drift on what a binstring is. The map
   * KEEPS each CpG's pattern rather than re-deriving it (as mrmp-build does),
   * because selection rewrites it twice and a third resolve pass would cost
   * more than the 4 bytes per CpG. */
  ms_binstring_map_t bm;
  ms_binstring_map(store, ns, label, voff, mincov, beta_thr, max_ambig,
                   min_fold, mrmp_impute_method(gh->flags), g_impute_seed,
                   sel->inc_all0, sel->inc_all1, &bm);
  const uint64_t n_cpg = bm.n_cpg;
  uint64_t n_pat = bm.n_pat, pat_cap = bm.n_pat ? bm.n_pat : 1;
  uint64_t *pkeys = bm.keys, *pcount = bm.count;
  uint32_t *pidx = bm.cpg_pat;
  uint64_t checksum = bm.checksum;
  (void)pat_cap;
  /* ownership moves here: the three arrays are freed below (pidx/pcount with
   * the selection scratch, pkeys with the write), so ms_binstring_map_free()
   * must NOT also run on `bm`. */
  uint64_t *pna_key = xcalloc(nw, sizeof(uint64_t), "pna key");
  for (uint32_t s2 = 0; s2 < ns; ++s2)
    pna_key[s2 / MRMP_TRITS_PER_WORD] = pna_key[s2 / MRMP_TRITS_PER_WORD] * 3 + 2;
  cfile_t cf;

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

  /* BANK type-1 floor: a pattern under it is dropped whole; its CpGs fold
   * into PNA through the empty-pattern path below, exactly as selection
   * casualties do. Pair sets are exempt -- their stringency is the
   * calibrated admission, not a pattern-size gate. */
  if (g_floor_active && ns > 2)
    for (uint64_t p = 0; p < n_pat; ++p)
      if (ncount[p] < g_floor_active) ncount[p] = 0;

  uint64_t n_cand = 0;
  uint32_t *ord2 = xcalloc(n_pat ? n_pat : 1, sizeof(uint32_t), "rank order");
  for (uint64_t p = 0; p < n_pat; ++p)
    if (ncount[p]) ord2[n_cand++] = (uint32_t)p;
  if (!n_cand) {
    /* Under the BANK floor an emptied set is an ANSWER, not an error: the
     * caller treats the node as emitting no type-1 and gives its pairs
     * mandatory resolvers. Everywhere else empty remains fatal. */
    if (g_floor_active) {
      free(pkeys); free(ncount); free(ord2); free(rank_of);
      free(pidx); free(pna_key);
      memset(out, 0, sizeof *out);
      return;
    }
    die("no CpG survived selection for this set", label[0]);
  }
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

/* Progress spinner for the per-node MRMP builds.
 *
 * Each build_subset_block() is a multi-second streaming pass with no
 * natural progress to report from inside, so a static line looks hung. A
 * detached thread repaints a frame while the caller works; the caller then
 * overwrites the whole line with the finished result.
 *
 * Terminal only. Redirected, this writes nothing at all -- \r animation in a
 * SLURM log is noise, and mrmp-build prints one line per node there instead. */
static volatile sig_atomic_t g_spin_run;
static char g_spin_msg[192];

static void *spin_worker(void *arg) {
  (void)arg;
  static const char *frame[] = {"\xe2\xa3\xbe","\xe2\xa3\xbd","\xe2\xa3\xbb",
                               "\xe2\xa2\xbf","\xe2\xa1\xbf","\xe2\xa3\x9f",
                               "\xe2\xa3\xaf","\xe2\xa3\xb7"};
  for (unsigned i = 0; g_spin_run; ++i) {
    fprintf(stderr, "\r\033[K  %s %s", frame[i % 8], g_spin_msg);
    fflush(stderr);
    usleep(120000);
  }
  return NULL;
}

static pthread_t g_spin_th;
static int       g_spin_on;

static void spin_start(int tty, const char *msg) {
  if (!tty) return;
  snprintf(g_spin_msg, sizeof g_spin_msg, "%s", msg);
  g_spin_run = 1;
  g_spin_on = (pthread_create(&g_spin_th, NULL, spin_worker, NULL) == 0);
  if (!g_spin_on) g_spin_run = 0;      /* no thread: fall back to no animation */
}

static void spin_stop(void) {
  if (!g_spin_on) return;
  g_spin_run = 0;
  pthread_join(g_spin_th, NULL);
  g_spin_on = 0;
  fprintf(stderr, "\r\033[K");
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
int main_mrmpset_inspect(const char *path, int show_patterns, uint32_t top_k) {
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
  /* The single-set report opens with `format`; this one did not, so the two
   * .mrmp reports disagreed on their own first line. */
  printf("  %-14s MRMPIDX1 v1, chain of %u sets\n", "format", s->n_sets);
  printf("  %-14s %s over %s CpGs (%.2f%% of the row space)\n", "patterns",
         commafmt_local(total_pat, cb), commafmt_local(total_cpg, cb2), pct);
  printf("  %-14s the remaining %.2f%% -- no set has a pattern there\n",
         "PNA", 100.0 - pct);
  printf("  %-14s %s CpG rows\n", "row space", commafmt_local(n_cpg_rows, cb));
  printf("  %-14s %s bytes\n", "on disk", commafmt_local(file_bytes, cb));
  printf("\n");

  /* Width follows the longest name, so a 24-char satellite cannot shove the
   * numeric columns out of line the way a fixed %-12s did. */
  /* No share column. As a fraction of patterns it was 100/n_sets on every row
   * of a satellite chain; as a fraction of summed CpGs the denominator
   * double-counts whatever two sets share, so neither reading was worth a
   * column. The absolute CpG count beside it says the same thing honestly. */
  printf("  %-*s  %7s  %8s  %12s  %s\n", wname, "set",
         "classes", "patterns", "CpGs", "members");
  for (uint32_t i = 0; i < s->n_sets; ++i) {
    mrmp_top_t *t = top[i];
    printf("  %-*s  %7u  %8u  %12s  ", wname, s->name[i],
           t->n_samples, t->n_patterns, commafmt_local(set_cpg[i], cb));
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

  if (show_patterns) {
    printf("\nset\tlabel\tpattern\tcpg_count\n");
    for (uint32_t i = 0; i < s->n_sets; ++i) {
      uint32_t lim = top_k < top[i]->n_patterns ? top_k : top[i]->n_patterns;
      for (uint32_t p = 0; p < lim; ++p)
        printf("%s\tP%u\t%s\t%llu\n", s->name[i], p + 1,
               top[i]->binstring[p], (unsigned long long)top[i]->count[p]);
    }
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

/* ----------------------------------------------------------------- mrmp-build
 *
 * One command builds EVERY node of a routing tree, each node an MRMP over its
 * own class subset. It replaces the old build + satellite-build + pool chain
 * for this shape: the tree IS the set collection, and a node's children are the
 * groups its own patterns cannot tell apart.
 *
 * Why per-node rebuild rather than partitioning one global's patterns: a
 * pattern must be consistent across EVERY class in its set, so each class
 * dropped relaxes the constraint and admits more CpGs. Measured on this
 * reference, the colon/small-intestine pair carries 21,146 segregating CpGs
 * inside the 33-class global and 102,694 as its own 2-class set -- ~5x the
 * evidence from nothing but removing the other 31 classes. The recursion GAINS
 * evidence as it descends, which is the opposite of a decision tree, where a
 * child always sees less data than its parent.
 *
 * The parent pointer lives in the SET NAME -- "root", "root.0", "root.0.1" --
 * so a node's parent is its name minus the last dotted component. The 128-byte
 * header has no room left (see mrmp.h), and a name costs nothing, survives
 * `cat`, and already prints in `inspect`.
 *
 * The split rule is single-linkage on the segregating-CpG graph. Two classes
 * that the node separates by <= --min-segregating CpGs MUST end up in the same
 * child, so the groups are the connected components of that graph. This is not
 * a heuristic: routing is HARD, a misroute is unrecoverable downstream, and
 * "these two are separated by more than N CpGs" is exactly the statement that
 * routing between them is reliable. Connected components also make the shape
 * independent of pattern order -- the tree is a function of (reference,
 * selection rule, min-segregating) and nothing else.
 *
 * Group separation is the MINIMUM over cross-group class pairs rather than the
 * count of patterns that split the whole groups apart. The weakest pair is what
 * actually breaks routing, and taking the minimum keeps the criterion
 * consistent with the component construction it feeds. */

typedef struct {
  void    **img;
  uint64_t *len;
  char    **name;
  uint32_t  n, cap;
} treeout_t;

static void tree_push(treeout_t *t, void *img, uint64_t len, const char *name) {
  if (t->n == t->cap) {
    t->cap = t->cap ? t->cap * 2 : 64;
    t->img  = realloc(t->img,  (size_t)t->cap * sizeof(void *));
    t->len  = realloc(t->len,  (size_t)t->cap * sizeof(uint64_t));
    t->name = realloc(t->name, (size_t)t->cap * sizeof(char *));
    if (!t->img || !t->len || !t->name) die("out of memory (tree)", NULL);
  }
  t->img[t->n] = img; t->len[t->n] = len;
  t->name[t->n] = strdup(name); ++t->n;
}

/* Segregating CpGs for every class pair of a freshly built block, read straight
 * out of its in-memory image -- the block is a standalone MRMPIDX1, so this is
 * the same walk ms_mrmp_top_read() does on a file, minus the file.
 *
 * seg[a*ns+b] sums the CpGs of patterns calling a '0' and b '1' or the reverse.
 * A '2' (no call) never counts, because a pattern that abstains on a class
 * carries no evidence about it. */
static uint64_t *tree_pair_seg(const void *img) {
  const mrmp_header_t *h = (const mrmp_header_t *)img;
  const uint32_t ns = h->n_samples, nw = mrmp_key_words(ns);
  const uint64_t stride = mrmp_pattern_stride(ns), npat = h->n_candidates;
  const char *base = (const char *)img + h->patterns_offset;
  uint64_t *seg = xcalloc((size_t)ns * ns, sizeof(uint64_t), "pair seg");
  char *bs = xcalloc((size_t)ns + 1, 1, "binstring");
  uint32_t *z = xcalloc(ns, sizeof(uint32_t), "zero side");
  uint32_t *o = xcalloc(ns, sizeof(uint32_t), "one side");
  for (uint64_t p = 0; p < npat; ++p) {
    const char *rec = base + p * stride;
    key_to_string((const uint64_t *)(const void *)rec, ns, bs);
    uint64_t cnt; memcpy(&cnt, rec + (uint64_t)nw * sizeof(uint64_t), sizeof cnt);
    if (!cnt) continue;
    uint32_t nz = 0, no = 0;
    for (uint32_t s = 0; s < ns; ++s) {
      if (bs[s] == '0') z[nz++] = s; else if (bs[s] == '1') o[no++] = s;
    }
    /* only the cross product carries a contrast, so a homogeneous pattern
     * (nz or no zero) costs nothing here */
    for (uint32_t i = 0; i < nz; ++i)
      for (uint32_t j = 0; j < no; ++j) {
        seg[(uint64_t)z[i] * ns + o[j]] += cnt;
        seg[(uint64_t)o[j] * ns + z[i]] += cnt;
      }
  }
  free(bs); free(z); free(o);
  return seg;
}

/* Connected components of {a--b : seg(a,b) <= min_seg}: classes the node cannot
 * reliably tell apart must ride to the same child together. Returns the group
 * count and fills grp[] with a 0-based group id per class, in first-appearance
 * order so the numbering is stable. */
static uint32_t tree_partition(const uint64_t *seg, uint32_t ns,
                              uint64_t min_seg, uint32_t *grp) {
  uint32_t *par = xcalloc(ns, sizeof(uint32_t), "union-find");
  for (uint32_t s = 0; s < ns; ++s) par[s] = s;
  for (uint32_t a = 0; a < ns; ++a)
    for (uint32_t b = a + 1; b < ns; ++b) {
      if (seg[(uint64_t)a * ns + b] > min_seg) continue;
      uint32_t ra = a, rb = b;
      while (par[ra] != ra) ra = par[ra] = par[par[ra]];
      while (par[rb] != rb) rb = par[rb] = par[par[rb]];
      if (ra != rb) par[ra > rb ? ra : rb] = ra < rb ? ra : rb;
    }
  uint32_t ng = 0;
  for (uint32_t s = 0; s < ns; ++s) grp[s] = UINT32_MAX;
  for (uint32_t s = 0; s < ns; ++s) {
    uint32_t r = s;
    while (par[r] != r) r = par[r];
    if (grp[r] == UINT32_MAX) grp[r] = ng++;
    grp[s] = grp[r];
  }
  free(par);
  return ng;
}

/* Largest edge on the minimum spanning tree of the pair graph -- the smallest
 * threshold at which single-linkage CONNECTS every class (merging edges
 * <= t), so any threshold strictly below it splits the node, and bottleneck-1
 * is the largest splitting threshold. Prim on the dense symmetric matrix;
 * n <= a few hundred classes, so O(n^2) is nothing. */
static uint64_t tree_mst_bottleneck(const uint64_t *seg, uint32_t ns) {
  uint64_t *d = xcalloc(ns, sizeof(uint64_t), "mst dist");
  uint8_t *in = xcalloc(ns, 1, "mst done");
  for (uint32_t i = 1; i < ns; ++i) d[i] = UINT64_MAX;
  uint64_t bneck = 0;
  for (uint32_t it = 0; it < ns; ++it) {
    uint32_t b = UINT32_MAX;
    for (uint32_t i = 0; i < ns; ++i)
      if (!in[i] && (b == UINT32_MAX || d[i] < d[b])) b = i;
    in[b] = 1;
    if (d[b] > bneck) bneck = d[b];
    for (uint32_t i = 0; i < ns; ++i)
      if (!in[i] && seg[(uint64_t)b * ns + i] < d[i])
        d[i] = seg[(uint64_t)b * ns + i];
  }
  free(d); free(in);
  return bneck;
}

/* Weakest cross-group pair, i.e. the separation the routing between two
 * children actually rests on. */
static uint64_t tree_group_gap(const uint64_t *seg, uint32_t ns,
                              const uint32_t *grp, uint32_t ga, uint32_t gb) {
  uint64_t lo = UINT64_MAX;
  for (uint32_t a = 0; a < ns; ++a) {
    if (grp[a] != ga) continue;
    for (uint32_t b = 0; b < ns; ++b) {
      if (grp[b] != gb) continue;
      uint64_t v = seg[(uint64_t)a * ns + b];
      if (v < lo) lo = v;
    }
  }
  return lo == UINT64_MAX ? 0 : lo;
}

static int u64cmp(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return x < y ? -1 : x > y;
}

/* The off-diagonal of seg[], sorted ascending; caller frees. */
static uint64_t *tree_pair_sorted(const uint64_t *seg, uint32_t ns, uint64_t *n) {
  *n = (uint64_t)ns * (ns - 1) / 2;
  uint64_t *v = xcalloc(*n ? *n : 1, sizeof(uint64_t), "pair list"), k = 0;
  for (uint32_t a = 0; a < ns; ++a)
    for (uint32_t b = a + 1; b < ns; ++b) v[k++] = seg[(uint64_t)a * ns + b];
  qsort(v, (size_t)*n, sizeof(uint64_t), u64cmp);
  return v;
}

/* Build this node's MRMP over its own classes, then recurse into the groups it
 * cannot separate. Depth-first, so the chain reads parent before child. */

/* CpG-weighted Hamming between every pair of a node's classes: the fraction of
 * this set's pattern CpG mass on which the two are called differently. It is
 * the metric the classifier scores with, so a small distance IS a predicted
 * confusion, and it reads the reference alone -- no test cell, no confusion
 * matrix -- so satellites can be chosen at build time. Distinct from
 * tree_pair_seg, which counts only CpGs where one is '0' and the other '1';
 * here a '2' against a '1' is a difference too, matching the 20260806
 * generator this reproduces. */
static double *sat_hamming(const void *img) {
  const mrmp_header_t *h = (const mrmp_header_t *)img;
  const uint32_t ns = h->n_samples, nw = mrmp_key_words(ns);
  const uint64_t stride = mrmp_pattern_stride(ns), npat = h->n_candidates;
  const char *base = (const char *)img + h->patterns_offset;
  double *d = xcalloc((size_t)ns * ns, sizeof(double), "sat hamming");
  char *bs = xcalloc((size_t)ns + 1, 1, "binstring");
  double tot = 0.0;
  for (uint64_t p = 0; p < npat; ++p) {
    const char *rec = base + p * stride;
    key_to_string((const uint64_t *)(const void *)rec, ns, bs);
    uint64_t cnt; memcpy(&cnt, rec + (uint64_t)nw * sizeof(uint64_t), sizeof cnt);
    if (!cnt) continue;
    tot += (double)cnt;
    for (uint32_t a = 0; a < ns; ++a)
      for (uint32_t b = a + 1; b < ns; ++b)
        if ((bs[a] == '1') != (bs[b] == '1')) {
          d[(size_t)a * ns + b] += (double)cnt;
          d[(size_t)b * ns + a] += (double)cnt;
        }
  }
  free(bs);
  if (tot > 0.0)
    for (size_t i = 0; i < (size_t)ns * ns; ++i) d[i] /= tot;
  return d;
}

/* A class name as it appears in a set NAME: the '@' separator and the dotted
 * parent rule both have to survive it, so anything else becomes '_'. */
static void sat_tag(const char *in, char *out, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; in[i] && j + 1 < cap; ++i) {
    char c = in[i];
    out[j++] = ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9')) ? c : '_';
  }
  out[j] = '\0';
}

/* Overlapping closest-N pairs of a childless node, each built as its own
 * 2-class set and appended to the same chain under the name <node>@<a>__<b>.
 *
 * OVERLAPPING, not a partition: the 20260806 measurement found a WPGMA
 * partition left 73.5% of the remaining error in pairs no satellite covered,
 * because a partition cannot cover a close pair straddling a block boundary.
 * Here a class joins as many pairs as name it -- on the mouse 33-class leaf
 * PAL-Inh lands in 20 of 135, which no partition can express.
 *
 * The point is the CpG budget, not the pairing: a node's pattern must hold
 * across every class it carries, so a wide node's filter is a conjunction that
 * starves exactly the pairs needing help. Rebuilt over 2 classes the same
 * contrast is far thicker -- MGE-Sst/PAL-Inh goes from 10 CpGs on the side a
 * rank column needs to 3,034. */
/* ---- per-pair LOO calibration (--cell-store / --cell-labels) ----------
 *
 * File-scope because the alternative is threading four more parameters
 * through tree_build -> tree_satellites -> tree_thin_pair for a feature
 * that is one optional table and a store path. Set once in main, read-only
 * during the build. */
/* ---- annealed split threshold (--anneal-min-seg) ----------------------
 * Redone at EVERY node: try HI first, and when nothing separates, drop to
 * the largest threshold at which the single-linkage graph disconnects (the
 * MST bottleneck minus one) -- the slowest possible peel -- stopping at LO.
 * A non-zero STEP quantizes that drop downward to the grid HI, HI-STEP, ...:
 * near-simultaneous disconnections then land in ONE split instead of a
 * cascade of near-duplicate single-peel levels, and the recorded thresholds
 * are round, fold-comparable values rather than jittery bottleneck-1s.
 * File-scope for the same reason as the calibration state below. */
static uint64_t g_anneal_hi = 0, g_anneal_lo = 0;   /* 0 = annealing off */
static uint64_t g_anneal_step = 0;                  /* 0 = continuous */

/* ---- BANK mode (--bank): the two-type flat artifact -------------------
 * Type 1: each SPLIT node's binstrings, pruned to patterns carrying at
 * least --min-pattern-cpgs CpGs -- tree-named for provenance, but nothing
 * routes. Type 2: calibrated 2-class rebuilds, ALL named root@A__B: every
 * terminal pair (a 2-class split group, or any pair inside an unsplittable
 * leaf, which emits no type-1 block of its own) plus every cross-split
 * pair whose seg support at the node where its classes part falls below
 * the --resolver-quantile cutoff (capped at the anneal HI, so a pair
 * separated at full stringency never gets a resolver). Featurize the
 * chain with --satellite-contrast replace and train with
 * classify-train --pool-nodes: pattern columns from type 1, one anchored
 * contrast per type-2 set, no routing anywhere. */
static int g_bank = 0;
static uint32_t g_min_pattern_cpgs = 2000;   /* type-1 builds only */
/* ONE knob spans the release family: --resolvers N|all. Every pair is
 * registered at the split where its classes part; the mandatory ones
 * (2-class groups, unsplittable leaves, floor failures) always get a
 * resolver, and the rest are ranked by their hard-block FOOTPRINT
 * (bank_footprint_pick, below) so the N thinnest pairs in total get one.
 * Default 0 = the mandatory pairs only; -1 = all = bank-full, the
 * expensive one, so it is asked for. This replaced --resolver-gate (a pair was judged
 * resolved when ONE type-1 pattern at ONE node separated it with >= N
 * CpGs): on fold 0 the gate's picks overlapped the pairs a held-out error
 * search wanted by almost nothing, and every excess error of the gated
 * lite was on a gated-out pair (20260913 entry). */
static int64_t g_resolvers = 0;
static const char *g_resolver_cache = NULL;  /* --resolver-cache DIR */
static uint32_t g_stride_k = 0, g_stride_n = 0;  /* --resolver-stride k/N */
/* mand: gets a resolver. never: the tree never parts the pair (a 2-class
 * split group, an unsplittable leaf, a floor failure) -- recorded so the
 * build can say how many such pairs the chosen resolvers cover. */
typedef struct { char *a, *b; uint64_t sup; int mand, never; } bank_pair_t;
static bank_pair_t *g_bp = NULL;
static int bank_pair_cmp(const void *x, const void *y) {
  const bank_pair_t *p = (const bank_pair_t *)x, *q = (const bank_pair_t *)y;
  int c = strcmp(p->a, q->a);
  return c ? c : strcmp(p->b, q->b);
}
static uint32_t g_bp_n = 0, g_bp_cap = 0;

static void bank_pair_add(const char *a, const char *b, uint64_t sup,
                          int mand, int never) {
  for (uint32_t i = 0; i < g_bp_n; ++i)
    if ((!strcmp(g_bp[i].a, a) && !strcmp(g_bp[i].b, b)) ||
        (!strcmp(g_bp[i].a, b) && !strcmp(g_bp[i].b, a))) {
      if (mand) g_bp[i].mand = 1;
      if (never) g_bp[i].never = 1;
      if (sup && (!g_bp[i].sup || sup < g_bp[i].sup)) g_bp[i].sup = sup;
      return;
    }
  if (g_bp_n == g_bp_cap) {
    g_bp_cap = g_bp_cap ? g_bp_cap * 2 : 256;
    g_bp = realloc(g_bp, (size_t)g_bp_cap * sizeof(bank_pair_t));
    if (!g_bp) die("out of memory (bank pairs)", NULL);
  }
  g_bp[g_bp_n].a = strdup(a); g_bp[g_bp_n].b = strdup(b);
  g_bp[g_bp_n].sup = sup; g_bp[g_bp_n].mand = mand;
  g_bp[g_bp_n].never = never;
  ++g_bp_n;
}

/* ---- --resolvers N: the footprint pick --------------------------------
 * Which cross pairs get a calibrated resolver is decided from the
 * reference alone. A pair's FOOTPRINT is the number of hard-block CpGs --
 * the union of every type-1 block's pattern CpGs, i.e. what the classifier
 * looks at before any resolver -- at which the two class pseudobulks are
 * both covered and differ by more than 0.5 in beta. At 4,096 sampled CpGs
 * a cell hits ~0.02% of the row space, so a 10k footprint is ~2 usable
 * CpGs and an 80k one ~16: the thinnest footprints are the pairs the hard
 * blocks cannot part in a sparse cell, and those get the resolvers. The N
 * thinnest pairs are marked, nothing else: the pairs the tree never parts
 * (a 2-class split group, an unsplittable leaf) have no separating CpG
 * by construction and rank at the very top on their own -- mouse ranks
 * 1, 2, 7, 8 of 820; all 15 human ones within the top 16 of 1,953 -- so
 * they need no special case, only a warning if N is so small that one is
 * left out (the model then has NO feature for that pair). --resolvers 0
 * is hard blocks only: no calibration, no cell store needed.
 *
 * Measured before adopting (fold 0, 20260913): the 100 thinnest pairs
 * overlap the 100 an inner-validation error search picked by 69/100
 * (mouse) and 58/100 (human) and train to within 0.5 point of them, where
 * a pattern-level count (whole patterns by binstring bit, max over sets)
 * managed 21/100 and the retired gate almost none: the pair needs a
 * per-CpG contrast against the actual betas, not a node summary. The 0.5
 * threshold and the both-covered rule are constants on purpose -- the
 * recipe has no knob a user should tune. */
static uint64_t *g_fp_key = NULL;         /* sort key: footprint per pair */
static int fp_cmp(const void *x, const void *y) {
  uint32_t i = *(const uint32_t *)x, j = *(const uint32_t *)y;
  if (g_fp_key[i] != g_fp_key[j]) return g_fp_key[i] < g_fp_key[j] ? -1 : 1;
  int c = strcmp(g_bp[i].a, g_bp[j].a);
  return c ? c : strcmp(g_bp[i].b, g_bp[j].b);
}
static void bank_footprint_pick(const char *store, char *const *slab,
                                const int64_t *voff, uint32_t ns,
                                const treeout_t *t, FILE *rep) {
  uint64_t want = g_resolvers < 0 ? g_bp_n : (uint64_t)g_resolvers;
  if (want > g_bp_n) want = g_bp_n;
  /* mand is set by rank alone from here; never-parted pairs rank at the
   * top by themselves and are only counted for the report */
  uint32_t nnever = 0;
  for (uint32_t i = 0; i < g_bp_n; ++i) {
    nnever += g_bp[i].never != 0; g_bp[i].mand = 0;
  }
  if (!want || !t->n) {
    if (!t->n)
      fprintf(rep, "\n[methscope] bank resolvers: no hard block to rank "
              "on; none built\n");
    else
      fprintf(rep, "\n[methscope] bank resolvers: none (--resolvers 0): "
              "hard blocks only\n");
    fprintf(rep, "  %u pair(s) the tree never parts (no type-1 pattern "
            "separates them), 0 covered by a resolver\n", nnever);
    if (nnever)
      fprintf(rep, "  WARNING: the model has NO feature for those pairs; "
              "give --resolvers\n");
    return;
  }
  /* 1. the hard-block CpG union */
  const uint64_t n_cpg = ((const mrmp_header_t *)t->img[0])->n_cpg;
  uint8_t *in = xcalloc(n_cpg, 1, "footprint mask");
  for (uint32_t k = 0; k < t->n; ++k) {
    const char *img = (const char *)t->img[k];
    const mrmp_header_t *h = (const mrmp_header_t *)(const void *)img;
    if (h->n_cpg != n_cpg) die("hard blocks disagree on the row space", NULL);
    uint32_t *owned = NULL; const uint32_t *m;
    if (h->flags & MRMP_FLAG_MEMB_RLE)
      m = owned = memb_decompress((const uint8_t *)(img + h->membership_offset),
                                  h->membership_bytes, n_cpg, h->n_candidates,
                                  (h->flags & MRMP_FLAG_MEMB_BGZF) != 0,
                                  "footprint membership");
    else m = (const uint32_t *)(const void *)(img + h->membership_offset);
    for (uint64_t i = 0; i < n_cpg; ++i)
      if (m[i] != MRMP_PNA_MEMBERSHIP) in[i] = 1;
    free(owned);
  }
  uint64_t nrow = 0;
  for (uint64_t i = 0; i < n_cpg; ++i) nrow += in[i];
  uint64_t *row = xcalloc(nrow ? nrow : 1, sizeof(uint64_t), "footprint rows");
  for (uint64_t i = 0, j = 0; i < n_cpg; ++i) if (in[i]) row[j++] = i;
  free(in);
  /* 2. every class's beta at those rows (NAN = uncovered) */
  float *beta = xcalloc((size_t)nrow * ns + 1, sizeof(float), "footprint betas");
  cfile_t cf = open_cfile((char *)store);
  for (uint32_t k = 0; k < ns; ++k) {
    cdata_t c = read_record_at(&cf, voff[k], slab[k]);
    if (c.n != n_cpg) die("reference and hard blocks disagree on rows", slab[k]);
    for (uint64_t j = 0; j < nrow; ++j) {
      uint64_t mu = f3_get_mu(&c, row[j]);
      beta[j * ns + k] = (mu && MU2cov(mu)) ? (float)MU2beta(mu) : NAN;
    }
    free_cdata(&c);
  }
  bgzf_close(cf.fh);
  free(row);
  /* 3. per pair: rows both covered and > 0.5 apart */
  uint64_t *fp = xcalloc((size_t)ns * ns, sizeof(uint64_t), "footprints");
  for (uint64_t j = 0; j < nrow; ++j) {
    const float *b = beta + j * ns;
    for (uint32_t a = 0; a < ns; ++a) {
      if (isnan(b[a])) continue;
      for (uint32_t c2 = a + 1; c2 < ns; ++c2)
        if (!isnan(b[c2]) && fabsf(b[a] - b[c2]) > 0.5f) ++fp[(size_t)a * ns + c2];
    }
  }
  free(beta);
  /* 4. rank the registry, mark the N thinnest */
  g_fp_key = xcalloc(g_bp_n ? g_bp_n : 1, sizeof(uint64_t), "pair footprints");
  for (uint32_t i = 0; i < g_bp_n; ++i) {
    uint32_t ia = UINT32_MAX, ib = UINT32_MAX;
    for (uint32_t k = 0; k < ns; ++k) {
      if (!strcmp(slab[k], g_bp[i].a)) ia = k;
      if (!strcmp(slab[k], g_bp[i].b)) ib = k;
    }
    if (ia == UINT32_MAX || ib == UINT32_MAX)
      die("bank pair names a class missing from the store", g_bp[i].a);
    if (ia > ib) { uint32_t x = ia; ia = ib; ib = x; }
    g_fp_key[i] = fp[(size_t)ia * ns + ib];
  }
  free(fp);
  uint32_t *ord = xcalloc(g_bp_n ? g_bp_n : 1, sizeof(uint32_t), "pair order");
  for (uint32_t i = 0; i < g_bp_n; ++i) ord[i] = i;
  qsort(ord, g_bp_n, sizeof(uint32_t), fp_cmp);
  uint64_t median = g_bp_n ? g_fp_key[ord[g_bp_n / 2]] : 0;
  for (uint32_t r = 0; r < want; ++r) g_bp[ord[r]].mand = 1;
  char b1[32], b2[32], b3[32];
  fprintf(rep, "\n[methscope] bank resolvers: the %" PRIu64 " of %u pairs "
          "with the thinnest footprint (hard-block CpGs both covered and "
          "> 0.5 apart; %s rows, median pair %s)\n", want, g_bp_n,
          commafmt_local(nrow, b1), commafmt_local(median, b2));
  for (uint32_t r = 0; r < want; ++r) {
    const bank_pair_t *q = &g_bp[ord[r]];
    fprintf(rep, "  %10s  %s | %s%s\n", commafmt_local(g_fp_key[ord[r]], b3),
            q->a, q->b, q->never ? "   (never parted)" : "");
  }
  uint32_t covered = 0;
  for (uint32_t i = 0; i < g_bp_n; ++i) covered += g_bp[i].never && g_bp[i].mand;
  fprintf(rep, "  %u pair(s) the tree never parts (no type-1 pattern "
          "separates them), %u covered by a resolver\n", nnever, covered);
  if (covered < nnever)
    fprintf(rep, "  WARNING: %u of them got no resolver -- the model has "
            "NO feature for those pairs; raise --resolvers\n",
            nnever - covered);
  free(ord); free(g_fp_key); g_fp_key = NULL;
}

static const char *g_cal_store = NULL;   /* per-cell fmt3 .cg (with .idx) */
static float g_cal_eps = 0.02f;          /* tolerance below best LOO macro */
static uint32_t g_cal_threads = 8;
static char **g_cal_cell = NULL, **g_cal_cls = NULL;  /* parallel arrays */
static uint32_t g_cal_n = 0;

static void calib_load_labels(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) die("cannot open --cell-labels", path);
  char *line = NULL; size_t cap = 0; ssize_t len;
  uint32_t alloc = 1024;
  g_cal_cell = xcalloc(alloc, sizeof(char *), "calib cells");
  g_cal_cls = xcalloc(alloc, sizeof(char *), "calib classes");
  while ((len = getline(&line, &cap, f)) > 0) {
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = '\0';
    if (!len) continue;
    char *tab = strchr(line, '\t');
    if (!tab || !tab[1]) die("--cell-labels wants cell<TAB>class", line);
    *tab = '\0';
    if (g_cal_n == alloc) {
      alloc <<= 1;
      g_cal_cell = realloc(g_cal_cell, (size_t)alloc * sizeof(char *));
      g_cal_cls = realloc(g_cal_cls, (size_t)alloc * sizeof(char *));
      if (!g_cal_cell || !g_cal_cls) die("out of memory", path);
    }
    g_cal_cell[g_cal_n] = strdup(line);
    g_cal_cls[g_cal_n] = strdup(tab + 1);
    ++g_cal_n;
  }
  free(line); fclose(f);
  if (!g_cal_n) die("--cell-labels is empty", path);
}

/* Override sel's (shrink_pseudocnt, min_sbeta_gap) with the pair's own
 * LOO-calibrated values. Returns 1 when the override happened; on any skip
 * (no calibration configured, no labelled cells, calibration declined) the
 * caller's sel is untouched and the build proceeds on defaults. */
static int calib_pair(char *const *lab, ms_select_opt_t *sel,
                      const char *sname, int tty, FILE *rep) {
  if (!g_cal_store) return 0;
  uint32_t nA = 0, nB = 0;
  for (uint32_t k = 0; k < g_cal_n; ++k) {
    if (!strcmp(g_cal_cls[k], lab[0])) ++nA;
    else if (!strcmp(g_cal_cls[k], lab[1])) ++nB;
  }
  if (!nA || !nB) {
    fprintf(rep, "  %s. %s: no labelled cells for %s; defaults kept\n",
            tty ? "\r\033[K" : "", sname, nA ? lab[1] : lab[0]);
    return 0;
  }
  char **ca = xcalloc(nA, sizeof(char *), "calib cells A");
  char **cb = xcalloc(nB, sizeof(char *), "calib cells B");
  nA = nB = 0;
  for (uint32_t k = 0; k < g_cal_n; ++k) {
    if (!strcmp(g_cal_cls[k], lab[0])) ca[nA++] = g_cal_cell[k];
    else if (!strcmp(g_cal_cls[k], lab[1])) cb[nB++] = g_cal_cell[k];
  }
  double a, G, v; uint32_t nc; uint64_t nadm; int grid = 0;
  { char m[192];
    snprintf(m, sizeof m, "[%.160s] calibrating", sname);
    spin_start(tty, m); }
  int ok = ms_pair_calibrate(g_cal_store, ca, nA, cb, nB, g_cal_eps,
                             g_cal_threads, &a, &G, &v, &nc, &nadm, &grid);
  spin_stop();
  free(ca); free(cb);
  if (!ok) return 0;
  sel->shrink_pseudocnt = (float)a;
  sel->min_sbeta_gap = (float)G;      /* gap form replaces the band */
  char b1[32];
  fprintf(rep, "  %s= calibrated %s: shrink %g, gap %.3f "
          "(LOO macro %.3f over %u cells; admits %s CpGs)%s\n",
          tty ? "\r\033[K" : "", sname, a, G, v, nc,
          commafmt_local(nadm, b1), grid ? " [full grid]" : "");
  fflush(rep);
  return 1;
}

/* THIN-PAIR relaxation: a 2-class leaf whose admitted pool is under
 * `relax_below` gets ONE soft satellite over the same pair, selected by
 * shrunk-beta GAP ORDER (--min-sbeta-gap) instead of the positional band --
 * coverage over purity, confined to soft columns the booster can weigh.
 *
 * Why: the band's anchors cost the Tnaive CD4/CD8 pair ~8x its evidence
 * (5,972 CpGs at gap >= 0.40 vs 733 admitted), and the recovered sites are
 * deep (3,730 at >= 10 reads both classes) and two-sided. Measured on held
 * -out cells, the single gap-selected sign feature matched the whole node
 * (test 87-88%), peaking at gap 0.35-0.40; 0.30 is a trap (train-test gap
 * -29 points on CD8 -- loose admission re-couples selection to the
 * reference). Hence the 0.35 default and a floor rather than always-on:
 * where the band already feeds a pair richly, relaxation only dilutes. */
static uint32_t tree_thin_pair(const char *store, uint32_t n,
                               char *const *lab, const int64_t *vo,
                               const mrmp_header_t *gh,
                               const ms_select_opt_t *sel, const char *name,
                               uint64_t n_kept, uint64_t relax_below,
                               float relax_gap, int tty, treeout_t *out,
                               FILE *rep) {
  if (n != 2 || !relax_below || n_kept >= relax_below) return 0;
  ms_select_opt_t rs = *sel;
  rs.min_sbeta_gap = relax_gap;          /* replaces the band */
  char sname[256];
  snprintf(sname, sizeof sname, "%s@gap", name);
  calib_pair(lab, &rs, sname, tty, rep); /* pair-specific (a, G) if enabled */
  subset_block_t s2;
  { char m[192]; snprintf(m, sizeof m, "[%.180s]", sname); spin_start(tty, m); }
  build_subset_block(store, 2, (char *const *)lab, (const int64_t *)vo, gh,
                     &rs, sname, &s2);
  spin_stop();
  tree_push(out, s2.img, s2.bytes, sname);
  char b1[32], b2[32];
  fprintf(rep, "  %s+ thin-pair satellite %s: %s CpGs (gap >= %.2f; node had %s)\n",
          tty ? "\r\033[K" : "", sname, commafmt_local(s2.n_kept, b1),
          relax_gap, commafmt_local(n_kept, b2));
  return 1;
}

static uint32_t tree_satellites(const char *store, const subset_block_t *sb,
                               uint32_t n, char *const *lab,
                               const int64_t *vo, const mrmp_header_t *gh,
                               const ms_select_opt_t *sel, const char *name,
                               uint32_t n_partner, float relax_gap, int tty,
                               treeout_t *out, FILE *rep) {
  /* A 2-class node needs no satellite: its only pair IS its own class pair, so
   * the satellite rebuilds the identical MRMP and hands the booster a second
   * copy of the column it already has. Seen on the human tree, where
   * root.1.0.9.1 (T.Cell.CD4/CD8, 2 patterns, 6,418 CpGs) generated a
   * satellite with exactly 2 patterns and 6,418 CpGs. At 3+ classes they are
   * genuinely different -- the same node's Dendritic/Macrophage/Monocyte
   * satellites carry 4,277-10,270 CpGs against the 3-class set's 5,798,
   * because dropping the third class admits CpGs its filter had spoiled. */
  /* Overlapping closest-N pairs, not a partition of the node's classes. A
   * partition puts each class in at most one block, and a cap on block size then
   * truncates -- measured on a 10-fold mouse arm, three quarters of the residual
   * error sat in confusion pairs no block covered, because two classes fell out
   * of the vocabulary entirely and one pair was split across two blocks. Here a
   * class appears in as many sets as it has close neighbours, so every pair that
   * matters is reachable.
   *
   * Pairs also buy more per column: pattern count grows as 2^N-2 while covered
   * error does not, and a 2-class set spent 2 pooled columns for 7.4% of error
   * where a 6-class one spent 62 for 4.6%. */
  if (n < 3 || !n_partner) return 0;
  /* Satellites are soft, relaxed evidence by design: select by the universal
   * shrunk-beta gap rather than the positional band (see tree_thin_pair for
   * the measurements behind 0.40). */
  ms_select_opt_t rs = *sel;
  rs.min_sbeta_gap = relax_gap;
  sel = &rs;
  double *d = sat_hamming(sb->img);
  uint8_t *want = xcalloc((size_t)n * n, 1, "satellite pairs");
  uint32_t *ord = xcalloc(n, sizeof(uint32_t), "near order");
  for (uint32_t a = 0; a < n; ++a) {
    uint32_t m = 0;
    for (uint32_t b = 0; b < n; ++b) if (b != a) ord[m++] = b;
    /* partial selection sort: N is small, and it keeps ties resolved by index
     * so a rebuild reproduces the same set */
    uint32_t take = n_partner < m ? n_partner : m;
    for (uint32_t i = 0; i < take; ++i) {
      uint32_t best = i;
      for (uint32_t j = i + 1; j < m; ++j) {
        double dj = d[(size_t)a * n + ord[j]], db = d[(size_t)a * n + ord[best]];
        if (dj < db || (dj == db && ord[j] < ord[best])) best = j;
      }
      uint32_t t = ord[i]; ord[i] = ord[best]; ord[best] = t;
      uint32_t x = a < ord[i] ? a : ord[i], y = a < ord[i] ? ord[i] : a;
      want[(size_t)x * n + y] = 1;
    }
  }
  free(ord); free(d);

  uint32_t made = 0;
  for (uint32_t a = 0; a < n; ++a)
    for (uint32_t b = a + 1; b < n; ++b) {
      if (!want[(size_t)a * n + b]) continue;
      char ta[128], tb[128], sname[512];
      sat_tag(lab[a], ta, sizeof ta); sat_tag(lab[b], tb, sizeof tb);
      snprintf(sname, sizeof sname, "%s%c%s__%s", name, MS_SAT_SEP, ta, tb);
      /* sat_tag maps every non-alphanumeric to '_' and truncates at 128, so it
       * is lossy: "IT-L5" and "IT_L5" produce the same tag. Two sets sharing a
       * name in one chain is silently wrong -- a by-name lookup takes the
       * first -- and nothing downstream checks it, so refuse here where the
       * offending pair can be named. */
      for (uint32_t q = 0; q < made; ++q)
        if (!strcmp(sname, out->name[out->n - made + q])) {
          char m[512];
          snprintf(m, sizeof m, "%.200s and an earlier pair of %.100s both "
                   "yield this satellite name; class names must stay distinct "
                   "after non-alphanumerics become '_'", sname, name);
          die(m, lab[a]);
        }
      char *two[2]; int64_t vv[2];
      two[0] = lab[a]; two[1] = lab[b]; vv[0] = vo[a]; vv[1] = vo[b];
      ms_select_opt_t ps = *sel;         /* per-pair (a, G) when enabled */
      calib_pair(two, &ps, sname, tty, rep);
      subset_block_t s2;
      { char m[192];
        snprintf(m, sizeof m, "[%.180s]", sname); spin_start(tty, m); }
      build_subset_block(store, 2, two, vv, gh, &ps, sname, &s2);
      spin_stop();
      tree_push(out, s2.img, s2.bytes, sname);
      ++made;
    }
  free(want);
  if (made) {
    char b1[32];
    fprintf(rep, "  %s+ %u satellite(s) over %s pairs of %s\n",
            tty ? "\r\033[K" : "", made, commafmt_local(made, b1), name);
  }
  return made;
}

static void tree_build(const char *store, char *const *slab, const int64_t *voff,
                       const uint32_t *idx, uint32_t n, const mrmp_header_t *gh,
                       const ms_select_opt_t *sel, const char *name,
                       uint64_t min_seg, uint32_t depth, uint32_t max_depth,
                       int dry, uint32_t sat_n, uint64_t relax_below,
                       float relax_gap, int tty, treeout_t *out,
                       FILE *rep) {
  char **lab = xcalloc(n, sizeof(char *), "node labels");
  int64_t *vo = xcalloc(n, sizeof(int64_t), "node offsets");
  for (uint32_t k = 0; k < n; ++k) { lab[k] = slab[idx[k]]; vo[k] = voff[idx[k]]; }

  char ind[80]; uint32_t w = depth * 2 < 72 ? depth * 2 : 72;
  memset(ind, ' ', w); ind[w] = '\0';

  /* BANK: a 2-class group emits no node -- the pair is a type-2 resolver,
   * built (and calibrated) uniformly after the tree walk. */
  if (g_bank && n == 2) {
    fprintf(rep, "%s%s  %s + %s -> resolver pair\n", tty ? "\r\033[K" : "",
            ind, lab[0], lab[1]);
    if (!dry) bank_pair_add(lab[0], lab[1], 0, 1, 1);
    free(lab); free(vo); return;
  }

  subset_block_t sb;
  /* A 2-class node's own MRMP IS the pair classifier, so it gets the pair's
   * calibrated (a, G) directly -- and a calibrated node needs no thin-pair
   * satellite, which would rebuild the identical selection under @gap. */
  ms_select_opt_t selc; int calibed = 0;
  if (n == 2 && !dry) {
    selc = *sel;
    calibed = calib_pair(lab, &selc, name, tty, rep);
    if (calibed) sel = &selc;
  }
  { char m[192]; snprintf(m, sizeof m, "[%s] %u classes", name, n);
    spin_start(tty, m); }
  build_subset_block(store, n, lab, vo, gh, sel, name, &sb);
  spin_stop();
  /* BANK defers the push: an unsplittable leaf emits nothing (its internal
   * pairs become resolvers; its collective identity is the parent's split
   * binstrings), so a block is only kept once a SPLIT is confirmed. */
  int pushed = 0;
  if (!g_bank) { tree_push(out, sb.img, sb.bytes, name); pushed = 1; }

  char b1[32], b2[32], b3[32], b4[32], b5[32];
  if (n < 2 || depth >= max_depth) {
    fprintf(rep, "%s%s%s  n=%-3u %7s pat %9s CpGs\n", tty ? "\r\033[K" : "",
            ind, name, n, commafmt_local(sb.n_pat, b1),
            commafmt_local(sb.n_kept, b2));
    if (g_bank) {
      if (!dry)
        for (uint32_t a = 0; a < n; ++a)
          for (uint32_t b = a + 1; b < n; ++b)
            bank_pair_add(lab[a], lab[b], 0, 1, 1);
      if (!pushed) free(sb.img);
    } else if (!dry)
      tree_satellites(store, &sb, n, lab, vo, gh, sel, name, sat_n,
                      relax_gap, tty, out, rep);
    free(lab); free(vo); return;
  }

  uint64_t *seg = tree_pair_seg(sb.img);
  uint64_t npair = 0, *sorted = tree_pair_sorted(seg, n, &npair);
  uint32_t *grp = xcalloc(n, sizeof(uint32_t), "grouping");
  /* The threshold this node's split actually uses. Fixed mode: --min-
   * segregating as given. Annealed mode: start at HI, and when nothing
   * separates there, drop exactly to the first disconnection (bottleneck-1)
   * -- the largest splitting threshold, hence the slowest peel -- unless it
   * is under the LO floor, which makes this node a genuine leaf. */
  uint64_t t_eff = min_seg, bneck = 0;
  uint32_t ng;
  if (g_anneal_hi && min_seg != UINT64_MAX) {
    t_eff = g_anneal_hi;
    ng = tree_partition(seg, n, t_eff, grp);
    if (ng < 2) {
      bneck = tree_mst_bottleneck(seg, n);
      if (bneck > g_anneal_lo) {           /* bneck-1 >= LO: split in range */
        t_eff = bneck - 1;
        /* quantize downward to the HI - k*STEP grid (never below LO) */
        if (g_anneal_step && t_eff < g_anneal_hi) {
          t_eff = g_anneal_hi
                - ((g_anneal_hi - t_eff + g_anneal_step - 1) / g_anneal_step)
                  * g_anneal_step;
          if (t_eff < g_anneal_lo) t_eff = g_anneal_lo;
        }
        ng = tree_partition(seg, n, t_eff, grp);
      } else t_eff = g_anneal_lo;          /* leaf; record the floor tried */
    }
  } else ng = tree_partition(seg, n, t_eff, grp);
  /* Record the threshold in the node's own header (the block was already
   * pushed by pointer, so the stored chain sees this too). */
  if (min_seg != UINT64_MAX) {
    mrmp_header_t *hh = (mrmp_header_t *)sb.img;
    hh->split_minseg = t_eff;
    hh->flags |= MRMP_FLAG_MINSEG;
  }
  if (g_bank && ng >= 2) {
    /* emission pruned, discovery not: rebuild this confirmed split with
     * the floor active and store THAT image; seg/partition above came
     * from the full-evidence build. Costs a second streaming pass per
     * split node, which is what keeps the tree's shape independent of
     * the emission floor. */
    subset_block_t sbf;
    { char m[192]; snprintf(m, sizeof m, "[%s] floor", name);
      spin_start(tty, m); }
    g_floor_active = g_min_pattern_cpgs;
    build_subset_block(store, n, lab, vo, gh, sel, name, &sbf);
    g_floor_active = 0;
    spin_stop();
    if (!sbf.img) {
      /* the split is real but nothing clears the floor: no type-1 block
       * at this level, so EVERY cross pair here is a mandatory resolver;
       * children still recurse and may emit their own type-1. Seen on
       * the 33-label bulk myeloid triplet. */
      fprintf(rep, "  %s! no pattern clears the floor at %s: its pairs go "
              "to resolvers\n", tty ? "\r\033[K" : "", name);
      free(sb.img); sb.img = NULL;
      if (!dry)
        for (uint32_t a = 0; a < n; ++a)
          for (uint32_t b = a + 1; b < n; ++b)
            if (grp[a] != grp[b])
              bank_pair_add(lab[a], lab[b], seg[(uint64_t)a * n + b], 1, 1);
    } else {
    if (min_seg != UINT64_MAX) {
      mrmp_header_t *fh2 = (mrmp_header_t *)sbf.img;
      fh2->split_minseg = t_eff;
      fh2->flags |= MRMP_FLAG_MINSEG;
    }
    if (!pushed) { tree_push(out, sbf.img, sbf.bytes, name); pushed = 1; }
    free(sb.img); sb.img = NULL;      /* the unpruned discovery image */
    /* every cross-group pair parts HERE: register it. Under --resolvers
     * all every pair is a resolver outright; under --resolvers N the
     * footprint pick (after the tree walk) decides which of them are. */
    if (!dry)
      for (uint32_t a = 0; a < n; ++a)
        for (uint32_t b = a + 1; b < n; ++b)
          if (grp[a] != grp[b])
            bank_pair_add(lab[a], lab[b], seg[(uint64_t)a * n + b],
                          g_resolvers < 0, 0);
    }                              /* sbf.img != NULL */
  }

  /* Report the node's pairwise separation range and the resulting split. */
  const char *bold = tty ? "\033[1m" : "";
  const char *cyan = tty ? "\033[36m" : "";
  const char *yellow = tty ? "\033[33m" : "";
  const char *reset = tty ? "\033[0m" : "";
  fprintf(rep, "%s%s%s%s  %u classes | %s patterns | %s CpGs\n",
          tty ? "\r\033[K" : "", bold, name, reset, n,
          commafmt_local(sb.n_pat, b1), commafmt_local(sb.n_kept, b2));
  if (dry)
    fprintf(rep, "%s  pair separation: min %s, median %s CpGs\n%s",
            cyan, commafmt_local(sorted[0], b3),
            commafmt_local(sorted[npair / 2], b4), reset);
  /* --flat carries min_seg as UINT64_MAX, which printed as
   * "18,446,744,073,709,551,615 CpGs" -- a number no reader can parse as
   * "there is no threshold". Say that instead. */
  if (min_seg == UINT64_MAX)
    fprintf(rep, "  split threshold: %snone%s (flat) -> %u group(s)\n",
            yellow, reset, ng);
  else if (g_anneal_hi && t_eff != g_anneal_hi)
    fprintf(rep, "  split threshold: %s%s%s CpGs (annealed from %s) -> "
            "%u group(s)\n", yellow, commafmt_local(t_eff, b5), reset,
            commafmt_local(g_anneal_hi, b4), ng);
  else
    fprintf(rep, "  split threshold: %s%s%s CpGs -> %u group(s)\n",
            yellow, commafmt_local(t_eff, b5), reset, ng);

  if (dry) {   /* what a threshold is actually chosen from: how the partition
                * moves as it rises, across this node's own observed pairs */
    /* Across the WHOLE distribution, not its bottom quarter: a deep tree wants
     * a HIGH threshold -- few groups per split, more levels -- and the first
     * version of this swept only to the 0.24 quantile, which hid exactly that
     * regime. */
    static const double QS[] = {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                               0.60, 0.70, 0.80, 0.90, 0.95, 0.98};
    fprintf(rep, "%s%sCandidate split thresholds%s\n", bold, cyan, reset);
    fprintf(rep, "%s  Lower thresholds make more groups; higher thresholds\n"
                "%s  merge more classes.\n%s  %10s %10s %8s %13s\n",
            ind, ind, ind, "percentile", "threshold", "groups", "largest group");
    for (uint32_t q = 0; q < sizeof QS / sizeof *QS; ++q) {
      uint64_t at = (uint64_t)(QS[q] * (double)npair);
      if (at >= npair) at = npair - 1;
      uint64_t t = sorted[at];
      uint32_t *g2 = xcalloc(n, sizeof(uint32_t), "sweep grouping");
      uint32_t ng2 = tree_partition(seg, n, t, g2);
      uint32_t big = 0;
      for (uint32_t a = 0; a < ng2; ++a) {
        uint32_t c2 = 0;
        for (uint32_t k2 = 0; k2 < n; ++k2) c2 += (g2[k2] == a);
        if (c2 > big) big = c2;
      }
      fprintf(rep, "%s  %9.0f%% %10s %8u %13u\n", ind, 100.0 * QS[q],
              commafmt_local(t, b1), ng2, big);
      free(g2);
    }
  }
  if (dry) {
    uint32_t show = n < 12 ? npair < 12 ? (uint32_t)npair : 12 : 12;
    fprintf(rep, "%s%sClosest class pairs by segregating CpGs%s\n"
                "%s  Smaller values mean weaker separation.\n", bold, cyan, reset,
            ind);
    for (uint32_t a = 0, k = 0; a < n && k < show; ++a)
      for (uint32_t b = a + 1; b < n && k < show; ++b)
        if (seg[(uint64_t)a * n + b] <= sorted[show - 1]) {
          fprintf(rep, "%s    %s vs %s: %s CpGs\n", ind, lab[a], lab[b],
                  commafmt_local(seg[(uint64_t)a * n + b], b1));
          ++k;
        }
  }

  if (ng < 2) {   /* nothing here separates them: an irreducible multi-class leaf */
    /* Under --flat this is the ANSWER, not a finding, so do not report 33
     * classes as a failure to split. min_seg == UINT64_MAX only happens there. */
    if (min_seg != UINT64_MAX) {
      const uint32_t show = n < 6 ? n : 6;
      fprintf(rep, "%s  ! unsplittable (%u classes):", ind, n);
      for (uint32_t k = 0; k < show; ++k) fprintf(rep, " %s", lab[k]);
      if (show < n) fprintf(rep, " ... (%u more)", n - show);
      fprintf(rep, "\n");
      if (g_anneal_hi && bneck)
        fprintf(rep, "%s    (anneal floor %s reached; a split needs "
                "threshold %s)\n", ind, commafmt_local(g_anneal_lo, b1),
                commafmt_local(bneck - 1, b2));
    }
    /* This is the node satellites are FOR: the classes it carries are the
     * ones its own evidence cannot separate, and a 2-class rebuild is where
     * the CpGs to separate them come from. */
    if (g_bank) {
      /* no split -> no type-1 block; every internal pair is a resolver */
      if (!dry)
        for (uint32_t a = 0; a < n; ++a)
          for (uint32_t b = a + 1; b < n; ++b)
            bank_pair_add(lab[a], lab[b], 0, 1, 1);
      if (!pushed) free(sb.img);
    } else if (!dry) {
      tree_satellites(store, &sb, n, lab, vo, gh, sel, name, sat_n,
                      relax_gap, tty, out, rep);
      tree_thin_pair(store, n, lab, vo, gh, sel, name, sb.n_kept,
                     calibed ? 0 : relax_below, relax_gap, tty, out, rep);
    }
    free(sorted); free(seg); free(grp); free(lab); free(vo); return;
  }
  for (uint32_t g = 0; g < ng; ++g) {
    uint32_t *sub = xcalloc(n, sizeof(uint32_t), "child idx"), m = 0;
    for (uint32_t k = 0; k < n; ++k) if (grp[k] == g) sub[m++] = idx[k];
    uint64_t gap = UINT64_MAX;
    for (uint32_t h = 0; h < ng; ++h)
      if (h != g) { uint64_t v = tree_group_gap(seg, n, grp, g, h);
                    if (v < gap) gap = v; }
    /* A singleton is already decided by this node's own call; giving it a child
     * MRMP would be a 1-class set, which has no contrast to describe. */
    if (m == 1) {
      fprintf(rep, "%s  - %-40s gap %s\n", ind, slab[sub[0]],
              commafmt_local(gap == UINT64_MAX ? 0 : gap, b1));
      free(sub); continue;
    }
    if (g_bank && m == 2) {
      fprintf(rep, "%s  + %s + %s -> resolver pair (gap %s)\n", ind,
              slab[sub[0]], slab[sub[1]],
              commafmt_local(gap == UINT64_MAX ? 0 : gap, b1));
      if (!dry) bank_pair_add(slab[sub[0]], slab[sub[1]], 0, 1, 1);
      free(sub); continue;
    }
    fprintf(rep, "%s  + %u classes, gap %s\n", ind, m,
            commafmt_local(gap == UINT64_MAX ? 0 : gap, b1));
    fflush(rep);
    if (dry) { free(sub); continue; }
    char cn[256]; snprintf(cn, sizeof cn, "%s.%u", name, g);
    tree_build(store, slab, voff, sub, m, gh, sel, cn, min_seg,
               depth + 1, max_depth, dry, sat_n, relax_below, relax_gap,
               tty, out, rep);
    free(sub);
  }
  /* A 2-class node that split into two singletons decides the pair with its
   * own booster all the same -- it qualifies for the thin-pair relaxation
   * exactly as an unsplittable pair does. */
  if (!dry && !g_bank)
    tree_thin_pair(store, n, lab, vo, gh, sel, name, sb.n_kept,
                   calibed ? 0 : relax_below, relax_gap, tty, out, rep);
  if (g_bank && !pushed) free(sb.img);
  free(sorted); free(seg); free(grp); free(lab); free(vo);
}

int main_mrmp_build(int argc, char *argv[]) {
  g_cmd = "mrmp-build";
  if (argc == 1) { char *h[2]; h[0] = argv[0]; h[1] = (char *)"-h";
                   (void)main_mrmp_build(2, h); return 1; }
  const char *pos[2] = {NULL, NULL}, *setname = "root";
  const char *cal_labels = NULL;
  int npos = 0, force = 0, flat = 0;
  uint32_t top = 10000;              /* --flat only: keep the top N by CpG count */
  int top_given = 0;
  const int dry = 0;                 /* the --dry-run mode is retired */
  /* Satellites are retired with the routing trees: the bank's resolvers
   * replaced them. The recursion still carries the parameters, pinned off. */
  const uint32_t sat_n = 0; const uint64_t relax_below = 0; const float relax_gap = 0.40f;
  uint64_t min_seg = 20000; uint32_t max_depth = 16;
  ms_select_opt_t sel; ms_select_defaults(&sel);
  sel.quiet = 1;                     /* one line per node, not per selection */
  mrmp_header_t gh; memset(&gh, 0, sizeof gh);
  gh.mincov = MRMP_DEF_MINCOV;       gh.beta_threshold = MRMP_DEF_BETA_THRESH;
  gh.max_ambig_frac = MRMP_DEF_MAX_AMBIG; gh.min_major_fold = MRMP_DEF_MIN_FOLD;
  /* majority is the default because every model shipped so far was built
   * that way -- it was the only behaviour -- so a rebuild reproduces them. */
  gh.flags |= mrmp_impute_flags(MRMP_IMPUTE_MAJORITY);
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stdout,
        "Usage: methscope mrmp-build [options] REF.cg OUT.mrmp\n"
        "       methscope mrmp-build --bank [options] REF.cg OUT.mrmp\n\n"
        "Build an MRMP from a labelled reference store: one record per class,\n"
        "record name = class name. The default is ONE set over every class,\n"
        "for upscale, the violation rule and export; --bank is the classifier\n"
        "artifact (the release models), a chain of blocks from an annealed\n"
        "split hierarchy plus calibrated 2-class resolvers.\n\n"
        "Workflow\n"
        "  methscope mrmp-build --top 500 REF.cg OUT.mrmp\n"
        "  methscope mrmp-build --bank --anneal-min-seg 10000,3000,1000 \\\n"
        "      --resolvers 100 --cell-store CELLS.cg --cell-labels CELLS.tsv \\\n"
        "      REF.cg OUT.mrmp\n\n"
        "Split discovery (the hierarchy behind a bank's binstring blocks)\n"
        "  --anneal-min-seg HI,LO[,STEP]\n"
        "                        Anneal the split threshold PER NODE:\n"
        "                        try HI, and when nothing separates,\n"
        "                        drop exactly to the first disconnection of the\n"
        "                        single-linkage graph (the largest threshold\n"
        "                        that splits anything -- the slowest possible\n"
        "                        peel), stopping at LO. A node unsplittable at\n"
        "                        LO is a leaf. Each node re-anneals from HI, so\n"
        "                        the root peels only its best-separated groups\n"
        "                        while deep leaves can still split fine\n"
        "                        structure. A STEP quantizes the drop to the\n"
        "                        grid HI, HI-STEP, ...: near-tied\n"
        "                        disconnections land in one split instead of a\n"
        "                        cascade of near-duplicate levels, and the\n"
        "                        recorded thresholds are round numbers. Every\n"
        "                        node records the threshold it used in its\n"
        "                        header (inspect --tree shows it). Required by\n"
        "                        --bank; a fixed threshold is HI,HI.\n"
        "  --max-depth N         Maximum split depth. Default: 16.\n"
        "\n"
        "Modes\n"
        "  --top N               Default mode only: keep the N patterns with the most\n"
        "                        CpGs, the rest folded into PNA. Default:\n"
        "                        10000; 0 keeps every pattern. The build ranks\n"
        "                        every candidate; this is the consumer's cut,\n"
        "                        applied here because a flat set has one\n"
        "                        consumer -- the same prune as mrmp-pool\n"
        "                        --pooled-top, for chains that must compete.\n"
        "  --bank                The classifier artifact: a FLAT chain of two\n"
        "                        feature types and no routing. Type 1: each split\n"
        "                        node's binstrings, pruned to patterns with\n"
        "                        >= --min-pattern-cpgs CpGs (tree-named for\n"
        "                        provenance only). Type 2: calibrated 2-class\n"
        "                        rebuilds named root@A__B -- every terminal\n"
        "                        pair, every pair inside an unsplittable leaf\n"
        "                        (which emits no type-1 of its own), and every\n"
        "                        cross-split pair --resolvers picks.\n"
        "                        Needs --anneal-min-seg and the calibration\n"
        "                        inputs. Featurize the result with\n"
        "                        --satellite-contrast replace and train with\n"
        "                        classify-train --data.\n"
        "  --min-pattern-cpgs N  A type-1 pattern must carry at least N CpGs\n"
        "                        to stay in its block; smaller ones fold into\n"
        "                        the background. Discovery is unaffected (it\n"
        "                        runs on full evidence), so the tree and the\n"
        "                        resolvers are the same at any N; only the\n"
        "                        hard blocks get fatter. Default 2000; 500\n"
        "                        is the release models' value.\n"
        "  --resolvers N|all     How many pairs get a calibrated resolver.\n"
        "                        Default 0: none, hard blocks only (no\n"
        "                        calibration, so no cell store needed).\n"
        "                        all = every pair, the bank-full model.\n"
        "                        N = the N thinnest pairs by hard-\n"
        "                        block FOOTPRINT -- the hard-block CpGs at\n"
        "                        which the two class pseudobulks are both\n"
        "                        covered and > 0.5 apart in beta -- the\n"
        "                        pairs a sparse cell cannot tell apart from\n"
        "                        the type-1 columns alone. Pairs the tree\n"
        "                        never parts rank at the top by themselves;\n"
        "                        the log warns if N leaves one out. 100 is\n"
        "                        the bank-lite release model. The build log\n"
        "                        lists every chosen pair with its footprint.\n"
        "  --resolver-cache DIR  Reuse resolver blocks across runs: before\n"
        "                        building root@A__B, look for DIR/A__B.mrmp\n"
        "                        (sanitized tags) and splice it in; else build\n"
        "                        it and save it there. A block embodies its\n"
        "                        calibration, so bank-full and bank-lite on\n"
        "                        the same reference + cells share one cache,\n"
        "                        and a rerun rebuilds nothing. The cache is\n"
        "                        valid ONLY for one (reference, cell store,\n"
        "                        labels, selection) combination -- point\n"
        "                        different builds at different directories.\n"
        "  --resolver-stride k/N Populate the cache in parallel: build only\n"
        "                        the resolver pairs with ordinal == k mod N\n"
        "                        (0-based); an uncached out-of-stride pair is\n"
        "                        skipped and the artifact is PARTIAL. Run N\n"
        "                        array jobs with the same command and\n"
        "                        k = 0..N-1, then one final run without the\n"
        "                        stride to assemble everything from cache.\n"
        "                        Requires --resolver-cache.\n\n"
        "Per-pair calibration\n"
        "  --cell-store F.cg     Per-cell fmt3 store (with F.cg.idx) holding\n"
        "                        the training cells. With --cell-labels, every\n"
        "                        2-class resolver gets its own\n"
        "                        (shrink-pseudocnt, gap) by leave-one-cell-out:\n"
        "                        selection pools drop the held-out cell (no\n"
        "                        leak, and the inner pool depth matches the\n"
        "                        final build), the anchored sign feature\n"
        "                        scores it, and the LEAST RESTRICTIVE grid\n"
        "                        cell within --calib-eps of the best LOO\n"
        "                        macro wins -- looser admission keeps more\n"
        "                        CpGs for sparse queries. Grid: pseudocount\n"
        "                        1-14, gap 0.28-0.53. Validated on fold-0:\n"
        "                        13 pairs, mean 2.8 points off the per-pair\n"
        "                        oracle, worst 10.7.\n"
        "  --cell-labels F.tsv   cell<TAB>class, class names as in REF.cg.\n"
        "                        Cells missing from the store index are\n"
        "                        skipped with a note.\n"
        "  --calib-eps E         Tolerance below the best LOO macro when\n"
        "                        picking the loosest grid cell. Default: 0.02.\n"
        "  --calib-threads N     Threads per calibration pass. Default: 8.\n\n"
        "Binstring: which pattern a CpG belongs to\n"
        "  --beta-threshold B    Call beta > B methylated. Default: 0.5.\n"
        "  --call-mindepth N     Reads a class needs at a CpG to be CALLED\n"
        "                        there; below it the class is ambiguous (see\n"
        "                        below). Per class per CpG, not a class\n"
        "                        filter. Default: 1. (was --mincov)\n"
        "  --include-all-0       Keep the all-unmethylated pattern.\n"
        "  --include-all-1       Keep the all-methylated pattern.\n"
        "  Ambiguous classes. A class is ambiguous at a CpG when it is below\n"
        "  --call-mindepth or sits exactly at --beta-threshold. Such a class\n"
        "  is imputed, by default from the majority call of the classes that\n"
        "  ARE confident at that CpG. --impute-ambiguous chooses the strategy,\n"
        "  `none` making the CpG PNA instead -- the strict reading, since a\n"
        "  binstring otherwise states something about a class with no\n"
        "  evidence behind it. The rest of this block tunes imputation and\n"
        "  means nothing under `none`.\n"        "  --impute-ambiguous S  How to fill an ambiguous class:\n"
        "                          none      the CpG is PNA\n"
        "                          majority  the majority call of the\n"
        "                                    confidently called classes AT\n"
        "                                    THAT CpG (more 1s -> 1, more\n"
        "                                    0s -> 0, exact tie -> 0)\n"
        "                                    [default]\n"
        "                          zero      always unmethylated\n"
        "                          one       always methylated\n"
        "                          random    an unbiased coin per (CpG,\n"
        "                                    class), from --impute-seed\n"
        "                        A nearest-neighbour strategy -- fill from\n"
        "                        the CpGs around it rather than the classes\n"
        "                        beside it -- is not implemented.\n"
        "  --impute-seed N       Seed for `random` (default 1). The draw is\n"
        "                        a hash of the seed, the CpG and the class,\n"
        "                        so it does not depend on row order or\n"
        "                        thread count. NOT stored in the artifact:\n"
        "                        keep the build command.\n"
        "  --min-major-fold F    `majority` only: trust the fill just where\n"
        "                        the majority sweeps, the larger side being\n"
        "                        >= F times the smaller (unanimous always\n"
        "                        qualifies). Otherwise the CpG is PNA after\n"
        "                        all, because filling from a near-even split\n"
        "                        fabricates calls. Default: 10. 0 fills\n"
        "                        whatever the majority is.\n"
        "  --max-ambig-frac F    PNA a CpG whose ambiguous classes exceed\n"
        "                        this fraction, whatever the strategy says.\n"
        "                        Default: 1.0 (off).\n"
        "\n"
        "Feature selection: which CpGs back a pattern\n"
        "  --call-band LO,HI     Keep a CpG only if every 0-class reads <= LO\n"
        "                        and every 1-class >= HI, on shrunk betas.\n"
        "                        Default: 0.30,0.70. A class with reads is\n"
        "                        tested even where its digit was imputed: an\n"
        "                        intermediate measurement is a real answer,\n"
        "                        and a band failure is never tolerated. `none`\n"
        "                        (LO == HI == --beta-threshold) filters\n"
        "                        nothing, which is the way to see a set before\n"
        "                        and after the band. (was --qfilter)\n"
        "  --shrink-pseudocnt A  Shrink the per-class beta to (M+A)/(M+U+2A)\n"
        "                        for BOTH the admission test and the ranking.\n"
        "                        In Bayesian terms this is the posterior mean\n"
        "                        under a Beta(A,A) prior: A is the PRIOR\n"
        "                        INERTIA, A pseudo-observations on each side\n"
        "                        pulling the estimate toward 0.5 until real\n"
        "                        votes outweigh them. Default: 3 -- a\n"
        "                        unanimous site needs depth 4 to clear the\n"
        "                        0.70 leg, one dissenting read pushes that to\n"
        "                        ~8. Without it a single read scores beta\n"
        "                        exactly 0 or 1, passing admission more easily\n"
        "                        than well-measured evidence and taking the\n"
        "                        maximum rank. 0 restores raw betas.\n"
        "  --feature-mindepth N  Reads a class needs at a CpG for its beta to\n"
        "                        count as EVIDENCE there. A class below it is\n"
        "                        not tested. Default: --call-mindepth; a class\n"
        "                        that could not be called is not evidence.\n"
        "                        (was --min-cg-depth)\n"
        "  --max-lowdepth-frac F Fraction of classes allowed below\n"
        "                        --feature-mindepth before the CpG is dropped\n"
        "                        as a feature. Default: 0. This is the only\n"
        "                        tolerance: untested classes may be waived,\n"
        "                        tested-and-failed ones may not.\n"
        "                        (was --max-frac-na)\n"
        "  --delta-mean-top N    Keep at most N CpGs per binstring, ranked by\n"
        "                        mean class gap. Default: 20000; 0 keeps all.\n"
        "\n"
        "Output\n"
        "  --name NAME           Root name. Default: root.\n"
        "  --force               Overwrite OUT.mrmp.\n\n"
        "Inspect the result with: inspect --tree OUT.mrmp\n");
      return 0;
    }
    else if (!strcmp(a, "--bank")) {
      g_bank = 1;
      if (i + 1 < argc && (!strcmp(argv[i + 1], "full") ||
                           !strcmp(argv[i + 1], "lite")))
        die("--bank full|lite is retired; say --resolvers all or "
            "--resolvers N", argv[i + 1]);
    }
    else if (!strcmp(a, "--min-pattern-cpgs") && i + 1 < argc)
      g_min_pattern_cpgs = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--pattern-floor"))
      die("--pattern-floor was renamed --min-pattern-cpgs", NULL);
    else if (!strcmp(a, "--resolvers") && i + 1 < argc) {
      const char *v = argv[++i];
      g_resolvers = !strcmp(v, "all") ? -1 : (int64_t)parse_u64(v, a);
    }
    else if (!strcmp(a, "--resolver-gate"))
      die("--resolver-gate is retired: the pairs that get a resolver are "
          "now the N thinnest by hard-block footprint, --resolvers N "
          "(or all)", NULL);
    else if (!strcmp(a, "--resolver-cache") && i + 1 < argc)
      g_resolver_cache = argv[++i];
    else if (!strcmp(a, "--resolver-stride") && i + 1 < argc) {
      const char *v = argv[++i]; char *end = NULL;
      g_stride_k = (uint32_t)strtoul(v, &end, 10);
      if (!end || *end != '/') die("--resolver-stride wants k/N", v);
      g_stride_n = (uint32_t)strtoul(end + 1, NULL, 10);
      if (!g_stride_n || g_stride_k >= g_stride_n)
        die("--resolver-stride needs 0 <= k < N", v);
    }
    else if (!strcmp(a, "--anneal-min-seg") && i + 1 < argc) {
      const char *v = argv[++i]; char *end = NULL;
      g_anneal_hi = strtoull(v, &end, 10);
      if (!end || *end != ',') die("--anneal-min-seg wants HI,LO[,STEP]", v);
      g_anneal_lo = strtoull(end + 1, &end, 10);
      if (end && *end == ',') g_anneal_step = strtoull(end + 1, NULL, 10);
      if (!g_anneal_hi || g_anneal_lo > g_anneal_hi)
        die("--anneal-min-seg needs HI >= LO and HI > 0", v);
    }
    /* One MRMP over every class, no routing. What mrmp-build meant before it
     * became the tree builder, kept because a flat global is still the right
     * artifact for deconvolution and for a reference too shallow to split. */
    else if (!strcmp(a, "--flat"))
      die("--flat is the default now: drop the flag (--bank is the other "
          "mode)", NULL);
    else if (!strcmp(a, "--top") && i + 1 < argc) {
      top = (uint32_t)parse_u64(argv[++i], a); top_given = 1;
    }
    else if (!strcmp(a, "--cell-store") && i + 1 < argc)
      g_cal_store = argv[++i];
    else if (!strcmp(a, "--cell-labels") && i + 1 < argc)
      cal_labels = argv[++i];
    else if (!strcmp(a, "--calib-eps") && i + 1 < argc)
      g_cal_eps = (float)atof(argv[++i]);
    else if (!strcmp(a, "--calib-threads") && i + 1 < argc)
      g_cal_threads = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--name") && i + 1 < argc) setname = argv[++i];
    else if (!strcmp(a, "--beta-threshold") && i + 1 < argc)
      gh.beta_threshold = (float)atof(argv[++i]);
    else if (!strcmp(a, "--impute-ambiguous") && i + 1 < argc) {
      const char *m = argv[++i];
      int meth = !strcmp(m, "none")     ? MRMP_IMPUTE_NONE
               : !strcmp(m, "majority") ? MRMP_IMPUTE_MAJORITY
               : !strcmp(m, "zero")     ? MRMP_IMPUTE_ZERO
               : !strcmp(m, "one")      ? MRMP_IMPUTE_ONE
               : !strcmp(m, "random")   ? MRMP_IMPUTE_RANDOM : -1;
      if (meth < 0) die("--impute-ambiguous: unknown strategy", m);
      gh.flags = (gh.flags & ~(MRMP_FLAG_IMPUTE | MRMP_FLAG_IMPUTE_METHOD_MASK))
               | mrmp_impute_flags(meth);
    }
    else if (!strcmp(a, "--impute-seed") && i + 1 < argc)
      g_impute_seed = parse_u64(argv[++i], a);
    else if (!strcmp(a, "--max-ambig-frac") && i + 1 < argc)
      gh.max_ambig_frac = (float)atof(argv[++i]);
    else if (!strcmp(a, "--min-major-fold") && i + 1 < argc)
      gh.min_major_fold = (float)atof(argv[++i]);
    else if ((!strcmp(a, "--max-lowdepth-frac") || !strcmp(a, "--max-frac-na"))
             && i + 1 < argc)
      sel.max_lowdepth_frac = (float)atof(argv[++i]);
    else if ((!strcmp(a, "--feature-mindepth") || !strcmp(a, "--min-cg-depth"))
             && i + 1 < argc)
      sel.feature_mindepth = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--shrink-pseudocnt") && i + 1 < argc)
      sel.shrink_pseudocnt = (float)atof(argv[++i]);
    else if (!strcmp(a, "--include-all-0")) sel.inc_all0 = 1;
    else if (!strcmp(a, "--include-all-1")) sel.inc_all1 = 1;
    else if (!strcmp(a, "--max-depth") && i + 1 < argc)
      max_depth = (uint32_t)parse_u64(argv[++i], a);
    else if ((!strcmp(a, "--call-mindepth") || !strcmp(a, "--mincov"))
             && i + 1 < argc)
      gh.mincov = (uint32_t)parse_u64(argv[++i], a);
    else if (!strcmp(a, "--delta-mean-top") && i + 1 < argc) {
      sel.delta_mean_top = (uint32_t)parse_u64(argv[++i], a);
    }
    else if ((!strcmp(a, "--call-band") || !strcmp(a, "--qfilter")) && i + 1 < argc) {
      const char *v = argv[++i];
      /* `none` is LO == HI == the call threshold, which is the identity: a
       * confident 0-class is below it and a confident 1-class above it, so
       * both legs always pass. Spelled out because a band that filters
       * nothing should not have to be discovered. LO == HI is therefore
       * legal; LO > HI still is not. */
      if (!strcmp(v, "none") || !strcmp(v, "off")) {
        sel.qfilter_lo = sel.qfilter_hi = gh.beta_threshold;
      } else {
      char *end = NULL;
      float lo = strtof(v, &end);
      if (!end || *end != ',') die("--call-band wants LO,HI (or none)", v);
      float hi = strtof(end + 1, NULL);
      if (!(lo >= 0.0f && hi <= 1.0f && lo <= hi))
        die("--call-band needs 0 <= LO <= HI <= 1", v);
      sel.qfilter_lo = lo; sel.qfilter_hi = hi;
      }
                     /* asking for the old rule selects it */
    }
    else if (!strcmp(a, "--force")) force = 1;
    else if (a[0] == '-') die("unrecognized or incomplete option", a);
    else if (npos < 2) pos[npos++] = a;
    else die("too many arguments", a);
  }
  if (npos != 2) die("need REF.cg and OUT.mrmp (see mrmp-build -h)", NULL);
  if (!!g_cal_store != !!cal_labels)
    die("--cell-store and --cell-labels go together", NULL);
  if (cal_labels) calib_load_labels(cal_labels);
  if (g_bank) {
    if (!g_anneal_hi)
      die("--bank discovers its splits by annealing; give --anneal-min-seg",
          NULL);
    if (!g_cal_store && !dry && g_resolvers)
      fprintf(stderr, "[methscope] mrmp-build: NOTE --bank without "
              "--cell-store: resolver pairs keep the DEFAULT selection "
              "(shrunk 0.30/0.70 band), uncalibrated -- the bulk-reference "
              "mode. Give per-cell data when you have it.\n");
    if (g_stride_n && !g_resolver_cache)
      die("--resolver-stride only populates a cache; give --resolver-cache",
          NULL);
    if (g_resolver_cache) {
      struct stat st;
      if (stat(g_resolver_cache, &st) || !S_ISDIR(st.st_mode))
        die("--resolver-cache is not an existing directory",
            g_resolver_cache);
    }
    /* NOTE the floor is NOT activated here: split discovery (seg
     * matrices, annealing, pair supports) must run on the FULL evidence.
     * Only the EMITTED type-1 block is pruned -- tree_build rebuilds a
     * confirmed split node with the floor active just for its image. */
  }
  flat = !g_bank;                    /* one set is the default mode */
  if (flat && g_anneal_hi)
    die("--anneal-min-seg discovers a bank's splits; give --bank", NULL);
  if (g_bank && top_given)
    die("--top cuts a single flat set; a bank's sets are budgeted by "
        "--min-pattern-cpgs and --resolvers, or by mrmp-pool", NULL);
  if (flat) min_seg = UINT64_MAX;   /* nothing can split */
  /* The floor stays OFF by default, including under --flat. It was briefly
   * defaulted on here, because the standalone satellite builder that --flat
   * replaced ran it -- but measurement says the protection is band-specific,
   * not builder-specific. At --call-band 0.15,0.75 held-out PAL-Inh read 0.422 on a
   * contrast its reference put at 0.043, and the floor was the right answer.
   * At 0.30,0.70 the same pair reads 0.080 against 0.078: the looser band
   * never selects the extreme, thinly-supported CpGs in the first place, so
   * the floor only deletes evidence -- 35% of that pair's thin side, for 52
   * errors against 47 on the mouse fold. Turn it on with a tight band. */
  const char *store = pos[0], *out = pos[1];
  if (!force && !dry) { struct stat st; if (!stat(out, &st)) die("output exists (use --force)", out); }

  uint32_t nstore = 0; int64_t *voff = NULL;
  char **slab = ms_read_store_index(store, &nstore, &voff);
  if (nstore < 2) die("reference holds fewer than two classes", store);
  uint32_t *idx = xcalloc(nstore, sizeof(uint32_t), "root idx");
  for (uint32_t k = 0; k < nstore; ++k) idx[k] = k;

  const int tty = isatty(STDERR_FILENO);
  /* The build report goes to stderr only. The SHAPE is not written anywhere:
   * `inspect --tree` derives it from the chain -- parent from the name, classes
   * from each block -- so a file here would be a second copy of something the
   * artifact already answers, free to drift. */
  FILE *rep = stderr;
  treeout_t t; memset(&t, 0, sizeof t);
  if (flat)
    fprintf(stderr, "[methscope] %s: %u classes, one MRMP over all of them "
            "(a tree of one level)\n", g_cmd, nstore);
  else if (g_anneal_hi)
    fprintf(stderr, "[methscope] %s: %u classes, split threshold annealed "
            "%" PRIu64 " -> %" PRIu64 " per node%s\n", g_cmd, nstore,
            g_anneal_hi, g_anneal_lo, dry ? " (dry run)" : "");
  else
    fprintf(stderr, "[methscope] %s: %u classes, split above %" PRIu64
            " segregating CpGs%s\n", g_cmd, nstore, min_seg,
            dry ? " (dry run)" : "");
  tree_build(store, slab, voff, idx, nstore, &gh, &sel, setname, min_seg,
             0, max_depth, dry, sat_n, relax_below, relax_gap,
             rep == stderr ? tty : 0, &t, rep);
  if (dry) return 0;

  /* ---- BANK type-2 pass: the resolver pairs, uniformly root@A__B ---- */
  if (g_bank) {
    g_floor_active = 0;                /* pair sets are calibration-gated */
    /* Sort the registry by (A, B) name so consecutive pairs share their
     * A-side class: with the extracted-cell cache in mrmp_calib.c, a class's
     * cells inflate once and serve its whole run of pairs. The sort is
     * deterministic, so --resolver-stride ordinals agree across array jobs
     * (and cache filenames are name-keyed, unaffected by order). */
    qsort(g_bp, g_bp_n, sizeof(bank_pair_t), bank_pair_cmp);
    uint32_t made = 0;
    if (g_resolvers < 0) {
      uint32_t nnever = 0;
      for (uint32_t i = 0; i < g_bp_n; ++i) nnever += g_bp[i].never != 0;
      fprintf(rep, "\n[methscope] bank resolvers: all %u pairs "
              "(bank-full)\n  %u pair(s) the tree never parts, %u covered "
              "by a resolver\n", g_bp_n, nnever, nnever);
    } else
      bank_footprint_pick(store, slab, voff, nstore, &t, rep);
    uint32_t ncached = 0, nskipped = 0, ordinal = 0;
    for (uint32_t i = 0; i < g_bp_n; ++i) {
      if (!g_bp[i].mand) continue;
      uint32_t ia = UINT32_MAX, ib = UINT32_MAX;
      for (uint32_t k = 0; k < nstore; ++k) {
        if (!strcmp(slab[k], g_bp[i].a)) ia = k;
        if (!strcmp(slab[k], g_bp[i].b)) ib = k;
      }
      if (ia == UINT32_MAX || ib == UINT32_MAX)
        die("bank pair names a class missing from the store", g_bp[i].a);
      char *two[2] = { slab[ia], slab[ib] };
      int64_t vv[2] = { voff[ia], voff[ib] };
      char ta[128], tb[128], sname[512];
      sat_tag(two[0], ta, sizeof ta); sat_tag(two[1], tb, sizeof tb);
      snprintf(sname, sizeof sname, "%s@%s__%s", setname, ta, tb);
      const uint32_t ord = ordinal++;      /* stable: registry order */
      char cpath[PATH_MAX];
      if (g_resolver_cache) {
        if (snprintf(cpath, sizeof cpath, "%s/%s__%s.mrmp",
                     g_resolver_cache, ta, tb) >= (int)sizeof cpath)
          die("resolver cache path too long", ta);
        FILE *cf = fopen(cpath, "rb");
        if (cf) {                          /* splice the cached block */
          fseek(cf, 0, SEEK_END);
          long sz = ftell(cf);
          if (sz > 0) {
            void *img = xcalloc(1, (size_t)sz, "cached resolver");
            fseek(cf, 0, SEEK_SET);
            if (fread(img, 1, (size_t)sz, cf) != (size_t)sz)
              die("short read on cached resolver", cpath);
            fclose(cf);
            tree_push(&t, img, (uint64_t)sz, sname);
            ++made; ++ncached;
            continue;
          }
          fclose(cf);
        }
      }
      if (g_stride_n && ord % g_stride_n != g_stride_k) { ++nskipped; continue; }
      ms_select_opt_t rs = sel;
      if (!calib_pair(two, &rs, sname, tty, rep) && rs.min_sbeta_gap <= 0.0f)
        /* no calibration (bulk reference): the band is the wrong default
         * for a 2-class rebuild -- close pairs can empty it outright
         * (measured: Dendritic.Cell pairs on the 33-label bulk pools).
         * Fall back to the validated satellite selection, the universal
         * shrunk-beta gap. */
        rs.min_sbeta_gap = 0.40f;
      subset_block_t s2;
      { char m[192]; snprintf(m, sizeof m, "[%.180s]", sname);
        spin_start(tty, m); }
      build_subset_block(store, 2, two, vv, &gh, &rs, sname, &s2);
      spin_stop();
      if (g_resolver_cache) {              /* save-then-rename: idempotent
                                            * under concurrent array jobs */
        char tpath[PATH_MAX];
        if (snprintf(tpath, sizeof tpath, "%s.tmp.%ld", cpath,
                     (long)getpid()) < (int)sizeof tpath) {
          FILE *tf = fopen(tpath, "wb");
          if (tf) {
            if (fwrite(s2.img, 1, (size_t)s2.bytes, tf) == s2.bytes) {
              fclose(tf);
              rename(tpath, cpath);
            } else { fclose(tf); unlink(tpath); }
          }
        }
      }
      tree_push(&t, s2.img, s2.bytes, sname);
      ++made;
    }
    fprintf(rep, "  %u resolver pair(s) built (%u from cache)\n",
            made, ncached);
    if (nskipped)
      fprintf(rep, "  WARNING: %u out-of-stride pair(s) skipped -- this "
              "artifact is PARTIAL; rerun without --resolver-stride once "
              "the cache is fully populated\n", nskipped);
  }

  ms_mrmp_chain_write(out, t.n, (const void *const *)t.img, t.len);
  /* --flat --top N: the consumer's cut, applied at write time because a flat
   * set has exactly one consumer and no other set to compete with. The same
   * prune mrmp-pool --pooled-top runs -- winners are a prefix of the set's own
   * CpG-count ranking -- so the two cannot drift. 0 keeps every pattern. */
  if (flat && top && t.n == 1) {
    ms_mrmpset_t *ch = ms_mrmpset_open(out);
    void *pimg = NULL; uint64_t pbytes = 0;
    uint32_t before = 0;
    { mrmp_reader_t r; mrmp_open_at(&r, out, ch->block_off[0], ch->block_bytes[0]);
      before = (uint32_t)r.h->n_candidates; mrmp_close(&r); }
    prune_block(out, ch->block_off[0], ch->block_bytes[0], top, NULL, &pimg, &pbytes);
    ms_mrmpset_free(ch);
    ms_mrmp_chain_write(out, 1, (const void *const *)&pimg, &pbytes);
    free(t.img[0]); t.img[0] = pimg; t.len[0] = pbytes;
    if (before > top)
      fprintf(stderr, "  --top %u: kept the %u patterns with the most CpGs of "
              "%u, the rest folded into PNA\n", top, top, before);
  }
  /* Also one file per node. A block is a byte-identical standalone MRMPIDX1, so
   * this needs no re-encode -- and it is what lets classify-featurize /
   * classify-train drive the tree per node with no new subcommand. */
  { char b1[32]; uint64_t tot = 0;
    for (uint32_t k = 0; k < t.n; ++k) tot += t.len[k];
    fprintf(stderr, "  %u node(s), %s bytes -> %s\n", t.n,
            commafmt_local(tot, b1), out); }
  for (uint32_t k = 0; k < t.n; ++k) { free(t.img[k]); free(t.name[k]); }
  free(t.img); free(t.len); free(t.name); free(idx);
  for (uint32_t k = 0; k < nstore; ++k) free(slab[k]);
  free(slab); free(voff);
  for (uint32_t k = 0; k < g_cal_n; ++k) { free(g_cal_cell[k]); free(g_cal_cls[k]); }
  free(g_cal_cell); free(g_cal_cls);
  return 0;
}
