// SPDX-License-Identifier: AGPL-3.0-or-later
/*
 * Build the compact membership-first index used by the residual upscale
 * decoder.  Exact MRMP membership is the primary grouping signal.  CpGs retain
 * genomic (YAME-universe) order inside a membership, while memberships are
 * packed in ternary-pattern prefix order so nearby blocks share similar MRMPs.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "methscope.h"

#define MSRI_PATTERN_LEN 35u

#pragma pack(push,1)
typedef struct {
  char magic[8];
  uint32_t version, flags, pattern_length, top_patterns;
  uint32_t target_bin_cpgs, n_bins, n_memberships, reserved32;
  uint64_t n_cpg, n_residual;
  uint64_t bin_offsets_offset, cpg_offset, rank_offset, group_offset, file_bytes;
  uint64_t pattern_checksum, top_checksum, reserved0, reserved1;
} msri_header_t;

typedef struct {
  uint64_t pattern_key;
  uint64_t output_offset;
  uint32_t count;
  uint32_t rank;
} msri_group_t;
#pragma pack(pop)

typedef struct {
  uint64_t key, output_offset;
  uint32_t count, rank, seen;
  uint8_t top;
} group_t;

typedef struct {
  uint64_t *keys;
  uint32_t *values; /* group index + 1; zero is empty */
  size_t cap, used;
} hash_t;

typedef struct {
  group_t *a;
  size_t n, cap;
} groups_t;

typedef struct {
  uint64_t *a;
  size_t n, cap;
} offsets_t;

static void fail(const char *msg) {
  fprintf(stderr, "[methscope] upscale-residual-index: %s\n", msg);
  exit(1);
}

static void fail_path(const char *msg, const char *path) {
  fprintf(stderr, "[methscope] upscale-residual-index: %s: %s\n", msg, path);
  exit(1);
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) fail("out of memory");
  return p;
}

static void *xcalloc(size_t n, size_t z) {
  if (z && n > SIZE_MAX / z) fail("allocation size overflow");
  void *p = calloc(n ? n : 1, z);
  if (!p) fail("out of memory");
  return p;
}

static void *xrealloc(void *p, size_t n) {
  void *q = realloc(p, n ? n : 1);
  if (!q) fail("out of memory");
  return q;
}

static uint64_t mix64(uint64_t x) {
  x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

static uint64_t checksum_add(uint64_t h, uint64_t key, uint64_t count) {
  return h ^ mix64(key + UINT64_C(0x9e3779b97f4a7c15) * (count + 1));
}

static uint64_t encode_pattern(const char *s) {
  uint64_t key = 0;
  for (uint32_t i = 0; i < MSRI_PATTERN_LEN; ++i) {
    unsigned d = (unsigned)(s[i] - '0');
    if (d > 2) fail("MRMP pattern must contain exactly 35 characters from 0,1,2");
    key = key * 3 + d;
  }
  if (s[MSRI_PATTERN_LEN] && s[MSRI_PATTERN_LEN] != '\t' &&
      s[MSRI_PATTERN_LEN] != ' ' && s[MSRI_PATTERN_LEN] != '\r' &&
      s[MSRI_PATTERN_LEN] != '\n')
    fail("MRMP pattern must contain exactly 35 characters");
  return key;
}

static void groups_push(groups_t *v, group_t g) {
  if (v->n == v->cap) {
    size_t nc = v->cap ? v->cap * 2 : 2048;
    if (nc < v->cap || nc > UINT32_MAX) fail("too many exact MRMP memberships");
    v->a = xrealloc(v->a, nc * sizeof(*v->a));
    v->cap = nc;
  }
  v->a[v->n++] = g;
}

static void offsets_push(offsets_t *v, uint64_t x) {
  if (v->n == v->cap) {
    size_t nc = v->cap ? v->cap * 2 : 1024;
    v->a = xrealloc(v->a, nc * sizeof(*v->a));
    v->cap = nc;
  }
  v->a[v->n++] = x;
}

static void hash_init(hash_t *h, size_t cap) {
  size_t c = 1;
  while (c < cap) {
    if (c > SIZE_MAX / 2) fail("hash capacity overflow");
    c <<= 1;
  }
  h->keys = xcalloc(c, sizeof(*h->keys));
  h->values = xcalloc(c, sizeof(*h->values));
  h->cap = c; h->used = 0;
}

static size_t hash_slot(const hash_t *h, uint64_t key) {
  size_t q = (size_t)mix64(key) & (h->cap - 1);
  while (h->values[q] && h->keys[q] != key) q = (q + 1) & (h->cap - 1);
  return q;
}

static uint32_t hash_get(const hash_t *h, uint64_t key) {
  size_t q = hash_slot(h, key);
  return h->values[q] ? h->values[q] - 1 : UINT32_MAX;
}

static void hash_rebuild(hash_t *h, size_t new_cap) {
  hash_t n;
  hash_init(&n, new_cap);
  for (size_t i = 0; i < h->cap; ++i) if (h->values[i]) {
    size_t q = hash_slot(&n, h->keys[i]);
    n.keys[q] = h->keys[i];
    n.values[q] = h->values[i];
    ++n.used;
  }
  free(h->keys); free(h->values); *h = n;
}

static void hash_put(hash_t *h, uint64_t key, uint32_t group_index) {
  if ((h->used + 1) * 10 >= h->cap * 7) hash_rebuild(h, h->cap * 2);
  size_t q = hash_slot(h, key);
  if (h->values[q]) fail("duplicate exact MRMP membership");
  h->keys[q] = key;
  h->values[q] = group_index + 1;
  ++h->used;
}

static int rank_cmp(const void *aa, const void *bb, void *ctx) {
  const groups_t *v = (const groups_t *)ctx;
  uint32_t a = *(const uint32_t *)aa, b = *(const uint32_t *)bb;
  const group_t *x = &v->a[a], *y = &v->a[b];
  if (x->count != y->count) return x->count < y->count ? 1 : -1;
  return x->key < y->key ? -1 : x->key > y->key;
}

static int similarity_cmp(const void *aa, const void *bb, void *ctx) {
  const groups_t *v = (const groups_t *)ctx;
  const group_t *x = &v->a[*(const uint32_t *)aa];
  const group_t *y = &v->a[*(const uint32_t *)bb];
  /* Numeric base-3 order is lexicographic order for fixed-length patterns,
   * placing memberships with long shared prefixes next to one another. */
  if (x->key != y->key) return x->key < y->key ? -1 : 1;
  return x->rank < y->rank ? -1 : x->rank > y->rank;
}

/* Portable qsort wrapper: the build targets glibc, where qsort_r has context
 * last. */
static const groups_t *sort_groups_ctx;
static int rank_cmp_plain(const void *a, const void *b) {
  return rank_cmp(a, b, (void *)sort_groups_ctx);
}
static int similarity_cmp_plain(const void *a, const void *b) {
  return similarity_cmp(a, b, (void *)sort_groups_ctx);
}

static uint64_t parse_u64(const char *s, const char **endp, const char *what) {
  errno = 0; char *e = NULL; unsigned long long x = strtoull(s, &e, 10);
  if (errno || e == s) {
    fprintf(stderr, "[methscope] upscale-residual-index: invalid %s\n", what);
    exit(1);
  }
  *endp = e; return (uint64_t)x;
}

static void write_all(FILE *f, const void *p, size_t n, const char *path) {
  if (n && fwrite(p, 1, n, f) != n) fail_path("write failed", path);
}

static uint64_t file_size(const char *path) {
  struct stat st;
  if (stat(path, &st) || st.st_size < 0) fail_path("cannot stat", path);
  return (uint64_t)st.st_size;
}

static int usage(void) {
  fprintf(stderr,
    "Usage: methscope upscale-residual-index --binstrings FILE --top-patterns FILE\\\n"
    "       --pattern-counts FILE -o INDEX.msri [--bin-cpgs 16384]\n\n"
    "Build a compact residual-CpG index. Exact MRMP membership is primary;\n"
    "CpGs remain in genomic index order within a membership; related pattern\n"
    "blocks are prefix-packed into approximately equal decoder heads.\n\n"
    "Inputs:\n"
    "  --binstrings FILE    one 35-symbol 0/1/2 MRMP string per genomic CpG\n"
    "  --top-patterns FILE  pinned top pattern table: PATTERN<TAB>P1..PN\n"
    "  --pattern-counts FILE full `uniq -c` pattern-count table (need not be ranked)\n"
    "  -o FILE              output MSRIDX1 binary index\n"
    "  --bin-cpgs N         target CpGs per decoder head (default 16384)\n"
    "  -h, --help           show this help\n");
  return 1;
}

int main_upscale_residual_index(int argc, char **argv) {
  const char *binstrings = NULL, *top_path = NULL, *counts_path = NULL;
  const char *out_path = NULL;
  uint32_t target = 16384;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) return usage();
    else if (!strcmp(argv[i], "--binstrings") && i + 1 < argc) binstrings = argv[++i];
    else if (!strcmp(argv[i], "--top-patterns") && i + 1 < argc) top_path = argv[++i];
    else if (!strcmp(argv[i], "--pattern-counts") && i + 1 < argc) counts_path = argv[++i];
    else if (!strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
    else if (!strcmp(argv[i], "--bin-cpgs") && i + 1 < argc) {
      const char *e; uint64_t x = parse_u64(argv[++i], &e, "--bin-cpgs");
      if (*e || !x || x > UINT32_MAX) fail("invalid --bin-cpgs");
      target = (uint32_t)x;
    } else {
      usage();
      fprintf(stderr, "[methscope] upscale-residual-index: bad option: %s\n", argv[i]);
      return 1;
    }
  }
  if (!binstrings || !top_path || !counts_path || !out_path) return usage();

  groups_t groups = {0};
  hash_t hash;
  hash_init(&hash, 4096);
  FILE *f = fopen(top_path, "r");
  if (!f) fail_path("cannot open top-pattern table", top_path);
  char line[4096];
  uint32_t expect_rank = 1;
  uint64_t top_checksum = 0;
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    uint64_t key = encode_pattern(line);
    char *p = line + MSRI_PATTERN_LEN;
    while (*p == '\t' || *p == ' ') ++p;
    if (*p != 'P' && *p != 'p') fail("top-pattern table requires P-number labels");
    const char *e; uint64_t r = parse_u64(p + 1, &e, "top pattern rank");
    while (*e == '\r' || *e == '\n' || *e == ' ' || *e == '\t') ++e;
    if (*e || r != expect_rank) fail("top-pattern ranks must be contiguous P1..PN");
    group_t g = {key, 0, 0, (uint32_t)r, 0, 1};
    groups_push(&groups, g);
    hash_put(&hash, key, (uint32_t)(groups.n - 1));
    top_checksum = checksum_add(top_checksum, key, r);
    ++expect_rank;
  }
  if (ferror(f) || fclose(f)) fail_path("error reading top-pattern table", top_path);
  uint32_t n_top = expect_rank - 1;
  if (!n_top) fail("empty top-pattern table");

  f = fopen(counts_path, "r");
  if (!f) fail_path("cannot open pattern-count table", counts_path);
  uint64_t n_cpg = 0, pattern_checksum = 0;
  while (fgets(line, sizeof(line), f)) {
    const char *p = line;
    while (*p == ' ' || *p == '\t') ++p;
    const char *e; uint64_t count = parse_u64(p, &e, "pattern count");
    while (*e == ' ' || *e == '\t') ++e;
    if (count > UINT32_MAX) fail("one exact membership exceeds uint32 count");
    uint64_t key = encode_pattern(e);
    uint32_t gi = hash_get(&hash, key);
    if (gi == UINT32_MAX) {
      group_t g = {key, 0, (uint32_t)count, 0, 0, 0};
      groups_push(&groups, g);
      gi = (uint32_t)(groups.n - 1);
      hash_put(&hash, key, gi);
    } else {
      if (groups.a[gi].count) fail("duplicate pattern in count table");
      groups.a[gi].count = (uint32_t)count;
    }
    if (UINT64_MAX - n_cpg < count) fail("CpG count overflow");
    n_cpg += count;
    pattern_checksum = checksum_add(pattern_checksum, key, count);
  }
  if (ferror(f) || fclose(f)) fail_path("error reading pattern-count table", counts_path);
  for (uint32_t i = 0; i < n_top; ++i)
    if (!groups.a[i].count) fail("a pinned top pattern is absent from pattern counts");
  if (groups.n <= n_top) fail("no residual MRMP memberships");

  size_t nr = groups.n - n_top;
  uint32_t *order = xmalloc(nr * sizeof(*order));
  for (size_t i = 0; i < nr; ++i) order[i] = n_top + (uint32_t)i;
  sort_groups_ctx = &groups;
  qsort(order, nr, sizeof(*order), rank_cmp_plain);
  for (size_t i = 0; i < nr; ++i) groups.a[order[i]].rank = n_top + 1 + (uint32_t)i;

  qsort(order, nr, sizeof(*order), similarity_cmp_plain);
  offsets_t bins = {0};
  offsets_push(&bins, 0);
  uint64_t residual = 0;
  uint32_t fill = 0;
  for (size_t oi = 0; oi < nr; ++oi) {
    group_t *g = &groups.a[order[oi]];
    if (g->count <= target) {
      if (fill && (uint64_t)fill + g->count > target) {
        offsets_push(&bins, residual);
        fill = 0;
      }
      g->output_offset = residual;
      residual += g->count;
      fill += g->count;
      if (fill == target) {
        offsets_push(&bins, residual);
        fill = 0;
      }
    } else {
      if (fill) {
        offsets_push(&bins, residual);
        fill = 0;
      }
      g->output_offset = residual;
      uint32_t left = g->count;
      while (left >= target) {
        residual += target;
        left -= target;
        offsets_push(&bins, residual);
      }
      if (left) {
        residual += left;
        fill = left;
      }
    }
  }
  if (bins.a[bins.n - 1] != residual) offsets_push(&bins, residual);
  if (residual > UINT32_MAX) fail("residual CpG count exceeds uint32 local IDs");
  if (bins.n - 1 > UINT32_MAX) fail("too many residual decoder heads");

  uint32_t *cpg = xmalloc((size_t)residual * sizeof(*cpg));
  uint32_t *rank = xmalloc((size_t)residual * sizeof(*rank));
  f = fopen(binstrings, "r");
  if (!f) fail_path("cannot open binstring table", binstrings);
  uint64_t genomic = 0, observed_residual = 0;
  while (fgets(line, sizeof(line), f)) {
    if (genomic > UINT32_MAX) fail("genomic CpG index exceeds uint32");
    uint64_t key = encode_pattern(line);
    uint32_t gi = hash_get(&hash, key);
    if (gi == UINT32_MAX) fail("binstring is absent from full pattern counts");
    group_t *g = &groups.a[gi];
    if (g->seen >= g->count) fail("binstring frequency exceeds count table");
    if (!g->top) {
      uint64_t dest = g->output_offset + g->seen;
      cpg[dest] = (uint32_t)genomic;
      rank[dest] = g->rank;
      ++observed_residual;
    }
    ++g->seen; ++genomic;
  }
  if (ferror(f) || fclose(f)) fail_path("error reading binstring table", binstrings);
  if (genomic != n_cpg || observed_residual != residual)
    fail("binstring and pattern-count totals disagree");
  for (size_t i = 0; i < groups.n; ++i)
    if (groups.a[i].seen != groups.a[i].count) fail("binstring membership count mismatch");

  msri_group_t *out_groups = xmalloc(nr * sizeof(*out_groups));
  for (size_t oi = 0; oi < nr; ++oi) {
    const group_t *g = &groups.a[order[oi]];
    out_groups[oi].pattern_key = g->key;
    out_groups[oi].output_offset = g->output_offset;
    out_groups[oi].count = g->count;
    out_groups[oi].rank = g->rank;
  }

  msri_header_t h;
  memset(&h, 0, sizeof(h));
  memcpy(h.magic, "MSRIDX1", 7);
  h.version = 1; h.flags = 1; h.pattern_length = MSRI_PATTERN_LEN;
  h.top_patterns = n_top; h.target_bin_cpgs = target;
  h.n_bins = (uint32_t)(bins.n - 1); h.n_memberships = (uint32_t)nr;
  h.n_cpg = n_cpg; h.n_residual = residual;
  h.bin_offsets_offset = sizeof(h);
  h.cpg_offset = h.bin_offsets_offset + bins.n * sizeof(*bins.a);
  h.rank_offset = h.cpg_offset + residual * sizeof(*cpg);
  h.group_offset = h.rank_offset + residual * sizeof(*rank);
  h.file_bytes = h.group_offset + nr * sizeof(*out_groups);
  h.pattern_checksum = pattern_checksum; h.top_checksum = top_checksum;
  if (sizeof(h) != 128 || sizeof(*out_groups) != 24) fail("internal MSRIDX1 layout error");

  f = fopen(out_path, "wb");
  if (!f) fail_path("cannot create index", out_path);
  write_all(f, &h, sizeof(h), out_path);
  write_all(f, bins.a, bins.n * sizeof(*bins.a), out_path);
  write_all(f, cpg, (size_t)residual * sizeof(*cpg), out_path);
  write_all(f, rank, (size_t)residual * sizeof(*rank), out_path);
  write_all(f, out_groups, nr * sizeof(*out_groups), out_path);
  if (fclose(f)) fail_path("error closing index", out_path);
  if (file_size(out_path) != h.file_bytes) fail_path("index size verification failed", out_path);

  char manifest[4096];
  if (snprintf(manifest, sizeof(manifest), "%s.tsv", out_path) >= (int)sizeof(manifest))
    fail("output path too long");
  f = fopen(manifest, "w");
  if (!f) fail_path("cannot create manifest", manifest);
  fprintf(f,
    "format\tMSRIDX1\nbinstrings\t%s\ntop_patterns_file\t%s\npattern_counts\t%s\n"
    "pattern_length\t%u\ntop_patterns\t%u\ntotal_cpgs\t%" PRIu64 "\n"
    "residual_cpgs\t%" PRIu64 "\nresidual_memberships\t%zu\n"
    "decoder_heads\t%u\ntarget_cpgs_per_head\t%u\n"
    "membership_order\tternary_prefix_similarity\n"
    "within_membership_order\tgenomic_cpg_index\n"
    "oversized_membership_split\tgenomic_cpg_index_chunks\n"
    "pattern_checksum\t%016" PRIx64 "\ntop_checksum\t%016" PRIx64 "\n"
    "file_bytes\t%" PRIu64 "\n",
    binstrings, top_path, counts_path, MSRI_PATTERN_LEN, n_top, n_cpg,
    residual, nr, h.n_bins, target, pattern_checksum, top_checksum, h.file_bytes);
  if (fclose(f)) fail_path("error closing manifest", manifest);

  uint64_t min_bin = UINT64_MAX, max_bin = 0;
  for (size_t i = 0; i + 1 < bins.n; ++i) {
    uint64_t z = bins.a[i + 1] - bins.a[i];
    if (z < min_bin) min_bin = z;
    if (z > max_bin) max_bin = z;
  }
  fprintf(stderr,
    "[methscope] upscale-residual-index: total=%" PRIu64
    " top=%u residual=%" PRIu64 " memberships=%zu heads=%u\n"
    "[methscope] upscale-residual-index: head CpGs min=%" PRIu64
    " mean=%.1f max=%" PRIu64 "\n"
    "[methscope] upscale-residual-index: wrote %s (%" PRIu64 " bytes) and %s\n",
    n_cpg, n_top, residual, nr, h.n_bins, min_bin,
    (double)residual / h.n_bins, max_bin, out_path, h.file_bytes, manifest);

  free(out_groups); free(rank); free(cpg); free(bins.a); free(order);
  free(hash.keys); free(hash.values); free(groups.a);
  return 0;
}
