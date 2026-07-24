// SPDX-License-Identifier: AGPL-3.0-or-later
/*
 * Whole-genome processing-unit index for UPDEC2.
 *
 * Real MRMP memberships are ordered by decreasing size (ternary key breaks
 * ties) and are never split.  CpGs retain genomic order inside a membership.
 * PNA (the all-2 pattern) is represented as logical singleton memberships,
 * placed after every real membership and packed in genomic order.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "methscope.h"

#define MSUI_PATTERN_LEN 35u
#define MSUI_UNIT_PURE 1u
#define MSUI_UNIT_PNA 2u
#define MSUI_UNIT_OVERSIZED 4u

#pragma pack(push,1)
typedef struct {
  char magic[8];
  uint32_t version, flags, pattern_length, target_unit_cpgs;
  uint32_t n_units, n_real_memberships, n_pna_units, reserved32;
  uint64_t n_cpg, n_real_cpg, n_pna_cpg;
  uint64_t unit_offset, cpg_offset, membership_offset, file_bytes;
  uint64_t pattern_checksum, reserved0, reserved1, reserved2;
} msui_header_t;

typedef struct {
  uint64_t output_offset;
  uint32_t first_membership;
  uint32_t membership_count; /* logical count; CpG count for PNA units */
  uint32_t cpg_count;
  uint32_t flags;
} msui_unit_t;

typedef struct {
  uint64_t pattern_key;
  uint64_t output_offset;
  uint32_t count;
  uint32_t unit;
} msui_membership_t;
#pragma pack(pop)

typedef struct {
  uint64_t key, output_offset;
  uint32_t count, seen, unit;
  uint8_t pna;
} group_t;

typedef struct {
  group_t *a;
  size_t n, cap;
} groups_t;

typedef struct {
  uint64_t *keys;
  uint32_t *values; /* group index + 1; zero is empty */
  size_t cap, used;
} hash_t;

typedef struct {
  msui_unit_t *a;
  size_t n, cap;
} units_t;

static void fail(const char *msg) {
  fprintf(stderr, "[methscope] _upscale index: %s\n", msg);
  exit(1);
}

static void fail_path(const char *msg, const char *path) {
  fprintf(stderr, "[methscope] _upscale index: %s: %s\n", msg, path);
  exit(1);
}

static void *xcalloc(size_t n, size_t z) {
  if (z && n > SIZE_MAX / z) fail("allocation size overflow");
  void *p = calloc(n ? n : 1, z);
  if (!p) fail("out of memory");
  return p;
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
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
  for (uint32_t i = 0; i < MSUI_PATTERN_LEN; ++i) {
    unsigned d = (unsigned)(s[i] - '0');
    if (d > 2) fail("MRMP pattern must contain exactly 35 characters from 0,1,2");
    key = key * 3 + d;
  }
  if (s[MSUI_PATTERN_LEN] && s[MSUI_PATTERN_LEN] != '\t' &&
      s[MSUI_PATTERN_LEN] != ' ' && s[MSUI_PATTERN_LEN] != '\r' &&
      s[MSUI_PATTERN_LEN] != '\n')
    fail("MRMP pattern must contain exactly 35 characters");
  return key;
}

static uint64_t pna_key(void) {
  uint64_t key = 0;
  for (uint32_t i = 0; i < MSUI_PATTERN_LEN; ++i) key = key * 3 + 2;
  return key;
}

static void groups_push(groups_t *v, group_t g) {
  if (v->n == v->cap) {
    size_t nc = v->cap ? v->cap * 2 : 4096;
    if (nc < v->cap || nc > UINT32_MAX) fail("too many MRMP memberships");
    v->a = xrealloc(v->a, nc * sizeof(*v->a));
    v->cap = nc;
  }
  v->a[v->n++] = g;
}

static void units_push(units_t *v, msui_unit_t u) {
  if (v->n == v->cap) {
    size_t nc = v->cap ? v->cap * 2 : 1024;
    v->a = xrealloc(v->a, nc * sizeof(*v->a));
    v->cap = nc;
  }
  v->a[v->n++] = u;
}

static void hash_init(hash_t *h, size_t cap) {
  size_t c = 1;
  while (c < cap) c <<= 1;
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

static void hash_rebuild(hash_t *h, size_t cap) {
  hash_t n;
  hash_init(&n, cap);
  for (size_t i = 0; i < h->cap; ++i) if (h->values[i]) {
    size_t q = hash_slot(&n, h->keys[i]);
    n.keys[q] = h->keys[i];
    n.values[q] = h->values[i];
    ++n.used;
  }
  free(h->keys); free(h->values); *h = n;
}

static void hash_put(hash_t *h, uint64_t key, uint32_t value) {
  if ((h->used + 1) * 10 >= h->cap * 7) hash_rebuild(h, h->cap * 2);
  size_t q = hash_slot(h, key);
  if (h->values[q]) fail("duplicate pattern in count table");
  h->keys[q] = key; h->values[q] = value + 1; ++h->used;
}

static const groups_t *sort_ctx;
static int group_cmp(const void *aa, const void *bb) {
  const group_t *a = &sort_ctx->a[*(const uint32_t *)aa];
  const group_t *b = &sort_ctx->a[*(const uint32_t *)bb];
  if (a->count != b->count) return a->count < b->count ? 1 : -1;
  return a->key < b->key ? -1 : a->key > b->key;
}

static uint64_t parse_u64(const char *s, const char **endp, const char *what) {
  errno = 0; char *e = NULL; unsigned long long x = strtoull(s, &e, 10);
  if (errno || e == s) {
    fprintf(stderr, "[methscope] _upscale index: invalid %s\n", what);
    exit(1);
  }
  *endp = e;
  return (uint64_t)x;
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
    "Usage: methscope _upscale index --binstrings FILE --pattern-counts FILE\n"
    "       -o UNITS.msui [--unit-cpgs 16384]\n\n"
    "Build the whole-genome processing-unit index used by UPDEC2. Real MRMP\n"
    "memberships are size-ranked and never split. PNA CpGs are implicit\n"
    "singleton memberships packed after all real memberships.\n\n"
    "  --binstrings FILE     one 35-symbol 0/1/2 string per genomic CpG\n"
    "  --pattern-counts FILE full `uniq -c` pattern-count table\n"
    "  -o FILE               output MSUIDX1 index\n"
    "  --unit-cpgs N         target CpGs per unit (default 16384)\n"
    "  --bin-cpgs N          deprecated alias for --unit-cpgs\n"
    "  -h, --help            show this help\n");
  return 1;
}

int main_upscale_residual_index(int argc, char **argv) {
  const char *binstrings = NULL, *counts_path = NULL, *out_path = NULL;
  uint32_t target = 16384;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage(); return 0;
    } else if (!strcmp(argv[i], "--binstrings") && i + 1 < argc) {
      binstrings = argv[++i];
    } else if (!strcmp(argv[i], "--pattern-counts") && i + 1 < argc) {
      counts_path = argv[++i];
    } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
      out_path = argv[++i];
    } else if ((!strcmp(argv[i], "--unit-cpgs") || !strcmp(argv[i], "--bin-cpgs"))
               && i + 1 < argc) {
      const char *e; uint64_t x = parse_u64(argv[++i], &e, "--unit-cpgs");
      if (*e || !x || x > UINT32_MAX) fail("invalid --unit-cpgs");
      target = (uint32_t)x;
    } else if (!strcmp(argv[i], "--top-patterns") && i + 1 < argc) {
      ++i; /* accepted temporarily so old scripts fail only on changed output semantics */
    } else {
      usage();
      fprintf(stderr, "[methscope] _upscale index: bad option: %s\n", argv[i]);
      return 1;
    }
  }
  if (!binstrings || !counts_path || !out_path) return usage();

  groups_t groups = {0};
  hash_t hash;
  hash_init(&hash, 4096);
  FILE *f = fopen(counts_path, "r");
  if (!f) fail_path("cannot open pattern-count table", counts_path);
  char line[4096];
  uint64_t total = 0, checksum = 0, pk = pna_key();
  uint32_t pna_group = UINT32_MAX;
  while (fgets(line, sizeof(line), f)) {
    const char *p = line;
    while (*p == ' ' || *p == '\t') ++p;
    const char *e; uint64_t count = parse_u64(p, &e, "pattern count");
    while (*e == ' ' || *e == '\t') ++e;
    if (!count || count > UINT32_MAX) fail("membership count is outside uint32 range");
    uint64_t key = encode_pattern(e);
    group_t g = {key, 0, (uint32_t)count, 0, UINT32_MAX, key == pk};
    groups_push(&groups, g);
    hash_put(&hash, key, (uint32_t)(groups.n - 1));
    if (key == pk) pna_group = (uint32_t)(groups.n - 1);
    if (UINT64_MAX - total < count) fail("CpG count overflow");
    total += count;
    checksum = checksum_add(checksum, key, count);
  }
  if (ferror(f) || fclose(f)) fail_path("error reading pattern-count table", counts_path);
  if (pna_group == UINT32_MAX) fail("pattern counts do not contain PNA (all-2)");
  if (groups.n < 2) fail("no real MRMP memberships");

  const size_t n_real_groups = groups.n - 1;
  uint32_t *order = xmalloc(n_real_groups * sizeof(*order));
  size_t q = 0;
  for (size_t i = 0; i < groups.n; ++i) if (!groups.a[i].pna) order[q++] = (uint32_t)i;
  sort_ctx = &groups;
  qsort(order, n_real_groups, sizeof(*order), group_cmp);

  units_t units = {0};
  uint64_t output = 0;
  size_t unit_first = 0;
  uint32_t unit_members = 0, unit_cpgs = 0;
  for (size_t oi = 0; oi < n_real_groups; ++oi) {
    group_t *g = &groups.a[order[oi]];
    if (unit_members && (uint64_t)unit_cpgs + g->count > target) {
      msui_unit_t u = {output - unit_cpgs, (uint32_t)unit_first,
                       unit_members, unit_cpgs, unit_members == 1 ? MSUI_UNIT_PURE : 0};
      units_push(&units, u);
      unit_first = oi; unit_members = 0; unit_cpgs = 0;
    }
    g->output_offset = output;
    g->unit = (uint32_t)units.n;
    output += g->count;
    unit_cpgs += g->count;
    ++unit_members;
    if (g->count > target) {
      msui_unit_t u = {output - unit_cpgs, (uint32_t)unit_first, 1, unit_cpgs,
                       MSUI_UNIT_PURE | MSUI_UNIT_OVERSIZED};
      units_push(&units, u);
      unit_first = oi + 1; unit_members = 0; unit_cpgs = 0;
    }
  }
  if (unit_members) {
    msui_unit_t u = {output - unit_cpgs, (uint32_t)unit_first,
                     unit_members, unit_cpgs, unit_members == 1 ? MSUI_UNIT_PURE : 0};
    units_push(&units, u);
  }
  const uint32_t n_real_units = (uint32_t)units.n;
  const uint64_t n_real_cpg = output;
  const uint64_t n_pna_cpg = groups.a[pna_group].count;
  for (uint64_t done = 0; done < n_pna_cpg;) {
    uint32_t z = (uint32_t)((n_pna_cpg - done) > target ? target : n_pna_cpg - done);
    msui_unit_t u = {n_real_cpg + done, UINT32_MAX, z, z, MSUI_UNIT_PNA};
    units_push(&units, u);
    done += z;
  }
  if (units.n > UINT32_MAX) fail("too many processing units");

  uint32_t *cpg = xmalloc((size_t)total * sizeof(*cpg));
  f = fopen(binstrings, "r");
  if (!f) fail_path("cannot open binstring table", binstrings);
  uint64_t genomic = 0, pna_seen = 0;
  while (fgets(line, sizeof(line), f)) {
    if (genomic > UINT32_MAX) fail("genomic CpG index exceeds uint32");
    uint64_t key = encode_pattern(line);
    uint32_t gi = hash_get(&hash, key);
    if (gi == UINT32_MAX) fail("binstring is absent from pattern-count table");
    group_t *g = &groups.a[gi];
    if (g->seen >= g->count) fail("binstring frequency exceeds count table");
    uint64_t dest = g->pna ? n_real_cpg + pna_seen++ : g->output_offset + g->seen;
    cpg[dest] = (uint32_t)genomic;
    ++g->seen; ++genomic;
  }
  if (ferror(f) || fclose(f)) fail_path("error reading binstring table", binstrings);
  if (genomic != total || pna_seen != n_pna_cpg)
    fail("binstring and pattern-count totals disagree");
  for (size_t i = 0; i < groups.n; ++i)
    if (groups.a[i].seen != groups.a[i].count) fail("binstring membership count mismatch");

  msui_membership_t *members = xmalloc(n_real_groups * sizeof(*members));
  for (size_t oi = 0; oi < n_real_groups; ++oi) {
    const group_t *g = &groups.a[order[oi]];
    members[oi].pattern_key = g->key;
    members[oi].output_offset = g->output_offset;
    members[oi].count = g->count;
    members[oi].unit = g->unit;
  }

  msui_header_t h;
  memset(&h, 0, sizeof(h));
  memcpy(h.magic, "MSUIDX1", 7);
  h.version = 1; h.flags = 1; h.pattern_length = MSUI_PATTERN_LEN;
  h.target_unit_cpgs = target; h.n_units = (uint32_t)units.n;
  h.n_real_memberships = (uint32_t)n_real_groups;
  h.n_pna_units = h.n_units - n_real_units;
  h.n_cpg = total; h.n_real_cpg = n_real_cpg; h.n_pna_cpg = n_pna_cpg;
  h.unit_offset = sizeof(h);
  h.cpg_offset = h.unit_offset + units.n * sizeof(*units.a);
  h.membership_offset = h.cpg_offset + total * sizeof(*cpg);
  h.file_bytes = h.membership_offset + n_real_groups * sizeof(*members);
  h.pattern_checksum = checksum;
  if (sizeof(h) != 128 || sizeof(*units.a) != 24 || sizeof(*members) != 24)
    fail("internal MSUIDX1 layout error");

  f = fopen(out_path, "wb");
  if (!f) fail_path("cannot create index", out_path);
  write_all(f, &h, sizeof(h), out_path);
  write_all(f, units.a, units.n * sizeof(*units.a), out_path);
  write_all(f, cpg, (size_t)total * sizeof(*cpg), out_path);
  write_all(f, members, n_real_groups * sizeof(*members), out_path);
  if (fclose(f)) fail_path("error closing index", out_path);
  if (file_size(out_path) != h.file_bytes) fail_path("index size verification failed", out_path);

  char manifest[4096];
  if (snprintf(manifest, sizeof(manifest), "%s.tsv", out_path) >= (int)sizeof(manifest))
    fail("output path too long");
  f = fopen(manifest, "w");
  if (!f) fail_path("cannot create manifest", manifest);
  uint32_t pure = 0, oversized = 0;
  for (uint32_t i = 0; i < n_real_units; ++i) {
    pure += !!(units.a[i].flags & MSUI_UNIT_PURE);
    oversized += !!(units.a[i].flags & MSUI_UNIT_OVERSIZED);
  }
  fprintf(f,
    "format\tMSUIDX1\nbinstrings\t%s\npattern_counts\t%s\n"
    "pattern_length\t%u\ntotal_cpgs\t%" PRIu64 "\nreal_cpgs\t%" PRIu64
    "\npna_cpgs\t%" PRIu64 "\nreal_memberships\t%zu\nunits\t%u\n"
    "real_units\t%u\nmembership_pure_units\t%u\nmixed_units\t%u\n"
    "pna_units\t%u\noversized_units\t%u\ntarget_cpgs_per_unit\t%u\n"
    "membership_order\tdecreasing_size_then_ternary_key\n"
    "within_membership_order\tgenomic_cpg_index\n"
    "real_membership_split\tfalse\npna_semantics\timplicit_singletons\n"
    "pattern_checksum\t%016" PRIx64 "\nfile_bytes\t%" PRIu64 "\n",
    binstrings, counts_path, MSUI_PATTERN_LEN, total, n_real_cpg, n_pna_cpg,
    n_real_groups, h.n_units, n_real_units, pure, n_real_units - pure,
    h.n_pna_units, oversized, target, checksum, h.file_bytes);
  if (fclose(f)) fail_path("error closing manifest", manifest);

  uint64_t min_unit = UINT64_MAX, max_unit = 0;
  for (size_t i = 0; i < units.n; ++i) {
    uint64_t z = units.a[i].cpg_count;
    if (z < min_unit) min_unit = z;
    if (z > max_unit) max_unit = z;
  }
  fprintf(stderr,
    "[methscope] _upscale index: CpGs=%" PRIu64 " real=%" PRIu64
    " PNA=%" PRIu64 " memberships=%zu\n"
    "[methscope] _upscale index: units=%u pure=%u mixed=%u PNA=%u oversized=%u\n"
    "[methscope] _upscale index: unit CpGs min=%" PRIu64
    " mean=%.1f max=%" PRIu64 "\n"
    "[methscope] _upscale index: wrote %s (%" PRIu64 " bytes) and %s\n",
    total, n_real_cpg, n_pna_cpg, n_real_groups, h.n_units, pure,
    n_real_units - pure, h.n_pna_units, oversized, min_unit,
    (double)total / h.n_units, max_unit, out_path, h.file_bytes, manifest);

  free(members); free(cpg); free(units.a); free(order);
  free(hash.keys); free(hash.values); free(groups.a);
  return 0;
}
