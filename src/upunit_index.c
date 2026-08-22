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
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "methscope.h"
#include "mrmp.h"

/* Units are an OUTPUT partition of the genome, so this builder reads the
 * reference STORE. It once also accepted a built .mrmp and a pair of text
 * tables; both were retired on 2026-08-22 because a .mrmp is a SELECTION.
 * mrmp-build drops the constant binstrings -- a CpG every class calls the
 * same way separates nothing, which is correct for a classifier -- but those
 * CpGs are 54% of hg38 (all-1 alone is 10.7M) and they still have to be
 * reconstructed. Reading the store keeps them, so coverage is 100% by
 * construction. The rule itself is ms_binstring_map(), shared with
 * mrmp-build, so the two cannot drift on what a binstring is.
 *
 * Pattern length is the reference's sample count, not a constant. The MSUIDX1
 * header has always carried it; only this builder assumed 35, which is what
 * the whole-genome Zhou reference happens to use. */
/* A membership's `pattern_key` is a base-3 packing of the binstring, which fits
 * a uint64 only up to 40 classes (3^41 > 2^64). Nothing anywhere INTERPRETS the
 * key -- upunit_index writes it, upscale-train copies it into the .updecx, and
 * no reader decodes it -- so above 40 the field carries a 64-bit hash of the
 * trits instead. Same width, same layout, both formats unchanged, and an
 * artifact at <= 40 classes is still byte-identical to what earlier builds
 * produced. MSUI_PATTERN_LEN_MAX is now the packing limit, not a class limit. */
#define MSUI_PATTERN_LEN_MAX 40u
static uint32_t g_pattern_len = 35u;
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
  msui_unit_t *a;
  size_t n, cap;
} units_t;

static void fail(const char *msg) {
  fprintf(stderr, "[methscope] upscale-set-units: %s\n", msg);
  exit(1);
}

static void fail_path(const char *msg, const char *path) {
  fprintf(stderr, "[methscope] upscale-set-units: %s: %s\n", msg, path);
  exit(1);
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

// splitmix64

static uint64_t mix64(uint64_t x) {
  x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

static uint64_t checksum_add(uint64_t h, uint64_t key, uint64_t count) {
  return h ^ mix64(key + UINT64_C(0x9e3779b97f4a7c15) /* golden ratio */
                   * (count + 1));
}

/* Identity for a binstring given as `len` trits (0/1/2). Base-3 packing when it
 * fits a uint64, so artifacts at <= 40 classes keep the exact bytes earlier
 * builds wrote; a 64-bit FNV-1a of the trits above that. Only ever compared and
 * stored, never decoded, so the two schemes need not agree. */
static uint64_t trits_key(const uint8_t *t, uint32_t len) {
  if (len <= MSUI_PATTERN_LEN_MAX) {
    uint64_t key = 0;
    for (uint32_t i = 0; i < len; ++i) key = key * 3 + t[i];
    return key;
  }
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t i = 0; i < len; ++i) h = (h ^ t[i]) * 1099511628211ULL;
  return h;
}

/* Unpack the base-3 key words mrmp-build produces back into one trit per class.
 * Class s lives in word s/MRMP_TRITS_PER_WORD, appended most-significant-first,
 * so a word holding `cnt` classes is decoded from the last of them backwards. */
static void key_words_to_trits(const uint64_t *key, uint32_t ns, uint8_t *out) {
  const uint32_t nw = mrmp_key_words(ns);
  for (uint32_t w = 0; w < nw; ++w) {
    uint32_t lo = w * MRMP_TRITS_PER_WORD;
    uint32_t hi = lo + MRMP_TRITS_PER_WORD; if (hi > ns) hi = ns;
    uint64_t v = key[w];
    for (uint32_t i = hi; i > lo; --i) { out[i - 1] = (uint8_t)(v % 3); v /= 3; }
  }
}

static uint64_t pna_key(void) {
  if (g_pattern_len > MSUI_PATTERN_LEN_MAX) {
    uint8_t *t = xmalloc(g_pattern_len);
    memset(t, 2, g_pattern_len);
    uint64_t k = trits_key(t, g_pattern_len);
    free(t);
    return k;
  }
  uint64_t key = 0;
  for (uint32_t i = 0; i < g_pattern_len; ++i) key = key * 3 + 2;
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
    fprintf(stderr, "[methscope] upscale-set-units: invalid %s\n", what);
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
  ms_help(stderr,
    "Usage: methscope upscale-set-units [options] REF.cg OUT.msui\n\n"
    "Build the whole-genome processing-unit index used by UPDEC2. Each CpG's\n"
    "binstring is derived from the reference store itself. Memberships are\n"
    "size-ranked and never split; PNA CpGs are implicit singleton memberships\n"
    "packed after all real memberships.\n\n"
    "Units are an OUTPUT partition, so this reads the STORE and not a .mrmp.\n"
    "A built .mrmp is a SELECTION -- mrmp-build drops the constant binstrings\n"
    "because a CpG every class calls the same way separates nothing, which is\n"
    "right for a classifier and fatal here: those CpGs are 54%% of the genome\n"
    "and they still need reconstructing. Reading the store keeps them, so\n"
    "every CpG has a real membership and coverage is 100%% by construction.\n\n"
    "  REF.cg                reference store (`--store REF.cg` also accepted)\n"
    "  OUT.msui              output MSUIDX1 index\n\n"
    "  --mincov N            min per-class coverage (default 1)\n"
    "  --beta-threshold B    call a class methylated above B (default 0.5).\n"
    "                        Must match mrmp-build's.\n"
    "  --unit-cpgs N         target CpGs per unit (default 16384)\n"
    "  --bin-cpgs N          deprecated alias for --unit-cpgs\n"
    "  -h, --help            show this help\n");
  return 1;
}

int main_upscale_set_units(int argc, char **argv) {
  const char *out_path = NULL, *store_path = NULL, *pos[2] = {NULL, NULL};
  int npos = 0;
  uint32_t target = 16384;
  uint32_t mincov = MS_BS_DEF_MINCOV;
  float beta_thr = MS_BS_DEF_BETA_THRESH;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage(); return 0;
    } else if (!strcmp(argv[i], "--store") && i + 1 < argc) {
      store_path = argv[++i];
    } else if (!strcmp(argv[i], "--mincov") && i + 1 < argc) {
      const char *e; uint64_t x = parse_u64(argv[++i], &e, "--mincov");
      if (*e || !x || x > UINT32_MAX) fail("invalid --mincov");
      mincov = (uint32_t)x;
    } else if (!strcmp(argv[i], "--beta-threshold") && i + 1 < argc) {
      beta_thr = (float)atof(argv[++i]);
      if (!(beta_thr > 0.0f && beta_thr < 1.0f)) fail("--beta-threshold must be in (0,1)");
    } else if ((!strcmp(argv[i], "--unit-cpgs") || !strcmp(argv[i], "--bin-cpgs"))
               && i + 1 < argc) {
      const char *e; uint64_t x = parse_u64(argv[++i], &e, "--unit-cpgs");
      if (*e || !x || x > UINT32_MAX) fail("invalid --unit-cpgs");
      target = (uint32_t)x;
    } else if (!strcmp(argv[i], "--top-patterns") && i + 1 < argc) {
      ++i; /* accepted temporarily so old scripts fail only on changed output semantics */
    } else if (argv[i][0] == '-') {
      usage();
      fprintf(stderr, "[methscope] upscale-set-units: bad option: %s\n", argv[i]);
      return 1;
    } else if (npos < 2) {
      pos[npos++] = argv[i];
    } else {
      fail_path("too many arguments", argv[i]);
    }
  }
  /* REF.cg may be given positionally or as --store, the spelling the recipes
   * in the lab journal already use. Either way it is the only input. */
  if (store_path) {
    if (npos != 1) { usage(); fail("--store takes only OUT.msui"); }
    out_path = pos[0];
  } else {
    if (npos != 2) { usage(); fail("need REF.cg and OUT.msui"); }
    /* Recipes predating 2026-08-22 passed a built .mrmp here. That mode is
     * gone -- see the header comment -- and the generic "no .idx" error it
     * would otherwise hit reads like a missing file rather than a retirement. */
    size_t n = strlen(pos[0]);
    if (n > 5 && !strcmp(pos[0] + n - 5, ".mrmp"))
      fail_path("units come from the reference store, not a built .mrmp "
                "(the .mrmp mode was retired: it drops the constant "
                "binstrings, which are 54% of the genome)", pos[0]);
    store_path = pos[0];
    out_path = pos[1];
  }

  groups_t groups = {0};
  uint64_t total = 0, checksum = 0, pk = pna_key();
  uint32_t pna_group = UINT32_MAX;
  size_t n_real_groups = 0;
  uint32_t *order = NULL;

  /* MRMPIDX1 artifact carries the candidate patterns already ranked (count
   * desc, key asc) plus a per-CpG membership rank, so the group set, order,
   * and CpG scatter are all read directly with no text parsing. */
  ms_binstring_map_t bm = {0};
  {
    /* Units are an OUTPUT partition: every CpG must land in one, so the
     * constant binstrings are KEPT (inc_all0/inc_all1 = 1). mrmp-build drops
     * them because a CpG every class calls the same way separates nothing --
     * true for a classifier, and the reason a selected .mrmp cannot define
     * units. The rule itself is ms_binstring_map(), shared with mrmp-build. */
    uint32_t ns = 0; int64_t *voff = NULL;
    char **lab = ms_read_store_index(store_path, &ns, &voff);
    if (!ns) fail_path("reference index is empty", store_path);
    g_pattern_len = ns;
    pk = pna_key();                       /* depends on g_pattern_len */
    ms_binstring_map(store_path, ns, lab, voff, mincov, beta_thr,
                     MS_BS_DEF_MAX_AMBIG, MS_BS_DEF_MIN_FOLD, 1, 1, &bm);
    for (uint32_t k = 0; k < ns; ++k) free(lab[k]);
    free(lab); free(voff);

    /* group index == intern index, so bm.cpg_pat indexes groups directly. */
    const uint32_t nw = mrmp_key_words(ns);
    uint8_t *trits = xmalloc(ns);
    for (uint64_t r = 0; r < bm.n_pat; ++r) {
      key_words_to_trits(bm.keys + r * nw, ns, trits);
      group_t g = {.key = trits_key(trits, ns), .output_offset = 0,
                   .count = (uint32_t)bm.count[r], .seen = 0,
                   .unit = UINT32_MAX, .pna = 0};
      if (!bm.count[r] || bm.count[r] > UINT32_MAX)
        fail("membership count is outside uint32 range");
      groups_push(&groups, g);
      total += bm.count[r];
      checksum = checksum_add(checksum, g.key, g.count);
    }
    free(trits);
    group_t pg = {.key = pk, .output_offset = 0, .count = (uint32_t)bm.pna_cpg,
                  .seen = 0, .unit = UINT32_MAX, .pna = 1};
    groups_push(&groups, pg);
    pna_group = (uint32_t)(groups.n - 1);
    total += bm.pna_cpg;
    checksum = checksum_add(checksum, pk, bm.pna_cpg);
    if (total != bm.n_cpg) fail("binstring counts do not cover all CpGs");
    if (groups.n < 2) fail("no real memberships");
    n_real_groups = groups.n - 1;
    /* NOT pre-ranked the way an artifact's candidates are: rank here, exactly
     * as the text path does. */
    order = xmalloc(n_real_groups * sizeof(*order));
    for (size_t i = 0; i < n_real_groups; ++i) order[i] = (uint32_t)i;
    sort_ctx = &groups;
    qsort(order, n_real_groups, sizeof(*order), group_cmp);
  }

  units_t units = {0};
  uint64_t output = 0;
  size_t unit_first = 0;
  uint32_t unit_members = 0, unit_cpgs = 0;
  for (size_t oi = 0; oi < n_real_groups; ++oi) {
    group_t *g = &groups.a[order[oi]];
    /* adding this membership would exceed the target unit size -> flush */
    if (unit_members && (uint64_t)unit_cpgs + g->count > target) {
      msui_unit_t u = {.output_offset = output - unit_cpgs,
                       .first_membership = (uint32_t)unit_first,
                       .membership_count = unit_members, .cpg_count = unit_cpgs,
                       .flags = unit_members == 1 ? MSUI_UNIT_PURE : 0};
      units_push(&units, u);
      unit_first = oi; unit_members = 0; unit_cpgs = 0;
    }
    g->output_offset = output;
    g->unit = (uint32_t)units.n;
    output += g->count;
    unit_cpgs += g->count;
    ++unit_members;
    /* this single membership is itself larger than target -> its own
       oversized unit */
    if (g->count > target) {
      msui_unit_t u = {.output_offset = output - unit_cpgs,
                       .first_membership = (uint32_t)unit_first,
                       .membership_count = 1, .cpg_count = unit_cpgs,
                       .flags = MSUI_UNIT_PURE | MSUI_UNIT_OVERSIZED};
      units_push(&units, u);
      unit_first = oi + 1; unit_members = 0; unit_cpgs = 0;
    }
  }
  if (unit_members) {
    msui_unit_t u = {.output_offset = output - unit_cpgs,
                     .first_membership = (uint32_t)unit_first,
                     .membership_count = unit_members, .cpg_count = unit_cpgs,
                     .flags = unit_members == 1 ? MSUI_UNIT_PURE : 0};
    units_push(&units, u);
  }
  const uint32_t n_real_units = (uint32_t)units.n;
  const uint64_t n_real_cpg = output;
  const uint64_t n_pna_cpg = groups.a[pna_group].count;
  for (uint64_t done = 0; done < n_pna_cpg;) {
    uint32_t z = (uint32_t)((n_pna_cpg - done) > target ? target : n_pna_cpg - done);
    msui_unit_t u = {.output_offset = n_real_cpg + done,
                     .first_membership = UINT32_MAX, .membership_count = z,
                     .cpg_count = z, .flags = MSUI_UNIT_PNA};
    units_push(&units, u);
    done += z;
  }
  if (units.n > UINT32_MAX) fail("too many processing units");

  uint32_t *cpg = xmalloc((size_t)total * sizeof(*cpg));
  uint64_t genomic = 0, pna_seen = 0;
  {
    /* cpg_pat is in genomic order, so CpGs land inside each unit in genomic
     * order exactly as the binstring scan would place them. */
    for (uint64_t i = 0; i < bm.n_cpg; ++i) {
      uint32_t r = bm.cpg_pat[i];
      uint64_t dest;
      if (r == MRMP_PNA_MEMBERSHIP) {
        dest = n_real_cpg + pna_seen++;
        ++groups.a[pna_group].seen;
      } else {
        if (r >= n_real_groups) fail("binstring index out of range");
        group_t *g = &groups.a[r];
        if (g->seen >= g->count) fail("binstring frequency exceeds count");
        dest = g->output_offset + g->seen;
        ++g->seen;
      }
      cpg[dest] = (uint32_t)i;
    }
    genomic = bm.n_cpg;
  }
  ms_binstring_map_free(&bm);            /* scatter done; groups own the counts */
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
  h.version = 1; h.flags = 1; h.pattern_length = g_pattern_len;
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

  FILE *f = fopen(out_path, "wb");
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
    "format\tMSUIDX1\nreference\t%s\nmincov\t%u\nbeta_threshold\t%.3f\n"
    "pattern_length\t%u\ntotal_cpgs\t%" PRIu64 "\nreal_cpgs\t%" PRIu64
    "\npna_cpgs\t%" PRIu64 "\nreal_memberships\t%zu\nunits\t%u\n"
    "real_units\t%u\nmembership_pure_units\t%u\nmixed_units\t%u\n"
    "pna_units\t%u\noversized_units\t%u\ntarget_cpgs_per_unit\t%u\n"
    "membership_order\tdecreasing_size_then_ternary_key\n"
    "within_membership_order\tgenomic_cpg_index\n"
    "real_membership_split\tfalse\npna_semantics\timplicit_singletons\n"
    "pattern_checksum\t%016" PRIx64 "\nfile_bytes\t%" PRIu64 "\n",
    store_path, mincov, (double)beta_thr,
    g_pattern_len, total, n_real_cpg, n_pna_cpg,
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
    "[methscope] upscale-set-units: CpGs=%" PRIu64 " real=%" PRIu64
    " PNA=%" PRIu64 " memberships=%zu\n"
    "[methscope] upscale-set-units: units=%u pure=%u mixed=%u PNA=%u oversized=%u\n"
    "[methscope] upscale-set-units: unit CpGs min=%" PRIu64
    " mean=%.1f max=%" PRIu64 "\n"
    "[methscope] upscale-set-units: wrote %s (%" PRIu64 " bytes) and %s\n",
    total, n_real_cpg, n_pna_cpg, n_real_groups, h.n_units, pure,
    n_real_units - pure, h.n_pna_units, oversized, min_unit,
    (double)total / h.n_units, max_unit, out_path, h.file_bytes, manifest);

  free(members); free(cpg); free(units.a); free(order);
  free(groups.a);
  return 0;
}

/* ---- `methscope inspect UNITS.msui` ------------------------------------- */

void ms_msui_report(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) fail_path("cannot open", path);
  msui_header_t h;
  if (fread(&h, 1, sizeof(h), f) != sizeof(h)) fail_path("truncated", path);
  if (fclose(f)) fail_path("error closing", path);
  if (memcmp(h.magic, "MSUIDX1", 7) || h.version != 1)
    fail_path("not a MSUIDX1 processing-unit index", path);
  printf("format\tMSUIDX1 v%u\n", h.version);
  printf("cpgs\t%" PRIu64 "\t(%" PRIu64 " real + %" PRIu64 " PNA)\n",
         h.n_cpg, h.n_real_cpg, h.n_pna_cpg);
  printf("pattern_length\t%u\n", h.pattern_length);
  printf("target_unit_cpgs\t%u\n", h.target_unit_cpgs);
  printf("units\t%u\t(%u PNA)\n", h.n_units, h.n_pna_units);
  printf("real_memberships\t%u\n", h.n_real_memberships);
  printf("pattern_checksum\t%016" PRIx64 "\n", h.pattern_checksum);
  printf("file_bytes\t%" PRIu64 "\n", h.file_bytes);
}
