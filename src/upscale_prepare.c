// SPDX-License-Identifier: AGPL-3.0-or-later
/* Build the compact raw-CG msur used by the global upscale trainer.
 *
 * The input .cg remains the immutable truth store.  This command writes only
 * deterministic YAME-compatible sampled CpG positions and MRMP summaries, so
 * it deliberately does not create downsampled .cg files or replicated TSVs. */
/* random_r/initstate_r are GNU extensions; needed before any header. */
#define _GNU_SOURCE
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
#include "cfile.h"
#include "summary.h"
#include "mrmp.h"

#include "msur.h"

#define MSUR_MAX_LEVELS 16

static void pdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] upscale-featurize: %s: %s\n", msg, arg);
  else fprintf(stderr, "[methscope] upscale-featurize: %s\n", msg);
  exit(1);
}

static void *xmalloc(size_t n, const char *what) {
  void *p = malloc(n ? n : 1);
  if (!p) pdie("out of memory", what);
  return p;
}

static uint64_t parse_u64(const char *s, const char *what) {
  errno = 0; char *end = NULL; unsigned long long v = strtoull(s, &end, 10);
  if (errno || end == s || *end) pdie("invalid integer", what);
  return (uint64_t)v;
}

/* Exact arithmetic used by YAME/src/dsample.c.  Keeping it here, rather than
 * changing the random protocol, lets an msur be checked against existing
 * `yame dsample -s SEED -r 1 -N N` simulations on this platform.
 *
 * The state is EXPLICIT rather than libc's global, because rand() is not
 * thread-safe and the replicate loop is the obvious axis to parallelise. This
 * is not a protocol change: glibc's rand()/srand() are random()/srandom() over
 * a 128-byte TYPE_3 state, so initstate_r() with the same size and seed yields
 * the identical sequence -- verified value-for-value before this was written. */
typedef struct { char st[128]; struct random_data d; } ms_rng_t;

static void ms_rng_seed(ms_rng_t *r, unsigned seed) {
  memset(&r->d, 0, sizeof r->d);
  initstate_r(seed, r->st, sizeof r->st, &r->d);
}

static double yame_rand01(ms_rng_t *r) {
  int32_t v; random_r(&r->d, &v);
  return (double)v / ((double)RAND_MAX + 1.0);
}

static void partial_fisher_yates(uint32_t *a, uint64_t n, uint64_t k,
                                 uint32_t *swap_j, ms_rng_t *rng) {
  if (k > n) k = n;
  for (uint64_t i = 0; i < k; ++i) {
    uint64_t j = i + (uint64_t)(yame_rand01(rng) * (double)(n - i));
    swap_j[i] = (uint32_t)j;
    uint32_t t = a[i]; a[i] = a[j]; a[j] = t;
  }
}

static void restore_partial_shuffle(uint32_t *a, uint32_t k,
                                    const uint32_t *swap_j) {
  while (k) {
    --k; uint32_t j = swap_j[k], t = a[k]; a[k] = a[j]; a[j] = t;
  }
}

static int pnum(const char *s) {
  if (!s || (s[0] != 'P' && s[0] != 'p')) return 0;
  char *end = NULL; long n = strtol(s + 1, &end, 10);
  return end != s + 1 && *end == '\0' && n > 0 && n <= INT_MAX ? (int)n : 0;
}

static int cmp_u32(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return (x > y) - (x < y);
}

static int usage(void) {
  ms_help(stderr,
    "Usage: methscope upscale-featurize [options] TRUTH.cg IN.mrmp OUT.msur\n\n"
    "Create a compact exact-YAME sampling msur for global upscale training.\n"
    "The original TRUTH.cg remains the truth store and is never copied.\n\n"
    "  TRUTH.cg         continuous format-3 YAME .cg truth store\n"
    "  IN.mrmp          MRMPIDX1 artifact (preferred) or an exported .cm\n"
    "  OUT.msur         output msur\n\n"
    "Options:\n"
    "  --reps N         deterministic simulations per --sample level (default 100)\n"
    "  --sample N[,N..] CpGs retained per cell/replicate (default 29000). Give a\n"
    "                   comma list to mix coverages: --reps applies to EACH level,\n"
    "                   so --reps 100 --sample 14701,29000,147009 writes 300\n"
    "                   replicates. Each level stores exactly its own record size.\n"
    "  --sample-logrange MIN,MAX\n"
    "                   continuous ladder instead of --sample: --reps is the TOTAL\n"
    "                   replicate count, each getting its own sample size spread\n"
    "                   in log space over [MIN,MAX]\n"
    "  --sample-skew A  bend that draw (default 0.5). A<1 puts more replicates at\n"
    "                   the DENSE end, where finer differences must be resolved;\n"
    "                   A=1 is plain log-uniform\n"
    "  --binarize       one read per observed CpG: replace each sampled beta with a\n"
    "                   Bernoulli(beta) draw, as `yame dsample -b` does\n"
    "  --patterns N     retain feature summaries P1..PN (default 1000)\n"
    "  --in-memory      inflate the truth store once and reuse it for all replicates\n"
    "  --manifest PATH  write provenance TSV (default OUT.msur.tsv)\n"
    "  -h, --help       show this help\n"
    "\n"
    "One msur serves many models. What is stored per (cell, replicate) is the raw\n"
    "summary -- beta and covered-CpG count for every pattern, plus the observed set\n"
    "-- NOT the encoder input. Which projection of that the encoder sees is chosen\n"
    "at TRAINING time, so a single msur covers every `upscale-train --features`\n"
    "mode (missing / count / beta / scalar) and any `--patterns P` up to the N used\n"
    "here. Only the *simulation* is fixed at this step: the cells, the replicate\n"
    "count, the coverage ladder and --binarize. So featurize at the widest pattern\n"
    "vocabulary you might ever want -- P can be narrowed later, never widened --\n"
    "and vary features and width by retraining, which costs minutes against the\n"
    "hours and tens of GB a rebuild here costs.\n");
  return 1;
}

static void write_or_die(FILE *fp, const void *p, size_t n, const char *path) {
  if (n && fwrite(p, 1, n, fp) != n) pdie("write failed", path);
}

int main_upscale_prepare(int argc, char *argv[]) {
  const char *out = NULL, *truth = NULL, *mrmp = NULL, *manifest = NULL;
  const char *pos[3] = {NULL, NULL, NULL};
  int npos = 0;
  uint32_t reps = 100, patterns = 1000;
  uint32_t levels[MSUR_MAX_LEVELS] = {29000}, n_levels = 1, max_sample = 29000;
  uint32_t log_min = 0, log_max = 0;   /* --sample-logrange */
  double log_skew = 0.5;               /* <1 leans dense; 1 = plain log-uniform */
  int in_memory = 0, binarize = 0;
  int embed_truth = 1;   /* always embed truth -- upscale-train requires it */
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage();
      return 0;
    }
    if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (uint32_t)parse_u64(argv[++i], "--reps");
    else if (!strcmp(argv[i], "--sample") && i + 1 < argc) {
      const char *s = argv[++i];
      n_levels = 0; max_sample = 0;
      for (;;) {
        const char *comma = strchr(s, ',');
        char buf[32];
        size_t len = comma ? (size_t)(comma - s) : strlen(s);
        if (!len || len >= sizeof(buf)) pdie("invalid --sample level", argv[i]);
        memcpy(buf, s, len); buf[len] = '\0';
        if (n_levels == MSUR_MAX_LEVELS) pdie("too many --sample levels", argv[i]);
        levels[n_levels] = (uint32_t)parse_u64(buf, "--sample");
        if (!levels[n_levels]) pdie("--sample level must be positive", argv[i]);
        if (levels[n_levels] > max_sample) max_sample = levels[n_levels];
        ++n_levels;
        if (!comma) break;
        s = comma + 1;
      }
    }
    else if (!strcmp(argv[i], "--sample-logrange") && i + 1 < argc) {
      const char *v = argv[++i], *comma = strchr(v, ',');
      char buf[32];
      if (!comma || (size_t)(comma - v) >= sizeof(buf))
        pdie("--sample-logrange wants MIN,MAX", v);
      memcpy(buf, v, (size_t)(comma - v)); buf[comma - v] = '\0';
      log_min = (uint32_t)parse_u64(buf, "--sample-logrange MIN");
      log_max = (uint32_t)parse_u64(comma + 1, "--sample-logrange MAX");
      if (!log_min || log_max < log_min)
        pdie("--sample-logrange needs 0 < MIN <= MAX", v);
    }
    else if (!strcmp(argv[i], "--sample-skew") && i + 1 < argc) {
      char *end = NULL; log_skew = strtod(argv[++i], &end);
      if (!end || *end || !(log_skew > 0)) pdie("--sample-skew must be > 0", argv[i]);
    }
    else if (!strcmp(argv[i], "--binarize")) binarize = 1;
    else if (!strcmp(argv[i], "--patterns") && i + 1 < argc) patterns = (uint32_t)parse_u64(argv[++i], "--patterns");
    else if (!strcmp(argv[i], "--in-memory")) in_memory = 1;
    else if (argv[i][0] == '-') { usage(); pdie("unrecognized or incomplete option", argv[i]); }
    else if (npos < 3) pos[npos++] = argv[i];
    else pdie("too many arguments", argv[i]);
  }
  if (npos != 3) {
    usage();
    pdie("need TRUTH.cg, IN.mrmp, and OUT.msur", NULL);
  }
  if (!reps || (!log_min && !n_levels) || !patterns) return usage();
  if ((uint64_t)reps * n_levels > UINT32_MAX) pdie("too many replicates", NULL);

  /* Per-replicate sample sizes.  Either the small fixed ladder (--sample, reps
   * per level) or a continuous one (--sample-logrange, reps total).
   *
   * The continuous draw is stratified in LOG space -- coverage effects are
   * log-scaled, which is why every ladder we have used halves -- so the range
   * is covered evenly and the msur stays reproducible (no RNG).  --sample-skew
   * bends it: u^skew with skew < 1 pushes replicates toward the DENSE end,
   * where the model must resolve finer differences and an even ladder therefore
   * under-samples.  skew = 1 is plain log-uniform. */
  uint32_t total_reps = log_min ? reps : reps * n_levels;
  uint32_t *rep_sample = xmalloc((size_t)total_reps * sizeof(*rep_sample),
                                 "per-replicate sample sizes");
  if (log_min) {
    double lo = log((double)log_min), span = log((double)log_max) - lo;
    max_sample = 0;
    for (uint32_t r = 0; r < total_reps; ++r) {
      double u = total_reps > 1 ? (double)r / (double)(total_reps - 1) : 1.0;
      double n = exp(lo + span * pow(u, log_skew));
      rep_sample[r] = (uint32_t)(n + 0.5);
      if (rep_sample[r] < 1) rep_sample[r] = 1;
      if (rep_sample[r] > max_sample) max_sample = rep_sample[r];
    }
  } else {
    for (uint32_t r = 0; r < total_reps; ++r) rep_sample[r] = levels[r / reps];
  }
  truth = pos[0]; mrmp = pos[1]; out = pos[2];
  if (sizeof(msur_header_t) != 72 || sizeof(msur_header3_t) != 80 ||
      sizeof(msur_rep_t) != 24) pdie("unexpected msur header layout", NULL);
  if (patterns > UINT16_MAX - 1) pdie("--patterns exceeds uint16 group format", NULL);

  /* Size the truth store first; the MRMP is then read against its CpG count. */
  cfile_t check = open_cfile((char *)truth);
  cdata_t first = read_cdata1(&check);
  if (!first.n) pdie("truth store is empty", truth);
  decompress_in_situ(&first);
  if (first.fmt != '3') pdie("truth store must be continuous format 3", truth);
  uint64_t n_cpg = first.n;
  uint32_t n_cells = 0;
  free_cdata(&first);
  for (;;) { cdata_t c = read_cdata1(&check); if (!c.n) { free_cdata(&c); break; } ++n_cells; free_cdata(&c); }
  ++n_cells; /* account for first record */
  bgzf_close(check.fh);
  if (n_cpg > UINT32_MAX) pdie("msur supports at most 2^32-1 CpGs", truth);

  /* Feature group of CpG i; PNA/background stays group 0 and is never a
   * feature.  Prefer the MRMPIDX1 artifact, so this map and the mask the model
   * ends up bundling derive from one source and cannot disagree; an
   * already-exported categorical .cm is still accepted. */
  /* A CHAIN featurizes as the concatenation of its sets: a CpG lands in one
   * pattern PER SET, so a tree's leaf contributes columns its root cannot
   * express. n_sets * n_cpg uint16 is 823 MB for a 14-set chain over hg38 --
   * paid once, then reused for every replicate. Constant binstrings never
   * appear: mrmp-build routes all-0 and all-1 to PNA unless asked otherwise,
   * so a node contributes only what separates something in its own subset. */
  uint32_t n_sets = 1, *col0 = NULL, ncol = patterns;
  if (ms_mrmp_is_artifact(mrmp)) {
    ms_mrmpset_t *probe = ms_mrmpset_open(mrmp);
    n_sets = probe->n_sets;
    ms_mrmpset_free(probe);
  }
  uint16_t *group = xmalloc((size_t)(n_sets > 1 ? n_sets : 1) * n_cpg * sizeof(*group),
                            "MRMP group map");
  if (n_sets > 1) {
    col0 = xmalloc((size_t)(n_sets + 1) * sizeof(*col0), "chain column offsets");
    ncol = ms_mrmp_group_map_chain(mrmp, group, n_cpg, patterns, col0, &n_sets);
    fprintf(stderr, "[methscope] upscale-featurize: chain of %u sets -> %u columns\n",
            n_sets, ncol);
  } else if (ms_mrmp_is_artifact(mrmp)) {
    ms_mrmp_group_map(mrmp, group, n_cpg, patterns);
  } else {
    cfile_t cmf = open_cfile((char *)mrmp);
    cdata_t cm = read_cdata1(&cmf);
    cdata_t extra = read_cdata1(&cmf);
    bgzf_close(cmf.fh);
    if (!cm.n || extra.n) pdie("MRMP must contain exactly one record", mrmp);
    free_cdata(&extra);
    decompress_in_situ(&cm);
    if (cm.fmt != '2') pdie("MRMP must be categorical format 2", mrmp);
    fmt2_set_aux(&cm);
    if (cm.n != n_cpg) pdie("MRMP and truth CpG counts differ", mrmp);
    for (uint64_t i = 0; i < n_cpg; ++i) {
      int p = pnum(f2_get_string(&cm, i));
      group[i] = (p > 0 && p <= (int)patterns) ? (uint16_t)p : 0;
    }
    free_cdata(&cm);
  }

  /* A full continuous hg38 x 207-cell store inflates to about 46 GiB.  On the
   * 120-GiB training node this trades ~750 GiB of repeated filesystem reads
   * for one read and makes the 100-replicate cache build CPU-bound. */
  cdata_t *memory_cells = NULL;
  uint32_t **memory_eligible = NULL;
  uint64_t *memory_n_eligible = NULL;
  if (in_memory) {
    fprintf(stderr, "[methscope] upscale-featurize: inflating %u truth cells in memory\n", n_cells);
    memory_cells = calloc(n_cells, sizeof(*memory_cells));
    memory_eligible = calloc(n_cells, sizeof(*memory_eligible));
    memory_n_eligible = calloc(n_cells, sizeof(*memory_n_eligible));
    if (!memory_cells || !memory_eligible || !memory_n_eligible)
      pdie("out of memory allocating truth-cell table", truth);
    cfile_t cf = open_cfile((char *)truth);
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
      cdata_t c = read_cdata1(&cf);
      if (!c.n) pdie("truth store changed while reading", truth);
      decompress_in_situ(&c);
      if (c.fmt != '3' || c.n != n_cpg) pdie("inconsistent truth record", truth);
      memory_cells[cell] = c;
      uint64_t ne = 0;
      for (uint64_t i = 0; i < n_cpg; ++i) ne += f3_get_mu(&c, i) != 0;
      memory_eligible[cell] = xmalloc((size_t)ne * sizeof(uint32_t), "eligible CpGs");
      memory_n_eligible[cell] = ne;
      uint64_t q = 0;
      for (uint64_t i = 0; i < n_cpg; ++i)
        if (f3_get_mu(&c, i)) memory_eligible[cell][q++] = (uint32_t)i;
    }
    cdata_t end = read_cdata1(&cf); if (end.n) pdie("truth store changed while reading", truth); free_cdata(&end);
    bgzf_close(cf.fh);
  }

  /* One level keeps the v2 fixed-width layout, so a single-level msur stays
   * byte-identical to one built before --sample took a list.  Mixed levels get
   * v3: per-replicate tables instead of padding every short record out to the
   * widest level (a 143-CpG replicate padded to 147,009 is 99.9% filler). */
  const uint64_t feat_bytes =
    (uint64_t)patterns * (sizeof(float) + sizeof(uint32_t));
  int v3 = 0;
  for (uint32_t r = 1; r < total_reps && !v3; ++r) v3 = rep_sample[r] != rep_sample[0];

  FILE *fp = fopen(out, "wb");
  if (!fp) pdie("cannot create output", out);
  msur_header3_t h3;
  memset(&h3, 0, sizeof(h3));
  msur_header_t *hp = &h3.h;
  memcpy(hp->magic, v3 ? MSUR3_MAGIC : MSUR2_MAGIC, 8);
  hp->version = v3 ? 3u : 2u;
  /* n_patterns is the COLUMN count, which for a chain is the concatenation
   * width, not --patterns. upscale-train narrows against this number, so
   * recording the flag instead would let it read 200 columns of a 658-wide
   * record and silently mis-slice every row. */
  hp->n_cells = n_cells; hp->n_reps = total_reps; hp->n_patterns = ncol;
  hp->n_cpg = n_cpg; hp->sampled_per_cell = max_sample;
  hp->flags = (embed_truth ? MSUR_F_TRUTH_U16 : 0)
    | (binarize ? MSUR_F_BINARIZED : 0)
    | (v3 ? MSUR_F_MIXED_SAMPLE : 0);
  const uint64_t head_bytes = v3 ? sizeof(h3) : sizeof(*hp);
  hp->groups_offset = head_bytes;
  hp->truth_offset = embed_truth ? hp->groups_offset + n_cpg * sizeof(uint16_t) : 0;
  uint64_t after_truth = hp->groups_offset + n_cpg * sizeof(uint16_t)
    + (embed_truth ? (uint64_t)n_cells * n_cpg * sizeof(uint16_t) : 0);

  msur_rep_t *reptab = NULL;
  if (v3) {
    /* record_bytes varies by level, so the whole table is computable up front
     * and every record stays O(1) addressable. */
    h3.rep_table_offset = after_truth;
    hp->records_offset = after_truth + (uint64_t)total_reps * sizeof(msur_rep_t);
    hp->record_bytes = 0;          /* sentinel: variable, consult the table */
    reptab = xmalloc((size_t)total_reps * sizeof(*reptab), "replicate table");
    uint64_t at = hp->records_offset;
    uint64_t bmb = msur_bitmap_bytes(n_cpg);
    uint32_t n_bitmap = 0;
    /* Escape hatch: pin the encoding instead of choosing by size.  "list"
     * reproduces pre-bitmap output byte for byte (and is how the two encodings
     * are checked for equivalence); "bitmap" forces it everywhere. */
    const char *enc_env = getenv("MSUR_OBSERVED_ENCODING");
    int force_list = enc_env && !strcmp(enc_env, "list");
    int force_bitmap = enc_env && !strcmp(enc_env, "bitmap");
    if (enc_env && !force_list && !force_bitmap && strcmp(enc_env, "auto"))
      pdie("MSUR_OBSERVED_ENCODING must be list, bitmap, or auto", enc_env);
    for (uint32_t r = 0; r < total_reps; ++r) {
      uint32_t s = rep_sample[r];
      /* Whichever observed-set encoding is smaller for THIS replicate. */
      uint64_t list_bytes = (uint64_t)s * sizeof(uint32_t);
      int bitmap = force_bitmap || (!force_list && bmb < list_bytes);
      reptab[r].sample = s;
      reptab[r].flags = bitmap ? MSUR_ENC_BITMAP : MSUR_ENC_LIST;
      reptab[r].record_bytes = feat_bytes + (bitmap ? bmb : list_bytes);
      reptab[r].offset = at;
      at += (uint64_t)n_cells * reptab[r].record_bytes;
      n_bitmap += bitmap != 0;
    }
    if (n_bitmap)
      fprintf(stderr, "[methscope] upscale-featurize: bitmap observed-set for "
              "%u/%u replicates (>= %" PRIu64 " CpGs); records %.1f GiB\n",
              n_bitmap, total_reps, bmb / sizeof(uint32_t),
              (double)(at - hp->records_offset) / (1024.0 * 1024.0 * 1024.0));
  } else {
    hp->records_offset = after_truth;
    hp->record_bytes = feat_bytes + (uint64_t)max_sample * sizeof(uint32_t);
  }
  write_or_die(fp, &h3, (size_t)head_bytes, out);
  write_or_die(fp, group, (size_t)n_cpg * sizeof(*group), out);

  if (embed_truth) {
    fprintf(stderr, "[methscope] upscale-featurize: writing quantized truth matrix\n");
    uint16_t *truth_row = xmalloc((size_t)n_cpg * sizeof(*truth_row), "truth u16 row");
    cfile_t tcf = {0}; if (!memory_cells) tcf = open_cfile((char *)truth);
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
      cdata_t c = memory_cells ? memory_cells[cell] : read_cdata1(&tcf);
      if (!memory_cells) { decompress_in_situ(&c); if (c.fmt != '3' || c.n != n_cpg) pdie("inconsistent truth record", truth); }
      for (uint64_t i = 0; i < n_cpg; ++i) {
        uint64_t mu = f3_get_mu(&c, i);
        /* u16 code = beta*65534; 65535 (UINT16_MAX) = missing */
        truth_row[i] = mu ? (uint16_t)llround(MU2beta(mu) * 65534.0) : UINT16_MAX;
      }
      write_or_die(fp, truth_row, (size_t)n_cpg * sizeof(*truth_row), out);
      if (!memory_cells) free_cdata(&c);
    }
    if (!memory_cells) bgzf_close(tcf.fh);
    free(truth_row);
  }

  uint32_t *eligible = memory_cells ? NULL : xmalloc((size_t)n_cpg * sizeof(*eligible), "sampling workspace");
  double *sum = xmalloc((size_t)ncol * sizeof(*sum), "MRMP sums");
  uint32_t *count = xmalloc((size_t)ncol * sizeof(*count), "MRMP counts");
  float *beta = xmalloc((size_t)ncol * sizeof(*beta), "feature beta");
  uint32_t *selected = xmalloc((size_t)max_sample * sizeof(*selected), "sampled positions");
  uint32_t *swap_j = xmalloc((size_t)max_sample * sizeof(*swap_j), "sampling swaps");
  /* Only allocated when some replicate is dense enough to prefer a bitmap. */
  uint64_t bitmap_bytes = msur_bitmap_bytes(n_cpg);
  unsigned char *bitmap = NULL;
  if (v3) for (uint32_t r = 0; r < total_reps && !bitmap; ++r)
    if (reptab[r].flags == MSUR_ENC_BITMAP)
      bitmap = xmalloc((size_t)bitmap_bytes, "observed-set bitmap");

  if (v3) write_or_die(fp, reptab, (size_t)total_reps * sizeof(*reptab), out);

  for (uint32_t r = 0; r < total_reps; ++r) {
    /* Replicates run level-major, so r = level*reps + rep.  With one level this
     * is the historical loop exactly, seeds and all, and a single-level msur
     * is byte-identical to one built before --sample took a list. */
    uint32_t sample = rep_sample[r];
    int rep_bitmap = v3 && reptab[r].flags == MSUR_ENC_BITMAP;
    fprintf(stderr,
      "[methscope] upscale-featurize: simulation %u/%u (sample %u%s%s)\n",
      r + 1, total_reps, sample, binarize ? ", binarized" : "",
      rep_bitmap ? ", bitmap" : "");
    ms_rng_t rng; ms_rng_seed(&rng, r + 1);  /* historical: --seed 1..N */
    cfile_t cf = {0};
    if (!memory_cells) cf = open_cfile((char *)truth);
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
      cdata_t c = memory_cells ? memory_cells[cell] : read_cdata1(&cf);
      if (!c.n) pdie("truth store changed while reading", truth);
      if (!memory_cells) {
        decompress_in_situ(&c);
        if (c.fmt != '3' || c.n != n_cpg) pdie("inconsistent truth record", truth);
      }
      uint64_t ne = memory_cells ? memory_n_eligible[cell] : 0;
      uint32_t *cell_eligible = memory_cells ? memory_eligible[cell] : eligible;
      if (!memory_cells)
        for (uint64_t i = 0; i < n_cpg; ++i) if (f3_get_mu(&c, i)) cell_eligible[ne++] = (uint32_t)i;
      if (ne < sample) pdie("cell has fewer eligible CpGs than --sample", truth);
      partial_fisher_yates(cell_eligible, ne, sample, swap_j, &rng);
      memset(sum, 0, (size_t)ncol * sizeof(*sum));
      memset(count, 0, (size_t)ncol * sizeof(*count));
      for (uint32_t k = 0; k < sample; ++k) {
        uint64_t pos = cell_eligible[k]; selected[k] = (uint32_t)pos;
        /* The binarise draw is per CpG, NOT per set: one observed read gives
         * one call, and every set that contains this CpG must see the same
         * one. Drawing inside the set loop would consume n_sets randoms per
         * CpG and desynchronise the flat path's RNG stream. */
        double b = MU2beta(f3_get_mu(&c, pos));
        int drawn = 0;
        for (uint32_t sx = 0; sx < n_sets; ++sx) {
          uint16_t g = group[(uint64_t)sx * n_cpg + pos];
          if (!g) continue;
          if (binarize && !drawn) { b = yame_rand01(&rng) < b ? 1.0 : 0.0; drawn = 1; }
          sum[g - 1] += b; ++count[g - 1];
        }
      }
      for (uint32_t g = 0; g < ncol; ++g) beta[g] = count[g] ? (float)(sum[g] / count[g]) : NAN;
      qsort(selected, sample, sizeof(*selected), cmp_u32);
      write_or_die(fp, beta, (size_t)ncol * sizeof(*beta), out);
      write_or_die(fp, count, (size_t)ncol * sizeof(*count), out);
      /* Exactly `sample` entries -- no padding.  In v3 the replicate table
       * carries this record's length, so shorter levels cost nothing; dense
       * levels switch to a whole-genome bitmap, which is both smaller and O(1)
       * to test.  Either way the membership set is identical. */
      if (rep_bitmap) {
        memset(bitmap, 0, (size_t)bitmap_bytes);
        for (uint32_t k = 0; k < sample; ++k)
          bitmap[selected[k] >> 3] |= (unsigned char)(1u << (selected[k] & 7));
        write_or_die(fp, bitmap, (size_t)bitmap_bytes, out);
      } else {
        write_or_die(fp, selected, (size_t)sample * sizeof(*selected), out);
      }
      restore_partial_shuffle(cell_eligible, sample, swap_j);
      if (!memory_cells) free_cdata(&c);
    }
    if (!memory_cells) {
      cdata_t end = read_cdata1(&cf); if (end.n) pdie("truth store changed while reading", truth); free_cdata(&end);
      bgzf_close(cf.fh);
    }
  }
  if (fclose(fp)) pdie("error closing output", out);

  char auto_manifest[PATH_MAX];
  if (!manifest) { if (snprintf(auto_manifest, sizeof(auto_manifest), "%s.tsv", out) >= (int)sizeof(auto_manifest)) pdie("output path too long", out); manifest = auto_manifest; }
  FILE *mf = fopen(manifest, "w"); if (!mf) pdie("cannot create manifest", manifest);
  char level_list[MSUR_MAX_LEVELS * 12];
  if (log_min) {
    snprintf(level_list, sizeof(level_list), "logrange %u-%u skew %.3g",
             log_min, log_max, log_skew);
  } else {
    int at = 0;
    for (uint32_t k = 0; k < n_levels; ++k)
      at += snprintf(level_list + at, sizeof(level_list) - (size_t)at,
                     k ? ",%u" : "%u", levels[k]);
  }
  fprintf(mf, "format\t%s\nversion\t%u\ntruth_cg\t%s\nmrmp\t%s\nn_cells\t%u\nn_cpg\t%" PRIu64 "\nn_reps\t%u\nreps_per_level\t%u\nsample_levels\t%s\nsampled_per_cell\t%u\nbinarized\t%s\nn_patterns\t%u\ntruth_encoding\t%s\ngroups_offset\t%" PRIu64 "\ntruth_offset\t%" PRIu64 "\nrecords_offset\t%" PRIu64 "\nrecord_bytes\t%" PRIu64 "\nrandom_protocol\tYAME_dsample_partial_fisher_yates_rand_seed_1_to_n\nfeature_columns\tP1..P%u (PNA excluded; a CHAIN concatenates its sets, so this is the total across them)\n", v3 ? "MSURAW3" : "MSURAW2", hp->version, truth, mrmp, n_cells, n_cpg, total_reps, log_min ? total_reps : reps, level_list, max_sample, binarize ? "yes (one read per observed CpG)" : "no", ncol, embed_truth ? "uint16_beta_0_65534_missing_65535" : "external_cg", hp->groups_offset, hp->truth_offset, hp->records_offset, hp->record_bytes, ncol);
  if (fclose(mf)) pdie("error closing manifest", manifest);
  if (memory_cells) {
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
      free_cdata(&memory_cells[cell]); free(memory_eligible[cell]);
    }
    free(memory_cells);
    free(memory_eligible); free(memory_n_eligible);
  }
  free(group); free(eligible); free(sum); free(count); free(beta); free(selected); free(swap_j);
  fprintf(stderr, "[methscope] upscale-featurize: wrote %s and %s\n", out, manifest);
  return 0;
}

/* ---- `methscope inspect DATA.msur` -------------------------------------- */

void ms_msur_report(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) pdie("cannot open", path);
  msur_header3_t h3;
  if (fread(&h3, 1, sizeof(h3), f) != sizeof(h3)) pdie("truncated", path);
  const msur_header_t *h = &h3.h;
  int v3 = msur_is_v3(h);
  if (!v3 && !msur_is_v2(h)) pdie("not a MSURAW2/3 training msur", path);
  msur_rep_t *reps = NULL;
  if (v3) {
    reps = xmalloc((size_t)h->n_reps * sizeof(*reps), "replicate table");
    if (fseek(f, (long)h3.rep_table_offset, SEEK_SET) ||
        fread(reps, sizeof(*reps), h->n_reps, f) != h->n_reps)
      pdie("truncated replicate table", path);
  }
  if (fclose(f)) pdie("error closing", path);
  int truth = (h->flags & MSUR_F_TRUTH_U16) && h->truth_offset;
  printf("format\t%s v%u\n", v3 ? "MSURAW3" : "MSURAW2", h->version);
  printf("cells\t%u\n", h->n_cells);
  printf("replicates\t%u\n", h->n_reps);
  printf("rows\t%" PRIu64 "\t(cells x replicates)\n",
         (uint64_t)h->n_cells * h->n_reps);
  printf("cpgs\t%" PRIu64 "\n", h->n_cpg);
  printf("patterns\t%u\n", h->n_patterns);
  printf("observed_beta\t%s\n", (h->flags & MSUR_F_BINARIZED)
         ? "binarized (one read per CpG)" : "continuous (full-depth truth)");
  if (v3) {
    uint32_t nb = 0;
    for (uint32_t r = 0; r < h->n_reps; ++r) nb += reps[r].flags == MSUR_ENC_BITMAP;
    if (nb) printf("observed_set\tbitmap for %u/%u replicates, sorted list for the rest\n",
                   nb, h->n_reps);
    else    printf("observed_set\tsorted list (all replicates)\n");
  } else {
    printf("observed_set\tsorted list (all replicates)\n");
  }
  printf("embedded_truth\t%s\n", truth ? "yes (trainable)" :
         "no (upscale-train will reject it)");
  printf("groups_bytes\t%" PRIu64 "\n", h->n_cpg * 2);
  if (truth)
    printf("truth_bytes\t%" PRIu64 "\n", (uint64_t)h->n_cells * h->n_cpg * 2);
  if (v3) {
    /* Collapse the per-replicate table to one line per distinct level. */
    uint64_t total = 0;
    printf("sampled_per_cell\tvariable (widest %u)\n", h->sampled_per_cell);
    for (uint32_t i = 0; i < h->n_reps; ++i) {
      total += (uint64_t)h->n_cells * reps[i].record_bytes;
      if (i && reps[i].sample == reps[i - 1].sample) continue;
      uint32_t n = 0;
      for (uint32_t j = i; j < h->n_reps && reps[j].sample == reps[i].sample; ++j) ++n;
      printf("  level\t%u CpGs\t%u replicates\trecord %" PRIu64 " B\n",
             reps[i].sample, n, reps[i].record_bytes);
    }
    printf("records_bytes\t%" PRIu64 "\n", total);
    free(reps);
  } else {
    printf("sampled_per_cell\t%u\n", h->sampled_per_cell);
    printf("record_bytes\t%" PRIu64 "\n", h->record_bytes);
    printf("records_bytes\t%" PRIu64 "\n",
           (uint64_t)h->n_cells * h->n_reps * h->record_bytes);
  }
}
