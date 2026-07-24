// SPDX-License-Identifier: AGPL-3.0-or-later
/* Build the compact raw-CG sidecar used by the global upscale trainer.
 *
 * The input .cg remains the immutable truth store.  This command writes only
 * deterministic YAME-compatible sampled CpG positions and MRMP summaries, so
 * it deliberately does not create downsampled .cg files or replicated TSVs. */
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

#define MSUR_MAGIC "MSURAW2\0"
#define MSUR_VERSION 2u
#define MSUR_F_TRUTH_U16 1u

typedef struct {
  char magic[8];
  uint32_t version, n_cells, n_reps, n_patterns;
  uint64_t n_cpg;
  uint32_t sampled_per_cell, flags;
  uint64_t groups_offset, truth_offset, records_offset, record_bytes;
} msur_header_t;

static void pdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] _upscale prepare: %s: %s\n", msg, arg);
  else fprintf(stderr, "[methscope] _upscale prepare: %s\n", msg);
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
 * changing the random protocol, lets a sidecar be checked against existing
 * `yame dsample -s SEED -r 1 -N N` simulations on this platform. */
static double yame_rand01(void) {
  return (double)rand() / ((double)RAND_MAX + 1.0);
}

static void partial_fisher_yates(uint32_t *a, uint64_t n, uint64_t k,
                                 uint32_t *swap_j) {
  if (k > n) k = n;
  for (uint64_t i = 0; i < k; ++i) {
    uint64_t j = i + (uint64_t)(yame_rand01() * (double)(n - i));
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
  fprintf(stderr,
    "Usage: methscope _upscale prepare -o OUT.msur --truth TRUTH.cg --mrmp MRMP.cm [options]\n\n"
    "Create a compact exact-YAME sampling sidecar for global upscale training.\n"
    "The original TRUTH.cg remains the truth store and is never copied.\n\n"
    "Options:\n"
    "  -o PATH          output sidecar (.msur; required)\n"
    "  --truth PATH     continuous format-3 YAME .cg truth store (required)\n"
    "  --mrmp PATH      categorical MRMP .cm with P1..Pn/PNA states (required)\n"
    "  --reps N         deterministic simulations, seeds 1..N (default 100)\n"
    "  --sample N       CpGs retained per cell/replicate (default 29000)\n"
    "  --patterns N     retain feature summaries P1..PN (default 1000)\n"
    "  --in-memory      inflate the truth store once and reuse it for all replicates\n"
    "  --embed-truth    store cell-major beta as uint16 (65535=missing)\n"
    "  --manifest PATH  write provenance TSV (default OUT.msur.tsv)\n"
    "  -h, --help       show this help\n");
  return 1;
}

static void write_or_die(FILE *fp, const void *p, size_t n, const char *path) {
  if (n && fwrite(p, 1, n, fp) != n) pdie("write failed", path);
}

int main_upscale_prepare(int argc, char *argv[]) {
  const char *out = NULL, *truth = NULL, *mrmp = NULL, *manifest = NULL;
  uint32_t reps = 100, patterns = 1000, sample = 29000;
  int in_memory = 0, embed_truth = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage();
      return 0;
    }
    if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--truth") && i + 1 < argc) truth = argv[++i];
    else if (!strcmp(argv[i], "--mrmp") && i + 1 < argc) mrmp = argv[++i];
    else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (uint32_t)parse_u64(argv[++i], "--reps");
    else if (!strcmp(argv[i], "--sample") && i + 1 < argc) sample = (uint32_t)parse_u64(argv[++i], "--sample");
    else if (!strcmp(argv[i], "--patterns") && i + 1 < argc) patterns = (uint32_t)parse_u64(argv[++i], "--patterns");
    else if (!strcmp(argv[i], "--in-memory")) in_memory = 1;
    else if (!strcmp(argv[i], "--embed-truth")) embed_truth = 1;
    else { usage(); pdie("unrecognized or incomplete option", argv[i]); }
  }
  if (!out || !truth || !mrmp || !reps || !sample || !patterns) return usage();
  if (sizeof(msur_header_t) != 72) pdie("unexpected sidecar header layout", NULL);
  if (patterns > UINT16_MAX - 1) pdie("--patterns exceeds uint16 group format", NULL);

  /* Load exactly one categorical MRMP record.  Its state at i is the feature
   * group of CpG i.  PNA/background remains group 0 and is never a feature. */
  cfile_t cmf = open_cfile((char *)mrmp);
  cdata_t cm = read_cdata1(&cmf);
  cdata_t extra = read_cdata1(&cmf);
  bgzf_close(cmf.fh);
  if (!cm.n || extra.n) pdie("MRMP must contain exactly one record", mrmp);
  free_cdata(&extra);
  decompress_in_situ(&cm);
  if (cm.fmt != '2') pdie("MRMP must be categorical format 2", mrmp);
  fmt2_set_aux(&cm);

  cfile_t check = open_cfile((char *)truth);
  cdata_t first = read_cdata1(&check);
  if (!first.n) pdie("truth store is empty", truth);
  decompress_in_situ(&first);
  if (first.fmt != '3') pdie("truth store must be continuous format 3", truth);
  uint64_t n_cpg = first.n;
  if (cm.n != n_cpg) pdie("MRMP and truth CpG counts differ", mrmp);
  uint32_t n_cells = 0;
  free_cdata(&first);
  for (;;) { cdata_t c = read_cdata1(&check); if (!c.n) { free_cdata(&c); break; } ++n_cells; free_cdata(&c); }
  ++n_cells; /* account for first record */
  bgzf_close(check.fh);
  if (n_cpg > UINT32_MAX) pdie("sidecar v1 supports at most 2^32-1 CpGs", truth);

  uint16_t *group = xmalloc((size_t)n_cpg * sizeof(*group), "MRMP group map");
  for (uint64_t i = 0; i < n_cpg; ++i) {
    int p = pnum(f2_get_string(&cm, i));
    group[i] = (p > 0 && p <= (int)patterns) ? (uint16_t)p : 0;
  }
  free_cdata(&cm);

  /* A full continuous hg38 x 207-cell store inflates to about 46 GiB.  On the
   * 120-GiB training node this trades ~750 GiB of repeated filesystem reads
   * for one read and makes the 100-replicate cache build CPU-bound. */
  cdata_t *memory_cells = NULL;
  uint32_t **memory_eligible = NULL;
  uint64_t *memory_n_eligible = NULL;
  if (in_memory) {
    fprintf(stderr, "[methscope] _upscale prepare: inflating %u truth cells in memory\n", n_cells);
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

  FILE *fp = fopen(out, "wb");
  if (!fp) pdie("cannot create output", out);
  msur_header_t h = {0};
  memcpy(h.magic, MSUR_MAGIC, 8); h.version = MSUR_VERSION; h.n_cells = n_cells;
  h.n_reps = reps; h.n_patterns = patterns; h.n_cpg = n_cpg; h.sampled_per_cell = sample;
  h.flags = embed_truth ? MSUR_F_TRUTH_U16 : 0;
  h.groups_offset = sizeof(h);
  h.truth_offset = embed_truth ? h.groups_offset + n_cpg * sizeof(uint16_t) : 0;
  h.records_offset = h.groups_offset + n_cpg * sizeof(uint16_t)
    + (embed_truth ? (uint64_t)n_cells * n_cpg * sizeof(uint16_t) : 0);
  h.record_bytes = (uint64_t)patterns * (sizeof(float) + sizeof(uint32_t))
    + (uint64_t)sample * sizeof(uint32_t);
  write_or_die(fp, &h, sizeof(h), out);
  write_or_die(fp, group, (size_t)n_cpg * sizeof(*group), out);

  if (embed_truth) {
    fprintf(stderr, "[methscope] _upscale prepare: writing quantized truth matrix\n");
    uint16_t *truth_row = xmalloc((size_t)n_cpg * sizeof(*truth_row), "truth u16 row");
    cfile_t tcf = {0}; if (!memory_cells) tcf = open_cfile((char *)truth);
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
      cdata_t c = memory_cells ? memory_cells[cell] : read_cdata1(&tcf);
      if (!memory_cells) { decompress_in_situ(&c); if (c.fmt != '3' || c.n != n_cpg) pdie("inconsistent truth record", truth); }
      for (uint64_t i = 0; i < n_cpg; ++i) {
        uint64_t mu = f3_get_mu(&c, i);
        truth_row[i] = mu ? (uint16_t)llround(MU2beta(mu) * 65534.0) : UINT16_MAX;
      }
      write_or_die(fp, truth_row, (size_t)n_cpg * sizeof(*truth_row), out);
      if (!memory_cells) free_cdata(&c);
    }
    if (!memory_cells) bgzf_close(tcf.fh);
    free(truth_row);
  }

  uint32_t *eligible = memory_cells ? NULL : xmalloc((size_t)n_cpg * sizeof(*eligible), "sampling workspace");
  double *sum = xmalloc((size_t)patterns * sizeof(*sum), "MRMP sums");
  uint32_t *count = xmalloc((size_t)patterns * sizeof(*count), "MRMP counts");
  float *beta = xmalloc((size_t)patterns * sizeof(*beta), "feature beta");
  uint32_t *selected = xmalloc((size_t)sample * sizeof(*selected), "sampled positions");
  uint32_t *swap_j = xmalloc((size_t)sample * sizeof(*swap_j), "sampling swaps");

  for (uint32_t rep = 0; rep < reps; ++rep) {
    fprintf(stderr, "[methscope] _upscale prepare: simulation %u/%u\n", rep + 1, reps);
    srand(rep + 1); /* exact historical convention: --seed 1..N */
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
      partial_fisher_yates(cell_eligible, ne, sample, swap_j);
      memset(sum, 0, (size_t)patterns * sizeof(*sum));
      memset(count, 0, (size_t)patterns * sizeof(*count));
      for (uint32_t k = 0; k < sample; ++k) {
        uint64_t pos = cell_eligible[k]; selected[k] = (uint32_t)pos;
        uint16_t g = group[pos];
        if (g) { sum[g - 1] += MU2beta(f3_get_mu(&c, pos)); ++count[g - 1]; }
      }
      for (uint32_t g = 0; g < patterns; ++g) beta[g] = count[g] ? (float)(sum[g] / count[g]) : NAN;
      qsort(selected, sample, sizeof(*selected), cmp_u32);
      write_or_die(fp, beta, (size_t)patterns * sizeof(*beta), out);
      write_or_die(fp, count, (size_t)patterns * sizeof(*count), out);
      write_or_die(fp, selected, (size_t)sample * sizeof(*selected), out);
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
  fprintf(mf, "format\tMSURAW2\nversion\t%u\ntruth_cg\t%s\nmrmp\t%s\nn_cells\t%u\nn_cpg\t%" PRIu64 "\nn_reps\t%u\nsampled_per_cell\t%u\nn_patterns\t%u\ntruth_encoding\t%s\ngroups_offset\t%" PRIu64 "\ntruth_offset\t%" PRIu64 "\nrecords_offset\t%" PRIu64 "\nrecord_bytes\t%" PRIu64 "\nrandom_protocol\tYAME_dsample_partial_fisher_yates_rand_seed_1_to_n\nfeature_columns\tP1..P%u (PNA excluded)\n", MSUR_VERSION, truth, mrmp, n_cells, n_cpg, reps, sample, patterns, embed_truth ? "uint16_beta_0_65534_missing_65535" : "external_cg", h.groups_offset, h.truth_offset, h.records_offset, h.record_bytes, patterns);
  if (fclose(mf)) pdie("error closing manifest", manifest);
  if (memory_cells) {
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
      free_cdata(&memory_cells[cell]); free(memory_eligible[cell]);
    }
    free(memory_cells);
    free(memory_eligible); free(memory_n_eligible);
  }
  free(group); free(eligible); free(sum); free(count); free(beta); free(selected); free(swap_j);
  fprintf(stderr, "[methscope] _upscale prepare: wrote %s and %s\n", out, manifest);
  return 0;
}
