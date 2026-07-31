// SPDX-License-Identifier: AGPL-3.0-or-later
/* MSFMAT1 -- a featurized record x pattern matrix, shared by the writer
 * (msfm.c) and its consumers (train.c, predict.c) so the two cannot drift.
 *
 * This is a `.msur` stripped to what a LABEL model needs. The upscale msur is
 * dominated by things only a CpG-level decoder uses -- the embedded truth
 * matrix (n_cells x n_cpg x 2 B: 4.97 TB for 84,601 cells), the CpG->pattern
 * groups map, and a per-record observed set. A classifier needs none of them:
 * its target is a class id, not 29.4M CpGs. What is left is the summary the
 * featurization already computes.
 *
 * It is also structurally simpler than msur. Every row is the same width, so
 * there is no replicate table, no variable-length addressing and no
 * list/bitmap encoding choice -- a sparsity-ladder replicate is just another
 * row. Addressing is record * n_patterns, and the whole file mmaps.
 *
 * Layout (offsets are absolute, sections in this order):
 *   [0]              header, 88 bytes
 *   [names_offset]   n_patterns NUL-terminated pattern names
 *   [rows_offset]    n_records  NUL-terminated record names
 *   [labels_offset]  u16 class id per record, then n_classes NUL-terminated
 *                    class names (sorted, so the id IS the model's class index)
 *   [levels_offset]  u32 covered-CpG total per record
 *   [beta_offset]    u16 per (record, pattern), row-major
 *   [count_offset]   u32 per (record, pattern), row-major -- only with F_COUNTS
 */
#ifndef MSFM_H
#define MSFM_H

#include <stdint.h>
#include <string.h>

#define MSFM_MAGIC "MSFMAT1"

/* Full per-pattern covered-CpG counts. Off by default: training reads only
 * betas, and the one aggregate it wants is the always-present level. */
#define MSFM_F_COUNTS 1u

/* Beta is u16 fixed point, exactly the convention msur uses for its truth
 * matrix: code/65534 in [0,1], and MSFM_NA for missing. Half the size of f32,
 * and 1/65534 is far finer than a beta backed by a few hundred CpGs warrants. */
#define MSFM_NA UINT16_MAX

static inline uint16_t msfm_encode(double beta) {
  if (!(beta >= 0.0) && !(beta <= 0.0)) return MSFM_NA;   /* NaN */
  if (beta < 0.0) beta = 0.0;
  if (beta > 1.0) beta = 1.0;
  return (uint16_t)(beta * 65534.0 + 0.5);
}
static inline double msfm_decode(uint16_t v) {
  return v == MSFM_NA ? (0.0 / 0.0) : (double)v / 65534.0;
}

#pragma pack(push, 1)
typedef struct {
  char     magic[8];        /* "MSFMAT1\0" */
  uint32_t version;         /* 1 */
  uint32_t n_records;
  uint32_t n_patterns;
  uint32_t n_classes;
  uint32_t flags;
  uint32_t reserved;
  uint64_t names_offset;
  uint64_t rows_offset;
  uint64_t labels_offset;
  uint64_t levels_offset;
  uint64_t beta_offset;
  uint64_t count_offset;    /* 0 when absent */
  uint64_t file_bytes;
} msfm_header_t;
#pragma pack(pop)

static inline int msfm_is(const msfm_header_t *h) {
  return !memcmp(h->magic, MSFM_MAGIC, 7) && h->version == 1;
}

/* An opened (mmapped) artifact. */
typedef struct {
  int              fd;
  void            *map;
  uint64_t         length;
  const msfm_header_t *header;
  const uint16_t  *beta;        /* n_records * n_patterns */
  const uint32_t  *count;       /* NULL unless F_COUNTS */
  const uint32_t  *levels;      /* n_records */
  const uint16_t  *class_id;    /* n_records */
  char           **pattern_names;
  char           **record_names;
  char           **class_names;
} ms_msfm_t;

/* Open/close. Returns 0 and fills err on a bad or truncated file. */
int  ms_msfm_open(ms_msfm_t *f, const char *path, char *err, size_t errn);
void ms_msfm_close(ms_msfm_t *f);

/* Materialize an artifact as the ms_matrix_t every label model already expects,
 * so `--data` changes only where the features come from, never what is done
 * with them. *labels_out gets one malloc'd label per record (NULL when the
 * artifact is unlabeled) and *levels_out a malloc'd per-record covered-CpG
 * total; both are caller-freed. Either out-pointer may be NULL. */
ms_matrix_t *ms_msfm_to_matrix(const char *path, char ***labels_out,
                               uint32_t **levels_out);

/* `methscope inspect` reporter. */
void ms_msfm_report(const char *path);

/* Sampled, threaded featurization (msfm_build.c). Downsamples and summarizes in
 * one pass: each cell is inflated once and every replicate is drawn from that
 * copy, summarized by scatter-add over the CpG->pattern map rather than a
 * genome-wide scan. rep_sample[r] is the target covered-CpG count for replicate
 * r (0 = keep native coverage). Output is (n_reps * n_cells) rows, replicate
 * major. Caller frees *beta_out, *levels_out, and *names_out (and its strings). */
void ms_msfm_build_sampled(const char *query, const char *mrmp, uint32_t patterns,
                           const uint32_t *rep_sample, uint32_t n_reps,
                           int binarize, uint64_t seed, unsigned threads,
                           uint16_t **beta_out, uint32_t **levels_out,
                           char ***names_out, uint32_t *n_cells_out,
                           uint32_t *ncol_out);

/* Subcommand entry point. */
int  main_classify_featurize(int argc, char *argv[]);

#endif /* MSFM_H */
